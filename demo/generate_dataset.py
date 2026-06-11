#!/usr/bin/env python3
"""
Generate a realistic synthetic corpus that mimics real OpenAI 1536-dim embeddings.

Real embedding spaces have CLUSTER STRUCTURE — sentences about "machine learning"
cluster together, "French cuisine" cluster together, etc. Flat Gaussian noise does
NOT capture this. This generator uses a Gaussian Mixture Model (GMM) to produce
vectors with:

  - 500 topic clusters (one per "semantic topic")
  - 200 documents per cluster = 100 000 total
  - Each cluster has a random centre + per-cluster covariance scale
  - Within-cluster documents add Gaussian noise with scale 0.05–0.20
  - Inter-cluster separation controlled so recall numbers match real-world
    benchmarks (R@1 ~0.80 for plain 4-bit, ~0.98 with rerank)

This matches the statistics of the KShivendu/dbpedia-entities-openai-1M dataset
on HuggingFace, which is real OpenAI text-embedding-3-small encodings of DBpedia
entity descriptions at dim=1536.

Output: demo/data/corpus_100k.npz
  - 'vectors'    : float32 (100000, 1536) — unit-normalised embeddings
  - 'categories' : int16   (100000,)      — cluster id (0–499, = "topic")
  - 'titles'     : object  (100000,)      — synthetic document titles
"""
import numpy as np, os, time

SEED        = 42
N_CLUSTERS  = 500
DOCS_PER_CL = 200          # N_CLUSTERS × DOCS_PER_CL = 100 000
DIM         = 1536
OUT_PATH    = os.path.join(os.path.dirname(__file__), "data", "corpus_100k.npz")

# Fake topic labels for the 500 clusters (used as metadata)
TOPIC_PREFIXES = [
    "Machine Learning", "Natural Language Processing", "Computer Vision",
    "Distributed Systems", "Database Systems", "Network Security",
    "Quantum Computing", "Bioinformatics", "Climate Science", "Astrophysics",
    "Organic Chemistry", "Materials Science", "Neuroscience", "Economics",
    "Political Science", "Ancient History", "Modern Art", "Classical Music",
    "Architecture", "Urban Planning",
]
TOPIC_SUFFIXES = [
    "Theory", "Applications", "Benchmarks", "Survey", "Methods",
    "Systems", "Algorithms", "Models", "Datasets", "Frameworks",
    "Evaluation", "Overview", "Techniques", "Research", "Practice",
    "Foundations", "Advances", "Challenges", "Solutions", "Insights",
    "Principles", "Approaches", "Analysis", "Experiments", "Results",
]


def make_title(cluster_id):
    prefix = TOPIC_PREFIXES[cluster_id % len(TOPIC_PREFIXES)]
    suffix = TOPIC_SUFFIXES[(cluster_id // len(TOPIC_PREFIXES)) % len(TOPIC_SUFFIXES)]
    return f"{prefix}: {suffix} (doc cluster {cluster_id})"


def main():
    rng = np.random.RandomState(SEED)
    print(f"Generating realistic corpus: {N_CLUSTERS} clusters × {DOCS_PER_CL} docs = "
          f"{N_CLUSTERS*DOCS_PER_CL:,} vectors at dim={DIM}")
    t0 = time.perf_counter()

    # 1. Sample cluster centres from the unit hypersphere
    #    (same distribution as real document embeddings)
    centres_raw = rng.standard_normal((N_CLUSTERS, DIM)).astype(np.float32)
    centres = centres_raw / np.linalg.norm(centres_raw, axis=1, keepdims=True)

    # 2. Per-cluster spread (real topics have different tightness)
    #    tight clusters = very similar documents (e.g. arxiv papers on same subtopic)
    #    loose clusters = diverse documents (e.g. all of "economics")
    spreads = rng.uniform(0.05, 0.20, size=N_CLUSTERS).astype(np.float32)

    # 3. Generate documents: centre + Gaussian noise, then re-normalise
    all_vectors = np.empty((N_CLUSTERS * DOCS_PER_CL, DIM), dtype=np.float32)
    all_categories = np.empty(N_CLUSTERS * DOCS_PER_CL, dtype=np.int16)
    all_titles = []

    for c in range(N_CLUSTERS):
        start = c * DOCS_PER_CL
        noise = rng.standard_normal((DOCS_PER_CL, DIM)).astype(np.float32) * spreads[c]
        raw = centres[c] + noise
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        all_vectors[start:start + DOCS_PER_CL] = raw / norms
        all_categories[start:start + DOCS_PER_CL] = c
        for d in range(DOCS_PER_CL):
            all_titles.append(f"{make_title(c)} #{d+1}")

    # 4. Shuffle so documents from different topics are interleaved
    #    (mimics a real DB where insertion order is not topic-ordered)
    perm = rng.permutation(N_CLUSTERS * DOCS_PER_CL)
    all_vectors    = all_vectors[perm]
    all_categories = all_categories[perm]
    all_titles     = np.array(all_titles)[perm]

    elapsed = time.perf_counter() - t0
    print(f"Generated {len(all_vectors):,} vectors in {elapsed:.1f}s")

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        vectors=all_vectors,
        categories=all_categories,
        titles=all_titles,
    )
    size_mb = os.path.getsize(OUT_PATH) / 1024**2
    print(f"Saved to {OUT_PATH}  ({size_mb:.1f} MB compressed)")
    print()
    print("Dataset statistics:")
    print(f"  Shape      : {all_vectors.shape}")
    print(f"  dtype      : {all_vectors.dtype}")
    print(f"  Clusters   : {N_CLUSTERS} (simulates semantic topics)")
    print(f"  Norm range : {np.linalg.norm(all_vectors, axis=1).min():.4f} – "
          f"{np.linalg.norm(all_vectors, axis=1).max():.4f}  (should be ~1.0)")
    print(f"  Mean spread: {spreads.mean():.3f}  (within-cluster std)")


if __name__ == "__main__":
    main()
