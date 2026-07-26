"""
turbovec — interactive quantized vector search (Hugging Face Space).

Builds a real turbovec index over a synthetic 32-topic embedding corpus and
lets you explore the accuracy/memory trade-off of bit-width, refine mode
(int8 / float16 / float32) and the cascade rerank factor — live.

Every number is measured from the actual Rust engine via the `turbovec`
Python wheel (built from source in the Dockerfile).
"""
import time
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from turbovec import TurboQuantIndex

# ── corpus ────────────────────────────────────────────────────────────────────
DIM, N, CLUSTERS, SEED = 256, 4000, 32, 42
TOPICS = [
    "Machine Learning", "Natural Language Processing", "Computer Vision",
    "Distributed Systems", "Databases", "Network Security", "Quantum Computing",
    "Bioinformatics", "Climate Science", "Astrophysics", "Organic Chemistry",
    "Materials Science", "Neuroscience", "Economics", "Political Science",
    "Ancient History", "Modern Art", "Classical Music", "Architecture",
    "Urban Planning", "Robotics", "Genomics", "Cryptography", "Game Theory",
    "Ecology", "Linguistics", "Philosophy", "Cartography", "Meteorology",
    "Paleontology", "Oceanography", "Sociology",
]


def make_corpus():
    rng = np.random.RandomState(SEED)
    cent = rng.standard_normal((CLUSTERS, DIM)).astype("float32")
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    spread = rng.uniform(0.05, 0.20, CLUSTERS).astype("float32")
    vecs = np.empty((N, DIM), "float32")
    cl = np.empty(N, "int32")
    for r in range(N):
        c = rng.randint(CLUSTERS)
        cl[r] = c
        v = cent[c] + rng.standard_normal(DIM).astype("float32") * spread[c]
        vecs[r] = v / np.linalg.norm(v)
    return vecs, cl, cent, spread


VECS, CL, CENT, SPREAD = make_corpus()

# Indexes are cached per (bit_width, refine) so repeated runs are instant.
_INDEX_CACHE = {}


def get_index(bit_width, refine):
    key = (bit_width, refine)
    if key not in _INDEX_CACHE:
        idx = TurboQuantIndex(
            DIM, bit_width=bit_width, refine=None if refine == "none" else refine
        )
        idx.add(VECS)
        _INDEX_CACHE[key] = idx
    return _INDEX_CACHE[key]


def mem_bytes(n, dim, bit_width, refine):
    coarse = n * (((dim * bit_width + 7) // 8) + 4)
    ref = {
        "none": 0,
        "int8": n * (dim + 4),
        "float16": n * dim * 2,
        "float32": n * dim * 4,
    }[refine]
    return n * dim * 4, coarse, ref


def brute_top1(q):
    return int(np.argmax(VECS @ q))


def _search(idx, q2, k, refine, rf):
    if refine != "none" and rf > 1:
        return idx.search(q2, k, rerank_factor=rf)
    return idx.search(q2, k)


# ── main callback ─────────────────────────────────────────────────────────────
def run(topic, bit_width, refine, rerank_factor, k):
    bit_width, k, rf = int(bit_width), int(k), int(rerank_factor)
    idx = get_index(bit_width, refine)
    rng = np.random.RandomState()

    # Recall@1 (is the returned #1 the exact brute-force nearest neighbour?)
    # + latency, over a batch of fresh queries. R@1 differentiates the modes;
    # a "true NN in top-k" metric saturates on well-separated clusters.
    hits, NQ, elapsed = 0, 40, 0.0
    for _ in range(NQ):
        cc = rng.randint(CLUSTERS)
        qq = CENT[cc] + rng.standard_normal(DIM).astype("float32") * SPREAD[cc]
        qq /= np.linalg.norm(qq)
        truth = brute_top1(qq)
        t0 = time.perf_counter()
        _, ix = _search(idx, qq.reshape(1, -1), k, refine, rf)
        elapsed += time.perf_counter() - t0
        if ix.shape[1] > 0 and int(ix[0, 0]) == truth:
            hits += 1
    recall, avg_ms = hits / NQ, elapsed / NQ * 1000

    # One displayed query from the chosen topic.
    c = TOPICS.index(topic)
    q = CENT[c] + rng.standard_normal(DIM).astype("float32") * SPREAD[c]
    q /= np.linalg.norm(q)
    truth = brute_top1(q)
    sc, ix = _search(idx, q.reshape(1, -1), k, refine, rf)
    rows = []
    for j in range(ix.shape[1]):
        slot = int(ix[0, j])
        rows.append([
            j + 1,
            slot,
            TOPICS[CL[slot]] if slot >= 0 else "—",
            round(float(sc[0, j]), 4),
            "★ true NN" if slot == truth else "",
        ])

    f32, coarse, ref = mem_bytes(N, DIM, bit_width, refine)
    total = coarse + ref
    comp = f32 / total
    mem_md = (
        f"### Memory\n"
        f"- float32 raw: **{f32/1e6:.2f} MB**\n"
        f"- turbovec index: **{coarse/1e6:.2f} MB**"
        + (f"  ·  +{refine} refine: **{ref/1e6:.2f} MB**" if ref else "")
        + f"\n- **{comp:.1f}× smaller** than float32"
    )
    rec_md = (
        f"### Quality\n"
        f"- **Recall@1 = {recall:.3f}**  ({hits}/{NQ} queries returned the exact nearest neighbour at rank 1)\n"
        f"- **{avg_ms:.2f} ms** per query  ·  {1000/max(avg_ms,1e-6):.0f} queries/sec"
    )

    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    ax.barh(["float32", "index", "+refine"],
            [f32 / 1e6, coarse / 1e6, ref / 1e6],
            color=["#d64545", "#2563eb", "#6d3cff"])
    ax.set_xlabel("MB")
    ax.invert_yaxis()
    for i, v in enumerate([f32 / 1e6, coarse / 1e6, ref / 1e6]):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    return rows, mem_md, rec_md, fig


# ── UI ────────────────────────────────────────────────────────────────────────
INTRO = """
# 🧭 turbovec — quantized vector search, live

A data-oblivious **2–4-bit vector quantizer + SIMD ANN engine** for embeddings.
Pick a topic and tune the knobs to see the **memory ↔ recall** trade-off measured
from the real Rust engine.

- **Bit width** — fewer bits ⇒ smaller index, lower coarse recall.
- **Refine mode** — stores originals for an exact re-rank: `int8` (4× smaller),
  **`float16`** *(new in this fork — half the size of float32, near-exact)*, or `float32`.
- **Rerank factor** — how many coarse candidates the exact re-rank re-scores.

Try `2-bit + float16 + rerank 8` vs `4-bit + none`: similar recall, very different memory.
"""

with gr.Blocks(title="turbovec — quantized vector search", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            topic = gr.Dropdown(TOPICS, value=TOPICS[0], label="Query topic")
            bit_width = gr.Radio(["2", "3", "4"], value="4", label="Bit width")
            refine = gr.Radio(
                ["none", "int8", "float16", "float32"],
                value="float16", label="Refine mode",
            )
            rerank_factor = gr.Radio(["1", "2", "4", "8"], value="4", label="Rerank factor")
            k = gr.Radio(["1", "10", "20"], value="10", label="k (results)")
            go = gr.Button("Search", variant="primary")
        with gr.Column(scale=2):
            mem = gr.Markdown()
            rec = gr.Markdown()
            plot = gr.Plot(label="Memory footprint")
    results = gr.Dataframe(
        headers=["rank", "slot", "cluster topic", "score", "ground truth"],
        label="Top-k results (last query)",
        wrap=True,
    )
    go.click(run, [topic, bit_width, refine, rerank_factor, k], [results, mem, rec, plot])
    demo.load(run, [topic, bit_width, refine, rerank_factor, k], [results, mem, rec, plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
