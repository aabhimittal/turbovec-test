#!/usr/bin/env python3
"""Recall benchmark: d=1536, sweeping bit_width × rerank_factor.

Shows that 2-bit + int8 re-rank recovers the recall of 4-bit at lower memory.

The rerank feature requires a TurboQuantIndex built with `refine="int8"` or
`refine="float32"`.  `search(queries, k, rerank_factor=R)` runs a coarse
scan for k*R candidates then re-scores them with the stored originals.

Run after `cd turbovec-python && maturin develop --release`:
    python benchmarks/suite/recall_d1536_4bit_rerank.py
"""
import os, json, time, numpy as np
from turbovec import TurboQuantIndex

DATA_DIR = os.path.expanduser("~/data/py-turboquant")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DIM = 1536
K = 10
SEED = 42
RERANK_FACTORS = [1, 2, 4, 8]
BIT_WIDTHS = [2, 4]
REFINE_MODE = "int8"   # "float32" for exact re-rank


def load_openai(dim):
    path = os.path.join(DATA_DIR, f"openai-{dim}.npy")
    all_vecs = np.load(path)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(all_vecs))
    database = all_vecs[idx[:100_000]].astype(np.float32)
    queries  = all_vecs[idx[100_000:101_000]].astype(np.float32)
    database /= np.linalg.norm(database, axis=-1, keepdims=True)
    queries  /= np.linalg.norm(queries,  axis=-1, keepdims=True)
    return database, queries


def recall_at_k(true_top1, predicted, k):
    return float(np.mean([true_top1[i] in predicted[i, :k] for i in range(len(true_top1))]))


def main():
    print(f"=== Rerank recall benchmark  d={DIM}  refine={REFINE_MODE} ===\n")

    database, queries = load_openai(DIM)
    true_top1 = np.argmax(queries @ database.T, axis=1)

    all_results = {}

    for bw in BIT_WIDTHS:
        # Build index with refine store
        t0 = time.time()
        idx = TurboQuantIndex(DIM, bit_width=bw, refine=REFINE_MODE)
        idx.add(database)
        build_s = time.time() - t0

        row = {"bit_width": bw, "build_s": round(build_s, 2), "rerank": {}}

        for rf in RERANK_FACTORS:
            t0 = time.time()
            if rf == 1:
                # rerank_factor=1: coarse only (same as plain search with exact scores)
                _, pred = idx.search(queries, k=K, rerank_factor=rf)
            else:
                _, pred = idx.search(queries, k=K, rerank_factor=rf)
            elapsed = time.time() - t0
            pred = np.array(pred)
            rec = recall_at_k(true_top1, pred, K)
            row["rerank"][str(rf)] = {"recall": round(rec, 4), "search_s": round(elapsed, 2)}
            print(
                f"  {bw}-bit  rerank_factor={rf:2d}"
                f"  recall@{K}={rec:.4f}"
                f"  search={elapsed:.2f}s"
            )

        all_results[str(bw)] = row
        print()

    # Also run plain 4-bit (no refine) as baseline
    print("  --- baseline (4-bit, no refine) ---")
    t0 = time.time()
    idx_plain = TurboQuantIndex(DIM, bit_width=4)
    idx_plain.add(database)
    _, pred_plain = idx_plain.search(queries, k=K)
    elapsed = time.time() - t0
    pred_plain = np.array(pred_plain)
    rec_plain = recall_at_k(true_top1, pred_plain, K)
    print(f"  4-bit plain  recall@{K}={rec_plain:.4f}  total={elapsed:.2f}s\n")
    all_results["4_plain"] = {"recall": round(rec_plain, 4), "total_s": round(elapsed, 2)}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "recall_d1536_rerank.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
