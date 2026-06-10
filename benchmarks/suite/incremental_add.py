#!/usr/bin/env python3
"""Benchmark: incremental add() performance.

Compares the wall time of 100 rounds of add-then-search with d=1536, 4-bit.
Each round adds 1 000 vectors then runs a 100-query search.

Before the incremental pack optimisation every add() discarded the entire
blocked layout (O(n·dim) work per add).  After the optimisation only the tail
blocks dirtied by the new vectors are repacked — O(batch·dim) per add.

Run after `cd turbovec-python && maturin develop --release`:
    python benchmarks/suite/incremental_add.py
"""
import os, time, json, numpy as np
from turbovec import TurboQuantIndex

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

DIM = 1536
BIT_WIDTH = 4
BATCH = 1_000     # vectors added per round
ROUNDS = 100      # add rounds
N_QUERY = 100     # queries per search call
K = 10
SEED = 42


def rand_vecs(n, dim, rng):
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def main():
    rng = np.random.RandomState(SEED)
    print(f"=== Incremental add benchmark  d={DIM}  {BIT_WIDTH}-bit ===")
    print(f"    {ROUNDS} rounds × {BATCH} vectors/round, {N_QUERY} queries/round\n")

    index = TurboQuantIndex(DIM, bit_width=BIT_WIDTH)
    queries = rand_vecs(N_QUERY, DIM, rng)

    add_times = []
    search_times = []

    for r in range(ROUNDS):
        batch = rand_vecs(BATCH, DIM, rng)

        t0 = time.perf_counter()
        index.add(batch)
        add_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        index.search(queries, k=K)
        search_times.append(time.perf_counter() - t0)

        if (r + 1) % 20 == 0:
            n = index.len()
            print(
                f"  round {r+1:3d}  n={n:7d}"
                f"  add={add_times[-1]*1e3:6.1f} ms"
                f"  search={search_times[-1]*1e3:6.1f} ms"
            )

    total_add    = sum(add_times)
    total_search = sum(search_times)
    mean_add     = total_add / ROUNDS * 1e3
    mean_search  = total_search / ROUNDS * 1e3

    print()
    print(f"  Total add time    : {total_add:.2f} s  (mean {mean_add:.1f} ms/round)")
    print(f"  Total search time : {total_search:.2f} s  (mean {mean_search:.1f} ms/round)")
    print(f"  Grand total       : {total_add + total_search:.2f} s")

    results = {
        "dim": DIM,
        "bit_width": BIT_WIDTH,
        "rounds": ROUNDS,
        "batch_per_round": BATCH,
        "n_queries": N_QUERY,
        "k": K,
        "total_add_s": round(total_add, 3),
        "total_search_s": round(total_search, 3),
        "mean_add_ms": round(mean_add, 2),
        "mean_search_ms": round(mean_search, 2),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "incremental_add.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
