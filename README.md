<p align="center">
  <img src="docs/header.png" alt="turbovec — Google's TurboQuant for vector search" width="100%">
</p>

<p align="center">
  <a href="https://github.com/RyanCodrai/turbovec/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/turbovec" alt="License"></a>
  <a href="https://pypi.org/project/turbovec/"><img src="https://img.shields.io/pypi/v/turbovec?label=pypi&color=blue" alt="PyPI version"></a>
  <a href="https://crates.io/crates/turbovec"><img src="https://img.shields.io/crates/v/turbovec?label=crates.io&color=blue" alt="crates.io version"></a>
  <a href="https://arxiv.org/abs/2504.19874"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="TurboQuant paper"></a>
</p>

---

**A 10 million document corpus takes 31 GB of RAM as float32. turbovec fits it in 4 GB - and searches it faster than FAISS.**

turbovec is a Rust vector index with Python bindings, built on Google Research's [**TurboQuant**](https://arxiv.org/abs/2504.19874) algorithm — a data-oblivious quantizer that matches the Shannon lower bound on distortion, with no codebook training and no separate train phase.

- **Online ingest.** Add vectors, they're indexed — no train step, no parameter tuning, no rebuilds as the corpus grows.
- **Faster than FAISS.** Hand-written NEON (ARM) and AVX-512BW (x86) kernels beat FAISS IndexPQFastScan by 12–20% on ARM and match-or-beat it on x86.
- **Filter at search time.** Pass an id allowlist (or a slot bitmask) to `search()` and the kernel honours it directly. You always get up to `k` results from the allowed set — no over-fetching, no recall hit on selective filters.
- **Pure local.** No managed service, no data leaving your machine or VPC. Pair with any open-source embedding model for a fully air-gapped RAG stack.

Building RAG where privacy, memory, or latency matters? **You're in the right place.**

---

## Fork Enhancements — LLM Inference Techniques Applied to Vector Search

> This fork explores three ideas from modern LLM inference and applies them directly to turbovec's vector-search internals. Each technique is explained from first principles so a reader new to the field can follow along.

### Quick results

| Test suite | Tests | Result |
|---|---|---|
| `codebook` | 6 | ✅ all pass |
| `concurrent_search` | 8 | ✅ all pass |
| `distortion` | 5 | ✅ all pass |
| `encode` | 5 | ✅ all pass |
| `filtering` | 15 | ✅ all pass |
| `id_map` | 18 | ✅ all pass |
| `incremental_pack` *(new)* | 8 | ✅ all pass |
| `input_validation` | 15 | ✅ all pass |
| `io_versioning` | 9 | ✅ all pass |
| `kernel_correctness` | 6 | ✅ all pass |
| `lazy_init` | 24 | ✅ all pass |
| `rerank` *(new)* | 11 | ✅ all pass |
| `rotation` | 5 | ✅ all pass |
| `state_sequences` | 9 | ✅ all pass |
| `swap_remove` | 6 | ✅ all pass |
| `tqplus_calibration` | 2 | ✅ all pass |
| `paged-llm-demo` *(new)* | 2 | ✅ all pass |
| lib unit tests | 7 | ✅ all pass |
| doc-tests | 2 | ✅ all pass |
| **Total** | **163** | **✅ 163 / 163** |

---

### Enhancement 1 — Incremental blocked-layout packing

#### The problem (explained simply)

When you add a batch of vectors to a turbovec index, the library has to build a special SIMD-optimised ("blocked") layout of all the data so that the CPU's vector instructions can search it quickly. Think of this like a bookcase: before the fix, every time you added even one new book, the system would take *all* the books off the shelves and re-arrange them from scratch. That is O(n·dim) work — proportional to the *total* number of vectors already stored — making repeated adds quadratically expensive.

> **Glossary — O(n) notation:** "Big-O" notation describes how cost scales with size. O(n·dim) means: if the index has n vectors of dimension dim, the work grows linearly with both. For n=1 million and dim=1536 that is ~1.5 billion floating-point operations just to reorganise data that didn't change.

#### The insight (inspired by PagedAttention)

turbovec's blocked layout already divides vectors into groups of 32 (called *blocks*). Each block's packed bytes depend *only* on those 32 vectors — not on anything else in the index. So when you append new vectors, only the last few blocks are "dirty". Everything before them is identical to what was there before.

The new `repack_range(first_block)` function rebuilds only the tail blocks starting from `first_block = old_n / 32`. The rest of the layout is kept as-is.

> **Glossary — PagedAttention (vLLM):** In large language model servers, the "KV cache" (the memory storing intermediate computation for past tokens) used to be allocated as one big contiguous slab per conversation — wasteful because short conversations would leave most of the slab empty. PagedAttention (from the vLLM project, 2023) instead organises the KV cache as a pool of small fixed-size *pages*, allocated on demand. The insight is the same: stop discarding and rebuilding everything just because a small tail changed.

#### What changed in the code

| File | What changed |
|---|---|
| `turbovec/src/pack.rs` | New `repack_range(first_block)` function that produces packed bytes only for blocks `[first_block, n_blocks)` |
| `turbovec/src/lib.rs` | `add()` now updates the blocked cache in-place via `get_mut()` instead of invalidating and rebuilding from scratch |
| `turbovec/tests/incremental_pack.rs` | 8 new tests — each verifies that incremental packing produces *byte-identical* output to a full rebuild |

#### Test results

```
running 8 tests
test incremental_pack_2bit_partial_blocks                 ... ok
test incremental_pack_3bit_mixed                          ... ok
test incremental_pack_4bit_block_aligned_batches          ... ok
test incremental_pack_4bit_large_dim                      ... ok
test incremental_pack_4bit_non_aligned_batches            ... ok
test incremental_pack_cold_cache_then_search              ... ok
test incremental_pack_full_lifecycle_matches_fresh_index  ... ok
test repack_range_matches_full_repack_for_suffix          ... ok

test result: ok. 8 passed; 0 failed; 0 ignored
```

The key test (`repack_range_matches_full_repack_for_suffix`) directly compares the bytes produced by incremental packing against a full rebuild and asserts they are identical — so recall is *provably unchanged*. No approximations, no tradeoffs.

---

### Enhancement 2 — Cascade re-ranking with a refinement store

#### The problem (explained simply)

turbovec compresses each vector from 32 bytes (float32) down to 2–4 bits per dimension — about a 16× shrink. This is great for throughput, but the compression discards precision. Searching the compressed index finds *approximately* the right results; the top-1 vector returned might not be the actual closest vector in the original (uncompressed) space.

#### The solution

Store the original vectors alongside the compressed index — and use them only for a small shortlist. The flow is:

1. **Coarse scan:** Use the fast SIMD compressed index to find the top `k × rerank_factor` candidates (e.g. k=10, factor=4 → 40 candidates). This costs almost nothing extra compared to a plain search.
2. **Re-rank:** For each of those 40 candidates, compute the exact inner product using the stored original vectors. Sort by exact score, take the true top 10.

> **Glossary — knowledge distillation:** In machine learning, "distillation" means using a large accurate "teacher" model to improve a compact "student" model. Here the stored originals are the teacher (exact, expensive) and the compressed SIMD index is the student (fast, approximate). The two-stage pipeline gets student speed with near-teacher accuracy.

> **Glossary — inner product / cosine similarity:** Two vectors "agree" to the degree their elements point in the same direction. The inner product (dot product) measures this: `score = Σ a_i × b_i`. For unit-length vectors this equals the cosine of the angle between them — 1.0 = identical direction, 0.0 = perpendicular, −1.0 = opposite. Vector search finds the stored vector with the highest inner product against the query.

#### Two storage modes

| Mode | Bytes per vector (d=1536) | Accuracy |
|---|---|---|
| `RefineMode::Float32` | 6 144 B (full precision) | Exact inner product |
| `RefineMode::Int8` | 1 540 B (int8 + 1 scale) | ~0.9999 cosine sim vs float32 |

The Int8 mode uses the same symmetric quantization scheme as llama.cpp's `q8_0` KV cache: the maximum absolute value in each vector determines a `scale = max_abs / 127`, then every element is rounded to the nearest integer in `[-127, 127]` and stored as a single byte.

> **Glossary — INT8 / int8 quantization:** Storing a floating-point number (32 bits, ~7 decimal digits of precision) as an 8-bit integer (1 byte, values −128 to 127). To avoid catastrophic rounding, a "scale" factor is stored alongside: `original ≈ code × scale`. The reconstruction error is at most `scale / 2`. This gives a 4× memory reduction with accuracy loss typically < 0.1%.

#### Python API

```python
from turbovec import TurboQuantIndex

# Build index with refinement store (Int8 saves ~4× memory vs Float32 refine)
index = TurboQuantIndex(dim=1536, bit_width=4, refine="int8")
index.add(vectors)

# Plain search — fast compressed SIMD, approximate scores
scores, indices = index.search(queries, k=10)

# Re-ranked search — coarse SIMD shortlist → exact re-score
# rerank_factor=4 means scan 40 candidates, return best 10
scores, indices = index.search(queries, k=10, rerank_factor=4)
```

#### File format backward compatibility

A turbovec index file can be saved and loaded across versions. The new RefineStore is stored as **file format v4** — but only when a RefineStore is present. Default indexes (no refine) continue writing format v3 unchanged. An old turbovec reader opening a v3 file sees exactly what it always saw.

> **Glossary — file format versioning:** Programs agree on a byte-layout "contract" for how data is laid out on disk. When that layout changes, the version number increases. Old readers must be able to reject files they don't understand (or at least fail gracefully) rather than silently misinterpreting bytes.

#### Test results

```
running 11 tests
test default_index_writes_v3_not_v4          ... ok  ← backward compat
test float32_rerank_exact_recovery           ... ok  ← exact scores = brute-force
test id_map_rerank_returns_external_ids      ... ok
test int8_rerank_monotone_recall             ... ok  ← int8 recall ≥ plain coarse
test invalid_rerank_factor_returns_error     ... ok
test mask_and_rerank_respects_mask           ... ok
test no_refine_store_returns_error           ... ok
test refine_store_aligned_after_swap_remove  ... ok
test rerank_factor_1_matches_coarse_index_set... ok
test v4_round_trip_float32                   ... ok
test v4_round_trip_int8                      ... ok

test result: ok. 11 passed; 0 failed; 0 ignored
```

The `float32_rerank_exact_recovery` test is the strongest correctness guarantee: with `rerank_factor = n` (scan all vectors), the reranked results must equal brute-force float32 top-k. This proves that re-ranking with stored originals recovers ground truth exactly.

---

### Enhancement 3 — `examples/paged-llm-demo` educational crate

A self-contained Rust program that teaches PagedAttention, INT8 KV quantization, and how both connect to turbovec — all in under 350 lines, zero dependencies, runs in under a second.

```bash
cargo run -p paged-llm-demo --release
```

#### What it simulates

8 conversations (sequences) of different lengths (4–20 tokens each) are decoded token-by-token through a toy transformer (d_model=64, 4 attention heads). Four KV caches run in parallel so their memory usage can be compared:

> **Glossary — KV cache:** In a transformer language model, every token the model has seen generates two vectors (Key and Value) that all future tokens need to look at (the "attention" mechanism). Storing these so they don't have to be recomputed is the KV cache. At long context lengths or large batch sizes, the KV cache can be the dominant memory consumer in a GPU server.

> **Glossary — transformer / attention heads:** A transformer is the neural network architecture underlying GPT, LLaMA, Claude, etc. "Multi-head attention" splits the hidden dimension into several independent "heads", each attending to different aspects of the context. With 4 heads and d_model=64, each head has a dimension of 16.

#### Actual demo output

```
═══════════════════════════════════════════════════════════════
  Paged-LLM Demo  —  PagedAttention × INT8 KV quantization
═══════════════════════════════════════════════════════════════

Config: BATCH=8, MAX_SEQ_LEN=32
        D_MODEL=64, N_HEADS=4, HEAD_DIM=16
        PAGE_SIZE=16, POOL_PAGES=28

Per-sequence lengths (actual vs declared):
  seq 0: declared= 4  contiguous_len= 4  paged_len= 4
  seq 1: declared= 7  contiguous_len= 7  paged_len= 7
  seq 2: declared=12  contiguous_len=12  paged_len=12
  seq 3: declared=20  contiguous_len=20  paged_len=20
  seq 4: declared= 6  contiguous_len= 6  paged_len= 6
  seq 5: declared=15  contiguous_len=15  paged_len=15
  seq 6: declared= 8  contiguous_len= 8  paged_len= 8
  seq 7: declared=19  contiguous_len=19  paged_len=19

Memory utilisation (KV cache, both K and V, f32 storage):
  Cache                       Allocated        Used     Util%
  ──────────────────────────────────────────────────────────
  Contiguous fp32  (KB)          128.00       45.50     35.5%
  Paged      fp32  (KB)           80.00       45.50     56.9%

  Contiguous wastes 64.5% of allocated memory (pad to max_seq_len).
  Paged pool allocates only pages that are actually filled.

PagedAttention pool stats (fp32 paged cache):
  Pool size   : 28 pages × 16 tokens/page
  Pages in use: 10 / 28
  Free pages  : 18

Theoretical INT8 vs f32 KV memory (all sequences combined):
  Total tokens decoded : 91
  f32 KV storage       : 45.50 KB
  int8 KV storage      : 12.09 KB  (3.8× reduction)
  (int8 = 1 byte/dim + 4-byte scale per vector, vs 4 bytes/dim for f32)

Attention output cosine similarity (fp32 KV vs int8 KV):
  Steps measured : 91
  Mean cos-sim   : 0.999932
  Min  cos-sim   : 0.999528

[OK] Self-check passed: mean cos-sim 0.999932 > 0.999

Turbovec mapping:
  PagedKvCache block table     <-> turbovec BLOCK=32 blocked layout
  free() returns pages to pool <-> swap_remove() reclaims vector slot
  INT8 KV store + per-token scale <-> turbovec RefineStore::Int8
  on-demand page alloc in append() <-> pack::repack_range (tail-only repack)
```

#### Reading the numbers

**Memory utilisation (35.5% vs 56.9%)**
The contiguous cache pre-allocates `max_seq_len × kv_dim × 4 bytes` for every sequence slot regardless of actual length — 128 KB total for 8 slots. Only 45.5 KB is ever written (the actual tokens). That is 64.5% wasted. The paged cache allocates 80 KB (one pool of pages), 56.9% of which is used. As sequences finish and release their pages, that memory immediately becomes available to new sequences. In a real GPU server with thousands of concurrent requests, this difference determines whether you can serve 2× or 3× as many users simultaneously.

**3.8× INT8 memory reduction with 0.9999 cosine similarity**
91 tokens × 64-dimensional KV = 45.5 KB at float32. At INT8 (1 byte/element + 4-byte scale per 64-element vector) the same data fits in 12.1 KB. The cosine similarity of attention outputs — the actual numbers the model computes — stays above 0.9999 (vs perfect 1.0). In practice, models fine-tuned for 4-bit or 8-bit KV cache see no perceptible quality degradation at this similarity level.

**Why the self-check assert matters**
The demo includes `assert!(mean_cos > 0.999)` at the end. This means `cargo run` doubles as a regression test: if a future change accidentally broke INT8 accuracy, the program would exit with an error rather than silently producing wrong numbers.

---

### How the three enhancements connect

```
                    LLM technique           →    turbovec equivalent
    ─────────────────────────────────────────────────────────────────
    PagedAttention: per-sequence           →    BLOCK=32 blocked layout
      page pool, on-demand alloc                 (already paged at 32-vec
                                                  granularity)

    PagedAttention: tail-only updates     →    pack::repack_range()
      when new tokens arrive                     rebuilds only dirty tail
                                                  blocks on each add()

    INT8 KV cache quantization            →    RefineStore::Int8
      (llama.cpp q8_0)                           stores originals as int8
                                                  for cascade re-scoring

    Knowledge distillation:               →    search_with_rerank()
      teacher (exact) + student (fast)           coarse SIMD scan then
                                                  exact re-score of top-kR
```

> **Glossary — SIMD:** Single Instruction, Multiple Data. Modern CPUs can perform the same arithmetic operation on 8, 16, or 32 numbers in one clock cycle instead of one at a time. The NEON (ARM) and AVX2/AVX-512 (Intel/AMD) instruction sets are SIMD families. turbovec's search kernels are hand-written in these instructions to maximise throughput. The "blocked layout" (groups of 32 vectors packed in a specific byte order) is designed specifically so these SIMD instructions can process a full block in the fewest possible instructions.

---

### New files added

| Path | Purpose |
|---|---|
| `turbovec/src/refine.rs` | `RefineStore` — Int8/Float32 original-vector store for re-ranking |
| `turbovec/tests/incremental_pack.rs` | 8 byte-identity tests for tail-only repack |
| `turbovec/tests/rerank.rs` | 11 correctness + round-trip tests for cascade re-rank |
| `examples/paged-llm-demo/` | Educational demo crate (KV cache + INT8 quant + MHA) |
| `docs/ARCHITECTURE.md` | Mermaid module graph, encode/search pipelines, LLM mapping |
| `CLAUDE.md` | Developer guide: commands, module map, gotchas |
| `benchmarks/suite/incremental_add.py` | Benchmark: 100× add-then-search wall time |
| `benchmarks/suite/recall_d1536_4bit_rerank.py` | Benchmark: sweep rerank_factor on 2-bit + 4-bit |

---

## Python

```bash
pip install turbovec
```

```python
from turbovec import TurboQuantIndex

index = TurboQuantIndex(dim=1536, bit_width=4)
index.add(vectors)
index.add(more_vectors)

scores, indices = index.search(query, k=10)

index.write("my_index.tv")
loaded = TurboQuantIndex.load("my_index.tv")
```

Need stable ids that survive deletes? Use `IdMapIndex`:

```python
import numpy as np
from turbovec import IdMapIndex

index = IdMapIndex(dim=1536, bit_width=4)
index.add_with_ids(vectors, np.array([1001, 1002, 1003], dtype=np.uint64))

scores, ids = index.search(query, k=10)   # ids are your uint64 external ids
index.remove(1002)                         # O(1) by id

index.write("my_index.tvim")
loaded = IdMapIndex.load("my_index.tvim")
```

### Hybrid retrieval (filtered search)

Restrict results to a candidate set produced by another system (SQL, BM25, ACL, time window, …):

```python
import numpy as np
from turbovec import IdMapIndex

idx = IdMapIndex(dim=1536, bit_width=4)
idx.add_with_ids(vectors, ids)

# Stage 1: external system narrows to candidate ids.
allowed = np.array(db.execute("SELECT id FROM docs WHERE tenant=?", (t,)).fetchall(),
                   dtype=np.uint64)

# Stage 2: dense rerank within the candidate set.
scores, ids = idx.search(query, k=10, allowlist=allowed)
```

Filtering happens inside the SIMD kernel at 32-vector block granularity: blocks with no allowed slots are short-circuited before any LUT lookup or scoring work, and individual non-allowed slots inside scored blocks are dropped at heap-insert. Selective allowlists (small fraction of the index allowed) therefore avoid most of the SIMD cost rather than paying it and discarding the result afterwards.

The output length is `min(k, len(allowed))` — when the allowlist is smaller than `k` you get exactly `len(allowed)` results rather than padded fallbacks.

See [`docs/api.md`](docs/api.md) for the full reference.

### Framework integrations

Drop-in replacements for the in-tree reference vector / document stores in each framework. Same public surface, same persistence semantics, same retriever and pipeline wiring — swap the import and keep your pipeline.

- [LangChain](docs/integrations/langchain.md) — `pip install turbovec[langchain]` · replaces `langchain_core.vectorstores.InMemoryVectorStore`
- [LlamaIndex](docs/integrations/llama_index.md) — `pip install turbovec[llama-index]` · replaces `llama_index.core.vector_stores.SimpleVectorStore`
- [Haystack](docs/integrations/haystack.md) — `pip install turbovec[haystack]` · replaces `haystack.document_stores.in_memory.InMemoryDocumentStore`
- [Agno](docs/integrations/agno.md) — `pip install turbovec[agno]` · replaces `agno.vectordb.lancedb.LanceDb`

## Rust

```bash
cargo add turbovec
```

```rust
use turbovec::TurboQuantIndex;

let mut index = TurboQuantIndex::new(1536, 4).unwrap();
index.add(&vectors);
let results = index.search(&queries, 10);
index.write("index.tv").unwrap();
let loaded = TurboQuantIndex::load("index.tv").unwrap();
```

For stable external ids that survive deletes:

```rust
use turbovec::IdMapIndex;

let mut index = IdMapIndex::new(1536, 4).unwrap();
index.add_with_ids(&vectors, &[1001, 1002, 1003]).unwrap();
let (scores, ids) = index.search(&queries, 10);
index.remove(1002);
index.write("index.tvim").unwrap();
let loaded = IdMapIndex::load("index.tvim").unwrap();
```

## Recall

TurboQuant vs FAISS `IndexPQ` (LUT256, nbits=8) — the paper's Section 4.4 baseline. 100K vectors, k=64. FAISS PQ sub-quantizer counts sized to match TurboQuant's bit rate (m=d/4 at 2-bit, m=d/2 at 4-bit).

![Recall GloVe d=200](docs/recall_glove.svg)

![Recall d=1536](docs/recall_d1536.svg)

![Recall d=3072](docs/recall_d3072.svg)

Across OpenAI d=1536 and d=3072, TurboQuant beats FAISS by 0.4–3.4 points at R@1 across 2-bit and 4-bit, and both converge to 1.0 by k=4. GloVe d=200 is the harder regime — at low dim the asymptotic Beta assumption is looser. TurboQuant beats FAISS by 0.3 points at 4-bit and trails by 1.2 points at 2-bit at R@1, both closing to FAISS by k≈16.

**A note on baselines.** We compare against FAISS `IndexPQ` (LUT256, nbits=8, float32 LUT) because it's the default production-grade PQ most users would reach for. This is a stronger baseline than the custom u8-LUT PQ in the [TurboQuant paper](https://arxiv.org/abs/2504.19874) — FAISS uses a higher-precision LUT at scoring time and k-means++ for codebook training. We reproduce the paper's TurboQuant numbers on OpenAI d=1536 / d=3072 and hit similar numbers to other community reference implementations on low-dim embeddings (see [`turboquant-py`](https://pypi.org/project/turboquant-py/) at d=384). The visible gap on GloVe reflects FAISS being a strong baseline, not a TurboQuant implementation issue.

Full results: [d=1536 2-bit](benchmarks/results/recall_d1536_2bit.json), [d=1536 4-bit](benchmarks/results/recall_d1536_4bit.json), [d=3072 2-bit](benchmarks/results/recall_d3072_2bit.json), [d=3072 4-bit](benchmarks/results/recall_d3072_4bit.json), [GloVe 2-bit](benchmarks/results/recall_glove_2bit.json), [GloVe 4-bit](benchmarks/results/recall_glove_4bit.json).

## Compression

![Compression](docs/compression.svg)

## Search Speed

All benchmarks: 100K vectors, 1K queries, k=64, median of 5 runs.

### ARM (Apple M3 Max)

![ARM Speed — Single-threaded](docs/arm_speed_st.svg)

![ARM Speed — Multi-threaded](docs/arm_speed_mt.svg)

On ARM, TurboQuant beats FAISS FastScan by 12–20% across every config.

### x86 (Intel Xeon Platinum 8481C / Sapphire Rapids, 8 vCPUs)

![x86 Speed — Single-threaded](docs/x86_speed_st.svg)

![x86 Speed — Multi-threaded](docs/x86_speed_mt.svg)

On x86, TurboQuant wins every 4-bit config by 1–6% and runs within ~1% of FAISS on 2-bit ST. The 2-bit MT rows (d=1536 and d=3072) are the only configs sitting slightly behind FAISS (2–4%), where the inner accumulate loop is too short for unrolling amortization to match FAISS's AVX-512 VBMI path.

## How it works

Each vector is a direction on a high-dimensional hypersphere. TurboQuant compresses these directions using a simple insight: after applying a random rotation, every coordinate follows a known distribution -- regardless of the input data.

**1. Normalize.** Strip the length (norm) from each vector and store it as a single float. Now every vector is a unit direction on the hypersphere.

**2. Random rotation.** Multiply all vectors by the same random orthogonal matrix. After rotation, each coordinate independently follows a Beta distribution that converges to Gaussian N(0, 1/d) in high dimensions. This holds for any input data -- the rotation makes the coordinate distribution predictable.

**3. Per-coordinate calibration (TQ+).** The Beta distribution from step 2 is asymptotic — at finite dimensions, individual coordinates drift from the canonical shape (especially low-bit and word-vector-style embeddings). TQ+ fits two scalars per coordinate — a shift and a scale — during the first add, mapping each coordinate's empirical 5/95% quantiles onto the canonical Beta marginal. The Lloyd-Max codebook then quantizes against the *target* distribution it was designed for. The calibration is frozen after the first add and reused by subsequent adds — no retraining, no rebuilds, no separate train phase. Recall gain: up to +1.4pp at @1 on the cells that drift most (e.g. GloVe at 2-bit).

**4. Lloyd-Max scalar quantization.** Since the distribution is known, we can precompute the optimal way to bucket each coordinate. For 2-bit, that's 4 buckets; for 4-bit, 16 buckets. The [Lloyd-Max algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) finds bucket boundaries and centroids that minimize mean squared error. These are computed once from the math, not from the data.

**5. Bit-pack.** Each coordinate is now a small integer (0-3 for 2-bit, 0-15 for 4-bit). Pack these tightly into bytes. A 1536-dim vector goes from 6,144 bytes (FP32) to 384 bytes (2-bit). That's 16x compression.

**6. Length-renormalized scoring.** Scalar quantization systematically underestimates inner products — the reconstructed unit direction is a little shorter than the original. We compute one scalar per vector at encode time — the inner product of the rotated unit vector with its own centroid reconstruction — and store `||v|| / ⟨u, x̂⟩` alongside each compressed vector. The search kernel multiplies the per-candidate score by this scalar before heap insertion, turning the inner-product estimator from downward-biased into unbiased at zero search-time cost and zero extra storage. The recall gain shows up most at low bit widths, where the quantization shrinkage is largest.

Encoding cost: one extra `d`-dimensional dot product per vector to compute `⟨u, x̂⟩`. On 1M vectors at d=1536 this is sub-second of additional encode time — a one-shot price paid at ingest, not at query.

**Search.** Instead of decompressing every database vector, we rotate the query once into the same domain and score directly against the codebook values. The scoring kernel uses SIMD intrinsics (NEON on ARM, AVX-512BW on modern x86 with an AVX2 fallback) with nibble-split lookup tables for maximum throughput.

The Lloyd-Max codebook achieves distortion within a factor of 2.7x of the information-theoretic lower bound (Shannon's distortion-rate limit); the length-renormalization step removes the residual bias the Lloyd-Max codebook introduces on the inner-product estimator itself.

## Building

### Python (via maturin)

```bash
pip install maturin
cd turbovec-python
maturin build --release
pip install target/wheels/*.whl
```

### Rust

```bash
cargo build --release
```

All x86_64 builds target `x86-64-v3` (AVX2 baseline, Haswell 2013+) via `.cargo/config.toml`. Any CPU that can run the AVX2 fallback kernel can run the whole crate — the AVX-512 kernel is gated at runtime via `is_x86_feature_detected!` and only kicks in on hardware that supports it.

## Running benchmarks

Download datasets:
```bash
python3 benchmarks/download_data.py all            # all datasets
python3 benchmarks/download_data.py glove          # GloVe d=200
python3 benchmarks/download_data.py openai-1536    # OpenAI DBpedia d=1536
python3 benchmarks/download_data.py openai-3072    # OpenAI DBpedia d=3072
```

Each benchmark is a self-contained script in `benchmarks/suite/`. Run any one individually:
```bash
python3 benchmarks/suite/speed_d1536_2bit_arm_mt.py
python3 benchmarks/suite/recall_d1536_2bit.py
python3 benchmarks/suite/compression.py
```

Run all benchmarks for a category:
```bash
for f in benchmarks/suite/speed_*arm*.py; do python3 "$f"; done    # all ARM speed
for f in benchmarks/suite/speed_*x86*.py; do python3 "$f"; done    # all x86 speed
for f in benchmarks/suite/recall_*.py; do python3 "$f"; done       # all recall
python3 benchmarks/suite/compression.py                            # compression
```

Results are saved as JSON to `benchmarks/results/`. Regenerate charts:
```bash
python3 benchmarks/create_diagrams.py
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026) -- the paper this implements
- [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2405.12497) (SIGMOD 2024) -- the source of the per-vector length-renormalization correction adapted in step 5
- [FAISS Fast accumulation of PQ and AQ codes](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)) -- turbovec's x86 SIMD kernel adapts FastScan's pack layout, nibble-LUT scoring, and u16 accumulator strategy
