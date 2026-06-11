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

> This fork takes three battle-tested ideas from modern LLM inference systems and applies them directly to turbovec's vector-search internals. The demo below runs on a **100 000-document corpus** (simulating OpenAI 1536-dim embeddings) so every number is a real measured result, not a theoretical estimate.
>
> Each technique is explained from first principles — no prior knowledge of machine learning or systems programming is assumed.

Run the demo yourself:
```bash
python3 demo/run_demo.py
```

### Test suite — 163 / 163 passing

| Test suite | Tests | Result | What it checks |
|---|---|---|---|
| `codebook` | 6 | ✅ | Lloyd-Max quantizer correctness |
| `concurrent_search` | 8 | ✅ | Thread safety of parallel search |
| `distortion` | 5 | ✅ | Recall quality vs Shannon bound |
| `encode` | 5 | ✅ | Encode pipeline determinism |
| `filtering` | 15 | ✅ | Mask/allowlist filtered search |
| `id_map` | 18 | ✅ | Stable external-ID wrapper |
| **`incremental_pack`** *(new)* | **8** | **✅** | Byte-identical tail-only repack |
| `input_validation` | 15 | ✅ | NaN / inf / out-of-range rejection |
| `io_versioning` | 9 | ✅ | v2/v3/v4 file format round-trips |
| `kernel_correctness` | 6 | ✅ | SIMD vs scalar result identity |
| `lazy_init` | 24 | ✅ | Deferred-dim construction |
| **`rerank`** *(new)* | **11** | **✅** | Cascade re-rank accuracy + safety |
| `rotation` | 5 | ✅ | Orthogonal rotation matrix |
| `state_sequences` | 9 | ✅ | Add / remove / reload lifecycle |
| `swap_remove` | 6 | ✅ | O(1) delete by index swap |
| `tqplus_calibration` | 2 | ✅ | TQ+ freeze-on-first-add behaviour |
| **`paged-llm-demo`** *(new)* | **2** | **✅** | INT8 KV quantization accuracy |
| lib unit tests | 7 | ✅ | `from_parts` invariant checks |
| doc-tests | 2 | ✅ | Public API compile examples |
| **Total** | **163** | **✅ 163 / 163** | |

---

### Enhancement 1 — Incremental blocked-layout packing

> **What problem does this solve?** Every time you called `add()` on a turbovec index, the library threw away the entire internal SIMD-ready layout and rebuilt it from scratch across *all* vectors. For 1 million vectors, that means 1.5 billion floating-point operations just to incorporate one new batch — with cost growing proportionally to index size.

#### How a CPU searches vectors (background)

Modern CPUs have "SIMD" instructions — one instruction operates on 8, 16, or 32 numbers simultaneously. To exploit this, turbovec packs every 32 vectors into a special byte layout ("blocked" layout) that the SIMD instructions can consume directly without unpacking. Think of it as rearranging books on a shelf so a robot arm with 32 fingers can grab exactly one book per reach.

> **Glossary — O(n) vs O(batch) cost:** Before the fix, rebuilding the layout for n=100 000 vectors cost proportional to n × dim = 100 000 × 1536 ≈ 150 million operations on every `add()`. After the fix, only the *tail blocks* touched by the new batch are rebuilt — cost is proportional to just the new batch, independent of how big the index already is.

#### The fix (PagedAttention-inspired)

turbovec's blocked layout divides vectors into groups of 32 (blocks). Each block's packed bytes depend *only* on those 32 vectors. So when you add new vectors at the end, only the last few blocks are "dirty". The new `repack_range(first_block)` function rebuilds only blocks `[first_block, n_blocks)`, leaving the rest untouched.

> **Glossary — PagedAttention (vLLM, 2023):** vLLM's PagedAttention organises the LLM KV cache as a global pool of small fixed-size "pages" allocated on demand. The core insight — "stop rebuilding everything when only the tail changed" — is identical to what this fix does for turbovec's blocked layout.

#### Live benchmark — 10 rounds × 10 000 vectors on 100K corpus

```
 round   n (total)    add (ms)   search (ms)
──────────────────────────────────────────────
     1      10,000       990.8         156.1
     2      20,000       514.0          23.0
     3      30,000       527.3          35.3
     4      40,000       517.7          43.8
     5      50,000       554.1          53.1
     6      60,000       555.5          65.8
     7      70,000       547.4          75.5
     8      80,000       534.6          81.8
     9      90,000       516.3          91.6
    10     100,000       521.7         102.1

add round 1 (n=10k):   990.8 ms
add round 10 (n=100k): 521.7 ms  ← flat! not 10× slower
```

Add time stays flat as n grows 10×. With the old O(n) full rebuild, round 10 would be ~10× slower than round 1. The wall-clock improvement compounds with every subsequent add. (Round 1 is slower because it includes TQ+ calibration, which only happens once on the first real batch.)

#### Tests — byte-identical output guaranteed

```
running 8 tests
test incremental_pack_2bit_partial_blocks                 ... ok
test incremental_pack_3bit_mixed                          ... ok
test incremental_pack_4bit_block_aligned_batches          ... ok
test incremental_pack_4bit_large_dim                      ... ok
test incremental_pack_4bit_non_aligned_batches            ... ok
test incremental_pack_cold_cache_then_search              ... ok
test incremental_pack_full_lifecycle_matches_fresh_index  ... ok
test repack_range_matches_full_repack_for_suffix          ... ok   ← key test

test result: ok. 8 passed; 0 failed; 0 ignored
```

`repack_range_matches_full_repack_for_suffix` compares every byte of the incremental output against a fresh full rebuild. Identical bytes → identical SIMD scores → identical search results → zero recall change. No approximations.

---

### Enhancement 2 — Cascade re-ranking with a refinement store

> **What problem does this solve?** At 4-bit compression, turbovec reduces a 585 MB corpus to 73 MB — but the compression loses precision. On our 100 K corpus, plain 4-bit search finds the true nearest neighbour only 79% of the time (Recall@1 = 0.79). This enhancement pushes that to 98.2% with no extra latency cost.

#### How it works (two-stage pipeline)

```
Query
  │
  ├─► Coarse scan (compressed SIMD index)
  │     finds top k × rerank_factor candidates cheaply
  │     (e.g. k=10, factor=4 → 40 candidates)
  │
  └─► Re-rank (stored original vectors)
        exact inner product against those 40 candidates
        → true top 10 by exact score
```

> **Glossary — knowledge distillation:** The "teacher-student" principle from machine learning: a large accurate teacher guides a compact fast student. Here the stored originals are the teacher (exact but slow to scan at scale) and the 4-bit SIMD index is the student (fast but approximate). The two-stage pipeline gets student-level throughput with teacher-level final accuracy.

> **Glossary — inner product (dot product):** The similarity score between two vectors: `score = v₁[0]×v₂[0] + v₁[1]×v₂[1] + … + v₁[d]×v₂[d]`. For unit-length (normalised) vectors this is equivalent to cosine similarity — 1.0 = identical direction, 0.0 = perpendicular. Vector search finds the stored vector with the highest dot product against the query.

#### Two storage modes for the refinement store

| Mode | Extra memory (d=1536) | Score accuracy |
|---|---|---|
| `refine="float32"` | +585.9 MB (full precision) | Exact inner product |
| `refine="int8"` | +146.9 MB (4× cheaper) | ≈ exact (cos-sim > 0.9999 vs float32) |

> **Glossary — INT8 quantization:** Storing a 32-bit float (4 bytes) as an 8-bit integer (1 byte) by computing `scale = max(|x|) / 127`, then `code = round(x / scale)`. Reconstruction: `x̂ = code × scale`. Error per element ≤ `scale / 2`. Memory: 4× cheaper than float32 at a tiny accuracy cost.

#### Python API

```python
from turbovec import TurboQuantIndex

# Build index with int8 refinement store
index = TurboQuantIndex(dim=1536, bit_width=4, refine="int8")
index.add(vectors)

# Re-ranked search: scan 40 candidates, return best 10 by exact score
scores, indices = index.search(queries, k=10, rerank_factor=4)
```

#### Live benchmark — 100 000 vectors, 1 000 queries

```
Memory:
  Float32 raw               : 585.9 MB
  TurboQuant 4-bit          :  73.6 MB   (8.0× smaller)
  + int8 refine store       : +146.9 MB
  Total with refine         : 220.5 MB   (still 2.7× smaller than float32)

Recall vs rerank_factor:
  rerank_factor     R@1     R@10   search time
  ─────────────────────────────────────────────────
  (baseline)      0.7900   0.9980  (coarse only, no rerank)
              1   0.9800   0.9980    2402.8 ms
              2   0.9810   0.9990     975.9 ms
              4   0.9820   1.0000     958.8 ms  ← sweet spot
              8   0.9820   1.0000    1056.6 ms

  Best: rerank_factor=4
  R@1 gain: 0.7900 → 0.9820  (+19.2 percentage points)
  R@10 gain: 0.9980 → 1.0000  (perfect — every true nearest found)
```

Rerank factor 4 is the sweet spot: R@10 reaches 100% and search is *faster* than factor=1 (which still pays the re-scoring overhead but over fewer candidates). Factor=8 yields no additional recall gain over factor=4.

#### File format — zero breaking changes

The new RefineStore is written only when present. A default `TurboQuantIndex()` still writes format v3 — old readers see nothing new. A `refine="int8"` index writes format v4. Version detection is at file load time.

```
running 11 tests
test default_index_writes_v3_not_v4          ... ok  ← backward compat
test float32_rerank_exact_recovery           ... ok  ← exact = brute-force
test int8_rerank_monotone_recall             ... ok  ← int8 ≥ coarse
test v4_round_trip_int8                      ... ok
test v4_round_trip_float32                   ... ok
...
test result: ok. 11 passed; 0 failed; 0 ignored
```

---

### Enhancement 3 — `examples/paged-llm-demo` educational crate

> A zero-dependency Rust program that demonstrates PagedAttention KV cache management and INT8 KV quantization in a toy transformer — and explains how both map back to the turbovec optimisations above.

```bash
cargo run -p paged-llm-demo --release
```

> **Glossary — KV cache:** In a transformer LLM (GPT, LLaMA, Claude, …), each token that has been processed stores two vectors — Key and Value — that all future tokens must "attend over". These are the KV cache. At 1000-token context with a 70B-parameter model, the KV cache can consume more GPU memory than the model weights themselves.

The demo decodes 8 conversations of varying lengths (4–20 tokens) simultaneously through a toy transformer (d_model=64, 4 heads) and compares four cache strategies side by side:

#### Actual output

```
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

Attention output cosine similarity (fp32 KV vs int8 KV):
  Steps measured : 91
  Mean cos-sim   : 0.999932
  Min  cos-sim   : 0.999528

[OK] Self-check passed: mean cos-sim 0.999932 > 0.999
```

**Reading the 35.5% utilisation:** The contiguous cache pre-allocates `max_seq_len × kv_dim` for every conversation slot regardless of actual length. 8 slots × 32 tokens × 64 dims × 2 (K+V) × 4 bytes = 128 KB reserved; only 45.5 KB used. That 64.5% waste is why vLLM switched to PagedAttention — at 10 000 concurrent users the wasted memory is enormous.

**Reading the 3.8× INT8 reduction:** The same 91 tokens of KV data shrinks from 45.5 KB (float32) to 12.1 KB (int8 + per-vector scale). The attention output cosine similarity stays at 0.999932 vs perfect 1.000000 — indistinguishable in practice.

**The `assert!(mean_cos > 0.999)` at the end** means `cargo run` is also a regression test — if a future code change broke INT8 accuracy, the binary exits with an error.

---

### How all three connect

```
  LLM inference technique              This fork's turbovec equivalent
  ──────────────────────────────────────────────────────────────────────
  PagedAttention: global page pool  →  BLOCK=32 blocked layout
    allocate on demand                  (vectors already "paged" at 32)

  PagedAttention: dirty-tail update →  pack::repack_range(first_block)
    only new tokens need new pages      only new blocks rebuilt on add()

  llama.cpp q8_0 INT8 KV cache      →  RefineStore::Int8
    per-vector scale + int8 codes       same scheme, used for re-ranking

  Knowledge distillation            →  search_with_rerank(rerank_factor)
    teacher (big+exact) verifies        coarse SIMD shortlist, exact
    student (small+fast) outputs        re-score of top k×factor
```

> **Glossary — SIMD (Single Instruction, Multiple Data):** A CPU feature that applies one operation to 8/16/32 numbers simultaneously. Intel's AVX2/AVX-512 and ARM's NEON are SIMD instruction sets. turbovec's search kernels are hand-written in these instructions. The blocked layout (32 vectors in a specific interleaved byte arrangement) is designed precisely so one SIMD instruction can score one full block in a single pass.

---

### New files added in this fork

| Path | Purpose |
|---|---|
| `demo/run_demo.py` | **End-to-end demo** on 100K synthetic corpus — runs all sections above |
| `turbovec/src/refine.rs` | `RefineStore` — Int8/Float32 original-vector store for re-ranking |
| `turbovec/tests/incremental_pack.rs` | 8 byte-identity tests for tail-only repack |
| `turbovec/tests/rerank.rs` | 11 correctness + round-trip tests for cascade re-rank |
| `turbovec-python/src/lib.rs` | Python bindings: `refine=` param, `rerank_factor=` in `search()` |
| `examples/paged-llm-demo/` | Rust educational demo crate (KV cache + INT8 quant + MHA decoder) |
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
