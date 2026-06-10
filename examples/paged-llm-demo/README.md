# paged-llm-demo

An educational Rust crate showing how three LLM inference techniques — **PagedAttention**, **INT8 KV quantization**, and **cascade re-ranking (distillation)** — map to the optimizations in the `turbovec` vector-search library.

```
cargo run -p paged-llm-demo --release
```

No external dependencies. Runs in under one second.

---

## What the demo shows

A batch of 8 sequences is decoded token-by-token through a toy single-layer
transformer (d_model=64, 4 heads, head_dim=16). Four KV caches run in parallel:

| Cache | K/V storage | Memory style |
|-------|-------------|--------------|
| Contiguous fp32 | exact float32 | Pre-allocated: `max_seq_len × kv_dim` per sequence |
| Paged fp32 | exact float32 | Global pool of 16-token pages; allocated on demand |
| Contiguous int8 | INT8 + scale | Same pre-allocation, K/V dequantized on write |
| Paged int8 | INT8 + scale | Paged pool + INT8 quantization |

The demo prints memory utilisation, pool stats, and cosine similarity of
attention outputs (fp32 KV vs int8 KV), then asserts `mean_cos > 0.999`.

---

## Concept → implementation mapping

### PagedAttention (Kwon et al., 2023 / vLLM)

Real vLLM maintains a global pool of fixed-size "blocks" (default 16 tokens).
Each sequence has a *block table* — a list of block IDs. Pages are allocated
on demand; when a sequence finishes its pages return to the pool immediately.

`PagedKvCache` in `src/kv_cache.rs` is a direct model:

```
pool_k / pool_v   →  flat float buffer, pool_pages × PAGE_SIZE × kv_dim
free_pages        →  free-list (Vec<usize>); pop to alloc, push to free
block_table[seq]  →  Vec<usize> of page IDs, one per 16 tokens
```

**Turbovec parallel**: turbovec's blocked layout already pages vectors in
BLOCK=32 chunks. Before this PR, every `add()` discarded and rebuilt the
entire layout. `pack::repack_range` now keeps existing blocks intact and
only rebuilds the dirty tail blocks — the same insight as PagedAttention
applied to ANN-index maintenance.

---

### INT8 KV quantization (llama.cpp `q8_0`)

llama.cpp's `q8_0` format stores each 32-element chunk of a K/V vector as
8-bit integers with a single `f16` scale. The demo uses a per-vector
symmetric int8 scheme (identical to turbovec's `RefineStore::Int8`):

```
scale   = max(|x_i|) / 127
code_i  = round(x_i / scale)   →  int8
x̂_i    = code_i × scale        ←  dequantized
```

Memory reduction: `kv_dim + 4` bytes (int8 codes + f32 scale) vs
`kv_dim × 4` bytes for f32. At d_model=64 that is ~3.8×; at d_model=4096
(LLaMA-class) it approaches 4×.

**Turbovec parallel**: `turbovec::RefineStore::Int8` stores original
pre-quantization vectors in the same format for exact re-ranking after a
coarse SIMD scan. The per-vector scale and int8 codes are identical to the
KV quantization scheme above.

---

### Cascade re-ranking / distillation

In knowledge distillation a large "teacher" model guides a compact "student".
Applied to retrieval: the low-bit SIMD index acts as the student (fast, cheap,
approximate) and the stored original vectors act as the teacher (exact score).

vLLM's speculative decoding uses the same two-stage idea: a small draft model
proposes tokens; the large target model verifies them.

**Turbovec implementation** (`turbovec::search_with_rerank`):
1. Coarse scan: `k × rerank_factor` candidates from the 2–4-bit SIMD index.
2. Re-rank: exact inner product against `RefineStore` (Int8 or Float32) for
   the shortlisted candidates. Return true top-k by exact score.

Recall improvement: 2-bit index + int8 re-rank typically matches 4-bit recall
at lower memory cost.

---

## File map

| File | Role |
|------|------|
| `src/kv_cache.rs` | `KvCache` trait; `ContiguousKvCache`; `PagedKvCache` |
| `src/quant.rs` | `QuantVec` — symmetric int8 quantize/dequantize; `cosine_sim` |
| `src/transformer.rs` | `AttentionWeights`; `decode_step`; MHA attention kernel |
| `src/main.rs` | Simulation loop; memory stats table; cosine-sim summary |

---

## Key numbers from a typical run

```
Memory utilisation (KV cache, both K and V, f32 storage):
  Contiguous fp32:  128.00 KB allocated  /  45.50 KB used  →  35.5% util
  Paged fp32:        80.00 KB allocated  /  45.50 KB used  →  56.9% util

  Contiguous wastes 64.5% of allocated memory (pad to max_seq_len).
  Paged pool allocates only pages that are actually filled.

Theoretical INT8 vs f32 KV memory:
  f32 KV:   45.50 KB
  int8 KV:  12.09 KB  (3.8× reduction)

Attention output cos-sim (fp32 KV vs int8 KV):
  Mean: 0.999932   Min: 0.999528
```
