# turbovec — Claude Code guide

## What this is

A Rust workspace implementing the **TurboQuant** algorithm: a data-oblivious
2–4 bit vector quantizer + SIMD approximate nearest-neighbour search library.
The typical use-case is compressing and searching embedding vectors for
RAG / semantic search, achieving 8–16× compression with near-optimal recall.

Two crates, one example, benchmark scripts:

```
turbovec/           # Core Rust library (the algorithm)
turbovec-python/    # PyO3 bindings + LangChain/LlamaIndex/Haystack/Agno integrations
examples/           # Runnable demos
benchmarks/suite/   # Python recall + speed benchmark scripts
```

## Build & test commands

```bash
# Rust — always run in release; debug is ~30x slower for SIMD workloads
cargo test --release

# Python bindings (requires maturin + a virtualenv)
cd turbovec-python
maturin develop --release
pytest tests/

# Example crates
cargo run -p paged-llm-demo --release      # educational LLM demo
cd examples/downstream-smoke && cargo run --release  # excluded from workspace

# Benchmarks (Python scripts, need turbovec-python installed)
python3 benchmarks/suite/recall_d1536_4bit.py
python3 benchmarks/suite/incremental_add.py
```

## Module map (turbovec/src/)

| File | Responsibility |
|---|---|
| `lib.rs` | Public API: `TurboQuantIndex`, `SearchResults`, `add`/`search`/`write`/`load`, `prepare`, `swap_remove`, `from_parts` |
| `encode.rs` | Encode pipeline: normalize → seeded-QR rotate → TQ+ calibrate → Lloyd-Max quantize → bit-plane pack |
| `search.rs` | SIMD LUT scoring kernels (NEON/AVX2/AVX-512BW), top-k heaps, mask filtering |
| `pack.rs` | Bit-plane rows → SIMD-blocked layout; `repack` / `repack_range` |
| `codebook.rs` | Lloyd-Max optimal quantizer over Beta(α,α) marginals |
| `rotation.rs` | Seeded QR-decomposition for the orthogonal random rotation matrix |
| `id_map.rs` | `IdMapIndex`: stable u64-ID wrapper over `TurboQuantIndex` |
| `io.rs` | `.tv` / `.tvim` file format (v2/v3/v4), read/write with version dispatch |
| `error.rs` | Typed errors: `AddError`, `ConstructError`, `RerankError` |
| `refine.rs` | Opt-in refinement store for cascade re-ranking (`RefineMode`, `RefineStore`) |

## Gotchas

**SIMD kernels are arch-specific and must stay numerically bit-identical.**
`search.rs` has three paths: NEON (aarch64), AVX2 (x86_64), AVX-512BW (x86_64
runtime-dispatched). All share `max_lut = 127` and matching FMA flush order
(`FLUSH_EVERY = 256`). Preserve these constants when editing — they prevent
u16 overflow and keep scores identical across architectures. NEON cannot be
tested in this CI environment (Linux x86_64); any changes to SIMD code must
be verified on an ARM machine.

**Blocked layout cache invalidation.**
`TurboQuantIndex` holds a lazy `OnceLock<BlockedCache>`. After `add`, the
cache is updated in-place (incremental repack from the first dirty block),
so only tail blocks are rebuilt. `swap_remove` fully invalidates it (it
reorders vectors non-deterministically). `search` takes `&self` and is
safe for concurrent access; `prepare()` warms caches eagerly before serving
traffic.

**TQ+ calibration is frozen by the first non-empty add.**
The constant `TQPLUS_MIN_SAMPLES` (1000) gates fitting. Once set,
`tqplus_shift`/`tqplus_scale` are never changed — all subsequent adds reuse
them. An empty first add is a no-op (deliberately); the calibration is only
set when the first real batch arrives. Loading a pre-TQ+ (v2) file produces
explicit identity calibration (`shift=0`, `scale=1`) so subsequent adds
behave consistently.

**dim must be a positive multiple of 8; bit_width ∈ {2, 3, 4}.**
Checked in constructors (`ConstructError`) and on lazy-first-add (`AddError`).

**Input validation via `first_invalid_coord`.**
Non-finite or `|v| >= 1e16` values panic on `add`/`search` and return
`AddError::InvalidInputValue` from `add_2d`. Validate untrusted input before
calling — the Python binding converts these panics to `ValueError`.

**File format versioning (`io.rs`).**
v2 = no TQ+; v3 = adds TQ+ calibration trailer; v4 = adds optional refine
store trailer. v4 is only written when a `RefineStore` is present; otherwise
the file is written as v3 so old readers are unaffected. v1 (no magic) is
refused with a rebuild hint. Add new version handling in `read_core_versioned`.

**`pack::repack_3bit` is dead code.**
3-bit search currently goes through the generic 4-bit nibble path. The
function is defined but has no callers — don't delete it (may be needed for a
future dedicated 3-bit kernel), but don't be surprised when a linter flags it.

**`.cargo/config.toml` pins `target-cpu=x86-64-v3`.**
This enables AVX2 as the baseline on x86. Do not raise to x86-64-v4 (AVX-512
baseline); that is commented in the file. AVX-512 is used via runtime
`is_x86_feature_detected!` dispatch, not a compile-time target requirement.

**`examples/downstream-smoke` is excluded from the workspace.**
Its `Cargo.toml` has no `[workspace]` table so it intentionally acts as an
independent downstream consumer of turbovec. It is NOT a member of the root
workspace, cannot be addressed with `-p`, and has its own lock file. Build it
with `cd examples/downstream-smoke && cargo run --release`.

## Rerank / refine store (opt-in, default untouched)

`TurboQuantIndex::new_with_refine(dim, bit_width, RefineMode::Int8)` or
`::Float32` enables an optional secondary store of the original input vectors
(pre-normalization). After construction, `search_with_rerank(queries, k,
rerank_factor)` does a coarse 2/4-bit scan for `k * rerank_factor` candidates
then re-scores them with the stored originals and returns exact inner-product
top-k. Files containing a refine store are written as v4; default users keep
producing v3.
