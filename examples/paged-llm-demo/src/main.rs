//! Paged-LLM Demo: Educational KV-cache simulation.
//!
//! Simulates a batch of 8 sequences with varying lengths (4–20 tokens each)
//! being decoded token-by-token.  For each batch we run four caches in
//! parallel so we can compare them side-by-side:
//!
//!   1. **ContiguousKvCache**  (fp32 K/V)  — naive pre-allocated baseline
//!   2. **PagedKvCache**       (fp32 K/V)  — PagedAttention-style paged pool
//!   3. **ContiguousKvCache**  (int8 K/V)  — as (1) but K/V quantized to int8
//!   4. **PagedKvCache**       (int8 K/V)  — as (2) but K/V quantized to int8
//!
//! After the decode loop the demo prints:
//!   • A memory-stats table: allocated vs used bytes for all four caches.
//!   • Per-step cosine similarities (fp32 vs int8 KV) and their mean.
//!   • An assertion that mean_cos > 0.999 (self-check).
//!
//! Run with:
//!   cargo run -p paged-llm-demo --release

mod kv_cache;
mod quant;
mod transformer;

use kv_cache::{ContiguousKvCache, KvCache, PagedKvCache, PAGE_SIZE};
use transformer::{decode_step, AttentionWeights, D_MODEL, HEAD_DIM, N_HEADS};

// ─── Tiny inline xorshift64 RNG ───────────────────────────────────────────────

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f32(&mut self) -> f32 {
        (self.next() as f64 / u64::MAX as f64) as f32
    }
    fn rand_vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32() * 2.0 - 1.0).collect()
    }
}

// ─── Simulation parameters ────────────────────────────────────────────────────

const BATCH: usize = 8;
const MAX_SEQ_LEN: usize = 32;
// Each sequence generates a different number of tokens (4..20).
const SEQ_LENS: [usize; BATCH] = [4, 7, 12, 20, 6, 15, 8, 19];
// Pool must be large enough for all sequences at max_seq_len.
// pages_needed = ceil(max_seq_len / PAGE_SIZE) * BATCH
const POOL_PAGES: usize = (MAX_SEQ_LEN / PAGE_SIZE + 1) * BATCH + 4;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Paged-LLM Demo  —  PagedAttention × INT8 KV quantization");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Config: BATCH={BATCH}, MAX_SEQ_LEN={MAX_SEQ_LEN}");
    println!("        D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, HEAD_DIM={HEAD_DIM}");
    println!("        PAGE_SIZE={PAGE_SIZE}, POOL_PAGES={POOL_PAGES}\n");

    // ── Initialise four caches ────────────────────────────────────────────────
    let mut cache_cont_f32 = ContiguousKvCache::new(BATCH, MAX_SEQ_LEN, N_HEADS, HEAD_DIM);
    let mut cache_paged_f32 = PagedKvCache::new(POOL_PAGES, BATCH, N_HEADS, HEAD_DIM);
    let mut cache_cont_i8  = ContiguousKvCache::new(BATCH, MAX_SEQ_LEN, N_HEADS, HEAD_DIM);
    let mut cache_paged_i8  = PagedKvCache::new(POOL_PAGES, BATCH, N_HEADS, HEAD_DIM);

    let weights = AttentionWeights::new_seeded(0xdeadbeef_cafebabe);
    let mut rng = Rng(0x1234567890abcdef);

    // cos-similarity of attention outputs: fp32 KV vs int8 KV
    let mut cos_sims: Vec<f32> = Vec::new();

    // ── Decode loop ───────────────────────────────────────────────────────────
    // Simulate token-by-token generation for the longest sequence, skipping
    // sequences that have already finished.
    let max_len = *SEQ_LENS.iter().max().unwrap();

    for step in 0..max_len {
        for seq in 0..BATCH {
            if step >= SEQ_LENS[seq] {
                continue; // this sequence already finished
            }
            let input = rng.rand_vec(D_MODEL);

            // fp32 path: contiguous cache; collect cos-sim against int8 path
            let step_out = decode_step(
                seq, &input, &weights,
                &mut cache_cont_f32,
                &mut cache_cont_i8,
            );
            cos_sims.push(step_out.cos_sim);

            // Also keep the paged caches in sync (same input, same weights).
            // Their sequence lengths and memory stats are what we display.
            let _step_paged = decode_step(
                seq, &input, &weights,
                &mut cache_paged_f32,
                &mut cache_paged_i8,
            );
        }
    }

    // ── Per-sequence stats ────────────────────────────────────────────────────
    println!("Per-sequence lengths (actual vs declared):");
    for seq in 0..BATCH {
        let len_cont  = cache_cont_f32.seq_len(seq);
        let len_paged = cache_paged_f32.seq_len(seq);
        println!(
            "  seq {seq}: declared={:2}  contiguous_len={:2}  paged_len={:2}",
            SEQ_LENS[seq], len_cont, len_paged
        );
    }
    println!();

    // ── Memory stats table ────────────────────────────────────────────────────
    let alloc_cf32 = cache_cont_f32.allocated_bytes(HEAD_DIM, N_HEADS);
    let alloc_pf32 = cache_paged_f32.allocated_bytes(HEAD_DIM, N_HEADS);
    let used_cf32  = cache_cont_f32.used_bytes(HEAD_DIM, N_HEADS);
    let used_pf32  = cache_paged_f32.used_bytes(HEAD_DIM, N_HEADS);

    fn pct(used: usize, alloc: usize) -> f32 {
        if alloc == 0 { 0.0 } else { used as f32 / alloc as f32 * 100.0 }
    }
    fn kb(b: usize) -> f32 { b as f32 / 1024.0 }

    println!("Memory utilisation (KV cache, both K and V, f32 storage):");
    println!(
        "  {:<26} {:>10}  {:>10}  {:>8}",
        "Cache", "Allocated", "Used", "Util%"
    );
    println!("  {}", "─".repeat(58));
    println!(
        "  {:<26} {:>10.2}  {:>10.2}  {:>7.1}%",
        "Contiguous fp32  (KB)",
        kb(alloc_cf32), kb(used_cf32), pct(used_cf32, alloc_cf32)
    );
    println!(
        "  {:<26} {:>10.2}  {:>10.2}  {:>7.1}%",
        "Paged      fp32  (KB)",
        kb(alloc_pf32), kb(used_pf32), pct(used_pf32, alloc_pf32)
    );
    println!();
    println!("  Contiguous wastes {:.1}% of allocated memory (pad to max_seq_len).",
        100.0 - pct(used_cf32, alloc_cf32));
    println!("  Paged pool allocates only pages that are actually filled.\n");

    // ── PagedAttention pool stats ─────────────────────────────────────────────
    let pages_used = cache_paged_f32.pool_pages() - cache_paged_f32.free_pages_remaining();
    println!("PagedAttention pool stats (fp32 paged cache):");
    println!("  Pool size   : {POOL_PAGES} pages × {PAGE_SIZE} tokens/page");
    println!("  Pages in use: {pages_used} / {POOL_PAGES}");
    println!("  Free pages  : {}", cache_paged_f32.free_pages_remaining());
    println!();

    // ── Theoretical INT8 savings ──────────────────────────────────────────────
    let total_tokens: usize = SEQ_LENS.iter().sum();
    let kv_dim = N_HEADS * HEAD_DIM;
    let f32_kv_bytes = total_tokens * kv_dim * 4 * 2; // ×2 for K and V
    let i8_kv_bytes  = total_tokens * (kv_dim + 4)   * 2;
    println!("Theoretical INT8 vs f32 KV memory (all sequences combined):");
    println!("  Total tokens decoded : {total_tokens}");
    println!("  f32 KV storage       : {:.2} KB", kb(f32_kv_bytes));
    println!("  int8 KV storage      : {:.2} KB  ({:.1}× reduction)",
        kb(i8_kv_bytes),
        f32_kv_bytes as f32 / i8_kv_bytes as f32);
    println!("  (int8 = 1 byte/dim + 4-byte scale per vector, vs 4 bytes/dim for f32)");
    println!();

    // ── Cosine similarity summary ─────────────────────────────────────────────
    println!("Attention output cosine similarity (fp32 KV vs int8 KV):");
    println!("  Steps measured : {}", cos_sims.len());
    let mean_cos = cos_sims.iter().sum::<f32>() / cos_sims.len() as f32;
    let min_cos  = cos_sims.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("  Mean cos-sim   : {mean_cos:.6}");
    println!("  Min  cos-sim   : {min_cos:.6}");

    assert!(
        mean_cos > 0.999,
        "INT8 KV accuracy regression: mean cos-sim {mean_cos:.6} <= 0.999"
    );
    println!("\n[OK] Self-check passed: mean cos-sim {mean_cos:.6} > 0.999");

    // ── Conceptual mapping to turbovec ───────────────────────────────────────
    println!();
    println!("Turbovec mapping:");
    println!("  PagedKvCache block table     <-> turbovec BLOCK=32 blocked layout");
    println!("  free() returns pages to pool <-> swap_remove() reclaims vector slot");
    println!("  INT8 KV store + per-token scale <-> turbovec RefineStore::Int8");
    println!("  on-demand page alloc in append() <-> pack::repack_range (tail-only repack)");
}
