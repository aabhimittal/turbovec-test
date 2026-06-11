//! Toy single-layer decoder-only transformer with multi-head attention (MHA).
//!
//! # Architecture
//!
//! d_model = 64, n_heads = 4, head_dim = 16
//!
//! Each decode step:
//!   1. Project input x ∈ ℝ^d_model  →  Q, K, V ∈ ℝ^{n_heads × head_dim}
//!      via random (but fixed) weight matrices (stand-ins for learned weights).
//!   2. Append (K, V) to the KV cache for the active sequence.
//!   3. Retrieve cached K, V for all past tokens (the "attend-over" window).
//!   4. Scaled dot-product attention:
//!        scores_i = Q · K_i / √head_dim    (for past token i)
//!        α        = softmax(scores)         (causal: only past tokens)
//!        out_head = Σ α_i V_i
//!   5. Concatenate all head outputs → d_model output vector.
//!
//! This is intentionally skeletal — no feed-forward layer, no learned weights,
//! no positional embedding — the goal is to show KV-cache behaviour, not to
//! produce meaningful outputs.
//!
//! # Connection to real LLMs
//!
//! In a real decoder (GPT-2, LLaMA, etc.):
//!  • Steps 1-5 repeat N_layers times, each with its own KV cache.
//!  • The KV cache grows one token per decode step; with batched inference
//!    the cache can be the dominant memory consumer.
//!  • PagedAttention (vLLM) replaces the per-sequence contiguous buffer with
//!    a global paged pool — exactly what `PagedKvCache` models here.

use crate::kv_cache::KvCache;
use crate::quant::{cosine_sim, QuantVec};

pub const D_MODEL: usize = 64;
pub const N_HEADS: usize = 4;
pub const HEAD_DIM: usize = D_MODEL / N_HEADS; // 16

/// Minimal weight matrices for a single attention layer.
/// Weights are deterministic pseudo-random (seeded with xorshift) so results
/// are reproducible without any external RNG dependency.
pub struct AttentionWeights {
    pub wq: Vec<f32>, // d_model × d_model  (row-major: row = output dim)
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
}

impl AttentionWeights {
    pub fn new_seeded(seed: u64) -> Self {
        let n = D_MODEL * D_MODEL;
        let mut state = seed;
        let gen = |s: &mut u64| -> f32 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            ((*s as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
        };
        let wq: Vec<f32> = (0..n).map(|_| gen(&mut state)).collect();
        let wk: Vec<f32> = (0..n).map(|_| gen(&mut state)).collect();
        let wv: Vec<f32> = (0..n).map(|_| gen(&mut state)).collect();
        Self { wq, wk, wv }
    }

    /// Project x ∈ ℝ^D_MODEL → out ∈ ℝ^D_MODEL via weight matrix W (row-major).
    pub fn project(w: &[f32], x: &[f32]) -> Vec<f32> {
        let d = D_MODEL;
        (0..d).map(|i| {
            let row = &w[i * d..(i + 1) * d];
            row.iter().zip(x).map(|(&w, &x)| w * x).sum()
        }).collect()
    }
}

/// Result of one decode step: output vector + cosine similarity between fp32
/// and int8-KV attention outputs (for the quant accuracy demo).
pub struct StepOutput {
    #[allow(dead_code)]
    pub out_f32: Vec<f32>,
    #[allow(dead_code)]
    pub out_i8: Vec<f32>,
    pub cos_sim: f32,
}

/// Run one decode step for sequence `seq_id`, appending to `cache_f32` and
/// `cache_i8` simultaneously so we can compare their attention outputs.
///
/// `input` is the current token embedding (d_model floats).
///
/// Returns the fp32 attention output, the int8-KV attention output, and their
/// cosine similarity.
pub fn decode_step(
    seq_id: usize,
    input: &[f32],
    weights: &AttentionWeights,
    cache_f32: &mut dyn KvCache,
    cache_i8: &mut dyn KvCache,
) -> StepOutput {
    // 1. Project to Q, K, V (shared; same for both cache paths)
    let q_full = AttentionWeights::project(&weights.wq, input);
    let k_full = AttentionWeights::project(&weights.wk, input);
    let v_full = AttentionWeights::project(&weights.wv, input);

    // 2a. Append exact fp32 K,V to fp32 cache
    cache_f32.append(seq_id, &k_full, &v_full);

    // 2b. Quantize K,V to int8 and store the *dequantized* (approximate)
    //     versions in the int8 cache.  This mirrors llama.cpp's q8_0 KV
    //     cache: keys/values are stored quantized, decoded on the fly during
    //     attention.  We store dequantized f32 here so the KvCache trait stays
    //     simple (no separate quant storage layer).
    let k_q = QuantVec::quantize(&k_full);
    let v_q = QuantVec::quantize(&v_full);
    let k_approx = k_q.dequantize();
    let v_approx = v_q.dequantize();
    cache_i8.append(seq_id, &k_approx, &v_approx);

    // 3. Retrieve full K,V histories for this sequence
    let keys_f32 = cache_f32.keys(seq_id);
    let vals_f32 = cache_f32.values(seq_id);
    let keys_i8  = cache_i8.keys(seq_id);
    let vals_i8  = cache_i8.values(seq_id);

    // 4. Multi-head scaled dot-product attention
    let out_f32 = mha_attention(&q_full, &keys_f32, &vals_f32);
    let out_i8  = mha_attention(&q_full, &keys_i8,  &vals_i8);

    let cos = cosine_sim(&out_f32, &out_i8);

    StepOutput { out_f32, out_i8, cos_sim: cos }
}

/// Scaled dot-product MHA over the full key/value history.
///
/// q      : current query,  d_model floats
/// keys   : all past keys,  n_tokens × d_model floats (row-major)
/// values : all past values, n_tokens × d_model floats
///
/// Returns: concatenated head outputs, d_model floats.
fn mha_attention(q: &[f32], keys: &[f32], values: &[f32]) -> Vec<f32> {
    let d = D_MODEL;
    let n_tok = keys.len() / d;
    assert_eq!(values.len(), n_tok * d);
    let scale = (HEAD_DIM as f32).sqrt().recip();
    let mut out = vec![0.0f32; d];

    for h in 0..N_HEADS {
        let h_start = h * HEAD_DIM;

        // Query slice for this head
        let q_h = &q[h_start..h_start + HEAD_DIM];

        // Compute attention scores: Q_h · K_h_t / sqrt(head_dim) for each t
        let scores: Vec<f32> = (0..n_tok).map(|t| {
            let k_t = &keys[t * d + h_start..t * d + h_start + HEAD_DIM];
            q_h.iter().zip(k_t).map(|(&qi, &ki)| qi * ki).sum::<f32>() * scale
        }).collect();

        // Softmax (numerically stable)
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_s: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum_e: f32 = exp_s.iter().sum();
        let alpha: Vec<f32> = exp_s.iter().map(|&e| e / sum_e).collect();

        // Weighted sum of values
        for t in 0..n_tok {
            let v_t = &values[t * d + h_start..t * d + h_start + HEAD_DIM];
            for (i, &vi) in v_t.iter().enumerate() {
                out[h_start + i] += alpha[t] * vi;
            }
        }
    }
    out
}
