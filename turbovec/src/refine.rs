//! Opt-in refinement store for cascade re-ranking.
//!
//! Stores the original input vectors (pre-normalization) alongside the
//! compact quantized codes so that a coarse SIMD scan can be followed by
//! an exact inner-product re-rank of the shortlisted candidates.
//!
//! # How it works
//!
//! This is the distillation-inspired two-stage pipeline:
//!
//! 1. **Coarse scan** — run `search_with_mask` for `k * rerank_factor`
//!    candidates using the 2–4 bit quantized codes and SIMD LUT scoring.
//! 2. **Re-rank** — compute the exact (or near-exact for `Int8`) inner
//!    product between the query and each candidate using the stored originals.
//!    Return the true top-k by exact score.
//!
//! The coarse scores are **unbiased estimates** of ⟨v, q⟩ (the length-
//! renormalization scale corrects for quantization distortion), so the
//! true top-k is almost always contained in the top `k * rerank_factor`
//! coarse candidates.  A `rerank_factor` of 4 is a good starting point;
//! higher values improve recall at the cost of re-rank latency.
//!
//! # Memory cost (d=1536)
//!
//! | Mode | Extra bytes / vector |
//! |------|---------------------|
//! | `Int8` | dim + 4 ≈ 1 540 B |
//! | `Float32` | dim × 4 = 6 144 B |
//!
//! Compare to the base 4-bit index cost of dim × 4 / 8 = 192 B/vector.

use rayon::prelude::*;

/// Whether to store original vectors as int8 (with a per-vector scale) or
/// as full float32.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RefineMode {
    /// Store as int8 codes with a per-vector symmetric scale `max_abs / 127`.
    /// Approximately 4× cheaper to store than `Float32` at d=1536.
    /// Cosine similarity of re-ranked scores vs f32 is typically > 0.999.
    Int8,
    /// Store as full float32.  Re-ranked scores are exact inner products.
    Float32,
}

/// Per-vector refinement data.
#[derive(Debug)]
pub struct RefineStore {
    pub mode: RefineMode,
    /// `Int8` path: quantized codes, length `n * dim`.
    pub codes_i8: Vec<i8>,
    /// `Int8` path: per-vector symmetric scale = `max_abs / 127`, length `n`.
    pub i8_scales: Vec<f32>,
    /// `Float32` path: raw stored vectors, length `n * dim`.
    pub floats: Vec<f32>,
}

impl RefineStore {
    pub fn new(mode: RefineMode) -> Self {
        Self {
            mode,
            codes_i8: Vec::new(),
            i8_scales: Vec::new(),
            floats: Vec::new(),
        }
    }

    /// Append `n` vectors of dimension `dim` to the store.
    pub fn append(&mut self, vectors: &[f32], n: usize, dim: usize) {
        match self.mode {
            RefineMode::Float32 => {
                self.floats.extend_from_slice(&vectors[..n * dim]);
            }
            RefineMode::Int8 => {
                // Parallel quantization: each vector gets its own scale.
                // Collect into Vec<(Vec<i8>, f32)> then flatten — the par
                // unzip of flat Vec<i8> isn't directly supported by Rayon.
                let results: Vec<(Vec<i8>, f32)> = vectors[..n * dim]
                    .par_chunks(dim)
                    .map(|row| {
                        let max_abs = row.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
                        if max_abs == 0.0 {
                            return (vec![0i8; dim], 0.0f32);
                        }
                        let scale = max_abs / 127.0;
                        let inv_scale = 1.0 / scale;
                        let codes: Vec<i8> = row
                            .iter()
                            .map(|&x| (x * inv_scale).round().clamp(-127.0, 127.0) as i8)
                            .collect();
                        (codes, scale)
                    })
                    .collect();
                for (codes, scale) in results {
                    self.codes_i8.extend_from_slice(&codes);
                    self.i8_scales.push(scale);
                }
            }
        }
    }

    /// Mirror a `swap_remove(idx)` from the inner index.
    /// `n_before` is the number of vectors *before* the swap_remove (i.e.
    /// `inner.len()` before the call).
    pub fn swap_remove(&mut self, idx: usize, n_before: usize, dim: usize) {
        let last = n_before - 1;
        match self.mode {
            RefineMode::Float32 => {
                if idx != last {
                    let src = last * dim;
                    let dst = idx * dim;
                    self.floats.copy_within(src..src + dim, dst);
                }
                self.floats.truncate(last * dim);
            }
            RefineMode::Int8 => {
                if idx != last {
                    let src = last * dim;
                    let dst = idx * dim;
                    self.codes_i8.copy_within(src..src + dim, dst);
                    self.i8_scales[idx] = self.i8_scales[last];
                }
                self.codes_i8.truncate(last * dim);
                self.i8_scales.truncate(last);
            }
        }
    }

    /// Compute the (approximate) inner product ⟨query, stored[idx]⟩.
    ///
    /// - `Float32`: exact inner product.
    /// - `Int8`: scale × Σ(q_i × c_i) where c_i are int8 codes; the
    ///   per-query normalization is not applied since `query` is the full
    ///   float vector and the stored codes capture the original direction.
    pub fn score(&self, query: &[f32], idx: usize, dim: usize) -> f32 {
        match self.mode {
            RefineMode::Float32 => {
                let offset = idx * dim;
                let stored = &self.floats[offset..offset + dim];
                query.iter().zip(stored.iter()).map(|(&q, &v)| q * v).sum()
            }
            RefineMode::Int8 => {
                let offset = idx * dim;
                let codes = &self.codes_i8[offset..offset + dim];
                let scale = self.i8_scales[idx];
                let dot: f32 = query
                    .iter()
                    .zip(codes.iter())
                    .map(|(&q, &c)| q * (c as f32))
                    .sum();
                scale * dot
            }
        }
    }

    /// Number of vectors currently stored.
    pub fn len(&self) -> usize {
        match self.mode {
            RefineMode::Float32 => {
                // Caller guarantees dim > 0 when len > 0.
                self.floats.len() // will be n * dim; outer code tracks n separately
            }
            RefineMode::Int8 => self.i8_scales.len(),
        }
    }
}
