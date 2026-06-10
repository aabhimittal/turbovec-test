//! INT8 KV-cache quantization, mirroring turbovec's `RefineStore::Int8`.
//!
//! Real systems (e.g. llama.cpp's `q8_0` KV cache) quantize each key / value
//! vector to 8-bit integers with a per-vector symmetric scale to halve KV
//! memory from 16-bit floats.  The idea maps exactly to turbovec's
//! `RefineStore` which stores original vectors as int8 for exact re-ranking.
//!
//! # Quantization scheme
//!
//! For a vector x of length d:
//!   scale = max(|x_i|) / 127          (symmetric, single scale per vector)
//!   code_i = round(x_i / scale)        → int8
//!   x̂_i   = code_i * scale             ← reconstructed
//!
//! The maximum absolute error per element is scale / 2 ≈ max_abs / 254.

/// A single quantized vector: 1 byte per dimension + 4-byte scale.
pub struct QuantVec {
    pub codes: Vec<i8>,
    pub scale: f32,
}

impl QuantVec {
    /// Quantize `v` into int8 codes with a symmetric per-vector scale.
    pub fn quantize(v: &[f32]) -> Self {
        let max_abs = v.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        if max_abs == 0.0 {
            return Self { codes: vec![0; v.len()], scale: 0.0 };
        }
        let scale = max_abs / 127.0;
        let inv = 1.0 / scale;
        let codes: Vec<i8> = v
            .iter()
            .map(|&x| (x * inv).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Self { codes, scale }
    }

    /// Reconstruct approximate float values from int8 codes.
    pub fn dequantize(&self) -> Vec<f32> {
        self.codes.iter().map(|&c| c as f32 * self.scale).collect()
    }
}

/// Compute cosine similarity between two equal-length float slices.
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Memory cost comparison (bytes) for a KV vector of length `dim` stored as
/// f32 vs int8 with a 4-byte scale.
#[allow(dead_code)]
pub fn memory_cost(dim: usize) -> (usize, usize) {
    let f32_bytes = dim * 4;
    let i8_bytes = dim + 4; // 1 byte/dim + 4-byte scale
    (f32_bytes, i8_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_cos_sim() {
        // A random-ish vector; cos-sim of reconstruction should be >0.999.
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37 - 8.0)).collect();
        let q = QuantVec::quantize(&v);
        let r = q.dequantize();
        assert!(cosine_sim(&v, &r) > 0.999, "cos-sim too low: {}", cosine_sim(&v, &r));
    }

    #[test]
    fn zero_vector() {
        let v = vec![0.0f32; 16];
        let q = QuantVec::quantize(&v);
        assert_eq!(q.scale, 0.0);
        assert!(q.dequantize().iter().all(|&x| x == 0.0));
    }
}
