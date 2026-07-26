//! WebAssembly bindings for turbovec.
//!
//! This crate compiles the **real** turbovec quantization + SIMD-fallback
//! search engine to `wasm32-unknown-unknown` and exposes a tiny JS-facing
//! API. The browser demo on GitHub Pages builds an index, adds vectors, and
//! searches — every number it shows is produced by the same Rust code that
//! runs natively, not a JavaScript re-implementation.
//!
//! Threads are unavailable in this target, so turbovec is built with
//! `default-features = false` (Rayon off); the sequential fallbacks in
//! `turbovec::par` produce bit-identical results.

use turbovec::{RefineMode, TurboQuantIndex};
use wasm_bindgen::prelude::*;

/// Improve panic messages in the browser console during development.
#[wasm_bindgen(start)]
pub fn init() {
    std::panic::set_hook(Box::new(|info| {
        // `console.error` via web-sys would be nicer, but keeping this crate
        // dependency-light: fall back to the default which wasm surfaces.
        let _ = info;
    }));
}

/// Result of a search: parallel arrays of candidate slot indices and scores.
#[wasm_bindgen]
pub struct WasmSearchResults {
    indices: Vec<i32>,
    scores: Vec<f32>,
}

#[wasm_bindgen]
impl WasmSearchResults {
    /// Candidate slot indices (row order = descending score). `-1` pads
    /// unfilled slots when fewer than `k` results exist.
    #[wasm_bindgen(getter)]
    pub fn indices(&self) -> Vec<i32> {
        self.indices.clone()
    }

    /// Scores aligned with `indices`. Exact inner products when a refine
    /// store is used, otherwise coarse quantized estimates.
    #[wasm_bindgen(getter)]
    pub fn scores(&self) -> Vec<f32> {
        self.scores.clone()
    }
}

/// A turbovec index, callable from JavaScript.
#[wasm_bindgen]
pub struct WasmIndex {
    inner: TurboQuantIndex,
    dim: usize,
    bit_width: usize,
    refine: Option<RefineMode>,
}

#[wasm_bindgen]
impl WasmIndex {
    /// Construct an index.
    ///
    /// * `dim` — vector dimensionality (positive multiple of 8).
    /// * `bit_width` — 2, 3, or 4.
    /// * `refine` — `undefined`/`"none"`, `"int8"`, `"float16"`, or `"float32"`.
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, bit_width: usize, refine: Option<String>) -> Result<WasmIndex, JsError> {
        let mode = match refine.as_deref() {
            None | Some("") | Some("none") => None,
            Some("int8") => Some(RefineMode::Int8),
            Some("float16") | Some("fp16") | Some("half") => Some(RefineMode::Float16),
            Some("float32") | Some("f32") => Some(RefineMode::Float32),
            Some(other) => {
                return Err(JsError::new(&format!(
                    "unknown refine mode {other:?} (use none/int8/float16/float32)"
                )))
            }
        };

        let inner = match mode {
            Some(m) => TurboQuantIndex::new_with_refine(dim, bit_width, m),
            None => TurboQuantIndex::new(dim, bit_width),
        }
        .map_err(|e| JsError::new(&format!("{e:?}")))?;

        Ok(WasmIndex { inner, dim, bit_width, refine: mode })
    }

    /// Add a flat row-major batch of `flat.len() / dim` vectors.
    pub fn add(&mut self, flat: &[f32]) -> Result<(), JsError> {
        if self.dim == 0 || flat.len() % self.dim != 0 {
            return Err(JsError::new(&format!(
                "input length {} is not a multiple of dim {}",
                flat.len(),
                self.dim
            )));
        }
        if let Some(bad) = flat.iter().position(|x| !x.is_finite() || x.abs() >= 1e16) {
            return Err(JsError::new(&format!(
                "input value at position {bad} is non-finite or out of range"
            )));
        }
        self.inner.add(flat);
        Ok(())
    }

    /// Search for the top-`k` neighbours of a single `query` vector.
    ///
    /// When `rerank_factor > 1` and the index has a refine store, runs the
    /// cascade re-rank; otherwise returns the coarse quantized top-k.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        rerank_factor: usize,
    ) -> Result<WasmSearchResults, JsError> {
        if query.len() != self.dim {
            return Err(JsError::new(&format!(
                "query length {} != dim {}",
                query.len(),
                self.dim
            )));
        }
        let res = if rerank_factor > 1 && self.inner.has_refine() {
            self.inner
                .search_with_rerank(query, k, rerank_factor)
                .map_err(|e| JsError::new(&format!("{e:?}")))?
        } else {
            self.inner.search(query, k)
        };
        Ok(WasmSearchResults {
            indices: res.indices.iter().map(|&i| i as i32).collect(),
            scores: res.scores.clone(),
        })
    }

    /// Number of vectors currently indexed.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Bytes used by the coarse quantized index (codes + per-vector scales).
    #[wasm_bindgen(js_name = coarseBytes)]
    pub fn coarse_bytes(&self) -> f64 {
        let n = self.inner.len() as f64;
        let bytes_per_row = ((self.dim * self.bit_width + 7) / 8) as f64;
        n * (bytes_per_row + 4.0) // + 4 bytes per-vector f32 scale
    }

    /// Extra bytes used by the optional refine store.
    #[wasm_bindgen(js_name = refineBytes)]
    pub fn refine_bytes(&self) -> f64 {
        let n = self.inner.len() as f64;
        let d = self.dim as f64;
        match self.refine {
            Some(RefineMode::Int8) => n * (d + 4.0),
            Some(RefineMode::Float16) => n * d * 2.0,
            Some(RefineMode::Float32) => n * d * 4.0,
            None => 0.0,
        }
    }

    /// Bytes a raw float32 store of the same vectors would use (the baseline).
    #[wasm_bindgen(js_name = fp32Bytes)]
    pub fn fp32_bytes(&self) -> f64 {
        self.inner.len() as f64 * self.dim as f64 * 4.0
    }
}
