//! Parallelism abstraction layer.
//!
//! The core crate uses Rayon data-parallelism on native targets (the
//! `parallel` feature, on by default). But some targets — most importantly
//! `wasm32-unknown-unknown`, which powers the in-browser demo — have no
//! threads and cannot link Rayon. Building with `--no-default-features`
//! turns Rayon off and swaps in the sequential shims below.
//!
//! The crate only ever reaches for four Rayon entry points:
//! `into_par_iter`, `par_iter_mut`, `par_chunks`, and `par_chunks_mut`.
//! Every downstream combinator it chains (`map`, `flat_map`, `zip`,
//! `enumerate`, `for_each`, `collect`, `unzip`, …) also exists on the
//! standard-library iterators, so the shims are exact drop-ins: the call
//! sites compile identically whether or not the feature is enabled.
//!
//! Correctness note: the sequential and parallel paths are numerically
//! identical because every parallel section here is embarrassingly parallel
//! (independent per-row / per-query work with no cross-item reduction), so
//! evaluation order never affects the result.

/// Re-export whichever prelude matches the active feature. Downstream code
/// writes `use crate::par::prelude::*;` and is agnostic to the backend.
#[cfg(feature = "parallel")]
pub mod prelude {
    pub use rayon::prelude::*;
}

#[cfg(not(feature = "parallel"))]
pub mod prelude {
    //! Sequential fallbacks that mirror the Rayon method names the crate
    //! uses, each backed by the equivalent `std` iterator.

    /// `into_par_iter()` → `into_iter()` for anything iterable.
    pub trait IntoParIterSeq: IntoIterator + Sized {
        #[inline]
        fn into_par_iter(self) -> <Self as IntoIterator>::IntoIter {
            self.into_iter()
        }
    }
    impl<T: IntoIterator> IntoParIterSeq for T {}

    /// Slice-level parallel entry points → their sequential `std` twins.
    pub trait SliceParSeq<T> {
        #[allow(dead_code)] // included for API parity; not all call sites use it
        fn par_iter(&self) -> core::slice::Iter<'_, T>;
        fn par_iter_mut(&mut self) -> core::slice::IterMut<'_, T>;
        fn par_chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, T>;
        fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T>;
    }

    impl<T> SliceParSeq<T> for [T] {
        #[inline]
        fn par_iter(&self) -> core::slice::Iter<'_, T> {
            self.iter()
        }
        #[inline]
        fn par_iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
            self.iter_mut()
        }
        #[inline]
        fn par_chunks(&self, chunk_size: usize) -> core::slice::Chunks<'_, T> {
            self.chunks(chunk_size)
        }
        #[inline]
        fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T> {
            self.chunks_mut(chunk_size)
        }
    }
}
