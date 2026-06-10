//! KV-cache implementations: contiguous (naive) vs paged (PagedAttention-style).
//!
//! # What is a KV cache?
//!
//! In transformer attention, each token generates a key vector and a value
//! vector that subsequent tokens need to attend over.  Caching these (the
//! "KV cache") avoids recomputing them on every decode step.
//!
//! A typical naive implementation pre-allocates a contiguous buffer of
//! shape `[max_seq_len, n_heads, head_dim]` for each sequence.  This is
//! simple but wastes memory: most sequences finish long before `max_seq_len`.
//!
//! # PagedAttention (Kwon et al., 2023)
//!
//! vLLM's PagedAttention organises the KV cache as a global pool of small
//! fixed-size "pages" (blocks).  Each sequence has a block table — a
//! list of page IDs — and pages are allocated on demand.  When a sequence
//! ends, its pages are released back to the pool.
//!
//! Benefits:
//! - **Near-zero internal fragmentation** (at most one partial page per sequence).
//! - **No external fragmentation** (shared page pool, any sequence can use any page).
//! - High batch utilisation because memory from finished sequences is instantly reusable.
//!
//! # Connection to turbovec
//!
//! `turbovec`'s SIMD blocked layout is already paged: the full index is
//! divided into 32-vector blocks. Before this feature, every `add()` call
//! discarded and rebuilt the entire blocked layout (O(n·dim) work).
//! The `pack::repack_range` optimisation introduced in this PR keeps the
//! existing pages intact and only rebuilds the dirty tail blocks —
//! exactly the same insight as PagedAttention applied to ANN search.

pub const PAGE_SIZE: usize = 16; // tokens per page (vLLM default is 16)

// ─── shared types ─────────────────────────────────────────────────────────────

// ─── KvCache trait ────────────────────────────────────────────────────────────

/// Minimal interface shared by both cache implementations.
pub trait KvCache {
    /// Append one token's KV pair to the given sequence slot.
    fn append(&mut self, seq_id: usize, k: &[f32], v: &[f32]);
    /// Retrieve all keys for a sequence (flattened, row-major over tokens).
    fn keys(&self, seq_id: usize) -> Vec<f32>;
    /// Retrieve all values for a sequence.
    fn values(&self, seq_id: usize) -> Vec<f32>;
    /// Free all pages belonging to `seq_id`.
    #[allow(dead_code)]
    fn free(&mut self, seq_id: usize);
    /// Current number of tokens stored for `seq_id`.
    fn seq_len(&self, seq_id: usize) -> usize;
    /// Total allocated memory in bytes.
    fn allocated_bytes(&self, head_dim: usize, n_heads: usize) -> usize;
    /// Total used memory in bytes.
    fn used_bytes(&self, head_dim: usize, n_heads: usize) -> usize;
}

// ─── ContiguousKvCache ────────────────────────────────────────────────────────

/// Naive pre-allocated KV cache.
///
/// Each sequence gets a contiguous buffer of `max_seq_len × kv_dim` floats
/// for K and the same for V, allocated upfront regardless of actual length.
/// This is the baseline prior to PagedAttention.
pub struct ContiguousKvCache {
    max_seq_len: usize,
    kv_dim: usize, // n_heads * head_dim
    /// keys[seq_id] = pre-allocated Vec of length max_seq_len * kv_dim.
    keys: Vec<Option<Vec<f32>>>,
    vals: Vec<Option<Vec<f32>>>,
    /// Logical length (number of tokens appended) per sequence.
    lens: Vec<usize>,
}

impl ContiguousKvCache {
    pub fn new(max_batch: usize, max_seq_len: usize, n_heads: usize, head_dim: usize) -> Self {
        let kv_dim = n_heads * head_dim;
        Self {
            max_seq_len,
            kv_dim,
            keys: (0..max_batch).map(|_| Some(vec![0.0f32; max_seq_len * kv_dim])).collect(),
            vals: (0..max_batch).map(|_| Some(vec![0.0f32; max_seq_len * kv_dim])).collect(),
            lens: vec![0; max_batch],
        }
    }
}

impl KvCache for ContiguousKvCache {
    fn append(&mut self, seq_id: usize, k: &[f32], v: &[f32]) {
        let t = self.lens[seq_id];
        assert!(t < self.max_seq_len, "sequence {seq_id} exceeded max_seq_len");
        let off = t * self.kv_dim;
        if let Some(buf) = &mut self.keys[seq_id] {
            buf[off..off + self.kv_dim].copy_from_slice(k);
        }
        if let Some(buf) = &mut self.vals[seq_id] {
            buf[off..off + self.kv_dim].copy_from_slice(v);
        }
        self.lens[seq_id] += 1;
    }

    fn keys(&self, seq_id: usize) -> Vec<f32> {
        let t = self.lens[seq_id];
        self.keys[seq_id]
            .as_ref()
            .map(|b| b[..t * self.kv_dim].to_vec())
            .unwrap_or_default()
    }

    fn values(&self, seq_id: usize) -> Vec<f32> {
        let t = self.lens[seq_id];
        self.vals[seq_id]
            .as_ref()
            .map(|b| b[..t * self.kv_dim].to_vec())
            .unwrap_or_default()
    }

    fn free(&mut self, seq_id: usize) {
        // Keep the allocation (it was pre-paid) but zero the logical length.
        // In a real system we'd return the buffer to a pool or track freed
        // slots; here we just reset the counter.
        self.lens[seq_id] = 0;
    }

    fn seq_len(&self, seq_id: usize) -> usize {
        self.lens[seq_id]
    }

    fn allocated_bytes(&self, head_dim: usize, n_heads: usize) -> usize {
        let kv_dim = n_heads * head_dim;
        let active = self.keys.iter().filter(|s| s.is_some()).count();
        // Each active sequence occupies 2 (K+V) × max_seq_len × kv_dim × 4 bytes.
        active * 2 * self.max_seq_len * kv_dim * 4
    }

    fn used_bytes(&self, head_dim: usize, n_heads: usize) -> usize {
        let kv_dim = n_heads * head_dim;
        self.lens.iter().map(|&t| 2 * t * kv_dim * 4).sum()
    }
}

// ─── PagedKvCache ─────────────────────────────────────────────────────────────

/// PagedAttention-style KV cache.
///
/// The global pool holds `pool_pages` pages.  Each page stores
/// `PAGE_SIZE` tokens × kv_dim floats for K, and the same for V.
///
/// Each sequence has a *block table*: a `Vec<usize>` of page IDs.
/// Pages are allocated from a free-list on demand and released when a
/// sequence ends.
pub struct PagedKvCache {
    kv_dim: usize,
    /// Flat pool: pool_k[page_id * PAGE_SIZE * kv_dim .. ] for K data.
    pool_k: Vec<f32>,
    pool_v: Vec<f32>,
    /// Free-list of available page IDs.
    free_pages: Vec<usize>,
    /// block_table[seq_id] = ordered list of page IDs assigned to that sequence.
    block_table: Vec<Vec<usize>>,
    /// Logical token count per sequence.
    lens: Vec<usize>,
    /// Total pool capacity in pages (for stats).
    pool_pages: usize,
}

impl PagedKvCache {
    pub fn new(pool_pages: usize, max_batch: usize, n_heads: usize, head_dim: usize) -> Self {
        let kv_dim = n_heads * head_dim;
        Self {
            kv_dim,
            pool_k: vec![0.0f32; pool_pages * PAGE_SIZE * kv_dim],
            pool_v: vec![0.0f32; pool_pages * PAGE_SIZE * kv_dim],
            free_pages: (0..pool_pages).rev().collect(), // stack: pop cheaply
            block_table: vec![Vec::new(); max_batch],
            lens: vec![0; max_batch],
            pool_pages,
        }
    }

    fn alloc_page(&mut self) -> usize {
        self.free_pages.pop().expect("KV cache pool exhausted")
    }

    fn page_offset(&self, page_id: usize, slot: usize) -> usize {
        (page_id * PAGE_SIZE + slot) * self.kv_dim
    }

    pub fn free_pages_remaining(&self) -> usize {
        self.free_pages.len()
    }

    #[allow(dead_code)]
    pub fn pool_pages(&self) -> usize {
        self.pool_pages
    }
}

impl KvCache for PagedKvCache {
    fn append(&mut self, seq_id: usize, k: &[f32], v: &[f32]) {
        let t = self.lens[seq_id];
        let page_slot = t % PAGE_SIZE;

        // Allocate a new page when we're at the start of a page slot.
        if page_slot == 0 {
            let page_id = self.alloc_page();
            self.block_table[seq_id].push(page_id);
        }

        let page_id = *self.block_table[seq_id].last().unwrap();
        let off = self.page_offset(page_id, page_slot);
        self.pool_k[off..off + self.kv_dim].copy_from_slice(k);
        self.pool_v[off..off + self.kv_dim].copy_from_slice(v);
        self.lens[seq_id] += 1;
    }

    fn keys(&self, seq_id: usize) -> Vec<f32> {
        let t = self.lens[seq_id];
        let mut out = Vec::with_capacity(t * self.kv_dim);
        for tok in 0..t {
            let page_idx = tok / PAGE_SIZE;
            let page_slot = tok % PAGE_SIZE;
            let page_id = self.block_table[seq_id][page_idx];
            let off = self.page_offset(page_id, page_slot);
            out.extend_from_slice(&self.pool_k[off..off + self.kv_dim]);
        }
        out
    }

    fn values(&self, seq_id: usize) -> Vec<f32> {
        let t = self.lens[seq_id];
        let mut out = Vec::with_capacity(t * self.kv_dim);
        for tok in 0..t {
            let page_idx = tok / PAGE_SIZE;
            let page_slot = tok % PAGE_SIZE;
            let page_id = self.block_table[seq_id][page_idx];
            let off = self.page_offset(page_id, page_slot);
            out.extend_from_slice(&self.pool_v[off..off + self.kv_dim]);
        }
        out
    }

    fn free(&mut self, seq_id: usize) {
        // Return all pages belonging to this sequence to the free-list.
        let pages = std::mem::take(&mut self.block_table[seq_id]);
        self.free_pages.extend(pages);
        self.lens[seq_id] = 0;
    }

    fn seq_len(&self, seq_id: usize) -> usize {
        self.lens[seq_id]
    }

    fn allocated_bytes(&self, head_dim: usize, n_heads: usize) -> usize {
        let kv_dim = n_heads * head_dim;
        let pages_in_use = self.pool_pages - self.free_pages.len();
        pages_in_use * 2 * PAGE_SIZE * kv_dim * 4
    }

    fn used_bytes(&self, head_dim: usize, n_heads: usize) -> usize {
        let kv_dim = n_heads * head_dim;
        self.lens.iter().map(|&t| 2 * t * kv_dim * 4).sum()
    }
}
