//! Bit-plane to SIMD-blocked layout repacking.
//!
//! Converts bit-plane packed codes into a layout optimised for SIMD scoring:
//! - x86: FAISS-style perm0-interleaved for AVX2 cross-lane compatibility
//! - ARM: Sequential layout for NEON

use crate::BLOCK;

/// Repack all bit-plane codes into SIMD-blocked layout.
/// Returns (blocked_codes, n_blocks).
///
/// Equivalent to `repack_range(packed_codes, n_vectors, bits, dim, 0)` —
/// provided as a convenience for full rebuilds (e.g. `from_parts`, `load`).
pub fn repack(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
) -> (Vec<u8>, usize) {
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let data = repack_range(packed_codes, n_vectors, bits, dim, 0);
    (data, n_blocks)
}

/// Repack bit-plane codes for blocks `[first_block, n_blocks)` only.
///
/// Used for incremental updates: after adding new vectors, only blocks from
/// `first_block = old_n_vectors / BLOCK` onward are dirty. Each block's
/// output is a pure function of the 32 vectors that make up that block, so
/// earlier blocks never need to be recomputed.
///
/// The returned `Vec<u8>` contains the SIMD-blocked bytes for those blocks
/// and should be appended to the existing blocked cache after truncating it
/// to `first_block * n_byte_groups * BLOCK` bytes.
pub fn repack_range(
    packed_codes: &[u8],
    n_vectors: usize,
    bits: usize,
    dim: usize,
    first_block: usize,
) -> Vec<u8> {
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;
    let bytes_per_row = bits * bytes_per_plane;

    let perm0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

    // Step 1: Extract packed nibble bytes for the range [first_block*BLOCK, n_vectors).
    // Index relative to the range start so `codes_flat[0]` is the first vector
    // in this range, avoiding allocation for the entire index on incremental adds.
    let range_start = first_block * BLOCK;
    let range_n = n_vectors.saturating_sub(range_start);
    if range_n == 0 {
        return Vec::new();
    }
    let range_blocks = n_blocks - first_block;
    let out_size = range_blocks * n_byte_groups * BLOCK;

    let mut codes_flat = vec![vec![0u8; n_byte_groups]; range_n];
    for rel_idx in 0..range_n {
        let vec_idx = range_start + rel_idx;
        for g in 0..n_byte_groups {
            let dim_start = g * codes_per_byte;
            let mut byte_val = 0u8;
            for c in 0..codes_per_byte {
                let j = dim_start + c;
                let byte_in_plane = j / 8;
                let bit_in_byte = 7 - (j % 8);
                let mask = 1u8 << bit_in_byte;

                let mut code = 0u8;
                for p in 0..bits {
                    let plane_byte = packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
                    if plane_byte & mask != 0 {
                        code |= 1 << p;
                    }
                }

                let shift = if bits == 3 {
                    (codes_per_byte - 1 - c) * 4
                } else {
                    (codes_per_byte - 1 - c) * bits
                };
                byte_val |= code << shift;
            }
            codes_flat[rel_idx][g] = byte_val;
        }
    }

    // Step 2: Pack into platform-specific layout for the range.
    pack_blocked_range(range_n, first_block, range_blocks, n_byte_groups, out_size, &codes_flat, &perm0)
}

#[cfg(target_arch = "x86_64")]
fn pack_blocked_range(
    range_n: usize,
    first_block: usize,
    range_blocks: usize,
    n_byte_groups: usize,
    out_size: usize,
    codes_flat: &[Vec<u8>],
    perm0: &[usize; 16],
) -> Vec<u8> {
    // FAISS layout: split each byte into hi/lo nibbles, interleave with perm0.
    // `codes_flat[i]` is relative to the range start; `i` is a relative index.
    let mut blocked = vec![0u8; out_size];
    for rel_block in 0..range_blocks {
        let block_idx = first_block + rel_block;
        let base_vec_abs = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (rel_block * n_byte_groups + g) * BLOCK;
            for j in 0..16 {
                let va_abs = base_vec_abs + perm0[j];
                let vb_abs = base_vec_abs + perm0[j] + 16;
                // Convert absolute indices to relative (within the range).
                let range_start = first_block * BLOCK;
                let va_rel = va_abs.wrapping_sub(range_start);
                let vb_rel = vb_abs.wrapping_sub(range_start);
                let ba = if va_rel < range_n { codes_flat[va_rel][g] } else { 0 };
                let bb = if vb_rel < range_n { codes_flat[vb_rel][g] } else { 0 };
                blocked[out_offset + j] = (ba >> 4) | ((bb >> 4) << 4);
                blocked[out_offset + 16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }
    }
    blocked
}

#[cfg(not(target_arch = "x86_64"))]
fn pack_blocked_range(
    range_n: usize,
    first_block: usize,
    range_blocks: usize,
    n_byte_groups: usize,
    out_size: usize,
    codes_flat: &[Vec<u8>],
    _perm0: &[usize; 16],
) -> Vec<u8> {
    // Sequential layout: each byte stored as-is, vectors in order.
    let range_start = first_block * BLOCK;
    let mut blocked = vec![0u8; out_size];
    for rel_block in 0..range_blocks {
        let block_idx = first_block + rel_block;
        let base_vec_abs = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (rel_block * n_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vi_abs = base_vec_abs + lane;
                let vi_rel = vi_abs.wrapping_sub(range_start);
                if vi_rel < range_n {
                    blocked[out_offset + lane] = codes_flat[vi_rel][g];
                }
            }
        }
    }
    blocked
}

/// Repack 3-bit codes into two blocked arrays:
/// - sub_codes: 2-bit nibble format from planes 0,1
/// - plane2: packed bits blocked by 32 vectors
pub fn repack_3bit(
    packed_codes: &[u8],
    n_vectors: usize,
    dim: usize,
) -> (Vec<u8>, Vec<u8>, usize) {
    let bytes_per_plane = dim / 8;
    let bytes_per_row = 3 * bytes_per_plane;
    let n_blocks = (n_vectors + BLOCK - 1) / BLOCK;

    let sub_byte_groups = dim / 4;
    let mut sub_codes = vec![0u8; n_blocks * sub_byte_groups * BLOCK];

    let plane2_byte_groups = bytes_per_plane;
    let mut plane2_blocked = vec![0u8; n_blocks * plane2_byte_groups * BLOCK];

    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;

        for g in 0..sub_byte_groups {
            let out_offset = (block_idx * sub_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n_vectors { continue; }

                let mut byte_val = 0u8;
                let dim_start = g * 4;
                for c in 0..4usize {
                    let j = dim_start + c;
                    let byte_in_plane = j / 8;
                    let bit_in_byte = 7 - (j % 8);
                    let mask = 1u8 << bit_in_byte;

                    let mut code = 0u8;
                    for p in 0..2usize {
                        let plane_byte = packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
                        if plane_byte & mask != 0 { code |= 1 << p; }
                    }
                    byte_val |= code << ((3 - c) * 2);
                }
                sub_codes[out_offset + lane] = byte_val;
            }
        }

        for g in 0..plane2_byte_groups {
            let out_offset = (block_idx * plane2_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n_vectors { continue; }
                plane2_blocked[out_offset + lane] = packed_codes[vec_idx * bytes_per_row + 2 * bytes_per_plane + g];
            }
        }
    }

    (sub_codes, plane2_blocked, n_blocks)
}
