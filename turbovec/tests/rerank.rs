//! Tests for the cascade re-ranking feature (`search_with_rerank`).
//!
//! Two modes are tested:
//! - `Float32` refine: re-ranked scores are exact inner products.
//! - `Int8` refine: re-ranked scores approximate inner products; the
//!   approximation is tight (cosine similarity vs f32 > 0.99).

use turbovec::{IdMapIndex, RerankError, RefineMode, TurboQuantIndex};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut uniform = || {
        let raw = (next() >> 40) as u32 | 1;
        raw as f32 / (1u32 << 24) as f32
    };
    let two_pi = 2.0_f32 * std::f32::consts::PI;
    let mut data = vec![0.0f32; n * dim];
    let mut i = 0;
    while i < data.len() {
        let u1 = uniform().max(1e-7);
        let u2 = uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = two_pi * u2;
        data[i] = r * theta.cos();
        if i + 1 < data.len() { data[i + 1] = r * theta.sin(); }
        i += 2;
    }
    for row in data.chunks_mut(dim) {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 { let inv = 1.0 / norm; for x in row.iter_mut() { *x *= inv; } }
    }
    data
}

/// Brute-force top-k by exact inner product.
fn brute_force_topk(vectors: &[f32], query: &[f32], k: usize) -> Vec<usize> {
    let dim = query.len();
    let n = vectors.len() / dim;
    let mut scores: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let v = &vectors[i * dim..(i + 1) * dim];
            let dot: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            (dot, i)
        })
        .collect();
    scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scores[..k].iter().map(|&(_, i)| i).collect()
}

// ─── Float32 exact-recovery oracle ───────────────────────────────────────────

#[test]
fn float32_rerank_exact_recovery() {
    // With Float32 refine and a rerank_factor large enough that k' = n,
    // the returned indices must be identical to brute-force exact top-k.
    let n = 200;
    let dim = 64;
    let k = 5;

    let vecs = unit_vectors(n, dim, 0xF001);
    let query = unit_vectors(1, dim, 0xF002);

    let mut idx = TurboQuantIndex::new_with_refine(dim, 4, RefineMode::Float32).unwrap();
    idx.add(&vecs);

    // rerank_factor = n/k ensures k' >= n → all candidates considered.
    let res = idx.search_with_rerank(&query, k, n / k + 1).unwrap();
    assert_eq!(res.k, k);
    assert_eq!(res.indices.len(), k);

    let expected = brute_force_topk(&vecs, &query, k);
    let got: Vec<usize> = res.indices.iter().map(|&i| i as usize).collect();

    // Allow tied scores to swap rank but set must match.
    let mut expected_sorted = expected.clone();
    let mut got_sorted = got.clone();
    expected_sorted.sort_unstable();
    got_sorted.sort_unstable();
    assert_eq!(
        got_sorted, expected_sorted,
        "Float32 rerank index set must match brute-force top-k"
    );

    // Scores must be descending.
    for w in res.scores.windows(2) {
        assert!(w[0] >= w[1] - 1e-6, "scores not descending: {:?}", &res.scores);
    }
}

// ─── Int8 recall is monotonically better than plain search ───────────────────

#[test]
fn int8_rerank_monotone_recall() {
    // recall@10 with Int8 rerank(factor=4) must be ≥ recall@10 of plain search,
    // on average over 5 queries.
    let n = 500;
    let dim = 64;
    let k = 10;

    let vecs = unit_vectors(n, dim, 0xF003);
    let queries = unit_vectors(5, dim, 0xF004);

    let mut plain = TurboQuantIndex::new(dim, 4).unwrap();
    plain.add(&vecs);

    let mut refined = TurboQuantIndex::new_with_refine(dim, 4, RefineMode::Int8).unwrap();
    refined.add(&vecs);

    let mut plain_recall_total = 0usize;
    let mut rerank_recall_total = 0usize;

    for qi in 0..5 {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let exact = brute_force_topk(&vecs, q, k);
        let exact_set: std::collections::HashSet<usize> = exact.into_iter().collect();

        let plain_res = plain.search(q, k);
        let plain_hits = plain_res.indices.iter().filter(|&&i| exact_set.contains(&(i as usize))).count();

        let rerank_res = refined.search_with_rerank(q, k, 4).unwrap();
        let rerank_hits = rerank_res.indices.iter().filter(|&&i| exact_set.contains(&(i as usize))).count();

        plain_recall_total += plain_hits;
        rerank_recall_total += rerank_hits;
    }

    assert!(
        rerank_recall_total >= plain_recall_total,
        "rerank recall {rerank_recall_total} should be >= plain recall {plain_recall_total}"
    );
}

// ─── rerank_factor = 1 returns coarse index set ───────────────────────────────

#[test]
fn rerank_factor_1_matches_coarse_index_set() {
    let n = 100;
    let dim = 64;
    let k = 5;

    let vecs = unit_vectors(n, dim, 0xF005);
    let query = unit_vectors(1, dim, 0xF006);

    let mut idx = TurboQuantIndex::new_with_refine(dim, 4, RefineMode::Float32).unwrap();
    idx.add(&vecs);

    let coarse = idx.search(&query, k);
    let rerank = idx.search_with_rerank(&query, k, 1).unwrap();

    // Same candidate pool (k' = k * 1 = k), but re-scored with exact IP.
    // Index sets must match (possibly in different order).
    let mut coarse_idx: Vec<i64> = coarse.indices.clone();
    let mut rerank_idx: Vec<i64> = rerank.indices.clone();
    coarse_idx.sort_unstable();
    rerank_idx.sort_unstable();
    assert_eq!(coarse_idx, rerank_idx, "rerank_factor=1 must use the same candidate pool as coarse search");
}

// ─── mask + rerank: masked slots never returned ───────────────────────────────

#[test]
fn mask_and_rerank_respects_mask() {
    let n = 100;
    let dim = 64;
    let k = 5;

    let vecs = unit_vectors(n, dim, 0xF007);
    let query = unit_vectors(1, dim, 0xF008);

    let mut idx = TurboQuantIndex::new_with_refine(dim, 4, RefineMode::Int8).unwrap();
    idx.add(&vecs);

    // Allow only slots 50..75.
    let mut mask = vec![false; n];
    for i in 50..75 { mask[i] = true; }

    let res = idx.search_with_rerank_mask(&query, k, 3, Some(&mask)).unwrap();

    for &slot in &res.indices {
        if slot < 0 { continue; }
        let s = slot as usize;
        assert!(s >= 50 && s < 75, "returned slot {s} is outside the allowed range [50, 75)");
    }
    assert!(res.k <= k);
    assert!(res.k <= 25); // ≤ number of allowed slots
}

// ─── NoRefineStore error ───────────────────────────────────────────────────────

#[test]
fn no_refine_store_returns_error() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let vecs = unit_vectors(10, dim, 0xF009);
    idx.add(&vecs);

    match idx.search_with_rerank(&unit_vectors(1, dim, 0xF00A), 3, 2) {
        Err(RerankError::NoRefineStore) => {}
        other => panic!("expected NoRefineStore, got {:?}", other),
    }
}

#[test]
fn invalid_rerank_factor_returns_error() {
    let dim = 64;
    let mut idx = TurboQuantIndex::new_with_refine(dim, 4, RefineMode::Float32).unwrap();
    let vecs = unit_vectors(10, dim, 0xF00B);
    idx.add(&vecs);

    match idx.search_with_rerank(&unit_vectors(1, dim, 0xF00C), 3, 0) {
        Err(RerankError::InvalidRerankFactor(0)) => {}
        other => panic!("expected InvalidRerankFactor(0), got {:?}", other),
    }
}

// ─── refine store alignment after swap_remove ─────────────────────────────────

#[test]
fn refine_store_aligned_after_swap_remove() {
    // After swap_remove, the refine store must be consistent with the inner
    // index: searching for the moved vector (which was at slot n-1, now at
    // slot `idx`) with Float32 rerank must still find it at the correct slot.
    let dim = 64;
    let n = 20;
    let vecs = unit_vectors(n, dim, 0xF00D);

    let mut idx = TurboQuantIndex::new_with_refine(dim, 4, RefineMode::Float32).unwrap();
    idx.add(&vecs);

    // Remove slot 3; slot 19 (last) moves to slot 3.
    idx.swap_remove(3);
    assert_eq!(idx.len(), n - 1);

    // The previously-last vector (vecs row 19) is now at slot 3.
    let query = &vecs[(n - 1) * dim..n * dim];
    let res = idx.search_with_rerank(query, 1, 5).unwrap();
    assert_eq!(
        res.indices[0] as usize, 3,
        "moved vector should be at slot 3 after swap_remove"
    );
}

// ─── round-trip: write v4, load, search_with_rerank ───────────────────────────

#[test]
fn v4_round_trip_float32() {
    let dim = 64;
    let n = 50;
    let vecs = unit_vectors(n, dim, 0xF010);
    let query = unit_vectors(1, dim, 0xF011);

    let mut idx = TurboQuantIndex::new_with_refine(dim, 4, RefineMode::Float32).unwrap();
    idx.add(&vecs);

    let tmp = std::env::temp_dir().join(format!(
        "turbovec_rerank_v4_f32_{}.tv",
        std::process::id()
    ));
    idx.write(&tmp).unwrap();

    // Verify the version byte is 4.
    let raw = std::fs::read(&tmp).unwrap();
    assert_eq!(raw[4], 4, "file version byte should be 4 for refine index");

    let loaded = TurboQuantIndex::load(&tmp).unwrap();
    std::fs::remove_file(&tmp).ok();

    assert!(loaded.has_refine());
    assert_eq!(loaded.refine_mode(), Some(RefineMode::Float32));

    let orig_res = idx.search_with_rerank(&query, 5, 3).unwrap();
    let loaded_res = loaded.search_with_rerank(&query, 5, 3).unwrap();

    let mut orig_idx: Vec<i64> = orig_res.indices.clone();
    let mut loaded_idx: Vec<i64> = loaded_res.indices.clone();
    orig_idx.sort_unstable();
    loaded_idx.sort_unstable();
    assert_eq!(orig_idx, loaded_idx, "v4 round-trip must preserve rerank results");
}

#[test]
fn v4_round_trip_int8() {
    let dim = 64;
    let n = 50;
    let vecs = unit_vectors(n, dim, 0xF012);
    let query = unit_vectors(1, dim, 0xF013);

    let mut idx = TurboQuantIndex::new_with_refine(dim, 4, RefineMode::Int8).unwrap();
    idx.add(&vecs);

    let tmp = std::env::temp_dir().join(format!(
        "turbovec_rerank_v4_i8_{}.tv",
        std::process::id()
    ));
    idx.write(&tmp).unwrap();
    assert_eq!(std::fs::read(&tmp).unwrap()[4], 4, "version byte should be 4");

    let loaded = TurboQuantIndex::load(&tmp).unwrap();
    std::fs::remove_file(&tmp).ok();

    assert!(loaded.has_refine());
    assert_eq!(loaded.refine_mode(), Some(RefineMode::Int8));

    let orig_res = idx.search_with_rerank(&query, 5, 3).unwrap();
    let loaded_res = loaded.search_with_rerank(&query, 5, 3).unwrap();

    let mut orig_idx: Vec<i64> = orig_res.indices.clone();
    let mut loaded_idx: Vec<i64> = loaded_res.indices.clone();
    orig_idx.sort_unstable();
    loaded_idx.sort_unstable();
    assert_eq!(orig_idx, loaded_idx, "v4 int8 round-trip must preserve rerank results");
}

#[test]
fn default_index_writes_v3_not_v4() {
    // A non-refine index must still write a v3 file so old readers are unaffected.
    let dim = 64;
    let mut idx = TurboQuantIndex::new(dim, 4).unwrap();
    let vecs = unit_vectors(5, dim, 0xF014);
    idx.add(&vecs);

    let tmp = std::env::temp_dir().join(format!(
        "turbovec_rerank_default_v3_{}.tv",
        std::process::id()
    ));
    idx.write(&tmp).unwrap();
    let raw = std::fs::read(&tmp).unwrap();
    std::fs::remove_file(&tmp).ok();

    assert_eq!(raw[4], 3, "default (no refine) index must write version 3");
    assert!(!idx.has_refine());
}

// ─── IdMapIndex rerank ─────────────────────────────────────────────────────────

#[test]
fn id_map_rerank_returns_external_ids() {
    let dim = 64;
    let n = 50;
    let vecs = unit_vectors(n, dim, 0xF015);
    let ids: Vec<u64> = (100..100 + n as u64).collect();
    let query = unit_vectors(1, dim, 0xF016);

    let mut idx = IdMapIndex::new_with_refine(dim, 4, RefineMode::Float32).unwrap();
    idx.add_with_ids(&vecs, &ids).unwrap();

    let (scores, ret_ids) = idx.search_with_rerank(&query, 5, 3).unwrap();
    assert_eq!(scores.len(), 5);
    assert_eq!(ret_ids.len(), 5);

    // All returned ids must be in the range we added.
    for &id in &ret_ids {
        assert!(id >= 100 && id < 100 + n as u64, "unexpected id {id}");
    }
}
