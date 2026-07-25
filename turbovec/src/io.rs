//! Read/write TurboVec index files.
//!
//! Two formats live here:
//! * `.tv` — [`TurboQuantIndex`](crate::TurboQuantIndex) — 4-byte magic
//!   "TVPI" + version + bit_width/dim/n_vectors header + packed codes +
//!   per-vector scales + (v3+) TQ+ per-coord calibration + (v4+) refine store.
//! * `.tvim` — [`IdMapIndex`](crate::IdMapIndex) — 4-byte magic "TVIM"
//!   + version + the same core-index payload + a trailing `slot_to_id`
//!   table of `u64` values.
//!
//! ## Format versioning
//!
//! | Version | Turbovec | Changes |
//! |---------|----------|---------|
//! | v1 | ≤ 0.4.3 | No magic; refused with rebuild hint |
//! | v2 | 0.4.4–0.5.x | Magic + version; no TQ+; loads as identity calibration |
//! | v3 | 0.6.x+ | Adds TQ+ calibration trailer |
//! | v4 | 0.9+ | Adds optional refine store trailer; only written when present |
//!
//! A v4 file is written only when the index was constructed with a
//! `RefineStore`. Default (no refine) indexes continue to write v3 so old
//! readers are unaffected.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::refine::{RefineMode, RefineStore};
use half::f16;

const TV_MAGIC: &[u8; 4] = b"TVPI";
const TV_VERSION_DEFAULT: u8 = 3;
const TV_VERSION_WITH_REFINE: u8 = 4;
const TVIM_MAGIC: &[u8; 4] = b"TVIM";
const TVIM_VERSION_DEFAULT: u8 = 3;
const TVIM_VERSION_WITH_REFINE: u8 = 4;

const REBUILD_HINT: &str =
    "Rebuild this index from the source vectors using turbovec 0.4.4 or later \
     (no in-place migration is provided; the format version 2 changes the meaning \
     of the per-vector scalar from ||v|| to a length-renormalization correction).";

/// Core payload — what a fully-deserialized index needs.
/// Tuple: (bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale, refine_store)
type CoreLoad = (usize, usize, usize, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, Option<RefineStore>);

/// `.tv` write — positional index.
pub fn write(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    refine: Option<&RefineStore>,
) -> io::Result<()> {
    let version = if refine.is_some() { TV_VERSION_WITH_REFINE } else { TV_VERSION_DEFAULT };
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TV_MAGIC)?;
    f.write_all(&[version])?;
    write_core(
        &mut f, bit_width, dim, n_vectors, packed_codes, scales,
        tqplus_shift, tqplus_scale,
    )?;
    if let Some(rs) = refine {
        write_refine_trailer(&mut f, rs, n_vectors, dim)?;
    }
    f.flush()?;
    Ok(())
}

/// `.tv` load — positional index. Transparently handles v2 (no TQ+),
/// v3 (with TQ+), and v4 (with refine store).
pub fn load(path: impl AsRef<Path>) -> io::Result<CoreLoad> {
    let mut f = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TV_MAGIC {
        if (2..=4).contains(&magic[0]) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "this .tv file was written by turbovec ≤ 0.4.3 (format \
                     version 1). It is incompatible with turbovec 0.4.4+ \
                     because the per-vector scalar's meaning changed. {}",
                    REBUILD_HINT,
                ),
            ));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a turbovec .tv file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    read_core_versioned(&mut f, version[0], TV_VERSION_WITH_REFINE, ".tv")
}

/// `.tvim` write — positional index plus the id-map side-tables.
pub fn write_id_map(
    path: impl AsRef<Path>,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    slot_to_id: &[u64],
    refine: Option<&RefineStore>,
) -> io::Result<()> {
    assert_eq!(
        slot_to_id.len(),
        n_vectors,
        "slot_to_id length {} does not match n_vectors {}",
        slot_to_id.len(),
        n_vectors,
    );

    let version = if refine.is_some() { TVIM_VERSION_WITH_REFINE } else { TVIM_VERSION_DEFAULT };
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(TVIM_MAGIC)?;
    f.write_all(&[version])?;
    write_core(
        &mut f, bit_width, dim, n_vectors, packed_codes, scales,
        tqplus_shift, tqplus_scale,
    )?;

    for &id in slot_to_id {
        f.write_all(&id.to_le_bytes())?;
    }
    if let Some(rs) = refine {
        write_refine_trailer(&mut f, rs, n_vectors, dim)?;
    }
    f.flush()?;
    Ok(())
}

/// `.tvim` load — positional index plus the id-map side-tables.
pub fn load_id_map(
    path: impl AsRef<Path>,
) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u64>, Option<RefineStore>)> {
    let mut f = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != TVIM_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a TVIM file: wrong magic",
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    if version[0] == 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "this .tvim file was written by turbovec ≤ 0.4.3 (format \
                 version 1). It is incompatible with turbovec 0.4.4+ \
                 because the per-vector scalar's meaning changed. {}",
                REBUILD_HINT,
            ),
        ));
    }
    let (bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale, refine_store) =
        read_core_versioned(&mut f, version[0], TVIM_VERSION_WITH_REFINE, ".tvim")?;

    let mut slot_to_id = Vec::with_capacity(n_vectors);
    let mut buf = [0u8; 8];
    for _ in 0..n_vectors {
        f.read_exact(&mut buf)?;
        slot_to_id.push(u64::from_le_bytes(buf));
    }

    // v4: refine trailer follows slot_to_id.
    let refine_store = if version[0] == 4 {
        Some(read_refine_trailer(&mut f, n_vectors, dim)?)
    } else {
        refine_store // None from versioned dispatch
    };

    Ok((
        bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale,
        slot_to_id, refine_store,
    ))
}

const CORE_HEADER_SIZE: usize = 9;

/// Core header + packed codes + per-vector scales + TQ+ calibration —
/// shared by `.tv` and `.tvim`.
fn write_core<W: Write>(
    w: &mut W,
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
) -> io::Result<()> {
    w.write_all(&[bit_width as u8])?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    w.write_all(&(n_vectors as u32).to_le_bytes())?;
    w.write_all(packed_codes)?;
    for &s in scales {
        w.write_all(&s.to_le_bytes())?;
    }
    // TQ+ trailer. n_calib == 0 means identity calibration (lazy index
    // with no add yet, or a loaded pre-TQ+ index that's been resaved);
    // otherwise must equal dim.
    assert!(
        tqplus_shift.len() == tqplus_scale.len()
            && (tqplus_shift.is_empty() || tqplus_shift.len() == dim),
        "TQ+ shift/scale must have equal length and either be empty or equal dim"
    );
    let n_calib = tqplus_shift.len() as u32;
    w.write_all(&n_calib.to_le_bytes())?;
    for &s in tqplus_shift {
        w.write_all(&s.to_le_bytes())?;
    }
    for &s in tqplus_scale {
        w.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

/// Write the v4 refine trailer.
///
/// Layout: `u8 mode (1=Int8, 2=Float32, 3=Float16)` followed by:
/// - Int8: `n` f32 scales, then `n * dim` i8 codes.
/// - Float32: `n * dim` f32 values (4 bytes each, LE).
/// - Float16: `n * dim` f16 values (2 bytes each, LE).
fn write_refine_trailer<W: Write>(
    w: &mut W,
    rs: &RefineStore,
    n_vectors: usize,
    dim: usize,
) -> io::Result<()> {
    match rs.mode {
        RefineMode::Int8 => {
            w.write_all(&[1u8])?;
            assert_eq!(rs.i8_scales.len(), n_vectors);
            assert_eq!(rs.codes_i8.len(), n_vectors * dim);
            for &s in &rs.i8_scales {
                w.write_all(&s.to_le_bytes())?;
            }
            for &c in &rs.codes_i8 {
                w.write_all(&[c as u8])?;
            }
        }
        RefineMode::Float32 => {
            w.write_all(&[2u8])?;
            assert_eq!(rs.floats.len(), n_vectors * dim);
            for &v in &rs.floats {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        RefineMode::Float16 => {
            w.write_all(&[3u8])?;
            assert_eq!(rs.halfs.len(), n_vectors * dim);
            for &v in &rs.halfs {
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

/// Read the v4 refine trailer.
fn read_refine_trailer<R: Read>(
    r: &mut R,
    n_vectors: usize,
    dim: usize,
) -> io::Result<RefineStore> {
    let mut mode_byte = [0u8; 1];
    r.read_exact(&mut mode_byte)?;
    match mode_byte[0] {
        1 => {
            let i8_scales = read_f32_array(r, n_vectors)?;
            let n_codes = n_vectors * dim;
            let mut bytes = vec![0u8; n_codes];
            r.read_exact(&mut bytes)?;
            let codes_i8: Vec<i8> = bytes.into_iter().map(|b| b as i8).collect();
            Ok(RefineStore {
                mode: RefineMode::Int8,
                codes_i8,
                i8_scales,
                halfs: Vec::new(),
                floats: Vec::new(),
            })
        }
        2 => {
            let floats = read_f32_array(r, n_vectors * dim)?;
            Ok(RefineStore {
                mode: RefineMode::Float32,
                codes_i8: Vec::new(),
                i8_scales: Vec::new(),
                halfs: Vec::new(),
                floats,
            })
        }
        3 => {
            let halfs = read_f16_array(r, n_vectors * dim)?;
            Ok(RefineStore {
                mode: RefineMode::Float16,
                codes_i8: Vec::new(),
                i8_scales: Vec::new(),
                halfs,
                floats: Vec::new(),
            })
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unknown refine mode byte {other} in v4 file \
                 (expected 1=Int8, 2=Float32, or 3=Float16)"
            ),
        )),
    }
}

/// Read the core payload, dispatching on the version byte. Knows about
/// v2 (no TQ+) and v3 (with TQ+); v4 deferred to caller for `.tvim`
/// (which must read slot_to_id between core and refine trailer).
fn read_core_versioned<R: Read>(
    r: &mut R,
    version: u8,
    expected: u8,
    label: &str,
) -> io::Result<CoreLoad> {
    match version {
        2 => read_core_v2(r),
        3 => read_core_v3(r),
        4 => read_core_v4(r),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported {label} format version: {version} (this build \
                 supports versions 2, 3, and {expected})",
            ),
        )),
    }
}

/// v2: header + codes + scales. Returns empty TQ+ vectors (identity calibration).
fn read_core_v2<R: Read>(r: &mut R) -> io::Result<CoreLoad> {
    let (bit_width, dim, n_vectors, packed_codes, scales) = read_header_codes_scales(r)?;
    Ok((bit_width, dim, n_vectors, packed_codes, scales, Vec::new(), Vec::new(), None))
}

/// v3: header + codes + scales + TQ+ trailer.
fn read_core_v3<R: Read>(r: &mut R) -> io::Result<CoreLoad> {
    let (bit_width, dim, n_vectors, packed_codes, scales) = read_header_codes_scales(r)?;

    let mut n_calib_bytes = [0u8; 4];
    r.read_exact(&mut n_calib_bytes)?;
    let n_calib = u32::from_le_bytes(n_calib_bytes) as usize;
    if n_calib != 0 && n_calib != dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid TQ+ n_calib {n_calib}: must be 0 or equal to dim {dim}"),
        ));
    }
    let tqplus_shift = read_f32_array(r, n_calib)?;
    let tqplus_scale = read_f32_array(r, n_calib)?;

    Ok((bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale, None))
}

/// v4: same as v3 core, then refine trailer.
/// For `.tv`: refine trailer immediately follows TQ+ data.
/// For `.tvim`: caller reads slot_to_id between TQ+ and refine; this function
/// returns `None` for refine and the caller handles v4 detection separately.
fn read_core_v4<R: Read>(r: &mut R) -> io::Result<CoreLoad> {
    let (bit_width, dim, n_vectors, packed_codes, scales) = read_header_codes_scales(r)?;

    let mut n_calib_bytes = [0u8; 4];
    r.read_exact(&mut n_calib_bytes)?;
    let n_calib = u32::from_le_bytes(n_calib_bytes) as usize;
    if n_calib != 0 && n_calib != dim {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid TQ+ n_calib {n_calib}: must be 0 or equal to dim {dim}"),
        ));
    }
    let tqplus_shift = read_f32_array(r, n_calib)?;
    let tqplus_scale = read_f32_array(r, n_calib)?;

    // For .tv files, read the refine trailer right after TQ+ data.
    // For .tvim, the caller handles this because slot_to_id comes first.
    let refine_store = read_refine_trailer(r, n_vectors, dim)?;

    Ok((bit_width, dim, n_vectors, packed_codes, scales, tqplus_shift, tqplus_scale, Some(refine_store)))
}

fn read_header_codes_scales<R: Read>(
    r: &mut R,
) -> io::Result<(usize, usize, usize, Vec<u8>, Vec<f32>)> {
    let mut header = [0u8; CORE_HEADER_SIZE];
    r.read_exact(&mut header)?;
    let bit_width = header[0] as usize;
    let dim = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let n_vectors = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;

    let packed_bytes = (dim / 8) * bit_width * n_vectors;
    let mut packed_codes = vec![0u8; packed_bytes];
    r.read_exact(&mut packed_codes)?;

    let scales = read_f32_array(r, n_vectors)?;
    Ok((bit_width, dim, n_vectors, packed_codes, scales))
}

fn read_f32_array<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn read_f16_array<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<f16>> {
    let mut bytes = vec![0u8; n * 2];
    r.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]))
        .collect())
}
