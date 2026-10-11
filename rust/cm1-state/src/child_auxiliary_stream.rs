//! Bounded-memory, grid-generic auxiliary/base-state publication primitives.
//!
//! The historical 300x300 producer materializes every carrier.  This module
//! retains its exact source arithmetic while writing horizontal slabs directly
//! into a single carrier-major arena, allowing the real 3000x3000 child to be
//! produced without a second multi-gigabyte in-memory state.

use crate::child_atmosphere::{
    child_atmosphere_carrier_layout_for_grid, ChildAtmosphereCarrierLayout,
    CHILD_ATMOSPHERE_ARENA_FILE, CHILD_ATMOSPHERE_MANIFEST_FILE, CHILD_ATMOSPHERE_RECEIPT_FILE,
    HARD_5M_CHILD_ATMOSPHERE_ARENA_SHA256, HARD_5M_CHILD_ATMOSPHERE_MANIFEST_SHA256,
    HARD_5M_CHILD_ATMOSPHERE_RECEIPT_SHA256, HARD_5M_STATIC_HEIGHT_MANIFEST_SHA256,
    HARD_5M_STATIC_HEIGHT_RECEIPT_SHA256, HARD_5M_STATIC_HEIGHT_VALIDATION_SHA256,
    HARD_5M_STATIC_HEIGHT_ZF_SHA256, HARD_5M_STATIC_HEIGHT_ZH_SHA256,
    HARD_5M_TERRAIN_MANIFEST_SHA256, HARD_5M_TERRAIN_PAYLOAD_SHA256,
    HARD_5M_TERRAIN_RECEIPT_SHA256, SEALED_CHILD_ATMOSPHERE_ARENA_SHA256,
    SEALED_CHILD_ATMOSPHERE_MANIFEST_SHA256, SEALED_CHILD_ATMOSPHERE_RECEIPT_SHA256,
};
use crate::child_auxiliary::{
    admit_child_auxiliary_base_capacity, child_auxiliary_base_layout_for_grid,
    derive_stretched_column_metrics, extrapolation_weights, fill_stretched_column_metrics,
    CHILD_AUXILIARY_GEOPOTENTIAL_FILE, CHILD_AUXILIARY_LIVE_DENSITY_FILE,
    CHILD_AUXILIARY_TERRAIN_OPEN_FILE, CHILD_LIVE_DENSITY_CARRIER_NAMES,
    CHILD_TERRAIN_MANIFEST_SHA256, CHILD_TERRAIN_PAYLOAD_SHA256, CHILD_TERRAIN_RECEIPT_SHA256,
    F02_PRODUCT_MANIFEST_SHA256, F02_PRODUCT_PAYLOAD_SHA256, F02_PRODUCT_RECEIPT_SHA256,
    F02_PRODUCT_RESTART_SHA256, F02_SOURCE_ARTIFACT_SHA256, F02_SOURCE_CONTRACT_SHA256,
    STATIC_HEIGHT_MANIFEST_SHA256, STATIC_VALIDATION_RECEIPT_SHA256, STATIC_ZF_SHA256,
    STATIC_ZH_SHA256,
};
use crate::child_full_state::{interpolate_source_profile, CHILD_BASE_WIND_FILE};
use crate::child_remap::{
    sealed_child_stretch_grid, ChildGridSpec, ACCEPTED_50M_CHILD_GRID, HARD_5M_CHILD_GRID,
};
use crate::json::JsonValue;
use crate::product_restart::read_initialized_product_restart;
use crate::sha256::{self, Sha256};
use crate::terrain_admission::terrain_open_carrier_shapes;
use crate::terrain_admission_science::{
    StretchedTerrainBaseColumnGenerator, CP, CPI, CPL, CPV, G, NGXY, P00, RD, RV,
};
use crate::{validate_sha256, Result, StateError};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHILD_AUXILIARY_STREAM_SCHEMA: &str = "gm1.source-coordinate-child-auxiliary-stream/v2";
pub const CHILD_AUXILIARY_STREAM_ABI: &str =
    "gm1.source-coordinate-child-auxiliary-base-f32le-c-order/v2";
pub const CHILD_AUXILIARY_STREAM_SLAB_ROWS: usize = 16;
pub const CHILD_AUXILIARY_STREAM_MANIFEST_SCHEMA: &str =
    "gm1.source-coordinate-child-auxiliary-stream-manifest/v2";
pub const CHILD_AUXILIARY_STREAM_RECEIPT_SCHEMA: &str =
    "gm1.source-coordinate-child-auxiliary-stream-receipt/v2";
pub const CHILD_AUXILIARY_STREAM_MANIFEST_FILE: &str = "manifest.json";
pub const CHILD_AUXILIARY_STREAM_RECEIPT_FILE: &str = "receipt.json";
pub const CHILD_AUXILIARY_STREAM_STATUS: &str = "PASS_SOURCE_BACKED_STREAMED_CHILD_AUXILIARY_BASE";
pub const CHILD_AUXILIARY_STREAM_CLAIM_LIMIT: &str = "source-backed CM1 r21.1 terrain-column hydrostatic base, restart-grid-preserved mutable/open carriers, selected psolver=3 physical pressure/density, exact binary32 zf geopotential, and terrain-aware staggered Rayleigh base wind for one admitted child grid; no closure, time integration, forecast, or synthetic meteorological detail claim";

static STREAM_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamCarrierLayout {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub offset_nbytes: u64,
    pub nbytes: u64,
}

#[derive(Clone, Debug)]
pub struct StreamedBaseProfiles {
    pub pi0: Vec<f32>,
    pub th0: Vec<f32>,
    pub qv0: Vec<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamAtmosphereIdentity {
    pub manifest_sha256: String,
    pub arena_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdmittedChildAuxiliaryStreamPreset {
    pub grid: ChildGridSpec,
    pub atmosphere_manifest_sha256: &'static str,
    pub atmosphere_arena_sha256: &'static str,
    pub atmosphere_receipt_sha256: &'static str,
    pub terrain_manifest_sha256: &'static str,
    pub terrain_payload_sha256: &'static str,
    pub terrain_receipt_sha256: &'static str,
    pub static_manifest_sha256: &'static str,
    pub static_zh_sha256: &'static str,
    pub static_zf_sha256: &'static str,
    pub static_receipt_sha256: Option<&'static str>,
    pub static_validation_sha256: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildAuxiliaryStreamReceipt {
    pub manifest_sha256: String,
    pub terrain_open_sha256: String,
    pub live_density_sha256: String,
    pub geopotential_sha256: String,
    pub base_wind_sha256: String,
    pub receipt_sha256: String,
    pub producer_source_sha256: String,
    pub producer_executable_sha256: String,
}

#[derive(Clone, Debug)]
struct LoadedSourceProfiles {
    base: StreamedBaseProfiles,
    u0: Vec<f32>,
    v0: Vec<f32>,
    hashes: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
struct StreamPayloadRecord {
    name: &'static str,
    shape: Vec<usize>,
    offset_nbytes: u64,
    nbytes: u64,
    sha256: String,
}

#[derive(Clone, Debug)]
struct StreamPayloadFile {
    name: &'static str,
    nbytes: u64,
    sha256: String,
    records: Vec<StreamPayloadRecord>,
}

pub trait AtmospherePlaneReader {
    fn read_plane_f32(&mut self, name: &str, prefix: &[usize]) -> Result<Vec<f32>>;
}

pub struct FileAtmospherePlaneReader {
    file: File,
    layout: BTreeMap<&'static str, ChildAtmosphereCarrierLayout>,
}

impl FileAtmospherePlaneReader {
    pub fn open_verified(
        directory: &Path,
        grid: ChildGridSpec,
        identity: &StreamAtmosphereIdentity,
    ) -> Result<Self> {
        for (value, label) in [
            (&identity.manifest_sha256, "stream atmosphere manifest"),
            (&identity.arena_sha256, "stream atmosphere arena"),
            (&identity.receipt_sha256, "stream atmosphere receipt"),
        ] {
            validate_sha256(value, label)?;
        }
        let expected_names = [
            CHILD_ATMOSPHERE_ARENA_FILE,
            CHILD_ATMOSPHERE_MANIFEST_FILE,
            CHILD_ATMOSPHERE_RECEIPT_FILE,
        ];
        let mut actual_names = fs::read_dir(directory)?
            .map(|entry| {
                let entry = entry?;
                if entry.file_type()?.is_symlink() || !entry.file_type()?.is_file() {
                    return Err(invalid("stream atmosphere contains a non-regular entry"));
                }
                entry
                    .file_name()
                    .into_string()
                    .map_err(|_| invalid("stream atmosphere contains a non-UTF8 name"))
            })
            .collect::<Result<Vec<_>>>()?;
        actual_names.sort();
        let mut expected_sorted = expected_names.map(str::to_owned);
        expected_sorted.sort();
        if actual_names != expected_sorted {
            return Err(invalid(
                "stream atmosphere inventory is not exactly three files",
            ));
        }
        for (name, expected, maximum) in [
            (
                CHILD_ATMOSPHERE_MANIFEST_FILE,
                identity.manifest_sha256.as_str(),
                1024 * 1024,
            ),
            (
                CHILD_ATMOSPHERE_RECEIPT_FILE,
                identity.receipt_sha256.as_str(),
                1024 * 1024,
            ),
        ] {
            let path = directory.join(name);
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum
            {
                return Err(invalid(format!(
                    "stream atmosphere {name} is not a bounded regular file"
                )));
            }
            let bytes = fs::read(path)?;
            if sha256::digest_hex(&bytes) != expected {
                return Err(invalid(format!(
                    "stream atmosphere {name} differs from its independent hash"
                )));
            }
        }
        let layout = child_atmosphere_carrier_layout_for_grid(grid)?;
        let expected_nbytes = layout
            .last()
            .map(|record| record.offset_nbytes + record.nbytes)
            .ok_or_else(|| invalid("stream atmosphere layout is empty"))?;
        let arena_path = directory.join(CHILD_ATMOSPHERE_ARENA_FILE);
        let metadata = fs::symlink_metadata(&arena_path)?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.len() != expected_nbytes
        {
            return Err(invalid(
                "stream atmosphere arena is not an exact regular file",
            ));
        }
        let mut reader = BufReader::with_capacity(1024 * 1024, File::open(&arena_path)?);
        let mut hasher = Sha256::default();
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
        if sha256::hex(&hasher.finalize()) != identity.arena_sha256 {
            return Err(invalid(
                "stream atmosphere arena differs from its independent hash",
            ));
        }
        Ok(Self {
            file: File::open(arena_path)?,
            layout: layout
                .into_iter()
                .map(|record| (record.name, record))
                .collect(),
        })
    }
}

impl AtmospherePlaneReader for FileAtmospherePlaneReader {
    fn read_plane_f32(&mut self, name: &str, prefix: &[usize]) -> Result<Vec<f32>> {
        let record = self
            .layout
            .get(name)
            .ok_or_else(|| invalid(format!("stream atmosphere has no carrier {name:?}")))?;
        if record.shape.len() < 2 || prefix.len() + 2 != record.shape.len() {
            return Err(invalid(format!(
                "stream atmosphere {name} plane prefix rank is invalid"
            )));
        }
        let mut plane_index = 0u64;
        for (index, extent) in prefix.iter().zip(&record.shape[..prefix.len()]) {
            if *index >= *extent {
                return Err(invalid(format!(
                    "stream atmosphere {name} plane index is outside its shape"
                )));
            }
            plane_index = plane_index
                .checked_mul(*extent as u64)
                .and_then(|value| value.checked_add(*index as u64))
                .ok_or_else(|| invalid("stream atmosphere plane index overflows"))?;
        }
        let plane_words = record.shape[record.shape.len() - 2]
            .checked_mul(record.shape[record.shape.len() - 1])
            .ok_or_else(|| invalid("stream atmosphere plane word count overflows"))?;
        let plane_nbytes = plane_words
            .checked_mul(4)
            .ok_or_else(|| invalid("stream atmosphere plane byte count overflows"))?;
        let offset = record
            .offset_nbytes
            .checked_add(
                plane_index
                    .checked_mul(plane_nbytes as u64)
                    .ok_or_else(|| invalid("stream atmosphere plane offset overflows"))?,
            )
            .ok_or_else(|| invalid("stream atmosphere plane offset overflows"))?;
        self.file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0u8; plane_nbytes];
        self.file.read_exact(&mut bytes)?;
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "stream atmosphere {name} plane contains a non-finite word"
            )));
        }
        Ok(values)
    }
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

pub fn admitted_child_auxiliary_stream_preset(
    grid: ChildGridSpec,
) -> Result<AdmittedChildAuxiliaryStreamPreset> {
    if grid == ACCEPTED_50M_CHILD_GRID {
        Ok(AdmittedChildAuxiliaryStreamPreset {
            grid,
            atmosphere_manifest_sha256: SEALED_CHILD_ATMOSPHERE_MANIFEST_SHA256,
            atmosphere_arena_sha256: SEALED_CHILD_ATMOSPHERE_ARENA_SHA256,
            atmosphere_receipt_sha256: SEALED_CHILD_ATMOSPHERE_RECEIPT_SHA256,
            terrain_manifest_sha256: CHILD_TERRAIN_MANIFEST_SHA256,
            terrain_payload_sha256: CHILD_TERRAIN_PAYLOAD_SHA256,
            terrain_receipt_sha256: CHILD_TERRAIN_RECEIPT_SHA256,
            static_manifest_sha256: STATIC_HEIGHT_MANIFEST_SHA256,
            static_zh_sha256: STATIC_ZH_SHA256,
            static_zf_sha256: STATIC_ZF_SHA256,
            static_receipt_sha256: None,
            static_validation_sha256: STATIC_VALIDATION_RECEIPT_SHA256,
        })
    } else if grid == HARD_5M_CHILD_GRID {
        Ok(AdmittedChildAuxiliaryStreamPreset {
            grid,
            atmosphere_manifest_sha256: HARD_5M_CHILD_ATMOSPHERE_MANIFEST_SHA256,
            atmosphere_arena_sha256: HARD_5M_CHILD_ATMOSPHERE_ARENA_SHA256,
            atmosphere_receipt_sha256: HARD_5M_CHILD_ATMOSPHERE_RECEIPT_SHA256,
            terrain_manifest_sha256: HARD_5M_TERRAIN_MANIFEST_SHA256,
            terrain_payload_sha256: HARD_5M_TERRAIN_PAYLOAD_SHA256,
            terrain_receipt_sha256: HARD_5M_TERRAIN_RECEIPT_SHA256,
            static_manifest_sha256: HARD_5M_STATIC_HEIGHT_MANIFEST_SHA256,
            static_zh_sha256: HARD_5M_STATIC_HEIGHT_ZH_SHA256,
            static_zf_sha256: HARD_5M_STATIC_HEIGHT_ZF_SHA256,
            static_receipt_sha256: Some(HARD_5M_STATIC_HEIGHT_RECEIPT_SHA256),
            static_validation_sha256: HARD_5M_STATIC_HEIGHT_VALIDATION_SHA256,
        })
    } else {
        Err(invalid(
            "streamed auxiliary publisher accepts only the sealed 50 m or 5 m child grid",
        ))
    }
}

pub fn terrain_open_stream_layout(grid: ChildGridSpec) -> Result<Vec<StreamCarrierLayout>> {
    let shapes = terrain_open_carrier_shapes(grid.nz(), grid.ny(), grid.nx())?;
    let mut offset_nbytes = 0u64;
    let mut records = Vec::with_capacity(shapes.len());
    for (name, shape) in shapes {
        let words = shape.iter().try_fold(1u64, |words, extent| {
            words
                .checked_mul(*extent as u64)
                .ok_or_else(|| invalid("streamed terrain/open carrier word count overflows"))
        })?;
        let nbytes = words
            .checked_mul(4)
            .ok_or_else(|| invalid("streamed terrain/open carrier byte count overflows"))?;
        records.push(StreamCarrierLayout {
            name,
            shape,
            offset_nbytes,
            nbytes,
        });
        offset_nbytes = offset_nbytes
            .checked_add(nbytes)
            .ok_or_else(|| invalid("streamed terrain/open carrier offset overflows"))?;
    }
    let expected = child_auxiliary_base_layout_for_grid(grid)?.terrain_open_nbytes;
    if offset_nbytes != expected {
        return Err(invalid(
            "streamed terrain/open carrier layout does not exhaust its arena",
        ));
    }
    Ok(records)
}

pub fn live_density_stream_layout(grid: ChildGridSpec) -> Result<Vec<StreamCarrierLayout>> {
    let mut offset_nbytes = 0u64;
    let mut records = Vec::with_capacity(CHILD_LIVE_DENSITY_CARRIER_NAMES.len());
    for name in CHILD_LIVE_DENSITY_CARRIER_NAMES {
        let shape = if name == "rf" {
            vec![grid.nz() + 1, grid.ny(), grid.nx()]
        } else {
            vec![grid.nz(), grid.ny(), grid.nx()]
        };
        let words = shape.iter().try_fold(1u64, |words, extent| {
            words
                .checked_mul(*extent as u64)
                .ok_or_else(|| invalid("streamed live-density carrier word count overflows"))
        })?;
        let nbytes = words
            .checked_mul(4)
            .ok_or_else(|| invalid("streamed live-density carrier byte count overflows"))?;
        records.push(StreamCarrierLayout {
            name,
            shape,
            offset_nbytes,
            nbytes,
        });
        offset_nbytes = offset_nbytes
            .checked_add(nbytes)
            .ok_or_else(|| invalid("streamed live-density carrier offset overflows"))?;
    }
    let expected = child_auxiliary_base_layout_for_grid(grid)?.live_density_nbytes;
    if offset_nbytes != expected {
        return Err(invalid(
            "streamed live-density layout does not exhaust its arena",
        ));
    }
    Ok(records)
}

pub fn base_wind_stream_layout(grid: ChildGridSpec) -> Result<Vec<StreamCarrierLayout>> {
    let shapes = [
        ("u0", vec![grid.nz(), grid.ny(), grid.nx() + 1]),
        ("v0", vec![grid.nz(), grid.ny() + 1, grid.nx()]),
    ];
    let mut offset_nbytes = 0u64;
    let mut records = Vec::with_capacity(shapes.len());
    for (name, shape) in shapes {
        let words = shape.iter().try_fold(1u64, |words, extent| {
            words
                .checked_mul(*extent as u64)
                .ok_or_else(|| invalid("streamed base-wind carrier word count overflows"))
        })?;
        let nbytes = words
            .checked_mul(4)
            .ok_or_else(|| invalid("streamed base-wind carrier byte count overflows"))?;
        records.push(StreamCarrierLayout {
            name,
            shape,
            offset_nbytes,
            nbytes,
        });
        offset_nbytes = offset_nbytes
            .checked_add(nbytes)
            .ok_or_else(|| invalid("streamed base-wind carrier offset overflows"))?;
    }
    let expected = child_auxiliary_base_layout_for_grid(grid)?.base_wind_nbytes;
    if offset_nbytes != expected {
        return Err(invalid(
            "streamed base-wind layout does not exhaust its arena",
        ));
    }
    Ok(records)
}

/// Write the ten hydrostatic base carriers, four base surface diagnostics, and
/// nine terrain/surface metric carriers. No mutable dynamics carrier is touched.
pub fn write_streamed_base_and_surface_regions(
    file: &mut File,
    grid: ChildGridSpec,
    terrain: &[f32],
    profiles: &StreamedBaseProfiles,
) -> Result<()> {
    let nx = grid.nx();
    let ny = grid.ny();
    let nz = grid.nz();
    if terrain.len() != ny * nx
        || terrain
            .iter()
            .any(|value| !value.is_finite() || *value >= grid.ztop_m() as f32)
        || profiles.pi0.len() != nz
        || profiles.th0.len() != nz
        || profiles.qv0.len() != nz
    {
        return Err(invalid(
            "streamed base/surface inputs differ from the admitted grid",
        ));
    }
    let stretch = sealed_child_stretch_grid()?;
    let sigma_mass = &stretch.sigma[1..=nz];
    let sigma_face = &stretch.sigmaf[1..=nz + 1];
    let child_bottom = extrapolation_weights(sigma_mass, sigma_face, false)?;
    let child_top = extrapolation_weights(sigma_mass, sigma_face, true)?;
    let generator = StretchedTerrainBaseColumnGenerator::new(
        sigma_mass,
        grid.ztop_m() as f32,
        200.0,
        &profiles.pi0,
        &profiles.th0,
        &profiles.qv0,
        child_bottom,
        child_top,
    )?;
    let records = terrain_open_stream_layout(grid)?;
    write_base_regions(
        file, grid, terrain, sigma_mass, sigma_face, &generator, &records,
    )?;
    write_surface_metric_regions(file, grid, terrain, &records)
}

fn write_base_regions(
    file: &mut File,
    grid: ChildGridSpec,
    terrain: &[f32],
    sigma_mass: &[f32],
    sigma_face: &[f32],
    generator: &StretchedTerrainBaseColumnGenerator,
    records: &[StreamCarrierLayout],
) -> Result<()> {
    let nx = grid.nx();
    let ny = grid.ny();
    let nz = grid.nz();
    let halo_nx = nx + 2 * NGXY;
    let halo_ny = ny + 2 * NGXY;
    let mut bytes = Vec::new();
    for slab_start in (0..halo_ny).step_by(CHILD_AUXILIARY_STREAM_SLAB_ROWS) {
        let slab_rows = (halo_ny - slab_start).min(CHILD_AUXILIARY_STREAM_SLAB_ROWS);
        let slab_words = slab_rows * halo_nx;
        let mut mass = (0..10 * (nz + 2))
            .map(|_| vec![0.0_f32; slab_words])
            .collect::<Vec<_>>();
        let mut surface = (0..4)
            .map(|_| vec![0.0_f32; slab_words])
            .collect::<Vec<_>>();
        for local_j in 0..slab_rows {
            let jh = slab_start + local_j;
            let j = (jh as isize - NGXY as isize).clamp(0, ny as isize - 1) as usize;
            for ih in 0..halo_nx {
                let i = (ih as isize - NGXY as isize).clamp(0, nx as isize - 1) as usize;
                let surface_height = terrain[j * nx + i];
                let metrics = derive_stretched_column_metrics(
                    surface_height,
                    sigma_mass,
                    sigma_face,
                    grid.ztop_m() as f32,
                )?;
                let column = generator.generate(surface_height, None, &metrics.c1, &metrics.c2)?;
                let mass_values = [
                    &column.pi0,
                    &column.prs0,
                    &column.th0,
                    &column.thv0,
                    &column.rth0,
                    &column.qv0,
                    &column.rho0,
                    &column.rr0,
                    &column.rf0,
                    &column.rrf0,
                ];
                let horizontal = local_j * halo_nx + ih;
                for (carrier, values) in mass_values.iter().enumerate() {
                    for (k, value) in values.iter().copied().enumerate() {
                        mass[carrier * (nz + 2) + k][horizontal] = value;
                    }
                }
                for (carrier, value) in [column.rho0s, column.pi0s, column.prs0s, column.rth0s]
                    .into_iter()
                    .enumerate()
                {
                    surface[carrier][horizontal] = value;
                }
            }
        }
        for carrier in 0..10 {
            for k in 0..nz + 2 {
                let record = &records[carrier];
                let word_offset = (k * halo_ny + slab_start) * halo_nx;
                write_f32_at(
                    file,
                    record.offset_nbytes + checked_bytes(word_offset, "base slab offset")?,
                    &mass[carrier * (nz + 2) + k],
                    &mut bytes,
                )?;
            }
        }
        for carrier in 0..4 {
            let record = &records[10 + carrier];
            let word_offset = slab_start * halo_nx;
            write_f32_at(
                file,
                record.offset_nbytes + checked_bytes(word_offset, "base surface slab offset")?,
                &surface[carrier],
                &mut bytes,
            )?;
        }
    }
    Ok(())
}

fn write_surface_metric_regions(
    file: &mut File,
    grid: ChildGridSpec,
    terrain: &[f32],
    records: &[StreamCarrierLayout],
) -> Result<()> {
    let nx = grid.nx();
    let ny = grid.ny();
    let halo_nx = nx + 2 * NGXY;
    let halo_ny = ny + 2 * NGXY;
    let top = grid.ztop_m() as f32;
    let rdx = 1.0_f32 / grid.dx_m() as f32;
    let rdy = 1.0_f32 / grid.dy_m() as f32;
    let terrain_at = |i: isize, j: isize| -> f32 {
        let bounded_i = i.clamp(0, nx as isize - 1) as usize;
        let bounded_j = j.clamp(0, ny as isize - 1) as usize;
        terrain[bounded_j * nx + bounded_i]
    };
    let x_flux = |face_i: isize, j: isize| -> f32 {
        flx6(
            terrain_at(face_i - 3, j),
            terrain_at(face_i - 2, j),
            terrain_at(face_i - 1, j),
            terrain_at(face_i, j),
            terrain_at(face_i + 1, j),
            terrain_at(face_i + 2, j),
        )
    };
    let y_flux = |i: isize, face_j: isize| -> f32 {
        flx6(
            terrain_at(i, face_j - 3),
            terrain_at(i, face_j - 2),
            terrain_at(i, face_j - 1),
            terrain_at(i, face_j),
            terrain_at(i, face_j + 1),
            terrain_at(i, face_j + 2),
        )
    };
    let mut bytes = Vec::new();
    for slab_start in (0..halo_ny).step_by(CHILD_AUXILIARY_STREAM_SLAB_ROWS) {
        let slab_rows = (halo_ny - slab_start).min(CHILD_AUXILIARY_STREAM_SLAB_ROWS);
        let slab_words = slab_rows * halo_nx;
        let mut output = (0..9)
            .map(|_| vec![0.0_f32; slab_words])
            .collect::<Vec<_>>();
        for local_j in 0..slab_rows {
            let jh = slab_start + local_j;
            let global_j = jh as isize - NGXY as isize;
            for ih in 0..halo_nx {
                let global_i = ih as isize - NGXY as isize;
                let p = local_j * halo_nx + ih;
                let surface = terrain_at(global_i, global_j);
                output[0][p] = surface;
                output[1][p] = top / (top - surface);
                output[2][p] = (top - surface) / top;
                let bounded_i = global_i.clamp(0, nx as isize - 1);
                let bounded_j = global_j.clamp(0, ny as isize - 1);
                output[7][p] =
                    (x_flux(bounded_i + 1, bounded_j) - x_flux(bounded_i, bounded_j)) * rdx;
                output[8][p] =
                    (y_flux(bounded_i, bounded_j + 1) - y_flux(bounded_i, bounded_j)) * rdy;
                if jh != 0 && ih != 0 {
                    let u_surface =
                        0.5 * (terrain_at(global_i - 1, global_j) + terrain_at(global_i, global_j));
                    let v_surface =
                        0.5 * (terrain_at(global_i, global_j - 1) + terrain_at(global_i, global_j));
                    output[3][p] = top / (top - u_surface);
                    output[4][p] = (top - u_surface) / top;
                    output[5][p] = top / (top - v_surface);
                    output[6][p] = (top - v_surface) / top;
                }
            }
        }
        for (carrier, values) in output.iter().enumerate() {
            let record = &records[14 + carrier];
            write_f32_at(
                file,
                record.offset_nbytes
                    + checked_bytes(slab_start * halo_nx, "surface metric slab offset")?,
                values,
                &mut bytes,
            )?;
        }
    }
    Ok(())
}

/// Write all 27 mutable/open-boundary carriers after the 23 base/surface
/// regions are present. Each source plane and padded output plane is bounded;
/// no three-dimensional mutable carrier is materialized.
pub fn write_streamed_mutable_regions<R: AtmospherePlaneReader>(
    file: &mut File,
    grid: ChildGridSpec,
    terrain: &[f32],
    atmosphere: &mut R,
) -> Result<()> {
    let records = terrain_open_stream_layout(grid)?;
    let by_name = records
        .iter()
        .map(|record| (record.name, record))
        .collect::<BTreeMap<_, _>>();
    let ppi = required_record(&by_name, "ppi")?;
    write_pressure_from_source(file, grid, atmosphere, "ppi", ppi)?;
    copy_record(file, ppi, required_record(&by_name, "pp3d")?)?;
    // The selected restart initialization overwrites ppx from ppi after the
    // restart-grid interpolation; the atmosphere ppx carrier is deliberately
    // not admitted at this seam (cm1.F cold-start disposition).
    copy_record(file, ppi, required_record(&by_name, "ppx")?)?;
    write_pressure_diagnostics(file, grid, atmosphere, &by_name)?;
    write_scalar_from_source(
        file,
        grid,
        atmosphere,
        "tha",
        required_record(&by_name, "tha")?,
    )?;
    copy_record(
        file,
        required_record(&by_name, "tha")?,
        required_record(&by_name, "th3d")?,
    )?;
    for name in ["thten", "thten1"] {
        write_zero_record(file, required_record(&by_name, name)?)?;
    }
    write_u_from_source(file, grid, atmosphere, required_record(&by_name, "ua")?)?;
    copy_record(
        file,
        required_record(&by_name, "ua")?,
        required_record(&by_name, "u3d")?,
    )?;
    for name in ["uten", "uten1"] {
        write_zero_record(file, required_record(&by_name, name)?)?;
    }
    write_v_from_source(file, grid, atmosphere, required_record(&by_name, "va")?)?;
    copy_record(
        file,
        required_record(&by_name, "va")?,
        required_record(&by_name, "v3d")?,
    )?;
    for name in ["vten", "vten1"] {
        write_zero_record(file, required_record(&by_name, name)?)?;
    }
    write_w_from_source(
        file,
        grid,
        terrain,
        atmosphere,
        required_record(&by_name, "wa")?,
    )?;
    copy_record(
        file,
        required_record(&by_name, "wa")?,
        required_record(&by_name, "w3d")?,
    )?;
    for name in ["wten", "wten1", "radbcw", "radbce", "radbcs", "radbcn"] {
        write_zero_record(file, required_record(&by_name, name)?)?;
    }
    Ok(())
}

/// Rebuild the four physical live pressure/density carriers without retaining
/// a full-domain base or mutable state. Mass planes are written immediately;
/// face density is then reconstructed in bounded horizontal slabs.
pub fn write_streamed_live_density<R: AtmospherePlaneReader>(
    output: &mut File,
    terrain_open: &mut File,
    grid: ChildGridSpec,
    terrain: &[f32],
    atmosphere: &mut R,
) -> Result<()> {
    if terrain.len() != grid.ny() * grid.nx() {
        return Err(invalid("streamed live-density terrain shape is invalid"));
    }
    let output_records = live_density_stream_layout(grid)?;
    let output_by_name = output_records
        .iter()
        .map(|record| (record.name, record))
        .collect::<BTreeMap<_, _>>();
    let terrain_records = terrain_open_stream_layout(grid)?;
    let terrain_by_name = terrain_records
        .iter()
        .map(|record| (record.name, record))
        .collect::<BTreeMap<_, _>>();
    let prs_record = required_record(&output_by_name, "prs")?;
    let rf_record = required_record(&output_by_name, "rf")?;
    let rho_record = required_record(&output_by_name, "rho")?;
    let rr_record = required_record(&output_by_name, "rr")?;
    let pi0_record = required_record(&terrain_by_name, "pi0")?;
    let th0_record = required_record(&terrain_by_name, "th0")?;
    let mut bytes = Vec::new();
    for k in 0..grid.nz() {
        let ppi = atmosphere.read_plane_f32("ppi", &[k])?;
        let tha = atmosphere.read_plane_f32("tha", &[k])?;
        let mut qv = atmosphere.read_plane_f32("qa", &[0, k])?;
        let pi0 = read_base_physical_plane(terrain_open, pi0_record, k + 1, grid)?;
        let th0 = read_base_physical_plane(terrain_open, th0_record, k + 1, grid)?;
        let plane_words = grid.ny() * grid.nx();
        if ppi.len() != plane_words
            || tha.len() != plane_words
            || qv.len() != plane_words
            || pi0.len() != plane_words
            || th0.len() != plane_words
        {
            return Err(invalid(
                "streamed live-density mass plane geometry is invalid",
            ));
        }
        let mut prs = vec![0.0_f32; plane_words];
        let mut rho = vec![0.0_f32; plane_words];
        let mut rr = vec![0.0_f32; plane_words];
        for word in 0..plane_words {
            qv[word] = qv[word].max(0.0);
            let total_pi = pi0[word] + ppi[word];
            let total_theta = th0[word] + tha[word];
            prs[word] = P00 * total_pi.powf(CP / RD);
            rho[word] = prs[word] / (total_theta * total_pi * (RD + qv[word] * RV));
            rr[word] = 1.0 / rho[word];
            if !total_pi.is_finite()
                || total_pi <= 0.0
                || !total_theta.is_finite()
                || total_theta <= 0.0
                || !prs[word].is_finite()
                || prs[word] <= 0.0
                || !rho[word].is_finite()
                || rho[word] <= 0.0
                || !rr[word].is_finite()
                || rr[word] <= 0.0
            {
                return Err(invalid(format!(
                    "streamed live pressure/density is non-physical at k={k},word={word}"
                )));
            }
        }
        write_record_plane(output, prs_record, k, &prs, &mut bytes)?;
        write_record_plane(output, rho_record, k, &rho, &mut bytes)?;
        write_record_plane(output, rr_record, k, &rr, &mut bytes)?;
    }

    let stretch = sealed_child_stretch_grid()?;
    let sigma = &stretch.sigma[1..=grid.nz()];
    let sigmaf = &stretch.sigmaf[1..=grid.nz() + 1];
    let bottom = extrapolation_weights(sigma, sigmaf, false)?;
    let top = extrapolation_weights(sigma, sigmaf, true)?;
    let nx = grid.nx();
    let mut zh = vec![0.0_f32; grid.nz()];
    let mut zf = vec![0.0_f32; grid.nz() + 1];
    let mut c1 = vec![0.0_f32; grid.nz() + 1];
    let mut c2 = vec![0.0_f32; grid.nz() + 1];
    for row_start in (0..grid.ny()).step_by(CHILD_AUXILIARY_STREAM_SLAB_ROWS) {
        let rows = CHILD_AUXILIARY_STREAM_SLAB_ROWS.min(grid.ny() - row_start);
        let slab_words = rows * nx;
        let mut rho = vec![0.0_f32; grid.nz() * slab_words];
        for k in 0..grid.nz() {
            let values = read_record_rows(output, rho_record, k, row_start, rows)?;
            rho[k * slab_words..(k + 1) * slab_words].copy_from_slice(&values);
        }
        let mut rf = vec![0.0_f32; (grid.nz() + 1) * slab_words];
        for local_word in 0..slab_words {
            let global_word = row_start * nx + local_word;
            fill_stretched_column_metrics(
                terrain[global_word],
                sigma,
                sigmaf,
                grid.ztop_m() as f32,
                &mut zh,
                &mut zf,
                &mut c1,
                &mut c2,
            )?;
            let density = |k: usize| rho[k * slab_words + local_word];
            rf[local_word] =
                bottom[0] * density(0) + bottom[1] * density(1) + bottom[2] * density(2);
            for face in 1..grid.nz() {
                rf[face * slab_words + local_word] =
                    c1[face] * density(face - 1) + c2[face] * density(face);
            }
            rf[grid.nz() * slab_words + local_word] = top[0] * density(grid.nz() - 1)
                + top[1] * density(grid.nz() - 2)
                + top[2] * density(grid.nz() - 3);
        }
        if rf.iter().any(|value| !value.is_finite() || *value <= 0.0) {
            return Err(invalid(
                "streamed live face-density reconstruction is non-positive",
            ));
        }
        for face in 0..=grid.nz() {
            write_record_rows(
                output,
                rf_record,
                face,
                row_start,
                rows,
                &rf[face * slab_words..(face + 1) * slab_words],
                &mut bytes,
            )?;
        }
    }
    Ok(())
}

/// Convert the exact static z-face height stream to binary32 geopotential in
/// one bounded chunk. Input and output files must already have their exact
/// admitted lengths.
pub fn write_streamed_geopotential(
    output: &mut File,
    static_zf: &mut File,
    grid: ChildGridSpec,
) -> Result<()> {
    let nbytes = child_auxiliary_base_layout_for_grid(grid)?.geopotential_nbytes;
    if static_zf.metadata()?.len() != nbytes || output.metadata()?.len() != nbytes {
        return Err(invalid(
            "streamed geopotential input/output length differs from the grid",
        ));
    }
    static_zf.seek(SeekFrom::Start(0))?;
    output.seek(SeekFrom::Start(0))?;
    let mut input = vec![0u8; 1024 * 1024];
    let mut encoded = Vec::with_capacity(input.len());
    let mut remaining = nbytes;
    while remaining != 0 {
        let count = usize::try_from(remaining.min(input.len() as u64))
            .expect("bounded geopotential chunk fits usize");
        static_zf.read_exact(&mut input[..count])?;
        if count % 4 != 0 {
            return Err(invalid("streamed static zf chunk is not f32 aligned"));
        }
        encoded.clear();
        for chunk in input[..count].chunks_exact(4) {
            let height = f32::from_le_bytes(chunk.try_into().expect("four-byte f32"));
            let geopotential = height * G;
            if !height.is_finite()
                || height < 0.0
                || !geopotential.is_finite()
                || geopotential < 0.0
            {
                return Err(invalid(
                    "streamed static zf/geopotential contains a non-physical word",
                ));
            }
            encoded.extend_from_slice(&geopotential.to_bits().to_le_bytes());
        }
        output.write_all(&encoded)?;
        remaining -= count as u64;
    }
    Ok(())
}

/// Reconstruct terrain-aware staggered Rayleigh base winds directly from the
/// exact static mass-height planes and the sealed one-dimensional source
/// profiles. No live velocity is consumed.
pub fn write_streamed_base_wind(
    output: &mut File,
    static_zh: &mut File,
    grid: ChildGridSpec,
    u_profile: &[f32],
    v_profile: &[f32],
) -> Result<()> {
    if u_profile.len() != grid.nz() || v_profile.len() != grid.nz() {
        return Err(invalid("streamed base-wind profile geometry is invalid"));
    }
    let layout = base_wind_stream_layout(grid)?;
    let expected_output = layout
        .last()
        .map(|record| record.offset_nbytes + record.nbytes)
        .ok_or_else(|| invalid("streamed base-wind layout is empty"))?;
    let expected_zh = checked_bytes(
        grid.nz() * grid.ny() * grid.nx(),
        "streamed static zh length",
    )?;
    if static_zh.metadata()?.len() != expected_zh || output.metadata()?.len() != expected_output {
        return Err(invalid(
            "streamed base-wind input/output length differs from the grid",
        ));
    }
    let u_record = &layout[0];
    let v_record = &layout[1];
    let mut bytes = Vec::new();
    for k in 0..grid.nz() {
        let zh = read_raw_f32_plane(static_zh, k, grid.ny(), grid.nx())?;
        if zh.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(invalid("streamed static zh contains a non-physical word"));
        }
        let mut u0 = Vec::with_capacity(grid.ny() * (grid.nx() + 1));
        for j in 0..grid.ny() {
            let row = j * grid.nx();
            for i in 0..=grid.nx() {
                let height = match i {
                    0 => zh[row],
                    value if value == grid.nx() => zh[row + grid.nx() - 1],
                    _ => f32_mul(0.5, f32_add(zh[row + i - 1], zh[row + i])),
                };
                u0.push(interpolate_source_profile(u_profile, height, "u0")?);
            }
        }
        let mut v0 = Vec::with_capacity((grid.ny() + 1) * grid.nx());
        for j in 0..=grid.ny() {
            for i in 0..grid.nx() {
                let height = match j {
                    0 => zh[i],
                    value if value == grid.ny() => zh[(grid.ny() - 1) * grid.nx() + i],
                    _ => f32_mul(
                        0.5,
                        f32_add(zh[(j - 1) * grid.nx() + i], zh[j * grid.nx() + i]),
                    ),
                };
                v0.push(interpolate_source_profile(v_profile, height, "v0")?);
            }
        }
        write_record_plane(output, u_record, k, &u0, &mut bytes)?;
        write_record_plane(output, v_record, k, &v0, &mut bytes)?;
    }
    Ok(())
}

fn required_record<'a>(
    records: &'a BTreeMap<&str, &StreamCarrierLayout>,
    name: &str,
) -> Result<&'a StreamCarrierLayout> {
    records
        .get(name)
        .copied()
        .ok_or_else(|| invalid(format!("streamed terrain/open layout lacks {name}")))
}

fn write_pressure_from_source<R: AtmospherePlaneReader>(
    file: &mut File,
    grid: ChildGridSpec,
    atmosphere: &mut R,
    source_name: &str,
    destination: &StreamCarrierLayout,
) -> Result<()> {
    let plane_words = (grid.ny() + 2) * (grid.nx() + 2);
    let mut bytes = Vec::new();
    write_zero_plane(file, destination, 0, plane_words, &mut bytes)?;
    for k in 0..grid.nz() {
        let physical = atmosphere.read_plane_f32(source_name, &[k])?;
        let padded = pressure_open_plane(&physical, grid.ny(), grid.nx())?;
        write_record_plane(file, destination, k + 1, &padded, &mut bytes)?;
    }
    write_zero_plane(file, destination, grid.nz() + 1, plane_words, &mut bytes)
}

fn write_pressure_diagnostics<R: AtmospherePlaneReader>(
    file: &mut File,
    grid: ChildGridSpec,
    atmosphere: &mut R,
    records: &BTreeMap<&str, &StreamCarrierLayout>,
) -> Result<()> {
    let nx = grid.nx();
    let ny = grid.ny();
    let nz = grid.nz();
    let pressure_words = (ny + 2) * (nx + 2);
    let mass_words = ny * nx;
    let piadv = required_record(records, "piadv")?;
    let ppterm = required_record(records, "ppterm")?;
    let thv = required_record(records, "thv")?;
    let pp3d = required_record(records, "pp3d")?;
    let base_pi = required_record(records, "pi0")?;
    let base_th = required_record(records, "th0")?;
    let mut bytes = Vec::new();
    for record in [piadv, ppterm, thv] {
        write_zero_plane(file, record, 0, pressure_words, &mut bytes)?;
        write_zero_plane(file, record, nz + 1, pressure_words, &mut bytes)?;
    }
    let reps = RV / RD;
    let cv = CP - RD;
    let cvv = CPV - RV;
    let rovcp = RD / CP;
    for k in 0..nz {
        let ppi_physical = atmosphere.read_plane_f32("ppi", &[k])?;
        let theta_physical = atmosphere.read_plane_f32("tha", &[k])?;
        let base_pi_physical = read_base_physical_plane(file, base_pi, k + 1, grid)?;
        let base_th_physical = read_base_physical_plane(file, base_th, k + 1, grid)?;
        let mut qv = atmosphere.read_plane_f32("qa", &[0, k])?;
        let mut liquid = vec![0.0_f32; mass_words];
        let mut solid = vec![0.0_f32; mass_words];
        for species in 1..=5 {
            let values = atmosphere.read_plane_f32("qa", &[species, k])?;
            let target = if species <= 2 {
                &mut liquid
            } else {
                &mut solid
            };
            for (accumulator, value) in target.iter_mut().zip(values) {
                *accumulator += value;
            }
        }
        for value in &mut qv {
            *value = value.max(0.0);
        }
        let mut thv_physical = vec![0.0_f32; mass_words];
        let mut ppterm_plane = vec![0.0_f32; pressure_words];
        for p in 0..mass_words {
            let total_pi = base_pi_physical[p] + ppi_physical[p];
            let total_theta = base_th_physical[p] + theta_physical[p];
            let qli = (liquid[p] + solid[p]).max(0.0);
            let cpli = CPL * liquid[p].max(0.0) + CPI * solid[p].max(0.0);
            thv_physical[p] = total_theta * (1.0 + reps * qv[p]) / (1.0 + qv[p] + qli);
            let cpm = CP + CPV * qv[p] + cpli;
            let cvm = 1.0 / (cv + cvv * qv[p] + cpli);
            let j = p / nx;
            let i = p % nx;
            ppterm_plane[(j + 1) * (nx + 2) + i + 1] = total_pi * rovcp * cpm * cvm;
        }
        let thv_plane = pressure_open_plane(&thv_physical, ny, nx)?;
        write_record_plane(file, thv, k + 1, &thv_plane, &mut bytes)?;
        write_record_plane(file, ppterm, k + 1, &ppterm_plane, &mut bytes)?;

        let pp3d_plane = read_record_plane(file, pp3d, k + 1)?;
        let base_live = read_base_pressure_live_plane(file, base_pi, k + 1, grid)?;
        let total = base_live
            .into_iter()
            .zip(pp3d_plane)
            .map(|(base, perturbation)| base + perturbation)
            .collect::<Vec<_>>();
        write_record_plane(file, piadv, k + 1, &total, &mut bytes)?;
    }
    write_zero_record(file, required_record(records, "ppten")?)
}

fn write_scalar_from_source<R: AtmospherePlaneReader>(
    file: &mut File,
    grid: ChildGridSpec,
    atmosphere: &mut R,
    source_name: &str,
    destination: &StreamCarrierLayout,
) -> Result<()> {
    let plane_words = (grid.ny() + 2 * NGXY) * (grid.nx() + 2 * NGXY);
    let mut bytes = Vec::new();
    write_zero_plane(file, destination, 0, plane_words, &mut bytes)?;
    for k in 0..grid.nz() {
        let physical = atmosphere.read_plane_f32(source_name, &[k])?;
        let padded = scalar_open_plane(&physical, grid.ny(), grid.nx())?;
        write_record_plane(file, destination, k + 1, &padded, &mut bytes)?;
    }
    write_zero_plane(file, destination, grid.nz() + 1, plane_words, &mut bytes)
}

fn write_u_from_source<R: AtmospherePlaneReader>(
    file: &mut File,
    grid: ChildGridSpec,
    atmosphere: &mut R,
    destination: &StreamCarrierLayout,
) -> Result<()> {
    let plane_words = (grid.ny() + 2 * NGXY) * (grid.nx() + 1 + 2 * NGXY);
    let mut bytes = Vec::new();
    write_zero_plane(file, destination, 0, plane_words, &mut bytes)?;
    for k in 0..grid.nz() {
        let physical = atmosphere.read_plane_f32("ua", &[k])?;
        let padded = u_open_plane(&physical, grid.ny(), grid.nx())?;
        write_record_plane(file, destination, k + 1, &padded, &mut bytes)?;
    }
    write_zero_plane(file, destination, grid.nz() + 1, plane_words, &mut bytes)
}

fn write_v_from_source<R: AtmospherePlaneReader>(
    file: &mut File,
    grid: ChildGridSpec,
    atmosphere: &mut R,
    destination: &StreamCarrierLayout,
) -> Result<()> {
    let plane_words = (grid.ny() + 1 + 2 * NGXY) * (grid.nx() + 2 * NGXY);
    let mut bytes = Vec::new();
    write_zero_plane(file, destination, 0, plane_words, &mut bytes)?;
    for k in 0..grid.nz() {
        let physical = atmosphere.read_plane_f32("va", &[k])?;
        let padded = v_open_plane(&physical, grid.ny(), grid.nx())?;
        write_record_plane(file, destination, k + 1, &padded, &mut bytes)?;
    }
    write_zero_plane(file, destination, grid.nz() + 1, plane_words, &mut bytes)
}

fn write_w_from_source<R: AtmospherePlaneReader>(
    file: &mut File,
    grid: ChildGridSpec,
    terrain: &[f32],
    atmosphere: &mut R,
    destination: &StreamCarrierLayout,
) -> Result<()> {
    let plane_words = (grid.ny() + 2 * NGXY) * (grid.nx() + 2 * NGXY);
    let mut bytes = Vec::new();
    write_zero_plane(file, destination, 0, plane_words, &mut bytes)?;
    let u0 = u_open_plane(
        &atmosphere.read_plane_f32("ua", &[0])?,
        grid.ny(),
        grid.nx(),
    )?;
    let v0 = v_open_plane(
        &atmosphere.read_plane_f32("va", &[0])?,
        grid.ny(),
        grid.nx(),
    )?;
    let bottom = w_surface_plane(&u0, &v0, terrain, grid)?;
    write_record_plane(file, destination, 1, &bottom, &mut bytes)?;
    for face in 1..grid.nz() {
        let physical = atmosphere.read_plane_f32("wa", &[face])?;
        let padded = w_open_plane(&physical, grid.ny(), grid.nx())?;
        write_record_plane(file, destination, face + 1, &padded, &mut bytes)?;
    }
    write_zero_plane(file, destination, grid.nz() + 1, plane_words, &mut bytes)?;
    write_zero_plane(file, destination, grid.nz() + 2, plane_words, &mut bytes)
}

fn pressure_open_plane(physical: &[f32], ny: usize, nx: usize) -> Result<Vec<f32>> {
    if physical.len() != ny * nx {
        return Err(invalid("pressure source plane geometry is invalid"));
    }
    let out_nx = nx + 2;
    let mut output = vec![0.0_f32; (ny + 2) * out_nx];
    for j in 0..ny {
        output[(j + 1) * out_nx + 1..(j + 1) * out_nx + nx + 1]
            .copy_from_slice(&physical[j * nx..(j + 1) * nx]);
    }
    for j in 0..=ny + 1 {
        output[j * out_nx] = output[j * out_nx + 1];
        output[j * out_nx + nx + 1] = output[j * out_nx + nx];
    }
    let south = output[out_nx..2 * out_nx].to_vec();
    output[..out_nx].copy_from_slice(&south);
    let north = output[ny * out_nx..(ny + 1) * out_nx].to_vec();
    output[(ny + 1) * out_nx..(ny + 2) * out_nx].copy_from_slice(&north);
    Ok(output)
}

fn scalar_open_plane(physical: &[f32], ny: usize, nx: usize) -> Result<Vec<f32>> {
    if physical.len() != ny * nx {
        return Err(invalid("scalar source plane geometry is invalid"));
    }
    let out_ny = ny + 2 * NGXY;
    let out_nx = nx + 2 * NGXY;
    let mut output = vec![0.0_f32; out_ny * out_nx];
    for j in 0..ny {
        output[(j + NGXY) * out_nx + NGXY..(j + NGXY) * out_nx + NGXY + nx]
            .copy_from_slice(&physical[j * nx..(j + 1) * nx]);
    }
    for j_native in 0..=ny as isize + 1 {
        let j = native_xy(j_native);
        for i_native in (1 - NGXY as isize)..=0 {
            output[j * out_nx + native_xy(i_native)] = output[j * out_nx + native_xy(1)];
        }
        for i_native in nx as isize + 1..=nx as isize + NGXY as isize {
            output[j * out_nx + native_xy(i_native)] = output[j * out_nx + native_xy(nx as isize)];
        }
    }
    for j_native in (1 - NGXY as isize)..=0 {
        for i_native in 0..=nx as isize + 1 {
            output[native_xy(j_native) * out_nx + native_xy(i_native)] =
                output[native_xy(1) * out_nx + native_xy(i_native)];
        }
    }
    for j_native in ny as isize + 1..=ny as isize + NGXY as isize {
        for i_native in 0..=nx as isize + 1 {
            output[native_xy(j_native) * out_nx + native_xy(i_native)] =
                output[native_xy(ny as isize) * out_nx + native_xy(i_native)];
        }
    }
    Ok(output)
}

fn u_open_plane(physical: &[f32], ny: usize, nx: usize) -> Result<Vec<f32>> {
    if physical.len() != ny * (nx + 1) {
        return Err(invalid("u source plane geometry is invalid"));
    }
    let out_ny = ny + 2 * NGXY;
    let out_nx = nx + 1 + 2 * NGXY;
    let mut output = vec![0.0_f32; out_ny * out_nx];
    for j in 0..ny {
        output[(j + NGXY) * out_nx + NGXY..(j + NGXY) * out_nx + NGXY + nx + 1]
            .copy_from_slice(&physical[j * (nx + 1)..(j + 1) * (nx + 1)]);
    }
    for j_native in 0..=ny as isize + 1 {
        let j = native_xy(j_native);
        for i_native in (1 - NGXY as isize)..=0 {
            output[j * out_nx + native_xy(i_native)] = output[j * out_nx + native_xy(1)];
        }
        for i_native in nx as isize + 2..=nx as isize + 1 + NGXY as isize {
            output[j * out_nx + native_xy(i_native)] =
                output[j * out_nx + native_xy(nx as isize + 1)];
        }
    }
    for j_native in (1 - NGXY as isize)..=0 {
        for i_native in 0..=nx as isize + 2 {
            output[native_xy(j_native) * out_nx + native_xy(i_native)] =
                output[native_xy(1) * out_nx + native_xy(i_native)];
        }
    }
    for j_native in ny as isize + 1..=ny as isize + NGXY as isize {
        for i_native in 0..=nx as isize + 2 {
            output[native_xy(j_native) * out_nx + native_xy(i_native)] =
                output[native_xy(ny as isize) * out_nx + native_xy(i_native)];
        }
    }
    Ok(output)
}

fn v_open_plane(physical: &[f32], ny: usize, nx: usize) -> Result<Vec<f32>> {
    if physical.len() != (ny + 1) * nx {
        return Err(invalid("v source plane geometry is invalid"));
    }
    let out_ny = ny + 1 + 2 * NGXY;
    let out_nx = nx + 2 * NGXY;
    let mut output = vec![0.0_f32; out_ny * out_nx];
    for j in 0..=ny {
        output[(j + NGXY) * out_nx + NGXY..(j + NGXY) * out_nx + NGXY + nx]
            .copy_from_slice(&physical[j * nx..(j + 1) * nx]);
    }
    for j_native in (1 - NGXY as isize)..=0 {
        for i_native in 0..=nx as isize + 1 {
            output[native_xy(j_native) * out_nx + native_xy(i_native)] =
                output[native_xy(1) * out_nx + native_xy(i_native)];
        }
    }
    for j_native in ny as isize + 2..=ny as isize + 1 + NGXY as isize {
        for i_native in 0..=nx as isize + 1 {
            output[native_xy(j_native) * out_nx + native_xy(i_native)] =
                output[native_xy(ny as isize + 1) * out_nx + native_xy(i_native)];
        }
    }
    for j_native in 0..=ny as isize + 2 {
        let j = native_xy(j_native);
        for i_native in (1 - NGXY as isize)..=0 {
            output[j * out_nx + native_xy(i_native)] = output[j * out_nx + native_xy(1)];
        }
        for i_native in nx as isize + 1..=nx as isize + NGXY as isize {
            output[j * out_nx + native_xy(i_native)] = output[j * out_nx + native_xy(nx as isize)];
        }
    }
    Ok(output)
}

fn w_open_plane(physical: &[f32], ny: usize, nx: usize) -> Result<Vec<f32>> {
    if physical.len() != ny * nx {
        return Err(invalid("w source plane geometry is invalid"));
    }
    scalar_open_plane(physical, ny, nx)
}

fn w_surface_plane(u: &[f32], v: &[f32], terrain: &[f32], grid: ChildGridSpec) -> Result<Vec<f32>> {
    let nx = grid.nx();
    let ny = grid.ny();
    let out_nx = nx + 2 * NGXY;
    let u_nx = nx + 1 + 2 * NGXY;
    let v_nx = nx + 2 * NGXY;
    if u.len() != (ny + 2 * NGXY) * u_nx
        || v.len() != (ny + 1 + 2 * NGXY) * v_nx
        || terrain.len() != ny * nx
    {
        return Err(invalid("w surface inputs have invalid geometry"));
    }
    let mut output = vec![0.0_f32; (ny + 2 * NGXY) * out_nx];
    for j_native in 0..=ny as isize + 1 {
        let j = native_xy(j_native);
        for i_native in 0..=nx as isize + 1 {
            let i = native_xy(i_native);
            let (gz, dzdx, dzdy) = surface_metric_triplet(terrain, grid, j, i);
            let u_sum = u[j * u_nx + i] + u[j * u_nx + i + 1];
            let v_sum = v[j * v_nx + i] + v[(j + 1) * v_nx + i];
            output[j * out_nx + i] = 0.5 * (u_sum * dzdx + v_sum * dzdy) * gz;
        }
    }
    Ok(output)
}

fn surface_metric_triplet(
    terrain: &[f32],
    grid: ChildGridSpec,
    jh: usize,
    ih: usize,
) -> (f32, f32, f32) {
    let nx = grid.nx();
    let ny = grid.ny();
    let at = |i: isize, j: isize| {
        let i = i.clamp(0, nx as isize - 1) as usize;
        let j = j.clamp(0, ny as isize - 1) as usize;
        terrain[j * nx + i]
    };
    let x_flux = |face_i: isize, j: isize| {
        flx6(
            at(face_i - 3, j),
            at(face_i - 2, j),
            at(face_i - 1, j),
            at(face_i, j),
            at(face_i + 1, j),
            at(face_i + 2, j),
        )
    };
    let y_flux = |i: isize, face_j: isize| {
        flx6(
            at(i, face_j - 3),
            at(i, face_j - 2),
            at(i, face_j - 1),
            at(i, face_j),
            at(i, face_j + 1),
            at(i, face_j + 2),
        )
    };
    let global_i = ih as isize - NGXY as isize;
    let global_j = jh as isize - NGXY as isize;
    let bounded_i = global_i.clamp(0, nx as isize - 1);
    let bounded_j = global_j.clamp(0, ny as isize - 1);
    let surface = at(global_i, global_j);
    let gz = grid.ztop_m() as f32 / (grid.ztop_m() as f32 - surface);
    let dzdx = (x_flux(bounded_i + 1, bounded_j) - x_flux(bounded_i, bounded_j))
        * (1.0 / grid.dx_m() as f32);
    let dzdy = (y_flux(bounded_i, bounded_j + 1) - y_flux(bounded_i, bounded_j))
        * (1.0 / grid.dy_m() as f32);
    (gz, dzdx, dzdy)
}

fn native_xy(value: isize) -> usize {
    usize::try_from(value + NGXY as isize - 1).expect("native index is inside allocation")
}

fn read_base_physical_plane(
    file: &mut File,
    record: &StreamCarrierLayout,
    k: usize,
    grid: ChildGridSpec,
) -> Result<Vec<f32>> {
    read_base_subplane(file, record, k, grid, NGXY, NGXY, grid.ny(), grid.nx())
}

fn read_base_pressure_live_plane(
    file: &mut File,
    record: &StreamCarrierLayout,
    k: usize,
    grid: ChildGridSpec,
) -> Result<Vec<f32>> {
    read_base_subplane(
        file,
        record,
        k,
        grid,
        NGXY - 1,
        NGXY - 1,
        grid.ny() + 2,
        grid.nx() + 2,
    )
}

#[allow(clippy::too_many_arguments)]
fn read_base_subplane(
    file: &mut File,
    record: &StreamCarrierLayout,
    k: usize,
    grid: ChildGridSpec,
    j_start: usize,
    i_start: usize,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>> {
    let halo_ny = grid.ny() + 2 * NGXY;
    let halo_nx = grid.nx() + 2 * NGXY;
    if record.shape != [grid.nz() + 2, halo_ny, halo_nx] || k >= grid.nz() + 2 {
        return Err(invalid("base subplane record geometry is invalid"));
    }
    let mut output = Vec::with_capacity(rows * columns);
    let mut bytes = vec![0u8; columns * 4];
    for row in 0..rows {
        let word = (k * halo_ny + j_start + row) * halo_nx + i_start;
        file.seek(SeekFrom::Start(
            record.offset_nbytes + checked_bytes(word, "base subplane offset")?,
        ))?;
        file.read_exact(&mut bytes)?;
        output.extend(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32"))),
        );
    }
    Ok(output)
}

fn read_record_plane(
    file: &mut File,
    record: &StreamCarrierLayout,
    plane: usize,
) -> Result<Vec<f32>> {
    if record.shape.len() != 3 || plane >= record.shape[0] {
        return Err(invalid("streamed record plane geometry is invalid"));
    }
    let plane_words = record.shape[1] * record.shape[2];
    file.seek(SeekFrom::Start(
        record.offset_nbytes + checked_bytes(plane * plane_words, "record plane offset")?,
    ))?;
    let mut bytes = vec![0u8; plane_words * 4];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect())
}

fn read_record_rows(
    file: &mut File,
    record: &StreamCarrierLayout,
    plane: usize,
    row_start: usize,
    rows: usize,
) -> Result<Vec<f32>> {
    if record.shape.len() != 3
        || plane >= record.shape[0]
        || rows == 0
        || row_start
            .checked_add(rows)
            .is_none_or(|end| end > record.shape[1])
    {
        return Err(invalid("streamed record row geometry is invalid"));
    }
    let row_words = record.shape[2];
    let plane_words = record.shape[1] * row_words;
    let start_word = plane * plane_words + row_start * row_words;
    file.seek(SeekFrom::Start(
        record.offset_nbytes + checked_bytes(start_word, "record row offset")?,
    ))?;
    let mut bytes = vec![0u8; rows * row_words * 4];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect())
}

fn write_record_plane(
    file: &mut File,
    record: &StreamCarrierLayout,
    plane: usize,
    values: &[f32],
    bytes: &mut Vec<u8>,
) -> Result<()> {
    if record.shape.len() != 3
        || plane >= record.shape[0]
        || values.len() != record.shape[1] * record.shape[2]
    {
        return Err(invalid(format!(
            "streamed carrier {} plane geometry is invalid",
            record.name
        )));
    }
    write_f32_at(
        file,
        record.offset_nbytes + checked_bytes(plane * values.len(), "record plane offset")?,
        values,
        bytes,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_record_rows(
    file: &mut File,
    record: &StreamCarrierLayout,
    plane: usize,
    row_start: usize,
    rows: usize,
    values: &[f32],
    bytes: &mut Vec<u8>,
) -> Result<()> {
    if record.shape.len() != 3
        || plane >= record.shape[0]
        || rows == 0
        || row_start
            .checked_add(rows)
            .is_none_or(|end| end > record.shape[1])
        || values.len() != rows * record.shape[2]
    {
        return Err(invalid(format!(
            "streamed carrier {} row geometry is invalid",
            record.name
        )));
    }
    let plane_words = record.shape[1] * record.shape[2];
    let start_word = plane * plane_words + row_start * record.shape[2];
    write_f32_at(
        file,
        record.offset_nbytes + checked_bytes(start_word, "record row offset")?,
        values,
        bytes,
    )
}

fn write_zero_plane(
    file: &mut File,
    record: &StreamCarrierLayout,
    plane: usize,
    plane_words: usize,
    bytes: &mut Vec<u8>,
) -> Result<()> {
    write_record_plane(file, record, plane, &vec![0.0_f32; plane_words], bytes)
}

fn write_zero_record(file: &mut File, record: &StreamCarrierLayout) -> Result<()> {
    file.seek(SeekFrom::Start(record.offset_nbytes))?;
    let buffer = vec![0u8; 1024 * 1024];
    let mut remaining = record.nbytes;
    while remaining != 0 {
        let count = usize::try_from(remaining.min(buffer.len() as u64))
            .expect("bounded zero write fits usize");
        file.write_all(&buffer[..count])?;
        remaining -= count as u64;
    }
    Ok(())
}

fn copy_record(
    file: &mut File,
    source: &StreamCarrierLayout,
    destination: &StreamCarrierLayout,
) -> Result<()> {
    if source.shape != destination.shape || source.nbytes != destination.nbytes {
        return Err(invalid("streamed carrier copy geometry differs"));
    }
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut copied = 0u64;
    while copied != source.nbytes {
        let count = usize::try_from((source.nbytes - copied).min(buffer.len() as u64))
            .expect("bounded carrier copy fits usize");
        file.seek(SeekFrom::Start(source.offset_nbytes + copied))?;
        file.read_exact(&mut buffer[..count])?;
        file.seek(SeekFrom::Start(destination.offset_nbytes + copied))?;
        file.write_all(&buffer[..count])?;
        copied += count as u64;
    }
    Ok(())
}

fn flx6(s1: f32, s2: f32, s3: f32, s4: f32, s5: f32, s6: f32) -> f32 {
    (37.0_f32 / 60.0_f32) * (s4 + s3)
        + (-8.0_f32 / 60.0_f32) * (s5 + s2)
        + (1.0_f32 / 60.0_f32) * (s6 + s1)
}

fn read_raw_f32_plane(
    file: &mut File,
    plane: usize,
    rows: usize,
    columns: usize,
) -> Result<Vec<f32>> {
    let plane_words = rows
        .checked_mul(columns)
        .ok_or_else(|| invalid("raw f32 plane word count overflows"))?;
    file.seek(SeekFrom::Start(checked_bytes(
        plane
            .checked_mul(plane_words)
            .ok_or_else(|| invalid("raw f32 plane offset overflows"))?,
        "raw f32 plane offset",
    )?))?;
    let mut bytes = vec![0u8; plane_words * 4];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect())
}

#[inline(always)]
fn f32_add(left: f32, right: f32) -> f32 {
    f32::from_bits((left + right).to_bits())
}

#[inline(always)]
fn f32_mul(left: f32, right: f32) -> f32 {
    f32::from_bits((left * right).to_bits())
}

fn checked_bytes(words: usize, label: &str) -> Result<u64> {
    (words as u64)
        .checked_mul(4)
        .ok_or_else(|| invalid(format!("{label} byte count overflows")))
}

fn write_f32_at(
    file: &mut File,
    offset_nbytes: u64,
    values: &[f32],
    bytes: &mut Vec<u8>,
) -> Result<()> {
    bytes.clear();
    bytes.reserve(
        values
            .len()
            .saturating_mul(4)
            .saturating_sub(bytes.capacity()),
    );
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    file.seek(SeekFrom::Start(offset_nbytes))?;
    file.write_all(bytes)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn publish_child_auxiliary_stream_new(
    grid: ChildGridSpec,
    atmosphere_directory: impl AsRef<Path>,
    terrain_directory: impl AsRef<Path>,
    static_height_directory: impl AsRef<Path>,
    static_validation_receipt: impl AsRef<Path>,
    source_product_restart: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
    free_bytes: u64,
    reserve_bytes: u64,
) -> Result<ChildAuxiliaryStreamReceipt> {
    let preset = admitted_child_auxiliary_stream_preset(grid)?;
    let capacity = admit_child_auxiliary_base_capacity(grid, free_bytes, reserve_bytes, 1)?;
    let atmosphere_identity = StreamAtmosphereIdentity {
        manifest_sha256: preset.atmosphere_manifest_sha256.to_owned(),
        arena_sha256: preset.atmosphere_arena_sha256.to_owned(),
        receipt_sha256: preset.atmosphere_receipt_sha256.to_owned(),
    };
    let mut atmosphere = FileAtmospherePlaneReader::open_verified(
        atmosphere_directory.as_ref(),
        grid,
        &atmosphere_identity,
    )?;
    let terrain = load_verified_terrain(terrain_directory.as_ref(), preset)?;
    verify_static_height_inputs(
        static_height_directory.as_ref(),
        static_validation_receipt.as_ref(),
        preset,
    )?;
    let source = load_source_profiles(source_product_restart.as_ref(), grid)?;
    let producer_source_sha256 = hash_regular_file(
        producer_source.as_ref(),
        Some(16 * 1024 * 1024),
        "streamed auxiliary producer source",
    )?
    .0;
    let producer_executable_sha256 = hash_regular_file(
        producer_executable.as_ref(),
        Some(128 * 1024 * 1024),
        "streamed auxiliary producer executable",
    )?
    .0;

    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace streamed auxiliary destination {}",
            output.display()
        )));
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(
            "streamed auxiliary destination parent does not exist",
        ));
    }
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("streamed auxiliary destination needs a UTF-8 name"))?;
    let staging = output.with_file_name(format!(
        ".{name}.partial-{}-{}",
        std::process::id(),
        STREAM_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&staging)?;
    let result = (|| -> Result<ChildAuxiliaryStreamReceipt> {
        let layout = child_auxiliary_base_layout_for_grid(grid)?;
        let terrain_path = staging.join(CHILD_AUXILIARY_TERRAIN_OPEN_FILE);
        let mut terrain_open = create_sized_file(&terrain_path, layout.terrain_open_nbytes)?;
        write_streamed_base_and_surface_regions(&mut terrain_open, grid, &terrain, &source.base)?;
        write_streamed_mutable_regions(&mut terrain_open, grid, &terrain, &mut atmosphere)?;
        terrain_open.sync_all()?;

        let density_path = staging.join(CHILD_AUXILIARY_LIVE_DENSITY_FILE);
        let mut density = create_sized_file(&density_path, layout.live_density_nbytes)?;
        write_streamed_live_density(
            &mut density,
            &mut terrain_open,
            grid,
            &terrain,
            &mut atmosphere,
        )?;
        density.sync_all()?;

        let static_directory = static_height_directory.as_ref();
        let geopotential_path = staging.join(CHILD_AUXILIARY_GEOPOTENTIAL_FILE);
        let mut geopotential = create_sized_file(&geopotential_path, layout.geopotential_nbytes)?;
        let mut zf = File::open(static_directory.join("zf.f32le"))?;
        write_streamed_geopotential(&mut geopotential, &mut zf, grid)?;
        geopotential.sync_all()?;

        let base_wind_path = staging.join(CHILD_BASE_WIND_FILE);
        let mut base_wind = create_sized_file(&base_wind_path, layout.base_wind_nbytes)?;
        let mut zh = File::open(static_directory.join("zh.f32le"))?;
        write_streamed_base_wind(&mut base_wind, &mut zh, grid, &source.u0, &source.v0)?;
        base_wind.sync_all()?;
        drop((terrain_open, density, geopotential, base_wind, zf, zh));

        let payloads = vec![
            hash_payload_file(
                &terrain_path,
                CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
                &terrain_open_stream_layout(grid)?,
            )?,
            hash_payload_file(
                &density_path,
                CHILD_AUXILIARY_LIVE_DENSITY_FILE,
                &live_density_stream_layout(grid)?,
            )?,
            hash_payload_file(
                &geopotential_path,
                CHILD_AUXILIARY_GEOPOTENTIAL_FILE,
                &[StreamCarrierLayout {
                    name: "total_geopotential_zf",
                    shape: vec![grid.nz() + 1, grid.ny(), grid.nx()],
                    offset_nbytes: 0,
                    nbytes: layout.geopotential_nbytes,
                }],
            )?,
            hash_payload_file(
                &base_wind_path,
                CHILD_BASE_WIND_FILE,
                &base_wind_stream_layout(grid)?,
            )?,
        ];
        let manifest = stream_manifest_value(
            preset,
            &atmosphere_identity,
            &source,
            &payloads,
            capacity.free_bytes,
            capacity.reserve_bytes,
            capacity.post_publish_free_bytes,
            &producer_source_sha256,
            &producer_executable_sha256,
        );
        let manifest_bytes = manifest.canonical_bytes();
        let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
        let template = stream_receipt_from_payloads(
            &payloads,
            manifest_sha256,
            producer_source_sha256.clone(),
            producer_executable_sha256.clone(),
        )?;
        let receipt_bytes = stream_receipt_value(&template).canonical_bytes();
        let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
        write_new_synced(
            &staging.join(CHILD_AUXILIARY_STREAM_MANIFEST_FILE),
            &manifest_bytes,
        )?;
        write_new_synced(
            &staging.join(CHILD_AUXILIARY_STREAM_RECEIPT_FILE),
            &receipt_bytes,
        )?;
        verify_stream_output_inventory(&staging)?;
        fs::rename(&staging, output)?;
        Ok(ChildAuxiliaryStreamReceipt {
            receipt_sha256,
            ..template
        })
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub fn inspect_child_auxiliary_stream(
    grid: ChildGridSpec,
    atmosphere_directory: impl AsRef<Path>,
    terrain_directory: impl AsRef<Path>,
    static_height_directory: impl AsRef<Path>,
    static_validation_receipt: impl AsRef<Path>,
    source_product_restart: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    artifact_directory: impl AsRef<Path>,
    free_bytes: u64,
    reserve_bytes: u64,
    expected_manifest_sha256: &str,
    expected_terrain_open_sha256: &str,
    expected_live_density_sha256: &str,
    expected_geopotential_sha256: &str,
    expected_base_wind_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<ChildAuxiliaryStreamReceipt> {
    for (value, label) in [
        (expected_manifest_sha256, "streamed auxiliary manifest"),
        (
            expected_terrain_open_sha256,
            "streamed auxiliary terrain/open",
        ),
        (
            expected_live_density_sha256,
            "streamed auxiliary live density",
        ),
        (
            expected_geopotential_sha256,
            "streamed auxiliary geopotential",
        ),
        (expected_base_wind_sha256, "streamed auxiliary base wind"),
        (expected_receipt_sha256, "streamed auxiliary receipt"),
    ] {
        validate_sha256(value, label)?;
    }
    let preset = admitted_child_auxiliary_stream_preset(grid)?;
    let capacity = admit_child_auxiliary_base_capacity(grid, free_bytes, reserve_bytes, 1)?;
    let atmosphere_identity = StreamAtmosphereIdentity {
        manifest_sha256: preset.atmosphere_manifest_sha256.to_owned(),
        arena_sha256: preset.atmosphere_arena_sha256.to_owned(),
        receipt_sha256: preset.atmosphere_receipt_sha256.to_owned(),
    };
    let _atmosphere = FileAtmospherePlaneReader::open_verified(
        atmosphere_directory.as_ref(),
        grid,
        &atmosphere_identity,
    )?;
    let _terrain = load_verified_terrain(terrain_directory.as_ref(), preset)?;
    verify_static_height_inputs(
        static_height_directory.as_ref(),
        static_validation_receipt.as_ref(),
        preset,
    )?;
    let source = load_source_profiles(source_product_restart.as_ref(), grid)?;
    let producer_source_sha256 = hash_regular_file(
        producer_source.as_ref(),
        Some(16 * 1024 * 1024),
        "streamed auxiliary producer source",
    )?
    .0;
    let producer_executable_sha256 = hash_regular_file(
        producer_executable.as_ref(),
        Some(128 * 1024 * 1024),
        "streamed auxiliary producer executable",
    )?
    .0;
    let artifact = artifact_directory.as_ref();
    verify_stream_output_inventory(artifact)?;
    let layout = child_auxiliary_base_layout_for_grid(grid)?;
    let payloads = vec![
        hash_payload_file(
            &artifact.join(CHILD_AUXILIARY_TERRAIN_OPEN_FILE),
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            &terrain_open_stream_layout(grid)?,
        )?,
        hash_payload_file(
            &artifact.join(CHILD_AUXILIARY_LIVE_DENSITY_FILE),
            CHILD_AUXILIARY_LIVE_DENSITY_FILE,
            &live_density_stream_layout(grid)?,
        )?,
        hash_payload_file(
            &artifact.join(CHILD_AUXILIARY_GEOPOTENTIAL_FILE),
            CHILD_AUXILIARY_GEOPOTENTIAL_FILE,
            &[StreamCarrierLayout {
                name: "total_geopotential_zf",
                shape: vec![grid.nz() + 1, grid.ny(), grid.nx()],
                offset_nbytes: 0,
                nbytes: layout.geopotential_nbytes,
            }],
        )?,
        hash_payload_file(
            &artifact.join(CHILD_BASE_WIND_FILE),
            CHILD_BASE_WIND_FILE,
            &base_wind_stream_layout(grid)?,
        )?,
    ];
    let observed_hashes = [
        payloads[0].sha256.as_str(),
        payloads[1].sha256.as_str(),
        payloads[2].sha256.as_str(),
        payloads[3].sha256.as_str(),
    ];
    let expected_hashes = [
        expected_terrain_open_sha256,
        expected_live_density_sha256,
        expected_geopotential_sha256,
        expected_base_wind_sha256,
    ];
    if observed_hashes != expected_hashes {
        return Err(invalid(
            "streamed auxiliary payload differs from an independent whole-file pin",
        ));
    }
    let expected_manifest = stream_manifest_value(
        preset,
        &atmosphere_identity,
        &source,
        &payloads,
        capacity.free_bytes,
        capacity.reserve_bytes,
        capacity.post_publish_free_bytes,
        &producer_source_sha256,
        &producer_executable_sha256,
    )
    .canonical_bytes();
    let observed_manifest = read_regular_bounded(
        &artifact.join(CHILD_AUXILIARY_STREAM_MANIFEST_FILE),
        4 * 1024 * 1024,
        "streamed auxiliary manifest",
    )?;
    if sha256::digest_hex(&observed_manifest) != expected_manifest_sha256
        || observed_manifest != expected_manifest
    {
        return Err(invalid(
            "streamed auxiliary manifest differs from exact recomputation",
        ));
    }
    let template = stream_receipt_from_payloads(
        &payloads,
        expected_manifest_sha256.to_owned(),
        producer_source_sha256,
        producer_executable_sha256,
    )?;
    let expected_receipt = stream_receipt_value(&template).canonical_bytes();
    let observed_receipt = read_regular_bounded(
        &artifact.join(CHILD_AUXILIARY_STREAM_RECEIPT_FILE),
        1024 * 1024,
        "streamed auxiliary receipt",
    )?;
    if sha256::digest_hex(&observed_receipt) != expected_receipt_sha256
        || observed_receipt != expected_receipt
    {
        return Err(invalid(
            "streamed auxiliary receipt differs from exact recomputation",
        ));
    }
    Ok(ChildAuxiliaryStreamReceipt {
        receipt_sha256: expected_receipt_sha256.to_owned(),
        ..template
    })
}

fn load_verified_terrain(
    directory: &Path,
    preset: AdmittedChildAuxiliaryStreamPreset,
) -> Result<Vec<f32>> {
    verify_exact_inventory(
        directory,
        &["manifest.json", "receipt.json", "terrain-msl.f32le"],
        "streamed child terrain",
    )?;
    require_hash(
        &directory.join("manifest.json"),
        preset.terrain_manifest_sha256,
        Some(1024 * 1024),
        "streamed child terrain manifest",
    )?;
    require_hash(
        &directory.join("receipt.json"),
        preset.terrain_receipt_sha256,
        Some(1024 * 1024),
        "streamed child terrain receipt",
    )?;
    let payload_path = directory.join("terrain-msl.f32le");
    let expected_nbytes = checked_bytes(
        preset.grid.ny() * preset.grid.nx(),
        "streamed terrain payload",
    )?;
    let (actual_hash, actual_nbytes) = hash_regular_file(
        &payload_path,
        Some(expected_nbytes),
        "streamed child terrain payload",
    )?;
    if actual_hash != preset.terrain_payload_sha256 || actual_nbytes != expected_nbytes {
        return Err(invalid(
            "streamed child terrain payload differs from its exact pin",
        ));
    }
    let mut file = File::open(payload_path)?;
    let mut bytes = vec![
        0u8;
        usize::try_from(expected_nbytes).map_err(|_| {
            invalid("streamed child terrain payload does not fit host address space")
        })?
    ];
    file.read_exact(&mut bytes)?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect::<Vec<_>>();
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid(
            "streamed child terrain contains a non-physical word",
        ));
    }
    Ok(values)
}

fn verify_static_height_inputs(
    directory: &Path,
    validation_receipt: &Path,
    preset: AdmittedChildAuxiliaryStreamPreset,
) -> Result<()> {
    if preset.static_receipt_sha256.is_some() {
        verify_exact_inventory(
            directory,
            &["manifest.json", "receipt.json", "zf.f32le", "zh.f32le"],
            "streamed child static heights",
        )?;
    } else {
        verify_exact_inventory(
            directory,
            &["manifest.json", "zf.f32le", "zh.f32le"],
            "streamed child static heights",
        )?;
    }
    require_hash(
        &directory.join("manifest.json"),
        preset.static_manifest_sha256,
        Some(1024 * 1024),
        "streamed child static manifest",
    )?;
    if let Some(expected) = preset.static_receipt_sha256 {
        require_hash(
            &directory.join("receipt.json"),
            expected,
            Some(1024 * 1024),
            "streamed child static receipt",
        )?;
    }
    require_hash(
        validation_receipt,
        preset.static_validation_sha256,
        Some(1024 * 1024),
        "streamed child static validation receipt",
    )?;
    for (name, expected, words) in [
        (
            "zh.f32le",
            preset.static_zh_sha256,
            preset.grid.nz() * preset.grid.ny() * preset.grid.nx(),
        ),
        (
            "zf.f32le",
            preset.static_zf_sha256,
            (preset.grid.nz() + 1) * preset.grid.ny() * preset.grid.nx(),
        ),
    ] {
        let expected_nbytes = checked_bytes(words, name)?;
        let (actual, nbytes) = hash_regular_file(
            &directory.join(name),
            Some(expected_nbytes),
            &format!("streamed child static {name}"),
        )?;
        if actual != expected || nbytes != expected_nbytes {
            return Err(invalid(format!(
                "streamed child static {name} differs from its exact pin"
            )));
        }
    }
    Ok(())
}

fn load_source_profiles(path: &Path, grid: ChildGridSpec) -> Result<LoadedSourceProfiles> {
    let source = read_initialized_product_restart(
        path,
        F02_PRODUCT_RESTART_SHA256,
        F02_PRODUCT_MANIFEST_SHA256,
        F02_PRODUCT_PAYLOAD_SHA256,
        F02_PRODUCT_RECEIPT_SHA256,
        F02_SOURCE_ARTIFACT_SHA256,
        F02_SOURCE_CONTRACT_SHA256,
    )?;
    let names = ["pi0", "qv0", "th0", "u0", "v0"];
    let mut raw = source.read_carriers_f32le_exact(&names)?;
    let mut values = BTreeMap::new();
    let mut hashes = BTreeMap::new();
    for name in names {
        let record = source
            .carrier(name)
            .ok_or_else(|| invalid(format!("source initialized product lacks {name}")))?;
        if record.shape != [grid.nz()] || record.nbytes != (grid.nz() * 4) as u64 {
            return Err(invalid(format!(
                "source initialized-product profile {name} shape differs from the child"
            )));
        }
        let bytes = raw
            .remove(name)
            .ok_or_else(|| invalid(format!("source profile extraction lost {name}")))?;
        if sha256::digest_hex(&bytes) != record.sha256 {
            return Err(invalid(format!(
                "source initialized-product profile {name} differs from its carrier pin"
            )));
        }
        let decoded = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
            .collect::<Vec<_>>();
        if decoded.len() != grid.nz()
            || decoded.iter().any(|value| !value.is_finite())
            || (name == "qv0" && decoded.iter().any(|value| *value < 0.0))
        {
            return Err(invalid(format!(
                "source initialized-product profile {name} is non-physical"
            )));
        }
        hashes.insert(name.to_owned(), record.sha256.clone());
        values.insert(name, decoded);
    }
    Ok(LoadedSourceProfiles {
        base: StreamedBaseProfiles {
            pi0: values.remove("pi0").expect("profile inserted"),
            th0: values.remove("th0").expect("profile inserted"),
            qv0: values.remove("qv0").expect("profile inserted"),
        },
        u0: values.remove("u0").expect("profile inserted"),
        v0: values.remove("v0").expect("profile inserted"),
        hashes,
    })
}

fn create_sized_file(path: &Path, nbytes: u64) -> Result<File> {
    let file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(path)?;
    file.set_len(nbytes)?;
    if file.metadata()?.len() != nbytes {
        return Err(invalid(format!(
            "streamed output {} length differs after allocation",
            path.display()
        )));
    }
    Ok(file)
}

fn hash_payload_file(
    path: &Path,
    name: &'static str,
    layout: &[StreamCarrierLayout],
) -> Result<StreamPayloadFile> {
    let expected_nbytes = layout
        .last()
        .map(|record| record.offset_nbytes + record.nbytes)
        .ok_or_else(|| invalid("streamed payload layout is empty"))?;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != expected_nbytes
    {
        return Err(invalid(format!(
            "streamed output {name} is not an exact regular file"
        )));
    }
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
    let mut whole = Sha256::default();
    let mut records = Vec::with_capacity(layout.len());
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    let mut expected_offset = 0u64;
    for record in layout {
        if record.offset_nbytes != expected_offset || record.nbytes % 4 != 0 {
            return Err(invalid(format!(
                "streamed output {name} carrier layout is not contiguous f32"
            )));
        }
        let mut carrier = Sha256::default();
        let mut remaining = record.nbytes;
        while remaining != 0 {
            let count = usize::try_from(remaining.min(buffer.len() as u64))
                .expect("bounded streamed hash chunk fits usize");
            reader.read_exact(&mut buffer[..count])?;
            whole.update(&buffer[..count]);
            carrier.update(&buffer[..count]);
            remaining -= count as u64;
        }
        records.push(StreamPayloadRecord {
            name: record.name,
            shape: record.shape.clone(),
            offset_nbytes: record.offset_nbytes,
            nbytes: record.nbytes,
            sha256: sha256::hex(&carrier.finalize()),
        });
        expected_offset = expected_offset
            .checked_add(record.nbytes)
            .ok_or_else(|| invalid("streamed output carrier offset overflows"))?;
    }
    if expected_offset != expected_nbytes {
        return Err(invalid(format!(
            "streamed output {name} layout does not exhaust the file"
        )));
    }
    let mut probe = [0u8; 1];
    if reader.read(&mut probe)? != 0 {
        return Err(invalid(format!(
            "streamed output {name} contains trailing bytes"
        )));
    }
    Ok(StreamPayloadFile {
        name,
        nbytes: expected_nbytes,
        sha256: sha256::hex(&whole.finalize()),
        records,
    })
}

fn hash_regular_file(path: &Path, maximum: Option<u64>, label: &str) -> Result<(String, u64)> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || maximum.is_some_and(|limit| metadata.len() > limit)
    {
        return Err(invalid(format!("{label} is not a bounded regular file")));
    }
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
    let mut hasher = Sha256::default();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    let mut observed = 0u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        observed += count as u64;
    }
    if observed != metadata.len() {
        return Err(invalid(format!("{label} length changed during hashing")));
    }
    Ok((sha256::hex(&hasher.finalize()), observed))
}

fn read_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        return Err(invalid(format!("{label} is not a bounded regular file")));
    }
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len()).map_err(|_| invalid(format!("{label} is too large")))?,
    );
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() as u64 != metadata.len() {
        return Err(invalid(format!("{label} length changed while reading")));
    }
    Ok(bytes)
}

fn require_hash(path: &Path, expected: &str, maximum: Option<u64>, label: &str) -> Result<()> {
    validate_sha256(expected, label)?;
    let (actual, _) = hash_regular_file(path, maximum, label)?;
    if actual != expected {
        return Err(invalid(format!("{label} differs from its exact pin")));
    }
    Ok(())
}

fn verify_exact_inventory(directory: &Path, expected: &[&str], label: &str) -> Result<()> {
    let mut actual = fs::read_dir(directory)?
        .map(|entry| {
            let entry = entry?;
            if entry.file_type()?.is_symlink() || !entry.file_type()?.is_file() {
                return Err(invalid(format!("{label} contains a non-regular entry")));
            }
            entry
                .file_name()
                .into_string()
                .map_err(|_| invalid(format!("{label} contains a non-UTF8 name")))
        })
        .collect::<Result<Vec<_>>>()?;
    actual.sort();
    let mut expected = expected
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    expected.sort();
    if actual != expected {
        return Err(invalid(format!("{label} inventory differs")));
    }
    Ok(())
}

fn verify_stream_output_inventory(directory: &Path) -> Result<()> {
    verify_exact_inventory(
        directory,
        &[
            CHILD_BASE_WIND_FILE,
            CHILD_AUXILIARY_GEOPOTENTIAL_FILE,
            CHILD_AUXILIARY_LIVE_DENSITY_FILE,
            CHILD_AUXILIARY_STREAM_MANIFEST_FILE,
            CHILD_AUXILIARY_STREAM_RECEIPT_FILE,
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
        ],
        "streamed auxiliary output",
    )
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn stream_manifest_value(
    preset: AdmittedChildAuxiliaryStreamPreset,
    atmosphere: &StreamAtmosphereIdentity,
    source: &LoadedSourceProfiles,
    payloads: &[StreamPayloadFile],
    free_bytes: u64,
    reserve_bytes: u64,
    post_publish_free_bytes: u64,
    producer_source_sha256: &str,
    producer_executable_sha256: &str,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_AUXILIARY_STREAM_ABI)),
        ("claim_limit", string(CHILD_AUXILIARY_STREAM_CLAIM_LIMIT)),
        (
            "capacity",
            object(vec![
                ("free_bytes", number(free_bytes)),
                ("planned_payload_copies", number(1)),
                ("post_publish_free_bytes", number(post_publish_free_bytes)),
                ("reserve_bytes", number(reserve_bytes)),
            ]),
        ),
        (
            "grid",
            object(vec![
                ("dx_m_f64_bits", number(preset.grid.dx_m().to_bits())),
                ("dy_m_f64_bits", number(preset.grid.dy_m().to_bits())),
                ("nx", number(preset.grid.nx() as u64)),
                ("ny", number(preset.grid.ny() as u64)),
                ("nz", number(preset.grid.nz() as u64)),
                ("ztop_m_f64_bits", number(preset.grid.ztop_m().to_bits())),
            ]),
        ),
        (
            "inputs",
            object(vec![
                (
                    "atmosphere",
                    object(vec![
                        ("arena_sha256", string(&atmosphere.arena_sha256)),
                        ("manifest_sha256", string(&atmosphere.manifest_sha256)),
                        ("receipt_sha256", string(&atmosphere.receipt_sha256)),
                    ]),
                ),
                (
                    "source_product",
                    object(vec![
                        ("artifact_sha256", string(F02_PRODUCT_RESTART_SHA256)),
                        ("manifest_sha256", string(F02_PRODUCT_MANIFEST_SHA256)),
                        ("payload_sha256", string(F02_PRODUCT_PAYLOAD_SHA256)),
                        ("receipt_sha256", string(F02_PRODUCT_RECEIPT_SHA256)),
                        ("source_artifact_sha256", string(F02_SOURCE_ARTIFACT_SHA256)),
                        ("source_contract_sha256", string(F02_SOURCE_CONTRACT_SHA256)),
                    ]),
                ),
                (
                    "source_profiles",
                    JsonValue::Object(
                        source
                            .hashes
                            .iter()
                            .map(|(name, hash)| (name.clone(), string(hash)))
                            .collect(),
                    ),
                ),
                (
                    "static_heights",
                    object(vec![
                        ("manifest_sha256", string(preset.static_manifest_sha256)),
                        (
                            "receipt_sha256",
                            preset
                                .static_receipt_sha256
                                .map(string)
                                .unwrap_or(JsonValue::Null),
                        ),
                        ("validation_sha256", string(preset.static_validation_sha256)),
                        ("zf_sha256", string(preset.static_zf_sha256)),
                        ("zh_sha256", string(preset.static_zh_sha256)),
                    ]),
                ),
                (
                    "terrain",
                    object(vec![
                        ("manifest_sha256", string(preset.terrain_manifest_sha256)),
                        ("payload_sha256", string(preset.terrain_payload_sha256)),
                        ("receipt_sha256", string(preset.terrain_receipt_sha256)),
                    ]),
                ),
            ]),
        ),
        (
            "outputs",
            JsonValue::Array(payloads.iter().map(payload_file_value).collect()),
        ),
        (
            "producer",
            object(vec![
                ("executable_sha256", string(producer_executable_sha256)),
                ("source_sha256", string(producer_source_sha256)),
            ]),
        ),
        ("schema", string(CHILD_AUXILIARY_STREAM_MANIFEST_SCHEMA)),
        ("status", string(CHILD_AUXILIARY_STREAM_STATUS)),
    ])
}

fn payload_file_value(file: &StreamPayloadFile) -> JsonValue {
    object(vec![
        (
            "carriers",
            JsonValue::Array(file.records.iter().map(payload_record_value).collect()),
        ),
        ("file", string(file.name)),
        ("nbytes", number(file.nbytes)),
        ("sha256", string(&file.sha256)),
    ])
}

fn payload_record_value(record: &StreamPayloadRecord) -> JsonValue {
    object(vec![
        ("dtype", string("little-endian-f32")),
        ("name", string(record.name)),
        ("nbytes", number(record.nbytes)),
        ("offset_nbytes", number(record.offset_nbytes)),
        ("order", string("C")),
        ("sha256", string(&record.sha256)),
        (
            "shape",
            JsonValue::Array(
                record
                    .shape
                    .iter()
                    .map(|value| number(*value as u64))
                    .collect(),
            ),
        ),
    ])
}

fn stream_receipt_from_payloads(
    payloads: &[StreamPayloadFile],
    manifest_sha256: String,
    producer_source_sha256: String,
    producer_executable_sha256: String,
) -> Result<ChildAuxiliaryStreamReceipt> {
    let find = |name: &str| {
        payloads
            .iter()
            .find(|payload| payload.name == name)
            .map(|payload| payload.sha256.clone())
            .ok_or_else(|| invalid(format!("streamed receipt lacks payload {name}")))
    };
    Ok(ChildAuxiliaryStreamReceipt {
        manifest_sha256,
        terrain_open_sha256: find(CHILD_AUXILIARY_TERRAIN_OPEN_FILE)?,
        live_density_sha256: find(CHILD_AUXILIARY_LIVE_DENSITY_FILE)?,
        geopotential_sha256: find(CHILD_AUXILIARY_GEOPOTENTIAL_FILE)?,
        base_wind_sha256: find(CHILD_BASE_WIND_FILE)?,
        receipt_sha256: String::new(),
        producer_source_sha256,
        producer_executable_sha256,
    })
}

fn stream_receipt_value(receipt: &ChildAuxiliaryStreamReceipt) -> JsonValue {
    object(vec![
        ("base_wind_sha256", string(&receipt.base_wind_sha256)),
        ("geopotential_sha256", string(&receipt.geopotential_sha256)),
        ("live_density_sha256", string(&receipt.live_density_sha256)),
        ("manifest_sha256", string(&receipt.manifest_sha256)),
        (
            "producer_executable_sha256",
            string(&receipt.producer_executable_sha256),
        ),
        (
            "producer_source_sha256",
            string(&receipt.producer_source_sha256),
        ),
        ("schema", string(CHILD_AUXILIARY_STREAM_RECEIPT_SCHEMA)),
        ("status", string(CHILD_AUXILIARY_STREAM_STATUS)),
        ("terrain_open_sha256", string(&receipt.terrain_open_sha256)),
    ])
}

fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

fn string(value: impl Into<String>) -> JsonValue {
    JsonValue::String(value.into())
}

fn number(value: impl ToString) -> JsonValue {
    JsonValue::Number(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child_full_state::build_base_wind_values_for_grid;
    use crate::child_remap::{ACCEPTED_50M_CHILD_GRID, HARD_5M_CHILD_GRID};
    use crate::terrain::HorizontalCrop;
    use crate::terrain_admission_science::{
        compute_child_live_density, compute_stretched_terrain_base, compute_surface_halos,
        pack_cold_start_mutable, PerturbationPolicy,
    };
    use std::collections::BTreeMap;
    use std::fs::{self, OpenOptions};
    use std::io::{Read, Seek};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_file(label: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cm1-state-{label}-{}-{stamp}.f32le",
            std::process::id()
        ))
    }

    #[derive(Default)]
    struct MemoryAtmospherePlaneReader {
        arrays: BTreeMap<&'static str, (Vec<usize>, Vec<f32>)>,
    }

    impl MemoryAtmospherePlaneReader {
        fn insert(&mut self, name: &'static str, shape: Vec<usize>, values: Vec<f32>) {
            assert_eq!(shape.iter().product::<usize>(), values.len());
            assert!(self.arrays.insert(name, (shape, values)).is_none());
        }
    }

    impl AtmospherePlaneReader for MemoryAtmospherePlaneReader {
        fn read_plane_f32(&mut self, name: &str, prefix: &[usize]) -> Result<Vec<f32>> {
            let (shape, values) = self
                .arrays
                .get(name)
                .ok_or_else(|| invalid(format!("test atmosphere has no carrier {name}")))?;
            if shape.len() < 2 || prefix.len() + 2 != shape.len() {
                return Err(invalid("test atmosphere plane prefix rank is invalid"));
            }
            let mut plane_index = 0usize;
            for (index, extent) in prefix.iter().zip(shape) {
                if index >= extent {
                    return Err(invalid("test atmosphere plane prefix is out of range"));
                }
                plane_index = plane_index
                    .checked_mul(*extent)
                    .and_then(|value| value.checked_add(*index))
                    .ok_or_else(|| invalid("test atmosphere plane index overflows"))?;
            }
            let plane_words = shape[shape.len() - 2] * shape[shape.len() - 1];
            let start = plane_index * plane_words;
            Ok(values[start..start + plane_words].to_vec())
        }
    }

    #[test]
    fn streamed_layout_preserves_50m_and_hard_5m_byte_counts() {
        let accepted = terrain_open_stream_layout(ACCEPTED_50M_CHILD_GRID).unwrap();
        assert_eq!(accepted.len(), 50);
        assert_eq!(
            accepted.last().unwrap().offset_nbytes + accepted.last().unwrap().nbytes,
            953_971_136
        );
        let hard = terrain_open_stream_layout(HARD_5M_CHILD_GRID).unwrap();
        assert_eq!(hard.len(), 50);
        assert_eq!(
            hard.last().unwrap().offset_nbytes + hard.last().unwrap().nbytes,
            92_416_017_536
        );
    }

    #[test]
    fn bounded_base_and_surface_regions_are_raw_bit_exact_to_full_frames() {
        let grid = ChildGridSpec::new(2, 2, 75, 7_500.0, 7_500.0, 15_000.0).unwrap();
        let terrain = vec![0.0_f32, 50.0, 100.0, 150.0];
        let stretch = sealed_child_stretch_grid().unwrap();
        let sigma = &stretch.sigma[1..=grid.nz()];
        let sigmaf = &stretch.sigmaf[1..=grid.nz() + 1];
        let mut zh = Vec::with_capacity(grid.nz() * grid.ny() * grid.nx());
        let mut c1 = Vec::with_capacity((grid.nz() + 1) * grid.ny() * grid.nx());
        let mut c2 = Vec::with_capacity((grid.nz() + 1) * grid.ny() * grid.nx());
        for k in 0..grid.nz() {
            for surface in &terrain {
                zh.push(*surface + sigma[k] * (15_000.0 - *surface) / 15_000.0);
            }
        }
        for face in 0..=grid.nz() {
            for surface in &terrain {
                let metrics =
                    derive_stretched_column_metrics(*surface, sigma, sigmaf, 15_000.0).unwrap();
                c1.push(metrics.c1[face]);
                c2.push(metrics.c2[face]);
            }
        }
        let profiles = StreamedBaseProfiles {
            pi0: (0..grid.nz()).map(|k| 0.999 - 0.0005 * k as f32).collect(),
            th0: (0..grid.nz()).map(|k| 300.0 + 0.1 * k as f32).collect(),
            qv0: (0..grid.nz()).map(|k| 0.01 - 0.00005 * k as f32).collect(),
        };
        let bottom = extrapolation_weights(sigma, sigmaf, false).unwrap();
        let top = extrapolation_weights(sigma, sigmaf, true).unwrap();
        let base = compute_stretched_terrain_base(
            &terrain,
            [grid.nz(), grid.ny(), grid.nx()],
            &zh,
            sigma,
            15_000.0,
            200.0,
            &profiles.pi0,
            &profiles.th0,
            &profiles.qv0,
            &c1,
            &c2,
            bottom,
            top,
        )
        .unwrap();
        let surface = compute_surface_halos(
            &terrain,
            [grid.ny(), grid.nx()],
            HorizontalCrop::full(grid.ny(), grid.nx()),
            [grid.dx_m(), grid.dy_m(), 200.0],
            grid.nz(),
        )
        .unwrap();
        let path = temporary_file("streamed-aux-base");
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let records = terrain_open_stream_layout(grid).unwrap();
        file.set_len(records.last().unwrap().offset_nbytes + records.last().unwrap().nbytes)
            .unwrap();
        write_streamed_base_and_surface_regions(&mut file, grid, &terrain, &profiles).unwrap();
        let expected = [
            &base.pi0,
            &base.prs0,
            &base.th0,
            &base.thv0,
            &base.rth0,
            &base.qv0,
            &base.rho0,
            &base.rr0,
            &base.rf0,
            &base.rrf0,
            &base.rho0s,
            &base.pi0s,
            &base.prs0s,
            &base.rth0s,
            &surface.terrain_msl_halo,
            &surface.gz_halo,
            &surface.rgz_halo,
            &surface.gzu_halo,
            &surface.rgzu_halo,
            &surface.gzv_halo,
            &surface.rgzv_halo,
            &surface.dzdx_halo,
            &surface.dzdy_halo,
        ];
        let mut actual = Vec::new();
        for (record, expected_values) in records.iter().zip(expected) {
            file.seek(SeekFrom::Start(record.offset_nbytes)).unwrap();
            actual.resize(record.nbytes as usize, 0);
            file.read_exact(&mut actual).unwrap();
            let expected_bytes = expected_values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected_bytes, "carrier {} differs", record.name);
        }
        drop(file);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn bounded_mutable_regions_are_raw_bit_exact_to_full_frame() {
        let grid = ChildGridSpec::new(2, 2, 75, 7_500.0, 7_500.0, 15_000.0).unwrap();
        let terrain = vec![0.0_f32, 50.0, 100.0, 150.0];
        let stretch = sealed_child_stretch_grid().unwrap();
        let sigma = &stretch.sigma[1..=grid.nz()];
        let sigmaf = &stretch.sigmaf[1..=grid.nz() + 1];
        let mut zh = Vec::with_capacity(grid.nz() * grid.ny() * grid.nx());
        let mut c1 = Vec::with_capacity((grid.nz() + 1) * grid.ny() * grid.nx());
        let mut c2 = Vec::with_capacity((grid.nz() + 1) * grid.ny() * grid.nx());
        for k in 0..grid.nz() {
            for surface in &terrain {
                zh.push(*surface + sigma[k] * (grid.ztop_m() as f32 - *surface) / 15_000.0);
            }
        }
        for face in 0..=grid.nz() {
            for surface in &terrain {
                let metrics =
                    derive_stretched_column_metrics(*surface, sigma, sigmaf, grid.ztop_m() as f32)
                        .unwrap();
                c1.push(metrics.c1[face]);
                c2.push(metrics.c2[face]);
            }
        }
        let profiles = StreamedBaseProfiles {
            pi0: (0..grid.nz()).map(|k| 0.999 - 0.0005 * k as f32).collect(),
            th0: (0..grid.nz()).map(|k| 300.0 + 0.1 * k as f32).collect(),
            qv0: (0..grid.nz()).map(|k| 0.01 - 0.00005 * k as f32).collect(),
        };
        let bottom = extrapolation_weights(sigma, sigmaf, false).unwrap();
        let top = extrapolation_weights(sigma, sigmaf, true).unwrap();
        let base = compute_stretched_terrain_base(
            &terrain,
            [grid.nz(), grid.ny(), grid.nx()],
            &zh,
            sigma,
            grid.ztop_m() as f32,
            200.0,
            &profiles.pi0,
            &profiles.th0,
            &profiles.qv0,
            &c1,
            &c2,
            bottom,
            top,
        )
        .unwrap();
        let surface = compute_surface_halos(
            &terrain,
            [grid.ny(), grid.nx()],
            HorizontalCrop::full(grid.ny(), grid.nx()),
            [grid.dx_m(), grid.dy_m(), 200.0],
            grid.nz(),
        )
        .unwrap();
        let mass_words = grid.nz() * grid.ny() * grid.nx();
        let ua = (0..grid.nz() * grid.ny() * (grid.nx() + 1))
            .map(|word| -2.0 + word as f32 * 0.0001)
            .collect::<Vec<_>>();
        let va = (0..grid.nz() * (grid.ny() + 1) * grid.nx())
            .map(|word| 1.0 - word as f32 * 0.00007)
            .collect::<Vec<_>>();
        let wa = (0..(grid.nz() + 1) * grid.ny() * grid.nx())
            .map(|word| word as f32 * 0.00003)
            .collect::<Vec<_>>();
        let ppi = (0..mass_words)
            .map(|word| -0.0003 + word as f32 * 0.0000001)
            .collect::<Vec<_>>();
        let ppx = (0..mass_words)
            .map(|word| 0.0002 - word as f32 * 0.00000004)
            .collect::<Vec<_>>();
        let tha = (0..mass_words)
            .map(|word| -0.2 + word as f32 * 0.0002)
            .collect::<Vec<_>>();
        let qa = (0..6 * mass_words)
            .map(|word| 0.000001 * ((word % 17) + 1) as f32)
            .collect::<Vec<_>>();
        let full = pack_cold_start_mutable(
            [grid.nz(), grid.ny(), grid.nx()],
            HorizontalCrop::full(grid.ny(), grid.nx()),
            &profiles.pi0,
            &profiles.th0,
            &ua,
            &va,
            &wa,
            &ppi,
            None,
            &tha,
            &qa,
            &base,
            &surface,
            PerturbationPolicy::PreserveRestartGrid,
        )
        .unwrap();
        let expected_density = compute_child_live_density(
            [grid.nz(), grid.ny(), grid.nx()],
            &ppi,
            &tha,
            &qa,
            &base,
            &c1,
            &c2,
            bottom,
            top,
        )
        .unwrap();
        let mut atmosphere = MemoryAtmospherePlaneReader::default();
        atmosphere.insert("ppi", vec![grid.nz(), grid.ny(), grid.nx()], ppi);
        atmosphere.insert("ppx", vec![grid.nz(), grid.ny(), grid.nx()], ppx);
        atmosphere.insert("tha", vec![grid.nz(), grid.ny(), grid.nx()], tha);
        atmosphere.insert("qa", vec![6, grid.nz(), grid.ny(), grid.nx()], qa);
        atmosphere.insert("ua", vec![grid.nz(), grid.ny(), grid.nx() + 1], ua);
        atmosphere.insert("va", vec![grid.nz(), grid.ny() + 1, grid.nx()], va);
        atmosphere.insert("wa", vec![grid.nz() + 1, grid.ny(), grid.nx()], wa);
        let path = temporary_file("streamed-aux-mutable");
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let records = terrain_open_stream_layout(grid).unwrap();
        file.set_len(records.last().unwrap().offset_nbytes + records.last().unwrap().nbytes)
            .unwrap();
        write_streamed_base_and_surface_regions(&mut file, grid, &terrain, &profiles).unwrap();
        write_streamed_mutable_regions(&mut file, grid, &terrain, &mut atmosphere).unwrap();
        assert_eq!(full.carriers.len(), records.len() - 23);
        let mut actual = Vec::new();
        for (record, expected) in records.iter().skip(23).zip(&full.carriers) {
            assert_eq!(record.name, expected.name);
            assert_eq!(record.shape, expected.shape);
            file.seek(SeekFrom::Start(record.offset_nbytes)).unwrap();
            actual.resize(record.nbytes as usize, 0);
            file.read_exact(&mut actual).unwrap();
            let expected_bytes = expected
                .values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected_bytes, "carrier {} differs", record.name);
        }
        let density_path = temporary_file("streamed-aux-density");
        let mut density_file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&density_path)
            .unwrap();
        let density_records = live_density_stream_layout(grid).unwrap();
        density_file
            .set_len(
                density_records.last().unwrap().offset_nbytes
                    + density_records.last().unwrap().nbytes,
            )
            .unwrap();
        write_streamed_live_density(
            &mut density_file,
            &mut file,
            grid,
            &terrain,
            &mut atmosphere,
        )
        .unwrap();
        assert_eq!(density_records.len(), expected_density.carriers.len());
        for (record, expected) in density_records.iter().zip(&expected_density.carriers) {
            assert_eq!(record.name, expected.name);
            assert_eq!(record.shape, expected.shape);
            density_file
                .seek(SeekFrom::Start(record.offset_nbytes))
                .unwrap();
            actual.resize(record.nbytes as usize, 0);
            density_file.read_exact(&mut actual).unwrap();
            let expected_bytes = expected
                .values
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected_bytes, "carrier {} differs", record.name);
        }
        drop(density_file);
        fs::remove_file(density_path).unwrap();
        drop(file);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn bounded_geopotential_and_base_wind_are_raw_bit_exact_to_full_values() {
        let grid = ChildGridSpec::new(2, 2, 75, 7_500.0, 7_500.0, 15_000.0).unwrap();
        let terrain = [0.0_f32, 50.0, 100.0, 150.0];
        let stretch = sealed_child_stretch_grid().unwrap();
        let sigma = &stretch.sigma[1..=grid.nz()];
        let sigmaf = &stretch.sigmaf[1..=grid.nz() + 1];
        let mut zh = Vec::with_capacity(grid.nz() * grid.ny() * grid.nx());
        let mut zf = Vec::with_capacity((grid.nz() + 1) * grid.ny() * grid.nx());
        for k in 0..grid.nz() {
            for surface in terrain {
                let metrics =
                    derive_stretched_column_metrics(surface, sigma, sigmaf, grid.ztop_m() as f32)
                        .unwrap();
                zh.push(metrics.zh[k]);
            }
        }
        for face in 0..=grid.nz() {
            for surface in terrain {
                let metrics =
                    derive_stretched_column_metrics(surface, sigma, sigmaf, grid.ztop_m() as f32)
                        .unwrap();
                zf.push(metrics.zf[face]);
            }
        }
        let u_profile = (0..grid.nz())
            .map(|k| -4.0 + k as f32 * 0.05)
            .collect::<Vec<_>>();
        let v_profile = (0..grid.nz())
            .map(|k| 2.0 - k as f32 * 0.03)
            .collect::<Vec<_>>();
        let (expected_u0, expected_v0) =
            build_base_wind_values_for_grid(grid, &zh, &u_profile, &v_profile).unwrap();

        let zh_path = temporary_file("streamed-static-zh");
        let mut zh_file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&zh_path)
            .unwrap();
        write_f32_at(&mut zh_file, 0, &zh, &mut Vec::new()).unwrap();
        let wind_path = temporary_file("streamed-base-wind");
        let mut wind_file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&wind_path)
            .unwrap();
        let wind_records = base_wind_stream_layout(grid).unwrap();
        wind_file
            .set_len(
                wind_records.last().unwrap().offset_nbytes + wind_records.last().unwrap().nbytes,
            )
            .unwrap();
        write_streamed_base_wind(&mut wind_file, &mut zh_file, grid, &u_profile, &v_profile)
            .unwrap();
        let mut actual = Vec::new();
        for (record, expected) in wind_records.iter().zip([&expected_u0, &expected_v0]) {
            wind_file
                .seek(SeekFrom::Start(record.offset_nbytes))
                .unwrap();
            actual.resize(record.nbytes as usize, 0);
            wind_file.read_exact(&mut actual).unwrap();
            let expected_bytes = expected
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected_bytes, "carrier {} differs", record.name);
        }

        let zf_path = temporary_file("streamed-static-zf");
        let mut zf_file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&zf_path)
            .unwrap();
        write_f32_at(&mut zf_file, 0, &zf, &mut Vec::new()).unwrap();
        let geopotential_path = temporary_file("streamed-geopotential");
        let mut geopotential_file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&geopotential_path)
            .unwrap();
        geopotential_file
            .set_len(
                child_auxiliary_base_layout_for_grid(grid)
                    .unwrap()
                    .geopotential_nbytes,
            )
            .unwrap();
        write_streamed_geopotential(&mut geopotential_file, &mut zf_file, grid).unwrap();
        geopotential_file.seek(SeekFrom::Start(0)).unwrap();
        actual.resize(zf.len() * 4, 0);
        geopotential_file.read_exact(&mut actual).unwrap();
        let expected_geopotential = zf
            .iter()
            .flat_map(|height| (*height * G).to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected_geopotential);

        drop(geopotential_file);
        drop(zf_file);
        drop(wind_file);
        drop(zh_file);
        for path in [zh_path, zf_path, wind_path, geopotential_path] {
            fs::remove_file(path).unwrap();
        }
    }
}
