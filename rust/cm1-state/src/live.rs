//! Strict CM1 forecast-checkpoint transport into a CUDA-ready carrier arena.
//!
//! The producer verifies a self-describing `CM1FCP1` artifact against an
//! independently supplied contract hash and artifact hash.  It then performs
//! every shape decision, physical crop, moisture packing, and CM1 A/current
//! duplication in Rust.  The published payload is therefore opaque to the
//! Python CUDA driver: Python only verifies bytes, copies one arena, and
//! exposes the already-described non-aliasing views.

use crate::forecast::{
    read_forecast_checkpoint_embedded, ForecastArray, ForecastContract, ForecastFrame,
    ForecastReceipt, SchedulerScalarValue,
};
use crate::json::JsonValue;
use crate::sha256::{self, Sha256};
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const LIVE_EXPORT_SCHEMA: &str = "gpuwm-cm1.cuda-live-export/v1";
pub const LIVE_EXPORT_RECEIPT_SCHEMA: &str = "gpuwm-cm1.cuda-live-export-receipt/v1";
pub const LIVE_CARRIER_ABI: &str = "gpuwm-cm1.cuda-live-carriers/v1";
pub const LIVE_PAYLOAD_FILE: &str = "carrier-payload.f32le";
pub const LIVE_MANIFEST_FILE: &str = "manifest.json";
pub const LIVE_RECEIPT_FILE: &str = "receipt.json";
pub const LIVE_PUBLICATION_CONTRACT: &str = "write payload, canonical manifest, and canonical receipt in a new sibling staging directory; sync every file; rename the directory only after all hashes validate; refuse an existing destination";
pub const LIVE_CLAIM_LIMIT: &str = "byte-exact physical-domain CM1 carrier transport only; terrain metrics, halos, derived pressure/density, pressure rebalance, closure state, boundary history, and forecast stepping remain deferred";

const SOURCE_PROFILE_ID: &str = "hrrr-cm1-real-state-v1";
const REAL_NX: usize = 300;
const REAL_NY: usize = 300;
const REAL_NZ: usize = 75;
const REAL_DT_S: f64 = 0.5;
const REAL_DX_M: f64 = 100.0;
const REAL_DY_M: f64 = 100.0;
const REAL_DZ_M: f64 = 200.0;
const ITEMSIZE: usize = 4;
const IO_CHUNK_WORDS: usize = 64 * 1024;

const SOURCE_ARRAYS: [&str; 22] = [
    "pi0",
    "pp3d",
    "prs0",
    "qc",
    "qg",
    "qi",
    "qr",
    "qs",
    "qv",
    "qv0",
    "refc_dbz",
    "refd_1000m_agl_dbz",
    "refd_4000m_agl_dbz",
    "rho0",
    "terrain_msl",
    "th0",
    "th3d",
    "u0",
    "u3d",
    "v0",
    "v3d",
    "w3d",
];

const PROFILE_SOURCES: [&str; 7] = ["pi0", "prs0", "qv0", "rho0", "th0", "u0", "v0"];
const MOISTURE_SOURCES: [&str; 6] = ["qv", "qc", "qr", "qi", "qs", "qg"];

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LiveAdmission {
    /// Validate the carrier schema and shape relationships without fixing the
    /// outer source dimensions.  Intended for unit tests and independently
    /// bounded validation exports.
    Generic,
    /// Require the sealed real-product 300x300x75, 100x100x200 m, dt=0.5 s
    /// startup checkpoint at step zero.
    RealProduct,
}

impl LiveAdmission {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic-schema",
            Self::RealProduct => "real-product-300x300x75",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalCrop {
    pub k_start: usize,
    pub j_start: usize,
    pub i_start: usize,
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
}

impl PhysicalCrop {
    pub fn full(nz: usize, ny: usize, nx: usize) -> Self {
        Self {
            k_start: 0,
            j_start: 0,
            i_start: 0,
            nz,
            ny,
            nx,
        }
    }

    pub(crate) fn validate(self, nz: usize, ny: usize, nx: usize) -> Result<()> {
        if self.nz == 0 || self.ny == 0 || self.nx == 0 {
            return Err(invalid("live-export crop extents must all be positive"));
        }
        let k_end = self
            .k_start
            .checked_add(self.nz)
            .ok_or_else(|| invalid("live-export crop k range overflows"))?;
        let j_end = self
            .j_start
            .checked_add(self.ny)
            .ok_or_else(|| invalid("live-export crop j range overflows"))?;
        let i_end = self
            .i_start
            .checked_add(self.nx)
            .ok_or_else(|| invalid("live-export crop i range overflows"))?;
        if k_end > nz || j_end > ny || i_end > nx {
            return Err(invalid(format!(
                "live-export crop k={}..{}, j={}..{}, i={}..{} exceeds source physical grid [{nz},{ny},{nx}]",
                self.k_start, k_end, self.j_start, j_end, self.i_start, i_end
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveExportReceipt {
    pub source_artifact_sha256: String,
    pub source_contract_sha256: String,
    pub payload_sha256: String,
    pub payload_nbytes: u64,
    pub manifest_sha256: String,
    pub manifest_nbytes: usize,
    pub carrier_count: usize,
    pub admission: LiveAdmission,
}

impl LiveExportReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("admission", string(self.admission.as_str())),
            ("carrier_abi", string(LIVE_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            ("claim_limit", string(LIVE_CLAIM_LIMIT)),
            ("manifest_nbytes", number(self.manifest_nbytes)),
            ("manifest_sha256", string(&self.manifest_sha256)),
            ("payload_nbytes", u64_number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            ("publication_contract", string(LIVE_PUBLICATION_CONTRACT)),
            ("schema", string(LIVE_EXPORT_RECEIPT_SCHEMA)),
            (
                "source_artifact_sha256",
                string(&self.source_artifact_sha256),
            ),
            (
                "source_contract_sha256",
                string(&self.source_contract_sha256),
            ),
        ]);
        String::from_utf8(value.canonical_bytes()).expect("canonical live receipt is UTF-8")
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GridShape {
    pub(crate) nz: usize,
    pub(crate) ny: usize,
    pub(crate) nx: usize,
}

#[derive(Clone, Debug)]
struct CarrierStats {
    finite_count: usize,
    nonzero_count: usize,
    min: f32,
    max: f32,
}

impl CarrierStats {
    fn from_values(name: &str, values: &[f32]) -> Result<Self> {
        if values.is_empty() {
            return Err(invalid(format!("live carrier {name} is empty")));
        }
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut nonzero_count = 0usize;
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "live carrier {name} contains non-finite f32 bits {:08x} at word {index}",
                    value.to_bits()
                )));
            }
            min = min.min(value);
            max = max.max(value);
            if value.to_bits() & 0x7fff_ffff != 0 {
                nonzero_count += 1;
            }
        }
        Ok(Self {
            finite_count: values.len(),
            nonzero_count,
            min,
            max,
        })
    }

    fn json(&self) -> JsonValue {
        object(vec![
            ("finite_count", number(self.finite_count)),
            ("maximum_f32_bits", f32_bits(self.max)),
            ("minimum_f32_bits", f32_bits(self.min)),
            ("nonzero_count", number(self.nonzero_count)),
        ])
    }
}

#[derive(Clone, Debug)]
struct CarrierRecord {
    name: String,
    sources: Vec<String>,
    role: String,
    shape: Vec<usize>,
    offset_nbytes: u64,
    nbytes: u64,
    sha256: String,
    stats: CarrierStats,
    duplicate_of: Option<String>,
    species_order: Vec<String>,
}

impl CarrierRecord {
    fn json(&self) -> JsonValue {
        let duplicate = self
            .duplicate_of
            .as_deref()
            .map(string)
            .unwrap_or(JsonValue::Null);
        object(vec![
            ("duplicate_of", duplicate),
            ("dtype", string("<f4")),
            ("itemsize", number(ITEMSIZE)),
            ("name", string(&self.name)),
            ("nbytes", u64_number(self.nbytes)),
            ("offset_nbytes", u64_number(self.offset_nbytes)),
            ("order", string("C")),
            ("role", string(&self.role)),
            (
                "shape",
                JsonValue::Array(self.shape.iter().copied().map(number).collect()),
            ),
            ("sha256", string(&self.sha256)),
            (
                "source_arrays",
                JsonValue::Array(self.sources.iter().map(|value| string(value)).collect()),
            ),
            (
                "species_order",
                JsonValue::Array(
                    self.species_order
                        .iter()
                        .map(|value| string(value))
                        .collect(),
                ),
            ),
            ("statistics", self.stats.json()),
        ])
    }
}

/// Verify and publish a CUDA-ready arena from one `CM1FCP1` checkpoint.
///
/// The destination directory must not exist.  `crop` is expressed on the
/// mass grid; the corresponding extra u, v, and w faces are retained.
pub fn export_checkpoint_live_new(
    input: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
    expected_contract_sha256: &str,
    expected_artifact_sha256: &str,
    crop: Option<PhysicalCrop>,
    admission: LiveAdmission,
) -> Result<LiveExportReceipt> {
    validate_sha256(
        expected_artifact_sha256,
        "expected checkpoint artifact SHA-256",
    )?;
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace live-export destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "live-export parent directory does not exist: {}",
            parent.display()
        )));
    }

    let (contract, restored) =
        read_forecast_checkpoint_embedded(input.as_ref(), expected_contract_sha256)?;
    if restored.receipt().artifact_sha256 != expected_artifact_sha256 {
        return Err(invalid(format!(
            "checkpoint artifact SHA-256 is {}; independently expected {expected_artifact_sha256}",
            restored.receipt().artifact_sha256
        )));
    }
    let grid = validate_contract_and_frame(&contract, restored.frame(), admission)?;
    let crop = crop.unwrap_or_else(|| PhysicalCrop::full(grid.nz, grid.ny, grid.nx));
    crop.validate(grid.nz, grid.ny, grid.nx)?;

    let staging = staging_path(output)?;
    fs::create_dir(&staging)?;
    let result = export_into_directory(
        &staging,
        &contract,
        restored.frame(),
        restored.receipt(),
        grid,
        crop,
        admission,
    );
    match result {
        Ok(receipt) => {
            if let Err(error) = fs::rename(&staging, output) {
                let _ = fs::remove_dir_all(&staging);
                return Err(error.into());
            }
            Ok(receipt)
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

fn export_into_directory(
    directory: &Path,
    contract: &ForecastContract,
    frame: &ForecastFrame,
    source_receipt: &ForecastReceipt,
    grid: GridShape,
    crop: PhysicalCrop,
    admission: LiveAdmission,
) -> Result<LiveExportReceipt> {
    let arrays = array_map(frame)?;
    for name in SOURCE_ARRAYS {
        validate_physical_range(name, array(&arrays, name)?, admission)?;
    }

    let payload_path = directory.join(LIVE_PAYLOAD_FILE);
    let payload_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&payload_path)?;
    let mut payload_writer = BufWriter::new(payload_file);
    let mut payload_hasher = Sha256::default();
    let mut offset = 0u64;
    let mut records = Vec::new();

    for name in PROFILE_SOURCES {
        let values = crop_profile(array(&arrays, name)?, grid.nz, crop)?;
        records.push(write_carrier(
            &mut payload_writer,
            &mut payload_hasher,
            &mut offset,
            name,
            &[name],
            "static-base-profile",
            vec![crop.nz],
            &values,
            None,
            &[],
        )?);
    }

    let u = crop_zyx(
        array(&arrays, "u3d")?,
        [grid.nz, grid.ny, grid.nx + 1],
        [crop.k_start, crop.j_start, crop.i_start],
        [crop.nz, crop.ny, crop.nx + 1],
        "u3d",
    )?;
    write_duplicate_pair(
        &mut payload_writer,
        &mut payload_hasher,
        &mut offset,
        &mut records,
        "ua",
        "u3d",
        "u3d",
        vec![crop.nz, crop.ny, crop.nx + 1],
        &u,
        "x-face",
    )?;

    let v = crop_zyx(
        array(&arrays, "v3d")?,
        [grid.nz, grid.ny + 1, grid.nx],
        [crop.k_start, crop.j_start, crop.i_start],
        [crop.nz, crop.ny + 1, crop.nx],
        "v3d",
    )?;
    write_duplicate_pair(
        &mut payload_writer,
        &mut payload_hasher,
        &mut offset,
        &mut records,
        "va",
        "v3d",
        "v3d",
        vec![crop.nz, crop.ny + 1, crop.nx],
        &v,
        "y-face",
    )?;

    let w = crop_zyx(
        array(&arrays, "w3d")?,
        [grid.nz + 1, grid.ny, grid.nx],
        [crop.k_start, crop.j_start, crop.i_start],
        [crop.nz + 1, crop.ny, crop.nx],
        "w3d",
    )?;
    write_duplicate_pair(
        &mut payload_writer,
        &mut payload_hasher,
        &mut offset,
        &mut records,
        "wa",
        "w3d",
        "w3d",
        vec![crop.nz + 1, crop.ny, crop.nx],
        &w,
        "z-face",
    )?;

    let pressure = crop_mass(array(&arrays, "pp3d")?, grid, crop, "pp3d")?;
    write_duplicate_pair(
        &mut payload_writer,
        &mut payload_hasher,
        &mut offset,
        &mut records,
        "ppi",
        "pp3d",
        "pp3d",
        vec![crop.nz, crop.ny, crop.nx],
        &pressure,
        "scalar-mass",
    )?;
    records.push(write_carrier(
        &mut payload_writer,
        &mut payload_hasher,
        &mut offset,
        "ppx",
        &["pp3d"],
        "persistent-fresh-pressure-helper",
        vec![crop.nz, crop.ny, crop.nx],
        &pressure,
        Some("ppi"),
        &[],
    )?);

    let theta = crop_mass(array(&arrays, "th3d")?, grid, crop, "th3d")?;
    write_duplicate_pair(
        &mut payload_writer,
        &mut payload_hasher,
        &mut offset,
        &mut records,
        "tha",
        "th3d",
        "th3d",
        vec![crop.nz, crop.ny, crop.nx],
        &theta,
        "scalar-mass",
    )?;

    let mut moisture = Vec::with_capacity(
        MOISTURE_SOURCES
            .len()
            .checked_mul(crop.nz)
            .and_then(|value| value.checked_mul(crop.ny))
            .and_then(|value| value.checked_mul(crop.nx))
            .ok_or_else(|| invalid("packed live moisture size overflows"))?,
    );
    for name in MOISTURE_SOURCES {
        moisture.extend(crop_mass(array(&arrays, name)?, grid, crop, name)?);
    }
    let species = MOISTURE_SOURCES.to_vec();
    records.push(write_carrier(
        &mut payload_writer,
        &mut payload_hasher,
        &mut offset,
        "qa",
        &species,
        "prognostic-a-packed-moisture",
        vec![species.len(), crop.nz, crop.ny, crop.nx],
        &moisture,
        None,
        &species,
    )?);
    records.push(write_carrier(
        &mut payload_writer,
        &mut payload_hasher,
        &mut offset,
        "q3d",
        &species,
        "stage-current-packed-moisture",
        vec![species.len(), crop.nz, crop.ny, crop.nx],
        &moisture,
        Some("qa"),
        &species,
    )?);

    let terrain = crop_surface(array(&arrays, "terrain_msl")?, grid, crop)?;
    records.push(write_carrier(
        &mut payload_writer,
        &mut payload_hasher,
        &mut offset,
        "terrain_msl",
        &["terrain_msl"],
        "static-surface",
        vec![crop.ny, crop.nx],
        &terrain,
        None,
        &[],
    )?);

    payload_writer.flush()?;
    let payload_file = payload_writer
        .into_inner()
        .map_err(|error| error.into_error())?;
    payload_file.sync_all()?;
    let payload_sha256 = sha256::hex(&payload_hasher.finalize());
    if payload_file.metadata()?.len() != offset {
        return Err(invalid(
            "live payload byte count changed after synchronized write",
        ));
    }

    let manifest = manifest_json(
        contract,
        source_receipt,
        grid,
        crop,
        admission,
        offset,
        &payload_sha256,
        &records,
        frame,
    );
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    write_new_synced(&directory.join(LIVE_MANIFEST_FILE), &manifest_bytes)?;

    let receipt = LiveExportReceipt {
        source_artifact_sha256: source_receipt.artifact_sha256.clone(),
        source_contract_sha256: contract.sha256(),
        payload_sha256,
        payload_nbytes: offset,
        manifest_sha256,
        manifest_nbytes: manifest_bytes.len(),
        carrier_count: records.len(),
        admission,
    };
    write_new_synced(
        &directory.join(LIVE_RECEIPT_FILE),
        receipt.canonical_json().as_bytes(),
    )?;
    Ok(receipt)
}

pub(crate) fn validate_contract_and_frame(
    contract: &ForecastContract,
    frame: &ForecastFrame,
    admission: LiveAdmission,
) -> Result<GridShape> {
    validate_contract_and_frame_role(contract, frame, admission, LiveFrameRole::Initialization)
}

/// Validate the independently sealed f01 history as a new real-data analysis
/// source.  The arrays retain their exact forecast-valid values; only the
/// consumer's model clock is restarted after the initialized-product export.
pub(crate) fn validate_contract_and_boundary_analysis_frame(
    contract: &ForecastContract,
    frame: &ForecastFrame,
    admission: LiveAdmission,
) -> Result<GridShape> {
    validate_contract_and_frame_role(contract, frame, admission, LiveFrameRole::BoundaryAnalysis)
}

/// Validate the sealed, coordinate-selected real-HRRR f02 ROI history.  The
/// source clock remains absolute; only the consuming GM1 lifecycle chooses a
/// new relative origin after product construction.
pub(crate) fn validate_contract_and_f02_roi_analysis_frame(
    contract: &ForecastContract,
    frame: &ForecastFrame,
    admission: LiveAdmission,
) -> Result<GridShape> {
    validate_contract_and_frame_role(contract, frame, admission, LiveFrameRole::F02RoiAnalysis)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LiveFrameRole {
    Initialization,
    BoundaryAnalysis,
    F02RoiAnalysis,
}

fn validate_contract_and_frame_role(
    contract: &ForecastContract,
    frame: &ForecastFrame,
    admission: LiveAdmission,
    role: LiveFrameRole,
) -> Result<GridShape> {
    if contract.profile_id() != SOURCE_PROFILE_ID {
        return Err(invalid(format!(
            "live export requires forecast profile {SOURCE_PROFILE_ID:?}; got {:?}",
            contract.profile_id()
        )));
    }
    let actual = contract
        .arrays()
        .iter()
        .map(|spec| spec.name())
        .collect::<BTreeSet<_>>();
    let expected = SOURCE_ARRAYS.into_iter().collect::<BTreeSet<_>>();
    if actual != expected {
        let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
        let extra = actual.difference(&expected).copied().collect::<Vec<_>>();
        return Err(invalid(format!(
            "live export source carrier inventory mismatch; missing={missing:?}, extra={extra:?}"
        )));
    }
    let mass = shape(contract, "pp3d")?;
    if mass.len() != 3 {
        return Err(invalid("pp3d must have z/y/x rank three"));
    }
    let grid = GridShape {
        nz: mass[0],
        ny: mass[1],
        nx: mass[2],
    };
    for name in ["th3d", "qv", "qc", "qr", "qi", "qs", "qg"] {
        require_shape(contract, name, &[grid.nz, grid.ny, grid.nx])?;
    }
    for name in PROFILE_SOURCES {
        require_shape(contract, name, &[grid.nz])?;
    }
    require_shape(contract, "u3d", &[grid.nz, grid.ny, grid.nx + 1])?;
    require_shape(contract, "v3d", &[grid.nz, grid.ny + 1, grid.nx])?;
    require_shape(contract, "w3d", &[grid.nz + 1, grid.ny, grid.nx])?;
    for name in [
        "terrain_msl",
        "refc_dbz",
        "refd_1000m_agl_dbz",
        "refd_4000m_agl_dbz",
    ] {
        require_shape(contract, name, &[grid.ny, grid.nx])?;
    }
    match role {
        LiveFrameRole::Initialization => {
            if frame.clock().step() != 0
                || frame.clock().model_time_s().to_bits() != 0.0f64.to_bits()
            {
                return Err(invalid(
                    "live initialization requires a step-zero, time-zero checkpoint",
                ));
            }
            require_scalar(frame, "boundary_ready", SchedulerScalarValue::Bool(false))?;
            require_scalar(
                frame,
                "forcing_valid_offset_s",
                SchedulerScalarValue::F64(0.0),
            )?;
            require_scalar(frame, "source_forecast_hour", SchedulerScalarValue::U64(0))?;
        }
        LiveFrameRole::BoundaryAnalysis => {
            if frame.clock().step() != 7200
                || frame.clock().model_time_s().to_bits() != 3600.0f64.to_bits()
            {
                return Err(invalid(
                    "real boundary-analysis source requires sealed f01 step 7200 at 3600 s",
                ));
            }
            require_scalar(frame, "boundary_ready", SchedulerScalarValue::Bool(true))?;
            require_scalar(
                frame,
                "forcing_valid_offset_s",
                SchedulerScalarValue::F64(3600.0),
            )?;
            require_scalar(frame, "source_forecast_hour", SchedulerScalarValue::U64(1))?;
        }
        LiveFrameRole::F02RoiAnalysis => {
            if frame.clock().step() != 14_400
                || frame.clock().model_time_s().to_bits() != 7200.0f64.to_bits()
            {
                return Err(invalid(
                    "real f02 ROI analysis source requires sealed step 14400 at 7200 s",
                ));
            }
            require_scalar(frame, "boundary_ready", SchedulerScalarValue::Bool(true))?;
            require_scalar(
                frame,
                "forcing_valid_offset_s",
                SchedulerScalarValue::F64(7200.0),
            )?;
            require_scalar(frame, "source_forecast_hour", SchedulerScalarValue::U64(2))?;
        }
    }
    if admission == LiveAdmission::RealProduct {
        if (grid.nx, grid.ny, grid.nz) != (REAL_NX, REAL_NY, REAL_NZ) {
            return Err(invalid(format!(
                "real-product live export requires [{REAL_NZ},{REAL_NY},{REAL_NX}], got [{},{},{}]",
                grid.nz, grid.ny, grid.nx
            )));
        }
        if frame.clock().dt_s().to_bits() != REAL_DT_S.to_bits() {
            return Err(invalid(format!(
                "real-product live export requires dt={REAL_DT_S} s"
            )));
        }
        for (name, expected) in [
            ("target_dx_m", REAL_DX_M),
            ("target_dy_m", REAL_DY_M),
            ("target_dz_m", REAL_DZ_M),
        ] {
            require_scalar(frame, name, SchedulerScalarValue::F64(expected))?;
        }
    }
    Ok(grid)
}

fn require_scalar(frame: &ForecastFrame, name: &str, expected: SchedulerScalarValue) -> Result<()> {
    let value = frame
        .scheduler_scalars()
        .iter()
        .find(|scalar| scalar.name() == name)
        .ok_or_else(|| invalid(format!("live source lacks scheduler scalar {name:?}")))?
        .value();
    let equal = match (value, expected) {
        (SchedulerScalarValue::F64(left), SchedulerScalarValue::F64(right)) => {
            left.to_bits() == right.to_bits()
        }
        _ => value == expected,
    };
    if !equal {
        return Err(invalid(format!(
            "live scheduler scalar {name:?} is {value:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn shape<'a>(contract: &'a ForecastContract, name: &str) -> Result<&'a [usize]> {
    contract
        .arrays()
        .iter()
        .find(|spec| spec.name() == name)
        .map(|spec| spec.shape())
        .ok_or_else(|| invalid(format!("forecast contract lacks array {name:?}")))
}

fn require_shape(contract: &ForecastContract, name: &str, expected: &[usize]) -> Result<()> {
    let actual = shape(contract, name)?;
    if actual != expected {
        return Err(invalid(format!(
            "live source array {name:?} shape is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn array_map(frame: &ForecastFrame) -> Result<BTreeMap<&str, &ForecastArray>> {
    let mut arrays = BTreeMap::new();
    for array in frame.arrays() {
        if arrays.insert(array.name(), array).is_some() {
            return Err(invalid(format!(
                "restored forecast repeats array {:?}",
                array.name()
            )));
        }
    }
    Ok(arrays)
}

fn array<'a>(arrays: &BTreeMap<&str, &'a ForecastArray>, name: &str) -> Result<&'a [f32]> {
    arrays
        .get(name)
        .map(|value| value.values())
        .ok_or_else(|| invalid(format!("restored forecast lacks array {name:?}")))
}

pub(crate) fn validate_physical_range(
    name: &str,
    values: &[f32],
    admission: LiveAdmission,
) -> Result<()> {
    let stats = CarrierStats::from_values(name, values)?;
    match name {
        "pi0" | "prs0" | "rho0" | "th0" => {
            if stats.min <= 0.0 {
                return Err(invalid(format!(
                    "live base carrier {name} must be strictly positive; minimum bits={:08x}",
                    stats.min.to_bits()
                )));
            }
        }
        "qv0" | "qv" => {
            if stats.min < 0.0 || stats.max >= 1.0 {
                return Err(invalid(format!(
                    "live vapor carrier {name} must remain in [0,1); range bits={:08x}..{:08x}",
                    stats.min.to_bits(),
                    stats.max.to_bits()
                )));
            }
        }
        "qc" | "qr" | "qi" | "qs" | "qg" => {
            if stats.min < 0.0 || stats.max >= 1.0 {
                return Err(invalid(format!(
                    "live hydrometeor carrier {name} must remain in [0,1); range bits={:08x}..{:08x}",
                    stats.min.to_bits(), stats.max.to_bits()
                )));
            }
        }
        "refc_dbz" | "refd_1000m_agl_dbz" | "refd_4000m_agl_dbz" => {
            if stats.min < -200.0 || stats.max > 100.0 {
                return Err(invalid(format!(
                    "direct reflectivity carrier {name} leaves [-200,100] dBZ"
                )));
            }
        }
        _ => {}
    }
    if admission == LiveAdmission::RealProduct {
        if name == "w3d" && stats.nonzero_count == 0 {
            return Err(invalid(
                "real-product live w3d is all zero; zero-w fallback is forbidden",
            ));
        }
        if name == "th3d" && stats.min.to_bits() == stats.max.to_bits() {
            return Err(invalid(
                "real-product live th3d is horizontally and vertically uniform",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_duplicate_pair(
    writer: &mut BufWriter<File>,
    payload_hasher: &mut Sha256,
    offset: &mut u64,
    records: &mut Vec<CarrierRecord>,
    a_name: &str,
    current_name: &str,
    source: &str,
    shape: Vec<usize>,
    values: &[f32],
    stagger: &str,
) -> Result<()> {
    records.push(write_carrier(
        writer,
        payload_hasher,
        offset,
        a_name,
        &[source],
        &format!("prognostic-a-{stagger}"),
        shape.clone(),
        values,
        None,
        &[],
    )?);
    records.push(write_carrier(
        writer,
        payload_hasher,
        offset,
        current_name,
        &[source],
        &format!("stage-current-{stagger}"),
        shape,
        values,
        Some(a_name),
        &[],
    )?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_carrier(
    writer: &mut BufWriter<File>,
    payload_hasher: &mut Sha256,
    offset: &mut u64,
    name: &str,
    sources: &[&str],
    role: &str,
    shape: Vec<usize>,
    values: &[f32],
    duplicate_of: Option<&str>,
    species_order: &[&str],
) -> Result<CarrierRecord> {
    let expected_words = shape.iter().try_fold(1usize, |count, extent| {
        count
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("live carrier {name} shape product overflows")))
    })?;
    if expected_words != values.len() {
        return Err(invalid(format!(
            "live carrier {name} contains {} words; shape {shape:?} requires {expected_words}",
            values.len()
        )));
    }
    let stats = CarrierStats::from_values(name, values)?;
    let nbytes = u64::try_from(
        values
            .len()
            .checked_mul(ITEMSIZE)
            .ok_or_else(|| invalid(format!("live carrier {name} byte count overflows")))?,
    )
    .map_err(|_| invalid(format!("live carrier {name} byte count exceeds u64")))?;
    let carrier_offset = *offset;
    let mut field_hasher = Sha256::default();
    let mut bytes = Vec::with_capacity(IO_CHUNK_WORDS * ITEMSIZE);
    for chunk in values.chunks(IO_CHUNK_WORDS) {
        bytes.clear();
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes)?;
        field_hasher.update(&bytes);
        payload_hasher.update(&bytes);
    }
    *offset = offset
        .checked_add(nbytes)
        .ok_or_else(|| invalid("live payload offset overflows u64"))?;
    Ok(CarrierRecord {
        name: name.to_string(),
        sources: sources.iter().map(|value| (*value).to_string()).collect(),
        role: role.to_string(),
        shape,
        offset_nbytes: carrier_offset,
        nbytes,
        sha256: sha256::hex(&field_hasher.finalize()),
        stats,
        duplicate_of: duplicate_of.map(str::to_string),
        species_order: species_order
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
    })
}

pub(crate) fn crop_profile(values: &[f32], full_nz: usize, crop: PhysicalCrop) -> Result<Vec<f32>> {
    if values.len() != full_nz {
        return Err(invalid(
            "live base profile length changed after contract validation",
        ));
    }
    Ok(values[crop.k_start..crop.k_start + crop.nz].to_vec())
}

pub(crate) fn crop_mass(
    values: &[f32],
    grid: GridShape,
    crop: PhysicalCrop,
    name: &str,
) -> Result<Vec<f32>> {
    crop_zyx(
        values,
        [grid.nz, grid.ny, grid.nx],
        [crop.k_start, crop.j_start, crop.i_start],
        [crop.nz, crop.ny, crop.nx],
        name,
    )
}

pub(crate) fn crop_surface(
    values: &[f32],
    grid: GridShape,
    crop: PhysicalCrop,
) -> Result<Vec<f32>> {
    let expected = grid
        .ny
        .checked_mul(grid.nx)
        .ok_or_else(|| invalid("live surface source size overflows"))?;
    if values.len() != expected {
        return Err(invalid("terrain length changed after contract validation"));
    }
    let mut output = Vec::with_capacity(crop.ny * crop.nx);
    for j in crop.j_start..crop.j_start + crop.ny {
        let first = j * grid.nx + crop.i_start;
        output.extend_from_slice(&values[first..first + crop.nx]);
    }
    Ok(output)
}

pub(crate) fn crop_zyx(
    values: &[f32],
    full: [usize; 3],
    start: [usize; 3],
    count: [usize; 3],
    name: &str,
) -> Result<Vec<f32>> {
    let expected = full.iter().try_fold(1usize, |size, extent| {
        size.checked_mul(*extent)
            .ok_or_else(|| invalid(format!("source {name} shape product overflows")))
    })?;
    if values.len() != expected {
        return Err(invalid(format!(
            "source {name} has {} words; expected {expected}",
            values.len()
        )));
    }
    for axis in 0..3 {
        if count[axis] == 0
            || start[axis]
                .checked_add(count[axis])
                .is_none_or(|end| end > full[axis])
        {
            return Err(invalid(format!(
                "source {name} crop axis {axis} start/count exceeds shape"
            )));
        }
    }
    let capacity = count.iter().try_fold(1usize, |size, extent| {
        size.checked_mul(*extent)
            .ok_or_else(|| invalid(format!("cropped {name} shape product overflows")))
    })?;
    let mut output = Vec::with_capacity(capacity);
    for k in start[0]..start[0] + count[0] {
        for j in start[1]..start[1] + count[1] {
            let first = (k * full[1] + j) * full[2] + start[2];
            output.extend_from_slice(&values[first..first + count[2]]);
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn manifest_json(
    contract: &ForecastContract,
    source_receipt: &ForecastReceipt,
    grid: GridShape,
    crop: PhysicalCrop,
    admission: LiveAdmission,
    payload_nbytes: u64,
    payload_sha256: &str,
    records: &[CarrierRecord],
    frame: &ForecastFrame,
) -> JsonValue {
    object(vec![
        ("admission", string(admission.as_str())),
        ("carrier_abi", string(LIVE_CARRIER_ABI)),
        (
            "carriers",
            JsonValue::Array(records.iter().map(CarrierRecord::json).collect()),
        ),
        ("claim_limit", string(LIVE_CLAIM_LIMIT)),
        (
            "crop",
            object(vec![
                (
                    "count_zyx",
                    JsonValue::Array([crop.nz, crop.ny, crop.nx].map(number).to_vec()),
                ),
                (
                    "origin_zyx",
                    JsonValue::Array(
                        [crop.k_start, crop.j_start, crop.i_start]
                            .map(number)
                            .to_vec(),
                    ),
                ),
                (
                    "stagger_rule",
                    string("mass crop plus one terminal face on each staggered axis"),
                ),
            ]),
        ),
        ("dtype", string("<f4")),
        ("itemsize", number(ITEMSIZE)),
        ("order", string("C")),
        (
            "payload",
            object(vec![
                ("file", string(LIVE_PAYLOAD_FILE)),
                ("nbytes", u64_number(payload_nbytes)),
                ("sha256", string(payload_sha256)),
            ]),
        ),
        ("schema", string(LIVE_EXPORT_SCHEMA)),
        (
            "source",
            object(vec![
                ("artifact_nbytes", number(source_receipt.artifact_nbytes)),
                ("artifact_sha256", string(&source_receipt.artifact_sha256)),
                ("contract_sha256", string(&contract.sha256())),
                ("dt_f64_bits", f64_bits(frame.clock().dt_s())),
                (
                    "model_time_f64_bits",
                    f64_bits(frame.clock().model_time_s()),
                ),
                ("profile_id", string(contract.profile_id())),
                ("step", u64_number(frame.clock().step())),
            ]),
        ),
        (
            "source_grid",
            object(vec![
                (
                    "shape_zyx",
                    JsonValue::Array([grid.nz, grid.ny, grid.nx].map(number).to_vec()),
                ),
                (
                    "target_dx_m_f64_bits",
                    f64_bits(scalar_f64(frame, "target_dx_m")),
                ),
                (
                    "target_dy_m_f64_bits",
                    f64_bits(scalar_f64(frame, "target_dy_m")),
                ),
                (
                    "target_dz_m_f64_bits",
                    f64_bits(scalar_f64(frame, "target_dz_m")),
                ),
            ]),
        ),
    ])
}

fn scalar_f64(frame: &ForecastFrame, name: &str) -> f64 {
    frame
        .scheduler_scalars()
        .iter()
        .find_map(|scalar| {
            (scalar.name() == name).then(|| match scalar.value() {
                SchedulerScalarValue::F64(value) => value,
                _ => unreachable!("validated scheduler scalar type"),
            })
        })
        .expect("validated scheduler scalar exists")
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("live-export destination needs a UTF-8 file name"))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| invalid(format!("system clock precedes epoch: {error}")))?
        .as_nanos();
    Ok(output.with_file_name(format!(".{name}.partial-{}-{nonce}", std::process::id())))
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{label} must be 64 lower-case hexadecimal characters"
        )));
    }
    Ok(())
}

fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    )
}

fn string(value: &str) -> JsonValue {
    JsonValue::String(value.to_string())
}

fn number(value: usize) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn u64_number(value: u64) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn f32_bits(value: f32) -> JsonValue {
    string(&format!("{:08x}", value.to_bits()))
}

fn f64_bits(value: f64) -> JsonValue {
    string(&format!("{:016x}", value.to_bits()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::{
        write_forecast_checkpoint_new, ComponentBlobSpec, ForecastArraySpec, ForecastClock,
        SchedulerScalar, SchedulerScalarSpec, SchedulerScalarType,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "gpuwm-cm1-live-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        path
    }

    fn spec(name: &str, shape: Vec<usize>) -> ForecastArraySpec {
        let mut bounds = Vec::new();
        for extent in &shape {
            bounds.push(1);
            bounds.push(i32::try_from(*extent).unwrap());
        }
        ForecastArraySpec::new(name, name, "1", "test", bounds, shape).unwrap()
    }

    fn fixture(root: &Path, mutate: impl Fn(&str, &mut Vec<f32>)) -> (PathBuf, String, String) {
        let (nz, ny, nx) = (3usize, 3usize, 4usize);
        let mut specs = Vec::new();
        for name in PROFILE_SOURCES {
            specs.push(spec(name, vec![nz]));
        }
        for name in ["pp3d", "th3d", "qv", "qc", "qr", "qi", "qs", "qg"] {
            specs.push(spec(name, vec![nz, ny, nx]));
        }
        specs.push(spec("u3d", vec![nz, ny, nx + 1]));
        specs.push(spec("v3d", vec![nz, ny + 1, nx]));
        specs.push(spec("w3d", vec![nz + 1, ny, nx]));
        for name in [
            "terrain_msl",
            "refc_dbz",
            "refd_1000m_agl_dbz",
            "refd_4000m_agl_dbz",
        ] {
            specs.push(spec(name, vec![ny, nx]));
        }
        let scalar_specs = vec![
            SchedulerScalarSpec::new("boundary_ready", SchedulerScalarType::Bool).unwrap(),
            SchedulerScalarSpec::new("forcing_valid_offset_s", SchedulerScalarType::F64).unwrap(),
            SchedulerScalarSpec::new("source_forecast_hour", SchedulerScalarType::U64).unwrap(),
            SchedulerScalarSpec::new("target_dx_m", SchedulerScalarType::F64).unwrap(),
            SchedulerScalarSpec::new("target_dy_m", SchedulerScalarType::F64).unwrap(),
            SchedulerScalarSpec::new("target_dz_m", SchedulerScalarType::F64).unwrap(),
        ];
        let contract = ForecastContract::new(
            SOURCE_PROFILE_ID,
            "11".repeat(32),
            "22".repeat(32),
            specs,
            scalar_specs,
            Vec::<ComponentBlobSpec>::new(),
        )
        .unwrap();
        let mut arrays = Vec::new();
        for source in contract.arrays() {
            let words = source.shape().iter().product::<usize>();
            let mut values = (0..words)
                .map(|index| 0.01 + index as f32 * 0.0001)
                .collect::<Vec<_>>();
            match source.name() {
                "pi0" | "prs0" | "rho0" | "th0" => {
                    values.iter_mut().for_each(|value| *value += 1.0)
                }
                "qv0" | "qv" | "qc" | "qr" | "qi" | "qs" | "qg" => {
                    values.iter_mut().for_each(|value| *value *= 0.001)
                }
                "refc_dbz" | "refd_1000m_agl_dbz" | "refd_4000m_agl_dbz" => {
                    values.iter_mut().for_each(|value| *value -= 10.0)
                }
                _ => {}
            }
            mutate(source.name(), &mut values);
            arrays.push(ForecastArray::new(source.name(), values));
        }
        let scalars = vec![
            SchedulerScalar::new("boundary_ready", SchedulerScalarValue::Bool(false)).unwrap(),
            SchedulerScalar::new("forcing_valid_offset_s", SchedulerScalarValue::F64(0.0)).unwrap(),
            SchedulerScalar::new("source_forecast_hour", SchedulerScalarValue::U64(0)).unwrap(),
            SchedulerScalar::new("target_dx_m", SchedulerScalarValue::F64(100.0)).unwrap(),
            SchedulerScalar::new("target_dy_m", SchedulerScalarValue::F64(100.0)).unwrap(),
            SchedulerScalar::new("target_dz_m", SchedulerScalarValue::F64(200.0)).unwrap(),
        ];
        let frame = ForecastFrame::new(
            ForecastClock::new(0, 0.0, 0.5).unwrap(),
            arrays,
            scalars,
            Vec::new(),
        );
        let artifact = root.join("input.cm1fcp");
        let receipt = write_forecast_checkpoint_new(&contract, &frame, &artifact).unwrap();
        (artifact, contract.sha256(), receipt.artifact_sha256)
    }

    #[test]
    fn strict_export_crops_staggers_duplicates_and_publishes_atomically() {
        let root = temporary("roundtrip");
        let (artifact, contract_sha, artifact_sha) = fixture(&root, |_, _| {});
        let output = root.join("export");
        let crop = PhysicalCrop {
            k_start: 1,
            j_start: 1,
            i_start: 1,
            nz: 2,
            ny: 2,
            nx: 2,
        };
        let receipt = export_checkpoint_live_new(
            &artifact,
            &output,
            &contract_sha,
            &artifact_sha,
            Some(crop),
            LiveAdmission::Generic,
        )
        .unwrap();
        assert_eq!(receipt.carrier_count, 21);
        assert!(output.join(LIVE_PAYLOAD_FILE).is_file());
        let manifest_bytes = fs::read(output.join(LIVE_MANIFEST_FILE)).unwrap();
        assert_eq!(sha256::digest_hex(&manifest_bytes), receipt.manifest_sha256);
        assert_eq!(
            manifest_bytes,
            crate::json::parse(&manifest_bytes)
                .unwrap()
                .canonical_bytes()
        );
        let manifest = crate::json::parse(&manifest_bytes).unwrap();
        assert_eq!(
            manifest.get("schema").unwrap().as_str().unwrap(),
            LIVE_EXPORT_SCHEMA
        );
        let carriers = manifest.get("carriers").unwrap().as_array().unwrap();
        assert_eq!(carriers.len(), 21);
        let names = carriers
            .iter()
            .map(|carrier| carrier.get("name").unwrap().as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "pi0",
                "prs0",
                "qv0",
                "rho0",
                "th0",
                "u0",
                "v0",
                "ua",
                "u3d",
                "va",
                "v3d",
                "wa",
                "w3d",
                "ppi",
                "pp3d",
                "ppx",
                "tha",
                "th3d",
                "qa",
                "q3d",
                "terrain_msl"
            ]
        );
        for (a, current) in [(7, 8), (9, 10), (11, 12), (13, 14), (16, 17), (18, 19)] {
            assert_eq!(
                carriers[a].get("sha256").unwrap().as_str().unwrap(),
                carriers[current].get("sha256").unwrap().as_str().unwrap()
            );
            assert_eq!(
                carriers[current]
                    .get("duplicate_of")
                    .unwrap()
                    .as_str()
                    .unwrap(),
                carriers[a].get("name").unwrap().as_str().unwrap()
            );
        }
        assert_eq!(
            carriers[15].get("duplicate_of").unwrap().as_str().unwrap(),
            "ppi"
        );
        assert_eq!(
            carriers[13].get("sha256").unwrap().as_str().unwrap(),
            carriers[15].get("sha256").unwrap().as_str().unwrap()
        );
        assert_eq!(
            carriers[7]
                .get("shape")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![2, 2, 3]
        );
        assert_eq!(
            carriers[11]
                .get("shape")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![3, 2, 2]
        );
        assert!(export_checkpoint_live_new(
            &artifact,
            &output,
            &contract_sha,
            &artifact_sha,
            Some(crop),
            LiveAdmission::Generic,
        )
        .unwrap_err()
        .to_string()
        .contains("refusing to replace"));
    }

    #[test]
    fn strict_export_refuses_hash_range_crop_and_real_profile_drift() {
        let root = temporary("negative");
        let (artifact, contract_sha, artifact_sha) = fixture(&root, |_, _| {});
        let error = export_checkpoint_live_new(
            &artifact,
            root.join("wrong-artifact"),
            &contract_sha,
            &"ff".repeat(32),
            None,
            LiveAdmission::Generic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("independently expected"));
        let error = export_checkpoint_live_new(
            &artifact,
            root.join("bad-crop"),
            &contract_sha,
            &artifact_sha,
            Some(PhysicalCrop {
                k_start: 2,
                j_start: 0,
                i_start: 0,
                nz: 2,
                ny: 1,
                nx: 1,
            }),
            LiveAdmission::Generic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exceeds source physical grid"));
        let error = export_checkpoint_live_new(
            &artifact,
            root.join("wrong-profile"),
            &contract_sha,
            &artifact_sha,
            None,
            LiveAdmission::RealProduct,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires [75,300,300]"));

        let bad_root = root.join("bad-values");
        fs::create_dir(&bad_root).unwrap();
        let (bad_artifact, bad_contract_sha, bad_artifact_sha) =
            fixture(&bad_root, |name, values| {
                if name == "qc" {
                    values[0] = -0.5;
                }
            });
        let error = export_checkpoint_live_new(
            bad_artifact,
            bad_root.join("export"),
            &bad_contract_sha,
            &bad_artifact_sha,
            None,
            LiveAdmission::Generic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("hydrometeor carrier qc"));
    }
}
