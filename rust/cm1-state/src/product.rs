//! Atomic Rust-owned CM1 real-product initialization carrier arena.
//!
//! This module composes the strict checkpoint transport, terrain/static
//! auxiliaries, and audited moist pressure/density completion without a
//! Python numerical seam.

use crate::forecast::{
    read_forecast_checkpoint_embedded, read_forecast_history_embedded, ForecastArray,
    ForecastContract, ForecastFrame, ForecastReceipt, SchedulerScalarValue,
};
use crate::initialization::{
    complete_moist_initialization, MoistInitializationAdmission, MoistInitializationConfig,
};
use crate::json::JsonValue;
use crate::live::{self, crop_mass, crop_zyx, GridShape, LiveAdmission, PhysicalCrop};
use crate::sha256::{self, Sha256};
use crate::terrain::{
    compute_terrain_static, HorizontalCrop, TerrainStaticAdmission, TerrainStaticConfig,
    TerrainStaticFrame, TERRAIN_STATIC_BASE_POLICY,
};
use crate::{Result, StateError};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const INITIALIZED_PRODUCT_SCHEMA: &str = "gpuwm-cm1.cuda-initialized-product-export/v1";
pub const INITIALIZED_PRODUCT_RECEIPT_SCHEMA: &str =
    "gpuwm-cm1.cuda-initialized-product-export-receipt/v1";
pub const INITIALIZED_PRODUCT_CARRIER_ABI: &str = "gpuwm-cm1.cuda-initialized-product-carriers/v1";
pub const INITIALIZED_PRODUCT_PAYLOAD_FILE: &str = "initialized-carriers.f32le";
pub const INITIALIZED_PRODUCT_MANIFEST_FILE: &str = "manifest.json";
pub const INITIALIZED_PRODUCT_RECEIPT_FILE: &str = "receipt.json";
pub const INITIALIZED_PRODUCT_PUBLICATION_CONTRACT: &str = "write the binary32 payload, canonical manifest, and canonical receipt into a new sibling staging directory; synchronize every file; rename the directory only after all hashes validate; refuse an existing destination";
pub const INITIALIZED_PRODUCT_CLAIM_LIMIT: &str = "byte-exact Rust-owned CM1 r21.1 physical-domain initialization carriers from an independently hashed real forecast state; includes terrain coordinates/static metrics and audited ibalance=1 moist pressure/density completion; rho0_3d/rr0/rf0/rrf0 retain the explicit replaceable horizontal-broadcast base policy; no halos, tendencies, closure state, pressure operator, dynamics, physics, trajectory, forecast, tornado, or model-validation claim";
pub const INITIALIZED_PRODUCT_STAGGER_RULE: &str = "all vertical levels retained; horizontal mass crop retains one terminal face on each u/v axis and all rigid-boundary z faces";

pub const INITIALIZED_PRODUCT_CARRIER_NAMES: [&str; 49] = [
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
    "terrain_msl",
    "zh",
    "zf",
    "mh",
    "rmh",
    "mf",
    "rmf",
    "c1",
    "c2",
    "rho0_3d",
    "rr0",
    "rf0",
    "rrf0",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "dzdx",
    "dzdy",
    "gx",
    "gxu",
    "gy",
    "gyv",
    "prs",
    "rho",
    "rr",
    "rf",
];

const SEALED_PRODUCT_ARTIFACT_SHA256: &str =
    "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f";
const SEALED_BOUNDARY_ANALYSIS_ARTIFACT_SHA256: &str =
    "28e89cae538cf349c713ea42630dd8cc664a084c0c88f71eac1b3a491f14d189";
const SEALED_F02_ROI_ANALYSIS_ARTIFACT_SHA256: &str =
    "356fe3022ce83909f5c14e7ce9a3639811d9f6dfc5ed9b33a0782cfffb7cd79d";
const SEALED_PRODUCT_CONTRACT_SHA256: &str =
    "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12";
const SEALED_F02_ROI_CONTRACT_SHA256: &str =
    "a9ccd6f0c6650fd89cd42f16977209f57695464181bcb62beff14e9a9f61d896";
const AUTHORITY_TARBALL_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
const AUTHORITY_FILES: [(&str, &str, &str); 7] = [
    (
        "src/adv_routines.F",
        "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00",
        "5990-6037,9255-9322",
    ),
    (
        "src/base.F",
        "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef",
        "2160-2190",
    ),
    (
        "src/bc.F",
        "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
        "12-172",
    ),
    (
        "src/constants.F",
        "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
        "7-18,69-140",
    ),
    (
        "src/init3d.F",
        "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
        "1733-1785,2365-2467",
    ),
    (
        "src/init_terrain.F",
        "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c",
        "175-297",
    ),
    (
        "src/param.F",
        "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
        "6325-6327,7789-7812,8162-8207,8464-8492,8603-8627,8988-9062",
    ),
];
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
const STATIC_AFTER_TERRAIN: [&str; 24] = [
    "zh", "zf", "mh", "rmh", "mf", "rmf", "c1", "c2", "rho0_3d", "rr0", "rf0", "rrf0", "gz", "rgz",
    "gzu", "rgzu", "gzv", "rgzv", "dzdx", "dzdy", "gx", "gxu", "gy", "gyv",
];
const ITEMSIZE: usize = 4;
const IO_CHUNK_WORDS: usize = 64 * 1024;

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitializedProductAdmission {
    Generic,
    RealProduct,
}

impl InitializedProductAdmission {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic-schema",
            Self::RealProduct => "real-product-300x300x75",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitializedProductSourceKind {
    InitialCheckpoint,
    BoundaryAnalysisHistory,
    BoundaryAnalysisHistoryF02Roi,
}

impl InitializedProductSourceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::InitialCheckpoint => "initial-checkpoint",
            Self::BoundaryAnalysisHistory => "boundary-analysis-history-f01",
            Self::BoundaryAnalysisHistoryF02Roi => "boundary-analysis-history-f02-roi",
        }
    }

    fn sealed_artifact_sha256(self) -> &'static str {
        match self {
            Self::InitialCheckpoint => SEALED_PRODUCT_ARTIFACT_SHA256,
            Self::BoundaryAnalysisHistory => SEALED_BOUNDARY_ANALYSIS_ARTIFACT_SHA256,
            Self::BoundaryAnalysisHistoryF02Roi => SEALED_F02_ROI_ANALYSIS_ARTIFACT_SHA256,
        }
    }

    fn sealed_contract_sha256(self) -> &'static str {
        match self {
            Self::InitialCheckpoint | Self::BoundaryAnalysisHistory => {
                SEALED_PRODUCT_CONTRACT_SHA256
            }
            Self::BoundaryAnalysisHistoryF02Roi => SEALED_F02_ROI_CONTRACT_SHA256,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitializedProductReceipt {
    pub source_artifact_sha256: String,
    pub source_contract_sha256: String,
    pub payload_sha256: String,
    pub payload_nbytes: u64,
    pub manifest_sha256: String,
    pub manifest_nbytes: usize,
    pub carrier_count: usize,
    pub admission: InitializedProductAdmission,
}

impl InitializedProductReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("admission", string(self.admission.as_str())),
            ("carrier_abi", string(INITIALIZED_PRODUCT_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            ("claim_limit", string(INITIALIZED_PRODUCT_CLAIM_LIMIT)),
            ("manifest_nbytes", number(self.manifest_nbytes)),
            ("manifest_sha256", string(&self.manifest_sha256)),
            ("payload_nbytes", u64_number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            (
                "publication_contract",
                string(INITIALIZED_PRODUCT_PUBLICATION_CONTRACT),
            ),
            ("schema", string(INITIALIZED_PRODUCT_RECEIPT_SCHEMA)),
            (
                "source_artifact_sha256",
                string(&self.source_artifact_sha256),
            ),
            (
                "source_contract_sha256",
                string(&self.source_contract_sha256),
            ),
        ]);
        String::from_utf8(value.canonical_bytes())
            .expect("canonical initialized-product receipt is UTF-8")
    }
}

#[derive(Clone, Debug)]
struct CarrierStats {
    finite_count: usize,
    nonzero_count: usize,
    minimum: f32,
    maximum: f32,
}

impl CarrierStats {
    fn new(name: &str, values: &[f32]) -> Result<Self> {
        if values.is_empty() {
            return Err(invalid(format!(
                "initialized-product carrier {name} is empty"
            )));
        }
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        let mut nonzero_count = 0usize;
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "initialized-product carrier {name} has non-finite word {index}"
                )));
            }
            minimum = minimum.min(value);
            maximum = maximum.max(value);
            if value.to_bits() & 0x7fff_ffff != 0 {
                nonzero_count += 1;
            }
        }
        Ok(Self {
            finite_count: values.len(),
            nonzero_count,
            minimum,
            maximum,
        })
    }

    fn json(&self) -> JsonValue {
        object(vec![
            ("finite_count", number(self.finite_count)),
            ("maximum_f32_bits", f32_bits(self.maximum)),
            ("minimum_f32_bits", f32_bits(self.minimum)),
            ("nonzero_count", number(self.nonzero_count)),
        ])
    }
}

#[derive(Clone, Debug)]
struct CarrierRecord {
    name: String,
    role: String,
    source_arrays: Vec<String>,
    shape: Vec<usize>,
    offset_nbytes: u64,
    nbytes: u64,
    sha256: String,
    stats: CarrierStats,
}

impl CarrierRecord {
    fn json(&self) -> JsonValue {
        object(vec![
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
                JsonValue::Array(self.source_arrays.iter().map(|item| string(item)).collect()),
            ),
            ("statistics", self.stats.json()),
        ])
    }
}

#[derive(Clone, Debug)]
struct CarrierView<'a> {
    name: &'static str,
    role: &'static str,
    source_arrays: &'static [&'static str],
    shape: Vec<usize>,
    values: &'a [f32],
}

/// Compose and atomically publish the fixed 49-carrier product arena.
#[allow(clippy::too_many_arguments)]
pub fn export_initialized_product_new(
    input: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
    expected_contract_sha256: &str,
    expected_artifact_sha256: &str,
    crop: Option<HorizontalCrop>,
    terrain_config: TerrainStaticConfig,
    moist_config: MoistInitializationConfig,
    admission: InitializedProductAdmission,
) -> Result<InitializedProductReceipt> {
    export_initialized_product_source_new(
        input,
        output_directory,
        expected_contract_sha256,
        expected_artifact_sha256,
        crop,
        terrain_config,
        moist_config,
        admission,
        InitializedProductSourceKind::InitialCheckpoint,
    )
}

/// Compose a new model-time-zero product state from the independently sealed
/// f01 real-HRRR history frame.  The atmospheric values and f01 provenance
/// remain unchanged; the consuming GM1 run owns a new relative model clock.
#[allow(clippy::too_many_arguments)]
pub fn export_initialized_product_history_new(
    input: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
    expected_contract_sha256: &str,
    expected_artifact_sha256: &str,
    crop: Option<HorizontalCrop>,
    terrain_config: TerrainStaticConfig,
    moist_config: MoistInitializationConfig,
    admission: InitializedProductAdmission,
) -> Result<InitializedProductReceipt> {
    export_initialized_product_source_new(
        input,
        output_directory,
        expected_contract_sha256,
        expected_artifact_sha256,
        crop,
        terrain_config,
        moist_config,
        admission,
        InitializedProductSourceKind::BoundaryAnalysisHistory,
    )
}

/// Compose a new GM1 product from the independently sealed real-HRRR f02
/// history whose physical target grid is centered on the observed low-level
/// reflectivity maximum.  This is a coordinate-selected remap, never an array
/// translation, and it retains the absolute f02 source clock in the manifest.
#[allow(clippy::too_many_arguments)]
pub fn export_initialized_product_f02_roi_history_new(
    input: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
    expected_contract_sha256: &str,
    expected_artifact_sha256: &str,
    crop: Option<HorizontalCrop>,
    terrain_config: TerrainStaticConfig,
    moist_config: MoistInitializationConfig,
    admission: InitializedProductAdmission,
) -> Result<InitializedProductReceipt> {
    export_initialized_product_source_new(
        input,
        output_directory,
        expected_contract_sha256,
        expected_artifact_sha256,
        crop,
        terrain_config,
        moist_config,
        admission,
        InitializedProductSourceKind::BoundaryAnalysisHistoryF02Roi,
    )
}

#[allow(clippy::too_many_arguments)]
fn export_initialized_product_source_new(
    input: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
    expected_contract_sha256: &str,
    expected_artifact_sha256: &str,
    crop: Option<HorizontalCrop>,
    terrain_config: TerrainStaticConfig,
    moist_config: MoistInitializationConfig,
    admission: InitializedProductAdmission,
    source_kind: InitializedProductSourceKind,
) -> Result<InitializedProductReceipt> {
    validate_sha256(
        expected_contract_sha256,
        "expected forecast-state contract SHA-256",
    )?;
    validate_sha256(
        expected_artifact_sha256,
        "expected forecast-state artifact SHA-256",
    )?;
    if admission == InitializedProductAdmission::RealProduct
        && (expected_contract_sha256 != source_kind.sealed_contract_sha256()
            || expected_artifact_sha256 != source_kind.sealed_artifact_sha256())
    {
        return Err(invalid(format!(
            "real-product initialized export admits only the independently sealed {} identity",
            source_kind.as_str()
        )));
    }
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace initialized-product destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "initialized-product parent directory does not exist: {}",
            parent.display()
        )));
    }

    let (contract, restored) = match source_kind {
        InitializedProductSourceKind::InitialCheckpoint => {
            read_forecast_checkpoint_embedded(input.as_ref(), expected_contract_sha256)?
        }
        InitializedProductSourceKind::BoundaryAnalysisHistory
        | InitializedProductSourceKind::BoundaryAnalysisHistoryF02Roi => {
            read_forecast_history_embedded(input.as_ref(), expected_contract_sha256)?
        }
    };
    if restored.receipt().artifact_sha256 != expected_artifact_sha256 {
        return Err(invalid(format!(
            "{} artifact SHA-256 is {}; independently expected {expected_artifact_sha256}",
            source_kind.as_str(),
            restored.receipt().artifact_sha256
        )));
    }
    let live_admission = match admission {
        InitializedProductAdmission::Generic => LiveAdmission::Generic,
        InitializedProductAdmission::RealProduct => LiveAdmission::RealProduct,
    };
    let grid = match source_kind {
        InitializedProductSourceKind::InitialCheckpoint => {
            live::validate_contract_and_frame(&contract, restored.frame(), live_admission)?
        }
        InitializedProductSourceKind::BoundaryAnalysisHistory => {
            live::validate_contract_and_boundary_analysis_frame(
                &contract,
                restored.frame(),
                live_admission,
            )?
        }
        InitializedProductSourceKind::BoundaryAnalysisHistoryF02Roi => {
            live::validate_contract_and_f02_roi_analysis_frame(
                &contract,
                restored.frame(),
                live_admission,
            )?
        }
    };
    let horizontal_crop = crop.unwrap_or_else(|| HorizontalCrop::full(grid.ny, grid.nx));
    let physical_crop = PhysicalCrop {
        k_start: 0,
        j_start: horizontal_crop.j_start,
        i_start: horizontal_crop.i_start,
        nz: grid.nz,
        ny: horizontal_crop.ny,
        nx: horizontal_crop.nx,
    };
    physical_crop.validate(grid.nz, grid.ny, grid.nx)?;

    let arrays = array_map(restored.frame())?;
    for name in SOURCE_ARRAYS {
        live::validate_physical_range(name, array(&arrays, name)?, live_admission)?;
    }
    let spacing_m = [
        scalar_f64(restored.frame(), "target_dx_m")?,
        scalar_f64(restored.frame(), "target_dy_m")?,
        scalar_f64(restored.frame(), "target_dz_m")?,
    ];
    let terrain_admission = match admission {
        InitializedProductAdmission::Generic => TerrainStaticAdmission::Generic,
        InitializedProductAdmission::RealProduct => TerrainStaticAdmission::RealProduct,
    };
    let terrain = compute_terrain_static(
        array(&arrays, "terrain_msl")?,
        array(&arrays, "rho0")?,
        [grid.nz, grid.ny, grid.nx],
        spacing_m,
        horizontal_crop,
        terrain_config,
        terrain_admission,
    )?;

    let u = crop_zyx(
        array(&arrays, "u3d")?,
        [grid.nz, grid.ny, grid.nx + 1],
        [0, physical_crop.j_start, physical_crop.i_start],
        [grid.nz, physical_crop.ny, physical_crop.nx + 1],
        "u3d",
    )?;
    let v = crop_zyx(
        array(&arrays, "v3d")?,
        [grid.nz, grid.ny + 1, grid.nx],
        [0, physical_crop.j_start, physical_crop.i_start],
        [grid.nz, physical_crop.ny + 1, physical_crop.nx],
        "v3d",
    )?;
    let w = crop_zyx(
        array(&arrays, "w3d")?,
        [grid.nz + 1, grid.ny, grid.nx],
        [0, physical_crop.j_start, physical_crop.i_start],
        [grid.nz + 1, physical_crop.ny, physical_crop.nx],
        "w3d",
    )?;
    let theta = crop_mass(array(&arrays, "th3d")?, grid, physical_crop, "th3d")?;
    let mut moisture = Vec::with_capacity(
        MOISTURE_SOURCES
            .len()
            .checked_mul(grid.nz)
            .and_then(|value| value.checked_mul(physical_crop.ny))
            .and_then(|value| value.checked_mul(physical_crop.nx))
            .ok_or_else(|| invalid("initialized-product packed moisture size overflows"))?,
    );
    for name in MOISTURE_SOURCES {
        moisture.extend(crop_mass(array(&arrays, name)?, grid, physical_crop, name)?);
    }
    let moist_admission = match admission {
        InitializedProductAdmission::Generic => MoistInitializationAdmission::Generic,
        InitializedProductAdmission::RealProduct => MoistInitializationAdmission::RealProduct,
    };
    let initialized = complete_moist_initialization(
        array(&arrays, "pi0")?,
        array(&arrays, "th0")?,
        &theta,
        &moisture,
        static_values(&terrain, "zh")?,
        static_values(&terrain, "c1")?,
        static_values(&terrain, "c2")?,
        terrain.bottom_fornberg(),
        terrain.top_fornberg(),
        [grid.nz, physical_crop.ny, physical_crop.nx],
        moist_config,
        moist_admission,
    )?;

    let staging = staging_path(output)?;
    fs::create_dir(&staging)?;
    let result = write_into_directory(
        &staging,
        &contract,
        restored.frame(),
        restored.receipt(),
        grid,
        horizontal_crop,
        admission,
        source_kind,
        terrain_config,
        moist_config,
        &arrays,
        &u,
        &v,
        &w,
        &theta,
        &moisture,
        &terrain,
        &initialized,
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

#[allow(clippy::too_many_arguments)]
fn write_into_directory(
    directory: &Path,
    contract: &ForecastContract,
    source_frame: &ForecastFrame,
    source_receipt: &ForecastReceipt,
    grid: GridShape,
    crop: HorizontalCrop,
    admission: InitializedProductAdmission,
    source_kind: InitializedProductSourceKind,
    terrain_config: TerrainStaticConfig,
    moist_config: MoistInitializationConfig,
    arrays: &BTreeMap<&str, &ForecastArray>,
    u: &[f32],
    v: &[f32],
    w: &[f32],
    theta: &[f32],
    moisture: &[f32],
    terrain: &TerrainStaticFrame,
    initialized: &crate::initialization::MoistInitializationFrame,
) -> Result<InitializedProductReceipt> {
    let mass_shape = vec![grid.nz, crop.ny, crop.nx];
    let mut views = Vec::<CarrierView<'_>>::with_capacity(49);
    for name in PROFILE_SOURCES {
        views.push(CarrierView {
            name,
            role: "static-base-profile",
            source_arrays: match name {
                "pi0" => &["pi0"],
                "prs0" => &["prs0"],
                "qv0" => &["qv0"],
                "rho0" => &["rho0"],
                "th0" => &["th0"],
                "u0" => &["u0"],
                "v0" => &["v0"],
                _ => unreachable!(),
            },
            shape: vec![grid.nz],
            values: array(arrays, name)?,
        });
    }
    for (name, role) in [
        ("ua", "prognostic-a-x-face"),
        ("u3d", "stage-current-x-face"),
    ] {
        views.push(CarrierView {
            name,
            role,
            source_arrays: &["u3d"],
            shape: vec![grid.nz, crop.ny, crop.nx + 1],
            values: u,
        });
    }
    for (name, role) in [
        ("va", "prognostic-a-y-face"),
        ("v3d", "stage-current-y-face"),
    ] {
        views.push(CarrierView {
            name,
            role,
            source_arrays: &["v3d"],
            shape: vec![grid.nz, crop.ny + 1, crop.nx],
            values: v,
        });
    }
    for (name, role) in [
        ("wa", "prognostic-a-z-face"),
        ("w3d", "stage-current-z-face"),
    ] {
        views.push(CarrierView {
            name,
            role,
            source_arrays: &["w3d"],
            shape: vec![grid.nz + 1, crop.ny, crop.nx],
            values: w,
        });
    }
    for (name, role) in [
        ("ppi", "balanced-pressure-perturbation"),
        ("pp3d", "stage-current-balanced-pressure-perturbation"),
        ("ppx", "fresh-balanced-pressure-helper"),
    ] {
        views.push(CarrierView {
            name,
            role,
            source_arrays: &[
                "pi0", "th0", "th3d", "qv", "qc", "qr", "qi", "qs", "qg", "zh",
            ],
            shape: mass_shape.clone(),
            values: initialized.ppi(),
        });
    }
    for (name, role) in [
        ("tha", "prognostic-a-theta-perturbation"),
        ("th3d", "stage-current-theta-perturbation"),
    ] {
        views.push(CarrierView {
            name,
            role,
            source_arrays: &["th3d"],
            shape: mass_shape.clone(),
            values: theta,
        });
    }
    for (name, role) in [
        ("qa", "prognostic-a-packed-moisture"),
        ("q3d", "stage-current-packed-moisture"),
    ] {
        views.push(CarrierView {
            name,
            role,
            source_arrays: &["qv", "qc", "qr", "qi", "qs", "qg"],
            shape: vec![6, grid.nz, crop.ny, crop.nx],
            values: moisture,
        });
    }
    let terrain_source = terrain
        .carrier("terrain_msl")
        .ok_or_else(|| invalid("terrain frame lacks terrain_msl"))?;
    views.push(CarrierView {
        name: "terrain_msl",
        role: terrain_source.role(),
        source_arrays: terrain_source.source_arrays(),
        shape: terrain_source.shape().to_vec(),
        values: terrain_source.values(),
    });
    for name in STATIC_AFTER_TERRAIN {
        let source = terrain
            .carrier(name)
            .ok_or_else(|| invalid(format!("terrain frame lacks {name}")))?;
        views.push(CarrierView {
            name,
            role: source.role(),
            source_arrays: source.source_arrays(),
            shape: source.shape().to_vec(),
            values: source.values(),
        });
    }
    for (name, role, values, sources, shape) in [
        (
            "prs",
            "diagnostic-pressure-pa",
            initialized.prs(),
            &["pi0", "ppi"] as &'static [&'static str],
            mass_shape.clone(),
        ),
        (
            "rho",
            "diagnostic-moist-density",
            initialized.rho(),
            &["prs", "th0", "th3d", "pi0", "ppi", "qv"],
            mass_shape.clone(),
        ),
        (
            "rr",
            "reciprocal-diagnostic-moist-density",
            initialized.rr(),
            &["rho"],
            mass_shape.clone(),
        ),
        (
            "rf",
            "z-face-diagnostic-moist-density",
            initialized.rf(),
            &["rho", "c1", "c2"],
            vec![grid.nz + 1, crop.ny, crop.nx],
        ),
    ] {
        views.push(CarrierView {
            name,
            role,
            source_arrays: sources,
            shape,
            values,
        });
    }
    if views.iter().map(|item| item.name).collect::<Vec<_>>() != INITIALIZED_PRODUCT_CARRIER_NAMES {
        return Err(invalid(
            "initialized-product internal carrier order drifted",
        ));
    }

    let payload_path = directory.join(INITIALIZED_PRODUCT_PAYLOAD_FILE);
    let payload_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&payload_path)?;
    let mut writer = BufWriter::new(payload_file);
    let mut aggregate = Sha256::default();
    let mut offset = 0u64;
    let mut records = Vec::with_capacity(views.len());
    for view in &views {
        records.push(write_carrier(
            &mut writer,
            &mut aggregate,
            &mut offset,
            view,
        )?);
    }
    writer.flush()?;
    let payload_file = writer.into_inner().map_err(|error| error.into_error())?;
    payload_file.sync_all()?;
    if payload_file.metadata()?.len() != offset {
        return Err(invalid(
            "initialized-product payload size changed after synchronized write",
        ));
    }
    let payload_sha256 = sha256::hex(&aggregate.finalize());
    let manifest = manifest_json(
        contract,
        source_frame,
        source_receipt,
        grid,
        crop,
        admission,
        source_kind,
        terrain_config,
        moist_config,
        terrain,
        offset,
        &payload_sha256,
        &records,
    );
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    write_new_synced(
        &directory.join(INITIALIZED_PRODUCT_MANIFEST_FILE),
        &manifest_bytes,
    )?;
    let receipt = InitializedProductReceipt {
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
        &directory.join(INITIALIZED_PRODUCT_RECEIPT_FILE),
        receipt.canonical_json().as_bytes(),
    )?;
    Ok(receipt)
}

fn write_carrier(
    writer: &mut BufWriter<File>,
    aggregate: &mut Sha256,
    offset: &mut u64,
    carrier: &CarrierView<'_>,
) -> Result<CarrierRecord> {
    let expected_words = product(&carrier.shape, carrier.name)?;
    if carrier.values.len() != expected_words {
        return Err(invalid(format!(
            "initialized-product carrier {} contains {} words; shape {:?} requires {expected_words}",
            carrier.name,
            carrier.values.len(),
            carrier.shape
        )));
    }
    let nbytes = u64::try_from(
        carrier
            .values
            .len()
            .checked_mul(ITEMSIZE)
            .ok_or_else(|| invalid("initialized-product carrier byte count overflows"))?,
    )
    .map_err(|_| invalid("initialized-product carrier byte count exceeds u64"))?;
    let carrier_offset = *offset;
    let mut field_hasher = Sha256::default();
    let mut bytes = Vec::with_capacity(IO_CHUNK_WORDS * ITEMSIZE);
    for chunk in carrier.values.chunks(IO_CHUNK_WORDS) {
        bytes.clear();
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes)?;
        field_hasher.update(&bytes);
        aggregate.update(&bytes);
    }
    *offset = offset
        .checked_add(nbytes)
        .ok_or_else(|| invalid("initialized-product payload offset overflows"))?;
    Ok(CarrierRecord {
        name: carrier.name.to_string(),
        role: carrier.role.to_string(),
        source_arrays: carrier
            .source_arrays
            .iter()
            .map(|item| (*item).to_string())
            .collect(),
        shape: carrier.shape.clone(),
        offset_nbytes: carrier_offset,
        nbytes,
        sha256: sha256::hex(&field_hasher.finalize()),
        stats: CarrierStats::new(carrier.name, carrier.values)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn manifest_json(
    contract: &ForecastContract,
    source_frame: &ForecastFrame,
    source_receipt: &ForecastReceipt,
    grid: GridShape,
    crop: HorizontalCrop,
    admission: InitializedProductAdmission,
    source_kind: InitializedProductSourceKind,
    terrain_config: TerrainStaticConfig,
    moist_config: MoistInitializationConfig,
    terrain: &TerrainStaticFrame,
    payload_nbytes: u64,
    payload_sha256: &str,
    records: &[CarrierRecord],
) -> JsonValue {
    let [g, cp, rd, rv, p00] = moist_config.constants_bits();
    object(vec![
        ("admission", string(admission.as_str())),
        (
            "authority",
            object(vec![
                ("release", string("CM1 21.1")),
                (
                    "source_files",
                    JsonValue::Array(
                        AUTHORITY_FILES
                            .iter()
                            .map(|(path, digest, anchors)| {
                                object(vec![
                                    ("anchors", string(anchors)),
                                    ("path", string(path)),
                                    ("sha256", string(digest)),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
            ]),
        ),
        ("base_policy", string(TERRAIN_STATIC_BASE_POLICY)),
        ("carrier_abi", string(INITIALIZED_PRODUCT_CARRIER_ABI)),
        (
            "carriers",
            JsonValue::Array(records.iter().map(CarrierRecord::json).collect()),
        ),
        ("claim_limit", string(INITIALIZED_PRODUCT_CLAIM_LIMIT)),
        (
            "configuration",
            object(vec![
                (
                    "hadvordrs",
                    number(usize::from(terrain_config.hadvordrs)),
                ),
                (
                    "hadvordrv",
                    number(usize::from(terrain_config.hadvordrv)),
                ),
                ("ibalance", number(usize::from(moist_config.ibalance))),
                ("iice", number(usize::from(moist_config.iice))),
                (
                    "lateral_bc_west_east_south_north",
                    JsonValue::Array(
                        terrain_config
                            .lateral_bc
                            .map(|item| number(usize::from(item)))
                            .to_vec(),
                    ),
                ),
                ("psolver", number(usize::from(moist_config.psolver))),
                ("setppi", JsonValue::Bool(moist_config.setppi)),
            ]),
        ),
        (
            "crop",
            object(vec![
                (
                    "count_yx",
                    JsonValue::Array([crop.ny, crop.nx].map(number).to_vec()),
                ),
                (
                    "origin_yx",
                    JsonValue::Array([crop.j_start, crop.i_start].map(number).to_vec()),
                ),
                ("stagger_rule", string(INITIALIZED_PRODUCT_STAGGER_RULE)),
            ]),
        ),
        ("dtype", string("<f4")),
        (
            "fornberg",
            object(vec![
                (
                    "bottom_cgs_f32_bits",
                    JsonValue::Array(terrain.bottom_fornberg().map(f32_bits).to_vec()),
                ),
                (
                    "source_contract",
                    string("param.F:8988-9052 DOUBLE PRECISION recurrence; default-REAL normalized coordinates and default-REAL stored weights"),
                ),
                (
                    "top_cgt_f32_bits",
                    JsonValue::Array(terrain.top_fornberg().map(f32_bits).to_vec()),
                ),
            ]),
        ),
        (
            "formula_contracts",
            object(vec![
                ("base_reciprocals", string("base.F:2160-2190")),
                (
                    "horizontal_gradients",
                    string("adv_routines.F:9255-9322 with flx6 at 6026-6037 and bc2d open-edge halos"),
                ),
                (
                    "moist_initialization",
                    string("init3d.F:1733-1785,2365-2467 with param.F:8988-9052 Fornberg coefficients"),
                ),
                (
                    "terrain_coordinates",
                    string("init_terrain.F:175-297"),
                ),
                (
                    "vertical_metrics",
                    string("param.F:8464-8492,8603-8627"),
                ),
            ]),
        ),
        (
            "initialization",
            object(vec![
                (
                    "constants_f32_bits",
                    object(vec![
                        ("cp", u32_bits(cp)),
                        ("g", u32_bits(g)),
                        ("p00", u32_bits(p00)),
                        ("rd", u32_bits(rd)),
                        ("rv", u32_bits(rv)),
                    ]),
                ),
                (
                    "corrected_carriers",
                    strings(&["ppi", "pp3d", "ppx", "prs", "rho", "rr", "rf"]),
                ),
                (
                    "duplicate_groups",
                    JsonValue::Array(
                        [
                            &["ua", "u3d"] as &[&str],
                            &["va", "v3d"],
                            &["wa", "w3d"],
                            &["ppi", "pp3d", "ppx"],
                            &["tha", "th3d"],
                            &["qa", "q3d"],
                        ]
                        .iter()
                        .map(|group| strings(group))
                        .collect(),
                    ),
                ),
                (
                    "moist_species_order",
                    strings(&["qv", "qc", "qr", "qi", "qs", "qg"]),
                ),
                (
                    "ppi_small_value_rule",
                    string("if abs(ppi) < 1.0e-6 then assign positive binary32 zero before duplicating ppi into pp3d and ppx"),
                ),
                (
                    "qv_density_rule",
                    string("thv uses source qv unchanged; only the density gas-constant factor uses max(0.0,qv)"),
                ),
                (
                    "source_preserved_carriers",
                    strings(&[
                        "pi0", "prs0", "qv0", "rho0", "th0", "u0", "v0", "ua", "u3d",
                        "va", "v3d", "wa", "w3d", "tha", "th3d", "qa", "q3d",
                        "terrain_msl",
                    ]),
                ),
                (
                    "statement_order",
                    string("init3d.F:1733-1785 then 2365-2467 in source loop and assignment order"),
                ),
            ]),
        ),
        ("itemsize", number(ITEMSIZE)),
        ("order", string("C")),
        (
            "payload",
            object(vec![
                ("file", string(INITIALIZED_PRODUCT_PAYLOAD_FILE)),
                ("nbytes", u64_number(payload_nbytes)),
                ("sha256", string(payload_sha256)),
            ]),
        ),
        ("schema", string(INITIALIZED_PRODUCT_SCHEMA)),
        (
            "source",
            object(vec![
                ("kind", string(source_kind.as_str())),
                ("artifact_nbytes", number(source_receipt.artifact_nbytes)),
                ("artifact_sha256", string(&source_receipt.artifact_sha256)),
                ("contract_sha256", string(&contract.sha256())),
                ("dt_f64_bits", f64_bits(source_frame.clock().dt_s())),
                (
                    "model_time_f64_bits",
                    f64_bits(source_frame.clock().model_time_s()),
                ),
                ("profile_id", string(contract.profile_id())),
                ("step", u64_number(source_frame.clock().step())),
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
                    f64_bits(scalar_f64(source_frame, "target_dx_m").unwrap()),
                ),
                (
                    "target_dy_m_f64_bits",
                    f64_bits(scalar_f64(source_frame, "target_dy_m").unwrap()),
                ),
                (
                    "target_dz_m_f64_bits",
                    f64_bits(scalar_f64(source_frame, "target_dz_m").unwrap()),
                ),
            ]),
        ),
    ])
}

fn array_map(frame: &ForecastFrame) -> Result<BTreeMap<&str, &ForecastArray>> {
    let mut arrays = BTreeMap::new();
    for item in frame.arrays() {
        if arrays.insert(item.name(), item).is_some() {
            return Err(invalid(format!(
                "initialized-product restored frame repeats array {:?}",
                item.name()
            )));
        }
    }
    Ok(arrays)
}

fn array<'a>(arrays: &BTreeMap<&str, &'a ForecastArray>, name: &str) -> Result<&'a [f32]> {
    arrays.get(name).map(|item| item.values()).ok_or_else(|| {
        invalid(format!(
            "initialized-product restored frame lacks array {name:?}"
        ))
    })
}

fn static_values<'a>(frame: &'a TerrainStaticFrame, name: &str) -> Result<&'a [f32]> {
    frame
        .carrier(name)
        .map(|carrier| carrier.values())
        .ok_or_else(|| invalid(format!("terrain frame lacks carrier {name:?}")))
}

fn scalar_f64(frame: &ForecastFrame, name: &str) -> Result<f64> {
    match frame
        .scheduler_scalars()
        .iter()
        .find(|scalar| scalar.name() == name)
        .ok_or_else(|| invalid(format!("initialized-product source lacks scalar {name:?}")))?
        .value()
    {
        SchedulerScalarValue::F64(value) => Ok(value),
        other => Err(invalid(format!(
            "initialized-product scalar {name:?} has wrong type {other:?}"
        ))),
    }
}

fn product(shape: &[usize], name: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("initialized-product {name} shape overflows")))
    })
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|item| item.to_str())
        .ok_or_else(|| invalid("initialized-product destination needs a UTF-8 file name"))?;
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

fn strings(values: &[&str]) -> JsonValue {
    JsonValue::Array(values.iter().map(|value| string(value)).collect())
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

fn u32_bits(value: u32) -> JsonValue {
    string(&format!("{value:08x}"))
}

fn f32_bits(value: f32) -> JsonValue {
    u32_bits(value.to_bits())
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

    fn temporary(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "gpuwm-cm1-product-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        path
    }

    fn spec(name: &str, shape: Vec<usize>) -> ForecastArraySpec {
        let mut bounds = Vec::with_capacity(shape.len() * 2);
        for extent in &shape {
            bounds.push(1);
            bounds.push(i32::try_from(*extent).unwrap());
        }
        ForecastArraySpec::new(name, name, "1", "test", bounds, shape).unwrap()
    }

    fn source_shape(name: &str, nz: usize, ny: usize, nx: usize) -> Vec<usize> {
        match name {
            "pi0" | "prs0" | "qv0" | "rho0" | "th0" | "u0" | "v0" => vec![nz],
            "pp3d" | "th3d" | "qv" | "qc" | "qr" | "qi" | "qs" | "qg" => {
                vec![nz, ny, nx]
            }
            "u3d" => vec![nz, ny, nx + 1],
            "v3d" => vec![nz, ny + 1, nx],
            "w3d" => vec![nz + 1, ny, nx],
            "terrain_msl" | "refc_dbz" | "refd_1000m_agl_dbz" | "refd_4000m_agl_dbz" => {
                vec![ny, nx]
            }
            other => panic!("unhandled product fixture source {other}"),
        }
    }

    fn fixture(root: &Path) -> (PathBuf, String, String) {
        let (nz, ny, nx) = (4usize, 4usize, 5usize);
        let specs = SOURCE_ARRAYS
            .iter()
            .map(|name| spec(name, source_shape(name, nz, ny, nx)))
            .collect::<Vec<_>>();
        let scalar_specs = vec![
            SchedulerScalarSpec::new("boundary_ready", SchedulerScalarType::Bool).unwrap(),
            SchedulerScalarSpec::new("forcing_valid_offset_s", SchedulerScalarType::F64).unwrap(),
            SchedulerScalarSpec::new("source_forecast_hour", SchedulerScalarType::U64).unwrap(),
            SchedulerScalarSpec::new("target_dx_m", SchedulerScalarType::F64).unwrap(),
            SchedulerScalarSpec::new("target_dy_m", SchedulerScalarType::F64).unwrap(),
            SchedulerScalarSpec::new("target_dz_m", SchedulerScalarType::F64).unwrap(),
        ];
        let contract = ForecastContract::new(
            "hrrr-cm1-real-state-v1",
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
                .map(|index| 0.01_f32 + index as f32 * 0.0001_f32)
                .collect::<Vec<_>>();
            match source.name() {
                "pi0" => values
                    .iter_mut()
                    .enumerate()
                    .for_each(|(k, value)| *value = 0.99_f32 - k as f32 * 0.01_f32),
                "prs0" => values.iter_mut().for_each(|value| *value += 90_000.0),
                "rho0" => values.iter_mut().for_each(|value| *value += 1.0),
                "th0" => values.iter_mut().for_each(|value| *value += 300.0),
                "qv0" | "qv" | "qc" | "qr" | "qi" | "qs" | "qg" => {
                    values.iter_mut().for_each(|value| *value *= 0.001)
                }
                "terrain_msl" => {
                    for (index, value) in values.iter_mut().enumerate() {
                        *value = 10.0_f32 + index as f32 * 0.25_f32;
                    }
                }
                "refc_dbz" | "refd_1000m_agl_dbz" | "refd_4000m_agl_dbz" => {
                    values.iter_mut().for_each(|value| *value -= 10.0)
                }
                _ => {}
            }
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

    fn terrain_config() -> TerrainStaticConfig {
        TerrainStaticConfig::new(5, 5, [2; 4]).unwrap()
    }

    fn moist_config() -> MoistInitializationConfig {
        MoistInitializationConfig::new(
            f32::from_bits(0x411c_f5c3),
            f32::from_bits(0x447b_6ccd),
            f32::from_bits(0x438f_851f),
            f32::from_bits(0x43e6_c000),
            f32::from_bits(0x47c3_5000),
            3,
            1,
            true,
            1,
        )
        .unwrap()
    }

    #[test]
    fn generic_combined_export_is_canonical_contiguous_and_non_aliasing() {
        let root = temporary("roundtrip");
        let (artifact, contract_sha, artifact_sha) = fixture(&root);
        let output = root.join("export");
        let receipt = export_initialized_product_new(
            &artifact,
            &output,
            &contract_sha,
            &artifact_sha,
            Some(HorizontalCrop {
                j_start: 1,
                i_start: 1,
                ny: 2,
                nx: 3,
            }),
            terrain_config(),
            moist_config(),
            InitializedProductAdmission::Generic,
        )
        .unwrap();
        assert_eq!(receipt.carrier_count, 49);
        let payload = fs::read(output.join(INITIALIZED_PRODUCT_PAYLOAD_FILE)).unwrap();
        assert_eq!(payload.len() as u64, receipt.payload_nbytes);
        assert_eq!(sha256::digest_hex(&payload), receipt.payload_sha256);
        let manifest_bytes = fs::read(output.join(INITIALIZED_PRODUCT_MANIFEST_FILE)).unwrap();
        assert_eq!(sha256::digest_hex(&manifest_bytes), receipt.manifest_sha256);
        assert_eq!(
            manifest_bytes,
            crate::json::parse(&manifest_bytes)
                .unwrap()
                .canonical_bytes()
        );
        let manifest = crate::json::parse(&manifest_bytes).unwrap();
        let carriers = manifest.get("carriers").unwrap().as_array().unwrap();
        assert_eq!(
            carriers
                .iter()
                .map(|item| item.get("name").unwrap().as_str().unwrap())
                .collect::<Vec<_>>(),
            INITIALIZED_PRODUCT_CARRIER_NAMES
        );
        let carrier = |name: &str| {
            carriers
                .iter()
                .find(|item| item.get("name").unwrap().as_str().unwrap() == name)
                .unwrap()
        };
        for group in [
            &["ua", "u3d"] as &[&str],
            &["va", "v3d"],
            &["wa", "w3d"],
            &["ppi", "pp3d", "ppx"],
            &["tha", "th3d"],
            &["qa", "q3d"],
        ] {
            let hashes = group
                .iter()
                .map(|name| carrier(name).get("sha256").unwrap().as_str().unwrap())
                .collect::<std::collections::BTreeSet<_>>();
            let offsets = group
                .iter()
                .map(|name| {
                    carrier(name)
                        .get("offset_nbytes")
                        .unwrap()
                        .as_u64()
                        .unwrap()
                })
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(hashes.len(), 1, "{group:?}");
            assert_eq!(offsets.len(), group.len(), "{group:?}");
        }
        assert!(export_initialized_product_new(
            &artifact,
            &output,
            &contract_sha,
            &artifact_sha,
            None,
            terrain_config(),
            moist_config(),
            InitializedProductAdmission::Generic,
        )
        .unwrap_err()
        .to_string()
        .contains("refusing to replace"));
    }

    #[test]
    fn combined_export_refuses_unsealed_product_hash_and_bad_crop() {
        let root = temporary("negative");
        let (artifact, contract_sha, artifact_sha) = fixture(&root);
        let error = export_initialized_product_new(
            &artifact,
            root.join("unsealed"),
            &contract_sha,
            &artifact_sha,
            None,
            terrain_config(),
            moist_config(),
            InitializedProductAdmission::RealProduct,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("sealed initial-checkpoint identity"),
            "{error}"
        );
        let error = export_initialized_product_new(
            &artifact,
            root.join("bad-crop"),
            &contract_sha,
            &artifact_sha,
            Some(HorizontalCrop {
                j_start: 3,
                i_start: 0,
                ny: 2,
                nx: 1,
            }),
            terrain_config(),
            moist_config(),
            InitializedProductAdmission::Generic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exceeds source physical grid"), "{error}");
        let error = export_initialized_product_new(
            &artifact,
            root.join("wrong-hash"),
            &contract_sha,
            &"ff".repeat(32),
            None,
            terrain_config(),
            moist_config(),
            InitializedProductAdmission::Generic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("independently expected"), "{error}");
    }
}
