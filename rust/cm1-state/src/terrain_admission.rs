//! Rust-owned terrain/base/open-boundary auxiliary restart contract.
//!
//! The artifact is a strict `CM1FCP1` specialization.  It complements the
//! independently verified 49-carrier `CM1IPR1` physical-domain restart with
//! CM1 source-valid base/halo arrays, terrain surface halos, held/stage
//! dynamics carriers, and all four persistent radiative-boundary histories.
//! The embedded identity binds the exact source restart, authority sources,
//! grid/crop, spacing, and admission profile.

use crate::forecast::{
    read_forecast_checkpoint_embedded, write_forecast_checkpoint_new, ComponentBlob,
    ComponentBlobSpec, ForecastArray, ForecastArraySpec, ForecastClock, ForecastContract,
    ForecastFrame, ForecastReceipt, RestoredForecast, SchedulerScalar, SchedulerScalarSpec,
    SchedulerScalarType, SchedulerScalarValue,
};
use crate::json::{self, JsonValue};
use crate::product_restart::{
    ProductRestartAdmission, RestoredProductRestart, INITIALIZED_PRODUCT_RESTART_SCHEMA,
};
use crate::sha256;
use crate::terrain::{
    compute_terrain_static, HorizontalCrop, TerrainStaticAdmission, TerrainStaticConfig,
};
use crate::terrain_admission_science::{
    compute_surface_halos, compute_terrain_base, pack_cold_start_mutable, MutableCarrier,
    PerturbationPolicy, RecoveryMetric, CP, G, NGXY,
};
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const TERRAIN_OPEN_PROFILE_ID: &str = "cm1-21.1-real-terrain-open-auxiliary/v1";
pub const TERRAIN_OPEN_IDENTITY_SCHEMA: &str = "gpuwm-cm1.terrain-open-auxiliary-identity/v1";
pub const TERRAIN_OPEN_RECEIPT_SCHEMA: &str = "gpuwm-cm1.terrain-open-auxiliary-restart-receipt/v1";
pub const TERRAIN_OPEN_CARRIER_ABI: &str = "gpuwm-cm1.terrain-open-auxiliary-carriers/v1";
pub const TERRAIN_OPEN_IDENTITY_BLOB: &str = "terrain_open_identity";
pub const TERRAIN_OPEN_IDENTITY_MAX_NBYTES: usize = 16 * 1024;
pub const TERRAIN_OPEN_CLAIM_LIMIT: &str = "source-ordered CM1 r21.1 terrain-varying hydrostatic base, family-valid cold-start halos, held/stage wind-pressure-theta carriers, and exact persistent west/east/south/north radiative-boundary histories, durably round-tripped through a source-bound Rust CM1FCP1 envelope; this auxiliary must be paired with its independently hashed CM1IPR1 physical-domain source and does not yet include moisture halos, SGS/TKE, dry-mass reduction state, ArWen component persistence, an evolved coupled large step, uninterrupted/resumed trajectory identity, forecast, tornado, or model-validation evidence";

pub const TERRAIN_OPEN_CARRIER_NAMES: [&str; 50] = [
    "pi0",
    "prs0",
    "th0",
    "thv0",
    "rth0",
    "qv0",
    "rho0",
    "rr0",
    "rf0",
    "rrf0",
    "rho0s",
    "pi0s",
    "prs0s",
    "rth0s",
    "terrain_msl_halo",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "dzdx",
    "dzdy",
    "ppi",
    "pp3d",
    "ppx",
    "piadv",
    "ppterm",
    "ppten",
    "thv",
    "tha",
    "th3d",
    "thten",
    "thten1",
    "ua",
    "u3d",
    "uten",
    "uten1",
    "va",
    "v3d",
    "vten",
    "vten1",
    "wa",
    "w3d",
    "wten",
    "wten1",
    "radbcw",
    "radbce",
    "radbcs",
    "radbcn",
];

const AUTHORITY_TARBALL_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
const REAL_SHAPE_ZYX: [usize; 3] = [75, 300, 300];
const REAL_SPACING_M: [f64; 3] = [100.0, 100.0, 200.0];
const REAL_DT_S: f64 = 0.5;
const SOURCE_NAMES: [&str; 11] = [
    "pi0",
    "th0",
    "qv0",
    "rho0",
    "ua",
    "va",
    "wa",
    "ppi",
    "tha",
    "qa",
    "terrain_msl",
];
const SCHEDULER_NAMES: [&str; 5] = [
    "acoustic_substep",
    "completed_large_steps",
    "large_step_boundary",
    "nrk",
    "radiation_history_initialized",
];
const AUTHORITY_FILES: [(&str, &str, &str); 12] = [
    (
        "src/adv_routines.F",
        "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00",
        "6026-6037,9255-9322",
    ),
    (
        "src/base.F",
        "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef",
        "625-749,2060-2243,3209",
    ),
    (
        "src/bc.F",
        "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
        "12-172,180-1206,1948-2172",
    ),
    (
        "src/cm1.F",
        "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
        "747-775,884-923,1381-1411,1559-1657,2680,3326-3367,4269",
    ),
    (
        "src/cm1libs.F",
        "830cd004f6e9abaa12b242ad7e330c5370ac4b6ed4450745f7badd2ac2d0a954",
        "16-62",
    ),
    (
        "src/init3d.F",
        "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
        "1733-1785,2280-2467",
    ),
    (
        "src/init_terrain.F",
        "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c",
        "175-297",
    ),
    (
        "src/param.F",
        "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
        "7789-7812,8162-8207,8988-9062",
    ),
    (
        "src/restart_read.F",
        "72a041737a4025ce445cd303cd169eb6ff4c001a1140f9fca9e9cdb918884e2a",
        "2341-2474",
    ),
    (
        "src/restart_write.F",
        "d9c21d7071a4548cb9d3ffee1658550ce0e2a4ee7db9c67758ac8c61c07cddba",
        "1851-1970",
    ),
    (
        "src/solve2.F",
        "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734",
        "1041-1049,1080-1173,1261-1269",
    ),
    (
        "src/sound.F",
        "5fe055e22d106e91b5bdbe35ad7c72b2bbbcfb5b01733126703fdc7a5977da54",
        "46-47,253-296",
    ),
];

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[inline]
fn index3(k: usize, j: usize, i: usize, ny: usize, nx: usize) -> usize {
    (k * ny + j) * nx + i
}

#[inline]
fn ordered_f32(bits: u32) -> u32 {
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

#[inline]
fn ulp_distance(left: f32, right: f32) -> u32 {
    ordered_f32(left.to_bits()).abs_diff(ordered_f32(right.to_bits()))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerrainOpenAdmission {
    Generic,
    RealProduct,
}

impl TerrainOpenAdmission {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic-schema",
            Self::RealProduct => "real-product-300x300x75",
        }
    }

    fn terrain(self) -> TerrainStaticAdmission {
        match self {
            Self::Generic => TerrainStaticAdmission::Generic,
            Self::RealProduct => TerrainStaticAdmission::RealProduct,
        }
    }
}

#[derive(Clone, Debug)]
struct SourceIdentity {
    restart_artifact_sha256: String,
    product_payload_sha256: String,
    product_manifest_sha256: String,
    product_receipt_sha256: String,
    source_artifact_sha256: String,
    source_contract_sha256: String,
}

impl SourceIdentity {
    fn from_restart(source: &RestoredProductRestart) -> Self {
        let receipt = source.receipt();
        Self {
            restart_artifact_sha256: receipt.artifact_sha256.clone(),
            product_payload_sha256: receipt.payload_sha256.clone(),
            product_manifest_sha256: receipt.product_manifest_sha256.clone(),
            product_receipt_sha256: receipt.product_receipt_sha256.clone(),
            source_artifact_sha256: receipt.source_artifact_sha256.clone(),
            source_contract_sha256: receipt.source_contract_sha256.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TerrainOpenBuild {
    contract: ForecastContract,
    frame: ForecastFrame,
    source: SourceIdentity,
    source_shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    crop: HorizontalCrop,
    admission: TerrainOpenAdmission,
    pi_recovery: RecoveryMetric,
    theta_recovery: RecoveryMetric,
}

impl TerrainOpenBuild {
    pub fn contract(&self) -> &ForecastContract {
        &self.contract
    }

    pub fn frame(&self) -> &ForecastFrame {
        &self.frame
    }

    pub fn source_restart_artifact_sha256(&self) -> &str {
        &self.source.restart_artifact_sha256
    }

    pub fn source_shape_zyx(&self) -> [usize; 3] {
        self.source_shape_zyx
    }

    pub fn spacing_m(&self) -> [f64; 3] {
        self.spacing_m
    }

    pub fn crop(&self) -> HorizontalCrop {
        self.crop
    }

    pub fn admission(&self) -> TerrainOpenAdmission {
        self.admission
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TerrainOpenReceipt {
    pub forecast: ForecastReceipt,
    pub source_restart_artifact_sha256: String,
    pub carrier_count: usize,
    pub pi_recovery_mismatch_count: u64,
    pub pi_recovery_max_ulp: u32,
    pub pi_recovery_max_abs_f32_bits: u32,
    pub theta_recovery_mismatch_count: u64,
    pub theta_recovery_max_ulp: u32,
    pub theta_recovery_max_abs_f32_bits: u32,
}

impl TerrainOpenReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("artifact_nbytes", number(self.forecast.artifact_nbytes)),
            ("artifact_sha256", string(&self.forecast.artifact_sha256)),
            ("carrier_abi", string(TERRAIN_OPEN_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            ("claim_limit", string(TERRAIN_OPEN_CLAIM_LIMIT)),
            ("contract_sha256", string(&self.forecast.contract_sha256)),
            ("payload_nbytes", number(self.forecast.payload_nbytes)),
            ("payload_sha256", string(&self.forecast.payload_sha256)),
            (
                "pi_recovery",
                recovery_json(
                    self.pi_recovery_mismatch_count,
                    self.pi_recovery_max_ulp,
                    self.pi_recovery_max_abs_f32_bits,
                ),
            ),
            ("schema", string(TERRAIN_OPEN_RECEIPT_SCHEMA)),
            (
                "source_restart_artifact_sha256",
                string(&self.source_restart_artifact_sha256),
            ),
            (
                "theta_recovery",
                recovery_json(
                    self.theta_recovery_mismatch_count,
                    self.theta_recovery_max_ulp,
                    self.theta_recovery_max_abs_f32_bits,
                ),
            ),
        ]);
        String::from_utf8(value.canonical_bytes()).expect("canonical terrain-open receipt is UTF-8")
    }
}

#[derive(Clone, Debug)]
pub struct RestoredTerrainOpen {
    contract: ForecastContract,
    forecast: RestoredForecast,
    source_restart_artifact_sha256: String,
    source_shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    crop: HorizontalCrop,
}

impl RestoredTerrainOpen {
    pub fn contract(&self) -> &ForecastContract {
        &self.contract
    }

    pub fn forecast(&self) -> &RestoredForecast {
        &self.forecast
    }

    pub fn source_restart_artifact_sha256(&self) -> &str {
        &self.source_restart_artifact_sha256
    }

    pub fn source_shape_zyx(&self) -> [usize; 3] {
        self.source_shape_zyx
    }

    pub fn spacing_m(&self) -> [f64; 3] {
        self.spacing_m
    }

    pub fn crop(&self) -> HorizontalCrop {
        self.crop
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TerrainOpenDiagnostics {
    pub carrier_count: usize,
    pub finite_words: u64,
    pub nonfinite_words: u64,
    pub radiative_history_nonzero_words: u64,
    pub terrain_min_f32_bits: u32,
    pub terrain_max_f32_bits: u32,
    pub low_level_pi_min_f32_bits: u32,
    pub low_level_pi_max_f32_bits: u32,
    pub surface_pi_min_f32_bits: u32,
    pub surface_pi_max_f32_bits: u32,
    pub hydrostatic_mismatch_count: u64,
    pub hydrostatic_max_ulp: u32,
    pub hydrostatic_max_abs_f32_bits: u32,
}

impl TerrainOpenDiagnostics {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("carrier_count", number(self.carrier_count)),
            ("finite_words", u64_number(self.finite_words)),
            (
                "hydrostatic_recurrence",
                recovery_json(
                    self.hydrostatic_mismatch_count,
                    self.hydrostatic_max_ulp,
                    self.hydrostatic_max_abs_f32_bits,
                ),
            ),
            (
                "low_level_pi_max_f32_bits",
                string(&format!("{:08x}", self.low_level_pi_max_f32_bits)),
            ),
            (
                "low_level_pi_min_f32_bits",
                string(&format!("{:08x}", self.low_level_pi_min_f32_bits)),
            ),
            ("nonfinite_words", u64_number(self.nonfinite_words)),
            (
                "radiative_history_nonzero_words",
                u64_number(self.radiative_history_nonzero_words),
            ),
            (
                "surface_pi_max_f32_bits",
                string(&format!("{:08x}", self.surface_pi_max_f32_bits)),
            ),
            (
                "surface_pi_min_f32_bits",
                string(&format!("{:08x}", self.surface_pi_min_f32_bits)),
            ),
            (
                "terrain_max_f32_bits",
                string(&format!("{:08x}", self.terrain_max_f32_bits)),
            ),
            (
                "terrain_min_f32_bits",
                string(&format!("{:08x}", self.terrain_min_f32_bits)),
            ),
        ]);
        String::from_utf8(value.canonical_bytes())
            .expect("canonical terrain-open diagnostics are UTF-8")
    }
}

/// Construct the complete cold-start auxiliary from one fully verified
/// `CM1IPR1` source.  The real-product admission is intentionally fixed to the
/// complete 300x300x75 domain at 100x100x200 m and dt=0.5 s.
pub fn build_terrain_open_cold_start(
    source: &RestoredProductRestart,
    crop: Option<HorizontalCrop>,
    spacing_m: [f64; 3],
    dt_s: f64,
    admission: TerrainOpenAdmission,
) -> Result<TerrainOpenBuild> {
    let [nz, source_ny, source_nx] = source_shape(source)?;
    let source_shape_zyx = [nz, source_ny, source_nx];
    validate_source_shape(source_shape_zyx)?;
    let crop = crop.unwrap_or_else(|| HorizontalCrop::full(source_ny, source_nx));
    validate_crop(crop, source_ny, source_nx)?;
    validate_spacing(spacing_m)?;
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(invalid("terrain-open dt must be positive and finite"));
    }
    if admission == TerrainOpenAdmission::RealProduct
        && (source.receipt().admission != ProductRestartAdmission::RealProduct
            || source_shape_zyx != REAL_SHAPE_ZYX
            || crop != HorizontalCrop::full(REAL_SHAPE_ZYX[1], REAL_SHAPE_ZYX[2])
            || spacing_m.map(f64::to_bits) != REAL_SPACING_M.map(f64::to_bits)
            || dt_s.to_bits() != REAL_DT_S.to_bits())
    {
        return Err(invalid(
            "real-product terrain-open admission requires the complete real CM1IPR1 300x300x75 domain at 100x100x200 m and dt=0.5 s",
        ));
    }

    let mut raw = source.read_carriers_f32le_exact(&SOURCE_NAMES)?;
    let pi_profile = take_f32(&mut raw, "pi0")?;
    let theta_profile = take_f32(&mut raw, "th0")?;
    let qv_profile = take_f32(&mut raw, "qv0")?;
    let rho0_profile = take_f32(&mut raw, "rho0")?;
    let ua = take_f32(&mut raw, "ua")?;
    let va = take_f32(&mut raw, "va")?;
    let wa = take_f32(&mut raw, "wa")?;
    let ppi = take_f32(&mut raw, "ppi")?;
    let tha = take_f32(&mut raw, "tha")?;
    let qa = take_f32(&mut raw, "qa")?;
    let terrain = take_f32(&mut raw, "terrain_msl")?;
    if !raw.is_empty() {
        return Err(invalid(
            "terrain-open source extraction left unknown carriers",
        ));
    }

    let terrain_config = TerrainStaticConfig::new(5, 5, [2, 2, 2, 2])?;
    let terrain_frame = compute_terrain_static(
        &terrain,
        &rho0_profile,
        source_shape_zyx,
        spacing_m,
        crop,
        terrain_config,
        admission.terrain(),
    )?;
    let base = compute_terrain_base(
        &terrain,
        source_shape_zyx,
        spacing_m,
        crop,
        &pi_profile,
        &theta_profile,
        &qv_profile,
        &terrain_frame,
    )?;
    drop(terrain_frame);
    let surface = compute_surface_halos(&terrain, [source_ny, source_nx], crop, spacing_m, nz)?;
    let mutable = pack_cold_start_mutable(
        source_shape_zyx,
        crop,
        &pi_profile,
        &theta_profile,
        &ua,
        &va,
        &wa,
        &ppi,
        None,
        &tha,
        &qa,
        &base,
        &surface,
        PerturbationPolicy::RepartitionTotal,
    )?;

    let source_identity = SourceIdentity::from_restart(source);
    let identity = identity_json(
        &source_identity,
        source_shape_zyx,
        spacing_m,
        crop,
        dt_s,
        admission,
    );
    let identity_bytes = identity.canonical_bytes();
    if identity_bytes.len() > TERRAIN_OPEN_IDENTITY_MAX_NBYTES {
        return Err(invalid("terrain-open identity exceeds its fixed bound"));
    }
    let config_bytes =
        config_json(source_shape_zyx, spacing_m, crop, dt_s, admission).canonical_bytes();
    let contract = terrain_open_contract(
        crop.ny,
        crop.nx,
        nz,
        sha256::digest_hex(&config_bytes),
        sha256::digest_hex(&identity_bytes),
    )?;

    let mut arrays = Vec::with_capacity(TERRAIN_OPEN_CARRIER_NAMES.len());
    arrays.push(ForecastArray::new("pi0", base.pi0));
    arrays.push(ForecastArray::new("prs0", base.prs0));
    arrays.push(ForecastArray::new("th0", base.th0));
    arrays.push(ForecastArray::new("thv0", base.thv0));
    arrays.push(ForecastArray::new("rth0", base.rth0));
    arrays.push(ForecastArray::new("qv0", base.qv0));
    arrays.push(ForecastArray::new("rho0", base.rho0));
    arrays.push(ForecastArray::new("rr0", base.rr0));
    arrays.push(ForecastArray::new("rf0", base.rf0));
    arrays.push(ForecastArray::new("rrf0", base.rrf0));
    arrays.push(ForecastArray::new("rho0s", base.rho0s));
    arrays.push(ForecastArray::new("pi0s", base.pi0s));
    arrays.push(ForecastArray::new("prs0s", base.prs0s));
    arrays.push(ForecastArray::new("rth0s", base.rth0s));
    arrays.push(ForecastArray::new(
        "terrain_msl_halo",
        surface.terrain_msl_halo,
    ));
    arrays.push(ForecastArray::new("gz", surface.gz_halo));
    arrays.push(ForecastArray::new("rgz", surface.rgz_halo));
    arrays.push(ForecastArray::new("gzu", surface.gzu_halo));
    arrays.push(ForecastArray::new("rgzu", surface.rgzu_halo));
    arrays.push(ForecastArray::new("gzv", surface.gzv_halo));
    arrays.push(ForecastArray::new("rgzv", surface.rgzv_halo));
    arrays.push(ForecastArray::new("dzdx", surface.dzdx_halo));
    arrays.push(ForecastArray::new("dzdy", surface.dzdy_halo));
    for MutableCarrier {
        name,
        role,
        shape,
        live_region,
        values,
    } in mutable.carriers
    {
        let contracted = contract
            .arrays()
            .iter()
            .find(|spec| spec.name() == name)
            .ok_or_else(|| invalid(format!("terrain-open mutable {name} is not contracted")))?;
        if contracted.shape() != shape || role.is_empty() || live_region.is_empty() {
            return Err(invalid(format!(
                "terrain-open mutable {name} metadata differs from its strict contract"
            )));
        }
        arrays.push(ForecastArray::new(name, values));
    }
    validate_array_names(arrays.iter().map(ForecastArray::name))?;
    for array in &arrays {
        if let Some((index, value)) = array
            .values()
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(invalid(format!(
                "terrain-open carrier {} contains a non-finite word at index {index}; bits={:08x}",
                array.name(),
                value.to_bits()
            )));
        }
    }

    let frame = ForecastFrame::new(
        ForecastClock::new(0, 0.0, dt_s)?,
        arrays,
        cold_start_scalars()?,
        vec![ComponentBlob::new(
            TERRAIN_OPEN_IDENTITY_BLOB,
            TERRAIN_OPEN_IDENTITY_SCHEMA,
            identity_bytes,
        )],
    );
    Ok(TerrainOpenBuild {
        contract,
        frame,
        source: source_identity,
        source_shape_zyx,
        spacing_m,
        crop,
        admission,
        pi_recovery: mutable.pi_recovery,
        theta_recovery: mutable.theta_recovery,
    })
}

pub fn write_terrain_open_restart_new(
    build: &TerrainOpenBuild,
    output: impl AsRef<Path>,
) -> Result<TerrainOpenReceipt> {
    let forecast = write_forecast_checkpoint_new(&build.contract, &build.frame, output)?;
    Ok(TerrainOpenReceipt {
        forecast,
        source_restart_artifact_sha256: build.source.restart_artifact_sha256.clone(),
        carrier_count: TERRAIN_OPEN_CARRIER_NAMES.len(),
        pi_recovery_mismatch_count: build.pi_recovery.mismatch_count,
        pi_recovery_max_ulp: build.pi_recovery.max_ulp,
        pi_recovery_max_abs_f32_bits: build.pi_recovery.max_abs_f32_bits,
        theta_recovery_mismatch_count: build.theta_recovery.mismatch_count,
        theta_recovery_max_ulp: build.theta_recovery.max_ulp,
        theta_recovery_max_abs_f32_bits: build.theta_recovery.max_abs_f32_bits,
    })
}

/// Read a terrain/open auxiliary only through independently supplied contract,
/// artifact, and source-restart hashes.  Generic CM1FCP1 validity is followed
/// by the specialization's exact carrier, scheduler, identity, authority and
/// source-binding checks.
pub fn read_terrain_open_restart(
    input: impl AsRef<Path>,
    expected_contract_sha256: &str,
    expected_artifact_sha256: &str,
    expected_source_restart_artifact_sha256: &str,
) -> Result<RestoredTerrainOpen> {
    validate_sha256(
        expected_contract_sha256,
        "terrain-open expected contract SHA-256",
    )?;
    validate_sha256(
        expected_artifact_sha256,
        "terrain-open expected artifact SHA-256",
    )?;
    validate_sha256(
        expected_source_restart_artifact_sha256,
        "terrain-open expected source-restart SHA-256",
    )?;
    let (contract, forecast) = read_forecast_checkpoint_embedded(input, expected_contract_sha256)?;
    if forecast.receipt().artifact_sha256 != expected_artifact_sha256 {
        return Err(invalid(format!(
            "terrain-open artifact SHA-256 is {}; independently expected {expected_artifact_sha256}",
            forecast.receipt().artifact_sha256
        )));
    }
    validate_specialized_contract(&contract)?;
    let blobs = forecast.frame().component_blobs();
    if blobs.len() != 1
        || blobs[0].name() != TERRAIN_OPEN_IDENTITY_BLOB
        || blobs[0].schema() != TERRAIN_OPEN_IDENTITY_SCHEMA
    {
        return Err(invalid("terrain-open identity component set is invalid"));
    }
    let identity = json::parse(blobs[0].bytes())
        .map_err(|error| invalid(format!("terrain-open identity: {error}")))?;
    if identity.canonical_bytes() != blobs[0].bytes() {
        return Err(invalid("terrain-open identity is not canonical JSON"));
    }
    exact_keys(
        &identity,
        &[
            "admission",
            "authority",
            "carrier_abi",
            "claim_limit",
            "configuration",
            "crop",
            "schema",
            "source_product",
            "source_restart_artifact_sha256",
        ],
        "terrain-open identity",
    )?;
    expect_string(
        &identity,
        "schema",
        TERRAIN_OPEN_IDENTITY_SCHEMA,
        "terrain-open identity",
    )?;
    expect_string(
        &identity,
        "carrier_abi",
        TERRAIN_OPEN_CARRIER_ABI,
        "terrain-open identity",
    )?;
    expect_string(
        &identity,
        "claim_limit",
        TERRAIN_OPEN_CLAIM_LIMIT,
        "terrain-open identity",
    )?;
    expect_string(
        &identity,
        "source_restart_artifact_sha256",
        expected_source_restart_artifact_sha256,
        "terrain-open identity",
    )?;
    let authority = identity
        .get("authority")
        .map_err(|error| invalid(format!("terrain-open identity: {error}")))?;
    validate_authority(authority)?;
    let source_product = identity
        .get("source_product")
        .map_err(|error| invalid(format!("terrain-open identity: {error}")))?;
    validate_source_product(source_product)?;
    let admission_text = identity
        .get("admission")
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("terrain-open identity: {error}")))?;
    let admission = match admission_text {
        "generic-schema" => TerrainOpenAdmission::Generic,
        "real-product-300x300x75" => TerrainOpenAdmission::RealProduct,
        other => {
            return Err(invalid(format!(
                "terrain-open identity admission {other:?} is unsupported"
            )))
        }
    };
    let configuration = identity
        .get("configuration")
        .map_err(|error| invalid(format!("terrain-open identity: {error}")))?;
    exact_keys(
        configuration,
        &[
            "admission",
            "dt_f64_bits",
            "hadvordrs",
            "hadvordrv",
            "lateral_bc_west_east_south_north",
            "ngxy",
            "psolver",
            "source_shape_zyx",
            "spacing_f64_bits_xyz",
            "target_shape_zyx",
        ],
        "terrain-open configuration",
    )?;
    let source_shape_zyx = usize_array(configuration, "source_shape_zyx", 3)?;
    let target_shape_zyx = usize_array(configuration, "target_shape_zyx", 3)?;
    let source_shape_zyx = [
        source_shape_zyx[0],
        source_shape_zyx[1],
        source_shape_zyx[2],
    ];
    validate_source_shape(source_shape_zyx)?;
    let target_shape_zyx = [
        target_shape_zyx[0],
        target_shape_zyx[1],
        target_shape_zyx[2],
    ];
    let spacing_bits = string_array(configuration, "spacing_f64_bits_xyz", 3)?;
    let spacing_m = [
        f64::from_bits(parse_hex_u64(&spacing_bits[0], "terrain-open dx bits")?),
        f64::from_bits(parse_hex_u64(&spacing_bits[1], "terrain-open dy bits")?),
        f64::from_bits(parse_hex_u64(&spacing_bits[2], "terrain-open dz bits")?),
    ];
    validate_spacing(spacing_m)?;
    let dt_bits = configuration
        .get("dt_f64_bits")
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("terrain-open configuration: {error}")))?;
    let dt_s = f64::from_bits(parse_hex_u64(dt_bits, "terrain-open dt bits")?);
    let crop_value = identity
        .get("crop")
        .map_err(|error| invalid(format!("terrain-open identity: {error}")))?;
    exact_keys(crop_value, &["count_yx", "origin_yx"], "terrain-open crop")?;
    let count = usize_array(crop_value, "count_yx", 2)?;
    let origin = usize_array(crop_value, "origin_yx", 2)?;
    let crop = HorizontalCrop {
        j_start: origin[0],
        i_start: origin[1],
        ny: count[0],
        nx: count[1],
    };
    validate_crop(crop, source_shape_zyx[1], source_shape_zyx[2])?;
    if target_shape_zyx != [source_shape_zyx[0], crop.ny, crop.nx] {
        return Err(invalid(
            "terrain-open target shape and identity crop are inconsistent",
        ));
    }
    let expected_configuration = config_json(source_shape_zyx, spacing_m, crop, dt_s, admission);
    if expected_configuration.canonical_bytes() != configuration.canonical_bytes() {
        return Err(invalid(
            "terrain-open configuration differs from the supported exact specialization",
        ));
    }
    if admission == TerrainOpenAdmission::RealProduct
        && (source_shape_zyx != REAL_SHAPE_ZYX
            || crop != HorizontalCrop::full(REAL_SHAPE_ZYX[1], REAL_SHAPE_ZYX[2])
            || spacing_m.map(f64::to_bits) != REAL_SPACING_M.map(f64::to_bits)
            || dt_s.to_bits() != REAL_DT_S.to_bits())
    {
        return Err(invalid("terrain-open real-product identity drifted"));
    }
    let expected_contract = terrain_open_contract(
        crop.ny,
        crop.nx,
        source_shape_zyx[0],
        sha256::digest_hex(&expected_configuration.canonical_bytes()),
        sha256::digest_hex(blobs[0].bytes()),
    )?;
    if expected_contract.canonical_json() != contract.canonical_json() {
        return Err(invalid(
            "terrain-open embedded contract differs from its exact identity-derived specialization",
        ));
    }
    validate_frame_semantics(forecast.frame(), dt_s)?;
    let source_restart_artifact_sha256 = expected_source_restart_artifact_sha256.to_string();
    Ok(RestoredTerrainOpen {
        contract,
        forecast,
        source_restart_artifact_sha256,
        source_shape_zyx,
        spacing_m,
        crop,
    })
}

/// Re-scan every restored carrier and evaluate the source hydrostatic
/// recurrence on all physical columns.  This is diagnostic evidence only; it
/// does not mutate or advance the restart.
pub fn diagnose_terrain_open(restored: &RestoredTerrainOpen) -> Result<TerrainOpenDiagnostics> {
    let arrays = restored
        .forecast
        .frame()
        .arrays()
        .iter()
        .map(|array| (array.name(), array.values()))
        .collect::<BTreeMap<_, _>>();
    validate_array_names(arrays.keys().copied())?;
    let mut finite_words = 0u64;
    let mut nonfinite_words = 0u64;
    for values in arrays.values() {
        for value in values.iter().copied() {
            if value.is_finite() {
                finite_words = finite_words
                    .checked_add(1)
                    .ok_or_else(|| invalid("terrain-open finite-word count overflows"))?;
            } else {
                nonfinite_words = nonfinite_words
                    .checked_add(1)
                    .ok_or_else(|| invalid("terrain-open nonfinite-word count overflows"))?;
            }
        }
    }
    let radiative_history_nonzero_words = ["radbcw", "radbce", "radbcs", "radbcn"]
        .into_iter()
        .map(|name| {
            arrays
                .get(name)
                .expect("validated terrain-open carrier exists")
                .iter()
                .filter(|value| value.to_bits() != 0)
                .count() as u64
        })
        .sum();

    let [nz, _, _] = restored.source_shape_zyx;
    let ny = restored.crop.ny;
    let nx = restored.crop.nx;
    let halo_ny = ny + 2 * NGXY;
    let halo_nx = nx + 2 * NGXY;
    let pi0 = arrays.get("pi0").expect("validated pi0 exists");
    let thv0 = arrays.get("thv0").expect("validated thv0 exists");
    let terrain = arrays
        .get("terrain_msl_halo")
        .expect("validated terrain halo exists");
    let pi0s = arrays.get("pi0s").expect("validated pi0s exists");
    let mut terrain_min = f32::INFINITY;
    let mut terrain_max = f32::NEG_INFINITY;
    let mut low_pi_min = f32::INFINITY;
    let mut low_pi_max = f32::NEG_INFINITY;
    let mut surface_pi_min = f32::INFINITY;
    let mut surface_pi_max = f32::NEG_INFINITY;
    let mut hydrostatic_mismatch_count = 0u64;
    let mut hydrostatic_max_ulp = 0u32;
    let mut hydrostatic_max_abs = 0.0_f32;
    let model_top = restored.spacing_m[2] as f32 * nz as f32;
    for j in 0..ny {
        let jh = j + NGXY;
        for i in 0..nx {
            let ih = i + NGXY;
            let surface_index = jh * halo_nx + ih;
            let surface = terrain[surface_index];
            terrain_min = terrain_min.min(surface);
            terrain_max = terrain_max.max(surface);
            surface_pi_min = surface_pi_min.min(pi0s[surface_index]);
            surface_pi_max = surface_pi_max.max(pi0s[surface_index]);
            let first = index3(1, jh, ih, halo_ny, halo_nx);
            low_pi_min = low_pi_min.min(pi0[first]);
            low_pi_max = low_pi_max.max(pi0[first]);
            let layer_depth = restored.spacing_m[2] as f32 * (model_top - surface) / model_top;
            for k in 2..=nz {
                let below = index3(k - 1, jh, ih, halo_ny, halo_nx);
                let current = index3(k, jh, ih, halo_ny, halo_nx);
                let predicted =
                    pi0[below] - G * layer_depth / (CP * 0.5_f32 * (thv0[current] + thv0[below]));
                if predicted.to_bits() != pi0[current].to_bits() {
                    hydrostatic_mismatch_count += 1;
                }
                hydrostatic_max_ulp =
                    hydrostatic_max_ulp.max(ulp_distance(predicted, pi0[current]));
                hydrostatic_max_abs = hydrostatic_max_abs.max((predicted - pi0[current]).abs());
            }
        }
    }
    Ok(TerrainOpenDiagnostics {
        carrier_count: arrays.len(),
        finite_words,
        nonfinite_words,
        radiative_history_nonzero_words,
        terrain_min_f32_bits: terrain_min.to_bits(),
        terrain_max_f32_bits: terrain_max.to_bits(),
        low_level_pi_min_f32_bits: low_pi_min.to_bits(),
        low_level_pi_max_f32_bits: low_pi_max.to_bits(),
        surface_pi_min_f32_bits: surface_pi_min.to_bits(),
        surface_pi_max_f32_bits: surface_pi_max.to_bits(),
        hydrostatic_mismatch_count,
        hydrostatic_max_ulp,
        hydrostatic_max_abs_f32_bits: hydrostatic_max_abs.to_bits(),
    })
}

fn source_shape(source: &RestoredProductRestart) -> Result<[usize; 3]> {
    let ppi = source
        .carrier("ppi")
        .ok_or_else(|| invalid("terrain-open source lacks ppi"))?;
    if ppi.shape.len() != 3 {
        return Err(invalid("terrain-open source ppi is not rank three"));
    }
    let shape = [ppi.shape[0], ppi.shape[1], ppi.shape[2]];
    let [nz, ny, nx] = shape;
    let expected = [
        ("pi0", vec![nz]),
        ("th0", vec![nz]),
        ("qv0", vec![nz]),
        ("rho0", vec![nz]),
        ("ua", vec![nz, ny, nx + 1]),
        ("va", vec![nz, ny + 1, nx]),
        ("wa", vec![nz + 1, ny, nx]),
        ("ppi", vec![nz, ny, nx]),
        ("tha", vec![nz, ny, nx]),
        ("qa", vec![6, nz, ny, nx]),
        ("terrain_msl", vec![ny, nx]),
    ];
    for (name, required) in expected {
        let carrier = source
            .carrier(name)
            .ok_or_else(|| invalid(format!("terrain-open source lacks {name}")))?;
        if carrier.shape != required {
            return Err(invalid(format!(
                "terrain-open source {name} shape {:?}; expected {required:?}",
                carrier.shape
            )));
        }
    }
    Ok(shape)
}

fn validate_crop(crop: HorizontalCrop, ny: usize, nx: usize) -> Result<()> {
    if crop.ny == 0
        || crop.nx == 0
        || crop.j_start.checked_add(crop.ny).is_none()
        || crop.i_start.checked_add(crop.nx).is_none()
        || crop.j_start + crop.ny > ny
        || crop.i_start + crop.nx > nx
    {
        return Err(invalid(
            "terrain-open crop is empty, overflowing, or out of range",
        ));
    }
    if crop != HorizontalCrop::full(ny, nx) {
        return Err(invalid(
            "terrain-open admission currently requires the full horizontal source domain so open-boundary terrain halos cannot sample beyond an interior crop",
        ));
    }
    Ok(())
}

fn validate_source_shape([nz, ny, nx]: [usize; 3]) -> Result<()> {
    if nz < 4 || ny == 0 || nx == 0 {
        return Err(invalid(
            "terrain-open source shape requires nz >= 4 and nonempty horizontal dimensions",
        ));
    }
    for (axis, extent, padding) in [
        ("z", nz, 3usize),
        ("y", ny, 2 * NGXY + 1),
        ("x", nx, 2 * NGXY + 1),
    ] {
        let allocated = extent
            .checked_add(padding)
            .ok_or_else(|| invalid(format!("terrain-open {axis} allocation extent overflows")))?;
        if allocated > i32::MAX as usize {
            return Err(invalid(format!(
                "terrain-open {axis} allocation extent exceeds the signed native-bound limit"
            )));
        }
    }
    Ok(())
}

fn validate_spacing(spacing_m: [f64; 3]) -> Result<()> {
    for (axis, value) in ["dx", "dy", "dz"].into_iter().zip(spacing_m) {
        if !value.is_finite() || value <= 0.0 || (value as f32) as f64 != value {
            return Err(invalid(format!(
                "terrain-open {axis} must be positive finite and exactly representable in CM1 default REAL"
            )));
        }
    }
    Ok(())
}

fn take_f32(raw: &mut BTreeMap<String, Vec<u8>>, name: &str) -> Result<Vec<f32>> {
    let bytes = raw
        .remove(name)
        .ok_or_else(|| invalid(format!("terrain-open extraction lacks {name}")))?;
    if bytes.len() % 4 != 0 {
        return Err(invalid(format!(
            "terrain-open source {name} is not f32 aligned"
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().expect("four bytes"))))
        .collect())
}

fn terrain_open_contract(
    ny: usize,
    nx: usize,
    nz: usize,
    config_sha256: String,
    execution_identity_sha256: String,
) -> Result<ForecastContract> {
    ForecastContract::new(
        TERRAIN_OPEN_PROFILE_ID,
        config_sha256,
        execution_identity_sha256,
        terrain_open_carrier_specs(nz, ny, nx)?,
        vec![
            SchedulerScalarSpec::new("acoustic_substep", SchedulerScalarType::I64)?,
            SchedulerScalarSpec::new("completed_large_steps", SchedulerScalarType::U64)?,
            SchedulerScalarSpec::new("large_step_boundary", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("nrk", SchedulerScalarType::I64)?,
            SchedulerScalarSpec::new("radiation_history_initialized", SchedulerScalarType::Bool)?,
        ],
        vec![ComponentBlobSpec::new(
            TERRAIN_OPEN_IDENTITY_BLOB,
            TERRAIN_OPEN_IDENTITY_SCHEMA,
            TERRAIN_OPEN_IDENTITY_MAX_NBYTES,
        )?],
    )
}

pub(crate) fn terrain_open_carrier_specs(
    nz: usize,
    ny: usize,
    nx: usize,
) -> Result<Vec<ForecastArraySpec>> {
    let mass_shape = vec![nz + 2, ny + 2 * NGXY, nx + 2 * NGXY];
    let pressure_shape = vec![nz + 2, ny + 2, nx + 2];
    let surface_shape = vec![ny + 2 * NGXY, nx + 2 * NGXY];
    let u_shape = vec![nz + 2, ny + 2 * NGXY, nx + 1 + 2 * NGXY];
    let v_shape = vec![nz + 2, ny + 1 + 2 * NGXY, nx + 2 * NGXY];
    let w_shape = vec![nz + 3, ny + 2 * NGXY, nx + 2 * NGXY];
    let we_shape = vec![nz + 2, ny + 2 * NGXY];
    let sn_shape = vec![nz + 2, nx + 2 * NGXY];
    let mass_bounds = bounds3(
        0,
        nz + 1,
        1 - NGXY as i32,
        ny + NGXY,
        1 - NGXY as i32,
        nx + NGXY,
    )?;
    let pressure_bounds = bounds3(0, nz + 1, 0, ny + 1, 0, nx + 1)?;
    let surface_bounds = bounds2(1 - NGXY as i32, ny + NGXY, 1 - NGXY as i32, nx + NGXY)?;
    let u_bounds = bounds3(
        0,
        nz + 1,
        1 - NGXY as i32,
        ny + NGXY,
        1 - NGXY as i32,
        nx + 1 + NGXY,
    )?;
    let v_bounds = bounds3(
        0,
        nz + 1,
        1 - NGXY as i32,
        ny + 1 + NGXY,
        1 - NGXY as i32,
        nx + NGXY,
    )?;
    let w_bounds = bounds3(
        0,
        nz + 2,
        1 - NGXY as i32,
        ny + NGXY,
        1 - NGXY as i32,
        nx + NGXY,
    )?;
    let we_bounds = bounds2(0, nz + 1, 1 - NGXY as i32, ny + NGXY)?;
    let sn_bounds = bounds2(0, nz + 1, 1 - NGXY as i32, nx + NGXY)?;
    let mut specs = Vec::with_capacity(TERRAIN_OPEN_CARRIER_NAMES.len());
    for name in [
        "pi0", "prs0", "th0", "thv0", "rth0", "qv0", "rho0", "rr0", "rf0", "rrf0",
    ] {
        specs.push(spec(
            name,
            base_meaning(name),
            base_units(name),
            "CM1 scalar/base allocation",
            mass_bounds.clone(),
            mass_shape.clone(),
        )?);
    }
    for name in [
        "rho0s",
        "pi0s",
        "prs0s",
        "rth0s",
        "terrain_msl_halo",
        "gz",
        "rgz",
        "gzu",
        "rgzu",
        "gzv",
        "rgzv",
        "dzdx",
        "dzdy",
    ] {
        specs.push(spec(
            name,
            base_meaning(name),
            base_units(name),
            "CM1 surface allocation",
            surface_bounds.clone(),
            surface_shape.clone(),
        )?);
    }
    for name in ["ppi", "pp3d", "ppx", "piadv", "ppterm", "ppten", "thv"] {
        specs.push(spec(
            name,
            mutable_meaning(name),
            mutable_units(name),
            "pressure live halo",
            pressure_bounds.clone(),
            pressure_shape.clone(),
        )?);
    }
    for name in ["tha", "th3d", "thten", "thten1"] {
        specs.push(spec(
            name,
            mutable_meaning(name),
            mutable_units(name),
            "scalar three-halo",
            mass_bounds.clone(),
            mass_shape.clone(),
        )?);
    }
    for name in ["ua", "u3d", "uten", "uten1"] {
        specs.push(spec(
            name,
            mutable_meaning(name),
            mutable_units(name),
            "x face three-halo",
            u_bounds.clone(),
            u_shape.clone(),
        )?);
    }
    for name in ["va", "v3d", "vten", "vten1"] {
        specs.push(spec(
            name,
            mutable_meaning(name),
            mutable_units(name),
            "y face three-halo",
            v_bounds.clone(),
            v_shape.clone(),
        )?);
    }
    for name in ["wa", "w3d", "wten", "wten1"] {
        specs.push(spec(
            name,
            mutable_meaning(name),
            mutable_units(name),
            "z face three-halo",
            w_bounds.clone(),
            w_shape.clone(),
        )?);
    }
    for name in ["radbcw", "radbce"] {
        specs.push(spec(
            name,
            mutable_meaning(name),
            "m s-2",
            "west/east history plane",
            we_bounds.clone(),
            we_shape.clone(),
        )?);
    }
    for name in ["radbcs", "radbcn"] {
        specs.push(spec(
            name,
            mutable_meaning(name),
            "m s-2",
            "south/north history plane",
            sn_bounds.clone(),
            sn_shape.clone(),
        )?);
    }
    validate_array_names(specs.iter().map(ForecastArraySpec::name))?;
    Ok(specs)
}

/// Exact source-order carrier shapes for streamed auxiliary artifacts.
///
/// These artifacts are streamed rather than published as one forecast
/// envelope: their inventory is larger than the two-GiB payload ceiling this
/// crate carried before [`crate::forecast::MAX_FORECAST_PAYLOAD_BYTES`] became
/// the named host-memory bound it is now.  The streaming decision is not
/// revisited here; this function describes layout only and performs no
/// carrier allocation.
pub(crate) fn terrain_open_carrier_shapes(
    nz: usize,
    ny: usize,
    nx: usize,
) -> Result<Vec<(&'static str, Vec<usize>)>> {
    let add = |value: usize, extra: usize, label: &str| {
        value
            .checked_add(extra)
            .ok_or_else(|| invalid(format!("terrain/open {label} extent overflows")))
    };
    let halo_y = add(ny, 2 * NGXY, "halo y")?;
    let halo_x = add(nx, 2 * NGXY, "halo x")?;
    let mass_shape = vec![add(nz, 2, "mass z")?, halo_y, halo_x];
    let pressure_shape = vec![
        add(nz, 2, "pressure z")?,
        add(ny, 2, "pressure y")?,
        add(nx, 2, "pressure x")?,
    ];
    let surface_shape = vec![halo_y, halo_x];
    let u_shape = vec![add(nz, 2, "u z")?, halo_y, add(nx, 1 + 2 * NGXY, "u x")?];
    let v_shape = vec![add(nz, 2, "v z")?, add(ny, 1 + 2 * NGXY, "v y")?, halo_x];
    let w_shape = vec![add(nz, 3, "w z")?, halo_y, halo_x];
    let we_shape = vec![add(nz, 2, "west/east z")?, halo_y];
    let sn_shape = vec![add(nz, 2, "south/north z")?, halo_x];
    let mut shapes = Vec::with_capacity(TERRAIN_OPEN_CARRIER_NAMES.len());
    for name in [
        "pi0", "prs0", "th0", "thv0", "rth0", "qv0", "rho0", "rr0", "rf0", "rrf0",
    ] {
        shapes.push((name, mass_shape.clone()));
    }
    for name in [
        "rho0s",
        "pi0s",
        "prs0s",
        "rth0s",
        "terrain_msl_halo",
        "gz",
        "rgz",
        "gzu",
        "rgzu",
        "gzv",
        "rgzv",
        "dzdx",
        "dzdy",
    ] {
        shapes.push((name, surface_shape.clone()));
    }
    for name in ["ppi", "pp3d", "ppx", "piadv", "ppterm", "ppten", "thv"] {
        shapes.push((name, pressure_shape.clone()));
    }
    for name in ["tha", "th3d", "thten", "thten1"] {
        shapes.push((name, mass_shape.clone()));
    }
    for name in ["ua", "u3d", "uten", "uten1"] {
        shapes.push((name, u_shape.clone()));
    }
    for name in ["va", "v3d", "vten", "vten1"] {
        shapes.push((name, v_shape.clone()));
    }
    for name in ["wa", "w3d", "wten", "wten1"] {
        shapes.push((name, w_shape.clone()));
    }
    for name in ["radbcw", "radbce"] {
        shapes.push((name, we_shape.clone()));
    }
    for name in ["radbcs", "radbcn"] {
        shapes.push((name, sn_shape.clone()));
    }
    if shapes.len() != TERRAIN_OPEN_CARRIER_NAMES.len()
        || shapes
            .iter()
            .map(|(name, _)| *name)
            .ne(TERRAIN_OPEN_CARRIER_NAMES)
    {
        return Err(invalid("terrain/open streamed carrier order drifted"));
    }
    Ok(shapes)
}

fn spec(
    name: &str,
    meaning: &str,
    units: &str,
    stagger: &str,
    bounds: Vec<i32>,
    shape: Vec<usize>,
) -> Result<ForecastArraySpec> {
    ForecastArraySpec::new(name, meaning, units, stagger, bounds, shape)
}

fn base_meaning(name: &str) -> &'static str {
    match name {
        "pi0" => "terrain-varying hydrostatic base Exner pressure",
        "prs0" => "terrain-varying hydrostatic base pressure",
        "th0" => "terrain-varying base potential temperature",
        "thv0" => "source-rebuilt terrain base virtual potential temperature",
        "rth0" => "reciprocal terrain base potential temperature",
        "qv0" => "terrain-varying base water vapor mixing ratio",
        "rho0" => "terrain-varying hydrostatic base dry density",
        "rr0" => "reciprocal terrain base dry density",
        "rf0" => "source-interpolated terrain base face density",
        "rrf0" => "reciprocal terrain base face density",
        "rho0s" => "surface base density from rf0 native k=1",
        "pi0s" => "hydrostatically recovered terrain surface Exner pressure",
        "prs0s" => "terrain surface base pressure",
        "rth0s" => "reciprocal extrapolated surface potential temperature",
        "terrain_msl_halo" => "real terrain after source open bc2d edge law",
        "gz" => "terrain Jacobian with source-valid surface halos",
        "rgz" => "reciprocal terrain Jacobian with source-valid surface halos",
        "gzu" => "u-face terrain Jacobian with source-valid surface halos",
        "rgzu" => "reciprocal u-face terrain Jacobian with source-valid surface halos",
        "gzv" => "v-face terrain Jacobian with source-valid surface halos",
        "rgzv" => "reciprocal v-face terrain Jacobian with source-valid surface halos",
        "dzdx" => "source flx6 terrain x gradient with open halos",
        "dzdy" => "source flx6 terrain y gradient with open halos",
        _ => "terrain/open auxiliary carrier",
    }
}

fn base_units(name: &str) -> &'static str {
    match name {
        "prs0" | "prs0s" => "Pa",
        "th0" | "thv0" => "K",
        "rth0" | "rth0s" => "K-1",
        "qv0" => "kg kg-1",
        "rho0" | "rf0" | "rho0s" => "kg m-3",
        "rr0" | "rrf0" => "m3 kg-1",
        "terrain_msl_halo" => "m",
        "dzdx" | "dzdy" | "pi0" | "pi0s" | "gz" | "rgz" | "gzu" | "rgzu" | "gzv" | "rgzv" => "1",
        _ => "1",
    }
}

fn mutable_meaning(name: &str) -> &'static str {
    match name {
        "ppi" => "base-repartitioned Exner A carrier",
        "pp3d" => "base-repartitioned current Exner carrier",
        "ppx" => "base-repartitioned held Exner pressure helper",
        "piadv" => "terrain total Exner advection helper",
        "ppterm" => "eqtset2 pressure equation term",
        "ppten" => "stage pressure tendency",
        "thv" => "moist theta-rho helper",
        "tha" => "base-repartitioned theta A carrier",
        "th3d" => "base-repartitioned current theta carrier",
        "thten" => "stage theta tendency",
        "thten1" => "held pre-RK theta tendency",
        "ua" => "zonal velocity A carrier",
        "u3d" => "current zonal velocity",
        "uten" => "stage zonal tendency",
        "uten1" => "held pre-RK zonal tendency",
        "va" => "meridional velocity A carrier",
        "v3d" => "current meridional velocity",
        "vten" => "stage meridional tendency",
        "vten1" => "held pre-RK meridional tendency",
        "wa" => "vertical velocity A carrier with terrain lower face",
        "w3d" => "current vertical velocity with terrain lower face",
        "wten" => "stage vertical tendency",
        "wten1" => "held pre-RK vertical tendency",
        "radbcw" => "persistent west radiative-boundary history",
        "radbce" => "persistent east radiative-boundary history",
        "radbcs" => "persistent south radiative-boundary history",
        "radbcn" => "persistent north radiative-boundary history",
        _ => "terrain/open mutable auxiliary carrier",
    }
}

fn mutable_units(name: &str) -> &'static str {
    match name {
        "tha" | "th3d" | "thv" => "K",
        "thten" | "thten1" => "K s-1",
        "ua" | "u3d" | "va" | "v3d" | "wa" | "w3d" => "m s-1",
        "uten" | "uten1" | "vten" | "vten1" | "wten" | "wten1" => "m s-2",
        "ppten" => "s-1",
        _ => "1",
    }
}

fn bounds2(a0: i32, a1: usize, b0: i32, b1: usize) -> Result<Vec<i32>> {
    Ok(vec![a0, i32_extent(a1)?, b0, i32_extent(b1)?])
}

fn bounds3(a0: i32, a1: usize, b0: i32, b1: usize, c0: i32, c1: usize) -> Result<Vec<i32>> {
    Ok(vec![
        a0,
        i32_extent(a1)?,
        b0,
        i32_extent(b1)?,
        c0,
        i32_extent(c1)?,
    ])
}

fn i32_extent(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| invalid("terrain-open bound does not fit i32"))
}

fn cold_start_scalars() -> Result<Vec<SchedulerScalar>> {
    Ok(vec![
        SchedulerScalar::new("acoustic_substep", SchedulerScalarValue::I64(0))?,
        SchedulerScalar::new("completed_large_steps", SchedulerScalarValue::U64(0))?,
        SchedulerScalar::new("large_step_boundary", SchedulerScalarValue::Bool(true))?,
        SchedulerScalar::new("nrk", SchedulerScalarValue::I64(0))?,
        SchedulerScalar::new(
            "radiation_history_initialized",
            SchedulerScalarValue::Bool(true),
        )?,
    ])
}

fn identity_json(
    source: &SourceIdentity,
    source_shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    crop: HorizontalCrop,
    dt_s: f64,
    admission: TerrainOpenAdmission,
) -> JsonValue {
    object(vec![
        ("admission", string(admission.as_str())),
        (
            "authority",
            object(vec![
                (
                    "source_files",
                    JsonValue::Array(
                        AUTHORITY_FILES
                            .iter()
                            .map(|(path, hash, anchors)| {
                                object(vec![
                                    ("anchors", string(anchors)),
                                    ("path", string(path)),
                                    ("sha256", string(hash)),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
            ]),
        ),
        ("carrier_abi", string(TERRAIN_OPEN_CARRIER_ABI)),
        ("claim_limit", string(TERRAIN_OPEN_CLAIM_LIMIT)),
        (
            "configuration",
            config_json(source_shape_zyx, spacing_m, crop, dt_s, admission),
        ),
        (
            "crop",
            object(vec![
                (
                    "count_yx",
                    JsonValue::Array(vec![number(crop.ny), number(crop.nx)]),
                ),
                (
                    "origin_yx",
                    JsonValue::Array(vec![number(crop.j_start), number(crop.i_start)]),
                ),
            ]),
        ),
        ("schema", string(TERRAIN_OPEN_IDENTITY_SCHEMA)),
        (
            "source_product",
            object(vec![
                ("manifest_sha256", string(&source.product_manifest_sha256)),
                ("payload_sha256", string(&source.product_payload_sha256)),
                ("receipt_sha256", string(&source.product_receipt_sha256)),
                ("schema", string(INITIALIZED_PRODUCT_RESTART_SCHEMA)),
                (
                    "source_artifact_sha256",
                    string(&source.source_artifact_sha256),
                ),
                (
                    "source_contract_sha256",
                    string(&source.source_contract_sha256),
                ),
            ]),
        ),
        (
            "source_restart_artifact_sha256",
            string(&source.restart_artifact_sha256),
        ),
    ])
}

fn config_json(
    source_shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    crop: HorizontalCrop,
    dt_s: f64,
    admission: TerrainOpenAdmission,
) -> JsonValue {
    object(vec![
        ("admission", string(admission.as_str())),
        ("dt_f64_bits", string(&format!("{:016x}", dt_s.to_bits()))),
        ("hadvordrs", number(5)),
        ("hadvordrv", number(5)),
        (
            "lateral_bc_west_east_south_north",
            JsonValue::Array(vec![number(2), number(2), number(2), number(2)]),
        ),
        ("ngxy", number(NGXY)),
        ("psolver", number(3)),
        (
            "source_shape_zyx",
            JsonValue::Array(source_shape_zyx.into_iter().map(number).collect()),
        ),
        (
            "spacing_f64_bits_xyz",
            JsonValue::Array(
                spacing_m
                    .into_iter()
                    .map(|value| string(&format!("{:016x}", value.to_bits())))
                    .collect(),
            ),
        ),
        (
            "target_shape_zyx",
            JsonValue::Array(vec![
                number(source_shape_zyx[0]),
                number(crop.ny),
                number(crop.nx),
            ]),
        ),
    ])
}

fn validate_specialized_contract(contract: &ForecastContract) -> Result<()> {
    if contract.profile_id() != TERRAIN_OPEN_PROFILE_ID {
        return Err(invalid(format!(
            "terrain-open profile is {:?}; expected {TERRAIN_OPEN_PROFILE_ID:?}",
            contract.profile_id()
        )));
    }
    validate_array_names(contract.arrays().iter().map(ForecastArraySpec::name))?;
    let schedulers = contract
        .scheduler_scalars()
        .iter()
        .map(SchedulerScalarSpec::name)
        .collect::<BTreeSet<_>>();
    if schedulers != SCHEDULER_NAMES.into_iter().collect::<BTreeSet<_>>() {
        return Err(invalid("terrain-open scheduler contract is not exact"));
    }
    if contract.component_blobs().len() != 1
        || contract.component_blobs()[0].name() != TERRAIN_OPEN_IDENTITY_BLOB
    {
        return Err(invalid("terrain-open component contract is not exact"));
    }
    Ok(())
}

fn validate_array_names<'a>(names: impl Iterator<Item = &'a str>) -> Result<()> {
    let actual = names.collect::<BTreeSet<_>>();
    let expected = TERRAIN_OPEN_CARRIER_NAMES
        .into_iter()
        .collect::<BTreeSet<_>>();
    if actual != expected || actual.len() != TERRAIN_OPEN_CARRIER_NAMES.len() {
        let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
        let extra = actual.difference(&expected).copied().collect::<Vec<_>>();
        return Err(invalid(format!(
            "terrain-open carrier set differs; missing={missing:?}, extra={extra:?}"
        )));
    }
    Ok(())
}

fn exact_keys(value: &JsonValue, keys: &[&str], label: &str) -> Result<()> {
    let object = value
        .as_object()
        .map_err(|error| invalid(format!("{label}: {error}")))?;
    let actual = object
        .iter()
        .map(|(key, _)| key.as_str())
        .collect::<BTreeSet<_>>();
    let expected = keys.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected || actual.len() != keys.len() {
        return Err(invalid(format!("{label} has an unknown or missing key")));
    }
    Ok(())
}

fn validate_authority(value: &JsonValue) -> Result<()> {
    exact_keys(
        value,
        &["source_files", "tarball_sha256"],
        "terrain-open authority",
    )?;
    expect_string(
        value,
        "tarball_sha256",
        AUTHORITY_TARBALL_SHA256,
        "terrain-open authority",
    )?;
    let source_files = value
        .get("source_files")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("terrain-open authority.source_files: {error}")))?;
    if source_files.len() != AUTHORITY_FILES.len() {
        return Err(invalid(format!(
            "terrain-open authority.source_files contains {} entries; expected {}",
            source_files.len(),
            AUTHORITY_FILES.len()
        )));
    }
    let mut actual = BTreeMap::new();
    for (index, source_file) in source_files.iter().enumerate() {
        let label = format!("terrain-open authority.source_files[{index}]");
        exact_keys(source_file, &["anchors", "path", "sha256"], &label)?;
        let path = source_file
            .get("path")
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("{label}.path: {error}")))?;
        let hash = source_file
            .get("sha256")
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("{label}.sha256: {error}")))?;
        validate_sha256(hash, &format!("{label}.sha256"))?;
        let anchors = source_file
            .get("anchors")
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("{label}.anchors: {error}")))?;
        if actual.insert(path, (hash, anchors)).is_some() {
            return Err(invalid(format!(
                "terrain-open authority contains duplicate source path {path:?}"
            )));
        }
    }
    for (path, expected_hash, expected_anchors) in AUTHORITY_FILES {
        let expected = (expected_hash, expected_anchors);
        if actual.get(path).copied() != Some(expected) {
            return Err(invalid(format!(
                "terrain-open authority source entry {path:?} differs from the pinned CM1 r21.1 authority"
            )));
        }
    }
    Ok(())
}

fn validate_source_product(value: &JsonValue) -> Result<()> {
    exact_keys(
        value,
        &[
            "manifest_sha256",
            "payload_sha256",
            "receipt_sha256",
            "schema",
            "source_artifact_sha256",
            "source_contract_sha256",
        ],
        "terrain-open source_product",
    )?;
    expect_string(
        value,
        "schema",
        INITIALIZED_PRODUCT_RESTART_SCHEMA,
        "terrain-open source_product",
    )?;
    for key in [
        "manifest_sha256",
        "payload_sha256",
        "receipt_sha256",
        "source_artifact_sha256",
        "source_contract_sha256",
    ] {
        let hash = value
            .get(key)
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("terrain-open source_product.{key}: {error}")))?;
        validate_sha256(hash, &format!("terrain-open source_product.{key}"))?;
    }
    Ok(())
}

fn validate_frame_semantics(frame: &ForecastFrame, identity_dt_s: f64) -> Result<()> {
    let clock = frame.clock();
    if clock.dt_s().to_bits() != identity_dt_s.to_bits() {
        return Err(invalid(format!(
            "terrain-open frame dt bits {:016x} differ from identity dt bits {:016x}",
            clock.dt_s().to_bits(),
            identity_dt_s.to_bits()
        )));
    }
    let mut actual_scalars = BTreeMap::new();
    for scalar in frame.scheduler_scalars() {
        if actual_scalars
            .insert(scalar.name(), scalar.value())
            .is_some()
        {
            return Err(invalid(format!(
                "terrain-open frame contains duplicate scheduler scalar {:?}",
                scalar.name()
            )));
        }
    }
    let expected_scalars = [
        ("acoustic_substep", SchedulerScalarValue::I64(0)),
        (
            "completed_large_steps",
            SchedulerScalarValue::U64(clock.step()),
        ),
        ("large_step_boundary", SchedulerScalarValue::Bool(true)),
        ("nrk", SchedulerScalarValue::I64(0)),
        (
            "radiation_history_initialized",
            SchedulerScalarValue::Bool(true),
        ),
    ];
    for (name, expected) in expected_scalars {
        if actual_scalars.get(name).copied() != Some(expected) {
            return Err(invalid(format!(
                "terrain-open frame scheduler scalar {name:?} is not restart-safe at a completed large-step boundary"
            )));
        }
    }
    for array in frame.arrays() {
        if let Some((word, value)) = array
            .values()
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(invalid(format!(
                "terrain-open carrier {:?} contains non-finite word {word}: {value:?}",
                array.name()
            )));
        }
    }
    Ok(())
}

fn expect_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = value
        .get(key)
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("{label}: {error}")))?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!(
            "{label} must be exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn usize_array(value: &JsonValue, key: &str, expected_len: usize) -> Result<Vec<usize>> {
    let values = value
        .get(key)
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("terrain-open {key}: {error}")))?;
    if values.len() != expected_len {
        return Err(invalid(format!(
            "terrain-open {key} contains {} values; expected {expected_len}",
            values.len()
        )));
    }
    values
        .iter()
        .map(|item| {
            let value = item
                .as_u64()
                .map_err(|error| invalid(format!("terrain-open {key}: {error}")))?;
            usize::try_from(value)
                .map_err(|_| invalid(format!("terrain-open {key} value does not fit usize")))
        })
        .collect()
}

fn string_array(value: &JsonValue, key: &str, expected_len: usize) -> Result<Vec<String>> {
    let values = value
        .get(key)
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("terrain-open {key}: {error}")))?;
    if values.len() != expected_len {
        return Err(invalid(format!(
            "terrain-open {key} contains {} values; expected {expected_len}",
            values.len()
        )));
    }
    values
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .map_err(|error| invalid(format!("terrain-open {key}: {error}")))
        })
        .collect()
}

fn parse_hex_u64(value: &str, label: &str) -> Result<u64> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!(
            "{label} is not canonical 16-digit lowercase hex"
        )));
    }
    u64::from_str_radix(value, 16).map_err(|_| invalid(format!("{label} is invalid")))
}

fn recovery_json(mismatch_count: u64, max_ulp: u32, max_abs_f32_bits: u32) -> JsonValue {
    object(vec![
        (
            "max_abs_f32_bits",
            string(&format!("{max_abs_f32_bits:08x}")),
        ),
        ("max_ulp", u64_number(u64::from(max_ulp))),
        ("mismatch_count", u64_number(mismatch_count)),
    ])
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child_remap::{CHILD_NX, CHILD_NY, CHILD_NZ};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn streamed_carrier_shapes_match_the_legacy_forecast_contract_at_50m() {
        let specs = terrain_open_carrier_specs(CHILD_NZ, CHILD_NY, CHILD_NX).unwrap();
        let shapes = terrain_open_carrier_shapes(CHILD_NZ, CHILD_NY, CHILD_NX).unwrap();
        assert_eq!(specs.len(), shapes.len());
        for (spec, (name, shape)) in specs.iter().zip(shapes) {
            assert_eq!(spec.name(), name);
            assert_eq!(spec.shape(), shape);
        }
    }

    fn fake_source() -> SourceIdentity {
        SourceIdentity {
            restart_artifact_sha256: "11".repeat(32),
            product_payload_sha256: "22".repeat(32),
            product_manifest_sha256: "33".repeat(32),
            product_receipt_sha256: "44".repeat(32),
            source_artifact_sha256: "55".repeat(32),
            source_contract_sha256: "66".repeat(32),
        }
    }

    fn fixture() -> (ForecastContract, ForecastFrame, String) {
        let shape = [4, 7, 8];
        let crop = HorizontalCrop::full(shape[1], shape[2]);
        let spacing = [100.0, 100.0, 200.0];
        let source = fake_source();
        let identity = identity_json(
            &source,
            shape,
            spacing,
            crop,
            0.5,
            TerrainOpenAdmission::Generic,
        );
        let identity_bytes = identity.canonical_bytes();
        let config = config_json(shape, spacing, crop, 0.5, TerrainOpenAdmission::Generic);
        let contract = terrain_open_contract(
            shape[1],
            shape[2],
            shape[0],
            sha256::digest_hex(&config.canonical_bytes()),
            sha256::digest_hex(&identity_bytes),
        )
        .unwrap();
        let arrays = contract
            .arrays()
            .iter()
            .map(|spec| {
                let words = spec.shape().iter().product();
                let mut values = vec![0.0; words];
                if spec.name() == "radbcw" {
                    values[words / 2] = 1.25;
                }
                ForecastArray::new(spec.name(), values)
            })
            .collect::<Vec<_>>();
        let frame = ForecastFrame::new(
            ForecastClock::new(3, 1.5, 0.5).unwrap(),
            arrays,
            vec![
                SchedulerScalar::new("acoustic_substep", SchedulerScalarValue::I64(0)).unwrap(),
                SchedulerScalar::new("completed_large_steps", SchedulerScalarValue::U64(3))
                    .unwrap(),
                SchedulerScalar::new("large_step_boundary", SchedulerScalarValue::Bool(true))
                    .unwrap(),
                SchedulerScalar::new("nrk", SchedulerScalarValue::I64(0)).unwrap(),
                SchedulerScalar::new(
                    "radiation_history_initialized",
                    SchedulerScalarValue::Bool(true),
                )
                .unwrap(),
            ],
            vec![ComponentBlob::new(
                TERRAIN_OPEN_IDENTITY_BLOB,
                TERRAIN_OPEN_IDENTITY_SCHEMA,
                identity_bytes,
            )],
        );
        (contract, frame, source.restart_artifact_sha256)
    }

    fn temp_file(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cm1-terrain-open-{label}-{}-{nonce}.cm1fcp",
            std::process::id()
        ))
    }

    #[test]
    fn contract_has_exact_source_family_geometry() {
        let (contract, _, _) = fixture();
        assert_eq!(contract.arrays().len(), TERRAIN_OPEN_CARRIER_NAMES.len());
        validate_specialized_contract(&contract).unwrap();
        assert_eq!(
            contract
                .arrays()
                .iter()
                .find(|spec| spec.name() == "pi0")
                .unwrap()
                .shape(),
            [6, 13, 14]
        );
        assert_eq!(
            contract
                .arrays()
                .iter()
                .find(|spec| spec.name() == "ppi")
                .unwrap()
                .shape(),
            [6, 9, 10]
        );
        assert_eq!(
            contract
                .arrays()
                .iter()
                .find(|spec| spec.name() == "radbcw")
                .unwrap()
                .shape(),
            [6, 13]
        );
    }

    #[test]
    fn nested_identity_provenance_is_exact() {
        let shape = [4, 7, 8];
        let identity = identity_json(
            &fake_source(),
            shape,
            [100.0, 100.0, 200.0],
            HorizontalCrop::full(shape[1], shape[2]),
            0.5,
            TerrainOpenAdmission::Generic,
        );
        validate_authority(identity.get("authority").unwrap()).unwrap();
        validate_source_product(identity.get("source_product").unwrap()).unwrap();

        let missing_authority_file = object(vec![
            (
                "source_files",
                JsonValue::Array(
                    AUTHORITY_FILES[..AUTHORITY_FILES.len() - 1]
                        .iter()
                        .map(|(path, hash, anchors)| {
                            object(vec![
                                ("anchors", string(anchors)),
                                ("path", string(path)),
                                ("sha256", string(hash)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
        ]);
        assert!(validate_authority(&missing_authority_file).is_err());

        let malformed_product = object(vec![
            ("manifest_sha256", string(&"AA".repeat(32))),
            ("payload_sha256", string(&"22".repeat(32))),
            ("receipt_sha256", string(&"33".repeat(32))),
            ("schema", string(INITIALIZED_PRODUCT_RESTART_SCHEMA)),
            ("source_artifact_sha256", string(&"44".repeat(32))),
            ("source_contract_sha256", string(&"55".repeat(32))),
        ]);
        assert!(validate_source_product(&malformed_product).is_err());
        assert!(validate_crop(HorizontalCrop::full(7, 8), 7, 8).is_ok());
        assert!(validate_crop(
            HorizontalCrop {
                j_start: 1,
                i_start: 1,
                ny: 5,
                nx: 6,
            },
            7,
            8,
        )
        .is_err());
        assert!(validate_source_shape([4, 7, 8]).is_ok());
        assert!(validate_source_shape([3, 7, 8]).is_err());
        assert!(validate_source_shape([4, usize::MAX, 8]).is_err());
    }

    #[test]
    fn restored_frame_requires_complete_finite_large_step_state() {
        let (_, frame, _) = fixture();
        validate_frame_semantics(&frame, 0.5).unwrap();
        assert!(validate_frame_semantics(&frame, 0.25).is_err());

        let bad_scalars = frame
            .scheduler_scalars()
            .iter()
            .map(|scalar| {
                let value = if scalar.name() == "completed_large_steps" {
                    SchedulerScalarValue::U64(frame.clock().step() - 1)
                } else {
                    scalar.value()
                };
                SchedulerScalar::new(scalar.name(), value).unwrap()
            })
            .collect();
        let midstep = ForecastFrame::new(
            frame.clock(),
            frame.arrays().to_vec(),
            bad_scalars,
            frame.component_blobs().to_vec(),
        );
        assert!(validate_frame_semantics(&midstep, 0.5).is_err());

        let nonfinite_arrays = frame
            .arrays()
            .iter()
            .map(|array| {
                let mut values = array.values().to_vec();
                if array.name() == "pi0" {
                    values[0] = f32::NAN;
                }
                ForecastArray::new(array.name(), values)
            })
            .collect();
        let nonfinite = ForecastFrame::new(
            frame.clock(),
            nonfinite_arrays,
            frame.scheduler_scalars().to_vec(),
            frame.component_blobs().to_vec(),
        );
        assert!(validate_frame_semantics(&nonfinite, 0.5).is_err());
    }

    #[test]
    fn radiative_history_is_durable_and_source_bound() {
        let (contract, frame, source_sha) = fixture();
        let path = temp_file("roundtrip");
        let receipt = write_forecast_checkpoint_new(&contract, &frame, &path).unwrap();
        let restored = read_terrain_open_restart(
            &path,
            &receipt.contract_sha256,
            &receipt.artifact_sha256,
            &source_sha,
        )
        .unwrap();
        let west = restored
            .forecast()
            .frame()
            .arrays()
            .iter()
            .find(|array| array.name() == "radbcw")
            .unwrap();
        assert_eq!(
            west.values()[west.values().len() / 2].to_bits(),
            1.25_f32.to_bits()
        );
        let error = read_terrain_open_restart(
            &path,
            &receipt.contract_sha256,
            &receipt.artifact_sha256,
            &"77".repeat(32),
        )
        .unwrap_err();
        assert!(error.to_string().contains("source_restart_artifact_sha256"));
        fs::remove_file(path).unwrap();
    }
}
