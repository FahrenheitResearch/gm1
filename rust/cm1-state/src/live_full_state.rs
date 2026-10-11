//! Strict live full-state checkpoint specialization for an evolved forecast.
//!
//! The CUDA/Python layer may transport bytes, but it does not select, reshape,
//! or reinterpret checkpoint state.  Rust owns the fixed v1 inventory, its
//! deterministic arena order, semantic validation, `CM1FCP1` envelope, and
//! exact restore directory.  Capture is admitted only at a completed CM1
//! large-step boundary.

use crate::forecast::{
    read_forecast_checkpoint_embedded, write_forecast_checkpoint_new, ComponentBlob,
    ComponentBlobSpec, ForecastArray, ForecastArraySpec, ForecastClock, ForecastContract,
    ForecastFrame, ForecastReceipt, RestoredForecast, SchedulerScalar, SchedulerScalarSpec,
    SchedulerScalarType, SchedulerScalarValue,
};
use crate::json::{self, JsonValue};
use crate::sha256::{self, Sha256};
#[cfg(test)]
use crate::terrain_admission::TERRAIN_OPEN_CARRIER_NAMES;
use crate::terrain_admission::{terrain_open_carrier_specs, TERRAIN_OPEN_CARRIER_ABI};
use crate::transition_restart::{
    WSM6_TRANSITION_ABI, WSM6_TRANSITION_ARRAY_NAMES, WSM6_TRANSITION_BRIDGE_ABI,
    WSM6_TRANSITION_CARRIER_ABI, WSM6_TRANSITION_COMPONENT_ABI,
    WSM6_TRANSITION_IN_MEMORY_RESTART_SCHEMA,
};
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const LIVE_FULL_STATE_PROFILE_ID: &str = "cm1-21.1-real-live-full-state/v1";
pub const LIVE_FULL_STATE_CARRIER_ABI: &str = "gpuwm-cm1.live-full-state-carriers/v1";
pub const LIVE_FULL_STATE_IDENTITY_SCHEMA: &str = "gpuwm-cm1.live-full-state-identity/v1";
pub const LIVE_FULL_STATE_SCHEDULER_EXTENSION_SCHEMA: &str =
    "gpuwm-cm1.live-full-state-scheduler-extension/v1";
pub const LIVE_FULL_STATE_LAYOUT_SCHEMA: &str = "gpuwm-cm1.live-full-state-layout/v1";
pub const LIVE_FULL_STATE_RESTORE_MANIFEST_SCHEMA: &str =
    "gpuwm-cm1.live-full-state-restore-directory/v1";
pub const LIVE_FULL_STATE_RECEIPT_SCHEMA: &str = "gpuwm-cm1.live-full-state-checkpoint-receipt/v1";
pub const LIVE_FULL_STATE_RESTORE_RECEIPT_SCHEMA: &str =
    "gpuwm-cm1.live-full-state-restore-receipt/v1";
pub const LIVE_FULL_STATE_IDENTITY_BLOB: &str = "live_full_state_identity";
pub const LIVE_FULL_STATE_SCHEDULER_BLOB: &str = "scheduler_extension";
pub const LIVE_FULL_STATE_ARENA_FILE: &str = "carrier-arena.f32le";
pub const LIVE_FULL_STATE_MANIFEST_FILE: &str = "manifest.json";
pub const LIVE_FULL_STATE_RECEIPT_FILE: &str = "receipt.json";
pub const LIVE_FULL_STATE_CAPTURE_PHASE: &str = "completed-large-step-boundary";
pub const LIVE_FULL_STATE_RESTORE_PHASE: &str =
    "before-arming-next-large-step-or-final-rk-transition";
pub const LIVE_FULL_STATE_PUBLICATION_CONTRACT: &str = "Rust validates the complete named binary32 arena, writes one new CM1FCP1 artifact atomically, and never replaces an existing target";
pub const LIVE_FULL_STATE_RESTORE_PUBLICATION_CONTRACT: &str = "Rust writes one exact binary32 arena plus canonical manifest and receipt into a new sibling staging directory, synchronizes them, then renames the directory without replacing an existing target";
pub const LIVE_FULL_STATE_CLAIM_LIMIT: &str = "exact completed-large-step capture and raw-bit restore of the current terrain/open hydrostatic, halo, A/current, held tendency, radiative-history, packed scalar, WSM6 transition-persistence, and closure TKE A-carrier inventories; acoustic, advection, and closure diagnostic scratch are explicitly recomputable, while mid-step tke3d/tketen, dry-mass reduction state, closure state not named by this ABI, live coupled scheduler integration, uninterrupted/resumed trajectory identity, forecast skill, tornado, and model validation remain unclaimed";

pub const LIVE_FULL_STATE_ARRAY_COUNT: usize = 76;
pub const LIVE_FULL_STATE_IDENTITY_MAX_NBYTES: usize = 64 * 1024;
pub const LIVE_FULL_STATE_SCHEDULER_MAX_NBYTES: usize = 64 * 1024;
pub const LIVE_FULL_STATE_REAL_NZ: usize = 75;
pub const LIVE_FULL_STATE_REAL_NY: usize = 300;
pub const LIVE_FULL_STATE_REAL_NX: usize = 300;
pub const LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES: u64 = 2_144_851_136;

const AUTHORITY_TARBALL_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
const REAL_SPACING_M: [f64; 3] = [100.0, 100.0, 200.0];
const REAL_DT_S: f64 = 0.5;
const CUDA_TERRAIN_DEFORMATION_SCHEMA: &str = "gpuwm-cm1.cuda-terrain-deformation/v1";
const CUDA_TERRAIN_DEFORMATION_ABI: &str = "gpuwm-cm1.cuda-terrain-deformation-kernels/v1";
const CUDA_SGS_TKE_SCHEMA: &str = "gpuwm-cm1.cuda-sgs-tke/v1";
const CUDA_SGS_TKE_ABI: &str = "gpuwm-cm1.cuda-sgs-tke-kernels/v1";
const SPECIES: [&str; 6] = ["qv", "qc", "qr", "qi", "qs", "qg"];
const SCHEDULER_NAMES: [&str; 8] = [
    "acoustic_substep",
    "completed_large_steps",
    "completed_wsm6_transitions",
    "large_step_boundary",
    "nrk",
    "radiation_history_initialized",
    "scalar_stage",
    "wsm6_transition_ready",
];
const SCALAR_ARRAY_NAMES: [&str; 4] = ["scalar.qa", "scalar.q3d", "scalar.qten", "scalar.qten1"];
const CLOSURE_ARRAY_NAMES: [&str; 1] = ["closure.tkea"];
const RECOMPUTABLE_FAMILIES: [&str; 5] = [
    "acoustic elliptic/tri-diagonal scratch",
    "mass-flux and scalar-advection flux scratch",
    "closure Nm, deformation, diffusivity and local-source scratch",
    "diagnostic reflectivity",
    "launch geometry and CUDA events",
];
const DUPLICATE_BINDINGS: [(&str, &str, &str); 3] = [
    ("pp3d", "state.pp3d", "padded physical interior"),
    ("th3d", "state.th3d", "three-halo physical interior"),
    ("scalar.q3d", "state.q3d", "exact packed physical array"),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LiveFullStateAdmission {
    Generic,
    RealProduct,
}

impl LiveFullStateAdmission {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic-schema-validation",
            Self::RealProduct => "real-product-300x300x75",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "generic-schema-validation" => Ok(Self::Generic),
            "real-product-300x300x75" => Ok(Self::RealProduct),
            other => Err(invalid(format!(
                "live full-state admission {other:?} is unknown"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiveFullStateConfig {
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
    pub spacing_m: [f64; 3],
    pub dt_s: f64,
    pub admission: LiveFullStateAdmission,
}

impl LiveFullStateConfig {
    pub fn real_product() -> Self {
        Self {
            nz: LIVE_FULL_STATE_REAL_NZ,
            ny: LIVE_FULL_STATE_REAL_NY,
            nx: LIVE_FULL_STATE_REAL_NX,
            spacing_m: REAL_SPACING_M,
            dt_s: REAL_DT_S,
            admission: LiveFullStateAdmission::RealProduct,
        }
    }

    pub fn generic(
        nz: usize,
        ny: usize,
        nx: usize,
        spacing_m: [f64; 3],
        dt_s: f64,
    ) -> Result<Self> {
        let value = Self {
            nz,
            ny,
            nx,
            spacing_m,
            dt_s,
            admission: LiveFullStateAdmission::Generic,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(self) -> Result<()> {
        if self.nz == 0 || self.ny == 0 || self.nx == 0 {
            return Err(invalid("live full-state grid extents must all be positive"));
        }
        for (axis, value) in self.spacing_m.iter().copied().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!(
                    "live full-state spacing axis {axis} must be finite and positive"
                )));
            }
        }
        if !self.dt_s.is_finite() || self.dt_s <= 0.0 {
            return Err(invalid("live full-state dt must be finite and positive"));
        }
        if self.admission == LiveFullStateAdmission::RealProduct
            && (self.nz, self.ny, self.nx, self.spacing_m, self.dt_s)
                != (
                    LIVE_FULL_STATE_REAL_NZ,
                    LIVE_FULL_STATE_REAL_NY,
                    LIVE_FULL_STATE_REAL_NX,
                    REAL_SPACING_M,
                    REAL_DT_S,
                )
        {
            return Err(invalid(
                "real-product live full-state admission requires exact 75x300x300, 100x100x200 m, dt=0.5 s geometry",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveFullStateSourceIdentities {
    pub initialized_product_restart_sha256: String,
    pub initialized_product_manifest_sha256: String,
    pub initialized_product_payload_sha256: String,
    pub terrain_open_restart_sha256: String,
    pub wsm6_transition_restart_sha256: String,
}

impl LiveFullStateSourceIdentities {
    pub fn new(
        initialized_product_restart_sha256: impl Into<String>,
        initialized_product_manifest_sha256: impl Into<String>,
        initialized_product_payload_sha256: impl Into<String>,
        terrain_open_restart_sha256: impl Into<String>,
        wsm6_transition_restart_sha256: impl Into<String>,
    ) -> Result<Self> {
        let value = Self {
            initialized_product_restart_sha256: initialized_product_restart_sha256.into(),
            initialized_product_manifest_sha256: initialized_product_manifest_sha256.into(),
            initialized_product_payload_sha256: initialized_product_payload_sha256.into(),
            terrain_open_restart_sha256: terrain_open_restart_sha256.into(),
            wsm6_transition_restart_sha256: wsm6_transition_restart_sha256.into(),
        };
        for (label, hash) in [
            (
                "initialized-product restart SHA-256",
                &value.initialized_product_restart_sha256,
            ),
            (
                "initialized-product manifest SHA-256",
                &value.initialized_product_manifest_sha256,
            ),
            (
                "initialized-product payload SHA-256",
                &value.initialized_product_payload_sha256,
            ),
            (
                "terrain/open restart SHA-256",
                &value.terrain_open_restart_sha256,
            ),
            (
                "WSM6 transition restart SHA-256",
                &value.wsm6_transition_restart_sha256,
            ),
        ] {
            validate_sha256(hash, label)?;
        }
        Ok(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveFullStateCheckpointReceipt {
    pub forecast: ForecastReceipt,
    pub raw_arena_sha256: String,
    pub raw_arena_nbytes: u64,
    pub completed_wsm6_transitions: u64,
    pub source: LiveFullStateSourceIdentities,
}

impl LiveFullStateCheckpointReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("artifact_nbytes", number(self.forecast.artifact_nbytes)),
            ("artifact_sha256", string(&self.forecast.artifact_sha256)),
            ("carrier_abi", string(LIVE_FULL_STATE_CARRIER_ABI)),
            ("carrier_count", number(self.forecast.arrays)),
            ("capture_phase", string(LIVE_FULL_STATE_CAPTURE_PHASE)),
            ("claim_limit", string(LIVE_FULL_STATE_CLAIM_LIMIT)),
            (
                "completed_wsm6_transitions",
                u64_number(self.completed_wsm6_transitions),
            ),
            ("contract_sha256", string(&self.forecast.contract_sha256)),
            ("payload_sha256", string(&self.forecast.payload_sha256)),
            (
                "publication_contract",
                string(LIVE_FULL_STATE_PUBLICATION_CONTRACT),
            ),
            ("raw_arena_nbytes", u64_number(self.raw_arena_nbytes)),
            ("raw_arena_sha256", string(&self.raw_arena_sha256)),
            ("schema", string(LIVE_FULL_STATE_RECEIPT_SCHEMA)),
            ("source", source_json(&self.source)),
            ("step", u64_number(self.forecast.step)),
        ]);
        canonical_string(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveFullStateRestoreReceipt {
    pub checkpoint_artifact_sha256: String,
    pub checkpoint_contract_sha256: String,
    pub manifest_sha256: String,
    pub manifest_nbytes: usize,
    pub payload_sha256: String,
    pub payload_nbytes: u64,
    pub carrier_count: usize,
    pub step: u64,
}

impl LiveFullStateRestoreReceipt {
    pub fn canonical_json(&self) -> String {
        canonical_string(object(vec![
            ("carrier_abi", string(LIVE_FULL_STATE_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            (
                "checkpoint_artifact_sha256",
                string(&self.checkpoint_artifact_sha256),
            ),
            (
                "checkpoint_contract_sha256",
                string(&self.checkpoint_contract_sha256),
            ),
            ("claim_limit", string(LIVE_FULL_STATE_CLAIM_LIMIT)),
            ("manifest_nbytes", number(self.manifest_nbytes)),
            ("manifest_sha256", string(&self.manifest_sha256)),
            ("payload_nbytes", u64_number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            (
                "publication_contract",
                string(LIVE_FULL_STATE_RESTORE_PUBLICATION_CONTRACT),
            ),
            ("schema", string(LIVE_FULL_STATE_RESTORE_RECEIPT_SCHEMA)),
            ("step", u64_number(self.step)),
        ]))
    }
}

#[derive(Clone, Debug)]
pub struct RestoredLiveFullState {
    contract: ForecastContract,
    forecast: RestoredForecast,
    config: LiveFullStateConfig,
    source: LiveFullStateSourceIdentities,
    raw_arena_sha256: String,
    scheduler_extension: Vec<u8>,
    completed_wsm6_transitions: u64,
}

impl RestoredLiveFullState {
    pub fn contract(&self) -> &ForecastContract {
        &self.contract
    }

    pub fn forecast(&self) -> &RestoredForecast {
        &self.forecast
    }

    pub fn config(&self) -> LiveFullStateConfig {
        self.config
    }

    pub fn source(&self) -> &LiveFullStateSourceIdentities {
        &self.source
    }

    pub fn raw_arena_sha256(&self) -> &str {
        &self.raw_arena_sha256
    }

    pub fn scheduler_extension(&self) -> &[u8] {
        &self.scheduler_extension
    }

    pub fn completed_wsm6_transitions(&self) -> u64 {
        self.completed_wsm6_transitions
    }
}

#[derive(Clone, Debug)]
struct ParsedIdentity {
    config: LiveFullStateConfig,
    source: LiveFullStateSourceIdentities,
    raw_arena_sha256: String,
    scheduler_extension_sha256: String,
}

#[derive(Clone, Debug)]
struct RestoreCarrier {
    name: String,
    shape: Vec<usize>,
    offset_nbytes: u64,
    nbytes: u64,
    sha256: String,
}

impl RestoreCarrier {
    fn json(&self) -> JsonValue {
        object(vec![
            ("dtype", string("<f4")),
            ("itemsize", number(4)),
            ("name", string(&self.name)),
            ("nbytes", u64_number(self.nbytes)),
            ("offset_nbytes", u64_number(self.offset_nbytes)),
            ("order", string("C")),
            (
                "shape",
                JsonValue::Array(self.shape.iter().copied().map(number).collect()),
            ),
            ("sha256", string(&self.sha256)),
        ])
    }
}

/// Return the Rust-owned production arena layout without opening CUDA.
pub fn live_full_state_real_layout_json() -> Result<String> {
    live_full_state_layout_json(LiveFullStateConfig::real_product())
}

pub fn live_full_state_layout_json(config: LiveFullStateConfig) -> Result<String> {
    config.validate()?;
    let contract = live_full_state_contract(config)?;
    let specs = contract.arrays();
    let mut offset = 0u64;
    let mut carriers = Vec::with_capacity(specs.len());
    for spec in specs {
        let nbytes = spec_nbytes(spec)?;
        carriers.push(object(vec![
            ("name", string(spec.name())),
            ("nbytes", u64_number(nbytes)),
            ("offset_nbytes", u64_number(offset)),
            (
                "shape",
                JsonValue::Array(spec.shape().iter().copied().map(number).collect()),
            ),
        ]));
        offset = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("live full-state layout byte count overflows"))?;
    }
    if config.admission == LiveFullStateAdmission::RealProduct
        && offset != LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES
    {
        return Err(invalid(format!(
            "real live full-state arena has {offset} bytes; sealed v1 requires {LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES}"
        )));
    }
    Ok(canonical_string(object(vec![
        ("admission", string(config.admission.as_str())),
        ("carrier_abi", string(LIVE_FULL_STATE_CARRIER_ABI)),
        ("carrier_count", number(specs.len())),
        ("carriers", JsonValue::Array(carriers)),
        ("claim_limit", string(LIVE_FULL_STATE_CLAIM_LIMIT)),
        ("contract_sha256", string(&contract.sha256())),
        ("dtype", string("<f4")),
        ("itemsize", number(4)),
        ("order", string("C")),
        ("payload_nbytes", u64_number(offset)),
        ("profile_id", string(LIVE_FULL_STATE_PROFILE_ID)),
        ("schema", string(LIVE_FULL_STATE_LAYOUT_SCHEMA)),
        (
            "shape_zyx",
            JsonValue::Array([config.nz, config.ny, config.nx].map(number).to_vec()),
        ),
    ])))
}

/// Build a strict checkpoint from one raw arena in the exact `layout` order.
#[allow(clippy::too_many_arguments)]
pub fn write_live_full_state_checkpoint_new(
    raw_arena: impl AsRef<Path>,
    output: impl AsRef<Path>,
    expected_raw_arena_sha256: &str,
    config: LiveFullStateConfig,
    source: LiveFullStateSourceIdentities,
    clock: ForecastClock,
    completed_wsm6_transitions: u64,
    scheduler_extension: &[u8],
) -> Result<LiveFullStateCheckpointReceipt> {
    config.validate()?;
    validate_sha256(
        expected_raw_arena_sha256,
        "expected raw live full-state arena SHA-256",
    )?;
    if clock.dt_s().to_bits() != config.dt_s.to_bits() {
        return Err(invalid(
            "live full-state clock dt differs from the contracted configuration",
        ));
    }
    if clock.step() != completed_wsm6_transitions {
        return Err(invalid(format!(
            "completed-large-step capture requires step {} to equal completed WSM6 transitions {completed_wsm6_transitions}",
            clock.step()
        )));
    }
    let scheduler_extension = validate_scheduler_extension(scheduler_extension)?;
    let scheduler_sha256 = sha256::digest_hex(&scheduler_extension);
    let identity = identity_json(
        config,
        &source,
        expected_raw_arena_sha256,
        &scheduler_sha256,
    )
    .canonical_bytes();
    if identity.len() > LIVE_FULL_STATE_IDENTITY_MAX_NBYTES {
        return Err(invalid("live full-state identity exceeds its v1 cap"));
    }
    let contract = live_full_state_contract(config)?;
    let arrays = read_raw_arena(raw_arena.as_ref(), &contract, expected_raw_arena_sha256)?;
    validate_array_semantics(&arrays, config)?;
    let scalars = boundary_scalars(clock.step(), completed_wsm6_transitions)?;
    let frame = ForecastFrame::new(
        clock,
        arrays,
        scalars,
        vec![
            ComponentBlob::new(
                LIVE_FULL_STATE_IDENTITY_BLOB,
                LIVE_FULL_STATE_IDENTITY_SCHEMA,
                identity,
            ),
            ComponentBlob::new(
                LIVE_FULL_STATE_SCHEDULER_BLOB,
                LIVE_FULL_STATE_SCHEDULER_EXTENSION_SCHEMA,
                scheduler_extension,
            ),
        ],
    );
    let forecast = write_forecast_checkpoint_new(&contract, &frame, output)?;
    Ok(LiveFullStateCheckpointReceipt {
        raw_arena_sha256: expected_raw_arena_sha256.to_string(),
        raw_arena_nbytes: array_bytes(&contract)?,
        completed_wsm6_transitions,
        source,
        forecast,
    })
}

/// Read and semantically revalidate one specialized full-state checkpoint.
pub fn read_live_full_state_checkpoint(
    input: impl AsRef<Path>,
    expected_contract_sha256: &str,
    expected_artifact_sha256: &str,
    expected_initialized_product_restart_sha256: &str,
) -> Result<RestoredLiveFullState> {
    validate_sha256(
        expected_artifact_sha256,
        "expected checkpoint artifact SHA-256",
    )?;
    validate_sha256(
        expected_initialized_product_restart_sha256,
        "expected initialized-product restart SHA-256",
    )?;
    let (contract, forecast) =
        read_forecast_checkpoint_embedded(input.as_ref(), expected_contract_sha256)?;
    if forecast.receipt().artifact_sha256 != expected_artifact_sha256 {
        return Err(invalid(format!(
            "live full-state artifact SHA-256 is {}; independently expected {expected_artifact_sha256}",
            forecast.receipt().artifact_sha256
        )));
    }
    let blobs = forecast.frame().component_blobs();
    let identity_blob = blobs
        .iter()
        .find(|blob| blob.name() == LIVE_FULL_STATE_IDENTITY_BLOB)
        .ok_or_else(|| invalid("live full-state identity blob is missing"))?;
    let scheduler_blob = blobs
        .iter()
        .find(|blob| blob.name() == LIVE_FULL_STATE_SCHEDULER_BLOB)
        .ok_or_else(|| invalid("live full-state scheduler extension blob is missing"))?;
    if blobs.len() != 2
        || identity_blob.schema() != LIVE_FULL_STATE_IDENTITY_SCHEMA
        || scheduler_blob.schema() != LIVE_FULL_STATE_SCHEDULER_EXTENSION_SCHEMA
    {
        return Err(invalid(
            "live full-state component blob inventory or schema changed",
        ));
    }
    let scheduler_extension = validate_scheduler_extension(scheduler_blob.bytes())?;
    let parsed = parse_identity(identity_blob.bytes())?;
    if parsed.source.initialized_product_restart_sha256
        != expected_initialized_product_restart_sha256
    {
        return Err(invalid(format!(
            "live full-state source initialized-product restart SHA-256 is {}; independently expected {expected_initialized_product_restart_sha256}",
            parsed.source.initialized_product_restart_sha256
        )));
    }
    let scheduler_sha256 = sha256::digest_hex(&scheduler_extension);
    if parsed.scheduler_extension_sha256 != scheduler_sha256 {
        return Err(invalid(
            "live full-state scheduler extension bytes differ from the identity hash",
        ));
    }
    let expected_identity = identity_json(
        parsed.config,
        &parsed.source,
        &parsed.raw_arena_sha256,
        &parsed.scheduler_extension_sha256,
    )
    .canonical_bytes();
    if expected_identity != identity_blob.bytes() {
        return Err(invalid(
            "live full-state identity contains unknown or changed semantics",
        ));
    }
    let expected_contract = live_full_state_contract(parsed.config)?;
    if contract != expected_contract {
        return Err(invalid(
            "live full-state embedded contract differs from the Rust-owned v1 inventory",
        ));
    }
    let completed_wsm6_transitions =
        validate_boundary_scalars(forecast.frame(), forecast.frame().clock().step())?;
    validate_array_semantics(forecast.frame().arrays(), parsed.config)?;
    let actual_raw_sha256 = arrays_sha256(forecast.frame().arrays());
    if actual_raw_sha256 != parsed.raw_arena_sha256 {
        return Err(invalid(format!(
            "restored live full-state raw arena SHA-256 is {actual_raw_sha256}; identity requires {}",
            parsed.raw_arena_sha256
        )));
    }
    Ok(RestoredLiveFullState {
        contract,
        forecast,
        config: parsed.config,
        source: parsed.source,
        raw_arena_sha256: parsed.raw_arena_sha256,
        scheduler_extension,
        completed_wsm6_transitions,
    })
}

/// Publish an exact raw arena plus canonical metadata for host/device restore.
pub fn restore_live_full_state_checkpoint_new(
    restored: &RestoredLiveFullState,
    output_directory: impl AsRef<Path>,
) -> Result<LiveFullStateRestoreReceipt> {
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace live full-state restore destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "live full-state restore parent does not exist: {}",
            parent.display()
        )));
    }
    let staging = staging_path(output)?;
    fs::create_dir(&staging)?;
    let result = restore_into_directory(restored, &staging);
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

fn live_full_state_contract(config: LiveFullStateConfig) -> Result<ForecastContract> {
    config.validate()?;
    ForecastContract::new(
        LIVE_FULL_STATE_PROFILE_ID,
        sha256::digest_hex(&configuration_json(config).canonical_bytes()),
        sha256::digest_hex(&execution_identity_json().canonical_bytes()),
        live_full_state_array_specs(config.nz, config.ny, config.nx)?,
        vec![
            SchedulerScalarSpec::new("acoustic_substep", SchedulerScalarType::I64)?,
            SchedulerScalarSpec::new("completed_large_steps", SchedulerScalarType::U64)?,
            SchedulerScalarSpec::new("completed_wsm6_transitions", SchedulerScalarType::U64)?,
            SchedulerScalarSpec::new("large_step_boundary", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("nrk", SchedulerScalarType::I64)?,
            SchedulerScalarSpec::new("radiation_history_initialized", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("scalar_stage", SchedulerScalarType::I64)?,
            SchedulerScalarSpec::new("wsm6_transition_ready", SchedulerScalarType::Bool)?,
        ],
        vec![
            ComponentBlobSpec::new(
                LIVE_FULL_STATE_IDENTITY_BLOB,
                LIVE_FULL_STATE_IDENTITY_SCHEMA,
                LIVE_FULL_STATE_IDENTITY_MAX_NBYTES,
            )?,
            ComponentBlobSpec::new(
                LIVE_FULL_STATE_SCHEDULER_BLOB,
                LIVE_FULL_STATE_SCHEDULER_EXTENSION_SCHEMA,
                LIVE_FULL_STATE_SCHEDULER_MAX_NBYTES,
            )?,
        ],
    )
}

fn live_full_state_array_specs(nz: usize, ny: usize, nx: usize) -> Result<Vec<ForecastArraySpec>> {
    let mut specs = terrain_open_carrier_specs(nz, ny, nx)?;
    let volume = vec![nz, ny, nx];
    let packed = vec![SPECIES.len(), nz, ny, nx];
    let surface = vec![ny, nx];
    for name in WSM6_TRANSITION_ARRAY_NAMES {
        let (shape, meaning, units, stagger) = if name == "state.q3d" {
            (
                packed.clone(),
                "WSM6 current packed six-species state",
                "kg kg-1",
                "physical packed species,scalar",
            )
        } else if name.starts_with("component.mp_") || name == "component.mp_sr" {
            (
                surface.clone(),
                "WSM6 component persistent surface accumulator",
                if name == "component.mp_sr" { "1" } else { "mm" },
                "physical surface",
            )
        } else {
            let units = match name {
                "state.prs" => "Pa",
                "state.rho" => "kg m-3",
                "state.rr" => "m3 kg-1",
                "state.th3d" => "K",
                "held.qpten" => "s-1",
                "held.qtten" | "component.h_diabatic" => "K s-1",
                "held.qvten" | "held.qcten" => "kg kg-1 s-1",
                name if name.starts_with("component.eff") => "m",
                _ => "1",
            };
            (
                volume.clone(),
                "WSM6 transition persistent volume carrier",
                units,
                "physical scalar",
            )
        };
        specs.push(ForecastArraySpec::new(
            name,
            meaning,
            units,
            stagger,
            one_bounds(&shape)?,
            shape,
        )?);
    }
    for name in SCALAR_ARRAY_NAMES {
        let meaning = match name {
            "scalar.qa" => "packed six-species A-time scalar state",
            "scalar.q3d" => "packed six-species current scalar state",
            "scalar.qten" => "packed six-species current scalar tendency",
            "scalar.qten1" => "packed six-species held pre-RK scalar tendency",
            _ => unreachable!(),
        };
        specs.push(ForecastArraySpec::new(
            name,
            meaning,
            if name.ends_with("ten") || name.ends_with("ten1") {
                "kg kg-1 s-1"
            } else {
                "kg kg-1"
            },
            "physical packed species,scalar",
            one_bounds(&packed)?,
            packed.clone(),
        )?);
    }
    for name in CLOSURE_ARRAY_NAMES {
        let shape = vec![nz + 1, ny, nx];
        specs.push(ForecastArraySpec::new(
            name,
            "Deardorff closure prognostic TKE A-time carrier",
            "m2 s-2",
            "physical z face",
            one_bounds(&shape)?,
            shape,
        )?);
    }
    specs.sort_by(|left, right| left.name().cmp(right.name()));
    let names = specs
        .iter()
        .map(ForecastArraySpec::name)
        .collect::<Vec<_>>();
    if specs.len() != LIVE_FULL_STATE_ARRAY_COUNT
        || names.iter().copied().collect::<BTreeSet<_>>().len() != specs.len()
    {
        return Err(invalid(format!(
            "live full-state v1 inventory has {} arrays or duplicate names; expected {LIVE_FULL_STATE_ARRAY_COUNT}",
            specs.len()
        )));
    }
    Ok(specs)
}

fn boundary_scalars(step: u64, completed_wsm6_transitions: u64) -> Result<Vec<SchedulerScalar>> {
    Ok(vec![
        SchedulerScalar::new("acoustic_substep", SchedulerScalarValue::I64(0))?,
        SchedulerScalar::new("completed_large_steps", SchedulerScalarValue::U64(step))?,
        SchedulerScalar::new(
            "completed_wsm6_transitions",
            SchedulerScalarValue::U64(completed_wsm6_transitions),
        )?,
        SchedulerScalar::new("large_step_boundary", SchedulerScalarValue::Bool(true))?,
        SchedulerScalar::new("nrk", SchedulerScalarValue::I64(0))?,
        SchedulerScalar::new(
            "radiation_history_initialized",
            SchedulerScalarValue::Bool(true),
        )?,
        SchedulerScalar::new("scalar_stage", SchedulerScalarValue::I64(0))?,
        SchedulerScalar::new("wsm6_transition_ready", SchedulerScalarValue::Bool(true))?,
    ])
}

fn validate_boundary_scalars(frame: &ForecastFrame, step: u64) -> Result<u64> {
    let map = frame
        .scheduler_scalars()
        .iter()
        .map(|scalar| (scalar.name(), scalar.value()))
        .collect::<BTreeMap<_, _>>();
    let expected_fixed = [
        ("acoustic_substep", SchedulerScalarValue::I64(0)),
        ("completed_large_steps", SchedulerScalarValue::U64(step)),
        ("large_step_boundary", SchedulerScalarValue::Bool(true)),
        ("nrk", SchedulerScalarValue::I64(0)),
        (
            "radiation_history_initialized",
            SchedulerScalarValue::Bool(true),
        ),
        ("scalar_stage", SchedulerScalarValue::I64(0)),
        ("wsm6_transition_ready", SchedulerScalarValue::Bool(true)),
    ];
    for (name, expected) in expected_fixed {
        if map.get(name).copied() != Some(expected) {
            return Err(invalid(format!(
                "live full-state scheduler scalar {name} is not the completed-boundary value {expected:?}"
            )));
        }
    }
    let completed = match map.get("completed_wsm6_transitions").copied() {
        Some(SchedulerScalarValue::U64(value)) => value,
        _ => {
            return Err(invalid(
                "live full-state completed_wsm6_transitions scalar is missing or mistyped",
            ))
        }
    };
    if completed != step {
        return Err(invalid(
            "live full-state WSM6 transition cadence differs from completed large steps",
        ));
    }
    Ok(completed)
}

fn read_raw_arena(
    path: &Path,
    contract: &ForecastContract,
    expected_sha256: &str,
) -> Result<Vec<ForecastArray>> {
    let link = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect live full-state raw arena {}: {error}",
            path.display()
        ))
    })?;
    if link.file_type().is_symlink() || !link.is_file() {
        return Err(invalid(
            "live full-state raw arena must be a regular non-symlink file",
        ));
    }
    let expected_nbytes = array_bytes(contract)?;
    if link.len() != expected_nbytes {
        return Err(invalid(format!(
            "live full-state raw arena has {} bytes; contract requires {expected_nbytes}",
            link.len()
        )));
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut aggregate = Sha256::default();
    let mut arrays = Vec::with_capacity(contract.arrays().len());
    let mut raw = vec![0u8; 1024 * 1024];
    for spec in contract.arrays() {
        let words = spec_words(spec)?;
        let mut values = Vec::with_capacity(words);
        let mut remaining = words;
        while remaining != 0 {
            let take_words = remaining.min(raw.len() / 4);
            let bytes = &mut raw[..take_words * 4];
            reader.read_exact(bytes)?;
            aggregate.update(bytes);
            for word in bytes.chunks_exact(4) {
                values.push(f32::from_bits(u32::from_le_bytes(
                    word.try_into().expect("four-byte raw word"),
                )));
            }
            remaining -= take_words;
        }
        arrays.push(ForecastArray::new(spec.name(), values));
    }
    let actual = sha256::hex(&aggregate.finalize());
    if actual != expected_sha256 {
        return Err(invalid(format!(
            "live full-state raw arena SHA-256 is {actual}; independently expected {expected_sha256}"
        )));
    }
    Ok(arrays)
}

fn validate_array_semantics(arrays: &[ForecastArray], config: LiveFullStateConfig) -> Result<()> {
    let map = arrays
        .iter()
        .map(|array| (array.name(), array.values()))
        .collect::<BTreeMap<_, _>>();
    if map.len() != LIVE_FULL_STATE_ARRAY_COUNT {
        return Err(invalid("live full-state array inventory changed"));
    }
    for array in arrays {
        for (index, value) in array.values().iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "live full-state carrier {} contains non-finite f32 bits {:08x} at word {index}",
                    array.name(),
                    value.to_bits()
                )));
            }
        }
    }
    for name in [
        "pi0",
        "prs0",
        "th0",
        "thv0",
        "rth0",
        "rho0",
        "rr0",
        "rf0",
        "rrf0",
        "rho0s",
        "pi0s",
        "prs0s",
        "rth0s",
        "gz",
        "rgz",
        "gzu",
        "rgzu",
        "gzv",
        "rgzv",
        "state.prs",
        "state.rho",
        "state.rr",
    ] {
        require_positive(array(&map, name)?, name)?;
    }
    for name in [
        "qv0",
        "scalar.qa",
        "scalar.q3d",
        "state.q3d",
        "closure.tkea",
    ] {
        require_nonnegative(array(&map, name)?, name)?;
    }
    for name in WSM6_TRANSITION_ARRAY_NAMES {
        if name.starts_with("component.eff")
            || name.starts_with("component.mp_")
            || name == "component.mp_sr"
        {
            require_nonnegative(array(&map, name)?, name)?;
        }
    }
    compare_padded_interior(
        array(&map, "pp3d")?,
        [config.nz + 2, config.ny + 2, config.nx + 2],
        [1, 1, 1],
        array(&map, "state.pp3d")?,
        [config.nz, config.ny, config.nx],
        "pp3d/state.pp3d",
    )?;
    compare_padded_interior(
        array(&map, "th3d")?,
        [config.nz + 2, config.ny + 6, config.nx + 6],
        [1, 3, 3],
        array(&map, "state.th3d")?,
        [config.nz, config.ny, config.nx],
        "th3d/state.th3d",
    )?;
    require_bits_equal(
        array(&map, "scalar.q3d")?,
        array(&map, "state.q3d")?,
        "scalar.q3d/state.q3d",
    )?;
    Ok(())
}

fn compare_padded_interior(
    padded: &[f32],
    full: [usize; 3],
    start: [usize; 3],
    physical: &[f32],
    count: [usize; 3],
    label: &str,
) -> Result<()> {
    let expected_full = full.iter().product::<usize>();
    let expected_physical = count.iter().product::<usize>();
    if padded.len() != expected_full || physical.len() != expected_physical {
        return Err(invalid(format!(
            "live full-state duplicate binding {label} has changed shapes"
        )));
    }
    let mut target = 0usize;
    for k in start[0]..start[0] + count[0] {
        for j in start[1]..start[1] + count[1] {
            let first = (k * full[1] + j) * full[2] + start[2];
            for value in &padded[first..first + count[2]] {
                if value.to_bits() != physical[target].to_bits() {
                    return Err(invalid(format!(
                        "live full-state duplicate binding {label} differs at physical word {target}"
                    )));
                }
                target += 1;
            }
        }
    }
    Ok(())
}

fn require_bits_equal(left: &[f32], right: &[f32], label: &str) -> Result<()> {
    if left.len() != right.len() {
        return Err(invalid(format!(
            "live full-state duplicate binding {label} lengths differ"
        )));
    }
    if let Some(index) = left
        .iter()
        .zip(right)
        .position(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err(invalid(format!(
            "live full-state duplicate binding {label} differs at word {index}"
        )));
    }
    Ok(())
}

fn require_positive(values: &[f32], name: &str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| *value <= 0.0) {
        return Err(invalid(format!(
            "live full-state carrier {name} must be positive; word {index} is {:08x}",
            values[index].to_bits()
        )));
    }
    Ok(())
}

fn require_nonnegative(values: &[f32], name: &str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| *value < 0.0) {
        return Err(invalid(format!(
            "live full-state carrier {name} must be nonnegative; word {index} is {:08x}",
            values[index].to_bits()
        )));
    }
    Ok(())
}

fn validate_scheduler_extension(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.is_empty() || bytes.len() > LIVE_FULL_STATE_SCHEDULER_MAX_NBYTES {
        return Err(invalid(format!(
            "live full-state scheduler extension must contain 1 through {LIVE_FULL_STATE_SCHEDULER_MAX_NBYTES} bytes"
        )));
    }
    let value = json::parse(bytes)
        .map_err(|error| invalid(format!("scheduler extension JSON is invalid: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid(
            "live full-state scheduler extension must be canonical sorted-key compact UTF-8 JSON",
        ));
    }
    exact_keys(&value, &["entries", "schema"], "scheduler extension")?;
    fixed_string(
        &value,
        "schema",
        LIVE_FULL_STATE_SCHEDULER_EXTENSION_SCHEMA,
        "scheduler extension",
    )?;
    let entries = value
        .get("entries")
        .and_then(JsonValue::as_object)
        .map_err(|error| invalid(format!("scheduler extension.entries: {error}")))?;
    for (name, _) in entries {
        validate_extension_name(name)?;
        if SCHEDULER_NAMES.contains(&name.as_str()) {
            return Err(invalid(format!(
                "scheduler extension key {name:?} collides with a typed core scheduler scalar"
            )));
        }
    }
    Ok(bytes.to_vec())
}

fn parse_identity(bytes: &[u8]) -> Result<ParsedIdentity> {
    if bytes.is_empty() || bytes.len() > LIVE_FULL_STATE_IDENTITY_MAX_NBYTES {
        return Err(invalid(
            "live full-state identity length is outside its v1 cap",
        ));
    }
    let value = json::parse(bytes)
        .map_err(|error| invalid(format!("live full-state identity is invalid JSON: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid("live full-state identity is not canonical JSON"));
    }
    exact_keys(
        &value,
        &[
            "admission",
            "authority_tarball_sha256",
            "capture_phase",
            "carrier_abi",
            "claim_limit",
            "configuration",
            "duplicate_bindings",
            "recomputable_families",
            "restore_phase",
            "raw_arena_sha256",
            "scheduler_extension_sha256",
            "schema",
            "source",
            "transition",
        ],
        "live full-state identity",
    )?;
    fixed_string(
        &value,
        "schema",
        LIVE_FULL_STATE_IDENTITY_SCHEMA,
        "live full-state identity",
    )?;
    let admission = LiveFullStateAdmission::parse(required_string(
        &value,
        "admission",
        "live full-state identity",
    )?)?;
    let configuration = value
        .get("configuration")
        .map_err(|error| invalid(format!("live full-state configuration: {error}")))?;
    exact_keys(
        configuration,
        &["dt_f64_bits", "shape_zyx", "spacing_m_f64_bits"],
        "live full-state configuration",
    )?;
    let shape = usize_array::<3>(configuration, "shape_zyx", "configuration")?;
    let spacing_bits = string_array::<3>(configuration, "spacing_m_f64_bits", "configuration")?;
    let config = LiveFullStateConfig {
        nz: shape[0],
        ny: shape[1],
        nx: shape[2],
        spacing_m: [
            parse_f64_bits(&spacing_bits[0], "dx")?,
            parse_f64_bits(&spacing_bits[1], "dy")?,
            parse_f64_bits(&spacing_bits[2], "dz")?,
        ],
        dt_s: parse_f64_bits(
            required_string(configuration, "dt_f64_bits", "configuration")?,
            "dt",
        )?,
        admission,
    };
    config.validate()?;
    let source_value = value
        .get("source")
        .map_err(|error| invalid(format!("live full-state source: {error}")))?;
    exact_keys(
        source_value,
        &[
            "initialized_product_manifest_sha256",
            "initialized_product_payload_sha256",
            "initialized_product_restart_sha256",
            "terrain_open_restart_sha256",
            "wsm6_transition_restart_sha256",
        ],
        "live full-state source",
    )?;
    let source = LiveFullStateSourceIdentities::new(
        required_string(source_value, "initialized_product_restart_sha256", "source")?,
        required_string(
            source_value,
            "initialized_product_manifest_sha256",
            "source",
        )?,
        required_string(source_value, "initialized_product_payload_sha256", "source")?,
        required_string(source_value, "terrain_open_restart_sha256", "source")?,
        required_string(source_value, "wsm6_transition_restart_sha256", "source")?,
    )?;
    let raw_arena_sha256 =
        required_string(&value, "raw_arena_sha256", "live full-state identity")?.to_string();
    let scheduler_extension_sha256 = required_string(
        &value,
        "scheduler_extension_sha256",
        "live full-state identity",
    )?
    .to_string();
    validate_sha256(&raw_arena_sha256, "identity raw arena SHA-256")?;
    validate_sha256(
        &scheduler_extension_sha256,
        "identity scheduler extension SHA-256",
    )?;
    Ok(ParsedIdentity {
        config,
        source,
        raw_arena_sha256,
        scheduler_extension_sha256,
    })
}

fn identity_json(
    config: LiveFullStateConfig,
    source: &LiveFullStateSourceIdentities,
    raw_arena_sha256: &str,
    scheduler_extension_sha256: &str,
) -> JsonValue {
    object(vec![
        ("admission", string(config.admission.as_str())),
        ("authority_tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
        ("capture_phase", string(LIVE_FULL_STATE_CAPTURE_PHASE)),
        ("carrier_abi", string(LIVE_FULL_STATE_CARRIER_ABI)),
        ("claim_limit", string(LIVE_FULL_STATE_CLAIM_LIMIT)),
        ("configuration", configuration_json(config)),
        (
            "duplicate_bindings",
            JsonValue::Array(
                DUPLICATE_BINDINGS
                    .iter()
                    .map(|(live, transition, relation)| {
                        object(vec![
                            ("live", string(live)),
                            ("relation", string(relation)),
                            ("transition", string(transition)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "recomputable_families",
            JsonValue::Array(
                RECOMPUTABLE_FAMILIES
                    .iter()
                    .map(|value| string(value))
                    .collect(),
            ),
        ),
        ("restore_phase", string(LIVE_FULL_STATE_RESTORE_PHASE)),
        ("raw_arena_sha256", string(raw_arena_sha256)),
        (
            "scheduler_extension_sha256",
            string(scheduler_extension_sha256),
        ),
        ("schema", string(LIVE_FULL_STATE_IDENTITY_SCHEMA)),
        ("source", source_json(source)),
        ("transition", execution_identity_json()),
    ])
}

fn configuration_json(config: LiveFullStateConfig) -> JsonValue {
    object(vec![
        ("dt_f64_bits", f64_bits(config.dt_s)),
        (
            "shape_zyx",
            JsonValue::Array([config.nz, config.ny, config.nx].map(number).to_vec()),
        ),
        (
            "spacing_m_f64_bits",
            JsonValue::Array(config.spacing_m.map(f64_bits).to_vec()),
        ),
    ])
}

fn execution_identity_json() -> JsonValue {
    object(vec![
        ("closure_sgs_abi", string(CUDA_SGS_TKE_ABI)),
        ("closure_sgs_schema", string(CUDA_SGS_TKE_SCHEMA)),
        (
            "closure_terrain_deformation_abi",
            string(CUDA_TERRAIN_DEFORMATION_ABI),
        ),
        (
            "closure_terrain_deformation_schema",
            string(CUDA_TERRAIN_DEFORMATION_SCHEMA),
        ),
        ("terrain_open_carrier_abi", string(TERRAIN_OPEN_CARRIER_ABI)),
        ("wsm6_bridge_abi", string(WSM6_TRANSITION_BRIDGE_ABI)),
        ("wsm6_carrier_abi", string(WSM6_TRANSITION_CARRIER_ABI)),
        ("wsm6_component_abi", string(WSM6_TRANSITION_COMPONENT_ABI)),
        (
            "wsm6_in_memory_restart_schema",
            string(WSM6_TRANSITION_IN_MEMORY_RESTART_SCHEMA),
        ),
        ("wsm6_transition_abi", string(WSM6_TRANSITION_ABI)),
    ])
}

fn source_json(source: &LiveFullStateSourceIdentities) -> JsonValue {
    object(vec![
        (
            "initialized_product_manifest_sha256",
            string(&source.initialized_product_manifest_sha256),
        ),
        (
            "initialized_product_payload_sha256",
            string(&source.initialized_product_payload_sha256),
        ),
        (
            "initialized_product_restart_sha256",
            string(&source.initialized_product_restart_sha256),
        ),
        (
            "terrain_open_restart_sha256",
            string(&source.terrain_open_restart_sha256),
        ),
        (
            "wsm6_transition_restart_sha256",
            string(&source.wsm6_transition_restart_sha256),
        ),
    ])
}

fn restore_into_directory(
    restored: &RestoredLiveFullState,
    directory: &Path,
) -> Result<LiveFullStateRestoreReceipt> {
    let payload_path = directory.join(LIVE_FULL_STATE_ARENA_FILE);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&payload_path)?;
    let mut writer = BufWriter::new(file);
    let mut payload_hasher = Sha256::default();
    let mut offset = 0u64;
    let mut carriers = Vec::new();
    for (spec, array) in restored
        .contract
        .arrays()
        .iter()
        .zip(restored.forecast.frame().arrays())
    {
        if spec.name() != array.name() {
            return Err(invalid("restored live full-state array order changed"));
        }
        let mut field_hasher = Sha256::default();
        let mut bytes = Vec::with_capacity(1024 * 1024);
        for chunk in array.values().chunks((1024 * 1024) / 4) {
            bytes.clear();
            for value in chunk {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            writer.write_all(&bytes)?;
            field_hasher.update(&bytes);
            payload_hasher.update(&bytes);
        }
        let nbytes = spec_nbytes(spec)?;
        carriers.push(RestoreCarrier {
            name: spec.name().to_string(),
            shape: spec.shape().to_vec(),
            offset_nbytes: offset,
            nbytes,
            sha256: sha256::hex(&field_hasher.finalize()),
        });
        offset = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("restore payload offset overflows"))?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    let payload_sha256 = sha256::hex(&payload_hasher.finalize());
    if payload_sha256 != restored.raw_arena_sha256 || offset != array_bytes(&restored.contract)? {
        return Err(invalid(
            "restored live full-state arena bytes differ from the checkpoint identity",
        ));
    }
    let manifest = restore_manifest_json(restored, &carriers, offset, &payload_sha256);
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    write_new_synced(
        &directory.join(LIVE_FULL_STATE_MANIFEST_FILE),
        &manifest_bytes,
    )?;
    let receipt = LiveFullStateRestoreReceipt {
        checkpoint_artifact_sha256: restored.forecast.receipt().artifact_sha256.clone(),
        checkpoint_contract_sha256: restored.contract.sha256(),
        manifest_sha256,
        manifest_nbytes: manifest_bytes.len(),
        payload_sha256,
        payload_nbytes: offset,
        carrier_count: carriers.len(),
        step: restored.forecast.frame().clock().step(),
    };
    write_new_synced(
        &directory.join(LIVE_FULL_STATE_RECEIPT_FILE),
        receipt.canonical_json().as_bytes(),
    )?;
    let actual = fs::read_dir(directory)?
        .map(|entry| {
            entry
                .map(|value| value.file_name().to_string_lossy().into_owned())
                .map_err(StateError::from)
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let expected = [
        LIVE_FULL_STATE_ARENA_FILE.to_string(),
        LIVE_FULL_STATE_MANIFEST_FILE.to_string(),
        LIVE_FULL_STATE_RECEIPT_FILE.to_string(),
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid(
            "live full-state restore directory inventory changed",
        ));
    }
    Ok(receipt)
}

fn restore_manifest_json(
    restored: &RestoredLiveFullState,
    carriers: &[RestoreCarrier],
    payload_nbytes: u64,
    payload_sha256: &str,
) -> JsonValue {
    let scalars = restored
        .forecast
        .frame()
        .scheduler_scalars()
        .iter()
        .map(|scalar| {
            let value = match scalar.value() {
                SchedulerScalarValue::I64(value) => JsonValue::Number(value.to_string()),
                SchedulerScalarValue::U64(value) => JsonValue::Number(value.to_string()),
                SchedulerScalarValue::F64(value) => f64_bits(value),
                SchedulerScalarValue::Bool(value) => JsonValue::Bool(value),
            };
            (scalar.name().to_string(), value)
        })
        .collect();
    let extension = json::parse(&restored.scheduler_extension)
        .expect("scheduler extension was validated before restore");
    object(vec![
        ("capture_phase", string(LIVE_FULL_STATE_CAPTURE_PHASE)),
        ("carrier_abi", string(LIVE_FULL_STATE_CARRIER_ABI)),
        (
            "carriers",
            JsonValue::Array(carriers.iter().map(RestoreCarrier::json).collect()),
        ),
        ("claim_limit", string(LIVE_FULL_STATE_CLAIM_LIMIT)),
        ("dtype", string("<f4")),
        ("itemsize", number(4)),
        ("order", string("C")),
        (
            "payload",
            object(vec![
                ("file", string(LIVE_FULL_STATE_ARENA_FILE)),
                ("nbytes", u64_number(payload_nbytes)),
                ("sha256", string(payload_sha256)),
            ]),
        ),
        ("restore_phase", string(LIVE_FULL_STATE_RESTORE_PHASE)),
        ("scheduler_extension", extension),
        ("scheduler_scalars", JsonValue::Object(scalars)),
        ("schema", string(LIVE_FULL_STATE_RESTORE_MANIFEST_SCHEMA)),
        ("source", source_json(&restored.source)),
        (
            "source_checkpoint",
            object(vec![
                (
                    "artifact_sha256",
                    string(&restored.forecast.receipt().artifact_sha256),
                ),
                ("contract_sha256", string(&restored.contract.sha256())),
                ("dt_f64_bits", f64_bits(restored.config.dt_s)),
                (
                    "model_time_f64_bits",
                    f64_bits(restored.forecast.frame().clock().model_time_s()),
                ),
                ("step", u64_number(restored.forecast.frame().clock().step())),
            ]),
        ),
    ])
}

fn arrays_sha256(arrays: &[ForecastArray]) -> String {
    let mut hasher = Sha256::default();
    let mut bytes = Vec::with_capacity(1024 * 1024);
    for array in arrays {
        for chunk in array.values().chunks((1024 * 1024) / 4) {
            bytes.clear();
            for value in chunk {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            hasher.update(&bytes);
        }
    }
    sha256::hex(&hasher.finalize())
}

fn array_bytes(contract: &ForecastContract) -> Result<u64> {
    contract.arrays().iter().try_fold(0u64, |total, spec| {
        total
            .checked_add(spec_nbytes(spec)?)
            .ok_or_else(|| invalid("live full-state arena byte total overflows"))
    })
}

fn spec_words(spec: &ForecastArraySpec) -> Result<usize> {
    spec.shape().iter().try_fold(1usize, |total, extent| {
        total.checked_mul(*extent).ok_or_else(|| {
            invalid(format!(
                "live full-state carrier {} shape product overflows",
                spec.name()
            ))
        })
    })
}

fn spec_nbytes(spec: &ForecastArraySpec) -> Result<u64> {
    let bytes = spec_words(spec)?
        .checked_mul(4)
        .ok_or_else(|| invalid("live full-state carrier byte count overflows"))?;
    u64::try_from(bytes).map_err(|_| invalid("live full-state byte count does not fit u64"))
}

fn one_bounds(shape: &[usize]) -> Result<Vec<i32>> {
    let mut bounds = Vec::with_capacity(shape.len() * 2);
    for extent in shape {
        bounds.push(1);
        bounds.push(
            i32::try_from(*extent)
                .map_err(|_| invalid("live full-state extent does not fit i32"))?,
        );
    }
    Ok(bounds)
}

fn array<'a>(map: &BTreeMap<&str, &'a [f32]>, name: &str) -> Result<&'a [f32]> {
    map.get(name)
        .copied()
        .ok_or_else(|| invalid(format!("live full-state carrier {name} is missing")))
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("live full-state destination needs a UTF-8 file name"))?;
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

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) -> Result<()> {
    let actual = value
        .as_object()
        .map_err(|error| invalid(format!("{label}: {error}")))?
        .iter()
        .map(|(key, _)| key.as_str())
        .collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid(format!(
            "{label} keys are {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn fixed_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = required_string(value, key, label)?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn required_string<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn usize_array<const N: usize>(value: &JsonValue, key: &str, label: &str) -> Result<[usize; N]> {
    let values = value
        .get(key)
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
    if values.len() != N {
        return Err(invalid(format!(
            "{label}.{key} has {} values; expected {N}",
            values.len()
        )));
    }
    let parsed = values
        .iter()
        .map(|item| {
            let value = item
                .as_u64()
                .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
            usize::try_from(value)
                .map_err(|_| invalid(format!("{label}.{key} value does not fit usize")))
        })
        .collect::<Result<Vec<_>>>()?;
    parsed
        .try_into()
        .map_err(|_| invalid(format!("{label}.{key} length changed")))
}

fn string_array<const N: usize>(value: &JsonValue, key: &str, label: &str) -> Result<[String; N]> {
    let values = value
        .get(key)
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
    if values.len() != N {
        return Err(invalid(format!(
            "{label}.{key} has {} values; expected {N}",
            values.len()
        )));
    }
    let parsed = values
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .map_err(|error| invalid(format!("{label}.{key}: {error}")))
        })
        .collect::<Result<Vec<_>>>()?;
    parsed
        .try_into()
        .map_err(|_| invalid(format!("{label}.{key} length changed")))
}

fn parse_f64_bits(value: &str, label: &str) -> Result<f64> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "live full-state {label} must be 16 lower-case hexadecimal f64 bits"
        )));
    }
    let bits = u64::from_str_radix(value, 16)
        .map_err(|_| invalid(format!("live full-state {label} f64 bits are invalid")))?;
    let result = f64::from_bits(bits);
    if !result.is_finite() {
        return Err(invalid(format!(
            "live full-state {label} f64 bits are non-finite"
        )));
    }
    Ok(result)
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

fn validate_extension_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        })
    {
        return Err(invalid(format!(
            "scheduler extension key {value:?} must use 1..128 lower-case ASCII [a-z0-9._-] characters"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
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

fn f64_bits(value: f64) -> JsonValue {
    string(&format!("{:016x}", value.to_bits()))
}

fn canonical_string(value: JsonValue) -> String {
    String::from_utf8(value.canonical_bytes()).expect("canonical live full-state JSON is UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRoot {
        path: PathBuf,
    }

    impl TestRoot {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "gpuwm-cm1-live-full-{label}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn scheduler_extension() -> Vec<u8> {
        br#"{"entries":{"coupled.scheduler.epoch":7},"schema":"gpuwm-cm1.live-full-state-scheduler-extension/v1"}"#.to_vec()
    }

    fn sources() -> LiveFullStateSourceIdentities {
        LiveFullStateSourceIdentities::new(
            "11".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
            "44".repeat(32),
            "55".repeat(32),
        )
        .unwrap()
    }

    fn fixture(config: LiveFullStateConfig) -> Vec<ForecastArray> {
        let specs = live_full_state_array_specs(config.nz, config.ny, config.nx).unwrap();
        let mut arrays = specs
            .iter()
            .map(|spec| {
                let fill = if ["qv0", "scalar.qa", "scalar.q3d", "state.q3d"].contains(&spec.name())
                {
                    0.001
                } else {
                    1.0
                };
                ForecastArray::new(spec.name(), vec![fill; spec_words(spec).unwrap()])
            })
            .collect::<Vec<_>>();
        let mut map = arrays
            .iter_mut()
            .map(|array| (array.name().to_string(), array))
            .collect::<BTreeMap<_, _>>();
        let physical_pp = vec![0.25; config.nz * config.ny * config.nx];
        let physical_th = vec![0.5; config.nz * config.ny * config.nx];
        let packed_q = vec![0.001; SPECIES.len() * config.nz * config.ny * config.nx];
        fill_interior(
            map.get_mut("pp3d").unwrap().values_mut(),
            [config.nz + 2, config.ny + 2, config.nx + 2],
            [1, 1, 1],
            [config.nz, config.ny, config.nx],
            &physical_pp,
        );
        map.get_mut("state.pp3d")
            .unwrap()
            .values_mut()
            .clone_from_slice(&physical_pp);
        fill_interior(
            map.get_mut("th3d").unwrap().values_mut(),
            [config.nz + 2, config.ny + 6, config.nx + 6],
            [1, 3, 3],
            [config.nz, config.ny, config.nx],
            &physical_th,
        );
        map.get_mut("state.th3d")
            .unwrap()
            .values_mut()
            .clone_from_slice(&physical_th);
        map.get_mut("scalar.q3d")
            .unwrap()
            .values_mut()
            .clone_from_slice(&packed_q);
        map.get_mut("state.q3d")
            .unwrap()
            .values_mut()
            .clone_from_slice(&packed_q);
        // Exercise preservation of signed zero and the smallest subnormal on a
        // carrier with no positivity constraint.
        map.get_mut("ppten").unwrap().values_mut()[0] = f32::from_bits(0x8000_0000);
        map.get_mut("ppten").unwrap().values_mut()[1] = f32::from_bits(0x0000_0001);
        drop(map);
        arrays
    }

    fn fill_interior(
        padded: &mut [f32],
        full: [usize; 3],
        start: [usize; 3],
        count: [usize; 3],
        physical: &[f32],
    ) {
        let mut source = 0;
        for k in start[0]..start[0] + count[0] {
            for j in start[1]..start[1] + count[1] {
                let first = (k * full[1] + j) * full[2] + start[2];
                padded[first..first + count[2]]
                    .clone_from_slice(&physical[source..source + count[2]]);
                source += count[2];
            }
        }
    }

    fn arena_bytes(arrays: &[ForecastArray]) -> Vec<u8> {
        arrays
            .iter()
            .flat_map(|array| array.values().iter())
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn production_layout_is_complete_and_inside_the_envelope_bound() {
        let config = LiveFullStateConfig::real_product();
        let contract = live_full_state_contract(config).unwrap();
        assert_eq!(contract.arrays().len(), LIVE_FULL_STATE_ARRAY_COUNT);
        assert_eq!(
            array_bytes(&contract).unwrap(),
            LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES
        );
        assert!(
            LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES as usize
                + LIVE_FULL_STATE_IDENTITY_MAX_NBYTES
                + LIVE_FULL_STATE_SCHEDULER_MAX_NBYTES
                <= crate::forecast::MAX_FORECAST_PAYLOAD_BYTES
        );
        let names = contract
            .arrays()
            .iter()
            .map(ForecastArraySpec::name)
            .collect::<BTreeSet<_>>();
        assert!(TERRAIN_OPEN_CARRIER_NAMES
            .iter()
            .all(|name| names.contains(name)));
        assert!(WSM6_TRANSITION_ARRAY_NAMES
            .iter()
            .all(|name| names.contains(name)));
        assert!(SCALAR_ARRAY_NAMES.iter().all(|name| names.contains(name)));
        assert!(CLOSURE_ARRAY_NAMES.iter().all(|name| names.contains(name)));
        let layout: JsonValue =
            json::parse(live_full_state_real_layout_json().unwrap().as_bytes()).unwrap();
        assert_eq!(
            layout.get("payload_nbytes").unwrap().as_u64().unwrap(),
            LIVE_FULL_STATE_REAL_PAYLOAD_NBYTES
        );
    }

    #[test]
    fn checkpoint_and_restore_preserve_every_raw_bit() {
        let root = TestRoot::new("roundtrip");
        let config = LiveFullStateConfig::generic(2, 3, 4, [10.0, 20.0, 30.0], 0.25).unwrap();
        let arrays = fixture(config);
        validate_array_semantics(&arrays, config).unwrap();
        let raw = arena_bytes(&arrays);
        let raw_sha = sha256::digest_hex(&raw);
        let raw_path = root.path.join("live.f32le");
        fs::write(&raw_path, &raw).unwrap();
        let checkpoint = root.path.join("state.cm1fcp");
        let receipt = write_live_full_state_checkpoint_new(
            &raw_path,
            &checkpoint,
            &raw_sha,
            config,
            sources(),
            ForecastClock::new(7, 1.75, 0.25).unwrap(),
            7,
            &scheduler_extension(),
        )
        .unwrap();
        let restored = read_live_full_state_checkpoint(
            &checkpoint,
            &receipt.forecast.contract_sha256,
            &receipt.forecast.artifact_sha256,
            &"11".repeat(32),
        )
        .unwrap();
        assert_eq!(restored.raw_arena_sha256(), raw_sha);
        assert_eq!(restored.scheduler_extension(), scheduler_extension());
        let destination = root.path.join("restored");
        let restore_receipt =
            restore_live_full_state_checkpoint_new(&restored, &destination).unwrap();
        assert_eq!(restore_receipt.payload_sha256, raw_sha);
        assert_eq!(
            fs::read(destination.join(LIVE_FULL_STATE_ARENA_FILE)).unwrap(),
            raw
        );
    }

    #[test]
    fn refuses_incoherent_duplicate_live_views_before_publication() {
        let root = TestRoot::new("duplicate");
        let config = LiveFullStateConfig::generic(2, 3, 4, [10.0, 20.0, 30.0], 0.25).unwrap();
        let mut arrays = fixture(config);
        arrays
            .iter_mut()
            .find(|array| array.name() == "state.pp3d")
            .unwrap()
            .values_mut()[0] = 0.5;
        let raw = arena_bytes(&arrays);
        let raw_path = root.path.join("bad.f32le");
        fs::write(&raw_path, &raw).unwrap();
        let output = root.path.join("bad.cm1fcp");
        let error = write_live_full_state_checkpoint_new(
            &raw_path,
            &output,
            &sha256::digest_hex(&raw),
            config,
            sources(),
            ForecastClock::new(1, 0.25, 0.25).unwrap(),
            1,
            &scheduler_extension(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("duplicate binding pp3d/state.pp3d"),
            "{error}"
        );
        assert!(!output.exists());
    }

    #[test]
    fn scheduler_extension_is_canonical_extensible_and_noncolliding() {
        validate_scheduler_extension(&scheduler_extension()).unwrap();
        let noncanonical =
            br#"{"schema":"gpuwm-cm1.live-full-state-scheduler-extension/v1", "entries":{}}"#;
        assert!(validate_scheduler_extension(noncanonical)
            .unwrap_err()
            .to_string()
            .contains("canonical"));
        let collision =
            br#"{"entries":{"nrk":1},"schema":"gpuwm-cm1.live-full-state-scheduler-extension/v1"}"#;
        assert!(validate_scheduler_extension(collision)
            .unwrap_err()
            .to_string()
            .contains("collides"));
    }
}
