//! Strict paired checkpoint/history publication for admitted GM1 child profiles.
//!
//! The CUDA side transports two synchronized little-endian float32 arenas at
//! a quiescent completed-large-step boundary.  Rust owns the exact persistent
//! and render inventories, scientific source seals, cadence validation,
//! `CM1FCP1`/`CM1FHS1` envelopes, overlap checks, and atomic pair publication.
//! The fresh step-zero child publication is intentionally not accepted here.

use crate::child_full_state::{
    CHILD_CHECKPOINT_ADMISSION_FILE, CHILD_FIRST_STEP_FILE, CHILD_FULL_STATE_MANIFEST_FILE,
    CHILD_FULL_STATE_RECEIPT_FILE,
};
use crate::forecast::{
    read_forecast_checkpoint_embedded, read_forecast_history_embedded,
    write_forecast_checkpoint_new, write_forecast_history_new, ComponentBlob, ComponentBlobSpec,
    ForecastArray, ForecastArraySpec, ForecastClock, ForecastContract, ForecastFrame,
    ForecastReceipt, RestoredForecast, SchedulerScalar, SchedulerScalarSpec, SchedulerScalarType,
    SchedulerScalarValue,
};
use crate::gm1_checkpoint::GM1_OPERATOR_SOURCE_KEYS;
use crate::json::{self, JsonValue};
use crate::sha256::{self, Sha256};
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHILD_POST_STEP_SCHEMA: &str = "gm1.real-50m-child-post-step-pair/v1";
pub const CHILD_POST_STEP_CHECKPOINT_PROFILE: &str = "gm1-real-50m-child-post-large-step/v1";
pub const CHILD_POST_STEP_HISTORY_PROFILE: &str = "gm1-real-50m-child-render-history/v1";
pub const CHILD_POST_STEP_CHECKPOINT_ABI: &str = "gm1.forecast-checkpoint-carriers/v1";
pub const CHILD_POST_STEP_HISTORY_ABI: &str = "gm1.real-50m-child-render-carriers/v1";
pub const CHILD_POST_STEP_IDENTITY_SCHEMA: &str = "gm1.real-50m-child-post-step-identity/v1";
pub const CHILD_POST_STEP_OPERATOR_SCHEMA: &str = "gm1.operator-source-manifest/v1";
pub const CHILD_POST_STEP_SCHEDULER_SCHEMA: &str = "gm1.scheduler-extension/v1";
pub const CHILD_POST_STEP_MANIFEST_SCHEMA: &str = "gm1.real-50m-child-post-step-pair-manifest/v1";
pub const CHILD_POST_STEP_RECEIPT_SCHEMA: &str = "gm1.real-50m-child-post-step-pair-receipt/v1";
pub const CHILD_POST_STEP_RESTORE_MANIFEST_SCHEMA: &str = "gm1.real-50m-child-post-step-restore/v1";
pub const CHILD_POST_STEP_RESTORE_RECEIPT_SCHEMA: &str =
    "gm1.real-50m-child-post-step-restore-receipt/v1";
pub const CHILD_POST_STEP_IDENTITY_BLOB: &str = "gm1_child_post_step_identity";
pub const CHILD_POST_STEP_OPERATOR_BLOB: &str = "gm1_child_operator_manifest";
pub const CHILD_POST_STEP_SCHEDULER_BLOB: &str = "gm1_child_scheduler_extension";
pub const IDEALIZED_POST_STEP_AUTHORITY_BLOB: &str = "gm1_idealized_isnd7_iinit1_source_authority";
pub const CHILD_POST_STEP_CHECKPOINT_FILE: &str = "checkpoint.cm1fcp";
pub const CHILD_POST_STEP_HISTORY_FILE: &str = "history.cm1fhs";
pub const CHILD_POST_STEP_MANIFEST_FILE: &str = "manifest.json";
pub const CHILD_POST_STEP_RECEIPT_FILE: &str = "receipt.json";
pub const CHILD_POST_STEP_RESTORE_CHECKPOINT_ARENA: &str = "checkpoint-arena.f32le";
pub const CHILD_POST_STEP_RESTORE_HISTORY_ARENA: &str = "history-arena.f32le";
pub const CHILD_POST_STEP_CLAIM_LIMIT: &str = "exact Rust-owned paired CM1FCP1 checkpoint and CM1FHS1 render-history persistence for one quiescent completed large step of the sealed real source-coordinate 50 m child; raw-bit carrier restore and checkpoint/history overlap identity are admitted, while live device binding, a completed child step, uninterrupted/resumed trajectory identity, forecast skill, tornado, and performance remain unclaimed until separately proved";

pub const IDEALIZED_POST_STEP_SOURCE_SCHEMA: &str =
    "gm1.idealized-isnd7-iinit1-source-authority/v1";
pub const IDEALIZED_POST_STEP_SOURCE_STATUS: &str =
    "ADMITTED_SOURCE_AUDITED_CM1_R21_1_ISND7_IINIT1";
pub const IDEALIZED_POST_STEP_CHECKPOINT_PROFILE: &str =
    "gm1-idealized-isnd7-iinit1-post-large-step/v1";
pub const IDEALIZED_POST_STEP_HISTORY_PROFILE: &str =
    "gm1-idealized-isnd7-iinit1-render-history/v1";
pub const IDEALIZED_POST_STEP_HISTORY_ABI: &str = "gm1.idealized-isnd7-iinit1-render-carriers/v1";
pub const IDEALIZED_POST_STEP_IDENTITY_SCHEMA: &str =
    "gm1.idealized-isnd7-iinit1-post-step-identity/v1";
pub const IDEALIZED_POST_STEP_MANIFEST_SCHEMA: &str =
    "gm1.idealized-isnd7-iinit1-post-step-pair-manifest/v1";
pub const IDEALIZED_POST_STEP_RECEIPT_SCHEMA: &str =
    "gm1.idealized-isnd7-iinit1-post-step-pair-receipt/v1";
pub const IDEALIZED_POST_STEP_RESTORE_MANIFEST_SCHEMA: &str =
    "gm1.idealized-isnd7-iinit1-post-step-restore/v1";
pub const IDEALIZED_POST_STEP_RESTORE_RECEIPT_SCHEMA: &str =
    "gm1.idealized-isnd7-iinit1-post-step-restore-receipt/v1";
pub const IDEALIZED_POST_STEP_CLAIM_LIMIT: &str = "exact Rust-owned paired CM1FCP1 checkpoint and CM1FHS1 render-history persistence for one completed large step of the source-audited CM1 r21.1 isnd=7 single-sounding, iinit=1 official warm-bubble, flat 50 m horizontal and uniform 200 m vertical idealized configuration; raw-bit carrier restore, independent artifact inspection, and checkpoint/history overlap identity are admitted, while native-CM1 bitwise trajectory identity, forecast skill, tornado, and production-real-child authority remain unclaimed";
pub const IDEALIZED_TERRAIN_SHA256: &str =
    "e1038a405281e3a4ba6a65048621871508e751e77f2289c359b50edd90be6914";
pub const IDEALIZED_ZF_SHA256: &str =
    "172dfc2d73a61c12ef59a4581c4567bacd638eca119a0dd02d83eb96614a387e";
pub const IDEALIZED_GEOPOTENTIAL_SHA256: &str =
    "5dc8fc4492cb5aca0d62264b695b3787f5c859f976630069cda5055f1256ff22";

pub const SEALED_CHILD_FULL_STATE_MANIFEST_SHA256: &str =
    "02d27058721b2d379f439bf4a4840c666eb879b9954518b518ffc47e0c999a7d";
pub const SEALED_CHILD_FIRST_STEP_SHA256: &str =
    "132f2f7f877abe9bbfac44900406ceae17c6de93d0bcfc9dacacefe8d993edeb";
pub const SEALED_CHILD_CHECKPOINT_ADMISSION_SHA256: &str =
    "23c0f09129084690c16cae85461d23f79f5645b430a64988242a6bee0119ba1a";
pub const SEALED_CHILD_FULL_STATE_RECEIPT_SHA256: &str =
    "792dc1ade597590e985b44c434a05c0c47751390bd5949520abb309527fdebb6";
pub const SEALED_CHILD_BASE_WIND_SHA256: &str =
    "adf38848863b1cb36c7918e5617d9f38b81443ddf965bc6358ef7dd60ffaf8dd";
pub const SEALED_CHILD_TERRAIN_SHA256: &str =
    "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91";
pub const SEALED_CHILD_ZF_SHA256: &str =
    "df5aef17965915406fe334600afe2f8f2e1ebb5d07665f71820eaac9bde4128f";
pub const SEALED_CHILD_GEOPOTENTIAL_SHA256: &str =
    "687bc3ebe34c0c07dc0c46418e8e6042f271e1bb1bcd119284b4a27ef80bca8c";
pub const SUPERSEDED_CHILD_MANIFEST_SHA256: &str =
    "148e40cfbc2105d31534d20e6c7b2937eef79a1128c425cd2fd06d44b71a3c3f";

pub const CHILD_POST_STEP_CHECKPOINT_ARRAY_COUNT: usize = 37;
pub const CHILD_POST_STEP_HISTORY_ARRAY_COUNT: usize = 13;
pub const CHILD_POST_STEP_CHECKPOINT_NBYTES: u64 = 976_696_992;
pub const CHILD_POST_STEP_HISTORY_NBYTES: u64 = 461_340_000;
pub const CHILD_NZ: usize = 75;
pub const CHILD_NY: usize = 300;
pub const CHILD_NX: usize = 300;
pub const CHILD_DT_S: f64 = 0.5;

const MAX_JSON_NBYTES: u64 = 1024 * 1024;
const MAX_BLOB_NBYTES: usize = 128 * 1024;
const READ_BLOCK_NBYTES: usize = 1024 * 1024;
const AUTHORITY_TARBALL_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

const CHECKPOINT_NAMES: [&str; CHILD_POST_STEP_CHECKPOINT_ARRAY_COUNT] = [
    "closure.tkea",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.h_diabatic",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_sr",
    "held.qcten",
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "pp3d",
    "ppi",
    "ppx",
    "prs",
    "q3d",
    "qa",
    "radbce",
    "radbcn",
    "radbcs",
    "radbcw",
    "rf",
    "rho",
    "rr",
    "th3d",
    "tha",
    "u3d",
    "ua",
    "v3d",
    "va",
    "w3d",
    "wa",
];

const HISTORY_NAMES: [&str; CHILD_POST_STEP_HISTORY_ARRAY_COUNT] = [
    "closure.tkea",
    "pp3d",
    "prs",
    "q3d",
    "rf",
    "rho",
    "terrain_msl",
    "th3d",
    "total_geopotential_zf",
    "u3d",
    "v3d",
    "w3d",
    "zf",
];

const OVERLAP_NAMES: [&str; 10] = [
    "closure.tkea",
    "pp3d",
    "prs",
    "q3d",
    "rf",
    "rho",
    "th3d",
    "u3d",
    "v3d",
    "w3d",
];

const CORE_SCALAR_NAMES: [&str; 17] = [
    "completed_large_steps",
    "completed_wsm6_bridge_calls",
    "completed_wsm6_component_calls",
    "completed_wsm6_transitions",
    "history_record",
    "large_step_boundary",
    "mass1_f64",
    "moist_buoyancy_applied",
    "moist_post_acoustic_ready",
    "moist_terms_ready",
    "next_stage",
    "post_stage_hook_active",
    "quiescent_capture",
    "radiation_history_initialized",
    "scalar_completed",
    "scalar_next_nrk",
    "solve1_ready",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChildPostStepAdmission {
    Production,
    IdealizedSingleSoundingWarmBubble,
    #[cfg(test)]
    IdealizedBoundedProof,
    BoundedProof,
}

impl ChildPostStepAdmission {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Production => "production-real-50m-child",
            Self::IdealizedSingleSoundingWarmBubble => {
                "idealized-isnd7-iinit1-single-sounding-warm-bubble"
            }
            #[cfg(test)]
            Self::IdealizedBoundedProof => "idealized-bounded-structural-proof",
            Self::BoundedProof => "bounded-structural-proof",
        }
    }

    const fn is_idealized(self) -> bool {
        match self {
            Self::IdealizedSingleSoundingWarmBubble => true,
            #[cfg(test)]
            Self::IdealizedBoundedProof => true,
            _ => false,
        }
    }

    const fn is_bounded_proof(self) -> bool {
        match self {
            Self::BoundedProof => true,
            #[cfg(test)]
            Self::IdealizedBoundedProof => true,
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChildPostStepConfig {
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
    pub admission: ChildPostStepAdmission,
}

impl ChildPostStepConfig {
    pub const fn production() -> Self {
        Self {
            nz: CHILD_NZ,
            ny: CHILD_NY,
            nx: CHILD_NX,
            admission: ChildPostStepAdmission::Production,
        }
    }

    pub const fn idealized_single_sounding_warm_bubble() -> Self {
        Self {
            nz: CHILD_NZ,
            ny: CHILD_NY,
            nx: CHILD_NX,
            admission: ChildPostStepAdmission::IdealizedSingleSoundingWarmBubble,
        }
    }

    #[cfg(test)]
    fn bounded_proof(nz: usize, ny: usize, nx: usize) -> Result<Self> {
        let value = Self {
            nz,
            ny,
            nx,
            admission: ChildPostStepAdmission::BoundedProof,
        };
        value.validate()?;
        Ok(value)
    }

    #[cfg(test)]
    fn idealized_bounded_proof(nz: usize, ny: usize, nx: usize) -> Result<Self> {
        let value = Self {
            nz,
            ny,
            nx,
            admission: ChildPostStepAdmission::IdealizedBoundedProof,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(self) -> Result<()> {
        if self.nz == 0 || self.ny == 0 || self.nx == 0 {
            return Err(invalid("child post-step domain contains a zero extent"));
        }
        if self.nz > CHILD_NZ || self.ny > CHILD_NY || self.nx > CHILD_NX {
            return Err(invalid(
                "child post-step domain exceeds the sealed 75x300x300 child",
            ));
        }
        if matches!(
            self.admission,
            ChildPostStepAdmission::Production
                | ChildPostStepAdmission::IdealizedSingleSoundingWarmBubble
        ) && self.nz == CHILD_NZ
            && self.ny == CHILD_NY
            && self.nx == CHILD_NX
        {
            return Ok(());
        }
        if !self.admission.is_bounded_proof() {
            return Err(invalid(
                "admitted child post-step profile requires 75x300x300",
            ));
        }
        Ok(())
    }
}

impl ChildPostStepConfig {
    const fn claim_limit(self) -> &'static str {
        if self.admission.is_idealized() {
            IDEALIZED_POST_STEP_CLAIM_LIMIT
        } else {
            CHILD_POST_STEP_CLAIM_LIMIT
        }
    }

    const fn checkpoint_profile(self) -> &'static str {
        if self.admission.is_idealized() {
            IDEALIZED_POST_STEP_CHECKPOINT_PROFILE
        } else {
            CHILD_POST_STEP_CHECKPOINT_PROFILE
        }
    }

    const fn history_profile(self) -> &'static str {
        if self.admission.is_idealized() {
            IDEALIZED_POST_STEP_HISTORY_PROFILE
        } else {
            CHILD_POST_STEP_HISTORY_PROFILE
        }
    }

    const fn history_abi(self) -> &'static str {
        if self.admission.is_idealized() {
            IDEALIZED_POST_STEP_HISTORY_ABI
        } else {
            CHILD_POST_STEP_HISTORY_ABI
        }
    }

    const fn identity_schema(self) -> &'static str {
        if self.admission.is_idealized() {
            IDEALIZED_POST_STEP_IDENTITY_SCHEMA
        } else {
            CHILD_POST_STEP_IDENTITY_SCHEMA
        }
    }

    const fn manifest_schema(self) -> &'static str {
        if self.admission.is_idealized() {
            IDEALIZED_POST_STEP_MANIFEST_SCHEMA
        } else {
            CHILD_POST_STEP_MANIFEST_SCHEMA
        }
    }

    const fn receipt_schema(self) -> &'static str {
        if self.admission.is_idealized() {
            IDEALIZED_POST_STEP_RECEIPT_SCHEMA
        } else {
            CHILD_POST_STEP_RECEIPT_SCHEMA
        }
    }

    const fn restore_manifest_schema(self) -> &'static str {
        if self.admission.is_idealized() {
            IDEALIZED_POST_STEP_RESTORE_MANIFEST_SCHEMA
        } else {
            CHILD_POST_STEP_RESTORE_MANIFEST_SCHEMA
        }
    }

    const fn restore_receipt_schema(self) -> &'static str {
        if self.admission.is_idealized() {
            IDEALIZED_POST_STEP_RESTORE_RECEIPT_SCHEMA
        } else {
            CHILD_POST_STEP_RESTORE_RECEIPT_SCHEMA
        }
    }

    const fn pair_status(self) -> &'static str {
        if self.admission.is_idealized() {
            "PASS_RUST_IDEALIZED_ISND7_IINIT1_PAIRED_POST_STEP_ENVELOPE"
        } else {
            "PASS_RUST_PAIRED_POST_STEP_ENVELOPE"
        }
    }

    const fn restore_status(self) -> &'static str {
        if self.admission.is_idealized() {
            "PASS_IDEALIZED_ISND7_IINIT1_RAW_BIT_RESTORE"
        } else {
            "PASS_RAW_BIT_RESTORE"
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildPostStepOperatorManifest {
    bytes: Vec<u8>,
    sha256: String,
    operator_count: usize,
}

impl ChildPostStepOperatorManifest {
    pub fn from_canonical_bytes(bytes: &[u8], expected_sha256: &str) -> Result<Self> {
        validate_sha256(expected_sha256, "expected child operator manifest SHA-256")?;
        if bytes.is_empty() || bytes.len() > MAX_BLOB_NBYTES {
            return Err(invalid(
                "child operator manifest length is outside its v1 cap",
            ));
        }
        let actual_sha256 = sha256::digest_hex(bytes);
        if actual_sha256 != expected_sha256 {
            return Err(invalid(
                "child operator manifest differs from its independent SHA-256",
            ));
        }
        let value = json::parse(bytes)
            .map_err(|error| invalid(format!("child operator manifest JSON: {error}")))?;
        if value.canonical_bytes() != bytes {
            return Err(invalid("child operator manifest is not canonical JSON"));
        }
        exact_keys(
            &value,
            &["model", "operators", "schema"],
            "child operator manifest",
        )?;
        fixed_string(
            &value,
            "schema",
            CHILD_POST_STEP_OPERATOR_SCHEMA,
            "child operator manifest",
        )?;
        fixed_string(&value, "model", "GM1", "child operator manifest")?;
        let operators = value
            .get("operators")
            .and_then(JsonValue::as_object)
            .map_err(|error| invalid(format!("child operator manifest.operators: {error}")))?;
        if operators.len() < GM1_OPERATOR_SOURCE_KEYS.len() || operators.len() > 128 {
            return Err(invalid(
                "child operator manifest has an unsupported operator count",
            ));
        }
        let names = operators
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>();
        for required in GM1_OPERATOR_SOURCE_KEYS {
            if !names.contains(required) {
                return Err(invalid(format!(
                    "child operator manifest omits required {required}"
                )));
            }
        }
        for (name, value) in operators {
            validate_extension_name(name)?;
            let hash = value
                .as_str()
                .map_err(|error| invalid(format!("child operator {name}: {error}")))?;
            validate_sha256(hash, &format!("child operator {name} SHA-256"))?;
        }
        Ok(Self {
            bytes: bytes.to_vec(),
            sha256: actual_sha256,
            operator_count: operators.len(),
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn operator_count(&self) -> usize {
        self.operator_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ChildFullStateSeal {
    directory: String,
    base_wind_sha256: String,
    manifest_sha256: String,
    first_step_sha256: String,
    checkpoint_admission_sha256: String,
    receipt_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IdealizedPostStepAuthority {
    bytes: Vec<u8>,
    sha256: String,
    rust_library_sha256: String,
    sounding_sha256: String,
    initial_mass_f64_bits: String,
}

impl IdealizedPostStepAuthority {
    pub fn from_canonical_bytes(bytes: &[u8], expected_sha256: &str) -> Result<Self> {
        validate_sha256(
            expected_sha256,
            "expected idealized post-step authority SHA-256",
        )?;
        if bytes.is_empty() || bytes.len() > MAX_BLOB_NBYTES {
            return Err(invalid(
                "idealized post-step authority length is outside its v1 cap",
            ));
        }
        let actual_sha256 = sha256::digest_hex(bytes);
        if actual_sha256 != expected_sha256 {
            return Err(invalid(
                "idealized post-step authority differs from its independent SHA-256",
            ));
        }
        let value = parse_canonical_object(bytes, "idealized post-step authority")?;
        let (rust_library_sha256, sounding_sha256, initial_mass_f64_bits) =
            validate_idealized_authority(&value)?;
        Ok(Self {
            bytes: bytes.to_vec(),
            sha256: actual_sha256,
            rust_library_sha256,
            sounding_sha256,
            initial_mass_f64_bits,
        })
    }

    pub fn from_file(path: impl AsRef<Path>, expected_sha256: &str) -> Result<Self> {
        let bytes = read_regular_bounded(
            path.as_ref(),
            MAX_BLOB_NBYTES as u64,
            "idealized post-step authority",
        )?;
        validate_sha256(
            expected_sha256,
            "expected idealized post-step authority SHA-256",
        )?;
        let actual_sha256 = sha256::digest_hex(&bytes);
        if actual_sha256 != expected_sha256 {
            return Err(invalid(
                "idealized post-step authority file differs from its independent SHA-256",
            ));
        }
        // Tracked evidence files use the repository's one-LF text convention.
        // The JSON object before that exact terminator must still be canonical;
        // CRLF, blank lines, and arbitrary trailing whitespace remain refused.
        let canonical = bytes.strip_suffix(b"\n").unwrap_or(&bytes);
        if canonical.ends_with(b"\n") || canonical.ends_with(b"\r") {
            return Err(invalid(
                "idealized post-step authority has a noncanonical terminator",
            ));
        }
        let value = parse_canonical_object(canonical, "idealized post-step authority")?;
        let (rust_library_sha256, sounding_sha256, initial_mass_f64_bits) =
            validate_idealized_authority(&value)?;
        Ok(Self {
            bytes,
            sha256: actual_sha256,
            rust_library_sha256,
            sounding_sha256,
            initial_mass_f64_bits,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PostStepSourceSeal {
    Production(ChildFullStateSeal),
    Idealized(IdealizedPostStepAuthority),
}

#[derive(Clone, Debug)]
struct ArenaRead {
    arrays: Vec<ForecastArray>,
    aggregate_sha256: String,
    carrier_sha256: BTreeMap<String, String>,
    nbytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildPostStepReceipt {
    pub checkpoint: ForecastReceipt,
    pub history: ForecastReceipt,
    pub manifest_sha256: String,
    pub receipt_sha256: String,
    pub checkpoint_raw_sha256: String,
    pub history_raw_sha256: String,
    pub operator_manifest_sha256: String,
    pub scheduler_extension_sha256: String,
    pub step: u64,
    config: ChildPostStepConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildPostStepRestoreReceipt {
    pub checkpoint_arena_sha256: String,
    pub history_arena_sha256: String,
    pub manifest_sha256: String,
    pub receipt_sha256: String,
    pub step: u64,
    config: ChildPostStepConfig,
}

impl ChildPostStepReceipt {
    pub fn canonical_json(&self) -> String {
        canonical_string(object(vec![
            (
                "checkpoint_artifact_sha256",
                string(&self.checkpoint.artifact_sha256),
            ),
            (
                "checkpoint_contract_sha256",
                string(&self.checkpoint.contract_sha256),
            ),
            ("checkpoint_raw_sha256", string(&self.checkpoint_raw_sha256)),
            ("claim_limit", string(self.config.claim_limit())),
            (
                "history_artifact_sha256",
                string(&self.history.artifact_sha256),
            ),
            (
                "history_contract_sha256",
                string(&self.history.contract_sha256),
            ),
            ("history_raw_sha256", string(&self.history_raw_sha256)),
            ("manifest_sha256", string(&self.manifest_sha256)),
            ("model", string("GM1")),
            (
                "operator_manifest_sha256",
                string(&self.operator_manifest_sha256),
            ),
            ("receipt_sha256", string(&self.receipt_sha256)),
            (
                "scheduler_extension_sha256",
                string(&self.scheduler_extension_sha256),
            ),
            ("schema", string(self.config.receipt_schema())),
            ("step", u64_number(self.step)),
        ]))
    }
}

impl ChildPostStepRestoreReceipt {
    pub fn canonical_json(&self) -> String {
        canonical_string(object(vec![
            (
                "checkpoint_arena_sha256",
                string(&self.checkpoint_arena_sha256),
            ),
            ("claim_limit", string(self.config.claim_limit())),
            ("history_arena_sha256", string(&self.history_arena_sha256)),
            ("manifest_sha256", string(&self.manifest_sha256)),
            ("model", string("GM1")),
            ("receipt_sha256", string(&self.receipt_sha256)),
            ("schema", string(self.config.restore_receipt_schema())),
            ("step", u64_number(self.step)),
        ]))
    }
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

/// Emit the exact production raw-arena layouts without reading numeric data.
pub fn child_post_step_layout_json(
    operator_manifest: &ChildPostStepOperatorManifest,
) -> Result<String> {
    let config = ChildPostStepConfig::production();
    let checkpoint = checkpoint_contract(config, operator_manifest)?;
    let history = history_contract(config, operator_manifest)?;
    let checkpoint_layout = layout_json(&checkpoint)?;
    let history_layout = layout_json(&history)?;
    let checkpoint_nbytes = contract_nbytes(&checkpoint)?;
    let history_nbytes = contract_nbytes(&history)?;
    if checkpoint_nbytes != CHILD_POST_STEP_CHECKPOINT_NBYTES
        || history_nbytes != CHILD_POST_STEP_HISTORY_NBYTES
    {
        return Err(invalid("production child post-step byte inventory drifted"));
    }
    Ok(canonical_string(object(vec![
        ("checkpoint", checkpoint_layout),
        (
            "checkpoint_admission_sha256",
            string(SEALED_CHILD_CHECKPOINT_ADMISSION_SHA256),
        ),
        (
            "child_full_state_manifest_sha256",
            string(SEALED_CHILD_FULL_STATE_MANIFEST_SHA256),
        ),
        ("claim_limit", string(CHILD_POST_STEP_CLAIM_LIMIT)),
        ("history", history_layout),
        ("model", string("GM1")),
        (
            "operator_manifest_sha256",
            string(operator_manifest.sha256()),
        ),
        ("schema", string("gm1.real-50m-child-post-step-layout/v1")),
    ])))
}

/// Emit the exact idealized raw-arena layouts and authority binding.
pub fn idealized_child_post_step_layout_json(
    operator_manifest: &ChildPostStepOperatorManifest,
    authority: &IdealizedPostStepAuthority,
) -> Result<String> {
    let config = ChildPostStepConfig::idealized_single_sounding_warm_bubble();
    let checkpoint = checkpoint_contract(config, operator_manifest)?;
    let history = history_contract(config, operator_manifest)?;
    let checkpoint_nbytes = contract_nbytes(&checkpoint)?;
    let history_nbytes = contract_nbytes(&history)?;
    if checkpoint_nbytes != CHILD_POST_STEP_CHECKPOINT_NBYTES
        || history_nbytes != CHILD_POST_STEP_HISTORY_NBYTES
    {
        return Err(invalid("idealized child post-step byte inventory drifted"));
    }
    Ok(canonical_string(object(vec![
        ("checkpoint", layout_json(&checkpoint)?),
        ("claim_limit", string(config.claim_limit())),
        ("history", layout_json(&history)?),
        (
            "idealized_source_authority_sha256",
            string(authority.sha256()),
        ),
        ("model", string("GM1")),
        (
            "operator_manifest_sha256",
            string(operator_manifest.sha256()),
        ),
        (
            "schema",
            string("gm1.idealized-isnd7-iinit1-post-step-layout/v1"),
        ),
    ])))
}

/// Atomically publish one synchronized post-large-step checkpoint/history pair.
#[allow(clippy::too_many_arguments)]
pub fn publish_child_post_step_new(
    child_full_state_directory: impl AsRef<Path>,
    checkpoint_raw_arena: impl AsRef<Path>,
    expected_checkpoint_raw_sha256: &str,
    history_raw_arena: impl AsRef<Path>,
    expected_history_raw_sha256: &str,
    operator_manifest: ChildPostStepOperatorManifest,
    scheduler_extension: &[u8],
    clock: ForecastClock,
    mass1_f64: f64,
    output_directory: impl AsRef<Path>,
) -> Result<ChildPostStepReceipt> {
    let source = PostStepSourceSeal::Production(validate_child_full_state(
        child_full_state_directory.as_ref(),
    )?);
    publish_child_post_step_config_new(
        source,
        checkpoint_raw_arena.as_ref(),
        expected_checkpoint_raw_sha256,
        history_raw_arena.as_ref(),
        expected_history_raw_sha256,
        operator_manifest,
        scheduler_extension,
        clock,
        mass1_f64,
        output_directory.as_ref(),
        ChildPostStepConfig::production(),
    )
}

/// Atomically publish one source-audited idealized completed-step pair.
#[allow(clippy::too_many_arguments)]
pub fn publish_idealized_child_post_step_new(
    authority: IdealizedPostStepAuthority,
    checkpoint_raw_arena: impl AsRef<Path>,
    expected_checkpoint_raw_sha256: &str,
    history_raw_arena: impl AsRef<Path>,
    expected_history_raw_sha256: &str,
    operator_manifest: ChildPostStepOperatorManifest,
    scheduler_extension: &[u8],
    clock: ForecastClock,
    mass1_f64: f64,
    output_directory: impl AsRef<Path>,
) -> Result<ChildPostStepReceipt> {
    if f64_bits_string(mass1_f64) != authority.initial_mass_f64_bits {
        return Err(invalid(
            "idealized post-step mass differs from its source authority",
        ));
    }
    publish_child_post_step_config_new(
        PostStepSourceSeal::Idealized(authority),
        checkpoint_raw_arena.as_ref(),
        expected_checkpoint_raw_sha256,
        history_raw_arena.as_ref(),
        expected_history_raw_sha256,
        operator_manifest,
        scheduler_extension,
        clock,
        mass1_f64,
        output_directory.as_ref(),
        ChildPostStepConfig::idealized_single_sounding_warm_bubble(),
    )
}

#[allow(clippy::too_many_arguments)]
fn publish_child_post_step_config_new(
    source: PostStepSourceSeal,
    checkpoint_raw_arena: &Path,
    expected_checkpoint_raw_sha256: &str,
    history_raw_arena: &Path,
    expected_history_raw_sha256: &str,
    operator_manifest: ChildPostStepOperatorManifest,
    scheduler_extension: &[u8],
    clock: ForecastClock,
    mass1_f64: f64,
    output_directory: &Path,
    config: ChildPostStepConfig,
) -> Result<ChildPostStepReceipt> {
    config.validate()?;
    validate_sha256(
        expected_checkpoint_raw_sha256,
        "expected child checkpoint raw SHA-256",
    )?;
    validate_sha256(
        expected_history_raw_sha256,
        "expected child history raw SHA-256",
    )?;
    validate_boundary(clock, mass1_f64)?;
    let scheduler_extension = validate_scheduler_extension(scheduler_extension)?;
    let scheduler_extension_sha256 = sha256::digest_hex(&scheduler_extension);
    if fs::symlink_metadata(output_directory).is_ok() {
        return Err(invalid(format!(
            "refusing to replace child post-step destination {}",
            output_directory.display()
        )));
    }
    let parent = output_directory
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid("child post-step output parent does not exist"));
    }

    let checkpoint_contract = checkpoint_contract(config, &operator_manifest)?;
    let history_contract = history_contract(config, &operator_manifest)?;
    let checkpoint = read_arena(
        checkpoint_raw_arena,
        &checkpoint_contract,
        expected_checkpoint_raw_sha256,
    )?;
    validate_checkpoint_semantics(&checkpoint.arrays)?;
    let history = read_arena(
        history_raw_arena,
        &history_contract,
        expected_history_raw_sha256,
    )?;
    validate_history_semantics(&history.arrays, config)?;
    validate_pair_identity(&checkpoint, &history, config)?;

    let identity = identity_json(
        config,
        &source,
        &operator_manifest,
        &scheduler_extension_sha256,
        expected_checkpoint_raw_sha256,
        expected_history_raw_sha256,
    )
    .canonical_bytes();
    let scalars = boundary_scalars(clock, mass1_f64)?;
    let checkpoint_frame = ForecastFrame::new(
        clock,
        checkpoint.arrays,
        scalars.clone(),
        component_blobs(
            config,
            &source,
            &identity,
            &operator_manifest,
            &scheduler_extension,
        ),
    );
    let history_frame = ForecastFrame::new(
        clock,
        history.arrays,
        scalars,
        component_blobs(
            config,
            &source,
            &identity,
            &operator_manifest,
            &scheduler_extension,
        ),
    );
    let staging = staging_path(output_directory)?;
    fs::create_dir(&staging)?;
    let result = (|| -> Result<ChildPostStepReceipt> {
        let checkpoint_receipt = write_forecast_checkpoint_new(
            &checkpoint_contract,
            &checkpoint_frame,
            staging.join(CHILD_POST_STEP_CHECKPOINT_FILE),
        )?;
        let history_receipt = write_forecast_history_new(
            &history_contract,
            &history_frame,
            staging.join(CHILD_POST_STEP_HISTORY_FILE),
        )?;
        let manifest = pair_manifest_json(
            config,
            &source,
            &operator_manifest,
            &scheduler_extension_sha256,
            expected_checkpoint_raw_sha256,
            expected_history_raw_sha256,
            &checkpoint_receipt,
            &history_receipt,
            mass1_f64,
        )
        .canonical_bytes();
        let manifest_sha256 = sha256::digest_hex(&manifest);
        write_new_synced(&staging.join(CHILD_POST_STEP_MANIFEST_FILE), &manifest)?;
        let receipt_value = pair_receipt_json(
            config,
            &manifest_sha256,
            &checkpoint_receipt,
            &history_receipt,
            expected_checkpoint_raw_sha256,
            expected_history_raw_sha256,
            operator_manifest.sha256(),
            &scheduler_extension_sha256,
        );
        let receipt = receipt_value.canonical_bytes();
        let receipt_sha256 = sha256::digest_hex(&receipt);
        write_new_synced(&staging.join(CHILD_POST_STEP_RECEIPT_FILE), &receipt)?;
        verify_pair_inventory(&staging)?;
        fs::rename(&staging, output_directory)?;
        sync_directory(parent)?;
        Ok(ChildPostStepReceipt {
            checkpoint: checkpoint_receipt,
            history: history_receipt,
            manifest_sha256,
            receipt_sha256,
            checkpoint_raw_sha256: expected_checkpoint_raw_sha256.to_owned(),
            history_raw_sha256: expected_history_raw_sha256.to_owned(),
            operator_manifest_sha256: operator_manifest.sha256().to_owned(),
            scheduler_extension_sha256,
            step: clock.step(),
            config,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

/// Inspect and recompute both embedded envelopes against independent file pins.
#[allow(clippy::too_many_arguments)]
pub fn inspect_child_post_step(
    child_full_state_directory: impl AsRef<Path>,
    pair_directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_receipt_sha256: &str,
    expected_checkpoint_artifact_sha256: &str,
    expected_history_artifact_sha256: &str,
    expected_operator_manifest_sha256: &str,
) -> Result<ChildPostStepReceipt> {
    let source = PostStepSourceSeal::Production(validate_child_full_state(
        child_full_state_directory.as_ref(),
    )?);
    inspect_child_post_step_config(
        source,
        pair_directory.as_ref(),
        expected_manifest_sha256,
        expected_receipt_sha256,
        expected_checkpoint_artifact_sha256,
        expected_history_artifact_sha256,
        expected_operator_manifest_sha256,
        ChildPostStepConfig::production(),
    )
}

/// Independently inspect and recompute one idealized pair against source pins.
#[allow(clippy::too_many_arguments)]
pub fn inspect_idealized_child_post_step(
    authority: IdealizedPostStepAuthority,
    pair_directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_receipt_sha256: &str,
    expected_checkpoint_artifact_sha256: &str,
    expected_history_artifact_sha256: &str,
    expected_operator_manifest_sha256: &str,
) -> Result<ChildPostStepReceipt> {
    inspect_child_post_step_config(
        PostStepSourceSeal::Idealized(authority),
        pair_directory.as_ref(),
        expected_manifest_sha256,
        expected_receipt_sha256,
        expected_checkpoint_artifact_sha256,
        expected_history_artifact_sha256,
        expected_operator_manifest_sha256,
        ChildPostStepConfig::idealized_single_sounding_warm_bubble(),
    )
}

#[allow(clippy::too_many_arguments)]
fn inspect_child_post_step_config(
    source: PostStepSourceSeal,
    pair_directory: &Path,
    expected_manifest_sha256: &str,
    expected_receipt_sha256: &str,
    expected_checkpoint_artifact_sha256: &str,
    expected_history_artifact_sha256: &str,
    expected_operator_manifest_sha256: &str,
    config: ChildPostStepConfig,
) -> Result<ChildPostStepReceipt> {
    for (value, label) in [
        (
            expected_manifest_sha256,
            "expected child post-step manifest SHA-256",
        ),
        (
            expected_receipt_sha256,
            "expected child post-step receipt SHA-256",
        ),
        (
            expected_checkpoint_artifact_sha256,
            "expected child checkpoint artifact SHA-256",
        ),
        (
            expected_history_artifact_sha256,
            "expected child history artifact SHA-256",
        ),
        (
            expected_operator_manifest_sha256,
            "expected child operator manifest SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    config.validate()?;
    verify_pair_inventory(pair_directory)?;
    let manifest_bytes = read_regular_bounded(
        &pair_directory.join(CHILD_POST_STEP_MANIFEST_FILE),
        MAX_JSON_NBYTES,
        "child post-step manifest",
    )?;
    let receipt_bytes = read_regular_bounded(
        &pair_directory.join(CHILD_POST_STEP_RECEIPT_FILE),
        MAX_JSON_NBYTES,
        "child post-step receipt",
    )?;
    if sha256::digest_hex(&manifest_bytes) != expected_manifest_sha256
        || sha256::digest_hex(&receipt_bytes) != expected_receipt_sha256
    {
        return Err(invalid(
            "child post-step manifest or receipt differs from its independent pin",
        ));
    }
    let manifest = parse_canonical_object(&manifest_bytes, "child post-step manifest")?;
    let receipt = parse_canonical_object(&receipt_bytes, "child post-step receipt")?;
    let parsed = parse_pair_manifest(&manifest, config)?;
    if !source_summary_matches(&parsed.source, &source) {
        return Err(invalid(
            "child post-step source full-state identity changed",
        ));
    }
    if parsed.operator_manifest_sha256 != expected_operator_manifest_sha256 {
        return Err(invalid(
            "child post-step operator manifest differs from its independent pin",
        ));
    }
    let checkpoint_path = pair_directory.join(CHILD_POST_STEP_CHECKPOINT_FILE);
    let history_path = pair_directory.join(CHILD_POST_STEP_HISTORY_FILE);
    let (embedded_checkpoint_contract, checkpoint) =
        read_forecast_checkpoint_embedded(&checkpoint_path, &parsed.checkpoint_contract_sha256)?;
    let (embedded_history_contract, history) =
        read_forecast_history_embedded(&history_path, &parsed.history_contract_sha256)?;
    if checkpoint.receipt().artifact_sha256 != expected_checkpoint_artifact_sha256
        || checkpoint.receipt().artifact_sha256 != parsed.checkpoint_artifact_sha256
        || history.receipt().artifact_sha256 != expected_history_artifact_sha256
        || history.receipt().artifact_sha256 != parsed.history_artifact_sha256
    {
        return Err(invalid(
            "child post-step forecast artifact identity changed",
        ));
    }
    let (operator_manifest, scheduler_extension, identity_bytes, embedded_authority) =
        validate_embedded_blobs(&checkpoint, expected_operator_manifest_sha256, config)?;
    validate_embedded_blobs_equal(
        &history,
        &operator_manifest,
        &scheduler_extension,
        &identity_bytes,
        embedded_authority.as_deref(),
        config,
    )?;
    if let PostStepSourceSeal::Idealized(authority) = &source {
        if embedded_authority.as_deref() != Some(authority.bytes()) {
            return Err(invalid(
                "embedded idealized authority differs from its independent source",
            ));
        }
    } else if embedded_authority.is_some() {
        return Err(invalid(
            "production child pair contains an idealized authority blob",
        ));
    }
    let expected_checkpoint_contract = checkpoint_contract(config, &operator_manifest)?;
    let expected_history_contract = history_contract(config, &operator_manifest)?;
    if embedded_checkpoint_contract != expected_checkpoint_contract
        || embedded_history_contract != expected_history_contract
    {
        return Err(invalid(
            "child post-step embedded contract differs from Rust-owned layout",
        ));
    }
    validate_restored_pair(
        &checkpoint,
        &history,
        config,
        &parsed,
        &source,
        &operator_manifest,
        &scheduler_extension,
        &identity_bytes,
    )?;
    validate_pair_receipt(
        config,
        &receipt,
        expected_manifest_sha256,
        checkpoint.receipt(),
        history.receipt(),
        &parsed,
    )?;
    Ok(ChildPostStepReceipt {
        checkpoint: checkpoint.receipt().clone(),
        history: history.receipt().clone(),
        manifest_sha256: expected_manifest_sha256.to_owned(),
        receipt_sha256: expected_receipt_sha256.to_owned(),
        checkpoint_raw_sha256: parsed.checkpoint_raw_sha256,
        history_raw_sha256: parsed.history_raw_sha256,
        operator_manifest_sha256: parsed.operator_manifest_sha256,
        scheduler_extension_sha256: parsed.scheduler_extension_sha256,
        step: checkpoint.frame().clock().step(),
        config,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParsedPairManifest {
    source: PostStepSourceSeal,
    operator_manifest_sha256: String,
    scheduler_extension_sha256: String,
    checkpoint_raw_sha256: String,
    history_raw_sha256: String,
    checkpoint_contract_sha256: String,
    checkpoint_artifact_sha256: String,
    history_contract_sha256: String,
    history_artifact_sha256: String,
    mass1_f64_bits: String,
    step: u64,
}

fn checkpoint_contract(
    config: ChildPostStepConfig,
    operator_manifest: &ChildPostStepOperatorManifest,
) -> Result<ForecastContract> {
    post_step_contract(
        config.checkpoint_profile(),
        config,
        operator_manifest,
        checkpoint_specs(config)?,
    )
}

fn history_contract(
    config: ChildPostStepConfig,
    operator_manifest: &ChildPostStepOperatorManifest,
) -> Result<ForecastContract> {
    post_step_contract(
        config.history_profile(),
        config,
        operator_manifest,
        history_specs(config)?,
    )
}

fn post_step_contract(
    profile: &str,
    config: ChildPostStepConfig,
    operator_manifest: &ChildPostStepOperatorManifest,
    arrays: Vec<ForecastArraySpec>,
) -> Result<ForecastContract> {
    ForecastContract::new(
        profile,
        sha256::digest_hex(&configuration_json(config).canonical_bytes()),
        sha256::digest_hex(
            &execution_contract_identity(config, operator_manifest).canonical_bytes(),
        ),
        arrays,
        scheduler_specs()?,
        component_specs(config)?,
    )
}

fn execution_contract_identity(
    config: ChildPostStepConfig,
    operator_manifest: &ChildPostStepOperatorManifest,
) -> JsonValue {
    if config.admission.is_idealized() {
        object(vec![
            (
                "idealized_source_schema",
                string(IDEALIZED_POST_STEP_SOURCE_SCHEMA),
            ),
            (
                "operator_manifest_sha256",
                string(operator_manifest.sha256()),
            ),
            (
                "schema",
                string("gm1.idealized-isnd7-iinit1-execution-identity/v1"),
            ),
        ])
    } else {
        object(vec![
            (
                "child_full_state_manifest_sha256",
                string(SEALED_CHILD_FULL_STATE_MANIFEST_SHA256),
            ),
            (
                "operator_manifest_sha256",
                string(operator_manifest.sha256()),
            ),
            ("schema", string("gm1.real-50m-child-execution-identity/v1")),
        ])
    }
}

fn checkpoint_specs(config: ChildPostStepConfig) -> Result<Vec<ForecastArraySpec>> {
    config.validate()?;
    let nz = config.nz;
    let ny = config.ny;
    let nx = config.nx;
    let volume = vec![nz, ny, nx];
    let uface = vec![nz, ny, nx + 1];
    let vface = vec![nz, ny + 1, nx];
    let zface = vec![nz + 1, ny, nx];
    let species = vec![6, nz, ny, nx];
    let surface = vec![ny, nx];
    let mut specs = Vec::with_capacity(CHECKPOINT_NAMES.len());
    for name in CHECKPOINT_NAMES {
        let (shape, meaning, units, stagger) = match name {
            "closure.tkea" => (
                zface.clone(),
                "prognostic Deardorff TKE A carrier",
                "m2 s-2",
                "z face",
            ),
            "component.effc" | "component.effi" | "component.effs" => (
                volume.clone(),
                "WSM6 persistent effective-size carrier",
                "m",
                "scalar",
            ),
            "component.h_diabatic" => (
                volume.clone(),
                "WSM6 persistent diabatic heating",
                "K s-1",
                "scalar",
            ),
            name if name.starts_with("component.mp_") => {
                let units = if name == "component.mp_sr" { "1" } else { "mm" };
                (
                    surface.clone(),
                    "WSM6 persistent surface carrier",
                    units,
                    "surface",
                )
            }
            name if name.starts_with("held.") => {
                let units = match name {
                    "held.qpten" => "s-1",
                    "held.qtten" => "K s-1",
                    _ => "kg kg-1 s-1",
                };
                (volume.clone(), "WSM6 held tendency", units, "scalar")
            }
            "qa" | "q3d" => (
                species.clone(),
                "packed six-species moisture carrier",
                "kg kg-1",
                "species,scalar",
            ),
            "ua" | "u3d" => (uface.clone(), "zonal velocity carrier", "m s-1", "x face"),
            "va" | "v3d" => (
                vface.clone(),
                "meridional velocity carrier",
                "m s-1",
                "y face",
            ),
            "wa" | "w3d" => (
                zface.clone(),
                "vertical velocity carrier",
                "m s-1",
                "z face",
            ),
            "rf" => (
                zface.clone(),
                "interface density factor",
                "model-specific",
                "z face",
            ),
            "radbcw" | "radbce" => (
                vec![nz + 2, ny + 6],
                "all-open radiation history",
                "varies",
                "west/east boundary plane",
            ),
            "radbcs" | "radbcn" => (
                vec![nz + 2, nx + 6],
                "all-open radiation history",
                "varies",
                "south/north boundary plane",
            ),
            "prs" => (volume.clone(), "diagnostic pressure", "Pa", "scalar"),
            "rho" => (volume.clone(), "diagnostic density", "kg m-3", "scalar"),
            "rr" => (volume.clone(), "reciprocal density", "m3 kg-1", "scalar"),
            "ppi" | "pp3d" | "ppx" => (volume.clone(), "Exner perturbation carrier", "1", "scalar"),
            "tha" | "th3d" => (volume.clone(), "theta perturbation carrier", "K", "scalar"),
            _ => return Err(invalid(format!("unknown child checkpoint carrier {name}"))),
        };
        specs.push(spec(name, meaning, units, stagger, shape)?);
    }
    validate_spec_names(&specs, &CHECKPOINT_NAMES, "child checkpoint")?;
    Ok(specs)
}

fn history_specs(config: ChildPostStepConfig) -> Result<Vec<ForecastArraySpec>> {
    config.validate()?;
    let nz = config.nz;
    let ny = config.ny;
    let nx = config.nx;
    let volume = vec![nz, ny, nx];
    let uface = vec![nz, ny, nx + 1];
    let vface = vec![nz, ny + 1, nx];
    let zface = vec![nz + 1, ny, nx];
    let species = vec![6, nz, ny, nx];
    let surface = vec![ny, nx];
    let mut specs = Vec::with_capacity(HISTORY_NAMES.len());
    for name in HISTORY_NAMES {
        let (shape, meaning, units, stagger) = match name {
            "closure.tkea" => (
                zface.clone(),
                "live Deardorff TKE A carrier",
                "m2 s-2",
                "z face",
            ),
            "pp3d" => (volume.clone(), "live Exner perturbation", "1", "scalar"),
            "prs" => (volume.clone(), "live diagnostic pressure", "Pa", "scalar"),
            "q3d" => (
                species.clone(),
                "live packed six-species moisture",
                "kg kg-1",
                "species,scalar",
            ),
            "rf" => (
                zface.clone(),
                "live interface density factor",
                "model-specific",
                "z face",
            ),
            "rho" => (
                volume.clone(),
                "live diagnostic density",
                "kg m-3",
                "scalar",
            ),
            "terrain_msl" => (
                surface.clone(),
                "sealed child terrain elevation",
                "m",
                "surface",
            ),
            "th3d" => (volume.clone(), "live theta perturbation", "K", "scalar"),
            "total_geopotential_zf" => (
                zface.clone(),
                "sealed total z-face geopotential",
                "m2 s-2",
                "z face",
            ),
            "u3d" => (uface.clone(), "live zonal velocity", "m s-1", "x face"),
            "v3d" => (vface.clone(), "live meridional velocity", "m s-1", "y face"),
            "w3d" => (zface.clone(), "live vertical velocity", "m s-1", "z face"),
            "zf" => (zface.clone(), "sealed child z-face height", "m", "z face"),
            _ => return Err(invalid(format!("unknown child history carrier {name}"))),
        };
        specs.push(spec(name, meaning, units, stagger, shape)?);
    }
    validate_spec_names(&specs, &HISTORY_NAMES, "child history")?;
    Ok(specs)
}

fn spec(
    name: &str,
    meaning: &str,
    units: &str,
    stagger: &str,
    shape: Vec<usize>,
) -> Result<ForecastArraySpec> {
    ForecastArraySpec::new(name, meaning, units, stagger, one_bounds(&shape)?, shape)
}

fn validate_spec_names<const N: usize>(
    specs: &[ForecastArraySpec],
    expected: &[&str; N],
    label: &str,
) -> Result<()> {
    let actual = specs
        .iter()
        .map(ForecastArraySpec::name)
        .collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected || specs.len() != N {
        return Err(invalid(format!("{label} carrier inventory drifted")));
    }
    Ok(())
}

fn scheduler_specs() -> Result<Vec<SchedulerScalarSpec>> {
    Ok(vec![
        SchedulerScalarSpec::new("completed_large_steps", SchedulerScalarType::U64)?,
        SchedulerScalarSpec::new("completed_wsm6_bridge_calls", SchedulerScalarType::U64)?,
        SchedulerScalarSpec::new("completed_wsm6_component_calls", SchedulerScalarType::U64)?,
        SchedulerScalarSpec::new("completed_wsm6_transitions", SchedulerScalarType::U64)?,
        SchedulerScalarSpec::new("history_record", SchedulerScalarType::U64)?,
        SchedulerScalarSpec::new("large_step_boundary", SchedulerScalarType::Bool)?,
        SchedulerScalarSpec::new("mass1_f64", SchedulerScalarType::F64)?,
        SchedulerScalarSpec::new("moist_buoyancy_applied", SchedulerScalarType::Bool)?,
        SchedulerScalarSpec::new("moist_post_acoustic_ready", SchedulerScalarType::Bool)?,
        SchedulerScalarSpec::new("moist_terms_ready", SchedulerScalarType::Bool)?,
        SchedulerScalarSpec::new("next_stage", SchedulerScalarType::I64)?,
        SchedulerScalarSpec::new("post_stage_hook_active", SchedulerScalarType::Bool)?,
        SchedulerScalarSpec::new("quiescent_capture", SchedulerScalarType::Bool)?,
        SchedulerScalarSpec::new("radiation_history_initialized", SchedulerScalarType::Bool)?,
        SchedulerScalarSpec::new("scalar_completed", SchedulerScalarType::Bool)?,
        SchedulerScalarSpec::new("scalar_next_nrk", SchedulerScalarType::I64)?,
        SchedulerScalarSpec::new("solve1_ready", SchedulerScalarType::Bool)?,
    ])
}

fn component_specs(config: ChildPostStepConfig) -> Result<Vec<ComponentBlobSpec>> {
    let mut specs = vec![
        ComponentBlobSpec::new(
            CHILD_POST_STEP_IDENTITY_BLOB,
            config.identity_schema(),
            MAX_BLOB_NBYTES,
        )?,
        ComponentBlobSpec::new(
            CHILD_POST_STEP_OPERATOR_BLOB,
            CHILD_POST_STEP_OPERATOR_SCHEMA,
            MAX_BLOB_NBYTES,
        )?,
        ComponentBlobSpec::new(
            CHILD_POST_STEP_SCHEDULER_BLOB,
            CHILD_POST_STEP_SCHEDULER_SCHEMA,
            MAX_BLOB_NBYTES,
        )?,
    ];
    if config.admission.is_idealized() {
        specs.push(ComponentBlobSpec::new(
            IDEALIZED_POST_STEP_AUTHORITY_BLOB,
            IDEALIZED_POST_STEP_SOURCE_SCHEMA,
            MAX_BLOB_NBYTES,
        )?);
    }
    Ok(specs)
}

fn component_blobs(
    config: ChildPostStepConfig,
    source: &PostStepSourceSeal,
    identity: &[u8],
    operator_manifest: &ChildPostStepOperatorManifest,
    scheduler_extension: &[u8],
) -> Vec<ComponentBlob> {
    let mut blobs = vec![
        ComponentBlob::new(
            CHILD_POST_STEP_IDENTITY_BLOB,
            config.identity_schema(),
            identity.to_vec(),
        ),
        ComponentBlob::new(
            CHILD_POST_STEP_OPERATOR_BLOB,
            CHILD_POST_STEP_OPERATOR_SCHEMA,
            operator_manifest.bytes().to_vec(),
        ),
        ComponentBlob::new(
            CHILD_POST_STEP_SCHEDULER_BLOB,
            CHILD_POST_STEP_SCHEDULER_SCHEMA,
            scheduler_extension.to_vec(),
        ),
    ];
    if let PostStepSourceSeal::Idealized(authority) = source {
        blobs.push(ComponentBlob::new(
            IDEALIZED_POST_STEP_AUTHORITY_BLOB,
            IDEALIZED_POST_STEP_SOURCE_SCHEMA,
            authority.bytes().to_vec(),
        ));
    }
    blobs
}

fn boundary_scalars(clock: ForecastClock, mass1_f64: f64) -> Result<Vec<SchedulerScalar>> {
    validate_boundary(clock, mass1_f64)?;
    let step = clock.step();
    Ok(vec![
        SchedulerScalar::new("completed_large_steps", SchedulerScalarValue::U64(step))?,
        SchedulerScalar::new(
            "completed_wsm6_bridge_calls",
            SchedulerScalarValue::U64(step),
        )?,
        SchedulerScalar::new(
            "completed_wsm6_component_calls",
            SchedulerScalarValue::U64(step),
        )?,
        SchedulerScalar::new(
            "completed_wsm6_transitions",
            SchedulerScalarValue::U64(step),
        )?,
        SchedulerScalar::new("history_record", SchedulerScalarValue::U64(step))?,
        SchedulerScalar::new("large_step_boundary", SchedulerScalarValue::Bool(true))?,
        SchedulerScalar::new("mass1_f64", SchedulerScalarValue::F64(mass1_f64))?,
        SchedulerScalar::new("moist_buoyancy_applied", SchedulerScalarValue::Bool(false))?,
        SchedulerScalar::new(
            "moist_post_acoustic_ready",
            SchedulerScalarValue::Bool(false),
        )?,
        SchedulerScalar::new("moist_terms_ready", SchedulerScalarValue::Bool(false))?,
        SchedulerScalar::new("next_stage", SchedulerScalarValue::I64(1))?,
        SchedulerScalar::new("post_stage_hook_active", SchedulerScalarValue::Bool(false))?,
        SchedulerScalar::new("quiescent_capture", SchedulerScalarValue::Bool(true))?,
        SchedulerScalar::new(
            "radiation_history_initialized",
            SchedulerScalarValue::Bool(true),
        )?,
        SchedulerScalar::new("scalar_completed", SchedulerScalarValue::Bool(false))?,
        SchedulerScalar::new("scalar_next_nrk", SchedulerScalarValue::I64(1))?,
        SchedulerScalar::new("solve1_ready", SchedulerScalarValue::Bool(false))?,
    ])
}

fn validate_boundary(clock: ForecastClock, mass1_f64: f64) -> Result<()> {
    if clock.dt_s().to_bits() != CHILD_DT_S.to_bits() {
        return Err(invalid(
            "child post-step dt differs from exact 0.5 s selection",
        ));
    }
    if clock.step() == 0 {
        return Err(invalid(
            "fresh child step zero is not a post-step checkpoint",
        ));
    }
    if clock.model_time_s().to_bits() != ((clock.step() as f64) * CHILD_DT_S).to_bits() {
        return Err(invalid(
            "child model time does not equal completed steps times dt",
        ));
    }
    if !mass1_f64.is_finite() || mass1_f64 <= 0.0 {
        return Err(invalid("child mass1 ledger must be finite and positive"));
    }
    Ok(())
}

fn read_arena(
    path: &Path,
    contract: &ForecastContract,
    expected_sha256: &str,
) -> Result<ArenaRead> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect child raw arena {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(
            "child raw arena must be a regular non-symlink file",
        ));
    }
    let expected_nbytes = contract_nbytes(contract)?;
    if metadata.len() != expected_nbytes {
        return Err(invalid(format!(
            "child raw arena has {} bytes; contract requires {expected_nbytes}",
            metadata.len()
        )));
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut aggregate = Sha256::default();
    let mut arrays = Vec::with_capacity(contract.arrays().len());
    let mut carrier_sha256 = BTreeMap::new();
    let mut block = vec![0u8; READ_BLOCK_NBYTES];
    for spec in contract.arrays() {
        let words = spec_words(spec)?;
        let mut values = Vec::with_capacity(words);
        let mut carrier = Sha256::default();
        let mut remaining = words;
        while remaining != 0 {
            let take_words = remaining.min(block.len() / 4);
            let bytes = &mut block[..take_words * 4];
            reader.read_exact(bytes)?;
            aggregate.update(bytes);
            carrier.update(bytes);
            for word in bytes.chunks_exact(4) {
                values.push(f32::from_bits(u32::from_le_bytes(
                    word.try_into().expect("four-byte word"),
                )));
            }
            remaining -= take_words;
        }
        carrier_sha256.insert(spec.name().to_owned(), sha256::hex(&carrier.finalize()));
        arrays.push(ForecastArray::new(spec.name(), values));
    }
    let aggregate_sha256 = sha256::hex(&aggregate.finalize());
    if aggregate_sha256 != expected_sha256 {
        return Err(invalid(format!(
            "child raw arena SHA-256 is {aggregate_sha256}; independently expected {expected_sha256}"
        )));
    }
    Ok(ArenaRead {
        arrays,
        aggregate_sha256,
        carrier_sha256,
        nbytes: expected_nbytes,
    })
}

fn validate_checkpoint_semantics(arrays: &[ForecastArray]) -> Result<()> {
    let map = array_map(arrays, CHILD_POST_STEP_CHECKPOINT_ARRAY_COUNT, "checkpoint")?;
    require_finite(arrays)?;
    for name in ["prs", "rho", "rr", "rf"] {
        require_positive(required_array(&map, name)?, name)?;
    }
    for name in [
        "qa",
        "q3d",
        "closure.tkea",
        "component.effc",
        "component.effi",
        "component.effs",
        "component.mp_graupelnc",
        "component.mp_graupelncv",
        "component.mp_rainnc",
        "component.mp_rainncv",
        "component.mp_snownc",
        "component.mp_snowncv",
        "component.mp_sr",
    ] {
        require_nonnegative(required_array(&map, name)?, name)?;
    }
    for (a, b) in [
        ("ua", "u3d"),
        ("va", "v3d"),
        ("wa", "w3d"),
        ("ppi", "pp3d"),
        ("tha", "th3d"),
        ("qa", "q3d"),
    ] {
        require_bits_equal(required_array(&map, a)?, required_array(&map, b)?, a, b)?;
    }
    Ok(())
}

fn validate_history_semantics(arrays: &[ForecastArray], config: ChildPostStepConfig) -> Result<()> {
    let map = array_map(arrays, CHILD_POST_STEP_HISTORY_ARRAY_COUNT, "history")?;
    require_finite(arrays)?;
    for name in ["prs", "rho", "rf"] {
        require_positive(required_array(&map, name)?, name)?;
    }
    for name in ["q3d", "closure.tkea"] {
        require_nonnegative(required_array(&map, name)?, name)?;
    }
    if config.admission == ChildPostStepAdmission::Production {
        for (name, expected) in [
            ("terrain_msl", SEALED_CHILD_TERRAIN_SHA256),
            ("zf", SEALED_CHILD_ZF_SHA256),
            ("total_geopotential_zf", SEALED_CHILD_GEOPOTENTIAL_SHA256),
        ] {
            let actual = values_sha256(required_array(&map, name)?);
            if actual != expected {
                return Err(invalid(format!(
                    "child history static carrier {name} differs from its sealed source"
                )));
            }
        }
    } else if config.admission == ChildPostStepAdmission::IdealizedSingleSoundingWarmBubble {
        for (name, expected) in [
            ("terrain_msl", IDEALIZED_TERRAIN_SHA256),
            ("zf", IDEALIZED_ZF_SHA256),
            ("total_geopotential_zf", IDEALIZED_GEOPOTENTIAL_SHA256),
        ] {
            let actual = values_sha256(required_array(&map, name)?);
            if actual != expected {
                return Err(invalid(format!(
                    "idealized history static carrier {name} differs from its source authority"
                )));
            }
        }
    }
    Ok(())
}

fn validate_pair_identity(
    checkpoint: &ArenaRead,
    history: &ArenaRead,
    config: ChildPostStepConfig,
) -> Result<()> {
    if checkpoint.aggregate_sha256.is_empty()
        || history.aggregate_sha256.is_empty()
        || checkpoint.nbytes != contract_nbytes_from_names(config, false)?
        || history.nbytes != contract_nbytes_from_names(config, true)?
    {
        return Err(invalid("child post-step arena accounting changed"));
    }
    for name in OVERLAP_NAMES {
        if checkpoint.carrier_sha256.get(name) != history.carrier_sha256.get(name) {
            return Err(invalid(format!(
                "child history carrier {name} differs raw-bit from the checkpoint"
            )));
        }
    }
    Ok(())
}

fn contract_nbytes_from_names(config: ChildPostStepConfig, history: bool) -> Result<u64> {
    let dummy = ChildPostStepOperatorManifest {
        bytes: Vec::new(),
        sha256: "00".repeat(32),
        operator_count: 0,
    };
    let contract = if history {
        history_contract(config, &dummy)?
    } else {
        checkpoint_contract(config, &dummy)?
    };
    contract_nbytes(&contract)
}

fn array_map<'a>(
    arrays: &'a [ForecastArray],
    expected_len: usize,
    label: &str,
) -> Result<BTreeMap<&'a str, &'a [f32]>> {
    let map = arrays
        .iter()
        .map(|array| (array.name(), array.values()))
        .collect::<BTreeMap<_, _>>();
    if arrays.len() != expected_len || map.len() != expected_len {
        return Err(invalid(format!("child {label} array inventory changed")));
    }
    Ok(map)
}

fn required_array<'a>(map: &'a BTreeMap<&str, &'a [f32]>, name: &str) -> Result<&'a [f32]> {
    map.get(name)
        .copied()
        .ok_or_else(|| invalid(format!("child carrier {name} is missing")))
}

fn require_finite(arrays: &[ForecastArray]) -> Result<()> {
    for array in arrays {
        if let Some(index) = array.values().iter().position(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "child carrier {} contains non-finite bits {:08x} at word {index}",
                array.name(),
                array.values()[index].to_bits()
            )));
        }
    }
    Ok(())
}

fn require_positive(values: &[f32], name: &str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| *value <= 0.0) {
        return Err(invalid(format!(
            "child carrier {name} is not positive at word {index}: {:08x}",
            values[index].to_bits()
        )));
    }
    Ok(())
}

fn require_nonnegative(values: &[f32], name: &str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| *value < 0.0) {
        return Err(invalid(format!(
            "child carrier {name} is negative at word {index}: {:08x}",
            values[index].to_bits()
        )));
    }
    Ok(())
}

fn require_bits_equal(left: &[f32], right: &[f32], a: &str, b: &str) -> Result<()> {
    if left.len() != right.len() {
        return Err(invalid(format!("child carriers {a}/{b} lengths differ")));
    }
    if let Some(index) = left
        .iter()
        .zip(right)
        .position(|(x, y)| x.to_bits() != y.to_bits())
    {
        return Err(invalid(format!(
            "child carriers {a}/{b} differ raw-bit at word {index}"
        )));
    }
    Ok(())
}

fn values_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::default();
    for value in values {
        digest.update(&value.to_bits().to_le_bytes());
    }
    sha256::hex(&digest.finalize())
}

fn arrays_sha256(arrays: &[ForecastArray]) -> String {
    let mut digest = Sha256::default();
    for array in arrays {
        for value in array.values() {
            digest.update(&value.to_bits().to_le_bytes());
        }
    }
    sha256::hex(&digest.finalize())
}

fn validate_child_full_state(directory: &Path) -> Result<ChildFullStateSeal> {
    let metadata = fs::symlink_metadata(directory).map_err(|error| {
        invalid(format!(
            "could not inspect child full-state {}: {error}",
            directory.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(
            "child full-state must be a regular non-symlink directory",
        ));
    }
    let expected = [
        "base-wind.f32le",
        CHILD_CHECKPOINT_ADMISSION_FILE,
        CHILD_FIRST_STEP_FILE,
        CHILD_FULL_STATE_MANIFEST_FILE,
        CHILD_FULL_STATE_RECEIPT_FILE,
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .to_str()
            .ok_or_else(|| invalid("child full-state contains a non-UTF-8 name"))?
            .to_owned();
        if !entry.file_type()?.is_file() {
            return Err(invalid("child full-state contains a non-regular member"));
        }
        actual.insert(name);
    }
    let actual_refs = actual.iter().map(String::as_str).collect::<BTreeSet<_>>();
    if actual_refs != expected {
        return Err(invalid("child full-state five-file inventory changed"));
    }
    if hash_regular(&directory.join(CHILD_FULL_STATE_MANIFEST_FILE))?
        == SUPERSEDED_CHILD_MANIFEST_SHA256
    {
        return Err(invalid(
            "superseded pre-clippy child full-state tuple is refused",
        ));
    }
    let seals = [
        ("base-wind.f32le", SEALED_CHILD_BASE_WIND_SHA256),
        (
            CHILD_FULL_STATE_MANIFEST_FILE,
            SEALED_CHILD_FULL_STATE_MANIFEST_SHA256,
        ),
        (CHILD_FIRST_STEP_FILE, SEALED_CHILD_FIRST_STEP_SHA256),
        (
            CHILD_CHECKPOINT_ADMISSION_FILE,
            SEALED_CHILD_CHECKPOINT_ADMISSION_SHA256,
        ),
        (
            CHILD_FULL_STATE_RECEIPT_FILE,
            SEALED_CHILD_FULL_STATE_RECEIPT_SHA256,
        ),
    ];
    for (name, expected) in seals {
        let path = directory.join(name);
        let actual = hash_regular(&path)?;
        if actual != expected {
            return Err(invalid(format!(
                "child full-state {name} differs from sealed SHA-256 {expected}"
            )));
        }
        if name.ends_with(".json") {
            let bytes = read_regular_bounded(&path, MAX_JSON_NBYTES, name)?;
            let _ = parse_canonical_object(&bytes, name)?;
        }
    }
    let directory_name = directory
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("child full-state directory requires a UTF-8 name"))?
        .to_owned();
    Ok(ChildFullStateSeal {
        directory: directory_name,
        base_wind_sha256: SEALED_CHILD_BASE_WIND_SHA256.to_owned(),
        manifest_sha256: SEALED_CHILD_FULL_STATE_MANIFEST_SHA256.to_owned(),
        first_step_sha256: SEALED_CHILD_FIRST_STEP_SHA256.to_owned(),
        checkpoint_admission_sha256: SEALED_CHILD_CHECKPOINT_ADMISSION_SHA256.to_owned(),
        receipt_sha256: SEALED_CHILD_FULL_STATE_RECEIPT_SHA256.to_owned(),
    })
}

#[cfg(test)]
fn test_source_seal(directory: &Path) -> ChildFullStateSeal {
    ChildFullStateSeal {
        directory: directory
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("bounded-proof-source")
            .to_owned(),
        base_wind_sha256: SEALED_CHILD_BASE_WIND_SHA256.to_owned(),
        manifest_sha256: SEALED_CHILD_FULL_STATE_MANIFEST_SHA256.to_owned(),
        first_step_sha256: SEALED_CHILD_FIRST_STEP_SHA256.to_owned(),
        checkpoint_admission_sha256: SEALED_CHILD_CHECKPOINT_ADMISSION_SHA256.to_owned(),
        receipt_sha256: SEALED_CHILD_FULL_STATE_RECEIPT_SHA256.to_owned(),
    }
}

fn source_json(source: &PostStepSourceSeal) -> JsonValue {
    match source {
        PostStepSourceSeal::Production(source) => object(vec![
            ("base_wind_sha256", string(&source.base_wind_sha256)),
            (
                "checkpoint_admission_sha256",
                string(&source.checkpoint_admission_sha256),
            ),
            ("directory", string(&source.directory)),
            ("first_step_sha256", string(&source.first_step_sha256)),
            ("manifest_sha256", string(&source.manifest_sha256)),
            ("receipt_sha256", string(&source.receipt_sha256)),
        ]),
        PostStepSourceSeal::Idealized(source) => object(vec![
            (
                "initial_mass_f64_bits",
                string(&source.initial_mass_f64_bits),
            ),
            ("rust_library_sha256", string(&source.rust_library_sha256)),
            ("schema", string(IDEALIZED_POST_STEP_SOURCE_SCHEMA)),
            ("sha256", string(&source.sha256)),
            ("sounding_sha256", string(&source.sounding_sha256)),
            ("status", string(IDEALIZED_POST_STEP_SOURCE_STATUS)),
        ]),
    }
}

fn parse_source(value: &JsonValue, config: ChildPostStepConfig) -> Result<PostStepSourceSeal> {
    if config.admission.is_idealized() {
        let label = "idealized post-step source authority";
        exact_keys(
            value,
            &[
                "initial_mass_f64_bits",
                "rust_library_sha256",
                "schema",
                "sha256",
                "sounding_sha256",
                "status",
            ],
            label,
        )?;
        fixed_string(value, "schema", IDEALIZED_POST_STEP_SOURCE_SCHEMA, label)?;
        fixed_string(value, "status", IDEALIZED_POST_STEP_SOURCE_STATUS, label)?;
        let initial_mass_f64_bits =
            required_string(value, "initial_mass_f64_bits", label)?.to_owned();
        let mass = parse_f64_bits(&initial_mass_f64_bits, "idealized initial mass")?;
        if mass <= 0.0 {
            return Err(invalid("idealized initial mass must be positive"));
        }
        return Ok(PostStepSourceSeal::Idealized(IdealizedPostStepAuthority {
            bytes: Vec::new(),
            sha256: required_sha(value, "sha256", label)?,
            rust_library_sha256: required_sha(value, "rust_library_sha256", label)?,
            sounding_sha256: required_sha(value, "sounding_sha256", label)?,
            initial_mass_f64_bits,
        }));
    }
    exact_keys(
        value,
        &[
            "base_wind_sha256",
            "checkpoint_admission_sha256",
            "directory",
            "first_step_sha256",
            "manifest_sha256",
            "receipt_sha256",
        ],
        "child source full-state",
    )?;
    let source = ChildFullStateSeal {
        directory: required_string(value, "directory", "child source full-state")?.to_owned(),
        base_wind_sha256: required_sha(value, "base_wind_sha256", "child source full-state")?,
        manifest_sha256: required_sha(value, "manifest_sha256", "child source full-state")?,
        first_step_sha256: required_sha(value, "first_step_sha256", "child source full-state")?,
        checkpoint_admission_sha256: required_sha(
            value,
            "checkpoint_admission_sha256",
            "child source full-state",
        )?,
        receipt_sha256: required_sha(value, "receipt_sha256", "child source full-state")?,
    };
    if source.manifest_sha256 == SUPERSEDED_CHILD_MANIFEST_SHA256 {
        return Err(invalid("superseded child source manifest is refused"));
    }
    Ok(PostStepSourceSeal::Production(source))
}

fn source_summary_matches(left: &PostStepSourceSeal, right: &PostStepSourceSeal) -> bool {
    source_json(left).canonical_bytes() == source_json(right).canonical_bytes()
}

fn source_manifest_key(config: ChildPostStepConfig) -> &'static str {
    if config.admission.is_idealized() {
        "source_idealized_authority"
    } else {
        "source_full_state"
    }
}

fn configuration_json(config: ChildPostStepConfig) -> JsonValue {
    if config.admission.is_idealized() {
        return object(vec![
            ("admission", string(config.admission.as_str())),
            (
                "domain_shape_zyx",
                usize_array(&[config.nz, config.ny, config.nx]),
            ),
            ("dt_f64_bits", f64_bits(CHILD_DT_S)),
            ("dx_m_f64_bits", f64_bits(50.0)),
            ("dy_m_f64_bits", f64_bits(50.0)),
            ("dz_m_f64_bits", f64_bits(200.0)),
            ("equation_set", number(2)),
            ("iinit", number(1)),
            ("irst", number(0)),
            ("isnd", number(7)),
            (
                "lateral_boundaries",
                JsonValue::Array([2usize, 2, 2, 2].map(number).to_vec()),
            ),
            ("microphysics", string("wsm6-mp6")),
            ("psolver", number(3)),
            ("terrain", string("flat-zero-terrain")),
            ("vertical_grid", string("uniform-200m")),
            ("ztop_m_f64_bits", f64_bits(15_000.0)),
        ]);
    }
    object(vec![
        ("admission", string(config.admission.as_str())),
        (
            "domain_shape_zyx",
            usize_array(&[config.nz, config.ny, config.nx]),
        ),
        ("dt_f64_bits", f64_bits(CHILD_DT_S)),
        ("dx_m_f64_bits", f64_bits(50.0)),
        ("dy_m_f64_bits", f64_bits(50.0)),
        ("equation_set", number(2)),
        (
            "grid_sha256",
            string("f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56"),
        ),
        ("imove", number(0)),
        (
            "lateral_boundaries",
            JsonValue::Array([2usize, 2, 2, 2].map(number).to_vec()),
        ),
        ("microphysics", string("wsm6-mp6")),
        ("psolver", number(3)),
        (
            "stretch_schema",
            string("gpuwm-cm1.stretch-z1-child-15km/v1"),
        ),
        ("terrain", JsonValue::Bool(true)),
        ("ztop_m_f64_bits", f64_bits(15_000.0)),
    ])
}

fn identity_json(
    config: ChildPostStepConfig,
    source: &PostStepSourceSeal,
    operator_manifest: &ChildPostStepOperatorManifest,
    scheduler_extension_sha256: &str,
    checkpoint_raw_sha256: &str,
    history_raw_sha256: &str,
) -> JsonValue {
    object(vec![
        (
            "authority",
            object(vec![
                ("cm1_version", string("r21.1")),
                ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
            ]),
        ),
        (
            "capture_phase",
            string("post-large-step-after-finish-cuda-gm1-large-step"),
        ),
        (
            "checkpoint_carrier_abi",
            string(CHILD_POST_STEP_CHECKPOINT_ABI),
        ),
        ("checkpoint_raw_sha256", string(checkpoint_raw_sha256)),
        ("claim_limit", string(config.claim_limit())),
        ("configuration", configuration_json(config)),
        ("history_carrier_abi", string(config.history_abi())),
        ("history_raw_sha256", string(history_raw_sha256)),
        ("model", string("GM1")),
        (
            "operator_manifest_sha256",
            string(operator_manifest.sha256()),
        ),
        (
            "scheduler_extension_sha256",
            string(scheduler_extension_sha256),
        ),
        ("schema", string(config.identity_schema())),
        (source_manifest_key(config), source_json(source)),
    ])
}

#[allow(clippy::too_many_arguments)]
fn pair_manifest_json(
    config: ChildPostStepConfig,
    source: &PostStepSourceSeal,
    operator_manifest: &ChildPostStepOperatorManifest,
    scheduler_extension_sha256: &str,
    checkpoint_raw_sha256: &str,
    history_raw_sha256: &str,
    checkpoint: &ForecastReceipt,
    history: &ForecastReceipt,
    mass1_f64: f64,
) -> JsonValue {
    object(vec![
        (
            "checkpoint",
            forecast_json(
                checkpoint,
                checkpoint_raw_sha256,
                CHILD_POST_STEP_CHECKPOINT_FILE,
            ),
        ),
        ("claim_limit", string(config.claim_limit())),
        ("configuration", configuration_json(config)),
        (
            "history",
            forecast_json(history, history_raw_sha256, CHILD_POST_STEP_HISTORY_FILE),
        ),
        ("mass1_f64_bits", f64_bits(mass1_f64)),
        ("model", string("GM1")),
        ("operator_count", number(operator_manifest.operator_count())),
        (
            "operator_manifest_sha256",
            string(operator_manifest.sha256()),
        ),
        (
            "scheduler_extension_sha256",
            string(scheduler_extension_sha256),
        ),
        ("schema", string(config.manifest_schema())),
        (source_manifest_key(config), source_json(source)),
        ("status", string(config.pair_status())),
        ("step", u64_number(checkpoint.step)),
    ])
}

fn forecast_json(receipt: &ForecastReceipt, raw_sha256: &str, file: &str) -> JsonValue {
    object(vec![
        ("artifact_nbytes", number(receipt.artifact_nbytes)),
        ("artifact_sha256", string(&receipt.artifact_sha256)),
        ("contract_sha256", string(&receipt.contract_sha256)),
        ("file", string(file)),
        ("payload_sha256", string(&receipt.payload_sha256)),
        ("raw_arena_sha256", string(raw_sha256)),
    ])
}

#[allow(clippy::too_many_arguments)]
fn pair_receipt_json(
    config: ChildPostStepConfig,
    manifest_sha256: &str,
    checkpoint: &ForecastReceipt,
    history: &ForecastReceipt,
    checkpoint_raw_sha256: &str,
    history_raw_sha256: &str,
    operator_manifest_sha256: &str,
    scheduler_extension_sha256: &str,
) -> JsonValue {
    object(vec![
        (
            "checkpoint_artifact_sha256",
            string(&checkpoint.artifact_sha256),
        ),
        ("checkpoint_raw_sha256", string(checkpoint_raw_sha256)),
        ("claim_limit", string(config.claim_limit())),
        ("history_artifact_sha256", string(&history.artifact_sha256)),
        ("history_raw_sha256", string(history_raw_sha256)),
        ("manifest_sha256", string(manifest_sha256)),
        ("model", string("GM1")),
        ("operator_manifest_sha256", string(operator_manifest_sha256)),
        (
            "scheduler_extension_sha256",
            string(scheduler_extension_sha256),
        ),
        ("schema", string(config.receipt_schema())),
        ("status", string(config.pair_status())),
        ("step", u64_number(checkpoint.step)),
    ])
}

fn parse_pair_manifest(
    value: &JsonValue,
    config: ChildPostStepConfig,
) -> Result<ParsedPairManifest> {
    let source_key = source_manifest_key(config);
    exact_keys(
        value,
        &[
            "checkpoint",
            "claim_limit",
            "configuration",
            "history",
            "mass1_f64_bits",
            "model",
            "operator_count",
            "operator_manifest_sha256",
            "scheduler_extension_sha256",
            "schema",
            source_key,
            "status",
            "step",
        ],
        "child post-step manifest",
    )?;
    fixed_string(
        value,
        "schema",
        config.manifest_schema(),
        "child post-step manifest",
    )?;
    fixed_string(value, "model", "GM1", "child post-step manifest")?;
    fixed_string(
        value,
        "status",
        config.pair_status(),
        "child post-step manifest",
    )?;
    let configuration = value
        .get("configuration")
        .map_err(|error| invalid(format!("child post-step configuration: {error}")))?;
    if configuration.canonical_bytes() != configuration_json(config).canonical_bytes() {
        return Err(invalid("child post-step configuration changed"));
    }
    let checkpoint = parse_forecast_ref(
        value
            .get("checkpoint")
            .map_err(|error| invalid(format!("child post-step checkpoint: {error}")))?,
        CHILD_POST_STEP_CHECKPOINT_FILE,
        "checkpoint",
    )?;
    let history = parse_forecast_ref(
        value
            .get("history")
            .map_err(|error| invalid(format!("child post-step history: {error}")))?,
        CHILD_POST_STEP_HISTORY_FILE,
        "history",
    )?;
    let mass1_f64_bits =
        required_string(value, "mass1_f64_bits", "child post-step manifest")?.to_owned();
    let mass1 = parse_f64_bits(&mass1_f64_bits, "child mass1")?;
    if mass1 <= 0.0 {
        return Err(invalid("child mass1 ledger must be positive"));
    }
    let step = required_u64(value, "step", "child post-step manifest")?;
    if step == 0 {
        return Err(invalid(
            "child post-step manifest cannot represent step zero",
        ));
    }
    Ok(ParsedPairManifest {
        source: parse_source(
            value
                .get(source_key)
                .map_err(|error| invalid(format!("child post-step source: {error}")))?,
            config,
        )?,
        operator_manifest_sha256: required_sha(
            value,
            "operator_manifest_sha256",
            "child post-step manifest",
        )?,
        scheduler_extension_sha256: required_sha(
            value,
            "scheduler_extension_sha256",
            "child post-step manifest",
        )?,
        checkpoint_raw_sha256: checkpoint.2,
        history_raw_sha256: history.2,
        checkpoint_contract_sha256: checkpoint.0,
        checkpoint_artifact_sha256: checkpoint.1,
        history_contract_sha256: history.0,
        history_artifact_sha256: history.1,
        mass1_f64_bits,
        step,
    })
}

fn parse_forecast_ref(
    value: &JsonValue,
    expected_file: &str,
    label: &str,
) -> Result<(String, String, String)> {
    exact_keys(
        value,
        &[
            "artifact_nbytes",
            "artifact_sha256",
            "contract_sha256",
            "file",
            "payload_sha256",
            "raw_arena_sha256",
        ],
        label,
    )?;
    fixed_string(value, "file", expected_file, label)?;
    let _ = required_u64(value, "artifact_nbytes", label)?;
    let _ = required_sha(value, "payload_sha256", label)?;
    Ok((
        required_sha(value, "contract_sha256", label)?,
        required_sha(value, "artifact_sha256", label)?,
        required_sha(value, "raw_arena_sha256", label)?,
    ))
}

fn validate_pair_receipt(
    config: ChildPostStepConfig,
    value: &JsonValue,
    manifest_sha256: &str,
    checkpoint: &ForecastReceipt,
    history: &ForecastReceipt,
    parsed: &ParsedPairManifest,
) -> Result<()> {
    let expected = pair_receipt_json(
        config,
        manifest_sha256,
        checkpoint,
        history,
        &parsed.checkpoint_raw_sha256,
        &parsed.history_raw_sha256,
        &parsed.operator_manifest_sha256,
        &parsed.scheduler_extension_sha256,
    );
    if value.canonical_bytes() != expected.canonical_bytes() {
        return Err(invalid("child post-step receipt semantics changed"));
    }
    Ok(())
}

type ValidatedEmbeddedBlobs = (
    ChildPostStepOperatorManifest,
    Vec<u8>,
    Vec<u8>,
    Option<Vec<u8>>,
);

fn validate_embedded_blobs(
    forecast: &RestoredForecast,
    expected_operator_sha256: &str,
    config: ChildPostStepConfig,
) -> Result<ValidatedEmbeddedBlobs> {
    let blobs = forecast
        .frame()
        .component_blobs()
        .iter()
        .map(|blob| (blob.name(), blob))
        .collect::<BTreeMap<_, _>>();
    let expected_len = if config.admission.is_idealized() {
        4
    } else {
        3
    };
    if blobs.len() != expected_len {
        return Err(invalid("child post-step component blob inventory changed"));
    }
    let identity = required_blob(
        &blobs,
        CHILD_POST_STEP_IDENTITY_BLOB,
        config.identity_schema(),
    )?
    .bytes()
    .to_vec();
    let operator_bytes = required_blob(
        &blobs,
        CHILD_POST_STEP_OPERATOR_BLOB,
        CHILD_POST_STEP_OPERATOR_SCHEMA,
    )?
    .bytes();
    let operator = ChildPostStepOperatorManifest::from_canonical_bytes(
        operator_bytes,
        expected_operator_sha256,
    )?;
    let scheduler = required_blob(
        &blobs,
        CHILD_POST_STEP_SCHEDULER_BLOB,
        CHILD_POST_STEP_SCHEDULER_SCHEMA,
    )?
    .bytes()
    .to_vec();
    let scheduler = validate_scheduler_extension(&scheduler)?;
    let authority = if config.admission.is_idealized() {
        Some(
            required_blob(
                &blobs,
                IDEALIZED_POST_STEP_AUTHORITY_BLOB,
                IDEALIZED_POST_STEP_SOURCE_SCHEMA,
            )?
            .bytes()
            .to_vec(),
        )
    } else {
        None
    };
    Ok((operator, scheduler, identity, authority))
}

fn validate_embedded_blobs_equal(
    history: &RestoredForecast,
    operator: &ChildPostStepOperatorManifest,
    scheduler: &[u8],
    identity: &[u8],
    authority: Option<&[u8]>,
    config: ChildPostStepConfig,
) -> Result<()> {
    let (history_operator, history_scheduler, history_identity, history_authority) =
        validate_embedded_blobs(history, operator.sha256(), config)?;
    if history_operator.bytes() != operator.bytes()
        || history_scheduler != scheduler
        || history_identity != identity
        || history_authority.as_deref() != authority
    {
        return Err(invalid(
            "child checkpoint/history embedded identities differ",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_restored_pair(
    checkpoint: &RestoredForecast,
    history: &RestoredForecast,
    config: ChildPostStepConfig,
    parsed: &ParsedPairManifest,
    source: &PostStepSourceSeal,
    operator: &ChildPostStepOperatorManifest,
    scheduler_extension: &[u8],
    identity_bytes: &[u8],
) -> Result<()> {
    if checkpoint.frame().clock() != history.frame().clock()
        || checkpoint.frame().clock().step() != parsed.step
    {
        return Err(invalid("child checkpoint/history clocks differ"));
    }
    let mass1 = parse_f64_bits(&parsed.mass1_f64_bits, "child mass1")?;
    validate_boundary(checkpoint.frame().clock(), mass1)?;
    validate_boundary_scalars(checkpoint.frame(), mass1)?;
    validate_boundary_scalars(history.frame(), mass1)?;
    validate_checkpoint_semantics(checkpoint.frame().arrays())?;
    validate_history_semantics(history.frame().arrays(), config)?;
    let checkpoint_read = restored_arena(checkpoint.frame().arrays());
    let history_read = restored_arena(history.frame().arrays());
    if checkpoint_read.aggregate_sha256 != parsed.checkpoint_raw_sha256
        || history_read.aggregate_sha256 != parsed.history_raw_sha256
    {
        return Err(invalid(
            "restored child arena differs from embedded raw hash",
        ));
    }
    validate_pair_identity(&checkpoint_read, &history_read, config)?;
    let scheduler_sha256 = sha256::digest_hex(scheduler_extension);
    if scheduler_sha256 != parsed.scheduler_extension_sha256 {
        return Err(invalid("child scheduler extension differs from manifest"));
    }
    let expected_identity = identity_json(
        config,
        source,
        operator,
        &scheduler_sha256,
        &parsed.checkpoint_raw_sha256,
        &parsed.history_raw_sha256,
    )
    .canonical_bytes();
    if identity_bytes != expected_identity {
        return Err(invalid(
            "child post-step embedded identity semantics changed",
        ));
    }
    Ok(())
}

fn restored_arena(arrays: &[ForecastArray]) -> ArenaRead {
    let mut carrier_sha256 = BTreeMap::new();
    let mut nbytes = 0u64;
    for array in arrays {
        carrier_sha256.insert(array.name().to_owned(), values_sha256(array.values()));
        nbytes += (array.values().len() as u64) * 4;
    }
    ArenaRead {
        arrays: arrays.to_vec(),
        aggregate_sha256: arrays_sha256(arrays),
        carrier_sha256,
        nbytes,
    }
}

fn validate_boundary_scalars(frame: &ForecastFrame, mass1_f64: f64) -> Result<()> {
    let expected = boundary_scalars(frame.clock(), mass1_f64)?;
    let actual = frame
        .scheduler_scalars()
        .iter()
        .map(|value| (value.name(), value.value()))
        .collect::<BTreeMap<_, _>>();
    let expected = expected
        .iter()
        .map(|value| (value.name(), value.value()))
        .collect::<BTreeMap<_, _>>();
    if actual != expected
        || actual.keys().copied().collect::<BTreeSet<_>>()
            != CORE_SCALAR_NAMES.into_iter().collect::<BTreeSet<_>>()
    {
        return Err(invalid(
            "child post-step scheduler scalars differ from safe cadence",
        ));
    }
    Ok(())
}

/// Restore exact checkpoint and history arenas into one new atomic directory.
#[allow(clippy::too_many_arguments)]
pub fn restore_child_post_step_new(
    child_full_state_directory: impl AsRef<Path>,
    pair_directory: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_receipt_sha256: &str,
    expected_checkpoint_artifact_sha256: &str,
    expected_history_artifact_sha256: &str,
    expected_operator_manifest_sha256: &str,
) -> Result<ChildPostStepRestoreReceipt> {
    let source = PostStepSourceSeal::Production(validate_child_full_state(
        child_full_state_directory.as_ref(),
    )?);
    restore_child_post_step_config_new(
        source,
        pair_directory.as_ref(),
        output_directory.as_ref(),
        expected_manifest_sha256,
        expected_receipt_sha256,
        expected_checkpoint_artifact_sha256,
        expected_history_artifact_sha256,
        expected_operator_manifest_sha256,
        ChildPostStepConfig::production(),
    )
}

/// Restore exact idealized checkpoint and history arenas into a new directory.
#[allow(clippy::too_many_arguments)]
pub fn restore_idealized_child_post_step_new(
    authority: IdealizedPostStepAuthority,
    pair_directory: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_receipt_sha256: &str,
    expected_checkpoint_artifact_sha256: &str,
    expected_history_artifact_sha256: &str,
    expected_operator_manifest_sha256: &str,
) -> Result<ChildPostStepRestoreReceipt> {
    restore_child_post_step_config_new(
        PostStepSourceSeal::Idealized(authority),
        pair_directory.as_ref(),
        output_directory.as_ref(),
        expected_manifest_sha256,
        expected_receipt_sha256,
        expected_checkpoint_artifact_sha256,
        expected_history_artifact_sha256,
        expected_operator_manifest_sha256,
        ChildPostStepConfig::idealized_single_sounding_warm_bubble(),
    )
}

#[allow(clippy::too_many_arguments)]
fn restore_child_post_step_config_new(
    source: PostStepSourceSeal,
    pair_directory: &Path,
    output_directory: &Path,
    expected_manifest_sha256: &str,
    expected_receipt_sha256: &str,
    expected_checkpoint_artifact_sha256: &str,
    expected_history_artifact_sha256: &str,
    expected_operator_manifest_sha256: &str,
    config: ChildPostStepConfig,
) -> Result<ChildPostStepRestoreReceipt> {
    let inspected = inspect_child_post_step_config(
        source,
        pair_directory,
        expected_manifest_sha256,
        expected_receipt_sha256,
        expected_checkpoint_artifact_sha256,
        expected_history_artifact_sha256,
        expected_operator_manifest_sha256,
        config,
    )?;
    if fs::symlink_metadata(output_directory).is_ok() {
        return Err(invalid(format!(
            "refusing to replace child post-step restore destination {}",
            output_directory.display()
        )));
    }
    let parent = output_directory
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid("child restore output parent does not exist"));
    }
    let manifest_bytes = read_regular_bounded(
        &pair_directory.join(CHILD_POST_STEP_MANIFEST_FILE),
        MAX_JSON_NBYTES,
        "child post-step manifest",
    )?;
    let manifest = parse_canonical_object(&manifest_bytes, "child post-step manifest")?;
    let parsed = parse_pair_manifest(&manifest, config)?;
    let (_, checkpoint) = read_forecast_checkpoint_embedded(
        pair_directory.join(CHILD_POST_STEP_CHECKPOINT_FILE),
        &parsed.checkpoint_contract_sha256,
    )?;
    let (_, history) = read_forecast_history_embedded(
        pair_directory.join(CHILD_POST_STEP_HISTORY_FILE),
        &parsed.history_contract_sha256,
    )?;
    let staging = staging_path(output_directory)?;
    fs::create_dir(&staging)?;
    let result = (|| -> Result<ChildPostStepRestoreReceipt> {
        let checkpoint_arena = staging.join(CHILD_POST_STEP_RESTORE_CHECKPOINT_ARENA);
        let history_arena = staging.join(CHILD_POST_STEP_RESTORE_HISTORY_ARENA);
        write_arrays_new(&checkpoint_arena, checkpoint.frame().arrays())?;
        write_arrays_new(&history_arena, history.frame().arrays())?;
        let checkpoint_arena_sha256 = hash_regular(&checkpoint_arena)?;
        let history_arena_sha256 = hash_regular(&history_arena)?;
        if checkpoint_arena_sha256 != inspected.checkpoint_raw_sha256
            || history_arena_sha256 != inspected.history_raw_sha256
        {
            return Err(invalid("child restored arena changed raw bits"));
        }
        let restore_manifest = restore_manifest_json(
            config,
            &inspected,
            checkpoint.frame().arrays(),
            history.frame().arrays(),
            &checkpoint_arena_sha256,
            &history_arena_sha256,
        )
        .canonical_bytes();
        let restore_manifest_sha256 = sha256::digest_hex(&restore_manifest);
        write_new_synced(
            &staging.join(CHILD_POST_STEP_MANIFEST_FILE),
            &restore_manifest,
        )?;
        let restore_receipt_value = object(vec![
            ("checkpoint_arena_sha256", string(&checkpoint_arena_sha256)),
            ("claim_limit", string(config.claim_limit())),
            ("history_arena_sha256", string(&history_arena_sha256)),
            ("manifest_sha256", string(&restore_manifest_sha256)),
            ("model", string("GM1")),
            ("schema", string(config.restore_receipt_schema())),
            (
                "source_pair_manifest_sha256",
                string(expected_manifest_sha256),
            ),
            ("status", string(config.restore_status())),
            ("step", u64_number(inspected.step)),
        ]);
        let restore_receipt = restore_receipt_value.canonical_bytes();
        let restore_receipt_sha256 = sha256::digest_hex(&restore_receipt);
        write_new_synced(
            &staging.join(CHILD_POST_STEP_RECEIPT_FILE),
            &restore_receipt,
        )?;
        verify_restore_inventory(&staging)?;
        fs::rename(&staging, output_directory)?;
        sync_directory(parent)?;
        Ok(ChildPostStepRestoreReceipt {
            checkpoint_arena_sha256,
            history_arena_sha256,
            manifest_sha256: restore_manifest_sha256,
            receipt_sha256: restore_receipt_sha256,
            step: inspected.step,
            config,
        })
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn restore_manifest_json(
    config: ChildPostStepConfig,
    inspected: &ChildPostStepReceipt,
    checkpoint: &[ForecastArray],
    history: &[ForecastArray],
    checkpoint_sha256: &str,
    history_sha256: &str,
) -> JsonValue {
    object(vec![
        (
            "checkpoint",
            restore_arena_json(
                CHILD_POST_STEP_RESTORE_CHECKPOINT_ARENA,
                checkpoint,
                checkpoint_sha256,
            ),
        ),
        (
            "checkpoint_artifact_sha256",
            string(&inspected.checkpoint.artifact_sha256),
        ),
        ("claim_limit", string(config.claim_limit())),
        (
            "history",
            restore_arena_json(
                CHILD_POST_STEP_RESTORE_HISTORY_ARENA,
                history,
                history_sha256,
            ),
        ),
        (
            "history_artifact_sha256",
            string(&inspected.history.artifact_sha256),
        ),
        ("model", string("GM1")),
        ("schema", string(config.restore_manifest_schema())),
        (
            "source_pair_manifest_sha256",
            string(&inspected.manifest_sha256),
        ),
        ("step", u64_number(inspected.step)),
    ])
}

fn restore_arena_json(file: &str, arrays: &[ForecastArray], sha256_value: &str) -> JsonValue {
    let mut offset = 0u64;
    let carriers = arrays
        .iter()
        .map(|array| {
            let nbytes = (array.values().len() as u64) * 4;
            let value = object(vec![
                ("name", string(array.name())),
                ("nbytes", number(nbytes)),
                ("offset_nbytes", number(offset)),
                ("sha256", string(values_sha256(array.values()))),
            ]);
            offset += nbytes;
            value
        })
        .collect();
    object(vec![
        ("carriers", JsonValue::Array(carriers)),
        ("dtype", string("little-endian-f32")),
        ("file", string(file)),
        ("nbytes", number(offset)),
        ("order", string("C")),
        ("sha256", string(sha256_value)),
    ])
}

fn validate_scheduler_extension(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.is_empty() || bytes.len() > MAX_BLOB_NBYTES {
        return Err(invalid(
            "child scheduler extension length is outside its v1 cap",
        ));
    }
    let value = parse_canonical_object(bytes, "child scheduler extension")?;
    exact_keys(&value, &["entries", "schema"], "child scheduler extension")?;
    fixed_string(
        &value,
        "schema",
        CHILD_POST_STEP_SCHEDULER_SCHEMA,
        "child scheduler extension",
    )?;
    let entries = value
        .get("entries")
        .and_then(JsonValue::as_object)
        .map_err(|error| invalid(format!("child scheduler extension.entries: {error}")))?;
    for (name, value) in entries {
        validate_extension_name(name)?;
        if CORE_SCALAR_NAMES.contains(&name.as_str()) {
            return Err(invalid(format!(
                "child scheduler extension key {name} collides with typed cadence"
            )));
        }
        match value {
            JsonValue::Bool(_) | JsonValue::Number(_) | JsonValue::String(_) => {}
            _ => {
                return Err(invalid(format!(
                    "child scheduler extension value {name} must be scalar"
                )))
            }
        }
    }
    Ok(bytes.to_vec())
}

fn layout_json(contract: &ForecastContract) -> Result<JsonValue> {
    let mut offset = 0u64;
    let mut carriers = Vec::with_capacity(contract.arrays().len());
    for spec in contract.arrays() {
        let nbytes = spec_nbytes(spec)?;
        carriers.push(object(vec![
            ("name", string(spec.name())),
            ("nbytes", number(nbytes)),
            ("offset_nbytes", number(offset)),
            ("shape", usize_array(spec.shape())),
        ]));
        offset = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("child layout byte count overflows"))?;
    }
    Ok(object(vec![
        ("carrier_count", number(contract.arrays().len())),
        ("carriers", JsonValue::Array(carriers)),
        ("contract_sha256", string(contract.sha256())),
        ("dtype", string("little-endian-f32")),
        ("nbytes", number(offset)),
        ("order", string("C")),
        ("profile", string(contract.profile_id())),
    ]))
}

fn contract_nbytes(contract: &ForecastContract) -> Result<u64> {
    contract.arrays().iter().try_fold(0u64, |total, spec| {
        total
            .checked_add(spec_nbytes(spec)?)
            .ok_or_else(|| invalid("child contract byte count overflows"))
    })
}

fn spec_nbytes(spec: &ForecastArraySpec) -> Result<u64> {
    let words = spec_words(spec)?;
    (words as u64)
        .checked_mul(4)
        .ok_or_else(|| invalid("child carrier byte count overflows"))
}

fn spec_words(spec: &ForecastArraySpec) -> Result<usize> {
    spec.shape().iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("child carrier {} shape overflows", spec.name())))
    })
}

fn one_bounds(shape: &[usize]) -> Result<Vec<i32>> {
    let mut bounds = Vec::with_capacity(shape.len() * 2);
    for extent in shape {
        let upper = i32::try_from(*extent)
            .map_err(|_| invalid("child carrier extent exceeds i32 bounds"))?;
        bounds.extend([1, upper]);
    }
    Ok(bounds)
}

fn required_blob<'a>(
    blobs: &'a BTreeMap<&str, &'a ComponentBlob>,
    name: &str,
    schema: &str,
) -> Result<&'a ComponentBlob> {
    let blob = blobs
        .get(name)
        .copied()
        .ok_or_else(|| invalid(format!("child component blob {name} is missing")))?;
    if blob.schema() != schema {
        return Err(invalid(format!(
            "child component blob {name} schema changed"
        )));
    }
    Ok(blob)
}

fn verify_pair_inventory(directory: &Path) -> Result<()> {
    verify_inventory(
        directory,
        &[
            CHILD_POST_STEP_CHECKPOINT_FILE,
            CHILD_POST_STEP_HISTORY_FILE,
            CHILD_POST_STEP_MANIFEST_FILE,
            CHILD_POST_STEP_RECEIPT_FILE,
        ],
        "child post-step pair",
    )
}

fn verify_restore_inventory(directory: &Path) -> Result<()> {
    verify_inventory(
        directory,
        &[
            CHILD_POST_STEP_RESTORE_CHECKPOINT_ARENA,
            CHILD_POST_STEP_RESTORE_HISTORY_ARENA,
            CHILD_POST_STEP_MANIFEST_FILE,
            CHILD_POST_STEP_RECEIPT_FILE,
        ],
        "child post-step restore",
    )
}

fn verify_inventory(directory: &Path, expected: &[&str], label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!(
            "{label} must be a regular non-symlink directory"
        )));
    }
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .to_str()
            .ok_or_else(|| invalid(format!("{label} contains a non-UTF-8 name")))?
            .to_owned();
        if !entry.file_type()?.is_file() {
            return Err(invalid(format!("{label} contains a non-regular member")));
        }
        actual.insert(name);
    }
    let actual = actual.iter().map(String::as_str).collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid(format!("{label} file inventory changed")));
    }
    Ok(())
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("child output requires a UTF-8 name"))?;
    let staging = output.with_file_name(format!(
        ".{name}.partial-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    if fs::symlink_metadata(&staging).is_ok() {
        return Err(invalid("child staging path already exists"));
    }
    Ok(staging)
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn write_arrays_new(path: &Path, arrays: &[ForecastArray]) -> Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    for array in arrays {
        for value in array.values() {
            writer.write_all(&value.to_bits().to_le_bytes())?;
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(not(windows))]
    File::open(path)?.sync_all()?;
    #[cfg(windows)]
    let _ = path;
    Ok(())
}

fn read_regular_bounded(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > limit {
        return Err(invalid(format!("{label} is not a bounded regular file")));
    }
    fs::read(path).map_err(Into::into)
}

fn hash_regular(path: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(format!("{} is not a regular file", path.display())));
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::default();
    let mut block = vec![0u8; READ_BLOCK_NBYTES];
    loop {
        let read = reader.read(&mut block)?;
        if read == 0 {
            break;
        }
        digest.update(&block[..read]);
    }
    Ok(sha256::hex(&digest.finalize()))
}

fn parse_canonical_object(bytes: &[u8], label: &str) -> Result<JsonValue> {
    let value = json::parse(bytes).map_err(|error| invalid(format!("{label}: {error}")))?;
    let _ = value
        .as_object()
        .map_err(|error| invalid(format!("{label}: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid(format!("{label} is not canonical JSON")));
    }
    Ok(value)
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{label} must be exactly 64 lower-case hexadecimal characters"
        )));
    }
    Ok(())
}

fn validate_extension_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-' | b'_')
        })
    {
        return Err(invalid(format!("invalid namespaced child key {value:?}")));
    }
    Ok(())
}

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) -> Result<()> {
    let object = value
        .as_object()
        .map_err(|error| invalid(format!("{label}: {error}")))?;
    let actual = object
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
    if required_string(value, key, label)? != expected {
        return Err(invalid(format!("{label}.{key} changed")));
    }
    Ok(())
}

fn required_string<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn required_sha(value: &JsonValue, key: &str, label: &str) -> Result<String> {
    let value = required_string(value, key, label)?;
    validate_sha256(value, &format!("{label}.{key}"))?;
    Ok(value.to_owned())
}

fn required_u64(value: &JsonValue, key: &str, label: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(JsonValue::as_u64)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn validate_idealized_authority(value: &JsonValue) -> Result<(String, String, String)> {
    let label = "idealized post-step authority";
    exact_keys(
        value,
        &[
            "authority",
            "domain",
            "initial_dry_mass",
            "initialization",
            "model",
            "schema",
            "source_sha256",
            "static_carrier_sha256",
            "status",
        ],
        label,
    )?;
    fixed_string(value, "model", "GM1", label)?;
    fixed_string(value, "schema", IDEALIZED_POST_STEP_SOURCE_SCHEMA, label)?;
    fixed_string(value, "status", IDEALIZED_POST_STEP_SOURCE_STATUS, label)?;

    let authority = value
        .get("authority")
        .map_err(|error| invalid(format!("{label}.authority: {error}")))?;
    let expected_authority = object(vec![
        ("cm1_version", string("r21.1")),
        ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
    ]);
    if authority.canonical_bytes() != expected_authority.canonical_bytes() {
        return Err(invalid("idealized post-step CM1 authority changed"));
    }

    let domain = value
        .get("domain")
        .map_err(|error| invalid(format!("{label}.domain: {error}")))?;
    let expected_domain = object(vec![
        (
            "bubble_center_xy_m_f32_bits",
            JsonValue::Array(vec![string("45ea6000"), string("45ea6000")]),
        ),
        (
            "domain_shape_zyx",
            usize_array(&[CHILD_NZ, CHILD_NY, CHILD_NX]),
        ),
        ("dt_s_f64_bits", string("3fe0000000000000")),
        (
            "origin_xy_m_f32_bits",
            JsonValue::Array(vec![string("00000000"), string("00000000")]),
        ),
        (
            "spacing_m_f64_bits",
            JsonValue::Array(vec![
                string("4049000000000000"),
                string("4049000000000000"),
                string("4069000000000000"),
            ]),
        ),
        ("ztop_m_f64_bits", string("40cd4c0000000000")),
    ]);
    if domain.canonical_bytes() != expected_domain.canonical_bytes() {
        return Err(invalid("idealized post-step geometry changed"));
    }

    let initialization = value
        .get("initialization")
        .map_err(|error| invalid(format!("{label}.initialization: {error}")))?;
    let expected_initialization = object(vec![
        ("equation_set", number(2)),
        ("iinit", number(1)),
        ("irst", number(0)),
        ("isnd", number(7)),
        (
            "warm_bubble",
            object(vec![
                ("amplitude_k_f32_bits", string("3f800000")),
                ("center_z_agl_m_f32_bits", string("44af0000")),
                ("horizontal_radius_m_f32_bits", string("461c4000")),
                ("maintain_relative_humidity", JsonValue::Bool(false)),
                ("vertical_radius_m_f32_bits", string("44af0000")),
            ]),
        ),
    ]);
    if initialization.canonical_bytes() != expected_initialization.canonical_bytes() {
        return Err(invalid(
            "idealized post-step initialization controls changed",
        ));
    }

    let sources = value
        .get("source_sha256")
        .map_err(|error| invalid(format!("{label}.source_sha256: {error}")))?;
    exact_keys(
        sources,
        &[
            "cm1_init3d_f",
            "cm1_param_f",
            "python_zero_disk_bridge",
            "rust_idealized_ffi",
            "rust_idealized_initialization",
            "rust_library",
            "sounding",
        ],
        "idealized post-step source pins",
    )?;
    let expected_sources = [
        (
            "cm1_init3d_f",
            "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
        ),
        (
            "cm1_param_f",
            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
        ),
        (
            "python_zero_disk_bridge",
            "8bb2389a5be60d1dd14254ef0acfa8a35297a0fa4e385b3dba7c8d53223efe69",
        ),
        (
            "rust_idealized_ffi",
            "955038c1202ec98492d61bdb1027841682a40e1f19e5ce76b2d5726d7d0cf69e",
        ),
        (
            "rust_idealized_initialization",
            "bb3f6883f2e2867bd1220d747c6501afe99f49762397d26315eab26546ea5872",
        ),
        (
            "rust_library",
            "10ba6473422e9cffac8955a6ef68b6d47e0b58571d7550da7250a19dabcd754d",
        ),
        (
            "sounding",
            "56df89107f712b4911a7eb53a5c4df637826ee02ca14fe5322eb0cd2916bdea0",
        ),
    ];
    for (name, expected) in expected_sources {
        let actual = required_sha(sources, name, "idealized post-step source pins")?;
        if actual != expected {
            return Err(invalid(format!(
                "idealized post-step source pin {name} changed"
            )));
        }
    }

    let mass = value
        .get("initial_dry_mass")
        .map_err(|error| invalid(format!("{label}.initial_dry_mass: {error}")))?;
    exact_keys(
        mass,
        &["mass1_f64_bits", "mass1_f64_le_sha256", "source_order"],
        "idealized initial dry mass",
    )?;
    fixed_string(
        mass,
        "mass1_f64_bits",
        "414e81d1397c6000",
        "idealized initial dry mass",
    )?;
    fixed_string(
        mass,
        "mass1_f64_le_sha256",
        "1e2cbf8b97061a4a4f84df86274e79d8d11985e4b1d200f42f91e138d71e370c",
        "idealized initial dry mass",
    )?;
    fixed_string(
        mass,
        "source_order",
        "binary32 products; k/j/i source order into binary64",
        "idealized initial dry mass",
    )?;

    let static_carriers = value
        .get("static_carrier_sha256")
        .map_err(|error| invalid(format!("{label}.static_carrier_sha256: {error}")))?;
    exact_keys(
        static_carriers,
        &["terrain_msl", "total_geopotential_zf", "zf"],
        "idealized static carrier pins",
    )?;
    for (name, expected) in [
        ("terrain_msl", IDEALIZED_TERRAIN_SHA256),
        ("total_geopotential_zf", IDEALIZED_GEOPOTENTIAL_SHA256),
        ("zf", IDEALIZED_ZF_SHA256),
    ] {
        if required_sha(static_carriers, name, "idealized static carrier pins")? != expected {
            return Err(invalid(format!(
                "idealized static carrier pin {name} changed"
            )));
        }
    }

    Ok((
        required_sha(sources, "rust_library", "idealized post-step source pins")?,
        required_sha(sources, "sounding", "idealized post-step source pins")?,
        required_string(mass, "mass1_f64_bits", "idealized initial dry mass")?.to_owned(),
    ))
}

fn parse_f64_bits(value: &str, label: &str) -> Result<f64> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!("{label} is not an exact binary64 word")));
    }
    let bits = u64::from_str_radix(value, 16)
        .map_err(|_| invalid(format!("{label} is not an exact binary64 word")))?;
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return Err(invalid(format!("{label} is not finite")));
    }
    Ok(value)
}

fn canonical_string(value: JsonValue) -> String {
    String::from_utf8(value.canonical_bytes()).expect("canonical JSON is UTF-8")
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

fn u64_number(value: u64) -> JsonValue {
    number(value)
}

fn usize_array(values: &[usize]) -> JsonValue {
    JsonValue::Array(values.iter().copied().map(number).collect())
}

fn f64_bits(value: f64) -> JsonValue {
    string(f64_bits_string(value))
}

fn f64_bits_string(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cm1-child-post-step-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn operator() -> ChildPostStepOperatorManifest {
        let operators = JsonValue::Object(
            GM1_OPERATOR_SOURCE_KEYS
                .iter()
                .enumerate()
                .map(|(index, name)| ((*name).to_owned(), string(format!("{:064x}", index + 1))))
                .collect(),
        );
        let bytes = object(vec![
            ("model", string("GM1")),
            ("operators", operators),
            ("schema", string(CHILD_POST_STEP_OPERATOR_SCHEMA)),
        ])
        .canonical_bytes();
        let sha = sha256::digest_hex(&bytes);
        ChildPostStepOperatorManifest::from_canonical_bytes(&bytes, &sha).unwrap()
    }

    fn scheduler() -> Vec<u8> {
        object(vec![
            ("entries", object(Vec::new())),
            ("schema", string(CHILD_POST_STEP_SCHEDULER_SCHEMA)),
        ])
        .canonical_bytes()
    }

    fn idealized_authority() -> IdealizedPostStepAuthority {
        let bytes = object(vec![
            (
                "authority",
                object(vec![
                    ("cm1_version", string("r21.1")),
                    ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
                ]),
            ),
            (
                "domain",
                object(vec![
                    (
                        "bubble_center_xy_m_f32_bits",
                        JsonValue::Array(vec![string("45ea6000"), string("45ea6000")]),
                    ),
                    (
                        "domain_shape_zyx",
                        usize_array(&[CHILD_NZ, CHILD_NY, CHILD_NX]),
                    ),
                    ("dt_s_f64_bits", string("3fe0000000000000")),
                    (
                        "origin_xy_m_f32_bits",
                        JsonValue::Array(vec![string("00000000"), string("00000000")]),
                    ),
                    (
                        "spacing_m_f64_bits",
                        JsonValue::Array(vec![
                            string("4049000000000000"),
                            string("4049000000000000"),
                            string("4069000000000000"),
                        ]),
                    ),
                    ("ztop_m_f64_bits", string("40cd4c0000000000")),
                ]),
            ),
            (
                "initial_dry_mass",
                object(vec![
                    ("mass1_f64_bits", string("414e81d1397c6000")),
                    (
                        "mass1_f64_le_sha256",
                        string("1e2cbf8b97061a4a4f84df86274e79d8d11985e4b1d200f42f91e138d71e370c"),
                    ),
                    (
                        "source_order",
                        string("binary32 products; k/j/i source order into binary64"),
                    ),
                ]),
            ),
            (
                "initialization",
                object(vec![
                    ("equation_set", number(2)),
                    ("iinit", number(1)),
                    ("irst", number(0)),
                    ("isnd", number(7)),
                    (
                        "warm_bubble",
                        object(vec![
                            ("amplitude_k_f32_bits", string("3f800000")),
                            ("center_z_agl_m_f32_bits", string("44af0000")),
                            ("horizontal_radius_m_f32_bits", string("461c4000")),
                            ("maintain_relative_humidity", JsonValue::Bool(false)),
                            ("vertical_radius_m_f32_bits", string("44af0000")),
                        ]),
                    ),
                ]),
            ),
            ("model", string("GM1")),
            ("schema", string(IDEALIZED_POST_STEP_SOURCE_SCHEMA)),
            (
                "source_sha256",
                object(vec![
                    (
                        "cm1_init3d_f",
                        string("9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28"),
                    ),
                    (
                        "cm1_param_f",
                        string("cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349"),
                    ),
                    (
                        "python_zero_disk_bridge",
                        string("8bb2389a5be60d1dd14254ef0acfa8a35297a0fa4e385b3dba7c8d53223efe69"),
                    ),
                    (
                        "rust_idealized_ffi",
                        string("955038c1202ec98492d61bdb1027841682a40e1f19e5ce76b2d5726d7d0cf69e"),
                    ),
                    (
                        "rust_idealized_initialization",
                        string("bb3f6883f2e2867bd1220d747c6501afe99f49762397d26315eab26546ea5872"),
                    ),
                    (
                        "rust_library",
                        string("10ba6473422e9cffac8955a6ef68b6d47e0b58571d7550da7250a19dabcd754d"),
                    ),
                    (
                        "sounding",
                        string("56df89107f712b4911a7eb53a5c4df637826ee02ca14fe5322eb0cd2916bdea0"),
                    ),
                ]),
            ),
            (
                "static_carrier_sha256",
                object(vec![
                    ("terrain_msl", string(IDEALIZED_TERRAIN_SHA256)),
                    (
                        "total_geopotential_zf",
                        string(IDEALIZED_GEOPOTENTIAL_SHA256),
                    ),
                    ("zf", string(IDEALIZED_ZF_SHA256)),
                ]),
            ),
            ("status", string(IDEALIZED_POST_STEP_SOURCE_STATUS)),
        ])
        .canonical_bytes();
        let sha = sha256::digest_hex(&bytes);
        IdealizedPostStepAuthority::from_canonical_bytes(&bytes, &sha).unwrap()
    }

    fn arrays(specs: &[ForecastArraySpec]) -> Vec<ForecastArray> {
        specs
            .iter()
            .map(|spec| ForecastArray::new(spec.name(), vec![1.0; spec_words(spec).unwrap()]))
            .collect()
    }

    fn write_arena(path: &Path, arrays: &[ForecastArray]) -> String {
        write_arrays_new(path, arrays).unwrap();
        hash_regular(path).unwrap()
    }

    #[test]
    fn production_layout_matches_sealed_byte_counts_and_orders() {
        let operator = operator();
        let checkpoint = checkpoint_contract(ChildPostStepConfig::production(), &operator).unwrap();
        let history = history_contract(ChildPostStepConfig::production(), &operator).unwrap();
        assert_eq!(
            contract_nbytes(&checkpoint).unwrap(),
            CHILD_POST_STEP_CHECKPOINT_NBYTES
        );
        assert_eq!(
            contract_nbytes(&history).unwrap(),
            CHILD_POST_STEP_HISTORY_NBYTES
        );
        assert_eq!(
            checkpoint
                .arrays()
                .iter()
                .map(ForecastArraySpec::name)
                .collect::<Vec<_>>(),
            CHECKPOINT_NAMES
        );
        assert_eq!(
            history
                .arrays()
                .iter()
                .map(ForecastArraySpec::name)
                .collect::<Vec<_>>(),
            HISTORY_NAMES
        );
        let layout = child_post_step_layout_json(&operator).unwrap();
        assert!(layout.contains("976696992"));
        assert!(layout.contains("461340000"));
        assert!(layout.contains(SEALED_CHILD_FULL_STATE_MANIFEST_SHA256));
    }

    #[test]
    fn bounded_pair_roundtrips_raw_bits_and_refuses_replacement() {
        let root = temp_root("roundtrip");
        fs::create_dir(&root).unwrap();
        let config = ChildPostStepConfig::bounded_proof(4, 2, 3).unwrap();
        let operator = operator();
        let checkpoint_arrays = arrays(&checkpoint_specs(config).unwrap());
        let history_arrays = arrays(&history_specs(config).unwrap());
        let checkpoint_raw = root.join("checkpoint.f32le");
        let history_raw = root.join("history.f32le");
        let checkpoint_sha = write_arena(&checkpoint_raw, &checkpoint_arrays);
        let history_sha = write_arena(&history_raw, &history_arrays);
        let pair = root.join("pair");
        let source = root.join("bounded-source");
        let source_seal = PostStepSourceSeal::Production(test_source_seal(&source));
        let receipt = publish_child_post_step_config_new(
            source_seal.clone(),
            &checkpoint_raw,
            &checkpoint_sha,
            &history_raw,
            &history_sha,
            operator.clone(),
            &scheduler(),
            ForecastClock::new(1, 0.5, 0.5).unwrap(),
            42.0,
            &pair,
            config,
        )
        .unwrap();
        assert!(publish_child_post_step_config_new(
            source_seal.clone(),
            &checkpoint_raw,
            &checkpoint_sha,
            &history_raw,
            &history_sha,
            operator.clone(),
            &scheduler(),
            ForecastClock::new(1, 0.5, 0.5).unwrap(),
            42.0,
            &pair,
            config,
        )
        .is_err());
        let inspected = inspect_child_post_step_config(
            source_seal.clone(),
            &pair,
            &receipt.manifest_sha256,
            &receipt.receipt_sha256,
            &receipt.checkpoint.artifact_sha256,
            &receipt.history.artifact_sha256,
            operator.sha256(),
            config,
        )
        .unwrap();
        assert_eq!(inspected.checkpoint_raw_sha256, checkpoint_sha);
        assert_eq!(inspected.history_raw_sha256, history_sha);
        let restore = root.join("restore");
        let restored = restore_child_post_step_config_new(
            source_seal,
            &pair,
            &restore,
            &receipt.manifest_sha256,
            &receipt.receipt_sha256,
            &receipt.checkpoint.artifact_sha256,
            &receipt.history.artifact_sha256,
            operator.sha256(),
            config,
        )
        .unwrap();
        assert_eq!(restored.checkpoint_arena_sha256, checkpoint_sha);
        assert_eq!(restored.history_arena_sha256, history_sha);
        assert_eq!(
            fs::read(restore.join(CHILD_POST_STEP_RESTORE_CHECKPOINT_ARENA)).unwrap(),
            fs::read(&checkpoint_raw).unwrap()
        );
        assert_eq!(
            fs::read(restore.join(CHILD_POST_STEP_RESTORE_HISTORY_ARENA)).unwrap(),
            fs::read(&history_raw).unwrap()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_idealized_pair_is_atomic_independently_inspected_and_profile_closed() {
        let root = temp_root("idealized-roundtrip");
        fs::create_dir(&root).unwrap();
        let config = ChildPostStepConfig::idealized_bounded_proof(4, 2, 3).unwrap();
        let production_config = ChildPostStepConfig::bounded_proof(4, 2, 3).unwrap();
        let operator = operator();
        let canonical_authority = idealized_authority();
        let authority_path = root.join("idealized-authority.json");
        let mut authority_file = canonical_authority.bytes().to_vec();
        authority_file.push(b'\n');
        fs::write(&authority_path, &authority_file).unwrap();
        let authority_file_sha = sha256::digest_hex(&authority_file);
        let authority =
            IdealizedPostStepAuthority::from_file(&authority_path, &authority_file_sha).unwrap();
        assert_eq!(authority.bytes(), authority_file);
        let source = PostStepSourceSeal::Idealized(authority.clone());
        let checkpoint_arrays = arrays(&checkpoint_specs(config).unwrap());
        let history_arrays = arrays(&history_specs(config).unwrap());
        let checkpoint_raw = root.join("checkpoint.f32le");
        let history_raw = root.join("history.f32le");
        let checkpoint_sha = write_arena(&checkpoint_raw, &checkpoint_arrays);
        let history_sha = write_arena(&history_raw, &history_arrays);
        let pair = root.join("pair");
        let mass1 = f64::from_bits(u64::from_str_radix("414e81d1397c6000", 16).unwrap());
        let receipt = publish_child_post_step_config_new(
            source.clone(),
            &checkpoint_raw,
            &checkpoint_sha,
            &history_raw,
            &history_sha,
            operator.clone(),
            &scheduler(),
            ForecastClock::new(1, 0.5, 0.5).unwrap(),
            mass1,
            &pair,
            config,
        )
        .unwrap();
        assert!(pair.is_dir());
        assert!(!root.read_dir().unwrap().any(|entry| entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".partial-")));
        let manifest = fs::read(pair.join(CHILD_POST_STEP_MANIFEST_FILE)).unwrap();
        let manifest = parse_canonical_object(&manifest, "test idealized manifest").unwrap();
        fixed_string(
            &manifest,
            "schema",
            IDEALIZED_POST_STEP_MANIFEST_SCHEMA,
            "test idealized manifest",
        )
        .unwrap();
        fixed_string(
            &manifest,
            "status",
            config.pair_status(),
            "test idealized manifest",
        )
        .unwrap();
        assert!(manifest.get("source_idealized_authority").is_ok());
        assert!(manifest.get("source_full_state").is_err());

        let inspected = inspect_child_post_step_config(
            source.clone(),
            &pair,
            &receipt.manifest_sha256,
            &receipt.receipt_sha256,
            &receipt.checkpoint.artifact_sha256,
            &receipt.history.artifact_sha256,
            operator.sha256(),
            config,
        )
        .unwrap();
        assert_eq!(inspected.checkpoint_raw_sha256, checkpoint_sha);
        assert_eq!(inspected.history_raw_sha256, history_sha);
        assert!(inspected
            .canonical_json()
            .contains(IDEALIZED_POST_STEP_RECEIPT_SCHEMA));

        let cross_profile = inspect_child_post_step_config(
            PostStepSourceSeal::Production(test_source_seal(&root.join("real-source"))),
            &pair,
            &receipt.manifest_sha256,
            &receipt.receipt_sha256,
            &receipt.checkpoint.artifact_sha256,
            &receipt.history.artifact_sha256,
            operator.sha256(),
            production_config,
        )
        .unwrap_err();
        assert!(cross_profile.to_string().contains("schema"));

        let restore = root.join("restore");
        let restored = restore_child_post_step_config_new(
            source,
            &pair,
            &restore,
            &receipt.manifest_sha256,
            &receipt.receipt_sha256,
            &receipt.checkpoint.artifact_sha256,
            &receipt.history.artifact_sha256,
            operator.sha256(),
            config,
        )
        .unwrap();
        assert_eq!(restored.checkpoint_arena_sha256, checkpoint_sha);
        assert_eq!(restored.history_arena_sha256, history_sha);
        assert!(restored
            .canonical_json()
            .contains(IDEALIZED_POST_STEP_RESTORE_RECEIPT_SCHEMA));

        let mut changed = authority.bytes().to_vec();
        *changed.last_mut().unwrap() ^= 1;
        assert!(
            IdealizedPostStepAuthority::from_canonical_bytes(&changed, authority.sha256()).is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn overlap_clock_hash_and_operator_refusals_are_closed() {
        let root = temp_root("refusals");
        fs::create_dir(&root).unwrap();
        let config = ChildPostStepConfig::bounded_proof(2, 1, 2).unwrap();
        let operator = operator();
        let checkpoint_arrays = arrays(&checkpoint_specs(config).unwrap());
        let mut history_arrays = arrays(&history_specs(config).unwrap());
        history_arrays
            .iter_mut()
            .find(|array| array.name() == "pp3d")
            .unwrap()
            .values_mut()[0] = 2.0;
        let checkpoint_raw = root.join("checkpoint.f32le");
        let history_raw = root.join("history.f32le");
        let checkpoint_sha = write_arena(&checkpoint_raw, &checkpoint_arrays);
        let history_sha = write_arena(&history_raw, &history_arrays);
        assert!(publish_child_post_step_config_new(
            PostStepSourceSeal::Production(test_source_seal(&root.join("source"))),
            &checkpoint_raw,
            &checkpoint_sha,
            &history_raw,
            &history_sha,
            operator.clone(),
            &scheduler(),
            ForecastClock::new(1, 0.5, 0.5).unwrap(),
            1.0,
            &root.join("pair"),
            config,
        )
        .unwrap_err()
        .to_string()
        .contains("differs raw-bit"));
        assert!(validate_boundary(ForecastClock::new(0, 0.0, 0.5).unwrap(), 1.0).is_err());
        assert!(validate_boundary(ForecastClock::new(1, 1.0, 0.5).unwrap(), 1.0).is_err());
        assert!(ChildPostStepOperatorManifest::from_canonical_bytes(
            operator.bytes(),
            &"ff".repeat(32)
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn superseded_full_state_tuple_is_explicitly_distinct() {
        assert_ne!(
            SUPERSEDED_CHILD_MANIFEST_SHA256,
            SEALED_CHILD_FULL_STATE_MANIFEST_SHA256
        );
        assert_eq!(SUPERSEDED_CHILD_MANIFEST_SHA256.len(), 64);
    }
}
