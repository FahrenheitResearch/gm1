//! Atomic lightweight cloud-volume transport derived from an accepted child history.
//!
//! The full CM1FHS1 history remains the scientific state.  This module emits
//! only exact binary32 `qc + qi` words for asynchronous GM1 View ingestion.

use crate::child_post_step::{
    CHILD_POST_STEP_CLAIM_LIMIT, CHILD_POST_STEP_MANIFEST_SCHEMA, CHILD_POST_STEP_RECEIPT_SCHEMA,
    IDEALIZED_POST_STEP_CLAIM_LIMIT, IDEALIZED_POST_STEP_MANIFEST_SCHEMA,
    IDEALIZED_POST_STEP_RECEIPT_SCHEMA, IDEALIZED_POST_STEP_SOURCE_SCHEMA,
    IDEALIZED_POST_STEP_SOURCE_STATUS,
};
use crate::forecast::{
    ForecastContract, RestoredForecast, inspect_forecast_checkpoint_embedded,
    read_forecast_history_embedded,
};
use crate::json::{self, JsonValue};
use crate::sha256::{self, Sha256};
use crate::{Result, StateError};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

mod production;
pub use production::{
    CHILD_PRODUCTION_VIEW_PACK_ABI, CHILD_PRODUCTION_VIEW_PACK_MANIFEST_SCHEMA,
    CHILD_PRODUCTION_VIEW_PACK_RECEIPT_SCHEMA, ChildProductionViewPackReceipt,
    inspect_child_production_view_pack, publish_child_production_view_pack_new,
};

pub const CHILD_VIEW_PACK_MANIFEST_SCHEMA: &str = "gm1.view-derived-cloud-volume-manifest/v1";
pub const CHILD_VIEW_PACK_RECEIPT_SCHEMA: &str = "gm1.view-derived-cloud-volume-receipt/v1";
pub const CHILD_VIEW_PACK_ABI: &str = "gm1.view-cloud-condensate-f32le-c-order/v1";
pub const CHILD_VIEW_PACK_PAYLOAD_FILE: &str = "cloud-condensate.f32le";
pub const CHILD_VIEW_PACK_MANIFEST_FILE: &str = "manifest.json";
pub const CHILD_VIEW_PACK_RECEIPT_FILE: &str = "receipt.json";
pub const CHILD_VIEW_PACK_STATUS: &str = "PASS_EXACT_QC_PLUS_QI_RENDER_TRANSPORT";
pub const CHILD_VIEW_PACK_CLAIM_LIMIT: &str = "exact non-quantized binary32 qc+qi render transport derived in source word order from one independently pinned, accepted GM1 50 m child CM1FHS1 history; the full history remains the scientific state and this atomic three-file directory is not a checkpoint, restart, independent meteorological product, forecast-skill, storm, tornado, or performance claim";

pub const CHILD_VIEW_PACK_NZ: usize = 75;
pub const CHILD_VIEW_PACK_NY: usize = 300;
pub const CHILD_VIEW_PACK_NX: usize = 300;
pub const CHILD_VIEW_PACK_WORDS: usize =
    CHILD_VIEW_PACK_NZ * CHILD_VIEW_PACK_NY * CHILD_VIEW_PACK_NX;
pub const CHILD_VIEW_PACK_NBYTES: usize = CHILD_VIEW_PACK_WORDS * 4;
pub const CHILD_VIEW_Q3D_WORDS: usize = 6 * CHILD_VIEW_PACK_WORDS;
pub const CHILD_VIEW_Q3D_NBYTES: usize = CHILD_VIEW_Q3D_WORDS * 4;
pub const CHILD_VIEW_SPECIES: [&str; 6] = ["qv", "qc", "qr", "qi", "qs", "qg"];
pub const CHILD_VIEW_FORMULA: &str = "out[word]=f32(q3d[1*words+word]+q3d[3*words+word])";

const REAL_PAIR_STATUS: &str = "PASS_RUST_PAIRED_POST_STEP_ENVELOPE";
const IDEALIZED_PAIR_STATUS: &str = "PASS_RUST_IDEALIZED_ISND7_IINIT1_PAIRED_POST_STEP_ENVELOPE";
const IDEALIZED_SOURCE_AUTHORITY_SHA256: &str =
    "d4857df7e121f48a27cc15b3ff8845b0f064a790f4954a08d24588af0d5f6366";
const IDEALIZED_READINESS_RECEIPT_SHA256: &str =
    "a18a837cd7f8676fc2eb165c0b518c9f9f871957664548289c45bc6a480a7228";
const IDEALIZED_PAIR_PUBLISHER_SHA256: &str =
    "511bc27d685bd28f9a0526e1e32908b5db3800e29f6f63f7aa31e4bc1fadb096";
const IDEALIZED_HISTORY_CONTRACT_SHA256: &str =
    "f519a89c8171f34b2ca0317fd864ec1a4cd40371e360a7f39a3d20b3acca4d5d";
const IDEALIZED_CHECKPOINT_CONTRACT_SHA256: &str =
    "cf11c4eb1b7a8a6823d71ae8a94b7fed2532f021e88bf0931e372c6555acf914";
const IDEALIZED_OPERATOR_MANIFEST_SHA256: &str =
    "82fa0c0669c3d6eee805446df29fc42ab4f0e1b0a9fae6690adbfaf587f1352d";
const IDEALIZED_SCHEDULER_EXTENSION_SHA256: &str =
    "6552ed7d7afd4115baad3406dffbb69e11d7a187af8d3591f6969c985bf64ac2";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourcePairProfile {
    ProductionRealChild,
    IdealizedSingleSoundingWarmBubble,
}

impl SourcePairProfile {
    fn from_envelopes(
        manifest_schema: &str,
        manifest_status: &str,
        receipt_schema: &str,
        receipt_status: &str,
    ) -> Result<Self> {
        match (
            manifest_schema,
            manifest_status,
            receipt_schema,
            receipt_status,
        ) {
            (
                CHILD_POST_STEP_MANIFEST_SCHEMA,
                REAL_PAIR_STATUS,
                CHILD_POST_STEP_RECEIPT_SCHEMA,
                REAL_PAIR_STATUS,
            ) => Ok(Self::ProductionRealChild),
            (
                IDEALIZED_POST_STEP_MANIFEST_SCHEMA,
                IDEALIZED_PAIR_STATUS,
                IDEALIZED_POST_STEP_RECEIPT_SCHEMA,
                IDEALIZED_PAIR_STATUS,
            ) => Ok(Self::IdealizedSingleSoundingWarmBubble),
            _ => Err(invalid(
                "source pair manifest/receipt schema and status do not form one admitted profile envelope",
            )),
        }
    }

    const fn source_key(self) -> &'static str {
        match self {
            Self::ProductionRealChild => "source_full_state",
            Self::IdealizedSingleSoundingWarmBubble => "source_idealized_authority",
        }
    }

    const fn manifest_schema(self) -> &'static str {
        match self {
            Self::ProductionRealChild => CHILD_POST_STEP_MANIFEST_SCHEMA,
            Self::IdealizedSingleSoundingWarmBubble => IDEALIZED_POST_STEP_MANIFEST_SCHEMA,
        }
    }

    const fn receipt_schema(self) -> &'static str {
        match self {
            Self::ProductionRealChild => CHILD_POST_STEP_RECEIPT_SCHEMA,
            Self::IdealizedSingleSoundingWarmBubble => IDEALIZED_POST_STEP_RECEIPT_SCHEMA,
        }
    }

    const fn status(self) -> &'static str {
        match self {
            Self::ProductionRealChild => REAL_PAIR_STATUS,
            Self::IdealizedSingleSoundingWarmBubble => IDEALIZED_PAIR_STATUS,
        }
    }

    const fn claim_limit(self) -> &'static str {
        match self {
            Self::ProductionRealChild => CHILD_POST_STEP_CLAIM_LIMIT,
            Self::IdealizedSingleSoundingWarmBubble => IDEALIZED_POST_STEP_CLAIM_LIMIT,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AdmittedSourcePair {
    pair_manifest_sha256: &'static str,
    pair_receipt_sha256: &'static str,
    history_contract_sha256: &'static str,
    history_artifact_sha256: &'static str,
    history_payload_sha256: &'static str,
    history_raw_sha256: &'static str,
    checkpoint_contract_sha256: &'static str,
    checkpoint_artifact_sha256: &'static str,
    checkpoint_payload_sha256: &'static str,
    checkpoint_raw_sha256: &'static str,
    operator_manifest_sha256: &'static str,
    scheduler_extension_sha256: &'static str,
    step: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourcePairAuthority {
    HistoricalExact,
    LiveSeries,
    RestoredSeries,
    ProductionSeries,
    IdealizedSingleSoundingWarmBubble,
}

impl SourcePairAuthority {
    fn for_science_envelope(
        profile: SourcePairProfile,
        manifest_sha256: &str,
        receipt_sha256: &str,
        history_contract_sha256: &str,
        checkpoint_contract_sha256: &str,
        operator_manifest_sha256: &str,
        scheduler_extension_sha256: &str,
    ) -> Result<Self> {
        if profile == SourcePairProfile::IdealizedSingleSoundingWarmBubble {
            for (value, label) in [
                (
                    IDEALIZED_READINESS_RECEIPT_SHA256,
                    "idealized readiness receipt SHA-256",
                ),
                (
                    IDEALIZED_PAIR_PUBLISHER_SHA256,
                    "idealized pair publisher SHA-256",
                ),
            ] {
                validate_sha256(value, label)?;
            }
            if history_contract_sha256 == IDEALIZED_HISTORY_CONTRACT_SHA256
                && checkpoint_contract_sha256 == IDEALIZED_CHECKPOINT_CONTRACT_SHA256
                && operator_manifest_sha256 == IDEALIZED_OPERATOR_MANIFEST_SHA256
                && scheduler_extension_sha256 == IDEALIZED_SCHEDULER_EXTENSION_SHA256
            {
                return Ok(Self::IdealizedSingleSoundingWarmBubble);
            }
            return Err(invalid(
                "idealized source pair contracts/operator/scheduler differ from sealed CPU readiness",
            ));
        }
        if manifest_sha256 == HISTORICAL_SOURCE_PAIR.pair_manifest_sha256
            && receipt_sha256 == HISTORICAL_SOURCE_PAIR.pair_receipt_sha256
        {
            if history_contract_sha256 == HISTORICAL_SOURCE_PAIR.history_contract_sha256
                && checkpoint_contract_sha256 == HISTORICAL_SOURCE_PAIR.checkpoint_contract_sha256
                && operator_manifest_sha256 == HISTORICAL_SOURCE_PAIR.operator_manifest_sha256
                && scheduler_extension_sha256 == HISTORICAL_SOURCE_PAIR.scheduler_extension_sha256
            {
                return Ok(Self::HistoricalExact);
            }
            return Err(invalid(
                "historical source pair hashes are coupled to a different science envelope",
            ));
        }
        if scheduler_extension_sha256 == LIVE_SERIES_SCHEDULER_EXTENSION_SHA256
            && history_contract_sha256 == LIVE_SERIES_HISTORY_CONTRACT_SHA256
            && checkpoint_contract_sha256 == LIVE_SERIES_CHECKPOINT_CONTRACT_SHA256
            && operator_manifest_sha256 == LIVE_SERIES_OPERATOR_MANIFEST_SHA256
        {
            return Ok(Self::LiveSeries);
        }
        if scheduler_extension_sha256 == LIVE_SERIES_SCHEDULER_EXTENSION_SHA256
            && history_contract_sha256 == RESTORED_SERIES_HISTORY_CONTRACT_SHA256
            && checkpoint_contract_sha256 == RESTORED_SERIES_CHECKPOINT_CONTRACT_SHA256
            && operator_manifest_sha256 == RESTORED_SERIES_OPERATOR_MANIFEST_SHA256
        {
            return Ok(Self::RestoredSeries);
        }
        if scheduler_extension_sha256 == LIVE_SERIES_SCHEDULER_EXTENSION_SHA256
            && history_contract_sha256 == PRODUCTION_SERIES_HISTORY_CONTRACT_SHA256
            && checkpoint_contract_sha256 == PRODUCTION_SERIES_CHECKPOINT_CONTRACT_SHA256
            && operator_manifest_sha256 == PRODUCTION_SERIES_OPERATOR_MANIFEST_SHA256
        {
            return Ok(Self::ProductionSeries);
        }
        Err(invalid(
            "source pair contracts/operator/scheduler do not identify an admitted live science envelope",
        ))
    }

    fn history_contract_sha256(self) -> &'static str {
        match self {
            Self::HistoricalExact => HISTORICAL_SOURCE_PAIR.history_contract_sha256,
            Self::LiveSeries => LIVE_SERIES_HISTORY_CONTRACT_SHA256,
            Self::RestoredSeries => RESTORED_SERIES_HISTORY_CONTRACT_SHA256,
            Self::ProductionSeries => PRODUCTION_SERIES_HISTORY_CONTRACT_SHA256,
            Self::IdealizedSingleSoundingWarmBubble => IDEALIZED_HISTORY_CONTRACT_SHA256,
        }
    }

    fn checkpoint_contract_sha256(self) -> &'static str {
        match self {
            Self::HistoricalExact => HISTORICAL_SOURCE_PAIR.checkpoint_contract_sha256,
            Self::LiveSeries => LIVE_SERIES_CHECKPOINT_CONTRACT_SHA256,
            Self::RestoredSeries => RESTORED_SERIES_CHECKPOINT_CONTRACT_SHA256,
            Self::ProductionSeries => PRODUCTION_SERIES_CHECKPOINT_CONTRACT_SHA256,
            Self::IdealizedSingleSoundingWarmBubble => IDEALIZED_CHECKPOINT_CONTRACT_SHA256,
        }
    }

    fn operator_manifest_sha256(self) -> &'static str {
        match self {
            Self::HistoricalExact => HISTORICAL_SOURCE_PAIR.operator_manifest_sha256,
            Self::LiveSeries => LIVE_SERIES_OPERATOR_MANIFEST_SHA256,
            Self::RestoredSeries => RESTORED_SERIES_OPERATOR_MANIFEST_SHA256,
            Self::ProductionSeries => PRODUCTION_SERIES_OPERATOR_MANIFEST_SHA256,
            Self::IdealizedSingleSoundingWarmBubble => IDEALIZED_OPERATOR_MANIFEST_SHA256,
        }
    }

    fn scheduler_extension_sha256(self) -> &'static str {
        match self {
            Self::HistoricalExact => HISTORICAL_SOURCE_PAIR.scheduler_extension_sha256,
            Self::LiveSeries => LIVE_SERIES_SCHEDULER_EXTENSION_SHA256,
            Self::RestoredSeries => LIVE_SERIES_SCHEDULER_EXTENSION_SHA256,
            Self::ProductionSeries => LIVE_SERIES_SCHEDULER_EXTENSION_SHA256,
            Self::IdealizedSingleSoundingWarmBubble => IDEALIZED_SCHEDULER_EXTENSION_SHA256,
        }
    }
}

const HISTORICAL_SOURCE_PAIR: AdmittedSourcePair = AdmittedSourcePair {
    pair_manifest_sha256: "c37e5bec8101b797fe516fe21f3474785f3bd2884337d646f2ba0f7fbe8961e5",
    pair_receipt_sha256: "2ad53c8a564c38c775fe138248b9f2cd4bc471f67168602c60f943026a80c36b",
    history_contract_sha256: "78869f9f9775f24e42aa843bb88b415ba975f6481318a00e8f05b45044c6549b",
    history_artifact_sha256: "94108b08e926a9d4ebc09029b66afcc3db4948bb1f769296a9e100d9e2f7bb4e",
    history_payload_sha256: "2f15aea48331fe0d383af2c72f5bf44471b71e2b1e2d3280c6c2c70e54ddf42d",
    history_raw_sha256: "09fb4b252c18388aedbbf821e75e39baec7a9263038080504edd629683699ba9",
    checkpoint_contract_sha256: "515fe73f4c4732fe33c1cc69714b280119d8934d4e1b1ceae4e083cf31e22fcd",
    checkpoint_artifact_sha256: "b5b4cc507ed58cc738311ff1c5429d391e0db99eaa33354d365eb7170591a10d",
    checkpoint_payload_sha256: "660db1f2316deef9ff7d2d16bd0e0530d29b88a5e43d0e740fbd8e083c97e4e1",
    checkpoint_raw_sha256: "ec5afcdcb1bba63a65b4428d61d4da65eadeb73093258bc8290e2974a862d0fd",
    operator_manifest_sha256: "c60142440dcb98bed84a784e2ff27405f2c1bd66afec32c3ef263c5f5e37cef8",
    scheduler_extension_sha256: "6552ed7d7afd4115baad3406dffbb69e11d7a187af8d3591f6969c985bf64ac2",
    step: 600,
};
const LIVE_SERIES_HISTORY_CONTRACT_SHA256: &str =
    "a5d3f1c784544cf0398a0499d51fc77792b74fcab8129755b9e43b872c9dbb90";
const LIVE_SERIES_CHECKPOINT_CONTRACT_SHA256: &str =
    "6726ade2c3224c4740ed5a363d7e5843803a0f2189a4f692152913874aed0425";
const LIVE_SERIES_OPERATOR_MANIFEST_SHA256: &str =
    "6112fb9302f57f82b09118b3f38c6ae8dd3d82c59eb14fd01940f2dc43428c09";
const RESTORED_SERIES_HISTORY_CONTRACT_SHA256: &str =
    "d2be76466ff9848abc2239edfb9df03013b1ccd08ee061142f05e1bd96458894";
const RESTORED_SERIES_CHECKPOINT_CONTRACT_SHA256: &str =
    "76290e07f9a48a7a172c0f6923effbf065c406966c806cdd6092563b7d1826c3";
const RESTORED_SERIES_OPERATOR_MANIFEST_SHA256: &str =
    "a52400db650092944f152276ed764ad3cb852473f9b8b9b805b7ed68b4c1f3e7";
const PRODUCTION_SERIES_HISTORY_CONTRACT_SHA256: &str =
    "2d2df2c9c4fb2df6b383ed3a09be61a4e420212e888ad6136f40e716acc88cdd";
const PRODUCTION_SERIES_CHECKPOINT_CONTRACT_SHA256: &str =
    "7e9b35fb970556741b14e448cd6199dd9e2cd23edbbb64ed5b8731eef108ac71";
const PRODUCTION_SERIES_OPERATOR_MANIFEST_SHA256: &str =
    "0e834606732821d6f54e770f3c41f01ae6fe546b5a5b04048091fa67bb58a5fe";
const LIVE_SERIES_SCHEDULER_EXTENSION_SHA256: &str =
    "6552ed7d7afd4115baad3406dffbb69e11d7a187af8d3591f6969c985bf64ac2";
const VERTICAL_GRID_SHA256: &str =
    "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56";
const SOURCE_FULL_STATE_MANIFEST_SHA256: &str =
    "02d27058721b2d379f439bf4a4840c666eb879b9954518b518ffc47e0c999a7d";
const SOURCE_PARENT_PLAN_SHA256: &str =
    "14609494f7c87937fcb5ec0db3f214809ed514380560c43beb18183e6eebce17";
const SOURCE_TERRAIN_SHA256: &str =
    "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91";
const SOURCE_ZF_SHA256: &str = "df5aef17965915406fe334600afe2f8f2e1ebb5d07665f71820eaac9bde4128f";
const SOURCE_GEOPOTENTIAL_SHA256: &str =
    "687bc3ebe34c0c07dc0c46418e8e6042f271e1bb1bcd119284b4a27ef80bca8c";
const CHILD_CENTER_LAT_F64_BITS: &str = "4044bb15cda77904";
const CHILD_CENTER_LON_F64_BITS: &str = "c0578c5ebb59704f";
const SOUTHWEST_LAT_F64_BITS: &str = "4044b279efdec50d";
const SOUTHWEST_LON_F64_BITS: &str = "c057921adeefb07a";
const NORTHEAST_LAT_F64_BITS: &str = "4044c3af62e702d3";
const NORTHEAST_LON_F64_BITS: &str = "c057869f8bc20426";
const DX_M_F64_BITS: &str = "4049000000000000";
const DY_M_F64_BITS: &str = "4049000000000000";
const DT_F64_BITS: &str = "3fe0000000000000";
const ZTOP_M_F64_BITS: &str = "40cd4c0000000000";
const MAX_JSON_NBYTES: u64 = 1024 * 1024;
const MAX_SOURCE_NBYTES: u64 = 4 * 1024 * 1024;
const MAX_EXECUTABLE_NBYTES: u64 = 128 * 1024 * 1024;
// One complete GM1 history payload at this module's pinned geometry: sixteen
// float32 words per gridpoint plus the staggered-face and surface remainder
// of 5*ny*nx + nz*ny + nz*nx words.  Nothing larger can be a legitimate
// history here, because the q3d shape check below pins the grid exactly.
const PINNED_HISTORY_PAYLOAD_NBYTES: u64 = 4
    * (16 * (CHILD_VIEW_PACK_NZ * CHILD_VIEW_PACK_NY * CHILD_VIEW_PACK_NX)
        + 5 * (CHILD_VIEW_PACK_NY * CHILD_VIEW_PACK_NX)
        + CHILD_VIEW_PACK_NZ * CHILD_VIEW_PACK_NY
        + CHILD_VIEW_PACK_NZ * CHILD_VIEW_PACK_NX) as u64;
// Defence in depth ahead of the manifest's exact `nbytes`, which is the real
// seal and is checked on the line above each use.  These stay well under the
// generic forecast envelope bound on purpose: this module does not admit an
// arbitrary grid, it pins [6,75,300,300] and refuses anything else, so a
// manifest declaring a multi-gibibyte pair is corrupt or hostile by
// construction.  Taking the generic 16 GiB envelope bound here would let such
// a pair be read whole -- the artifact buffer plus one float32 copy of every
// carrier -- before the pinned q3d shape refused it, for a capacity this
// module can never use.
const MAX_HISTORY_NBYTES: u64 = 1024 * 1024 * 1024;
const MAX_CHECKPOINT_NBYTES: u64 = 2 * 1024 * 1024 * 1024;
const _: () = assert!(
    MAX_HISTORY_NBYTES > PINNED_HISTORY_PAYLOAD_NBYTES,
    "the child view pack history gate must admit one complete history at its own pinned geometry"
);
const _: () = assert!(
    MAX_CHECKPOINT_NBYTES > MAX_HISTORY_NBYTES
        && MAX_CHECKPOINT_NBYTES < crate::forecast::MAX_FORECAST_ENVELOPE_BYTES as u64,
    "the child view pack pair gates are narrower than the generic forecast envelope bound"
);
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildViewPackReceipt {
    pub manifest_sha256: String,
    pub payload_sha256: String,
    pub receipt_sha256: String,
    pub source_q3d_sha256: String,
    pub source_qc_sha256: String,
    pub source_qi_sha256: String,
    pub minimum_f32_bits: u32,
    pub maximum_f32_bits: u32,
    pub nonzero_count: usize,
    pub step: u64,
    pub model_time_f64_bits: String,
}

#[derive(Clone, Debug)]
struct SourceIdentity {
    profile: SourcePairProfile,
    pair_manifest_sha256: String,
    pair_receipt_sha256: String,
    history_artifact_sha256: String,
    history_contract_sha256: String,
    history_payload_sha256: String,
    history_raw_sha256: String,
    operator_manifest_sha256: String,
    q3d_sha256: String,
    qc_sha256: String,
    qi_sha256: String,
    step: u64,
    model_time_f64_bits: String,
}

#[derive(Clone, Debug)]
struct PackBuild {
    contract: ForecastContract,
    restored: RestoredForecast,
    payload: Vec<u8>,
    payload_sha256: String,
    minimum_f32_bits: u32,
    maximum_f32_bits: u32,
    nonzero_count: usize,
    source: SourceIdentity,
    timings: PackValidationTimings,
}

#[derive(Clone, Copy, Debug)]
struct PackValidationTimings {
    metadata: Duration,
    checkpoint: Duration,
    history: Duration,
    parallel_pair: Duration,
    derive: Duration,
    total: Duration,
}

fn require_real_v1_profile(build: &PackBuild) -> Result<()> {
    if build.source.profile != SourcePairProfile::ProductionRealChild {
        return Err(invalid(
            "legacy qc+qi v1 coordinates are production-real-child-only; idealized sources require the profile-neutral production-view v2 pack",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn publish_child_view_pack_new(
    pair_directory: impl AsRef<Path>,
    expected_pair_manifest_sha256: &str,
    expected_pair_receipt_sha256: &str,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    producer_commit: &str,
    output_directory: impl AsRef<Path>,
) -> Result<ChildViewPackReceipt> {
    let operation_started = Instant::now();
    let pair_directory = pair_directory.as_ref();
    let producer_source_sha256 = hash_regular_bounded(
        producer_source.as_ref(),
        MAX_SOURCE_NBYTES,
        "child View-pack producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable.as_ref(),
        MAX_EXECUTABLE_NBYTES,
        "child View-pack producer executable",
    )?;
    validate_commit(producer_commit)?;
    let build = build_pack(
        pair_directory,
        expected_pair_manifest_sha256,
        expected_pair_receipt_sha256,
    )?;
    require_real_v1_profile(&build)?;
    let manifest_bytes = manifest_value(
        &build,
        &producer_source_sha256,
        &producer_executable_sha256,
        producer_commit,
    )
    .canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let receipt_bytes = receipt_value(
        &build,
        &manifest_sha256,
        &producer_source_sha256,
        &producer_executable_sha256,
        producer_commit,
    )
    .canonical_bytes();
    let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    let terminal_started = Instant::now();
    publish_directory(
        output_directory.as_ref(),
        &build.payload,
        &manifest_bytes,
        &receipt_bytes,
    )?;
    report_stage_timings(
        "publish",
        &build.timings,
        "atomic_publish",
        terminal_started.elapsed(),
        operation_started.elapsed(),
    );
    Ok(ChildViewPackReceipt {
        manifest_sha256,
        payload_sha256: build.payload_sha256,
        receipt_sha256,
        source_q3d_sha256: build.source.q3d_sha256,
        source_qc_sha256: build.source.qc_sha256,
        source_qi_sha256: build.source.qi_sha256,
        minimum_f32_bits: build.minimum_f32_bits,
        maximum_f32_bits: build.maximum_f32_bits,
        nonzero_count: build.nonzero_count,
        step: build.source.step,
        model_time_f64_bits: build.source.model_time_f64_bits,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn inspect_child_view_pack(
    pair_directory: impl AsRef<Path>,
    expected_pair_manifest_sha256: &str,
    expected_pair_receipt_sha256: &str,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    producer_commit: &str,
    directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_payload_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<ChildViewPackReceipt> {
    let operation_started = Instant::now();
    for (value, label) in [
        (
            expected_manifest_sha256,
            "expected View-pack manifest SHA-256",
        ),
        (
            expected_payload_sha256,
            "expected View-pack payload SHA-256",
        ),
        (
            expected_receipt_sha256,
            "expected View-pack receipt SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    let producer_source_sha256 = hash_regular_bounded(
        producer_source.as_ref(),
        MAX_SOURCE_NBYTES,
        "child View-pack producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable.as_ref(),
        MAX_EXECUTABLE_NBYTES,
        "child View-pack producer executable",
    )?;
    validate_commit(producer_commit)?;
    let build = build_pack(
        pair_directory.as_ref(),
        expected_pair_manifest_sha256,
        expected_pair_receipt_sha256,
    )?;
    require_real_v1_profile(&build)?;
    let expected_manifest = manifest_value(
        &build,
        &producer_source_sha256,
        &producer_executable_sha256,
        producer_commit,
    )
    .canonical_bytes();
    let actual_manifest_sha256 = sha256::digest_hex(&expected_manifest);
    let expected_receipt = receipt_value(
        &build,
        &actual_manifest_sha256,
        &producer_source_sha256,
        &producer_executable_sha256,
        producer_commit,
    )
    .canonical_bytes();
    let terminal_started = Instant::now();
    let directory = directory.as_ref();
    verify_inventory(directory, &pack_inventory(), "child View-pack")?;
    let payload = read_regular_bounded(
        &directory.join(CHILD_VIEW_PACK_PAYLOAD_FILE),
        CHILD_VIEW_PACK_NBYTES as u64,
        "child View-pack payload",
    )?;
    let manifest = read_regular_bounded(
        &directory.join(CHILD_VIEW_PACK_MANIFEST_FILE),
        MAX_JSON_NBYTES,
        "child View-pack manifest",
    )?;
    let receipt = read_regular_bounded(
        &directory.join(CHILD_VIEW_PACK_RECEIPT_FILE),
        MAX_JSON_NBYTES,
        "child View-pack receipt",
    )?;
    if payload != build.payload || manifest != expected_manifest || receipt != expected_receipt {
        return Err(invalid(
            "child View-pack bytes differ from the independent source recomputation",
        ));
    }
    for (actual, expected, label) in [
        (
            sha256::digest_hex(&manifest),
            expected_manifest_sha256,
            "child View-pack manifest",
        ),
        (
            sha256::digest_hex(&payload),
            expected_payload_sha256,
            "child View-pack payload",
        ),
        (
            sha256::digest_hex(&receipt),
            expected_receipt_sha256,
            "child View-pack receipt",
        ),
    ] {
        if actual != expected {
            return Err(invalid(format!(
                "{label} SHA-256 is {actual}; expected {expected}"
            )));
        }
    }
    report_stage_timings(
        "inspect",
        &build.timings,
        "pack_compare",
        terminal_started.elapsed(),
        operation_started.elapsed(),
    );
    Ok(ChildViewPackReceipt {
        manifest_sha256: actual_manifest_sha256,
        payload_sha256: build.payload_sha256,
        receipt_sha256: sha256::digest_hex(&receipt),
        source_q3d_sha256: build.source.q3d_sha256,
        source_qc_sha256: build.source.qc_sha256,
        source_qi_sha256: build.source.qi_sha256,
        minimum_f32_bits: build.minimum_f32_bits,
        maximum_f32_bits: build.maximum_f32_bits,
        nonzero_count: build.nonzero_count,
        step: build.source.step,
        model_time_f64_bits: build.source.model_time_f64_bits,
    })
}

fn build_pack(
    pair_directory: &Path,
    expected_pair_manifest_sha256: &str,
    expected_pair_receipt_sha256: &str,
) -> Result<PackBuild> {
    let total_started = Instant::now();
    validate_sha256(
        expected_pair_manifest_sha256,
        "expected source pair manifest SHA-256",
    )?;
    validate_sha256(
        expected_pair_receipt_sha256,
        "expected source pair receipt SHA-256",
    )?;
    verify_inventory(pair_directory, &pair_inventory(), "source pair")?;
    let manifest_bytes = read_regular_bounded(
        &pair_directory.join("manifest.json"),
        MAX_JSON_NBYTES,
        "source pair manifest",
    )?;
    let receipt_bytes = read_regular_bounded(
        &pair_directory.join("receipt.json"),
        MAX_JSON_NBYTES,
        "source pair receipt",
    )?;
    if sha256::digest_hex(&manifest_bytes) != expected_pair_manifest_sha256
        || sha256::digest_hex(&receipt_bytes) != expected_pair_receipt_sha256
    {
        return Err(invalid("source pair manifest or receipt SHA-256 changed"));
    }
    let manifest = canonical_json(&manifest_bytes, "source pair manifest")?;
    let receipt = canonical_json(&receipt_bytes, "source pair receipt")?;
    let profile = SourcePairProfile::from_envelopes(
        &string_member(&manifest, "schema", "source pair manifest")?,
        &string_member(&manifest, "status", "source pair manifest")?,
        &string_member(&receipt, "schema", "source pair receipt")?,
        &string_member(&receipt, "status", "source pair receipt")?,
    )?;
    let candidate_checkpoint = member(&manifest, "checkpoint", "source pair manifest")?;
    let candidate_history = member(&manifest, "history", "source pair manifest")?;
    let candidate_checkpoint_contract = string_member(
        candidate_checkpoint,
        "contract_sha256",
        "source pair checkpoint",
    )?;
    let candidate_history_contract =
        string_member(candidate_history, "contract_sha256", "source pair history")?;
    let candidate_operator = string_member(
        &manifest,
        "operator_manifest_sha256",
        "source pair manifest",
    )?;
    let candidate_scheduler = string_member(
        &manifest,
        "scheduler_extension_sha256",
        "source pair manifest",
    )?;
    exact_keys(
        &manifest,
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
            profile.source_key(),
            "status",
            "step",
        ],
        "source pair manifest",
    )?;
    fixed_string(
        &manifest,
        "schema",
        profile.manifest_schema(),
        "source pair manifest",
    )?;
    fixed_string(
        &manifest,
        "status",
        profile.status(),
        "source pair manifest",
    )?;
    fixed_string(&manifest, "model", "GM1", "source pair manifest")?;
    fixed_string(
        &manifest,
        "claim_limit",
        profile.claim_limit(),
        "source pair manifest",
    )?;
    fixed_string(
        &receipt,
        "schema",
        profile.receipt_schema(),
        "source pair receipt",
    )?;
    fixed_string(&receipt, "status", profile.status(), "source pair receipt")?;
    fixed_string(
        &receipt,
        "manifest_sha256",
        expected_pair_manifest_sha256,
        "source pair receipt",
    )?;
    let step = u64_member(&manifest, "step", "source pair manifest")?;
    if step == 0 || u64_member(&receipt, "step", "source pair receipt")? != step {
        return Err(invalid(
            "source pair step must be positive and agree with its receipt",
        ));
    }
    let configuration = member(&manifest, "configuration", "source pair manifest")?;
    validate_source_pair_configuration(profile, configuration)?;
    let source = member(&manifest, profile.source_key(), "source pair manifest")?;
    if profile == SourcePairProfile::IdealizedSingleSoundingWarmBubble {
        validate_idealized_source_summary(source)?;
    }
    let authority = SourcePairAuthority::for_science_envelope(
        profile,
        expected_pair_manifest_sha256,
        expected_pair_receipt_sha256,
        &candidate_history_contract,
        &candidate_checkpoint_contract,
        &candidate_operator,
        &candidate_scheduler,
    )?;
    if authority == SourcePairAuthority::HistoricalExact && step != HISTORICAL_SOURCE_PAIR.step {
        return Err(invalid("historical source pair step changed"));
    }
    validate_source_pair_summary(profile, authority, source)?;
    let operator_manifest_sha256 = string_member(
        &manifest,
        "operator_manifest_sha256",
        "source pair manifest",
    )?;
    let scheduler_extension_sha256 = string_member(
        &manifest,
        "scheduler_extension_sha256",
        "source pair manifest",
    )?;
    if operator_manifest_sha256 != authority.operator_manifest_sha256()
        || scheduler_extension_sha256 != authority.scheduler_extension_sha256()
    {
        return Err(invalid(
            "source pair operator/scheduler identity is not admitted for this series",
        ));
    }
    let expected_operator_count = match authority {
        SourcePairAuthority::HistoricalExact => 52,
        SourcePairAuthority::IdealizedSingleSoundingWarmBubble => 57,
        SourcePairAuthority::LiveSeries
        | SourcePairAuthority::RestoredSeries
        | SourcePairAuthority::ProductionSeries => 56,
    };
    if u64_member(&manifest, "operator_count", "source pair manifest")? != expected_operator_count {
        return Err(invalid("source pair operator count changed"));
    }
    fixed_string(
        &manifest,
        "mass1_f64_bits",
        if profile == SourcePairProfile::IdealizedSingleSoundingWarmBubble {
            "414e81d1397c6000"
        } else {
            "415bdb8f1a3286e0"
        },
        "source pair manifest",
    )?;
    fixed_string(
        &receipt,
        "operator_manifest_sha256",
        &operator_manifest_sha256,
        "source pair receipt",
    )?;
    fixed_string(
        &receipt,
        "scheduler_extension_sha256",
        &scheduler_extension_sha256,
        "source pair receipt",
    )?;

    let checkpoint = member(&manifest, "checkpoint", "source pair manifest")?;
    exact_keys(
        checkpoint,
        &[
            "artifact_nbytes",
            "artifact_sha256",
            "contract_sha256",
            "file",
            "payload_sha256",
            "raw_arena_sha256",
        ],
        "source pair checkpoint",
    )?;
    fixed_string(
        checkpoint,
        "file",
        "checkpoint.cm1fcp",
        "source pair checkpoint",
    )?;
    fixed_string(
        checkpoint,
        "contract_sha256",
        authority.checkpoint_contract_sha256(),
        "source pair checkpoint",
    )?;
    let checkpoint_artifact_nbytes =
        u64_member(checkpoint, "artifact_nbytes", "source pair checkpoint")?;
    let checkpoint_artifact_sha256 =
        string_member(checkpoint, "artifact_sha256", "source pair checkpoint")?;
    let checkpoint_payload_sha256 =
        string_member(checkpoint, "payload_sha256", "source pair checkpoint")?;
    let checkpoint_raw_sha256 =
        string_member(checkpoint, "raw_arena_sha256", "source pair checkpoint")?;
    for (value, label) in [
        (
            &checkpoint_artifact_sha256,
            "source checkpoint artifact SHA-256",
        ),
        (
            &checkpoint_payload_sha256,
            "source checkpoint payload SHA-256",
        ),
        (
            &checkpoint_raw_sha256,
            "source checkpoint raw-arena SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }

    let history = member(&manifest, "history", "source pair manifest")?;
    exact_keys(
        history,
        &[
            "artifact_nbytes",
            "artifact_sha256",
            "contract_sha256",
            "file",
            "payload_sha256",
            "raw_arena_sha256",
        ],
        "source pair history",
    )?;
    fixed_string(history, "file", "history.cm1fhs", "source pair history")?;
    fixed_string(
        history,
        "contract_sha256",
        authority.history_contract_sha256(),
        "source pair history",
    )?;
    let history_artifact_nbytes = u64_member(history, "artifact_nbytes", "source pair history")?;
    let history_artifact_sha256 = string_member(history, "artifact_sha256", "source pair history")?;
    let history_payload_sha256 = string_member(history, "payload_sha256", "source pair history")?;
    let history_raw_sha256 = string_member(history, "raw_arena_sha256", "source pair history")?;
    for (value, label) in [
        (&history_artifact_sha256, "source history artifact SHA-256"),
        (&history_payload_sha256, "source history payload SHA-256"),
        (&history_raw_sha256, "source history raw-arena SHA-256"),
    ] {
        validate_sha256(value, label)?;
    }
    if authority == SourcePairAuthority::HistoricalExact
        && (history_artifact_sha256 != HISTORICAL_SOURCE_PAIR.history_artifact_sha256
            || history_payload_sha256 != HISTORICAL_SOURCE_PAIR.history_payload_sha256
            || history_raw_sha256 != HISTORICAL_SOURCE_PAIR.history_raw_sha256
            || checkpoint_artifact_sha256 != HISTORICAL_SOURCE_PAIR.checkpoint_artifact_sha256
            || checkpoint_payload_sha256 != HISTORICAL_SOURCE_PAIR.checkpoint_payload_sha256
            || checkpoint_raw_sha256 != HISTORICAL_SOURCE_PAIR.checkpoint_raw_sha256)
    {
        return Err(invalid(
            "historical source pair artifact/payload/raw tuple changed",
        ));
    }
    fixed_string(
        &receipt,
        "checkpoint_artifact_sha256",
        &checkpoint_artifact_sha256,
        "source pair receipt",
    )?;
    fixed_string(
        &receipt,
        "checkpoint_raw_sha256",
        &checkpoint_raw_sha256,
        "source pair receipt",
    )?;
    fixed_string(
        &receipt,
        "history_artifact_sha256",
        &history_artifact_sha256,
        "source pair receipt",
    )?;
    fixed_string(
        &receipt,
        "history_raw_sha256",
        &history_raw_sha256,
        "source pair receipt",
    )?;
    let expected_receipt = object(vec![
        (
            "checkpoint_artifact_sha256",
            string(&checkpoint_artifact_sha256),
        ),
        ("checkpoint_raw_sha256", string(&checkpoint_raw_sha256)),
        ("claim_limit", string(profile.claim_limit())),
        ("history_artifact_sha256", string(&history_artifact_sha256)),
        ("history_raw_sha256", string(&history_raw_sha256)),
        ("manifest_sha256", string(expected_pair_manifest_sha256)),
        ("model", string("GM1")),
        (
            "operator_manifest_sha256",
            string(&operator_manifest_sha256),
        ),
        (
            "scheduler_extension_sha256",
            string(&scheduler_extension_sha256),
        ),
        ("schema", string(profile.receipt_schema())),
        ("status", string(profile.status())),
        ("step", u64_number(step)),
    ]);
    if receipt.canonical_bytes() != expected_receipt.canonical_bytes() {
        return Err(invalid(
            "source pair receipt contains foreign, missing, or mixed authority fields",
        ));
    }

    let checkpoint_path = pair_directory.join("checkpoint.cm1fcp");
    let checkpoint_metadata = fs::symlink_metadata(&checkpoint_path)?;
    if checkpoint_metadata.file_type().is_symlink()
        || !checkpoint_metadata.is_file()
        || checkpoint_metadata.len() != checkpoint_artifact_nbytes
        || checkpoint_metadata.len() > MAX_CHECKPOINT_NBYTES
    {
        return Err(invalid(
            "source checkpoint type/size differs from its bounded manifest",
        ));
    }
    let history_path = pair_directory.join("history.cm1fhs");
    let history_metadata = fs::symlink_metadata(&history_path)?;
    if history_metadata.file_type().is_symlink()
        || !history_metadata.is_file()
        || history_metadata.len() != history_artifact_nbytes
        || history_metadata.len() > MAX_HISTORY_NBYTES
    {
        return Err(invalid(
            "source history type/size differs from its bounded manifest",
        ));
    }
    let metadata_elapsed = total_started.elapsed();
    let parallel_started = Instant::now();
    let (checkpoint_elapsed, (contract, restored, history_elapsed)) = thread::scope(|scope| {
        let checkpoint = scope.spawn(|| -> Result<Duration> {
            let started = Instant::now();
            let (contract, inspected) = inspect_forecast_checkpoint_embedded(
                &checkpoint_path,
                authority.checkpoint_contract_sha256(),
            )?;
            if contract.sha256() != authority.checkpoint_contract_sha256()
                || inspected.receipt().artifact_sha256 != checkpoint_artifact_sha256
                || inspected.receipt().payload_sha256 != checkpoint_payload_sha256
                || inspected.clock().step() != step
                || inspected.clock().dt_s().to_bits() != 0.5f64.to_bits()
                || inspected.array_payload_sha256() != checkpoint_raw_sha256
            {
                return Err(invalid(
                    "source checkpoint embedded identity, raw payload, or cadence changed",
                ));
            }
            Ok(started.elapsed())
        });
        let history = scope.spawn(|| -> Result<_> {
            let started = Instant::now();
            let (contract, restored) =
                read_forecast_history_embedded(&history_path, authority.history_contract_sha256())?;
            if contract.sha256() != authority.history_contract_sha256()
                || restored.receipt().artifact_sha256 != history_artifact_sha256
                || restored.receipt().payload_sha256 != history_payload_sha256
                || restored.frame().clock().step() != step
                || restored.frame().clock().dt_s().to_bits() != 0.5f64.to_bits()
                || restored.array_payload_sha256() != history_raw_sha256
            {
                return Err(invalid(
                    "source history embedded identity, raw payload, or cadence changed",
                ));
            }
            Ok((contract, restored, started.elapsed()))
        });
        let checkpoint = checkpoint
            .join()
            .map_err(|_| invalid("source checkpoint validation worker panicked"))??;
        let history = history
            .join()
            .map_err(|_| invalid("source history validation worker panicked"))??;
        Ok::<_, StateError>((checkpoint, history))
    })?;
    let parallel_pair_elapsed = parallel_started.elapsed();
    let derive_started = Instant::now();
    let q3d_spec = contract
        .arrays()
        .iter()
        .find(|spec| spec.name() == "q3d")
        .ok_or_else(|| invalid("source history contract lacks q3d"))?;
    if q3d_spec.shape()
        != [
            CHILD_VIEW_SPECIES.len(),
            CHILD_VIEW_PACK_NZ,
            CHILD_VIEW_PACK_NY,
            CHILD_VIEW_PACK_NX,
        ]
    {
        return Err(invalid("source q3d is not exact [6,75,300,300] C order"));
    }
    let q3d = restored
        .frame()
        .arrays()
        .iter()
        .find(|array| array.name() == "q3d")
        .ok_or_else(|| invalid("source history payload lacks q3d"))?
        .values();
    if q3d.len() != CHILD_VIEW_Q3D_WORDS {
        return Err(invalid("source q3d word count changed"));
    }
    let q3d_sha256 = hash_f32_words(q3d);
    let qc = &q3d[CHILD_VIEW_PACK_WORDS..2 * CHILD_VIEW_PACK_WORDS];
    let qi = &q3d[3 * CHILD_VIEW_PACK_WORDS..4 * CHILD_VIEW_PACK_WORDS];
    let qc_sha256 = hash_f32_words(qc);
    let qi_sha256 = hash_f32_words(qi);
    let mut payload = Vec::with_capacity(CHILD_VIEW_PACK_NBYTES);
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut nonzero_count = 0usize;
    for word in 0..CHILD_VIEW_PACK_WORDS {
        let qc_value = qc[word];
        let qi_value = qi[word];
        if !qc_value.is_finite() || !qi_value.is_finite() || qc_value < 0.0 || qi_value < 0.0 {
            return Err(invalid(format!(
                "source qc or qi word {word} is non-finite or negative"
            )));
        }
        let value = qc_value + qi_value;
        if !value.is_finite() || value < 0.0 {
            return Err(invalid(format!(
                "derived cloud-condensate word {word} is non-finite or negative"
            )));
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        nonzero_count += usize::from(value.to_bits() & 0x7fff_ffff != 0);
        payload.extend_from_slice(&value.to_le_bytes());
    }
    if payload.len() != CHILD_VIEW_PACK_NBYTES {
        return Err(invalid("derived cloud-condensate byte count changed"));
    }
    let model_time = restored.frame().clock().model_time_s();
    if model_time.to_bits() != (step as f64 * 0.5).to_bits() {
        return Err(invalid("source history model time differs from step*dt"));
    }
    let derive_elapsed = derive_started.elapsed();
    Ok(PackBuild {
        contract,
        restored,
        payload_sha256: sha256::digest_hex(&payload),
        payload,
        minimum_f32_bits: minimum.to_bits(),
        maximum_f32_bits: maximum.to_bits(),
        nonzero_count,
        source: SourceIdentity {
            profile,
            pair_manifest_sha256: expected_pair_manifest_sha256.to_owned(),
            pair_receipt_sha256: expected_pair_receipt_sha256.to_owned(),
            history_artifact_sha256,
            history_payload_sha256,
            history_raw_sha256,
            operator_manifest_sha256,
            history_contract_sha256: authority.history_contract_sha256().to_owned(),
            q3d_sha256,
            qc_sha256,
            qi_sha256,
            step,
            model_time_f64_bits: format!("{:016x}", model_time.to_bits()),
        },
        timings: PackValidationTimings {
            metadata: metadata_elapsed,
            checkpoint: checkpoint_elapsed,
            history: history_elapsed,
            parallel_pair: parallel_pair_elapsed,
            derive: derive_elapsed,
            total: total_started.elapsed(),
        },
    })
}

fn manifest_value(
    build: &PackBuild,
    producer_source_sha256: &str,
    producer_executable_sha256: &str,
    producer_commit: &str,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_VIEW_PACK_ABI)),
        ("claim_limit", string(CHILD_VIEW_PACK_CLAIM_LIMIT)),
        (
            "coordinates",
            object(vec![
                ("center_lat_f64_bits", string(CHILD_CENTER_LAT_F64_BITS)),
                ("center_lon_f64_bits", string(CHILD_CENTER_LON_F64_BITS)),
                ("dx_m_f64_bits", string(DX_M_F64_BITS)),
                ("dy_m_f64_bits", string(DY_M_F64_BITS)),
                (
                    "northeast_mass_lat_f64_bits",
                    string(NORTHEAST_LAT_F64_BITS),
                ),
                (
                    "northeast_mass_lon_f64_bits",
                    string(NORTHEAST_LON_F64_BITS),
                ),
                (
                    "source_parent_plan_sha256",
                    string(SOURCE_PARENT_PLAN_SHA256),
                ),
                (
                    "southwest_mass_lat_f64_bits",
                    string(SOUTHWEST_LAT_F64_BITS),
                ),
                (
                    "southwest_mass_lon_f64_bits",
                    string(SOUTHWEST_LON_F64_BITS),
                ),
                ("vertical_grid_sha256", string(VERTICAL_GRID_SHA256)),
                ("ztop_m_f64_bits", string(ZTOP_M_F64_BITS)),
            ]),
        ),
        ("dtype", string("little-endian-f32")),
        ("formula", string(CHILD_VIEW_FORMULA)),
        ("model", string("GM1")),
        ("order", string("C[k,j,i]")),
        (
            "payload",
            object(vec![
                ("file", string(CHILD_VIEW_PACK_PAYLOAD_FILE)),
                ("nbytes", number(CHILD_VIEW_PACK_NBYTES)),
                ("sha256", string(&build.payload_sha256)),
                (
                    "shape_kji",
                    number_array(&[CHILD_VIEW_PACK_NZ, CHILD_VIEW_PACK_NY, CHILD_VIEW_PACK_NX]),
                ),
            ]),
        ),
        (
            "producer",
            object(vec![
                ("commit", string(producer_commit)),
                ("executable_sha256", string(producer_executable_sha256)),
                ("source_sha256", string(producer_source_sha256)),
            ]),
        ),
        ("quantized", JsonValue::Bool(false)),
        ("schema", string(CHILD_VIEW_PACK_MANIFEST_SCHEMA)),
        ("source", source_value(&build.source)),
        (
            "statistics",
            statistics_value(
                build.minimum_f32_bits,
                build.maximum_f32_bits,
                build.nonzero_count,
            ),
        ),
        (
            "static_identity",
            object(vec![
                (
                    "total_geopotential_zf_sha256",
                    string(SOURCE_GEOPOTENTIAL_SHA256),
                ),
                ("terrain_msl_sha256", string(SOURCE_TERRAIN_SHA256)),
                ("zf_sha256", string(SOURCE_ZF_SHA256)),
            ]),
        ),
        ("status", string(CHILD_VIEW_PACK_STATUS)),
        ("synthetic_detail", JsonValue::Bool(false)),
    ])
}

fn receipt_value(
    build: &PackBuild,
    manifest_sha256: &str,
    producer_source_sha256: &str,
    producer_executable_sha256: &str,
    producer_commit: &str,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_VIEW_PACK_ABI)),
        ("all_finite", JsonValue::Bool(true)),
        ("claim_limit", string(CHILD_VIEW_PACK_CLAIM_LIMIT)),
        ("formula", string(CHILD_VIEW_FORMULA)),
        ("manifest_sha256", string(manifest_sha256)),
        ("model", string("GM1")),
        ("payload_nbytes", number(CHILD_VIEW_PACK_NBYTES)),
        ("payload_sha256", string(&build.payload_sha256)),
        (
            "producer",
            object(vec![
                ("commit", string(producer_commit)),
                ("executable_sha256", string(producer_executable_sha256)),
                ("source_sha256", string(producer_source_sha256)),
            ]),
        ),
        ("quantized", JsonValue::Bool(false)),
        ("schema", string(CHILD_VIEW_PACK_RECEIPT_SCHEMA)),
        ("source", source_value(&build.source)),
        (
            "statistics",
            statistics_value(
                build.minimum_f32_bits,
                build.maximum_f32_bits,
                build.nonzero_count,
            ),
        ),
        ("status", string(CHILD_VIEW_PACK_STATUS)),
        ("synthetic_detail", JsonValue::Bool(false)),
    ])
}

fn source_value(source: &SourceIdentity) -> JsonValue {
    object(vec![
        (
            "history",
            object(vec![
                ("artifact_sha256", string(&source.history_artifact_sha256)),
                ("contract_sha256", string(&source.history_contract_sha256)),
                ("payload_sha256", string(&source.history_payload_sha256)),
                ("raw_arena_sha256", string(&source.history_raw_sha256)),
            ]),
        ),
        ("model_time_f64_bits", string(&source.model_time_f64_bits)),
        (
            "operator_manifest_sha256",
            string(&source.operator_manifest_sha256),
        ),
        ("pair_manifest_sha256", string(&source.pair_manifest_sha256)),
        ("pair_receipt_sha256", string(&source.pair_receipt_sha256)),
        (
            "q3d",
            object(vec![
                ("nbytes", number(CHILD_VIEW_Q3D_NBYTES)),
                ("qc_sha256", string(&source.qc_sha256)),
                ("qi_sha256", string(&source.qi_sha256)),
                ("sha256", string(&source.q3d_sha256)),
                (
                    "shape_species_kji",
                    number_array(&[
                        CHILD_VIEW_SPECIES.len(),
                        CHILD_VIEW_PACK_NZ,
                        CHILD_VIEW_PACK_NY,
                        CHILD_VIEW_PACK_NX,
                    ]),
                ),
                ("species_order", string_array(&CHILD_VIEW_SPECIES)),
            ]),
        ),
        ("step", u64_number(source.step)),
    ])
}

fn statistics_value(minimum: u32, maximum: u32, nonzero_count: usize) -> JsonValue {
    object(vec![
        ("all_finite", JsonValue::Bool(true)),
        ("maximum_f32_bits", string(&format!("{maximum:08x}"))),
        ("minimum_f32_bits", string(&format!("{minimum:08x}"))),
        ("nonnegative", JsonValue::Bool(true)),
        ("nonzero_count", number(nonzero_count)),
        ("word_count", number(CHILD_VIEW_PACK_WORDS)),
    ])
}

fn hash_f32_words(values: &[f32]) -> String {
    let mut hasher = Sha256::default();
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
    sha256::hex(&hasher.finalize())
}

fn report_stage_timings(
    operation: &str,
    timings: &PackValidationTimings,
    terminal_label: &str,
    terminal: Duration,
    operation_total: Duration,
) {
    eprintln!(
        "gm1-child-view-pack: timings operation={operation} metadata_seconds={:.6} checkpoint_worker_seconds={:.6} history_worker_seconds={:.6} parallel_pair_wall_seconds={:.6} derive_seconds={:.6} build_total_seconds={:.6} {terminal_label}_seconds={:.6} operation_total_seconds={:.6}",
        timings.metadata.as_secs_f64(),
        timings.checkpoint.as_secs_f64(),
        timings.history.as_secs_f64(),
        timings.parallel_pair.as_secs_f64(),
        timings.derive.as_secs_f64(),
        timings.total.as_secs_f64(),
        terminal.as_secs_f64(),
        operation_total.as_secs_f64(),
    );
}

fn publish_directory(output: &Path, payload: &[u8], manifest: &[u8], receipt: &[u8]) -> Result<()> {
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace child View-pack destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "child View-pack destination parent does not exist: {}",
            parent.display()
        )));
    }
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("child View-pack destination needs a UTF-8 name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let staging =
        output.with_file_name(format!(".{name}.partial-{}-{sequence}", std::process::id()));
    fs::create_dir(&staging)?;
    let result = (|| -> Result<()> {
        write_new_synced(&staging.join(CHILD_VIEW_PACK_PAYLOAD_FILE), payload)?;
        write_new_synced(&staging.join(CHILD_VIEW_PACK_MANIFEST_FILE), manifest)?;
        write_new_synced(&staging.join(CHILD_VIEW_PACK_RECEIPT_FILE), receipt)?;
        verify_inventory(&staging, &pack_inventory(), "staged child View-pack")?;
        fs::rename(&staging, output)?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        for file in pack_inventory() {
            let _ = fs::remove_file(staging.join(file));
        }
        let _ = fs::remove_dir(&staging);
    }
    result
}

fn pair_inventory() -> BTreeSet<String> {
    BTreeSet::from([
        "checkpoint.cm1fcp".to_owned(),
        "history.cm1fhs".to_owned(),
        "manifest.json".to_owned(),
        "receipt.json".to_owned(),
    ])
}

fn pack_inventory() -> BTreeSet<String> {
    BTreeSet::from([
        CHILD_VIEW_PACK_PAYLOAD_FILE.to_owned(),
        CHILD_VIEW_PACK_MANIFEST_FILE.to_owned(),
        CHILD_VIEW_PACK_RECEIPT_FILE.to_owned(),
    ])
}

fn verify_inventory(directory: &Path, expected: &BTreeSet<String>, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(directory)
        .map_err(|error| invalid(format!("could not inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!("{label} is not a real directory")));
    }
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(invalid(format!("{label} contains a non-regular entry")));
        }
        actual.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| invalid(format!("{label} contains a non-UTF-8 name")))?,
        );
    }
    if &actual != expected {
        return Err(invalid(format!(
            "{label} inventory is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn canonical_json(bytes: &[u8], label: &str) -> Result<JsonValue> {
    let value = json::parse(bytes).map_err(|error| invalid(format!("{label}: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid(format!("{label} is not canonical compact JSON")));
    }
    Ok(value)
}

fn member<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a JsonValue> {
    value
        .get(key)
        .map_err(|error| invalid(format!("{label}: {error}")))
}

fn string_member(value: &JsonValue, key: &str, label: &str) -> Result<String> {
    member(value, key, label)?
        .as_str()
        .map(str::to_owned)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn u64_member(value: &JsonValue, key: &str, label: &str) -> Result<u64> {
    member(value, key, label)?
        .as_u64()
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn fixed_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    if string_member(value, key, label)? != expected {
        return Err(invalid(format!("{label}.{key} differs from {expected:?}")));
    }
    Ok(())
}

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) -> Result<()> {
    let observed = value
        .as_object()
        .map_err(|error| invalid(format!("{label}: {error}")))?
        .iter()
        .map(|(key, _)| key.as_str())
        .collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if observed != expected {
        return Err(invalid(format!(
            "{label} keys are {observed:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn live_series_configuration_value() -> JsonValue {
    object(vec![
        ("admission", string("production-real-50m-child")),
        (
            "domain_shape_zyx",
            number_array(&[CHILD_VIEW_PACK_NZ, CHILD_VIEW_PACK_NY, CHILD_VIEW_PACK_NX]),
        ),
        ("dt_f64_bits", string(DT_F64_BITS)),
        ("dx_m_f64_bits", string(DX_M_F64_BITS)),
        ("dy_m_f64_bits", string(DY_M_F64_BITS)),
        ("equation_set", number(2)),
        ("grid_sha256", string(VERTICAL_GRID_SHA256)),
        ("imove", number(0)),
        ("lateral_boundaries", number_array(&[2, 2, 2, 2])),
        ("microphysics", string("wsm6-mp6")),
        ("psolver", number(3)),
        (
            "stretch_schema",
            string("gpuwm-cm1.stretch-z1-child-15km/v1"),
        ),
        ("terrain", JsonValue::Bool(true)),
        ("ztop_m_f64_bits", string(ZTOP_M_F64_BITS)),
    ])
}

fn idealized_configuration_value() -> JsonValue {
    object(vec![
        (
            "admission",
            string("idealized-isnd7-iinit1-single-sounding-warm-bubble"),
        ),
        (
            "domain_shape_zyx",
            number_array(&[CHILD_VIEW_PACK_NZ, CHILD_VIEW_PACK_NY, CHILD_VIEW_PACK_NX]),
        ),
        ("dt_f64_bits", string(DT_F64_BITS)),
        ("dx_m_f64_bits", string(DX_M_F64_BITS)),
        ("dy_m_f64_bits", string(DY_M_F64_BITS)),
        ("dz_m_f64_bits", string("4069000000000000")),
        ("equation_set", number(2)),
        ("iinit", number(1)),
        ("irst", number(0)),
        ("isnd", number(7)),
        ("lateral_boundaries", number_array(&[2, 2, 2, 2])),
        ("microphysics", string("wsm6-mp6")),
        ("psolver", number(3)),
        ("terrain", string("flat-zero-terrain")),
        ("vertical_grid", string("uniform-200m")),
        ("ztop_m_f64_bits", string(ZTOP_M_F64_BITS)),
    ])
}

fn validate_source_pair_configuration(
    profile: SourcePairProfile,
    configuration: &JsonValue,
) -> Result<()> {
    let expected = match profile {
        SourcePairProfile::ProductionRealChild => live_series_configuration_value(),
        SourcePairProfile::IdealizedSingleSoundingWarmBubble => idealized_configuration_value(),
    };
    if configuration.canonical_bytes() != expected.canonical_bytes() {
        return Err(invalid(format!(
            "source pair configuration differs from the fixed {:?} 300x300x75 science envelope",
            profile
        )));
    }
    Ok(())
}

fn validate_idealized_source_summary(source: &JsonValue) -> Result<()> {
    let label = "idealized source authority summary";
    exact_keys(
        source,
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
    fixed_string(source, "initial_mass_f64_bits", "414e81d1397c6000", label)?;
    fixed_string(
        source,
        "rust_library_sha256",
        "10ba6473422e9cffac8955a6ef68b6d47e0b58571d7550da7250a19dabcd754d",
        label,
    )?;
    fixed_string(source, "schema", IDEALIZED_POST_STEP_SOURCE_SCHEMA, label)?;
    fixed_string(source, "sha256", IDEALIZED_SOURCE_AUTHORITY_SHA256, label)?;
    fixed_string(
        source,
        "sounding_sha256",
        "56df89107f712b4911a7eb53a5c4df637826ee02ca14fe5322eb0cd2916bdea0",
        label,
    )?;
    fixed_string(source, "status", IDEALIZED_POST_STEP_SOURCE_STATUS, label)
}

fn validate_source_pair_summary(
    profile: SourcePairProfile,
    authority: SourcePairAuthority,
    source: &JsonValue,
) -> Result<()> {
    match profile {
        SourcePairProfile::ProductionRealChild => {
            if authority == SourcePairAuthority::IdealizedSingleSoundingWarmBubble {
                return Err(invalid(
                    "idealized authority cannot be relabeled as a production-real source pair",
                ));
            }
            if source.canonical_bytes() != source_full_state_value(authority).canonical_bytes() {
                return Err(invalid("source full-state identity changed"));
            }
            Ok(())
        }
        SourcePairProfile::IdealizedSingleSoundingWarmBubble => {
            if authority != SourcePairAuthority::IdealizedSingleSoundingWarmBubble {
                return Err(invalid(
                    "production-real authority cannot be relabeled as an idealized source pair",
                ));
            }
            validate_idealized_source_summary(source)
        }
    }
}

fn source_full_state_value(authority: SourcePairAuthority) -> JsonValue {
    let directory = match authority {
        SourcePairAuthority::RestoredSeries | SourcePairAuthority::ProductionSeries => "full-state",
        SourcePairAuthority::HistoricalExact | SourcePairAuthority::LiveSeries => {
            "gm1-t300-child-full-state-v1"
        }
        SourcePairAuthority::IdealizedSingleSoundingWarmBubble => {
            return JsonValue::Null;
        }
    };
    object(vec![
        (
            "base_wind_sha256",
            string("adf38848863b1cb36c7918e5617d9f38b81443ddf965bc6358ef7dd60ffaf8dd"),
        ),
        (
            "checkpoint_admission_sha256",
            string("23c0f09129084690c16cae85461d23f79f5645b430a64988242a6bee0119ba1a"),
        ),
        ("directory", string(directory)),
        (
            "first_step_sha256",
            string("132f2f7f877abe9bbfac44900406ceae17c6de93d0bcfc9dacacefe8d993edeb"),
        ),
        ("manifest_sha256", string(SOURCE_FULL_STATE_MANIFEST_SHA256)),
        (
            "receipt_sha256",
            string("792dc1ade597590e985b44c434a05c0c47751390bd5949520abb309527fdebb6"),
        ),
    ])
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn read_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("could not inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        return Err(invalid(format!(
            "{label} is not a bounded real regular file"
        )));
    }
    let mut file = File::open(path)?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)?;
    if bytes.len() as u64 != metadata.len() {
        return Err(invalid(format!("{label} size changed while reading")));
    }
    Ok(bytes)
}

fn hash_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("could not inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        return Err(invalid(format!(
            "{label} is not a bounded real regular file"
        )));
    }
    let mut reader = File::open(path)?;
    let mut hasher = Sha256::default();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut total = 0u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| invalid(format!("{label} byte count overflow")))?;
    }
    if total != metadata.len() {
        return Err(invalid(format!("{label} size changed while hashing")));
    }
    Ok(sha256::hex(&hasher.finalize()))
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!("{label} is not exact lowercase SHA-256")));
    }
    Ok(())
}

fn validate_commit(value: &str) -> Result<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(
            "producer commit must be an exact lowercase 40-hex Git identity",
        ));
    }
    Ok(())
}

fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

fn string(value: &str) -> JsonValue {
    JsonValue::String(value.to_owned())
}

fn number(value: usize) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn u64_number(value: u64) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn number_array(values: &[usize]) -> JsonValue {
    JsonValue::Array(values.iter().map(|value| number(*value)).collect())
}

fn string_array(values: &[&str]) -> JsonValue {
    JsonValue::Array(values.iter().map(|value| string(value)).collect())
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn formula_is_one_source_order_binary32_add() {
        let qc = f32::from_bits(0x3a83_126f);
        let qi = f32::from_bits(0x3980_0041);
        let value = qc + qi;
        assert_eq!(value.to_bits(), 0x3aa3_127f);
        assert_eq!(
            CHILD_VIEW_FORMULA,
            "out[word]=f32(q3d[1*words+word]+q3d[3*words+word])"
        );
    }

    #[test]
    fn atomic_pack_directory_refuses_replacement_and_extra_files() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("gm1-child-view-pack-{nonce}"));
        fs::create_dir(&root).unwrap();
        let output = root.join("step-00000001");
        publish_directory(&output, &[0u8; 4], b"{}", b"{}").unwrap();
        verify_inventory(&output, &pack_inventory(), "test pack").unwrap();
        assert!(publish_directory(&output, &[0u8; 4], b"{}", b"{}").is_err());
        let extra = output.join("ready");
        fs::write(&extra, b"bad").unwrap();
        assert!(verify_inventory(&output, &pack_inventory(), "test pack").is_err());
        fs::remove_file(extra).unwrap();
        for file in pack_inventory() {
            fs::remove_file(output.join(file)).unwrap();
        }
        fs::remove_dir(output).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn the_pair_size_gates_stay_narrower_than_the_generic_envelope_bound() {
        // This module admits one pinned geometry, not an arbitrary grid, so its
        // file-size gates are sized for that geometry.  Taking the generic
        // forecast envelope bound here would let a corrupt manifest have a
        // multi-gibibyte file read whole -- the artifact plus one float32 copy
        // of every carrier -- before the pinned [6,75,300,300] q3d shape
        // refused it, for a capacity this module can never use.
        assert_eq!(MAX_HISTORY_NBYTES, 1024 * 1024 * 1024);
        assert_eq!(MAX_CHECKPOINT_NBYTES, 2 * 1024 * 1024 * 1024);
        // 6.75M gridpoints: one complete history of it is well under the gate.
        assert_eq!(
            CHILD_VIEW_PACK_NZ * CHILD_VIEW_PACK_NY * CHILD_VIEW_PACK_NX,
            6_750_000
        );
        assert_eq!(PINNED_HISTORY_PAYLOAD_NBYTES, 433_980_000);
        assert!(PINNED_HISTORY_PAYLOAD_NBYTES < MAX_HISTORY_NBYTES);
        assert!(MAX_CHECKPOINT_NBYTES < crate::forecast::MAX_FORECAST_ENVELOPE_BYTES as u64);
    }

    #[test]
    fn source_species_and_payload_geometry_are_exact() {
        assert_eq!(CHILD_VIEW_SPECIES, ["qv", "qc", "qr", "qi", "qs", "qg"]);
        assert_eq!(CHILD_VIEW_PACK_WORDS, 6_750_000);
        assert_eq!(CHILD_VIEW_PACK_NBYTES, 27_000_000);
        assert_eq!(CHILD_VIEW_Q3D_NBYTES, 162_000_000);
    }

    #[test]
    fn source_pair_authority_retains_fixture_and_routes_science_envelopes() {
        assert_eq!(
            SourcePairAuthority::for_science_envelope(
                SourcePairProfile::ProductionRealChild,
                HISTORICAL_SOURCE_PAIR.pair_manifest_sha256,
                HISTORICAL_SOURCE_PAIR.pair_receipt_sha256,
                HISTORICAL_SOURCE_PAIR.history_contract_sha256,
                HISTORICAL_SOURCE_PAIR.checkpoint_contract_sha256,
                HISTORICAL_SOURCE_PAIR.operator_manifest_sha256,
                HISTORICAL_SOURCE_PAIR.scheduler_extension_sha256,
            )
            .unwrap(),
            SourcePairAuthority::HistoricalExact,
        );
        assert_eq!(
            SourcePairAuthority::for_science_envelope(
                SourcePairProfile::ProductionRealChild,
                &"0".repeat(64),
                &"1".repeat(64),
                LIVE_SERIES_HISTORY_CONTRACT_SHA256,
                LIVE_SERIES_CHECKPOINT_CONTRACT_SHA256,
                LIVE_SERIES_OPERATOR_MANIFEST_SHA256,
                LIVE_SERIES_SCHEDULER_EXTENSION_SHA256,
            )
            .unwrap(),
            SourcePairAuthority::LiveSeries,
        );
        assert_eq!(
            SourcePairAuthority::for_science_envelope(
                SourcePairProfile::ProductionRealChild,
                &"2".repeat(64),
                &"1".repeat(64),
                RESTORED_SERIES_HISTORY_CONTRACT_SHA256,
                RESTORED_SERIES_CHECKPOINT_CONTRACT_SHA256,
                RESTORED_SERIES_OPERATOR_MANIFEST_SHA256,
                LIVE_SERIES_SCHEDULER_EXTENSION_SHA256,
            )
            .unwrap(),
            SourcePairAuthority::RestoredSeries,
        );
        assert_eq!(
            SourcePairAuthority::for_science_envelope(
                SourcePairProfile::ProductionRealChild,
                &"3".repeat(64),
                &"4".repeat(64),
                PRODUCTION_SERIES_HISTORY_CONTRACT_SHA256,
                PRODUCTION_SERIES_CHECKPOINT_CONTRACT_SHA256,
                PRODUCTION_SERIES_OPERATOR_MANIFEST_SHA256,
                LIVE_SERIES_SCHEDULER_EXTENSION_SHA256,
            )
            .unwrap(),
            SourcePairAuthority::ProductionSeries,
        );
        assert!(
            SourcePairAuthority::for_science_envelope(
                SourcePairProfile::ProductionRealChild,
                HISTORICAL_SOURCE_PAIR.pair_manifest_sha256,
                HISTORICAL_SOURCE_PAIR.pair_receipt_sha256,
                LIVE_SERIES_HISTORY_CONTRACT_SHA256,
                LIVE_SERIES_CHECKPOINT_CONTRACT_SHA256,
                LIVE_SERIES_OPERATOR_MANIFEST_SHA256,
                LIVE_SERIES_SCHEDULER_EXTENSION_SHA256,
            )
            .is_err()
        );
        assert!(
            SourcePairAuthority::for_science_envelope(
                SourcePairProfile::ProductionRealChild,
                &"2".repeat(64),
                &"3".repeat(64),
                RESTORED_SERIES_HISTORY_CONTRACT_SHA256,
                LIVE_SERIES_CHECKPOINT_CONTRACT_SHA256,
                RESTORED_SERIES_OPERATOR_MANIFEST_SHA256,
                LIVE_SERIES_SCHEDULER_EXTENSION_SHA256,
            )
            .is_err()
        );
        assert!(
            SourcePairAuthority::for_science_envelope(
                SourcePairProfile::ProductionRealChild,
                &"3".repeat(64),
                &"4".repeat(64),
                PRODUCTION_SERIES_HISTORY_CONTRACT_SHA256,
                RESTORED_SERIES_CHECKPOINT_CONTRACT_SHA256,
                PRODUCTION_SERIES_OPERATOR_MANIFEST_SHA256,
                LIVE_SERIES_SCHEDULER_EXTENSION_SHA256,
            )
            .is_err()
        );
        assert_eq!(
            SourcePairAuthority::LiveSeries.history_contract_sha256(),
            LIVE_SERIES_HISTORY_CONTRACT_SHA256,
        );
        assert_eq!(
            SourcePairAuthority::LiveSeries.operator_manifest_sha256(),
            LIVE_SERIES_OPERATOR_MANIFEST_SHA256,
        );
        assert_eq!(
            SourcePairAuthority::RestoredSeries.history_contract_sha256(),
            RESTORED_SERIES_HISTORY_CONTRACT_SHA256,
        );
        assert_eq!(
            SourcePairAuthority::RestoredSeries.operator_manifest_sha256(),
            RESTORED_SERIES_OPERATOR_MANIFEST_SHA256,
        );
        assert_eq!(
            SourcePairAuthority::ProductionSeries.history_contract_sha256(),
            PRODUCTION_SERIES_HISTORY_CONTRACT_SHA256,
        );
        assert_eq!(
            SourcePairAuthority::ProductionSeries.operator_manifest_sha256(),
            PRODUCTION_SERIES_OPERATOR_MANIFEST_SHA256,
        );
        assert_eq!(
            source_full_state_value(SourcePairAuthority::RestoredSeries)
                .get("directory")
                .and_then(JsonValue::as_str)
                .unwrap(),
            "full-state",
        );
        assert_eq!(
            source_full_state_value(SourcePairAuthority::ProductionSeries)
                .get("directory")
                .and_then(JsonValue::as_str)
                .unwrap(),
            "full-state",
        );
        assert_eq!(
            source_full_state_value(SourcePairAuthority::LiveSeries)
                .get("directory")
                .and_then(JsonValue::as_str)
                .unwrap(),
            "gm1-t300-child-full-state-v1",
        );
    }

    #[test]
    fn idealized_pair_profile_binds_final_readiness_but_not_pair_artifact_hashes() {
        let profile = SourcePairProfile::from_envelopes(
            IDEALIZED_POST_STEP_MANIFEST_SCHEMA,
            IDEALIZED_PAIR_STATUS,
            IDEALIZED_POST_STEP_RECEIPT_SCHEMA,
            IDEALIZED_PAIR_STATUS,
        )
        .unwrap();
        assert_eq!(
            profile,
            SourcePairProfile::IdealizedSingleSoundingWarmBubble
        );
        assert_eq!(profile.source_key(), "source_idealized_authority");
        assert!(
            SourcePairProfile::from_envelopes(
                IDEALIZED_POST_STEP_MANIFEST_SCHEMA,
                IDEALIZED_PAIR_STATUS,
                CHILD_POST_STEP_RECEIPT_SCHEMA,
                REAL_PAIR_STATUS,
            )
            .is_err()
        );
        validate_source_pair_configuration(profile, &idealized_configuration_value()).unwrap();
        let source = object(vec![
            ("initial_mass_f64_bits", string("414e81d1397c6000")),
            (
                "rust_library_sha256",
                string("10ba6473422e9cffac8955a6ef68b6d47e0b58571d7550da7250a19dabcd754d"),
            ),
            ("schema", string(IDEALIZED_POST_STEP_SOURCE_SCHEMA)),
            ("sha256", string(IDEALIZED_SOURCE_AUTHORITY_SHA256)),
            (
                "sounding_sha256",
                string("56df89107f712b4911a7eb53a5c4df637826ee02ca14fe5322eb0cd2916bdea0"),
            ),
            ("status", string(IDEALIZED_POST_STEP_SOURCE_STATUS)),
        ]);
        validate_idealized_source_summary(&source).unwrap();
        assert_eq!(
            SourcePairAuthority::for_science_envelope(
                profile,
                &"0".repeat(64),
                &"1".repeat(64),
                IDEALIZED_HISTORY_CONTRACT_SHA256,
                IDEALIZED_CHECKPOINT_CONTRACT_SHA256,
                IDEALIZED_OPERATOR_MANIFEST_SHA256,
                IDEALIZED_SCHEDULER_EXTENSION_SHA256,
            )
            .unwrap(),
            SourcePairAuthority::IdealizedSingleSoundingWarmBubble,
        );
        let error = SourcePairAuthority::for_science_envelope(
            profile,
            &"0".repeat(64),
            &"1".repeat(64),
            IDEALIZED_HISTORY_CONTRACT_SHA256,
            IDEALIZED_CHECKPOINT_CONTRACT_SHA256,
            &"4".repeat(64),
            IDEALIZED_SCHEDULER_EXTENSION_SHA256,
        )
        .unwrap_err();
        assert!(error.to_string().contains("sealed CPU readiness"));
        assert_eq!(
            IDEALIZED_READINESS_RECEIPT_SHA256,
            "a18a837cd7f8676fc2eb165c0b518c9f9f871957664548289c45bc6a480a7228"
        );
        assert_eq!(
            IDEALIZED_PAIR_PUBLISHER_SHA256,
            "511bc27d685bd28f9a0526e1e32908b5db3800e29f6f63f7aa31e4bc1fadb096"
        );
    }
}
