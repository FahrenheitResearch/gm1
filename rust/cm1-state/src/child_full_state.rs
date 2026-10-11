//! Atomic admission for the complete real 50 m GM1 child initialization.
//!
//! Large numeric publications remain in their independently sealed sibling
//! directories.  This module hashes every referenced file, validates the
//! exact fixed inventory, and atomically publishes a small reference envelope
//! for CUDA initialization, first-step history/render capture, and subsequent
//! completed-large-step checkpoint capture.  The fresh initialization boundary
//! is deliberately not mislabeled as a completed-large-step checkpoint.

use crate::child_artifact::{
    admit_artifact_with_receipt, json_sha256_at, json_u64_at, Anchor, ArtifactSpec, PayloadSpec,
    ReceiptSource,
};
use crate::child_atmosphere::child_atmosphere_arena_nbytes_for_grid;
use crate::child_atmosphere::{
    CHILD_ATMOSPHERE_ABI, CHILD_ATMOSPHERE_ARENA_FILE, CHILD_ATMOSPHERE_GRID_SCHEMA,
    CHILD_ATMOSPHERE_GRID_STATUS, CHILD_ATMOSPHERE_MANIFEST_FILE, CHILD_ATMOSPHERE_RECEIPT_FILE,
    CHILD_ATMOSPHERE_SCHEMA, CHILD_ATMOSPHERE_STATUS,
};
use crate::child_auxiliary::{
    child_auxiliary_base_layout_for_grid, CHILD_AUXILIARY_GEOPOTENTIAL_FILE,
    CHILD_AUXILIARY_LIVE_DENSITY_FILE, CHILD_AUXILIARY_MANIFEST_FILE,
    CHILD_AUXILIARY_MANIFEST_SCHEMA, CHILD_AUXILIARY_RECEIPT_FILE, CHILD_AUXILIARY_RECEIPT_SCHEMA,
    CHILD_AUXILIARY_STATUS, CHILD_AUXILIARY_TERRAIN_OPEN_FILE, CHILD_LIVE_DENSITY_ABI,
    GEOPOTENTIAL_ABI, SOURCE_MASS_Z_SHA256,
};
use crate::child_remap::{
    ChildGridSpec, ACCEPTED_50M_CHILD_GRID, CHILD_NX, CHILD_NY, CHILD_NZ, CHILD_STRETCH_ABI,
    CHILD_STRETCH_GRID_SHA256, CHILD_STRETCH_SCHEMA,
};
use crate::child_static_heights::{
    CHILD_STATIC_HEIGHT_GRID_SCHEMA, CHILD_STATIC_HEIGHT_GRID_STATUS,
};
use crate::child_terrain::{
    CHILD_TERRAIN_GRID_SCHEMA, CHILD_TERRAIN_GRID_STATUS, CHILD_TERRAIN_MANIFEST_FILE,
    CHILD_TERRAIN_PAYLOAD_FILE, CHILD_TERRAIN_RECEIPT_FILE, CHILD_TERRAIN_SCHEMA,
    CHILD_TERRAIN_STATUS,
};
use crate::json::JsonValue;
use crate::product_restart::read_initialized_product_restart_self_anchored;
use crate::sha256::{self, Sha256};
use crate::terrain_admission::TERRAIN_OPEN_CARRIER_ABI;
use crate::{Result, StateError};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHILD_FULL_STATE_SCHEMA: &str = "gm1.real-50m-child-full-state/v1";
pub const CHILD_FULL_STATE_MANIFEST_SCHEMA: &str = "gm1.real-50m-child-full-state-manifest/v1";
pub const CHILD_FIRST_STEP_SCHEMA: &str = "gm1.real-50m-child-first-step/v1";
pub const CHILD_CHECKPOINT_ADMISSION_SCHEMA: &str = "gm1.real-50m-child-checkpoint-admission/v1";
pub const CHILD_FULL_STATE_RECEIPT_SCHEMA: &str = "gm1.real-50m-child-full-state-receipt/v1";
pub const CHILD_FULL_STATE_MANIFEST_FILE: &str = "manifest.json";
pub const CHILD_FIRST_STEP_FILE: &str = "first-step.json";
pub const CHILD_CHECKPOINT_ADMISSION_FILE: &str = "checkpoint-admission.json";
pub const CHILD_FULL_STATE_RECEIPT_FILE: &str = "receipt.json";
pub const CHILD_BASE_WIND_FILE: &str = "base-wind.f32le";
pub const CHILD_BASE_WIND_ABI: &str = "gm1.real-50m-child-terrain-base-wind-f32/v1";
pub const CHILD_FULL_STATE_STATUS: &str = "PASS_REAL_CHILD_READY_FOR_FIRST_LARGE_STEP";
pub const CHILD_FULL_STATE_CLAIM_LIMIT: &str = "exact hash-bound admission of the real source-coordinate 50 m child atmosphere, terrain, stretched-grid/static heights, source-ordered base/open-boundary auxiliary, live density, geopotential, and independently diagnosed fresh closure state; the artifact is a durable atomic reference envelope rather than a numeric repack, the fresh step-zero boundary is not a completed-large-step checkpoint, and no child timestep, uninterrupted/resumed identity, forecast skill, tornado, or performance claim is made";

pub const CHILD_CLOSURE_SCHEMA: &str = "gm1.source-coordinate-child-stretched-closure/v3";
pub const CHILD_CLOSURE_MANIFEST_SCHEMA: &str =
    "gm1.source-coordinate-child-stretched-closure-manifest/v3";
pub const CHILD_CLOSURE_RECEIPT_SCHEMA: &str =
    "gm1.source-coordinate-child-stretched-closure-receipt/v3";
pub const CHILD_CLOSURE_ABI: &str =
    "gm1.source-coordinate-child-stretched-closure-f32le-c-order/v3";
pub const CHILD_CLOSURE_STATUS: &str = "PASS_SOURCE_BACKED_STRETCHED_CHILD_FRESH_CLOSURE";
pub const CHILD_CLOSURE_FILE: &str = "closure.f32le";
pub const CHILD_CLOSURE_MANIFEST_FILE: &str = "manifest.json";
pub const CHILD_CLOSURE_RECEIPT_FILE: &str = "receipt.json";
pub const CHILD_CLOSURE_KERNEL_SHA256: &str =
    "5264deec03cc61fe3e6e96874cd215b6e62e44fa7adc071b4808b9d2b26470ea";
pub const CHILD_CLOSURE_GENERIC_MODULE_SHA256: &str =
    "cc5fcb4b440a35bfe8b3ee6fc976f1f3c2eabccd97ef21aeca58ee5edd17b7e0";
pub const CHILD_CLOSURE_CALCNM_KERNEL_SHA256: &str =
    "40ddf9aa767b997f8e6d592ffcc8e7f0007f91e81bc122f0af192511b0295aac";
pub const CHILD_CLOSURE_STRETCHED_CALCDEF_KERNEL_SHA256: &str =
    "3114457a038061913bf760be0a3a6b24c1e56d753f249d77d224af59a7a987fd";

/// The corrected stretched-z closure this composition admits by default.
///
/// These are the bytes the current oracle published; they are recorded so the
/// envelope can say which corrected closure it was composed from, and so a
/// rerun that reproduces them is recognisable.  They are not the admission
/// gate: any closure artifact that is internally consistent with its own
/// receipt, carries the v3 schema and status, covers the admitted child grid,
/// and is not the discredited uniform-z result is admitted.
pub const CORRECTED_CHILD_CLOSURE_MANIFEST_SHA256: &str =
    "6ff7d5a61d8e7597c46b33887311ada5e1c763f18e2a822f5d5329392eb23ee5";
pub const CORRECTED_CHILD_CLOSURE_ARENA_SHA256: &str =
    "cf8a3731006959a23445ea9af94405d1812235de18144329d0b830dd4c5a8281";
pub const CORRECTED_CHILD_CLOSURE_RECEIPT_SHA256: &str =
    "8b471481978dd3a8cecbe7f324a0f7ad55c93ff6c982e4e5dd3694ccbd6945ed";
pub const CORRECTED_CHILD_CLOSURE_DEFV_SHA256: &str =
    "35c0ce245ef7b647331cf6796b492d81bd1d6da6d63bfe19f21c3e1bce5c4167";
pub const CORRECTED_CHILD_CLOSURE_DEFH_SHA256: &str =
    "930f8c270bdc59e8ba2d6d37301bbfaab79de9d81f3cbe490fc74b14c22f0184";

/// The superseded closure and the uniform-z negative control its deformation
/// carriers were byte-identical to.  Refused by name, with the reason.
pub const REFUSED_UNIFORM_Z_CLOSURE_MANIFEST_SHA256: &str =
    "7aba693cd3a5c984c5a0d969bffa0793f80dfea138aa661da34a86f69982d9d9";
pub const REFUSED_UNIFORM_Z_CLOSURE_ARENA_SHA256: &str =
    "dc277233d7bead2009749e09f402fdff6c072323a57568fef270dbe6904dca6c";
pub const REFUSED_UNIFORM_Z_CLOSURE_RECEIPT_SHA256: &str =
    "9972f6e981dc99e5635a844b70906328339535450bc57cdb01e61469b8080985";
pub const REFUSED_UNIFORM_Z_DEFV_SHA256: &str =
    "d7f044de8f9dea821f3f71616d64c5cdc770617ec92c189d8f46936d3dc05a78";
pub const REFUSED_UNIFORM_Z_DEFH_SHA256: &str =
    "0ba40b06156908886c5eb6560e9030c19bc36a37913c3f50f1b84735c948d24d";

const CHILD_BASE_WIND_NBYTES: u64 = 54_180_000;
const CHILD_U0_NBYTES: u64 = 27_090_000;
const CHILD_V0_NBYTES: u64 = 27_090_000;
const JSON_LIMIT: u64 = 2 * 1024 * 1024;
const SOURCE_LIMIT: u64 = 8 * 1024 * 1024;
const EXECUTABLE_LIMIT: u64 = 128 * 1024 * 1024;
const TARGET_DT_F64_BITS: &str = "3fe0000000000000";
const ZERO_F64_BITS: &str = "0000000000000000";
const AUTHORITY_TARBALL_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

const CLOSURE_CARRIERS: [(&str, &[usize]); 13] = [
    ("nm", &[76, 300, 300]),
    ("defv", &[76, 300, 300]),
    ("defh", &[76, 300, 300]),
    ("grdscl", &[75, 300, 300]),
    ("lenscl", &[76, 300, 300]),
    ("dissten", &[76, 300, 300]),
    ("kmh", &[76, 300, 300]),
    ("kmv", &[76, 300, 300]),
    ("khh", &[76, 300, 300]),
    ("khv", &[76, 300, 300]),
    ("tkea", &[76, 300, 300]),
    ("tke3d", &[76, 300, 300]),
    ("tketen", &[76, 300, 300]),
];

const WSM_PERSISTENCE: [&str; 15] = [
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "held.qcten",
    "component.h_diabatic",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_sr",
];
const WSM_HELD_ZERO: [&str; 4] = ["held.qpten", "held.qtten", "held.qvten", "held.qcten"];
const WSM_COMPONENT_PERSISTENCE: [&str; 11] = [
    "component.h_diabatic",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_sr",
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileSeal {
    name: String,
    nbytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VerifiedArtifact {
    role: &'static str,
    directory: String,
    schema: String,
    abi: String,
    status: String,
    manifest_sha256: String,
    receipt_sha256: String,
    files: Vec<FileSeal>,
}

impl VerifiedArtifact {
    fn from_admitted(
        admitted: &crate::child_artifact::AdmittedArtifact,
        directory: String,
    ) -> Self {
        Self {
            role: admitted.role,
            directory,
            schema: admitted.schema.clone(),
            abi: admitted
                .abi
                .clone()
                .unwrap_or_else(|| admitted.schema.clone()),
            status: admitted.status.clone(),
            manifest_sha256: admitted.manifest_sha256.clone(),
            receipt_sha256: admitted.receipt_sha256.clone(),
            files: admitted
                .files
                .iter()
                .map(|file| FileSeal {
                    name: file.name.clone(),
                    nbytes: file.nbytes,
                    sha256: file.sha256.clone(),
                })
                .collect(),
        }
    }
}

/// The one parent every child producer in a composition must name.
#[derive(Clone, Debug, Eq, PartialEq)]
struct BoundParent {
    checkpoint_sha256: String,
    plan_sha256: String,
}

#[derive(Clone, Debug)]
struct Prepared {
    atmosphere: VerifiedArtifact,
    auxiliary: VerifiedArtifact,
    terrain: VerifiedArtifact,
    static_heights: VerifiedArtifact,
    closure: VerifiedArtifact,
    parent: BoundParent,
    base_wind: BaseWind,
    producer_source_sha256: String,
    producer_executable_sha256: String,
}

#[derive(Clone, Debug)]
struct BaseWind {
    bytes: Vec<u8>,
    sha256: String,
    u0_sha256: String,
    v0_sha256: String,
    source_u0_sha256: String,
    source_v0_sha256: String,
    /// The initialization product these profiles were actually read from,
    /// recorded so a reader of the envelope can re-open it.
    source_product_manifest_sha256: String,
    source_product_payload_sha256: String,
    source_product_restart_sha256: String,
}

struct Documents {
    manifest: Vec<u8>,
    first_step: Vec<u8>,
    checkpoint: Vec<u8>,
    receipt: Vec<u8>,
    result: ChildFullStateReceipt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildFullStateReceipt {
    pub base_wind_sha256: String,
    pub manifest_sha256: String,
    pub first_step_sha256: String,
    pub checkpoint_admission_sha256: String,
    pub receipt_sha256: String,
    pub producer_source_sha256: String,
    pub producer_executable_sha256: String,
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[allow(clippy::too_many_arguments)]
pub fn publish_child_full_state_new(
    atmosphere_directory: impl AsRef<Path>,
    auxiliary_directory: impl AsRef<Path>,
    terrain_directory: impl AsRef<Path>,
    static_height_directory: impl AsRef<Path>,
    static_validation_receipt: impl AsRef<Path>,
    source_product_restart: impl AsRef<Path>,
    closure_directory: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
) -> Result<ChildFullStateReceipt> {
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace child full-state destination {}",
            output.display()
        )));
    }
    let output_parent = canonical_output_parent(output)?;
    let prepared = prepare(
        atmosphere_directory.as_ref(),
        auxiliary_directory.as_ref(),
        terrain_directory.as_ref(),
        static_height_directory.as_ref(),
        static_validation_receipt.as_ref(),
        source_product_restart.as_ref(),
        closure_directory.as_ref(),
        producer_source.as_ref(),
        producer_executable.as_ref(),
        &output_parent,
    )?;
    let documents = documents(&prepared)?;
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("child full-state destination needs a UTF-8 name"))?;
    let staging = output.with_file_name(format!(
        ".{name}.partial-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&staging)?;
    let publication = (|| -> Result<()> {
        write_new_synced(
            &staging.join(CHILD_BASE_WIND_FILE),
            &prepared.base_wind.bytes,
        )?;
        write_new_synced(
            &staging.join(CHILD_FULL_STATE_MANIFEST_FILE),
            &documents.manifest,
        )?;
        write_new_synced(&staging.join(CHILD_FIRST_STEP_FILE), &documents.first_step)?;
        write_new_synced(
            &staging.join(CHILD_CHECKPOINT_ADMISSION_FILE),
            &documents.checkpoint,
        )?;
        write_new_synced(
            &staging.join(CHILD_FULL_STATE_RECEIPT_FILE),
            &documents.receipt,
        )?;
        verify_output_inventory(&staging)?;
        fs::rename(&staging, output)?;
        sync_directory(&output_parent)?;
        Ok(())
    })();
    if let Err(error) = publication {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    Ok(documents.result)
}

#[allow(clippy::too_many_arguments)]
pub fn inspect_child_full_state(
    atmosphere_directory: impl AsRef<Path>,
    auxiliary_directory: impl AsRef<Path>,
    terrain_directory: impl AsRef<Path>,
    static_height_directory: impl AsRef<Path>,
    static_validation_receipt: impl AsRef<Path>,
    source_product_restart: impl AsRef<Path>,
    closure_directory: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    artifact_directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_first_step_sha256: &str,
    expected_checkpoint_admission_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<ChildFullStateReceipt> {
    for (value, label) in [
        (expected_manifest_sha256, "child full-state manifest"),
        (expected_first_step_sha256, "child first-step contract"),
        (
            expected_checkpoint_admission_sha256,
            "child checkpoint admission",
        ),
        (expected_receipt_sha256, "child full-state receipt"),
    ] {
        validate_sha256(value, label)?;
    }
    let artifact = artifact_directory.as_ref();
    let output_parent = canonical_existing_parent(artifact)?;
    let prepared = prepare(
        atmosphere_directory.as_ref(),
        auxiliary_directory.as_ref(),
        terrain_directory.as_ref(),
        static_height_directory.as_ref(),
        static_validation_receipt.as_ref(),
        source_product_restart.as_ref(),
        closure_directory.as_ref(),
        producer_source.as_ref(),
        producer_executable.as_ref(),
        &output_parent,
    )?;
    let documents = documents(&prepared)?;
    verify_output_inventory(artifact)?;
    compare_file_exact(
        &artifact.join(CHILD_BASE_WIND_FILE),
        &prepared.base_wind.bytes,
        &prepared.base_wind.sha256,
        "child base wind",
    )?;
    for (name, expected_bytes, expected_sha) in [
        (
            CHILD_FULL_STATE_MANIFEST_FILE,
            documents.manifest.as_slice(),
            expected_manifest_sha256,
        ),
        (
            CHILD_FIRST_STEP_FILE,
            documents.first_step.as_slice(),
            expected_first_step_sha256,
        ),
        (
            CHILD_CHECKPOINT_ADMISSION_FILE,
            documents.checkpoint.as_slice(),
            expected_checkpoint_admission_sha256,
        ),
        (
            CHILD_FULL_STATE_RECEIPT_FILE,
            documents.receipt.as_slice(),
            expected_receipt_sha256,
        ),
    ] {
        let actual = read_regular_bounded(&artifact.join(name), JSON_LIMIT, name)?;
        if actual != expected_bytes || sha256::digest_hex(&actual) != expected_sha {
            return Err(invalid(format!(
                "child full-state {name} differs from exact recomputation or independent pin"
            )));
        }
    }
    Ok(documents.result)
}

// ---------------------------------------------------------------------------
// Receipt-anchored admission of the five independently published producers.
//
// Each spec below names the file inventory an artifact's ABI defines and where
// inside that artifact's own receipt the digests, byte counts and bound
// identities live.  Nothing is pinned to one historical run: a producer rerun
// from a parent at a different completed step publishes different bytes and is
// admitted, provided it is internally consistent and agrees with its siblings.
// ---------------------------------------------------------------------------

const ATMOSPHERE_SPEC: ArtifactSpec = ArtifactSpec {
    role: "atmosphere",
    manifest_file: CHILD_ATMOSPHERE_MANIFEST_FILE,
    receipt_file: CHILD_ATMOSPHERE_RECEIPT_FILE,
    schema_path: &["schema"],
    receipt_schemas: &[CHILD_ATMOSPHERE_SCHEMA, CHILD_ATMOSPHERE_GRID_SCHEMA],
    status_path: &["status"],
    statuses: &[CHILD_ATMOSPHERE_STATUS, CHILD_ATMOSPHERE_GRID_STATUS],
    manifest_sha_path: &["manifest_sha256"],
    abi: Some(Anchor::Receipt(&["abi"])),
    payloads: &[PayloadSpec {
        file: CHILD_ATMOSPHERE_ARENA_FILE,
        sha_path: &["arena_sha256"],
        nbytes_path: Some(&["arena_nbytes"]),
    }],
    parent_checkpoint: Some(Anchor::Receipt(&["parent_checkpoint_sha256"])),
    parent_plan: Some(Anchor::Receipt(&["parent_plan_sha256"])),
};

const AUXILIARY_SPEC: ArtifactSpec = ArtifactSpec {
    role: "auxiliary",
    manifest_file: CHILD_AUXILIARY_MANIFEST_FILE,
    receipt_file: CHILD_AUXILIARY_RECEIPT_FILE,
    schema_path: &["schema"],
    receipt_schemas: &[CHILD_AUXILIARY_RECEIPT_SCHEMA],
    status_path: &["status"],
    statuses: &[CHILD_AUXILIARY_STATUS],
    manifest_sha_path: &["manifest_sha256"],
    abi: Some(Anchor::Receipt(&["artifact_schema"])),
    // The auxiliary receipt is a short digest envelope and declares no byte
    // counts, so there is no receipt path to anchor one to.  These three
    // payloads are instead sized in `require_composed_grid` against the carrier
    // layout `child_auxiliary_base_layout_for_grid` computes for the child this
    // envelope composes, which is stronger than a self-declared count: it is
    // checked against the geometry, not against the producer's own claim.
    payloads: &[
        PayloadSpec {
            file: CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            sha_path: &["terrain_open_sha256"],
            nbytes_path: None,
        },
        PayloadSpec {
            file: CHILD_AUXILIARY_LIVE_DENSITY_FILE,
            sha_path: &["live_density_sha256"],
            nbytes_path: None,
        },
        PayloadSpec {
            file: CHILD_AUXILIARY_GEOPOTENTIAL_FILE,
            sha_path: &["geopotential_sha256"],
            nbytes_path: None,
        },
    ],
    // The auxiliary receipt is a short digest envelope; the parent it was
    // derived from is recorded in its manifest, which that receipt anchors.
    parent_checkpoint: Some(Anchor::Manifest(&["parent", "checkpoint_sha256"])),
    parent_plan: Some(Anchor::Manifest(&["parent", "plan_sha256"])),
};

const TERRAIN_SPEC: ArtifactSpec = ArtifactSpec {
    role: "terrain",
    manifest_file: CHILD_TERRAIN_MANIFEST_FILE,
    receipt_file: CHILD_TERRAIN_RECEIPT_FILE,
    schema_path: &["schema"],
    receipt_schemas: &[CHILD_TERRAIN_SCHEMA, CHILD_TERRAIN_GRID_SCHEMA],
    status_path: &["status"],
    statuses: &[CHILD_TERRAIN_STATUS, CHILD_TERRAIN_GRID_STATUS],
    manifest_sha_path: &["manifest_sha256"],
    abi: Some(Anchor::Receipt(&["abi"])),
    payloads: &[PayloadSpec {
        file: CHILD_TERRAIN_PAYLOAD_FILE,
        sha_path: &["payload_sha256"],
        nbytes_path: Some(&["payload_nbytes"]),
    }],
    parent_checkpoint: Some(Anchor::Receipt(&["parent_checkpoint_sha256"])),
    parent_plan: Some(Anchor::Receipt(&["parent_plan_sha256"])),
};

/// The original 300/50 static-height export, anchored by the separately
/// committed CUDA validation receipt that records its manifest digest and both
/// payload digests and byte counts.
const STATIC_LEGACY_SPEC: ArtifactSpec = ArtifactSpec {
    role: "static_heights",
    manifest_file: "manifest.json",
    receipt_file: "receipt.json",
    schema_path: &["schema"],
    receipt_schemas: &[STATIC_VALIDATION_SCHEMA],
    status_path: &["status"],
    statuses: &[STATIC_VALIDATION_STATUS],
    manifest_sha_path: &["static_height_export", "manifest_sha256"],
    abi: Some(Anchor::Receipt(&["static_height_export", "abi"])),
    payloads: &[
        PayloadSpec {
            file: "zh.f32le",
            sha_path: &["static_height_export", "payloads", "zh", "sha256"],
            nbytes_path: Some(&["static_height_export", "payloads", "zh", "nbytes"]),
        },
        PayloadSpec {
            file: "zf.f32le",
            sha_path: &["static_height_export", "payloads", "zf", "sha256"],
            nbytes_path: Some(&["static_height_export", "payloads", "zf", "nbytes"]),
        },
    ],
    parent_checkpoint: Some(Anchor::Receipt(&[
        "cuda_static_metrics",
        "launch_receipt",
        "source",
        "parent_checkpoint_sha256",
    ])),
    parent_plan: Some(Anchor::Receipt(&[
        "cuda_static_metrics",
        "launch_receipt",
        "source",
        "parent_plan_sha256",
    ])),
};

/// The grid-generic static-height producer, which publishes its own receipt.
const STATIC_GRID_SPEC: ArtifactSpec = ArtifactSpec {
    role: "static_heights",
    manifest_file: "manifest.json",
    receipt_file: "receipt.json",
    schema_path: &["schema"],
    receipt_schemas: &[CHILD_STATIC_HEIGHT_GRID_SCHEMA],
    status_path: &["status"],
    statuses: &[CHILD_STATIC_HEIGHT_GRID_STATUS],
    manifest_sha_path: &["manifest_sha256"],
    abi: Some(Anchor::Receipt(&["abi"])),
    payloads: &[
        PayloadSpec {
            file: "zh.f32le",
            sha_path: &["payloads", "zh", "sha256"],
            nbytes_path: Some(&["payloads", "zh", "nbytes"]),
        },
        PayloadSpec {
            file: "zf.f32le",
            sha_path: &["payloads", "zf", "sha256"],
            nbytes_path: Some(&["payloads", "zf", "nbytes"]),
        },
    ],
    parent_checkpoint: None,
    parent_plan: None,
};

const CLOSURE_SPEC: ArtifactSpec = ArtifactSpec {
    role: "closure",
    manifest_file: CHILD_CLOSURE_MANIFEST_FILE,
    receipt_file: CHILD_CLOSURE_RECEIPT_FILE,
    schema_path: &["schema"],
    receipt_schemas: &[CHILD_CLOSURE_RECEIPT_SCHEMA],
    status_path: &["status"],
    statuses: &[CHILD_CLOSURE_STATUS],
    manifest_sha_path: &["manifest_sha256"],
    abi: Some(Anchor::Receipt(&["abi"])),
    payloads: &[PayloadSpec {
        file: CHILD_CLOSURE_FILE,
        sha_path: &["payload_sha256"],
        nbytes_path: Some(&["payload_nbytes"]),
    }],
    // The closure is a pure function of the atmosphere, auxiliary and terrain
    // it consumed; its manifest records those three, and `bind_closure_inputs`
    // requires them to be the artifacts admitted here.
    parent_checkpoint: None,
    parent_plan: None,
};

const STATIC_VALIDATION_SCHEMA: &str =
    "gm1.cuda-source-coordinate-50m-child-terrain-metrics-validation/v1";
const STATIC_VALIDATION_STATUS: &str =
    "PASS_REAL_SOURCE_TERRAIN_AND_CUDA_STATIC_METRICS_300X300X75";

#[allow(clippy::too_many_arguments)]
fn prepare(
    atmosphere_directory: &Path,
    auxiliary_directory: &Path,
    terrain_directory: &Path,
    static_height_directory: &Path,
    static_validation_receipt: &Path,
    source_product_restart: &Path,
    closure_directory: &Path,
    producer_source: &Path,
    producer_executable: &Path,
    output_parent: &Path,
) -> Result<Prepared> {
    // Every consumed artifact must be a canonical sibling of the publication,
    // so the envelope can reference it by name and a reader can find it.
    let mut names = Vec::new();
    for (role, directory) in [
        ("atmosphere", atmosphere_directory),
        ("auxiliary", auxiliary_directory),
        ("terrain", terrain_directory),
        ("static_heights", static_height_directory),
        ("closure", closure_directory),
    ] {
        names.push(sibling_directory_name(directory, output_parent, role)?);
    }
    let atmosphere = admit_artifact_with_receipt(
        atmosphere_directory,
        &ATMOSPHERE_SPEC,
        ReceiptSource::InDirectory,
    )?;
    let auxiliary = admit_artifact_with_receipt(
        auxiliary_directory,
        &AUXILIARY_SPEC,
        ReceiptSource::InDirectory,
    )?;
    let terrain =
        admit_artifact_with_receipt(terrain_directory, &TERRAIN_SPEC, ReceiptSource::InDirectory)?;
    let static_heights = if static_height_directory.join("receipt.json").is_file() {
        let admitted = admit_artifact_with_receipt(
            static_height_directory,
            &STATIC_GRID_SPEC,
            ReceiptSource::InDirectory,
        )?;
        bind_static_validation_receipt(static_validation_receipt, &admitted)?;
        admitted
    } else {
        // The original 300/50 export predates the in-directory receipt; its
        // separately committed CUDA validation receipt records the manifest
        // digest and both payload digests and byte counts, so it anchors the
        // directory exactly as an in-directory receipt would.
        admit_artifact_with_receipt(
            static_height_directory,
            &STATIC_LEGACY_SPEC,
            ReceiptSource::External(static_validation_receipt),
        )?
    };
    let closure =
        admit_artifact_with_receipt(closure_directory, &CLOSURE_SPEC, ReceiptSource::InDirectory)?;

    require_composed_grid(
        &atmosphere,
        &auxiliary,
        &terrain,
        &static_heights,
        ACCEPTED_50M_CHILD_GRID,
    )?;
    let parent = agree_on_one_parent(&[&atmosphere, &auxiliary, &terrain, &static_heights])?;
    // The product the parent run was itself initialized from, as the child's
    // own atmosphere producer recorded it.  The base wind below must come from
    // exactly that product.
    let parent_source_product_manifest_sha256 = json_sha256_at(
        &atmosphere.manifest,
        &["parent", "source_product_manifest_sha256"],
        "child atmosphere manifest",
    )?;
    bind_auxiliary_to_atmosphere(&auxiliary, &atmosphere)?;
    bind_static_heights_to_terrain(&static_heights, &terrain)?;
    bind_closure_inputs(&closure, &atmosphere, &auxiliary, &terrain)?;
    refuse_discredited_closure(&closure)?;
    require_closure_grid_and_layout(&closure, ACCEPTED_50M_CHILD_GRID)?;

    let producer_source_sha256 = hash_regular_bounded(
        producer_source,
        SOURCE_LIMIT,
        "child full-state producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable,
        EXECUTABLE_LIMIT,
        "child full-state producer executable",
    )?;
    let zh_record = static_heights
        .file("zh.f32le")
        .ok_or_else(|| invalid("admitted static heights lost their zh record"))?
        .clone();
    let base_wind = build_base_wind(
        source_product_restart,
        static_height_directory,
        &zh_record.sha256,
        zh_record.nbytes,
        &parent_source_product_manifest_sha256,
    )?;
    let [atmosphere_name, auxiliary_name, terrain_name, static_name, closure_name]: [String; 5] =
        names
            .try_into()
            .map_err(|_| invalid("child full-state sibling inventory drifted"))?;
    Ok(Prepared {
        atmosphere: VerifiedArtifact::from_admitted(&atmosphere, atmosphere_name),
        auxiliary: VerifiedArtifact::from_admitted(&auxiliary, auxiliary_name),
        terrain: VerifiedArtifact::from_admitted(&terrain, terrain_name),
        static_heights: VerifiedArtifact::from_admitted(&static_heights, static_name),
        closure: VerifiedArtifact::from_admitted(&closure, closure_name),
        parent,
        base_wind,
        producer_source_sha256,
        producer_executable_sha256,
    })
}

/// Every admitted payload must be sized for the child this envelope composes.
///
/// All seven referenced payloads are covered - the terrain surface, both static
/// height decks, the 28-carrier atmosphere arena and all three auxiliary arenas
/// - plus the base wind this module builds itself, which is written from the
/// admitted zh and checked against `CHILD_U0_NBYTES`/`CHILD_V0_NBYTES` where it
/// is assembled.
///
/// The envelope's base wind, first-step history windows and 37-carrier
/// checkpoint map are all written for one child geometry.  Composing producers
/// built for another would publish an envelope whose carrier extents do not
/// describe the arenas it points at, so every consumer would mmap past the end
/// of one payload and short of another.  The geometry is checked against the
/// bytes the producers actually published, not assumed.
fn require_composed_grid(
    atmosphere: &crate::child_artifact::AdmittedArtifact,
    auxiliary: &crate::child_artifact::AdmittedArtifact,
    terrain: &crate::child_artifact::AdmittedArtifact,
    static_heights: &crate::child_artifact::AdmittedArtifact,
    grid: ChildGridSpec,
) -> Result<()> {
    let words = |levels: usize| -> Result<u64> {
        u64::try_from(levels)
            .ok()
            .and_then(|value| value.checked_mul(grid.ny() as u64))
            .and_then(|value| value.checked_mul(grid.nx() as u64))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| invalid("composed child payload byte count overflowed"))
    };
    for (artifact, file, expected, what) in [
        (
            terrain,
            CHILD_TERRAIN_PAYLOAD_FILE,
            (grid.ny() as u64) * (grid.nx() as u64) * 4,
            "terrain surface",
        ),
        (
            static_heights,
            "zh.f32le",
            words(grid.nz())?,
            "mass heights",
        ),
        (
            static_heights,
            "zf.f32le",
            words(grid.nz() + 1)?,
            "face heights",
        ),
    ] {
        let record = artifact
            .file(file)
            .ok_or_else(|| invalid(format!("admitted {} lost its {file} record", artifact.role)))?;
        if record.nbytes != expected {
            return Err(invalid(format!(
                "admitted {what} payload {file} is {} bytes, but the {}x{}x{} child this envelope composes needs {expected}; the published envelope would describe extents that do not match the arenas it points at",
                record.nbytes,
                grid.nz(),
                grid.ny(),
                grid.nx()
            )));
        }
    }
    // The remaining four payloads are sized by the carrier layouts their own
    // producers publish for a child grid, so they are checked against exactly
    // those layouts rather than against a byte count frozen to one child.
    let auxiliary_layout = child_auxiliary_base_layout_for_grid(grid)?;
    for (artifact, file, expected, what) in [
        (
            atmosphere,
            CHILD_ATMOSPHERE_ARENA_FILE,
            child_atmosphere_arena_nbytes_for_grid(grid)?,
            "28-carrier atmosphere arena",
        ),
        (
            auxiliary,
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            auxiliary_layout.terrain_open_nbytes,
            "50-carrier terrain/open estate",
        ),
        (
            auxiliary,
            CHILD_AUXILIARY_LIVE_DENSITY_FILE,
            auxiliary_layout.live_density_nbytes,
            "live-density estate",
        ),
        (
            auxiliary,
            CHILD_AUXILIARY_GEOPOTENTIAL_FILE,
            auxiliary_layout.geopotential_nbytes,
            "total-geopotential estate",
        ),
    ] {
        let record = artifact
            .file(file)
            .ok_or_else(|| invalid(format!("admitted {} lost its {file} record", artifact.role)))?;
        if record.nbytes != expected {
            return Err(invalid(format!(
                "admitted {what} payload {file} is {} bytes, but the {}x{}x{} child this envelope composes needs {expected}; the published envelope would describe extents that do not match the arenas it points at",
                record.nbytes,
                grid.nz(),
                grid.ny(),
                grid.nx()
            )));
        }
    }
    Ok(())
}

/// Every producer that records a parent must record the same one.
///
/// Without this, an atmosphere interpolated from one parent checkpoint could be
/// composed with terrain, auxiliary state and static heights built from a
/// different parent run.  The child would then start from a state that never
/// existed anywhere: perturbation carriers from one trajectory sitting on a
/// terrain and base state from another.
fn agree_on_one_parent(
    artifacts: &[&crate::child_artifact::AdmittedArtifact],
) -> Result<BoundParent> {
    let mut bound: Option<(BoundParent, &'static str)> = None;
    let mut naming: Vec<&'static str> = Vec::new();
    for artifact in artifacts {
        let (Some(checkpoint), Some(plan)) = (
            artifact.parent_checkpoint_sha256.as_ref(),
            artifact.parent_plan_sha256.as_ref(),
        ) else {
            continue;
        };
        naming.push(artifact.role);
        let candidate = BoundParent {
            checkpoint_sha256: checkpoint.clone(),
            plan_sha256: plan.clone(),
        };
        match bound.as_ref() {
            None => bound = Some((candidate, artifact.role)),
            Some((existing, first)) if *existing != candidate => {
                return Err(invalid(format!(
                    "child {} names parent checkpoint {} / plan {}, but child {} names checkpoint {} / plan {}; composing them would blend two parent runs into one child initialization",
                    artifact.role,
                    candidate.checkpoint_sha256,
                    candidate.plan_sha256,
                    first,
                    existing.checkpoint_sha256,
                    existing.plan_sha256
                )));
            }
            Some(_) => {}
        }
    }
    // One producer naming a parent is not agreement.  The atmosphere, the
    // auxiliary estate and the terrain each record the parent they were built
    // from, so at least those three must name it and name the same one; the
    // static heights and the closure are bound to the terrain and to the three
    // artifacts they consumed instead, by `bind_static_heights_to_terrain` and
    // `bind_closure_inputs`.
    if naming.len() < 2 {
        return Err(invalid(format!(
            "only {} of the composed child's producers recorded the parent checkpoint and validated plan it was built from ({:?}); with fewer than two independent statements there is nothing to agree, and an artifact remapped from another parent run would be composed in unchallenged",
            naming.len(),
            naming
        )));
    }
    bound.map(|(value, _)| value).ok_or_else(|| {
        invalid(
            "no child producer recorded the parent checkpoint and validated plan it was built from; the composed child could not be re-bound to any parent",
        )
    })
}

/// The auxiliary estate is derived from the atmosphere it names.
fn bind_auxiliary_to_atmosphere(
    auxiliary: &crate::child_artifact::AdmittedArtifact,
    atmosphere: &crate::child_artifact::AdmittedArtifact,
) -> Result<()> {
    let arena = atmosphere
        .file(CHILD_ATMOSPHERE_ARENA_FILE)
        .ok_or_else(|| invalid("admitted atmosphere lost its arena record"))?;
    for (path, expected, what) in [
        (
            ["source_atmosphere", "arena_sha256"],
            arena.sha256.as_str(),
            "arena",
        ),
        (
            ["source_atmosphere", "manifest_sha256"],
            atmosphere.manifest_sha256.as_str(),
            "manifest",
        ),
        (
            ["source_atmosphere", "receipt_sha256"],
            atmosphere.receipt_sha256.as_str(),
            "receipt",
        ),
    ] {
        let recorded = json_sha256_at(&auxiliary.manifest, &path, "auxiliary manifest")?;
        if recorded != expected {
            return Err(invalid(format!(
                "auxiliary estate was published from atmosphere {what} {recorded}, but the atmosphere admitted here is {expected}; the child's open-boundary and base-state carriers would not match the perturbation carriers they surround"
            )));
        }
    }
    Ok(())
}

/// The independent CUDA validation receipt must name the heights being composed.
///
/// When the static-height artifact publishes its own receipt, that receipt
/// anchors the payloads; this additionally requires the separately supplied
/// validation receipt - the record that the CUDA metric construction actually
/// passed - to be about the same heights.  Without it the argument would be
/// accepted and ignored, and an unvalidated height export could be composed
/// under a validation record belonging to a different one.
fn bind_static_validation_receipt(
    path: &Path,
    static_heights: &crate::child_artifact::AdmittedArtifact,
) -> Result<()> {
    let (_, receipt) = crate::child_artifact::read_canonical_json(
        path,
        JSON_LIMIT,
        "child static-height validation receipt",
    )?;
    for anchor in [
        ["static_height_export", "manifest_sha256"].as_slice(),
        ["manifest_sha256"].as_slice(),
    ] {
        if let Ok(recorded) =
            json_sha256_at(&receipt, anchor, "child static-height validation receipt")
        {
            if recorded == static_heights.manifest_sha256 {
                return Ok(());
            }
            return Err(invalid(format!(
                "child static-height validation receipt validates manifest {recorded}, but the heights admitted here are {}; the composed child would carry a CUDA metric validation record belonging to a different height export",
                static_heights.manifest_sha256
            )));
        }
    }
    Err(invalid(
        "child static-height validation receipt records no static-height manifest digest; the heights being composed would carry no evidence that their CUDA metric construction ever passed",
    ))
}

/// The static heights are the CUDA metric result of the terrain they name.
fn bind_static_heights_to_terrain(
    static_heights: &crate::child_artifact::AdmittedArtifact,
    terrain: &crate::child_artifact::AdmittedArtifact,
) -> Result<()> {
    let payload = terrain
        .file(CHILD_TERRAIN_PAYLOAD_FILE)
        .ok_or_else(|| invalid("admitted terrain lost its payload record"))?;
    for (path, expected, what) in [
        (
            ["source_terrain", "payload_sha256"],
            payload.sha256.as_str(),
            "payload",
        ),
        (
            ["source_terrain", "manifest_sha256"],
            terrain.manifest_sha256.as_str(),
            "manifest",
        ),
        (
            ["source_terrain", "receipt_sha256"],
            terrain.receipt_sha256.as_str(),
            "receipt",
        ),
    ] {
        let recorded = json_sha256_at(&static_heights.receipt, &path, "static-height receipt")?;
        if recorded != expected {
            return Err(invalid(format!(
                "static heights were derived from terrain {what} {recorded}, but the terrain admitted here is {expected}; every vertical metric, face height and terrain-following coefficient would describe a surface the child does not have"
            )));
        }
    }
    Ok(())
}

/// The closure is a pure function of the three artifacts its manifest names.
fn bind_closure_inputs(
    closure: &crate::child_artifact::AdmittedArtifact,
    atmosphere: &crate::child_artifact::AdmittedArtifact,
    auxiliary: &crate::child_artifact::AdmittedArtifact,
    terrain: &crate::child_artifact::AdmittedArtifact,
) -> Result<()> {
    let arena = atmosphere
        .file(CHILD_ATMOSPHERE_ARENA_FILE)
        .ok_or_else(|| invalid("admitted atmosphere lost its arena record"))?;
    let terrain_payload = terrain
        .file(CHILD_TERRAIN_PAYLOAD_FILE)
        .ok_or_else(|| invalid("admitted terrain lost its payload record"))?;
    let mut expectations: Vec<(&[&str], &str, &str)> = vec![
        (
            &["inputs", "atmosphere", "arena_sha256"],
            arena.sha256.as_str(),
            "atmosphere arena",
        ),
        (
            &["inputs", "atmosphere", "manifest_sha256"],
            atmosphere.manifest_sha256.as_str(),
            "atmosphere manifest",
        ),
        (
            &["inputs", "atmosphere", "receipt_sha256"],
            atmosphere.receipt_sha256.as_str(),
            "atmosphere receipt",
        ),
        (
            &["inputs", "auxiliary", "manifest_sha256"],
            auxiliary.manifest_sha256.as_str(),
            "auxiliary manifest",
        ),
        (
            &["inputs", "auxiliary", "receipt_sha256"],
            auxiliary.receipt_sha256.as_str(),
            "auxiliary receipt",
        ),
        (
            &["inputs", "terrain", "manifest_sha256"],
            terrain.manifest_sha256.as_str(),
            "terrain manifest",
        ),
        (
            &["inputs", "terrain", "payload_sha256"],
            terrain_payload.sha256.as_str(),
            "terrain payload",
        ),
        (
            &["inputs", "terrain", "receipt_sha256"],
            terrain.receipt_sha256.as_str(),
            "terrain receipt",
        ),
    ];
    for (file, key) in [
        (CHILD_AUXILIARY_TERRAIN_OPEN_FILE, "terrain_open_sha256"),
        (CHILD_AUXILIARY_LIVE_DENSITY_FILE, "live_density_sha256"),
        (CHILD_AUXILIARY_GEOPOTENTIAL_FILE, "geopotential_sha256"),
    ] {
        let record = auxiliary
            .file(file)
            .ok_or_else(|| invalid(format!("admitted auxiliary lost its {file} record")))?;
        let path: &'static [&'static str] = match key {
            "terrain_open_sha256" => &["inputs", "auxiliary", "terrain_open_sha256"],
            "live_density_sha256" => &["inputs", "auxiliary", "live_density_sha256"],
            _ => &["inputs", "auxiliary", "geopotential_sha256"],
        };
        expectations.push((path, record.sha256.as_str(), file));
    }
    for (path, expected, what) in expectations {
        let recorded = json_sha256_at(&closure.manifest, path, "closure manifest")?;
        if recorded != expected {
            return Err(invalid(format!(
                "closure was computed from {what} {recorded}, but the {what} admitted here is {expected}; the child would start from subgrid coefficients and TKE diagnosed on a different state than the one it integrates"
            )));
        }
    }
    Ok(())
}

/// Refuse the superseded uniform-z closure, by name and by fingerprint.
///
/// `evidence/validation/gm1-t300-child-stretched-closure-v3/receipt.json`
/// records that the historical closure's `defv` and `defh` carriers are
/// byte-identical to the sealed uniform-z negative control, so they are not the
/// stretched-grid deformation at all; the fresh `km`, `kh` and TKE the child
/// starts from were computed from them.  Composing that artifact would give the
/// child a subgrid closure diagnosed for a vertical grid it does not have,
/// which is exactly the near-ground mixing a tornado-scale vortex depends on.
///
/// The identity check refuses the known artifact; the carrier check refuses any
/// future rerun that reproduces the same defect under a new digest.
fn refuse_discredited_closure(closure: &crate::child_artifact::AdmittedArtifact) -> Result<()> {
    let payload = closure
        .file(CHILD_CLOSURE_FILE)
        .ok_or_else(|| invalid("admitted closure lost its payload record"))?;
    if closure.manifest_sha256 == REFUSED_UNIFORM_Z_CLOSURE_MANIFEST_SHA256
        || closure.receipt_sha256 == REFUSED_UNIFORM_Z_CLOSURE_RECEIPT_SHA256
        || payload.sha256 == REFUSED_UNIFORM_Z_CLOSURE_ARENA_SHA256
    {
        return Err(invalid(
            "the superseded uniform-z child closure is refused: its defv/defh carriers are byte-identical to the sealed uniform-z negative control, so the child's fresh km/kh and TKE were diagnosed for a vertical grid the child does not have; rebuild the closure from the corrected stretched-z oracle",
        ));
    }
    let carriers = closure
        .manifest
        .get("carriers")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("closure manifest carriers: {error}")))?;
    for record in carriers {
        let name = record
            .get("name")
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("closure manifest carrier name: {error}")))?;
        if name != "defv" && name != "defh" {
            continue;
        }
        let digest = record
            .get("sha256")
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("closure manifest carrier sha256: {error}")))?;
        let refused = if name == "defv" {
            REFUSED_UNIFORM_Z_DEFV_SHA256
        } else {
            REFUSED_UNIFORM_Z_DEFH_SHA256
        };
        if digest == refused {
            return Err(invalid(format!(
                "closure carrier {name} is byte-identical to the sealed uniform-z negative control, so this closure did not use the stretched-grid deformation; the child's fresh km/kh and TKE would be diagnosed for a vertical grid it does not have"
            )));
        }
    }
    Ok(())
}

/// The closure must cover the child grid this envelope composes, with the exact
/// thirteen carriers in source order and no gap or overlap in the arena.
fn require_closure_grid_and_layout(
    closure: &crate::child_artifact::AdmittedArtifact,
    grid: ChildGridSpec,
) -> Result<()> {
    for (key, expected) in [("nx", grid.nx()), ("ny", grid.ny()), ("nz", grid.nz())] {
        let recorded = json_u64_at(&closure.manifest, &["grid", key], "closure manifest")?;
        if recorded != expected as u64 {
            return Err(invalid(format!(
                "closure covers {key}={recorded} but the child being composed has {key}={expected}; every subgrid coefficient would be read at an offset belonging to a different column"
            )));
        }
    }
    let carriers = closure
        .manifest
        .get("carriers")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("closure manifest carriers: {error}")))?;
    if carriers.len() != CLOSURE_CARRIERS.len() {
        return Err(invalid(format!(
            "closure publishes {} carriers; the child's first large step reads exactly {}",
            carriers.len(),
            CLOSURE_CARRIERS.len()
        )));
    }
    let mut offset = 0u64;
    for (record, (name, _)) in carriers.iter().zip(CLOSURE_CARRIERS.iter()) {
        let recorded_name = record
            .get("name")
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("closure manifest carrier name: {error}")))?;
        if recorded_name != *name {
            return Err(invalid(format!(
                "closure carrier {} is published where {name} is read; the child would upload one closure field into another's device allocation",
                recorded_name
            )));
        }
        let levels = if *name == "grdscl" {
            grid.nz()
        } else {
            grid.nz() + 1
        };
        let expected_shape = [levels, grid.ny(), grid.nx()];
        let recorded_shape = record
            .get("shape")
            .and_then(JsonValue::as_array)
            .map_err(|error| invalid(format!("closure manifest carrier shape: {error}")))?;
        if recorded_shape.len() != 3 {
            return Err(invalid(format!(
                "closure carrier {name} shape is not rank 3"
            )));
        }
        for (index, extent) in expected_shape.iter().enumerate() {
            let recorded = recorded_shape[index]
                .as_u64()
                .map_err(|error| invalid(format!("closure carrier {name} shape: {error}")))?;
            if recorded != *extent as u64 {
                return Err(invalid(format!(
                    "closure carrier {name} axis {index} is {recorded}; the child reads {extent}"
                )));
            }
        }
        let nbytes = expected_shape
            .iter()
            .try_fold(4u64, |value, extent| value.checked_mul(*extent as u64))
            .ok_or_else(|| invalid("closure carrier size overflowed"))?;
        let recorded_offset = record
            .get("offset_nbytes")
            .and_then(JsonValue::as_u64)
            .map_err(|error| invalid(format!("closure carrier {name} offset: {error}")))?;
        let recorded_nbytes = record
            .get("nbytes")
            .and_then(JsonValue::as_u64)
            .map_err(|error| invalid(format!("closure carrier {name} nbytes: {error}")))?;
        if recorded_offset != offset || recorded_nbytes != nbytes {
            return Err(invalid(format!(
                "closure carrier {name} occupies [{recorded_offset},{recorded_nbytes}) but the child reads [{offset},{nbytes}); the arena would be read with a gap or an overlap"
            )));
        }
        offset += nbytes;
    }
    let payload = closure
        .file(CHILD_CLOSURE_FILE)
        .ok_or_else(|| invalid("admitted closure lost its payload record"))?;
    if payload.nbytes != offset {
        return Err(invalid(format!(
            "closure payload is {} bytes; its own thirteen carriers exhaust {offset}",
            payload.nbytes
        )));
    }
    Ok(())
}

/// Build the child's terrain-staggered base wind from the parent's own
/// initialization product.
///
/// The product is admitted by the identity it records itself (its embedded
/// manifest and receipt are rehashed, its payload is rehashed carrier by
/// carrier), and is then required to be the very product the child's producers
/// name as the initialization of the parent they were built from.  That second
/// requirement is the one that matters scientifically: `u0`/`v0` here become
/// the child's Rayleigh base-state profiles, so a product from a different run
/// would give the child base-state winds from one trajectory underneath
/// perturbation carriers from another - a state that never existed anywhere.
fn build_base_wind(
    source_product_restart: &Path,
    static_height_directory: &Path,
    zh_sha256: &str,
    zh_nbytes: u64,
    parent_source_product_manifest_sha256: &str,
) -> Result<BaseWind> {
    let source = read_initialized_product_restart_self_anchored(source_product_restart)?;
    if source.receipt().product_manifest_sha256 != parent_source_product_manifest_sha256 {
        return Err(invalid(format!(
            "base-wind source is initialized product {}, but this child's producers were built from a parent initialized by product {}; the child's u0/v0 base-state profiles would come from a different run than its perturbation carriers",
            source.receipt().product_manifest_sha256,
            parent_source_product_manifest_sha256
        )));
    }
    let mut raw = source.read_carriers_f32le_exact(&["u0", "v0"])?;
    let u_record = source
        .carrier("u0")
        .ok_or_else(|| invalid("sealed f02 initialized product lacks u0"))?;
    let v_record = source
        .carrier("v0")
        .ok_or_else(|| invalid("sealed f02 initialized product lacks v0"))?;
    for (name, record) in [("u0", u_record), ("v0", v_record)] {
        if record.shape != [CHILD_NZ] || record.nbytes != (CHILD_NZ * 4) as u64 {
            return Err(invalid(format!(
                "sealed f02 source base-wind profile {name} shape drifted"
            )));
        }
    }
    let u_profile = decode_f32_profile(
        raw.remove("u0")
            .ok_or_else(|| invalid("f02 source extraction lost u0"))?,
        "u0",
    )?;
    let v_profile = decode_f32_profile(
        raw.remove("v0")
            .ok_or_else(|| invalid("f02 source extraction lost v0"))?,
        "v0",
    )?;
    let zh_bytes = read_regular_bounded(
        &static_height_directory.join("zh.f32le"),
        zh_nbytes,
        "child static zh",
    )?;
    if zh_bytes.len() as u64 != zh_nbytes || sha256::digest_hex(&zh_bytes) != zh_sha256 {
        return Err(invalid(
            "child static zh changed between admission and base-wind reconstruction; the terrain-following base wind would be interpolated onto heights that are no longer published",
        ));
    }
    let zh = decode_f32_words(&zh_bytes, CHILD_NZ * CHILD_NY * CHILD_NX, "child zh")?;
    let (u0, v0) =
        build_base_wind_values_for_grid(ACCEPTED_50M_CHILD_GRID, &zh, &u_profile, &v_profile)?;
    if u0.len() * 4 != CHILD_U0_NBYTES as usize || v0.len() * 4 != CHILD_V0_NBYTES as usize {
        return Err(invalid("child base-wind output shape drifted"));
    }
    let u_bytes = encode_f32(&u0);
    let v_bytes = encode_f32(&v0);
    let u0_sha256 = sha256::digest_hex(&u_bytes);
    let v0_sha256 = sha256::digest_hex(&v_bytes);
    let mut bytes = Vec::with_capacity(CHILD_BASE_WIND_NBYTES as usize);
    bytes.extend_from_slice(&u_bytes);
    bytes.extend_from_slice(&v_bytes);
    if bytes.len() as u64 != CHILD_BASE_WIND_NBYTES {
        return Err(invalid("child base-wind arena size drifted"));
    }
    let sha256 = sha256::digest_hex(&bytes);
    Ok(BaseWind {
        bytes,
        sha256,
        u0_sha256,
        v0_sha256,
        source_u0_sha256: u_record.sha256.clone(),
        source_v0_sha256: v_record.sha256.clone(),
        source_product_manifest_sha256: source.receipt().product_manifest_sha256.clone(),
        source_product_payload_sha256: source.receipt().payload_sha256.clone(),
        source_product_restart_sha256: source.receipt().artifact_sha256.clone(),
    })
}

pub(crate) fn build_base_wind_values_for_grid(
    grid: ChildGridSpec,
    zh: &[f32],
    u_profile: &[f32],
    v_profile: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    if zh.len() != grid.nz() * grid.ny() * grid.nx()
        || u_profile.len() != grid.nz()
        || v_profile.len() != grid.nz()
    {
        return Err(invalid("child base-wind input geometry is invalid"));
    }
    let mut u0 = Vec::with_capacity(grid.nz() * grid.ny() * (grid.nx() + 1));
    for k in 0..grid.nz() {
        for j in 0..grid.ny() {
            let row = (k * grid.ny() + j) * grid.nx();
            for i in 0..=grid.nx() {
                let height = match i {
                    0 => zh[row],
                    value if value == grid.nx() => zh[row + grid.nx() - 1],
                    _ => f32_mul(0.5, f32_add(zh[row + i - 1], zh[row + i])),
                };
                u0.push(interpolate_source_profile(u_profile, height, "u0")?);
            }
        }
    }
    let mut v0 = Vec::with_capacity(grid.nz() * (grid.ny() + 1) * grid.nx());
    for k in 0..grid.nz() {
        for j in 0..=grid.ny() {
            for i in 0..grid.nx() {
                let height = match j {
                    0 => zh[k * grid.ny() * grid.nx() + i],
                    value if value == grid.ny() => {
                        zh[((k * grid.ny() + grid.ny() - 1) * grid.nx()) + i]
                    }
                    _ => {
                        let south = zh[((k * grid.ny() + j - 1) * grid.nx()) + i];
                        let north = zh[((k * grid.ny() + j) * grid.nx()) + i];
                        f32_mul(0.5, f32_add(south, north))
                    }
                };
                v0.push(interpolate_source_profile(v_profile, height, "v0")?);
            }
        }
    }
    Ok((u0, v0))
}

pub(crate) fn interpolate_source_profile(profile: &[f32], height: f32, label: &str) -> Result<f32> {
    if profile.len() < 2 || !height.is_finite() || height < 0.0 {
        return Err(invalid(format!(
            "{label} source-profile interpolation received invalid input"
        )));
    }
    let first = 100.0_f32;
    let spacing = 200.0_f32;
    let last = first + spacing * (profile.len() as f32 - 1.0);
    if height > last {
        return Err(invalid(format!(
            "child {label} staggered height {height:?} exceeds the sealed source profile support {last:?}"
        )));
    }
    let lower = if height <= first {
        0
    } else {
        (((height - first) / spacing).floor() as usize).min(profile.len() - 2)
    };
    let lower_z = first + spacing * lower as f32;
    let fraction = f32_div(f32_sub(height, lower_z), spacing);
    let delta = f32_sub(profile[lower + 1], profile[lower]);
    let value = f32_add(profile[lower], f32_mul(delta, fraction));
    if !value.is_finite() {
        return Err(invalid(format!(
            "child {label} interpolation produced a non-finite value"
        )));
    }
    Ok(value)
}

fn decode_f32_profile(bytes: Vec<u8>, label: &str) -> Result<Vec<f32>> {
    decode_f32_words(&bytes, CHILD_NZ, label)
}

fn decode_f32_words(bytes: &[u8], words: usize, label: &str) -> Result<Vec<f32>> {
    if bytes.len() != words * 4 {
        return Err(invalid(format!(
            "{label} byte length differs from its shape"
        )));
    }
    let mut values = Vec::with_capacity(words);
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_bits(u32::from_le_bytes(chunk.try_into().expect("four bytes")));
        if !value.is_finite() {
            return Err(invalid(format!(
                "{label} contains non-finite bits {:08x} at word {index}",
                value.to_bits()
            )));
        }
        values.push(value);
    }
    Ok(values)
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

#[inline(always)]
fn f32_add(left: f32, right: f32) -> f32 {
    f32::from_bits((left + right).to_bits())
}

#[inline(always)]
fn f32_sub(left: f32, right: f32) -> f32 {
    f32::from_bits((left - right).to_bits())
}

#[inline(always)]
fn f32_mul(left: f32, right: f32) -> f32 {
    f32::from_bits((left * right).to_bits())
}

#[inline(always)]
fn f32_div(left: f32, right: f32) -> f32 {
    f32::from_bits((left / right).to_bits())
}

fn documents(prepared: &Prepared) -> Result<Documents> {
    let manifest = manifest_json(prepared).canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest);
    let first_step = first_step_json(prepared, &manifest_sha256).canonical_bytes();
    let first_step_sha256 = sha256::digest_hex(&first_step);
    let checkpoint = checkpoint_json(prepared, &manifest_sha256).canonical_bytes();
    let checkpoint_admission_sha256 = sha256::digest_hex(&checkpoint);
    let receipt = receipt_json(
        prepared,
        &manifest_sha256,
        &first_step_sha256,
        &checkpoint_admission_sha256,
    )
    .canonical_bytes();
    let receipt_sha256 = sha256::digest_hex(&receipt);
    let result = ChildFullStateReceipt {
        base_wind_sha256: prepared.base_wind.sha256.clone(),
        manifest_sha256,
        first_step_sha256,
        checkpoint_admission_sha256,
        receipt_sha256,
        producer_source_sha256: prepared.producer_source_sha256.clone(),
        producer_executable_sha256: prepared.producer_executable_sha256.clone(),
    };
    Ok(Documents {
        manifest,
        first_step,
        checkpoint,
        receipt,
        result,
    })
}

fn manifest_json(prepared: &Prepared) -> JsonValue {
    object(vec![
        (
            "artifacts",
            object(vec![
                ("atmosphere", artifact_json(&prepared.atmosphere)),
                ("auxiliary", artifact_json(&prepared.auxiliary)),
                ("closure", artifact_json(&prepared.closure)),
                ("static_heights", artifact_json(&prepared.static_heights)),
                ("terrain", artifact_json(&prepared.terrain)),
            ]),
        ),
        (
            "authority",
            object(vec![
                ("cm1_release", string("CM1 r21.1")),
                ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
            ]),
        ),
        ("base_wind", base_wind_json(&prepared.base_wind)),
        ("claim_limit", string(CHILD_FULL_STATE_CLAIM_LIMIT)),
        ("configuration", configuration_json()),
        (
            "closure_operator",
            object(vec![
                (
                    "calcnm_kernel_sha256",
                    string(CHILD_CLOSURE_CALCNM_KERNEL_SHA256),
                ),
                (
                    "generic_turbsmag_module_sha256",
                    string(CHILD_CLOSURE_GENERIC_MODULE_SHA256),
                ),
                (
                    "generic_turbsmag_kernel_sha256",
                    string(CHILD_CLOSURE_KERNEL_SHA256),
                ),
                (
                    "stretched_calcdef_kernel_sha256",
                    string(CHILD_CLOSURE_STRETCHED_CALCDEF_KERNEL_SHA256),
                ),
                (
                    "superseded_uniform_z_artifact_refused",
                    object(vec![
                        (
                            "arena_sha256",
                            string(REFUSED_UNIFORM_Z_CLOSURE_ARENA_SHA256),
                        ),
                        ("defh_sha256", string(REFUSED_UNIFORM_Z_DEFH_SHA256)),
                        ("defv_sha256", string(REFUSED_UNIFORM_Z_DEFV_SHA256)),
                        (
                            "manifest_sha256",
                            string(REFUSED_UNIFORM_Z_CLOSURE_MANIFEST_SHA256),
                        ),
                        (
                            "receipt_sha256",
                            string(REFUSED_UNIFORM_Z_CLOSURE_RECEIPT_SHA256),
                        ),
                    ]),
                ),
            ]),
        ),
        ("model", string("GM1")),
        (
            "numeric_interfaces",
            object(vec![
                ("atmosphere", string(CHILD_ATMOSPHERE_ABI)),
                (
                    "auxiliary_manifest",
                    string(CHILD_AUXILIARY_MANIFEST_SCHEMA),
                ),
                ("closure", string(CHILD_CLOSURE_ABI)),
                ("geopotential", string(GEOPOTENTIAL_ABI)),
                ("live_density", string(CHILD_LIVE_DENSITY_ABI)),
                ("terrain_open", string(TERRAIN_OPEN_CARRIER_ABI)),
            ]),
        ),
        (
            "parent_provenance",
            object(vec![
                (
                    "checkpoint_sha256",
                    string(&prepared.parent.checkpoint_sha256),
                ),
                ("plan_sha256", string(&prepared.parent.plan_sha256)),
                ("synthetic_atmospheric_data", boolean(false)),
                ("synthetic_translation", boolean(false)),
            ]),
        ),
        (
            "producer",
            object(vec![
                (
                    "executable_sha256",
                    string(&prepared.producer_executable_sha256),
                ),
                ("source_sha256", string(&prepared.producer_source_sha256)),
            ]),
        ),
        ("schema", string(CHILD_FULL_STATE_MANIFEST_SCHEMA)),
        ("state_schema", string(CHILD_FULL_STATE_SCHEMA)),
        ("static_composition", static_composition_json(prepared)),
        ("status", string(CHILD_FULL_STATE_STATUS)),
    ])
}

fn configuration_json() -> JsonValue {
    object(vec![
        ("cadence", string("fresh-child-initialization")),
        (
            "domain_shape_zyx",
            usize_array(&[CHILD_NZ, CHILD_NY, CHILD_NX]),
        ),
        ("dx_m_f64_bits", string("4049000000000000")),
        ("dy_m_f64_bits", string("4049000000000000")),
        ("imove", number(0)),
        ("open_boundaries_wesn", usize_array(&[2, 2, 2, 2])),
        ("stretch_abi", string(CHILD_STRETCH_ABI)),
        ("stretch_grid_sha256", string(CHILD_STRETCH_GRID_SHA256)),
        ("stretch_schema", string(CHILD_STRETCH_SCHEMA)),
        ("target_dt_seconds_f64_bits", string(TARGET_DT_F64_BITS)),
        ("terrain", boolean(true)),
        ("ztop_m_f64_bits", string("40cd4c0000000000")),
    ])
}

fn static_composition_json(prepared: &Prepared) -> JsonValue {
    object(vec![
        ("materialization", string("reconstruct on CUDA from the exact terrain payload and admitted stretch profiles; validate the combined 23-carrier hash before use")),
        ("names", string_array(&[
            "terrain_msl", "zh", "zf", "mh", "rmh", "mf", "rmf", "c1", "c2", "gz",
            "rgz", "gzu", "rgzu", "gzv", "rgzv", "dzdx", "dzdy", "gx", "gxu", "gy",
            "gyv", "rds", "rdsf",
        ])),
        (
            "static_heights_manifest_sha256",
            string(&prepared.static_heights.manifest_sha256),
        ),
        (
            "static_heights_receipt_sha256",
            string(&prepared.static_heights.receipt_sha256),
        ),
    ])
}

fn first_step_json(prepared: &Prepared, manifest_sha256: &str) -> JsonValue {
    let history = [
        source_ref(
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            "pp3d",
            Some(&[1, 76, 1, 301, 1, 301]),
        ),
        source_ref(
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            "th3d",
            Some(&[1, 76, 3, 303, 3, 303]),
        ),
        source_ref("atmosphere", CHILD_ATMOSPHERE_ARENA_FILE, "q3d", None),
        source_ref(
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            "u3d",
            Some(&[1, 76, 3, 303, 3, 304]),
        ),
        source_ref(
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            "v3d",
            Some(&[1, 76, 3, 304, 3, 303]),
        ),
        source_ref(
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            "w3d",
            Some(&[1, 77, 3, 303, 3, 303]),
        ),
        source_ref("auxiliary", CHILD_AUXILIARY_LIVE_DENSITY_FILE, "prs", None),
        source_ref("auxiliary", CHILD_AUXILIARY_LIVE_DENSITY_FILE, "rho", None),
        source_ref("auxiliary", CHILD_AUXILIARY_LIVE_DENSITY_FILE, "rf", None),
        source_ref("terrain", CHILD_TERRAIN_PAYLOAD_FILE, "terrain_msl", None),
        source_ref("static_heights", "zf.f32le", "zf", None),
        source_ref(
            "auxiliary",
            CHILD_AUXILIARY_GEOPOTENTIAL_FILE,
            "total_geopotential_zf",
            None,
        ),
        source_ref("closure", CHILD_CLOSURE_FILE, "tkea", None),
    ];
    object(vec![
        (
            "checkpoint_phase",
            string("not-a-completed-large-step-checkpoint"),
        ),
        (
            "carrier_dispositions",
            object(vec![
                (
                    "a_to_current_raw_bit_pairs",
                    JsonValue::Array(
                        [
                            ["ua", "u3d"],
                            ["va", "v3d"],
                            ["wa", "w3d"],
                            ["ppi", "pp3d"],
                            ["tha", "th3d"],
                            ["qa", "q3d"],
                        ]
                        .into_iter()
                        .map(|pair| string_array(&pair))
                        .collect(),
                    ),
                ),
                (
                    "component_persistence",
                    string_array(&WSM_COMPONENT_PERSISTENCE),
                ),
                ("held_positive_zero", string_array(&WSM_HELD_ZERO)),
                (
                    "ppx_post_restart",
                    string("raw-bit copy of ppi after selected getset psolver=3"),
                ),
            ]),
        ),
        (
            "closure",
            object(vec![
                ("abi", string(CHILD_CLOSURE_ABI)),
                ("carrier_order", closure_layout_json()),
                ("fresh_tke_required", boolean(true)),
                ("tke3d_raw_copy_of_tkea", boolean(true)),
                ("tketen_positive_zero", boolean(true)),
            ]),
        ),
        (
            "cuda_initialization_artifacts",
            string_array(&[
                &prepared.atmosphere.directory,
                &prepared.terrain.directory,
                &prepared.static_heights.directory,
                &prepared.auxiliary.directory,
                &prepared.closure.directory,
            ]),
        ),
        (
            "history_capture",
            object(vec![
                (
                    "capture_from",
                    string("live CUDA carriers after the first complete large step"),
                ),
                (
                    "initial_reference_layout",
                    JsonValue::Array(history.into_iter().collect()),
                ),
                ("source_reference_is_not_evolved_output", boolean(true)),
            ]),
        ),
        (
            "lifecycle",
            object(vec![
                ("completed_large_step", boolean(false)),
                ("completed_large_steps", number(0)),
                ("first_large_step_ready", boolean(true)),
                ("model_time_seconds_f64_bits", string(ZERO_F64_BITS)),
                ("next_stage", number(1)),
                ("open_boundary_histories_initialized", boolean(true)),
                ("step_index", number(0)),
                ("target_dt_seconds_f64_bits", string(TARGET_DT_F64_BITS)),
            ]),
        ),
        ("manifest_sha256", string(manifest_sha256)),
        ("model", string("GM1")),
        (
            "physical_base_views",
            JsonValue::Array(vec![
                source_ref(
                    "auxiliary",
                    CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
                    "pi0",
                    Some(&[1, 76, 3, 303, 3, 303]),
                ),
                source_ref(
                    "auxiliary",
                    CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
                    "th0",
                    Some(&[1, 76, 3, 303, 3, 303]),
                ),
                source_ref(
                    "auxiliary",
                    CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
                    "qv0",
                    Some(&[1, 76, 3, 303, 3, 303]),
                ),
            ]),
        ),
        (
            "render_capture",
            object(vec![
                (
                    "continuous_renderer_input",
                    string("live post-step history, never native-HRRR source pixels"),
                ),
                (
                    "minimum_live_carriers",
                    string_array(&[
                        "pp3d",
                        "th3d",
                        "q3d",
                        "u3d",
                        "v3d",
                        "w3d",
                        "prs",
                        "rho",
                        "rf",
                        "terrain_msl",
                        "zf",
                        "total_geopotential_zf",
                        "closure.tkea",
                    ]),
                ),
                (
                    "wsm6_component_persistence",
                    string_array(&WSM_COMPONENT_PERSISTENCE),
                ),
                ("wsm6_held_positive_zero", string_array(&WSM_HELD_ZERO)),
            ]),
        ),
        (
            "terrain_rayleigh_base_wind",
            object(vec![
                ("abi", string(CHILD_BASE_WIND_ABI)),
                ("file", string(CHILD_BASE_WIND_FILE)),
                ("u0_shape_zyx", usize_array(&[75, 300, 301])),
                ("v0_shape_zyx", usize_array(&[75, 301, 300])),
            ]),
        ),
        ("schema", string(CHILD_FIRST_STEP_SCHEMA)),
        ("status", string("READY_FOR_FIRST_COMPLETE_LARGE_STEP")),
    ])
}

fn checkpoint_json(_prepared: &Prepared, manifest_sha256: &str) -> JsonValue {
    let mut sources = Vec::new();
    for name in ["ua", "u3d"] {
        sources.push(checkpoint_source(
            name,
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            &[1, 76, 3, 303, 3, 304],
        ));
    }
    for name in ["va", "v3d"] {
        sources.push(checkpoint_source(
            name,
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            &[1, 76, 3, 304, 3, 303],
        ));
    }
    for name in ["wa", "w3d"] {
        sources.push(checkpoint_source(
            name,
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            &[1, 77, 3, 303, 3, 303],
        ));
    }
    for name in ["ppi", "pp3d", "ppx"] {
        sources.push(checkpoint_source(
            name,
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            &[1, 76, 1, 301, 1, 301],
        ));
    }
    for name in ["tha", "th3d"] {
        sources.push(checkpoint_source(
            name,
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            &[1, 76, 3, 303, 3, 303],
        ));
    }
    for name in ["qa", "q3d"] {
        sources.push(checkpoint_source(
            name,
            "atmosphere",
            CHILD_ATMOSPHERE_ARENA_FILE,
            &[],
        ));
    }
    for name in ["prs", "rho", "rr", "rf"] {
        sources.push(checkpoint_source(
            name,
            "auxiliary",
            CHILD_AUXILIARY_LIVE_DENSITY_FILE,
            &[],
        ));
    }
    for name in ["radbcw", "radbce", "radbcs", "radbcn"] {
        sources.push(checkpoint_source(
            name,
            "auxiliary",
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
            &[],
        ));
    }
    for name in WSM_PERSISTENCE {
        sources.push(checkpoint_source(
            name,
            "atmosphere",
            CHILD_ATMOSPHERE_ARENA_FILE,
            &[],
        ));
    }
    sources.push(checkpoint_source(
        "closure.tkea",
        "closure",
        CHILD_CLOSURE_FILE,
        &[],
    ));
    sources.sort_by(|left, right| {
        left.get("name")
            .and_then(JsonValue::as_str)
            .unwrap_or_default()
            .cmp(
                right
                    .get("name")
                    .and_then(JsonValue::as_str)
                    .unwrap_or_default(),
            )
    });
    object(vec![
        ("admission_boundary", string("after-finish-cuda-gm1-large-step")),
        ("carrier_abi", string("gm1.forecast-checkpoint-carriers/v1")),
        ("carrier_count", number(sources.len())),
        ("initialization_boundary_admitted_as_checkpoint", boolean(false)),
        ("manifest_sha256", string(manifest_sha256)),
        ("model", string("GM1")),
        ("persistent_sources", JsonValue::Array(sources)),
        ("required_live_cadence", object(vec![
            ("completed_large_steps_minimum", number(1)),
            ("large_step_boundary", boolean(true)),
            ("moist_bridge_flags_all_false", boolean(true)),
            ("next_stage", number(1)),
            ("post_stage_hook_active", boolean(false)),
            ("quiescent_capture", boolean(true)),
            ("scalar_completed", boolean(false)),
            ("scalar_next_nrk", number(1)),
            ("solve1_ready", boolean(false)),
        ])),
        ("schema", string(CHILD_CHECKPOINT_ADMISSION_SCHEMA)),
        ("source_identity", string("bind the complete child full-state manifest SHA-256 plus all nested artifact hashes; never reuse the f02 parent-only checkpoint source tuple")),
        ("status", string("ADMIT_AFTER_FIRST_COMPLETED_LARGE_STEP_ONLY")),
    ])
}

fn receipt_json(
    prepared: &Prepared,
    manifest_sha256: &str,
    first_step_sha256: &str,
    checkpoint_admission_sha256: &str,
) -> JsonValue {
    object(vec![
        (
            "checkpoint_admission_sha256",
            string(checkpoint_admission_sha256),
        ),
        ("base_wind_sha256", string(&prepared.base_wind.sha256)),
        ("claim_limit", string(CHILD_FULL_STATE_CLAIM_LIMIT)),
        ("first_step_sha256", string(first_step_sha256)),
        ("manifest_sha256", string(manifest_sha256)),
        ("model", string("GM1")),
        (
            "producer_executable_sha256",
            string(&prepared.producer_executable_sha256),
        ),
        (
            "producer_source_sha256",
            string(&prepared.producer_source_sha256),
        ),
        ("schema", string(CHILD_FULL_STATE_RECEIPT_SCHEMA)),
        ("status", string(CHILD_FULL_STATE_STATUS)),
    ])
}

fn artifact_json(artifact: &VerifiedArtifact) -> JsonValue {
    object(vec![
        ("abi", string(&artifact.abi)),
        ("directory", string(&artifact.directory)),
        (
            "files",
            JsonValue::Array(
                artifact
                    .files
                    .iter()
                    .map(|file| {
                        object(vec![
                            ("name", string(&file.name)),
                            ("nbytes", number(file.nbytes)),
                            ("sha256", string(&file.sha256)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "reference_rule",
            string("sibling directory; rehash every file before mmap/upload"),
        ),
        ("role", string(artifact.role)),
        ("schema", string(&artifact.schema)),
        ("status", string(&artifact.status)),
    ])
}

fn base_wind_json(base: &BaseWind) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_BASE_WIND_ABI)),
        ("authority", string("CM1 r21.1 base.F:674-723 staggered zu/zv interpolation and solve1.F:413-427 terrain-Rayleigh consumption")),
        ("carriers", JsonValue::Array(vec![
            object(vec![
                ("dtype", string("little-endian-f32")),
                ("name", string("u0")),
                ("nbytes", number(CHILD_U0_NBYTES)),
                ("offset_nbytes", number(0)),
                ("order", string("C")),
                ("sha256", string(&base.u0_sha256)),
                ("shape", usize_array(&[75, 300, 301])),
            ]),
            object(vec![
                ("dtype", string("little-endian-f32")),
                ("name", string("v0")),
                ("nbytes", number(CHILD_V0_NBYTES)),
                ("offset_nbytes", number(CHILD_U0_NBYTES)),
                ("order", string("C")),
                ("sha256", string(&base.v0_sha256)),
                ("shape", usize_array(&[75, 301, 300])),
            ]),
        ])),
        ("file", string(CHILD_BASE_WIND_FILE)),
        ("nbytes", number(CHILD_BASE_WIND_NBYTES)),
        ("sha256", string(&base.sha256)),
        ("source", object(vec![
            ("mass_z_f32_sha256", string(SOURCE_MASS_Z_SHA256)),
            ("product_manifest_sha256", string(&base.source_product_manifest_sha256)),
            ("product_payload_sha256", string(&base.source_product_payload_sha256)),
            ("product_restart_sha256", string(&base.source_product_restart_sha256)),
            ("u0_profile_sha256", string(&base.source_u0_sha256)),
            ("v0_profile_sha256", string(&base.source_v0_sha256)),
        ])),
        ("source_profile_seam", string("the sealed f02 initialized product retains domain-mean u0/v0 at 100,300,...14900 m rather than the original sounding; reconstruct each child staggered terrain face with f32 piecewise-linear interpolation and a first-segment continuation below 100 m, never from live ua/va")),
    ])
}

fn closure_layout_json() -> JsonValue {
    let mut offset = 0u64;
    JsonValue::Array(
        CLOSURE_CARRIERS
            .iter()
            .map(|(name, shape)| {
                let nbytes = shape
                    .iter()
                    .fold(4u64, |value, extent| value * *extent as u64);
                let value = object(vec![
                    ("name", string(*name)),
                    ("nbytes", number(nbytes)),
                    ("offset_nbytes", number(offset)),
                    ("shape", usize_array(shape)),
                ]);
                offset += nbytes;
                value
            })
            .collect(),
    )
}

fn source_ref(
    artifact: &str,
    file: &str,
    carrier: &str,
    physical_slice_exclusive_zyx: Option<&[usize]>,
) -> JsonValue {
    let mut entries = vec![
        ("artifact", string(artifact)),
        ("carrier", string(carrier)),
        ("file", string(file)),
    ];
    if let Some(slice) = physical_slice_exclusive_zyx {
        entries.push(("physical_slice_exclusive_zyx", usize_array(slice)));
    }
    object(entries)
}

fn checkpoint_source(
    name: &str,
    artifact: &str,
    file: &str,
    physical_slice_exclusive_zyx: &[usize],
) -> JsonValue {
    let mut entries = vec![
        ("artifact", string(artifact)),
        ("file", string(file)),
        ("name", string(name)),
    ];
    if !physical_slice_exclusive_zyx.is_empty() {
        entries.push((
            "physical_slice_exclusive_zyx",
            usize_array(physical_slice_exclusive_zyx),
        ));
    }
    object(entries)
}

fn sibling_directory_name(directory: &Path, output_parent: &Path, label: &str) -> Result<String> {
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!(
            "{label} artifact must be a regular non-symlink directory"
        )));
    }
    let canonical = fs::canonicalize(directory)?;
    if canonical.parent() != Some(output_parent) {
        return Err(invalid(format!(
            "{label} artifact must be a sibling of the child full-state publication"
        )));
    }
    canonical
        .file_name()
        .and_then(|value| value.to_str())
        .map(str::to_owned)
        .ok_or_else(|| invalid(format!("{label} artifact needs a UTF-8 directory name")))
}

fn canonical_output_parent(output: &Path) -> Result<PathBuf> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let metadata = fs::symlink_metadata(parent)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(
            "child full-state destination parent must be a regular directory",
        ));
    }
    Ok(fs::canonicalize(parent)?)
}

fn canonical_existing_parent(artifact: &Path) -> Result<PathBuf> {
    let metadata = fs::symlink_metadata(artifact)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(
            "child full-state artifact must be a regular non-symlink directory",
        ));
    }
    artifact
        .canonicalize()?
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| invalid("child full-state artifact has no parent"))
}

fn verify_output_inventory(directory: &Path) -> Result<()> {
    let expected = [
        CHILD_BASE_WIND_FILE,
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
            .ok_or_else(|| invalid("child full-state output contains a non-UTF-8 name"))?
            .to_owned();
        if !entry.file_type()?.is_file() {
            return Err(invalid(
                "child full-state output contains a non-file member",
            ));
        }
        actual.insert(name);
    }
    let expected = expected
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid("child full-state output inventory drifted"));
    }
    Ok(())
}

fn read_regular_bounded(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > limit {
        return Err(invalid(format!(
            "{label} is not a bounded regular non-symlink file"
        )));
    }
    let capacity = usize::try_from(metadata.len())
        .map_err(|_| invalid(format!("{label} is too large for this host")))?;
    let mut bytes = Vec::with_capacity(capacity);
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() != capacity {
        return Err(invalid(format!("{label} size changed while reading")));
    }
    Ok(bytes)
}

fn hash_regular_bounded(path: &Path, limit: u64, label: &str) -> Result<String> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > limit {
        return Err(invalid(format!(
            "{label} is not a bounded regular non-symlink file"
        )));
    }
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut hash = Sha256::default();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut total = 0u64;
    loop {
        let amount = reader.read(&mut buffer)?;
        if amount == 0 {
            break;
        }
        total = total
            .checked_add(amount as u64)
            .ok_or_else(|| invalid(format!("{label} size overflowed")))?;
        if total > limit {
            return Err(invalid(format!("{label} exceeds its byte limit")));
        }
        hash.update(&buffer[..amount]);
    }
    if total != metadata.len() {
        return Err(invalid(format!("{label} changed size while hashing")));
    }
    Ok(sha256::hex(&hash.finalize()))
}

fn compare_file_exact(
    path: &Path,
    expected: &[u8],
    expected_sha256: &str,
    label: &str,
) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != expected.len() as u64
    {
        return Err(invalid(format!(
            "{label} type or size differs from recomputation"
        )));
    }
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut hash = Sha256::default();
    let mut offset = 0usize;
    let mut buffer = vec![0u8; 1024 * 1024];
    while offset != expected.len() {
        let amount = (expected.len() - offset).min(buffer.len());
        reader.read_exact(&mut buffer[..amount])?;
        hash.update(&buffer[..amount]);
        if buffer[..amount] != expected[offset..offset + amount] {
            return Err(invalid(format!(
                "{label} differs from exact source-backed recomputation at or before byte {offset}"
            )));
        }
        offset += amount;
    }
    let mut extra = [0u8; 1];
    if reader.read(&mut extra)? != 0 || sha256::hex(&hash.finalize()) != expected_sha256 {
        return Err(invalid(format!(
            "{label} differs from its independently expected SHA-256"
        )));
    }
    Ok(())
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes)?;
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

const fn boolean(value: bool) -> JsonValue {
    JsonValue::Bool(value)
}

fn usize_array(values: &[usize]) -> JsonValue {
    JsonValue::Array(values.iter().copied().map(number).collect())
}

fn string_array(values: &[&str]) -> JsonValue {
    JsonValue::Array(values.iter().copied().map(string).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child_terrain::CHILD_TERRAIN_ABI;
    use std::sync::atomic::{AtomicU64, Ordering as TestOrdering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static TEST_NONCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn admitted_closure_is_the_corrected_stretched_oracle_not_the_refused_uniform_z_one() {
        // The project's own downstream impact map records the superseded
        // artifact as REFUSED_STRETCHED_Z_ORACLE.  Nothing in this module may
        // point at it as the closure to compose.
        assert_eq!(
            CORRECTED_CHILD_CLOSURE_MANIFEST_SHA256,
            "6ff7d5a61d8e7597c46b33887311ada5e1c763f18e2a822f5d5329392eb23ee5"
        );
        assert_eq!(
            CORRECTED_CHILD_CLOSURE_ARENA_SHA256,
            "cf8a3731006959a23445ea9af94405d1812235de18144329d0b830dd4c5a8281"
        );
        assert_eq!(
            CORRECTED_CHILD_CLOSURE_RECEIPT_SHA256,
            "8b471481978dd3a8cecbe7f324a0f7ad55c93ff6c982e4e5dd3694ccbd6945ed"
        );
        for refused in [
            REFUSED_UNIFORM_Z_CLOSURE_MANIFEST_SHA256,
            REFUSED_UNIFORM_Z_CLOSURE_ARENA_SHA256,
            REFUSED_UNIFORM_Z_CLOSURE_RECEIPT_SHA256,
        ] {
            assert_ne!(refused, CORRECTED_CHILD_CLOSURE_MANIFEST_SHA256);
            assert_ne!(refused, CORRECTED_CHILD_CLOSURE_ARENA_SHA256);
            assert_ne!(refused, CORRECTED_CHILD_CLOSURE_RECEIPT_SHA256);
        }
        assert_eq!(
            CHILD_CLOSURE_SCHEMA,
            "gm1.source-coordinate-child-stretched-closure/v3"
        );
        assert_eq!(
            CHILD_CLOSURE_RECEIPT_SCHEMA,
            "gm1.source-coordinate-child-stretched-closure-receipt/v3"
        );
        assert_eq!(
            CHILD_CLOSURE_STATUS,
            "PASS_SOURCE_BACKED_STRETCHED_CHILD_FRESH_CLOSURE"
        );
    }

    #[test]
    fn closure_layout_is_exact_and_exhausts_one_arena() {
        let value = closure_layout_json();
        let records = value.as_array().unwrap();
        assert_eq!(records.len(), 13);
        assert_eq!(records[0].get("name").unwrap().as_str().unwrap(), "nm");
        assert_eq!(records[10].get("name").unwrap().as_str().unwrap(), "tkea");
        let last = records.last().unwrap();
        let offset = last.get("offset_nbytes").unwrap().as_u64().unwrap();
        let nbytes = last.get("nbytes").unwrap().as_u64().unwrap();
        assert_eq!(offset + nbytes, 355_320_000);
    }

    #[test]
    fn checkpoint_map_has_exact_37_unique_carriers_and_is_not_step_zero() {
        let prepared = fixture_prepared();
        let value = checkpoint_json(&prepared, &"3".repeat(64));
        assert_eq!(
            value
                .get("initialization_boundary_admitted_as_checkpoint")
                .unwrap(),
            &JsonValue::Bool(false)
        );
        let sources = value.get("persistent_sources").unwrap().as_array().unwrap();
        assert_eq!(sources.len(), 37);
        let names = sources
            .iter()
            .map(|value| value.get("name").unwrap().as_str().unwrap())
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), 37);
        assert!(names.contains("closure.tkea"));
        assert!(names.contains("radbcw"));
        assert!(names.contains("component.mp_sr"));
    }

    #[test]
    fn first_step_contract_is_fresh_boundary_and_targets_half_second() {
        let prepared = fixture_prepared();
        let value = first_step_json(&prepared, &"3".repeat(64));
        let lifecycle = value.get("lifecycle").unwrap();
        assert_eq!(
            lifecycle.get("completed_large_step").unwrap(),
            &JsonValue::Bool(false)
        );
        assert_eq!(
            lifecycle
                .get("target_dt_seconds_f64_bits")
                .unwrap()
                .as_str()
                .unwrap(),
            TARGET_DT_F64_BITS
        );
    }

    #[test]
    fn published_manifest_records_the_parent_the_producers_named() {
        let prepared = fixture_prepared();
        let value = manifest_json(&prepared);
        let provenance = value.get("parent_provenance").unwrap();
        assert_eq!(
            provenance
                .get("checkpoint_sha256")
                .unwrap()
                .as_str()
                .unwrap(),
            &"ab".repeat(32)
        );
        assert_eq!(
            provenance.get("plan_sha256").unwrap().as_str().unwrap(),
            &"cd".repeat(32)
        );
    }

    // ---- receipt-anchored admission -------------------------------------

    /// The published byte counts of the accepted 50 m child, written out here
    /// rather than recomputed, so this test fails if the sizing rule drifts.
    const ACCEPTED_ATMOSPHERE_ARENA_NBYTES: u64 = 840_600_000;
    const ACCEPTED_TERRAIN_OPEN_NBYTES: u64 = 953_971_136;
    const ACCEPTED_LIVE_DENSITY_NBYTES: u64 = 108_360_000;
    const ACCEPTED_GEOPOTENTIAL_NBYTES: u64 = 27_360_000;

    fn accepted_auxiliary() -> crate::child_artifact::AdmittedArtifact {
        fake_auxiliary(
            ACCEPTED_TERRAIN_OPEN_NBYTES,
            ACCEPTED_LIVE_DENSITY_NBYTES,
            ACCEPTED_GEOPOTENTIAL_NBYTES,
        )
    }

    #[test]
    fn every_composed_payload_is_sized_for_the_child_being_composed() {
        let grid = ACCEPTED_50M_CHILD_GRID;
        assert_eq!(
            child_atmosphere_arena_nbytes_for_grid(grid).unwrap(),
            ACCEPTED_ATMOSPHERE_ARENA_NBYTES
        );
        let layout = child_auxiliary_base_layout_for_grid(grid).unwrap();
        assert_eq!(layout.terrain_open_nbytes, ACCEPTED_TERRAIN_OPEN_NBYTES);
        assert_eq!(layout.live_density_nbytes, ACCEPTED_LIVE_DENSITY_NBYTES);
        assert_eq!(layout.geopotential_nbytes, ACCEPTED_GEOPOTENTIAL_NBYTES);

        let terrain = fake_sized("terrain", CHILD_TERRAIN_PAYLOAD_FILE, 300 * 300 * 4);
        let static_heights = fake_static((75 * 300 * 300 * 4) as u64, (76 * 300 * 300 * 4) as u64);
        let atmosphere = fake_sized(
            "atmosphere",
            CHILD_ATMOSPHERE_ARENA_FILE,
            ACCEPTED_ATMOSPHERE_ARENA_NBYTES,
        );
        let auxiliary = accepted_auxiliary();
        require_composed_grid(&atmosphere, &auxiliary, &terrain, &static_heights, grid).unwrap();

        // A 5 m child's terrain surface under a 50 m envelope.
        let wide = fake_sized("terrain", CHILD_TERRAIN_PAYLOAD_FILE, 3000 * 3000 * 4);
        let error = require_composed_grid(&atmosphere, &auxiliary, &wide, &static_heights, grid)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("do not match the arenas it points at"),
            "{error}"
        );

        // A deck with one level too few.
        let short = fake_static((74 * 300 * 300 * 4) as u64, (76 * 300 * 300 * 4) as u64);
        let error = require_composed_grid(&atmosphere, &auxiliary, &terrain, &short, grid)
            .unwrap_err()
            .to_string();
        assert!(error.contains("mass heights"), "{error}");

        // The 5 m child's own 28-carrier arena, offered to a 50 m envelope.
        // Word-aligned and nonzero, so only a real size check catches it.
        let foreign_arena = fake_sized(
            "atmosphere",
            CHILD_ATMOSPHERE_ARENA_FILE,
            child_atmosphere_arena_nbytes_for_grid(
                ChildGridSpec::new(3000, 3000, CHILD_NZ, 5.0, 5.0, 15_000.0).unwrap(),
            )
            .unwrap(),
        );
        let error =
            require_composed_grid(&foreign_arena, &auxiliary, &terrain, &static_heights, grid)
                .unwrap_err()
                .to_string();
        assert!(error.contains("28-carrier atmosphere arena"), "{error}");

        // One auxiliary payload short by exactly one carrier plane.
        for (index, what) in [
            (0usize, "50-carrier terrain/open estate"),
            (1, "live-density estate"),
            (2, "total-geopotential estate"),
        ] {
            let mut sizes = [
                ACCEPTED_TERRAIN_OPEN_NBYTES,
                ACCEPTED_LIVE_DENSITY_NBYTES,
                ACCEPTED_GEOPOTENTIAL_NBYTES,
            ];
            sizes[index] -= (300 * 300 * 4) as u64;
            let wrong = fake_auxiliary(sizes[0], sizes[1], sizes[2]);
            let error = require_composed_grid(&atmosphere, &wrong, &terrain, &static_heights, grid)
                .unwrap_err()
                .to_string();
            assert!(error.contains(what), "{error}");
        }
    }

    #[test]
    fn one_parent_is_required_across_every_producer() {
        let atmosphere = fake_admitted("atmosphere", Some(("11", "22")));
        let terrain = fake_admitted("terrain", Some(("11", "22")));
        let agreeing = agree_on_one_parent(&[&atmosphere, &terrain]).unwrap();
        assert_eq!(agreeing.checkpoint_sha256, "11".repeat(32));

        let foreign = fake_admitted("auxiliary", Some(("33", "44")));
        let error = agree_on_one_parent(&[&atmosphere, &foreign]).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("blend two parent runs"), "{message}");

        // One producer naming a parent is not agreement: the gate needs at
        // least two independent statements to compare.
        let anonymous = fake_admitted("closure", None);
        let error = agree_on_one_parent(&[&atmosphere, &anonymous]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("with fewer than two independent statements"),
            "{error}"
        );

        let error = agree_on_one_parent(&[&anonymous]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("with fewer than two independent statements"),
            "{error}"
        );
    }

    #[test]
    fn a_closure_carrying_the_uniform_z_deformation_is_refused_under_any_digest() {
        // Exactly the defect the evidence names: defv/defh byte-identical to
        // the sealed uniform-z negative control, published under fresh
        // manifest/receipt/payload digests so no hash pin would catch it.
        let closure = fake_closure(
            REFUSED_UNIFORM_Z_DEFV_SHA256,
            REFUSED_UNIFORM_Z_DEFH_SHA256,
            "99".repeat(32),
        );
        let error = refuse_discredited_closure(&closure).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("uniform-z negative control"), "{message}");
        assert!(message.contains("vertical grid"), "{message}");

        let corrected = fake_closure(
            CORRECTED_CHILD_CLOSURE_DEFV_SHA256,
            CORRECTED_CHILD_CLOSURE_DEFH_SHA256,
            "99".repeat(32),
        );
        refuse_discredited_closure(&corrected).unwrap();
    }

    #[test]
    fn the_superseded_closure_artifact_is_refused_by_identity() {
        let closure = fake_closure(
            CORRECTED_CHILD_CLOSURE_DEFV_SHA256,
            CORRECTED_CHILD_CLOSURE_DEFH_SHA256,
            REFUSED_UNIFORM_Z_CLOSURE_ARENA_SHA256.to_owned(),
        );
        let error = refuse_discredited_closure(&closure).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("superseded uniform-z child closure is refused"),
            "{error}"
        );
    }

    #[test]
    fn closure_layout_must_cover_the_composed_child_grid() {
        let closure = fake_closure(
            CORRECTED_CHILD_CLOSURE_DEFV_SHA256,
            CORRECTED_CHILD_CLOSURE_DEFH_SHA256,
            "99".repeat(32),
        );
        require_closure_grid_and_layout(&closure, ACCEPTED_50M_CHILD_GRID).unwrap();

        let other = ChildGridSpec::new(150, 150, CHILD_NZ, 100.0, 100.0, 15_000.0).unwrap();
        let error = require_closure_grid_and_layout(&closure, other).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("belonging to a different column"),
            "{error}"
        );
    }

    #[test]
    fn an_artifact_whose_receipt_does_not_anchor_its_own_manifest_is_refused() {
        let root = temp_root("full-state-admit");
        let directory = root.join("terrain");
        fs::create_dir_all(&directory).unwrap();
        let payload = vec![7u8; 32];
        let payload_sha = sha256::digest_hex(&payload);
        fs::write(directory.join(CHILD_TERRAIN_PAYLOAD_FILE), &payload).unwrap();
        let manifest = object(vec![("model", string("GM1"))]).canonical_bytes();
        fs::write(directory.join(CHILD_TERRAIN_MANIFEST_FILE), &manifest).unwrap();
        let receipt = |manifest_sha: &str| {
            object(vec![
                ("abi", string(CHILD_TERRAIN_ABI)),
                ("manifest_sha256", string(manifest_sha)),
                ("parent_checkpoint_sha256", string("ab".repeat(32))),
                ("parent_plan_sha256", string("cd".repeat(32))),
                ("payload_nbytes", number(payload.len())),
                ("payload_sha256", string(&payload_sha)),
                ("schema", string(CHILD_TERRAIN_SCHEMA)),
                ("status", string(CHILD_TERRAIN_STATUS)),
            ])
            .canonical_bytes()
        };
        fs::write(
            directory.join(CHILD_TERRAIN_RECEIPT_FILE),
            receipt(&"ff".repeat(32)),
        )
        .unwrap();
        let error = admit_artifact_with_receipt(
            &directory,
            &TERRAIN_SPEC,
            crate::child_artifact::ReceiptSource::InDirectory,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("not the manifest its own receipt anchors"),
            "{error}"
        );

        // Repair only the anchor: the same directory is then admitted, and its
        // recorded parent identity is carried out for the composing gate.
        fs::write(
            directory.join(CHILD_TERRAIN_RECEIPT_FILE),
            receipt(&sha256::digest_hex(&manifest)),
        )
        .unwrap();
        let admitted = admit_artifact_with_receipt(
            &directory,
            &TERRAIN_SPEC,
            crate::child_artifact::ReceiptSource::InDirectory,
        )
        .unwrap();
        assert_eq!(
            admitted.parent_checkpoint_sha256.as_deref(),
            Some("ab".repeat(32).as_str())
        );
        assert_eq!(
            admitted.file(CHILD_TERRAIN_PAYLOAD_FILE).unwrap().sha256,
            payload_sha
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_tampered_payload_is_refused_against_its_own_receipt() {
        let root = temp_root("full-state-payload");
        let directory = root.join("terrain");
        fs::create_dir_all(&directory).unwrap();
        let payload = vec![7u8; 32];
        let payload_sha = sha256::digest_hex(&payload);
        fs::write(directory.join(CHILD_TERRAIN_PAYLOAD_FILE), vec![8u8; 32]).unwrap();
        let manifest = object(vec![("model", string("GM1"))]).canonical_bytes();
        fs::write(directory.join(CHILD_TERRAIN_MANIFEST_FILE), &manifest).unwrap();
        fs::write(
            directory.join(CHILD_TERRAIN_RECEIPT_FILE),
            object(vec![
                ("abi", string(CHILD_TERRAIN_ABI)),
                ("manifest_sha256", string(sha256::digest_hex(&manifest))),
                ("parent_checkpoint_sha256", string("ab".repeat(32))),
                ("parent_plan_sha256", string("cd".repeat(32))),
                ("payload_nbytes", number(payload.len())),
                ("payload_sha256", string(&payload_sha)),
                ("schema", string(CHILD_TERRAIN_SCHEMA)),
                ("status", string(CHILD_TERRAIN_STATUS)),
            ])
            .canonical_bytes(),
        )
        .unwrap();
        let error = admit_artifact_with_receipt(
            &directory,
            &TERRAIN_SPEC,
            crate::child_artifact::ReceiptSource::InDirectory,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("differs from the digest its own receipt records"),
            "{error}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    // ---- fixtures --------------------------------------------------------

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let serial = TEST_NONCE.fetch_add(1, TestOrdering::Relaxed);
        let root = std::env::temp_dir().join(format!("gm1-{label}-{nonce}-{serial}"));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn fake_admitted(
        role: &'static str,
        parent: Option<(&str, &str)>,
    ) -> crate::child_artifact::AdmittedArtifact {
        crate::child_artifact::AdmittedArtifact {
            role,
            directory: PathBuf::from(format!("fixture-{role}")),
            schema: "fixture-schema".to_owned(),
            status: "fixture-status".to_owned(),
            abi: Some("fixture-abi".to_owned()),
            manifest_sha256: "0".repeat(64),
            receipt_sha256: "1".repeat(64),
            manifest: object(Vec::new()),
            receipt: object(Vec::new()),
            files: Vec::new(),
            parent_checkpoint_sha256: parent.map(|(value, _)| value.repeat(32)),
            parent_plan_sha256: parent.map(|(_, value)| value.repeat(32)),
        }
    }

    fn fake_sized(
        role: &'static str,
        file: &str,
        nbytes: u64,
    ) -> crate::child_artifact::AdmittedArtifact {
        let mut artifact = fake_admitted(role, None);
        artifact.files = vec![crate::child_artifact::AdmittedFile {
            name: file.to_owned(),
            nbytes,
            sha256: "aa".repeat(32),
        }];
        artifact
    }

    fn fake_auxiliary(
        terrain_open: u64,
        live_density: u64,
        geopotential: u64,
    ) -> crate::child_artifact::AdmittedArtifact {
        let mut artifact = fake_admitted("auxiliary", None);
        artifact.files = vec![
            crate::child_artifact::AdmittedFile {
                name: CHILD_AUXILIARY_TERRAIN_OPEN_FILE.to_owned(),
                nbytes: terrain_open,
                sha256: "dd".repeat(32),
            },
            crate::child_artifact::AdmittedFile {
                name: CHILD_AUXILIARY_LIVE_DENSITY_FILE.to_owned(),
                nbytes: live_density,
                sha256: "ee".repeat(32),
            },
            crate::child_artifact::AdmittedFile {
                name: CHILD_AUXILIARY_GEOPOTENTIAL_FILE.to_owned(),
                nbytes: geopotential,
                sha256: "ff".repeat(32),
            },
        ];
        artifact
    }

    fn fake_static(zh: u64, zf: u64) -> crate::child_artifact::AdmittedArtifact {
        let mut artifact = fake_admitted("static_heights", None);
        artifact.files = vec![
            crate::child_artifact::AdmittedFile {
                name: "zf.f32le".to_owned(),
                nbytes: zf,
                sha256: "bb".repeat(32),
            },
            crate::child_artifact::AdmittedFile {
                name: "zh.f32le".to_owned(),
                nbytes: zh,
                sha256: "cc".repeat(32),
            },
        ];
        artifact
    }

    fn fake_closure(
        defv: &str,
        defh: &str,
        payload_sha256: String,
    ) -> crate::child_artifact::AdmittedArtifact {
        let grid = ACCEPTED_50M_CHILD_GRID;
        let mut offset = 0u64;
        let mut carriers = Vec::new();
        for (name, _) in CLOSURE_CARRIERS {
            let levels = if name == "grdscl" {
                grid.nz()
            } else {
                grid.nz() + 1
            };
            let shape = [levels, grid.ny(), grid.nx()];
            let nbytes = shape
                .iter()
                .fold(4u64, |value, extent| value * *extent as u64);
            let digest = match name {
                "defv" => defv.to_owned(),
                "defh" => defh.to_owned(),
                _ => "ee".repeat(32),
            };
            carriers.push(object(vec![
                ("name", string(name)),
                ("nbytes", number(nbytes)),
                ("offset_nbytes", number(offset)),
                ("sha256", string(digest)),
                ("shape", usize_array(&shape)),
            ]));
            offset += nbytes;
        }
        let manifest = object(vec![
            ("carriers", JsonValue::Array(carriers)),
            (
                "grid",
                object(vec![
                    ("nx", number(grid.nx())),
                    ("ny", number(grid.ny())),
                    ("nz", number(grid.nz())),
                ]),
            ),
        ]);
        crate::child_artifact::AdmittedArtifact {
            role: "closure",
            directory: PathBuf::from("fixture-closure"),
            schema: CHILD_CLOSURE_RECEIPT_SCHEMA.to_owned(),
            status: CHILD_CLOSURE_STATUS.to_owned(),
            abi: Some(CHILD_CLOSURE_ABI.to_owned()),
            manifest_sha256: "0".repeat(64),
            receipt_sha256: "1".repeat(64),
            manifest,
            receipt: object(Vec::new()),
            files: vec![crate::child_artifact::AdmittedFile {
                name: CHILD_CLOSURE_FILE.to_owned(),
                nbytes: offset,
                sha256: payload_sha256,
            }],
            parent_checkpoint_sha256: None,
            parent_plan_sha256: None,
        }
    }

    fn fixture_prepared() -> Prepared {
        Prepared {
            atmosphere: fixture_artifact("atmosphere"),
            auxiliary: fixture_artifact("auxiliary"),
            terrain: fixture_artifact("terrain"),
            static_heights: fixture_artifact("static_heights"),
            closure: fixture_artifact("closure"),
            parent: BoundParent {
                checkpoint_sha256: "ab".repeat(32),
                plan_sha256: "cd".repeat(32),
            },
            base_wind: fixture_base_wind(),
            producer_source_sha256: "1".repeat(64),
            producer_executable_sha256: "2".repeat(64),
        }
    }

    fn fixture_artifact(role: &'static str) -> VerifiedArtifact {
        VerifiedArtifact {
            role,
            directory: format!("fixture-{role}"),
            schema: "fixture-schema".to_owned(),
            abi: "fixture-abi".to_owned(),
            status: "fixture-status".to_owned(),
            manifest_sha256: "0".repeat(64),
            receipt_sha256: "1".repeat(64),
            files: Vec::new(),
        }
    }

    fn fixture_base_wind() -> BaseWind {
        BaseWind {
            bytes: Vec::new(),
            sha256: "4".repeat(64),
            u0_sha256: "5".repeat(64),
            v0_sha256: "6".repeat(64),
            source_u0_sha256: "7".repeat(64),
            source_v0_sha256: "8".repeat(64),
            source_product_manifest_sha256: "9".repeat(64),
            source_product_payload_sha256: "a".repeat(64),
            source_product_restart_sha256: "b".repeat(64),
        }
    }
}
