//! Atomic completed-step discovery and off-thread strict history admission.

use crate::{
    input::{Gm1Dataset, load_artifact},
    pack::{is_pack_directory, load_pack_directory},
};
use cm1_state::child_post_step::{
    CHILD_POST_STEP_HISTORY_PROFILE, CHILD_POST_STEP_MANIFEST_SCHEMA,
    CHILD_POST_STEP_RECEIPT_SCHEMA, IDEALIZED_POST_STEP_HISTORY_PROFILE,
    IDEALIZED_POST_STEP_MANIFEST_SCHEMA, IDEALIZED_POST_STEP_RECEIPT_SCHEMA,
    IDEALIZED_POST_STEP_SOURCE_SCHEMA, IDEALIZED_POST_STEP_SOURCE_STATUS,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver},
    thread,
    time::Duration,
};

pub const PAIR_MANIFEST_SCHEMA: &str = CHILD_POST_STEP_MANIFEST_SCHEMA;
pub const PAIR_RECEIPT_SCHEMA: &str = CHILD_POST_STEP_RECEIPT_SCHEMA;
pub const PAIR_STATUS: &str = "PASS_RUST_PAIRED_POST_STEP_ENVELOPE";
pub const IDEALIZED_PAIR_MANIFEST_SCHEMA: &str = IDEALIZED_POST_STEP_MANIFEST_SCHEMA;
pub const IDEALIZED_PAIR_RECEIPT_SCHEMA: &str = IDEALIZED_POST_STEP_RECEIPT_SCHEMA;
pub const IDEALIZED_PAIR_STATUS: &str =
    "PASS_RUST_IDEALIZED_ISND7_IINIT1_PAIRED_POST_STEP_ENVELOPE";
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
pub const HISTORY_FILE: &str = "history.cm1fhs";
pub const CHECKPOINT_FILE: &str = "checkpoint.cm1fcp";
pub const MANIFEST_FILE: &str = "manifest.json";
pub const RECEIPT_FILE: &str = "receipt.json";
const MAX_JSON_BYTES: u64 = 128 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(650);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WatchRefusal {
    pub step: Option<u64>,
    pub directory: PathBuf,
    pub reason: String,
}

#[derive(Debug)]
pub struct WatchScan {
    pub accepted: Option<Gm1Dataset>,
    pub refusals: Vec<WatchRefusal>,
}

/// Strict, startup-only inventory of all atomically visible exact cloud packs.
/// Full history/pair admission remains a separate adapter; this fast path is
/// intentionally limited to the canonical three-file, non-quantized pack ABI.
#[derive(Debug)]
pub struct PackTimelineScan {
    pub frames: Vec<Gm1Dataset>,
    pub refusals: Vec<WatchRefusal>,
}

#[derive(Debug)]
pub enum WatchEvent {
    Accepted(Gm1Dataset),
    Refused(WatchRefusal),
}

#[derive(Clone, Debug)]
struct AdmittedHistory {
    step: u64,
    history_path: PathBuf,
    history_artifact_sha256: String,
    history_payload_sha256: String,
    history_contract_sha256: String,
    profile: PairSourceProfile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PairSourceProfile {
    ProductionRealChild,
    IdealizedSingleSoundingWarmBubble,
}

impl PairSourceProfile {
    fn from_envelopes(
        manifest_schema: &str,
        manifest_status: &str,
        receipt_schema: &str,
        receipt_status: &str,
    ) -> Result<Self, String> {
        match (
            manifest_schema,
            manifest_status,
            receipt_schema,
            receipt_status,
        ) {
            (PAIR_MANIFEST_SCHEMA, PAIR_STATUS, PAIR_RECEIPT_SCHEMA, PAIR_STATUS) => {
                Ok(Self::ProductionRealChild)
            }
            (
                IDEALIZED_PAIR_MANIFEST_SCHEMA,
                IDEALIZED_PAIR_STATUS,
                IDEALIZED_PAIR_RECEIPT_SCHEMA,
                IDEALIZED_PAIR_STATUS,
            ) => Ok(Self::IdealizedSingleSoundingWarmBubble),
            _ => Err(
                "pair manifest/receipt schema and status do not form one admitted profile envelope"
                    .to_owned(),
            ),
        }
    }

    const fn history_profile(self) -> &'static str {
        match self {
            Self::ProductionRealChild => CHILD_POST_STEP_HISTORY_PROFILE,
            Self::IdealizedSingleSoundingWarmBubble => IDEALIZED_POST_STEP_HISTORY_PROFILE,
        }
    }
}

#[derive(Debug, Deserialize)]
struct PairManifest {
    schema: String,
    status: String,
    step: u64,
    checkpoint: PairArtifact,
    configuration: PairConfiguration,
    history: PairArtifact,
    operator_manifest_sha256: String,
    scheduler_extension_sha256: String,
    #[serde(default)]
    source_full_state: Option<serde_json::Value>,
    #[serde(default)]
    source_idealized_authority: Option<IdealizedSourceAuthoritySummary>,
}

#[derive(Debug, Deserialize)]
struct PairConfiguration {
    domain_shape_zyx: [usize; 3],
    #[serde(default)]
    grid_sha256: Option<String>,
    #[serde(default)]
    admission: Option<String>,
    #[serde(default)]
    dt_f64_bits: Option<String>,
    #[serde(default)]
    dx_m_f64_bits: Option<String>,
    #[serde(default)]
    dy_m_f64_bits: Option<String>,
    #[serde(default)]
    dz_m_f64_bits: Option<String>,
    #[serde(default)]
    equation_set: Option<u64>,
    #[serde(default)]
    iinit: Option<u64>,
    #[serde(default)]
    irst: Option<u64>,
    #[serde(default)]
    isnd: Option<u64>,
    #[serde(default)]
    lateral_boundaries: Option<[u64; 4]>,
    #[serde(default)]
    microphysics: Option<String>,
    #[serde(default)]
    psolver: Option<u64>,
    #[serde(default)]
    terrain: Option<serde_json::Value>,
    #[serde(default)]
    vertical_grid: Option<String>,
    #[serde(default)]
    ztop_m_f64_bits: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct IdealizedSourceAuthoritySummary {
    initial_mass_f64_bits: String,
    rust_library_sha256: String,
    schema: String,
    sha256: String,
    sounding_sha256: String,
    status: String,
}

#[derive(Debug, Deserialize)]
struct PairArtifact {
    artifact_nbytes: u64,
    artifact_sha256: String,
    contract_sha256: String,
    file: String,
    payload_sha256: String,
    raw_arena_sha256: String,
}

#[derive(Debug, Deserialize)]
struct PairReceipt {
    schema: String,
    status: String,
    step: u64,
    manifest_sha256: String,
    checkpoint_artifact_sha256: String,
    checkpoint_raw_sha256: String,
    history_artifact_sha256: String,
    history_raw_sha256: String,
    operator_manifest_sha256: String,
    scheduler_extension_sha256: String,
}

pub fn load_initial_watch_dataset(root: &Path) -> Result<Gm1Dataset, String> {
    let scan = scan_watch_dir(root, None);
    scan.accepted.ok_or_else(|| {
        let reasons = scan
            .refusals
            .iter()
            .map(|refusal| format!("{}: {}", refusal.directory.display(), refusal.reason))
            .collect::<Vec<_>>()
            .join("; ");
        if reasons.is_empty() {
            format!(
                "watch directory {} has no atomically completed visible step-* frame",
                root.display()
            )
        } else {
            format!("watch directory has no admissible frame: {reasons}")
        }
    })
}

pub fn spawn_watch(root: PathBuf, last_step: u64) -> Receiver<WatchEvent> {
    let (sender, receiver) = mpsc::channel();
    thread::Builder::new()
        .name("gm1-view-atomic-watch".to_owned())
        .spawn(move || {
            let mut cursor = last_step;
            let mut last_refusal = None::<String>;
            loop {
                let scan = scan_watch_dir(&root, Some(cursor));
                if let Some(dataset) = scan.accepted {
                    cursor = dataset.step;
                    last_refusal = None;
                    if sender.send(WatchEvent::Accepted(dataset)).is_err() {
                        return;
                    }
                }
                if let Some(refusal) = scan.refusals.into_iter().max_by_key(|item| item.step) {
                    let fingerprint = format!(
                        "{:?}|{}|{}",
                        refusal.step,
                        refusal.directory.display(),
                        refusal.reason
                    );
                    if last_refusal.as_deref() != Some(&fingerprint) {
                        last_refusal = Some(fingerprint);
                        if sender.send(WatchEvent::Refused(refusal)).is_err() {
                            return;
                        }
                    }
                }
                thread::sleep(POLL_INTERVAL);
            }
        })
        .expect("GM1 View watch thread must start");
    receiver
}

pub fn scan_pack_timeline(root: &Path) -> PackTimelineScan {
    let mut frames = Vec::new();
    let mut refusals = Vec::new();
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) => {
            return PackTimelineScan {
                frames,
                refusals: vec![WatchRefusal {
                    step: None,
                    directory: root.to_path_buf(),
                    reason: format!("timeline root is unreadable: {error}"),
                }],
            };
        }
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let Some(step) = parse_step_directory(name) else {
            continue;
        };
        let directory = entry.path();
        if !entry
            .file_type()
            .is_ok_and(|kind| kind.is_dir() && !kind.is_symlink())
            || !is_pack_directory(&directory)
        {
            continue;
        }
        match load_pack_directory(&directory) {
            Ok(dataset)
                if dataset.step == step
                    && dataset
                        .fields
                        .iter()
                        .all(|field| field.shape == [75, 300, 300]) =>
            {
                frames.push(dataset);
            }
            Ok(_) => refusals.push(WatchRefusal {
                step: Some(step),
                directory,
                reason: "timeline cloud-pack step/grid differs from its atomic directory"
                    .to_owned(),
            }),
            Err(reason) => refusals.push(WatchRefusal {
                step: Some(step),
                directory,
                reason: format!("strict timeline cloud-pack admission refused: {reason}"),
            }),
        }
    }
    frames.sort_by_key(|frame| frame.step);
    frames.dedup_by_key(|frame| frame.step);
    PackTimelineScan { frames, refusals }
}

pub fn scan_watch_dir(root: &Path, after_step: Option<u64>) -> WatchScan {
    scan_watch_dir_with(root, after_step, |admitted| {
        load_artifact(&admitted.history_path, &admitted.history_contract_sha256)
    })
}

fn scan_watch_dir_with<F>(root: &Path, after_step: Option<u64>, mut loader: F) -> WatchScan
where
    F: FnMut(&AdmittedHistory) -> Result<Gm1Dataset, String>,
{
    let mut refusals = Vec::new();
    let mut candidates = Vec::new();
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) => {
            return WatchScan {
                accepted: None,
                refusals: vec![WatchRefusal {
                    step: None,
                    directory: root.to_path_buf(),
                    reason: format!("watch root is unreadable: {error}"),
                }],
            };
        }
    };
    let mut maximum_visible_step = None::<u64>;
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                refusals.push(WatchRefusal {
                    step: None,
                    directory: root.to_path_buf(),
                    reason: format!("watch entry is unreadable: {error}"),
                });
                continue;
            }
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let Some(step) = parse_step_directory(name) else {
            continue;
        };
        maximum_visible_step = Some(maximum_visible_step.map_or(step, |current| current.max(step)));
        if after_step.is_some_and(|cursor| step <= cursor) {
            continue;
        }
        match entry.file_type() {
            Ok(kind) if kind.is_dir() && !kind.is_symlink() => {
                candidates.push((step, entry.path()))
            }
            Ok(_) => refusals.push(WatchRefusal {
                step: Some(step),
                directory: entry.path(),
                reason: "visible step entry is not a real directory".to_owned(),
            }),
            Err(error) => refusals.push(WatchRefusal {
                step: Some(step),
                directory: entry.path(),
                reason: format!("visible step file type is unreadable: {error}"),
            }),
        }
    }
    candidates.sort_by_key(|(step, _)| *step);
    let mut accepted = None;
    for (directory_step, directory) in candidates {
        if is_pack_directory(&directory) {
            let dataset = match load_pack_directory(&directory) {
                Ok(dataset) => dataset,
                Err(reason) => {
                    refusals.push(WatchRefusal {
                        step: Some(directory_step),
                        directory,
                        reason: format!("strict cloud-pack admission refused: {reason}"),
                    });
                    continue;
                }
            };
            if dataset.step != directory_step
                || dataset
                    .fields
                    .iter()
                    .any(|field| field.shape != [75, 300, 300])
            {
                refusals.push(WatchRefusal {
                    step: Some(directory_step),
                    directory,
                    reason: "decoded cloud-pack step/grid differs from its atomic directory"
                        .to_owned(),
                });
                continue;
            }
            accepted = Some(dataset);
            continue;
        }
        let admitted = match admit_history_directory(&directory, directory_step) {
            Ok(admitted) => admitted,
            Err(reason) => {
                refusals.push(WatchRefusal {
                    step: Some(directory_step),
                    directory,
                    reason,
                });
                continue;
            }
        };
        let dataset = match loader(&admitted) {
            Ok(dataset) => dataset,
            Err(reason) => {
                refusals.push(WatchRefusal {
                    step: Some(directory_step),
                    directory,
                    reason: format!("strict history decode refused: {reason}"),
                });
                continue;
            }
        };
        if dataset.step != admitted.step
            || dataset.artifact_sha256 != admitted.history_artifact_sha256
            || dataset.payload_sha256 != admitted.history_payload_sha256
            || dataset.contract_sha256 != admitted.history_contract_sha256
            || dataset.profile_id != admitted.profile.history_profile()
        {
            refusals.push(WatchRefusal {
                step: Some(directory_step),
                directory,
                reason: "decoded history identity/profile differs from the atomic pair manifest"
                    .to_owned(),
            });
            continue;
        }
        if dataset
            .fields
            .iter()
            .any(|field| field.shape != [75, 300, 300])
        {
            refusals.push(WatchRefusal {
                step: Some(directory_step),
                directory,
                reason: "decoded render fields differ from the admitted [75,300,300] grid"
                    .to_owned(),
            });
            continue;
        }
        accepted = Some(dataset);
    }
    if let Some(cursor) = after_step
        && cursor > 0
        && maximum_visible_step.is_none_or(|step| step < cursor)
    {
        refusals.push(WatchRefusal {
            step: maximum_visible_step,
            directory: root.to_path_buf(),
            reason: format!(
                "watch timeline rolled back below admitted cursor step {cursor}; highest visible step is {maximum_visible_step:?}"
            ),
        });
    }
    WatchScan { accepted, refusals }
}

fn admit_history_directory(
    directory: &Path,
    directory_step: u64,
) -> Result<AdmittedHistory, String> {
    verify_inventory(directory)?;
    let manifest_path = directory.join(MANIFEST_FILE);
    let receipt_path = directory.join(RECEIPT_FILE);
    let manifest_bytes = read_bounded(&manifest_path, "pair manifest")?;
    let receipt_bytes = read_bounded(&receipt_path, "pair receipt")?;
    let manifest: PairManifest = parse_canonical(&manifest_bytes, "pair manifest")?;
    let receipt: PairReceipt = parse_canonical(&receipt_bytes, "pair receipt")?;
    let profile = PairSourceProfile::from_envelopes(
        &manifest.schema,
        &manifest.status,
        &receipt.schema,
        &receipt.status,
    )?;
    if manifest.step != directory_step || receipt.step != directory_step {
        return Err("directory, manifest, and receipt step disagree".to_owned());
    }
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    if receipt.manifest_sha256 != manifest_sha256 {
        return Err("receipt does not bind the canonical manifest bytes".to_owned());
    }
    for (value, label) in [
        (
            &manifest.checkpoint.artifact_sha256,
            "checkpoint artifact SHA-256",
        ),
        (
            &manifest.checkpoint.contract_sha256,
            "checkpoint contract SHA-256",
        ),
        (
            &manifest.checkpoint.payload_sha256,
            "checkpoint payload SHA-256",
        ),
        (
            &manifest.checkpoint.raw_arena_sha256,
            "checkpoint raw arena SHA-256",
        ),
        (
            &manifest.history.artifact_sha256,
            "history artifact SHA-256",
        ),
        (
            &manifest.history.contract_sha256,
            "history contract SHA-256",
        ),
        (&manifest.history.payload_sha256, "history payload SHA-256"),
        (
            &manifest.history.raw_arena_sha256,
            "history raw arena SHA-256",
        ),
        (
            &manifest.operator_manifest_sha256,
            "operator manifest SHA-256",
        ),
        (
            &manifest.scheduler_extension_sha256,
            "scheduler extension SHA-256",
        ),
        (
            &receipt.history_artifact_sha256,
            "receipt history artifact SHA-256",
        ),
        (
            &receipt.checkpoint_artifact_sha256,
            "receipt checkpoint artifact SHA-256",
        ),
        (
            &receipt.checkpoint_raw_sha256,
            "receipt checkpoint raw SHA-256",
        ),
        (&receipt.history_raw_sha256, "receipt history raw SHA-256"),
        (
            &receipt.operator_manifest_sha256,
            "receipt operator manifest SHA-256",
        ),
        (
            &receipt.scheduler_extension_sha256,
            "receipt scheduler extension SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    validate_pair_profile(&manifest, profile)?;
    if manifest.checkpoint.file != CHECKPOINT_FILE || manifest.history.file != HISTORY_FILE {
        return Err("manifest checkpoint/history filename changed".to_owned());
    }
    if receipt.checkpoint_artifact_sha256 != manifest.checkpoint.artifact_sha256
        || receipt.checkpoint_raw_sha256 != manifest.checkpoint.raw_arena_sha256
        || receipt.history_artifact_sha256 != manifest.history.artifact_sha256
        || receipt.history_raw_sha256 != manifest.history.raw_arena_sha256
        || receipt.operator_manifest_sha256 != manifest.operator_manifest_sha256
        || receipt.scheduler_extension_sha256 != manifest.scheduler_extension_sha256
    {
        return Err("receipt and manifest history/execution identities disagree".to_owned());
    }
    let history_path = directory.join(HISTORY_FILE);
    let metadata = fs::metadata(&history_path)
        .map_err(|error| format!("history metadata is unavailable: {error}"))?;
    if !metadata.is_file() || metadata.len() != manifest.history.artifact_nbytes {
        return Err("history file type or byte length differs from the manifest".to_owned());
    }
    let observed = sha256_file(&history_path)?;
    if observed != manifest.history.artifact_sha256 {
        return Err("history artifact SHA-256 differs from the manifest".to_owned());
    }
    Ok(AdmittedHistory {
        step: directory_step,
        history_path,
        history_artifact_sha256: manifest.history.artifact_sha256,
        history_payload_sha256: manifest.history.payload_sha256,
        history_contract_sha256: manifest.history.contract_sha256,
        profile,
    })
}

fn validate_pair_profile(
    manifest: &PairManifest,
    profile: PairSourceProfile,
) -> Result<(), String> {
    if manifest.configuration.domain_shape_zyx != [75, 300, 300] {
        return Err("pair domain is not the admitted [75,300,300] GM1 domain".to_owned());
    }
    match profile {
        PairSourceProfile::ProductionRealChild => {
            let grid = manifest
                .configuration
                .grid_sha256
                .as_deref()
                .ok_or_else(|| "real-child pair configuration lacks grid SHA-256".to_owned())?;
            validate_sha256(grid, "grid SHA-256")?;
            if manifest.source_full_state.is_none() || manifest.source_idealized_authority.is_some()
            {
                return Err(
                    "real-child pair requires only its production source full-state".to_owned(),
                );
            }
            Ok(())
        }
        PairSourceProfile::IdealizedSingleSoundingWarmBubble => {
            let source = manifest
                .source_idealized_authority
                .as_ref()
                .ok_or_else(|| "idealized pair lacks its typed source authority".to_owned())?;
            if manifest.source_full_state.is_some() {
                return Err(
                    "idealized pair must not carry production-real source full-state authority"
                        .to_owned(),
                );
            }
            for (value, label) in [
                (
                    &source.rust_library_sha256,
                    "idealized Rust library SHA-256",
                ),
                (&source.sha256, "idealized source authority SHA-256"),
                (&source.sounding_sha256, "idealized sounding SHA-256"),
            ] {
                validate_sha256(value, label)?;
            }
            if manifest.configuration.grid_sha256.is_some()
                || manifest.configuration.admission.as_deref()
                    != Some("idealized-isnd7-iinit1-single-sounding-warm-bubble")
                || manifest.configuration.dt_f64_bits.as_deref() != Some("3fe0000000000000")
                || manifest.configuration.dx_m_f64_bits.as_deref() != Some("4049000000000000")
                || manifest.configuration.dy_m_f64_bits.as_deref() != Some("4049000000000000")
                || manifest.configuration.dz_m_f64_bits.as_deref() != Some("4069000000000000")
                || manifest.configuration.equation_set != Some(2)
                || manifest.configuration.iinit != Some(1)
                || manifest.configuration.irst != Some(0)
                || manifest.configuration.isnd != Some(7)
                || manifest.configuration.lateral_boundaries != Some([2, 2, 2, 2])
                || manifest.configuration.microphysics.as_deref() != Some("wsm6-mp6")
                || manifest.configuration.psolver != Some(3)
                || manifest.configuration.terrain.as_ref()
                    != Some(&serde_json::Value::String("flat-zero-terrain".to_owned()))
                || manifest.configuration.vertical_grid.as_deref() != Some("uniform-200m")
                || manifest.configuration.ztop_m_f64_bits.as_deref() != Some("40cd4c0000000000")
            {
                return Err(
                    "idealized pair configuration differs from the exact flat dz=200 m profile"
                        .to_owned(),
                );
            }
            if source.initial_mass_f64_bits != "414e81d1397c6000"
                || source.rust_library_sha256
                    != "10ba6473422e9cffac8955a6ef68b6d47e0b58571d7550da7250a19dabcd754d"
                || source.schema != IDEALIZED_POST_STEP_SOURCE_SCHEMA
                || source.sha256 != IDEALIZED_SOURCE_AUTHORITY_SHA256
                || source.sounding_sha256
                    != "56df89107f712b4911a7eb53a5c4df637826ee02ca14fe5322eb0cd2916bdea0"
                || source.status != IDEALIZED_POST_STEP_SOURCE_STATUS
            {
                return Err("idealized pair source authority summary changed".to_owned());
            }
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
            if manifest.history.contract_sha256 != IDEALIZED_HISTORY_CONTRACT_SHA256
                || manifest.checkpoint.contract_sha256 != IDEALIZED_CHECKPOINT_CONTRACT_SHA256
                || manifest.operator_manifest_sha256 != IDEALIZED_OPERATOR_MANIFEST_SHA256
                || manifest.scheduler_extension_sha256 != IDEALIZED_SCHEDULER_EXTENSION_SHA256
            {
                return Err(
                    "idealized pair contracts/operator/scheduler differ from sealed CPU readiness"
                        .to_owned(),
                );
            }
            Ok(())
        }
    }
}

fn verify_inventory(directory: &Path) -> Result<(), String> {
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let metadata = fs::symlink_metadata(entry.path()).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err("step inventory contains a symlink or non-file".to_owned());
        }
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "step inventory has a non-UTF-8 filename".to_owned())?;
        observed.insert(name);
    }
    let projected = BTreeSet::from([
        HISTORY_FILE.to_owned(),
        MANIFEST_FILE.to_owned(),
        RECEIPT_FILE.to_owned(),
    ]);
    let mut complete_pair = projected.clone();
    complete_pair.insert(CHECKPOINT_FILE.to_owned());
    if observed != projected && observed != complete_pair {
        return Err(format!(
            "step inventory is neither exact history projection nor exact four-file pair: {observed:?}"
        ));
    }
    Ok(())
}

fn parse_step_directory(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("step-")?;
    if digits.len() != 8 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn read_bounded(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::metadata(path).map_err(|error| format!("{label} metadata: {error}"))?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_JSON_BYTES {
        return Err(format!("{label} is empty, oversized, or not a file"));
    }
    fs::read(path).map_err(|error| format!("{label} read failed: {error}"))
}

fn parse_canonical<T>(bytes: &[u8], label: &str) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("{label} JSON: {error}"))?;
    let canonical = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
    if canonical != bytes {
        return Err(format!("{label} is not canonical compact JSON"));
    }
    serde_json::from_value(value).map_err(|error| format!("{label} fields: {error}"))
}

fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} is not lowercase SHA-256"));
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::{
        CHECKPOINT_FILE, HISTORY_FILE, IDEALIZED_PAIR_MANIFEST_SCHEMA,
        IDEALIZED_PAIR_RECEIPT_SCHEMA, IDEALIZED_PAIR_STATUS, MANIFEST_FILE, PAIR_MANIFEST_SCHEMA,
        PAIR_RECEIPT_SCHEMA, PAIR_STATUS, RECEIPT_FILE, scan_watch_dir_with, sha256_bytes,
    };
    use crate::input::{Gm1Dataset, VolumeField, f32_sha256};
    use serde_json::{Value, json};
    use std::{
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "gm1-view-watch-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        path
    }

    fn canonical(value: &Value) -> Vec<u8> {
        serde_json::to_vec(value).unwrap()
    }

    fn make_pair(root: &Path, name: &str, step: u64) {
        let directory = root.join(name);
        fs::create_dir(&directory).unwrap();
        let history = format!("history-{step}").into_bytes();
        fs::write(directory.join(HISTORY_FILE), &history).unwrap();
        let artifact = sha256_bytes(&history);
        let contract = "11".repeat(32);
        let payload = "22".repeat(32);
        let raw = "33".repeat(32);
        let operator = "44".repeat(32);
        let scheduler = "55".repeat(32);
        let manifest = json!({
            "checkpoint": {
                "artifact_nbytes": 4,
                "artifact_sha256": "77".repeat(32),
                "contract_sha256": "88".repeat(32),
                "file": CHECKPOINT_FILE,
                "payload_sha256": "99".repeat(32),
                "raw_arena_sha256": "aa".repeat(32),
            },
            "configuration": {"domain_shape_zyx": [75, 300, 300], "grid_sha256": "66".repeat(32)},
            "history": {
                "artifact_nbytes": history.len(),
                "artifact_sha256": artifact,
                "contract_sha256": contract,
                "file": HISTORY_FILE,
                "payload_sha256": payload,
                "raw_arena_sha256": raw,
            },
            "operator_manifest_sha256": operator,
            "scheduler_extension_sha256": scheduler,
            "schema": PAIR_MANIFEST_SCHEMA,
            "source_full_state": {},
            "status": PAIR_STATUS,
            "step": step,
        });
        let manifest_bytes = canonical(&manifest);
        let manifest_sha = sha256_bytes(&manifest_bytes);
        fs::write(directory.join(MANIFEST_FILE), &manifest_bytes).unwrap();
        let receipt = json!({
            "checkpoint_artifact_sha256": "77".repeat(32),
            "checkpoint_raw_sha256": "aa".repeat(32),
            "history_artifact_sha256": artifact,
            "history_raw_sha256": raw,
            "manifest_sha256": manifest_sha,
            "operator_manifest_sha256": operator,
            "scheduler_extension_sha256": scheduler,
            "schema": PAIR_RECEIPT_SCHEMA,
            "status": PAIR_STATUS,
            "step": step,
        });
        fs::write(directory.join(RECEIPT_FILE), canonical(&receipt)).unwrap();
    }

    fn make_idealized_pair(root: &Path, name: &str, step: u64) {
        let directory = root.join(name);
        fs::create_dir(&directory).unwrap();
        let history = format!("bounded-idealized-history-{step}").into_bytes();
        fs::write(directory.join(HISTORY_FILE), &history).unwrap();
        let artifact = sha256_bytes(&history);
        let raw = "33".repeat(32);
        let operator = "82fa0c0669c3d6eee805446df29fc42ab4f0e1b0a9fae6690adbfaf587f1352d";
        let scheduler = "6552ed7d7afd4115baad3406dffbb69e11d7a187af8d3591f6969c985bf64ac2";
        let manifest = json!({
            "checkpoint": {
                "artifact_nbytes": 4,
                "artifact_sha256": "77".repeat(32),
                "contract_sha256": "cf11c4eb1b7a8a6823d71ae8a94b7fed2532f021e88bf0931e372c6555acf914",
                "file": CHECKPOINT_FILE,
                "payload_sha256": "99".repeat(32),
                "raw_arena_sha256": "aa".repeat(32),
            },
            "configuration": {
                "admission": "idealized-isnd7-iinit1-single-sounding-warm-bubble",
                "domain_shape_zyx": [75, 300, 300],
                "dt_f64_bits": "3fe0000000000000",
                "dx_m_f64_bits": "4049000000000000",
                "dy_m_f64_bits": "4049000000000000",
                "dz_m_f64_bits": "4069000000000000",
                "equation_set": 2,
                "iinit": 1,
                "irst": 0,
                "isnd": 7,
                "lateral_boundaries": [2, 2, 2, 2],
                "microphysics": "wsm6-mp6",
                "psolver": 3,
                "terrain": "flat-zero-terrain",
                "vertical_grid": "uniform-200m",
                "ztop_m_f64_bits": "40cd4c0000000000",
            },
            "history": {
                "artifact_nbytes": history.len(),
                "artifact_sha256": artifact,
                "contract_sha256": "f519a89c8171f34b2ca0317fd864ec1a4cd40371e360a7f39a3d20b3acca4d5d",
                "file": HISTORY_FILE,
                "payload_sha256": "22".repeat(32),
                "raw_arena_sha256": raw,
            },
            "operator_manifest_sha256": operator,
            "scheduler_extension_sha256": scheduler,
            "schema": IDEALIZED_PAIR_MANIFEST_SCHEMA,
            "source_idealized_authority": {
                "initial_mass_f64_bits": "414e81d1397c6000",
                "rust_library_sha256": "10ba6473422e9cffac8955a6ef68b6d47e0b58571d7550da7250a19dabcd754d",
                "schema": "gm1.idealized-isnd7-iinit1-source-authority/v1",
                "sha256": "d4857df7e121f48a27cc15b3ff8845b0f064a790f4954a08d24588af0d5f6366",
                "sounding_sha256": "56df89107f712b4911a7eb53a5c4df637826ee02ca14fe5322eb0cd2916bdea0",
                "status": "ADMITTED_SOURCE_AUDITED_CM1_R21_1_ISND7_IINIT1",
            },
            "status": IDEALIZED_PAIR_STATUS,
            "step": step,
        });
        let manifest_bytes = canonical(&manifest);
        let manifest_sha = sha256_bytes(&manifest_bytes);
        fs::write(directory.join(MANIFEST_FILE), &manifest_bytes).unwrap();
        let receipt = json!({
            "checkpoint_artifact_sha256": "77".repeat(32),
            "checkpoint_raw_sha256": "aa".repeat(32),
            "history_artifact_sha256": artifact,
            "history_raw_sha256": raw,
            "manifest_sha256": manifest_sha,
            "operator_manifest_sha256": operator,
            "scheduler_extension_sha256": scheduler,
            "schema": IDEALIZED_PAIR_RECEIPT_SCHEMA,
            "status": IDEALIZED_PAIR_STATUS,
            "step": step,
        });
        fs::write(directory.join(RECEIPT_FILE), canonical(&receipt)).unwrap();
    }

    fn stub_dataset(admitted: &super::AdmittedHistory) -> Result<Gm1Dataset, String> {
        let values = vec![0.0; 75 * 300 * 300];
        Ok(Gm1Dataset {
            config_sha256: String::new(),
            terrain_msl_m: None,
            face_height_m: None,
            tornado_surface_fields: Vec::new(),
            source_path: admitted.history_path.clone(),
            artifact_kind: "test history".to_owned(),
            artifact_sha256: admitted.history_artifact_sha256.clone(),
            payload_sha256: admitted.history_payload_sha256.clone(),
            contract_sha256: admitted.history_contract_sha256.clone(),
            profile_id: admitted.profile.history_profile().to_owned(),
            step: admitted.step,
            model_time_s: admitted.step as f64,
            dt_s: 0.5,
            fields: vec![VolumeField {
                id: "cloud_condensate".to_owned(),
                label: "cloud".to_owned(),
                units: "kg kg-1".to_owned(),
                meaning: "test".to_owned(),
                source: admitted
                    .history_path
                    .parent()
                    .unwrap()
                    .display()
                    .to_string(),
                shape: [75, 300, 300],
                values_sha256: f32_sha256(&values),
                finite_min: 0.0,
                finite_max: 0.0,
                values,
            }],
        })
    }

    #[test]
    fn hidden_partial_directory_is_ignored_until_atomic_rename() {
        let root = temp_root("atomic");
        make_pair(&root, ".step-00000001.staging", 1);
        let hidden = scan_watch_dir_with(&root, None, stub_dataset);
        assert!(hidden.accepted.is_none());
        assert!(hidden.refusals.is_empty());
        fs::rename(
            root.join(".step-00000001.staging"),
            root.join("step-00000001"),
        )
        .unwrap();
        let visible = scan_watch_dir_with(&root, None, stub_dataset);
        assert_eq!(visible.accepted.unwrap().step, 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn malformed_newer_frame_is_refused_without_replacing_last_good() {
        let root = temp_root("refusal");
        make_pair(&root, "step-00000001", 1);
        let first = scan_watch_dir_with(&root, None, stub_dataset);
        assert_eq!(first.accepted.unwrap().step, 1);
        make_pair(&root, "step-00000002", 2);
        let receipt_path = root.join("step-00000002").join(RECEIPT_FILE);
        let mut receipt: Value = serde_json::from_slice(&fs::read(&receipt_path).unwrap()).unwrap();
        receipt["manifest_sha256"] = Value::String("ff".repeat(32));
        fs::write(&receipt_path, canonical(&receipt)).unwrap();
        let next = scan_watch_dir_with(&root, Some(1), stub_dataset);
        assert!(next.accepted.is_none());
        assert_eq!(next.refusals.len(), 1);
        assert_eq!(next.refusals[0].step, Some(2));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sequential_completed_frames_advance_monotonically() {
        let root = temp_root("monotonic");
        make_pair(&root, "step-00000001", 1);
        let first = scan_watch_dir_with(&root, None, stub_dataset);
        assert_eq!(first.accepted.unwrap().step, 1);
        make_pair(&root, "step-00000003", 3);
        let third = scan_watch_dir_with(&root, Some(1), stub_dataset);
        assert_eq!(third.accepted.unwrap().step, 3);
        let duplicate = scan_watch_dir_with(&root, Some(3), stub_dataset);
        assert!(duplicate.accepted.is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn visible_timeline_rollback_below_cursor_is_refused() {
        let root = temp_root("rollback");
        make_pair(&root, "step-00000001", 1);
        let rollback = scan_watch_dir_with(&root, Some(3), stub_dataset);
        assert!(rollback.accepted.is_none());
        assert_eq!(rollback.refusals.len(), 1);
        assert_eq!(rollback.refusals[0].step, Some(1));
        assert!(rollback.refusals[0].reason.contains("rolled back"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn idealized_profile_accepts_bounded_metadata_only_under_final_authority() {
        let root = temp_root("idealized-authority");
        make_idealized_pair(&root, "step-00000001", 1);
        let scan = scan_watch_dir_with(&root, None, stub_dataset);
        assert_eq!(scan.accepted.unwrap().step, 1);
        assert!(scan.refusals.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn idealized_foreign_operator_is_refused_with_self_consistent_pair_hashes() {
        let root = temp_root("idealized-foreign-operator");
        make_idealized_pair(&root, "step-00000001", 1);
        let directory = root.join("step-00000001");
        let manifest_path = directory.join(MANIFEST_FILE);
        let receipt_path = directory.join(RECEIPT_FILE);
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["operator_manifest_sha256"] = Value::String("ee".repeat(32));
        let manifest_bytes = canonical(&manifest);
        fs::write(&manifest_path, &manifest_bytes).unwrap();
        let mut receipt: Value = serde_json::from_slice(&fs::read(&receipt_path).unwrap()).unwrap();
        receipt["manifest_sha256"] = Value::String(sha256_bytes(&manifest_bytes));
        receipt["operator_manifest_sha256"] = Value::String("ee".repeat(32));
        fs::write(&receipt_path, canonical(&receipt)).unwrap();
        let scan = scan_watch_dir_with(&root, None, stub_dataset);
        assert!(scan.accepted.is_none());
        assert_eq!(scan.refusals.len(), 1);
        assert!(scan.refusals[0].reason.contains("sealed CPU readiness"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn idealized_pair_refuses_mixed_production_source_authority() {
        let root = temp_root("idealized-mixed-source");
        make_idealized_pair(&root, "step-00000001", 1);
        let directory = root.join("step-00000001");
        let manifest_path = directory.join(MANIFEST_FILE);
        let receipt_path = directory.join(RECEIPT_FILE);
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["source_full_state"] = json!({"directory": "foreign-real-child"});
        let manifest_bytes = canonical(&manifest);
        fs::write(&manifest_path, &manifest_bytes).unwrap();
        let mut receipt: Value = serde_json::from_slice(&fs::read(&receipt_path).unwrap()).unwrap();
        receipt["manifest_sha256"] = Value::String(sha256_bytes(&manifest_bytes));
        fs::write(&receipt_path, canonical(&receipt)).unwrap();
        let scan = scan_watch_dir_with(&root, None, stub_dataset);
        assert!(scan.accepted.is_none());
        assert_eq!(scan.refusals.len(), 1);
        assert!(
            scan.refusals[0]
                .reason
                .contains("must not carry production-real")
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mixed_real_and_idealized_envelope_is_refused_before_decode() {
        let root = temp_root("mixed-profile");
        make_idealized_pair(&root, "step-00000001", 1);
        let receipt_path = root.join("step-00000001").join(RECEIPT_FILE);
        let mut receipt: Value = serde_json::from_slice(&fs::read(&receipt_path).unwrap()).unwrap();
        receipt["schema"] = Value::String(PAIR_RECEIPT_SCHEMA.to_owned());
        receipt["status"] = Value::String(PAIR_STATUS.to_owned());
        fs::write(&receipt_path, canonical(&receipt)).unwrap();
        let scan = scan_watch_dir_with(&root, None, stub_dataset);
        assert!(scan.accepted.is_none());
        assert_eq!(scan.refusals.len(), 1);
        assert!(
            scan.refusals[0]
                .reason
                .contains("one admitted profile envelope")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
