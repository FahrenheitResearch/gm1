//! Durable persistence for the 21-array initialized-product WSM6 transition.
//!
//! `CM1W6R1` owns the checkpoint envelope, metadata, validation, hashing, and
//! atomic publication.  Its capture ingress and restore egress are raw named
//! binary32 files so Python/CUDA orchestration performs only opaque byte
//! transport and cannot cast, reorder, derive, or silently omit a carrier.

use crate::geopotential::WSM6_GEOPOTENTIAL_CARRIER_ABI;
use crate::json::JsonValue;
use crate::product::INITIALIZED_PRODUCT_CARRIER_ABI;
use crate::sha256::{self, Sha256};
use crate::{Result, StateError};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const WSM6_TRANSITION_CHECKPOINT_MAGIC: [u8; 8] = *b"CM1W6R1\0";
pub const WSM6_TRANSITION_CHECKPOINT_SCHEMA: &str =
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-checkpoint/v1";
pub const WSM6_TRANSITION_CHECKPOINT_RECEIPT_SCHEMA: &str =
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-checkpoint-receipt/v1";
pub const WSM6_TRANSITION_RESTORE_MANIFEST_SCHEMA: &str =
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-restore-directory/v1";
pub const WSM6_TRANSITION_RESTORE_RECEIPT_SCHEMA: &str =
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-restore-receipt/v1";
pub const WSM6_TRANSITION_CARRIER_ABI: &str =
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-restart-carriers/v1";
pub const WSM6_TRANSITION_ABI: &str = "gpuwm-cm1.initialized-product-eqtset2-wsm6-transition/v1";
pub const WSM6_TRANSITION_IN_MEMORY_RESTART_SCHEMA: &str =
    "gpuwm-cm1.initialized-product-eqtset2-wsm6-restart/v1";
pub const WSM6_TRANSITION_BRIDGE_ABI: &str = "gpuwm-cm1.eqtset2-wsm6-bridge/v1";
pub const WSM6_TRANSITION_COMPONENT_ABI: &str = "gpuwm-cm1.arwen-wsm6-component/v1";
pub const WSM6_TRANSITION_CAPTURE_PHASE: &str = "post-transition-after-held-recapture";
pub const WSM6_TRANSITION_RESTORE_PHASE: &str = "fresh-bridge-before-next-final-rk-remove";
pub const WSM6_TRANSITION_CLAIM_LIMIT: &str = "exact durable persistence and restore of these 21 transition-local arrays and cadence only; not all 49 product or A-time arrays, halos, boundary history, closure or dynamics state, scheduler or model clock, idiss, efall, reflectivity, coupled trajectory, forecast, tornado, or model-validation claim";
pub const WSM6_TRANSITION_PUBLICATION_CONTRACT: &str = "create a new same-directory temporary file, stream and synchronize the complete CM1W6R1 envelope, publish through a create-new hard link, then unlink only the owned temporary; never overwrite the target";
pub const WSM6_TRANSITION_RESTORE_PUBLICATION_CONTRACT: &str = "write 21 exact named binary32 files and one canonical manifest into a new sibling staging directory; synchronize every file; rename only after the checkpoint and all field hashes revalidate; refuse an existing destination";
pub const WSM6_TRANSITION_ARRAY_NAMES: [&str; 21] = [
    "state.pp3d",
    "state.th3d",
    "state.prs",
    "state.rho",
    "state.rr",
    "state.q3d",
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

const ROLES: [&str; 21] = [
    "state-current-pressure-perturbation",
    "state-current-theta-perturbation",
    "diagnostic-pressure-pa",
    "diagnostic-moist-density",
    "reciprocal-diagnostic-moist-density",
    "state-current-packed-moisture",
    "CM1-held-next-step-Exner-tendency",
    "CM1-held-next-step-theta-tendency",
    "CM1-held-next-step-qv-tendency",
    "CM1-held-next-step-qc-tendency",
    "component-held-heating",
    "component-effective-radius-cloud",
    "component-effective-radius-ice",
    "component-effective-radius-snow",
    "component-surface-rain-total",
    "component-surface-rain-increment",
    "component-surface-snow-total",
    "component-surface-snow-increment",
    "component-surface-graupel-total",
    "component-surface-graupel-increment",
    "component-surface-frozen-precipitation-ratio",
];
const SPECIES: [&str; 6] = ["qv", "qc", "qr", "qi", "qs", "qg"];
const SEALED_SOURCE_ARTIFACT_SHA256: &str =
    "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f";
const SEALED_SOURCE_CONTRACT_SHA256: &str =
    "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12";
const PREFIX_BYTES: u64 = 16;
const MAX_HEADER_BYTES: u64 = 1024 * 1024;
const MAX_PAYLOAD_BYTES: u64 = 1024 * 1024 * 1024;
const IO_CHUNK_BYTES: usize = 1024 * 1024;
const NZ: usize = 75;
const SOURCE_NY: usize = 300;
const SOURCE_NX: usize = 300;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

const HEADER_KEYS: &[&str] = &[
    "capture_phase",
    "carrier_abi",
    "carriers",
    "claim_limit",
    "configuration",
    "dtype",
    "itemsize",
    "order",
    "payload_nbytes",
    "payload_sha256",
    "restore_phase",
    "schema",
    "source",
    "transition",
];
const CONFIGURATION_KEYS: &[&str] = &[
    "completed_bridge_calls",
    "completed_component_calls",
    "completed_transitions",
    "crop_count_yx",
    "crop_origin_yx",
    "dt_f64_bits",
    "dx_f64_bits",
    "dy_f64_bits",
    "dz_f64_bits",
    "efall",
    "eqtset",
    "idiss",
    "native_ptype_context",
    "nz",
    "reflectivity",
    "semantic_component_id",
];
const SOURCE_KEYS: &[&str] = &[
    "geopotential_manifest_sha256",
    "geopotential_payload_sha256",
    "product_manifest_sha256",
    "product_payload_sha256",
    "source_artifact_sha256",
    "source_contract_sha256",
];
const TRANSITION_KEYS: &[&str] = &[
    "bridge_abi",
    "component_abi",
    "geopotential_carrier_abi",
    "in_memory_restart_schema",
    "product_carrier_abi",
    "transition_abi",
];
const CARRIER_KEYS: &[&str] = &[
    "dtype",
    "itemsize",
    "name",
    "nbytes",
    "offset_nbytes",
    "order",
    "role",
    "shape",
    "sha256",
    "species_order",
    "statistics",
];
const STATS_KEYS: &[&str] = &[
    "finite_count",
    "maximum_f32_bits",
    "minimum_f32_bits",
    "nonzero_count",
];
const RESTORE_MANIFEST_KEYS: &[&str] = &[
    "arrays",
    "capture_phase",
    "carrier_abi",
    "claim_limit",
    "completed_transitions",
    "restore_phase",
    "schema",
    "source",
    "source_checkpoint_artifact_sha256",
];
const RESTORE_ARRAY_KEYS: &[&str] = &[
    "dtype",
    "file",
    "itemsize",
    "name",
    "nbytes",
    "order",
    "role",
    "shape",
    "sha256",
    "species_order",
    "statistics",
];

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionCheckpointConfig {
    pub j_start: usize,
    pub i_start: usize,
    pub ny: usize,
    pub nx: usize,
    pub completed_bridge_calls: u64,
    pub completed_component_calls: u64,
    pub completed_transitions: u64,
}

impl TransitionCheckpointConfig {
    pub fn new(
        j_start: usize,
        i_start: usize,
        ny: usize,
        nx: usize,
        completed_bridge_calls: u64,
        completed_component_calls: u64,
        completed_transitions: u64,
    ) -> Result<Self> {
        if ny == 0
            || nx == 0
            || j_start.checked_add(ny).is_none()
            || i_start.checked_add(nx).is_none()
            || j_start + ny > SOURCE_NY
            || i_start + nx > SOURCE_NX
        {
            return Err(invalid(
                "transition checkpoint crop is empty, overflows, or exceeds 300x300",
            ));
        }
        if completed_transitions == 0
            || completed_bridge_calls != completed_transitions
            || completed_component_calls != completed_transitions
        {
            return Err(invalid(
                "transition checkpoint completed bridge, component, and transition counts must be equal and at least one",
            ));
        }
        Ok(Self {
            j_start,
            i_start,
            ny,
            nx,
            completed_bridge_calls,
            completed_component_calls,
            completed_transitions,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionSourceIdentities {
    pub product_manifest_sha256: String,
    pub product_payload_sha256: String,
    pub geopotential_manifest_sha256: String,
    pub geopotential_payload_sha256: String,
    pub source_artifact_sha256: String,
    pub source_contract_sha256: String,
}

impl TransitionSourceIdentities {
    pub fn new(
        product_manifest_sha256: impl Into<String>,
        product_payload_sha256: impl Into<String>,
        geopotential_manifest_sha256: impl Into<String>,
        geopotential_payload_sha256: impl Into<String>,
        source_artifact_sha256: impl Into<String>,
        source_contract_sha256: impl Into<String>,
    ) -> Result<Self> {
        let result = Self {
            product_manifest_sha256: product_manifest_sha256.into(),
            product_payload_sha256: product_payload_sha256.into(),
            geopotential_manifest_sha256: geopotential_manifest_sha256.into(),
            geopotential_payload_sha256: geopotential_payload_sha256.into(),
            source_artifact_sha256: source_artifact_sha256.into(),
            source_contract_sha256: source_contract_sha256.into(),
        };
        for (value, label) in [
            (&result.product_manifest_sha256, "product manifest SHA-256"),
            (&result.product_payload_sha256, "product payload SHA-256"),
            (
                &result.geopotential_manifest_sha256,
                "geopotential manifest SHA-256",
            ),
            (
                &result.geopotential_payload_sha256,
                "geopotential payload SHA-256",
            ),
            (&result.source_artifact_sha256, "source artifact SHA-256"),
            (&result.source_contract_sha256, "source contract SHA-256"),
        ] {
            validate_sha256(value, label)?;
        }
        if result.source_artifact_sha256 != SEALED_SOURCE_ARTIFACT_SHA256
            || result.source_contract_sha256 != SEALED_SOURCE_CONTRACT_SHA256
        {
            return Err(invalid(
                "transition checkpoint source is not the sealed corrected-v3 artifact and contract",
            ));
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionCarrier {
    pub name: String,
    pub role: String,
    pub shape: Vec<usize>,
    pub offset_nbytes: u64,
    pub nbytes: u64,
    pub sha256: String,
    pub finite_count: usize,
    pub nonzero_count: usize,
    pub minimum_f32_bits: u32,
    pub maximum_f32_bits: u32,
}

impl TransitionCarrier {
    fn statistics_json(&self) -> JsonValue {
        object(vec![
            ("finite_count", number(self.finite_count)),
            (
                "maximum_f32_bits",
                string(&format!("{:08x}", self.maximum_f32_bits)),
            ),
            (
                "minimum_f32_bits",
                string(&format!("{:08x}", self.minimum_f32_bits)),
            ),
            ("nonzero_count", number(self.nonzero_count)),
        ])
    }

    fn header_json(&self) -> JsonValue {
        object(vec![
            ("dtype", string("<f4")),
            ("itemsize", number(4)),
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
            ("species_order", species_json(&self.name)),
            ("statistics", self.statistics_json()),
        ])
    }

    fn restore_json(&self) -> JsonValue {
        object(vec![
            ("dtype", string("<f4")),
            ("file", string(&capture_file_name(&self.name))),
            ("itemsize", number(4)),
            ("name", string(&self.name)),
            ("nbytes", u64_number(self.nbytes)),
            ("order", string("C")),
            ("role", string(&self.role)),
            (
                "shape",
                JsonValue::Array(self.shape.iter().copied().map(number).collect()),
            ),
            ("sha256", string(&self.sha256)),
            ("species_order", species_json(&self.name)),
            ("statistics", self.statistics_json()),
        ])
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionCheckpointReceipt {
    pub artifact_sha256: String,
    pub artifact_nbytes: u64,
    pub header_sha256: String,
    pub header_nbytes: usize,
    pub payload_sha256: String,
    pub payload_nbytes: u64,
    pub carrier_count: usize,
    pub completed_transitions: u64,
    pub source: TransitionSourceIdentities,
}

impl TransitionCheckpointReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("artifact_nbytes", u64_number(self.artifact_nbytes)),
            ("artifact_sha256", string(&self.artifact_sha256)),
            ("capture_phase", string(WSM6_TRANSITION_CAPTURE_PHASE)),
            ("carrier_abi", string(WSM6_TRANSITION_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            ("claim_limit", string(WSM6_TRANSITION_CLAIM_LIMIT)),
            (
                "completed_transitions",
                u64_number(self.completed_transitions),
            ),
            ("header_nbytes", number(self.header_nbytes)),
            ("header_sha256", string(&self.header_sha256)),
            ("payload_nbytes", u64_number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            (
                "publication_contract",
                string(WSM6_TRANSITION_PUBLICATION_CONTRACT),
            ),
            ("restore_phase", string(WSM6_TRANSITION_RESTORE_PHASE)),
            ("schema", string(WSM6_TRANSITION_CHECKPOINT_RECEIPT_SCHEMA)),
            (
                "source_artifact_sha256",
                string(&self.source.source_artifact_sha256),
            ),
            (
                "source_contract_sha256",
                string(&self.source.source_contract_sha256),
            ),
            (
                "source_geopotential_manifest_sha256",
                string(&self.source.geopotential_manifest_sha256),
            ),
            (
                "source_geopotential_payload_sha256",
                string(&self.source.geopotential_payload_sha256),
            ),
            (
                "source_product_manifest_sha256",
                string(&self.source.product_manifest_sha256),
            ),
            (
                "source_product_payload_sha256",
                string(&self.source.product_payload_sha256),
            ),
        ]);
        String::from_utf8(value.canonical_bytes())
            .expect("canonical transition checkpoint receipt is UTF-8")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransitionRestoreReceipt {
    pub checkpoint_artifact_sha256: String,
    pub manifest_sha256: String,
    pub manifest_nbytes: usize,
    pub payload_sha256: String,
    pub payload_nbytes: u64,
    pub carrier_count: usize,
    pub completed_transitions: u64,
}

impl TransitionRestoreReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("carrier_abi", string(WSM6_TRANSITION_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            (
                "checkpoint_artifact_sha256",
                string(&self.checkpoint_artifact_sha256),
            ),
            ("claim_limit", string(WSM6_TRANSITION_CLAIM_LIMIT)),
            (
                "completed_transitions",
                u64_number(self.completed_transitions),
            ),
            ("manifest_nbytes", number(self.manifest_nbytes)),
            ("manifest_sha256", string(&self.manifest_sha256)),
            ("payload_nbytes", u64_number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            (
                "publication_contract",
                string(WSM6_TRANSITION_RESTORE_PUBLICATION_CONTRACT),
            ),
            ("schema", string(WSM6_TRANSITION_RESTORE_RECEIPT_SCHEMA)),
        ]);
        String::from_utf8(value.canonical_bytes())
            .expect("canonical transition restore receipt is UTF-8")
    }
}

#[derive(Clone, Debug)]
pub struct TransitionCheckpoint {
    path: PathBuf,
    payload_offset: u64,
    header_bytes: Vec<u8>,
    config: TransitionCheckpointConfig,
    source: TransitionSourceIdentities,
    carriers: Vec<TransitionCarrier>,
    receipt: TransitionCheckpointReceipt,
}

impl TransitionCheckpoint {
    pub fn config(&self) -> &TransitionCheckpointConfig {
        &self.config
    }

    pub fn source(&self) -> &TransitionSourceIdentities {
        &self.source
    }

    pub fn carriers(&self) -> &[TransitionCarrier] {
        &self.carriers
    }

    pub fn receipt(&self) -> &TransitionCheckpointReceipt {
        &self.receipt
    }
}

pub fn write_wsm6_transition_checkpoint_new(
    capture_directory: impl AsRef<Path>,
    output: impl AsRef<Path>,
    config: TransitionCheckpointConfig,
    source: TransitionSourceIdentities,
) -> Result<TransitionCheckpointReceipt> {
    let capture_directory = capture_directory.as_ref();
    verify_capture_inventory(capture_directory)?;
    let mut carriers = Vec::with_capacity(WSM6_TRANSITION_ARRAY_NAMES.len());
    let mut aggregate = Sha256::default();
    let mut offset = 0u64;
    for (index, name) in WSM6_TRANSITION_ARRAY_NAMES.iter().copied().enumerate() {
        let path = capture_directory.join(capture_file_name(name));
        let shape = expected_shape(name, config.ny, config.nx);
        let expected_nbytes = expected_nbytes(&shape, name)?;
        let scan = scan_regular_file(&path, expected_nbytes, name, Some(&mut aggregate))?;
        carriers.push(TransitionCarrier {
            name: name.to_string(),
            role: ROLES[index].to_string(),
            shape,
            offset_nbytes: offset,
            nbytes: expected_nbytes,
            sha256: scan.sha256,
            finite_count: scan.finite_count,
            nonzero_count: scan.nonzero_count,
            minimum_f32_bits: scan.minimum_f32_bits,
            maximum_f32_bits: scan.maximum_f32_bits,
        });
        offset = offset
            .checked_add(expected_nbytes)
            .ok_or_else(|| invalid("transition checkpoint payload offset overflows"))?;
    }
    if offset > MAX_PAYLOAD_BYTES {
        return Err(invalid(
            "transition checkpoint payload exceeds the v1 one-GiB limit",
        ));
    }
    let payload_sha256 = sha256::hex(&aggregate.finalize());
    let header = header_value(&config, &source, &carriers, offset, &payload_sha256);
    let header_bytes = header.canonical_bytes();
    if header_bytes.len() as u64 > MAX_HEADER_BYTES {
        return Err(invalid(
            "transition checkpoint header exceeds the one-MiB limit",
        ));
    }
    let mut prefix = Vec::with_capacity(PREFIX_BYTES as usize);
    prefix.extend_from_slice(&WSM6_TRANSITION_CHECKPOINT_MAGIC);
    prefix.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());

    let output = output.as_ref();
    refuse_existing_file(output, "transition checkpoint destination")?;
    let temporary = temporary_file_path(output)?;
    let result = (|| -> Result<TransitionCheckpointReceipt> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let mut writer = BufWriter::new(file);
        let mut artifact = Sha256::default();
        let mut copied_aggregate = Sha256::default();
        writer.write_all(&prefix)?;
        writer.write_all(&header_bytes)?;
        artifact.update(&prefix);
        artifact.update(&header_bytes);
        for carrier in &carriers {
            let path = capture_directory.join(capture_file_name(&carrier.name));
            let (mut reader, actual_nbytes) = open_regular_file(&path, &carrier.name)?;
            if actual_nbytes != carrier.nbytes {
                return Err(invalid(format!(
                    "capture file {} size changed after preflight",
                    carrier.name
                )));
            }
            let scan = scan_stream(
                &mut reader,
                carrier.nbytes,
                &carrier.name,
                Some(&mut writer),
                Some(&mut copied_aggregate),
                Some(&mut artifact),
            )?;
            ensure_eof(&mut reader, &carrier.name)?;
            require_scan_matches(&scan, carrier, "capture file changed after preflight")?;
        }
        let copied_payload_sha256 = sha256::hex(&copied_aggregate.finalize());
        if copied_payload_sha256 != payload_sha256 {
            return Err(invalid(
                "transition capture aggregate changed after preflight",
            ));
        }
        writer.flush()?;
        let file = writer.into_inner().map_err(|error| error.into_error())?;
        file.sync_all()?;
        let artifact_nbytes = PREFIX_BYTES
            .checked_add(header_bytes.len() as u64)
            .and_then(|value| value.checked_add(offset))
            .ok_or_else(|| invalid("transition checkpoint size overflows"))?;
        if file.metadata()?.len() != artifact_nbytes {
            return Err(invalid(
                "transition checkpoint size changed after synchronized write",
            ));
        }
        drop(file);
        let receipt = TransitionCheckpointReceipt {
            artifact_sha256: sha256::hex(&artifact.finalize()),
            artifact_nbytes,
            header_sha256: sha256::digest_hex(&header_bytes),
            header_nbytes: header_bytes.len(),
            payload_sha256: payload_sha256.clone(),
            payload_nbytes: offset,
            carrier_count: carriers.len(),
            completed_transitions: config.completed_transitions,
            source: source.clone(),
        };
        fs::hard_link(&temporary, output).map_err(|error| {
            invalid(format!(
                "could not publish new transition checkpoint {} without replacement: {error}",
                output.display()
            ))
        })?;
        fs::remove_file(&temporary)?;
        Ok(receipt)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub fn read_wsm6_transition_checkpoint(
    input: impl AsRef<Path>,
    expected_artifact_sha256: &str,
    expected_config: &TransitionCheckpointConfig,
    expected_source: &TransitionSourceIdentities,
) -> Result<TransitionCheckpoint> {
    validate_sha256(
        expected_artifact_sha256,
        "expected transition checkpoint SHA-256",
    )?;
    let input = input.as_ref();
    let (mut reader, file_nbytes) = open_regular_file(input, "transition checkpoint")?;
    if file_nbytes < PREFIX_BYTES {
        return Err(invalid(
            "transition checkpoint is truncated before its prefix",
        ));
    }
    if file_nbytes
        > PREFIX_BYTES
            .checked_add(MAX_HEADER_BYTES)
            .and_then(|value| value.checked_add(MAX_PAYLOAD_BYTES))
            .expect("v1 maximum fits u64")
    {
        return Err(invalid("transition checkpoint exceeds the v1 size limit"));
    }
    let mut prefix = [0u8; PREFIX_BYTES as usize];
    reader.read_exact(&mut prefix)?;
    if prefix[..8] != WSM6_TRANSITION_CHECKPOINT_MAGIC {
        return Err(invalid(
            "transition checkpoint magic/version does not identify CM1W6R1",
        ));
    }
    let header_nbytes = u64::from_le_bytes(prefix[8..].try_into().expect("fixed prefix"));
    if header_nbytes > MAX_HEADER_BYTES {
        return Err(invalid("transition checkpoint header exceeds the v1 limit"));
    }
    let mut header_bytes = vec![
        0u8;
        usize::try_from(header_nbytes).map_err(|_| invalid(
            "transition header size does not fit this platform"
        ))?
    ];
    reader.read_exact(&mut header_bytes)?;
    let header = parse_canonical_json(&header_bytes, "transition checkpoint header")?;
    let (config, source, carriers, payload_nbytes, payload_sha256) =
        validate_header(&header, expected_config, expected_source)?;
    let expected_nbytes = PREFIX_BYTES
        .checked_add(header_nbytes)
        .and_then(|value| value.checked_add(payload_nbytes))
        .ok_or_else(|| invalid("transition checkpoint size overflows"))?;
    if file_nbytes != expected_nbytes {
        return Err(invalid(format!(
            "transition checkpoint contains {file_nbytes} bytes; header and carriers require {expected_nbytes}"
        )));
    }
    let mut artifact = Sha256::default();
    artifact.update(&prefix);
    artifact.update(&header_bytes);
    let mut aggregate = Sha256::default();
    for carrier in &carriers {
        let scan = scan_stream(
            &mut reader,
            carrier.nbytes,
            &carrier.name,
            None,
            Some(&mut aggregate),
            Some(&mut artifact),
        )?;
        require_scan_matches(&scan, carrier, "checkpoint carrier validation")?;
    }
    ensure_eof(&mut reader, "transition checkpoint")?;
    let actual_payload_sha256 = sha256::hex(&aggregate.finalize());
    if actual_payload_sha256 != payload_sha256 {
        return Err(invalid(format!(
            "transition payload SHA-256 is {actual_payload_sha256}; header declares {payload_sha256}"
        )));
    }
    let artifact_sha256 = sha256::hex(&artifact.finalize());
    if artifact_sha256 != expected_artifact_sha256 {
        return Err(invalid(format!(
            "transition checkpoint SHA-256 is {artifact_sha256}; independently expected {expected_artifact_sha256}"
        )));
    }
    let receipt = TransitionCheckpointReceipt {
        artifact_sha256,
        artifact_nbytes: file_nbytes,
        header_sha256: sha256::digest_hex(&header_bytes),
        header_nbytes: header_bytes.len(),
        payload_sha256,
        payload_nbytes,
        carrier_count: carriers.len(),
        completed_transitions: config.completed_transitions,
        source: source.clone(),
    };
    Ok(TransitionCheckpoint {
        path: input.to_path_buf(),
        payload_offset: PREFIX_BYTES + header_nbytes,
        header_bytes,
        config,
        source,
        carriers,
        receipt,
    })
}

pub fn restore_wsm6_transition_checkpoint_new(
    checkpoint: &TransitionCheckpoint,
    output_directory: impl AsRef<Path>,
) -> Result<TransitionRestoreReceipt> {
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace transition restore destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "transition restore parent does not exist: {}",
            parent.display()
        )));
    }
    let staging = temporary_directory_path(output)?;
    fs::create_dir(&staging)?;
    let result = (|| -> Result<TransitionRestoreReceipt> {
        let (mut reader, file_nbytes) =
            open_regular_file(&checkpoint.path, "transition checkpoint")?;
        if file_nbytes != checkpoint.receipt.artifact_nbytes {
            return Err(invalid(
                "transition checkpoint size changed after verified read",
            ));
        }
        let prefix_header_len = usize::try_from(checkpoint.payload_offset)
            .map_err(|_| invalid("transition prefix/header does not fit this platform"))?;
        let mut prefix_header = vec![0u8; prefix_header_len];
        reader.read_exact(&mut prefix_header)?;
        let mut expected_prefix = Vec::with_capacity(PREFIX_BYTES as usize);
        expected_prefix.extend_from_slice(&WSM6_TRANSITION_CHECKPOINT_MAGIC);
        expected_prefix.extend_from_slice(&(checkpoint.header_bytes.len() as u64).to_le_bytes());
        if prefix_header[..PREFIX_BYTES as usize] != expected_prefix
            || prefix_header[PREFIX_BYTES as usize..] != checkpoint.header_bytes
        {
            return Err(invalid(
                "transition checkpoint prefix or header changed after verified read",
            ));
        }
        let mut artifact = Sha256::default();
        artifact.update(&prefix_header);
        let mut aggregate = Sha256::default();
        for carrier in &checkpoint.carriers {
            let path = staging.join(capture_file_name(&carrier.name));
            let file = OpenOptions::new().write(true).create_new(true).open(path)?;
            let mut writer = BufWriter::new(file);
            let scan = scan_stream(
                &mut reader,
                carrier.nbytes,
                &carrier.name,
                Some(&mut writer),
                Some(&mut aggregate),
                Some(&mut artifact),
            )?;
            require_scan_matches(&scan, carrier, "transition restore carrier validation")?;
            writer.flush()?;
            let file = writer.into_inner().map_err(|error| error.into_error())?;
            file.sync_all()?;
            if file.metadata()?.len() != carrier.nbytes {
                return Err(invalid(format!(
                    "restored carrier {} size changed after sync",
                    carrier.name
                )));
            }
        }
        ensure_eof(&mut reader, "transition checkpoint")?;
        let payload_sha256 = sha256::hex(&aggregate.finalize());
        if payload_sha256 != checkpoint.receipt.payload_sha256 {
            return Err(invalid(
                "transition payload changed after verified read; restore refused",
            ));
        }
        let artifact_sha256 = sha256::hex(&artifact.finalize());
        if artifact_sha256 != checkpoint.receipt.artifact_sha256 {
            return Err(invalid(
                "transition checkpoint changed after verified read; restore refused",
            ));
        }
        let manifest = restore_manifest_value(checkpoint);
        validate_restore_manifest(&manifest, checkpoint)?;
        let manifest_bytes = manifest.canonical_bytes();
        write_new_synced(&staging.join("manifest.json"), &manifest_bytes)?;
        verify_restore_inventory(&staging)?;
        fs::rename(&staging, output)?;
        Ok(TransitionRestoreReceipt {
            checkpoint_artifact_sha256: artifact_sha256,
            manifest_sha256: sha256::digest_hex(&manifest_bytes),
            manifest_nbytes: manifest_bytes.len(),
            payload_sha256,
            payload_nbytes: checkpoint.receipt.payload_nbytes,
            carrier_count: checkpoint.carriers.len(),
            completed_transitions: checkpoint.config.completed_transitions,
        })
    })();
    if result.is_err() {
        cleanup_owned_staging_directory(&staging);
    }
    result
}

#[derive(Debug)]
struct Scan {
    sha256: String,
    finite_count: usize,
    nonzero_count: usize,
    minimum_f32_bits: u32,
    maximum_f32_bits: u32,
}

fn scan_regular_file(
    path: &Path,
    expected_nbytes: u64,
    name: &str,
    aggregate: Option<&mut Sha256>,
) -> Result<Scan> {
    let (mut reader, actual_nbytes) = open_regular_file(path, name)?;
    if actual_nbytes != expected_nbytes {
        return Err(invalid(format!(
            "capture file {name} contains {actual_nbytes} bytes; shape requires {expected_nbytes}"
        )));
    }
    let scan = scan_stream(&mut reader, expected_nbytes, name, None, aggregate, None)?;
    ensure_eof(&mut reader, name)?;
    Ok(scan)
}

fn scan_stream<R: Read>(
    reader: &mut R,
    nbytes: u64,
    name: &str,
    mut writer: Option<&mut BufWriter<File>>,
    mut aggregate: Option<&mut Sha256>,
    mut artifact: Option<&mut Sha256>,
) -> Result<Scan> {
    if nbytes == 0 || !nbytes.is_multiple_of(4) {
        return Err(invalid(format!(
            "transition carrier {name} byte count is empty or unaligned"
        )));
    }
    let mut remaining = nbytes;
    let mut buffer = vec![0u8; IO_CHUNK_BYTES];
    let mut field = Sha256::default();
    let mut finite_count = 0usize;
    let mut nonzero_count = 0usize;
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    while remaining != 0 {
        let take = usize::try_from(remaining.min(buffer.len() as u64))
            .expect("bounded I/O chunk fits usize");
        reader.read_exact(&mut buffer[..take])?;
        let bytes = &buffer[..take];
        if let Some(destination) = writer.as_deref_mut() {
            destination.write_all(bytes)?;
        }
        field.update(bytes);
        if let Some(value) = aggregate.as_deref_mut() {
            value.update(bytes);
        }
        if let Some(value) = artifact.as_deref_mut() {
            value.update(bytes);
        }
        for word in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes(word.try_into().expect("four-byte word"));
            if !value.is_finite() {
                return Err(invalid(format!(
                    "transition carrier {name} contains non-finite word {finite_count}"
                )));
            }
            finite_count += 1;
            if value.to_bits() & 0x7fff_ffff != 0 {
                nonzero_count += 1;
            }
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        remaining -= take as u64;
    }
    Ok(Scan {
        sha256: sha256::hex(&field.finalize()),
        finite_count,
        nonzero_count,
        minimum_f32_bits: minimum.to_bits(),
        maximum_f32_bits: maximum.to_bits(),
    })
}

fn require_scan_matches(scan: &Scan, carrier: &TransitionCarrier, context: &str) -> Result<()> {
    if scan.sha256 != carrier.sha256
        || scan.finite_count != carrier.finite_count
        || scan.nonzero_count != carrier.nonzero_count
        || scan.minimum_f32_bits != carrier.minimum_f32_bits
        || scan.maximum_f32_bits != carrier.maximum_f32_bits
    {
        return Err(invalid(format!(
            "{context}: transition carrier {} bytes or statistics differ",
            carrier.name
        )));
    }
    Ok(())
}

fn header_value(
    config: &TransitionCheckpointConfig,
    source: &TransitionSourceIdentities,
    carriers: &[TransitionCarrier],
    payload_nbytes: u64,
    payload_sha256: &str,
) -> JsonValue {
    object(vec![
        ("capture_phase", string(WSM6_TRANSITION_CAPTURE_PHASE)),
        ("carrier_abi", string(WSM6_TRANSITION_CARRIER_ABI)),
        (
            "carriers",
            JsonValue::Array(
                carriers
                    .iter()
                    .map(TransitionCarrier::header_json)
                    .collect(),
            ),
        ),
        ("claim_limit", string(WSM6_TRANSITION_CLAIM_LIMIT)),
        ("configuration", configuration_json(config)),
        ("dtype", string("<f4")),
        ("itemsize", number(4)),
        ("order", string("C")),
        ("payload_nbytes", u64_number(payload_nbytes)),
        ("payload_sha256", string(payload_sha256)),
        ("restore_phase", string(WSM6_TRANSITION_RESTORE_PHASE)),
        ("schema", string(WSM6_TRANSITION_CHECKPOINT_SCHEMA)),
        ("source", source_json(source)),
        ("transition", transition_json()),
    ])
}

fn configuration_json(config: &TransitionCheckpointConfig) -> JsonValue {
    object(vec![
        (
            "completed_bridge_calls",
            u64_number(config.completed_bridge_calls),
        ),
        (
            "completed_component_calls",
            u64_number(config.completed_component_calls),
        ),
        (
            "completed_transitions",
            u64_number(config.completed_transitions),
        ),
        (
            "crop_count_yx",
            JsonValue::Array([config.ny, config.nx].into_iter().map(number).collect()),
        ),
        (
            "crop_origin_yx",
            JsonValue::Array(
                [config.j_start, config.i_start]
                    .into_iter()
                    .map(number)
                    .collect(),
            ),
        ),
        ("dt_f64_bits", string("3fe0000000000000")),
        ("dx_f64_bits", string("4059000000000000")),
        ("dy_f64_bits", string("4059000000000000")),
        ("dz_f64_bits", string("4069000000000000")),
        ("efall", JsonValue::Bool(false)),
        ("eqtset", number(2)),
        ("idiss", JsonValue::Bool(false)),
        ("native_ptype_context", number(5)),
        ("nz", number(NZ)),
        ("reflectivity", JsonValue::Bool(false)),
        ("semantic_component_id", string("wsm6-mp6")),
    ])
}

fn source_json(source: &TransitionSourceIdentities) -> JsonValue {
    object(vec![
        (
            "geopotential_manifest_sha256",
            string(&source.geopotential_manifest_sha256),
        ),
        (
            "geopotential_payload_sha256",
            string(&source.geopotential_payload_sha256),
        ),
        (
            "product_manifest_sha256",
            string(&source.product_manifest_sha256),
        ),
        (
            "product_payload_sha256",
            string(&source.product_payload_sha256),
        ),
        (
            "source_artifact_sha256",
            string(&source.source_artifact_sha256),
        ),
        (
            "source_contract_sha256",
            string(&source.source_contract_sha256),
        ),
    ])
}

fn transition_json() -> JsonValue {
    object(vec![
        ("bridge_abi", string(WSM6_TRANSITION_BRIDGE_ABI)),
        ("component_abi", string(WSM6_TRANSITION_COMPONENT_ABI)),
        (
            "geopotential_carrier_abi",
            string(WSM6_GEOPOTENTIAL_CARRIER_ABI),
        ),
        (
            "in_memory_restart_schema",
            string(WSM6_TRANSITION_IN_MEMORY_RESTART_SCHEMA),
        ),
        (
            "product_carrier_abi",
            string(INITIALIZED_PRODUCT_CARRIER_ABI),
        ),
        ("transition_abi", string(WSM6_TRANSITION_ABI)),
    ])
}

fn validate_header(
    header: &JsonValue,
    expected_config: &TransitionCheckpointConfig,
    expected_source: &TransitionSourceIdentities,
) -> Result<(
    TransitionCheckpointConfig,
    TransitionSourceIdentities,
    Vec<TransitionCarrier>,
    u64,
    String,
)> {
    exact_keys(header, HEADER_KEYS, "transition checkpoint header")?;
    fixed_string(
        header,
        "schema",
        WSM6_TRANSITION_CHECKPOINT_SCHEMA,
        "transition checkpoint header",
    )?;
    fixed_string(
        header,
        "carrier_abi",
        WSM6_TRANSITION_CARRIER_ABI,
        "transition checkpoint header",
    )?;
    fixed_string(
        header,
        "capture_phase",
        WSM6_TRANSITION_CAPTURE_PHASE,
        "transition checkpoint header",
    )?;
    fixed_string(
        header,
        "restore_phase",
        WSM6_TRANSITION_RESTORE_PHASE,
        "transition checkpoint header",
    )?;
    fixed_string(
        header,
        "claim_limit",
        WSM6_TRANSITION_CLAIM_LIMIT,
        "transition checkpoint header",
    )?;
    fixed_string(header, "dtype", "<f4", "transition checkpoint header")?;
    fixed_string(header, "order", "C", "transition checkpoint header")?;
    fixed_u64(header, "itemsize", 4, "transition checkpoint header")?;
    let transition = member(header, "transition", "transition checkpoint header")?;
    exact_keys(
        transition,
        TRANSITION_KEYS,
        "transition checkpoint transition",
    )?;
    if transition != &transition_json() {
        return Err(invalid(
            "transition checkpoint ABI inventory differs from the frozen v1 seam",
        ));
    }
    let configuration = member(header, "configuration", "transition checkpoint header")?;
    exact_keys(
        configuration,
        CONFIGURATION_KEYS,
        "transition checkpoint configuration",
    )?;
    if configuration != &configuration_json(expected_config) {
        return Err(invalid(
            "transition checkpoint configuration or cadence differs from the independent expectation",
        ));
    }
    let source_value = member(header, "source", "transition checkpoint header")?;
    exact_keys(source_value, SOURCE_KEYS, "transition checkpoint source")?;
    if source_value != &source_json(expected_source) {
        return Err(invalid(
            "transition checkpoint product, geopotential, artifact, or contract source differs from the independent expectation",
        ));
    }
    let payload_nbytes = member_u64(header, "payload_nbytes", "transition checkpoint header")?;
    if payload_nbytes == 0
        || payload_nbytes > MAX_PAYLOAD_BYTES
        || !payload_nbytes.is_multiple_of(4)
    {
        return Err(invalid(
            "transition checkpoint payload is empty, unaligned, or exceeds the v1 limit",
        ));
    }
    let payload_sha256 =
        member_str(header, "payload_sha256", "transition checkpoint header")?.to_string();
    validate_sha256(&payload_sha256, "transition payload SHA-256")?;
    let carrier_values = member(header, "carriers", "transition checkpoint header")?
        .as_array()
        .map_err(|error| invalid(format!("transition checkpoint carriers: {error}")))?;
    if carrier_values.len() != WSM6_TRANSITION_ARRAY_NAMES.len() {
        return Err(invalid(
            "transition checkpoint does not contain exactly 21 carriers",
        ));
    }
    let mut carriers = Vec::with_capacity(carrier_values.len());
    let mut offset = 0u64;
    for (index, (value, expected_name)) in carrier_values
        .iter()
        .zip(WSM6_TRANSITION_ARRAY_NAMES)
        .enumerate()
    {
        let label = format!("transition checkpoint carriers[{index}]");
        exact_keys(value, CARRIER_KEYS, &label)?;
        fixed_string(value, "dtype", "<f4", &label)?;
        fixed_string(value, "order", "C", &label)?;
        fixed_u64(value, "itemsize", 4, &label)?;
        fixed_string(value, "name", expected_name, &label)?;
        fixed_string(value, "role", ROLES[index], &label)?;
        if member(value, "species_order", &label)? != &species_json(expected_name) {
            return Err(invalid(format!(
                "{label}.species_order differs from the packed-moisture ABI"
            )));
        }
        let shape = usize_array(member(value, "shape", &label)?, &format!("{label}.shape"))?;
        let expected_shape = expected_shape(expected_name, expected_config.ny, expected_config.nx);
        if shape != expected_shape {
            return Err(invalid(format!(
                "{label}.shape is {shape:?}; required {expected_shape:?}"
            )));
        }
        let nbytes = expected_nbytes(&shape, expected_name)?;
        fixed_u64(value, "nbytes", nbytes, &label)?;
        fixed_u64(value, "offset_nbytes", offset, &label)?;
        let digest = member_str(value, "sha256", &label)?.to_string();
        validate_sha256(&digest, &format!("{label}.sha256"))?;
        let statistics = member(value, "statistics", &label)?;
        exact_keys(statistics, STATS_KEYS, &format!("{label}.statistics"))?;
        let words = usize::try_from(nbytes / 4)
            .map_err(|_| invalid(format!("{label} word count exceeds this platform")))?;
        let finite_count = usize_from_u64(
            member_u64(statistics, "finite_count", &format!("{label}.statistics"))?,
            &format!("{label}.finite_count"),
        )?;
        let nonzero_count = usize_from_u64(
            member_u64(statistics, "nonzero_count", &format!("{label}.statistics"))?,
            &format!("{label}.nonzero_count"),
        )?;
        if finite_count != words || nonzero_count > words {
            return Err(invalid(format!(
                "{label} statistics counts differ from the shape"
            )));
        }
        let minimum_f32_bits = parse_hex_u32(member_str(
            statistics,
            "minimum_f32_bits",
            &format!("{label}.statistics"),
        )?)?;
        let maximum_f32_bits = parse_hex_u32(member_str(
            statistics,
            "maximum_f32_bits",
            &format!("{label}.statistics"),
        )?)?;
        if !f32::from_bits(minimum_f32_bits).is_finite()
            || !f32::from_bits(maximum_f32_bits).is_finite()
        {
            return Err(invalid(format!("{label} minimum or maximum is non-finite")));
        }
        carriers.push(TransitionCarrier {
            name: expected_name.to_string(),
            role: ROLES[index].to_string(),
            shape,
            offset_nbytes: offset,
            nbytes,
            sha256: digest,
            finite_count,
            nonzero_count,
            minimum_f32_bits,
            maximum_f32_bits,
        });
        offset = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("transition carrier offset overflows"))?;
    }
    if offset != payload_nbytes {
        return Err(invalid(format!(
            "transition carriers cover {offset} bytes; payload declares {payload_nbytes}"
        )));
    }
    Ok((
        expected_config.clone(),
        expected_source.clone(),
        carriers,
        payload_nbytes,
        payload_sha256,
    ))
}

fn restore_manifest_value(checkpoint: &TransitionCheckpoint) -> JsonValue {
    object(vec![
        (
            "arrays",
            JsonValue::Array(
                checkpoint
                    .carriers
                    .iter()
                    .map(TransitionCarrier::restore_json)
                    .collect(),
            ),
        ),
        ("capture_phase", string(WSM6_TRANSITION_CAPTURE_PHASE)),
        ("carrier_abi", string(WSM6_TRANSITION_CARRIER_ABI)),
        ("claim_limit", string(WSM6_TRANSITION_CLAIM_LIMIT)),
        (
            "completed_transitions",
            u64_number(checkpoint.config.completed_transitions),
        ),
        ("restore_phase", string(WSM6_TRANSITION_RESTORE_PHASE)),
        ("schema", string(WSM6_TRANSITION_RESTORE_MANIFEST_SCHEMA)),
        ("source", source_json(&checkpoint.source)),
        (
            "source_checkpoint_artifact_sha256",
            string(&checkpoint.receipt.artifact_sha256),
        ),
    ])
}

fn validate_restore_manifest(
    manifest: &JsonValue,
    checkpoint: &TransitionCheckpoint,
) -> Result<()> {
    exact_keys(
        manifest,
        RESTORE_MANIFEST_KEYS,
        "transition restore manifest",
    )?;
    fixed_string(
        manifest,
        "schema",
        WSM6_TRANSITION_RESTORE_MANIFEST_SCHEMA,
        "transition restore manifest",
    )?;
    let arrays = member(manifest, "arrays", "transition restore manifest")?
        .as_array()
        .map_err(|error| invalid(format!("transition restore arrays: {error}")))?;
    if arrays.len() != checkpoint.carriers.len() {
        return Err(invalid("transition restore array inventory drifted"));
    }
    for (index, (array, carrier)) in arrays.iter().zip(&checkpoint.carriers).enumerate() {
        exact_keys(
            array,
            RESTORE_ARRAY_KEYS,
            &format!("transition restore arrays[{index}]"),
        )?;
        if array.canonical_bytes() != carrier.restore_json().canonical_bytes() {
            return Err(invalid(format!(
                "transition restore array {} metadata drifted",
                carrier.name
            )));
        }
    }
    Ok(())
}

fn expected_shape(name: &str, ny: usize, nx: usize) -> Vec<usize> {
    match name {
        "state.q3d" => vec![6, NZ, ny, nx],
        "component.mp_rainnc"
        | "component.mp_rainncv"
        | "component.mp_snownc"
        | "component.mp_snowncv"
        | "component.mp_graupelnc"
        | "component.mp_graupelncv"
        | "component.mp_sr" => vec![ny, nx],
        _ => vec![NZ, ny, nx],
    }
}

fn expected_nbytes(shape: &[usize], name: &str) -> Result<u64> {
    let words = shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("transition carrier {name} shape overflows")))
    })?;
    u64::try_from(
        words
            .checked_mul(4)
            .ok_or_else(|| invalid(format!("transition carrier {name} bytes overflow")))?,
    )
    .map_err(|_| invalid(format!("transition carrier {name} bytes exceed u64")))
}

fn species_json(name: &str) -> JsonValue {
    if name == "state.q3d" {
        JsonValue::Array(SPECIES.into_iter().map(string).collect())
    } else {
        JsonValue::Array(Vec::new())
    }
}

fn capture_file_name(name: &str) -> String {
    format!("{name}.f32le")
}

fn verify_capture_inventory(directory: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(directory).map_err(|error| {
        invalid(format!(
            "could not inspect transition capture directory {}: {error}",
            directory.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid("transition capture input is not a real directory"));
    }
    let expected = WSM6_TRANSITION_ARRAY_NAMES
        .iter()
        .map(|name| capture_file_name(name))
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| invalid("transition capture directory has a non-UTF-8 entry"))?;
        actual.insert(name);
    }
    if actual != expected {
        return Err(invalid(format!(
            "transition capture inventory is {actual:?}; exactly {expected:?} is required"
        )));
    }
    for name in WSM6_TRANSITION_ARRAY_NAMES {
        let path = directory.join(capture_file_name(name));
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(invalid(format!(
                "transition capture entry {} is not a real regular file",
                path.display()
            )));
        }
    }
    Ok(())
}

fn verify_restore_inventory(directory: &Path) -> Result<()> {
    let mut expected = WSM6_TRANSITION_ARRAY_NAMES
        .iter()
        .map(|name| capture_file_name(name))
        .collect::<BTreeSet<_>>();
    expected.insert("manifest.json".to_string());
    let actual = fs::read_dir(directory)?
        .map(|entry| {
            entry?
                .file_name()
                .into_string()
                .map_err(|_| std::io::Error::other("non-UTF-8 restore entry"))
        })
        .collect::<std::io::Result<BTreeSet<_>>>()?;
    if actual != expected {
        return Err(invalid(
            "transition restore directory inventory drifted before publication",
        ));
    }
    Ok(())
}

fn cleanup_owned_staging_directory(directory: &Path) {
    for name in WSM6_TRANSITION_ARRAY_NAMES {
        let _ = fs::remove_file(directory.join(capture_file_name(name)));
    }
    let _ = fs::remove_file(directory.join("manifest.json"));
    // Deliberately non-recursive: an unexpected entry must never broaden the
    // cleanup target beyond the files this publisher owns by exact name.
    let _ = fs::remove_dir(directory);
}

fn open_regular_file(path: &Path, label: &str) -> Result<(BufReader<File>, u64)> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(format!(
            "{label} is not a real regular file: {}",
            path.display()
        )));
    }
    Ok((BufReader::new(File::open(path)?), metadata.len()))
}

fn ensure_eof<R: Read>(reader: &mut R, label: &str) -> Result<()> {
    let mut byte = [0u8; 1];
    if reader.read(&mut byte)? != 0 {
        return Err(invalid(format!("{label} contains trailing bytes")));
    }
    Ok(())
}

fn refuse_existing_file(path: &Path, label: &str) -> Result<()> {
    if fs::symlink_metadata(path).is_ok() {
        return Err(invalid(format!(
            "refusing to replace {label} {}",
            path.display()
        )));
    }
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "{label} parent does not exist: {}",
            parent.display()
        )));
    }
    Ok(())
}

fn temporary_file_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("transition checkpoint needs a UTF-8 file name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(output.with_file_name(format!(".{name}.partial-{}-{sequence}", std::process::id())))
}

fn temporary_directory_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("transition restore needs a UTF-8 directory name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(output.with_file_name(format!(".{name}.restore-{}-{sequence}", std::process::id())))
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn parse_canonical_json(bytes: &[u8], label: &str) -> Result<JsonValue> {
    let value =
        crate::json::parse(bytes).map_err(|error| invalid(format!("invalid {label}: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid(format!(
            "{label} is not canonical sorted-key compact JSON"
        )));
    }
    Ok(value)
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
            "{label} keys are {actual:?}; exactly {expected:?} is required"
        )));
    }
    Ok(())
}

fn member<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a JsonValue> {
    value
        .get(key)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn member_str<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    member(value, key, label)?
        .as_str()
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn member_u64(value: &JsonValue, key: &str, label: &str) -> Result<u64> {
    member(value, key, label)?
        .as_u64()
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn fixed_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = member_str(value, key, label)?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} is {actual:?}; required {expected:?}"
        )));
    }
    Ok(())
}

fn fixed_u64(value: &JsonValue, key: &str, expected: u64, label: &str) -> Result<()> {
    let actual = member_u64(value, key, label)?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} is {actual}; required {expected}"
        )));
    }
    Ok(())
}

fn usize_array(value: &JsonValue, label: &str) -> Result<Vec<usize>> {
    value
        .as_array()
        .map_err(|error| invalid(format!("{label}: {error}")))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let raw = value
                .as_u64()
                .map_err(|error| invalid(format!("{label}[{index}]: {error}")))?;
            usize_from_u64(raw, &format!("{label}[{index}]"))
        })
        .collect()
}

fn usize_from_u64(value: u64, label: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid(format!("{label} exceeds this platform")))
}

fn parse_hex_u32(value: &str) -> Result<u32> {
    if value.len() != 8
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(
            "transition binary32 bits must be eight lower-case hexadecimal characters",
        ));
    }
    u32::from_str_radix(value, 16).map_err(|_| invalid("transition binary32 bits do not parse"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestRoot {
        path: PathBuf,
        temporary_parent: PathBuf,
    }

    impl TestRoot {
        fn new(label: &str) -> Self {
            let temporary_parent = std::env::temp_dir();
            let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = temporary_parent.join(format!(
                "gpuwm-cm1-wsm6-transition-{label}-{}-{sequence}",
                std::process::id()
            ));
            assert_eq!(path.parent(), Some(temporary_parent.as_path()));
            fs::create_dir(&path).expect("create owned transition test root");
            Self {
                path,
                temporary_parent,
            }
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let owned_name = self
                .path
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|value| value.starts_with("gpuwm-cm1-wsm6-transition-"));
            if owned_name && self.path.parent() == Some(self.temporary_parent.as_path()) {
                let _ = fs::remove_dir_all(&self.path);
            }
        }
    }

    fn test_config() -> TransitionCheckpointConfig {
        TransitionCheckpointConfig::new(142, 142, 2, 3, 7, 7, 7).unwrap()
    }

    fn test_source() -> TransitionSourceIdentities {
        TransitionSourceIdentities::new(
            "11".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
            "44".repeat(32),
            SEALED_SOURCE_ARTIFACT_SHA256,
            SEALED_SOURCE_CONTRACT_SHA256,
        )
        .unwrap()
    }

    fn make_capture(root: &Path, config: &TransitionCheckpointConfig) -> PathBuf {
        let capture = root.join("capture");
        fs::create_dir(&capture).unwrap();
        for (field_index, name) in WSM6_TRANSITION_ARRAY_NAMES.iter().enumerate() {
            let shape = expected_shape(name, config.ny, config.nx);
            let words = expected_nbytes(&shape, name).unwrap() as usize / 4;
            let mut bytes = Vec::with_capacity(words * 4);
            for word_index in 0..words {
                let value = (field_index as f32 + 1.0) * 0.25 + word_index as f32 * 0.000_125;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            fs::write(capture.join(capture_file_name(name)), bytes).unwrap();
        }
        capture
    }

    fn write_checkpoint(
        capture: &Path,
        path: &Path,
        config: &TransitionCheckpointConfig,
        source: &TransitionSourceIdentities,
    ) -> TransitionCheckpointReceipt {
        write_wsm6_transition_checkpoint_new(capture, path, config.clone(), source.clone()).unwrap()
    }

    fn read_checkpoint(
        path: &Path,
        receipt: &TransitionCheckpointReceipt,
        config: &TransitionCheckpointConfig,
        source: &TransitionSourceIdentities,
    ) -> TransitionCheckpoint {
        read_wsm6_transition_checkpoint(path, &receipt.artifact_sha256, config, source).unwrap()
    }

    #[test]
    fn deterministic_write_read_restore_is_byte_exact() {
        let root = TestRoot::new("roundtrip");
        let config = test_config();
        let source = test_source();
        let capture = make_capture(&root.path, &config);
        let first_path = root.path.join("first.cm1w6r");
        let second_path = root.path.join("second.cm1w6r");
        let first = write_checkpoint(&capture, &first_path, &config, &source);
        let second = write_checkpoint(&capture, &second_path, &config, &source);
        assert_eq!(first, second);
        assert_eq!(
            fs::read(&first_path).unwrap(),
            fs::read(&second_path).unwrap()
        );
        assert_eq!(first.carrier_count, 21);
        assert_eq!(first.completed_transitions, 7);
        assert!(first
            .canonical_json()
            .contains(WSM6_TRANSITION_CHECKPOINT_RECEIPT_SCHEMA));

        let checkpoint = read_checkpoint(&first_path, &first, &config, &source);
        assert_eq!(checkpoint.carriers().len(), 21);
        assert_eq!(checkpoint.carriers()[5].name, "state.q3d");
        assert_eq!(checkpoint.carriers()[5].shape, [6, 75, 2, 3]);
        assert_eq!(checkpoint.config(), &config);
        assert_eq!(checkpoint.source(), &source);

        let restored = root.path.join("restored");
        let restore_receipt =
            restore_wsm6_transition_checkpoint_new(&checkpoint, &restored).unwrap();
        assert_eq!(
            restore_receipt.checkpoint_artifact_sha256,
            first.artifact_sha256
        );
        assert_eq!(restore_receipt.carrier_count, 21);
        for name in WSM6_TRANSITION_ARRAY_NAMES {
            let file = capture_file_name(name);
            assert_eq!(
                fs::read(capture.join(&file)).unwrap(),
                fs::read(restored.join(&file)).unwrap(),
                "{name}"
            );
        }
        let manifest_bytes = fs::read(restored.join("manifest.json")).unwrap();
        let manifest = parse_canonical_json(&manifest_bytes, "test restore manifest").unwrap();
        validate_restore_manifest(&manifest, &checkpoint).unwrap();
        assert_eq!(
            restore_receipt.manifest_sha256,
            sha256::digest_hex(&manifest_bytes)
        );
        verify_restore_inventory(&restored).unwrap();
    }

    #[test]
    fn refuses_missing_extra_nonfinite_and_nonregular_capture_entries() {
        let root = TestRoot::new("capture-refusals");
        let config = test_config();
        let source = test_source();
        let capture = make_capture(&root.path, &config);
        fs::write(capture.join("extra.bin"), b"unexpected").unwrap();
        let error = write_wsm6_transition_checkpoint_new(
            &capture,
            root.path.join("extra.cm1w6r"),
            config.clone(),
            source.clone(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("exactly") && error.contains("extra.bin"),
            "{error}"
        );
        fs::remove_file(capture.join("extra.bin")).unwrap();

        let missing = capture.join(capture_file_name("state.pp3d"));
        let original = fs::read(&missing).unwrap();
        fs::remove_file(&missing).unwrap();
        let error = write_wsm6_transition_checkpoint_new(
            &capture,
            root.path.join("missing.cm1w6r"),
            config.clone(),
            source.clone(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("exactly") && error.contains("state.pp3d"),
            "{error}"
        );
        fs::write(&missing, &original).unwrap();

        let nonregular = capture.join(capture_file_name("held.qpten"));
        let nonregular_bytes = fs::read(&nonregular).unwrap();
        fs::remove_file(&nonregular).unwrap();
        fs::create_dir(&nonregular).unwrap();
        let error = write_wsm6_transition_checkpoint_new(
            &capture,
            root.path.join("directory-entry.cm1w6r"),
            config.clone(),
            source.clone(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("not a real regular file"), "{error}");
        fs::remove_dir(&nonregular).unwrap();
        fs::write(&nonregular, nonregular_bytes).unwrap();

        let nonfinite = capture.join(capture_file_name("component.effc"));
        let mut bytes = fs::read(&nonfinite).unwrap();
        bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(&nonfinite, bytes).unwrap();
        let error = write_wsm6_transition_checkpoint_new(
            &capture,
            root.path.join("nonfinite.cm1w6r"),
            config,
            source,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("non-finite") && error.contains("component.effc"),
            "{error}"
        );
    }

    #[test]
    fn refuses_wrong_source_cadence_and_existing_destinations() {
        let root = TestRoot::new("identity-refusals");
        let config = test_config();
        let source = test_source();
        let capture = make_capture(&root.path, &config);
        let checkpoint_path = root.path.join("state.cm1w6r");
        let receipt = write_checkpoint(&capture, &checkpoint_path, &config, &source);

        let error = TransitionCheckpointConfig::new(142, 142, 2, 3, 7, 6, 7)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("must be equal") && error.contains("at least one"),
            "{error}"
        );

        let error = TransitionSourceIdentities::new(
            "11".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
            "44".repeat(32),
            "55".repeat(32),
            SEALED_SOURCE_CONTRACT_SHA256,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("sealed corrected-v3"), "{error}");

        let wrong_source = TransitionSourceIdentities::new(
            "aa".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
            "44".repeat(32),
            SEALED_SOURCE_ARTIFACT_SHA256,
            SEALED_SOURCE_CONTRACT_SHA256,
        )
        .unwrap();
        let error = read_wsm6_transition_checkpoint(
            &checkpoint_path,
            &receipt.artifact_sha256,
            &config,
            &wrong_source,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("independent expectation"), "{error}");

        let wrong_config = TransitionCheckpointConfig::new(142, 142, 2, 3, 8, 8, 8).unwrap();
        let error = read_wsm6_transition_checkpoint(
            &checkpoint_path,
            &receipt.artifact_sha256,
            &wrong_config,
            &source,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("configuration or cadence"), "{error}");

        let error = write_wsm6_transition_checkpoint_new(
            &capture,
            &checkpoint_path,
            config.clone(),
            source.clone(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("refusing to replace"), "{error}");
        let checkpoint = read_checkpoint(&checkpoint_path, &receipt, &config, &source);
        let destination = root.path.join("existing-restore");
        fs::create_dir(&destination).unwrap();
        let error = restore_wsm6_transition_checkpoint_new(&checkpoint, &destination)
            .unwrap_err()
            .to_string();
        assert!(error.contains("refusing to replace"), "{error}");
    }

    #[test]
    fn refuses_payload_corruption_size_drift_and_post_read_mutation() {
        let root = TestRoot::new("artifact-refusals");
        let config = test_config();
        let source = test_source();
        let capture = make_capture(&root.path, &config);
        let checkpoint_path = root.path.join("state.cm1w6r");
        let receipt = write_checkpoint(&capture, &checkpoint_path, &config, &source);
        let original = fs::read(&checkpoint_path).unwrap();

        let mut corrupt = original.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        let corrupt_path = root.path.join("corrupt.cm1w6r");
        fs::write(&corrupt_path, &corrupt).unwrap();
        let error = read_wsm6_transition_checkpoint(
            &corrupt_path,
            &sha256::digest_hex(&corrupt),
            &config,
            &source,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("carrier") && error.contains("differ"),
            "{error}"
        );

        let mut truncated = original.clone();
        truncated.pop();
        let truncated_path = root.path.join("truncated.cm1w6r");
        fs::write(&truncated_path, &truncated).unwrap();
        let error = read_wsm6_transition_checkpoint(
            &truncated_path,
            &sha256::digest_hex(&truncated),
            &config,
            &source,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("header and carriers require"), "{error}");

        let mut appended = original.clone();
        appended.push(0);
        let appended_path = root.path.join("appended.cm1w6r");
        fs::write(&appended_path, &appended).unwrap();
        let error = read_wsm6_transition_checkpoint(
            &appended_path,
            &sha256::digest_hex(&appended),
            &config,
            &source,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("header and carriers require"), "{error}");

        let verified = read_checkpoint(&checkpoint_path, &receipt, &config, &source);
        fs::write(&checkpoint_path, corrupt).unwrap();
        let destination = root.path.join("refused-restore");
        let error = restore_wsm6_transition_checkpoint_new(&verified, &destination)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("carrier") || error.contains("changed"),
            "{error}"
        );
        assert!(!destination.exists());
    }

    #[test]
    fn refuses_rehashed_unknown_header_keys_and_has_disjoint_magic() {
        assert_ne!(
            WSM6_TRANSITION_CHECKPOINT_MAGIC,
            crate::product_restart::INITIALIZED_PRODUCT_RESTART_MAGIC
        );
        assert_ne!(
            WSM6_TRANSITION_CHECKPOINT_MAGIC,
            crate::forecast::FORECAST_CHECKPOINT_MAGIC
        );
        assert_ne!(
            WSM6_TRANSITION_CHECKPOINT_MAGIC,
            crate::snapshot::CHECKPOINT_MAGIC
        );

        let root = TestRoot::new("header-refusal");
        let config = test_config();
        let source = test_source();
        let capture = make_capture(&root.path, &config);
        let checkpoint_path = root.path.join("state.cm1w6r");
        write_checkpoint(&capture, &checkpoint_path, &config, &source);
        let bytes = fs::read(&checkpoint_path).unwrap();
        let header_nbytes = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut header = json::parse(&bytes[16..16 + header_nbytes]).unwrap();
        match &mut header {
            JsonValue::Object(entries) => {
                entries.push(("unknown".to_string(), JsonValue::Bool(true)));
            }
            _ => unreachable!(),
        }
        let header = header.canonical_bytes();
        let mut changed = Vec::new();
        changed.extend_from_slice(&WSM6_TRANSITION_CHECKPOINT_MAGIC);
        changed.extend_from_slice(&(header.len() as u64).to_le_bytes());
        changed.extend_from_slice(&header);
        changed.extend_from_slice(&bytes[16 + header_nbytes..]);
        let changed_path = root.path.join("unknown.cm1w6r");
        fs::write(&changed_path, &changed).unwrap();
        let error = read_wsm6_transition_checkpoint(
            &changed_path,
            &sha256::digest_hex(&changed),
            &config,
            &source,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("keys") && error.contains("unknown"),
            "{error}"
        );
    }
}
