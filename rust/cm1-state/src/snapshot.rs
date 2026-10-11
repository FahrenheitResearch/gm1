//! Deterministic Rust-native checkpoint and history envelopes.
//!
//! This slice deliberately serializes only the state that the Rust
//! scientific initializer actually owns.  Its checkpoint is an exact
//! post-GETSET startup snapshot, not a forecast restart: missing moisture,
//! closure, held-tendency, boundary-history, scheduler, and component state
//! are named in the artifact and make any broader restore claim invalid.

use crate::json::{self, JsonValue};
use crate::scientific::{
    Bounds3, FieldFamily, ScientificField, ScientificState, Stagger, NX, NY, NZ, SCIENTIFIC_SCHEMA,
};
use crate::sha256;
use crate::{Result, StateError};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHECKPOINT_SCHEMA: &str = "gpuwm-cm1.checkpoint/startup-post-getset-v1";
pub const HISTORY_SCHEMA: &str = "gpuwm-cm1.history/post-getset-observational-v1";
pub const RECEIPT_SCHEMA: &str = "gpuwm-cm1.rust-snapshot-receipt/v1";
pub const AUTHORITY_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
pub const CHECKPOINT_MAGIC: [u8; 8] = *b"CM1CKP1\0";
pub const HISTORY_MAGIC: [u8; 8] = *b"CM1HST1\0";
pub const MAX_HEADER_BYTES: usize = 256 * 1024;
pub const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

const PREFIX_BYTES: usize = 16;
const MAX_ENVELOPE_BYTES: usize = PREFIX_BYTES + MAX_HEADER_BYTES + MAX_PAYLOAD_BYTES;
const PROFILE: &str = "sealed-product-120x120x40/v1";
const REPRESENTATION: &str =
    "raw IEEE-754 binary32 little-endian; fields use native Fortran i-fastest order";
pub(crate) const ATOMIC_WRITE_CONTRACT: &str =
    "create-new same-directory temporary file, write_all, sync_all, create-new hard-link publish, unlink owned temporary; target is never overwritten";

const CHECKPOINT_CAPABILITY: &str = "startup-post-getset-state-only";
const HISTORY_CAPABILITY: &str = "observational-post-getset-history-only";
const CHECKPOINT_CLAIM_LIMIT: &str = "exact Rust scientific-initializer state at the first nonrestart post-GETSET startup boundary; no forecast timestep, mid-step state, moisture carriers, TKE/SGS, derived rho/rr/rf/prs/ppx, tendencies, radiative-boundary history, component persistence, scheduler clocks, uninterrupted/resumed identity, CUDA, or product-restart claim";
const HISTORY_CLAIM_LIMIT: &str = "observational physical-domain slice of the Rust scientific initializer at model time zero; no forecast trajectory, halos, restart semantics, NetCDF compatibility, moisture, closure, precipitation, reflectivity, CUDA, or weather-render claim";

const CHECKPOINT_DEFERRED: &[&str] = &[
    "moisture carriers and semantic species mapping",
    "TKE/SGS prognostic, tendency, and coefficient state",
    "derived rho/rr/rf/prs/ppx state",
    "held and stage tendencies",
    "radiative-boundary history",
    "ArWen component persistence and precipitation accumulators",
    "forecast output/restart clocks and domain translation",
    "deterministic reduction identity",
];

const HISTORY_FIELDS: &[&str] = &[
    "pi0", "prs0", "rho0", "thv0", "th0", "qv0", "u0", "v0", "u3d", "v3d", "w3d", "pp3d", "th3d",
];

const TOP_KEYS: &[&str] = &[
    "arrays",
    "authority_sha256",
    "byte_order",
    "capability",
    "claim_limit",
    "clock",
    "deferred",
    "payload_nbytes",
    "payload_order",
    "payload_sha256",
    "profile",
    "representation",
    "schema",
    "source_state_schema",
];

const ARRAY_KEYS: &[&str] = &[
    "bounds", "meaning", "name", "nbytes", "offset", "rank", "role", "sha256", "shape", "stagger",
    "units",
];

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The two independently versioned Rust envelopes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotKind {
    Checkpoint,
    History,
}

impl SnapshotKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Checkpoint => "checkpoint",
            Self::History => "history",
        }
    }

    const fn magic(self) -> [u8; 8] {
        match self {
            Self::Checkpoint => CHECKPOINT_MAGIC,
            Self::History => HISTORY_MAGIC,
        }
    }

    const fn schema(self) -> &'static str {
        match self {
            Self::Checkpoint => CHECKPOINT_SCHEMA,
            Self::History => HISTORY_SCHEMA,
        }
    }

    const fn capability(self) -> &'static str {
        match self {
            Self::Checkpoint => CHECKPOINT_CAPABILITY,
            Self::History => HISTORY_CAPABILITY,
        }
    }

    const fn claim_limit(self) -> &'static str {
        match self {
            Self::Checkpoint => CHECKPOINT_CLAIM_LIMIT,
            Self::History => HISTORY_CLAIM_LIMIT,
        }
    }
}

/// Metadata for one strictly ordered array in an envelope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SnapshotArrayInfo {
    pub name: String,
    pub meaning: String,
    pub units: String,
    pub role: String,
    pub stagger: String,
    pub rank: usize,
    pub bounds: Vec<i32>,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub nbytes: usize,
    pub sha256: String,
}

/// One fully verified, owned restored array.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestoredArray {
    info: SnapshotArrayInfo,
    bytes: Vec<u8>,
}

impl RestoredArray {
    pub fn info(&self) -> &SnapshotArrayInfo {
        &self.info
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn word_bits(&self, index: usize) -> Option<u32> {
        let offset = index.checked_mul(4)?;
        let word: [u8; 4] = self.bytes.get(offset..offset + 4)?.try_into().ok()?;
        Some(u32::from_le_bytes(word))
    }
}

/// A fully verified startup checkpoint or observational history artifact.
#[derive(Clone, Debug)]
pub struct RestoredSnapshot {
    kind: SnapshotKind,
    arrays: Vec<RestoredArray>,
    artifact_sha256: String,
    header_sha256: String,
    payload_sha256: String,
    artifact_nbytes: usize,
    payload_nbytes: usize,
}

impl RestoredSnapshot {
    pub fn kind(&self) -> SnapshotKind {
        self.kind
    }

    pub fn arrays(&self) -> &[RestoredArray] {
        &self.arrays
    }

    pub fn array(&self, name: &str) -> Option<&RestoredArray> {
        self.arrays.iter().find(|array| array.info.name == name)
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn receipt(&self) -> ArtifactReceipt {
        ArtifactReceipt {
            kind: self.kind,
            artifact_sha256: self.artifact_sha256.clone(),
            header_sha256: self.header_sha256.clone(),
            payload_sha256: self.payload_sha256.clone(),
            artifact_nbytes: self.artifact_nbytes,
            payload_nbytes: self.payload_nbytes,
            arrays: self.arrays.len(),
        }
    }
}

/// Deterministic content receipt.  The filesystem path is intentionally absent
/// so byte-identical artifacts produce byte-identical receipt JSON anywhere.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactReceipt {
    pub kind: SnapshotKind,
    pub artifact_sha256: String,
    pub header_sha256: String,
    pub payload_sha256: String,
    pub artifact_nbytes: usize,
    pub payload_nbytes: usize,
    pub arrays: usize,
}

impl ArtifactReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("arrays", number(self.arrays)),
            ("artifact_nbytes", number(self.artifact_nbytes)),
            ("artifact_sha256", string(&self.artifact_sha256)),
            ("atomic_write_contract", string(ATOMIC_WRITE_CONTRACT)),
            ("capability", string(self.kind.capability())),
            ("header_sha256", string(&self.header_sha256)),
            ("kind", string(self.kind.as_str())),
            ("payload_nbytes", number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            ("schema", string(RECEIPT_SCHEMA)),
            ("snapshot_schema", string(self.kind.schema())),
        ]);
        String::from_utf8(value.canonical_bytes()).expect("canonical JSON is UTF-8")
    }
}

#[derive(Clone, Debug)]
struct EncodedArray {
    info: SnapshotArrayInfo,
    bytes: Vec<u8>,
}

struct ArrayDescriptor<'a> {
    name: &'a str,
    meaning: &'a str,
    units: &'a str,
    role: &'a str,
    stagger: &'a str,
    bounds: Vec<i32>,
    shape: Vec<usize>,
}

#[derive(Clone, Debug)]
struct EncodedEnvelope {
    kind: SnapshotKind,
    bytes: Vec<u8>,
    header: Vec<u8>,
    arrays: Vec<EncodedArray>,
    payload_sha256: String,
}

impl EncodedEnvelope {
    fn receipt(&self) -> ArtifactReceipt {
        ArtifactReceipt {
            kind: self.kind,
            artifact_sha256: sha256::digest_hex(&self.bytes),
            header_sha256: sha256::digest_hex(&self.header),
            payload_sha256: self.payload_sha256.clone(),
            artifact_nbytes: self.bytes.len(),
            payload_nbytes: self.arrays.iter().map(|array| array.bytes.len()).sum(),
            arrays: self.arrays.len(),
        }
    }
}

/// Atomically write a deterministic checkpoint of the currently implemented
/// post-GETSET scientific initializer.  The destination must not exist.
pub fn write_checkpoint_new(
    state: &ScientificState,
    output: impl AsRef<Path>,
) -> Result<ArtifactReceipt> {
    write_new(SnapshotKind::Checkpoint, state, output.as_ref())
}

/// Atomically write a deterministic physical-domain observational history at
/// the currently implemented post-GETSET startup boundary.
pub fn write_history_new(
    state: &ScientificState,
    output: impl AsRef<Path>,
) -> Result<ArtifactReceipt> {
    write_new(SnapshotKind::History, state, output.as_ref())
}

/// Read and fully validate the exact startup checkpoint.  Even an internally
/// rehashed mutation is refused because the sealed scientific state is
/// independently regenerated and compared array by array.
pub fn read_checkpoint(input: impl AsRef<Path>) -> Result<RestoredSnapshot> {
    read_exact(SnapshotKind::Checkpoint, input.as_ref())
}

/// Read and fully validate the exact observational history slice.
pub fn read_history(input: impl AsRef<Path>) -> Result<RestoredSnapshot> {
    read_exact(SnapshotKind::History, input.as_ref())
}

fn write_new(
    kind: SnapshotKind,
    state: &ScientificState,
    output: &Path,
) -> Result<ArtifactReceipt> {
    let envelope = encode(kind, state)?;
    atomic_write_new(output, &envelope.bytes)?;
    Ok(envelope.receipt())
}

fn encode(kind: SnapshotKind, state: &ScientificState) -> Result<EncodedEnvelope> {
    let mut arrays = match kind {
        SnapshotKind::Checkpoint => checkpoint_arrays(state),
        SnapshotKind::History => history_arrays(state)?,
    };
    let mut offset = 0usize;
    let mut payload = Vec::new();
    for array in &mut arrays {
        array.info.offset = offset;
        offset = offset
            .checked_add(array.bytes.len())
            .ok_or_else(|| invalid("snapshot payload size overflow"))?;
        if offset > MAX_PAYLOAD_BYTES {
            return Err(invalid(format!(
                "snapshot payload exceeds {MAX_PAYLOAD_BYTES} bytes"
            )));
        }
        payload.extend_from_slice(&array.bytes);
    }
    let payload_sha256 = sha256::digest_hex(&payload);
    let header_value = header_value(kind, &arrays, &payload_sha256, payload.len());
    let header = header_value.canonical_bytes();
    if header.len() > MAX_HEADER_BYTES {
        return Err(invalid(format!(
            "snapshot header exceeds {MAX_HEADER_BYTES} bytes"
        )));
    }
    let mut bytes = Vec::with_capacity(PREFIX_BYTES + header.len() + payload.len());
    bytes.extend_from_slice(&kind.magic());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&payload);
    Ok(EncodedEnvelope {
        kind,
        bytes,
        header,
        arrays,
        payload_sha256,
    })
}

fn checkpoint_arrays(state: &ScientificState) -> Vec<EncodedArray> {
    let mut arrays = Vec::with_capacity(state.coordinates().len() + state.fields().len());
    for coordinate in state.coordinates() {
        let (meaning, units) = coordinate_semantics(coordinate.name());
        arrays.push(encoded_array(
            ArrayDescriptor {
                name: coordinate.name(),
                meaning,
                units,
                role: "coordinate-complete-allocation",
                stagger: coordinate_stagger(coordinate.name()),
                bounds: vec![coordinate.lower(), coordinate.upper()],
                shape: vec![coordinate.values().len()],
            },
            coordinate.values(),
        ));
    }
    for field in state.fields() {
        arrays.push(encoded_field(field, field.bounds(), "complete-allocation"));
    }
    arrays
}

fn history_arrays(state: &ScientificState) -> Result<Vec<EncodedArray>> {
    let mut arrays = Vec::with_capacity(6 + HISTORY_FIELDS.len());
    for coordinate in state.coordinates() {
        let (lower, upper) = physical_coordinate_bounds(coordinate.name())?;
        let start = usize::try_from(lower - coordinate.lower()).map_err(|_| {
            invalid(format!(
                "coordinate {} physical lower bound",
                coordinate.name()
            ))
        })?;
        let end = usize::try_from(upper - coordinate.lower() + 1).map_err(|_| {
            invalid(format!(
                "coordinate {} physical upper bound",
                coordinate.name()
            ))
        })?;
        let values = coordinate.values().get(start..end).ok_or_else(|| {
            invalid(format!(
                "coordinate {} does not contain its physical history slice",
                coordinate.name()
            ))
        })?;
        let (meaning, units) = coordinate_semantics(coordinate.name());
        arrays.push(encoded_array(
            ArrayDescriptor {
                name: coordinate.name(),
                meaning,
                units,
                role: "coordinate-physical-domain",
                stagger: coordinate_stagger(coordinate.name()),
                bounds: vec![lower, upper],
                shape: vec![values.len()],
            },
            values,
        ));
    }
    for name in HISTORY_FIELDS {
        let field = state
            .field(name)
            .ok_or_else(|| invalid(format!("scientific state is missing history field {name}")))?;
        arrays.push(encoded_field(
            field,
            physical_field_bounds(field.stagger()),
            "observational-physical-domain",
        ));
    }
    Ok(arrays)
}

fn encoded_field(field: &ScientificField, bounds: Bounds3, scope: &str) -> EncodedArray {
    let mut values = Vec::with_capacity(bounds.words());
    for k in bounds.k_lo..=bounds.k_hi {
        for j in bounds.j_lo..=bounds.j_hi {
            for i in bounds.i_lo..=bounds.i_hi {
                values.push(
                    field
                        .value(i, j, k)
                        .expect("requested bounds are contained in the scientific field"),
                );
            }
        }
    }
    let role = format!("{}-{scope}", family_name(field.family()));
    let (meaning, units) = field_semantics(field.name());
    encoded_array(
        ArrayDescriptor {
            name: field.name(),
            meaning,
            units,
            role: &role,
            stagger: field.stagger().as_str(),
            bounds: vec![
                bounds.i_lo,
                bounds.i_hi,
                bounds.j_lo,
                bounds.j_hi,
                bounds.k_lo,
                bounds.k_hi,
            ],
            shape: bounds.shape_ijk().to_vec(),
        },
        &values,
    )
}

fn encoded_array(descriptor: ArrayDescriptor<'_>, values: &[f32]) -> EncodedArray {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let info = SnapshotArrayInfo {
        name: descriptor.name.to_owned(),
        meaning: descriptor.meaning.to_owned(),
        units: descriptor.units.to_owned(),
        role: descriptor.role.to_owned(),
        stagger: descriptor.stagger.to_owned(),
        rank: descriptor.shape.len(),
        bounds: descriptor.bounds,
        shape: descriptor.shape,
        offset: 0,
        nbytes: bytes.len(),
        sha256: sha256::digest_hex(&bytes),
    };
    EncodedArray { info, bytes }
}

fn coordinate_semantics(name: &str) -> (&'static str, &'static str) {
    match name {
        "xh" => ("x mass-point coordinate", "m"),
        "xf" => ("x face coordinate", "m"),
        "yh" => ("y mass-point coordinate", "m"),
        "yf" => ("y face coordinate", "m"),
        "zh" => ("geometric mass-level height", "m"),
        "zf" => ("geometric interface height", "m"),
        _ => ("unknown coordinate", "unknown"),
    }
}

fn field_semantics(name: &str) -> (&'static str, &'static str) {
    match name {
        "pi0" => ("base nondimensional Exner pressure", "1"),
        "prs0" => ("base pressure", "Pa"),
        "rho0" => ("base dry-air density", "kg m-3"),
        "thv0" => ("base virtual potential temperature", "K"),
        "th0" => ("base potential temperature", "K"),
        "rth0" => ("reciprocal base potential temperature", "K-1"),
        "qv0" => ("base water-vapor mixing ratio", "kg kg-1"),
        "u0" => ("base zonal wind", "m s-1"),
        "v0" => ("base meridional wind", "m s-1"),
        "ua" => ("zonal velocity A time-level carrier", "m s-1"),
        "u3d" => ("current zonal velocity", "m s-1"),
        "va" => ("meridional velocity A time-level carrier", "m s-1"),
        "v3d" => ("current meridional velocity", "m s-1"),
        "wa" => ("vertical velocity A time-level carrier", "m s-1"),
        "w3d" => ("current vertical velocity", "m s-1"),
        "ppi" => ("Exner pressure perturbation A carrier", "1"),
        "pp3d" => ("current Exner pressure perturbation", "1"),
        "tha" => ("potential-temperature perturbation A carrier", "K"),
        "th3d" => ("current potential-temperature perturbation", "K"),
        _ => ("unknown scientific field", "unknown"),
    }
}

fn physical_coordinate_bounds(name: &str) -> Result<(i32, i32)> {
    match name {
        "xh" => Ok((1, NX)),
        "xf" => Ok((1, NX + 1)),
        "yh" => Ok((1, NY)),
        "yf" => Ok((1, NY + 1)),
        "zh" => Ok((1, NZ)),
        "zf" => Ok((1, NZ + 1)),
        _ => Err(invalid(format!(
            "unknown scientific coordinate {name:?}; history cannot infer a physical slice"
        ))),
    }
}

fn coordinate_stagger(name: &str) -> &'static str {
    match name {
        "xh" => "x_mass",
        "xf" => "x_face",
        "yh" => "y_mass",
        "yf" => "y_face",
        "zh" => "z_mass_flat_reduction",
        "zf" => "z_face_flat_reduction",
        _ => "unknown",
    }
}

fn physical_field_bounds(stagger: Stagger) -> Bounds3 {
    match stagger {
        Stagger::Scalar => Bounds3::new(1, NX, 1, NY, 1, NZ),
        Stagger::UFace => Bounds3::new(1, NX + 1, 1, NY, 1, NZ),
        Stagger::VFace => Bounds3::new(1, NX, 1, NY + 1, 1, NZ),
        Stagger::WFace => Bounds3::new(1, NX, 1, NY, 1, NZ + 1),
    }
}

fn family_name(family: FieldFamily) -> &'static str {
    match family {
        FieldFamily::Base => "base",
        FieldFamily::Core => "core",
    }
}

fn header_value(
    kind: SnapshotKind,
    arrays: &[EncodedArray],
    payload_sha256: &str,
    payload_nbytes: usize,
) -> JsonValue {
    let array_values = arrays
        .iter()
        .map(|array| {
            object(vec![
                (
                    "bounds",
                    JsonValue::Array(
                        array
                            .info
                            .bounds
                            .iter()
                            .map(|value| integer(*value))
                            .collect(),
                    ),
                ),
                ("meaning", string(&array.info.meaning)),
                ("name", string(&array.info.name)),
                ("nbytes", number(array.info.nbytes)),
                ("offset", number(array.info.offset)),
                ("rank", number(array.info.rank)),
                ("role", string(&array.info.role)),
                ("sha256", string(&array.info.sha256)),
                (
                    "shape",
                    JsonValue::Array(
                        array
                            .info
                            .shape
                            .iter()
                            .map(|value| number(*value))
                            .collect(),
                    ),
                ),
                ("stagger", string(&array.info.stagger)),
                ("units", string(&array.info.units)),
            ])
        })
        .collect();
    let payload_order = arrays
        .iter()
        .map(|array| string(&array.info.name))
        .collect();
    let deferred = CHECKPOINT_DEFERRED
        .iter()
        .map(|entry| string(entry))
        .collect();
    object(vec![
        ("arrays", JsonValue::Array(array_values)),
        ("authority_sha256", string(AUTHORITY_SHA256)),
        ("byte_order", string("little-endian")),
        ("capability", string(kind.capability())),
        ("claim_limit", string(kind.claim_limit())),
        (
            "clock",
            object(vec![
                ("acoustic_substep", number(0)),
                ("boundary", string("startup-post-getset-before-first-solve")),
                ("model_time_f64_bits", string("0000000000000000")),
                ("nstep", number(0)),
                ("rk_stage", number(0)),
            ]),
        ),
        ("deferred", JsonValue::Array(deferred)),
        ("payload_nbytes", number(payload_nbytes)),
        ("payload_order", JsonValue::Array(payload_order)),
        ("payload_sha256", string(payload_sha256)),
        ("profile", string(PROFILE)),
        ("representation", string(REPRESENTATION)),
        ("schema", string(kind.schema())),
        ("source_state_schema", string(SCIENTIFIC_SCHEMA)),
    ])
}

fn read_exact(kind: SnapshotKind, input: &Path) -> Result<RestoredSnapshot> {
    let bytes = read_preflighted(input)?;
    if bytes.len() < PREFIX_BYTES {
        return Err(invalid(format!(
            "{} is too short for a {} envelope",
            input.display(),
            kind.as_str()
        )));
    }
    if bytes[..8] != kind.magic() {
        return Err(invalid(format!(
            "{} magic does not identify a {} artifact",
            input.display(),
            kind.as_str()
        )));
    }
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().expect("fixed prefix"));
    let header_len = usize::try_from(header_len)
        .map_err(|_| invalid("snapshot header length does not fit this platform"))?;
    if header_len > MAX_HEADER_BYTES {
        return Err(invalid(format!(
            "snapshot header length {header_len} exceeds {MAX_HEADER_BYTES}"
        )));
    }
    let payload_start = PREFIX_BYTES
        .checked_add(header_len)
        .ok_or_else(|| invalid("snapshot header offset overflow"))?;
    if payload_start > bytes.len() {
        return Err(invalid("snapshot header length extends past the artifact"));
    }
    let header = &bytes[PREFIX_BYTES..payload_start];
    let parsed = json::parse(header)
        .map_err(|error| invalid(format!("snapshot header JSON is invalid: {error}")))?;
    if parsed.canonical_bytes() != header {
        return Err(invalid(
            "snapshot header is not canonical sorted-key compact UTF-8 JSON",
        ));
    }
    exact_keys(&parsed, TOP_KEYS, "snapshot header")?;
    let array_headers = parsed
        .get("arrays")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("snapshot header.arrays: {error}")))?;
    for array in array_headers {
        exact_keys(array, ARRAY_KEYS, "snapshot array header")?;
    }
    fixed_string(&parsed, "schema", kind.schema(), "snapshot header")?;
    fixed_string(
        &parsed,
        "authority_sha256",
        AUTHORITY_SHA256,
        "snapshot header",
    )?;
    fixed_string(&parsed, "profile", PROFILE, "snapshot header")?;
    fixed_string(
        &parsed,
        "source_state_schema",
        SCIENTIFIC_SCHEMA,
        "snapshot header",
    )?;

    let state = ScientificState::initialize_sealed_v1();
    let expected = encode(kind, &state)?;
    if header != expected.header {
        return Err(invalid(format!(
            "{} header does not match the sealed {} contract; unknown fields, clocks, geometry, identities, hashes, or coverage are refused",
            kind.as_str(),
            kind.capability()
        )));
    }
    let payload = &bytes[payload_start..];
    let expected_payload_nbytes: usize =
        expected.arrays.iter().map(|array| array.bytes.len()).sum();
    if payload.len() != expected_payload_nbytes {
        return Err(invalid(format!(
            "{} payload has {} bytes; expected {}",
            kind.as_str(),
            payload.len(),
            expected_payload_nbytes
        )));
    }
    let payload_sha256 = sha256::digest_hex(payload);
    if payload_sha256 != expected.payload_sha256 {
        return Err(invalid(format!(
            "{} payload SHA-256 is {}; expected sealed {}",
            kind.as_str(),
            payload_sha256,
            expected.payload_sha256
        )));
    }
    let mut restored = Vec::with_capacity(expected.arrays.len());
    for array in expected.arrays {
        let start = array.info.offset;
        let end = start + array.info.nbytes;
        let actual = payload
            .get(start..end)
            .ok_or_else(|| invalid(format!("array {} extends past payload", array.info.name)))?;
        if actual != array.bytes {
            return Err(invalid(format!(
                "array {} differs from the independently regenerated sealed scientific state even if artifact hashes were updated",
                array.info.name
            )));
        }
        restored.push(RestoredArray {
            info: array.info,
            bytes: actual.to_vec(),
        });
    }
    Ok(RestoredSnapshot {
        kind,
        arrays: restored,
        artifact_sha256: sha256::digest_hex(&bytes),
        header_sha256: sha256::digest_hex(header),
        payload_sha256,
        artifact_nbytes: bytes.len(),
        payload_nbytes: payload.len(),
    })
}

fn read_preflighted(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "cannot inspect snapshot input {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.file_type().is_file() {
        return Err(invalid(format!(
            "snapshot input {} is not a regular non-symlink file",
            path.display()
        )));
    }
    let declared = usize::try_from(metadata.len())
        .map_err(|_| invalid("snapshot file length does not fit this platform"))?;
    if declared > MAX_ENVELOPE_BYTES {
        return Err(invalid(format!(
            "snapshot input has {declared} bytes; maximum is {MAX_ENVELOPE_BYTES}"
        )));
    }
    let mut file = File::open(path).map_err(|error| {
        invalid(format!(
            "cannot open snapshot input {}: {error}",
            path.display()
        ))
    })?;
    let opened_len = usize::try_from(
        file.metadata()
            .map_err(|error| invalid(format!("cannot inspect opened snapshot: {error}")))?
            .len(),
    )
    .map_err(|_| invalid("opened snapshot length does not fit this platform"))?;
    if opened_len != declared {
        return Err(invalid("snapshot size changed during metadata preflight"));
    }
    let mut bytes = Vec::with_capacity(declared);
    file.read_to_end(&mut bytes)
        .map_err(|error| invalid(format!("cannot read snapshot {}: {error}", path.display())))?;
    if bytes.len() != declared {
        return Err(invalid(format!(
            "snapshot size changed while reading: expected {declared}, read {}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

pub(crate) fn atomic_write_new(target: &Path, bytes: &[u8]) -> Result<()> {
    atomic_write_new_parts(target, &[bytes])
}

/// Publish the concatenation of `parts` under [`ATOMIC_WRITE_CONTRACT`].
///
/// The published bytes are exactly `parts` joined in order, so this is the
/// same artifact [`atomic_write_new`] would write from one buffer.  A producer
/// that already holds a multi-gibibyte payload hands it over as its own part
/// instead of copying it into a second full-size buffer first.
pub(crate) fn atomic_write_new_parts(target: &Path, parts: &[&[u8]]) -> Result<()> {
    if target.exists() {
        return Err(invalid(format!(
            "snapshot output already exists and will not be overwritten: {}",
            target.display()
        )));
    }
    let parent = target.parent().ok_or_else(|| {
        invalid(format!(
            "snapshot output has no parent directory: {}",
            target.display()
        ))
    })?;
    let parent_metadata = fs::metadata(parent).map_err(|error| {
        invalid(format!(
            "cannot inspect snapshot output directory {}: {error}",
            parent.display()
        ))
    })?;
    if !parent_metadata.is_dir() {
        return Err(invalid(format!(
            "snapshot output parent is not a directory: {}",
            parent.display()
        )));
    }
    let file_name = target
        .file_name()
        .ok_or_else(|| invalid("snapshot output must name a file"))?
        .to_string_lossy();
    let mut last_collision = None;
    for _ in 0..64 {
        let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{file_name}.cm1tmp-{}-{sequence}",
            std::process::id()
        ));
        let opened = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary);
        let mut file = match opened {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                last_collision = Some(error);
                continue;
            }
            Err(error) => {
                return Err(invalid(format!(
                    "cannot create atomic snapshot temporary {}: {error}",
                    temporary.display()
                )))
            }
        };
        let written = (|| -> std::io::Result<()> {
            let bytes = parts.iter().map(|part| part.len() as u64).sum::<u64>();
            let started = std::time::Instant::now();
            for part in parts {
                file.write_all(part)?;
            }
            crate::publish_timing::record("write", started.elapsed().as_secs_f64(), bytes);
            let started = std::time::Instant::now();
            file.sync_all()?;
            drop(file);
            crate::publish_timing::record("fsync", started.elapsed().as_secs_f64(), bytes);
            let started = std::time::Instant::now();
            fs::hard_link(&temporary, target)?;
            fs::remove_file(&temporary)?;
            crate::publish_timing::record("link", started.elapsed().as_secs_f64(), 0);
            Ok(())
        })();
        if let Err(error) = written {
            let _ = fs::remove_file(&temporary);
            return Err(invalid(format!(
                "atomic snapshot publish to {} failed without overwriting the target: {error}",
                target.display()
            )));
        }
        return Ok(());
    }
    Err(invalid(format!(
        "could not reserve an atomic snapshot temporary after 64 collisions: {}",
        last_collision
            .map(|error| error.to_string())
            .unwrap_or_else(|| "unknown collision".to_owned())
    )))
}

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) -> Result<()> {
    let mut actual: Vec<_> = value
        .as_object()
        .map_err(|error| invalid(format!("{label}: {error}")))?
        .iter()
        .map(|(key, _)| key.as_str())
        .collect();
    actual.sort_unstable();
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err(invalid(format!(
            "{label} keys are {:?}; expected exactly {:?}",
            actual, expected
        )));
    }
    Ok(())
}

fn fixed_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = value
        .get(key)
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} is {actual:?}; expected {expected:?}"
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

fn string(value: &str) -> JsonValue {
    JsonValue::String(value.to_owned())
}

fn number(value: usize) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn integer(value: i32) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    static STATE: OnceLock<ScientificState> = OnceLock::new();

    fn state() -> &'static ScientificState {
        STATE.get_or_init(ScientificState::initialize_sealed_v1)
    }

    fn temp_path(label: &str) -> PathBuf {
        let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "cm1-state-{label}-{}-{sequence}",
            std::process::id()
        ))
    }

    #[test]
    fn checkpoint_bytes_are_deterministic_and_restore_exactly() {
        let first = encode(SnapshotKind::Checkpoint, state()).unwrap();
        let second = encode(SnapshotKind::Checkpoint, state()).unwrap();
        assert_eq!(first.bytes, second.bytes);
        assert_eq!(first.arrays.len(), 25);
        let path = temp_path("checkpoint.cm1ckpt");
        atomic_write_new(&path, &first.bytes).unwrap();
        let restored = read_checkpoint(&path).unwrap();
        assert_eq!(restored.kind(), SnapshotKind::Checkpoint);
        assert_eq!(restored.arrays().len(), 25);
        assert_eq!(restored.array("tha").unwrap().word_bits(0), Some(0));
        assert_eq!(restored.receipt(), first.receipt());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn history_is_physical_domain_only_and_roundtrips() {
        let encoded = encode(SnapshotKind::History, state()).unwrap();
        assert_eq!(encoded.arrays.len(), 6 + HISTORY_FIELDS.len());
        let xh = encoded
            .arrays
            .iter()
            .find(|array| array.info.name == "xh")
            .unwrap();
        assert_eq!(xh.info.bounds, vec![1, NX]);
        assert_eq!(xh.info.shape, vec![NX as usize]);
        let u = encoded
            .arrays
            .iter()
            .find(|array| array.info.name == "u3d")
            .unwrap();
        assert_eq!(u.info.bounds, vec![1, NX + 1, 1, NY, 1, NZ]);
        assert_eq!(
            u.info.shape,
            vec![(NX + 1) as usize, NY as usize, NZ as usize]
        );
        assert!(encoded.arrays.iter().all(|array| array.info.name != "ua"));
        let path = temp_path("history.cm1hist");
        atomic_write_new(&path, &encoded.bytes).unwrap();
        let restored = read_history(&path).unwrap();
        assert_eq!(restored.receipt(), encoded.receipt());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn payload_corruption_and_wrong_kind_are_refused() {
        let mut encoded = encode(SnapshotKind::Checkpoint, state()).unwrap();
        let last = encoded.bytes.len() - 1;
        encoded.bytes[last] ^= 1;
        let path = temp_path("corrupt.cm1ckpt");
        atomic_write_new(&path, &encoded.bytes).unwrap();
        let error = read_checkpoint(&path).unwrap_err().to_string();
        assert!(error.contains("payload SHA-256"), "{error}");
        fs::remove_file(path).unwrap();

        let history = encode(SnapshotKind::History, state()).unwrap();
        let path = temp_path("wrong-kind.cm1hist");
        atomic_write_new(&path, &history.bytes).unwrap();
        let error = read_checkpoint(&path).unwrap_err().to_string();
        assert!(
            error.contains("magic does not identify a checkpoint"),
            "{error}"
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn noncanonical_or_rehashed_header_is_refused() {
        let encoded = encode(SnapshotKind::Checkpoint, state()).unwrap();
        let mut bytes = encoded.bytes.clone();
        let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let comma = bytes[PREFIX_BYTES..PREFIX_BYTES + header_len]
            .iter()
            .position(|byte| *byte == b',')
            .unwrap()
            + PREFIX_BYTES;
        bytes.insert(comma + 1, b' ');
        bytes[8..16].copy_from_slice(&((header_len + 1) as u64).to_le_bytes());
        let path = temp_path("noncanonical.cm1ckpt");
        atomic_write_new(&path, &bytes).unwrap();
        let error = read_checkpoint(&path).unwrap_err().to_string();
        assert!(error.contains("not canonical"), "{error}");
        fs::remove_file(path).unwrap();

        let mut parsed = json::parse(&encoded.header).unwrap();
        if let JsonValue::Object(entries) = &mut parsed {
            let capability = entries
                .iter_mut()
                .find(|(key, _)| key == "capability")
                .unwrap();
            capability.1 = string("full-forecast-restart");
        }
        let forged_header = parsed.canonical_bytes();
        let payload = &encoded.bytes[PREFIX_BYTES + encoded.header.len()..];
        let mut forged = Vec::new();
        forged.extend_from_slice(&CHECKPOINT_MAGIC);
        forged.extend_from_slice(&(forged_header.len() as u64).to_le_bytes());
        forged.extend_from_slice(&forged_header);
        forged.extend_from_slice(payload);
        let path = temp_path("forged-claim.cm1ckpt");
        atomic_write_new(&path, &forged).unwrap();
        let error = read_checkpoint(&path).unwrap_err().to_string();
        assert!(error.contains("does not match the sealed"), "{error}");
        fs::remove_file(path).unwrap();

        let mut payload = encoded.bytes[PREFIX_BYTES + encoded.header.len()..].to_vec();
        let last = payload.len() - 1;
        payload[last] ^= 1;
        let mut parsed = json::parse(&encoded.header).unwrap();
        if let JsonValue::Object(entries) = &mut parsed {
            entries
                .iter_mut()
                .find(|(key, _)| key == "payload_sha256")
                .unwrap()
                .1 = string(&sha256::digest_hex(&payload));
            let arrays = entries
                .iter_mut()
                .find(|(key, _)| key == "arrays")
                .unwrap()
                .1
                .as_array()
                .unwrap()
                .to_vec();
            let mut arrays = arrays;
            let last_array = arrays.last_mut().unwrap();
            if let JsonValue::Object(metadata) = last_array {
                let last_info = encoded.arrays.last().unwrap();
                let changed =
                    &payload[last_info.info.offset..last_info.info.offset + last_info.info.nbytes];
                metadata
                    .iter_mut()
                    .find(|(key, _)| key == "sha256")
                    .unwrap()
                    .1 = string(&sha256::digest_hex(changed));
            }
            entries
                .iter_mut()
                .find(|(key, _)| key == "arrays")
                .unwrap()
                .1 = JsonValue::Array(arrays);
        }
        let forged_header = parsed.canonical_bytes();
        let mut forged = Vec::new();
        forged.extend_from_slice(&CHECKPOINT_MAGIC);
        forged.extend_from_slice(&(forged_header.len() as u64).to_le_bytes());
        forged.extend_from_slice(&forged_header);
        forged.extend_from_slice(&payload);
        let path = temp_path("rehashed-payload.cm1ckpt");
        atomic_write_new(&path, &forged).unwrap();
        let error = read_checkpoint(&path).unwrap_err().to_string();
        assert!(error.contains("does not match the sealed"), "{error}");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn atomic_publish_refuses_overwrite_and_receipt_is_canonical() {
        let path = temp_path("no-overwrite.cm1ckpt");
        fs::write(&path, b"owned-existing-bytes").unwrap();
        let error = atomic_write_new(&path, b"replacement")
            .unwrap_err()
            .to_string();
        assert!(error.contains("will not be overwritten"), "{error}");
        assert_eq!(fs::read(&path).unwrap(), b"owned-existing-bytes");
        fs::remove_file(path).unwrap();

        let receipt = encode(SnapshotKind::Checkpoint, state()).unwrap().receipt();
        let json = receipt.canonical_json();
        let parsed = json::parse(json.as_bytes()).unwrap();
        assert_eq!(parsed.canonical_bytes(), json.as_bytes());
        assert!(json.contains(CHECKPOINT_CAPABILITY));
        assert!(json.contains(ATOMIC_WRITE_CONTRACT));
    }

    #[test]
    fn schema_lists_every_array_key_and_named_deferred_family() {
        let encoded = encode(SnapshotKind::Checkpoint, state()).unwrap();
        let parsed = json::parse(&encoded.header).unwrap();
        let arrays = parsed.get("arrays").unwrap().as_array().unwrap();
        for array in arrays {
            exact_keys(array, ARRAY_KEYS, "array").unwrap();
        }
        let deferred = parsed.get("deferred").unwrap().as_array().unwrap();
        assert_eq!(deferred.len(), CHECKPOINT_DEFERRED.len());
        assert!(CHECKPOINT_CLAIM_LIMIT.contains("no forecast timestep"));
        assert!(HISTORY_CLAIM_LIMIT.contains("no forecast trajectory"));
    }

    #[test]
    fn tracked_startup_receipt_is_bounded_canonical_json() {
        let bytes = include_bytes!("../evidence/startup-checkpoint-history-v1.json");
        assert!(!bytes.contains(&b'\r'));
        let parsed = json::parse(bytes).unwrap();
        let canonical = parsed.canonical_bytes();
        let canonical_parsed = json::parse(&canonical).unwrap();
        assert_eq!(canonical_parsed.canonical_bytes(), canonical);
        let text = std::str::from_utf8(bytes).unwrap();
        assert!(text.contains("b569a09e92116093605bf7c326aa3dfc5cfd9c71f3f2330ac00c9eaac802fe54"));
        assert!(text.contains("3e49a229bf2ef8db94d2dc3b54d62e58acbef07167bd03953e4c8824224350bf"));
        assert!(text.contains("not a forecast restart"));
        assert!(text.contains("no forecast trajectory"));
    }
}
