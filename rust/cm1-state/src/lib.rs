//! Rust ownership boundary for the structural zero-step container.
//!
//! The format is deliberately narrow:
//!
//! ```text
//! CM1SZS1\0 | u64-le JSON length | canonical UTF-8 JSON | field payloads
//! ```
//!
//! The ten field payloads are contiguous little-endian binary32 arrays in
//! C-order `(z, y, x)`.  This crate verifies the full envelope before exposing
//! metadata or writing a round trip.  The structural container remains a
//! sentinel-only contract; the separate [`scientific`] module implements one
//! source-anchored, sealed scientific-initialization profile.

mod child_artifact;
pub mod analytic_soundings;
pub mod anelastic_balance;
pub mod singleton_fft;
pub mod arwen_history;
pub mod base_state_wind;
pub mod cartesian_restart;
pub mod child_atmosphere;
pub mod child_auxiliary;
pub mod child_auxiliary_stream;
pub mod child_full_state;
pub mod child_post_step;
pub mod child_remap;
pub mod child_static_heights;
pub mod child_terrain;
pub mod child_view_pack;
pub mod cm1_namelist;
pub mod domain_grid_checkpoint;
pub mod domain_location;
pub mod forcing_history_export;
pub mod forecast;
pub mod geopotential;
pub mod getcape;
pub mod gm1_checkpoint;
pub mod hrrr;
pub mod hrrr_future;
pub mod idealized_ffi;
pub mod idealized_initialization;
pub mod initial_perturbation;
pub mod initialization;
mod json;
pub mod live;
pub mod live_full_state;
pub mod nws_reflectivity;
mod png;
pub mod product;
pub mod product_restart;
#[doc(hidden)]
pub mod publish_timing;
pub mod rayleigh_profiles;
pub mod reduced_history;
pub mod scientific;
mod sha256;
pub mod snapshot;
pub mod sounding_report;
pub mod storm_initialization;
pub mod tc_vortex;
pub mod terrain;
pub mod terrain_admission;
mod terrain_admission_science;
pub mod transition_restart;

use json::JsonValue;
use std::fmt;
use std::fs;
use std::path::Path;

pub const MAGIC: [u8; 8] = *b"CM1SZS1\0";
pub const ZERO_STEP_SCHEMA: &str = "gpuwm-cm1.structural-zero-step/v1";
pub const STATE_SCHEMA: &str = "gpuwm-cm1.structural-state/v1";
pub const STATE_PLAN_SCHEMA: &str = "gpuwm-cm1.structural-state-plan/v1";
pub const ARRAY_LAYOUT: &str = "<f4 C-order zyx";
pub const MAX_HEADER_BYTES: usize = 64 * 1024;
pub const MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_ENVELOPE_BYTES: usize = PREFIX_BYTES + MAX_HEADER_BYTES + MAX_PAYLOAD_BYTES;

const AUTHORITY_SHA256: &str = "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
const CLAIM_LIMIT: &str = "layout/interior-sentinel self-test only; not CM1 base-state or perturbation initialization, CUDA residency, a native oracle, or forecast output";
const PREFIX_BYTES: usize = MAGIC.len() + 8;

const HEADER_KEYS: &[&str] = &[
    "claim_limit",
    "payload_byte_order",
    "payload_order",
    "payloads",
    "plan",
    "plan_sha256",
    "schema",
    "state_schema",
    "total_payload_bytes",
];

const PLAN_KEYS: &[&str] = &[
    "authority_sha256",
    "claim_limit",
    "config_sha256",
    "deferred_state_families",
    "fields",
    "grid",
    "lane",
    "schema",
    "status",
    "total_nbytes",
];

const GRID_KEYS: &[&str] = &[
    "axis_order",
    "coordinates",
    "dimensions",
    "fortran_bounds",
    "halo",
    "spacing_m",
    "stagger_geometry",
];

const FIELD_KEYS: &[&str] = &[
    "dtype",
    "geometry",
    "initialization",
    "itemsize",
    "live_region_policy",
    "meaning",
    "name",
    "nbytes",
    "output_name",
    "persistence",
    "units",
];

const GEOMETRY_KEYS: &[&str] = &[
    "allocated_shape_zyx",
    "allocation_native_bounds",
    "confirmed_interior_native_bounds",
    "confirmed_interior_slices_zyx",
    "halo_authority",
    "stagger",
];

const FIELD_NAMES: [&str; 10] = [
    "ua", "u3d", "va", "v3d", "wa", "w3d", "ppi", "pp3d", "tha", "th3d",
];

#[derive(Clone, Copy)]
struct FieldSpec {
    name: &'static str,
    meaning: &'static str,
    units: &'static str,
    stagger: &'static str,
    live_region: &'static str,
    persistence: &'static str,
    output_name: &'static str,
}

#[derive(Clone, Copy)]
struct CoordinateSpec {
    name: &'static str,
    axis: &'static str,
    location: &'static str,
    face: bool,
}

const COORDINATE_SPECS: [CoordinateSpec; 4] = [
    CoordinateSpec {
        name: "xh",
        axis: "x",
        location: "mass",
        face: false,
    },
    CoordinateSpec {
        name: "xf",
        axis: "x",
        location: "face",
        face: true,
    },
    CoordinateSpec {
        name: "yh",
        axis: "y",
        location: "mass",
        face: false,
    },
    CoordinateSpec {
        name: "yf",
        axis: "y",
        location: "face",
        face: true,
    },
];

#[derive(Clone, Copy)]
struct DeferredSpec {
    name: &'static str,
    native_names: &'static [&'static str],
    gate: &'static str,
    reason: &'static str,
}

const DEFERRED_SPECS: [DeferredSpec; 4] = [
    DeferredSpec {
        name: "grid-metrics-and-base-state-values",
        native_names: &[
            "xh", "xf", "yh", "yf", "zh", "zf", "mh", "rmh", "mf", "rmf", "pi0", "rho0", "prs0",
            "th0", "qv0", "rr0", "rf0", "u0", "v0",
        ],
        gate: "O-GRID",
        reason: concat!(
            "allocation geometry is source-known, but coordinate rounding, sounding, ",
            "hydrostatic base values, and exact native live regions lack an executable oracle"
        ),
    },
    DeferredSpec {
        name: "packed-moisture",
        native_names: &["qa", "q3d", "qten", "qpten", "qtten", "qvten", "qcten"],
        gate: "O-ARWEN-WSM6",
        reason: concat!(
            "species order, energy carriers, and component persistence await the public ",
            "CM1-neutral ArWen seam"
        ),
    },
    DeferredSpec {
        name: "sgs-tke",
        native_names: &["tkea", "tke3d", "tketen", "kmh", "kmv", "khh", "khv"],
        gate: "O-CLOSURES",
        reason: "closure-specific bounds and live regions have no native sentinel fixture",
    },
    DeferredSpec {
        name: "tendencies-boundary-history-and-clocks",
        native_names: &[
            "uten",
            "uten1",
            "vten",
            "vten1",
            "wten",
            "wten1",
            "ppten",
            "thten",
            "thten1",
            "radbcw",
            "radbce",
            "radbcs",
            "radbcn",
            "nrk",
            "acoustic_substep",
        ],
        gate: "O-INSTRUMENT",
        reason: "exact lifetime, halo access, and restart cadence remain uninstrumented",
    },
];

const FIELD_SPECS: [FieldSpec; 10] = [
    FieldSpec {
        name: "ua",
        meaning: "zonal velocity A carrier",
        units: "m s-1",
        stagger: "x-face",
        live_region: "x-face-a-time-level",
        persistence: "restart",
        output_name: "u",
    },
    FieldSpec {
        name: "u3d",
        meaning: "current zonal velocity",
        units: "m s-1",
        stagger: "x-face",
        live_region: "x-face-current-3d",
        persistence: "stage",
        output_name: "u",
    },
    FieldSpec {
        name: "va",
        meaning: "meridional velocity A carrier",
        units: "m s-1",
        stagger: "y-face",
        live_region: "y-face-a-time-level",
        persistence: "restart",
        output_name: "v",
    },
    FieldSpec {
        name: "v3d",
        meaning: "current meridional velocity",
        units: "m s-1",
        stagger: "y-face",
        live_region: "y-face-current-3d",
        persistence: "stage",
        output_name: "v",
    },
    FieldSpec {
        name: "wa",
        meaning: "vertical velocity A carrier",
        units: "m s-1",
        stagger: "z-face",
        live_region: "z-face-a-time-level",
        persistence: "restart",
        output_name: "w",
    },
    FieldSpec {
        name: "w3d",
        meaning: "current vertical velocity",
        units: "m s-1",
        stagger: "z-face",
        live_region: "z-face-current-3d",
        persistence: "stage",
        output_name: "w",
    },
    FieldSpec {
        name: "ppi",
        meaning: "Exner perturbation A carrier",
        units: "1",
        stagger: "scalar",
        live_region: "scalar-a-time-level",
        persistence: "restart",
        output_name: "ppi",
    },
    FieldSpec {
        name: "pp3d",
        meaning: "current Exner perturbation",
        units: "1",
        stagger: "scalar",
        live_region: "scalar-current-3d",
        persistence: "stage",
        output_name: "ppi",
    },
    FieldSpec {
        name: "tha",
        meaning: "theta perturbation A carrier",
        units: "K",
        stagger: "scalar",
        live_region: "scalar-a-time-level",
        persistence: "restart",
        output_name: "thpert",
    },
    FieldSpec {
        name: "th3d",
        meaning: "current theta perturbation",
        units: "K",
        stagger: "scalar",
        live_region: "scalar-current-3d",
        persistence: "stage",
        output_name: "thpert",
    },
];

#[derive(Debug)]
pub struct StateError {
    message: String,
}

impl StateError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for StateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for StateError {}

impl From<std::io::Error> for StateError {
    fn from(error: std::io::Error) -> Self {
        Self::invalid(format!("I/O error: {error}"))
    }
}

pub type Result<T> = std::result::Result<T, StateError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PayloadInfo {
    pub name: String,
    pub nbytes: u64,
    pub offset: u64,
    pub sha256: String,
}

#[derive(Clone, Debug)]
struct Payload {
    info: PayloadInfo,
    bytes: Vec<u8>,
}

/// Fully verified structural state container.
#[derive(Clone, Debug)]
pub struct Container {
    header: JsonValue,
    canonical_header: Vec<u8>,
    payloads: Vec<Payload>,
    artifact_sha256: String,
}

impl Container {
    /// Parse and validate every v1 envelope, schema, geometry, and payload invariant.
    pub fn parse(blob: &[u8]) -> Result<Self> {
        if blob.len() > MAX_ENVELOPE_BYTES {
            return Err(StateError::invalid(format!(
                "structural zero-step envelope exceeds the {MAX_ENVELOPE_BYTES}-byte v1 size limit"
            )));
        }
        if blob.len() < PREFIX_BYTES {
            return Err(StateError::invalid(
                "structural zero-step envelope is truncated",
            ));
        }
        if blob[..MAGIC.len()] != MAGIC {
            return Err(StateError::invalid(
                "structural zero-step magic/version mismatch",
            ));
        }
        let header_length = u64::from_le_bytes(
            blob[MAGIC.len()..PREFIX_BYTES]
                .try_into()
                .expect("prefix length checked"),
        );
        let header_length = usize::try_from(header_length)
            .map_err(|_| StateError::invalid("JSON header length exceeds this platform"))?;
        if header_length > MAX_HEADER_BYTES {
            return Err(StateError::invalid(format!(
                "JSON header length {header_length} exceeds the {MAX_HEADER_BYTES}-byte v1 safety cap"
            )));
        }
        let header_end = PREFIX_BYTES
            .checked_add(header_length)
            .ok_or_else(|| StateError::invalid("JSON header offset overflow"))?;
        if header_end > blob.len() {
            return Err(StateError::invalid(
                "structural zero-step JSON header is truncated",
            ));
        }

        let raw_header = &blob[PREFIX_BYTES..header_end];
        let header = json::parse(raw_header)
            .map_err(|error| StateError::invalid(format!("invalid JSON header: {error}")))?;
        let canonical_header = header.canonical_bytes();
        if raw_header != canonical_header {
            return Err(StateError::invalid(
                "structural zero-step JSON header is not canonical Python sorted-key compact JSON",
            ));
        }
        let payload_region = &blob[header_end..];
        let infos = validate_header(&header, payload_region)?;
        let payloads = infos
            .into_iter()
            .map(|info| {
                let start = info.offset as usize;
                let end = start + info.nbytes as usize;
                Payload {
                    info,
                    bytes: payload_region[start..end].to_vec(),
                }
            })
            .collect();

        Ok(Self {
            header,
            canonical_header,
            payloads,
            artifact_sha256: sha256::digest_hex(blob),
        })
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).map_err(|error| {
            StateError::invalid(format!("could not inspect {}: {error}", path.display()))
        })?;
        if !metadata.is_file() {
            return Err(StateError::invalid(format!(
                "structural zero-step input is not a regular file: {}",
                path.display()
            )));
        }
        if metadata.len() > MAX_ENVELOPE_BYTES as u64 {
            return Err(StateError::invalid(format!(
                "structural zero-step envelope {} is {} bytes and exceeds the {MAX_ENVELOPE_BYTES}-byte v1 size limit",
                path.display(),
                metadata.len()
            )));
        }
        let bytes = fs::read(path).map_err(|error| {
            StateError::invalid(format!("could not read {}: {error}", path.display()))
        })?;
        Self::parse(&bytes)
    }

    /// Reconstruct the canonical envelope.  For parsed inputs this is byte-exact.
    pub fn encode(&self) -> Vec<u8> {
        let payload_bytes: usize = self
            .payloads
            .iter()
            .map(|payload| payload.bytes.len())
            .sum();
        let mut output =
            Vec::with_capacity(PREFIX_BYTES + self.canonical_header.len() + payload_bytes);
        output.extend_from_slice(&MAGIC);
        output.extend_from_slice(&(self.canonical_header.len() as u64).to_le_bytes());
        output.extend_from_slice(&self.canonical_header);
        for payload in &self.payloads {
            output.extend_from_slice(&payload.bytes);
        }
        output
    }

    /// Write a verified, reconstructed envelope and verify the bytes read back.
    pub fn write_roundtrip(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let encoded = self.encode();
        if sha256::digest_hex(&encoded) != self.artifact_sha256 {
            return Err(StateError::invalid(
                "internal round-trip changed the artifact SHA-256",
            ));
        }
        fs::write(path, &encoded).map_err(|error| {
            StateError::invalid(format!("could not write {}: {error}", path.display()))
        })?;
        let written = fs::read(path).map_err(|error| {
            StateError::invalid(format!(
                "could not read round-trip output {}: {error}",
                path.display()
            ))
        })?;
        if written != encoded {
            return Err(StateError::invalid(format!(
                "round-trip output {} is not byte-exact",
                path.display()
            )));
        }
        let reparsed = Self::parse(&written)?;
        if reparsed.artifact_sha256 != self.artifact_sha256 {
            return Err(StateError::invalid(
                "round-trip verification changed the artifact identity",
            ));
        }
        Ok(())
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn payloads(&self) -> impl ExactSizeIterator<Item = &PayloadInfo> {
        self.payloads.iter().map(|payload| &payload.info)
    }

    /// Borrow one verified contiguous `<f4` C-order `(z,y,x)` field payload.
    pub fn payload_bytes(&self, name: &str) -> Option<&[u8]> {
        self.payloads
            .iter()
            .find(|payload| payload.info.name == name)
            .map(|payload| payload.bytes.as_slice())
    }

    pub fn total_payload_bytes(&self) -> u64 {
        self.payloads
            .iter()
            .map(|payload| payload.info.nbytes)
            .sum()
    }

    pub fn canonical_header(&self) -> &[u8] {
        &self.canonical_header
    }

    pub fn inspect_json(&self) -> String {
        let mut output = format!(
            "{{\"array_layout\":\"{ARRAY_LAYOUT}\",\"artifact_sha256\":\"{}\",\"payload_count\":{},\"payloads\":[",
            self.artifact_sha256,
            self.payloads.len()
        );
        for (index, payload) in self.payloads.iter().enumerate() {
            if index != 0 {
                output.push(',');
            }
            output.push_str(&format!(
                "{{\"name\":\"{}\",\"nbytes\":{},\"offset\":{},\"sha256\":\"{}\"}}",
                payload.info.name, payload.info.nbytes, payload.info.offset, payload.info.sha256
            ));
        }
        output.push_str(&format!(
            "],\"schema\":\"{ZERO_STEP_SCHEMA}\",\"state_schema\":\"{STATE_SCHEMA}\",\"total_payload_bytes\":{}}}",
            self.total_payload_bytes()
        ));
        output
    }

    pub fn verify_internal_roundtrip(&self) -> Result<()> {
        let encoded = self.encode();
        if sha256::digest_hex(&encoded) != self.artifact_sha256 {
            return Err(StateError::invalid(
                "canonical reconstruction is not byte-exact",
            ));
        }
        Self::parse(&encoded)?;
        Ok(())
    }

    pub fn schema(&self) -> &str {
        // The constructor validates this fixed value; keep the header retained
        // so future additive inspection fields do not need a second parser.
        self.header
            .get("schema")
            .expect("validated schema key")
            .as_str()
            .expect("validated schema string")
    }
}

fn validate_header(header: &JsonValue, payload_region: &[u8]) -> Result<Vec<PayloadInfo>> {
    exact_keys(header, HEADER_KEYS, "top-level header")?;
    fixed_string(header, "schema", ZERO_STEP_SCHEMA, "top-level header")?;
    fixed_string(header, "state_schema", STATE_SCHEMA, "top-level header")?;
    fixed_string(header, "payload_byte_order", "little", "top-level header")?;
    fixed_string(
        header,
        "payload_order",
        "plan-field-order",
        "top-level header",
    )?;
    fixed_string(header, "claim_limit", CLAIM_LIMIT, "top-level header")?;

    let plan = required(header, "plan", "top-level header")?;
    let field_contracts = validate_plan(plan)?;
    let plan_sha = required_string(header, "plan_sha256", "top-level header")?;
    validate_sha256(plan_sha, "plan_sha256")?;
    let actual_plan_sha = sha256::digest_hex(&plan.canonical_bytes());
    if plan_sha != actual_plan_sha {
        return Err(StateError::invalid(format!(
            "plan_sha256 mismatch: header has {plan_sha}, computed {actual_plan_sha}"
        )));
    }

    let total_payload_bytes = required_u64(header, "total_payload_bytes", "top-level header")?;
    if total_payload_bytes != payload_region.len() as u64 {
        return Err(StateError::invalid(format!(
            "total_payload_bytes is {total_payload_bytes}, but the envelope contains {} bytes",
            payload_region.len()
        )));
    }

    let manifest = required(header, "payloads", "top-level header")?
        .as_array()
        .map_err(StateError::invalid)?;
    if manifest.len() != FIELD_NAMES.len() {
        return Err(StateError::invalid(format!(
            "payload manifest has {} rows; v1 requires {}",
            manifest.len(),
            FIELD_NAMES.len()
        )));
    }

    let mut expected_offset = 0u64;
    let mut infos = Vec::with_capacity(FIELD_NAMES.len());
    for (index, row) in manifest.iter().enumerate() {
        exact_keys(row, &["name", "nbytes", "offset", "sha256"], "payload row")?;
        let name = required_string(row, "name", "payload row")?;
        if name != FIELD_NAMES[index] {
            return Err(StateError::invalid(format!(
                "payload row {index} is {name:?}; fixed v1 order requires {:?}",
                FIELD_NAMES[index]
            )));
        }
        let nbytes = required_u64(row, "nbytes", "payload row")?;
        if nbytes != field_contracts[index].nbytes {
            return Err(StateError::invalid(format!(
                "payload {name} has {nbytes} bytes; plan field declares {}",
                field_contracts[index].nbytes
            )));
        }
        let offset = required_u64(row, "offset", "payload row")?;
        if offset != expected_offset {
            return Err(StateError::invalid(format!(
                "payload {name} offset is {offset}; contiguous v1 offset is {expected_offset}"
            )));
        }
        let end = offset
            .checked_add(nbytes)
            .ok_or_else(|| StateError::invalid(format!("payload {name} range overflows u64")))?;
        if end > total_payload_bytes {
            return Err(StateError::invalid(format!(
                "payload {name} extends beyond the payload region"
            )));
        }
        let digest = required_string(row, "sha256", "payload row")?;
        validate_sha256(digest, &format!("payload {name} sha256"))?;
        let start_usize = usize::try_from(offset)
            .map_err(|_| StateError::invalid(format!("payload {name} offset is too large")))?;
        let end_usize = usize::try_from(end)
            .map_err(|_| StateError::invalid(format!("payload {name} end is too large")))?;
        let actual_digest = sha256::digest_hex(&payload_region[start_usize..end_usize]);
        if digest != actual_digest {
            return Err(StateError::invalid(format!(
                "payload hash mismatch for {name}: header has {digest}, computed {actual_digest}"
            )));
        }
        validate_sentinel_payload(
            name,
            &payload_region[start_usize..end_usize],
            field_contracts[index],
        )?;
        infos.push(PayloadInfo {
            name: name.to_string(),
            nbytes,
            offset,
            sha256: digest.to_string(),
        });
        expected_offset = end;
    }
    if expected_offset != total_payload_bytes {
        return Err(StateError::invalid(format!(
            "payload manifest accounts for {expected_offset} bytes, not {total_payload_bytes}"
        )));
    }
    Ok(infos)
}

#[derive(Clone, Copy)]
struct GridContract {
    nx: i64,
    ny: i64,
    nz: i64,
    hx: i64,
    hy: i64,
    hz: i64,
}

#[derive(Clone, Copy, Debug)]
struct FieldContract {
    nbytes: u64,
    shape_zyx: [usize; 3],
    slices_zyx: [(usize, usize); 3],
}

fn validate_plan(plan: &JsonValue) -> Result<Vec<FieldContract>> {
    exact_keys(plan, PLAN_KEYS, "state plan")?;
    fixed_string(plan, "schema", STATE_PLAN_SCHEMA, "state plan")?;
    fixed_string(
        plan,
        "status",
        "structural-allocation-contract-only",
        "state plan",
    )?;
    fixed_string(plan, "claim_limit", CLAIM_LIMIT, "state plan")?;
    fixed_string(plan, "authority_sha256", AUTHORITY_SHA256, "state plan")?;
    let config_hash = required_string(plan, "config_sha256", "state plan")?;
    validate_sha256(config_hash, "config_sha256")?;
    let lane = required_string(plan, "lane", "state plan")?;
    if !matches!(lane, "dry-dycore" | "product" | "native-context") {
        return Err(StateError::invalid(format!(
            "state-plan lane {lane:?} is not sealed"
        )));
    }
    validate_deferred_state_families(required(plan, "deferred_state_families", "state plan")?)?;

    let grid = validate_grid(required(plan, "grid", "state plan")?, lane)?;
    let fields = required(plan, "fields", "state plan")?
        .as_array()
        .map_err(|_| StateError::invalid("state-plan fields must be an array"))?;
    if fields.len() != FIELD_SPECS.len() {
        return Err(StateError::invalid(format!(
            "state plan has {} fields; v1 requires {}",
            fields.len(),
            FIELD_SPECS.len()
        )));
    }
    let mut field_contracts = Vec::with_capacity(FIELD_SPECS.len());
    let mut total = 0u64;
    for (field, spec) in fields.iter().zip(FIELD_SPECS) {
        let contract = validate_field(field, spec, grid)?;
        total = total
            .checked_add(contract.nbytes)
            .ok_or_else(|| StateError::invalid("state-plan byte count overflows u64"))?;
        field_contracts.push(contract);
    }
    let declared_total = required_u64(plan, "total_nbytes", "state plan")?;
    if declared_total != total {
        return Err(StateError::invalid(format!(
            "state-plan total_nbytes is {declared_total}; field sum is {total}"
        )));
    }
    Ok(field_contracts)
}

fn validate_grid(grid: &JsonValue, lane: &str) -> Result<GridContract> {
    exact_keys(grid, GRID_KEYS, "grid")?;
    fixed_string(grid, "axis_order", "zyx", "grid")?;
    fixed_string(grid, "fortran_bounds", "inclusive", "grid")?;
    let dimensions = required(grid, "dimensions", "grid")?;
    exact_keys(dimensions, &["nx", "ny", "nz"], "grid dimensions")?;
    let nx = positive_i64(dimensions, "nx", "grid dimensions")?;
    let ny = positive_i64(dimensions, "ny", "grid dimensions")?;
    let nz = positive_i64(dimensions, "nz", "grid dimensions")?;
    let (expected_dimensions, expected_spacing) = match lane {
        "dry-dycore" => ((16, 16, 16), [1000.0, 1000.0, 1000.0]),
        "product" | "native-context" => ((120, 120, 40), [1000.0, 1000.0, 500.0]),
        _ => return Err(StateError::invalid("state-plan lane is not sealed")),
    };
    if (nx, ny, nz) != expected_dimensions {
        return Err(StateError::invalid(format!(
            "grid dimensions {nx}x{ny}x{nz} differ from the {lane} v1 profile {}x{}x{}",
            expected_dimensions.0, expected_dimensions.1, expected_dimensions.2
        )));
    }
    let halo = required(grid, "halo", "grid")?;
    exact_keys(halo, &["x", "y", "z"], "grid halo")?;
    let hx = required_i64(halo, "x", "grid halo")?;
    let hy = required_i64(halo, "y", "grid halo")?;
    let hz = required_i64(halo, "z", "grid halo")?;
    if (hx, hy, hz) != (3, 3, 1) {
        return Err(StateError::invalid(format!(
            "v1 halo is x=3,y=3,z=1; header has x={hx},y={hy},z={hz}"
        )));
    }
    let spacing = required(grid, "spacing_m", "grid")?
        .as_array()
        .map_err(|_| StateError::invalid("spacing_m must be an array"))?;
    if spacing.len() != 3 {
        return Err(StateError::invalid(
            "spacing_m must be three finite positive JSON numbers",
        ));
    }
    let spacing_values = [
        spacing[0].as_f64().map_err(StateError::invalid)?,
        spacing[1].as_f64().map_err(StateError::invalid)?,
        spacing[2].as_f64().map_err(StateError::invalid)?,
    ];
    if spacing_values
        .iter()
        .any(|number| !number.is_finite() || *number <= 0.0)
    {
        return Err(StateError::invalid(
            "spacing_m must be three finite positive JSON numbers",
        ));
    }
    if spacing_values != expected_spacing {
        return Err(StateError::invalid(format!(
            "grid spacing {spacing_values:?} differs from the {lane} v1 profile {expected_spacing:?}"
        )));
    }
    let contract = GridContract {
        nx,
        ny,
        nz,
        hx,
        hy,
        hz,
    };
    validate_coordinates(required(grid, "coordinates", "grid")?, contract)?;
    validate_stagger_geometry(required(grid, "stagger_geometry", "grid")?, contract)?;
    Ok(contract)
}

fn validate_coordinates(value: &JsonValue, grid: GridContract) -> Result<()> {
    let rows = value
        .as_array()
        .map_err(|_| StateError::invalid("grid coordinates must be an array"))?;
    if rows.len() != COORDINATE_SPECS.len() {
        return Err(StateError::invalid(
            "v1 grid coordinates must contain xh, xf, yh, and yf metadata",
        ));
    }
    for (row, spec) in rows.iter().zip(COORDINATE_SPECS) {
        let label = format!("coordinate {}", spec.name);
        exact_keys(
            row,
            &[
                "axis",
                "count",
                "location",
                "name",
                "native_bounds",
                "status",
                "units",
            ],
            &label,
        )?;
        fixed_string(row, "name", spec.name, &label)?;
        fixed_string(row, "axis", spec.axis, &label)?;
        fixed_string(row, "location", spec.location, &label)?;
        fixed_string(row, "units", "m", &label)?;
        fixed_string(
            row,
            "status",
            "allocation-only-values-deferred-to-O-GRID",
            &label,
        )?;
        let (width, halo) = if spec.axis == "x" {
            (grid.nx, grid.hx)
        } else {
            (grid.ny, grid.hy)
        };
        let expected = Bounds {
            lower: 1 - halo,
            upper: width + halo + i64::from(spec.face),
        };
        fixed_u64(row, "count", expected.count()?, &label)?;
        let actual = bounds(required(row, "native_bounds", &label)?, &label)?;
        if actual != expected {
            return Err(StateError::invalid(format!(
                "{label} native bounds {actual:?} differ from {expected:?}"
            )));
        }
    }
    Ok(())
}

fn validate_stagger_geometry(value: &JsonValue, grid: GridContract) -> Result<()> {
    exact_keys(
        value,
        &["scalar", "x-face", "y-face", "z-face"],
        "grid stagger_geometry",
    )?;
    for stagger in ["scalar", "x-face", "y-face", "z-face"] {
        validate_geometry(
            required(value, stagger, "grid stagger_geometry")?,
            stagger,
            &format!("grid stagger_geometry.{stagger}"),
            grid,
        )?;
    }
    Ok(())
}

fn validate_deferred_state_families(value: &JsonValue) -> Result<()> {
    let rows = value
        .as_array()
        .map_err(|_| StateError::invalid("deferred_state_families must be an array"))?;
    if rows.len() != DEFERRED_SPECS.len() {
        return Err(StateError::invalid(format!(
            "deferred_state_families has {} rows; v1 requires {}",
            rows.len(),
            DEFERRED_SPECS.len()
        )));
    }
    for (row, spec) in rows.iter().zip(DEFERRED_SPECS) {
        let label = format!("deferred state family {}", spec.name);
        exact_keys(row, &["gate", "name", "native_names", "reason"], &label)?;
        fixed_string(row, "name", spec.name, &label)?;
        fixed_string(row, "gate", spec.gate, &label)?;
        fixed_string(row, "reason", spec.reason, &label)?;
        let native_names = required(row, "native_names", &label)?
            .as_array()
            .map_err(|_| StateError::invalid(format!("{label}.native_names must be an array")))?;
        if native_names.len() != spec.native_names.len() {
            return Err(StateError::invalid(format!(
                "{label}.native_names has {} entries; v1 requires {}",
                native_names.len(),
                spec.native_names.len()
            )));
        }
        for (index, (actual, expected)) in native_names.iter().zip(spec.native_names).enumerate() {
            let actual = actual.as_str().map_err(|_| {
                StateError::invalid(format!("{label}.native_names[{index}] must be a string"))
            })?;
            if actual != *expected {
                return Err(StateError::invalid(format!(
                    "{label}.native_names[{index}] is {actual:?}; v1 requires {expected:?}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_field(field: &JsonValue, spec: FieldSpec, grid: GridContract) -> Result<FieldContract> {
    exact_keys(field, FIELD_KEYS, &format!("field {}", spec.name))?;
    fixed_string(field, "name", spec.name, &format!("field {}", spec.name))?;
    fixed_string(field, "dtype", "<f4", &format!("field {}", spec.name))?;
    fixed_u64(field, "itemsize", 4, &format!("field {}", spec.name))?;
    fixed_string(
        field,
        "live_region_policy",
        spec.live_region,
        &format!("field {}", spec.name),
    )?;
    fixed_string(
        field,
        "meaning",
        spec.meaning,
        &format!("field {}", spec.name),
    )?;
    fixed_string(field, "units", spec.units, &format!("field {}", spec.name))?;
    fixed_string(
        field,
        "persistence",
        spec.persistence,
        &format!("field {}", spec.name),
    )?;
    fixed_string(
        field,
        "output_name",
        spec.output_name,
        &format!("field {}", spec.name),
    )?;
    let initialization = required(field, "initialization", &format!("field {}", spec.name))?;
    exact_keys(
        initialization,
        &[
            "confirmed_interior_word",
            "outside_confirmed_interior_word",
            "status",
        ],
        &format!("field {} initialization", spec.name),
    )?;
    fixed_string(
        initialization,
        "confirmed_interior_word",
        "0x00000000",
        &format!("field {} initialization", spec.name),
    )?;
    fixed_string(
        initialization,
        "outside_confirmed_interior_word",
        "0x7fc00000",
        &format!("field {} initialization", spec.name),
    )?;
    fixed_string(
        initialization,
        "status",
        "structural-sentinel-only",
        &format!("field {} initialization", spec.name),
    )?;

    let geometry = required(field, "geometry", &format!("field {}", spec.name))?;
    let geometry_contract = validate_geometry(
        geometry,
        spec.stagger,
        &format!("field {} geometry", spec.name),
        grid,
    )?;
    let declared_nbytes = required_u64(field, "nbytes", &format!("field {}", spec.name))?;
    if declared_nbytes != geometry_contract.nbytes {
        return Err(StateError::invalid(format!(
            "field {} nbytes is {declared_nbytes}; <f4 zyx geometry requires {}",
            spec.name, geometry_contract.nbytes
        )));
    }
    Ok(geometry_contract)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Bounds {
    lower: i64,
    upper: i64,
}

impl Bounds {
    fn count(self) -> Result<u64> {
        if self.upper < self.lower {
            return Err(StateError::invalid(format!(
                "native bounds {}..{} are reversed",
                self.lower, self.upper
            )));
        }
        u64::try_from(self.upper - self.lower + 1)
            .map_err(|_| StateError::invalid("native-bounds count is out of range"))
    }
}

fn validate_geometry(
    geometry: &JsonValue,
    stagger: &str,
    label: &str,
    grid: GridContract,
) -> Result<FieldContract> {
    exact_keys(geometry, GEOMETRY_KEYS, label)?;
    fixed_string(
        geometry,
        "halo_authority",
        "unconfirmed-outside-physical-interior",
        label,
    )?;
    fixed_string(geometry, "stagger", stagger, label)?;

    let x_extra = i64::from(stagger == "x-face");
    let y_extra = i64::from(stagger == "y-face");
    let z_extra = i64::from(stagger == "z-face");
    let expected_confirmed = [
        Bounds {
            lower: 1,
            upper: grid.nx + x_extra,
        },
        Bounds {
            lower: 1,
            upper: grid.ny + y_extra,
        },
        Bounds {
            lower: 1,
            upper: grid.nz + z_extra,
        },
    ];
    let expected_allocated = [
        Bounds {
            lower: 1 - grid.hx,
            upper: grid.nx + x_extra + grid.hx,
        },
        Bounds {
            lower: 1 - grid.hy,
            upper: grid.ny + y_extra + grid.hy,
        },
        Bounds {
            lower: 1 - grid.hz,
            upper: grid.nz + z_extra + grid.hz,
        },
    ];

    let allocated = bounds_xyz(
        required(geometry, "allocation_native_bounds", label)?,
        &format!("{label} allocation_native_bounds"),
    )?;
    let confirmed = bounds_xyz(
        required(geometry, "confirmed_interior_native_bounds", label)?,
        &format!("{label} confirmed_interior_native_bounds"),
    )?;
    if allocated != expected_allocated {
        return Err(StateError::invalid(format!(
            "{label} allocation bounds differ from the fixed halo/stagger geometry"
        )));
    }
    if confirmed != expected_confirmed {
        return Err(StateError::invalid(format!(
            "{label} confirmed bounds differ from its physical staggered interior"
        )));
    }

    let shape_values = required(geometry, "allocated_shape_zyx", label)?
        .as_array()
        .map_err(|_| StateError::invalid(format!("{label} shape must be an array")))?;
    if shape_values.len() != 3 {
        return Err(StateError::invalid(format!(
            "{label} allocated_shape_zyx must have three entries"
        )));
    }
    let shape = [
        shape_values[0].as_u64().map_err(StateError::invalid)?,
        shape_values[1].as_u64().map_err(StateError::invalid)?,
        shape_values[2].as_u64().map_err(StateError::invalid)?,
    ];
    let expected_shape = [
        allocated[2].count()?,
        allocated[1].count()?,
        allocated[0].count()?,
    ];
    if shape != expected_shape {
        return Err(StateError::invalid(format!(
            "{label} allocated_shape_zyx {shape:?} disagrees with z,y,x bounds {expected_shape:?}"
        )));
    }

    let slices = required(geometry, "confirmed_interior_slices_zyx", label)?
        .as_array()
        .map_err(|_| StateError::invalid(format!("{label} slices must be an array")))?;
    if slices.len() != 3 {
        return Err(StateError::invalid(format!(
            "{label} confirmed_interior_slices_zyx must have three entries"
        )));
    }
    let mut slices_zyx = [(0usize, 0usize); 3];
    for (position, axis) in [2usize, 1, 0].into_iter().enumerate() {
        let slice = &slices[position];
        exact_keys(slice, &["start", "stop"], &format!("{label} slice"))?;
        let start = required_u64(slice, "start", &format!("{label} slice"))?;
        let stop = required_u64(slice, "stop", &format!("{label} slice"))?;
        let expected_start = u64::try_from(confirmed[axis].lower - allocated[axis].lower)
            .map_err(|_| StateError::invalid("confirmed slice start is out of range"))?;
        let expected_stop = expected_start + confirmed[axis].count()?;
        if (start, stop) != (expected_start, expected_stop) || stop > shape[position] {
            return Err(StateError::invalid(format!(
                "{label} zyx slice {position} is {start}:{stop}; expected {expected_start}:{expected_stop}"
            )));
        }
        slices_zyx[position] = (
            usize::try_from(start)
                .map_err(|_| StateError::invalid(format!("{label} slice start is too large")))?,
            usize::try_from(stop)
                .map_err(|_| StateError::invalid(format!("{label} slice stop is too large")))?,
        );
    }

    let nbytes = shape
        .into_iter()
        .try_fold(4u64, |product, dimension| product.checked_mul(dimension))
        .ok_or_else(|| StateError::invalid(format!("{label} byte count overflows u64")))?;
    let shape_zyx = [
        usize::try_from(shape[0])
            .map_err(|_| StateError::invalid(format!("{label} z shape is too large")))?,
        usize::try_from(shape[1])
            .map_err(|_| StateError::invalid(format!("{label} y shape is too large")))?,
        usize::try_from(shape[2])
            .map_err(|_| StateError::invalid(format!("{label} x shape is too large")))?,
    ];
    Ok(FieldContract {
        nbytes,
        shape_zyx,
        slices_zyx,
    })
}

fn validate_sentinel_payload(name: &str, bytes: &[u8], contract: FieldContract) -> Result<()> {
    const INTERIOR_WORD: [u8; 4] = 0x0000_0000u32.to_le_bytes();
    const HALO_WORD: [u8; 4] = 0x7fc0_0000u32.to_le_bytes();
    if bytes.len() as u64 != contract.nbytes {
        return Err(StateError::invalid(format!(
            "payload {name} length changed before sentinel validation"
        )));
    }
    let [z_count, y_count, x_count] = contract.shape_zyx;
    let [(z_start, z_stop), (y_start, y_stop), (x_start, x_stop)] = contract.slices_zyx;
    for z in 0..z_count {
        let z_interior = (z_start..z_stop).contains(&z);
        for y in 0..y_count {
            let yz_interior = z_interior && (y_start..y_stop).contains(&y);
            for x in 0..x_count {
                let expected = if yz_interior && (x_start..x_stop).contains(&x) {
                    &INTERIOR_WORD
                } else {
                    &HALO_WORD
                };
                let word_index = (z * y_count + y)
                    .checked_mul(x_count)
                    .and_then(|row| row.checked_add(x))
                    .and_then(|word| word.checked_mul(4))
                    .ok_or_else(|| {
                        StateError::invalid(format!("payload {name} word offset overflows usize"))
                    })?;
                if bytes[word_index..word_index + 4] != *expected {
                    return Err(StateError::invalid(format!(
                        "payload {name} violates the structural sentinel policy at allocated z/y/x {z}/{y}/{x}"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn bounds_xyz(value: &JsonValue, label: &str) -> Result<[Bounds; 3]> {
    exact_keys(value, &["x", "y", "z"], label)?;
    Ok([
        bounds(required(value, "x", label)?, &format!("{label}.x"))?,
        bounds(required(value, "y", label)?, &format!("{label}.y"))?,
        bounds(required(value, "z", label)?, &format!("{label}.z"))?,
    ])
}

fn bounds(value: &JsonValue, label: &str) -> Result<Bounds> {
    exact_keys(value, &["lower", "upper"], label)?;
    let result = Bounds {
        lower: required_i64(value, "lower", label)?,
        upper: required_i64(value, "upper", label)?,
    };
    result.count()?;
    Ok(result)
}

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) -> Result<()> {
    let object = value
        .as_object()
        .map_err(|_| StateError::invalid(format!("{label} must be an object")))?;
    let mut actual: Vec<&str> = object.iter().map(|(key, _)| key.as_str()).collect();
    actual.sort_unstable();
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err(StateError::invalid(format!(
            "{label} keys differ: actual={actual:?}, expected={expected:?}"
        )));
    }
    Ok(())
}

fn required<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a JsonValue> {
    value
        .get(key)
        .map_err(|_| StateError::invalid(format!("{label} is missing key {key:?}")))
}

fn required_string<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    required(value, key, label)?
        .as_str()
        .map_err(|_| StateError::invalid(format!("{label}.{key} must be a string")))
}

fn required_u64(value: &JsonValue, key: &str, label: &str) -> Result<u64> {
    required(value, key, label)?
        .as_u64()
        .map_err(|_| StateError::invalid(format!("{label}.{key} must be an unsigned integer")))
}

fn required_i64(value: &JsonValue, key: &str, label: &str) -> Result<i64> {
    required(value, key, label)?
        .as_i64()
        .map_err(|_| StateError::invalid(format!("{label}.{key} must be a signed integer")))
}

fn positive_i64(value: &JsonValue, key: &str, label: &str) -> Result<i64> {
    let result = required_i64(value, key, label)?;
    if result <= 0 {
        return Err(StateError::invalid(format!(
            "{label}.{key} must be positive"
        )));
    }
    Ok(result)
}

fn fixed_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = required_string(value, key, label)?;
    if actual != expected {
        return Err(StateError::invalid(format!(
            "{label}.{key} is {actual:?}; v1 requires {expected:?}"
        )));
    }
    Ok(())
}

fn fixed_u64(value: &JsonValue, key: &str, expected: u64, label: &str) -> Result<()> {
    let actual = required_u64(value, key, label)?;
    if actual != expected {
        return Err(StateError::invalid(format!(
            "{label}.{key} is {actual}; v1 requires {expected}"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(StateError::invalid(format!(
            "{label} must be a lower-case 64-digit SHA-256 hex digest"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::ffi::OsString;
    use std::fs::File;
    use std::process::Command;

    fn string(value: &str) -> JsonValue {
        JsonValue::String(value.to_string())
    }

    fn number(value: u64) -> JsonValue {
        JsonValue::Number(value.to_string())
    }

    fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
        JsonValue::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        )
    }

    fn bounds_value(lower: i64, upper: i64) -> JsonValue {
        object(vec![
            ("lower", JsonValue::Number(lower.to_string())),
            ("upper", JsonValue::Number(upper.to_string())),
        ])
    }

    fn geometry(stagger: &str, nx: i64, ny: i64, nz: i64) -> (JsonValue, FieldContract) {
        let extra_x = i64::from(stagger == "x-face");
        let extra_y = i64::from(stagger == "y-face");
        let extra_z = i64::from(stagger == "z-face");
        let allocated = [
            Bounds {
                lower: -2,
                upper: nx + extra_x + 3,
            },
            Bounds {
                lower: -2,
                upper: ny + extra_y + 3,
            },
            Bounds {
                lower: 0,
                upper: nz + extra_z + 1,
            },
        ];
        let confirmed = [
            Bounds {
                lower: 1,
                upper: nx + extra_x,
            },
            Bounds {
                lower: 1,
                upper: ny + extra_y,
            },
            Bounds {
                lower: 1,
                upper: nz + extra_z,
            },
        ];
        let shape = [
            allocated[2].count().unwrap(),
            allocated[1].count().unwrap(),
            allocated[0].count().unwrap(),
        ];
        let slice_pairs: Vec<(u64, u64)> = [2usize, 1, 0]
            .into_iter()
            .map(|axis| {
                let start = (confirmed[axis].lower - allocated[axis].lower) as u64;
                (start, start + confirmed[axis].count().unwrap())
            })
            .collect();
        let slices = slice_pairs
            .iter()
            .map(|(start, stop)| object(vec![("start", number(*start)), ("stop", number(*stop))]))
            .collect();
        let geometry = object(vec![
            (
                "allocated_shape_zyx",
                JsonValue::Array(shape.into_iter().map(number).collect()),
            ),
            (
                "allocation_native_bounds",
                object(vec![
                    ("x", bounds_value(allocated[0].lower, allocated[0].upper)),
                    ("y", bounds_value(allocated[1].lower, allocated[1].upper)),
                    ("z", bounds_value(allocated[2].lower, allocated[2].upper)),
                ]),
            ),
            (
                "confirmed_interior_native_bounds",
                object(vec![
                    ("x", bounds_value(confirmed[0].lower, confirmed[0].upper)),
                    ("y", bounds_value(confirmed[1].lower, confirmed[1].upper)),
                    ("z", bounds_value(confirmed[2].lower, confirmed[2].upper)),
                ]),
            ),
            ("confirmed_interior_slices_zyx", JsonValue::Array(slices)),
            (
                "halo_authority",
                string("unconfirmed-outside-physical-interior"),
            ),
            ("stagger", string(stagger)),
        ]);
        let nbytes = shape.into_iter().product::<u64>() * 4;
        let contract = FieldContract {
            nbytes,
            shape_zyx: shape.map(|value| value as usize),
            slices_zyx: [
                (slice_pairs[0].0 as usize, slice_pairs[0].1 as usize),
                (slice_pairs[1].0 as usize, slice_pairs[1].1 as usize),
                (slice_pairs[2].0 as usize, slice_pairs[2].1 as usize),
            ],
        };
        (geometry, contract)
    }

    fn sentinel_payload(contract: FieldContract) -> Vec<u8> {
        let mut payload = Vec::with_capacity(contract.nbytes as usize);
        let [z_count, y_count, x_count] = contract.shape_zyx;
        let [(z_start, z_stop), (y_start, y_stop), (x_start, x_stop)] = contract.slices_zyx;
        for z in 0..z_count {
            for y in 0..y_count {
                for x in 0..x_count {
                    let word = if (z_start..z_stop).contains(&z)
                        && (y_start..y_stop).contains(&y)
                        && (x_start..x_stop).contains(&x)
                    {
                        0x0000_0000u32
                    } else {
                        0x7fc0_0000u32
                    };
                    payload.extend_from_slice(&word.to_le_bytes());
                }
            }
        }
        payload
    }

    fn coordinate_rows(nx: i64, ny: i64) -> JsonValue {
        JsonValue::Array(
            COORDINATE_SPECS
                .into_iter()
                .map(|spec| {
                    let width = if spec.axis == "x" { nx } else { ny };
                    let expected = Bounds {
                        lower: -2,
                        upper: width + 3 + i64::from(spec.face),
                    };
                    object(vec![
                        ("axis", string(spec.axis)),
                        ("count", number(expected.count().unwrap())),
                        ("location", string(spec.location)),
                        ("name", string(spec.name)),
                        (
                            "native_bounds",
                            bounds_value(expected.lower, expected.upper),
                        ),
                        (
                            "status",
                            string("allocation-only-values-deferred-to-O-GRID"),
                        ),
                        ("units", string("m")),
                    ])
                })
                .collect(),
        )
    }

    fn deferred_rows() -> JsonValue {
        JsonValue::Array(
            DEFERRED_SPECS
                .into_iter()
                .map(|spec| {
                    object(vec![
                        ("gate", string(spec.gate)),
                        ("name", string(spec.name)),
                        (
                            "native_names",
                            JsonValue::Array(
                                spec.native_names.iter().map(|name| string(name)).collect(),
                            ),
                        ),
                        ("reason", string(spec.reason)),
                    ])
                })
                .collect(),
        )
    }

    fn fixture() -> Vec<u8> {
        let nx = 16;
        let ny = 16;
        let nz = 16;
        let mut field_values = Vec::new();
        let mut payloads = Vec::new();
        let mut payload_manifest = Vec::new();
        let mut offset = 0u64;
        for spec in FIELD_SPECS {
            let (geometry, contract) = geometry(spec.stagger, nx, ny, nz);
            field_values.push(object(vec![
                ("dtype", string("<f4")),
                ("geometry", geometry),
                (
                    "initialization",
                    object(vec![
                        ("confirmed_interior_word", string("0x00000000")),
                        ("outside_confirmed_interior_word", string("0x7fc00000")),
                        ("status", string("structural-sentinel-only")),
                    ]),
                ),
                ("itemsize", number(4)),
                ("live_region_policy", string(spec.live_region)),
                ("meaning", string(spec.meaning)),
                ("name", string(spec.name)),
                ("nbytes", number(contract.nbytes)),
                ("output_name", string(spec.output_name)),
                ("persistence", string(spec.persistence)),
                ("units", string(spec.units)),
            ]));
            let payload = sentinel_payload(contract);
            payload_manifest.push(object(vec![
                ("name", string(spec.name)),
                ("nbytes", number(contract.nbytes)),
                ("offset", number(offset)),
                ("sha256", string(&sha256::digest_hex(&payload))),
            ]));
            offset += contract.nbytes;
            payloads.push(payload);
        }
        let scalar_geometry = geometry("scalar", nx, ny, nz).0;
        let x_geometry = geometry("x-face", nx, ny, nz).0;
        let y_geometry = geometry("y-face", nx, ny, nz).0;
        let z_geometry = geometry("z-face", nx, ny, nz).0;
        let plan = object(vec![
            ("authority_sha256", string(AUTHORITY_SHA256)),
            ("claim_limit", string(CLAIM_LIMIT)),
            ("config_sha256", string(&"1".repeat(64))),
            ("deferred_state_families", deferred_rows()),
            ("fields", JsonValue::Array(field_values)),
            (
                "grid",
                object(vec![
                    ("axis_order", string("zyx")),
                    ("coordinates", coordinate_rows(nx, ny)),
                    (
                        "dimensions",
                        object(vec![
                            ("nx", number(nx as u64)),
                            ("ny", number(ny as u64)),
                            ("nz", number(nz as u64)),
                        ]),
                    ),
                    ("fortran_bounds", string("inclusive")),
                    (
                        "halo",
                        object(vec![("x", number(3)), ("y", number(3)), ("z", number(1))]),
                    ),
                    (
                        "spacing_m",
                        JsonValue::Array(vec![
                            JsonValue::Number("1000.0".into()),
                            JsonValue::Number("1000.0".into()),
                            JsonValue::Number("1000.0".into()),
                        ]),
                    ),
                    (
                        "stagger_geometry",
                        object(vec![
                            ("scalar", scalar_geometry),
                            ("x-face", x_geometry),
                            ("y-face", y_geometry),
                            ("z-face", z_geometry),
                        ]),
                    ),
                ]),
            ),
            ("lane", string("dry-dycore")),
            ("schema", string(STATE_PLAN_SCHEMA)),
            ("status", string("structural-allocation-contract-only")),
            ("total_nbytes", number(offset)),
        ]);
        let plan_sha = sha256::digest_hex(&plan.canonical_bytes());
        let header = object(vec![
            ("claim_limit", string(CLAIM_LIMIT)),
            ("payload_byte_order", string("little")),
            ("payload_order", string("plan-field-order")),
            ("payloads", JsonValue::Array(payload_manifest)),
            ("plan", plan),
            ("plan_sha256", string(&plan_sha)),
            ("schema", string(ZERO_STEP_SCHEMA)),
            ("state_schema", string(STATE_SCHEMA)),
            ("total_payload_bytes", number(offset)),
        ]);
        let header = header.canonical_bytes();
        let mut envelope = Vec::new();
        envelope.extend_from_slice(&MAGIC);
        envelope.extend_from_slice(&(header.len() as u64).to_le_bytes());
        envelope.extend_from_slice(&header);
        for payload in payloads {
            envelope.extend_from_slice(&payload);
        }
        envelope
    }

    fn split(blob: &[u8]) -> (&[u8], &[u8]) {
        let header_len = u64::from_le_bytes(blob[8..16].try_into().unwrap()) as usize;
        (&blob[16..16 + header_len], &blob[16 + header_len..])
    }

    fn member_mut<'a>(value: &'a mut JsonValue, key: &str) -> &'a mut JsonValue {
        match value {
            JsonValue::Object(entries) => entries
                .iter_mut()
                .find(|(candidate, _)| candidate == key)
                .map(|(_, value)| value)
                .unwrap_or_else(|| panic!("missing test key {key}")),
            _ => panic!("test value is not an object"),
        }
    }

    fn array_mut(value: &mut JsonValue) -> &mut Vec<JsonValue> {
        match value {
            JsonValue::Array(values) => values,
            _ => panic!("test value is not an array"),
        }
    }

    fn parsed_header(blob: &[u8]) -> JsonValue {
        let (header, _) = split(blob);
        json::parse(header).expect("fixture header parses")
    }

    fn refresh_plan_sha(header: &mut JsonValue) {
        let digest = {
            let plan = member_mut(header, "plan");
            sha256::digest_hex(&plan.canonical_bytes())
        };
        *member_mut(header, "plan_sha256") = string(&digest);
    }

    fn reframe(header: &JsonValue, payload: &[u8]) -> Vec<u8> {
        let header = header.canonical_bytes();
        let mut changed = Vec::with_capacity(PREFIX_BYTES + header.len() + payload.len());
        changed.extend_from_slice(&MAGIC);
        changed.extend_from_slice(&(header.len() as u64).to_le_bytes());
        changed.extend_from_slice(&header);
        changed.extend_from_slice(payload);
        changed
    }

    #[test]
    fn verifies_and_byte_exactly_round_trips_v1() {
        let bytes = fixture();
        let container = Container::parse(&bytes).expect("fixture validates");
        assert_eq!(container.encode(), bytes);
        assert_eq!(container.schema(), ZERO_STEP_SCHEMA);
        assert_eq!(container.payloads().len(), 10);
        assert_eq!(
            container.payload_bytes("ua").unwrap().len(),
            container.payloads().next().unwrap().nbytes as usize
        );
        assert!(container.payload_bytes("not-a-field").is_none());
        assert!(container.inspect_json().contains(ARRAY_LAYOUT));
        container
            .verify_internal_roundtrip()
            .expect("round trip verifies");
    }

    #[test]
    fn rejects_noncanonical_header() {
        let bytes = fixture();
        let (header, payload) = split(&bytes);
        let mut changed_header = vec![b' '];
        changed_header.extend_from_slice(header);
        let mut changed = Vec::new();
        changed.extend_from_slice(&MAGIC);
        changed.extend_from_slice(&(changed_header.len() as u64).to_le_bytes());
        changed.extend_from_slice(&changed_header);
        changed.extend_from_slice(payload);
        assert!(Container::parse(&changed)
            .unwrap_err()
            .to_string()
            .contains("not canonical"));
    }

    #[test]
    fn rejects_payload_corruption() {
        let mut bytes = fixture();
        *bytes.last_mut().unwrap() ^= 1;
        assert!(Container::parse(&bytes)
            .unwrap_err()
            .to_string()
            .contains("payload hash mismatch"));
    }

    #[test]
    fn rejects_wrong_magic_and_truncation() {
        let mut bytes = fixture();
        bytes[0] ^= 1;
        assert!(Container::parse(&bytes)
            .unwrap_err()
            .to_string()
            .contains("magic/version"));
        assert!(Container::parse(b"CM1")
            .unwrap_err()
            .to_string()
            .contains("truncated"));
    }

    #[test]
    fn rejects_manifest_offset_even_with_canonical_reframe() {
        let bytes = fixture();
        let (header, payload) = split(&bytes);
        let mut header = json::parse(header).unwrap();
        let manifest = match header.get("payloads").unwrap() {
            JsonValue::Array(values) => values.clone(),
            _ => unreachable!(),
        };
        let mut manifest = manifest;
        let second = match &mut manifest[1] {
            JsonValue::Object(entries) => entries,
            _ => unreachable!(),
        };
        let (_, offset) = second.iter_mut().find(|(key, _)| key == "offset").unwrap();
        *offset = number(1);
        let entries = match &mut header {
            JsonValue::Object(entries) => entries,
            _ => unreachable!(),
        };
        entries
            .iter_mut()
            .find(|(key, _)| key == "payloads")
            .unwrap()
            .1 = JsonValue::Array(manifest);
        let header = header.canonical_bytes();
        let mut changed = Vec::new();
        changed.extend_from_slice(&MAGIC);
        changed.extend_from_slice(&(header.len() as u64).to_le_bytes());
        changed.extend_from_slice(&header);
        changed.extend_from_slice(payload);
        assert!(Container::parse(&changed)
            .unwrap_err()
            .to_string()
            .contains("contiguous"));
    }

    #[test]
    fn rejects_oversized_header_and_envelope_before_decode() {
        assert_eq!(MAX_HEADER_BYTES, 65_536);
        assert_eq!(MAX_PAYLOAD_BYTES, 67_108_864);
        assert_eq!(MAX_ENVELOPE_BYTES, 67_174_416);

        let mut oversized_header = Vec::from(MAGIC);
        oversized_header.extend_from_slice(&((MAX_HEADER_BYTES + 1) as u64).to_le_bytes());
        assert!(Container::parse(&oversized_header)
            .unwrap_err()
            .to_string()
            .contains("JSON header length"));

        // The bytes deliberately have invalid magic.  Receiving the envelope
        // size error proves the cap is checked before prefix or JSON decode.
        let oversized_envelope = vec![0u8; MAX_ENVELOPE_BYTES + 1];
        let error = Container::parse(&oversized_envelope)
            .unwrap_err()
            .to_string();
        assert!(error.contains("envelope exceeds"), "{error}");
    }

    /// A scratch directory on the build volume, wherever cargo is building.
    ///
    /// This test needs somewhere to make a sixteen-gibibyte *sparse* file. The
    /// build directory is the right volume -- it is known to have room for the
    /// build itself, and on many hosts `/tmp` is a tmpfs where even a sparse
    /// file of that declared length is the wrong thing to create. But it is not
    /// always `$CARGO_MANIFEST_DIR/target`: `CARGO_TARGET_DIR` moves it, which
    /// is an ordinary cargo configuration (a shared target directory across
    /// worktrees, for one), and this test used to fail with a bare "No such
    /// file or directory" whenever it was set -- a real build-layout setting
    /// reported as a container-preflight failure.
    fn build_scratch_directory() -> std::path::PathBuf {
        let root = match env::var_os("CARGO_TARGET_DIR") {
            Some(value) if !value.is_empty() => std::path::PathBuf::from(value),
            _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("target"),
        };
        fs::create_dir_all(&root).expect("create the build scratch directory");
        root
    }

    #[test]
    fn read_metadata_preflights_oversized_file_before_allocation() {
        let path = build_scratch_directory()
            .join(format!("oversized-preflight-{}.cm1szs", std::process::id()));
        let file = File::create(&path).expect("create sparse test input on the build volume");
        file.set_len((MAX_ENVELOPE_BYTES + 1) as u64)
            .expect("set sparse test length");
        drop(file);
        let error = Container::read(&path).unwrap_err().to_string();
        fs::remove_file(&path).expect("remove owned sparse test input");
        assert!(
            error.contains("envelope") && error.contains("size limit"),
            "{error}"
        );
    }

    #[test]
    fn rejects_rehashed_lane_dimension_and_spacing_drift() {
        let bytes = fixture();
        let payload = split(&bytes).1.to_vec();

        let mut dimension_header = parsed_header(&bytes);
        {
            let plan = member_mut(&mut dimension_header, "plan");
            let grid = member_mut(plan, "grid");
            let dimensions = member_mut(grid, "dimensions");
            *member_mut(dimensions, "nx") = number(17);
        }
        refresh_plan_sha(&mut dimension_header);
        let error = Container::parse(&reframe(&dimension_header, &payload))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("differ from the dry-dycore v1 profile"),
            "{error}"
        );

        let mut spacing_header = parsed_header(&bytes);
        {
            let plan = member_mut(&mut spacing_header, "plan");
            let grid = member_mut(plan, "grid");
            array_mut(member_mut(grid, "spacing_m"))[2] = JsonValue::Number("500.0".into());
        }
        refresh_plan_sha(&mut spacing_header);
        let error = Container::parse(&reframe(&spacing_header, &payload))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("grid spacing") && error.contains("dry-dycore"),
            "{error}"
        );

        let mut lane_header = parsed_header(&bytes);
        *member_mut(member_mut(&mut lane_header, "plan"), "lane") = string("product");
        refresh_plan_sha(&mut lane_header);
        let error = Container::parse(&reframe(&lane_header, &payload))
            .unwrap_err()
            .to_string();
        assert!(error.contains("product v1 profile"), "{error}");
    }

    #[test]
    fn rejects_rehashed_coordinate_semantic_drift() {
        let bytes = fixture();
        let payload = split(&bytes).1.to_vec();
        let mut header = parsed_header(&bytes);
        {
            let plan = member_mut(&mut header, "plan");
            let grid = member_mut(plan, "grid");
            let coordinates = array_mut(member_mut(grid, "coordinates"));
            *member_mut(&mut coordinates[0], "count") = number(999);
        }
        refresh_plan_sha(&mut header);
        let error = Container::parse(&reframe(&header, &payload))
            .unwrap_err()
            .to_string();
        assert!(error.contains("coordinate xh.count"), "{error}");
    }

    #[test]
    fn rejects_rehashed_stagger_geometry_semantic_drift() {
        let bytes = fixture();
        let payload = split(&bytes).1.to_vec();
        let mut header = parsed_header(&bytes);
        {
            let plan = member_mut(&mut header, "plan");
            let grid = member_mut(plan, "grid");
            let staggers = member_mut(grid, "stagger_geometry");
            let scalar = member_mut(staggers, "scalar");
            array_mut(member_mut(scalar, "allocated_shape_zyx"))[0] = number(999);
        }
        refresh_plan_sha(&mut header);
        let error = Container::parse(&reframe(&header, &payload))
            .unwrap_err()
            .to_string();
        assert!(error.contains("grid stagger_geometry.scalar"), "{error}");
    }

    #[test]
    fn rejects_rehashed_deferred_inventory_semantic_drift() {
        let bytes = fixture();
        let payload = split(&bytes).1.to_vec();
        let mut header = parsed_header(&bytes);
        {
            let plan = member_mut(&mut header, "plan");
            let deferred = array_mut(member_mut(plan, "deferred_state_families"));
            *member_mut(&mut deferred[0], "gate") = string("O-W999-FORGED");
        }
        refresh_plan_sha(&mut header);
        let error = Container::parse(&reframe(&header, &payload))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("deferred state family") && error.contains("gate"),
            "{error}"
        );
    }

    #[test]
    fn rejects_rehashed_non_sentinel_halo_and_interior_words() {
        let bytes = fixture();
        let original_payload = split(&bytes).1.to_vec();
        // ua is [z=18,y=22,x=23], with confirmed slices z=1:17,
        // y=3:19,x=3:20.  Exercise one halo and one interior word.
        let interior_word_offset = ((22 + 3) * 23 + 3) * 4;
        for word_offset in [0usize, interior_word_offset] {
            let mut payload = original_payload.clone();
            payload[word_offset] ^= 1;
            let mut header = parsed_header(&bytes);
            let digest = sha256::digest_hex(&payload[..36_432]);
            let manifest = array_mut(member_mut(&mut header, "payloads"));
            *member_mut(&mut manifest[0], "sha256") = string(&digest);
            let error = Container::parse(&reframe(&header, &payload))
                .unwrap_err()
                .to_string();
            assert!(error.contains("structural sentinel policy"), "{error}");
        }
    }

    #[test]
    fn python_dry_fixture_is_cross_language_byte_exact() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let repository = manifest_dir
            .parent()
            .and_then(Path::parent)
            .expect("crate is nested at rust/cm1-state");
        let script = concat!(
            "import sys\n",
            "from pathlib import Path\n",
            "from cm1core.config import load_config\n",
            "from cm1core.zero_step import build_structural_zero_step\n",
            "root=Path(sys.argv[1])\n",
            "blob=build_structural_zero_step(load_config(root/'examples'/'dry-psolver2-bootstrap.toml'))\n",
            "sys.stdout.buffer.write(blob)\n",
        );
        // PEP 394 leaves `python` optional and most current distributions
        // ship only `python3`, so hard-coding `python` made this test fail on
        // a working environment for a reason that has nothing to do with the
        // bytes it checks.  This is the crate's only cross-language byte
        // check, so a missing PATH alias must never read as a Rust/Python
        // divergence in the zero-step container.  GM1_PYTHON names an
        // interpreter explicitly; otherwise every candidate is tried in turn
        // and every failure is reported, whether the interpreter was absent
        // or was present and could not be executed.
        let explicit = env::var_os("GM1_PYTHON");
        let mut candidates: Vec<OsString> = Vec::new();
        if let Some(value) = explicit {
            candidates.push(value);
        } else {
            candidates.extend(["python3", "python"].iter().map(OsString::from));
        }
        let mut attempt = None;
        let mut missing = Vec::new();
        for interpreter in &candidates {
            match Command::new(interpreter)
                .args(["-B", "-c", script])
                .arg(repository)
                .env("PYTHONDONTWRITEBYTECODE", "1")
                .env("PYTHONPATH", repository.join("src"))
                .current_dir(repository)
                .output()
            {
                Ok(value) => {
                    attempt = Some(value);
                    break;
                }
                Err(error) => missing.push(format!("{}: {error}", interpreter.to_string_lossy())),
            }
        }
        let output = attempt.unwrap_or_else(|| {
            panic!(
                "no Python interpreter ran this fixture; tried {}. Set GM1_PYTHON \
                 to the interpreter that can import cm1core",
                missing.join(", ")
            )
        });
        assert!(
            output.status.success(),
            "Python fixture producer failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout.len(), 372_467);

        let container = Container::parse(&output.stdout).expect("Rust accepts Python v1 bytes");
        assert_eq!(
            container.artifact_sha256(),
            "eb480f2da26aacf62bf345a2c13036cbbc131d92949ba5cce3793b4aae498e30"
        );
        assert_eq!(container.total_payload_bytes(), 358_688);
        assert_eq!(container.encode(), output.stdout);
        assert_eq!(
            container
                .header
                .get("plan_sha256")
                .unwrap()
                .as_str()
                .unwrap(),
            "2b4cd83ed9c0ce045df3abee8468cc2349626f74afe0e4dcef43ea428f8b92eb"
        );
    }
}
