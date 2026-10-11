//! Generic deterministic forecast checkpoint and history envelopes.
//!
//! Unlike the sealed startup-only formats in [`crate::snapshot`], this module
//! accepts caller-supplied complete-step state.  The caller must provide an
//! explicit contract naming every required float32 carrier, scheduler scalar,
//! and opaque component-persistence blob.  The Rust path owns deterministic
//! ordering, serialization, validation, receipts, and atomic publication; it
//! does not yet copy state from or restore state to live CUDA allocations.

use crate::json::{self, JsonValue};
use crate::publish_timing;
use crate::sha256;
use crate::snapshot::{atomic_write_new_parts, ATOMIC_WRITE_CONTRACT};
use crate::{Result, StateError};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{Read, Seek};
use std::path::Path;
use std::thread;
use std::time::Instant;

pub const FORECAST_CHECKPOINT_SCHEMA: &str = "gpuwm-cm1.forecast-checkpoint/generic-v1";
pub const FORECAST_HISTORY_SCHEMA: &str = "gpuwm-cm1.forecast-history/generic-v1";
pub const FORECAST_RECEIPT_SCHEMA: &str = "gpuwm-cm1.forecast-rust-receipt/v1";
pub const FORECAST_CHECKPOINT_MAGIC: [u8; 8] = *b"CM1FCP1\0";
pub const FORECAST_HISTORY_MAGIC: [u8; 8] = *b"CM1FHS1\0";
pub const MAX_FORECAST_HEADER_BYTES: usize = 1024 * 1024;
/// Host-memory bound for one forecast payload.
///
/// This is not a container bound.  Nothing in `CM1FCP1`/`CM1FHS1` is 32-bit:
/// the sixteen-byte prefix is eight magic bytes plus the header length as a
/// little-endian `u64`, and every payload `offset`, every `nbytes`, and
/// `payload_nbytes` itself is a JSON integer carried as an exact decimal
/// string by [`crate::json::JsonValue`] and read back with `as_u64`.  A v1
/// envelope has addressed payloads far past 2^32 bytes since it was written,
/// so raising this number needs no versioned successor magic and leaves every
/// artifact already on disk byte-identical and readable.
///
/// What the bound buys is a preflighted ceiling on host allocation.  Admission
/// reads the whole artifact into one buffer and materializes one float32
/// copy of every carrier out of it, so admitting a payload of `n`
/// bytes costs [`forecast_admission_host_bytes`] of them.  Every forecast
/// read prices its own artifact against this number from the header length in
/// the prefix, before a buffer for it exists, and then reserves that buffer
/// fallibly: a payload past the ceiling, or one this host cannot give back,
/// is refused by name instead of aborting the reader on allocation.
///
/// Sixteen gibibytes is the value because its admission working set, 32 GiB,
/// is what a host driving a 32 GiB device already provides.  A complete GM1
/// history profile costs sixteen float32 words per gridpoint plus a
/// staggered-face and surface remainder of `5*ny*nx + nz*ny + nz*nx` words,
/// so the capacity is just under 268435456 gridpoints and its exact value
/// depends on the aspect ratio: 512x512x1023 (268173312 gridpoints) prices at
/// 17172525056 bytes and publishes, while 512x512x1024 -- exactly 268435456
/// gridpoints -- prices at 17189306368 and is refused.  A 600x600x180 frame
/// costs 3.87 GiB and a 1000x1000x80 frame 4.79 GiB.  CM1 itself caps no
/// payload -- `writeout.F:9818` opens the history file with `recl=(nx*ny*4)`
/// and `writeout.F:8842` streams one record per model level -- so this
/// ceiling is this port's host-resource policy and stays named as one.
pub const MAX_FORECAST_PAYLOAD_BYTES: usize = 16 * 1024 * 1024 * 1024;
/// Peak host bytes one payload byte costs at admission: the artifact buffer
/// the reader fills from the file, plus the float32 carriers materialized out
/// of it.
pub const FORECAST_ADMISSION_HOST_BYTES_PER_PAYLOAD_BYTE: usize = 2;
pub const MAX_FORECAST_ARRAYS: usize = 4096;
pub const MAX_FORECAST_SCALARS: usize = 4096;
pub const MAX_FORECAST_COMPONENT_BLOBS: usize = 1024;

const PREFIX_BYTES: usize = 16;
/// Largest admissible artifact: prefix, header and payload.  Readers that
/// gate a forecast artifact by file size take their bound from here rather
/// than from a second number that would refuse a frame this module admits.
pub const MAX_FORECAST_ENVELOPE_BYTES: usize =
    PREFIX_BYTES + MAX_FORECAST_HEADER_BYTES + MAX_FORECAST_PAYLOAD_BYTES;
// Every payload offset in this module is a `usize`.  On a 32-bit target the
// bound above would wrap to a smaller number than it spells and each offset
// would silently truncate, so refuse to build rather than publish an envelope
// whose offsets do not address their own payload.
const _: () = assert!(
    usize::BITS >= 64,
    "the forecast envelope requires a 64-bit usize to address MAX_FORECAST_PAYLOAD_BYTES"
);
const AUTHORITY_SHA256: &str = "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
const REPRESENTATION: &str = "arrays are raw IEEE-754 binary32 little-endian in caller-contracted dimension order; component blobs are opaque bytes";
const LIVE_STATE_WIRING: &str = "deferred-no-cuda-device-copy-or-forecast-driver-restore";
const CHECKPOINT_CAPABILITY: &str = "caller-contracted-complete-step-forecast-state";
const HISTORY_CAPABILITY: &str = "caller-contracted-observational-forecast-frame";
const CHECKPOINT_CLAIM_LIMIT: &str = "deterministic generic complete-step carrier/scalar/component envelope only; semantic CM1 inventory completeness is exactly the supplied contract, and live CUDA capture/restore plus uninterrupted/resumed trajectory identity remain deferred";
const HISTORY_CLAIM_LIMIT: &str = "deterministic generic observational forecast frame only; semantic field selection is exactly the supplied contract, and live CUDA capture, NetCDF compatibility, weather rendering, and trajectory validation remain deferred";

const TOP_KEYS: &[&str] = &[
    "arrays",
    "authority_sha256",
    "capability",
    "claim_limit",
    "clock",
    "component_blobs",
    "contract",
    "contract_sha256",
    "live_state_wiring",
    "payload_nbytes",
    "payload_order",
    "payload_sha256",
    "representation",
    "scheduler_scalars",
    "schema",
];
const CONTRACT_KEYS: &[&str] = &[
    "array_specs",
    "component_blob_specs",
    "config_sha256",
    "execution_identity_sha256",
    "profile_id",
    "scheduler_scalar_specs",
];
const ARRAY_SPEC_KEYS: &[&str] = &["bounds", "meaning", "name", "shape", "stagger", "units"];
const SCALAR_SPEC_KEYS: &[&str] = &["name", "type"];
const COMPONENT_SPEC_KEYS: &[&str] = &["max_nbytes", "name", "schema"];
const ARRAY_HEADER_KEYS: &[&str] = &["name", "nbytes", "offset", "sha256"];
const SCALAR_HEADER_KEYS: &[&str] = &["name", "type", "value"];
const COMPONENT_HEADER_KEYS: &[&str] = &["name", "nbytes", "offset", "schema", "sha256"];
const CLOCK_KEYS: &[&str] = &["dt_f64_bits", "model_time_f64_bits", "step"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForecastArtifactKind {
    Checkpoint,
    History,
}

impl ForecastArtifactKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Checkpoint => "forecast-checkpoint",
            Self::History => "forecast-history",
        }
    }

    const fn magic(self) -> [u8; 8] {
        match self {
            Self::Checkpoint => FORECAST_CHECKPOINT_MAGIC,
            Self::History => FORECAST_HISTORY_MAGIC,
        }
    }

    const fn schema(self) -> &'static str {
        match self {
            Self::Checkpoint => FORECAST_CHECKPOINT_SCHEMA,
            Self::History => FORECAST_HISTORY_SCHEMA,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerScalarType {
    I64,
    U64,
    F64,
    Bool,
}

impl SchedulerScalarType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::I64 => "i64",
            Self::U64 => "u64",
            Self::F64 => "f64-bits",
            Self::Bool => "bool",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SchedulerScalarValue {
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
}

impl SchedulerScalarValue {
    pub const fn scalar_type(self) -> SchedulerScalarType {
        match self {
            Self::I64(_) => SchedulerScalarType::I64,
            Self::U64(_) => SchedulerScalarType::U64,
            Self::F64(_) => SchedulerScalarType::F64,
            Self::Bool(_) => SchedulerScalarType::Bool,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForecastArraySpec {
    name: String,
    meaning: String,
    units: String,
    stagger: String,
    bounds: Vec<i32>,
    shape: Vec<usize>,
}

impl ForecastArraySpec {
    pub fn new(
        name: impl Into<String>,
        meaning: impl Into<String>,
        units: impl Into<String>,
        stagger: impl Into<String>,
        bounds: Vec<i32>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        let spec = Self {
            name: name.into(),
            meaning: meaning.into(),
            units: units.into(),
            stagger: stagger.into(),
            bounds,
            shape,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The semantic columns a field table carries, so an envelope read back
    /// can be re-emitted as the exact table its writer handed in.
    pub fn meaning(&self) -> &str {
        &self.meaning
    }

    pub fn units(&self) -> &str {
        &self.units
    }

    pub fn stagger(&self) -> &str {
        &self.stagger
    }

    pub fn bounds(&self) -> &[i32] {
        &self.bounds
    }

    fn words(&self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |count, extent| {
            count
                .checked_mul(*extent)
                .ok_or_else(|| invalid(format!("array {} shape product overflows", self.name)))
        })
    }

    fn validate(&self) -> Result<()> {
        validate_name(&self.name, "array carrier name")?;
        validate_text(&self.meaning, 256, "array meaning")?;
        validate_text(&self.units, 64, "array units")?;
        validate_text(&self.stagger, 64, "array stagger")?;
        if self.shape.is_empty() || self.shape.len() > 8 {
            return Err(invalid(format!(
                "array {} rank is {}; expected 1 through 8",
                self.name,
                self.shape.len()
            )));
        }
        if self.shape.contains(&0) {
            return Err(invalid(format!(
                "array {} shape contains a zero extent",
                self.name
            )));
        }
        if self.bounds.len() != 2 * self.shape.len() {
            return Err(invalid(format!(
                "array {} has {} bounds values; expected {}",
                self.name,
                self.bounds.len(),
                2 * self.shape.len()
            )));
        }
        for (axis, extent) in self.shape.iter().enumerate() {
            let lower = self.bounds[2 * axis];
            let upper = self.bounds[2 * axis + 1];
            if upper < lower {
                return Err(invalid(format!(
                    "array {} axis {axis} bounds {lower}:{upper} are reversed",
                    self.name
                )));
            }
            let bounded_extent = i64::from(upper) - i64::from(lower) + 1;
            if usize::try_from(bounded_extent).ok() != Some(*extent) {
                return Err(invalid(format!(
                    "array {} axis {axis} bounds {lower}:{upper} imply {bounded_extent} values; shape declares {extent}",
                    self.name
                )));
            }
        }
        let nbytes = self
            .words()?
            .checked_mul(4)
            .ok_or_else(|| invalid(format!("array {} byte count overflows", self.name)))?;
        if nbytes > MAX_FORECAST_PAYLOAD_BYTES {
            return Err(invalid(format!(
                "array {} alone has {nbytes} bytes; maximum forecast payload is {MAX_FORECAST_PAYLOAD_BYTES}",
                self.name
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerScalarSpec {
    name: String,
    scalar_type: SchedulerScalarType,
}

impl SchedulerScalarSpec {
    pub fn new(name: impl Into<String>, scalar_type: SchedulerScalarType) -> Result<Self> {
        let spec = Self {
            name: name.into(),
            scalar_type,
        };
        validate_name(&spec.name, "scheduler scalar name")?;
        Ok(spec)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentBlobSpec {
    name: String,
    schema: String,
    max_nbytes: usize,
}

impl ComponentBlobSpec {
    pub fn new(
        name: impl Into<String>,
        schema: impl Into<String>,
        max_nbytes: usize,
    ) -> Result<Self> {
        let spec = Self {
            name: name.into(),
            schema: schema.into(),
            max_nbytes,
        };
        validate_name(&spec.name, "component blob name")?;
        validate_identity(&spec.schema, 128, "component blob schema")?;
        if spec.max_nbytes > MAX_FORECAST_PAYLOAD_BYTES {
            return Err(invalid(format!(
                "component blob {} maximum {} exceeds forecast payload maximum {}",
                spec.name, spec.max_nbytes, MAX_FORECAST_PAYLOAD_BYTES
            )));
        }
        Ok(spec)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForecastContract {
    profile_id: String,
    config_sha256: String,
    execution_identity_sha256: String,
    arrays: Vec<ForecastArraySpec>,
    scheduler_scalars: Vec<SchedulerScalarSpec>,
    component_blobs: Vec<ComponentBlobSpec>,
}

impl ForecastContract {
    pub fn new(
        profile_id: impl Into<String>,
        config_sha256: impl Into<String>,
        execution_identity_sha256: impl Into<String>,
        mut arrays: Vec<ForecastArraySpec>,
        mut scheduler_scalars: Vec<SchedulerScalarSpec>,
        mut component_blobs: Vec<ComponentBlobSpec>,
    ) -> Result<Self> {
        let profile_id = profile_id.into();
        let config_sha256 = config_sha256.into();
        let execution_identity_sha256 = execution_identity_sha256.into();
        validate_identity(&profile_id, 128, "forecast profile id")?;
        validate_sha256(&config_sha256, "forecast config SHA-256")?;
        validate_sha256(
            &execution_identity_sha256,
            "forecast execution identity SHA-256",
        )?;
        if arrays.is_empty() {
            return Err(invalid(
                "forecast contract requires at least one named float32 array carrier",
            ));
        }
        if arrays.len() > MAX_FORECAST_ARRAYS {
            return Err(invalid(format!(
                "forecast contract has {} arrays; maximum is {MAX_FORECAST_ARRAYS}",
                arrays.len()
            )));
        }
        if scheduler_scalars.len() > MAX_FORECAST_SCALARS {
            return Err(invalid(format!(
                "forecast contract has {} scheduler scalars; maximum is {MAX_FORECAST_SCALARS}",
                scheduler_scalars.len()
            )));
        }
        if component_blobs.len() > MAX_FORECAST_COMPONENT_BLOBS {
            return Err(invalid(format!(
                "forecast contract has {} component blobs; maximum is {MAX_FORECAST_COMPONENT_BLOBS}",
                component_blobs.len()
            )));
        }
        arrays.sort_by(|left, right| left.name.cmp(&right.name));
        scheduler_scalars.sort_by(|left, right| left.name.cmp(&right.name));
        component_blobs.sort_by(|left, right| left.name.cmp(&right.name));
        reject_spec_duplicates(
            arrays.iter().map(|spec| spec.name.as_str()),
            "array carrier contract",
        )?;
        reject_spec_duplicates(
            scheduler_scalars.iter().map(|spec| spec.name.as_str()),
            "scheduler scalar contract",
        )?;
        reject_spec_duplicates(
            component_blobs.iter().map(|spec| spec.name.as_str()),
            "component blob contract",
        )?;
        let mut global = BTreeSet::new();
        for (category, name) in arrays
            .iter()
            .map(|spec| ("array", spec.name.as_str()))
            .chain(
                scheduler_scalars
                    .iter()
                    .map(|spec| ("scheduler scalar", spec.name.as_str())),
            )
            .chain(
                component_blobs
                    .iter()
                    .map(|spec| ("component blob", spec.name.as_str())),
            )
        {
            if !global.insert(name) {
                return Err(invalid(format!(
                    "forecast contract name {name:?} is reused across carrier categories (at {category})"
                )));
            }
        }
        let array_nbytes = arrays.iter().try_fold(0usize, |total, spec| {
            let nbytes = spec
                .words()?
                .checked_mul(4)
                .ok_or_else(|| invalid(format!("array {} byte count overflows", spec.name)))?;
            total
                .checked_add(nbytes)
                .ok_or_else(|| invalid("forecast contract array byte total overflows"))
        })?;
        let contracted_max = component_blobs
            .iter()
            .try_fold(array_nbytes, |total, spec| {
                total.checked_add(spec.max_nbytes).ok_or_else(|| {
                    invalid("forecast contract array/component byte total overflows")
                })
            })?;
        if contracted_max > MAX_FORECAST_PAYLOAD_BYTES {
            return Err(invalid(format!(
                "forecast contract permits {contracted_max} array/component bytes; maximum is {MAX_FORECAST_PAYLOAD_BYTES}"
            )));
        }
        Ok(Self {
            profile_id,
            config_sha256,
            execution_identity_sha256,
            arrays,
            scheduler_scalars,
            component_blobs,
        })
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn arrays(&self) -> &[ForecastArraySpec] {
        &self.arrays
    }

    pub fn scheduler_scalars(&self) -> &[SchedulerScalarSpec] {
        &self.scheduler_scalars
    }

    pub fn component_blobs(&self) -> &[ComponentBlobSpec] {
        &self.component_blobs
    }

    pub fn sha256(&self) -> String {
        sha256::digest_hex(&contract_json(self).canonical_bytes())
    }

    /// Canonical contract JSON embedded verbatim in forecast artifacts.
    pub fn canonical_json(&self) -> String {
        String::from_utf8(contract_json(self).canonical_bytes())
            .expect("canonical forecast contract JSON is UTF-8")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForecastClock {
    step: u64,
    model_time_s: f64,
    dt_s: f64,
}

impl ForecastClock {
    pub fn new(step: u64, model_time_s: f64, dt_s: f64) -> Result<Self> {
        if !model_time_s.is_finite() || model_time_s < 0.0 {
            return Err(invalid(format!(
                "forecast model time must be finite and nonnegative; got {model_time_s:?}"
            )));
        }
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(invalid(format!(
                "forecast dt must be finite and positive; got {dt_s:?}"
            )));
        }
        Ok(Self {
            step,
            model_time_s,
            dt_s,
        })
    }

    pub fn step(self) -> u64 {
        self.step
    }

    pub fn model_time_s(self) -> f64 {
        self.model_time_s
    }

    pub fn dt_s(self) -> f64 {
        self.dt_s
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForecastArray {
    name: String,
    values: Vec<f32>,
}

impl ForecastArray {
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    #[cfg(test)]
    pub(crate) fn values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerScalar {
    name: String,
    value: SchedulerScalarValue,
}

impl SchedulerScalar {
    pub fn new(name: impl Into<String>, value: SchedulerScalarValue) -> Result<Self> {
        let scalar = Self {
            name: name.into(),
            value,
        };
        validate_name(&scalar.name, "scheduler scalar name")?;
        if let SchedulerScalarValue::F64(value) = scalar.value {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "scheduler scalar {} must contain a finite f64; got {value:?}",
                    scalar.name
                )));
            }
        }
        Ok(scalar)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> SchedulerScalarValue {
        self.value
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentBlob {
    name: String,
    schema: String,
    bytes: Vec<u8>,
}

impl ComponentBlob {
    pub fn new(name: impl Into<String>, schema: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            schema: schema.into(),
            bytes,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForecastFrame {
    clock: ForecastClock,
    arrays: Vec<ForecastArray>,
    scheduler_scalars: Vec<SchedulerScalar>,
    component_blobs: Vec<ComponentBlob>,
}

impl ForecastFrame {
    pub fn new(
        clock: ForecastClock,
        arrays: Vec<ForecastArray>,
        scheduler_scalars: Vec<SchedulerScalar>,
        component_blobs: Vec<ComponentBlob>,
    ) -> Self {
        Self {
            clock,
            arrays,
            scheduler_scalars,
            component_blobs,
        }
    }

    pub fn clock(&self) -> ForecastClock {
        self.clock
    }

    pub fn arrays(&self) -> &[ForecastArray] {
        &self.arrays
    }

    pub fn scheduler_scalars(&self) -> &[SchedulerScalar] {
        &self.scheduler_scalars
    }

    pub fn component_blobs(&self) -> &[ComponentBlob] {
        &self.component_blobs
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForecastReceipt {
    pub kind: ForecastArtifactKind,
    pub contract_sha256: String,
    pub artifact_sha256: String,
    pub header_sha256: String,
    pub payload_sha256: String,
    pub artifact_nbytes: usize,
    pub payload_nbytes: usize,
    pub arrays: usize,
    pub scheduler_scalars: usize,
    pub component_blobs: usize,
    pub step: u64,
    pub model_time_f64_bits: String,
    pub dt_f64_bits: String,
}

impl ForecastReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("arrays", number(self.arrays)),
            ("artifact_nbytes", number(self.artifact_nbytes)),
            ("artifact_sha256", string(&self.artifact_sha256)),
            ("atomic_write_contract", string(ATOMIC_WRITE_CONTRACT)),
            ("capability", string(self.kind.capability())),
            ("component_blobs", number(self.component_blobs)),
            ("contract_sha256", string(&self.contract_sha256)),
            ("dt_f64_bits", string(&self.dt_f64_bits)),
            ("header_sha256", string(&self.header_sha256)),
            ("kind", string(self.kind.as_str())),
            ("live_state_wiring", string(LIVE_STATE_WIRING)),
            ("model_time_f64_bits", string(&self.model_time_f64_bits)),
            ("payload_nbytes", number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            ("scheduler_scalars", number(self.scheduler_scalars)),
            ("schema", string(FORECAST_RECEIPT_SCHEMA)),
            ("snapshot_schema", string(self.kind.schema())),
            ("step", u64_number(self.step)),
        ]);
        String::from_utf8(value.canonical_bytes()).expect("canonical JSON is UTF-8")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RestoredForecast {
    kind: ForecastArtifactKind,
    frame: ForecastFrame,
    receipt: ForecastReceipt,
    array_payload_sha256: String,
}

impl RestoredForecast {
    pub fn kind(&self) -> ForecastArtifactKind {
        self.kind
    }

    pub fn frame(&self) -> &ForecastFrame {
        &self.frame
    }

    pub fn receipt(&self) -> &ForecastReceipt {
        &self.receipt
    }

    /// SHA-256 of all contracted float32 carrier bytes in canonical payload
    /// order, computed from the validated little-endian envelope bytes.
    pub fn array_payload_sha256(&self) -> &str {
        &self.array_payload_sha256
    }
}

/// A fully authenticated forecast envelope whose array bytes were validated
/// without allocating a second resident copy of every float32 carrier.
///
/// This is intentionally inspection-only: callers that need carrier values
/// must use [`RestoredForecast`]. The raw array payload digest covers every
/// contracted array byte in canonical payload order.
#[derive(Clone, Debug, PartialEq)]
pub struct InspectedForecast {
    kind: ForecastArtifactKind,
    clock: ForecastClock,
    receipt: ForecastReceipt,
    array_payload_sha256: String,
}

impl InspectedForecast {
    pub fn kind(&self) -> ForecastArtifactKind {
        self.kind
    }

    pub fn clock(&self) -> ForecastClock {
        self.clock
    }

    pub fn receipt(&self) -> &ForecastReceipt {
        &self.receipt
    }

    pub fn array_payload_sha256(&self) -> &str {
        &self.array_payload_sha256
    }
}

#[derive(Clone, Debug)]
struct NormalizedFrame<'a> {
    clock: ForecastClock,
    arrays: Vec<(&'a ForecastArraySpec, &'a ForecastArray)>,
    scalars: Vec<(&'a SchedulerScalarSpec, &'a SchedulerScalar)>,
    blobs: Vec<(&'a ComponentBlobSpec, &'a ComponentBlob)>,
}

/// One encoded artifact held as its three published parts.
///
/// Publication hands the parts straight to the atomic writer, so a payload is
/// never copied into a second full-size buffer on its way to disk.  The
/// concatenation is still exactly what the reader sees, and [`Self::bytes`]
/// produces it for callers that want the whole artifact in memory.
#[cfg(test)]
#[derive(Clone, Debug)]
struct EncodedForecast {
    prefix: [u8; PREFIX_BYTES],
    header: Vec<u8>,
    payload: Vec<u8>,
    receipt: ForecastReceipt,
}

#[cfg(test)]
impl EncodedForecast {
    fn parts(&self) -> [&[u8]; 3] {
        [&self.prefix, &self.header, &self.payload]
    }

    fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.receipt.artifact_nbytes);
        for part in self.parts() {
            bytes.extend_from_slice(part);
        }
        bytes
    }
}

#[derive(Clone, Copy)]
enum ArrayDecode {
    Materialize,
    Inspect,
}

struct DecodedForecast {
    arrays: Option<Vec<ForecastArray>>,
    array_payload_sha256: String,
    blobs: Vec<ComponentBlob>,
    clock: ForecastClock,
    receipt: ForecastReceipt,
    scalars: Vec<SchedulerScalar>,
}

/// Peak host bytes required to admit one forecast payload of `payload_nbytes`.
///
/// Two callers preflight with this.  Producers price a frame before they pay
/// for the device-to-host transfer that would produce it; every reader in
/// this crate prices the artifact it is about to hold, from the header length
/// its prefix declares, before the buffer for it exists.  The refusal names
/// the ceiling that stops an unbounded read from aborting the process on
/// allocation.
pub fn forecast_admission_host_bytes(payload_nbytes: usize) -> Result<usize> {
    if payload_nbytes > MAX_FORECAST_PAYLOAD_BYTES {
        return Err(invalid(format!(
            "forecast payload of {payload_nbytes} bytes exceeds the {MAX_FORECAST_PAYLOAD_BYTES}-byte host admission ceiling; reading it would allocate the artifact plus one float32 copy of every carrier, and an unbounded payload aborts the reader on allocation instead of being refused by name"
        )));
    }
    payload_nbytes
        .checked_mul(FORECAST_ADMISSION_HOST_BYTES_PER_PAYLOAD_BYTE)
        .ok_or_else(|| invalid("forecast admission host byte count overflows"))
}

pub fn write_forecast_checkpoint_new(
    contract: &ForecastContract,
    frame: &ForecastFrame,
    output: impl AsRef<Path>,
) -> Result<ForecastReceipt> {
    write_new(
        ForecastArtifactKind::Checkpoint,
        contract,
        frame,
        output.as_ref(),
    )
}

pub fn write_forecast_history_new(
    contract: &ForecastContract,
    frame: &ForecastFrame,
    output: impl AsRef<Path>,
) -> Result<ForecastReceipt> {
    write_new(
        ForecastArtifactKind::History,
        contract,
        frame,
        output.as_ref(),
    )
}

pub fn read_forecast_checkpoint(
    contract: &ForecastContract,
    input: impl AsRef<Path>,
) -> Result<RestoredForecast> {
    read_forecast(ForecastArtifactKind::Checkpoint, contract, input.as_ref())
}

/// Read a self-describing checkpoint only when its embedded contract has the
/// independently supplied expected SHA-256.
pub fn read_forecast_checkpoint_embedded(
    input: impl AsRef<Path>,
    expected_contract_sha256: &str,
) -> Result<(ForecastContract, RestoredForecast)> {
    read_forecast_embedded(
        ForecastArtifactKind::Checkpoint,
        input.as_ref(),
        expected_contract_sha256,
    )
}

/// Fully validate a self-describing checkpoint without materializing a second
/// in-memory copy of all float32 carriers.
pub fn inspect_forecast_checkpoint_embedded(
    input: impl AsRef<Path>,
    expected_contract_sha256: &str,
) -> Result<(ForecastContract, InspectedForecast)> {
    inspect_forecast_embedded(
        ForecastArtifactKind::Checkpoint,
        input.as_ref(),
        expected_contract_sha256,
    )
}

pub fn read_forecast_history(
    contract: &ForecastContract,
    input: impl AsRef<Path>,
) -> Result<RestoredForecast> {
    read_forecast(ForecastArtifactKind::History, contract, input.as_ref())
}

/// Read a self-describing history only when its embedded contract has the
/// independently supplied expected SHA-256.
///
/// This is the renderer-facing door: the artifact remains self-describing,
/// while the caller must anchor admission to a receipt or another trusted
/// source instead of accepting an arbitrary contract from the same file.
pub fn read_forecast_history_embedded(
    input: impl AsRef<Path>,
    expected_contract_sha256: &str,
) -> Result<(ForecastContract, RestoredForecast)> {
    read_forecast_embedded(
        ForecastArtifactKind::History,
        input.as_ref(),
        expected_contract_sha256,
    )
}

fn read_forecast_embedded(
    kind: ForecastArtifactKind,
    input: &Path,
    expected_contract_sha256: &str,
) -> Result<(ForecastContract, RestoredForecast)> {
    validate_sha256(
        expected_contract_sha256,
        "expected forecast contract SHA-256",
    )?;
    let bytes = read_preflighted(input)?;
    let contract = read_embedded_contract_bytes(kind, input, &bytes, expected_contract_sha256)?;
    let restored = read_forecast_bytes(kind, &contract, input, bytes)?;
    Ok((contract, restored))
}

fn inspect_forecast_embedded(
    kind: ForecastArtifactKind,
    input: &Path,
    expected_contract_sha256: &str,
) -> Result<(ForecastContract, InspectedForecast)> {
    validate_sha256(
        expected_contract_sha256,
        "expected forecast contract SHA-256",
    )?;
    let bytes = read_preflighted(input)?;
    let contract = read_embedded_contract_bytes(kind, input, &bytes, expected_contract_sha256)?;
    let decoded = decode_forecast_bytes(kind, &contract, input, bytes, ArrayDecode::Inspect)?;
    let inspected = InspectedForecast {
        kind,
        clock: decoded.clock,
        receipt: decoded.receipt,
        array_payload_sha256: decoded.array_payload_sha256,
    };
    Ok((contract, inspected))
}

fn write_new(
    kind: ForecastArtifactKind,
    contract: &ForecastContract,
    frame: &ForecastFrame,
    output: &Path,
) -> Result<ForecastReceipt> {
    let unsealed = encode_forecast_unsealed(kind, contract, frame)?;
    write_unsealed(unsealed, output)
}

/// [`write_forecast_checkpoint_new`], consuming the frame.
///
/// The published bytes and receipt are identical.  Each carrier's float32
/// words are released as soon as they are copied into the payload, so peak
/// host memory is one payload plus the largest carrier instead of two whole
/// payloads -- for an 8 GiB checkpoint, the difference between 8 and 16 GiB
/// resident.
pub fn write_forecast_checkpoint_new_owned(
    contract: &ForecastContract,
    frame: ForecastFrame,
    output: impl AsRef<Path>,
) -> Result<ForecastReceipt> {
    let unsealed =
        encode_forecast_unsealed_owned(ForecastArtifactKind::Checkpoint, contract, frame)?;
    write_unsealed(unsealed, output.as_ref())
}

/// [`write_forecast_history_new`], consuming the frame (see
/// [`write_forecast_checkpoint_new_owned`]).
pub fn write_forecast_history_new_owned(
    contract: &ForecastContract,
    frame: ForecastFrame,
    output: impl AsRef<Path>,
) -> Result<ForecastReceipt> {
    let unsealed = encode_forecast_unsealed_owned(ForecastArtifactKind::History, contract, frame)?;
    write_unsealed(unsealed, output.as_ref())
}

/// The float32 array section of one forecast payload, filled in place.
///
/// A producer whose carriers already exist as published little-endian bytes
/// (a raw arena on disk) reads each carrier straight into its window here and
/// hands the whole section to [`write_forecast_checkpoint_new_raw`] or
/// [`write_forecast_history_new_raw`].  Nothing is decoded into float32 words
/// and re-encoded, and no second payload-sized buffer exists; the published
/// artifact is byte-identical to the one the frame writers produce from the
/// same words.
pub struct RawArrayPayload<'a> {
    contract: &'a ForecastContract,
    payload: Vec<u8>,
    ranges: Vec<std::ops::Range<usize>>,
}

impl<'a> RawArrayPayload<'a> {
    /// One zeroed window per contracted carrier, in payload order, with the
    /// contract's component-blob maxima reserved behind them.
    pub fn new(contract: &'a ForecastContract) -> Result<Self> {
        let mut ranges = Vec::with_capacity(contract.arrays.len());
        let mut arrays_nbytes = 0usize;
        for spec in &contract.arrays {
            let end = spec
                .words()?
                .checked_mul(4)
                .and_then(|nbytes| arrays_nbytes.checked_add(nbytes))
                .ok_or_else(|| invalid(format!("array {} byte count overflows", spec.name)))?;
            ranges.push(arrays_nbytes..end);
            arrays_nbytes = end;
        }
        let reserved = contract
            .component_blobs
            .iter()
            .try_fold(arrays_nbytes, |total, spec| {
                total.checked_add(spec.max_nbytes)
            })
            .ok_or_else(|| invalid("forecast payload byte count overflows"))?;
        let mut payload = Vec::new();
        payload.try_reserve_exact(reserved).map_err(|_| {
            invalid(format!(
                "publishing this frame needs a {reserved}-byte payload arena, which this host did not give back; the contracted maximum is {MAX_FORECAST_PAYLOAD_BYTES} bytes"
            ))
        })?;
        payload.resize(arrays_nbytes, 0);
        Ok(Self {
            contract,
            payload,
            ranges,
        })
    }

    /// The published-byte window of one contracted carrier, by name.
    pub fn carrier_mut(&mut self, name: &str) -> Option<&mut [u8]> {
        let index = self
            .contract
            .arrays
            .iter()
            .position(|spec| spec.name == name)?;
        let range = self.ranges[index].clone();
        Some(&mut self.payload[range])
    }
}

/// [`write_forecast_checkpoint_new`] from carriers already in published
/// byte form (see [`RawArrayPayload`]).
pub fn write_forecast_checkpoint_new_raw(
    contract: &ForecastContract,
    arrays: RawArrayPayload<'_>,
    clock: ForecastClock,
    scheduler_scalars: Vec<SchedulerScalar>,
    component_blobs: Vec<ComponentBlob>,
    output: impl AsRef<Path>,
) -> Result<ForecastReceipt> {
    let unsealed = encode_forecast_unsealed_raw(
        ForecastArtifactKind::Checkpoint,
        contract,
        arrays,
        clock,
        scheduler_scalars,
        component_blobs,
    )?;
    write_unsealed(unsealed, output.as_ref())
}

/// [`write_forecast_history_new`] from carriers already in published byte
/// form (see [`RawArrayPayload`]).
pub fn write_forecast_history_new_raw(
    contract: &ForecastContract,
    arrays: RawArrayPayload<'_>,
    clock: ForecastClock,
    scheduler_scalars: Vec<SchedulerScalar>,
    component_blobs: Vec<ComponentBlob>,
    output: impl AsRef<Path>,
) -> Result<ForecastReceipt> {
    let unsealed = encode_forecast_unsealed_raw(
        ForecastArtifactKind::History,
        contract,
        arrays,
        clock,
        scheduler_scalars,
        component_blobs,
    )?;
    write_unsealed(unsealed, output.as_ref())
}

fn encode_forecast_unsealed_raw(
    kind: ForecastArtifactKind,
    contract: &ForecastContract,
    arrays: RawArrayPayload<'_>,
    clock: ForecastClock,
    scheduler_scalars: Vec<SchedulerScalar>,
    component_blobs: Vec<ComponentBlob>,
) -> Result<UnsealedForecast> {
    if !std::ptr::eq(arrays.contract, contract) {
        return Err(invalid(
            "the raw array payload was laid out for another forecast contract; its carrier windows would be published under this contract's names and offsets",
        ));
    }
    let frame = ForecastFrame::new(clock, Vec::new(), scheduler_scalars, component_blobs);
    let tail = normalize_tail(contract, &frame)?;
    let RawArrayPayload {
        mut payload,
        ranges,
        ..
    } = arrays;
    let blobs = append_blobs(&mut payload, &tail);
    let specs = contract.arrays.iter().collect::<Vec<_>>();
    finish_encoding_parts(
        kind,
        contract,
        clock,
        &specs,
        &tail.scalars,
        &tail.blobs,
        payload,
        PayloadLayout {
            arrays: ranges,
            blobs,
        },
    )
}

/// Publish an encoded artifact while its whole-artifact SHA-256 is computed.
///
/// The digest reads the same three parts the atomic writer writes, on a
/// second thread, so the payload is hashed during the write instead of in a
/// separate pass before it.
fn write_unsealed(unsealed: UnsealedForecast, output: &Path) -> Result<ForecastReceipt> {
    let (written, artifact_sha256) = std::thread::scope(|scope| {
        let hashing = scope.spawn(|| unsealed.artifact_sha256());
        let written = atomic_write_new_parts(output, &unsealed.parts());
        (written, hashing.join().expect("artifact SHA-256 thread"))
    });
    written?;
    Ok(unsealed.into_receipt(artifact_sha256))
}

fn normalize<'a>(
    contract: &'a ForecastContract,
    frame: &'a ForecastFrame,
) -> Result<NormalizedFrame<'a>> {
    validate_inventory(
        contract.arrays.iter().map(|spec| spec.name.as_str()),
        frame.arrays.iter().map(|array| array.name.as_str()),
        "float32 array carriers",
    )?;
    let mut normalized = normalize_tail(contract, frame)?;

    let mut arrays = Vec::with_capacity(contract.arrays.len());
    for spec in &contract.arrays {
        let array = frame
            .arrays
            .iter()
            .find(|array| array.name == spec.name)
            .expect("inventory validation found the required array");
        let expected_words = spec.words()?;
        if array.values.len() != expected_words {
            return Err(invalid(format!(
                "array carrier {} has {} words; contract shape {:?} requires {expected_words}",
                spec.name,
                array.values.len(),
                spec.shape
            )));
        }
        arrays.push((spec, array));
    }
    normalized.arrays = arrays;
    Ok(normalized)
}

/// The scheduler scalars and component blobs of `frame` in contract order,
/// validated; the arrays are left empty for the caller.
fn normalize_tail<'a>(
    contract: &'a ForecastContract,
    frame: &'a ForecastFrame,
) -> Result<NormalizedFrame<'a>> {
    validate_inventory(
        contract
            .scheduler_scalars
            .iter()
            .map(|spec| spec.name.as_str()),
        frame.scalars_names(),
        "scheduler scalars",
    )?;
    validate_inventory(
        contract
            .component_blobs
            .iter()
            .map(|spec| spec.name.as_str()),
        frame.component_blobs.iter().map(|blob| blob.name.as_str()),
        "component blobs",
    )?;

    let mut scalars = Vec::with_capacity(contract.scheduler_scalars.len());
    for spec in &contract.scheduler_scalars {
        let scalar = frame
            .scheduler_scalars
            .iter()
            .find(|scalar| scalar.name == spec.name)
            .expect("inventory validation found the required scalar");
        if scalar.value.scalar_type() != spec.scalar_type {
            return Err(invalid(format!(
                "scheduler scalar {} has type {}; contract requires {}",
                spec.name,
                scalar.value.scalar_type().as_str(),
                spec.scalar_type.as_str()
            )));
        }
        scalars.push((spec, scalar));
    }
    let mut blobs = Vec::with_capacity(contract.component_blobs.len());
    for spec in &contract.component_blobs {
        let blob = frame
            .component_blobs
            .iter()
            .find(|blob| blob.name == spec.name)
            .expect("inventory validation found the required component blob");
        if blob.schema != spec.schema {
            return Err(invalid(format!(
                "component blob {} schema is {:?}; contract requires {:?}",
                spec.name, blob.schema, spec.schema
            )));
        }
        if blob.bytes.len() > spec.max_nbytes {
            return Err(invalid(format!(
                "component blob {} has {} bytes; contract maximum is {}",
                spec.name,
                blob.bytes.len(),
                spec.max_nbytes
            )));
        }
        blobs.push((spec, blob));
    }
    Ok(NormalizedFrame {
        clock: frame.clock,
        arrays: Vec::new(),
        scalars,
        blobs,
    })
}

impl ForecastFrame {
    fn scalars_names(&self) -> impl Iterator<Item = &str> {
        self.scheduler_scalars
            .iter()
            .map(|scalar| scalar.name.as_str())
    }
}

#[cfg(test)]
fn encode_forecast(
    kind: ForecastArtifactKind,
    contract: &ForecastContract,
    frame: &ForecastFrame,
) -> Result<EncodedForecast> {
    let unsealed = encode_forecast_unsealed(kind, contract, frame)?;
    let artifact_sha256 = unsealed.artifact_sha256();
    Ok(unsealed.seal_with(artifact_sha256))
}

/// An encoded artifact whose whole-artifact digest is not computed yet.
struct UnsealedForecast {
    prefix: [u8; PREFIX_BYTES],
    header: Vec<u8>,
    payload: Vec<u8>,
    receipt: ForecastReceipt,
}

impl UnsealedForecast {
    fn parts(&self) -> [&[u8]; 3] {
        [&self.prefix, &self.header, &self.payload]
    }

    /// Hash the three parts in publication order rather than a
    /// concatenation: the digest is identical, and the payload is never
    /// copied to compute it.
    fn artifact_sha256(&self) -> String {
        let started = Instant::now();
        let mut artifact_digest = sha256::Sha256::default();
        artifact_digest.update(&self.prefix);
        artifact_digest.update(&self.header);
        artifact_digest.update(&self.payload);
        let digest = sha256::hex(&artifact_digest.finalize());
        publish_timing::record(
            "hash.artifact",
            started.elapsed().as_secs_f64(),
            self.receipt.artifact_nbytes as u64,
        );
        digest
    }

    fn into_receipt(self, artifact_sha256: String) -> ForecastReceipt {
        let mut receipt = self.receipt;
        receipt.artifact_sha256 = artifact_sha256;
        receipt
    }

    #[cfg(test)]
    fn seal_with(self, artifact_sha256: String) -> EncodedForecast {
        let mut receipt = self.receipt;
        receipt.artifact_sha256 = artifact_sha256;
        EncodedForecast {
            prefix: self.prefix,
            header: self.header,
            payload: self.payload,
            receipt,
        }
    }
}

/// The payload bytes of one float32 carrier, appended in place.
///
/// On a little-endian target the in-memory words already are the published
/// little-endian bytes, so they are copied as one block; elsewhere each word
/// is converted.  Either way the payload holds `value.to_bits().to_le_bytes()`
/// for every word, in order.
fn append_words(payload: &mut Vec<u8>, values: &[f32]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: f32 has no padding and every byte of its representation is
        // initialized; the slice covers exactly `values.len() * 4` bytes of
        // the live `values` allocation, and it is only read.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        payload.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    for value in values {
        payload.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

/// Price the payload before a byte of it is built, so the arena is one exact
/// allocation instead of a doubling series.
///
/// This is not the ceiling's enforcement point and does not repeat it:
/// `ForecastContract::new` refuses any contract whose array words plus
/// component maxima exceed MAX_FORECAST_PAYLOAD_BYTES, and `normalize` pins
/// every array to its contracted word count and every blob under its
/// contracted maximum, so the total here is bounded by that already admitted
/// total.  A second refusal on this line could never fire and would tell a
/// future reader the encoder is the ceiling, which it is not.
fn reserve_payload(normalized: &NormalizedFrame<'_>) -> Result<Vec<u8>> {
    let mut payload_nbytes = 0usize;
    for (spec, array) in &normalized.arrays {
        payload_nbytes = array
            .values
            .len()
            .checked_mul(4)
            .and_then(|nbytes| payload_nbytes.checked_add(nbytes))
            .ok_or_else(|| invalid(format!("array {} byte count overflows", spec.name)))?;
    }
    for (spec, blob) in &normalized.blobs {
        payload_nbytes = payload_nbytes
            .checked_add(blob.bytes.len())
            .ok_or_else(|| invalid(format!("component {} byte count overflows", spec.name)))?;
    }
    let mut payload = Vec::new();
    payload.try_reserve_exact(payload_nbytes).map_err(|_| {
        invalid(format!(
            "publishing this frame needs a {payload_nbytes}-byte payload arena, which this host did not give back; the contracted maximum is {MAX_FORECAST_PAYLOAD_BYTES} bytes"
        ))
    })?;
    Ok(payload)
}

/// Offsets of every array and blob inside the assembled payload.
struct PayloadLayout {
    arrays: Vec<std::ops::Range<usize>>,
    blobs: Vec<std::ops::Range<usize>>,
}

fn encode_forecast_unsealed(
    kind: ForecastArtifactKind,
    contract: &ForecastContract,
    frame: &ForecastFrame,
) -> Result<UnsealedForecast> {
    let normalized = normalize(contract, frame)?;
    let mut payload = reserve_payload(&normalized)?;
    let started = Instant::now();
    let mut arrays = Vec::with_capacity(normalized.arrays.len());
    for (_, array) in &normalized.arrays {
        // Serialize straight into the arena.  A per-carrier staging buffer
        // would double the peak for no gain: the appended slice is the same
        // bytes, and hashing it in place is what the header records.
        let offset = payload.len();
        append_words(&mut payload, &array.values);
        arrays.push(offset..payload.len());
    }
    let blobs = append_blobs(&mut payload, &normalized);
    publish_timing::record(
        "encode.copy",
        started.elapsed().as_secs_f64(),
        payload.len() as u64,
    );
    finish_encoding(
        kind,
        contract,
        &normalized,
        payload,
        PayloadLayout { arrays, blobs },
    )
}

fn encode_forecast_unsealed_owned(
    kind: ForecastArtifactKind,
    contract: &ForecastContract,
    mut frame: ForecastFrame,
) -> Result<UnsealedForecast> {
    // Validate the complete frame and resolve, for each contracted array in
    // payload order, which frame slot holds it.
    let (order, mut payload) = {
        let normalized = normalize(contract, &frame)?;
        let order = normalized
            .arrays
            .iter()
            .map(|(_, array)| {
                frame
                    .arrays
                    .iter()
                    .position(|candidate| std::ptr::eq(candidate, *array))
                    .expect("normalized arrays are borrowed from this frame")
            })
            .collect::<Vec<_>>();
        (order, reserve_payload(&normalized)?)
    };
    let started = Instant::now();
    let mut arrays = Vec::with_capacity(order.len());
    for index in order {
        let values = std::mem::take(&mut frame.arrays[index].values);
        let offset = payload.len();
        append_words(&mut payload, &values);
        // Release this carrier's words before the next one is copied.
        drop(values);
        arrays.push(offset..payload.len());
    }
    publish_timing::record(
        "encode.copy",
        started.elapsed().as_secs_f64(),
        payload.len() as u64,
    );
    // The arrays are moved out; the scalars, blobs and clock are still the
    // frame's and were validated above.  Re-derive their contract order
    // against the emptied frame's own inventory.
    let tail = normalize_tail(contract, &frame)?;
    let blobs = append_blobs(&mut payload, &tail);
    let specs = contract.arrays.iter().collect::<Vec<_>>();
    finish_encoding_parts(
        kind,
        contract,
        tail.clock,
        &specs,
        &tail.scalars,
        &tail.blobs,
        payload,
        PayloadLayout { arrays, blobs },
    )
}

fn append_blobs(
    payload: &mut Vec<u8>,
    normalized: &NormalizedFrame<'_>,
) -> Vec<std::ops::Range<usize>> {
    normalized
        .blobs
        .iter()
        .map(|(_, blob)| {
            let offset = payload.len();
            payload.extend_from_slice(&blob.bytes);
            offset..payload.len()
        })
        .collect()
}

fn finish_encoding(
    kind: ForecastArtifactKind,
    contract: &ForecastContract,
    normalized: &NormalizedFrame<'_>,
    payload: Vec<u8>,
    layout: PayloadLayout,
) -> Result<UnsealedForecast> {
    let specs = normalized
        .arrays
        .iter()
        .map(|(spec, _)| *spec)
        .collect::<Vec<_>>();
    finish_encoding_parts(
        kind,
        contract,
        normalized.clock,
        &specs,
        &normalized.scalars,
        &normalized.blobs,
        payload,
        layout,
    )
}

/// SHA-256 of every array and blob range and of the whole payload, in
/// parallel.  Each digest reads exactly the bytes a serial pass would.
fn range_and_payload_digests(
    payload: &[u8],
    ranges: &[std::ops::Range<usize>],
) -> (Vec<String>, String) {
    const PARALLEL_MIN_BYTES: usize = 8 << 20;
    if payload.len() < PARALLEL_MIN_BYTES || ranges.is_empty() {
        let digests = ranges
            .iter()
            .map(|range| sha256::digest_hex(&payload[range.clone()]))
            .collect();
        return (digests, sha256::digest_hex(payload));
    }
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(2, 8)
        - 1;
    // Balance the ranges over the workers by bytes, largest first.
    let mut indices = (0..ranges.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&index| std::cmp::Reverse(ranges[index].len()));
    let mut groups: Vec<(usize, Vec<usize>)> = (0..workers.min(ranges.len()))
        .map(|_| (0usize, Vec::new()))
        .collect();
    for index in indices {
        let lightest = groups
            .iter_mut()
            .min_by_key(|(bytes, _)| *bytes)
            .expect("at least one worker group");
        lightest.0 += ranges[index].len();
        lightest.1.push(index);
    }
    std::thread::scope(|scope| {
        let whole = scope.spawn(|| sha256::digest_hex(payload));
        let handles = groups
            .into_iter()
            .map(|(_, members)| {
                scope.spawn(move || {
                    members
                        .into_iter()
                        .map(|index| (index, sha256::digest_hex(&payload[ranges[index].clone()])))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        let mut digests = vec![String::new(); ranges.len()];
        for handle in handles {
            for (index, digest) in handle.join().expect("range SHA-256 thread") {
                digests[index] = digest;
            }
        }
        (digests, whole.join().expect("payload SHA-256 thread"))
    })
}

#[allow(clippy::too_many_arguments)]
fn finish_encoding_parts(
    kind: ForecastArtifactKind,
    contract: &ForecastContract,
    clock: ForecastClock,
    array_specs: &[&ForecastArraySpec],
    scalars: &[(&SchedulerScalarSpec, &SchedulerScalar)],
    blobs: &[(&ComponentBlobSpec, &ComponentBlob)],
    payload: Vec<u8>,
    layout: PayloadLayout,
) -> Result<UnsealedForecast> {
    let contract_value = contract_json(contract);
    let contract_sha256 = sha256::digest_hex(&contract_value.canonical_bytes());
    let mut ranges = layout.arrays.clone();
    ranges.extend(layout.blobs.iter().cloned());
    let started = Instant::now();
    let (digests, payload_sha256) = range_and_payload_digests(&payload, &ranges);
    publish_timing::record(
        "hash.carriers_and_payload",
        started.elapsed().as_secs_f64(),
        2 * payload.len() as u64,
    );
    let mut array_headers = Vec::with_capacity(array_specs.len());
    let mut component_headers = Vec::with_capacity(blobs.len());
    let mut payload_order = Vec::with_capacity(array_specs.len() + blobs.len());
    for ((spec, range), digest) in array_specs.iter().zip(&layout.arrays).zip(&digests) {
        array_headers.push(object(vec![
            ("name", string(&spec.name)),
            ("nbytes", number(range.len())),
            ("offset", number(range.start)),
            ("sha256", string(digest)),
        ]));
        payload_order.push(string(&format!("array:{}", spec.name)));
    }
    for (((spec, _), range), digest) in blobs
        .iter()
        .zip(&layout.blobs)
        .zip(&digests[layout.arrays.len()..])
    {
        component_headers.push(object(vec![
            ("name", string(&spec.name)),
            ("nbytes", number(range.len())),
            ("offset", number(range.start)),
            ("schema", string(&spec.schema)),
            ("sha256", string(digest)),
        ]));
        payload_order.push(string(&format!("component:{}", spec.name)));
    }
    let scalar_headers = scalars
        .iter()
        .map(|(spec, scalar)| scalar_json(spec, scalar.value))
        .collect();
    let header_value = object(vec![
        ("arrays", JsonValue::Array(array_headers)),
        ("authority_sha256", string(AUTHORITY_SHA256)),
        ("capability", string(kind.capability())),
        ("claim_limit", string(kind.claim_limit())),
        ("clock", clock_json(clock)),
        ("component_blobs", JsonValue::Array(component_headers)),
        ("contract", contract_value),
        ("contract_sha256", string(&contract_sha256)),
        ("live_state_wiring", string(LIVE_STATE_WIRING)),
        ("payload_nbytes", number(payload.len())),
        ("payload_order", JsonValue::Array(payload_order)),
        ("payload_sha256", string(&payload_sha256)),
        ("representation", string(REPRESENTATION)),
        ("scheduler_scalars", JsonValue::Array(scalar_headers)),
        ("schema", string(kind.schema())),
    ]);
    let header = header_value.canonical_bytes();
    if header.len() > MAX_FORECAST_HEADER_BYTES {
        return Err(invalid(format!(
            "forecast header has {} bytes; maximum is {MAX_FORECAST_HEADER_BYTES}",
            header.len()
        )));
    }
    let mut prefix = [0u8; PREFIX_BYTES];
    prefix[..8].copy_from_slice(&kind.magic());
    prefix[8..].copy_from_slice(&(header.len() as u64).to_le_bytes());
    let artifact_nbytes = PREFIX_BYTES
        .checked_add(header.len())
        .and_then(|value| value.checked_add(payload.len()))
        .ok_or_else(|| invalid("forecast artifact byte count overflows"))?;
    let receipt = ForecastReceipt {
        kind,
        contract_sha256,
        artifact_sha256: String::new(),
        header_sha256: sha256::digest_hex(&header),
        payload_sha256,
        artifact_nbytes,
        payload_nbytes: payload.len(),
        arrays: array_specs.len(),
        scheduler_scalars: scalars.len(),
        component_blobs: blobs.len(),
        step: clock.step,
        model_time_f64_bits: f64_bits(clock.model_time_s),
        dt_f64_bits: f64_bits(clock.dt_s),
    };
    Ok(UnsealedForecast {
        prefix,
        header,
        payload,
        receipt,
    })
}

fn read_forecast(
    kind: ForecastArtifactKind,
    contract: &ForecastContract,
    input: &Path,
) -> Result<RestoredForecast> {
    let bytes = read_preflighted(input)?;
    read_forecast_bytes(kind, contract, input, bytes)
}

fn read_forecast_bytes(
    kind: ForecastArtifactKind,
    contract: &ForecastContract,
    input: &Path,
    bytes: Vec<u8>,
) -> Result<RestoredForecast> {
    let decoded = decode_forecast_bytes(kind, contract, input, bytes, ArrayDecode::Materialize)?;
    let frame = ForecastFrame::new(
        decoded.clock,
        decoded
            .arrays
            .expect("materialized forecast decoding returns arrays"),
        decoded.scalars,
        decoded.blobs,
    );
    normalize(contract, &frame)?;
    Ok(RestoredForecast {
        kind,
        frame,
        receipt: decoded.receipt,
        array_payload_sha256: decoded.array_payload_sha256,
    })
}

fn decode_forecast_bytes(
    kind: ForecastArtifactKind,
    contract: &ForecastContract,
    input: &Path,
    bytes: Vec<u8>,
    array_decode: ArrayDecode,
) -> Result<DecodedForecast> {
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
        .map_err(|_| invalid("forecast header length does not fit this platform"))?;
    if header_len > MAX_FORECAST_HEADER_BYTES {
        return Err(invalid(format!(
            "forecast header length {header_len} exceeds {MAX_FORECAST_HEADER_BYTES}"
        )));
    }
    let payload_start = PREFIX_BYTES
        .checked_add(header_len)
        .ok_or_else(|| invalid("forecast header offset overflow"))?;
    if payload_start > bytes.len() {
        return Err(invalid("forecast header length extends past the artifact"));
    }
    let header = &bytes[PREFIX_BYTES..payload_start];
    let parsed = json::parse(header)
        .map_err(|error| invalid(format!("forecast header JSON is invalid: {error}")))?;
    if parsed.canonical_bytes() != header {
        return Err(invalid(
            "forecast header is not canonical sorted-key compact UTF-8 JSON",
        ));
    }
    exact_keys(&parsed, TOP_KEYS, "forecast header")?;
    fixed_string(&parsed, "schema", kind.schema(), "forecast header")?;
    fixed_string(
        &parsed,
        "authority_sha256",
        AUTHORITY_SHA256,
        "forecast header",
    )?;
    fixed_string(&parsed, "capability", kind.capability(), "forecast header")?;
    fixed_string(
        &parsed,
        "claim_limit",
        kind.claim_limit(),
        "forecast header",
    )?;
    fixed_string(
        &parsed,
        "live_state_wiring",
        LIVE_STATE_WIRING,
        "forecast header",
    )?;
    fixed_string(&parsed, "representation", REPRESENTATION, "forecast header")?;

    let expected_contract = contract_json(contract);
    let actual_contract = parsed
        .get("contract")
        .map_err(|error| invalid(format!("forecast header.contract: {error}")))?;
    validate_contract_object(actual_contract)?;
    if actual_contract.canonical_bytes() != expected_contract.canonical_bytes() {
        return Err(invalid(format!(
            "forecast artifact contract does not match required contract {}",
            contract.sha256()
        )));
    }
    let contract_sha256 = contract.sha256();
    fixed_string(
        &parsed,
        "contract_sha256",
        &contract_sha256,
        "forecast header",
    )?;
    let clock = parse_clock(
        parsed
            .get("clock")
            .map_err(|error| invalid(format!("forecast header.clock: {error}")))?,
    )?;
    let scalars = parse_scalars(
        parsed
            .get("scheduler_scalars")
            .map_err(|error| invalid(format!("forecast header.scheduler_scalars: {error}")))?,
        contract,
    )?;
    let payload = &bytes[payload_start..];
    let declared_payload = required_usize(&parsed, "payload_nbytes", "forecast header")?;
    if declared_payload != payload.len() {
        return Err(invalid(format!(
            "forecast payload has {} bytes; header declares {declared_payload}",
            payload.len()
        )));
    }
    let expected_array_nbytes = contract.arrays.iter().try_fold(0usize, |total, spec| {
        let nbytes = spec
            .words()?
            .checked_mul(4)
            .ok_or_else(|| invalid(format!("array {} byte count overflows", spec.name)))?;
        total
            .checked_add(nbytes)
            .ok_or_else(|| invalid("forecast array byte total overflows"))
    })?;
    let array_payload = payload
        .get(..expected_array_nbytes)
        .ok_or_else(|| invalid("forecast array payload extends past the artifact payload"))?;
    let (artifact_sha256, payload_sha256, array_payload_sha256, decoded) = thread::scope(
        |scope| -> Result<_> {
            let artifact_worker = scope.spawn(|| sha256::digest_hex(&bytes));
            let payload_worker = scope.spawn(|| sha256::digest_hex(payload));
            let array_worker = scope.spawn(|| sha256::digest_hex(array_payload));

            let decoded = (|| -> Result<_> {
                let array_headers = parsed
                    .get("arrays")
                    .map_err(|error| invalid(format!("forecast header.arrays: {error}")))?;
                let (arrays, array_end, mut expected_order) = match array_decode {
                    ArrayDecode::Materialize => {
                        let (arrays, end, order) = parse_arrays(array_headers, contract, payload)?;
                        (Some(arrays), end, order)
                    }
                    ArrayDecode::Inspect => {
                        let (end, order) = validate_arrays(array_headers, contract, payload)?;
                        (None, end, order)
                    }
                };
                if array_end != expected_array_nbytes {
                    return Err(invalid(format!(
                        "validated forecast array payload has {array_end} bytes; contract requires {expected_array_nbytes}"
                    )));
                }
                let (blobs, final_offset, component_order) = parse_components(
                    parsed.get("component_blobs").map_err(|error| {
                        invalid(format!("forecast header.component_blobs: {error}"))
                    })?,
                    contract,
                    payload,
                    array_end,
                )?;
                expected_order.extend(component_order);
                if final_offset != payload.len() {
                    return Err(invalid(format!(
                        "forecast payload has {} trailing bytes after the contracted arrays and component blobs",
                        payload.len() - final_offset
                    )));
                }
                let actual_order = parsed
                    .get("payload_order")
                    .and_then(JsonValue::as_array)
                    .map_err(|error| invalid(format!("forecast header.payload_order: {error}")))?;
                let actual_order: Vec<&str> = actual_order
                    .iter()
                    .map(JsonValue::as_str)
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|error| invalid(format!("forecast header.payload_order: {error}")))?;
                let expected_refs: Vec<&str> = expected_order.iter().map(String::as_str).collect();
                if actual_order != expected_refs {
                    return Err(invalid(format!(
                        "forecast payload order is {actual_order:?}; expected {expected_refs:?}"
                    )));
                }
                Ok((arrays, blobs))
            })();

            let artifact_sha256 = artifact_worker
                .join()
                .map_err(|_| invalid("forecast artifact SHA-256 worker panicked"))?;
            let payload_sha256 = payload_worker
                .join()
                .map_err(|_| invalid("forecast payload SHA-256 worker panicked"))?;
            let array_payload_sha256 = array_worker
                .join()
                .map_err(|_| invalid("forecast array SHA-256 worker panicked"))?;
            Ok((
                artifact_sha256,
                payload_sha256,
                array_payload_sha256,
                decoded,
            ))
        },
    )?;
    fixed_string(
        &parsed,
        "payload_sha256",
        &payload_sha256,
        "forecast header",
    )?;
    let (arrays, blobs) = decoded?;
    let receipt = ForecastReceipt {
        kind,
        contract_sha256,
        artifact_sha256,
        header_sha256: sha256::digest_hex(header),
        payload_sha256,
        artifact_nbytes: bytes.len(),
        payload_nbytes: payload.len(),
        arrays: contract.arrays.len(),
        scheduler_scalars: scalars.len(),
        component_blobs: blobs.len(),
        step: clock.step,
        model_time_f64_bits: f64_bits(clock.model_time_s),
        dt_f64_bits: f64_bits(clock.dt_s),
    };
    Ok(DecodedForecast {
        arrays,
        array_payload_sha256,
        blobs,
        clock,
        receipt,
        scalars,
    })
}

fn read_embedded_contract_bytes(
    kind: ForecastArtifactKind,
    input: &Path,
    bytes: &[u8],
    expected_contract_sha256: &str,
) -> Result<ForecastContract> {
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
        .map_err(|_| invalid("forecast header length does not fit this platform"))?;
    if header_len > MAX_FORECAST_HEADER_BYTES {
        return Err(invalid(format!(
            "forecast header length {header_len} exceeds {MAX_FORECAST_HEADER_BYTES}"
        )));
    }
    let payload_start = PREFIX_BYTES
        .checked_add(header_len)
        .ok_or_else(|| invalid("forecast header offset overflow"))?;
    if payload_start > bytes.len() {
        return Err(invalid("forecast header length extends past the artifact"));
    }
    let header = &bytes[PREFIX_BYTES..payload_start];
    let parsed = json::parse(header)
        .map_err(|error| invalid(format!("forecast header JSON is invalid: {error}")))?;
    if parsed.canonical_bytes() != header {
        return Err(invalid(
            "forecast header is not canonical sorted-key compact UTF-8 JSON",
        ));
    }
    exact_keys(&parsed, TOP_KEYS, "forecast header")?;
    fixed_string(&parsed, "schema", kind.schema(), "forecast header")?;
    fixed_string(
        &parsed,
        "authority_sha256",
        AUTHORITY_SHA256,
        "forecast header",
    )?;
    fixed_string(&parsed, "capability", kind.capability(), "forecast header")?;
    fixed_string(
        &parsed,
        "claim_limit",
        kind.claim_limit(),
        "forecast header",
    )?;
    fixed_string(
        &parsed,
        "live_state_wiring",
        LIVE_STATE_WIRING,
        "forecast header",
    )?;
    fixed_string(&parsed, "representation", REPRESENTATION, "forecast header")?;

    let contract_value = parsed
        .get("contract")
        .map_err(|error| invalid(format!("forecast header.contract: {error}")))?;
    validate_contract_object(contract_value)?;
    let embedded_sha256 = sha256::digest_hex(&contract_value.canonical_bytes());
    fixed_string(
        &parsed,
        "contract_sha256",
        &embedded_sha256,
        "forecast header",
    )?;
    if embedded_sha256 != expected_contract_sha256 {
        return Err(invalid(format!(
            "forecast artifact contract SHA-256 is {embedded_sha256}; independently expected {expected_contract_sha256}"
        )));
    }
    let contract = parse_contract_object(contract_value)?;
    if contract.sha256() != embedded_sha256 {
        return Err(invalid(
            "parsed forecast contract does not reproduce its embedded canonical SHA-256",
        ));
    }
    Ok(contract)
}

fn parse_contract_object(value: &JsonValue) -> Result<ForecastContract> {
    validate_contract_object(value)?;
    let required_string = |key: &str| -> Result<String> {
        value
            .get(key)
            .and_then(JsonValue::as_str)
            .map(str::to_owned)
            .map_err(|error| invalid(format!("forecast contract.{key}: {error}")))
    };

    let mut arrays = Vec::new();
    for entry in value
        .get("array_specs")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("forecast contract.array_specs: {error}")))?
    {
        let field_string = |key: &str| -> Result<String> {
            entry
                .get(key)
                .and_then(JsonValue::as_str)
                .map(str::to_owned)
                .map_err(|error| invalid(format!("forecast array contract.{key}: {error}")))
        };
        let bounds = entry
            .get("bounds")
            .and_then(JsonValue::as_array)
            .map_err(|error| invalid(format!("forecast array contract.bounds: {error}")))?
            .iter()
            .map(|item| {
                let value = item
                    .as_i64()
                    .map_err(|error| invalid(format!("forecast array bound: {error}")))?;
                i32::try_from(value)
                    .map_err(|_| invalid(format!("forecast array bound {value} does not fit i32")))
            })
            .collect::<Result<Vec<_>>>()?;
        let shape = entry
            .get("shape")
            .and_then(JsonValue::as_array)
            .map_err(|error| invalid(format!("forecast array contract.shape: {error}")))?
            .iter()
            .map(|item| {
                let value = item
                    .as_u64()
                    .map_err(|error| invalid(format!("forecast array shape: {error}")))?;
                usize::try_from(value).map_err(|_| {
                    invalid(format!("forecast array extent {value} does not fit usize"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        arrays.push(ForecastArraySpec::new(
            field_string("name")?,
            field_string("meaning")?,
            field_string("units")?,
            field_string("stagger")?,
            bounds,
            shape,
        )?);
    }

    let mut scalars = Vec::new();
    for entry in value
        .get("scheduler_scalar_specs")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("forecast contract.scheduler_scalar_specs: {error}")))?
    {
        let name = entry
            .get("name")
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("forecast scalar contract.name: {error}")))?;
        let raw_type = entry
            .get("type")
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("forecast scalar contract.type: {error}")))?;
        let scalar_type = match raw_type {
            "i64" => SchedulerScalarType::I64,
            "u64" => SchedulerScalarType::U64,
            "f64-bits" => SchedulerScalarType::F64,
            "bool" => SchedulerScalarType::Bool,
            other => {
                return Err(invalid(format!(
                    "forecast scalar contract type {other:?} is unknown"
                )))
            }
        };
        scalars.push(SchedulerScalarSpec::new(name, scalar_type)?);
    }

    let mut components = Vec::new();
    for entry in value
        .get("component_blob_specs")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("forecast contract.component_blob_specs: {error}")))?
    {
        let name = entry
            .get("name")
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("forecast component contract.name: {error}")))?;
        let schema = entry
            .get("schema")
            .and_then(JsonValue::as_str)
            .map_err(|error| invalid(format!("forecast component contract.schema: {error}")))?;
        let max_nbytes = required_usize(entry, "max_nbytes", "forecast component contract")?;
        components.push(ComponentBlobSpec::new(name, schema, max_nbytes)?);
    }

    ForecastContract::new(
        required_string("profile_id")?,
        required_string("config_sha256")?,
        required_string("execution_identity_sha256")?,
        arrays,
        scalars,
        components,
    )
}

fn parse_arrays(
    value: &JsonValue,
    contract: &ForecastContract,
    payload: &[u8],
) -> Result<(Vec<ForecastArray>, usize, Vec<String>)> {
    let mut arrays = Vec::with_capacity(contract.arrays.len());
    let (offset, order) = walk_arrays(value, contract, payload, |spec, bytes| {
        let values = bytes
            .chunks_exact(4)
            .map(|word| {
                f32::from_bits(u32::from_le_bytes(
                    word.try_into().expect("four-byte chunk"),
                ))
            })
            .collect();
        arrays.push(ForecastArray::new(spec.name.clone(), values));
        Ok(())
    })?;
    Ok((arrays, offset, order))
}

fn validate_arrays(
    value: &JsonValue,
    contract: &ForecastContract,
    payload: &[u8],
) -> Result<(usize, Vec<String>)> {
    walk_arrays(value, contract, payload, |_spec, _bytes| Ok(()))
}

fn walk_arrays<F>(
    value: &JsonValue,
    contract: &ForecastContract,
    payload: &[u8],
    mut visit: F,
) -> Result<(usize, Vec<String>)>
where
    F: FnMut(&ForecastArraySpec, &[u8]) -> Result<()>,
{
    let entries = value
        .as_array()
        .map_err(|error| invalid(format!("forecast arrays: {error}")))?;
    if entries.len() != contract.arrays.len() {
        return Err(invalid(format!(
            "forecast artifact has {} array headers; contract requires {}",
            entries.len(),
            contract.arrays.len()
        )));
    }
    let mut offset = 0usize;
    let mut order = Vec::with_capacity(entries.len());
    for (entry, spec) in entries.iter().zip(&contract.arrays) {
        exact_keys(entry, ARRAY_HEADER_KEYS, "forecast array header")?;
        fixed_string(entry, "name", &spec.name, "forecast array header")?;
        let declared_offset = required_usize(entry, "offset", "forecast array header")?;
        if declared_offset != offset {
            return Err(invalid(format!(
                "forecast array {} offset is {declared_offset}; expected contiguous {offset}",
                spec.name
            )));
        }
        let nbytes = spec
            .words()?
            .checked_mul(4)
            .ok_or_else(|| invalid(format!("array {} byte count overflows", spec.name)))?;
        fixed_usize(entry, "nbytes", nbytes, "forecast array header")?;
        let end = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("forecast array payload offset overflow"))?;
        let bytes = payload.get(offset..end).ok_or_else(|| {
            invalid(format!(
                "forecast array {} extends past the payload",
                spec.name
            ))
        })?;
        fixed_string(
            entry,
            "sha256",
            &sha256::digest_hex(bytes),
            "forecast array header",
        )?;
        visit(spec, bytes)?;
        order.push(format!("array:{}", spec.name));
        offset = end;
    }
    Ok((offset, order))
}

fn parse_components(
    value: &JsonValue,
    contract: &ForecastContract,
    payload: &[u8],
    mut offset: usize,
) -> Result<(Vec<ComponentBlob>, usize, Vec<String>)> {
    let entries = value
        .as_array()
        .map_err(|error| invalid(format!("forecast component blobs: {error}")))?;
    if entries.len() != contract.component_blobs.len() {
        return Err(invalid(format!(
            "forecast artifact has {} component blob headers; contract requires {}",
            entries.len(),
            contract.component_blobs.len()
        )));
    }
    let mut blobs = Vec::with_capacity(entries.len());
    let mut order = Vec::with_capacity(entries.len());
    for (entry, spec) in entries.iter().zip(&contract.component_blobs) {
        exact_keys(entry, COMPONENT_HEADER_KEYS, "forecast component header")?;
        fixed_string(entry, "name", &spec.name, "forecast component header")?;
        fixed_string(entry, "schema", &spec.schema, "forecast component header")?;
        let declared_offset = required_usize(entry, "offset", "forecast component header")?;
        if declared_offset != offset {
            return Err(invalid(format!(
                "forecast component {} offset is {declared_offset}; expected contiguous {offset}",
                spec.name
            )));
        }
        let nbytes = required_usize(entry, "nbytes", "forecast component header")?;
        if nbytes > spec.max_nbytes {
            return Err(invalid(format!(
                "forecast component {} has {nbytes} bytes; contract maximum is {}",
                spec.name, spec.max_nbytes
            )));
        }
        let end = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("forecast component payload offset overflow"))?;
        let bytes = payload.get(offset..end).ok_or_else(|| {
            invalid(format!(
                "forecast component {} extends past the payload",
                spec.name
            ))
        })?;
        fixed_string(
            entry,
            "sha256",
            &sha256::digest_hex(bytes),
            "forecast component header",
        )?;
        blobs.push(ComponentBlob::new(
            spec.name.clone(),
            spec.schema.clone(),
            bytes.to_vec(),
        ));
        order.push(format!("component:{}", spec.name));
        offset = end;
    }
    Ok((blobs, offset, order))
}

fn parse_scalars(value: &JsonValue, contract: &ForecastContract) -> Result<Vec<SchedulerScalar>> {
    let entries = value
        .as_array()
        .map_err(|error| invalid(format!("forecast scheduler scalars: {error}")))?;
    if entries.len() != contract.scheduler_scalars.len() {
        return Err(invalid(format!(
            "forecast artifact has {} scheduler scalars; contract requires {}",
            entries.len(),
            contract.scheduler_scalars.len()
        )));
    }
    let mut scalars = Vec::with_capacity(entries.len());
    for (entry, spec) in entries.iter().zip(&contract.scheduler_scalars) {
        exact_keys(entry, SCALAR_HEADER_KEYS, "forecast scheduler scalar")?;
        fixed_string(entry, "name", &spec.name, "forecast scheduler scalar")?;
        fixed_string(
            entry,
            "type",
            spec.scalar_type.as_str(),
            "forecast scheduler scalar",
        )?;
        let value = match spec.scalar_type {
            SchedulerScalarType::I64 => SchedulerScalarValue::I64(
                entry
                    .get("value")
                    .and_then(JsonValue::as_i64)
                    .map_err(|error| invalid(format!("scalar {}: {error}", spec.name)))?,
            ),
            SchedulerScalarType::U64 => SchedulerScalarValue::U64(
                entry
                    .get("value")
                    .and_then(JsonValue::as_u64)
                    .map_err(|error| invalid(format!("scalar {}: {error}", spec.name)))?,
            ),
            SchedulerScalarType::F64 => {
                let bits = entry
                    .get("value")
                    .and_then(JsonValue::as_str)
                    .map_err(|error| invalid(format!("scalar {}: {error}", spec.name)))?;
                SchedulerScalarValue::F64(parse_f64_bits(bits, &format!("scalar {}", spec.name))?)
            }
            SchedulerScalarType::Bool => match entry
                .get("value")
                .map_err(|error| invalid(format!("scalar {}: {error}", spec.name)))?
            {
                JsonValue::Bool(value) => SchedulerScalarValue::Bool(*value),
                _ => {
                    return Err(invalid(format!(
                        "scalar {} value is not a JSON boolean",
                        spec.name
                    )))
                }
            },
        };
        scalars.push(SchedulerScalar::new(spec.name.clone(), value)?);
    }
    Ok(scalars)
}

fn parse_clock(value: &JsonValue) -> Result<ForecastClock> {
    exact_keys(value, CLOCK_KEYS, "forecast clock")?;
    let step = value
        .get("step")
        .and_then(JsonValue::as_u64)
        .map_err(|error| invalid(format!("forecast clock.step: {error}")))?;
    let model_time_bits = value
        .get("model_time_f64_bits")
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("forecast clock.model_time_f64_bits: {error}")))?;
    let dt_bits = value
        .get("dt_f64_bits")
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("forecast clock.dt_f64_bits: {error}")))?;
    ForecastClock::new(
        step,
        parse_f64_bits(model_time_bits, "forecast model time")?,
        parse_f64_bits(dt_bits, "forecast dt")?,
    )
}

fn contract_json(contract: &ForecastContract) -> JsonValue {
    let arrays = contract
        .arrays
        .iter()
        .map(|spec| {
            object(vec![
                (
                    "bounds",
                    JsonValue::Array(spec.bounds.iter().map(|value| i32_number(*value)).collect()),
                ),
                ("meaning", string(&spec.meaning)),
                ("name", string(&spec.name)),
                (
                    "shape",
                    JsonValue::Array(spec.shape.iter().map(|value| number(*value)).collect()),
                ),
                ("stagger", string(&spec.stagger)),
                ("units", string(&spec.units)),
            ])
        })
        .collect();
    let scalars = contract
        .scheduler_scalars
        .iter()
        .map(|spec| {
            object(vec![
                ("name", string(&spec.name)),
                ("type", string(spec.scalar_type.as_str())),
            ])
        })
        .collect();
    let components = contract
        .component_blobs
        .iter()
        .map(|spec| {
            object(vec![
                ("max_nbytes", number(spec.max_nbytes)),
                ("name", string(&spec.name)),
                ("schema", string(&spec.schema)),
            ])
        })
        .collect();
    object(vec![
        ("array_specs", JsonValue::Array(arrays)),
        ("component_blob_specs", JsonValue::Array(components)),
        ("config_sha256", string(&contract.config_sha256)),
        (
            "execution_identity_sha256",
            string(&contract.execution_identity_sha256),
        ),
        ("profile_id", string(&contract.profile_id)),
        ("scheduler_scalar_specs", JsonValue::Array(scalars)),
    ])
}

fn validate_contract_object(value: &JsonValue) -> Result<()> {
    exact_keys(value, CONTRACT_KEYS, "forecast contract")?;
    for entry in value
        .get("array_specs")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("forecast contract.array_specs: {error}")))?
    {
        exact_keys(entry, ARRAY_SPEC_KEYS, "forecast array contract")?;
    }
    for entry in value
        .get("scheduler_scalar_specs")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("forecast contract.scheduler_scalar_specs: {error}")))?
    {
        exact_keys(entry, SCALAR_SPEC_KEYS, "forecast scalar contract")?;
    }
    for entry in value
        .get("component_blob_specs")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("forecast contract.component_blob_specs: {error}")))?
    {
        exact_keys(entry, COMPONENT_SPEC_KEYS, "forecast component contract")?;
    }
    Ok(())
}

fn scalar_json(spec: &SchedulerScalarSpec, value: SchedulerScalarValue) -> JsonValue {
    let encoded = match value {
        SchedulerScalarValue::I64(value) => JsonValue::Number(value.to_string()),
        SchedulerScalarValue::U64(value) => JsonValue::Number(value.to_string()),
        SchedulerScalarValue::F64(value) => string(&f64_bits(value)),
        SchedulerScalarValue::Bool(value) => JsonValue::Bool(value),
    };
    object(vec![
        ("name", string(&spec.name)),
        ("type", string(spec.scalar_type.as_str())),
        ("value", encoded),
    ])
}

fn clock_json(clock: ForecastClock) -> JsonValue {
    object(vec![
        ("dt_f64_bits", string(&f64_bits(clock.dt_s))),
        ("model_time_f64_bits", string(&f64_bits(clock.model_time_s))),
        ("step", u64_number(clock.step)),
    ])
}

fn f64_bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn parse_f64_bits(value: &str, label: &str) -> Result<f64> {
    if value.len() != 16
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(invalid(format!(
            "{label} f64 bits must be exactly 16 lowercase hexadecimal characters"
        )));
    }
    let bits = u64::from_str_radix(value, 16)
        .map_err(|error| invalid(format!("{label} f64 bits are invalid: {error}")))?;
    Ok(f64::from_bits(bits))
}

/// Exact payload byte count an artifact of `artifact_nbytes` carries under a
/// header of `header_nbytes`.
///
/// The artifact is the sixteen-byte prefix, then the header, then the
/// payload.  A declared header that does not fit inside the artifact leaves
/// no payload; the decoder refuses that shape by name immediately afterwards,
/// so this saturates rather than inventing a length.
fn forecast_payload_nbytes(artifact_nbytes: usize, header_nbytes: usize) -> usize {
    artifact_nbytes
        .saturating_sub(PREFIX_BYTES)
        .saturating_sub(header_nbytes)
}

/// Peak host bytes admitting the artifact open as `file` will cost.
///
/// Reads only the sixteen-byte prefix, rewinds, and prices the payload it
/// declares with [`forecast_admission_host_bytes`].  This is where the
/// payload ceiling becomes a read gate: a declared payload past it is refused
/// by name here, before any buffer for it exists.
fn forecast_read_admission_host_bytes(
    artifact_nbytes: usize,
    file: &mut File,
    path: &Path,
) -> Result<usize> {
    if artifact_nbytes < PREFIX_BYTES {
        // Too short to declare a header; the decoder refuses it by name and
        // the whole file is smaller than one prefix.
        return forecast_admission_host_bytes(0);
    }
    let mut prefix = [0u8; PREFIX_BYTES];
    file.read_exact(&mut prefix).map_err(|error| {
        invalid(format!(
            "cannot read the forecast prefix of {}: {error}",
            path.display()
        ))
    })?;
    file.rewind().map_err(|error| {
        invalid(format!(
            "cannot rewind forecast artifact {} after its prefix: {error}",
            path.display()
        ))
    })?;
    let header_nbytes = u64::from_le_bytes(prefix[8..16].try_into().expect("fixed prefix"));
    let header_nbytes = usize::try_from(header_nbytes).unwrap_or(usize::MAX);
    forecast_admission_host_bytes(forecast_payload_nbytes(artifact_nbytes, header_nbytes))
}

fn read_preflighted(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "cannot inspect forecast artifact {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.file_type().is_file() {
        return Err(invalid(format!(
            "forecast artifact {} is not a regular non-symlink file",
            path.display()
        )));
    }
    let declared = usize::try_from(metadata.len())
        .map_err(|_| invalid("forecast artifact length does not fit this platform"))?;
    if declared > MAX_FORECAST_ENVELOPE_BYTES {
        return Err(invalid(format!(
            "forecast artifact has {declared} bytes; maximum is {MAX_FORECAST_ENVELOPE_BYTES}"
        )));
    }
    let mut file = File::open(path).map_err(|error| {
        invalid(format!(
            "cannot open forecast artifact {}: {error}",
            path.display()
        ))
    })?;
    let opened = usize::try_from(
        file.metadata()
            .map_err(|error| invalid(format!("cannot inspect opened forecast artifact: {error}")))?
            .len(),
    )
    .map_err(|_| invalid("opened forecast artifact length does not fit this platform"))?;
    if opened != declared {
        return Err(invalid(
            "forecast artifact size changed during metadata preflight",
        ));
    }
    // Price the read before the buffer exists.  The sixteen-byte prefix
    // carries the header length, so the payload -- and with it the host cost
    // of admitting this artifact -- is exact here rather than estimated, and
    // an artifact past the ceiling is refused by name without a byte of it
    // being held.
    let admission_host_bytes = forecast_read_admission_host_bytes(declared, &mut file, path)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(declared).map_err(|_| {
        invalid(format!(
            "admitting forecast artifact {} costs {admission_host_bytes} host bytes -- the {declared}-byte artifact plus one float32 copy of every carrier -- which this host did not give back; refused by name rather than aborting the reader on allocation",
            path.display()
        ))
    })?;
    file.read_to_end(&mut bytes).map_err(|error| {
        invalid(format!(
            "cannot read forecast artifact {}: {error}",
            path.display()
        ))
    })?;
    if bytes.len() != declared {
        return Err(invalid(format!(
            "forecast artifact size changed while reading: expected {declared}, read {}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

fn validate_inventory<'a>(
    expected: impl Iterator<Item = &'a str>,
    actual: impl Iterator<Item = &'a str>,
    label: &str,
) -> Result<()> {
    let expected: BTreeSet<&str> = expected.collect();
    let mut actual_set = BTreeSet::new();
    for name in actual {
        validate_name(name, label)?;
        if !actual_set.insert(name) {
            return Err(invalid(format!(
                "duplicate {label} entry {name:?} is refused"
            )));
        }
    }
    let missing: Vec<_> = expected.difference(&actual_set).copied().collect();
    let extra: Vec<_> = actual_set.difference(&expected).copied().collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(invalid(format!(
            "{label} do not match the contract; missing={missing:?}, extra={extra:?}"
        )));
    }
    Ok(())
}

fn reject_spec_duplicates<'a>(names: impl Iterator<Item = &'a str>, label: &str) -> Result<()> {
    let mut seen = BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(invalid(format!(
                "duplicate {label} name {name:?} is refused"
            )));
        }
    }
    Ok(())
}

fn validate_name(value: &str, label: &str) -> Result<()> {
    if value.is_empty() || value.len() > 64 {
        return Err(invalid(format!(
            "{label} must contain 1 through 64 ASCII characters"
        )));
    }
    let mut bytes = value.bytes();
    let first = bytes.next().expect("nonempty name");
    if !first.is_ascii_lowercase() {
        return Err(invalid(format!(
            "{label} {value:?} must begin with an ASCII lowercase letter"
        )));
    }
    if !bytes
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte))
    {
        return Err(invalid(format!(
            "{label} {value:?} contains a character outside [a-z0-9._-]"
        )));
    }
    Ok(())
}

fn validate_identity(value: &str, max_len: usize, label: &str) -> Result<()> {
    if value.is_empty() || value.len() > max_len {
        return Err(invalid(format!(
            "{label} must contain 1 through {max_len} ASCII characters"
        )));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"._/-".contains(&byte))
    {
        return Err(invalid(format!(
            "{label} {value:?} contains a character outside [A-Za-z0-9._/-]"
        )));
    }
    Ok(())
}

fn validate_text(value: &str, max_len: usize, label: &str) -> Result<()> {
    if value.is_empty() || value.len() > max_len || value.chars().any(char::is_control) {
        return Err(invalid(format!(
            "{label} must contain 1 through {max_len} non-control UTF-8 bytes"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(invalid(format!(
            "{label} must be exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
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
            "{label} keys are {actual:?}; expected exactly {expected:?}"
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

fn fixed_usize(value: &JsonValue, key: &str, expected: usize, label: &str) -> Result<()> {
    let actual = required_usize(value, key, label)?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} is {actual}; expected {expected}"
        )));
    }
    Ok(())
}

fn required_usize(value: &JsonValue, key: &str, label: &str) -> Result<usize> {
    let raw = value
        .get(key)
        .and_then(JsonValue::as_u64)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
    usize::try_from(raw).map_err(|_| invalid(format!("{label}.{key} does not fit this platform")))
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

fn i32_number(value: i32) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn hash(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn contract() -> ForecastContract {
        ForecastContract::new(
            "cm1-generic-test/v1",
            hash(0x11),
            hash(0x22),
            vec![
                ForecastArraySpec::new(
                    "th3d",
                    "current potential-temperature perturbation",
                    "K",
                    "scalar",
                    vec![1, 2, 1, 2],
                    vec![2, 2],
                )
                .unwrap(),
                ForecastArraySpec::new(
                    "pp3d",
                    "current Exner pressure perturbation",
                    "1",
                    "scalar",
                    vec![1, 2, 1, 2],
                    vec![2, 2],
                )
                .unwrap(),
            ],
            vec![
                SchedulerScalarSpec::new("history_record", SchedulerScalarType::U64).unwrap(),
                SchedulerScalarSpec::new("domainlocx", SchedulerScalarType::F64).unwrap(),
                SchedulerScalarSpec::new("moving_frame_active", SchedulerScalarType::Bool).unwrap(),
                SchedulerScalarSpec::new("restart_slot", SchedulerScalarType::I64).unwrap(),
            ],
            vec![ComponentBlobSpec::new("arwen_wsm6", "arwen-wsm6-persistence/v1", 64).unwrap()],
        )
        .unwrap()
    }

    fn frame() -> ForecastFrame {
        ForecastFrame::new(
            ForecastClock::new(600, 3600.0, 6.0).unwrap(),
            vec![
                ForecastArray::new("th3d", vec![1.0, -0.0, f32::from_bits(1), 4.0]),
                ForecastArray::new("pp3d", vec![0.1, 0.2, 0.3, 0.4]),
            ],
            vec![
                SchedulerScalar::new("moving_frame_active", SchedulerScalarValue::Bool(true))
                    .unwrap(),
                SchedulerScalar::new("restart_slot", SchedulerScalarValue::I64(-1)).unwrap(),
                SchedulerScalar::new("domainlocx", SchedulerScalarValue::F64(45_000.0)).unwrap(),
                SchedulerScalar::new("history_record", SchedulerScalarValue::U64(4)).unwrap(),
            ],
            vec![ComponentBlob::new(
                "arwen_wsm6",
                "arwen-wsm6-persistence/v1",
                vec![0, 1, 2, 3, 0xff],
            )],
        )
    }

    /// Envelopes published before the payload ceiling was raised, produced by
    /// `cargo run --example forecast_fixture` and pinned in
    /// `evidence/generic-forecast-envelope-v1.json`.
    const PRE_CEILING_HISTORY: &[u8] =
        include_bytes!("../evidence/generic-forecast-envelope-v1.cm1fhs");
    const PRE_CEILING_CHECKPOINT: &[u8] =
        include_bytes!("../evidence/generic-forecast-envelope-v1.cm1fcp");
    const PRE_CEILING_CONTRACT_SHA256: &str =
        "3cb0542e0b118b1a1f4c0bfeabf74c48b761f68dc1f66d2e7be077f7f88be430";
    const PRE_CEILING_HISTORY_SHA256: &str =
        "7acbcfcc8ca41901e852968140d1a1453667cee33213e553536a5dce026f68f9";
    const PRE_CEILING_CHECKPOINT_SHA256: &str =
        "579687b14f161bd489653c2ee440f06f48856ca9c3fc8dc85d73775655a517d5";

    fn temp_path(label: &str) -> PathBuf {
        let sequence = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "cm1-forecast-{label}-{}-{sequence}",
            std::process::id()
        ))
    }

    /// A contract and frame whose payload crosses the parallel-digest
    /// threshold, with carriers of unequal size and non-repeating words.
    fn large_contract_and_frame() -> (ForecastContract, ForecastFrame) {
        let shapes = [
            ("qv", 1_500_000usize),
            ("u3d", 700_001),
            ("w3d", 3),
            ("prs", 900_000),
        ];
        let specs = shapes
            .iter()
            .map(|(name, words)| {
                ForecastArraySpec::new(
                    *name,
                    "large carrier",
                    "1",
                    "scalar",
                    vec![0, *words as i32 - 1],
                    vec![*words],
                )
                .unwrap()
            })
            .collect();
        let contract = ForecastContract::new(
            "cm1-generic-test/v1",
            hash(0x33),
            hash(0x44),
            specs,
            vec![SchedulerScalarSpec::new("domainlocx", SchedulerScalarType::F64).unwrap()],
            vec![ComponentBlobSpec::new("arwen_wsm6", "arwen-wsm6-persistence/v1", 64).unwrap()],
        )
        .unwrap();
        let arrays = shapes
            .iter()
            .enumerate()
            .map(|(carrier, (name, words))| {
                let values = (0..*words)
                    .map(|index| {
                        f32::from_bits(
                            ((index as u32).wrapping_mul(2_654_435_761) >> 12)
                                | ((carrier as u32 + 1) << 24),
                        )
                    })
                    .collect();
                ForecastArray::new(*name, values)
            })
            .collect();
        let frame = ForecastFrame::new(
            ForecastClock::new(12, 6.0, 0.5).unwrap(),
            arrays,
            vec![SchedulerScalar::new("domainlocx", SchedulerScalarValue::F64(-3.25)).unwrap()],
            vec![ComponentBlob::new(
                "arwen_wsm6",
                "arwen-wsm6-persistence/v1",
                vec![9; 17],
            )],
        );
        (contract, frame)
    }

    #[test]
    fn parallel_range_digests_equal_the_serial_digests() {
        let payload = (0..(21u32 << 20))
            .map(|index| (index.wrapping_mul(2_654_435_761) >> 11) as u8)
            .collect::<Vec<_>>();
        let ranges = vec![
            0..1,
            1..(5 << 20),
            (5 << 20)..(5 << 20),
            (5 << 20)..(13 << 20) + 3,
            (13 << 20) + 3..(21 << 20),
        ];
        let (digests, whole) = range_and_payload_digests(&payload, &ranges);
        for (range, digest) in ranges.iter().zip(&digests) {
            assert_eq!(digest, &sha256::digest_hex(&payload[range.clone()]));
        }
        assert_eq!(whole, sha256::digest_hex(&payload));
    }

    #[test]
    fn owned_borrowed_and_in_memory_encodings_are_the_same_artifact() {
        for (contract, frame) in [(contract(), frame()), large_contract_and_frame()] {
            for kind in [
                ForecastArtifactKind::Checkpoint,
                ForecastArtifactKind::History,
            ] {
                let encoded = encode_forecast(kind, &contract, &frame).unwrap();
                let borrowed_path = temp_path("borrowed");
                let owned_path = temp_path("owned");
                let (borrowed, owned) = match kind {
                    ForecastArtifactKind::Checkpoint => (
                        write_forecast_checkpoint_new(&contract, &frame, &borrowed_path).unwrap(),
                        write_forecast_checkpoint_new_owned(&contract, frame.clone(), &owned_path)
                            .unwrap(),
                    ),
                    ForecastArtifactKind::History => (
                        write_forecast_history_new(&contract, &frame, &borrowed_path).unwrap(),
                        write_forecast_history_new_owned(&contract, frame.clone(), &owned_path)
                            .unwrap(),
                    ),
                };
                let bytes = fs::read(&borrowed_path).unwrap();
                assert_eq!(bytes, encoded.bytes());
                assert_eq!(fs::read(&owned_path).unwrap(), bytes);
                assert_eq!(borrowed, encoded.receipt);
                assert_eq!(owned, encoded.receipt);
                assert_eq!(borrowed.artifact_sha256, sha256::digest_hex(&bytes));
                fs::remove_file(&borrowed_path).unwrap();
                fs::remove_file(&owned_path).unwrap();
            }
        }
    }

    #[test]
    fn a_raw_byte_payload_publishes_the_frame_writers_artifact() {
        for (contract, frame) in [(contract(), frame()), large_contract_and_frame()] {
            for kind in [
                ForecastArtifactKind::Checkpoint,
                ForecastArtifactKind::History,
            ] {
                let encoded = encode_forecast(kind, &contract, &frame).unwrap();
                let mut raw = RawArrayPayload::new(&contract).unwrap();
                for array in frame.arrays() {
                    let window = raw.carrier_mut(array.name()).unwrap();
                    for (target, value) in window.chunks_exact_mut(4).zip(array.values()) {
                        target.copy_from_slice(&value.to_bits().to_le_bytes());
                    }
                }
                assert!(raw.carrier_mut("not-a-carrier").is_none());
                let path = temp_path("raw");
                let scalars = frame.scheduler_scalars().to_vec();
                let blobs = frame.component_blobs().to_vec();
                let receipt = match kind {
                    ForecastArtifactKind::Checkpoint => write_forecast_checkpoint_new_raw(
                        &contract,
                        raw,
                        frame.clock(),
                        scalars,
                        blobs,
                        &path,
                    ),
                    ForecastArtifactKind::History => write_forecast_history_new_raw(
                        &contract,
                        raw,
                        frame.clock(),
                        scalars,
                        blobs,
                        &path,
                    ),
                }
                .unwrap();
                assert_eq!(fs::read(&path).unwrap(), encoded.bytes());
                assert_eq!(receipt, encoded.receipt);
                fs::remove_file(&path).unwrap();
            }
        }
        // A payload laid out for one contract is refused under another.
        let (large, _) = large_contract_and_frame();
        let small = contract();
        let frame = frame();
        let path = temp_path("raw-foreign");
        let refusal = write_forecast_checkpoint_new_raw(
            &small,
            RawArrayPayload::new(&large).unwrap(),
            frame.clock(),
            frame.scheduler_scalars().to_vec(),
            frame.component_blobs().to_vec(),
            &path,
        )
        .unwrap_err()
        .to_string();
        assert!(refusal.contains("another forecast contract"), "{refusal}");
        assert!(!path.exists());
    }

    #[test]
    fn an_owned_frame_is_refused_exactly_as_a_borrowed_one() {
        let contract = contract();
        let mut short = frame();
        short.arrays[1].values.pop();
        let path = temp_path("owned-refused");
        let borrowed = write_forecast_checkpoint_new(&contract, &short, &path)
            .unwrap_err()
            .to_string();
        let owned = write_forecast_checkpoint_new_owned(&contract, short, &path)
            .unwrap_err()
            .to_string();
        assert_eq!(borrowed, owned);
        assert!(!path.exists());
    }

    #[test]
    fn contract_is_sorted_and_rejects_invalid_or_duplicate_specs() {
        let contract = contract();
        assert_eq!(contract.arrays[0].name, "pp3d");
        assert_eq!(contract.arrays[1].name, "th3d");
        assert_eq!(contract.component_blobs[0].name, "arwen_wsm6");
        assert!(
            ForecastArraySpec::new("bad", "bad bounds", "1", "scalar", vec![1, 2], vec![3])
                .unwrap_err()
                .to_string()
                .contains("imply 2 values")
        );
        let duplicate = ForecastContract::new(
            "duplicate/v1",
            hash(1),
            hash(2),
            vec![contract.arrays[0].clone(), contract.arrays[0].clone()],
            Vec::new(),
            Vec::new(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            duplicate.contains("duplicate array carrier contract"),
            "{duplicate}"
        );
        let oversized = ForecastContract::new(
            "oversized/v1",
            hash(1),
            hash(2),
            vec![contract.arrays[0].clone()],
            Vec::new(),
            vec![
                ComponentBlobSpec::new(
                    "component_a",
                    "component-a/v1",
                    MAX_FORECAST_PAYLOAD_BYTES / 2 + 1,
                )
                .unwrap(),
                ComponentBlobSpec::new(
                    "component_b",
                    "component-b/v1",
                    MAX_FORECAST_PAYLOAD_BYTES / 2 + 1,
                )
                .unwrap(),
            ],
        )
        .unwrap_err()
        .to_string();
        assert!(oversized.contains("array/component bytes"), "{oversized}");
    }

    #[test]
    fn generic_checkpoint_is_deterministic_and_roundtrips_all_types() {
        let contract = contract();
        let frame = frame();
        let first = encode_forecast(ForecastArtifactKind::Checkpoint, &contract, &frame).unwrap();
        let mut reversed = frame.clone();
        reversed.arrays.reverse();
        reversed.scheduler_scalars.reverse();
        let second =
            encode_forecast(ForecastArtifactKind::Checkpoint, &contract, &reversed).unwrap();
        assert_eq!(first.bytes(), second.bytes());
        assert_eq!(first.receipt, second.receipt);
        assert_eq!(first.receipt.step, 600);
        assert_eq!(first.receipt.model_time_f64_bits, f64_bits(3600.0));
        assert_eq!(first.receipt.dt_f64_bits, f64_bits(6.0));

        let path = temp_path("roundtrip.cm1fcp");
        write_forecast_checkpoint_new(&contract, &frame, &path).unwrap();
        let restored = read_forecast_checkpoint(&contract, &path).unwrap();
        assert_eq!(restored.receipt, first.receipt);
        let normalized_original = normalize(&contract, &frame).unwrap();
        let normalized_restored = normalize(&contract, restored.frame()).unwrap();
        for ((_, original), (_, actual)) in normalized_original
            .arrays
            .iter()
            .zip(&normalized_restored.arrays)
        {
            let original_bits: Vec<_> = original
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect();
            let actual_bits: Vec<_> = actual.values.iter().map(|value| value.to_bits()).collect();
            assert_eq!(original_bits, actual_bits);
        }
        assert_eq!(
            restored.frame.scheduler_scalars,
            normalized_frame_scalars(&frame, &contract)
        );
        assert_eq!(restored.frame.component_blobs, frame.component_blobs);
        fs::remove_file(path).unwrap();
    }

    fn normalized_frame_scalars(
        frame: &ForecastFrame,
        contract: &ForecastContract,
    ) -> Vec<SchedulerScalar> {
        contract
            .scheduler_scalars
            .iter()
            .map(|spec| {
                frame
                    .scheduler_scalars
                    .iter()
                    .find(|scalar| scalar.name == spec.name)
                    .unwrap()
                    .clone()
            })
            .collect()
    }

    #[test]
    fn generic_history_is_distinct_and_wrong_kind_refuses() {
        let contract = contract();
        let frame = frame();
        let checkpoint =
            encode_forecast(ForecastArtifactKind::Checkpoint, &contract, &frame).unwrap();
        let history = encode_forecast(ForecastArtifactKind::History, &contract, &frame).unwrap();
        assert_ne!(checkpoint.bytes(), history.bytes());
        assert_eq!(
            checkpoint.receipt.payload_sha256,
            history.receipt.payload_sha256
        );
        let path = temp_path("history.cm1fhs");
        write_forecast_history_new(&contract, &frame, &path).unwrap();
        let restored = read_forecast_history(&contract, &path).unwrap();
        assert_eq!(restored.kind(), ForecastArtifactKind::History);
        let error = read_forecast_checkpoint(&contract, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("magic does not identify"), "{error}");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn embedded_forecast_contract_requires_an_independent_hash_anchor() {
        let contract = contract();
        for (label, kind) in [
            (
                "embedded-checkpoint.cm1fcp",
                ForecastArtifactKind::Checkpoint,
            ),
            ("embedded-history.cm1fhs", ForecastArtifactKind::History),
        ] {
            let path = temp_path(label);
            match kind {
                ForecastArtifactKind::Checkpoint => {
                    write_forecast_checkpoint_new(&contract, &frame(), &path).unwrap();
                }
                ForecastArtifactKind::History => {
                    write_forecast_history_new(&contract, &frame(), &path).unwrap();
                }
            }
            let (embedded, restored) = match kind {
                ForecastArtifactKind::Checkpoint => {
                    read_forecast_checkpoint_embedded(&path, &contract.sha256()).unwrap()
                }
                ForecastArtifactKind::History => {
                    read_forecast_history_embedded(&path, &contract.sha256()).unwrap()
                }
            };
            assert_eq!(embedded, contract);
            assert_eq!(embedded.canonical_json(), contract.canonical_json());
            assert_eq!(restored.kind(), kind);
            assert_eq!(restored.receipt().contract_sha256, contract.sha256());
            if kind == ForecastArtifactKind::Checkpoint {
                let (inspected_contract, inspected) =
                    inspect_forecast_checkpoint_embedded(&path, &contract.sha256()).unwrap();
                assert_eq!(inspected_contract, contract);
                assert_eq!(inspected.kind(), kind);
                assert_eq!(inspected.clock(), restored.frame().clock());
                assert_eq!(inspected.receipt(), restored.receipt());
                assert_eq!(
                    inspected.array_payload_sha256(),
                    restored.array_payload_sha256()
                );
                let raw_arrays = restored
                    .frame()
                    .arrays()
                    .iter()
                    .flat_map(|array| array.values().iter().flat_map(|value| value.to_le_bytes()))
                    .collect::<Vec<_>>();
                assert_eq!(
                    inspected.array_payload_sha256(),
                    sha256::digest_hex(&raw_arrays)
                );
            }

            let error = match kind {
                ForecastArtifactKind::Checkpoint => {
                    read_forecast_checkpoint_embedded(&path, &hash(0xee)).unwrap_err()
                }
                ForecastArtifactKind::History => {
                    read_forecast_history_embedded(&path, &hash(0xee)).unwrap_err()
                }
            }
            .to_string();
            assert!(error.contains("independently expected"), "{error}");
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn missing_extra_and_duplicate_carriers_are_all_refused() {
        let contract = contract();
        let base = frame();
        let mut cases = Vec::new();

        let mut missing = base.clone();
        missing.arrays.pop();
        cases.push((missing, "missing=[\"pp3d\"]"));
        let mut extra = base.clone();
        extra.arrays.push(ForecastArray::new("extra", vec![0.0; 4]));
        cases.push((extra, "extra=[\"extra\"]"));
        let mut duplicate = base.clone();
        duplicate.arrays.push(duplicate.arrays[0].clone());
        cases.push((duplicate, "duplicate float32 array carriers"));

        let mut missing_scalar = base.clone();
        missing_scalar.scheduler_scalars.pop();
        cases.push((missing_scalar, "scheduler scalars do not match"));
        let mut duplicate_scalar = base.clone();
        duplicate_scalar
            .scheduler_scalars
            .push(duplicate_scalar.scheduler_scalars[0].clone());
        cases.push((duplicate_scalar, "duplicate scheduler scalars"));
        let mut extra_scalar = base.clone();
        extra_scalar
            .scheduler_scalars
            .push(SchedulerScalar::new("extra_scalar", SchedulerScalarValue::U64(0)).unwrap());
        cases.push((extra_scalar, "extra=[\"extra_scalar\"]"));

        let mut missing_blob = base.clone();
        missing_blob.component_blobs.clear();
        cases.push((missing_blob, "component blobs do not match"));
        let mut extra_blob = base.clone();
        extra_blob
            .component_blobs
            .push(ComponentBlob::new("extra_component", "extra/v1", vec![]));
        cases.push((extra_blob, "extra=[\"extra_component\"]"));
        let mut duplicate_blob = base.clone();
        duplicate_blob
            .component_blobs
            .push(duplicate_blob.component_blobs[0].clone());
        cases.push((duplicate_blob, "duplicate component blobs"));

        for (frame, expected) in cases {
            let error = encode_forecast(ForecastArtifactKind::Checkpoint, &contract, &frame)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn type_shape_schema_clock_and_contract_drift_refuse() {
        assert!(ForecastClock::new(1, f64::NAN, 6.0).is_err());
        assert!(ForecastClock::new(1, 6.0, 0.0).is_err());
        let contract = contract();
        let mut bad_shape = frame();
        bad_shape.arrays[0].values.pop();
        assert!(
            encode_forecast(ForecastArtifactKind::Checkpoint, &contract, &bad_shape)
                .unwrap_err()
                .to_string()
                .contains("contract shape")
        );
        let mut bad_type = frame();
        bad_type.scheduler_scalars[0] =
            SchedulerScalar::new("moving_frame_active", SchedulerScalarValue::U64(1)).unwrap();
        assert!(
            encode_forecast(ForecastArtifactKind::Checkpoint, &contract, &bad_type)
                .unwrap_err()
                .to_string()
                .contains("contract requires bool")
        );
        let mut bad_schema = frame();
        bad_schema.component_blobs[0].schema = "wrong/v1".into();
        assert!(
            encode_forecast(ForecastArtifactKind::Checkpoint, &contract, &bad_schema)
                .unwrap_err()
                .to_string()
                .contains("schema is")
        );

        let path = temp_path("contract-drift.cm1fcp");
        write_forecast_checkpoint_new(&contract, &frame(), &path).unwrap();
        let other = ForecastContract::new(
            "other/v1",
            hash(0x11),
            hash(0x22),
            contract.arrays.clone(),
            contract.scheduler_scalars.clone(),
            contract.component_blobs.clone(),
        )
        .unwrap();
        let error = read_forecast_checkpoint(&other, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("contract does not match"), "{error}");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn corruption_noncanonical_header_and_overwrite_refuse() {
        let contract = contract();
        let frame = frame();
        let encoded = encode_forecast(ForecastArtifactKind::Checkpoint, &contract, &frame).unwrap();
        let mut corrupted = encoded.bytes();
        let last = corrupted.len() - 1;
        corrupted[last] ^= 1;
        let path = temp_path("corrupt.cm1fcp");
        crate::snapshot::atomic_write_new(&path, &corrupted).unwrap();
        let error = read_forecast_checkpoint(&contract, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("payload_sha256"), "{error}");
        fs::remove_file(path).unwrap();

        let mut noncanonical = encoded.bytes();
        let header_len = u64::from_le_bytes(noncanonical[8..16].try_into().unwrap()) as usize;
        let comma = noncanonical[PREFIX_BYTES..PREFIX_BYTES + header_len]
            .iter()
            .position(|byte| *byte == b',')
            .unwrap()
            + PREFIX_BYTES;
        noncanonical.insert(comma + 1, b' ');
        noncanonical[8..16].copy_from_slice(&((header_len + 1) as u64).to_le_bytes());
        let path = temp_path("noncanonical.cm1fcp");
        crate::snapshot::atomic_write_new(&path, &noncanonical).unwrap();
        let error = read_forecast_checkpoint(&contract, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not canonical"), "{error}");
        fs::remove_file(path).unwrap();

        let path = temp_path("overwrite.cm1fcp");
        write_forecast_checkpoint_new(&contract, &frame, &path).unwrap();
        let original = fs::read(&path).unwrap();
        let error = write_forecast_checkpoint_new(&contract, &frame, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("will not be overwritten"), "{error}");
        assert_eq!(fs::read(&path).unwrap(), original);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn startup_magics_and_schemas_remain_disjoint_and_unchanged() {
        assert_eq!(crate::snapshot::CHECKPOINT_MAGIC, *b"CM1CKP1\0");
        assert_eq!(crate::snapshot::HISTORY_MAGIC, *b"CM1HST1\0");
        assert_eq!(FORECAST_CHECKPOINT_MAGIC, *b"CM1FCP1\0");
        assert_eq!(FORECAST_HISTORY_MAGIC, *b"CM1FHS1\0");
        assert_ne!(
            FORECAST_CHECKPOINT_SCHEMA,
            crate::snapshot::CHECKPOINT_SCHEMA
        );
        assert_ne!(FORECAST_HISTORY_SCHEMA, crate::snapshot::HISTORY_SCHEMA);
        assert!(CHECKPOINT_CLAIM_LIMIT.contains("live CUDA capture/restore"));
        assert_eq!(
            LIVE_STATE_WIRING,
            "deferred-no-cuda-device-copy-or-forecast-driver-restore"
        );
    }

    #[test]
    fn tracked_generic_forecast_receipt_is_bounded_json() {
        let bytes = include_bytes!("../evidence/generic-forecast-envelope-v1.json");
        assert!(!bytes.contains(&b'\r'));
        let parsed = json::parse(bytes).unwrap();
        let canonical = parsed.canonical_bytes();
        assert_eq!(
            json::parse(&canonical).unwrap().canonical_bytes(),
            canonical
        );
        let text = std::str::from_utf8(bytes).unwrap();
        assert!(text.contains("579687b14f161bd489653c2ee440f06f48856ca9c3fc8dc85d73775655a517d5"));
        assert!(text.contains("7acbcfcc8ca41901e852968140d1a1453667cee33213e553536a5dce026f68f9"));
        assert!(text.contains("deferred-no-cuda-device-copy-or-forecast-driver-restore"));
        assert!(text.contains("missing-extra-duplicate-array-scalar-component"));

        // The record enumerates the artifacts it seals, and those are the two
        // committed beside it -- name, byte count and digest.  Without this a
        // hash in the record could outlive the file it describes.
        let sealed = parsed.get("sealed_artifacts").unwrap();
        for (key, suffix, stored, digest) in [
            (
                "checkpoint",
                "cm1fcp",
                PRE_CEILING_CHECKPOINT,
                PRE_CEILING_CHECKPOINT_SHA256,
            ),
            (
                "history",
                "cm1fhs",
                PRE_CEILING_HISTORY,
                PRE_CEILING_HISTORY_SHA256,
            ),
        ] {
            let entry = sealed.get(key).unwrap();
            assert_eq!(
                entry.get("path").unwrap().as_str().unwrap(),
                format!("generic-forecast-envelope-v1.{suffix}")
            );
            assert_eq!(
                entry.get("nbytes").unwrap().as_u64().unwrap(),
                stored.len() as u64
            );
            assert_eq!(entry.get("sha256").unwrap().as_str().unwrap(), digest);
            assert_eq!(sha256::digest_hex(stored), digest);
        }
    }

    #[test]
    fn envelopes_published_under_the_old_ceiling_still_read_and_re_encode_exactly() {
        // The container did not change when the ceiling moved, so an artifact
        // written before it must admit without any compatibility branch, and
        // re-encoding what it holds must reproduce it byte for byte.
        for (stored, kind, expected_sha256) in [
            (
                PRE_CEILING_HISTORY,
                ForecastArtifactKind::History,
                PRE_CEILING_HISTORY_SHA256,
            ),
            (
                PRE_CEILING_CHECKPOINT,
                ForecastArtifactKind::Checkpoint,
                PRE_CEILING_CHECKPOINT_SHA256,
            ),
        ] {
            assert_eq!(
                sha256::digest_hex(stored),
                expected_sha256,
                "the committed {} fixture is not the pinned pre-ceiling artifact",
                kind.as_str()
            );
            assert_eq!(&stored[..8], &kind.magic());
            // The length field the pre-ceiling writer emitted is the 64-bit
            // one this reader still parses.
            let header_len = u64::from_le_bytes(stored[8..16].try_into().unwrap());
            assert_eq!(
                PREFIX_BYTES + usize::try_from(header_len).unwrap() + 37,
                stored.len()
            );

            let path = temp_path("pre-ceiling");
            crate::snapshot::atomic_write_new(&path, stored).unwrap();
            let (contract, restored) = match kind {
                ForecastArtifactKind::History => {
                    read_forecast_history_embedded(&path, PRE_CEILING_CONTRACT_SHA256).unwrap()
                }
                ForecastArtifactKind::Checkpoint => {
                    read_forecast_checkpoint_embedded(&path, PRE_CEILING_CONTRACT_SHA256).unwrap()
                }
            };
            assert_eq!(contract.sha256(), PRE_CEILING_CONTRACT_SHA256);
            assert_eq!(contract.profile_id(), "cm1-forecast-api-evidence/v1");
            let receipt = restored.receipt();
            assert_eq!(receipt.artifact_sha256, expected_sha256);
            assert_eq!(receipt.artifact_nbytes, stored.len());
            assert_eq!(receipt.payload_nbytes, 37);
            assert_eq!(receipt.step, 600);
            assert_eq!(receipt.arrays, 2);
            assert_eq!(receipt.scheduler_scalars, 4);
            assert_eq!(receipt.component_blobs, 1);
            let restored_frame = restored.frame();
            assert_eq!(
                restored_frame
                    .arrays()
                    .iter()
                    .find(|array| array.name() == "th3d")
                    .unwrap()
                    .values(),
                &[1.0, -0.0, f32::from_bits(1), 4.0]
            );
            assert_eq!(
                restored_frame.component_blobs()[0].bytes(),
                &[0, 1, 2, 3, 0xff]
            );

            let re_encoded = encode_forecast(kind, &contract, restored_frame).unwrap();
            assert_eq!(
                re_encoded.bytes(),
                stored,
                "the raised ceiling changed the published {} bytes",
                kind.as_str()
            );
            assert_eq!(&re_encoded.receipt, receipt);
            fs::remove_file(&path).unwrap();
        }
    }

    #[test]
    fn the_container_addresses_payloads_past_two_gibibytes() {
        // The old 2 GiB number was host-resource policy, never a field width:
        // both the prefix length and every JSON byte count are 64-bit, and the
        // proof is that a value past u32::MAX survives a canonical round trip.
        let past_u32 = 4_294_967_296u64 * 3;
        let encoded = object(vec![("nbytes", u64_number(past_u32))]).canonical_bytes();
        assert_eq!(encoded, br#"{"nbytes":12884901888}"#);
        assert_eq!(
            json::parse(&encoded)
                .unwrap()
                .get("nbytes")
                .unwrap()
                .as_u64()
                .unwrap(),
            past_u32
        );

        // The grid the old ceiling refused: 600x600x180 at the twelve-carrier,
        // sixteen-word GM1 history profile.  Nothing here allocates a payload;
        // the contract prices it, which is where the refusal used to land.
        let (nx, ny, nz) = (600usize, 600usize, 180usize);
        let mass = vec![nz, ny, nx];
        let faces = vec![nz + 1, ny, nx];
        let mut specs = Vec::new();
        for (name, shape) in [
            ("closure.tkea", faces.clone()),
            ("pp3d", mass.clone()),
            ("prs", mass.clone()),
            ("q3d", vec![6, nz, ny, nx]),
            ("rf", faces.clone()),
            ("rho", mass.clone()),
            ("terrain_msl", vec![ny, nx]),
            ("th3d", mass.clone()),
            ("u3d", vec![nz, ny, nx + 1]),
            ("v3d", vec![nz, ny + 1, nx]),
            ("w3d", faces.clone()),
            ("zf", faces.clone()),
        ] {
            let bounds = shape
                .iter()
                .flat_map(|extent| [1i32, i32::try_from(*extent).unwrap()])
                .collect::<Vec<_>>();
            specs.push(ForecastArraySpec::new(name, name, "1", "grid", bounds, shape).unwrap());
        }
        let payload_nbytes = specs
            .iter()
            .map(|spec| spec.words().unwrap() * 4)
            .sum::<usize>();
        assert_eq!(payload_nbytes, 4_155_264_000);
        assert!(payload_nbytes > 2 * 1024 * 1024 * 1024);
        let contract = ForecastContract::new(
            "gm1.cartesian-history/v1",
            hash(1),
            hash(2),
            specs,
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(contract.arrays().len(), 12);
        assert_eq!(
            forecast_admission_host_bytes(payload_nbytes).unwrap(),
            2 * payload_nbytes
        );
    }

    #[test]
    fn the_payload_ceiling_is_finite_and_refuses_by_name() {
        assert_eq!(MAX_FORECAST_PAYLOAD_BYTES, 16 * 1024 * 1024 * 1024);
        assert_eq!(
            forecast_admission_host_bytes(MAX_FORECAST_PAYLOAD_BYTES).unwrap(),
            32 * 1024 * 1024 * 1024
        );
        let refusal = forecast_admission_host_bytes(MAX_FORECAST_PAYLOAD_BYTES + 1)
            .unwrap_err()
            .to_string();
        assert!(
            refusal.contains(&MAX_FORECAST_PAYLOAD_BYTES.to_string()),
            "{refusal}"
        );
        assert!(
            refusal.contains("aborts the reader on allocation"),
            "{refusal}"
        );

        // The contract prices the whole inventory, so an over-ceiling grid is
        // refused before a single carrier word is copied to the host.
        let over = ForecastArraySpec::new(
            "th3d",
            "current potential-temperature perturbation",
            "K",
            "scalar",
            vec![1, i32::MAX, 1, 2],
            vec![usize::try_from(i32::MAX).unwrap(), 2],
        )
        .unwrap();
        let mut specs = Vec::new();
        for index in 0..2 {
            let mut spec = over.clone();
            spec.name = format!("carrier_{index}");
            specs.push(spec);
        }
        let refusal =
            ForecastContract::new("over/v1", hash(1), hash(2), specs, Vec::new(), Vec::new())
                .unwrap_err()
                .to_string();
        assert!(refusal.contains("array/component bytes"), "{refusal}");
    }

    #[test]
    fn the_contract_is_the_only_place_the_payload_ceiling_is_enforced() {
        // `ForecastContract::new` refuses any contract whose array words plus
        // component maxima exceed the ceiling (:461 below), and `normalize`
        // pins every array to its contracted word count and every component
        // blob under its contracted maximum.  A frame that reaches the encoder
        // is therefore already under the ceiling, which is why the encoder
        // carries no second ceiling refusal: it could never fire, and would
        // tell a reader the encoder is the enforcement point when it is not.
        let contract = contract();
        let frame = frame();
        let contracted_max = contract
            .arrays()
            .iter()
            .map(|spec| spec.words().unwrap() * 4)
            .sum::<usize>()
            + contract
                .component_blobs
                .iter()
                .map(|spec| spec.max_nbytes)
                .sum::<usize>();
        assert!(contracted_max <= MAX_FORECAST_PAYLOAD_BYTES);
        let encoded = encode_forecast(ForecastArtifactKind::History, &contract, &frame).unwrap();
        assert!(encoded.receipt.payload_nbytes <= contracted_max);

        // The two gates that hold that invariant, each refusing by name.
        let mut wide = frame.clone();
        wide.arrays[0] = ForecastArray::new("th3d", vec![0.0; 5]);
        let refusal = encode_forecast(ForecastArtifactKind::History, &contract, &wide)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("contract shape"), "{refusal}");
        let mut fat = frame.clone();
        fat.component_blobs[0] =
            ComponentBlob::new("arwen_wsm6", "arwen-wsm6-persistence/v1", vec![0u8; 65]);
        let refusal = encode_forecast(ForecastArtifactKind::History, &contract, &fat)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("contract maximum is 64"), "{refusal}");
    }

    #[test]
    fn every_read_prices_its_own_artifact_against_the_admission_ceiling() {
        // The ceiling is only a bound if a reader consults it before it
        // allocates.  `read_preflighted` takes the header length out of the
        // sixteen-byte prefix, prices the payload it implies, and refuses by
        // name -- so the arithmetic below is the production gate, not a helper
        // nothing calls.
        assert_eq!(forecast_payload_nbytes(PREFIX_BYTES + 100 + 37, 100), 37);
        // A header that does not fit the artifact leaves no payload rather
        // than a wrapped length; the decoder refuses that shape by name.
        assert_eq!(forecast_payload_nbytes(PREFIX_BYTES + 10, usize::MAX), 0);
        assert_eq!(
            forecast_admission_host_bytes(forecast_payload_nbytes(
                PREFIX_BYTES + 1024 + MAX_FORECAST_PAYLOAD_BYTES,
                1024,
            ))
            .unwrap(),
            2 * MAX_FORECAST_PAYLOAD_BYTES
        );
        let refusal = forecast_admission_host_bytes(forecast_payload_nbytes(
            PREFIX_BYTES + 1024 + MAX_FORECAST_PAYLOAD_BYTES + 1,
            1024,
        ))
        .unwrap_err()
        .to_string();
        assert!(
            refusal.contains("aborts the reader on allocation"),
            "{refusal}"
        );

        // And the gate is wired: a real published artifact reads back through
        // it, priced from its own prefix.
        let contract = contract();
        let frame = frame();
        let path = temp_path("priced.cm1fhs");
        let receipt = write_forecast_history_new(&contract, &frame, &path).unwrap();
        let mut file = File::open(&path).unwrap();
        assert_eq!(
            forecast_read_admission_host_bytes(receipt.artifact_nbytes, &mut file, &path).unwrap(),
            2 * receipt.payload_nbytes
        );
        // The prefix read is rewound, so the reader still sees the whole file.
        let (_, restored) = read_forecast_history_embedded(&path, &contract.sha256()).unwrap();
        assert_eq!(restored.receipt().artifact_sha256, receipt.artifact_sha256);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn publication_writes_exactly_the_bytes_a_single_buffer_write_would_have() {
        // Publication hands three parts to the atomic writer so a large payload
        // is never copied into a second full-size buffer.  The comparison has
        // to be against bytes that never went through the parts writer at all,
        // or it only proves the writer agrees with itself: `fs::write` of the
        // concatenation is that independent side.
        let contract = contract();
        let frame = frame();
        let encoded = encode_forecast(ForecastArtifactKind::History, &contract, &frame).unwrap();
        let mut joined = Vec::new();
        joined.extend_from_slice(&encoded.prefix);
        joined.extend_from_slice(&encoded.header);
        joined.extend_from_slice(&encoded.payload);
        assert_eq!(joined.len(), encoded.receipt.artifact_nbytes);
        assert_eq!(sha256::digest_hex(&joined), encoded.receipt.artifact_sha256);

        let parts_path = temp_path("parts.cm1fhs");
        let whole_path = temp_path("whole.cm1fhs");
        write_forecast_history_new(&contract, &frame, &parts_path).unwrap();
        fs::write(&whole_path, &joined).unwrap();
        let published = fs::read(&parts_path).unwrap();
        assert_eq!(published, fs::read(&whole_path).unwrap());
        assert_eq!(published, joined);
        // And the published file is admissible: the split into parts did not
        // move a byte of the header or the payload.
        let (read_contract, restored) =
            read_forecast_history_embedded(&parts_path, &contract.sha256()).unwrap();
        assert_eq!(read_contract.sha256(), contract.sha256());
        assert_eq!(
            restored.receipt().artifact_sha256,
            encoded.receipt.artifact_sha256
        );
        fs::remove_file(&parts_path).unwrap();
        fs::remove_file(&whole_path).unwrap();
    }
}
