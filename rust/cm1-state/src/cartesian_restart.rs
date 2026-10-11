//! Durable restart artifact for the configurable Cartesian GM1 lane.
//!
//! CM1 r21.1 writes a restart file from `src/restart_write.F:10` on the
//! `dorestart` decision of `src/cm1.F:3186`, after the large step that
//! reached `rsttim` has completed (`src/cm1.F:4246-4311`), and reads it back
//! through `src/restart_read.F:19` when `irst=1` selects file number `rstnum`
//! (`README.namelist:227-232`, `README.restart:28-49`).  CM1 saves, in its own
//! words, "all information that is carried from one timestep to the next".
//!
//! This module is the durable half of that for GM1's Cartesian lane.  Rust
//! owns the envelope, the identity binding, the byte-exactness and the atomic
//! publication; the lane owns its own carrier inventory, because that
//! inventory is a property of the built CUDA runtime -- the named allocations
//! of its device buffer registry -- and not of any grid constant.  The lane
//! declares it in one canonical inventory manifest whose SHA-256 is bound into
//! the contract, so a restart written by one runtime can never be restored
//! into a runtime whose registry differs by a single name, shape or dtype.
//!
//! Nothing here is pinned to one geometry, one timestep or one historical
//! product.  The two existing GM1 checkpoint profiles are:
//! `gm1_checkpoint.rs`, sealed to a 300x300x75 100 m real-data run at
//! `dt = 0.5 s` with a fixed 37-carrier inventory, and `live_full_state.rs`,
//! geometry-generic but with a fixed 76-carrier terrain-open inventory that a
//! Cartesian runtime does not allocate.  Neither can carry an arbitrary
//! Cartesian run, which is why the inventory is declared rather than compiled
//! in.
//!
//! A carrier whose live dtype is binary64 is carried as its exact little-endian
//! binary32 word pairs, with a trailing extent of 2 in the contract shape and
//! its true dtype recorded in the inventory.  No value is converted.

use crate::forecast::{
    read_forecast_checkpoint_embedded, write_forecast_checkpoint_new, ComponentBlob,
    ComponentBlobSpec, ForecastArray, ForecastArraySpec, ForecastClock, ForecastContract,
    ForecastFrame, ForecastReceipt, RestoredForecast, SchedulerScalar, SchedulerScalarSpec,
    SchedulerScalarType, SchedulerScalarValue,
};
use crate::json::{self, JsonValue};
use crate::sha256;
use crate::{Result, StateError};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const CARTESIAN_RESTART_MODEL: &str = "GM1";
pub const CARTESIAN_RESTART_LANE: &str = "cartesian";
pub const CARTESIAN_RESTART_PROFILE_ID: &str = "gm1.cartesian-restart/v1";
pub const CARTESIAN_RESTART_CARRIER_ABI: &str = "gm1.cartesian-restart-carriers/v1";
pub const CARTESIAN_RESTART_INVENTORY_SCHEMA: &str = "gm1.cartesian-restart-inventory/v1";
pub const CARTESIAN_RESTART_LAYOUT_SCHEMA: &str = "gm1.cartesian-restart-layout/v1";
pub const CARTESIAN_RESTART_RECEIPT_SCHEMA: &str = "gm1.cartesian-restart-receipt/v1";
pub const CARTESIAN_RESTART_MANIFEST_SCHEMA: &str = "gm1.cartesian-restart-directory/v1";
pub const CARTESIAN_RESTART_RESTORE_RECEIPT_SCHEMA: &str =
    "gm1.cartesian-restart-restore-receipt/v1";
pub const CARTESIAN_RESTART_SCHEDULER_SCHEMA: &str = "gm1.cartesian-restart-scheduler/v1";
pub const CARTESIAN_RESTART_INVENTORY_BLOB: &str = "cartesian_restart_inventory";
pub const CARTESIAN_RESTART_SCHEDULER_BLOB: &str = "cartesian_restart_scheduler";
pub const CARTESIAN_RESTART_CAPTURE_PHASE: &str = "post-solve3-completed-large-step";
pub const CARTESIAN_RESTART_RESTORE_PHASE: &str = "before-the-next-large-step";
pub const CARTESIAN_RESTART_ARENA_FILE: &str = "cartesian-carriers.f32le";
pub const CARTESIAN_RESTART_MANIFEST_FILE: &str = "manifest.json";
pub const CARTESIAN_RESTART_RECEIPT_FILE: &str = "receipt.json";
pub const CARTESIAN_RESTART_BLOB_MAX_NBYTES: usize = 1024 * 1024;
pub const CARTESIAN_RESTART_MAX_CARRIERS: usize = 4096;
pub const CARTESIAN_RESTART_PUBLICATION_CONTRACT: &str = "Rust validates the complete declared arena and every metadata field, publishes one new synchronized CM1FCP1 artifact atomically, and never replaces an existing target";
pub const CARTESIAN_RESTART_CLAIM_LIMIT: &str = "exact post-solve3 completed-large-step persistence and raw-bit restore of the declared Cartesian registry inventory, its clock, its domain-location integral and its dry-mass reference; the inventory itself is declared by the producing runtime and bound by hash, so this artifact proves byte identity and configuration identity, not that the declaring runtime enumerated every word its own next step reads";

/// Scheduler scalars every Cartesian restart carries, in contract order.
const SCHEDULER_NAMES: [(&str, SchedulerScalarType); 8] = [
    ("completed_large_steps", SchedulerScalarType::U64),
    ("completed_physics_step", SchedulerScalarType::U64),
    ("completed_solve3_step", SchedulerScalarType::U64),
    ("domainlocx_m", SchedulerScalarType::F64),
    ("domainlocy_m", SchedulerScalarType::F64),
    ("large_step_boundary", SchedulerScalarType::Bool),
    ("mass1_f64", SchedulerScalarType::F64),
    ("quiescent_capture", SchedulerScalarType::Bool),
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CartesianRestartCarrier {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl CartesianRestartCarrier {
    fn contract_shape(&self) -> Vec<usize> {
        let mut shape = self.shape.clone();
        if self.dtype == "<f8" {
            shape.push(2);
        }
        shape
    }
}

/// One lane-declared, hash-bound carrier inventory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CartesianRestartInventory {
    bytes: Vec<u8>,
    sha256: String,
    carriers: Vec<CartesianRestartCarrier>,
    /// SHA-256 of the producing run's restart identity: the projection of
    /// its sealed configuration that the restored carriers' meaning depends
    /// on.  `README.restart:16-24` lets a second leg change the namelist, so
    /// the whole document is deliberately not what is bound here.
    identity_sha256: String,
    source_identity_sha256: String,
    grid_shape_zyx: [usize; 3],
}

impl CartesianRestartInventory {
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() || bytes.len() > CARTESIAN_RESTART_BLOB_MAX_NBYTES {
            return Err(invalid(
                "Cartesian restart inventory length is outside its v1 cap",
            ));
        }
        let value = json::parse(bytes).map_err(|error| {
            invalid(format!("Cartesian restart inventory is invalid JSON: {error}"))
        })?;
        if value.canonical_bytes() != bytes {
            return Err(invalid(
                "Cartesian restart inventory is not canonical JSON; the envelope binds the exact bytes",
            ));
        }
        exact_keys(
            &value,
            &[
                "carriers",
                "identity_sha256",
                "dt_f64_bits",
                "grid_shape_zyx",
                "grid_spacing_m_f64_bits",
                "lane",
                "model",
                "schema",
                "source_identity_sha256",
            ],
            "Cartesian restart inventory",
        )?;
        fixed_string(
            &value,
            "schema",
            CARTESIAN_RESTART_INVENTORY_SCHEMA,
            "Cartesian restart inventory",
        )?;
        fixed_string(
            &value,
            "model",
            CARTESIAN_RESTART_MODEL,
            "Cartesian restart inventory",
        )?;
        fixed_string(
            &value,
            "lane",
            CARTESIAN_RESTART_LANE,
            "Cartesian restart inventory",
        )?;
        let identity_sha256 = required_sha(&value, "identity_sha256", "Cartesian restart inventory")?;
        let source_identity_sha256 = required_sha(
            &value,
            "source_identity_sha256",
            "Cartesian restart inventory",
        )?;
        let grid_shape_zyx =
            usize_array::<3>(&value, "grid_shape_zyx", "Cartesian restart inventory")?;
        if grid_shape_zyx.iter().any(|extent| *extent == 0) {
            return Err(invalid(
                "Cartesian restart inventory grid extents must all be positive",
            ));
        }
        let dt = f64_from_bits(&value, "dt_f64_bits", "Cartesian restart inventory")?;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(invalid(
                "Cartesian restart inventory dt must be a finite positive step",
            ));
        }
        let spacing = value
            .get("grid_spacing_m_f64_bits")
            .and_then(JsonValue::as_array)
            .map_err(|error| {
                invalid(format!(
                    "Cartesian restart inventory.grid_spacing_m_f64_bits: {error}"
                ))
            })?;
        if spacing.len() != 3 {
            return Err(invalid(
                "Cartesian restart inventory spacing must name three axes",
            ));
        }
        for item in spacing {
            let word = bits_from_str(item.as_str().map_err(|error| {
                invalid(format!("Cartesian restart inventory spacing: {error}"))
            })?)?;
            if !word.is_finite() || word <= 0.0 {
                return Err(invalid(
                    "Cartesian restart inventory spacing must be finite and positive on every axis",
                ));
            }
        }
        let declared = value
            .get("carriers")
            .and_then(JsonValue::as_array)
            .map_err(|error| invalid(format!("Cartesian restart inventory.carriers: {error}")))?;
        if declared.is_empty() || declared.len() > CARTESIAN_RESTART_MAX_CARRIERS {
            return Err(invalid(format!(
                "Cartesian restart inventory declares {} carriers; v1 admits 1 through {CARTESIAN_RESTART_MAX_CARRIERS}",
                declared.len()
            )));
        }
        let mut carriers = Vec::with_capacity(declared.len());
        let mut names = BTreeSet::new();
        let mut previous: Option<String> = None;
        for item in declared {
            exact_keys(item, &["dtype", "name", "shape"], "Cartesian restart carrier")?;
            let name = required_string(item, "name", "Cartesian restart carrier")?.to_string();
            let dtype = required_string(item, "dtype", "Cartesian restart carrier")?.to_string();
            if dtype != "<f4" && dtype != "<f8" {
                return Err(invalid(format!(
                    "Cartesian restart carrier {name} declares dtype {dtype:?}; v1 carries little-endian <f4 and <f8 words only"
                )));
            }
            let extents = item
                .get("shape")
                .and_then(JsonValue::as_array)
                .map_err(|error| invalid(format!("Cartesian restart carrier {name}: {error}")))?;
            if extents.is_empty() || extents.len() > 7 {
                return Err(invalid(format!(
                    "Cartesian restart carrier {name} rank is {}; v1 admits 1 through 7",
                    extents.len()
                )));
            }
            let mut shape = Vec::with_capacity(extents.len());
            for extent in extents {
                let value = extent
                    .as_u64()
                    .map_err(|error| invalid(format!("Cartesian restart carrier {name}: {error}")))?;
                let value = usize::try_from(value)
                    .map_err(|_| invalid(format!("Cartesian restart carrier {name} extent exceeds usize")))?;
                if value == 0 {
                    return Err(invalid(format!(
                        "Cartesian restart carrier {name} declares a zero extent"
                    )));
                }
                shape.push(value);
            }
            if !names.insert(name.clone()) {
                return Err(invalid(format!(
                    "Cartesian restart inventory names {name} twice"
                )));
            }
            if let Some(previous) = previous.as_ref() {
                if previous.as_str() >= name.as_str() {
                    return Err(invalid(
                        "Cartesian restart inventory carriers must be listed in strict name order",
                    ));
                }
            }
            previous = Some(name.clone());
            carriers.push(CartesianRestartCarrier { name, shape, dtype });
        }
        Ok(Self {
            bytes: bytes.to_vec(),
            sha256: sha256::digest_hex(bytes),
            carriers,
            identity_sha256,
            source_identity_sha256,
            grid_shape_zyx,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn carriers(&self) -> &[CartesianRestartCarrier] {
        &self.carriers
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub fn source_identity_sha256(&self) -> &str {
        &self.source_identity_sha256
    }

    pub fn grid_shape_zyx(&self) -> [usize; 3] {
        self.grid_shape_zyx
    }
}

#[derive(Clone, Debug)]
pub struct CartesianRestartReceipt {
    pub raw_arena_sha256: String,
    pub raw_arena_nbytes: u64,
    pub inventory_sha256: String,
    pub carrier_count: usize,
    pub forecast: ForecastReceipt,
}

impl CartesianRestartReceipt {
    pub fn canonical_json(&self) -> String {
        canonical_string(object(vec![
            ("artifact_sha256", string(&self.forecast.artifact_sha256)),
            ("capture_phase", string(CARTESIAN_RESTART_CAPTURE_PHASE)),
            ("carrier_abi", string(CARTESIAN_RESTART_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            ("claim_limit", string(CARTESIAN_RESTART_CLAIM_LIMIT)),
            ("contract_sha256", string(&self.forecast.contract_sha256)),
            ("inventory_sha256", string(&self.inventory_sha256)),
            ("model", string(CARTESIAN_RESTART_MODEL)),
            ("lane", string(CARTESIAN_RESTART_LANE)),
            ("artifact_nbytes", number(self.forecast.artifact_nbytes)),
            ("payload_nbytes", number(self.forecast.payload_nbytes)),
            ("payload_sha256", string(&self.forecast.payload_sha256)),
            ("profile_id", string(CARTESIAN_RESTART_PROFILE_ID)),
            (
                "publication_contract",
                string(CARTESIAN_RESTART_PUBLICATION_CONTRACT),
            ),
            ("raw_arena_nbytes", u64_number(self.raw_arena_nbytes)),
            ("raw_arena_sha256", string(&self.raw_arena_sha256)),
            ("schema", string(CARTESIAN_RESTART_RECEIPT_SCHEMA)),
            ("step", u64_number(self.forecast.step)),
            ("model_time_f64_bits", string(&self.forecast.model_time_f64_bits)),
            ("dt_f64_bits", string(&self.forecast.dt_f64_bits)),
        ]))
    }
}

#[derive(Clone, Debug)]
pub struct RestoredCartesianRestart {
    contract: ForecastContract,
    forecast: RestoredForecast,
    inventory: CartesianRestartInventory,
    scheduler: Vec<u8>,
}

impl RestoredCartesianRestart {
    pub fn contract(&self) -> &ForecastContract {
        &self.contract
    }

    pub fn forecast(&self) -> &RestoredForecast {
        &self.forecast
    }

    pub fn inventory(&self) -> &CartesianRestartInventory {
        &self.inventory
    }

    pub fn scheduler_bytes(&self) -> &[u8] {
        &self.scheduler
    }
}

pub fn cartesian_restart_layout_json(inventory: &CartesianRestartInventory) -> Result<String> {
    let contract = cartesian_contract(inventory)?;
    let mut offset = 0u64;
    let mut carriers = Vec::with_capacity(contract.arrays().len());
    for (spec, declared) in contract.arrays().iter().zip(inventory.carriers()) {
        let nbytes = spec_nbytes(spec)?;
        carriers.push(object(vec![
            ("dtype", string(&declared.dtype)),
            ("name", string(spec.name())),
            ("nbytes", u64_number(nbytes)),
            ("offset_nbytes", u64_number(offset)),
            (
                "shape",
                JsonValue::Array(declared.shape.iter().copied().map(number).collect()),
            ),
        ]));
        offset = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("Cartesian restart layout byte count overflows"))?;
    }
    Ok(canonical_string(object(vec![
        ("capture_phase", string(CARTESIAN_RESTART_CAPTURE_PHASE)),
        ("carrier_abi", string(CARTESIAN_RESTART_CARRIER_ABI)),
        ("carrier_count", number(contract.arrays().len())),
        ("carriers", JsonValue::Array(carriers)),
        ("claim_limit", string(CARTESIAN_RESTART_CLAIM_LIMIT)),
        ("identity_sha256", string(inventory.identity_sha256())),
        ("contract_sha256", string(&contract.sha256())),
        ("inventory_sha256", string(inventory.sha256())),
        ("lane", string(CARTESIAN_RESTART_LANE)),
        ("model", string(CARTESIAN_RESTART_MODEL)),
        ("order", string("C")),
        ("payload_nbytes", u64_number(offset)),
        ("profile_id", string(CARTESIAN_RESTART_PROFILE_ID)),
        ("restore_phase", string(CARTESIAN_RESTART_RESTORE_PHASE)),
        ("schema", string(CARTESIAN_RESTART_LAYOUT_SCHEMA)),
        (
            "source_identity_sha256",
            string(inventory.source_identity_sha256()),
        ),
    ])))
}

#[allow(clippy::too_many_arguments)]
pub fn write_cartesian_restart_new(
    raw_arena: impl AsRef<Path>,
    output: impl AsRef<Path>,
    expected_raw_arena_sha256: &str,
    inventory: CartesianRestartInventory,
    clock: ForecastClock,
    completed_physics_step: u64,
    completed_solve3_step: u64,
    domainlocx_m: f64,
    domainlocy_m: f64,
    mass1_f64: f64,
    scheduler: &[u8],
) -> Result<CartesianRestartReceipt> {
    validate_sha256(expected_raw_arena_sha256, "expected Cartesian arena SHA-256")?;
    validate_boundary_clock(clock)?;
    if completed_physics_step != clock.step() || completed_solve3_step != clock.step() {
        return Err(invalid(
            "Cartesian restart requires the physics and solve3 counters to stand at the completed large step; anywhere else the carriers mix two model times",
        ));
    }
    if !mass1_f64.is_finite() || mass1_f64 <= 0.0 {
        return Err(invalid(
            "Cartesian restart dry-mass reference must be finite and positive",
        ));
    }
    for (label, value) in [("domainlocx_m", domainlocx_m), ("domainlocy_m", domainlocy_m)] {
        if !value.is_finite() {
            return Err(invalid(format!(
                "Cartesian restart {label} must be finite; cm1.F:3512-3513 integrates it once per large step"
            )));
        }
    }
    let scheduler = validate_scheduler(scheduler)?;
    let contract = cartesian_contract(&inventory)?;
    let arrays = read_raw_arena(raw_arena.as_ref(), &contract, expected_raw_arena_sha256)?;
    let frame = ForecastFrame::new(
        clock,
        arrays,
        vec![
            SchedulerScalar::new(
                "completed_large_steps",
                SchedulerScalarValue::U64(clock.step()),
            )?,
            SchedulerScalar::new(
                "completed_physics_step",
                SchedulerScalarValue::U64(completed_physics_step),
            )?,
            SchedulerScalar::new(
                "completed_solve3_step",
                SchedulerScalarValue::U64(completed_solve3_step),
            )?,
            SchedulerScalar::new("domainlocx_m", SchedulerScalarValue::F64(domainlocx_m))?,
            SchedulerScalar::new("domainlocy_m", SchedulerScalarValue::F64(domainlocy_m))?,
            SchedulerScalar::new("large_step_boundary", SchedulerScalarValue::Bool(true))?,
            SchedulerScalar::new("mass1_f64", SchedulerScalarValue::F64(mass1_f64))?,
            SchedulerScalar::new("quiescent_capture", SchedulerScalarValue::Bool(true))?,
        ],
        vec![
            ComponentBlob::new(
                CARTESIAN_RESTART_INVENTORY_BLOB,
                CARTESIAN_RESTART_INVENTORY_SCHEMA,
                inventory.bytes().to_vec(),
            ),
            ComponentBlob::new(
                CARTESIAN_RESTART_SCHEDULER_BLOB,
                CARTESIAN_RESTART_SCHEDULER_SCHEMA,
                scheduler,
            ),
        ],
    );
    let forecast = write_forecast_checkpoint_new(&contract, &frame, output)?;
    Ok(CartesianRestartReceipt {
        raw_arena_sha256: expected_raw_arena_sha256.to_string(),
        raw_arena_nbytes: array_bytes(&contract)?,
        inventory_sha256: inventory.sha256().to_string(),
        carrier_count: inventory.carriers().len(),
        forecast,
    })
}

pub fn read_cartesian_restart(
    input: impl AsRef<Path>,
    expected_contract_sha256: &str,
    expected_artifact_sha256: &str,
    expected_inventory_sha256: &str,
) -> Result<RestoredCartesianRestart> {
    validate_sha256(
        expected_artifact_sha256,
        "expected Cartesian restart artifact SHA-256",
    )?;
    validate_sha256(
        expected_inventory_sha256,
        "expected Cartesian restart inventory SHA-256",
    )?;
    let (contract, forecast) =
        read_forecast_checkpoint_embedded(input.as_ref(), expected_contract_sha256)?;
    if contract.profile_id() != CARTESIAN_RESTART_PROFILE_ID {
        return Err(invalid(format!(
            "artifact profile is {:?}; expected {CARTESIAN_RESTART_PROFILE_ID}",
            contract.profile_id()
        )));
    }
    if forecast.receipt().artifact_sha256 != expected_artifact_sha256 {
        return Err(invalid(
            "Cartesian restart artifact SHA-256 differs from the independently recorded value",
        ));
    }
    let inventory_bytes = blob_bytes(&forecast, CARTESIAN_RESTART_INVENTORY_BLOB)?;
    let inventory = CartesianRestartInventory::from_canonical_bytes(&inventory_bytes)?;
    if inventory.sha256() != expected_inventory_sha256 {
        return Err(invalid(format!(
            "the artifact carries inventory {} and the live runtime declares {expected_inventory_sha256}; a restart may only be restored into the registry it was captured from",
            inventory.sha256()
        )));
    }
    let scheduler = blob_bytes(&forecast, CARTESIAN_RESTART_SCHEDULER_BLOB)?;
    validate_scheduler(&scheduler)?;
    let rebuilt = cartesian_contract(&inventory)?;
    if rebuilt.sha256() != contract.sha256() {
        return Err(invalid(
            "the embedded contract does not rebuild from the embedded inventory",
        ));
    }
    Ok(RestoredCartesianRestart {
        contract,
        forecast,
        inventory,
        scheduler,
    })
}

pub fn restore_cartesian_restart_new(
    restored: &RestoredCartesianRestart,
    output_directory: impl AsRef<Path>,
) -> Result<String> {
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace Cartesian restore destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "Cartesian restore parent does not exist: {}",
            parent.display()
        )));
    }
    let staging = staging_path(output)?;
    fs::create_dir(&staging)?;
    match restore_into_directory(restored, &staging) {
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

fn restore_into_directory(
    restored: &RestoredCartesianRestart,
    directory: &Path,
) -> Result<String> {
    let arena = directory.join(CARTESIAN_RESTART_ARENA_FILE);
    let mut digest = sha256::Sha256::default();
    let mut offset = 0u64;
    let mut entries = Vec::with_capacity(restored.inventory.carriers().len());
    {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&arena)?;
        let mut writer = BufWriter::new(file);
        for (spec, declared) in restored
            .contract
            .arrays()
            .iter()
            .zip(restored.inventory.carriers())
        {
            let values = restored
                .forecast
                .frame()
                .arrays()
                .iter()
                .find(|array| array.name() == spec.name())
                .map(ForecastArray::values)
                .ok_or_else(|| {
                    invalid(format!("restored artifact lacks carrier {}", spec.name()))
                })?;
            let nbytes = spec_nbytes(spec)?;
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            if bytes.len() as u64 != nbytes {
                return Err(invalid(format!(
                    "restored carrier {} has {} bytes; the contract requires {nbytes}",
                    spec.name(),
                    bytes.len()
                )));
            }
            writer.write_all(&bytes)?;
            digest.update(&bytes);
            entries.push(object(vec![
                ("dtype", string(&declared.dtype)),
                ("name", string(spec.name())),
                ("nbytes", u64_number(nbytes)),
                ("offset_nbytes", u64_number(offset)),
                (
                    "shape",
                    JsonValue::Array(declared.shape.iter().copied().map(number).collect()),
                ),
            ]));
            offset = offset
                .checked_add(nbytes)
                .ok_or_else(|| invalid("Cartesian restore byte count overflows"))?;
        }
        let file = writer
            .into_inner()
            .map_err(|error| StateError::from(error.into_error()))?;
        file.sync_all()?;
    }
    let arena_sha256 = sha256::hex(&digest.finalize());
    let manifest = canonical_string(object(vec![
        ("arena_file", string(CARTESIAN_RESTART_ARENA_FILE)),
        ("arena_nbytes", u64_number(offset)),
        ("arena_sha256", string(&arena_sha256)),
        ("carrier_abi", string(CARTESIAN_RESTART_CARRIER_ABI)),
        ("carriers", JsonValue::Array(entries)),
        ("identity_sha256", string(restored.inventory.identity_sha256())),
        (
            "contract_sha256",
            string(&restored.contract.sha256()),
        ),
        ("dtype_default", string("<f4")),
        (
            "inventory_sha256",
            string(restored.inventory.sha256()),
        ),
        ("lane", string(CARTESIAN_RESTART_LANE)),
        ("model", string(CARTESIAN_RESTART_MODEL)),
        ("order", string("C")),
        ("restore_phase", string(CARTESIAN_RESTART_RESTORE_PHASE)),
        ("schema", string(CARTESIAN_RESTART_MANIFEST_SCHEMA)),
        (
            "source_identity_sha256",
            string(restored.inventory.source_identity_sha256()),
        ),
    ]));
    write_new_synced(
        &directory.join(CARTESIAN_RESTART_MANIFEST_FILE),
        manifest.as_bytes(),
    )?;
    let clock = restored.forecast.frame().clock();
    let receipt = canonical_string(object(vec![
        ("arena_file", string(CARTESIAN_RESTART_ARENA_FILE)),
        ("arena_sha256", string(&arena_sha256)),
        (
            "artifact_sha256",
            string(&restored.forecast.receipt().artifact_sha256),
        ),
        ("carrier_count", number(restored.contract.arrays().len())),
        ("claim_limit", string(CARTESIAN_RESTART_CLAIM_LIMIT)),
        ("contract_sha256", string(&restored.contract.sha256())),
        ("dt_f64_bits", f64_bits(clock.dt_s())),
        ("inventory_sha256", string(restored.inventory.sha256())),
        ("manifest_file", string(CARTESIAN_RESTART_MANIFEST_FILE)),
        ("model_time_f64_bits", f64_bits(clock.model_time_s())),
        ("restore_phase", string(CARTESIAN_RESTART_RESTORE_PHASE)),
        ("schema", string(CARTESIAN_RESTART_RESTORE_RECEIPT_SCHEMA)),
        ("scheduler", scheduler_json(&restored.scheduler)?),
        (
            "scheduler_scalars",
            scheduler_scalars_json(restored.forecast.frame().scheduler_scalars()),
        ),
        ("step", u64_number(clock.step())),
    ]));
    write_new_synced(
        &directory.join(CARTESIAN_RESTART_RECEIPT_FILE),
        receipt.as_bytes(),
    )?;
    Ok(receipt)
}

fn cartesian_contract(inventory: &CartesianRestartInventory) -> Result<ForecastContract> {
    let mut specs = Vec::with_capacity(inventory.carriers().len());
    for carrier in inventory.carriers() {
        let shape = carrier.contract_shape();
        specs.push(ForecastArraySpec::new(
            carrier.name.as_str(),
            "Cartesian runtime device registry carrier, raw words",
            "model-specific",
            if carrier.dtype == "<f8" {
                "registry allocation, binary64 as binary32 word pairs"
            } else {
                "registry allocation"
            },
            one_bounds(&shape)?,
            shape,
        )?);
    }
    ForecastContract::new(
        CARTESIAN_RESTART_PROFILE_ID,
        inventory.identity_sha256().to_string(),
        inventory.sha256().to_string(),
        specs,
        SCHEDULER_NAMES
            .into_iter()
            .map(|(name, kind)| SchedulerScalarSpec::new(name, kind))
            .collect::<Result<Vec<_>>>()?,
        vec![
            ComponentBlobSpec::new(
                CARTESIAN_RESTART_INVENTORY_BLOB,
                CARTESIAN_RESTART_INVENTORY_SCHEMA,
                CARTESIAN_RESTART_BLOB_MAX_NBYTES,
            )?,
            ComponentBlobSpec::new(
                CARTESIAN_RESTART_SCHEDULER_BLOB,
                CARTESIAN_RESTART_SCHEDULER_SCHEMA,
                CARTESIAN_RESTART_BLOB_MAX_NBYTES,
            )?,
        ],
    )
}

fn validate_boundary_clock(clock: ForecastClock) -> Result<()> {
    if clock.step() == 0 {
        return Err(invalid(
            "a Cartesian restart requires at least one completed large step; capturing at step zero would persist the initialized source state under a forecast clock",
        ));
    }
    if !clock.dt_s().is_finite() || clock.dt_s() <= 0.0 {
        return Err(invalid("Cartesian restart dt must be finite and positive"));
    }
    let expected = (clock.step() as f64) * clock.dt_s();
    if clock.model_time_s().to_bits() != expected.to_bits() {
        return Err(invalid(
            "Cartesian restart model time does not equal completed_large_steps * dt at the safe boundary",
        ));
    }
    Ok(())
}

fn validate_scheduler(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.is_empty() || bytes.len() > CARTESIAN_RESTART_BLOB_MAX_NBYTES {
        return Err(invalid(
            "Cartesian restart scheduler blob length is outside its v1 cap",
        ));
    }
    let value = json::parse(bytes)
        .map_err(|error| invalid(format!("Cartesian restart scheduler is invalid JSON: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid(
            "Cartesian restart scheduler blob is not canonical JSON",
        ));
    }
    exact_keys(&value, &["entries", "schema"], "Cartesian restart scheduler")?;
    fixed_string(
        &value,
        "schema",
        CARTESIAN_RESTART_SCHEDULER_SCHEMA,
        "Cartesian restart scheduler",
    )?;
    let entries = value
        .get("entries")
        .and_then(JsonValue::as_object)
        .map_err(|error| invalid(format!("Cartesian restart scheduler.entries: {error}")))?;
    for (name, _) in entries {
        if name.is_empty()
            || name.len() > 128
            || !name.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
            })
        {
            return Err(invalid(format!(
                "Cartesian restart scheduler key {name:?} must use 1..128 [a-z0-9._-] characters"
            )));
        }
    }
    Ok(bytes.to_vec())
}

fn scheduler_scalars_json(scalars: &[SchedulerScalar]) -> JsonValue {
    JsonValue::Object(
        scalars
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
            .collect(),
    )
}

fn scheduler_json(bytes: &[u8]) -> Result<JsonValue> {
    json::parse(bytes)
        .map_err(|error| invalid(format!("Cartesian restart scheduler is invalid JSON: {error}")))
}

fn blob_bytes(forecast: &RestoredForecast, name: &str) -> Result<Vec<u8>> {
    forecast
        .frame()
        .component_blobs()
        .iter()
        .find(|blob| blob.name() == name)
        .map(|blob| blob.bytes().to_vec())
        .ok_or_else(|| invalid(format!("Cartesian restart artifact lacks blob {name}")))
}

fn read_raw_arena(
    path: &Path,
    contract: &ForecastContract,
    expected_sha256: &str,
) -> Result<Vec<ForecastArray>> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect Cartesian raw arena {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(
            "Cartesian raw arena must be a regular non-symlink file",
        ));
    }
    let expected_nbytes = array_bytes(contract)?;
    if metadata.len() != expected_nbytes {
        return Err(invalid(format!(
            "Cartesian raw arena has {} bytes; the declared inventory requires {expected_nbytes}",
            metadata.len()
        )));
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = sha256::Sha256::default();
    let mut arrays = Vec::with_capacity(contract.arrays().len());
    let mut word = [0u8; 4];
    for spec in contract.arrays() {
        let words = usize::try_from(spec_nbytes(spec)? / 4)
            .map_err(|_| invalid("Cartesian carrier word count exceeds usize"))?;
        let mut values = Vec::with_capacity(words);
        for _ in 0..words {
            reader.read_exact(&mut word)?;
            digest.update(&word);
            values.push(f32::from_le_bytes(word));
        }
        arrays.push(ForecastArray::new(spec.name(), values));
    }
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(invalid("Cartesian raw arena carries trailing bytes"));
    }
    let actual = sha256::hex(&digest.finalize());
    if actual != expected_sha256 {
        return Err(invalid(format!(
            "Cartesian raw arena SHA-256 is {actual}; the caller declared {expected_sha256}"
        )));
    }
    Ok(arrays)
}

fn array_bytes(contract: &ForecastContract) -> Result<u64> {
    contract.arrays().iter().try_fold(0u64, |total, spec| {
        total
            .checked_add(spec_nbytes(spec)?)
            .ok_or_else(|| invalid("Cartesian arena byte total overflows"))
    })
}

fn spec_nbytes(spec: &ForecastArraySpec) -> Result<u64> {
    let words = spec.shape().iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("Cartesian carrier {} shape overflows", spec.name())))
    })?;
    let bytes = words
        .checked_mul(4)
        .ok_or_else(|| invalid("Cartesian carrier byte count overflows"))?;
    u64::try_from(bytes).map_err(|_| invalid("Cartesian carrier byte count exceeds u64"))
}

fn one_bounds(shape: &[usize]) -> Result<Vec<i32>> {
    let mut bounds = Vec::with_capacity(shape.len() * 2);
    for extent in shape {
        bounds.push(1);
        bounds.push(
            i32::try_from(*extent).map_err(|_| invalid("Cartesian extent does not fit i32"))?,
        );
    }
    Ok(bounds)
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("Cartesian restore destination needs a UTF-8 file name"))?;
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
        .map(|(name, _)| name.as_str())
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

fn required_sha(value: &JsonValue, key: &str, label: &str) -> Result<String> {
    let result = required_string(value, key, label)?.to_string();
    validate_sha256(&result, &format!("{label}.{key}"))?;
    Ok(result)
}

fn usize_array<const N: usize>(value: &JsonValue, key: &str, label: &str) -> Result<[usize; N]> {
    let values = value
        .get(key)
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
    if values.len() != N {
        return Err(invalid(format!("{label}.{key} must contain {N} values")));
    }
    values
        .iter()
        .map(|item| {
            let value = item
                .as_u64()
                .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
            usize::try_from(value).map_err(|_| invalid(format!("{label}.{key} exceeds usize")))
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| invalid(format!("{label}.{key} length changed")))
}

fn f64_from_bits(value: &JsonValue, key: &str, label: &str) -> Result<f64> {
    bits_from_str(required_string(value, key, label)?)
}

fn bits_from_str(value: &str) -> Result<f64> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(
            "a binary64 word must be exactly 16 lower-case hexadecimal characters",
        ));
    }
    u64::from_str_radix(value, 16)
        .map(f64::from_bits)
        .map_err(|_| invalid("binary64 word is not hexadecimal"))
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
    String::from_utf8(value.canonical_bytes()).expect("canonical Cartesian restart JSON is UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "cartesian-restart-{label}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn inventory_bytes() -> Vec<u8> {
        // Names must be listed in strict order; "physics.asq" is the binary64
        // carrier, so its contract shape gains a trailing extent of two.
        let value = object(vec![
            (
                "carriers",
                JsonValue::Array(vec![
                    object(vec![
                        ("dtype", string("<f4")),
                        ("name", string("field.th3d")),
                        (
                            "shape",
                            JsonValue::Array(vec![number(2), number(3), number(4)]),
                        ),
                    ]),
                    object(vec![
                        ("dtype", string("<f4")),
                        ("name", string("padded.radbcw")),
                        ("shape", JsonValue::Array(vec![number(3), number(5)])),
                    ]),
                    object(vec![
                        ("dtype", string("<f8")),
                        ("name", string("physics.asq")),
                        ("shape", JsonValue::Array(vec![number(6)])),
                    ]),
                ]),
            ),
            ("dt_f64_bits", f64_bits(0.5)),
            (
                "grid_shape_zyx",
                JsonValue::Array(vec![number(2), number(3), number(4)]),
            ),
            (
                "grid_spacing_m_f64_bits",
                JsonValue::Array(vec![f64_bits(1000.0), f64_bits(1000.0), f64_bits(500.0)]),
            ),
            ("identity_sha256", string(&"a".repeat(64))),
            ("lane", string(CARTESIAN_RESTART_LANE)),
            ("model", string(CARTESIAN_RESTART_MODEL)),
            ("schema", string(CARTESIAN_RESTART_INVENTORY_SCHEMA)),
            ("source_identity_sha256", string(&"b".repeat(64))),
        ]);
        value.canonical_bytes()
    }

    fn scheduler_bytes() -> Vec<u8> {
        object(vec![
            (
                "entries",
                object(vec![("completed_large_steps", u64_number(8))]),
            ),
            ("schema", string(CARTESIAN_RESTART_SCHEDULER_SCHEMA)),
        ])
        .canonical_bytes()
    }

    /// 24 + 15 f32 words plus 6 f64 values as 12 f32 words.
    fn arena_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        for index in 0..(24 + 15) {
            bytes.extend_from_slice(&(index as f32).to_le_bytes());
        }
        for index in 0..6u64 {
            bytes.extend_from_slice(&(index as f64 * 0.125).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn an_inventory_must_be_canonical_complete_and_ordered() {
        let bytes = inventory_bytes();
        let inventory = CartesianRestartInventory::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(inventory.carriers().len(), 3);
        assert_eq!(inventory.grid_shape_zyx(), [2, 3, 4]);
        assert_eq!(inventory.identity_sha256(), "a".repeat(64));

        let mut padded = bytes.clone();
        padded.push(b' ');
        assert!(CartesianRestartInventory::from_canonical_bytes(&padded).is_err());

        let reordered = String::from_utf8(bytes.clone())
            .unwrap()
            .replace("field.th3d", "zzz.th3d");
        assert!(CartesianRestartInventory::from_canonical_bytes(reordered.as_bytes()).is_err());

        let retyped = String::from_utf8(bytes).unwrap().replace("<f8", "<i4");
        assert!(CartesianRestartInventory::from_canonical_bytes(retyped.as_bytes()).is_err());
    }

    #[test]
    fn the_layout_places_every_carrier_and_doubles_a_binary64_extent() {
        let inventory = CartesianRestartInventory::from_canonical_bytes(&inventory_bytes()).unwrap();
        let layout = json::parse(cartesian_restart_layout_json(&inventory).unwrap().as_bytes())
            .unwrap();
        assert_eq!(layout.get("payload_nbytes").unwrap().as_u64().unwrap(), 204);
        let carriers = layout.get("carriers").unwrap().as_array().unwrap();
        assert_eq!(carriers.len(), 3);
        assert_eq!(
            carriers[2].get("nbytes").unwrap().as_u64().unwrap(),
            48,
            "six binary64 values occupy twelve binary32 words"
        );
        assert_eq!(
            carriers[2].get("offset_nbytes").unwrap().as_u64().unwrap(),
            (24 + 15) * 4
        );
    }

    #[test]
    fn a_captured_arena_restores_raw_word_exactly() {
        let root = TestRoot::new("roundtrip");
        let raw = root.join("arena.f32le");
        let payload = arena_bytes();
        fs::write(&raw, &payload).unwrap();
        let raw_sha = sha256::digest_hex(&payload);
        let inventory = CartesianRestartInventory::from_canonical_bytes(&inventory_bytes()).unwrap();
        let output = root.join("cm1rst_000001.cm1fcp");
        let receipt = write_cartesian_restart_new(
            &raw,
            &output,
            &raw_sha,
            inventory.clone(),
            ForecastClock::new(8, 4.0, 0.5).unwrap(),
            8,
            8,
            120.0,
            -30.0,
            3764.5,
            &scheduler_bytes(),
        )
        .unwrap();
        assert_eq!(receipt.carrier_count, 3);
        assert_eq!(receipt.raw_arena_nbytes, payload.len() as u64);

        let restored = read_cartesian_restart(
            &output,
            &receipt.forecast.contract_sha256,
            &receipt.forecast.artifact_sha256,
            inventory.sha256(),
        )
        .unwrap();
        let directory = root.join("restored");
        let json_text = restore_cartesian_restart_new(&restored, &directory).unwrap();
        assert!(json_text.contains("\"step\":8"));
        let written = fs::read(directory.join(CARTESIAN_RESTART_ARENA_FILE)).unwrap();
        assert_eq!(written, payload, "the restored arena is the captured bytes");

        // The whole-artifact digest covers the scheduler scalars too.
        let receipt_text =
            fs::read_to_string(directory.join(CARTESIAN_RESTART_RECEIPT_FILE)).unwrap();
        // The typed scheduler scalars ride in the restore receipt, so the
        // resume reads the clock, the cm1.F:3512-3513 domain location and the
        // dry-mass reference from the same authenticated envelope as the words.
        assert!(receipt_text.contains("\"domainlocx_m\":\"405e000000000000\""));
        assert!(receipt_text.contains("\"domainlocy_m\":\"c03e000000000000\""));
        assert!(receipt_text.contains("\"completed_physics_step\":8"));
    }

    #[test]
    fn an_inventory_from_another_runtime_is_refused_on_read() {
        let root = TestRoot::new("identity");
        let raw = root.join("arena.f32le");
        let payload = arena_bytes();
        fs::write(&raw, &payload).unwrap();
        let inventory = CartesianRestartInventory::from_canonical_bytes(&inventory_bytes()).unwrap();
        let output = root.join("cm1rst_000001.cm1fcp");
        let receipt = write_cartesian_restart_new(
            &raw,
            &output,
            &sha256::digest_hex(&payload),
            inventory,
            ForecastClock::new(8, 4.0, 0.5).unwrap(),
            8,
            8,
            0.0,
            0.0,
            1.0,
            &scheduler_bytes(),
        )
        .unwrap();
        let error = read_cartesian_restart(
            &output,
            &receipt.forecast.contract_sha256,
            &receipt.forecast.artifact_sha256,
            &"c".repeat(64),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("captured from"));
    }

    #[test]
    fn a_capture_is_refused_before_any_large_step_has_completed() {
        let root = TestRoot::new("step-zero");
        let raw = root.join("arena.f32le");
        let payload = arena_bytes();
        fs::write(&raw, &payload).unwrap();
        let inventory = CartesianRestartInventory::from_canonical_bytes(&inventory_bytes()).unwrap();
        let error = write_cartesian_restart_new(
            &raw,
            &root.join("zero.cm1fcp"),
            &sha256::digest_hex(&payload),
            inventory,
            ForecastClock::new(0, 0.0, 0.5).unwrap(),
            0,
            0,
            0.0,
            0.0,
            1.0,
            &scheduler_bytes(),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("at least one completed large step"));
    }

    #[test]
    fn a_clock_off_its_own_step_product_is_refused() {
        let root = TestRoot::new("clock");
        let raw = root.join("arena.f32le");
        let payload = arena_bytes();
        fs::write(&raw, &payload).unwrap();
        let inventory = CartesianRestartInventory::from_canonical_bytes(&inventory_bytes()).unwrap();
        let error = write_cartesian_restart_new(
            &raw,
            &root.join("skewed.cm1fcp"),
            &sha256::digest_hex(&payload),
            inventory,
            ForecastClock::new(8, 4.25, 0.5).unwrap(),
            8,
            8,
            0.0,
            0.0,
            1.0,
            &scheduler_bytes(),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("completed_large_steps * dt"));
    }

    #[test]
    fn an_arena_that_does_not_match_its_declared_digest_is_refused() {
        let root = TestRoot::new("digest");
        let raw = root.join("arena.f32le");
        fs::write(&raw, arena_bytes()).unwrap();
        let inventory = CartesianRestartInventory::from_canonical_bytes(&inventory_bytes()).unwrap();
        let error = write_cartesian_restart_new(
            &raw,
            &root.join("bad.cm1fcp"),
            &"d".repeat(64),
            inventory,
            ForecastClock::new(8, 4.0, 0.5).unwrap(),
            8,
            8,
            0.0,
            0.0,
            1.0,
            &scheduler_bytes(),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("SHA-256 is"));
    }

    #[test]
    fn an_arena_of_the_wrong_length_is_refused_before_it_is_read() {
        let root = TestRoot::new("length");
        let raw = root.join("arena.f32le");
        let mut payload = arena_bytes();
        payload.truncate(payload.len() - 4);
        fs::write(&raw, &payload).unwrap();
        let inventory = CartesianRestartInventory::from_canonical_bytes(&inventory_bytes()).unwrap();
        let error = write_cartesian_restart_new(
            &raw,
            &root.join("short.cm1fcp"),
            &sha256::digest_hex(&payload),
            inventory,
            ForecastClock::new(8, 4.0, 0.5).unwrap(),
            8,
            8,
            0.0,
            0.0,
            1.0,
            &scheduler_bytes(),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("declared inventory requires"));
    }

    #[test]
    fn a_capture_never_replaces_an_existing_artifact() {
        let root = TestRoot::new("exists");
        let raw = root.join("arena.f32le");
        let payload = arena_bytes();
        fs::write(&raw, &payload).unwrap();
        let inventory = CartesianRestartInventory::from_canonical_bytes(&inventory_bytes()).unwrap();
        let output = root.join("cm1rst_000001.cm1fcp");
        for attempt in 0..2 {
            let result = write_cartesian_restart_new(
                &raw,
                &output,
                &sha256::digest_hex(&payload),
                inventory.clone(),
                ForecastClock::new(8, 4.0, 0.5).unwrap(),
                8,
                8,
                0.0,
                0.0,
                1.0,
                &scheduler_bytes(),
            );
            assert_eq!(result.is_ok(), attempt == 0);
        }
    }

    #[test]
    fn the_physics_counters_must_stand_at_the_completed_step() {
        let root = TestRoot::new("counters");
        let raw = root.join("arena.f32le");
        let payload = arena_bytes();
        fs::write(&raw, &payload).unwrap();
        let inventory = CartesianRestartInventory::from_canonical_bytes(&inventory_bytes()).unwrap();
        let error = write_cartesian_restart_new(
            &raw,
            &root.join("mixed.cm1fcp"),
            &sha256::digest_hex(&payload),
            inventory,
            ForecastClock::new(8, 4.0, 0.5).unwrap(),
            7,
            8,
            0.0,
            0.0,
            1.0,
            &scheduler_bytes(),
        )
        .unwrap_err();
        assert!(format!("{error}").contains("mix two model times"));
    }
}
