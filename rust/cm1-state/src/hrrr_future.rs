//! Strict publication of future HRRR boundary frames for a GM1 run.
//!
//! The caller provides independently pinned real-source checksums, complete
//! decoded windows, and an already admitted initialization-time history.  Rust
//! validates the entire provenance chain, preserves absolute HRRR source
//! clocks, maps the future frames, and atomically publishes deterministic
//! `CM1FHS1` artifacts plus a canonical series receipt.

use crate::forecast::{
    read_forecast_history, read_forecast_history_embedded, write_forecast_history_new,
    ForecastContract, ForecastReceipt, SchedulerScalarValue,
};
use crate::hrrr::{
    configuration_sha256, execution_identity_sha256, forecast_contract, map_to_cm1, Cm1MappedFrame,
    Cm1TargetGrid, DecodedHrrrWindow, HrrrLambertGrid, MappingStats, DECODED_WINDOW_SCHEMA,
    REFLECTIVITY_CONTRACT, VERTICAL_MOTION_CONTRACT,
};
use crate::json::JsonValue;
use crate::sha256::{self, Sha256};
use crate::snapshot::atomic_write_new;
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

pub const GM1_HRRR_FUTURE_BOUNDARY_SERIES_SCHEMA: &str = "gm1.hrrr-future-boundary-series/v1";
pub const GM1_HRRR_FUTURE_BOUNDARY_ABI: &str = "gm1-hrrr-future-boundaries/v1";

const MODEL: &str = "GM1";
const CLAIM_LIMIT: &str = "real HRRR f02/f03 decoded-window admission, Rust CM1-grid mapping, absolute/relative source-clock binding, deterministic CM1FHS1 publication, source reflectivity extrema, and orientation/scan proof only; native-HRRR block geometry is source-admission evidence, not an evolved 100 m GM1 field, temporal boundary interpolation, live CUDA boundary application, evolved forecast identity, or tornado detection";
const INITIAL_SOURCE_OFFSET_S: f64 = 3_600.0;
const FUTURE_HOURS: [u32; 2] = [2, 3];
const OUTPUT_NAMES: [&str; 2] = ["boundary-f02.cm1fhs", "boundary-f03.cm1fhs"];
const RAW_ROLE_SUFFIXES: [&str; 4] = [
    ".soilf02.grib2",
    ".soilf03.grib2",
    ".wrfnatf02.grib2",
    ".wrfnatf03.grib2",
];
const GATE_KEYS: [&str; 16] = [
    "status",
    "cycle",
    "valid_times",
    "forecast_hours",
    "source_forecast_hours",
    "model_forcing_hours",
    "series_count",
    "atmosphere_selected_per_time",
    "hybrid_levels",
    "soil_selected_per_time",
    "grid",
    "window_zero_based_inclusive",
    "window_shape",
    "output",
    "qice_mapping",
    "cross_time_inventory",
];
const STORM_METADATA_HEADER: &str = "index\tdiscipline\tcategory\tparameter\tcenter\tsubcenter\tmaster_table_version\tlocal_table_version\tlevel_type\tlevel_value\tsecond_level_type\tsecond_level_value\tmember\tpdt\tdrt\tnx\tny\tscan_mode\tbitmap\tcount\tfinite\tmissing\tminimum\tmaximum\tfilename";

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[derive(Clone, Debug)]
pub struct FutureBoundarySeriesRequest {
    pub raw_source_root: PathBuf,
    pub decoded_root: PathBuf,
    pub storm_source_root: PathBuf,
    pub storm_dump_f02: PathBuf,
    pub storm_dump_f03: PathBuf,
    pub baseline_f01_history: PathBuf,
    pub output_directory: PathBuf,
    pub expected_contract_sha256: String,
    pub expected_sha256s_sha256: String,
    pub expected_storm_receipt_sha256: String,
    pub expected_storm_f02_sha256: String,
    pub expected_storm_f03_sha256: String,
    pub source_grid: HrrrLambertGrid,
    pub target: Cm1TargetGrid,
    pub model_dt_s: f64,
}

#[derive(Clone, Debug)]
pub struct FutureBoundarySeriesReceipt {
    value: JsonValue,
}

impl FutureBoundarySeriesReceipt {
    pub fn canonical_json(&self) -> String {
        String::from_utf8(self.value.canonical_bytes())
            .expect("canonical future-boundary receipt is UTF-8")
    }

    pub fn sha256(&self) -> String {
        sha256::digest_hex(&self.value.canonical_bytes())
    }
}

#[derive(Clone, Debug)]
struct FileDigest {
    name: String,
    nbytes: u64,
    sha256: String,
}

#[derive(Clone, Debug)]
struct RawSourceProof {
    sha256s: FileDigest,
    fetch_manifest: FileDigest,
    files: Vec<FileDigest>,
}

#[derive(Clone, Debug)]
struct StormSourceProof {
    receipt: FileDigest,
    f02: FileDigest,
    f03: FileDigest,
}

#[derive(Clone, Debug)]
struct DecodedProof {
    gate: FileDigest,
    inventory: FileDigest,
    tree_sha256: String,
    storm_f02_metadata: FileDigest,
    storm_f03_metadata: FileDigest,
    storm_crop_sha256: String,
    orientation_scan_audit: JsonValue,
}

#[derive(Clone, Debug)]
struct DiagnosticExtrema {
    minimum: f32,
    minimum_i: usize,
    minimum_j: usize,
    minimum_lat: f64,
    minimum_lon: f64,
    maximum: f32,
    maximum_i: usize,
    maximum_j: usize,
    maximum_lat: f64,
    maximum_lon: f64,
}

#[derive(Clone, Debug)]
struct PublishedFrame {
    forecast_hour: u32,
    source_fingerprint_sha256: String,
    stats: MappingStats,
    refc: DiagnosticExtrema,
    refd_1000: DiagnosticExtrema,
    refd_4000: DiagnosticExtrema,
    receipt: ForecastReceipt,
}

struct ReceiptBuild<'a> {
    request: &'a FutureBoundarySeriesRequest,
    contract: &'a ForecastContract,
    config_sha256: &'a str,
    execution_sha256: &'a str,
    raw: &'a RawSourceProof,
    storm: &'a StormSourceProof,
    decoded: &'a DecodedProof,
    baseline: &'a ForecastReceipt,
    frames: &'a [PublishedFrame; 2],
}

pub fn publish_future_boundaries_new(
    request: &FutureBoundarySeriesRequest,
) -> Result<FutureBoundarySeriesReceipt> {
    validate_request(request)?;
    let raw = validate_raw_source(request)?;
    let storm = validate_storm_source(request)?;
    let decoded = validate_decoded_source(request)?;

    let config_sha256 =
        configuration_sha256(request.source_grid, request.target, request.model_dt_s)?;
    let execution_sha256 = execution_identity_sha256();
    let contract = forecast_contract(
        request.target,
        config_sha256.clone(),
        execution_sha256.clone(),
    )?;
    if contract.sha256() != request.expected_contract_sha256 {
        return Err(invalid(format!(
            "recomputed HRRR/GM1 contract {} does not match independently expected {}",
            contract.sha256(),
            request.expected_contract_sha256
        )));
    }
    let (_, baseline) = read_forecast_history_embedded(
        &request.baseline_f01_history,
        &request.expected_contract_sha256,
    )?;
    validate_frame_clock_and_scalars(
        baseline.frame(),
        1,
        7_200,
        INITIAL_SOURCE_OFFSET_S,
        request.model_dt_s,
    )?;

    let source2 = DecodedHrrrWindow::read_directory(
        request.decoded_root.join("atmosphere-f02"),
        request.source_grid,
        2,
    )?;
    let source3 = DecodedHrrrWindow::read_directory(
        request.decoded_root.join("atmosphere-f03"),
        request.source_grid,
        3,
    )?;
    if source2.source_fingerprint_sha256() == source3.source_fingerprint_sha256() {
        return Err(invalid(
            "f02 and f03 decoded source fingerprints unexpectedly alias",
        ));
    }

    let staging = staging_path(&request.output_directory)?;
    fs::create_dir(&staging).map_err(|error| {
        invalid(format!(
            "create future-boundary staging directory {}: {error}",
            staging.display()
        ))
    })?;
    let publication = (|| {
        let frame2 = map_publish_one(
            source2,
            &contract,
            request.target,
            request.model_dt_s,
            &staging.join(OUTPUT_NAMES[0]),
        )?;
        let frame3 = map_publish_one(
            source3,
            &contract,
            request.target,
            request.model_dt_s,
            &staging.join(OUTPUT_NAMES[1]),
        )?;
        let frames = [frame2, frame3];
        let receipt = build_receipt(ReceiptBuild {
            request,
            contract: &contract,
            config_sha256: &config_sha256,
            execution_sha256: &execution_sha256,
            raw: &raw,
            storm: &storm,
            decoded: &decoded,
            baseline: baseline.receipt(),
            frames: &frames,
        })?;
        atomic_write_new(
            &staging.join("future-boundary-series-receipt.json"),
            &receipt.value.canonical_bytes(),
        )?;
        Ok::<_, StateError>(receipt)
    })();
    match publication {
        Ok(receipt) => {
            fs::rename(&staging, &request.output_directory).map_err(|error| {
                invalid(format!(
                    "publish future-boundary directory {}: {error}",
                    request.output_directory.display()
                ))
            })?;
            Ok(receipt)
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

fn validate_request(request: &FutureBoundarySeriesRequest) -> Result<()> {
    for (label, digest) in [
        (
            "forecast contract",
            request.expected_contract_sha256.as_str(),
        ),
        ("SHA256SUMS", request.expected_sha256s_sha256.as_str()),
        (
            "storm transport receipt",
            request.expected_storm_receipt_sha256.as_str(),
        ),
        (
            "storm f02 subset",
            request.expected_storm_f02_sha256.as_str(),
        ),
        (
            "storm f03 subset",
            request.expected_storm_f03_sha256.as_str(),
        ),
    ] {
        validate_sha256(digest, label)?;
    }
    if !request.model_dt_s.is_finite() || request.model_dt_s <= 0.0 {
        return Err(invalid("model dt must be finite and positive"));
    }
    if request.output_directory.exists() {
        return Err(invalid(format!(
            "refusing to replace {}",
            request.output_directory.display()
        )));
    }
    let parent = request
        .output_directory
        .parent()
        .ok_or_else(|| invalid("future-boundary output must have a parent directory"))?;
    let metadata = fs::symlink_metadata(parent).map_err(|error| {
        invalid(format!(
            "inspect future-boundary output parent {}: {error}",
            parent.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(
            "future-boundary output parent must be a real directory",
        ));
    }
    Ok(())
}

fn validate_raw_source(request: &FutureBoundarySeriesRequest) -> Result<RawSourceProof> {
    let sums_path = request.raw_source_root.join("SHA256SUMS");
    let sha256s = hash_regular_file(&sums_path, Some("SHA256SUMS"))?;
    if sha256s.sha256 != request.expected_sha256s_sha256 {
        return Err(invalid(format!(
            "SHA256SUMS digest {} does not match expected {}",
            sha256s.sha256, request.expected_sha256s_sha256
        )));
    }
    if sha256s.nbytes > 16 * 1024 {
        return Err(invalid("SHA256SUMS exceeds the bounded text size"));
    }
    let text = fs::read_to_string(&sums_path)
        .map_err(|error| invalid(format!("read {}: {error}", sums_path.display())))?;
    let mut roles = BTreeMap::<String, (String, String)>::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let (digest, name) = line.split_once("  ").ok_or_else(|| {
            invalid(format!(
                "SHA256SUMS line {} is not '<digest><two spaces><name>'",
                line_number + 1
            ))
        })?;
        validate_sha256(digest, "SHA256SUMS entry")?;
        validate_plain_filename(name)?;
        let suffix = RAW_ROLE_SUFFIXES
            .iter()
            .find(|suffix| name.ends_with(**suffix))
            .ok_or_else(|| invalid(format!("unexpected SHA256SUMS source role {name:?}")))?;
        if roles
            .insert(
                (*suffix).to_string(),
                (name.to_string(), digest.to_string()),
            )
            .is_some()
        {
            return Err(invalid(format!(
                "duplicate SHA256SUMS source role {suffix}"
            )));
        }
    }
    if roles.len() != RAW_ROLE_SUFFIXES.len()
        || RAW_ROLE_SUFFIXES
            .iter()
            .any(|suffix| !roles.contains_key(*suffix))
    {
        return Err(invalid(
            "SHA256SUMS must contain exactly f02/f03 atmosphere and soil sources",
        ));
    }
    let mut files = Vec::new();
    for suffix in RAW_ROLE_SUFFIXES {
        let (name, expected) = roles.get(suffix).expect("role inventory checked");
        let actual = hash_regular_file(&request.raw_source_root.join(name), Some(name))?;
        if &actual.sha256 != expected {
            return Err(invalid(format!(
                "raw source {name} digest {} does not match SHA256SUMS {expected}",
                actual.sha256
            )));
        }
        files.push(actual);
    }
    let fetch_manifest = hash_regular_file(
        &request.raw_source_root.join("fetch-manifest.json"),
        Some("fetch-manifest.json"),
    )?;
    Ok(RawSourceProof {
        sha256s,
        fetch_manifest,
        files,
    })
}

fn validate_storm_source(request: &FutureBoundarySeriesRequest) -> Result<StormSourceProof> {
    let entries = exact_entry_names(&request.storm_source_root)?;
    if entries.len() != 3 || !entries.contains("download-receipt.json") {
        return Err(invalid(
            "storm source directory must contain exactly one receipt and f02/f03 subsets",
        ));
    }
    let f02_name = one_name_with_suffix(&entries, ".wrfnatf02.storm.grib2")?;
    let f03_name = one_name_with_suffix(&entries, ".wrfnatf03.storm.grib2")?;
    let receipt = hash_regular_file(
        &request.storm_source_root.join("download-receipt.json"),
        Some("download-receipt.json"),
    )?;
    let f02 = hash_regular_file(&request.storm_source_root.join(&f02_name), Some(&f02_name))?;
    let f03 = hash_regular_file(&request.storm_source_root.join(&f03_name), Some(&f03_name))?;
    for (label, actual, expected) in [
        (
            "storm transport receipt",
            receipt.sha256.as_str(),
            request.expected_storm_receipt_sha256.as_str(),
        ),
        (
            "storm f02 subset",
            f02.sha256.as_str(),
            request.expected_storm_f02_sha256.as_str(),
        ),
        (
            "storm f03 subset",
            f03.sha256.as_str(),
            request.expected_storm_f03_sha256.as_str(),
        ),
    ] {
        if actual != expected {
            return Err(invalid(format!(
                "{label} digest {actual} does not match expected {expected}"
            )));
        }
    }
    Ok(StormSourceProof { receipt, f02, f03 })
}

fn validate_decoded_source(request: &FutureBoundarySeriesRequest) -> Result<DecodedProof> {
    let expected_top = [
        "atmosphere-f02",
        "atmosphere-f03",
        "gate.txt",
        "inventory.tsv",
        "soil-f02",
        "soil-f03",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<BTreeSet<_>>();
    if exact_entry_names(&request.decoded_root)? != expected_top {
        return Err(invalid(
            "decoded root inventory is not the exact f02/f03 atmosphere, soil, gate, and inventory set",
        ));
    }
    validate_gate(&request.decoded_root.join("gate.txt"), request.source_grid)?;
    let (window_nx, window_ny) = request.source_grid.window_shape();
    let soil_bytes = window_nx
        .checked_mul(window_ny)
        .and_then(|cells| cells.checked_mul(9))
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| invalid("soil decoded byte count overflows"))? as u64;
    for hour in FUTURE_HOURS {
        let directory = request.decoded_root.join(format!("soil-f{hour:02}"));
        let expected = ["SOILT.f32le".to_string(), "SOILW.f32le".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        if exact_entry_names(&directory)? != expected {
            return Err(invalid(format!("soil f{hour:02} inventory is not exact")));
        }
        for name in ["SOILT.f32le", "SOILW.f32le"] {
            let digest = hash_regular_file(&directory.join(name), Some(name))?;
            if digest.nbytes != soil_bytes {
                return Err(invalid(format!(
                    "soil f{hour:02} {name} has {} bytes; expected {soil_bytes}",
                    digest.nbytes
                )));
            }
        }
    }
    let gate = hash_regular_file(&request.decoded_root.join("gate.txt"), Some("gate.txt"))?;
    let inventory = hash_regular_file(
        &request.decoded_root.join("inventory.tsv"),
        Some("inventory.tsv"),
    )?;
    let mut crop_hasher = Sha256::default();
    let storm_f02_metadata = validate_storm_dump(
        &request.storm_dump_f02,
        &request.decoded_root.join("atmosphere-f02"),
        request.source_grid,
        &mut crop_hasher,
    )?;
    let storm_f03_metadata = validate_storm_dump(
        &request.storm_dump_f03,
        &request.decoded_root.join("atmosphere-f03"),
        request.source_grid,
        &mut crop_hasher,
    )?;
    let tree_sha256 = fingerprint_tree(&request.decoded_root)?;
    let orientation_scan_audit =
        build_orientation_scan_audit(&request.decoded_root, request.source_grid, request.target)?;
    Ok(DecodedProof {
        gate,
        inventory,
        tree_sha256,
        storm_f02_metadata,
        storm_f03_metadata,
        storm_crop_sha256: sha256::hex(&crop_hasher.finalize()),
        orientation_scan_audit,
    })
}

fn map_publish_one(
    source: DecodedHrrrWindow,
    contract: &ForecastContract,
    target: Cm1TargetGrid,
    model_dt_s: f64,
    output: &Path,
) -> Result<PublishedFrame> {
    let mapped = map_to_cm1(&source, target)?;
    let forecast_hour = mapped.forecast_hour();
    if !FUTURE_HOURS.contains(&forecast_hour) {
        return Err(invalid(format!(
            "unexpected future source hour {forecast_hour}"
        )));
    }
    let refc = diagnostic_extrema(&mapped, target, "refc_dbz")?;
    let refd_1000 = diagnostic_extrema(&mapped, target, "refd_1000m_agl_dbz")?;
    let refd_4000 = diagnostic_extrema(&mapped, target, "refd_4000m_agl_dbz")?;
    let stats = mapped.stats().clone();
    let source_fingerprint_sha256 = mapped.source_fingerprint_sha256().to_string();
    let frame = mapped.into_forecast_frame(model_dt_s)?;
    let expected_time = f64::from(forecast_hour) * 3_600.0;
    let expected_step = (expected_time / model_dt_s).round() as u64;
    validate_frame_clock_and_scalars(
        &frame,
        forecast_hour,
        expected_step,
        expected_time,
        model_dt_s,
    )?;
    let receipt = write_forecast_history_new(contract, &frame, output)?;
    let restored = read_forecast_history(contract, output)?;
    if restored.receipt() != &receipt || !frames_identical(restored.frame(), &frame) {
        return Err(invalid(format!(
            "published f{forecast_hour:02} history failed byte/semantic restore identity"
        )));
    }
    Ok(PublishedFrame {
        forecast_hour,
        source_fingerprint_sha256,
        stats,
        refc,
        refd_1000,
        refd_4000,
        receipt,
    })
}

fn frames_identical(
    left: &crate::forecast::ForecastFrame,
    right: &crate::forecast::ForecastFrame,
) -> bool {
    left.clock() == right.clock()
        && left.arrays().len() == right.arrays().len()
        && left.scheduler_scalars().len() == right.scheduler_scalars().len()
        && left.component_blobs().len() == right.component_blobs().len()
        && right.arrays().iter().all(|expected| {
            left.arrays()
                .iter()
                .find(|actual| actual.name() == expected.name())
                .is_some_and(|actual| {
                    actual
                        .values()
                        .iter()
                        .zip(expected.values())
                        .all(|(left, right)| left.to_bits() == right.to_bits())
                        && actual.values().len() == expected.values().len()
                })
        })
        && right.scheduler_scalars().iter().all(|expected| {
            left.scheduler_scalars()
                .iter()
                .find(|actual| actual.name() == expected.name())
                .is_some_and(|actual| scalar_values_identical(actual.value(), expected.value()))
        })
        && right.component_blobs().iter().all(|expected| {
            left.component_blobs()
                .iter()
                .find(|actual| actual.name() == expected.name())
                .is_some_and(|actual| {
                    actual.schema() == expected.schema() && actual.bytes() == expected.bytes()
                })
        })
}

fn scalar_values_identical(left: SchedulerScalarValue, right: SchedulerScalarValue) -> bool {
    match (left, right) {
        (SchedulerScalarValue::I64(left), SchedulerScalarValue::I64(right)) => left == right,
        (SchedulerScalarValue::U64(left), SchedulerScalarValue::U64(right)) => left == right,
        (SchedulerScalarValue::Bool(left), SchedulerScalarValue::Bool(right)) => left == right,
        (SchedulerScalarValue::F64(left), SchedulerScalarValue::F64(right)) => {
            left.to_bits() == right.to_bits()
        }
        _ => false,
    }
}

fn validate_frame_clock_and_scalars(
    frame: &crate::forecast::ForecastFrame,
    forecast_hour: u32,
    expected_step: u64,
    expected_time_s: f64,
    expected_dt_s: f64,
) -> Result<()> {
    let clock = frame.clock();
    if clock.step() != expected_step
        || clock.model_time_s().to_bits() != expected_time_s.to_bits()
        || clock.dt_s().to_bits() != expected_dt_s.to_bits()
    {
        return Err(invalid(format!(
            "f{forecast_hour:02} absolute forecast clock does not match step={expected_step}, time={expected_time_s}, dt={expected_dt_s}"
        )));
    }
    let expected = [
        (
            "boundary_ready",
            SchedulerScalarValue::Bool(forecast_hour > 0),
        ),
        (
            "forcing_valid_offset_s",
            SchedulerScalarValue::F64(expected_time_s),
        ),
        (
            "source_forecast_hour",
            SchedulerScalarValue::U64(u64::from(forecast_hour)),
        ),
    ];
    for (name, value) in expected {
        let actual = frame
            .scheduler_scalars()
            .iter()
            .find(|scalar| scalar.name() == name)
            .ok_or_else(|| invalid(format!("f{forecast_hour:02} lacks scalar {name}")))?
            .value();
        let identical = match (actual, value) {
            (SchedulerScalarValue::Bool(left), SchedulerScalarValue::Bool(right)) => left == right,
            (SchedulerScalarValue::U64(left), SchedulerScalarValue::U64(right)) => left == right,
            (SchedulerScalarValue::F64(left), SchedulerScalarValue::F64(right)) => {
                left.to_bits() == right.to_bits()
            }
            _ => false,
        };
        if !identical {
            return Err(invalid(format!(
                "f{forecast_hour:02} scalar {name} is not the exact expected value"
            )));
        }
    }
    Ok(())
}

fn diagnostic_extrema(
    mapped: &Cm1MappedFrame,
    target: Cm1TargetGrid,
    name: &str,
) -> Result<DiagnosticExtrema> {
    let values = mapped
        .array(name)
        .ok_or_else(|| invalid(format!("mapped frame lacks diagnostic {name}")))?;
    if values.len() != target.nx * target.ny {
        return Err(invalid(format!(
            "diagnostic {name} is not the target surface shape"
        )));
    }
    let mut minimum_index = 0usize;
    let mut maximum_index = 0usize;
    for index in 1..values.len() {
        if values[index] < values[minimum_index] {
            minimum_index = index;
        }
        if values[index] > values[maximum_index] {
            maximum_index = index;
        }
    }
    let minimum_i = minimum_index % target.nx;
    let minimum_j = minimum_index / target.nx;
    let maximum_i = maximum_index % target.nx;
    let maximum_j = maximum_index / target.nx;
    let (minimum_lat, minimum_lon) = target.mass_point_lat_lon(minimum_i, minimum_j)?;
    let (maximum_lat, maximum_lon) = target.mass_point_lat_lon(maximum_i, maximum_j)?;
    Ok(DiagnosticExtrema {
        minimum: values[minimum_index],
        minimum_i,
        minimum_j,
        minimum_lat,
        minimum_lon,
        maximum: values[maximum_index],
        maximum_i,
        maximum_j,
        maximum_lat,
        maximum_lon,
    })
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("future-boundary output name is not portable UTF-8"))?;
    let parent = output
        .parent()
        .ok_or_else(|| invalid("future-boundary output lacks a parent"))?;
    let staging = parent.join(format!(".{name}.partial-{}", std::process::id()));
    if staging.exists() {
        return Err(invalid(format!(
            "stale future-boundary staging path exists: {}",
            staging.display()
        )));
    }
    Ok(staging)
}

fn validate_gate(path: &Path, grid: HrrrLambertGrid) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("inspect {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > 64 * 1024 {
        return Err(invalid("decoded gate must be a bounded regular file"));
    }
    let text = fs::read_to_string(path)
        .map_err(|error| invalid(format!("read {}: {error}", path.display())))?;
    let mut values = BTreeMap::new();
    for (line_number, line) in text.lines().enumerate() {
        let (key, value) = line.split_once('\t').ok_or_else(|| {
            invalid(format!(
                "decoded gate line {} lacks one tab separator",
                line_number + 1
            ))
        })?;
        if values.insert(key, value).is_some() {
            return Err(invalid(format!("decoded gate repeats key {key}")));
        }
    }
    let actual_keys = values.keys().copied().collect::<BTreeSet<_>>();
    let expected_keys = GATE_KEYS.into_iter().collect::<BTreeSet<_>>();
    if actual_keys != expected_keys {
        return Err(invalid("decoded gate key inventory is not exact"));
    }
    for (key, expected) in [
        ("status", "PASS"),
        ("forecast_hours", "2,3"),
        ("source_forecast_hours", "2,3"),
        ("model_forcing_hours", "0,1"),
        ("series_count", "2"),
        ("atmosphere_selected_per_time", "561"),
        ("hybrid_levels", "50"),
        ("soil_selected_per_time", "18"),
        ("window_shape", "26x26"),
        (
            "output",
            "FP32 little-endian south-to-north row-major; 3-D level-major",
        ),
    ] {
        if values.get(key).copied() != Some(expected) {
            return Err(invalid(format!(
                "decoded gate {key} is not expected value {expected:?}"
            )));
        }
    }
    let (i, j) = grid.window_origin();
    let (nx, ny) = grid.window_shape();
    let expected_window = format!("i={i}..{} j={j}..{}", i + nx - 1, j + ny - 1);
    if values.get("window_zero_based_inclusive").copied() != Some(expected_window.as_str()) {
        return Err(invalid(
            "decoded gate window does not match caller-selected source grid",
        ));
    }
    if !values
        .get("valid_times")
        .is_some_and(|value| value.contains("+02h") && value.contains("+03h"))
        || !values
            .get("qice_mapping")
            .is_some_and(|value| value.starts_with("PASS "))
        || !values
            .get("cross_time_inventory")
            .is_some_and(|value| value.starts_with("PASS "))
    {
        return Err(invalid(
            "decoded gate lacks exact f02/f03 time, QICE, or cross-time PASS evidence",
        ));
    }
    Ok(())
}

fn build_orientation_scan_audit(
    decoded_root: &Path,
    source: HrrrLambertGrid,
    target: Cm1TargetGrid,
) -> Result<JsonValue> {
    let (source_i, source_j) = source.window_origin();
    let (source_nx, source_ny) = source.window_shape();
    let source_i_end = source_i + source_nx - 1;
    let source_j_end = source_j + source_ny - 1;
    let mut source_i_checks = 0usize;
    let mut source_j_checks = 0usize;
    for local_j in 0..source_ny {
        for local_i in 0..source_nx - 1 {
            let (_, west_lon) = source
                .grid_point_lat_lon((source_i + local_i) as f64, (source_j + local_j) as f64)?;
            let (_, east_lon) = source
                .grid_point_lat_lon((source_i + local_i + 1) as f64, (source_j + local_j) as f64)?;
            if normalized_longitude_delta(east_lon, west_lon) <= 0.0 {
                return Err(invalid(
                    "source-window i axis is not strictly eastward; transpose/scan audit failed",
                ));
            }
            source_i_checks += 1;
        }
    }
    for local_j in 0..source_ny - 1 {
        for local_i in 0..source_nx {
            let (south_lat, _) = source
                .grid_point_lat_lon((source_i + local_i) as f64, (source_j + local_j) as f64)?;
            let (north_lat, _) = source
                .grid_point_lat_lon((source_i + local_i) as f64, (source_j + local_j + 1) as f64)?;
            if north_lat <= south_lat {
                return Err(invalid(
                    "source-window j axis is not strictly northward; transpose/scan audit failed",
                ));
            }
            source_j_checks += 1;
        }
    }

    let mut target_i_checks = 0usize;
    let mut target_j_checks = 0usize;
    for j in 0..target.ny {
        for i in 0..target.nx - 1 {
            let (_, west_lon) = target.mass_point_lat_lon(i, j)?;
            let (_, east_lon) = target.mass_point_lat_lon(i + 1, j)?;
            if normalized_longitude_delta(east_lon, west_lon) <= 0.0 {
                return Err(invalid(
                    "target mass-grid i axis is not strictly eastward; transpose audit failed",
                ));
            }
            target_i_checks += 1;
        }
    }
    for j in 0..target.ny - 1 {
        for i in 0..target.nx {
            let (south_lat, _) = target.mass_point_lat_lon(i, j)?;
            let (north_lat, _) = target.mass_point_lat_lon(i, j + 1)?;
            if north_lat <= south_lat {
                return Err(invalid(
                    "target mass-grid j axis is not strictly northward; transpose audit failed",
                ));
            }
            target_j_checks += 1;
        }
    }

    let source_corners = object(vec![
        (
            "northeast",
            source_point_json(source, source_i_end, source_j_end)?,
        ),
        (
            "northwest",
            source_point_json(source, source_i, source_j_end)?,
        ),
        (
            "southeast",
            source_point_json(source, source_i_end, source_j)?,
        ),
        ("southwest", source_point_json(source, source_i, source_j)?),
    ]);
    let target_corners = object(vec![
        (
            "northeast",
            target_point_json(target, target.nx - 1, target.ny - 1)?,
        ),
        ("northwest", target_point_json(target, 0, target.ny - 1)?),
        ("southeast", target_point_json(target, target.nx - 1, 0)?),
        ("southwest", target_point_json(target, 0, 0)?),
    ]);
    let source_frames = FUTURE_HOURS
        .iter()
        .map(|hour| {
            let atmosphere = decoded_root.join(format!("atmosphere-f{hour:02}"));
            Ok(object(vec![
                ("forecast_hour", number(*hour as usize)),
                (
                    "refc_dbz",
                    source_diagnostic_extrema_json(&atmosphere.join("REFC.f32le"), source)?,
                ),
                (
                    "refd_1000m_agl_dbz",
                    source_diagnostic_extrema_json(&atmosphere.join("REFD-1000.f32le"), source)?,
                ),
                (
                    "refd_4000m_agl_dbz",
                    source_diagnostic_extrema_json(&atmosphere.join("REFD-4000.f32le"), source)?,
                ),
            ]))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(object(vec![
        (
            "crop_index_contract",
            string("source offset=(j*full_nx+i)*8; output offset=(local_j*window_nx+local_i)*4; every selected f64 word is independently narrowed and raw-byte compared"),
        ),
        (
            "decoded_order",
            string("FP32 little-endian south-to-north row-major; 3-D level-major"),
        ),
        ("result", string("PASS_NO_TRANSPOSE_OR_SCAN_DISCONTINUITY")),
        ("source_corners", source_corners),
        (
            "source_i_adjacencies_checked",
            number(source_i_checks),
        ),
        (
            "source_j_adjacencies_checked",
            number(source_j_checks),
        ),
        ("source_reflectivity_extrema", JsonValue::Array(source_frames)),
        ("source_scan_mode", string("0x40")),
        ("target_corners", target_corners),
        (
            "target_i_adjacencies_checked",
            number(target_i_checks),
        ),
        (
            "target_j_adjacencies_checked",
            number(target_j_checks),
        ),
    ]))
}

fn source_diagnostic_extrema_json(path: &Path, grid: HrrrLambertGrid) -> Result<JsonValue> {
    let (window_nx, window_ny) = grid.window_shape();
    let expected_bytes = window_nx
        .checked_mul(window_ny)
        .and_then(|cells| cells.checked_mul(4))
        .ok_or_else(|| invalid("source diagnostic byte count overflows"))?;
    let bytes =
        fs::read(path).map_err(|error| invalid(format!("read {}: {error}", path.display())))?;
    if bytes.len() != expected_bytes {
        return Err(invalid(format!(
            "source diagnostic {} has {} bytes; expected {expected_bytes}",
            path.display(),
            bytes.len()
        )));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("fixed f32 word")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!(
            "source diagnostic {} contains non-finite data",
            path.display()
        )));
    }
    let mut minimum_index = 0usize;
    let mut maximum_index = 0usize;
    for index in 1..values.len() {
        if values[index] < values[minimum_index] {
            minimum_index = index;
        }
        if values[index] > values[maximum_index] {
            maximum_index = index;
        }
    }
    let (origin_i, origin_j) = grid.window_origin();
    let minimum_local_i = minimum_index % window_nx;
    let minimum_local_j = minimum_index / window_nx;
    let maximum_local_i = maximum_index % window_nx;
    let maximum_local_j = maximum_index / window_nx;
    let minimum_full_i = origin_i + minimum_local_i;
    let minimum_full_j = origin_j + minimum_local_j;
    let maximum_full_i = origin_i + maximum_local_i;
    let maximum_full_j = origin_j + maximum_local_j;
    let (minimum_lat, minimum_lon) =
        grid.grid_point_lat_lon(minimum_full_i as f64, minimum_full_j as f64)?;
    let (maximum_lat, maximum_lon) =
        grid.grid_point_lat_lon(maximum_full_i as f64, maximum_full_j as f64)?;
    Ok(object(vec![
        (
            "maximum",
            object(vec![
                ("dbz_f32_bits", f32_bits(values[maximum_index])),
                ("source_full_i", number(maximum_full_i)),
                ("source_full_j", number(maximum_full_j)),
                ("source_local_i", number(maximum_local_i)),
                ("source_local_j", number(maximum_local_j)),
                ("latitude_f64_bits", f64_bits(maximum_lat)),
                ("longitude_f64_bits", f64_bits(maximum_lon)),
            ]),
        ),
        (
            "minimum",
            object(vec![
                ("dbz_f32_bits", f32_bits(values[minimum_index])),
                ("source_full_i", number(minimum_full_i)),
                ("source_full_j", number(minimum_full_j)),
                ("source_local_i", number(minimum_local_i)),
                ("source_local_j", number(minimum_local_j)),
                ("latitude_f64_bits", f64_bits(minimum_lat)),
                ("longitude_f64_bits", f64_bits(minimum_lon)),
            ]),
        ),
    ]))
}

fn source_point_json(grid: HrrrLambertGrid, i: usize, j: usize) -> Result<JsonValue> {
    let (lat, lon) = grid.grid_point_lat_lon(i as f64, j as f64)?;
    Ok(object(vec![
        ("full_i", number(i)),
        ("full_j", number(j)),
        ("latitude_f64_bits", f64_bits(lat)),
        ("longitude_f64_bits", f64_bits(lon)),
    ]))
}

fn target_point_json(target: Cm1TargetGrid, i: usize, j: usize) -> Result<JsonValue> {
    let (lat, lon) = target.mass_point_lat_lon(i, j)?;
    Ok(object(vec![
        ("i", number(i)),
        ("j", number(j)),
        ("latitude_f64_bits", f64_bits(lat)),
        ("longitude_f64_bits", f64_bits(lon)),
    ]))
}

fn normalized_longitude_delta(value: f64, origin: f64) -> f64 {
    let mut delta = (value - origin) % 360.0;
    if delta >= 180.0 {
        delta -= 360.0;
    }
    if delta < -180.0 {
        delta += 360.0;
    }
    delta
}

fn validate_storm_dump(
    dump: &Path,
    atmosphere: &Path,
    grid: HrrrLambertGrid,
    crop_hasher: &mut Sha256,
) -> Result<FileDigest> {
    let metadata_path = dump.join("metadata.tsv");
    let metadata_digest = hash_regular_file(&metadata_path, Some("metadata.tsv"))?;
    if metadata_digest.nbytes > 256 * 1024 {
        return Err(invalid("storm dump metadata exceeds bounded size"));
    }
    let text = fs::read_to_string(&metadata_path)
        .map_err(|error| invalid(format!("read {}: {error}", metadata_path.display())))?;
    let mut lines = text.lines();
    if lines.next() != Some(STORM_METADATA_HEADER) {
        return Err(invalid("storm dump metadata header mismatch"));
    }
    let rows = lines.collect::<Vec<_>>();
    let expected_indices = (0usize..=50).chain([52usize, 53usize]).collect::<Vec<_>>();
    if rows.len() != expected_indices.len() {
        return Err(invalid(format!(
            "storm dump has {} rows; expected {}",
            rows.len(),
            expected_indices.len()
        )));
    }
    let mut expected_entries = BTreeSet::from(["metadata.tsv".to_string()]);
    let (full_nx, full_ny) = grid.full_shape();
    for (row, expected_index) in rows.iter().zip(expected_indices) {
        let columns = row.split('\t').collect::<Vec<_>>();
        if columns.len() != 25 {
            return Err(invalid("storm dump metadata row width mismatch"));
        }
        let index = parse_usize(columns[0], "storm metadata index")?;
        if index != expected_index
            || columns[1] != "0"
            || columns[4] != "7"
            || columns[5] != "0"
            || columns[6] != "2"
            || columns[7] != "1"
            || columns[10] != "255"
            || columns[11] != "0"
            || columns[12] != "-"
            || columns[13] != "0"
            || columns[14] != "3"
            || parse_usize(columns[15], "storm metadata nx")? != full_nx
            || parse_usize(columns[16], "storm metadata ny")? != full_ny
            || columns[17] != "0x40"
            || columns[18] != "false"
            || parse_usize(columns[19], "storm metadata count")? != full_nx * full_ny
            || columns[19] != columns[20]
            || columns[21] != "0"
        {
            return Err(invalid(format!(
                "storm dump metadata invariant mismatch at index {index}"
            )));
        }
        let expected_filename = format!("field-{index:04}.f64le");
        if columns[24] != expected_filename {
            return Err(invalid(format!(
                "storm dump filename mismatch at index {index}"
            )));
        }
        let output_name = match index {
            0..=49 => {
                if columns[2] != "2"
                    || columns[3] != "8"
                    || columns[8] != "105"
                    || parse_usize(columns[9], "VVEL level")? != index + 1
                {
                    return Err(invalid(format!(
                        "storm dump VVEL semantics mismatch at index {index}"
                    )));
                }
                format!("VVEL-{:02}.f32le", index + 1)
            }
            50 => {
                if columns[2] != "16"
                    || columns[3] != "196"
                    || columns[8] != "10"
                    || columns[9] != "0"
                {
                    return Err(invalid("storm dump REFC semantics mismatch"));
                }
                "REFC.f32le".to_string()
            }
            52 | 53 => {
                let level = if index == 52 { "1000" } else { "4000" };
                if columns[2] != "16"
                    || columns[3] != "195"
                    || columns[8] != "103"
                    || columns[9] != level
                {
                    return Err(invalid(format!(
                        "storm dump REFD semantics mismatch at index {index}"
                    )));
                }
                format!("REFD-{level}.f32le")
            }
            _ => unreachable!(),
        };
        expected_entries.insert(expected_filename.clone());
        verify_f64_crop(
            &dump.join(&expected_filename),
            &atmosphere.join(&output_name),
            grid,
            crop_hasher,
        )?;
    }
    if exact_entry_names(dump)? != expected_entries {
        return Err(invalid("storm dump file inventory is not exact"));
    }
    Ok(metadata_digest)
}

fn verify_f64_crop(
    source: &Path,
    output: &Path,
    grid: HrrrLambertGrid,
    aggregate: &mut Sha256,
) -> Result<()> {
    let (full_nx, full_ny) = grid.full_shape();
    let (window_i, window_j) = grid.window_origin();
    let (window_nx, window_ny) = grid.window_shape();
    let source_metadata = regular_metadata(source)?;
    let expected_source_bytes = full_nx
        .checked_mul(full_ny)
        .and_then(|cells| cells.checked_mul(8))
        .ok_or_else(|| invalid("storm f64 source size overflows"))?
        as u64;
    if source_metadata.len() != expected_source_bytes {
        return Err(invalid(format!(
            "storm f64 source {} has {} bytes; expected {expected_source_bytes}",
            source.display(),
            source_metadata.len()
        )));
    }
    let expected_output_bytes = window_nx
        .checked_mul(window_ny)
        .and_then(|cells| cells.checked_mul(4))
        .ok_or_else(|| invalid("storm f32 crop size overflows"))?;
    let output_bytes =
        fs::read(output).map_err(|error| invalid(format!("read {}: {error}", output.display())))?;
    if output_bytes.len() != expected_output_bytes {
        return Err(invalid(format!(
            "storm f32 crop {} has {} bytes; expected {expected_output_bytes}",
            output.display(),
            output_bytes.len()
        )));
    }
    let mut source_file = File::open(source)
        .map_err(|error| invalid(format!("open {}: {error}", source.display())))?;
    let mut row = vec![0u8; window_nx * 8];
    let mut expected = Vec::with_capacity(expected_output_bytes);
    for j in window_j..window_j + window_ny {
        let byte_offset = (j * full_nx + window_i)
            .checked_mul(8)
            .ok_or_else(|| invalid("storm crop seek offset overflows"))?;
        source_file
            .seek(SeekFrom::Start(byte_offset as u64))
            .and_then(|_| source_file.read_exact(&mut row))
            .map_err(|error| invalid(format!("read storm crop {}: {error}", source.display())))?;
        for word in row.chunks_exact(8) {
            let value = f64::from_le_bytes(word.try_into().expect("fixed f64 word"));
            let narrowed = value as f32;
            if !value.is_finite() || !narrowed.is_finite() {
                return Err(invalid(format!(
                    "storm crop {} contains non-finite or f32-overflowing data",
                    source.display()
                )));
            }
            expected.extend_from_slice(&narrowed.to_le_bytes());
        }
    }
    if expected != output_bytes {
        return Err(invalid(format!(
            "storm crop {} is not the exact Rust f64-to-f32 window",
            output.display()
        )));
    }
    aggregate.update(output.file_name().unwrap().to_string_lossy().as_bytes());
    aggregate.update(&[0]);
    aggregate.update(&output_bytes);
    Ok(())
}

fn regular_metadata(path: &Path) -> Result<fs::Metadata> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("inspect {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(format!(
            "{} must be a regular non-symlink file",
            path.display()
        )));
    }
    Ok(metadata)
}

fn hash_regular_file(path: &Path, receipt_name: Option<&str>) -> Result<FileDigest> {
    let metadata = regular_metadata(path)?;
    let mut stream =
        File::open(path).map_err(|error| invalid(format!("open {}: {error}", path.display())))?;
    let mut hasher = Sha256::default();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut count = 0u64;
    loop {
        let read = stream
            .read(&mut buffer)
            .map_err(|error| invalid(format!("read {}: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        count = count
            .checked_add(read as u64)
            .ok_or_else(|| invalid("file byte count overflows u64"))?;
    }
    if count != metadata.len() {
        return Err(invalid(format!(
            "{} changed length during hashing",
            path.display()
        )));
    }
    Ok(FileDigest {
        name: receipt_name
            .map(str::to_string)
            .or_else(|| {
                path.file_name()
                    .map(|value| value.to_string_lossy().into_owned())
            })
            .ok_or_else(|| invalid("hashed file lacks a name"))?,
        nbytes: count,
        sha256: sha256::hex(&hasher.finalize()),
    })
}

fn fingerprint_tree(root: &Path) -> Result<String> {
    let root_meta = fs::symlink_metadata(root)
        .map_err(|error| invalid(format!("inspect {}: {error}", root.display())))?;
    if root_meta.file_type().is_symlink() || !root_meta.is_dir() {
        return Err(invalid("decoded tree root must be a real directory"));
    }
    let mut files = Vec::new();
    collect_tree_files(root, root, &mut files)?;
    files.sort();
    let mut aggregate = Sha256::default();
    for relative in files {
        let name = relative.to_string_lossy().replace('\\', "/");
        let digest = hash_regular_file(&root.join(&relative), Some(&name))?;
        aggregate.update(&(name.len() as u64).to_le_bytes());
        aggregate.update(name.as_bytes());
        aggregate.update(&digest.nbytes.to_le_bytes());
        aggregate.update(digest.sha256.as_bytes());
    }
    Ok(sha256::hex(&aggregate.finalize()))
}

fn collect_tree_files(root: &Path, current: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(current)
        .map_err(|error| invalid(format!("read directory {}: {error}", current.display())))?
    {
        let entry = entry.map_err(|error| {
            invalid(format!(
                "read directory entry {}: {error}",
                current.display()
            ))
        })?;
        let metadata = fs::symlink_metadata(entry.path())
            .map_err(|error| invalid(format!("inspect {}: {error}", entry.path().display())))?;
        if metadata.file_type().is_symlink() {
            return Err(invalid(format!(
                "decoded tree contains symlink {}",
                entry.path().display()
            )));
        }
        if metadata.is_dir() {
            collect_tree_files(root, &entry.path(), output)?;
        } else if metadata.is_file() {
            let relative = entry
                .path()
                .strip_prefix(root)
                .map_err(|_| invalid("decoded tree path escaped root"))?
                .to_path_buf();
            output.push(relative);
        } else {
            return Err(invalid(format!(
                "decoded tree contains unsupported entry {}",
                entry.path().display()
            )));
        }
    }
    Ok(())
}

fn exact_entry_names(path: &Path) -> Result<BTreeSet<String>> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("inspect {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!(
            "{} must be a real directory",
            path.display()
        )));
    }
    fs::read_dir(path)
        .map_err(|error| invalid(format!("read directory {}: {error}", path.display())))?
        .map(|entry| {
            let entry = entry.map_err(|error| {
                invalid(format!("read directory entry {}: {error}", path.display()))
            })?;
            entry
                .file_name()
                .into_string()
                .map_err(|_| invalid("directory contains a non-UTF-8 filename"))
        })
        .collect()
}

fn one_name_with_suffix(entries: &BTreeSet<String>, suffix: &str) -> Result<String> {
    let matches = entries
        .iter()
        .filter(|name| name.ends_with(suffix))
        .cloned()
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(invalid(format!(
            "expected exactly one storm source ending {suffix:?}"
        )));
    }
    Ok(matches[0].clone())
}

fn validate_plain_filename(name: &str) -> Result<()> {
    let path = Path::new(name);
    if name.is_empty()
        || path.is_absolute()
        || path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err(invalid(format!("unsafe source filename {name:?}")));
    }
    Ok(())
}

fn parse_usize(value: &str, label: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| invalid(format!("{label} is not an unsigned integer")))
}

fn build_receipt(input: ReceiptBuild<'_>) -> Result<FutureBoundarySeriesReceipt> {
    let ReceiptBuild {
        request,
        contract,
        config_sha256,
        execution_sha256,
        raw,
        storm,
        decoded,
        baseline,
        frames,
    } = input;
    if frames[0].forecast_hour != 2 || frames[1].forecast_hour != 3 {
        return Err(invalid("future frames are not ordered f02 then f03"));
    }
    let baseline_artifact =
        hash_regular_file(&request.baseline_f01_history, Some("baseline-f01.cm1fhs"))?;
    if baseline_artifact.sha256 != baseline.artifact_sha256 {
        return Err(invalid(
            "baseline f01 file digest changed after Rust history admission",
        ));
    }
    let value = object(vec![
        (
            "authority",
            object(vec![
                ("model_source", string("CM1 r21.1")),
                ("source_role", string("upstream scientific authority only")),
            ]),
        ),
        (
            "baseline_f01",
            object(vec![
                ("artifact", file_digest_json(&baseline_artifact)),
                ("receipt", forecast_receipt_json(baseline)),
                ("source_forecast_hour", number(1)),
                ("source_offset_s_f64_bits", f64_bits(INITIAL_SOURCE_OFFSET_S)),
            ]),
        ),
        ("claim_limit", string(CLAIM_LIMIT)),
        ("config_sha256", string(config_sha256)),
        ("contract_sha256", string(&contract.sha256())),
        (
            "decoded_source",
            object(vec![
                ("decoded_window_schema", string(DECODED_WINDOW_SCHEMA)),
                ("gate", file_digest_json(&decoded.gate)),
                ("inventory", file_digest_json(&decoded.inventory)),
                (
                    "orientation_scan_audit",
                    decoded.orientation_scan_audit.clone(),
                ),
                (
                    "storm_crop_payload_sha256",
                    string(&decoded.storm_crop_sha256),
                ),
                (
                    "storm_f02_metadata",
                    file_digest_json(&decoded.storm_f02_metadata),
                ),
                (
                    "storm_f03_metadata",
                    file_digest_json(&decoded.storm_f03_metadata),
                ),
                ("tree_sha256", string(&decoded.tree_sha256)),
            ]),
        ),
        ("execution_identity_sha256", string(execution_sha256)),
        (
            "frames",
            JsonValue::Array(vec![
                published_frame_json(&frames[0], request.target, 0.0, 3_600.0)?,
                published_frame_json(&frames[1], request.target, 3_600.0, 7_200.0)?,
            ]),
        ),
        (
            "model_clock",
            object(vec![
                ("dt_s_f64_bits", f64_bits(request.model_dt_s)),
                (
                    "initialization_source_offset_s_f64_bits",
                    f64_bits(INITIAL_SOURCE_OFFSET_S),
                ),
                ("relative_model_origin_s_f64_bits", f64_bits(0.0)),
                (
                    "semantics",
                    string("absolute HRRR source clocks are preserved; relative GM1 time is source time minus the f01 initialization offset; RK stage consumers must supply their explicit clock point"),
                ),
            ]),
        ),
        ("model", string(MODEL)),
        (
            "presentation_role",
            string("source-admission evidence; native HRRR resolution/block geometry; not the user-facing evolved 100 m GM1 storm field"),
        ),
        (
            "operator_contracts",
            object(vec![
                ("reflectivity", string(REFLECTIVITY_CONTRACT)),
                ("vertical_motion", string(VERTICAL_MOTION_CONTRACT)),
            ]),
        ),
        (
            "raw_source",
            object(vec![
                ("fetch_manifest", file_digest_json(&raw.fetch_manifest)),
                (
                    "files",
                    JsonValue::Array(raw.files.iter().map(file_digest_json).collect()),
                ),
                ("sha256s", file_digest_json(&raw.sha256s)),
            ]),
        ),
        ("schema", string(GM1_HRRR_FUTURE_BOUNDARY_SERIES_SCHEMA)),
        ("status", string("PASS_REAL_HRRR_F02_F03_RUST_MAPPED")),
        (
            "storm_source",
            object(vec![
                ("f02_subset", file_digest_json(&storm.f02)),
                ("f03_subset", file_digest_json(&storm.f03)),
                ("transport_receipt", file_digest_json(&storm.receipt)),
            ]),
        ),
        ("target", target_json(request.target)),
    ]);
    Ok(FutureBoundarySeriesReceipt { value })
}

fn published_frame_json(
    frame: &PublishedFrame,
    target: Cm1TargetGrid,
    relative_start_s: f64,
    relative_end_s: f64,
) -> Result<JsonValue> {
    let absolute_end_s = f64::from(frame.forecast_hour) * 3_600.0;
    let absolute_start_s = absolute_end_s - 3_600.0;
    Ok(object(vec![
        ("artifact", forecast_receipt_json(&frame.receipt)),
        (
            "clock_interval",
            object(vec![
                ("absolute_source_end_s_f64_bits", f64_bits(absolute_end_s)),
                (
                    "absolute_source_start_s_f64_bits",
                    f64_bits(absolute_start_s),
                ),
                ("relative_model_end_s_f64_bits", f64_bits(relative_end_s)),
                (
                    "relative_model_start_s_f64_bits",
                    f64_bits(relative_start_s),
                ),
            ]),
        ),
        (
            "diagnostics",
            object(vec![
                ("refc_dbz", extrema_json(&frame.refc, target)),
                ("refd_1000m_agl_dbz", extrema_json(&frame.refd_1000, target)),
                ("refd_4000m_agl_dbz", extrema_json(&frame.refd_4000, target)),
            ]),
        ),
        ("forecast_hour", number(frame.forecast_hour as usize)),
        (
            "output_name",
            string(OUTPUT_NAMES[(frame.forecast_hour - 2) as usize]),
        ),
        (
            "source_fingerprint_sha256",
            string(&frame.source_fingerprint_sha256),
        ),
        ("stats", stats_json(&frame.stats)),
        ("target", target_json(target)),
    ]))
}

fn extrema_json(extrema: &DiagnosticExtrema, target: Cm1TargetGrid) -> JsonValue {
    let maximum_edge_distance = extrema
        .maximum_i
        .min(extrema.maximum_j)
        .min(target.nx - 1 - extrema.maximum_i)
        .min(target.ny - 1 - extrema.maximum_j);
    let minimum_edge_distance = extrema
        .minimum_i
        .min(extrema.minimum_j)
        .min(target.nx - 1 - extrema.minimum_i)
        .min(target.ny - 1 - extrema.minimum_j);
    object(vec![
        (
            "maximum",
            object(vec![
                ("dbz_f32_bits", f32_bits(extrema.maximum)),
                (
                    "distance_to_nearest_edge_cells",
                    number(maximum_edge_distance),
                ),
                ("i", number(extrema.maximum_i)),
                ("j", number(extrema.maximum_j)),
                ("latitude_f64_bits", f64_bits(extrema.maximum_lat)),
                ("longitude_f64_bits", f64_bits(extrema.maximum_lon)),
            ]),
        ),
        (
            "minimum",
            object(vec![
                ("dbz_f32_bits", f32_bits(extrema.minimum)),
                (
                    "distance_to_nearest_edge_cells",
                    number(minimum_edge_distance),
                ),
                ("i", number(extrema.minimum_i)),
                ("j", number(extrema.minimum_j)),
                ("latitude_f64_bits", f64_bits(extrema.minimum_lat)),
                ("longitude_f64_bits", f64_bits(extrema.minimum_lon)),
            ]),
        ),
    ])
}

fn stats_json(stats: &MappingStats) -> JsonValue {
    object(vec![
        (
            "nonzero_w_cells",
            JsonValue::Number(stats.nonzero_w_cells.to_string()),
        ),
        ("refc_max_dbz_f32_bits", f32_bits(stats.refc_max_dbz)),
        ("refc_min_dbz_f32_bits", f32_bits(stats.refc_min_dbz)),
        (
            "refd_1000_max_dbz_f32_bits",
            f32_bits(stats.refd_1000_max_dbz),
        ),
        (
            "refd_4000_max_dbz_f32_bits",
            f32_bits(stats.refd_4000_max_dbz),
        ),
        ("w_max_mps_f32_bits", f32_bits(stats.w_max_mps)),
        ("w_min_mps_f32_bits", f32_bits(stats.w_min_mps)),
    ])
}

fn file_digest_json(digest: &FileDigest) -> JsonValue {
    object(vec![
        ("name", string(&digest.name)),
        ("nbytes", JsonValue::Number(digest.nbytes.to_string())),
        ("sha256", string(&digest.sha256)),
    ])
}

fn forecast_receipt_json(receipt: &ForecastReceipt) -> JsonValue {
    object(vec![
        (
            "artifact_nbytes",
            JsonValue::Number(receipt.artifact_nbytes.to_string()),
        ),
        ("artifact_sha256", string(&receipt.artifact_sha256)),
        ("contract_sha256", string(&receipt.contract_sha256)),
        ("dt_f64_bits", string(&receipt.dt_f64_bits)),
        ("header_sha256", string(&receipt.header_sha256)),
        ("model_time_f64_bits", string(&receipt.model_time_f64_bits)),
        (
            "payload_nbytes",
            JsonValue::Number(receipt.payload_nbytes.to_string()),
        ),
        ("payload_sha256", string(&receipt.payload_sha256)),
        ("step", JsonValue::Number(receipt.step.to_string())),
    ])
}

fn target_json(target: Cm1TargetGrid) -> JsonValue {
    object(vec![
        ("center_lat_f64_bits", f64_bits(target.center_lat_deg)),
        ("center_lon_f64_bits", f64_bits(target.center_lon_deg)),
        ("dx_m_f64_bits", f64_bits(target.dx_m)),
        ("dy_m_f64_bits", f64_bits(target.dy_m)),
        ("dz_m_f64_bits", f64_bits(target.dz_m)),
        ("nx", number(target.nx)),
        ("ny", number(target.ny)),
        ("nz", number(target.nz)),
    ])
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!(
            "{label} must be 64 lowercase hexadecimal characters"
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

fn f32_bits(value: f32) -> JsonValue {
    string(&format!("{:08x}", value.to_bits()))
}

fn f64_bits(value: f64) -> JsonValue {
    string(&format!("{:016x}", value.to_bits()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_clock_intervals_are_absolute_and_run_relative() {
        assert_eq!(INITIAL_SOURCE_OFFSET_S.to_bits(), 3_600.0_f64.to_bits());
        let f02_absolute = f64::from(FUTURE_HOURS[0]) * 3_600.0;
        let f03_absolute = f64::from(FUTURE_HOURS[1]) * 3_600.0;
        assert_eq!(f02_absolute.to_bits(), 7_200.0_f64.to_bits());
        assert_eq!(f03_absolute.to_bits(), 10_800.0_f64.to_bits());
        assert_eq!(
            (f02_absolute - INITIAL_SOURCE_OFFSET_S).to_bits(),
            3_600.0_f64.to_bits()
        );
        assert_eq!(
            (f03_absolute - INITIAL_SOURCE_OFFSET_S).to_bits(),
            7_200.0_f64.to_bits()
        );
    }

    #[test]
    fn sha_and_plain_filename_validation_fail_closed() {
        assert!(validate_sha256(&"a".repeat(64), "test").is_ok());
        assert!(validate_sha256(&"A".repeat(64), "test").is_err());
        assert!(validate_sha256("short", "test").is_err());
        assert!(validate_plain_filename("one.grib2").is_ok());
        assert!(validate_plain_filename("../one.grib2").is_err());
        assert!(validate_plain_filename("dir/one.grib2").is_err());
    }

    #[test]
    fn strict_storm_roles_do_not_admit_core_only_inventory() {
        let core_only = BTreeSet::from(["download-receipt.json".to_string()]);
        assert!(one_name_with_suffix(&core_only, ".wrfnatf02.storm.grib2").is_err());
        let complete = BTreeSet::from([
            "download-receipt.json".to_string(),
            "cycle.wrfnatf02.storm.grib2".to_string(),
            "cycle.wrfnatf03.storm.grib2".to_string(),
        ]);
        assert_eq!(
            one_name_with_suffix(&complete, ".wrfnatf02.storm.grib2").unwrap(),
            "cycle.wrfnatf02.storm.grib2"
        );
    }

    #[test]
    fn longitude_direction_is_continuous_across_the_dateline() {
        assert_eq!(normalized_longitude_delta(-179.0, 179.0), 2.0);
        assert_eq!(normalized_longitude_delta(179.0, -179.0), -2.0);
        assert!(normalized_longitude_delta(-94.0, -95.0) > 0.0);
    }
}
