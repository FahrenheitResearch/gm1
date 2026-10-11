//! Additive production-science transport derived from one authenticated history read.
//!
//! The v1 `qc + qi` pack remains unchanged. This v2 sibling carries its exact
//! cloud payload plus source words needed for scientifically honest wind,
//! pressure, terrain, and height views. Reflectivity is deliberately absent
//! until a source carrier or a separately admitted source-exact diagnostic is
//! available.

use super::{
    CHILD_VIEW_FORMULA, CHILD_VIEW_PACK_NX, CHILD_VIEW_PACK_NY, CHILD_VIEW_PACK_NZ,
    CHILD_VIEW_PACK_PAYLOAD_FILE, MAX_EXECUTABLE_NBYTES, MAX_JSON_NBYTES, MAX_SOURCE_NBYTES,
    PackBuild, SourceIdentity, TEMP_COUNTER, build_pack, hash_f32_words, invalid, number,
    number_array, object, read_regular_bounded, report_stage_timings, source_value, string,
    string_array, validate_commit, validate_sha256, verify_inventory, write_new_synced,
};
use crate::Result;
use crate::forecast::{ForecastContract, RestoredForecast};
use crate::json::JsonValue;
use crate::sha256;
use std::collections::BTreeSet;
use std::fs;
#[cfg(unix)]
use std::fs::File;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

pub const CHILD_PRODUCTION_VIEW_PACK_MANIFEST_SCHEMA: &str = "gm1.production-view-pack-manifest/v2";
pub const CHILD_PRODUCTION_VIEW_PACK_RECEIPT_SCHEMA: &str = "gm1.production-view-pack-receipt/v2";
pub const CHILD_PRODUCTION_VIEW_PACK_ABI: &str = "gm1.production-view-multifield-f32le-c-order/v2";
const CHILD_PRODUCTION_VIEW_PACK_STATUS: &str =
    "PASS_EXACT_MULTI_FIELD_VIEW_TRANSPORT_NO_REFLECTIVITY";
const CHILD_PRODUCTION_VIEW_PACK_CLAIM_LIMIT: &str = "one-pass additive, non-quantized binary32 view transport from one independently authenticated GM1 CM1FHS1 history: exact qc+qi plus unmodified u3d/v3d/w3d/prs/terrain_msl/zf/total_geopotential_zf source words; raw staggered winds enable a separately pinned exact updraft/vorticity derivation, but this pack contains no reflectivity carrier, invented boundary values, checkpoint, restart, independent meteorological product, forecast-skill, storm, tornado, or performance claim";
const MANIFEST_FILE: &str = "manifest.json";
const RECEIPT_FILE: &str = "receipt.json";
const RAW_COPY_FORMULA: &str = "out[word]=source_carrier[word] (IEEE-754 binary32 word exact)";

#[derive(Clone, Copy, Debug)]
struct RawChannelSpec {
    name: &'static str,
    file: &'static str,
    source_carrier: &'static str,
    role: &'static str,
    units: &'static str,
    stagger: &'static str,
    shape: &'static [usize],
}

const RAW_CHANNEL_SPECS: [RawChannelSpec; 7] = [
    RawChannelSpec {
        name: "u_staggered_ms",
        file: "u3d.f32le",
        source_carrier: "u3d",
        role: "eastward wind on native x faces",
        units: "m s-1",
        stagger: "x-face C-grid",
        shape: &[75, 300, 301],
    },
    RawChannelSpec {
        name: "v_staggered_ms",
        file: "v3d.f32le",
        source_carrier: "v3d",
        role: "northward wind on native y faces",
        units: "m s-1",
        stagger: "y-face C-grid",
        shape: &[75, 301, 300],
    },
    RawChannelSpec {
        name: "w_staggered_ms",
        file: "w3d.f32le",
        source_carrier: "w3d",
        role: "vertical velocity on native z faces",
        units: "m s-1",
        stagger: "z-face C-grid",
        shape: &[76, 300, 300],
    },
    RawChannelSpec {
        name: "pressure_pa",
        file: "prs.f32le",
        source_carrier: "prs",
        role: "physical pressure",
        units: "Pa",
        stagger: "mass",
        shape: &[75, 300, 300],
    },
    RawChannelSpec {
        name: "terrain_msl_m",
        file: "terrain_msl.f32le",
        source_carrier: "terrain_msl",
        role: "terrain height above mean sea level",
        units: "m",
        stagger: "horizontal mass",
        shape: &[300, 300],
    },
    RawChannelSpec {
        name: "static_zf_m",
        file: "zf.f32le",
        source_carrier: "zf",
        role: "static geometric height on vertical faces",
        units: "m",
        stagger: "z-face mass column",
        shape: &[76, 300, 300],
    },
    RawChannelSpec {
        name: "total_geopotential_zf_m",
        file: "total-geopotential-zf.f32le",
        source_carrier: "total_geopotential_zf",
        role: "terrain-following total geopotential height on vertical faces",
        units: "m",
        stagger: "z-face mass column",
        shape: &[76, 300, 300],
    },
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildProductionViewPackReceipt {
    pub manifest_sha256: String,
    pub payload_set_sha256: String,
    pub receipt_sha256: String,
    pub channel_sha256: Vec<(String, String)>,
    pub step: u64,
    pub model_time_f64_bits: String,
}

#[derive(Clone, Debug)]
struct ChannelBuild {
    name: &'static str,
    file: &'static str,
    role: &'static str,
    units: &'static str,
    stagger: &'static str,
    shape: Vec<usize>,
    source_carriers: Vec<&'static str>,
    source_sha256: Vec<String>,
    formula: &'static str,
    payload: Vec<u8>,
    payload_sha256: String,
    minimum_f32_bits: u32,
    maximum_f32_bits: u32,
    nonzero_count: usize,
    nonnegative: bool,
}

#[derive(Clone, Debug)]
struct ProductionBuild {
    channels: Vec<ChannelBuild>,
    payload_set_sha256: String,
    source: SourceIdentity,
    timings: super::PackValidationTimings,
    science_derive: Duration,
}

#[allow(clippy::too_many_arguments)]
pub fn publish_child_production_view_pack_new(
    pair_directory: impl AsRef<Path>,
    expected_pair_manifest_sha256: &str,
    expected_pair_receipt_sha256: &str,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    producer_commit: &str,
    output_directory: impl AsRef<Path>,
) -> Result<ChildProductionViewPackReceipt> {
    let operation_started = Instant::now();
    let producer_source_sha256 = super::hash_regular_bounded(
        producer_source.as_ref(),
        MAX_SOURCE_NBYTES,
        "production View-pack producer source",
    )?;
    let producer_executable_sha256 = super::hash_regular_bounded(
        producer_executable.as_ref(),
        MAX_EXECUTABLE_NBYTES,
        "production View-pack producer executable",
    )?;
    validate_commit(producer_commit)?;
    let build = build_production_pack(build_pack(
        pair_directory.as_ref(),
        expected_pair_manifest_sha256,
        expected_pair_receipt_sha256,
    )?)?;
    let manifest = manifest_value(
        &build,
        &producer_source_sha256,
        &producer_executable_sha256,
        producer_commit,
    )
    .canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest);
    let receipt = receipt_value(
        &build,
        &manifest_sha256,
        &producer_source_sha256,
        &producer_executable_sha256,
        producer_commit,
    )
    .canonical_bytes();
    let receipt_sha256 = sha256::digest_hex(&receipt);
    let terminal_started = Instant::now();
    publish_production_directory(
        output_directory.as_ref(),
        &build.channels,
        &manifest,
        &receipt,
    )?;
    report_production_timings(
        "publish-production",
        &build,
        "atomic_publish",
        terminal_started.elapsed(),
        operation_started.elapsed(),
    );
    Ok(public_receipt(&build, manifest_sha256, receipt_sha256))
}

#[allow(clippy::too_many_arguments)]
pub fn inspect_child_production_view_pack(
    pair_directory: impl AsRef<Path>,
    expected_pair_manifest_sha256: &str,
    expected_pair_receipt_sha256: &str,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    producer_commit: &str,
    directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_payload_set_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<ChildProductionViewPackReceipt> {
    let operation_started = Instant::now();
    for (value, label) in [
        (
            expected_manifest_sha256,
            "expected production View-pack manifest SHA-256",
        ),
        (
            expected_payload_set_sha256,
            "expected production View-pack payload-set SHA-256",
        ),
        (
            expected_receipt_sha256,
            "expected production View-pack receipt SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    let producer_source_sha256 = super::hash_regular_bounded(
        producer_source.as_ref(),
        MAX_SOURCE_NBYTES,
        "production View-pack producer source",
    )?;
    let producer_executable_sha256 = super::hash_regular_bounded(
        producer_executable.as_ref(),
        MAX_EXECUTABLE_NBYTES,
        "production View-pack producer executable",
    )?;
    validate_commit(producer_commit)?;
    let build = build_production_pack(build_pack(
        pair_directory.as_ref(),
        expected_pair_manifest_sha256,
        expected_pair_receipt_sha256,
    )?)?;
    if build.payload_set_sha256 != expected_payload_set_sha256 {
        return Err(invalid(format!(
            "independently recomputed production View-pack payload set is {}; expected {expected_payload_set_sha256}",
            build.payload_set_sha256
        )));
    }
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
    verify_inventory(
        directory,
        &production_inventory(&build.channels),
        "production View-pack",
    )?;
    for channel in &build.channels {
        let actual = read_regular_bounded(
            &directory.join(channel.file),
            channel.payload.len() as u64,
            &format!("production View-pack channel {}", channel.name),
        )?;
        if actual != channel.payload {
            return Err(invalid(format!(
                "production View-pack channel {} differs from independent source recomputation",
                channel.name
            )));
        }
    }
    let actual_manifest = read_regular_bounded(
        &directory.join(MANIFEST_FILE),
        MAX_JSON_NBYTES,
        "production View-pack manifest",
    )?;
    let actual_receipt = read_regular_bounded(
        &directory.join(RECEIPT_FILE),
        MAX_JSON_NBYTES,
        "production View-pack receipt",
    )?;
    if actual_manifest != expected_manifest || actual_receipt != expected_receipt {
        return Err(invalid(
            "production View-pack manifest or receipt differs from independent recomputation",
        ));
    }
    for (actual, expected, label) in [
        (
            sha256::digest_hex(&actual_manifest),
            expected_manifest_sha256,
            "production View-pack manifest",
        ),
        (
            sha256::digest_hex(&actual_receipt),
            expected_receipt_sha256,
            "production View-pack receipt",
        ),
    ] {
        if actual != expected {
            return Err(invalid(format!(
                "{label} SHA-256 is {actual}; expected {expected}"
            )));
        }
    }
    report_production_timings(
        "inspect-production",
        &build,
        "pack_compare",
        terminal_started.elapsed(),
        operation_started.elapsed(),
    );
    Ok(public_receipt(
        &build,
        actual_manifest_sha256,
        sha256::digest_hex(&actual_receipt),
    ))
}

fn build_production_pack(base: PackBuild) -> Result<ProductionBuild> {
    let started = Instant::now();
    let PackBuild {
        contract,
        restored,
        payload,
        payload_sha256,
        minimum_f32_bits,
        maximum_f32_bits,
        nonzero_count,
        source,
        timings,
    } = base;
    let mut channels = Vec::with_capacity(1 + RAW_CHANNEL_SPECS.len());
    channels.push(ChannelBuild {
        name: "cloud_condensate_kgkg",
        file: CHILD_VIEW_PACK_PAYLOAD_FILE,
        role: "exact qc plus qi cloud condensate",
        units: "kg kg-1",
        stagger: "mass",
        shape: vec![CHILD_VIEW_PACK_NZ, CHILD_VIEW_PACK_NY, CHILD_VIEW_PACK_NX],
        source_carriers: vec!["q3d.qc", "q3d.qi"],
        source_sha256: vec![source.qc_sha256.clone(), source.qi_sha256.clone()],
        formula: CHILD_VIEW_FORMULA,
        payload,
        payload_sha256,
        minimum_f32_bits,
        maximum_f32_bits,
        nonzero_count,
        nonnegative: true,
    });
    for spec in RAW_CHANNEL_SPECS {
        channels.push(raw_channel(&contract, &restored, spec)?);
    }
    let payload_set_sha256 = payload_set_sha256(&channels);
    Ok(ProductionBuild {
        channels,
        payload_set_sha256,
        source,
        timings,
        science_derive: started.elapsed(),
    })
}

fn raw_channel(
    contract: &ForecastContract,
    restored: &RestoredForecast,
    spec: RawChannelSpec,
) -> Result<ChannelBuild> {
    let contracted = contract
        .arrays()
        .iter()
        .find(|candidate| candidate.name() == spec.source_carrier)
        .ok_or_else(|| {
            invalid(format!(
                "source history contract lacks {}",
                spec.source_carrier
            ))
        })?;
    if contracted.shape() != spec.shape {
        return Err(invalid(format!(
            "source {} shape is {:?}; expected {:?}",
            spec.source_carrier,
            contracted.shape(),
            spec.shape
        )));
    }
    let values = restored
        .frame()
        .arrays()
        .iter()
        .find(|candidate| candidate.name() == spec.source_carrier)
        .ok_or_else(|| {
            invalid(format!(
                "source history payload lacks {}",
                spec.source_carrier
            ))
        })?
        .values();
    let expected_words = spec.shape.iter().try_fold(1usize, |count, extent| {
        count
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("{} shape product overflows", spec.name)))
    })?;
    let (payload, minimum_f32_bits, maximum_f32_bits, nonzero_count, nonnegative) =
        encode_finite_words(values, expected_words, spec.name)?;
    let source_sha256 = hash_f32_words(values);
    let payload_sha256 = sha256::digest_hex(&payload);
    if payload_sha256 != source_sha256 {
        return Err(invalid(format!(
            "raw channel {} did not preserve source binary32 words",
            spec.name
        )));
    }
    Ok(ChannelBuild {
        name: spec.name,
        file: spec.file,
        role: spec.role,
        units: spec.units,
        stagger: spec.stagger,
        shape: spec.shape.to_vec(),
        source_carriers: vec![spec.source_carrier],
        source_sha256: vec![source_sha256],
        formula: RAW_COPY_FORMULA,
        payload,
        payload_sha256,
        minimum_f32_bits,
        maximum_f32_bits,
        nonzero_count,
        nonnegative,
    })
}

fn encode_finite_words(
    values: &[f32],
    expected_words: usize,
    label: &str,
) -> Result<(Vec<u8>, u32, u32, usize, bool)> {
    if values.len() != expected_words {
        return Err(invalid(format!(
            "{label} contains {} words; expected {expected_words}",
            values.len()
        )));
    }
    let mut payload = Vec::with_capacity(expected_words * 4);
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut nonzero_count = 0usize;
    let mut nonnegative = true;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(invalid(format!("{label} word {index} is non-finite")));
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        nonzero_count += usize::from(value.to_bits() & 0x7fff_ffff != 0);
        nonnegative &= value >= 0.0;
        payload.extend_from_slice(&value.to_le_bytes());
    }
    Ok((
        payload,
        minimum.to_bits(),
        maximum.to_bits(),
        nonzero_count,
        nonnegative,
    ))
}

fn manifest_value(
    build: &ProductionBuild,
    producer_source_sha256: &str,
    producer_executable_sha256: &str,
    producer_commit: &str,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_PRODUCTION_VIEW_PACK_ABI)),
        ("channels", channels_value(&build.channels)),
        (
            "claim_limit",
            string(CHILD_PRODUCTION_VIEW_PACK_CLAIM_LIMIT),
        ),
        ("model", string("GM1")),
        ("optional_channels", optional_channels_value()),
        ("payload_set_sha256", string(&build.payload_set_sha256)),
        (
            "producer",
            producer_value(
                producer_source_sha256,
                producer_executable_sha256,
                producer_commit,
            ),
        ),
        (
            "qc_plus_qi_v1_payload_sha256",
            string(&build.channels[0].payload_sha256),
        ),
        ("schema", string(CHILD_PRODUCTION_VIEW_PACK_MANIFEST_SCHEMA)),
        ("source", source_value(&build.source)),
        ("status", string(CHILD_PRODUCTION_VIEW_PACK_STATUS)),
        ("synthetic_detail", JsonValue::Bool(false)),
    ])
}

fn receipt_value(
    build: &ProductionBuild,
    manifest_sha256: &str,
    producer_source_sha256: &str,
    producer_executable_sha256: &str,
    producer_commit: &str,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_PRODUCTION_VIEW_PACK_ABI)),
        ("channels", channel_inventory_value(&build.channels)),
        (
            "claim_limit",
            string(CHILD_PRODUCTION_VIEW_PACK_CLAIM_LIMIT),
        ),
        ("manifest_sha256", string(manifest_sha256)),
        ("model", string("GM1")),
        ("payload_set_sha256", string(&build.payload_set_sha256)),
        (
            "producer",
            producer_value(
                producer_source_sha256,
                producer_executable_sha256,
                producer_commit,
            ),
        ),
        ("schema", string(CHILD_PRODUCTION_VIEW_PACK_RECEIPT_SCHEMA)),
        ("source", source_value(&build.source)),
        ("status", string(CHILD_PRODUCTION_VIEW_PACK_STATUS)),
        ("synthetic_detail", JsonValue::Bool(false)),
    ])
}

fn channels_value(channels: &[ChannelBuild]) -> JsonValue {
    JsonValue::Array(
        channels
            .iter()
            .map(|channel| {
                object(vec![
                    ("dtype", string("little-endian-f32")),
                    ("file", string(channel.file)),
                    ("formula", string(channel.formula)),
                    ("name", string(channel.name)),
                    ("nbytes", number(channel.payload.len())),
                    ("order", string("C")),
                    ("payload_sha256", string(&channel.payload_sha256)),
                    ("role", string(channel.role)),
                    ("shape", number_array(&channel.shape)),
                    ("source_carriers", string_array(&channel.source_carriers)),
                    (
                        "source_sha256",
                        JsonValue::Array(
                            channel
                                .source_sha256
                                .iter()
                                .map(|value| string(value))
                                .collect(),
                        ),
                    ),
                    ("stagger", string(channel.stagger)),
                    (
                        "statistics",
                        object(vec![
                            ("all_finite", JsonValue::Bool(true)),
                            (
                                "maximum_f32_bits",
                                string(&format!("{:08x}", channel.maximum_f32_bits)),
                            ),
                            (
                                "minimum_f32_bits",
                                string(&format!("{:08x}", channel.minimum_f32_bits)),
                            ),
                            ("nonnegative", JsonValue::Bool(channel.nonnegative)),
                            ("nonzero_count", number(channel.nonzero_count)),
                            ("word_count", number(channel.payload.len() / 4)),
                        ]),
                    ),
                    ("units", string(channel.units)),
                ])
            })
            .collect(),
    )
}

fn channel_inventory_value(channels: &[ChannelBuild]) -> JsonValue {
    JsonValue::Array(
        channels
            .iter()
            .map(|channel| {
                object(vec![
                    ("file", string(channel.file)),
                    ("name", string(channel.name)),
                    ("nbytes", number(channel.payload.len())),
                    ("payload_sha256", string(&channel.payload_sha256)),
                    ("shape", number_array(&channel.shape)),
                ])
            })
            .collect(),
    )
}

fn payload_set_sha256(channels: &[ChannelBuild]) -> String {
    sha256::digest_hex(&channel_inventory_value(channels).canonical_bytes())
}

fn optional_channels_value() -> JsonValue {
    object(vec![
        (
            "reflectivity_dbz",
            object(vec![
                (
                    "reason",
                    string(
                        "the admitted 13-carrier history contains no reflectivity carrier and no source-exact reflectivity diagnostic is bound",
                    ),
                ),
                ("status", string("MISSING_SOURCE_CARRIER")),
            ]),
        ),
        (
            "updraft_mass_center_ms",
            object(vec![
                (
                    "reason",
                    string(
                        "exact native w3d z-face words are present; mass-center derivation is deferred and must pin its formula",
                    ),
                ),
                ("status", string("DERIVABLE_FROM_EXACT_W3D")),
            ]),
        ),
        (
            "vertical_vorticity_s",
            object(vec![
                (
                    "reason",
                    string(
                        "exact native u3d/v3d face words are present; C-grid vorticity and boundary disposition are deferred and must pin their formula",
                    ),
                ),
                ("status", string("DERIVABLE_FROM_EXACT_U3D_V3D")),
            ]),
        ),
    ])
}

fn producer_value(source: &str, executable: &str, commit: &str) -> JsonValue {
    object(vec![
        ("commit", string(commit)),
        ("executable_sha256", string(executable)),
        ("source_sha256", string(source)),
    ])
}

fn public_receipt(
    build: &ProductionBuild,
    manifest_sha256: String,
    receipt_sha256: String,
) -> ChildProductionViewPackReceipt {
    ChildProductionViewPackReceipt {
        manifest_sha256,
        payload_set_sha256: build.payload_set_sha256.clone(),
        receipt_sha256,
        channel_sha256: build
            .channels
            .iter()
            .map(|channel| (channel.name.to_owned(), channel.payload_sha256.clone()))
            .collect(),
        step: build.source.step,
        model_time_f64_bits: build.source.model_time_f64_bits.clone(),
    }
}

fn report_production_timings(
    operation: &str,
    build: &ProductionBuild,
    terminal_label: &str,
    terminal: Duration,
    operation_total: Duration,
) {
    report_stage_timings(
        operation,
        &build.timings,
        terminal_label,
        terminal,
        operation_total,
    );
    eprintln!(
        "gm1-child-view-pack: timings operation={operation} production_science_derive_seconds={:.6} channel_count={} payload_bytes={}",
        build.science_derive.as_secs_f64(),
        build.channels.len(),
        build
            .channels
            .iter()
            .map(|channel| channel.payload.len() as u64)
            .sum::<u64>(),
    );
}

fn publish_production_directory(
    output: &Path,
    channels: &[ChannelBuild],
    manifest: &[u8],
    receipt: &[u8],
) -> Result<()> {
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace production View-pack destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "production View-pack destination parent does not exist: {}",
            parent.display()
        )));
    }
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("production View-pack destination needs a UTF-8 name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let staging =
        output.with_file_name(format!(".{name}.partial-{}-{sequence}", std::process::id()));
    fs::create_dir(&staging)?;
    let inventory = production_inventory(channels);
    let result = (|| -> Result<()> {
        for channel in channels {
            write_new_synced(&staging.join(channel.file), &channel.payload)?;
        }
        write_new_synced(&staging.join(MANIFEST_FILE), manifest)?;
        write_new_synced(&staging.join(RECEIPT_FILE), receipt)?;
        verify_inventory(&staging, &inventory, "staged production View-pack")?;
        fs::rename(&staging, output)?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        for file in &inventory {
            let _ = fs::remove_file(staging.join(file));
        }
        let _ = fs::remove_dir(&staging);
    }
    result
}

fn production_inventory(channels: &[ChannelBuild]) -> BTreeSet<String> {
    let mut files = channels
        .iter()
        .map(|channel| channel.file.to_owned())
        .collect::<BTreeSet<_>>();
    files.insert(MANIFEST_FILE.to_owned());
    files.insert(RECEIPT_FILE.to_owned());
    files
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn dummy_channel() -> ChannelBuild {
        ChannelBuild {
            name: "dummy",
            file: "dummy.f32le",
            role: "test",
            units: "1",
            stagger: "mass",
            shape: vec![1],
            source_carriers: vec!["dummy"],
            source_sha256: vec![sha256::digest_hex(&0.0f32.to_le_bytes())],
            formula: RAW_COPY_FORMULA,
            payload: 0.0f32.to_le_bytes().to_vec(),
            payload_sha256: sha256::digest_hex(&0.0f32.to_le_bytes()),
            minimum_f32_bits: 0,
            maximum_f32_bits: 0,
            nonzero_count: 0,
            nonnegative: true,
        }
    }

    #[test]
    fn production_inventory_is_additive_and_reflectivity_is_explicitly_missing() {
        assert_eq!(RAW_CHANNEL_SPECS.len() + 1, 8);
        assert_eq!(RAW_CHANNEL_SPECS[0].shape, &[75, 300, 301]);
        assert_eq!(RAW_CHANNEL_SPECS[1].shape, &[75, 301, 300]);
        assert_eq!(RAW_CHANNEL_SPECS[2].shape, &[76, 300, 300]);
        assert!(
            RAW_CHANNEL_SPECS
                .iter()
                .any(|spec| spec.source_carrier == "prs")
        );
        assert!(
            RAW_CHANNEL_SPECS
                .iter()
                .any(|spec| spec.source_carrier == "terrain_msl")
        );
        let optional = optional_channels_value().canonical_bytes();
        let text = std::str::from_utf8(&optional).unwrap();
        assert!(text.contains("MISSING_SOURCE_CARRIER"));
        assert!(text.contains("DERIVABLE_FROM_EXACT_U3D_V3D"));
        assert!(text.contains("DERIVABLE_FROM_EXACT_W3D"));
    }

    #[test]
    fn raw_encoder_is_bit_exact_and_refuses_foreign_words() {
        let values = [f32::from_bits(0x8000_0000), 1.25, -2.5];
        let (payload, minimum, maximum, nonzero, nonnegative) =
            encode_finite_words(&values, values.len(), "test").unwrap();
        assert_eq!(
            payload,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(minimum, (-2.5f32).to_bits());
        assert_eq!(maximum, 1.25f32.to_bits());
        assert_eq!(nonzero, 2);
        assert!(!nonnegative);
        assert!(encode_finite_words(&values, 2, "test").is_err());
        assert!(encode_finite_words(&[f32::NAN], 1, "test").is_err());
    }

    #[test]
    fn production_directory_is_atomic_exact_inventory_and_no_replace() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("gm1-production-view-pack-{nonce}"));
        fs::create_dir(&root).unwrap();
        let output = root.join("step-00000001");
        let channels = vec![dummy_channel()];
        publish_production_directory(&output, &channels, b"{}", b"{}").unwrap();
        verify_inventory(
            &output,
            &production_inventory(&channels),
            "test production pack",
        )
        .unwrap();
        assert!(publish_production_directory(&output, &channels, b"{}", b"{}").is_err());
        let extra = output.join("foreign");
        fs::write(&extra, b"bad").unwrap();
        assert!(
            verify_inventory(
                &output,
                &production_inventory(&channels),
                "test production pack"
            )
            .is_err()
        );
        fs::remove_file(extra).unwrap();
        for file in production_inventory(&channels) {
            fs::remove_file(output.join(file)).unwrap();
        }
        fs::remove_dir(output).unwrap();
        fs::remove_dir(root).unwrap();
    }
}
