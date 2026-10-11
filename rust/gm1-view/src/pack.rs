//! Strict admission for the atomic, non-quantized GM1 cloud render transport.
//!
//! This is a display transport for exact `qc + qi` binary32 words.  The source
//! CM1FHS history remains the scientific state and is independently pinned by
//! every admitted directory.

use crate::input::{Gm1Dataset, VolumeField};
use cm1_state::child_view_pack::{
    CHILD_VIEW_FORMULA, CHILD_VIEW_PACK_ABI, CHILD_VIEW_PACK_CLAIM_LIMIT,
    CHILD_VIEW_PACK_MANIFEST_FILE, CHILD_VIEW_PACK_MANIFEST_SCHEMA, CHILD_VIEW_PACK_NBYTES,
    CHILD_VIEW_PACK_NX, CHILD_VIEW_PACK_NY, CHILD_VIEW_PACK_NZ, CHILD_VIEW_PACK_PAYLOAD_FILE,
    CHILD_VIEW_PACK_RECEIPT_FILE, CHILD_VIEW_PACK_RECEIPT_SCHEMA, CHILD_VIEW_PACK_STATUS,
    CHILD_VIEW_PACK_WORDS, CHILD_VIEW_Q3D_NBYTES, CHILD_VIEW_SPECIES,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::Path, time::Instant};

pub const ADMITTED_PRODUCER_COMMIT: &str = "5eb5de9c034fffc50ac8c3cbcbd88b899b0b3583";
pub const ADMITTED_PRODUCER_SOURCE_SHA256: &str =
    "9296e55e4628bb1330d2110853a8e9733d23a0c5aafcfaceb6085c199018ed95";
pub const ADMITTED_PRODUCER_EXECUTABLE_SHA256: &str =
    "044bc8caecc09b81cec388119edab6f5a9642e04b74a4f633c99e96f4333a7bd";
pub const REBUILT_PRODUCER_EXECUTABLE_SHA256: &str =
    "968f420f267e825f104a1b3b43f820979008228f53bc676adeffbce7a36c32a1";
pub const FINAL_STEP_PRODUCER_COMMIT: &str = "e373c3bb57d45fe5bcf8ea3ef70b2640aa0492d4";
pub const FINAL_STEP_PRODUCER_SOURCE_SHA256: &str =
    "eb8b5cfe107fd063e709f0de8eef2421d908ce1c5e42ba0e3260c6d71aa3aed9";
pub const FINAL_STEP_PRODUCER_EXECUTABLE_SHA256: &str =
    "02ea9a57ece25cdfe1a11c1760f980b27f89a612a8c26f8f26c599ac54f957e9";
pub const LIVE_SERIES_PRODUCER_COMMIT: &str = "eff2bec1ab48f31067adea7d543c7d46ce460a1b";
pub const LIVE_SERIES_PRODUCER_SOURCE_SHA256: &str =
    "bad7907154a7867086ce2505b3454d144bfab4a2c371af34a45237c93e8db25d";
pub const LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256: &str =
    "9c90b643bb311fdd38e56d0101173034bdc5b8d1fc5c33418633f5ad28f24dd8";
pub const PARALLEL_LIVE_SERIES_PRODUCER_COMMIT: &str = "4dd4e1715a696a8abd105a1acc70b6dc2878fc20";
pub const PARALLEL_LIVE_SERIES_PRODUCER_SOURCE_SHA256: &str =
    "c23282f1e2ace9baaeb0d5e5d9608ec53a0ba706ee75f6166c8bf7bbefd26b57";
pub const PARALLEL_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256: &str =
    "9630ce3f8f0855659ceab446a7e60d4cd12015d3480d867103de44259b43378f";
pub const RESTORED_LIVE_SERIES_PRODUCER_COMMIT: &str = "b361cc6ccc67cfa556e711dad46a263593eafb62";
pub const RESTORED_LIVE_SERIES_PRODUCER_SOURCE_SHA256: &str =
    "4982dc0de68229a3f60a70713949bf9a6bf5781b79b909eca43ad0849a38dba8";
pub const RESTORED_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256: &str =
    "399fffddd52303b7f7c2cb4b3e0b517337ce825411b305b5cc297db590a9fb18";
pub const ADMITTED_HISTORY_CONTRACT_SHA256: &str =
    "78869f9f9775f24e42aa843bb88b415ba975f6481318a00e8f05b45044c6549b";
pub const FINAL_STEP_HISTORY_CONTRACT_SHA256: &str =
    "a5d3f1c784544cf0398a0499d51fc77792b74fcab8129755b9e43b872c9dbb90";
pub const RESTORED_LIVE_SERIES_HISTORY_CONTRACT_SHA256: &str =
    "d2be76466ff9848abc2239edfb9df03013b1ccd08ee061142f05e1bd96458894";
const RESTORED_LIVE_SERIES_OPERATOR_MANIFEST_SHA256: &str =
    "a52400db650092944f152276ed764ad3cb852473f9b8b9b805b7ed68b4c1f3e7";
const MAX_JSON_BYTES: u64 = 128 * 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ProducerIdentity {
    commit: String,
    executable_sha256: String,
    source_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct HistoryIdentity {
    artifact_sha256: String,
    contract_sha256: String,
    payload_sha256: String,
    raw_arena_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Q3dIdentity {
    nbytes: usize,
    qc_sha256: String,
    qi_sha256: String,
    sha256: String,
    shape_species_kji: [usize; 4],
    species_order: [String; 6],
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct SourceIdentity {
    history: HistoryIdentity,
    model_time_f64_bits: String,
    operator_manifest_sha256: String,
    pair_manifest_sha256: String,
    pair_receipt_sha256: String,
    q3d: Q3dIdentity,
    step: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct PackStatistics {
    all_finite: bool,
    maximum_f32_bits: String,
    minimum_f32_bits: String,
    nonnegative: bool,
    nonzero_count: usize,
    word_count: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Coordinates {
    center_lat_f64_bits: String,
    center_lon_f64_bits: String,
    dx_m_f64_bits: String,
    dy_m_f64_bits: String,
    northeast_mass_lat_f64_bits: String,
    northeast_mass_lon_f64_bits: String,
    source_parent_plan_sha256: String,
    southwest_mass_lat_f64_bits: String,
    southwest_mass_lon_f64_bits: String,
    vertical_grid_sha256: String,
    ztop_m_f64_bits: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct StaticIdentity {
    terrain_msl_sha256: String,
    total_geopotential_zf_sha256: String,
    zf_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct PayloadDescriptor {
    file: String,
    nbytes: usize,
    sha256: String,
    shape_kji: [usize; 3],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackManifest {
    abi: String,
    claim_limit: String,
    coordinates: Coordinates,
    dtype: String,
    formula: String,
    model: String,
    order: String,
    payload: PayloadDescriptor,
    producer: ProducerIdentity,
    quantized: bool,
    schema: String,
    source: SourceIdentity,
    statistics: PackStatistics,
    static_identity: StaticIdentity,
    status: String,
    synthetic_detail: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackReceipt {
    abi: String,
    all_finite: bool,
    claim_limit: String,
    formula: String,
    manifest_sha256: String,
    model: String,
    payload_nbytes: usize,
    payload_sha256: String,
    producer: ProducerIdentity,
    quantized: bool,
    schema: String,
    source: SourceIdentity,
    statistics: PackStatistics,
    status: String,
    synthetic_detail: bool,
}

pub fn is_pack_directory(directory: &Path) -> bool {
    directory.join(CHILD_VIEW_PACK_PAYLOAD_FILE).exists()
}

pub fn load_pack_directory(directory: &Path) -> Result<Gm1Dataset, String> {
    let started = Instant::now();
    verify_inventory(directory)?;
    let manifest_bytes = read_bounded(
        &directory.join(CHILD_VIEW_PACK_MANIFEST_FILE),
        "cloud-pack manifest",
    )?;
    let receipt_bytes = read_bounded(
        &directory.join(CHILD_VIEW_PACK_RECEIPT_FILE),
        "cloud-pack receipt",
    )?;
    let manifest: PackManifest = parse_canonical(&manifest_bytes, "cloud-pack manifest")?;
    let receipt: PackReceipt = parse_canonical(&receipt_bytes, "cloud-pack receipt")?;
    validate_envelopes(directory, &manifest_bytes, &manifest, &receipt)?;

    let payload_path = directory.join(CHILD_VIEW_PACK_PAYLOAD_FILE);
    let payload = fs::read(&payload_path)
        .map_err(|error| format!("cloud-pack payload is unreadable: {error}"))?;
    if payload.len() != CHILD_VIEW_PACK_NBYTES {
        return Err(format!(
            "cloud-pack payload has {} bytes, expected {CHILD_VIEW_PACK_NBYTES}",
            payload.len()
        ));
    }
    let payload_sha256 = sha256_bytes(&payload);
    if payload_sha256 != manifest.payload.sha256 {
        return Err("cloud-pack payload SHA-256 differs from the canonical manifest".to_owned());
    }
    let (values, observed) = decode_and_measure(&payload)?;
    if observed != manifest.statistics {
        return Err("cloud-pack payload statistics differ from both sealed envelopes".to_owned());
    }
    let model_time_s = decode_f64_bits(&manifest.source.model_time_f64_bits, "model time")?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    eprintln!(
        "gm1-view: admitted exact {}-byte qc+qi pack step {} in {:.3}s ({})",
        payload.len(),
        manifest.source.step,
        started.elapsed().as_secs_f64(),
        payload_sha256
    );
    Ok(Gm1Dataset {
        config_sha256: String::new(),
        terrain_msl_m: None,
        face_height_m: None,
        tornado_surface_fields: Vec::new(),
        source_path: directory.to_path_buf(),
        artifact_kind: "atomic exact qc+qi render transport".to_owned(),
        artifact_sha256: manifest_sha256,
        payload_sha256: payload_sha256.clone(),
        contract_sha256: manifest.source.history.contract_sha256.clone(),
        profile_id: CHILD_VIEW_PACK_ABI.to_owned(),
        step: manifest.source.step,
        model_time_s,
        // The admitted producer validates the fixed child's exact 0.5 s clock
        // before publication. The value is deliberately not inferred from step/time.
        dt_s: 0.5,
        fields: vec![VolumeField {
            id: "cloud_condensate".to_owned(),
            label: "Resolved cloud condensate".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "WSM6 non-precipitating cloud water plus cloud ice".to_owned(),
            source: format!(
                "{CHILD_VIEW_FORMULA}; exact non-quantized pack; source q3d {}",
                manifest.source.q3d.sha256
            ),
            shape: [CHILD_VIEW_PACK_NZ, CHILD_VIEW_PACK_NY, CHILD_VIEW_PACK_NX],
            values,
            values_sha256: payload_sha256,
            finite_min: f32::from_bits(parse_hex_u32(
                &manifest.statistics.minimum_f32_bits,
                "minimum bits",
            )?),
            finite_max: f32::from_bits(parse_hex_u32(
                &manifest.statistics.maximum_f32_bits,
                "maximum bits",
            )?),
        }],
    })
}

fn validate_envelopes(
    directory: &Path,
    manifest_bytes: &[u8],
    manifest: &PackManifest,
    receipt: &PackReceipt,
) -> Result<(), String> {
    if manifest.schema != CHILD_VIEW_PACK_MANIFEST_SCHEMA
        || receipt.schema != CHILD_VIEW_PACK_RECEIPT_SCHEMA
        || manifest.abi != CHILD_VIEW_PACK_ABI
        || receipt.abi != CHILD_VIEW_PACK_ABI
        || manifest.status != CHILD_VIEW_PACK_STATUS
        || receipt.status != CHILD_VIEW_PACK_STATUS
    {
        return Err("cloud-pack schema, ABI, or PASS status changed".to_owned());
    }
    if manifest.claim_limit != CHILD_VIEW_PACK_CLAIM_LIMIT
        || receipt.claim_limit != CHILD_VIEW_PACK_CLAIM_LIMIT
        || manifest.formula != CHILD_VIEW_FORMULA
        || receipt.formula != CHILD_VIEW_FORMULA
        || manifest.model != "GM1"
        || receipt.model != "GM1"
        || manifest.dtype != "little-endian-f32"
        || manifest.order != "C[k,j,i]"
        || manifest.quantized
        || receipt.quantized
        || manifest.synthetic_detail
        || receipt.synthetic_detail
        || !receipt.all_finite
    {
        return Err("cloud-pack scientific disposition or exact formula changed".to_owned());
    }
    if manifest.payload.file != CHILD_VIEW_PACK_PAYLOAD_FILE
        || manifest.payload.nbytes != CHILD_VIEW_PACK_NBYTES
        || receipt.payload_nbytes != CHILD_VIEW_PACK_NBYTES
        || manifest.payload.shape_kji
            != [CHILD_VIEW_PACK_NZ, CHILD_VIEW_PACK_NY, CHILD_VIEW_PACK_NX]
        || manifest.statistics.word_count != CHILD_VIEW_PACK_WORDS
        || !manifest.statistics.all_finite
        || !manifest.statistics.nonnegative
    {
        return Err("cloud-pack payload layout or finite/nonnegative contract changed".to_owned());
    }
    let manifest_sha256 = sha256_bytes(manifest_bytes);
    if receipt.manifest_sha256 != manifest_sha256
        || receipt.payload_sha256 != manifest.payload.sha256
        || receipt.producer != manifest.producer
        || receipt.source != manifest.source
        || receipt.statistics != manifest.statistics
    {
        return Err(
            "cloud-pack receipt does not bind the exact manifest/source/payload".to_owned(),
        );
    }
    validate_coordinates(&manifest.coordinates)?;
    validate_static_identity(&manifest.static_identity)?;
    validate_source(&manifest.source)?;
    if !authority_is_admitted(&manifest.producer, &manifest.source) {
        return Err(
            "cloud-pack producer/science authority pair is not admitted as one coupled lineage"
                .to_owned(),
        );
    }
    for (value, label) in [
        (&manifest.payload.sha256, "payload SHA-256"),
        (&receipt.manifest_sha256, "manifest SHA-256"),
        (&manifest.producer.source_sha256, "producer source SHA-256"),
        (
            &manifest.producer.executable_sha256,
            "producer executable SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    let step_name = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "cloud-pack directory has no UTF-8 step name".to_owned())?;
    if step_name != format!("step-{:08}", manifest.source.step) {
        return Err("cloud-pack directory name and sealed source step disagree".to_owned());
    }
    Ok(())
}

#[cfg(test)]
fn producer_is_admitted(producer: &ProducerIdentity) -> bool {
    (producer.source_sha256 == ADMITTED_PRODUCER_SOURCE_SHA256
        && producer.commit == ADMITTED_PRODUCER_COMMIT
        && (producer.executable_sha256 == ADMITTED_PRODUCER_EXECUTABLE_SHA256
            || producer.executable_sha256 == REBUILT_PRODUCER_EXECUTABLE_SHA256))
        || (producer.source_sha256 == FINAL_STEP_PRODUCER_SOURCE_SHA256
            && producer.commit == FINAL_STEP_PRODUCER_COMMIT
            && producer.executable_sha256 == FINAL_STEP_PRODUCER_EXECUTABLE_SHA256)
        || (producer.source_sha256 == LIVE_SERIES_PRODUCER_SOURCE_SHA256
            && producer.commit == LIVE_SERIES_PRODUCER_COMMIT
            && producer.executable_sha256 == LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256)
        || (producer.source_sha256 == PARALLEL_LIVE_SERIES_PRODUCER_SOURCE_SHA256
            && producer.commit == PARALLEL_LIVE_SERIES_PRODUCER_COMMIT
            && producer.executable_sha256 == PARALLEL_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256)
        || (producer.source_sha256 == RESTORED_LIVE_SERIES_PRODUCER_SOURCE_SHA256
            && producer.commit == RESTORED_LIVE_SERIES_PRODUCER_COMMIT
            && producer.executable_sha256 == RESTORED_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256)
}

fn authority_is_admitted(producer: &ProducerIdentity, source: &SourceIdentity) -> bool {
    let historical_producer = producer.source_sha256 == ADMITTED_PRODUCER_SOURCE_SHA256
        && producer.commit == ADMITTED_PRODUCER_COMMIT
        && (producer.executable_sha256 == ADMITTED_PRODUCER_EXECUTABLE_SHA256
            || producer.executable_sha256 == REBUILT_PRODUCER_EXECUTABLE_SHA256);
    let final_step_fixture_producer = producer.source_sha256 == FINAL_STEP_PRODUCER_SOURCE_SHA256
        && producer.commit == FINAL_STEP_PRODUCER_COMMIT
        && producer.executable_sha256 == FINAL_STEP_PRODUCER_EXECUTABLE_SHA256;
    let live_series_producer = (producer.source_sha256 == LIVE_SERIES_PRODUCER_SOURCE_SHA256
        && producer.commit == LIVE_SERIES_PRODUCER_COMMIT
        && producer.executable_sha256 == LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256)
        || (producer.source_sha256 == PARALLEL_LIVE_SERIES_PRODUCER_SOURCE_SHA256
            && producer.commit == PARALLEL_LIVE_SERIES_PRODUCER_COMMIT
            && producer.executable_sha256 == PARALLEL_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256);
    let restored_live_series_producer = producer.source_sha256
        == RESTORED_LIVE_SERIES_PRODUCER_SOURCE_SHA256
        && producer.commit == RESTORED_LIVE_SERIES_PRODUCER_COMMIT
        && producer.executable_sha256 == RESTORED_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256;
    (historical_producer && historical_source_fixture(source))
        || (final_step_fixture_producer && final_step_source_fixture(source))
        || (live_series_producer && live_series_source_is_admitted(source))
        || (restored_live_series_producer && restored_live_series_source_is_admitted(source))
}

fn historical_source_fixture(source: &SourceIdentity) -> bool {
    source.history.artifact_sha256
        == "94108b08e926a9d4ebc09029b66afcc3db4948bb1f769296a9e100d9e2f7bb4e"
        && source.history.contract_sha256 == ADMITTED_HISTORY_CONTRACT_SHA256
        && source.history.payload_sha256
            == "2f15aea48331fe0d383af2c72f5bf44471b71e2b1e2d3280c6c2c70e54ddf42d"
        && source.history.raw_arena_sha256
            == "09fb4b252c18388aedbbf821e75e39baec7a9263038080504edd629683699ba9"
        && source.model_time_f64_bits == "4072c00000000000"
        && source.operator_manifest_sha256
            == "c60142440dcb98bed84a784e2ff27405f2c1bd66afec32c3ef263c5f5e37cef8"
        && source.pair_manifest_sha256
            == "c37e5bec8101b797fe516fe21f3474785f3bd2884337d646f2ba0f7fbe8961e5"
        && source.pair_receipt_sha256
            == "2ad53c8a564c38c775fe138248b9f2cd4bc471f67168602c60f943026a80c36b"
        && source.q3d.sha256 == "31bebd4aba014bf99e2db78f78af62be18c88556534ed1ae2e2c41354d06145f"
        && source.q3d.qc_sha256
            == "12d3fdb2e931086a5e96ce820ec6bfaa2dfcbdbd0bc96bd9cd3d57e039a9fc71"
        && source.q3d.qi_sha256
            == "fb94cbeb6304f6c1cfa494746c28ef7e441a27876cf8e264878a2585b3b8fad8"
        && source.step == 600
}

fn final_step_source_fixture(source: &SourceIdentity) -> bool {
    source.history.artifact_sha256
        == "dbd57a7f5f00d19ccd4929c87219294dce6c84567fc49b41f0c49a26f00c7ee0"
        && source.history.contract_sha256 == FINAL_STEP_HISTORY_CONTRACT_SHA256
        && source.history.payload_sha256
            == "51080963b505c7872963de751d93f054f9c8c9565efdb7320eae2deab613f567"
        && source.history.raw_arena_sha256
            == "4999f36bbe8c0e14a7abb13ec641512c9a58f5d860ab598b05958a9a4464e5dd"
        && source.model_time_f64_bits == "3fe0000000000000"
        && source.operator_manifest_sha256
            == "6112fb9302f57f82b09118b3f38c6ae8dd3d82c59eb14fd01940f2dc43428c09"
        && source.pair_manifest_sha256
            == "687b4de7ee61de115dea88536e1f9e7ff1f58bf155285f08215d1e7800f862d4"
        && source.pair_receipt_sha256
            == "849e153bbc995f72f75a4410cefa5c8a4a58ccdce3a8648a997512427b222e2b"
        && source.q3d.sha256 == "0055b6ecfe010ef205d68bbe3d0bc4a071220af679cde4af402c3e75518d7661"
        && source.q3d.qc_sha256
            == "0e0eda30bcca186dd4066225bf5e8ad9b3897f9691bfef345b5d1dcfd2287115"
        && source.q3d.qi_sha256
            == "d2c34f2fa64ba4c8c9f0db8bda2b3ba19ff70dea5cca2e1b69ce94c65fa191b1"
        && source.step == 1
}

fn live_series_source_is_admitted(source: &SourceIdentity) -> bool {
    source.step > 0
        && source.history.contract_sha256 == FINAL_STEP_HISTORY_CONTRACT_SHA256
        && source.operator_manifest_sha256
            == "6112fb9302f57f82b09118b3f38c6ae8dd3d82c59eb14fd01940f2dc43428c09"
        && decode_f64_bits(&source.model_time_f64_bits, "model time")
            .is_ok_and(|time| time.to_bits() == (source.step as f64 * 0.5).to_bits())
}

fn restored_live_series_source_is_admitted(source: &SourceIdentity) -> bool {
    source.step > 0
        && source.history.contract_sha256 == RESTORED_LIVE_SERIES_HISTORY_CONTRACT_SHA256
        && source.operator_manifest_sha256 == RESTORED_LIVE_SERIES_OPERATOR_MANIFEST_SHA256
        && decode_f64_bits(&source.model_time_f64_bits, "model time")
            .is_ok_and(|time| time.to_bits() == (source.step as f64 * 0.5).to_bits())
}

fn validate_coordinates(value: &Coordinates) -> Result<(), String> {
    let expected = Coordinates {
        center_lat_f64_bits: "4044bb15cda77904".to_owned(),
        center_lon_f64_bits: "c0578c5ebb59704f".to_owned(),
        dx_m_f64_bits: "4049000000000000".to_owned(),
        dy_m_f64_bits: "4049000000000000".to_owned(),
        northeast_mass_lat_f64_bits: "4044c3af62e702d3".to_owned(),
        northeast_mass_lon_f64_bits: "c057869f8bc20426".to_owned(),
        source_parent_plan_sha256:
            "14609494f7c87937fcb5ec0db3f214809ed514380560c43beb18183e6eebce17".to_owned(),
        southwest_mass_lat_f64_bits: "4044b279efdec50d".to_owned(),
        southwest_mass_lon_f64_bits: "c057921adeefb07a".to_owned(),
        vertical_grid_sha256: "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56"
            .to_owned(),
        ztop_m_f64_bits: "40cd4c0000000000".to_owned(),
    };
    if value != &expected {
        return Err("cloud-pack georeference or sealed vertical grid changed".to_owned());
    }
    Ok(())
}

fn validate_static_identity(value: &StaticIdentity) -> Result<(), String> {
    let expected = StaticIdentity {
        terrain_msl_sha256: "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91"
            .to_owned(),
        total_geopotential_zf_sha256:
            "687bc3ebe34c0c07dc0c46418e8e6042f271e1bb1bcd119284b4a27ef80bca8c".to_owned(),
        zf_sha256: "df5aef17965915406fe334600afe2f8f2e1ebb5d07665f71820eaac9bde4128f".to_owned(),
    };
    if value != &expected {
        return Err("cloud-pack static terrain/height identity changed".to_owned());
    }
    Ok(())
}

fn validate_source(source: &SourceIdentity) -> Result<(), String> {
    for (value, label) in [
        (&source.history.artifact_sha256, "history artifact SHA-256"),
        (&source.history.contract_sha256, "history contract SHA-256"),
        (&source.history.payload_sha256, "history payload SHA-256"),
        (&source.history.raw_arena_sha256, "history raw SHA-256"),
        (
            &source.operator_manifest_sha256,
            "operator manifest SHA-256",
        ),
        (&source.pair_manifest_sha256, "pair manifest SHA-256"),
        (&source.pair_receipt_sha256, "pair receipt SHA-256"),
        (&source.q3d.sha256, "q3d SHA-256"),
        (&source.q3d.qc_sha256, "qc SHA-256"),
        (&source.q3d.qi_sha256, "qi SHA-256"),
    ] {
        validate_sha256(value, label)?;
    }
    if source.q3d.nbytes != CHILD_VIEW_Q3D_NBYTES
        || source.q3d.shape_species_kji
            != [
                CHILD_VIEW_SPECIES.len(),
                CHILD_VIEW_PACK_NZ,
                CHILD_VIEW_PACK_NY,
                CHILD_VIEW_PACK_NX,
            ]
        || source
            .q3d
            .species_order
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != CHILD_VIEW_SPECIES
    {
        return Err("cloud-pack source q3d layout/species contract changed".to_owned());
    }
    let model_time = decode_f64_bits(&source.model_time_f64_bits, "model time")?;
    if source.step == 0
        || !model_time.is_finite()
        || model_time < 0.0
        || model_time.to_bits() != (source.step as f64 * 0.5).to_bits()
    {
        return Err(
            "cloud-pack step/model time is not the exact positive step*0.5 cadence".to_owned(),
        );
    }
    Ok(())
}

fn decode_and_measure(payload: &[u8]) -> Result<(Vec<f32>, PackStatistics), String> {
    if !payload.len().is_multiple_of(4) {
        return Err("cloud-pack payload is not whole binary32 words".to_owned());
    }
    let mut values = Vec::with_capacity(payload.len() / 4);
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut nonzero_count = 0usize;
    for bytes in payload.chunks_exact(4) {
        let value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
        if !value.is_finite() || value < 0.0 {
            return Err("cloud-pack payload contains a non-finite or negative word".to_owned());
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        nonzero_count += usize::from(value != 0.0);
        values.push(value);
    }
    Ok((
        values,
        PackStatistics {
            all_finite: true,
            maximum_f32_bits: format!("{:08x}", maximum.to_bits()),
            minimum_f32_bits: format!("{:08x}", minimum.to_bits()),
            nonnegative: true,
            nonzero_count,
            word_count: payload.len() / 4,
        },
    ))
}

fn verify_inventory(directory: &Path) -> Result<(), String> {
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("cloud-pack directory is unreadable: {error}"))?
    {
        let entry = entry.map_err(|error| error.to_string())?;
        let metadata = fs::symlink_metadata(entry.path()).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err("cloud-pack inventory contains a symlink or non-file".to_owned());
        }
        observed.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| "cloud-pack has a non-UTF-8 filename".to_owned())?,
        );
    }
    let expected = BTreeSet::from([
        CHILD_VIEW_PACK_PAYLOAD_FILE.to_owned(),
        CHILD_VIEW_PACK_MANIFEST_FILE.to_owned(),
        CHILD_VIEW_PACK_RECEIPT_FILE.to_owned(),
    ]);
    if observed != expected {
        return Err(format!(
            "cloud-pack exact three-file inventory changed: {observed:?}"
        ));
    }
    Ok(())
}

fn read_bounded(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} metadata is unavailable: {error}"))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > MAX_JSON_BYTES
    {
        return Err(format!("{label} is not a bounded regular file"));
    }
    fs::read(path).map_err(|error| format!("{label} is unreadable: {error}"))
}

fn parse_canonical<T: for<'de> Deserialize<'de>>(bytes: &[u8], label: &str) -> Result<T, String> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("{label} JSON is invalid: {error}"))?;
    let canonical = serde_json::to_vec(&value)
        .map_err(|error| format!("{label} canonicalization failed: {error}"))?;
    if bytes != canonical {
        return Err(format!("{label} is not canonical compact JSON"));
    }
    serde_json::from_value(value).map_err(|error| format!("{label} schema is invalid: {error}"))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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

fn parse_hex_u32(value: &str, label: &str) -> Result<u32, String> {
    if value.len() != 8
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} is not eight lowercase hex digits"));
    }
    u32::from_str_radix(value, 16).map_err(|error| format!("{label} is invalid: {error}"))
}

fn decode_f64_bits(value: &str, label: &str) -> Result<f64, String> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} is not sixteen lowercase hex digits"));
    }
    let bits = u64::from_str_radix(value, 16)
        .map_err(|error| format!("{label} bits are invalid: {error}"))?;
    Ok(f64::from_bits(bits))
}

#[cfg(test)]
mod tests {
    use super::{
        ADMITTED_PRODUCER_COMMIT, ADMITTED_PRODUCER_EXECUTABLE_SHA256,
        ADMITTED_PRODUCER_SOURCE_SHA256, FINAL_STEP_PRODUCER_COMMIT,
        FINAL_STEP_PRODUCER_EXECUTABLE_SHA256, FINAL_STEP_PRODUCER_SOURCE_SHA256, HistoryIdentity,
        LIVE_SERIES_PRODUCER_COMMIT, LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256,
        LIVE_SERIES_PRODUCER_SOURCE_SHA256, PARALLEL_LIVE_SERIES_PRODUCER_COMMIT,
        PARALLEL_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256,
        PARALLEL_LIVE_SERIES_PRODUCER_SOURCE_SHA256, ProducerIdentity, Q3dIdentity,
        REBUILT_PRODUCER_EXECUTABLE_SHA256, RESTORED_LIVE_SERIES_HISTORY_CONTRACT_SHA256,
        RESTORED_LIVE_SERIES_OPERATOR_MANIFEST_SHA256, RESTORED_LIVE_SERIES_PRODUCER_COMMIT,
        RESTORED_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256,
        RESTORED_LIVE_SERIES_PRODUCER_SOURCE_SHA256, SourceIdentity, authority_is_admitted,
        decode_and_measure, parse_canonical, producer_is_admitted, validate_source,
        verify_inventory,
    };
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn scratch(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "gm1-view-pack-{name}-{}-{}",
            std::process::id(),
            SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir(&path).expect("create scratch");
        path
    }

    #[test]
    fn exact_binary32_decode_measures_raw_values() {
        let words = [0.0_f32, 1.25, 0.5, 4.0];
        let payload = words
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let (decoded, stats) = decode_and_measure(&payload).expect("decode");
        assert_eq!(decoded, words);
        assert_eq!(stats.minimum_f32_bits, "00000000");
        assert_eq!(stats.maximum_f32_bits, "40800000");
        assert_eq!(stats.nonzero_count, 3);
        let mut corrupt = payload;
        corrupt[0..4].copy_from_slice(&(-1.0_f32).to_le_bytes());
        assert!(decode_and_measure(&corrupt).is_err());
    }

    #[test]
    fn canonical_json_and_exact_inventory_fail_closed() {
        let canonical = b"{\"a\":1}";
        let parsed: serde_json::Value = parse_canonical(canonical, "test").expect("canonical");
        assert_eq!(parsed["a"], 1);
        assert!(parse_canonical::<serde_json::Value>(b"{ \"a\": 1 }\n", "test").is_err());

        let directory = scratch("inventory");
        for name in ["cloud-condensate.f32le", "manifest.json", "receipt.json"] {
            fs::write(directory.join(name), b"x").expect("write inventory fixture");
        }
        verify_inventory(&directory).expect("exact three files");
        fs::write(directory.join("ready.json"), b"x").expect("write extra");
        assert!(verify_inventory(&directory).is_err());
        fs::remove_dir_all(directory).expect("remove scratch");
    }

    #[test]
    fn producer_allowlist_requires_one_exact_commit_executable_pair() {
        let producer = |commit: &str, executable: &str, source: &str| ProducerIdentity {
            commit: commit.to_owned(),
            executable_sha256: executable.to_owned(),
            source_sha256: source.to_owned(),
        };
        assert!(producer_is_admitted(&producer(
            ADMITTED_PRODUCER_COMMIT,
            ADMITTED_PRODUCER_EXECUTABLE_SHA256,
            ADMITTED_PRODUCER_SOURCE_SHA256,
        )));
        assert!(producer_is_admitted(&producer(
            ADMITTED_PRODUCER_COMMIT,
            REBUILT_PRODUCER_EXECUTABLE_SHA256,
            ADMITTED_PRODUCER_SOURCE_SHA256,
        )));
        assert!(producer_is_admitted(&producer(
            FINAL_STEP_PRODUCER_COMMIT,
            FINAL_STEP_PRODUCER_EXECUTABLE_SHA256,
            FINAL_STEP_PRODUCER_SOURCE_SHA256,
        )));
        assert!(producer_is_admitted(&producer(
            LIVE_SERIES_PRODUCER_COMMIT,
            LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256,
            LIVE_SERIES_PRODUCER_SOURCE_SHA256,
        )));
        assert!(producer_is_admitted(&producer(
            PARALLEL_LIVE_SERIES_PRODUCER_COMMIT,
            PARALLEL_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256,
            PARALLEL_LIVE_SERIES_PRODUCER_SOURCE_SHA256,
        )));
        assert!(producer_is_admitted(&producer(
            RESTORED_LIVE_SERIES_PRODUCER_COMMIT,
            RESTORED_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256,
            RESTORED_LIVE_SERIES_PRODUCER_SOURCE_SHA256,
        )));
        assert!(!producer_is_admitted(&producer(
            &"0".repeat(40),
            ADMITTED_PRODUCER_EXECUTABLE_SHA256,
            ADMITTED_PRODUCER_SOURCE_SHA256,
        )));
        assert!(!producer_is_admitted(&producer(
            &"0".repeat(40),
            REBUILT_PRODUCER_EXECUTABLE_SHA256,
            ADMITTED_PRODUCER_SOURCE_SHA256,
        )));
        assert!(!producer_is_admitted(&producer(
            ADMITTED_PRODUCER_COMMIT,
            REBUILT_PRODUCER_EXECUTABLE_SHA256,
            &"0".repeat(64),
        )));
        assert!(!producer_is_admitted(&producer(
            FINAL_STEP_PRODUCER_COMMIT,
            REBUILT_PRODUCER_EXECUTABLE_SHA256,
            FINAL_STEP_PRODUCER_SOURCE_SHA256,
        )));
        assert!(!producer_is_admitted(&producer(
            ADMITTED_PRODUCER_COMMIT,
            FINAL_STEP_PRODUCER_EXECUTABLE_SHA256,
            FINAL_STEP_PRODUCER_SOURCE_SHA256,
        )));
        assert!(!producer_is_admitted(&producer(
            LIVE_SERIES_PRODUCER_COMMIT,
            FINAL_STEP_PRODUCER_EXECUTABLE_SHA256,
            LIVE_SERIES_PRODUCER_SOURCE_SHA256,
        )));
        assert!(!producer_is_admitted(&producer(
            RESTORED_LIVE_SERIES_PRODUCER_COMMIT,
            PARALLEL_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256,
            RESTORED_LIVE_SERIES_PRODUCER_SOURCE_SHA256,
        )));
    }

    #[test]
    fn coupled_authority_retains_fixtures_and_accepts_new_series_data_hashes() {
        let producer = |commit: &str, executable: &str, source: &str| ProducerIdentity {
            commit: commit.to_owned(),
            executable_sha256: executable.to_owned(),
            source_sha256: source.to_owned(),
        };
        let source = |final_step: bool| SourceIdentity {
            history: HistoryIdentity {
                artifact_sha256: if final_step {
                    "dbd57a7f5f00d19ccd4929c87219294dce6c84567fc49b41f0c49a26f00c7ee0"
                } else {
                    "94108b08e926a9d4ebc09029b66afcc3db4948bb1f769296a9e100d9e2f7bb4e"
                }
                .to_owned(),
                contract_sha256: if final_step {
                    super::FINAL_STEP_HISTORY_CONTRACT_SHA256
                } else {
                    super::ADMITTED_HISTORY_CONTRACT_SHA256
                }
                .to_owned(),
                payload_sha256: if final_step {
                    "51080963b505c7872963de751d93f054f9c8c9565efdb7320eae2deab613f567"
                } else {
                    "2f15aea48331fe0d383af2c72f5bf44471b71e2b1e2d3280c6c2c70e54ddf42d"
                }
                .to_owned(),
                raw_arena_sha256: if final_step {
                    "4999f36bbe8c0e14a7abb13ec641512c9a58f5d860ab598b05958a9a4464e5dd"
                } else {
                    "09fb4b252c18388aedbbf821e75e39baec7a9263038080504edd629683699ba9"
                }
                .to_owned(),
            },
            model_time_f64_bits: if final_step {
                "3fe0000000000000"
            } else {
                "4072c00000000000"
            }
            .to_owned(),
            operator_manifest_sha256: if final_step {
                "6112fb9302f57f82b09118b3f38c6ae8dd3d82c59eb14fd01940f2dc43428c09"
            } else {
                "c60142440dcb98bed84a784e2ff27405f2c1bd66afec32c3ef263c5f5e37cef8"
            }
            .to_owned(),
            pair_manifest_sha256: if final_step {
                "687b4de7ee61de115dea88536e1f9e7ff1f58bf155285f08215d1e7800f862d4"
            } else {
                "c37e5bec8101b797fe516fe21f3474785f3bd2884337d646f2ba0f7fbe8961e5"
            }
            .to_owned(),
            pair_receipt_sha256: if final_step {
                "849e153bbc995f72f75a4410cefa5c8a4a58ccdce3a8648a997512427b222e2b"
            } else {
                "2ad53c8a564c38c775fe138248b9f2cd4bc471f67168602c60f943026a80c36b"
            }
            .to_owned(),
            q3d: Q3dIdentity {
                nbytes: 162_000_000,
                qc_sha256: if final_step {
                    "0e0eda30bcca186dd4066225bf5e8ad9b3897f9691bfef345b5d1dcfd2287115"
                } else {
                    "12d3fdb2e931086a5e96ce820ec6bfaa2dfcbdbd0bc96bd9cd3d57e039a9fc71"
                }
                .to_owned(),
                qi_sha256: if final_step {
                    "d2c34f2fa64ba4c8c9f0db8bda2b3ba19ff70dea5cca2e1b69ce94c65fa191b1"
                } else {
                    "fb94cbeb6304f6c1cfa494746c28ef7e441a27876cf8e264878a2585b3b8fad8"
                }
                .to_owned(),
                sha256: if final_step {
                    "0055b6ecfe010ef205d68bbe3d0bc4a071220af679cde4af402c3e75518d7661"
                } else {
                    "31bebd4aba014bf99e2db78f78af62be18c88556534ed1ae2e2c41354d06145f"
                }
                .to_owned(),
                shape_species_kji: [6, 75, 300, 300],
                species_order: ["qv", "qc", "qr", "qi", "qs", "qg"].map(str::to_owned),
            },
            step: if final_step { 1 } else { 600 },
        };
        let historical = source(false);
        let final_step = source(true);
        let historical_producer = producer(
            ADMITTED_PRODUCER_COMMIT,
            ADMITTED_PRODUCER_EXECUTABLE_SHA256,
            ADMITTED_PRODUCER_SOURCE_SHA256,
        );
        let final_fixture_producer = producer(
            FINAL_STEP_PRODUCER_COMMIT,
            FINAL_STEP_PRODUCER_EXECUTABLE_SHA256,
            FINAL_STEP_PRODUCER_SOURCE_SHA256,
        );
        let live_producer = producer(
            LIVE_SERIES_PRODUCER_COMMIT,
            LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256,
            LIVE_SERIES_PRODUCER_SOURCE_SHA256,
        );
        let parallel_live_producer = producer(
            PARALLEL_LIVE_SERIES_PRODUCER_COMMIT,
            PARALLEL_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256,
            PARALLEL_LIVE_SERIES_PRODUCER_SOURCE_SHA256,
        );
        let restored_live_producer = producer(
            RESTORED_LIVE_SERIES_PRODUCER_COMMIT,
            RESTORED_LIVE_SERIES_PRODUCER_EXECUTABLE_SHA256,
            RESTORED_LIVE_SERIES_PRODUCER_SOURCE_SHA256,
        );
        assert!(authority_is_admitted(&historical_producer, &historical));
        assert!(authority_is_admitted(&final_fixture_producer, &final_step));
        assert!(authority_is_admitted(&live_producer, &final_step));
        assert!(authority_is_admitted(&parallel_live_producer, &final_step));

        let mut next = final_step.clone();
        next.step = 2;
        next.model_time_f64_bits = format!("{:016x}", 1.0_f64.to_bits());
        next.history.artifact_sha256 = "10".repeat(32);
        next.history.payload_sha256 = "20".repeat(32);
        next.history.raw_arena_sha256 = "30".repeat(32);
        next.pair_manifest_sha256 = "40".repeat(32);
        next.pair_receipt_sha256 = "50".repeat(32);
        next.q3d.sha256 = "60".repeat(32);
        next.q3d.qc_sha256 = "70".repeat(32);
        next.q3d.qi_sha256 = "80".repeat(32);
        assert!(validate_source(&next).is_ok());
        assert!(authority_is_admitted(&live_producer, &next));
        assert!(authority_is_admitted(&parallel_live_producer, &next));
        assert!(!authority_is_admitted(&final_fixture_producer, &next));
        assert!(!authority_is_admitted(&historical_producer, &final_step));

        let mut restored = next.clone();
        restored.history.contract_sha256 = RESTORED_LIVE_SERIES_HISTORY_CONTRACT_SHA256.to_owned();
        restored.operator_manifest_sha256 =
            RESTORED_LIVE_SERIES_OPERATOR_MANIFEST_SHA256.to_owned();
        assert!(validate_source(&restored).is_ok());
        assert!(authority_is_admitted(&restored_live_producer, &restored));
        assert!(!authority_is_admitted(&live_producer, &restored));
        assert!(!authority_is_admitted(&parallel_live_producer, &restored));
        assert!(!authority_is_admitted(&restored_live_producer, &next));
        let mut mixed_restored = restored;
        mixed_restored.operator_manifest_sha256 =
            "6112fb9302f57f82b09118b3f38c6ae8dd3d82c59eb14fd01940f2dc43428c09".to_owned();
        assert!(!authority_is_admitted(
            &restored_live_producer,
            &mixed_restored
        ));

        let mut foreign_operator = next.clone();
        foreign_operator.operator_manifest_sha256 = "90".repeat(32);
        assert!(!authority_is_admitted(&live_producer, &foreign_operator));
        let mut rollback_clock = next.clone();
        rollback_clock.model_time_f64_bits = format!("{:016x}", 0.5_f64.to_bits());
        assert!(validate_source(&rollback_clock).is_err());
        assert!(!authority_is_admitted(&live_producer, &rollback_clock));
        let mut corrupt_layout = next;
        corrupt_layout.q3d.shape_species_kji[1] = 74;
        assert!(validate_source(&corrupt_layout).is_err());
    }
}
