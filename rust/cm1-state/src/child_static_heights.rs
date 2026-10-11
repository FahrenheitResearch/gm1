//! Bounded Rust producer for terrain-following child `zh` and `zf` heights.
//!
//! Each output level is streamed directly to its final staging file using the
//! binary32 operation order of CM1 r21.1 `init_terrain` and the sealed
//! `stretch_z=1` profiles. The accepted 300/50 CUDA artifact remains an
//! independent raw-bit oracle; this module adds a grid-parameterized v2
//! envelope for the real 15 km child domain.

use crate::child_remap::{
    sealed_child_stretch_grid, ChildGridSpec, CHILD_STRETCH_ABI, CHILD_STRETCH_GRID_SHA256,
    CHILD_STRETCH_SCHEMA, CM1_R21_AUTHORITY_SHA256,
};
use crate::child_terrain::{
    CHILD_TERRAIN_ABI, CHILD_TERRAIN_GRID_ABI, CHILD_TERRAIN_GRID_MANIFEST_SCHEMA,
    CHILD_TERRAIN_GRID_SCHEMA, CHILD_TERRAIN_MANIFEST_FILE, CHILD_TERRAIN_MANIFEST_SCHEMA,
    CHILD_TERRAIN_PAYLOAD_FILE, CHILD_TERRAIN_RECEIPT_FILE, CHILD_TERRAIN_SCHEMA,
};
use crate::json::JsonValue;
use crate::sha256::{self, Sha256};
use crate::{Result, StateError};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHILD_STATIC_HEIGHT_GRID_SCHEMA: &str = "gm1.source-coordinate-child-static-heights/v2";
pub const CHILD_STATIC_HEIGHT_GRID_MANIFEST_SCHEMA: &str =
    "gm1.source-coordinate-child-static-heights-manifest/v2";
pub const CHILD_STATIC_HEIGHT_GRID_ABI: &str = "gm1.source-coordinate-child-static-heights-f32/v2";
pub const CHILD_STATIC_HEIGHT_GRID_STATUS: &str = "PASS_REAL_SOURCE_TERRAIN_CHILD_STATIC_HEIGHTS";
pub const CHILD_STATIC_HEIGHT_GRID_CLAIM_LIMIT: &str = "two exact terrain-following height carriers, zh and zf, computed with the CM1 r21.1 init_terrain binary32 source order from one independently SHA-bound real source-coordinate child terrain and the sealed 75-level stretch_z=1 profiles; no other static metric, atmospheric remap, hydrostatic/base state, open-boundary state, closure, checkpoint, timestep, forecast-skill, storm, or tornado claim";

pub const CHILD_STATIC_HEIGHT_MANIFEST_FILE: &str = "manifest.json";
pub const CHILD_STATIC_HEIGHT_RECEIPT_FILE: &str = "receipt.json";
pub const CHILD_STATIC_HEIGHT_ZH_FILE: &str = "zh.f32le";
pub const CHILD_STATIC_HEIGHT_ZF_FILE: &str = "zf.f32le";
pub const CM1_INIT_TERRAIN_SHA256: &str =
    "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c";
pub const CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256: &str =
    "905bec56020e2bf1a73c0c854a9d5f462ccc372334ab3ceee2ad50f041be2f18";

const JSON_MAX_BYTES: u64 = 1024 * 1024;
const SOURCE_MAX_BYTES: u64 = 4 * 1024 * 1024;
const EXECUTABLE_MAX_BYTES: u64 = 64 * 1024 * 1024;
const STREAM_CHUNK_BYTES: usize = 1024 * 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildTerrainArtifactIdentity {
    pub manifest_sha256: String,
    pub payload_sha256: String,
    pub receipt_sha256: String,
}

impl ChildTerrainArtifactIdentity {
    pub fn new(
        manifest_sha256: impl Into<String>,
        payload_sha256: impl Into<String>,
        receipt_sha256: impl Into<String>,
    ) -> Result<Self> {
        let identity = Self {
            manifest_sha256: manifest_sha256.into(),
            payload_sha256: payload_sha256.into(),
            receipt_sha256: receipt_sha256.into(),
        };
        for (label, value) in [
            ("terrain manifest", identity.manifest_sha256.as_str()),
            ("terrain payload", identity.payload_sha256.as_str()),
            ("terrain receipt", identity.receipt_sha256.as_str()),
        ] {
            validate_sha256(value, label)?;
        }
        Ok(identity)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChildStaticHeightGridReceipt {
    pub grid: ChildGridSpec,
    pub manifest_sha256: String,
    pub zh_sha256: String,
    pub zf_sha256: String,
    pub receipt_sha256: String,
    pub producer_source_sha256: String,
    pub producer_executable_sha256: String,
    pub terrain: ChildTerrainArtifactIdentity,
}

#[derive(Clone, Debug)]
struct HeightFileRecord {
    name: &'static str,
    file: &'static str,
    shape: Vec<usize>,
    nbytes: u64,
    sha256: String,
    minimum_f32_bits: u32,
    maximum_f32_bits: u32,
}

#[derive(Clone, Debug)]
struct PreparedStaticInputs {
    grid: ChildGridSpec,
    terrain: ChildTerrainArtifactIdentity,
    terrain_words: Vec<f32>,
    producer_source_sha256: String,
    producer_executable_sha256: String,
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

fn add32(left: f32, right: f32) -> f32 {
    left + right
}

fn sub32(left: f32, right: f32) -> f32 {
    left - right
}

fn mul32(left: f32, right: f32) -> f32 {
    left * right
}

fn div32(left: f32, right: f32) -> f32 {
    left / right
}

fn terrain_height(surface: f32, sigma: f32) -> f32 {
    let top = 15_000.0f32;
    add32(surface, div32(mul32(sigma, sub32(top, surface)), top))
}

fn height_nbytes(grid: ChildGridSpec, levels: usize) -> Result<u64> {
    u64::try_from(levels)
        .ok()
        .and_then(|value| value.checked_mul(grid.ny() as u64))
        .and_then(|value| value.checked_mul(grid.nx() as u64))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| invalid("child static-height byte count overflowed"))
}

pub fn child_static_height_nbytes_for_grid(grid: ChildGridSpec) -> Result<(u64, u64)> {
    Ok((
        height_nbytes(grid, grid.nz())?,
        height_nbytes(grid, grid.nz() + 1)?,
    ))
}

fn prepare_inputs(
    grid: ChildGridSpec,
    terrain_directory: &Path,
    terrain: ChildTerrainArtifactIdentity,
    init_terrain_source: &Path,
    producer_source: &Path,
    producer_executable: &Path,
) -> Result<PreparedStaticInputs> {
    verify_terrain_inventory(terrain_directory)?;
    verify_exact_hash(
        &terrain_directory.join(CHILD_TERRAIN_MANIFEST_FILE),
        JSON_MAX_BYTES,
        &terrain.manifest_sha256,
        "child terrain manifest",
    )?;
    verify_exact_hash(
        &terrain_directory.join(CHILD_TERRAIN_RECEIPT_FILE),
        JSON_MAX_BYTES,
        &terrain.receipt_sha256,
        "child terrain receipt",
    )?;
    let terrain_nbytes = height_nbytes(grid, 1)?;
    let terrain_bytes = read_regular_exact(
        &terrain_directory.join(CHILD_TERRAIN_PAYLOAD_FILE),
        terrain_nbytes,
        "child terrain payload",
    )?;
    if sha256::digest_hex(&terrain_bytes) != terrain.payload_sha256 {
        return Err(invalid(
            "child terrain payload differs from its exact identity",
        ));
    }
    let manifest_bytes = read_regular_bounded(
        &terrain_directory.join(CHILD_TERRAIN_MANIFEST_FILE),
        JSON_MAX_BYTES,
        "child terrain manifest",
    )?;
    validate_terrain_manifest(&manifest_bytes, grid, &terrain)?;
    verify_exact_hash(
        init_terrain_source,
        SOURCE_MAX_BYTES,
        CM1_INIT_TERRAIN_SHA256,
        "CM1 init_terrain.F authority",
    )?;
    let producer_source_sha256 = hash_regular_bounded(
        producer_source,
        SOURCE_MAX_BYTES,
        "child static-height producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable,
        EXECUTABLE_MAX_BYTES,
        "child static-height producer executable",
    )?;
    let mut terrain_words = Vec::with_capacity(grid.nx() * grid.ny());
    for word in terrain_bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(word.try_into().expect("four-byte terrain word"));
        if !value.is_finite() || value >= 15_000.0 {
            return Err(invalid(
                "child terrain contains a non-finite value or reaches model top",
            ));
        }
        terrain_words.push(value);
    }
    Ok(PreparedStaticInputs {
        grid,
        terrain,
        terrain_words,
        producer_source_sha256,
        producer_executable_sha256,
    })
}

fn validate_terrain_manifest(
    bytes: &[u8],
    grid: ChildGridSpec,
    identity: &ChildTerrainArtifactIdentity,
) -> Result<()> {
    let value = crate::json::parse(bytes)
        .map_err(|error| invalid(format!("child terrain manifest JSON: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid("child terrain manifest is not canonical JSON"));
    }
    let payload = value
        .get("payload")
        .map_err(|error| invalid(format!("child terrain manifest: {error}")))?;
    let shape = payload
        .get("shape_yx")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("child terrain payload shape: {error}")))?;
    let shape_values = shape
        .iter()
        .map(|item| {
            let number = item
                .as_u64()
                .map_err(|error| invalid(format!("child terrain payload shape: {error}")))?;
            usize::try_from(number).map_err(|_| invalid("child terrain payload shape is invalid"))
        })
        .collect::<Result<Vec<_>>>()?;
    let expected_schema = if grid.nx() == 300 && grid.ny() == 300 {
        CHILD_TERRAIN_MANIFEST_SCHEMA
    } else {
        CHILD_TERRAIN_GRID_MANIFEST_SCHEMA
    };
    let expected_abi = if grid.nx() == 300 && grid.ny() == 300 {
        CHILD_TERRAIN_ABI
    } else {
        CHILD_TERRAIN_GRID_ABI
    };
    if value.get("schema").and_then(JsonValue::as_str).ok() != Some(expected_schema)
        || value.get("abi").and_then(JsonValue::as_str).ok() != Some(expected_abi)
        || shape_values != [grid.ny(), grid.nx()]
        || payload.get("sha256").and_then(JsonValue::as_str).ok()
            != Some(identity.payload_sha256.as_str())
        || payload.get("nbytes").and_then(JsonValue::as_u64).ok() != Some(height_nbytes(grid, 1)?)
        || value.get("synthetic_translation").ok() != Some(&JsonValue::Bool(false))
    {
        return Err(invalid(
            "child terrain manifest schema, grid, payload, or source disposition drifted",
        ));
    }
    Ok(())
}

fn stream_height_file(
    writer: &mut dyn Write,
    terrain: &[f32],
    levels: &[f32],
    grid: ChildGridSpec,
    name: &'static str,
    file: &'static str,
) -> Result<HeightFileRecord> {
    if terrain.len() != grid.nx() * grid.ny() || levels.is_empty() {
        return Err(invalid("child static-height input shape drifted"));
    }
    let expected_nbytes = height_nbytes(grid, levels.len())?;
    let mut hasher = Sha256::default();
    let mut buffer = Vec::with_capacity(STREAM_CHUNK_BYTES);
    let mut nbytes = 0u64;
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for sigma in levels {
        for surface in terrain {
            let value = terrain_height(*surface, *sigma);
            if !value.is_finite() {
                return Err(invalid(
                    "child static-height transform produced non-finite data",
                ));
            }
            minimum = minimum.min(value);
            maximum = maximum.max(value);
            buffer.extend_from_slice(&value.to_le_bytes());
            if buffer.len() == STREAM_CHUNK_BYTES {
                writer.write_all(&buffer)?;
                hasher.update(&buffer);
                nbytes += buffer.len() as u64;
                buffer.clear();
            }
        }
    }
    if !buffer.is_empty() {
        writer.write_all(&buffer)?;
        hasher.update(&buffer);
        nbytes += buffer.len() as u64;
    }
    if nbytes != expected_nbytes {
        return Err(invalid("child static-height streamed byte count drifted"));
    }
    Ok(HeightFileRecord {
        name,
        file,
        shape: vec![levels.len(), grid.ny(), grid.nx()],
        nbytes,
        sha256: sha256::hex(&hasher.finalize()),
        minimum_f32_bits: minimum.to_bits(),
        maximum_f32_bits: maximum.to_bits(),
    })
}

fn create_height_file(
    path: &Path,
    terrain: &[f32],
    levels: &[f32],
    grid: ChildGridSpec,
    name: &'static str,
    file_name: &'static str,
) -> Result<HeightFileRecord> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::with_capacity(STREAM_CHUNK_BYTES, file);
    let record = stream_height_file(&mut writer, terrain, levels, grid, name, file_name)?;
    writer.flush()?;
    let file = writer
        .into_inner()
        .map_err(|error| invalid(format!("could not finalize child static height: {error}")))?;
    file.sync_all()?;
    Ok(record)
}

fn compare_height_file(
    path: &Path,
    terrain: &[f32],
    levels: &[f32],
    grid: ChildGridSpec,
    name: &'static str,
    file_name: &'static str,
) -> Result<HeightFileRecord> {
    let file = File::open(path)?;
    let mut comparator = CompareWriter::new(BufReader::with_capacity(STREAM_CHUNK_BYTES, file));
    let record = stream_height_file(&mut comparator, terrain, levels, grid, name, file_name)?;
    comparator.finish()?;
    Ok(record)
}

fn build_height_files(
    prepared: &PreparedStaticInputs,
    directory: &Path,
) -> Result<(HeightFileRecord, HeightFileRecord)> {
    let vertical = sealed_child_stretch_grid()?;
    let zh = create_height_file(
        &directory.join(CHILD_STATIC_HEIGHT_ZH_FILE),
        &prepared.terrain_words,
        &vertical.sigma[1..=prepared.grid.nz()],
        prepared.grid,
        "zh",
        CHILD_STATIC_HEIGHT_ZH_FILE,
    )?;
    let zf = create_height_file(
        &directory.join(CHILD_STATIC_HEIGHT_ZF_FILE),
        &prepared.terrain_words,
        &vertical.sigmaf[1..=prepared.grid.nz() + 1],
        prepared.grid,
        "zf",
        CHILD_STATIC_HEIGHT_ZF_FILE,
    )?;
    Ok((zh, zf))
}

fn compare_height_files(
    prepared: &PreparedStaticInputs,
    directory: &Path,
) -> Result<(HeightFileRecord, HeightFileRecord)> {
    let vertical = sealed_child_stretch_grid()?;
    let zh = compare_height_file(
        &directory.join(CHILD_STATIC_HEIGHT_ZH_FILE),
        &prepared.terrain_words,
        &vertical.sigma[1..=prepared.grid.nz()],
        prepared.grid,
        "zh",
        CHILD_STATIC_HEIGHT_ZH_FILE,
    )?;
    let zf = compare_height_file(
        &directory.join(CHILD_STATIC_HEIGHT_ZF_FILE),
        &prepared.terrain_words,
        &vertical.sigmaf[1..=prepared.grid.nz() + 1],
        prepared.grid,
        "zf",
        CHILD_STATIC_HEIGHT_ZF_FILE,
    )?;
    Ok((zh, zf))
}

#[allow(clippy::too_many_arguments)]
pub fn publish_child_static_heights_for_grid_new(
    grid: ChildGridSpec,
    terrain_directory: impl AsRef<Path>,
    terrain: ChildTerrainArtifactIdentity,
    init_terrain_source: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
) -> Result<ChildStaticHeightGridReceipt> {
    let prepared = prepare_inputs(
        grid,
        terrain_directory.as_ref(),
        terrain,
        init_terrain_source.as_ref(),
        producer_source.as_ref(),
        producer_executable.as_ref(),
    )?;
    let output = output_directory.as_ref();
    let staging = create_staging(output)?;
    let result = (|| -> Result<ChildStaticHeightGridReceipt> {
        let (zh, zf) = build_height_files(&prepared, &staging)?;
        let manifest_bytes = manifest_value(&prepared, &zh, &zf).canonical_bytes();
        let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
        let receipt_template = ChildStaticHeightGridReceipt {
            grid,
            manifest_sha256,
            zh_sha256: zh.sha256.clone(),
            zf_sha256: zf.sha256.clone(),
            receipt_sha256: String::new(),
            producer_source_sha256: prepared.producer_source_sha256.clone(),
            producer_executable_sha256: prepared.producer_executable_sha256.clone(),
            terrain: prepared.terrain.clone(),
        };
        let receipt_bytes = receipt_value(&receipt_template, &zh, &zf).canonical_bytes();
        let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
        write_new_synced(
            &staging.join(CHILD_STATIC_HEIGHT_MANIFEST_FILE),
            &manifest_bytes,
        )?;
        write_new_synced(
            &staging.join(CHILD_STATIC_HEIGHT_RECEIPT_FILE),
            &receipt_bytes,
        )?;
        verify_static_inventory(&staging)?;
        fs::rename(&staging, output)?;
        Ok(ChildStaticHeightGridReceipt {
            receipt_sha256,
            ..receipt_template
        })
    })();
    if result.is_err() {
        cleanup_staging(&staging);
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub fn read_child_static_heights_for_grid_artifact(
    grid: ChildGridSpec,
    terrain_directory: impl AsRef<Path>,
    terrain: ChildTerrainArtifactIdentity,
    init_terrain_source: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_zh_sha256: &str,
    expected_zf_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<ChildStaticHeightGridReceipt> {
    for (label, value) in [
        ("static-height manifest", expected_manifest_sha256),
        ("static-height zh", expected_zh_sha256),
        ("static-height zf", expected_zf_sha256),
        ("static-height receipt", expected_receipt_sha256),
    ] {
        validate_sha256(value, label)?;
    }
    let prepared = prepare_inputs(
        grid,
        terrain_directory.as_ref(),
        terrain,
        init_terrain_source.as_ref(),
        producer_source.as_ref(),
        producer_executable.as_ref(),
    )?;
    let directory = directory.as_ref();
    verify_static_inventory(directory)?;
    let manifest_bytes = read_regular_bounded(
        &directory.join(CHILD_STATIC_HEIGHT_MANIFEST_FILE),
        JSON_MAX_BYTES,
        "child static-height manifest",
    )?;
    let receipt_bytes = read_regular_bounded(
        &directory.join(CHILD_STATIC_HEIGHT_RECEIPT_FILE),
        JSON_MAX_BYTES,
        "child static-height receipt",
    )?;
    if sha256::digest_hex(&manifest_bytes) != expected_manifest_sha256
        || sha256::digest_hex(&receipt_bytes) != expected_receipt_sha256
        || hash_regular_exact(
            &directory.join(CHILD_STATIC_HEIGHT_ZH_FILE),
            height_nbytes(grid, grid.nz())?,
            "child static zh",
        )? != expected_zh_sha256
        || hash_regular_exact(
            &directory.join(CHILD_STATIC_HEIGHT_ZF_FILE),
            height_nbytes(grid, grid.nz() + 1)?,
            "child static zf",
        )? != expected_zf_sha256
    {
        return Err(invalid(
            "child static-height artifact differs from its independent hashes",
        ));
    }
    let (zh, zf) = compare_height_files(&prepared, directory)?;
    if zh.sha256 != expected_zh_sha256 || zf.sha256 != expected_zf_sha256 {
        return Err(invalid(
            "child static-height artifact differs from its Rust source recomputation",
        ));
    }
    if manifest_bytes != manifest_value(&prepared, &zh, &zf).canonical_bytes() {
        return Err(invalid(
            "child static-height manifest differs from its source-bound contract",
        ));
    }
    let mut receipt = ChildStaticHeightGridReceipt {
        grid,
        manifest_sha256: expected_manifest_sha256.to_owned(),
        zh_sha256: expected_zh_sha256.to_owned(),
        zf_sha256: expected_zf_sha256.to_owned(),
        receipt_sha256: String::new(),
        producer_source_sha256: prepared.producer_source_sha256.clone(),
        producer_executable_sha256: prepared.producer_executable_sha256.clone(),
        terrain: prepared.terrain.clone(),
    };
    if receipt_bytes != receipt_value(&receipt, &zh, &zf).canonical_bytes() {
        return Err(invalid(
            "child static-height receipt differs from its source-bound contract",
        ));
    }
    receipt.receipt_sha256 = expected_receipt_sha256.to_owned();
    Ok(receipt)
}

fn manifest_value(
    prepared: &PreparedStaticInputs,
    zh: &HeightFileRecord,
    zf: &HeightFileRecord,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_STATIC_HEIGHT_GRID_ABI)),
        ("artifact_schema", string(CHILD_STATIC_HEIGHT_GRID_SCHEMA)),
        ("authority", authority_value()),
        ("claim_limit", string(CHILD_STATIC_HEIGHT_GRID_CLAIM_LIMIT)),
        ("grid", grid_value(prepared.grid)),
        (
            "payloads",
            object(vec![("zf", height_value(zf)), ("zh", height_value(zh))]),
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
        ("schema", string(CHILD_STATIC_HEIGHT_GRID_MANIFEST_SCHEMA)),
        (
            "source_terrain",
            terrain_value(&prepared.terrain, prepared.grid),
        ),
        ("status", string(CHILD_STATIC_HEIGHT_GRID_STATUS)),
        ("synthetic_detail", JsonValue::Bool(false)),
        ("synthetic_translation", JsonValue::Bool(false)),
        ("transform", transform_value()),
        ("vertical", vertical_value()),
    ])
}

fn receipt_value(
    receipt: &ChildStaticHeightGridReceipt,
    zh: &HeightFileRecord,
    zf: &HeightFileRecord,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_STATIC_HEIGHT_GRID_ABI)),
        ("all_finite", JsonValue::Bool(true)),
        ("claim_limit", string(CHILD_STATIC_HEIGHT_GRID_CLAIM_LIMIT)),
        ("grid", grid_value(receipt.grid)),
        ("manifest_sha256", string(&receipt.manifest_sha256)),
        (
            "payloads",
            object(vec![("zf", height_value(zf)), ("zh", height_value(zh))]),
        ),
        (
            "producer_executable_sha256",
            string(&receipt.producer_executable_sha256),
        ),
        (
            "producer_source_sha256",
            string(&receipt.producer_source_sha256),
        ),
        ("schema", string(CHILD_STATIC_HEIGHT_GRID_SCHEMA)),
        (
            "source_terrain",
            terrain_value(&receipt.terrain, receipt.grid),
        ),
        ("status", string(CHILD_STATIC_HEIGHT_GRID_STATUS)),
        ("synthetic_detail", JsonValue::Bool(false)),
        ("synthetic_translation", JsonValue::Bool(false)),
        ("vertical", vertical_value()),
    ])
}

fn height_value(record: &HeightFileRecord) -> JsonValue {
    object(vec![
        ("dtype", string("little-endian-f32")),
        ("file", string(record.file)),
        (
            "maximum_f32_bits",
            string(&format!("{:08x}", record.maximum_f32_bits)),
        ),
        (
            "minimum_f32_bits",
            string(&format!("{:08x}", record.minimum_f32_bits)),
        ),
        ("name", string(record.name)),
        ("nbytes", u64_number(record.nbytes)),
        ("order", string("C[k,j,i]")),
        ("sha256", string(&record.sha256)),
        ("shape_kji", number_array(&record.shape)),
    ])
}

fn terrain_value(identity: &ChildTerrainArtifactIdentity, grid: ChildGridSpec) -> JsonValue {
    let (schema, abi) = if grid.nx() == 300 && grid.ny() == 300 {
        (CHILD_TERRAIN_SCHEMA, CHILD_TERRAIN_ABI)
    } else {
        (CHILD_TERRAIN_GRID_SCHEMA, CHILD_TERRAIN_GRID_ABI)
    };
    object(vec![
        ("abi", string(abi)),
        ("manifest_sha256", string(&identity.manifest_sha256)),
        ("payload_sha256", string(&identity.payload_sha256)),
        ("receipt_sha256", string(&identity.receipt_sha256)),
        ("schema", string(schema)),
    ])
}

fn grid_value(grid: ChildGridSpec) -> JsonValue {
    object(vec![
        ("dx_m_f64_bits", f64_bits(grid.dx_m())),
        ("dy_m_f64_bits", f64_bits(grid.dy_m())),
        ("nx", number(grid.nx())),
        ("ny", number(grid.ny())),
        ("nz", number(grid.nz())),
        ("ztop_m_f64_bits", f64_bits(grid.ztop_m())),
    ])
}

fn authority_value() -> JsonValue {
    object(vec![
        (
            "cuda_cross_oracle_source_sha256",
            string(CUDA_STRETCH_Z1_TERRAIN_SOURCE_SHA256),
        ),
        (
            "init_terrain",
            object(vec![
                ("anchors", string("160-297")),
                ("sha256", string(CM1_INIT_TERRAIN_SHA256)),
            ]),
        ),
        ("model", string("GM1")),
        ("model_source", string("CM1 r21.1")),
        ("tarball_sha256", string(CM1_R21_AUTHORITY_SHA256)),
    ])
}

fn vertical_value() -> JsonValue {
    object(vec![
        ("abi", string(CHILD_STRETCH_ABI)),
        ("combined_grid_sha256", string(CHILD_STRETCH_GRID_SHA256)),
        ("schema", string(CHILD_STRETCH_SCHEMA)),
    ])
}

fn transform_value() -> JsonValue {
    object(vec![
        (
            "formula",
            string("f32(surface + f32(f32(sigma * f32(ztop-surface)) / ztop))"),
        ),
        ("output_order", string("level-major C[k,j,i]")),
        (
            "rounding",
            string("each subtraction, multiplication, division, and addition is one binary32 round-to-nearest-ties-to-even operation in CM1/CUDA source order"),
        ),
        ("stream_chunk_bytes", number(STREAM_CHUNK_BYTES)),
    ])
}

struct CompareWriter<R> {
    reader: R,
    offset: u64,
}

impl<R: Read> CompareWriter<R> {
    fn new(reader: R) -> Self {
        Self { reader, offset: 0 }
    }

    fn finish(mut self) -> Result<()> {
        let mut extra = [0u8; 1];
        if self.reader.read(&mut extra)? != 0 {
            return Err(invalid("child static-height file has trailing bytes"));
        }
        Ok(())
    }
}

impl<R: Read> Write for CompareWriter<R> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let mut observed = vec![0u8; bytes.len()];
        self.reader.read_exact(&mut observed)?;
        if observed != bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "child static-height bytes differ at or before offset {}",
                    self.offset
                ),
            ));
        }
        self.offset += bytes.len() as u64;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn create_staging(output: &Path) -> Result<std::path::PathBuf> {
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace child static-height destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "child static-height destination parent does not exist: {}",
            parent.display()
        )));
    }
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("child static-height destination needs a UTF-8 name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let staging =
        output.with_file_name(format!(".{name}.partial-{}-{sequence}", std::process::id()));
    fs::create_dir(&staging)?;
    Ok(staging)
}

fn cleanup_staging(staging: &Path) {
    for file in [
        CHILD_STATIC_HEIGHT_MANIFEST_FILE,
        CHILD_STATIC_HEIGHT_RECEIPT_FILE,
        CHILD_STATIC_HEIGHT_ZH_FILE,
        CHILD_STATIC_HEIGHT_ZF_FILE,
    ] {
        let _ = fs::remove_file(staging.join(file));
    }
    let _ = fs::remove_dir(staging);
}

fn verify_static_inventory(directory: &Path) -> Result<()> {
    verify_inventory(
        directory,
        &[
            CHILD_STATIC_HEIGHT_MANIFEST_FILE,
            CHILD_STATIC_HEIGHT_RECEIPT_FILE,
            CHILD_STATIC_HEIGHT_ZF_FILE,
            CHILD_STATIC_HEIGHT_ZH_FILE,
        ],
        "child static-height",
    )
}

fn verify_terrain_inventory(directory: &Path) -> Result<()> {
    verify_inventory(
        directory,
        &[
            CHILD_TERRAIN_MANIFEST_FILE,
            CHILD_TERRAIN_PAYLOAD_FILE,
            CHILD_TERRAIN_RECEIPT_FILE,
        ],
        "child terrain",
    )
}

fn verify_inventory(directory: &Path, expected: &[&str], label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(directory)
        .map_err(|error| invalid(format!("could not inspect {label} directory: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!("{label} input is not a real directory")));
    }
    let expected = expected
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(invalid(format!("{label} contains a non-regular entry")));
        }
        actual.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| invalid(format!("{label} contains a non-UTF-8 name")))?,
        );
    }
    if actual != expected {
        return Err(invalid(format!(
            "{label} inventory is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn read_regular_exact(path: &Path, expected: u64, label: &str) -> Result<Vec<u8>> {
    let bytes = read_regular_bounded(path, expected, label)?;
    if bytes.len() as u64 != expected {
        return Err(invalid(format!(
            "{label} has {} bytes; expected {expected}",
            bytes.len()
        )));
    }
    Ok(bytes)
}

fn read_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("could not inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        return Err(invalid(format!("{label} is not a bounded regular file")));
    }
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len()).map_err(|_| invalid(format!("{label} is too large")))?,
    );
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() as u64 != metadata.len() {
        return Err(invalid(format!("{label} size changed while reading")));
    }
    Ok(bytes)
}

fn hash_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<String> {
    Ok(sha256::digest_hex(&read_regular_bounded(
        path, maximum, label,
    )?))
}

fn hash_regular_exact(path: &Path, expected: u64, label: &str) -> Result<String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("could not inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != expected {
        return Err(invalid(format!("{label} is not the expected regular file")));
    }
    let mut reader = BufReader::with_capacity(STREAM_CHUNK_BYTES, File::open(path)?);
    let mut hasher = Sha256::default();
    let mut buffer = vec![0u8; STREAM_CHUNK_BYTES];
    let mut remaining = expected;
    while remaining != 0 {
        let amount = usize::try_from(remaining.min(STREAM_CHUNK_BYTES as u64))
            .expect("bounded hash chunk fits usize");
        reader.read_exact(&mut buffer[..amount])?;
        hasher.update(&buffer[..amount]);
        remaining -= amount as u64;
    }
    Ok(sha256::hex(&hasher.finalize()))
}

fn verify_exact_hash(path: &Path, maximum: u64, expected: &str, label: &str) -> Result<()> {
    if hash_regular_bounded(path, maximum, label)? != expected {
        return Err(invalid(format!("{label} differs from its exact SHA-256")));
    }
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!(
            "{label} is not a lowercase SHA-256 digest"
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

fn u64_number(value: u64) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn number_array(values: &[usize]) -> JsonValue {
    JsonValue::Array(values.iter().map(|value| number(*value)).collect())
}

fn f64_bits(value: f64) -> JsonValue {
    string(&format!("{:016x}", value.to_bits()))
}

/// Read one published static-height artifact's identity from its own receipt.
///
/// The grid-generic publisher writes a receipt that records the child grid, its
/// own manifest digest, both payload digests, the producer identities, and the
/// terrain the heights were computed from.  Reading that back is what lets a
/// consumer admit any published heights - at any admitted child grid, from any
/// parent - instead of one frozen 300/50 or 3000/5 identity.
///
/// The refusal is the property, not the bytes: a receipt that does not anchor
/// the manifest present beside it, or that describes a grid whose zh/zf byte
/// counts are not the files on disk, would hand the atmosphere producer heights
/// whose vertical structure does not match the arena it is about to fill.
pub fn read_child_static_height_identity(
    directory: impl AsRef<Path>,
) -> Result<ChildStaticHeightGridReceipt> {
    let directory = directory.as_ref();
    let manifest_path = directory.join(CHILD_STATIC_HEIGHT_MANIFEST_FILE);
    let receipt_path = directory.join(CHILD_STATIC_HEIGHT_RECEIPT_FILE);
    let manifest_sha256 = hash_regular_bounded(
        &manifest_path,
        JSON_MAX_BYTES,
        "child static-height manifest",
    )?;
    let (receipt_bytes, receipt) = crate::child_artifact::read_canonical_json(
        &receipt_path,
        JSON_MAX_BYTES,
        "child static-height receipt",
    )?;
    let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    let field = |path: &[&str]| {
        crate::child_artifact::json_string_at(&receipt, path, "child static-height receipt")
    };
    if field(&["schema"])? != CHILD_STATIC_HEIGHT_GRID_SCHEMA
        || field(&["status"])? != CHILD_STATIC_HEIGHT_GRID_STATUS
        || field(&["abi"])? != CHILD_STATIC_HEIGHT_GRID_ABI
    {
        return Err(invalid(
            "child static-height receipt is not the grid-generic static-height family; its zh/zf layout is undescribed and the atmosphere producer would read heights at another artifact's offsets",
        ));
    }
    if crate::child_artifact::json_sha256_at(
        &receipt,
        &["manifest_sha256"],
        "child static-height receipt",
    )? != manifest_sha256
    {
        return Err(invalid(
            "child static-height manifest bytes are not the manifest its own receipt anchors; the published heights and the published layout describe different artifacts",
        ));
    }
    let grid_value =
        crate::child_artifact::json_at(&receipt, &["grid"], "child static-height receipt")?;
    let extent = |key: &str| {
        crate::child_artifact::json_u64_at(grid_value, &[key], "child static-height grid").and_then(
            |value| {
                usize::try_from(value)
                    .map_err(|_| invalid(format!("child static-height grid {key} overflows")))
            },
        )
    };
    let bits = |key: &str| -> Result<f64> {
        let word =
            crate::child_artifact::json_string_at(grid_value, &[key], "child static-height grid")?;
        if word.len() != 16
            || !word
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(invalid(format!(
                "child static-height grid {key} is not sixteen lower-case hex digits"
            )));
        }
        Ok(f64::from_bits(u64::from_str_radix(&word, 16).map_err(
            |_| invalid(format!("child static-height grid {key} does not parse")),
        )?))
    };
    let grid = ChildGridSpec::new(
        extent("nx")?,
        extent("ny")?,
        extent("nz")?,
        bits("dx_m_f64_bits")?,
        bits("dy_m_f64_bits")?,
        bits("ztop_m_f64_bits")?,
    )?;
    let (zh_nbytes, zf_nbytes) = child_static_height_nbytes_for_grid(grid)?;
    let mut digests = Vec::new();
    for (name, file, expected_nbytes) in [
        ("zh", CHILD_STATIC_HEIGHT_ZH_FILE, zh_nbytes),
        ("zf", CHILD_STATIC_HEIGHT_ZF_FILE, zf_nbytes),
    ] {
        let recorded_nbytes = crate::child_artifact::json_u64_at(
            &receipt,
            &["payloads", name, "nbytes"],
            "child static-height receipt",
        )?;
        let path = directory.join(file);
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(invalid(format!(
                "child static-height {name} is not a regular file"
            )));
        }
        if recorded_nbytes != expected_nbytes || metadata.len() != expected_nbytes {
            return Err(invalid(format!(
                "child static-height {name} is {} bytes and its receipt records {recorded_nbytes}, but the {}x{}x{} grid it declares needs {expected_nbytes}; the atmosphere producer would interpolate onto a column depth these heights do not have",
                metadata.len(),
                grid.nz(),
                grid.ny(),
                grid.nx()
            )));
        }
        digests.push(crate::child_artifact::json_sha256_at(
            &receipt,
            &["payloads", name, "sha256"],
            "child static-height receipt",
        )?);
    }
    let terrain = ChildTerrainArtifactIdentity::new(
        crate::child_artifact::json_sha256_at(
            &receipt,
            &["source_terrain", "manifest_sha256"],
            "child static-height receipt",
        )?,
        crate::child_artifact::json_sha256_at(
            &receipt,
            &["source_terrain", "payload_sha256"],
            "child static-height receipt",
        )?,
        crate::child_artifact::json_sha256_at(
            &receipt,
            &["source_terrain", "receipt_sha256"],
            "child static-height receipt",
        )?,
    )?;
    Ok(ChildStaticHeightGridReceipt {
        grid,
        manifest_sha256,
        zh_sha256: digests[0].clone(),
        zf_sha256: digests[1].clone(),
        receipt_sha256,
        producer_source_sha256: crate::child_artifact::json_sha256_at(
            &receipt,
            &["producer_source_sha256"],
            "child static-height receipt",
        )?,
        producer_executable_sha256: crate::child_artifact::json_sha256_at(
            &receipt,
            &["producer_executable_sha256"],
            "child static-height receipt",
        )?,
        terrain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child_remap::{ACCEPTED_50M_CHILD_GRID, HARD_5M_CHILD_GRID};

    #[test]
    fn static_height_layout_scales_without_allocation() {
        assert_eq!(
            child_static_height_nbytes_for_grid(ACCEPTED_50M_CHILD_GRID).unwrap(),
            (27_000_000, 27_360_000)
        );
        assert_eq!(
            child_static_height_nbytes_for_grid(HARD_5M_CHILD_GRID).unwrap(),
            (2_700_000_000, 2_736_000_000)
        );
    }

    #[test]
    fn terrain_height_uses_the_pinned_binary32_source_order() {
        let surface = f32::from_bits(0x43a1_2345);
        let sigma = f32::from_bits(0x44cb_1357);
        let expected = add32(
            surface,
            div32(mul32(sigma, sub32(15_000.0, surface)), 15_000.0),
        );
        assert_eq!(terrain_height(surface, sigma).to_bits(), expected.to_bits());
    }

    #[test]
    fn receipt_refuses_synthetic_dispositions() {
        let terrain =
            ChildTerrainArtifactIdentity::new("11".repeat(32), "22".repeat(32), "33".repeat(32))
                .unwrap();
        let record = HeightFileRecord {
            name: "zh",
            file: CHILD_STATIC_HEIGHT_ZH_FILE,
            shape: vec![75, 300, 300],
            nbytes: 27_000_000,
            sha256: "44".repeat(32),
            minimum_f32_bits: 0,
            maximum_f32_bits: 1,
        };
        let receipt = ChildStaticHeightGridReceipt {
            grid: ACCEPTED_50M_CHILD_GRID,
            manifest_sha256: "55".repeat(32),
            zh_sha256: record.sha256.clone(),
            zf_sha256: "66".repeat(32),
            receipt_sha256: String::new(),
            producer_source_sha256: "77".repeat(32),
            producer_executable_sha256: "88".repeat(32),
            terrain,
        };
        let value = receipt_value(&receipt, &record, &record);
        assert_eq!(
            value.get("synthetic_detail").unwrap(),
            &JsonValue::Bool(false)
        );
        assert_eq!(
            value.get("synthetic_translation").unwrap(),
            &JsonValue::Bool(false)
        );
    }
}
