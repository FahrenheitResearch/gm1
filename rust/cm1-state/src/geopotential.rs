//! Exact Rust-owned CM1 z-face height to ArWen WSM6 geopotential transform.
//!
//! The auxiliary is deliberately separate from the frozen 49-carrier ABI.
//! It binds to an independently verified `CM1IPR1` artifact and publishes one
//! binary32 `total_geopotential_zf` carrier for the public ArWen WSM6 `phb`
//! operand.  The consumer-owned `php` operand is exact zero at initialization.
//! Binary32 multiply/divide is not an identity for every `zf` word, so the
//! envelope records measured recovery error instead of claiming impossible
//! raw-bit height recovery.

use crate::json::JsonValue;
use crate::product::INITIALIZED_PRODUCT_CARRIER_ABI;
use crate::product_restart::{RestoredProductRestart, INITIALIZED_PRODUCT_RESTART_SCHEMA};
use crate::sha256;
use crate::{Result, StateError};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const WSM6_GEOPOTENTIAL_SCHEMA: &str = "gpuwm-cm1.arwen-wsm6-geopotential-auxiliary/v1";
pub const WSM6_GEOPOTENTIAL_RECEIPT_SCHEMA: &str =
    "gpuwm-cm1.arwen-wsm6-geopotential-auxiliary-receipt/v1";
pub const WSM6_GEOPOTENTIAL_CARRIER_ABI: &str = "gpuwm-cm1.arwen-wsm6-total-geopotential/v1";
pub const WSM6_GEOPOTENTIAL_PAYLOAD_FILE: &str = "total-geopotential-zf.f32le";
pub const WSM6_GEOPOTENTIAL_MANIFEST_FILE: &str = "manifest.json";
pub const WSM6_GEOPOTENTIAL_RECEIPT_FILE: &str = "receipt.json";
pub const WSM6_GEOPOTENTIAL_CARRIER_NAME: &str = "total_geopotential_zf";
pub const WSM6_GEOPOTENTIAL_GRAVITY_F32_BITS: u32 = 0x411c_f5c3;
pub const WSM6_GEOPOTENTIAL_CLAIM_LIMIT: &str = "exact Rust binary32 CM1 zf-to-ArWen total-geopotential transform and measured WSM6 recovery bounds; no raw zf recovery identity, php evolution, halos, open-boundary history or tendencies, microphysics execution, component persistence, coupled trajectory, forecast, tornado, or model-validation claim";
pub const WSM6_GEOPOTENTIAL_PUBLICATION_CONTRACT: &str = "write the exact binary32 payload, canonical manifest, and canonical receipt into a new sibling staging directory; synchronize every file; rename only after all source, transform, and output hashes validate; refuse an existing destination";
pub const WSM6_GEOPOTENTIAL_COMPARISON_STATEMENT: &str = "geopotential = binary32(zf * G_binary32); recovered_z = binary32(geopotential / G_binary32), in that Rust statement order with IEEE-754 round-to-nearest ties-to-even";
pub const WSM6_GEOPOTENTIAL_RECOVERY_METRIC: &str = "for finite nonnegative zf and recovered_z, ULP distance is the absolute difference of monotonically ordered IEEE-754 binary32 bit patterns; absolute and relative errors use exact binary32-to-binary64 conversion and relative error divides by abs(zf), with exact zero mapped to zero relative error";

const CONSUMER_COMMIT: &str = "abb5ff270f263f15b89a272e9b17648406ef0f1d";
const CONSUMER_FILES: [(&str, &str, &str); 2] = [
    (
        "gpuwm/core/constants.py",
        "76c8c11d1a75d17fd54d0a80646cec9dde04f649247ec2347d3d32e6c0500e9e",
        "24",
    ),
    (
        "gpuwm/core/wsm6.py",
        "3ba72f8e6f3e4453d7c864e0556804a89da2f161f238043ab7670191e5d6af3d",
        "102-111",
    ),
];
const MAX_JSON_BYTES: u64 = 1024 * 1024;
const MAX_PAYLOAD_BYTES: u64 = 512 * 1024 * 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeopotentialRecoveryMetrics {
    pub word_count: usize,
    pub recovered_count: usize,
    pub mismatch_count: usize,
    pub maximum_ulp_distance: u32,
    pub maximum_absolute_error_m_f64_bits: u64,
    pub maximum_relative_error_f64_bits: u64,
}

impl GeopotentialRecoveryMetrics {
    fn json(&self) -> JsonValue {
        object(vec![
            (
                "comparison_statement_order",
                string(WSM6_GEOPOTENTIAL_COMPARISON_STATEMENT),
            ),
            (
                "maximum_absolute_error_m_f64_bits",
                string(&format!("{:016x}", self.maximum_absolute_error_m_f64_bits)),
            ),
            (
                "maximum_relative_error_f64_bits",
                string(&format!("{:016x}", self.maximum_relative_error_f64_bits)),
            ),
            (
                "maximum_ulp_distance",
                u64_number(u64::from(self.maximum_ulp_distance)),
            ),
            ("metric", string(WSM6_GEOPOTENTIAL_RECOVERY_METRIC)),
            ("mismatch_count", number(self.mismatch_count)),
            ("recovered_count", number(self.recovered_count)),
            ("word_count", number(self.word_count)),
        ])
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wsm6GeopotentialReceipt {
    pub manifest_sha256: String,
    pub manifest_nbytes: usize,
    pub payload_sha256: String,
    pub payload_nbytes: u64,
    pub receipt_sha256: String,
    pub source_product_manifest_sha256: String,
    pub source_product_payload_sha256: String,
    pub source_restart_artifact_sha256: String,
    pub recovery: GeopotentialRecoveryMetrics,
}

impl Wsm6GeopotentialReceipt {
    fn receipt_value(&self) -> JsonValue {
        object(vec![
            ("carrier_abi", string(WSM6_GEOPOTENTIAL_CARRIER_ABI)),
            ("carrier_count", number(1)),
            ("claim_limit", string(WSM6_GEOPOTENTIAL_CLAIM_LIMIT)),
            ("manifest_nbytes", number(self.manifest_nbytes)),
            ("manifest_sha256", string(&self.manifest_sha256)),
            ("payload_nbytes", u64_number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            (
                "publication_contract",
                string(WSM6_GEOPOTENTIAL_PUBLICATION_CONTRACT),
            ),
            ("recovery", self.recovery.json()),
            ("schema", string(WSM6_GEOPOTENTIAL_RECEIPT_SCHEMA)),
            (
                "source_product_manifest_sha256",
                string(&self.source_product_manifest_sha256),
            ),
            (
                "source_product_payload_sha256",
                string(&self.source_product_payload_sha256),
            ),
            (
                "source_restart_artifact_sha256",
                string(&self.source_restart_artifact_sha256),
            ),
        ])
    }

    pub fn canonical_json(&self) -> String {
        String::from_utf8(self.receipt_value().canonical_bytes())
            .expect("canonical geopotential receipt is UTF-8")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Wsm6GeopotentialAuxiliary {
    pub receipt: Wsm6GeopotentialReceipt,
    pub shape: Vec<usize>,
}

#[derive(Debug)]
struct Transform {
    payload: Vec<u8>,
    payload_sha256: String,
    minimum_f32_bits: u32,
    maximum_f32_bits: u32,
    nonzero_count: usize,
    recovery: GeopotentialRecoveryMetrics,
}

/// Derive and atomically publish the exact ArWen WSM6 geopotential auxiliary.
pub fn export_wsm6_geopotential_new(
    source: &RestoredProductRestart,
    output_directory: impl AsRef<Path>,
) -> Result<Wsm6GeopotentialReceipt> {
    let zf = source
        .carrier("zf")
        .ok_or_else(|| invalid("initialized-product restart lacks required zf carrier"))?;
    if zf.shape.len() != 3 || zf.shape.contains(&0) {
        return Err(invalid(format!(
            "initialized-product zf shape {:?} is not non-empty [nz+1,ny,nx]",
            zf.shape
        )));
    }
    let zf_bytes = source.read_carrier_f32le_exact("zf")?;
    let transform = transform_zf(&zf_bytes)?;
    let manifest = manifest_value(source, &transform);
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let mut receipt = Wsm6GeopotentialReceipt {
        manifest_sha256,
        manifest_nbytes: manifest_bytes.len(),
        payload_sha256: transform.payload_sha256.clone(),
        payload_nbytes: transform.payload.len() as u64,
        receipt_sha256: String::new(),
        source_product_manifest_sha256: source.receipt().product_manifest_sha256.clone(),
        source_product_payload_sha256: source.receipt().payload_sha256.clone(),
        source_restart_artifact_sha256: source.receipt().artifact_sha256.clone(),
        recovery: transform.recovery.clone(),
    };
    let receipt_bytes = receipt.receipt_value().canonical_bytes();
    receipt.receipt_sha256 = sha256::digest_hex(&receipt_bytes);

    let output = output_directory.as_ref();
    refuse_existing_directory(output)?;
    let staging = staging_path(output)?;
    fs::create_dir(&staging)?;
    let result = (|| -> Result<()> {
        write_new_synced(
            &staging.join(WSM6_GEOPOTENTIAL_PAYLOAD_FILE),
            &transform.payload,
        )?;
        write_new_synced(
            &staging.join(WSM6_GEOPOTENTIAL_MANIFEST_FILE),
            &manifest_bytes,
        )?;
        write_new_synced(
            &staging.join(WSM6_GEOPOTENTIAL_RECEIPT_FILE),
            &receipt_bytes,
        )?;
        verify_exact_directory_inventory(&staging)?;
        if fs::metadata(staging.join(WSM6_GEOPOTENTIAL_PAYLOAD_FILE))?.len()
            != receipt.payload_nbytes
        {
            return Err(invalid(
                "geopotential payload size changed after synchronized write",
            ));
        }
        fs::rename(&staging, output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result?;
    Ok(receipt)
}

/// Recompute the transform from the independently verified source restart and
/// require the published auxiliary to be byte-identical to that result.
#[allow(clippy::too_many_arguments)]
pub fn read_wsm6_geopotential_auxiliary(
    source: &RestoredProductRestart,
    directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_payload_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<Wsm6GeopotentialAuxiliary> {
    for (value, label) in [
        (
            expected_manifest_sha256,
            "expected geopotential manifest SHA-256",
        ),
        (
            expected_payload_sha256,
            "expected geopotential payload SHA-256",
        ),
        (
            expected_receipt_sha256,
            "expected geopotential receipt SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    let directory = directory.as_ref();
    verify_exact_directory_inventory(directory)?;
    let manifest_bytes = read_regular_bounded(
        &directory.join(WSM6_GEOPOTENTIAL_MANIFEST_FILE),
        MAX_JSON_BYTES,
        "geopotential manifest",
    )?;
    let payload_bytes = read_regular_bounded(
        &directory.join(WSM6_GEOPOTENTIAL_PAYLOAD_FILE),
        MAX_PAYLOAD_BYTES,
        "geopotential payload",
    )?;
    let receipt_bytes = read_regular_bounded(
        &directory.join(WSM6_GEOPOTENTIAL_RECEIPT_FILE),
        MAX_JSON_BYTES,
        "geopotential receipt",
    )?;
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let payload_sha256 = sha256::digest_hex(&payload_bytes);
    let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    for (actual, expected, label) in [
        (
            manifest_sha256.as_str(),
            expected_manifest_sha256,
            "geopotential manifest",
        ),
        (
            payload_sha256.as_str(),
            expected_payload_sha256,
            "geopotential payload",
        ),
        (
            receipt_sha256.as_str(),
            expected_receipt_sha256,
            "geopotential receipt",
        ),
    ] {
        if actual != expected {
            return Err(invalid(format!(
                "{label} SHA-256 is {actual}; independently expected {expected}"
            )));
        }
    }
    parse_canonical_json(&manifest_bytes, "geopotential manifest")?;
    parse_canonical_json(&receipt_bytes, "geopotential receipt")?;

    let zf = source
        .carrier("zf")
        .ok_or_else(|| invalid("initialized-product restart lacks required zf carrier"))?;
    let zf_bytes = source.read_carrier_f32le_exact("zf")?;
    let transform = transform_zf(&zf_bytes)?;
    if payload_bytes != transform.payload {
        return Err(invalid(
            "geopotential payload is not the exact Rust binary32 transform of source zf",
        ));
    }
    let expected_manifest = manifest_value(source, &transform).canonical_bytes();
    if manifest_bytes != expected_manifest {
        return Err(invalid(
            "geopotential manifest differs from the exact source-bound v1 contract",
        ));
    }
    let mut receipt = Wsm6GeopotentialReceipt {
        manifest_sha256,
        manifest_nbytes: manifest_bytes.len(),
        payload_sha256,
        payload_nbytes: payload_bytes.len() as u64,
        receipt_sha256: String::new(),
        source_product_manifest_sha256: source.receipt().product_manifest_sha256.clone(),
        source_product_payload_sha256: source.receipt().payload_sha256.clone(),
        source_restart_artifact_sha256: source.receipt().artifact_sha256.clone(),
        recovery: transform.recovery,
    };
    let expected_receipt = receipt.receipt_value().canonical_bytes();
    if receipt_bytes != expected_receipt {
        return Err(invalid(
            "geopotential receipt differs from the exact source-bound v1 contract",
        ));
    }
    receipt.receipt_sha256 = receipt_sha256;
    Ok(Wsm6GeopotentialAuxiliary {
        receipt,
        shape: zf.shape.clone(),
    })
}

fn transform_zf(zf_bytes: &[u8]) -> Result<Transform> {
    if zf_bytes.is_empty() || !zf_bytes.len().is_multiple_of(4) {
        return Err(invalid(
            "source zf bytes are empty or not aligned binary32 words",
        ));
    }
    let gravity = f32::from_bits(WSM6_GEOPOTENTIAL_GRAVITY_F32_BITS);
    let word_count = zf_bytes.len() / 4;
    let mut payload = Vec::with_capacity(zf_bytes.len());
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut nonzero_count = 0usize;
    let mut recovered_count = 0usize;
    let mut maximum_ulp_distance = 0u32;
    let mut maximum_absolute_error = 0.0f64;
    let mut maximum_relative_error = 0.0f64;
    for (index, word) in zf_bytes.chunks_exact(4).enumerate() {
        let zf = f32::from_le_bytes(word.try_into().expect("four-byte source word"));
        if !zf.is_finite() || zf.is_sign_negative() {
            return Err(invalid(format!(
                "source zf word {index} is not finite nonnegative binary32"
            )));
        }
        let geopotential = zf * gravity;
        if !geopotential.is_finite() {
            return Err(invalid(format!(
                "geopotential word {index} overflowed binary32"
            )));
        }
        payload.extend_from_slice(&geopotential.to_le_bytes());
        minimum = minimum.min(geopotential);
        maximum = maximum.max(geopotential);
        if geopotential.to_bits() & 0x7fff_ffff != 0 {
            nonzero_count += 1;
        }
        let recovered = geopotential / gravity;
        let ulp_distance = recovered.to_bits().abs_diff(zf.to_bits());
        maximum_ulp_distance = maximum_ulp_distance.max(ulp_distance);
        if ulp_distance == 0 {
            recovered_count += 1;
        }
        let absolute_error = (f64::from(recovered) - f64::from(zf)).abs();
        let relative_error = if zf == 0.0 {
            0.0
        } else {
            absolute_error / f64::from(zf).abs()
        };
        maximum_absolute_error = maximum_absolute_error.max(absolute_error);
        maximum_relative_error = maximum_relative_error.max(relative_error);
    }
    Ok(Transform {
        payload_sha256: sha256::digest_hex(&payload),
        payload,
        minimum_f32_bits: minimum.to_bits(),
        maximum_f32_bits: maximum.to_bits(),
        nonzero_count,
        recovery: GeopotentialRecoveryMetrics {
            word_count,
            recovered_count,
            mismatch_count: word_count - recovered_count,
            maximum_ulp_distance,
            maximum_absolute_error_m_f64_bits: maximum_absolute_error.to_bits(),
            maximum_relative_error_f64_bits: maximum_relative_error.to_bits(),
        },
    })
}

fn manifest_value(source: &RestoredProductRestart, transform: &Transform) -> JsonValue {
    let zf = source
        .carrier("zf")
        .expect("verified initialized-product carrier inventory contains zf");
    object(vec![
        (
            "authority",
            object(vec![
                (
                    "cm1_product_gravity_f32_bits",
                    string(&format!("{:08x}", WSM6_GEOPOTENTIAL_GRAVITY_F32_BITS)),
                ),
                ("consumer_commit", string(CONSUMER_COMMIT)),
                (
                    "consumer_gravity_f32_bits",
                    string(&format!("{:08x}", WSM6_GEOPOTENTIAL_GRAVITY_F32_BITS)),
                ),
                (
                    "consumer_source_files",
                    JsonValue::Array(
                        CONSUMER_FILES
                            .iter()
                            .map(|(path, digest, anchors)| {
                                object(vec![
                                    ("anchors", string(anchors)),
                                    ("path", string(path)),
                                    ("sha256", string(digest)),
                                ])
                            })
                            .collect(),
                    ),
                ),
                (
                    "gravity_identity",
                    string("PASS: CM1 initialized-product and ArWen WSM6 consumer gravity are the same binary32 word"),
                ),
            ]),
        ),
        (
            "carrier",
            object(vec![
                ("dtype", string("<f4")),
                ("itemsize", number(4)),
                ("name", string(WSM6_GEOPOTENTIAL_CARRIER_NAME)),
                ("nbytes", u64_number(transform.payload.len() as u64)),
                ("offset_nbytes", number(0)),
                ("order", string("C")),
                ("recovery", transform.recovery.json()),
                ("role", string("ArWen-WSM6-base-total-geopotential")),
                (
                    "shape",
                    JsonValue::Array(zf.shape.iter().copied().map(number).collect()),
                ),
                ("sha256", string(&transform.payload_sha256)),
                (
                    "source_carrier",
                    object(vec![
                        ("name", string("zf")),
                        ("nbytes", u64_number(zf.nbytes)),
                        ("offset_nbytes", u64_number(zf.offset_nbytes)),
                        ("sha256", string(&zf.sha256)),
                    ]),
                ),
                (
                    "statistics",
                    object(vec![
                        ("finite_count", number(transform.recovery.word_count)),
                        (
                            "maximum_f32_bits",
                            string(&format!("{:08x}", transform.maximum_f32_bits)),
                        ),
                        (
                            "minimum_f32_bits",
                            string(&format!("{:08x}", transform.minimum_f32_bits)),
                        ),
                        ("nonzero_count", number(transform.nonzero_count)),
                    ]),
                ),
                ("units", string("m2 s-2")),
            ]),
        ),
        ("carrier_abi", string(WSM6_GEOPOTENTIAL_CARRIER_ABI)),
        ("claim_limit", string(WSM6_GEOPOTENTIAL_CLAIM_LIMIT)),
        (
            "consumer_binding",
            object(vec![
                ("phb", string(WSM6_GEOPOTENTIAL_CARRIER_NAME)),
                ("php", string("consumer-owned exact binary32 zero")),
                (
                    "wsm6_height_expression",
                    string("z8w = (phb + state.php) / DTYPE(c.G)"),
                ),
                ("zf_raw_identity", JsonValue::Bool(false)),
            ]),
        ),
        ("dtype", string("<f4")),
        ("itemsize", number(4)),
        ("order", string("C")),
        (
            "payload",
            object(vec![
                ("file", string(WSM6_GEOPOTENTIAL_PAYLOAD_FILE)),
                ("nbytes", u64_number(transform.payload.len() as u64)),
                ("sha256", string(&transform.payload_sha256)),
            ]),
        ),
        ("schema", string(WSM6_GEOPOTENTIAL_SCHEMA)),
        (
            "source_product",
            object(vec![
                ("carrier_abi", string(INITIALIZED_PRODUCT_CARRIER_ABI)),
                (
                    "manifest_sha256",
                    string(&source.receipt().product_manifest_sha256),
                ),
                (
                    "payload_sha256",
                    string(&source.receipt().payload_sha256),
                ),
                (
                    "receipt_sha256",
                    string(&source.receipt().product_receipt_sha256),
                ),
                (
                    "source_artifact_sha256",
                    string(&source.receipt().source_artifact_sha256),
                ),
                (
                    "source_contract_sha256",
                    string(&source.receipt().source_contract_sha256),
                ),
            ]),
        ),
        (
            "source_restart",
            object(vec![
                (
                    "artifact_nbytes",
                    u64_number(source.receipt().artifact_nbytes),
                ),
                (
                    "artifact_sha256",
                    string(&source.receipt().artifact_sha256),
                ),
                ("schema", string(INITIALIZED_PRODUCT_RESTART_SCHEMA)),
            ]),
        ),
        (
            "transform",
            object(vec![
                (
                    "consumer_gravity_f32_bits",
                    string(&format!("{:08x}", WSM6_GEOPOTENTIAL_GRAVITY_F32_BITS)),
                ),
                (
                    "operation",
                    string("one binary32 multiplication per C-order zf word: total_geopotential_zf = zf * consumer_gravity"),
                ),
                (
                    "rounding",
                    string("IEEE-754 binary32 round-to-nearest ties-to-even after multiplication"),
                ),
                ("source_carrier_name", string("zf")),
            ]),
        ),
    ])
}

fn refuse_existing_directory(output: &Path) -> Result<()> {
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace geopotential destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "geopotential destination parent does not exist: {}",
            parent.display()
        )));
    }
    Ok(())
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("geopotential destination needs a UTF-8 directory name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(output.with_file_name(format!(".{name}.partial-{}-{sequence}", std::process::id())))
}

fn verify_exact_directory_inventory(directory: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(directory).map_err(|error| {
        invalid(format!(
            "could not inspect geopotential directory {}: {error}",
            directory.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!(
            "geopotential input is not a real directory: {}",
            directory.display()
        )));
    }
    let expected = BTreeSet::from([
        WSM6_GEOPOTENTIAL_MANIFEST_FILE.to_string(),
        WSM6_GEOPOTENTIAL_PAYLOAD_FILE.to_string(),
        WSM6_GEOPOTENTIAL_RECEIPT_FILE.to_string(),
    ]);
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| invalid("geopotential directory contains a non-UTF-8 name"))?;
        actual.insert(name);
    }
    if actual != expected {
        return Err(invalid(format!(
            "geopotential directory inventory is {actual:?}; exactly {expected:?} is required"
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

fn read_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
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
    if metadata.len() > maximum {
        return Err(invalid(format!(
            "{label} has {} bytes and exceeds the {maximum}-byte limit",
            metadata.len()
        )));
    }
    let length = usize::try_from(metadata.len())
        .map_err(|_| invalid(format!("{label} size does not fit this platform")))?;
    let mut file = File::open(path)?;
    let mut bytes = vec![0u8; length];
    file.read_exact(&mut bytes)?;
    let mut extra = [0u8; 1];
    if file.read(&mut extra)? != 0 {
        return Err(invalid(format!("{label} contains trailing bytes")));
    }
    Ok(bytes)
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

    #[test]
    fn transform_is_source_ordered_and_records_non_identity() {
        let gravity = f32::from_bits(WSM6_GEOPOTENTIAL_GRAVITY_F32_BITS);
        let values = [0.0f32, 1.0, f32::from_bits(0x4477_f3cd), 15_000.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let transformed = transform_zf(&bytes).unwrap();
        assert_eq!(transformed.payload.len(), bytes.len());
        for (index, word) in transformed.payload.chunks_exact(4).enumerate() {
            let actual = f32::from_le_bytes(word.try_into().unwrap());
            assert_eq!(actual.to_bits(), (values[index] * gravity).to_bits());
        }
        assert_eq!(transformed.recovery.word_count, values.len());
        assert_eq!(
            transformed.recovery.recovered_count + transformed.recovery.mismatch_count,
            values.len()
        );
        assert!(transformed.recovery.maximum_ulp_distance <= 1);
    }

    #[test]
    fn transform_refuses_empty_negative_and_nonfinite_zf() {
        assert!(transform_zf(&[]).unwrap_err().to_string().contains("empty"));
        for value in [-1.0f32, f32::NAN, f32::INFINITY] {
            let error = transform_zf(&value.to_le_bytes()).unwrap_err().to_string();
            assert!(error.contains("finite nonnegative"), "{error}");
        }
    }
}
