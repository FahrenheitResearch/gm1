//! Receipt-anchored admission for the independently published child artifacts.
//!
//! Every producer in the child chain publishes one directory holding a
//! canonical JSON manifest, a canonical JSON receipt, and one or more raw
//! little-endian float32 payloads.  Historically the composing gate admitted
//! those directories by a fixed SHA-256 per file, which pinned the whole chain
//! to one run: a parent advanced to a different step, or a producer rerun after
//! a corrected operator, published different bytes and was refused even though
//! nothing about it was wrong.
//!
//! What a consumer actually needs is weaker than "these exact bytes" and
//! stronger than "some bytes":
//!
//!  * the directory holds exactly the files the artifact's ABI defines - no
//!    more, no fewer, none of them a symlink or a directory;
//!  * the manifest and receipt are canonical sorted-key compact JSON, so one
//!    document has exactly one byte spelling and a digest can bind it;
//!  * the receipt declares an admitted schema and a passed status, so a
//!    partial, refused or foreign artifact is not consumed as a finished one;
//!  * the receipt's recorded manifest digest is the manifest that is actually
//!    present, so the published layout and the published receipt describe the
//!    same artifact; and
//!  * every payload's digest and byte count are the ones the receipt records.
//!
//! A directory that satisfies all of that can be consumed without knowing
//! which run produced it.  A directory that fails any of it would hand a
//! consumer carrier offsets, shapes and dispositions taken from one artifact
//! and bytes taken from another - the concrete wrong result this module
//! refuses.
//!
//! Cross-artifact agreement (all five producers naming one parent, one
//! atmosphere, one terrain) is a separate property and is checked by the
//! composing gate in `child_full_state`.

use crate::json::JsonValue;
use crate::sha256;
use crate::{Result, StateError};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) const JSON_MAX_BYTES: u64 = 2 * 1024 * 1024;
pub(crate) const PAYLOAD_MAX_BYTES: u64 = 128 * 1024 * 1024 * 1024;

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

/// Where one recorded identity lives inside a published artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Anchor {
    Receipt(&'static [&'static str]),
    Manifest(&'static [&'static str]),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PayloadSpec {
    pub file: &'static str,
    pub sha_path: &'static [&'static str],
    pub nbytes_path: Option<&'static [&'static str]>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ArtifactSpec {
    pub role: &'static str,
    pub manifest_file: &'static str,
    pub receipt_file: &'static str,
    /// Path inside the receipt to the schema string.
    pub schema_path: &'static [&'static str],
    /// Accepted values of that schema string.
    pub receipt_schemas: &'static [&'static str],
    /// Path inside the receipt to the status string.
    pub status_path: &'static [&'static str],
    /// Accepted values of that status string.
    pub statuses: &'static [&'static str],
    /// Path inside the receipt to the digest of the manifest bytes.
    pub manifest_sha_path: &'static [&'static str],
    /// Path to the artifact's numeric ABI string, when it publishes one.
    pub abi: Option<Anchor>,
    pub payloads: &'static [PayloadSpec],
    /// Path to the parent checkpoint identity, when this producer records one.
    pub parent_checkpoint: Option<Anchor>,
    /// Path to the validated parent plan identity, when it records one.
    pub parent_plan: Option<Anchor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdmittedFile {
    pub name: String,
    pub nbytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub(crate) struct AdmittedArtifact {
    pub role: &'static str,
    #[allow(dead_code)]
    pub directory: PathBuf,
    pub schema: String,
    pub status: String,
    pub abi: Option<String>,
    pub manifest_sha256: String,
    pub receipt_sha256: String,
    pub manifest: JsonValue,
    pub receipt: JsonValue,
    pub files: Vec<AdmittedFile>,
    pub parent_checkpoint_sha256: Option<String>,
    pub parent_plan_sha256: Option<String>,
}

impl AdmittedArtifact {
    pub fn file(&self, name: &str) -> Option<&AdmittedFile> {
        self.files.iter().find(|file| file.name == name)
    }
}

/// Read one canonical JSON document.
pub(crate) fn read_canonical_json(
    path: &Path,
    maximum: u64,
    label: &str,
) -> Result<(Vec<u8>, JsonValue)> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(format!("{label} is not a regular file")));
    }
    if metadata.len() == 0 || metadata.len() > maximum {
        return Err(invalid(format!(
            "{label} is empty or exceeds the {maximum}-byte bound"
        )));
    }
    let bytes = fs::read(path)?;
    if bytes.len() as u64 != metadata.len() {
        return Err(invalid(format!("{label} changed size while being read")));
    }
    let value = crate::json::parse(&bytes)
        .map_err(|error| invalid(format!("{label} is invalid JSON: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid(format!(
            "{label} is not canonical sorted-key compact JSON; two byte spellings of one document would hash differently and no digest could bind it"
        )));
    }
    Ok((bytes, value))
}

pub(crate) fn json_at<'a>(
    value: &'a JsonValue,
    path: &[&str],
    label: &str,
) -> Result<&'a JsonValue> {
    let mut current = value;
    for key in path {
        current = current
            .get(key)
            .map_err(|error| invalid(format!("{label}.{}: {error}", path.join("."))))?;
    }
    Ok(current)
}

pub(crate) fn json_string_at(value: &JsonValue, path: &[&str], label: &str) -> Result<String> {
    json_at(value, path, label)?
        .as_str()
        .map(str::to_owned)
        .map_err(|error| invalid(format!("{label}.{}: {error}", path.join("."))))
}

pub(crate) fn json_u64_at(value: &JsonValue, path: &[&str], label: &str) -> Result<u64> {
    json_at(value, path, label)?
        .as_u64()
        .map_err(|error| invalid(format!("{label}.{}: {error}", path.join("."))))
}

pub(crate) fn json_sha256_at(value: &JsonValue, path: &[&str], label: &str) -> Result<String> {
    let word = json_string_at(value, path, label)?;
    validate_sha256(&word, &format!("{label}.{}", path.join(".")))?;
    Ok(word)
}

pub(crate) fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{label} must be exactly 64 lower-case hexadecimal characters"
        )));
    }
    Ok(())
}

fn anchored(
    anchor: Anchor,
    manifest: &JsonValue,
    receipt: &JsonValue,
    label: &str,
) -> Result<String> {
    match anchor {
        Anchor::Receipt(path) => json_string_at(receipt, path, &format!("{label} receipt")),
        Anchor::Manifest(path) => json_string_at(manifest, path, &format!("{label} manifest")),
    }
}

/// Where the receipt that anchors an artifact lives.
///
/// Most producers publish the receipt inside the artifact directory.  The
/// original 300/50 static-height export predates that convention and is
/// anchored by a separately committed CUDA validation receipt instead; that
/// receipt records the manifest digest and every payload digest and byte
/// count, so it anchors the directory exactly as an in-directory receipt would.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ReceiptSource<'a> {
    InDirectory,
    External(&'a Path),
}

/// Admit one published child artifact against an explicitly located receipt.
pub(crate) fn admit_artifact_with_receipt(
    directory: &Path,
    spec: &ArtifactSpec,
    receipt_source: ReceiptSource<'_>,
) -> Result<AdmittedArtifact> {
    let role = spec.role;
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!(
            "{role} artifact path is not a regular directory"
        )));
    }
    let mut expected = BTreeSet::new();
    expected.insert(spec.manifest_file.to_owned());
    if matches!(receipt_source, ReceiptSource::InDirectory) {
        expected.insert(spec.receipt_file.to_owned());
    }
    for payload in spec.payloads {
        expected.insert(payload.file.to_owned());
    }
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .to_str()
            .ok_or_else(|| invalid(format!("{role} artifact contains a non-UTF-8 name")))?
            .to_owned();
        if !actual.insert(name.clone()) {
            return Err(invalid(format!("{role} artifact inventory is duplicated")));
        }
        if !entry.file_type()?.is_file() {
            return Err(invalid(format!(
                "{role} artifact member {name:?} is not a regular file"
            )));
        }
    }
    if actual != expected {
        return Err(invalid(format!(
            "{role} artifact inventory {actual:?} differs from the {expected:?} its ABI defines; a consumer would mmap a file no manifest describes or miss one it requires"
        )));
    }

    let (manifest_bytes, manifest) = read_canonical_json(
        &directory.join(spec.manifest_file),
        JSON_MAX_BYTES,
        &format!("{role} manifest"),
    )?;
    let receipt_path = match receipt_source {
        ReceiptSource::InDirectory => directory.join(spec.receipt_file),
        ReceiptSource::External(path) => path.to_path_buf(),
    };
    let (receipt_bytes, receipt) =
        read_canonical_json(&receipt_path, JSON_MAX_BYTES, &format!("{role} receipt"))?;
    let schema = json_string_at(&receipt, spec.schema_path, &format!("{role} receipt"))?;
    if !spec.receipt_schemas.contains(&schema.as_str()) {
        return Err(invalid(format!(
            "{role} receipt schema {schema:?} is not one of {:?}; its carrier layout is undescribed and a consumer would read the payload at offsets that belong to another artifact",
            spec.receipt_schemas
        )));
    }
    let status = json_string_at(&receipt, spec.status_path, &format!("{role} receipt"))?;
    if !spec.statuses.contains(&status.as_str()) {
        return Err(invalid(format!(
            "{role} receipt status {status:?} is not one of {:?}; a partial, refused or superseded candidate would be composed into the child as if it had passed",
            spec.statuses
        )));
    }
    let manifest_sha256 =
        json_sha256_at(&receipt, spec.manifest_sha_path, &format!("{role} receipt"))?;
    if sha256::digest_hex(&manifest_bytes) != manifest_sha256 {
        return Err(invalid(format!(
            "{role} manifest bytes are not the manifest its own receipt anchors; the published layout and the published receipt describe different artifacts"
        )));
    }

    let mut files = vec![AdmittedFile {
        name: spec.manifest_file.to_owned(),
        nbytes: manifest_bytes.len() as u64,
        sha256: manifest_sha256.clone(),
    }];
    if matches!(receipt_source, ReceiptSource::InDirectory) {
        files.push(AdmittedFile {
            name: spec.receipt_file.to_owned(),
            nbytes: receipt_bytes.len() as u64,
            sha256: sha256::digest_hex(&receipt_bytes),
        });
    }
    for payload in spec.payloads {
        let expected_sha = json_sha256_at(&receipt, payload.sha_path, &format!("{role} receipt"))?;
        let path = directory.join(payload.file);
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(invalid(format!(
                "{role} payload {} is not a regular file",
                payload.file
            )));
        }
        let nbytes = metadata.len();
        if nbytes == 0 || nbytes > PAYLOAD_MAX_BYTES {
            return Err(invalid(format!(
                "{role} payload {} is empty or exceeds the composition byte bound",
                payload.file
            )));
        }
        if let Some(nbytes_path) = payload.nbytes_path {
            let recorded = json_u64_at(&receipt, nbytes_path, &format!("{role} receipt"))?;
            if recorded != nbytes {
                return Err(invalid(format!(
                    "{role} payload {} is {nbytes} bytes but its receipt records {recorded}; the published carrier layout does not fit the published bytes",
                    payload.file
                )));
            }
        }
        let actual_sha = hash_file_streaming(&path, nbytes)?;
        if actual_sha != expected_sha {
            return Err(invalid(format!(
                "{role} payload {} differs from the digest its own receipt records",
                payload.file
            )));
        }
        files.push(AdmittedFile {
            name: payload.file.to_owned(),
            nbytes,
            sha256: actual_sha,
        });
    }
    files.sort_by(|left, right| left.name.cmp(&right.name));

    let abi = match spec.abi {
        Some(anchor) => Some(anchored(anchor, &manifest, &receipt, role)?),
        None => None,
    };
    let parent_checkpoint_sha256 = match spec.parent_checkpoint {
        Some(anchor) => {
            let value = anchored(anchor, &manifest, &receipt, role)?;
            validate_sha256(&value, &format!("{role} parent checkpoint identity"))?;
            Some(value)
        }
        None => None,
    };
    let parent_plan_sha256 = match spec.parent_plan {
        Some(anchor) => {
            let value = anchored(anchor, &manifest, &receipt, role)?;
            validate_sha256(&value, &format!("{role} parent plan identity"))?;
            Some(value)
        }
        None => None,
    };
    Ok(AdmittedArtifact {
        role,
        directory: directory.to_path_buf(),
        schema,
        status,
        abi,
        manifest_sha256,
        receipt_sha256: sha256::digest_hex(&receipt_bytes),
        manifest,
        receipt,
        files,
        parent_checkpoint_sha256,
        parent_plan_sha256,
    })
}

fn hash_file_streaming(path: &Path, nbytes: u64) -> Result<String> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut digest = sha256::Sha256::default();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut total = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total += read as u64;
        if total > nbytes {
            return Err(invalid(format!(
                "{} grew while it was being hashed",
                path.display()
            )));
        }
        digest.update(&buffer[..read]);
    }
    if total != nbytes {
        return Err(invalid(format!(
            "{} shrank while it was being hashed",
            path.display()
        )));
    }
    Ok(sha256::hex(&digest.finalize()))
}
