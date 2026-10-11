//! Streaming checkpoint persistence for the 49-carrier initialized product.
//!
//! `CM1IPR1` embeds the canonical initialized-product manifest and receipt,
//! then appends the already-described binary32 carrier arena byte-for-byte.
//! Read and restore paths verify every carrier, aggregate payload, source
//! identity, shape, ABI, and artifact hash before exposing a restore handle.

use crate::json::{self, JsonValue};
use crate::product::{
    INITIALIZED_PRODUCT_CARRIER_ABI, INITIALIZED_PRODUCT_CARRIER_NAMES,
    INITIALIZED_PRODUCT_CLAIM_LIMIT, INITIALIZED_PRODUCT_MANIFEST_FILE,
    INITIALIZED_PRODUCT_PAYLOAD_FILE, INITIALIZED_PRODUCT_PUBLICATION_CONTRACT,
    INITIALIZED_PRODUCT_RECEIPT_FILE, INITIALIZED_PRODUCT_RECEIPT_SCHEMA,
    INITIALIZED_PRODUCT_SCHEMA, INITIALIZED_PRODUCT_STAGGER_RULE,
};
use crate::sha256::{self, Sha256};
use crate::terrain::TERRAIN_STATIC_BASE_POLICY;
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const INITIALIZED_PRODUCT_RESTART_MAGIC: [u8; 8] = *b"CM1IPR1\0";
pub const INITIALIZED_PRODUCT_RESTART_SCHEMA: &str = "gpuwm-cm1.initialized-product-restart/v1";
pub const INITIALIZED_PRODUCT_RESTART_RECEIPT_SCHEMA: &str =
    "gpuwm-cm1.initialized-product-restart-receipt/v1";
pub const INITIALIZED_PRODUCT_RESTORE_RECEIPT_SCHEMA: &str =
    "gpuwm-cm1.initialized-product-restore-receipt/v1";
pub const INITIALIZED_PRODUCT_RESTART_CAPABILITY: &str =
    "initialized-product-physical-domain-restart";
pub const INITIALIZED_PRODUCT_RESTART_CLAIM_LIMIT: &str = "exact persistence and restore of all 49 Rust initialized-product physical-domain carriers; no halos, open-boundary history or tendencies, closure state, component persistence, scheduler advancement, uninterrupted/resumed trajectory identity, dynamics, physics, forecast, tornado, or model-validation claim";
pub const INITIALIZED_PRODUCT_RESTART_REPRESENTATION: &str =
    "CM1IPR1 prefix, canonical UTF-8 JSON header, then the exact initialized-carriers.f32le arena";
pub const INITIALIZED_PRODUCT_RESTART_PUBLICATION_CONTRACT: &str = "create a new same-directory temporary file, stream and synchronize the complete envelope, publish through a create-new hard link, then unlink only the owned temporary; never overwrite the target";
pub const INITIALIZED_PRODUCT_RESTORE_PUBLICATION_CONTRACT: &str = "write the exact product payload, canonical embedded manifest, and canonical embedded receipt into a new sibling staging directory; synchronize every file; rename only after the checkpoint and all carrier hashes revalidate; refuse an existing destination";

const PREFIX_BYTES: u64 = 16;
const MAX_HEADER_BYTES: u64 = 1024 * 1024;
const MAX_PAYLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const IO_CHUNK_BYTES: usize = 1024 * 1024;
const SOURCE_PROFILE_ID: &str = "hrrr-cm1-real-state-v1";
const REAL_PRODUCT_DT_S: f64 = 0.5;
const REAL_PRODUCT_DT_F64_BITS: &str = "3fe0000000000000";
/// The first real sources this lane admitted, kept as lineage evidence.
///
/// They are not admission gates: the caller supplies the source artifact and
/// contract identities it independently expects, and every source that is a
/// known kind at the sealed timestep on a completed-step clock is admitted.
pub const FIRST_ADMITTED_REAL_PRODUCT_ARTIFACT_SHA256: &str =
    "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f";
pub const FIRST_ADMITTED_BOUNDARY_ANALYSIS_ARTIFACT_SHA256: &str =
    "28e89cae538cf349c713ea42630dd8cc664a084c0c88f71eac1b3a491f14d189";
pub const FIRST_ADMITTED_F02_ROI_ANALYSIS_ARTIFACT_SHA256: &str =
    "356fe3022ce83909f5c14e7ce9a3639811d9f6dfc5ed9b33a0782cfffb7cd79d";
pub const FIRST_ADMITTED_REAL_PRODUCT_CONTRACT_SHA256: &str =
    "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12";
pub const FIRST_ADMITTED_F02_ROI_CONTRACT_SHA256: &str =
    "a9ccd6f0c6650fd89cd42f16977209f57695464181bcb62beff14e9a9f61d896";
const LEGACY_INITIALIZED_PRODUCT_CLAIM_LIMIT: &str = "byte-exact Rust-owned CM1 r21.1 physical-domain initialization carriers from an independently hashed real checkpoint; includes terrain coordinates/static metrics and audited ibalance=1 moist pressure/density completion; rho0_3d/rr0/rf0/rrf0 retain the explicit replaceable horizontal-broadcast base policy; no halos, open-boundary history or tendencies, closure state, pressure operator, dynamics, physics, trajectory, forecast, tornado, or model-validation claim";
const AUTHORITY_TARBALL_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
const AUTHORITY_FILES: [(&str, &str, &str); 7] = [
    (
        "src/adv_routines.F",
        "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00",
        "5990-6037,9255-9322",
    ),
    (
        "src/base.F",
        "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef",
        "2160-2190",
    ),
    (
        "src/bc.F",
        "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
        "12-172",
    ),
    (
        "src/constants.F",
        "71cb7f92163a247c6f76df9d9d563f9fe4e1470123d0480a8c16762c2e0c8f94",
        "7-18,69-140",
    ),
    (
        "src/init3d.F",
        "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
        "1733-1785,2365-2467",
    ),
    (
        "src/init_terrain.F",
        "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c",
        "175-297",
    ),
    (
        "src/param.F",
        "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
        "6325-6327,7789-7812,8162-8207,8464-8492,8603-8627,8988-9062",
    ),
];

const MANIFEST_KEYS: &[&str] = &[
    "admission",
    "authority",
    "base_policy",
    "carrier_abi",
    "carriers",
    "claim_limit",
    "configuration",
    "crop",
    "dtype",
    "formula_contracts",
    "fornberg",
    "initialization",
    "itemsize",
    "order",
    "payload",
    "schema",
    "source",
    "source_grid",
];
const RECEIPT_KEYS: &[&str] = &[
    "admission",
    "carrier_abi",
    "carrier_count",
    "claim_limit",
    "manifest_nbytes",
    "manifest_sha256",
    "payload_nbytes",
    "payload_sha256",
    "publication_contract",
    "schema",
    "source_artifact_sha256",
    "source_contract_sha256",
];
const CARRIER_KEYS: &[&str] = &[
    "dtype",
    "itemsize",
    "name",
    "nbytes",
    "offset_nbytes",
    "order",
    "role",
    "shape",
    "sha256",
    "source_arrays",
    "statistics",
];
const STATS_KEYS: &[&str] = &[
    "finite_count",
    "maximum_f32_bits",
    "minimum_f32_bits",
    "nonzero_count",
];
const HEADER_KEYS: &[&str] = &[
    "admission",
    "capability",
    "carrier_abi",
    "carrier_count",
    "claim_limit",
    "payload_nbytes",
    "payload_sha256",
    "product_manifest",
    "product_manifest_nbytes",
    "product_manifest_sha256",
    "product_receipt",
    "product_receipt_nbytes",
    "product_receipt_sha256",
    "representation",
    "schema",
    "source_artifact_sha256",
    "source_contract_sha256",
];

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductRestartAdmission {
    Generic,
    RealProduct,
}

impl ProductRestartAdmission {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic-schema",
            Self::RealProduct => "real-product-300x300x75",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "generic-schema" => Ok(Self::Generic),
            "real-product-300x300x75" => Ok(Self::RealProduct),
            other => Err(invalid(format!(
                "initialized-product admission {other:?} is unsupported"
            ))),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductRestartCarrier {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset_nbytes: u64,
    pub nbytes: u64,
    pub sha256: String,
    pub finite_count: usize,
    pub nonzero_count: usize,
    pub minimum_f32_bits: u32,
    pub maximum_f32_bits: u32,
}

#[derive(Clone, Debug)]
struct VerifiedProductExport {
    payload_path: PathBuf,
    manifest: JsonValue,
    manifest_bytes: Vec<u8>,
    manifest_sha256: String,
    receipt: JsonValue,
    receipt_bytes: Vec<u8>,
    receipt_sha256: String,
    payload_sha256: String,
    payload_nbytes: u64,
    source_artifact_sha256: String,
    source_contract_sha256: String,
    admission: ProductRestartAdmission,
    carriers: Vec<ProductRestartCarrier>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductRestartReceipt {
    pub admission: ProductRestartAdmission,
    pub artifact_sha256: String,
    pub artifact_nbytes: u64,
    pub header_sha256: String,
    pub header_nbytes: usize,
    pub payload_sha256: String,
    pub payload_nbytes: u64,
    pub product_manifest_sha256: String,
    pub product_receipt_sha256: String,
    pub source_artifact_sha256: String,
    pub source_contract_sha256: String,
    pub carrier_count: usize,
}

impl ProductRestartReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("admission", string(self.admission.as_str())),
            ("artifact_nbytes", u64_number(self.artifact_nbytes)),
            ("artifact_sha256", string(&self.artifact_sha256)),
            ("capability", string(INITIALIZED_PRODUCT_RESTART_CAPABILITY)),
            ("carrier_abi", string(INITIALIZED_PRODUCT_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            (
                "claim_limit",
                string(INITIALIZED_PRODUCT_RESTART_CLAIM_LIMIT),
            ),
            ("header_nbytes", number(self.header_nbytes)),
            ("header_sha256", string(&self.header_sha256)),
            ("payload_nbytes", u64_number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            (
                "product_manifest_sha256",
                string(&self.product_manifest_sha256),
            ),
            (
                "product_receipt_sha256",
                string(&self.product_receipt_sha256),
            ),
            (
                "publication_contract",
                string(INITIALIZED_PRODUCT_RESTART_PUBLICATION_CONTRACT),
            ),
            ("schema", string(INITIALIZED_PRODUCT_RESTART_RECEIPT_SCHEMA)),
            (
                "source_artifact_sha256",
                string(&self.source_artifact_sha256),
            ),
            (
                "source_contract_sha256",
                string(&self.source_contract_sha256),
            ),
        ]);
        String::from_utf8(value.canonical_bytes())
            .expect("canonical initialized-product restart receipt is UTF-8")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductRestoreReceipt {
    pub restart_artifact_sha256: String,
    pub product_manifest_sha256: String,
    pub product_receipt_sha256: String,
    pub payload_sha256: String,
    pub payload_nbytes: u64,
    pub source_artifact_sha256: String,
    pub source_contract_sha256: String,
    pub carrier_count: usize,
}

impl ProductRestoreReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("carrier_abi", string(INITIALIZED_PRODUCT_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            (
                "claim_limit",
                string(INITIALIZED_PRODUCT_RESTART_CLAIM_LIMIT),
            ),
            ("payload_nbytes", u64_number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            (
                "product_manifest_sha256",
                string(&self.product_manifest_sha256),
            ),
            (
                "product_receipt_sha256",
                string(&self.product_receipt_sha256),
            ),
            (
                "publication_contract",
                string(INITIALIZED_PRODUCT_RESTORE_PUBLICATION_CONTRACT),
            ),
            (
                "restart_artifact_sha256",
                string(&self.restart_artifact_sha256),
            ),
            ("schema", string(INITIALIZED_PRODUCT_RESTORE_RECEIPT_SCHEMA)),
            (
                "source_artifact_sha256",
                string(&self.source_artifact_sha256),
            ),
            (
                "source_contract_sha256",
                string(&self.source_contract_sha256),
            ),
        ]);
        String::from_utf8(value.canonical_bytes())
            .expect("canonical initialized-product restore receipt is UTF-8")
    }
}

#[derive(Clone, Debug)]
pub struct RestoredProductRestart {
    checkpoint_path: PathBuf,
    payload_offset: u64,
    header_bytes: Vec<u8>,
    manifest_bytes: Vec<u8>,
    receipt_bytes: Vec<u8>,
    carriers: Vec<ProductRestartCarrier>,
    receipt: ProductRestartReceipt,
}

/// The source domain one initialized product was built for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProductSourceGrid {
    pub shape_zyx: [usize; 3],
    pub dx_m: f64,
    pub dy_m: f64,
    pub dz_m: f64,
}

impl RestoredProductRestart {
    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }

    /// The extent and spacing the product's own manifest records.
    ///
    /// A consumer that places a child window inside this domain reads the
    /// parent geometry here rather than assuming one: the window's index
    /// mapping divides by exactly this spacing and strides by exactly this
    /// extent.  The values were already structurally validated when the
    /// restart was admitted, so this only decodes them.
    pub fn source_grid(&self) -> Result<ProductSourceGrid> {
        let manifest = parse_canonical_json(&self.manifest_bytes, "product manifest")?;
        let source_grid = member(&manifest, "source_grid", "product manifest")?;
        let shape = usize_array(
            member(source_grid, "shape_zyx", "product manifest.source_grid")?,
            3,
            "product manifest.source_grid.shape_zyx",
        )?;
        let mut spacing = [0.0f64; 3];
        for (index, key) in [
            "target_dx_m_f64_bits",
            "target_dy_m_f64_bits",
            "target_dz_m_f64_bits",
        ]
        .into_iter()
        .enumerate()
        {
            let bits = member_str(source_grid, key, "product manifest.source_grid")?;
            let value = f64::from_bits(u64::from_str_radix(bits, 16).map_err(|_| {
                invalid(format!("product manifest.source_grid.{key} does not parse"))
            })?);
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!(
                    "product manifest.source_grid.{key} decodes to {value:?}; a non-finite or non-positive spacing would divide every index mapping derived from this domain by zero"
                )));
            }
            spacing[index] = value;
        }
        Ok(ProductSourceGrid {
            shape_zyx: [shape[0], shape[1], shape[2]],
            dx_m: spacing[0],
            dy_m: spacing[1],
            dz_m: spacing[2],
        })
    }

    pub fn carriers(&self) -> &[ProductRestartCarrier] {
        &self.carriers
    }

    pub fn carrier(&self, name: &str) -> Option<&ProductRestartCarrier> {
        self.carriers.iter().find(|carrier| carrier.name == name)
    }

    pub fn receipt(&self) -> &ProductRestartReceipt {
        &self.receipt
    }

    /// Revalidate the complete checkpoint and return one exact carrier slice.
    /// This deliberately scans every carrier so a caller cannot consume one
    /// valid field from an otherwise corrupted restart artifact.
    pub fn read_carrier_f32le_exact(&self, name: &str) -> Result<Vec<u8>> {
        self.read_carriers_f32le_exact(&[name])?
            .remove(name)
            .ok_or_else(|| invalid(format!("restart extraction lost carrier {name:?}")))
    }

    /// Revalidate the complete checkpoint once and return several exact slices.
    ///
    /// Every requested field is captured during the same aggregate/hash scan.
    /// This avoids weakening validation or rescanning a large real checkpoint
    /// once per field.
    pub fn read_carriers_f32le_exact(&self, names: &[&str]) -> Result<BTreeMap<String, Vec<u8>>> {
        if names.is_empty() {
            return Err(invalid(
                "restart multi-carrier extraction requires at least one name",
            ));
        }
        let mut requested = BTreeSet::new();
        let mut captured = BTreeMap::new();
        for name in names {
            if !requested.insert(*name) {
                return Err(invalid(format!(
                    "restart multi-carrier extraction contains duplicate name {name:?}"
                )));
            }
            let carrier = self
                .carrier(name)
                .ok_or_else(|| invalid(format!("restart contains no carrier {name:?}")))?;
            let expected_nbytes = usize::try_from(carrier.nbytes).map_err(|_| {
                invalid(format!("carrier {name:?} size does not fit this platform"))
            })?;
            captured.insert((*name).to_string(), Vec::with_capacity(expected_nbytes));
        }
        let (mut reader, file_nbytes) =
            open_regular_file(&self.checkpoint_path, "initialized-product restart")?;
        if file_nbytes != self.receipt.artifact_nbytes {
            return Err(invalid(
                "restart artifact size changed after the verified read",
            ));
        }
        let prefix_and_header_nbytes = usize::try_from(self.payload_offset)
            .map_err(|_| invalid("restart prefix and header do not fit this platform"))?;
        let mut prefix_and_header = vec![0u8; prefix_and_header_nbytes];
        reader.read_exact(&mut prefix_and_header)?;
        let mut expected_prefix = Vec::with_capacity(PREFIX_BYTES as usize);
        expected_prefix.extend_from_slice(&INITIALIZED_PRODUCT_RESTART_MAGIC);
        expected_prefix.extend_from_slice(&(self.header_bytes.len() as u64).to_le_bytes());
        if prefix_and_header[..PREFIX_BYTES as usize] != expected_prefix
            || prefix_and_header[PREFIX_BYTES as usize..] != self.header_bytes
        {
            return Err(invalid(
                "restart prefix or canonical header changed after the verified read",
            ));
        }
        let mut artifact_hasher = Sha256::default();
        artifact_hasher.update(&prefix_and_header);
        scan_payload(
            &mut reader,
            &self.carriers,
            &self.receipt.payload_sha256,
            None,
            Some(&mut artifact_hasher),
            Some(&mut captured),
        )?;
        ensure_reader_eof(&mut reader, "initialized-product restart")?;
        for name in names {
            let carrier = self
                .carrier(name)
                .expect("requested carrier was prevalidated");
            let expected_nbytes =
                usize::try_from(carrier.nbytes).expect("requested carrier size was prevalidated");
            let actual_nbytes = captured
                .get(*name)
                .expect("requested capture entry exists")
                .len();
            if actual_nbytes != expected_nbytes {
                return Err(invalid(format!(
                    "captured carrier {name:?} contains {actual_nbytes} bytes; metadata requires {expected_nbytes}"
                )));
            }
        }
        let artifact_sha256 = sha256::hex(&artifact_hasher.finalize());
        if artifact_sha256 != self.receipt.artifact_sha256 {
            return Err(invalid(
                "restart artifact changed after the verified read; carrier extraction refused",
            ));
        }
        Ok(captured)
    }
}

/// Verify a published 49-carrier arena and atomically persist it as CM1IPR1.
#[allow(clippy::too_many_arguments)]
pub fn write_initialized_product_restart_new(
    product_directory: impl AsRef<Path>,
    output: impl AsRef<Path>,
    expected_product_manifest_sha256: &str,
    expected_product_payload_sha256: &str,
    expected_product_receipt_sha256: &str,
    expected_source_artifact_sha256: &str,
    expected_source_contract_sha256: &str,
) -> Result<ProductRestartReceipt> {
    let product = verify_product_export(
        product_directory.as_ref(),
        expected_product_manifest_sha256,
        expected_product_payload_sha256,
        expected_product_receipt_sha256,
        expected_source_artifact_sha256,
        expected_source_contract_sha256,
    )?;
    let header = restart_header(&product);
    let header_bytes = header.canonical_bytes();
    if header_bytes.len() as u64 > MAX_HEADER_BYTES {
        return Err(invalid(format!(
            "initialized-product restart header exceeds {MAX_HEADER_BYTES} bytes"
        )));
    }
    let mut prefix = Vec::with_capacity(PREFIX_BYTES as usize);
    prefix.extend_from_slice(&INITIALIZED_PRODUCT_RESTART_MAGIC);
    prefix.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());

    let output = output.as_ref();
    refuse_existing_file(output, "initialized-product restart destination")?;
    let temporary = temporary_file_path(output)?;
    let result = (|| -> Result<ProductRestartReceipt> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let mut writer = BufWriter::new(file);
        let mut artifact_hasher = Sha256::default();
        writer.write_all(&prefix)?;
        writer.write_all(&header_bytes)?;
        artifact_hasher.update(&prefix);
        artifact_hasher.update(&header_bytes);

        let mut source = BufReader::new(File::open(&product.payload_path)?);
        let copied = stream_exact(
            &mut source,
            &mut writer,
            product.payload_nbytes,
            Some(&mut artifact_hasher),
        )?;
        if copied.sha256 != product.payload_sha256 {
            return Err(invalid(
                "initialized-product payload changed after preflight verification",
            ));
        }
        ensure_reader_eof(&mut source, "initialized-product payload")?;
        writer.flush()?;
        let file = writer.into_inner().map_err(|error| error.into_error())?;
        file.sync_all()?;
        let artifact_nbytes = PREFIX_BYTES
            .checked_add(header_bytes.len() as u64)
            .and_then(|value| value.checked_add(product.payload_nbytes))
            .ok_or_else(|| invalid("restart artifact byte count overflows"))?;
        if file.metadata()?.len() != artifact_nbytes {
            return Err(invalid(
                "restart artifact size changed after synchronized write",
            ));
        }
        let receipt = ProductRestartReceipt {
            admission: product.admission,
            artifact_sha256: sha256::hex(&artifact_hasher.finalize()),
            artifact_nbytes,
            header_sha256: sha256::digest_hex(&header_bytes),
            header_nbytes: header_bytes.len(),
            payload_sha256: product.payload_sha256.clone(),
            payload_nbytes: product.payload_nbytes,
            product_manifest_sha256: product.manifest_sha256.clone(),
            product_receipt_sha256: product.receipt_sha256.clone(),
            source_artifact_sha256: product.source_artifact_sha256.clone(),
            source_contract_sha256: product.source_contract_sha256.clone(),
            carrier_count: product.carriers.len(),
        };
        fs::hard_link(&temporary, output).map_err(|error| {
            invalid(format!(
                "could not publish new restart target {} without replacement: {error}",
                output.display()
            ))
        })?;
        fs::remove_file(&temporary)?;
        Ok(receipt)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Read and fully scan one CM1IPR1 artifact against independent identities.
#[allow(clippy::too_many_arguments)]
pub fn read_initialized_product_restart(
    input: impl AsRef<Path>,
    expected_restart_artifact_sha256: &str,
    expected_product_manifest_sha256: &str,
    expected_product_payload_sha256: &str,
    expected_product_receipt_sha256: &str,
    expected_source_artifact_sha256: &str,
    expected_source_contract_sha256: &str,
) -> Result<RestoredProductRestart> {
    read_initialized_product_restart_inner(
        input.as_ref(),
        Some(expected_restart_artifact_sha256),
        expected_product_manifest_sha256,
        expected_product_payload_sha256,
        expected_product_receipt_sha256,
        expected_source_artifact_sha256,
        expected_source_contract_sha256,
    )
}

/// The digests one initialized-product restart declares about itself.
struct DeclaredRestartIdentity {
    product_manifest_sha256: String,
    product_payload_sha256: String,
    product_receipt_sha256: String,
    source_artifact_sha256: String,
    source_contract_sha256: String,
}

/// Read only the bounded prefix and header of a restart artifact.
///
/// The header is at most `MAX_HEADER_BYTES`, so this never touches the payload.
fn read_declared_restart_identity(input: &Path) -> Result<DeclaredRestartIdentity> {
    let (mut reader, file_nbytes) = open_regular_file(input, "initialized-product restart")?;
    if file_nbytes < PREFIX_BYTES {
        return Err(invalid(
            "initialized-product restart is truncated before its prefix",
        ));
    }
    let mut prefix = [0u8; PREFIX_BYTES as usize];
    reader.read_exact(&mut prefix)?;
    if prefix[..8] != INITIALIZED_PRODUCT_RESTART_MAGIC {
        return Err(invalid("restart magic/version does not identify CM1IPR1"));
    }
    let header_nbytes = u64::from_le_bytes(prefix[8..16].try_into().expect("fixed prefix"));
    if header_nbytes > MAX_HEADER_BYTES {
        return Err(invalid(format!(
            "restart header length {header_nbytes} exceeds {MAX_HEADER_BYTES}"
        )));
    }
    let header_nbytes_usize = usize::try_from(header_nbytes)
        .map_err(|_| invalid("restart header length does not fit this platform"))?;
    let mut header_bytes = vec![0u8; header_nbytes_usize];
    reader.read_exact(&mut header_bytes)?;
    let parsed = parse_canonical_json(&header_bytes, "restart header")?;
    let read = |key: &str| -> Result<String> {
        let value = member_str(&parsed, key, "restart header")?.to_string();
        validate_sha256(&value, &format!("restart header.{key}"))?;
        Ok(value)
    };
    Ok(DeclaredRestartIdentity {
        product_manifest_sha256: read("product_manifest_sha256")?,
        product_payload_sha256: read("payload_sha256")?,
        product_receipt_sha256: read("product_receipt_sha256")?,
        source_artifact_sha256: read("source_artifact_sha256")?,
        source_contract_sha256: read("source_contract_sha256")?,
    })
}

/// Read one initialized-product restart by the identity it records itself.
///
/// Every digest this reader would otherwise be handed from outside is instead
/// taken from the artifact's own header and then required to equal an
/// independent recomputation of the bytes it describes: the embedded product
/// manifest and receipt are rehashed, the payload is rehashed carrier by
/// carrier against the manifest's per-carrier digests, and the whole-artifact
/// digest is derived from the prefix, header and payload actually read.
/// Nothing is pinned to one historical publication, and nothing is taken on
/// trust: a restart whose header disagrees with its own bytes is still refused.
///
/// What this deliberately does not decide is *which* product the caller wanted.
/// A caller that needs a particular lineage must bind the returned
/// `product_manifest_sha256` to an identity it obtained independently.  The
/// child composition does exactly that, against the parent its own producers
/// name; without that bind, base-state profiles from one run could be composed
/// with perturbation carriers from another.
pub fn read_initialized_product_restart_self_anchored(
    input: impl AsRef<Path>,
) -> Result<RestoredProductRestart> {
    let input = input.as_ref();
    let declared = read_declared_restart_identity(input)?;
    read_initialized_product_restart_inner(
        input,
        None,
        &declared.product_manifest_sha256,
        &declared.product_payload_sha256,
        &declared.product_receipt_sha256,
        &declared.source_artifact_sha256,
        &declared.source_contract_sha256,
    )
}

#[allow(clippy::too_many_arguments)]
fn read_initialized_product_restart_inner(
    input: &Path,
    expected_restart_artifact_sha256: Option<&str>,
    expected_product_manifest_sha256: &str,
    expected_product_payload_sha256: &str,
    expected_product_receipt_sha256: &str,
    expected_source_artifact_sha256: &str,
    expected_source_contract_sha256: &str,
) -> Result<RestoredProductRestart> {
    for (value, label) in [
        (
            expected_product_manifest_sha256,
            "expected product manifest SHA-256",
        ),
        (
            expected_product_payload_sha256,
            "expected product payload SHA-256",
        ),
        (
            expected_product_receipt_sha256,
            "expected product receipt SHA-256",
        ),
        (
            expected_source_artifact_sha256,
            "expected source artifact SHA-256",
        ),
        (
            expected_source_contract_sha256,
            "expected source contract SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    if let Some(value) = expected_restart_artifact_sha256 {
        validate_sha256(value, "expected restart artifact SHA-256")?;
    }
    let (mut reader, file_nbytes) = open_regular_file(input, "initialized-product restart")?;
    if file_nbytes < PREFIX_BYTES {
        return Err(invalid(
            "initialized-product restart is truncated before its prefix",
        ));
    }
    if file_nbytes
        > PREFIX_BYTES
            .checked_add(MAX_HEADER_BYTES)
            .and_then(|value| value.checked_add(MAX_PAYLOAD_BYTES))
            .expect("restart maximum fits u64")
    {
        return Err(invalid(
            "initialized-product restart exceeds the bounded envelope size",
        ));
    }
    let mut prefix = [0u8; PREFIX_BYTES as usize];
    reader.read_exact(&mut prefix)?;
    if prefix[..8] != INITIALIZED_PRODUCT_RESTART_MAGIC {
        return Err(invalid("restart magic/version does not identify CM1IPR1"));
    }
    let header_nbytes = u64::from_le_bytes(prefix[8..16].try_into().expect("fixed prefix"));
    if header_nbytes > MAX_HEADER_BYTES {
        return Err(invalid(format!(
            "restart header length {header_nbytes} exceeds {MAX_HEADER_BYTES}"
        )));
    }
    let header_nbytes_usize = usize::try_from(header_nbytes)
        .map_err(|_| invalid("restart header length does not fit this platform"))?;
    let mut header_bytes = vec![0u8; header_nbytes_usize];
    reader.read_exact(&mut header_bytes)?;
    let parsed = parse_canonical_json(&header_bytes, "restart header")?;
    exact_keys(&parsed, HEADER_KEYS, "restart header")?;
    fixed_string(
        &parsed,
        "schema",
        INITIALIZED_PRODUCT_RESTART_SCHEMA,
        "restart header",
    )?;
    fixed_string(
        &parsed,
        "capability",
        INITIALIZED_PRODUCT_RESTART_CAPABILITY,
        "restart header",
    )?;
    fixed_string(
        &parsed,
        "carrier_abi",
        INITIALIZED_PRODUCT_CARRIER_ABI,
        "restart header",
    )?;
    fixed_string(
        &parsed,
        "claim_limit",
        INITIALIZED_PRODUCT_RESTART_CLAIM_LIMIT,
        "restart header",
    )?;
    fixed_string(
        &parsed,
        "representation",
        INITIALIZED_PRODUCT_RESTART_REPRESENTATION,
        "restart header",
    )?;
    let manifest = parsed
        .get("product_manifest")
        .map_err(|error| invalid(format!("restart header.product_manifest: {error}")))?
        .clone();
    let receipt = parsed
        .get("product_receipt")
        .map_err(|error| invalid(format!("restart header.product_receipt: {error}")))?
        .clone();
    let manifest_bytes = manifest.canonical_bytes();
    let receipt_bytes = receipt.canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    fixed_string(
        &parsed,
        "product_manifest_sha256",
        expected_product_manifest_sha256,
        "restart header",
    )?;
    fixed_string(
        &parsed,
        "product_receipt_sha256",
        expected_product_receipt_sha256,
        "restart header",
    )?;
    if manifest_sha256 != expected_product_manifest_sha256
        || receipt_sha256 != expected_product_receipt_sha256
    {
        return Err(invalid(
            "embedded product manifest or receipt bytes differ from their independent hash",
        ));
    }
    fixed_u64(
        &parsed,
        "product_manifest_nbytes",
        manifest_bytes.len() as u64,
        "restart header",
    )?;
    fixed_u64(
        &parsed,
        "product_receipt_nbytes",
        receipt_bytes.len() as u64,
        "restart header",
    )?;
    let product = validate_product_values(
        input.parent().unwrap_or_else(|| Path::new(".")),
        Path::new("embedded-payload"),
        manifest,
        manifest_bytes.clone(),
        manifest_sha256,
        receipt,
        receipt_bytes.clone(),
        receipt_sha256,
        expected_product_payload_sha256,
        expected_source_artifact_sha256,
        expected_source_contract_sha256,
    )?;
    fixed_string(
        &parsed,
        "admission",
        product.admission.as_str(),
        "restart header",
    )?;
    fixed_u64(
        &parsed,
        "carrier_count",
        product.carriers.len() as u64,
        "restart header",
    )?;
    fixed_u64(
        &parsed,
        "payload_nbytes",
        product.payload_nbytes,
        "restart header",
    )?;
    fixed_string(
        &parsed,
        "payload_sha256",
        &product.payload_sha256,
        "restart header",
    )?;
    fixed_string(
        &parsed,
        "source_artifact_sha256",
        expected_source_artifact_sha256,
        "restart header",
    )?;
    fixed_string(
        &parsed,
        "source_contract_sha256",
        expected_source_contract_sha256,
        "restart header",
    )?;
    let payload_offset = PREFIX_BYTES
        .checked_add(header_nbytes)
        .ok_or_else(|| invalid("restart payload offset overflows"))?;
    let expected_file_nbytes = payload_offset
        .checked_add(product.payload_nbytes)
        .ok_or_else(|| invalid("restart artifact size overflows"))?;
    if file_nbytes != expected_file_nbytes {
        return Err(invalid(format!(
            "restart artifact contains {file_nbytes} bytes; header and payload require {expected_file_nbytes}"
        )));
    }
    let mut artifact_hasher = Sha256::default();
    artifact_hasher.update(&prefix);
    artifact_hasher.update(&header_bytes);
    scan_payload(
        &mut reader,
        &product.carriers,
        &product.payload_sha256,
        None,
        Some(&mut artifact_hasher),
        None,
    )?;
    ensure_reader_eof(&mut reader, "initialized-product restart")?;
    let artifact_sha256 = sha256::hex(&artifact_hasher.finalize());
    if let Some(expected) = expected_restart_artifact_sha256 {
        if artifact_sha256 != expected {
            return Err(invalid(format!(
                "restart artifact SHA-256 is {artifact_sha256}; independently expected {expected}"
            )));
        }
    }
    let receipt_value = ProductRestartReceipt {
        admission: product.admission,
        artifact_sha256,
        artifact_nbytes: file_nbytes,
        header_sha256: sha256::digest_hex(&header_bytes),
        header_nbytes: header_bytes.len(),
        payload_sha256: product.payload_sha256,
        payload_nbytes: product.payload_nbytes,
        product_manifest_sha256: product.manifest_sha256,
        product_receipt_sha256: product.receipt_sha256,
        source_artifact_sha256: product.source_artifact_sha256,
        source_contract_sha256: product.source_contract_sha256,
        carrier_count: product.carriers.len(),
    };
    Ok(RestoredProductRestart {
        checkpoint_path: input.to_path_buf(),
        payload_offset,
        header_bytes,
        manifest_bytes,
        receipt_bytes,
        carriers: product.carriers,
        receipt: receipt_value,
    })
}

/// Restore the exact initialized-product directory from a previously verified
/// checkpoint.  The checkpoint is fully rehashed during the copy so a
/// post-read mutation cannot cross the restore boundary.
pub fn restore_initialized_product_restart_new(
    restored: &RestoredProductRestart,
    output_directory: impl AsRef<Path>,
) -> Result<ProductRestoreReceipt> {
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace initialized-product restore destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "restore parent directory does not exist: {}",
            parent.display()
        )));
    }
    let staging = temporary_directory_path(output)?;
    fs::create_dir(&staging)?;
    let result = (|| -> Result<ProductRestoreReceipt> {
        let (mut reader, file_nbytes) =
            open_regular_file(&restored.checkpoint_path, "initialized-product restart")?;
        if file_nbytes != restored.receipt.artifact_nbytes {
            return Err(invalid(
                "restart artifact size changed after the verified read",
            ));
        }
        let prefix_and_header_nbytes = usize::try_from(restored.payload_offset)
            .map_err(|_| invalid("restart prefix and header do not fit this platform"))?;
        let mut prefix_and_header = vec![0u8; prefix_and_header_nbytes];
        reader.read_exact(&mut prefix_and_header)?;
        let header_start = PREFIX_BYTES as usize;
        if prefix_and_header[..8] != INITIALIZED_PRODUCT_RESTART_MAGIC
            || prefix_and_header[header_start..] != restored.header_bytes
        {
            return Err(invalid(
                "restart prefix or canonical header changed after the verified read",
            ));
        }
        let mut artifact_hasher = Sha256::default();
        artifact_hasher.update(&prefix_and_header);
        let payload_path = staging.join(INITIALIZED_PRODUCT_PAYLOAD_FILE);
        let payload_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&payload_path)?;
        let mut writer = BufWriter::new(payload_file);
        scan_payload(
            &mut reader,
            &restored.carriers,
            &restored.receipt.payload_sha256,
            Some(&mut writer),
            Some(&mut artifact_hasher),
            None,
        )?;
        ensure_reader_eof(&mut reader, "initialized-product restart")?;
        writer.flush()?;
        let payload_file = writer.into_inner().map_err(|error| error.into_error())?;
        payload_file.sync_all()?;
        if payload_file.metadata()?.len() != restored.receipt.payload_nbytes {
            return Err(invalid("restored payload byte count changed after sync"));
        }
        drop(payload_file);
        let artifact_sha256 = sha256::hex(&artifact_hasher.finalize());
        if artifact_sha256 != restored.receipt.artifact_sha256 {
            return Err(invalid(
                "restart artifact changed after the verified read; restore refused",
            ));
        }
        write_new_synced(
            &staging.join(INITIALIZED_PRODUCT_MANIFEST_FILE),
            &restored.manifest_bytes,
        )?;
        write_new_synced(
            &staging.join(INITIALIZED_PRODUCT_RECEIPT_FILE),
            &restored.receipt_bytes,
        )?;
        if let Err(error) = fs::rename(&staging, output) {
            return Err(error.into());
        }
        Ok(ProductRestoreReceipt {
            restart_artifact_sha256: artifact_sha256,
            product_manifest_sha256: restored.receipt.product_manifest_sha256.clone(),
            product_receipt_sha256: restored.receipt.product_receipt_sha256.clone(),
            payload_sha256: restored.receipt.payload_sha256.clone(),
            payload_nbytes: restored.receipt.payload_nbytes,
            source_artifact_sha256: restored.receipt.source_artifact_sha256.clone(),
            source_contract_sha256: restored.receipt.source_contract_sha256.clone(),
            carrier_count: restored.carriers.len(),
        })
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn verify_product_export(
    directory: &Path,
    expected_manifest_sha256: &str,
    expected_payload_sha256: &str,
    expected_receipt_sha256: &str,
    expected_source_artifact_sha256: &str,
    expected_source_contract_sha256: &str,
) -> Result<VerifiedProductExport> {
    for (value, label) in [
        (
            expected_manifest_sha256,
            "expected product manifest SHA-256",
        ),
        (expected_payload_sha256, "expected product payload SHA-256"),
        (expected_receipt_sha256, "expected product receipt SHA-256"),
        (
            expected_source_artifact_sha256,
            "expected source artifact SHA-256",
        ),
        (
            expected_source_contract_sha256,
            "expected source contract SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    let metadata = fs::symlink_metadata(directory).map_err(|error| {
        invalid(format!(
            "could not inspect initialized-product directory {}: {error}",
            directory.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!(
            "initialized-product input is not a real directory: {}",
            directory.display()
        )));
    }
    let required = BTreeSet::from([
        INITIALIZED_PRODUCT_MANIFEST_FILE.to_string(),
        INITIALIZED_PRODUCT_PAYLOAD_FILE.to_string(),
        INITIALIZED_PRODUCT_RECEIPT_FILE.to_string(),
    ]);
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| invalid("initialized-product directory contains a non-UTF-8 name"))?;
        actual.insert(name);
    }
    if actual != required {
        return Err(invalid(format!(
            "initialized-product directory inventory is {actual:?}; exactly {required:?} is required"
        )));
    }
    let manifest_path = directory.join(INITIALIZED_PRODUCT_MANIFEST_FILE);
    let payload_path = directory.join(INITIALIZED_PRODUCT_PAYLOAD_FILE);
    let receipt_path = directory.join(INITIALIZED_PRODUCT_RECEIPT_FILE);
    let manifest_bytes =
        read_regular_bounded(&manifest_path, MAX_HEADER_BYTES, "product manifest")?;
    let receipt_bytes = read_regular_bounded(&receipt_path, MAX_HEADER_BYTES, "product receipt")?;
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    if manifest_sha256 != expected_manifest_sha256 {
        return Err(invalid(format!(
            "product manifest SHA-256 is {manifest_sha256}; independently expected {expected_manifest_sha256}"
        )));
    }
    if receipt_sha256 != expected_receipt_sha256 {
        return Err(invalid(format!(
            "product receipt SHA-256 is {receipt_sha256}; independently expected {expected_receipt_sha256}"
        )));
    }
    let manifest = parse_canonical_json(&manifest_bytes, "product manifest")?;
    let receipt = parse_canonical_json(&receipt_bytes, "product receipt")?;
    let product = validate_product_values(
        directory,
        &payload_path,
        manifest,
        manifest_bytes,
        manifest_sha256,
        receipt,
        receipt_bytes,
        receipt_sha256,
        expected_payload_sha256,
        expected_source_artifact_sha256,
        expected_source_contract_sha256,
    )?;
    let (mut payload, nbytes) = open_regular_file(&payload_path, "initialized-product payload")?;
    if nbytes != product.payload_nbytes {
        return Err(invalid(format!(
            "initialized-product payload has {nbytes} bytes; manifest requires {}",
            product.payload_nbytes
        )));
    }
    scan_payload(
        &mut payload,
        &product.carriers,
        &product.payload_sha256,
        None,
        None,
        None,
    )?;
    ensure_reader_eof(&mut payload, "initialized-product payload")?;
    Ok(product)
}

#[allow(clippy::too_many_arguments)]
fn validate_product_values(
    _directory: &Path,
    payload_path: &Path,
    manifest: JsonValue,
    manifest_bytes: Vec<u8>,
    manifest_sha256: String,
    receipt: JsonValue,
    receipt_bytes: Vec<u8>,
    receipt_sha256: String,
    expected_payload_sha256: &str,
    expected_source_artifact_sha256: &str,
    expected_source_contract_sha256: &str,
) -> Result<VerifiedProductExport> {
    exact_keys(&manifest, MANIFEST_KEYS, "product manifest")?;
    exact_keys(&receipt, RECEIPT_KEYS, "product receipt")?;
    fixed_string(
        &manifest,
        "schema",
        INITIALIZED_PRODUCT_SCHEMA,
        "product manifest",
    )?;
    fixed_string(
        &manifest,
        "carrier_abi",
        INITIALIZED_PRODUCT_CARRIER_ABI,
        "product manifest",
    )?;
    let product_claim = member_str(&manifest, "claim_limit", "product manifest")?;
    if product_claim != INITIALIZED_PRODUCT_CLAIM_LIMIT
        && product_claim != LEGACY_INITIALIZED_PRODUCT_CLAIM_LIMIT
    {
        return Err(invalid("product manifest claim boundary is unsupported"));
    }
    fixed_string(
        &manifest,
        "base_policy",
        TERRAIN_STATIC_BASE_POLICY,
        "product manifest",
    )?;
    fixed_string(&manifest, "dtype", "<f4", "product manifest")?;
    fixed_string(&manifest, "order", "C", "product manifest")?;
    fixed_u64(&manifest, "itemsize", 4, "product manifest")?;
    let admission_text = member_str(&manifest, "admission", "product manifest")?;
    let admission = ProductRestartAdmission::parse(admission_text)?;
    validate_authority(member(&manifest, "authority", "product manifest")?)?;
    validate_configuration(member(&manifest, "configuration", "product manifest")?)?;
    validate_formula_contracts(member(&manifest, "formula_contracts", "product manifest")?)?;
    validate_fornberg(member(&manifest, "fornberg", "product manifest")?)?;
    validate_initialization(member(&manifest, "initialization", "product manifest")?)?;

    let source_grid = member(&manifest, "source_grid", "product manifest")?;
    exact_keys(
        source_grid,
        &[
            "shape_zyx",
            "target_dx_m_f64_bits",
            "target_dy_m_f64_bits",
            "target_dz_m_f64_bits",
        ],
        "product manifest.source_grid",
    )?;
    let source_shape = usize_array(
        member(source_grid, "shape_zyx", "product manifest.source_grid")?,
        3,
        "product manifest.source_grid.shape_zyx",
    )?;
    if source_shape.contains(&0) {
        return Err(invalid("product source grid extents must be positive"));
    }
    for key in [
        "target_dx_m_f64_bits",
        "target_dy_m_f64_bits",
        "target_dz_m_f64_bits",
    ] {
        validate_hex_bits(
            member_str(source_grid, key, "product manifest.source_grid")?,
            16,
            &format!("product manifest.source_grid.{key}"),
        )?;
    }
    if admission == ProductRestartAdmission::RealProduct
        && (source_shape != [75, 300, 300]
            || member_str(source_grid, "target_dx_m_f64_bits", "source_grid")?
                != "4059000000000000"
            || member_str(source_grid, "target_dy_m_f64_bits", "source_grid")?
                != "4059000000000000"
            || member_str(source_grid, "target_dz_m_f64_bits", "source_grid")?
                != "4069000000000000")
    {
        // This is the real lane's own admitted parent domain, not a run
        // identity: the CUDA carrier arena, the 300x300 checkpoint crop and the
        // 200 m base-state profile support are all compiled against it, so a
        // different extent or spacing would be uploaded into allocations sized
        // for this one and interpolated against heights the source never has.
        return Err(invalid(
            "real-product source grid is not the admitted 300x300x75, 100 m x 100 m x 200 m parent domain; the resident carrier arena, checkpoint crop and base-state profile support are all sized for that domain, so another one would be read past its allocation",
        ));
    }

    let crop = member(&manifest, "crop", "product manifest")?;
    exact_keys(
        crop,
        &["count_yx", "origin_yx", "stagger_rule"],
        "product manifest.crop",
    )?;
    fixed_string(
        crop,
        "stagger_rule",
        INITIALIZED_PRODUCT_STAGGER_RULE,
        "product manifest.crop",
    )?;
    let count = usize_array(
        member(crop, "count_yx", "product manifest.crop")?,
        2,
        "product manifest.crop.count_yx",
    )?;
    let origin = usize_array(
        member(crop, "origin_yx", "product manifest.crop")?,
        2,
        "product manifest.crop.origin_yx",
    )?;
    if count.contains(&0)
        || origin[0].checked_add(count[0]).is_none()
        || origin[1].checked_add(count[1]).is_none()
        || origin[0] + count[0] > source_shape[1]
        || origin[1] + count[1] > source_shape[2]
    {
        return Err(invalid(
            "product crop is empty, overflows, or exceeds the source physical grid",
        ));
    }

    let source = member(&manifest, "source", "product manifest")?;
    let source_kind = match member(source, "kind", "product manifest.source") {
        Ok(value) => {
            exact_keys(
                source,
                &[
                    "artifact_nbytes",
                    "artifact_sha256",
                    "contract_sha256",
                    "dt_f64_bits",
                    "kind",
                    "model_time_f64_bits",
                    "profile_id",
                    "step",
                ],
                "product manifest.source",
            )?;
            value
                .as_str()
                .map_err(|error| invalid(format!("product manifest.source.kind: {error}")))?
        }
        Err(_) => {
            exact_keys(
                source,
                &[
                    "artifact_nbytes",
                    "artifact_sha256",
                    "contract_sha256",
                    "dt_f64_bits",
                    "model_time_f64_bits",
                    "profile_id",
                    "step",
                ],
                "product manifest.source",
            )?;
            "initial-checkpoint"
        }
    };
    let source_artifact_sha256 =
        member_str(source, "artifact_sha256", "product manifest.source")?.to_string();
    let source_contract_sha256 =
        member_str(source, "contract_sha256", "product manifest.source")?.to_string();
    validate_sha256(&source_artifact_sha256, "product source artifact SHA-256")?;
    validate_sha256(&source_contract_sha256, "product source contract SHA-256")?;
    if source_artifact_sha256 != expected_source_artifact_sha256
        || source_contract_sha256 != expected_source_contract_sha256
    {
        return Err(invalid(
            "product source artifact or contract identity differs from the independent admission",
        ));
    }
    if member_u64(source, "artifact_nbytes", "product manifest.source")? == 0 {
        return Err(invalid(
            "product source artifact byte count must be positive",
        ));
    }
    validate_hex_bits(
        member_str(source, "dt_f64_bits", "product manifest.source")?,
        16,
        "product manifest.source.dt_f64_bits",
    )?;
    validate_hex_bits(
        member_str(source, "model_time_f64_bits", "product manifest.source")?,
        16,
        "product manifest.source.model_time_f64_bits",
    )?;
    let profile_id = member_str(source, "profile_id", "product manifest.source")?;
    if profile_id.is_empty() {
        return Err(invalid("product source profile_id is empty"));
    }
    let _ = member_u64(source, "step", "product manifest.source")?;
    if admission == ProductRestartAdmission::RealProduct {
        admit_real_product_source(
            source_kind,
            profile_id,
            member_str(source, "dt_f64_bits", "product manifest.source")?,
            member_u64(source, "step", "product manifest.source")?,
            member_str(source, "model_time_f64_bits", "product manifest.source")?,
        )?;
    }

    let payload = member(&manifest, "payload", "product manifest")?;
    exact_keys(
        payload,
        &["file", "nbytes", "sha256"],
        "product manifest.payload",
    )?;
    fixed_string(
        payload,
        "file",
        INITIALIZED_PRODUCT_PAYLOAD_FILE,
        "product manifest.payload",
    )?;
    let payload_nbytes = member_u64(payload, "nbytes", "product manifest.payload")?;
    if payload_nbytes == 0 || payload_nbytes > MAX_PAYLOAD_BYTES || payload_nbytes % 4 != 0 {
        return Err(invalid(
            "product payload byte count is empty, unaligned, or exceeds the restart limit",
        ));
    }
    let payload_sha256 = member_str(payload, "sha256", "product manifest.payload")?.to_string();
    validate_sha256(&payload_sha256, "product payload SHA-256")?;
    if payload_sha256 != expected_payload_sha256 {
        return Err(invalid(format!(
            "product payload SHA-256 is {payload_sha256}; independently expected {expected_payload_sha256}"
        )));
    }

    let carrier_values = member(&manifest, "carriers", "product manifest")?
        .as_array()
        .map_err(|error| invalid(format!("product manifest.carriers: {error}")))?;
    if carrier_values.len() != INITIALIZED_PRODUCT_CARRIER_NAMES.len() {
        return Err(invalid(format!(
            "product manifest contains {} carriers; ABI requires {}",
            carrier_values.len(),
            INITIALIZED_PRODUCT_CARRIER_NAMES.len()
        )));
    }
    let mut carriers = Vec::with_capacity(carrier_values.len());
    let mut next_offset = 0u64;
    for (index, (value, expected_name)) in carrier_values
        .iter()
        .zip(INITIALIZED_PRODUCT_CARRIER_NAMES)
        .enumerate()
    {
        let label = format!("product manifest.carriers[{index}]");
        exact_keys(value, CARRIER_KEYS, &label)?;
        fixed_string(value, "dtype", "<f4", &label)?;
        fixed_string(value, "order", "C", &label)?;
        fixed_u64(value, "itemsize", 4, &label)?;
        fixed_string(value, "name", expected_name, &label)?;
        let role = member_str(value, "role", &label)?;
        if role.is_empty() {
            return Err(invalid(format!("{label}.role is empty")));
        }
        let source_arrays = member(value, "source_arrays", &label)?
            .as_array()
            .map_err(|error| invalid(format!("{label}.source_arrays: {error}")))?;
        if source_arrays.is_empty()
            || source_arrays
                .iter()
                .any(|item| item.as_str().map_or(true, str::is_empty))
        {
            return Err(invalid(format!(
                "{label}.source_arrays must contain non-empty strings"
            )));
        }
        let shape = usize_array_any(member(value, "shape", &label)?, &format!("{label}.shape"))?;
        let expected_shape =
            expected_carrier_shape(expected_name, source_shape[0], count[0], count[1]);
        if shape != expected_shape {
            return Err(invalid(format!(
                "{label}.shape is {shape:?}; ABI requires {expected_shape:?}"
            )));
        }
        let words = shape_product(&shape, expected_name)?;
        let nbytes = u64::try_from(
            words
                .checked_mul(4)
                .ok_or_else(|| invalid(format!("carrier {expected_name} byte count overflows")))?,
        )
        .map_err(|_| invalid(format!("carrier {expected_name} byte count exceeds u64")))?;
        fixed_u64(value, "nbytes", nbytes, &label)?;
        fixed_u64(value, "offset_nbytes", next_offset, &label)?;
        let digest = member_str(value, "sha256", &label)?.to_string();
        validate_sha256(&digest, &format!("{label}.sha256"))?;
        let stats = member(value, "statistics", &label)?;
        exact_keys(stats, STATS_KEYS, &format!("{label}.statistics"))?;
        let finite_count = usize_from_u64(
            member_u64(stats, "finite_count", &format!("{label}.statistics"))?,
            &format!("{label}.statistics.finite_count"),
        )?;
        let nonzero_count = usize_from_u64(
            member_u64(stats, "nonzero_count", &format!("{label}.statistics"))?,
            &format!("{label}.statistics.nonzero_count"),
        )?;
        if finite_count != words || nonzero_count > words {
            return Err(invalid(format!(
                "{label}.statistics counts are inconsistent with {words} binary32 words"
            )));
        }
        let minimum_f32_bits = parse_hex_u32(
            member_str(stats, "minimum_f32_bits", &format!("{label}.statistics"))?,
            &format!("{label}.statistics.minimum_f32_bits"),
        )?;
        let maximum_f32_bits = parse_hex_u32(
            member_str(stats, "maximum_f32_bits", &format!("{label}.statistics"))?,
            &format!("{label}.statistics.maximum_f32_bits"),
        )?;
        if !f32::from_bits(minimum_f32_bits).is_finite()
            || !f32::from_bits(maximum_f32_bits).is_finite()
        {
            return Err(invalid(format!(
                "{label}.statistics minimum or maximum is non-finite"
            )));
        }
        carriers.push(ProductRestartCarrier {
            name: expected_name.to_string(),
            shape,
            offset_nbytes: next_offset,
            nbytes,
            sha256: digest,
            finite_count,
            nonzero_count,
            minimum_f32_bits,
            maximum_f32_bits,
        });
        next_offset = next_offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("product carrier offsets overflow"))?;
    }
    if next_offset != payload_nbytes {
        return Err(invalid(format!(
            "product carriers cover {next_offset} bytes; payload declares {payload_nbytes}"
        )));
    }
    validate_duplicate_groups(&carriers)?;

    fixed_string(
        &receipt,
        "schema",
        INITIALIZED_PRODUCT_RECEIPT_SCHEMA,
        "product receipt",
    )?;
    fixed_string(&receipt, "admission", admission.as_str(), "product receipt")?;
    fixed_string(
        &receipt,
        "carrier_abi",
        INITIALIZED_PRODUCT_CARRIER_ABI,
        "product receipt",
    )?;
    if member_str(&receipt, "claim_limit", "product receipt")? != product_claim {
        return Err(invalid(
            "product receipt claim boundary differs from the manifest",
        ));
    }
    fixed_string(
        &receipt,
        "publication_contract",
        INITIALIZED_PRODUCT_PUBLICATION_CONTRACT,
        "product receipt",
    )?;
    fixed_u64(
        &receipt,
        "carrier_count",
        carriers.len() as u64,
        "product receipt",
    )?;
    fixed_u64(
        &receipt,
        "manifest_nbytes",
        manifest_bytes.len() as u64,
        "product receipt",
    )?;
    fixed_string(
        &receipt,
        "manifest_sha256",
        &manifest_sha256,
        "product receipt",
    )?;
    fixed_u64(
        &receipt,
        "payload_nbytes",
        payload_nbytes,
        "product receipt",
    )?;
    fixed_string(
        &receipt,
        "payload_sha256",
        &payload_sha256,
        "product receipt",
    )?;
    fixed_string(
        &receipt,
        "source_artifact_sha256",
        &source_artifact_sha256,
        "product receipt",
    )?;
    fixed_string(
        &receipt,
        "source_contract_sha256",
        &source_contract_sha256,
        "product receipt",
    )?;

    Ok(VerifiedProductExport {
        payload_path: payload_path.to_path_buf(),
        manifest,
        manifest_bytes,
        manifest_sha256,
        receipt,
        receipt_bytes,
        receipt_sha256,
        payload_sha256,
        payload_nbytes,
        source_artifact_sha256,
        source_contract_sha256,
        admission,
        carriers,
    })
}

fn validate_authority(value: &JsonValue) -> Result<()> {
    exact_keys(
        value,
        &["release", "source_files", "tarball_sha256"],
        "product manifest.authority",
    )?;
    fixed_string(value, "release", "CM1 21.1", "product manifest.authority")?;
    fixed_string(
        value,
        "tarball_sha256",
        AUTHORITY_TARBALL_SHA256,
        "product manifest.authority",
    )?;
    let files = member(value, "source_files", "product manifest.authority")?
        .as_array()
        .map_err(|error| invalid(format!("product manifest.authority.source_files: {error}")))?;
    if files.len() != AUTHORITY_FILES.len() {
        return Err(invalid(
            "product authority source-file inventory is incomplete or extended",
        ));
    }
    for (index, (file, (path, digest, anchors))) in files.iter().zip(AUTHORITY_FILES).enumerate() {
        let label = format!("product manifest.authority.source_files[{index}]");
        exact_keys(file, &["anchors", "path", "sha256"], &label)?;
        fixed_string(file, "path", path, &label)?;
        fixed_string(file, "sha256", digest, &label)?;
        fixed_string(file, "anchors", anchors, &label)?;
    }
    Ok(())
}

fn validate_configuration(value: &JsonValue) -> Result<()> {
    exact_keys(
        value,
        &[
            "hadvordrs",
            "hadvordrv",
            "ibalance",
            "iice",
            "lateral_bc_west_east_south_north",
            "psolver",
            "setppi",
        ],
        "product manifest.configuration",
    )?;
    for (key, expected) in [
        ("hadvordrs", 5),
        ("hadvordrv", 5),
        ("ibalance", 1),
        ("iice", 1),
        ("psolver", 3),
    ] {
        fixed_u64(value, key, expected, "product manifest.configuration")?;
    }
    fixed_bool(value, "setppi", true, "product manifest.configuration")?;
    let bcs = usize_array(
        member(
            value,
            "lateral_bc_west_east_south_north",
            "product manifest.configuration",
        )?,
        4,
        "product manifest.configuration.lateral_bc_west_east_south_north",
    )?;
    if bcs != [2, 2, 2, 2] {
        return Err(invalid(
            "product configuration is not the admitted four-open-boundary tuple",
        ));
    }
    Ok(())
}

fn validate_formula_contracts(value: &JsonValue) -> Result<()> {
    let expected = object(vec![
        ("base_reciprocals", string("base.F:2160-2190")),
        (
            "horizontal_gradients",
            string("adv_routines.F:9255-9322 with flx6 at 6026-6037 and bc2d open-edge halos"),
        ),
        (
            "moist_initialization",
            string("init3d.F:1733-1785,2365-2467 with param.F:8988-9052 Fornberg coefficients"),
        ),
        ("terrain_coordinates", string("init_terrain.F:175-297")),
        ("vertical_metrics", string("param.F:8464-8492,8603-8627")),
    ]);
    if value != &expected {
        return Err(invalid(
            "product formula contracts differ from the source-audited v1 tuple",
        ));
    }
    Ok(())
}

fn validate_fornberg(value: &JsonValue) -> Result<()> {
    let weights = JsonValue::Array(
        ["3ff00000", "bfa00000", "3ec00000"]
            .into_iter()
            .map(string)
            .collect(),
    );
    let expected = object(vec![
        ("bottom_cgs_f32_bits", weights.clone()),
        (
            "source_contract",
            string("param.F:8988-9052 DOUBLE PRECISION recurrence; default-REAL normalized coordinates and default-REAL stored weights"),
        ),
        ("top_cgt_f32_bits", weights),
    ]);
    if value != &expected {
        return Err(invalid(
            "product Fornberg contract or exact binary32 weights drifted",
        ));
    }
    Ok(())
}

fn validate_initialization(value: &JsonValue) -> Result<()> {
    let expected = object(vec![
        (
            "constants_f32_bits",
            object(vec![
                ("cp", string("447b6ccd")),
                ("g", string("411cf5c3")),
                ("p00", string("47c35000")),
                ("rd", string("438f851f")),
                ("rv", string("43e6c000")),
            ]),
        ),
        (
            "corrected_carriers",
            strings(&["ppi", "pp3d", "ppx", "prs", "rho", "rr", "rf"]),
        ),
        (
            "duplicate_groups",
            JsonValue::Array(
                [
                    &["ua", "u3d"] as &[&str],
                    &["va", "v3d"],
                    &["wa", "w3d"],
                    &["ppi", "pp3d", "ppx"],
                    &["tha", "th3d"],
                    &["qa", "q3d"],
                ]
                .into_iter()
                .map(strings)
                .collect(),
            ),
        ),
        (
            "moist_species_order",
            strings(&["qv", "qc", "qr", "qi", "qs", "qg"]),
        ),
        (
            "ppi_small_value_rule",
            string("if abs(ppi) < 1.0e-6 then assign positive binary32 zero before duplicating ppi into pp3d and ppx"),
        ),
        (
            "qv_density_rule",
            string("thv uses source qv unchanged; only the density gas-constant factor uses max(0.0,qv)"),
        ),
        (
            "source_preserved_carriers",
            strings(&[
                "pi0", "prs0", "qv0", "rho0", "th0", "u0", "v0", "ua", "u3d",
                "va", "v3d", "wa", "w3d", "tha", "th3d", "qa", "q3d",
                "terrain_msl",
            ]),
        ),
        (
            "statement_order",
            string("init3d.F:1733-1785 then 2365-2467 in source loop and assignment order"),
        ),
    ]);
    if value != &expected {
        return Err(invalid(
            "product initialization constants, statement order, carrier groups, or rules drifted",
        ));
    }
    Ok(())
}

fn validate_duplicate_groups(carriers: &[ProductRestartCarrier]) -> Result<()> {
    for group in [
        &["ua", "u3d"] as &[&str],
        &["va", "v3d"],
        &["wa", "w3d"],
        &["ppi", "pp3d", "ppx"],
        &["tha", "th3d"],
        &["qa", "q3d"],
    ] {
        let first = carriers
            .iter()
            .find(|carrier| carrier.name == group[0])
            .expect("fixed carrier exists");
        for name in &group[1..] {
            let other = carriers
                .iter()
                .find(|carrier| carrier.name == *name)
                .expect("fixed carrier exists");
            if first.shape != other.shape
                || first.nbytes != other.nbytes
                || first.sha256 != other.sha256
                || first.finite_count != other.finite_count
                || first.nonzero_count != other.nonzero_count
                || first.minimum_f32_bits != other.minimum_f32_bits
                || first.maximum_f32_bits != other.maximum_f32_bits
            {
                return Err(invalid(format!(
                    "product duplicate carrier group {group:?} is not byte-identical"
                )));
            }
            if first.offset_nbytes == other.offset_nbytes {
                return Err(invalid(format!(
                    "product duplicate carrier group {group:?} aliases one payload offset"
                )));
            }
        }
    }
    Ok(())
}

fn expected_carrier_shape(name: &str, nz: usize, ny: usize, nx: usize) -> Vec<usize> {
    match name {
        "pi0" | "prs0" | "qv0" | "rho0" | "th0" | "u0" | "v0" => vec![nz],
        "ua" | "u3d" => vec![nz, ny, nx + 1],
        "va" | "v3d" => vec![nz, ny + 1, nx],
        "wa" | "w3d" => vec![nz + 1, ny, nx],
        "ppi" | "pp3d" | "ppx" | "tha" | "th3d" | "prs" | "rho" | "rr" | "zh" | "mh" | "rmh"
        | "rho0_3d" | "rr0" => vec![nz, ny, nx],
        "qa" | "q3d" => vec![6, nz, ny, nx],
        "terrain_msl" | "gz" | "rgz" | "dzdx" | "dzdy" => vec![ny, nx],
        "zf" | "mf" | "rmf" | "c1" | "c2" | "rf0" | "rrf0" | "gx" | "gy" | "rf" => {
            vec![nz + 1, ny, nx]
        }
        "gzu" | "rgzu" => vec![ny, nx + 1],
        "gzv" | "rgzv" => vec![ny + 1, nx],
        "gxu" => vec![nz + 1, ny, nx + 1],
        "gyv" => vec![nz + 1, ny + 1, nx],
        other => panic!("unhandled fixed initialized-product carrier {other}"),
    }
}

#[derive(Debug)]
struct StreamDigest {
    sha256: String,
}

fn stream_exact<R: Read, W: Write>(
    reader: &mut R,
    writer: &mut W,
    nbytes: u64,
    mut artifact_hasher: Option<&mut Sha256>,
) -> Result<StreamDigest> {
    let mut hasher = Sha256::default();
    let mut remaining = nbytes;
    let mut buffer = vec![0u8; IO_CHUNK_BYTES];
    while remaining != 0 {
        let take =
            usize::try_from(remaining.min(buffer.len() as u64)).expect("bounded chunk fits usize");
        reader.read_exact(&mut buffer[..take])?;
        writer.write_all(&buffer[..take])?;
        hasher.update(&buffer[..take]);
        if let Some(artifact) = artifact_hasher.as_deref_mut() {
            artifact.update(&buffer[..take]);
        }
        remaining -= take as u64;
    }
    Ok(StreamDigest {
        sha256: sha256::hex(&hasher.finalize()),
    })
}

fn scan_payload<R: Read>(
    reader: &mut R,
    carriers: &[ProductRestartCarrier],
    expected_payload_sha256: &str,
    mut writer: Option<&mut BufWriter<File>>,
    mut artifact_hasher: Option<&mut Sha256>,
    mut captures: Option<&mut BTreeMap<String, Vec<u8>>>,
) -> Result<()> {
    let mut aggregate = Sha256::default();
    let mut buffer = vec![0u8; IO_CHUNK_BYTES];
    for carrier in carriers {
        let mut field = Sha256::default();
        let mut remaining = carrier.nbytes;
        let mut finite_count = 0usize;
        let mut nonzero_count = 0usize;
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        while remaining != 0 {
            let take = usize::try_from(remaining.min(buffer.len() as u64))
                .expect("bounded chunk fits usize");
            reader.read_exact(&mut buffer[..take])?;
            let bytes = &buffer[..take];
            field.update(bytes);
            aggregate.update(bytes);
            if let Some(artifact) = artifact_hasher.as_deref_mut() {
                artifact.update(bytes);
            }
            if let Some(destination) = writer.as_deref_mut() {
                destination.write_all(bytes)?;
            }
            if let Some(captures) = captures.as_deref_mut() {
                if let Some(destination) = captures.get_mut(&carrier.name) {
                    destination.extend_from_slice(bytes);
                }
            }
            for word in bytes.chunks_exact(4) {
                let value = f32::from_le_bytes(word.try_into().expect("four-byte word"));
                if !value.is_finite() {
                    return Err(invalid(format!(
                        "initialized-product carrier {} contains a non-finite word at index {}",
                        carrier.name, finite_count
                    )));
                }
                finite_count += 1;
                if value.to_bits() & 0x7fff_ffff != 0 {
                    nonzero_count += 1;
                }
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
            remaining -= take as u64;
        }
        let digest = sha256::hex(&field.finalize());
        if digest != carrier.sha256 {
            return Err(invalid(format!(
                "initialized-product carrier {} SHA-256 is {digest}; manifest declares {}",
                carrier.name, carrier.sha256
            )));
        }
        if finite_count != carrier.finite_count
            || nonzero_count != carrier.nonzero_count
            || minimum.to_bits() != carrier.minimum_f32_bits
            || maximum.to_bits() != carrier.maximum_f32_bits
        {
            return Err(invalid(format!(
                "initialized-product carrier {} statistics differ from its manifest",
                carrier.name
            )));
        }
    }
    let digest = sha256::hex(&aggregate.finalize());
    if digest != expected_payload_sha256 {
        return Err(invalid(format!(
            "initialized-product aggregate payload SHA-256 is {digest}; expected {expected_payload_sha256}"
        )));
    }
    Ok(())
}

fn restart_header(product: &VerifiedProductExport) -> JsonValue {
    object(vec![
        ("admission", string(product.admission.as_str())),
        ("capability", string(INITIALIZED_PRODUCT_RESTART_CAPABILITY)),
        ("carrier_abi", string(INITIALIZED_PRODUCT_CARRIER_ABI)),
        ("carrier_count", number(product.carriers.len())),
        (
            "claim_limit",
            string(INITIALIZED_PRODUCT_RESTART_CLAIM_LIMIT),
        ),
        ("payload_nbytes", u64_number(product.payload_nbytes)),
        ("payload_sha256", string(&product.payload_sha256)),
        ("product_manifest", product.manifest.clone()),
        (
            "product_manifest_nbytes",
            number(product.manifest_bytes.len()),
        ),
        ("product_manifest_sha256", string(&product.manifest_sha256)),
        ("product_receipt", product.receipt.clone()),
        (
            "product_receipt_nbytes",
            number(product.receipt_bytes.len()),
        ),
        ("product_receipt_sha256", string(&product.receipt_sha256)),
        (
            "representation",
            string(INITIALIZED_PRODUCT_RESTART_REPRESENTATION),
        ),
        ("schema", string(INITIALIZED_PRODUCT_RESTART_SCHEMA)),
        (
            "source_artifact_sha256",
            string(&product.source_artifact_sha256),
        ),
        (
            "source_contract_sha256",
            string(&product.source_contract_sha256),
        ),
    ])
}

fn open_regular_file(path: &Path, label: &str) -> Result<(BufReader<File>, u64)> {
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
    let file = File::open(path)?;
    Ok((BufReader::new(file), metadata.len()))
}

fn read_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let (mut reader, nbytes) = open_regular_file(path, label)?;
    if nbytes > maximum {
        return Err(invalid(format!(
            "{label} has {nbytes} bytes and exceeds the {maximum}-byte limit"
        )));
    }
    let length = usize::try_from(nbytes)
        .map_err(|_| invalid(format!("{label} size does not fit this platform")))?;
    let mut bytes = vec![0u8; length];
    reader.read_exact(&mut bytes)?;
    ensure_reader_eof(&mut reader, label)?;
    Ok(bytes)
}

fn parse_canonical_json(bytes: &[u8], label: &str) -> Result<JsonValue> {
    let parsed =
        json::parse(bytes).map_err(|error| invalid(format!("invalid {label}: {error}")))?;
    if parsed.canonical_bytes() != bytes {
        return Err(invalid(format!(
            "{label} is not canonical sorted-key compact JSON"
        )));
    }
    Ok(parsed)
}

fn ensure_reader_eof<R: Read>(reader: &mut R, label: &str) -> Result<()> {
    let mut byte = [0u8; 1];
    if reader.read(&mut byte)? != 0 {
        return Err(invalid(format!("{label} contains trailing bytes")));
    }
    Ok(())
}

fn refuse_existing_file(path: &Path, label: &str) -> Result<()> {
    if fs::symlink_metadata(path).is_ok() {
        return Err(invalid(format!(
            "refusing to replace {label} {}",
            path.display()
        )));
    }
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "{label} parent directory does not exist: {}",
            parent.display()
        )));
    }
    Ok(())
}

fn temporary_file_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("restart destination needs a UTF-8 file name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(output.with_file_name(format!(".{name}.partial-{}-{sequence}", std::process::id())))
}

fn temporary_directory_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("restore destination needs a UTF-8 directory name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(output.with_file_name(format!(".{name}.restore-{}-{sequence}", std::process::id())))
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
        .map(|(key, _)| key.as_str())
        .collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid(format!(
            "{label} keys are {actual:?}; exactly {expected:?} is required"
        )));
    }
    Ok(())
}

fn member<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a JsonValue> {
    value
        .get(key)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

/// Admit one real-product restart source by the properties a reader needs.
///
/// The source artifact and contract identities are separately required to equal
/// the caller's independent expectation, so this is not about which run produced
/// the source.  It checks that the source is a kind this distribution has a
/// reader for, at the timestep the real lane is compiled for, carrying a clock
/// the model could actually have produced.
///
/// This replaced a whitelist of exactly three historical artifacts pinned by
/// SHA-256, byte count, step and model time.  Under that whitelist a real lane
/// advanced to any other analysis hour, or a source republished after a
/// producer fix, was refused with nothing wrong with it.
fn admit_real_product_source(
    source_kind: &str,
    profile_id: &str,
    dt_bits: &str,
    step: u64,
    model_time_bits: &str,
) -> Result<()> {
    if profile_id != SOURCE_PROFILE_ID {
        return Err(invalid(format!(
            "real-product restart source profile is {profile_id:?}; this reader decodes only {SOURCE_PROFILE_ID:?} carriers, and another profile's arena would be split into carriers at offsets that belong to different fields"
        )));
    }
    if dt_bits != REAL_PRODUCT_DT_F64_BITS {
        return Err(invalid(format!(
            "real-product restart source advances at dt bits {dt_bits}; the real lane's acoustic and RK cadence is derived from dt={REAL_PRODUCT_DT_S} s ({REAL_PRODUCT_DT_F64_BITS}), so another timestep would integrate this state on a cadence it was never advanced by"
        )));
    }
    validate_hex_bits(
        model_time_bits,
        16,
        "product manifest.source.model_time_f64_bits",
    )?;
    let model_time = f64::from_bits(
        u64::from_str_radix(model_time_bits, 16)
            .map_err(|_| invalid("product manifest.source.model_time_f64_bits does not parse"))?,
    );
    // CM1 r21.1 advances the model clock by exactly one `dbldt` per completed
    // large step (`src/cm1.F:3510`, whose comment is "do not change mtime
    // anywhere else") and increments `nstep` once per step (`src/cm1.F:3171`);
    // the restart writer persists both (`src/restart_write.F:364` and
    // `src/restart_write.F:403`).  dt=0.5 s is a power of two, so CM1's
    // repeated addition is exactly `step * dt`.
    let expected_model_time = step as f64 * REAL_PRODUCT_DT_S;
    if !model_time.is_finite() || model_time.to_bits() != expected_model_time.to_bits() {
        return Err(invalid(format!(
            "real-product restart source records step {step} with model time {model_time:?}; a completed large step at dt={REAL_PRODUCT_DT_S} s is {expected_model_time:?}, so this source was not captured at a step boundary and every carrier would be dated by a clock that never produced it"
        )));
    }
    match source_kind {
        "initial-checkpoint" => {
            if step != 0 {
                return Err(invalid(format!(
                    "real-product initial-checkpoint source records completed step {step}; an initial checkpoint is the analysis a run starts from, so a nonzero step means an evolved state is being read as an initialization"
                )));
            }
        }
        "boundary-analysis-history-f01" | "boundary-analysis-history-f02-roi" => {
            if step == 0 {
                return Err(invalid(format!(
                    "real-product {source_kind} source records step zero; a boundary analysis history is a later analysis valid time, so step zero means the initial analysis is being read as a future boundary"
                )));
            }
        }
        other => {
            return Err(invalid(format!(
                "real-product restart source kind {other:?} has no reader in this distribution; its carrier meaning, valid time and boundary role are undefined here, so it would be consumed as whichever kind happened to be assumed"
            )));
        }
    }
    Ok(())
}

fn member_str<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    member(value, key, label)?
        .as_str()
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn member_u64(value: &JsonValue, key: &str, label: &str) -> Result<u64> {
    member(value, key, label)?
        .as_u64()
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn fixed_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = member_str(value, key, label)?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} is {actual:?}; required {expected:?}"
        )));
    }
    Ok(())
}

fn fixed_u64(value: &JsonValue, key: &str, expected: u64, label: &str) -> Result<()> {
    let actual = member_u64(value, key, label)?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} is {actual}; required {expected}"
        )));
    }
    Ok(())
}

fn fixed_bool(value: &JsonValue, key: &str, expected: bool, label: &str) -> Result<()> {
    let actual = member(value, key, label)?;
    if actual != &JsonValue::Bool(expected) {
        return Err(invalid(format!(
            "{label}.{key} is not the required boolean {expected}"
        )));
    }
    Ok(())
}

fn usize_array(value: &JsonValue, length: usize, label: &str) -> Result<Vec<usize>> {
    let values = usize_array_any(value, label)?;
    if values.len() != length {
        return Err(invalid(format!(
            "{label} has {} entries; exactly {length} are required",
            values.len()
        )));
    }
    Ok(values)
}

fn usize_array_any(value: &JsonValue, label: &str) -> Result<Vec<usize>> {
    let values = value
        .as_array()
        .map_err(|error| invalid(format!("{label}: {error}")))?;
    if values.is_empty() {
        return Err(invalid(format!("{label} is empty")));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let raw = item
                .as_u64()
                .map_err(|error| invalid(format!("{label}[{index}]: {error}")))?;
            usize_from_u64(raw, &format!("{label}[{index}]"))
        })
        .collect()
}

fn usize_from_u64(value: u64, label: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid(format!("{label} exceeds this platform")))
}

fn shape_product(shape: &[usize], name: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("carrier {name} shape overflows")))
    })
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    validate_hex_bits(value, 64, label)
}

fn validate_hex_bits(value: &str, length: usize, label: &str) -> Result<()> {
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{label} must be {length} lower-case hexadecimal characters"
        )));
    }
    Ok(())
}

fn parse_hex_u32(value: &str, label: &str) -> Result<u32> {
    validate_hex_bits(value, 8, label)?;
    u32::from_str_radix(value, 16).map_err(|_| invalid(format!("{label} is not binary32 bits")))
}

fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    )
}

fn strings(values: &[&str]) -> JsonValue {
    JsonValue::Array(values.iter().copied().map(string).collect())
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
    use std::sync::atomic::{AtomicU64, Ordering};

    fn bits(value: f64) -> String {
        format!("{:016x}", value.to_bits())
    }

    /// The real-product source is admitted by what a reader needs, not by the
    /// identity of one of three historical artifacts.
    #[test]
    fn any_completed_step_real_product_source_is_admitted() {
        // The three historical sources stay admitted.
        admit_real_product_source(
            "initial-checkpoint",
            SOURCE_PROFILE_ID,
            REAL_PRODUCT_DT_F64_BITS,
            0,
            &bits(0.0),
        )
        .unwrap();
        admit_real_product_source(
            "boundary-analysis-history-f01",
            SOURCE_PROFILE_ID,
            REAL_PRODUCT_DT_F64_BITS,
            7_200,
            &bits(3_600.0),
        )
        .unwrap();
        admit_real_product_source(
            "boundary-analysis-history-f02-roi",
            SOURCE_PROFILE_ID,
            REAL_PRODUCT_DT_F64_BITS,
            14_400,
            &bits(7_200.0),
        )
        .unwrap();
        // And so is any other completed large step of the same kinds.
        for step in [2u64, 1_200, 21_600, 28_800, 43_200] {
            admit_real_product_source(
                "boundary-analysis-history-f02-roi",
                SOURCE_PROFILE_ID,
                REAL_PRODUCT_DT_F64_BITS,
                step,
                &bits(step as f64 * REAL_PRODUCT_DT_S),
            )
            .unwrap();
        }
    }

    #[test]
    fn real_product_source_refusals_name_the_wrong_result_they_prevent() {
        let error = admit_real_product_source(
            "boundary-analysis-history-f01",
            SOURCE_PROFILE_ID,
            REAL_PRODUCT_DT_F64_BITS,
            7_200,
            &bits(3_600.25),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("not captured at a step boundary"), "{error}");

        let error = admit_real_product_source(
            "initial-checkpoint",
            SOURCE_PROFILE_ID,
            REAL_PRODUCT_DT_F64_BITS,
            7_200,
            &bits(3_600.0),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("evolved state is being read as an initialization"),
            "{error}"
        );

        let error = admit_real_product_source(
            "boundary-analysis-history-f01",
            SOURCE_PROFILE_ID,
            REAL_PRODUCT_DT_F64_BITS,
            0,
            &bits(0.0),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("read as a future boundary"), "{error}");

        let error = admit_real_product_source(
            "some-other-history",
            SOURCE_PROFILE_ID,
            REAL_PRODUCT_DT_F64_BITS,
            7_200,
            &bits(3_600.0),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("has no reader in this distribution"),
            "{error}"
        );

        let error = admit_real_product_source(
            "initial-checkpoint",
            "some-other-profile",
            REAL_PRODUCT_DT_F64_BITS,
            0,
            &bits(0.0),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("offsets that belong to different fields"),
            "{error}"
        );

        let error = admit_real_product_source(
            "initial-checkpoint",
            SOURCE_PROFILE_ID,
            &bits(0.25),
            0,
            &bits(0.0),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("cadence it was never advanced by"),
            "{error}"
        );
    }

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
    const SOURCE_ARTIFACT: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";
    const SOURCE_CONTRACT: &str =
        "2222222222222222222222222222222222222222222222222222222222222222";

    struct Fixture {
        directory: PathBuf,
        manifest_sha256: String,
        payload_sha256: String,
        receipt_sha256: String,
    }

    fn temporary(label: &str) -> PathBuf {
        let sequence = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "gpuwm-cm1-product-restart-{label}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create owned test directory");
        path
    }

    fn duplicate_seed(name: &str) -> usize {
        match name {
            "ua" | "u3d" => 7,
            "va" | "v3d" => 9,
            "wa" | "w3d" => 11,
            "ppi" | "pp3d" | "ppx" => 13,
            "tha" | "th3d" => 17,
            "qa" | "q3d" => 19,
            _ => {
                INITIALIZED_PRODUCT_CARRIER_NAMES
                    .iter()
                    .position(|candidate| *candidate == name)
                    .expect("fixed name")
                    + 31
            }
        }
    }

    fn fixture(root: &Path) -> Fixture {
        let directory = root.join("product");
        fs::create_dir(&directory).unwrap();
        let (nz, ny, nx) = (2usize, 2usize, 3usize);
        let mut payload = Vec::new();
        let mut carrier_json = Vec::new();
        let mut offset = 0u64;
        for name in INITIALIZED_PRODUCT_CARRIER_NAMES {
            let shape = expected_carrier_shape(name, nz, ny, nx);
            let words = shape_product(&shape, name).unwrap();
            let seed = duplicate_seed(name);
            let values = (0..words)
                .map(|index| seed as f32 + index as f32 * 0.125)
                .collect::<Vec<_>>();
            let mut bytes = Vec::with_capacity(words * 4);
            for value in &values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let nbytes = bytes.len() as u64;
            carrier_json.push(object(vec![
                ("dtype", string("<f4")),
                ("itemsize", number(4)),
                ("name", string(name)),
                ("nbytes", u64_number(nbytes)),
                ("offset_nbytes", u64_number(offset)),
                ("order", string("C")),
                ("role", string("test-physical-domain-carrier")),
                (
                    "shape",
                    JsonValue::Array(shape.iter().copied().map(number).collect()),
                ),
                ("sha256", string(&sha256::digest_hex(&bytes))),
                ("source_arrays", strings(&["test-source"])),
                (
                    "statistics",
                    object(vec![
                        ("finite_count", number(words)),
                        (
                            "maximum_f32_bits",
                            string(&format!("{:08x}", maximum.to_bits())),
                        ),
                        (
                            "minimum_f32_bits",
                            string(&format!("{:08x}", minimum.to_bits())),
                        ),
                        ("nonzero_count", number(words)),
                    ]),
                ),
            ]));
            payload.extend_from_slice(&bytes);
            offset += nbytes;
        }
        let payload_sha256 = sha256::digest_hex(&payload);
        let authority = object(vec![
            ("release", string("CM1 21.1")),
            (
                "source_files",
                JsonValue::Array(
                    AUTHORITY_FILES
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
            ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
        ]);
        let manifest = object(vec![
            ("admission", string("generic-schema")),
            ("authority", authority),
            ("base_policy", string(TERRAIN_STATIC_BASE_POLICY)),
            ("carrier_abi", string(INITIALIZED_PRODUCT_CARRIER_ABI)),
            ("carriers", JsonValue::Array(carrier_json)),
            ("claim_limit", string(INITIALIZED_PRODUCT_CLAIM_LIMIT)),
            (
                "configuration",
                object(vec![
                    ("hadvordrs", number(5)),
                    ("hadvordrv", number(5)),
                    ("ibalance", number(1)),
                    ("iice", number(1)),
                    (
                        "lateral_bc_west_east_south_north",
                        JsonValue::Array([2usize; 4].into_iter().map(number).collect()),
                    ),
                    ("psolver", number(3)),
                    ("setppi", JsonValue::Bool(true)),
                ]),
            ),
            (
                "crop",
                object(vec![
                    (
                        "count_yx",
                        JsonValue::Array([ny, nx].into_iter().map(number).collect()),
                    ),
                    (
                        "origin_yx",
                        JsonValue::Array([0usize, 0].into_iter().map(number).collect()),
                    ),
                    ("stagger_rule", string(INITIALIZED_PRODUCT_STAGGER_RULE)),
                ]),
            ),
            ("dtype", string("<f4")),
            (
                "formula_contracts",
                object(vec![
                    ("base_reciprocals", string("base.F:2160-2190")),
                    (
                        "horizontal_gradients",
                        string("adv_routines.F:9255-9322 with flx6 at 6026-6037 and bc2d open-edge halos"),
                    ),
                    (
                        "moist_initialization",
                        string("init3d.F:1733-1785,2365-2467 with param.F:8988-9052 Fornberg coefficients"),
                    ),
                    ("terrain_coordinates", string("init_terrain.F:175-297")),
                    ("vertical_metrics", string("param.F:8464-8492,8603-8627")),
                ]),
            ),
            (
                "fornberg",
                object(vec![
                    (
                        "bottom_cgs_f32_bits",
                        strings(&["3ff00000", "bfa00000", "3ec00000"]),
                    ),
                    (
                        "source_contract",
                        string("param.F:8988-9052 DOUBLE PRECISION recurrence; default-REAL normalized coordinates and default-REAL stored weights"),
                    ),
                    (
                        "top_cgt_f32_bits",
                        strings(&["3ff00000", "bfa00000", "3ec00000"]),
                    ),
                ]),
            ),
            (
                "initialization",
                object(vec![
                    (
                        "constants_f32_bits",
                        object(vec![
                            ("cp", string("447b6ccd")),
                            ("g", string("411cf5c3")),
                            ("p00", string("47c35000")),
                            ("rd", string("438f851f")),
                            ("rv", string("43e6c000")),
                        ]),
                    ),
                    (
                        "corrected_carriers",
                        strings(&["ppi", "pp3d", "ppx", "prs", "rho", "rr", "rf"]),
                    ),
                    (
                        "duplicate_groups",
                        JsonValue::Array(
                            [
                                &["ua", "u3d"] as &[&str],
                                &["va", "v3d"],
                                &["wa", "w3d"],
                                &["ppi", "pp3d", "ppx"],
                                &["tha", "th3d"],
                                &["qa", "q3d"],
                            ]
                            .into_iter()
                            .map(strings)
                            .collect(),
                        ),
                    ),
                    (
                        "moist_species_order",
                        strings(&["qv", "qc", "qr", "qi", "qs", "qg"]),
                    ),
                    (
                        "ppi_small_value_rule",
                        string("if abs(ppi) < 1.0e-6 then assign positive binary32 zero before duplicating ppi into pp3d and ppx"),
                    ),
                    (
                        "qv_density_rule",
                        string("thv uses source qv unchanged; only the density gas-constant factor uses max(0.0,qv)"),
                    ),
                    (
                        "source_preserved_carriers",
                        strings(&[
                            "pi0", "prs0", "qv0", "rho0", "th0", "u0", "v0",
                            "ua", "u3d", "va", "v3d", "wa", "w3d", "tha", "th3d",
                            "qa", "q3d", "terrain_msl",
                        ]),
                    ),
                    (
                        "statement_order",
                        string("init3d.F:1733-1785 then 2365-2467 in source loop and assignment order"),
                    ),
                ]),
            ),
            ("itemsize", number(4)),
            ("order", string("C")),
            (
                "payload",
                object(vec![
                    ("file", string(INITIALIZED_PRODUCT_PAYLOAD_FILE)),
                    ("nbytes", u64_number(payload.len() as u64)),
                    ("sha256", string(&payload_sha256)),
                ]),
            ),
            ("schema", string(INITIALIZED_PRODUCT_SCHEMA)),
            (
                "source",
                object(vec![
                    ("artifact_nbytes", number(123)),
                    ("artifact_sha256", string(SOURCE_ARTIFACT)),
                    ("contract_sha256", string(SOURCE_CONTRACT)),
                    ("dt_f64_bits", string("3fe0000000000000")),
                    ("model_time_f64_bits", string("0000000000000000")),
                    ("profile_id", string("generic-test-profile")),
                    ("step", number(0)),
                ]),
            ),
            (
                "source_grid",
                object(vec![
                    (
                        "shape_zyx",
                        JsonValue::Array([nz, ny, nx].into_iter().map(number).collect()),
                    ),
                    ("target_dx_m_f64_bits", string("4059000000000000")),
                    ("target_dy_m_f64_bits", string("4059000000000000")),
                    ("target_dz_m_f64_bits", string("4069000000000000")),
                ]),
            ),
        ]);
        let manifest_bytes = manifest.canonical_bytes();
        let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
        let receipt = object(vec![
            ("admission", string("generic-schema")),
            ("carrier_abi", string(INITIALIZED_PRODUCT_CARRIER_ABI)),
            ("carrier_count", number(49)),
            ("claim_limit", string(INITIALIZED_PRODUCT_CLAIM_LIMIT)),
            ("manifest_nbytes", number(manifest_bytes.len())),
            ("manifest_sha256", string(&manifest_sha256)),
            ("payload_nbytes", u64_number(payload.len() as u64)),
            ("payload_sha256", string(&payload_sha256)),
            (
                "publication_contract",
                string(INITIALIZED_PRODUCT_PUBLICATION_CONTRACT),
            ),
            ("schema", string(INITIALIZED_PRODUCT_RECEIPT_SCHEMA)),
            ("source_artifact_sha256", string(SOURCE_ARTIFACT)),
            ("source_contract_sha256", string(SOURCE_CONTRACT)),
        ]);
        let receipt_bytes = receipt.canonical_bytes();
        let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
        fs::write(directory.join(INITIALIZED_PRODUCT_PAYLOAD_FILE), payload).unwrap();
        fs::write(
            directory.join(INITIALIZED_PRODUCT_MANIFEST_FILE),
            manifest_bytes,
        )
        .unwrap();
        fs::write(
            directory.join(INITIALIZED_PRODUCT_RECEIPT_FILE),
            receipt_bytes,
        )
        .unwrap();
        Fixture {
            directory,
            manifest_sha256,
            payload_sha256,
            receipt_sha256,
        }
    }

    fn write_restart(fixture: &Fixture, path: &Path) -> ProductRestartReceipt {
        write_initialized_product_restart_new(
            &fixture.directory,
            path,
            &fixture.manifest_sha256,
            &fixture.payload_sha256,
            &fixture.receipt_sha256,
            SOURCE_ARTIFACT,
            SOURCE_CONTRACT,
        )
        .unwrap()
    }

    fn read_restart(path: &Path, receipt: &ProductRestartReceipt) -> RestoredProductRestart {
        read_initialized_product_restart(
            path,
            &receipt.artifact_sha256,
            &receipt.product_manifest_sha256,
            &receipt.payload_sha256,
            &receipt.product_receipt_sha256,
            &receipt.source_artifact_sha256,
            &receipt.source_contract_sha256,
        )
        .unwrap()
    }

    #[test]
    fn deterministic_write_read_restore_and_rewrite_are_byte_exact() {
        let root = temporary("roundtrip");
        let fixture = fixture(&root);
        let first_path = root.join("first.cm1ipr");
        let second_path = root.join("second.cm1ipr");
        let first = write_restart(&fixture, &first_path);
        let second = write_restart(&fixture, &second_path);
        assert_eq!(first, second);
        assert_eq!(
            fs::read(&first_path).unwrap(),
            fs::read(&second_path).unwrap()
        );
        assert_eq!(first.carrier_count, 49);
        assert!(first
            .canonical_json()
            .contains(INITIALIZED_PRODUCT_RESTART_RECEIPT_SCHEMA));

        let checkpoint = read_restart(&first_path, &first);
        assert_eq!(checkpoint.carriers().len(), 49);
        assert_eq!(checkpoint.carrier("qa").unwrap().shape, [6, 2, 2, 3]);
        assert!(checkpoint.carrier("not-a-carrier").is_none());
        let restored_directory = root.join("restored");
        let restore_receipt =
            restore_initialized_product_restart_new(&checkpoint, &restored_directory).unwrap();
        assert_eq!(
            restore_receipt.restart_artifact_sha256,
            first.artifact_sha256
        );
        for name in [
            INITIALIZED_PRODUCT_MANIFEST_FILE,
            INITIALIZED_PRODUCT_PAYLOAD_FILE,
            INITIALIZED_PRODUCT_RECEIPT_FILE,
        ] {
            assert_eq!(
                fs::read(fixture.directory.join(name)).unwrap(),
                fs::read(restored_directory.join(name)).unwrap(),
                "{name}"
            );
        }
        let third_path = root.join("third.cm1ipr");
        let third = write_initialized_product_restart_new(
            &restored_directory,
            &third_path,
            &fixture.manifest_sha256,
            &fixture.payload_sha256,
            &fixture.receipt_sha256,
            SOURCE_ARTIFACT,
            SOURCE_CONTRACT,
        )
        .unwrap();
        assert_eq!(third, first);
        assert_eq!(fs::read(third_path).unwrap(), fs::read(first_path).unwrap());
    }

    #[test]
    fn geopotential_auxiliary_is_source_bound_and_strictly_readable() {
        let root = temporary("geopotential");
        let fixture = fixture(&root);
        let checkpoint_path = root.join("state.cm1ipr");
        let checkpoint_receipt = write_restart(&fixture, &checkpoint_path);
        let source = read_restart(&checkpoint_path, &checkpoint_receipt);
        let output = root.join("geopotential");
        let receipt = crate::geopotential::export_wsm6_geopotential_new(&source, &output)
            .expect("source-bound auxiliary export");
        assert_eq!(receipt.recovery.word_count, 3 * 2 * 3);
        assert_eq!(
            receipt.receipt_sha256,
            sha256::digest_hex(
                &fs::read(output.join(crate::geopotential::WSM6_GEOPOTENTIAL_RECEIPT_FILE))
                    .unwrap()
            )
        );
        let auxiliary = crate::geopotential::read_wsm6_geopotential_auxiliary(
            &source,
            &output,
            &receipt.manifest_sha256,
            &receipt.payload_sha256,
            &receipt.receipt_sha256,
        )
        .expect("strict auxiliary read");
        assert_eq!(auxiliary.shape, [3, 2, 3]);
        assert_eq!(auxiliary.receipt, receipt);
        let error = crate::geopotential::export_wsm6_geopotential_new(&source, &output)
            .unwrap_err()
            .to_string();
        assert!(error.contains("refusing to replace"), "{error}");

        let payload_path = output.join(crate::geopotential::WSM6_GEOPOTENTIAL_PAYLOAD_FILE);
        let mut payload = fs::read(&payload_path).unwrap();
        payload[0] ^= 1;
        fs::write(&payload_path, payload).unwrap();
        let error = crate::geopotential::read_wsm6_geopotential_auxiliary(
            &source,
            &output,
            &receipt.manifest_sha256,
            &receipt.payload_sha256,
            &receipt.receipt_sha256,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("payload SHA-256"), "{error}");
    }

    #[test]
    fn refuses_wrong_independent_identities_and_existing_destinations() {
        let root = temporary("identities");
        let fixture = fixture(&root);
        let checkpoint_path = root.join("state.cm1ipr");
        let receipt = write_restart(&fixture, &checkpoint_path);
        let error = write_initialized_product_restart_new(
            &fixture.directory,
            root.join("wrong-source.cm1ipr"),
            &fixture.manifest_sha256,
            &fixture.payload_sha256,
            &fixture.receipt_sha256,
            &"ff".repeat(32),
            SOURCE_CONTRACT,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("independent admission"), "{error}");
        let error = read_initialized_product_restart(
            &checkpoint_path,
            &"ff".repeat(32),
            &receipt.product_manifest_sha256,
            &receipt.payload_sha256,
            &receipt.product_receipt_sha256,
            SOURCE_ARTIFACT,
            SOURCE_CONTRACT,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("artifact SHA-256"), "{error}");
        let error = write_initialized_product_restart_new(
            &fixture.directory,
            &checkpoint_path,
            &fixture.manifest_sha256,
            &fixture.payload_sha256,
            &fixture.receipt_sha256,
            SOURCE_ARTIFACT,
            SOURCE_CONTRACT,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("refusing to replace"), "{error}");
        let restored = read_restart(&checkpoint_path, &receipt);
        let destination = root.join("existing-restore");
        fs::create_dir(&destination).unwrap();
        let error = restore_initialized_product_restart_new(&restored, &destination)
            .unwrap_err()
            .to_string();
        assert!(error.contains("refusing to replace"), "{error}");
    }

    #[test]
    fn refuses_payload_corruption_trailing_bytes_and_post_read_mutation() {
        let root = temporary("mutations");
        let fixture = fixture(&root);
        let checkpoint_path = root.join("state.cm1ipr");
        let receipt = write_restart(&fixture, &checkpoint_path);
        let mut bytes = fs::read(&checkpoint_path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        let corrupt_path = root.join("corrupt.cm1ipr");
        fs::write(&corrupt_path, &bytes).unwrap();
        let corrupt_sha = sha256::digest_hex(&bytes);
        let error = read_initialized_product_restart(
            &corrupt_path,
            &corrupt_sha,
            &receipt.product_manifest_sha256,
            &receipt.payload_sha256,
            &receipt.product_receipt_sha256,
            SOURCE_ARTIFACT,
            SOURCE_CONTRACT,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("carrier") && error.contains("SHA-256"),
            "{error}"
        );

        let mut truncated = fs::read(&checkpoint_path).unwrap();
        truncated.pop();
        let truncated_path = root.join("truncated.cm1ipr");
        fs::write(&truncated_path, &truncated).unwrap();
        let error = read_initialized_product_restart(
            &truncated_path,
            &sha256::digest_hex(&truncated),
            &receipt.product_manifest_sha256,
            &receipt.payload_sha256,
            &receipt.product_receipt_sha256,
            SOURCE_ARTIFACT,
            SOURCE_CONTRACT,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("header and payload require"), "{error}");

        let mut appended = fs::read(&checkpoint_path).unwrap();
        appended.push(0);
        let appended_path = root.join("appended.cm1ipr");
        fs::write(&appended_path, &appended).unwrap();
        let error = read_initialized_product_restart(
            &appended_path,
            &sha256::digest_hex(&appended),
            &receipt.product_manifest_sha256,
            &receipt.payload_sha256,
            &receipt.product_receipt_sha256,
            SOURCE_ARTIFACT,
            SOURCE_CONTRACT,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("header and payload require"), "{error}");

        let verified = read_restart(&checkpoint_path, &receipt);
        let mut changed = fs::read(&checkpoint_path).unwrap();
        *changed.last_mut().unwrap() ^= 1;
        fs::write(&checkpoint_path, changed).unwrap();
        let error = restore_initialized_product_restart_new(&verified, root.join("refused"))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("carrier") || error.contains("changed"),
            "{error}"
        );
    }

    #[test]
    fn refuses_rehashed_unknown_header_key_and_shape_drift() {
        let root = temporary("semantic-drift");
        let fixture = fixture(&root);
        let checkpoint_path = root.join("state.cm1ipr");
        let receipt = write_restart(&fixture, &checkpoint_path);
        let bytes = fs::read(&checkpoint_path).unwrap();
        let header_length = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut header = json::parse(&bytes[16..16 + header_length]).unwrap();
        if let JsonValue::Object(entries) = &mut header {
            entries.push(("unknown".to_string(), JsonValue::Bool(true)));
        }
        let header = header.canonical_bytes();
        let mut changed = Vec::new();
        changed.extend_from_slice(&INITIALIZED_PRODUCT_RESTART_MAGIC);
        changed.extend_from_slice(&(header.len() as u64).to_le_bytes());
        changed.extend_from_slice(&header);
        changed.extend_from_slice(&bytes[16 + header_length..]);
        let changed_path = root.join("unknown.cm1ipr");
        fs::write(&changed_path, &changed).unwrap();
        let error = read_initialized_product_restart(
            &changed_path,
            &sha256::digest_hex(&changed),
            &receipt.product_manifest_sha256,
            &receipt.payload_sha256,
            &receipt.product_receipt_sha256,
            SOURCE_ARTIFACT,
            SOURCE_CONTRACT,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("keys") && error.contains("unknown"),
            "{error}"
        );

        let manifest_path = fixture.directory.join(INITIALIZED_PRODUCT_MANIFEST_FILE);
        let mut manifest = json::parse(&fs::read(&manifest_path).unwrap()).unwrap();
        let carriers = match member_mut(&mut manifest, "carriers") {
            JsonValue::Array(values) => values,
            _ => unreachable!(),
        };
        let shape = match member_mut(&mut carriers[0], "shape") {
            JsonValue::Array(values) => values,
            _ => unreachable!(),
        };
        shape[0] = number(3);
        let manifest_bytes = manifest.canonical_bytes();
        let manifest_sha = sha256::digest_hex(&manifest_bytes);
        fs::write(&manifest_path, &manifest_bytes).unwrap();
        let receipt_path = fixture.directory.join(INITIALIZED_PRODUCT_RECEIPT_FILE);
        let mut product_receipt = json::parse(&fs::read(&receipt_path).unwrap()).unwrap();
        *member_mut(&mut product_receipt, "manifest_nbytes") = number(manifest_bytes.len());
        *member_mut(&mut product_receipt, "manifest_sha256") = string(&manifest_sha);
        let product_receipt_bytes = product_receipt.canonical_bytes();
        let product_receipt_sha = sha256::digest_hex(&product_receipt_bytes);
        fs::write(&receipt_path, product_receipt_bytes).unwrap();
        let error = write_initialized_product_restart_new(
            &fixture.directory,
            root.join("shape-drift.cm1ipr"),
            &manifest_sha,
            &fixture.payload_sha256,
            &product_receipt_sha,
            SOURCE_ARTIFACT,
            SOURCE_CONTRACT,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("shape") && error.contains("ABI requires"),
            "{error}"
        );
    }

    fn member_mut<'a>(value: &'a mut JsonValue, key: &str) -> &'a mut JsonValue {
        match value {
            JsonValue::Object(entries) => entries
                .iter_mut()
                .find(|(candidate, _)| candidate == key)
                .map(|(_, value)| value)
                .unwrap(),
            _ => panic!("expected object"),
        }
    }
}
