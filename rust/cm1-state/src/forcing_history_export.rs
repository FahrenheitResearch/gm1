//! Rust-owned CM1FHS1 forcing-history extraction.
//!
//! The caller supplies an independently trusted embedded-contract SHA-256.
//! Rust validates the complete forecast-history envelope, then publishes one
//! deterministic little-endian float32 arena and exact TSV/JSON offsets in a
//! new directory.  No Python history parser participates in this boundary.

use crate::forecast::{read_forecast_history_embedded, ForecastArray, ForecastArraySpec};
use crate::json::{self, JsonValue};
use crate::sha256::{self, Sha256};
use crate::{Result, StateError};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const GM1_FORCING_HISTORY_EXPORT_ABI: &str = "gm1.forcing-history-export/v1";
pub const GM1_FORCING_HISTORY_EXPORT_SCHEMA: &str = "gm1.forcing-history-export-manifest/v1";
pub const GM1_FORCING_HISTORY_EXPORT_RECEIPT_SCHEMA: &str = "gm1.forcing-history-export-receipt/v1";
pub const GM1_FORCING_HISTORY_RAW_FILE: &str = "arrays.f32le";
pub const GM1_FORCING_HISTORY_TABLE_FILE: &str = "arrays.tsv";
pub const GM1_FORCING_HISTORY_MANIFEST_FILE: &str = "manifest.json";
pub const GM1_FORCING_HISTORY_PUBLICATION_CONTRACT: &str = "Rust validates the complete CM1FHS1 envelope against an independently supplied embedded-contract SHA-256, writes one deterministic carrier arena plus offsets into a new sibling staging directory, and renames without replacing an existing destination";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForcingHistoryExportReceipt {
    pub array_count: usize,
    pub contract_sha256: String,
    pub source_artifact_sha256: String,
    pub raw_sha256: String,
    pub raw_nbytes: u64,
    pub table_sha256: String,
    pub table_nbytes: usize,
    pub manifest_sha256: String,
    pub manifest_nbytes: usize,
    pub step: u64,
    pub model_time_f64_bits: String,
    pub dt_f64_bits: String,
}

impl ForcingHistoryExportReceipt {
    pub fn canonical_json(&self) -> String {
        canonical_string(object(vec![
            ("abi", string(GM1_FORCING_HISTORY_EXPORT_ABI)),
            ("array_count", number(self.array_count)),
            ("contract_sha256", string(&self.contract_sha256)),
            ("dt_f64_bits", string(&self.dt_f64_bits)),
            ("manifest_nbytes", number(self.manifest_nbytes)),
            ("manifest_sha256", string(&self.manifest_sha256)),
            ("model", string("GM1")),
            ("model_time_f64_bits", string(&self.model_time_f64_bits)),
            (
                "publication_contract",
                string(GM1_FORCING_HISTORY_PUBLICATION_CONTRACT),
            ),
            ("raw_nbytes", u64_number(self.raw_nbytes)),
            ("raw_sha256", string(&self.raw_sha256)),
            ("schema", string(GM1_FORCING_HISTORY_EXPORT_RECEIPT_SCHEMA)),
            (
                "source_artifact_sha256",
                string(&self.source_artifact_sha256),
            ),
            ("step", u64_number(self.step)),
            ("table_nbytes", number(self.table_nbytes)),
            ("table_sha256", string(&self.table_sha256)),
        ]))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExportArray {
    name: String,
    shape: Vec<usize>,
    offset_nbytes: u64,
    nbytes: u64,
    sha256: String,
}

impl ExportArray {
    fn json(&self) -> JsonValue {
        object(vec![
            ("dtype", string("<f4")),
            ("name", string(&self.name)),
            ("nbytes", u64_number(self.nbytes)),
            ("offset_nbytes", u64_number(self.offset_nbytes)),
            ("order", string("C")),
            (
                "shape",
                JsonValue::Array(self.shape.iter().copied().map(number).collect()),
            ),
            ("sha256", string(&self.sha256)),
        ])
    }

    fn tsv(&self) -> String {
        let shape = self
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{}\t<f4\tC\t{}\t{}\t{}\t{}\n",
            self.name, shape, self.offset_nbytes, self.nbytes, self.sha256
        )
    }
}

/// Export every validated named f32 carrier from a CM1FHS1 artifact.
///
/// `expected_contract_sha256` must come from an independent trusted receipt.
/// `output_directory` must not exist; publication is an atomic sibling rename.
pub fn export_forcing_history_new(
    input: impl AsRef<Path>,
    expected_contract_sha256: &str,
    output_directory: impl AsRef<Path>,
) -> Result<ForcingHistoryExportReceipt> {
    let input = input.as_ref();
    require_regular_nonsymlink(input, "forcing history input")?;
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace forcing-history export {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent_metadata = fs::symlink_metadata(parent).map_err(|error| {
        invalid(format!(
            "could not inspect forcing-history export parent {}: {error}",
            parent.display()
        ))
    })?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(invalid(
            "forcing-history export parent must be a real directory, not a symlink",
        ));
    }
    let (contract, restored) = read_forecast_history_embedded(input, expected_contract_sha256)?;
    let staging = staging_path(output)?;
    fs::create_dir(&staging)?;
    match export_into_directory(&contract, &restored, &staging) {
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

fn export_into_directory(
    contract: &crate::forecast::ForecastContract,
    restored: &crate::forecast::RestoredForecast,
    directory: &Path,
) -> Result<ForcingHistoryExportReceipt> {
    let frame_arrays = restored
        .frame()
        .arrays()
        .iter()
        .map(|array| (array.name(), array))
        .collect::<BTreeMap<_, _>>();
    if frame_arrays.len() != contract.arrays().len() {
        return Err(invalid(
            "validated forcing history changed its named carrier inventory",
        ));
    }
    let raw_path = directory.join(GM1_FORCING_HISTORY_RAW_FILE);
    let raw_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&raw_path)?;
    let mut writer = BufWriter::new(raw_file);
    let mut aggregate = Sha256::default();
    let mut offset = 0u64;
    let mut arrays = Vec::with_capacity(contract.arrays().len());
    for spec in contract.arrays() {
        let array = frame_arrays
            .get(spec.name())
            .copied()
            .ok_or_else(|| invalid(format!("forcing carrier {} is missing", spec.name())))?;
        let (field_sha256, nbytes) = write_array(&mut writer, &mut aggregate, spec, array)?;
        arrays.push(ExportArray {
            name: spec.name().to_string(),
            shape: spec.shape().to_vec(),
            offset_nbytes: offset,
            nbytes,
            sha256: field_sha256,
        });
        offset = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("forcing-history arena offset overflows"))?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    let raw_sha256 = sha256::hex(&aggregate.finalize());
    let table = arrays.iter().map(ExportArray::tsv).collect::<String>();
    let table_bytes = table.as_bytes();
    let table_sha256 = sha256::digest_hex(table_bytes);
    write_new_synced(&directory.join(GM1_FORCING_HISTORY_TABLE_FILE), table_bytes)?;
    let contract_json = json::parse(contract.canonical_json().as_bytes()).map_err(invalid)?;
    let source_receipt =
        json::parse(restored.receipt().canonical_json().as_bytes()).map_err(invalid)?;
    let manifest = object(vec![
        (
            "arrays",
            JsonValue::Array(arrays.iter().map(ExportArray::json).collect()),
        ),
        ("contract", contract_json),
        ("contract_sha256", string(&contract.sha256())),
        ("dtype", string("<f4")),
        ("model", string("GM1")),
        ("order", string("C")),
        (
            "publication_contract",
            string(GM1_FORCING_HISTORY_PUBLICATION_CONTRACT),
        ),
        ("raw_file", string(GM1_FORCING_HISTORY_RAW_FILE)),
        ("raw_nbytes", u64_number(offset)),
        ("raw_sha256", string(&raw_sha256)),
        ("schema", string(GM1_FORCING_HISTORY_EXPORT_SCHEMA)),
        ("source_history", source_receipt),
        ("table_file", string(GM1_FORCING_HISTORY_TABLE_FILE)),
        ("table_sha256", string(&table_sha256)),
    ]);
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    write_new_synced(
        &directory.join(GM1_FORCING_HISTORY_MANIFEST_FILE),
        &manifest_bytes,
    )?;
    let actual = fs::read_dir(directory)?
        .map(|entry| {
            entry
                .map(|value| value.file_name().to_string_lossy().into_owned())
                .map_err(StateError::from)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut actual_sorted = actual;
    actual_sorted.sort();
    let mut expected = vec![
        GM1_FORCING_HISTORY_MANIFEST_FILE.to_string(),
        GM1_FORCING_HISTORY_RAW_FILE.to_string(),
        GM1_FORCING_HISTORY_TABLE_FILE.to_string(),
    ];
    expected.sort();
    if actual_sorted != expected {
        return Err(invalid(
            "forcing-history staging directory contains an unexpected file",
        ));
    }
    Ok(ForcingHistoryExportReceipt {
        array_count: arrays.len(),
        contract_sha256: contract.sha256(),
        source_artifact_sha256: restored.receipt().artifact_sha256.clone(),
        raw_sha256,
        raw_nbytes: offset,
        table_sha256,
        table_nbytes: table_bytes.len(),
        manifest_sha256,
        manifest_nbytes: manifest_bytes.len(),
        step: restored.receipt().step,
        model_time_f64_bits: restored.receipt().model_time_f64_bits.clone(),
        dt_f64_bits: restored.receipt().dt_f64_bits.clone(),
    })
}

fn write_array(
    writer: &mut BufWriter<File>,
    aggregate: &mut Sha256,
    spec: &ForecastArraySpec,
    array: &ForecastArray,
) -> Result<(String, u64)> {
    let expected_words = spec.shape().iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("forcing carrier {} shape overflows", spec.name())))
    })?;
    if array.name() != spec.name() || array.values().len() != expected_words {
        return Err(invalid(format!(
            "forcing carrier {} differs from its validated contract",
            spec.name()
        )));
    }
    let mut field = Sha256::default();
    let mut bytes = Vec::with_capacity(1024 * 1024);
    for values in array.values().chunks((1024 * 1024) / 4) {
        bytes.clear();
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        writer.write_all(&bytes)?;
        aggregate.update(&bytes);
        field.update(&bytes);
    }
    let nbytes = expected_words
        .checked_mul(4)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| {
            invalid(format!(
                "forcing carrier {} byte count overflows",
                spec.name()
            ))
        })?;
    Ok((sha256::hex(&field.finalize()), nbytes))
}

fn require_regular_nonsymlink(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(format!(
            "{label} must be a regular non-symlink file"
        )));
    }
    Ok(())
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid("forcing-history output requires a UTF-8 directory name"))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("system clock precedes UNIX epoch"))?
        .as_nanos();
    let path = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    if fs::symlink_metadata(&path).is_ok() {
        return Err(invalid("forcing-history staging path already exists"));
    }
    Ok(path)
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
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

fn canonical_string(value: JsonValue) -> String {
    String::from_utf8(value.canonical_bytes()).expect("canonical export JSON is UTF-8")
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::{
        write_forecast_history_new, ForecastArray, ForecastArraySpec, ForecastClock,
        ForecastContract, ForecastFrame,
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new(label: &str) -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "gm1-forcing-export-{label}-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture(root: &TestRoot) -> (PathBuf, ForecastContract) {
        let contract = ForecastContract::new(
            "gm1-forcing-test/v1",
            "11".repeat(32),
            "22".repeat(32),
            vec![
                ForecastArraySpec::new(
                    "z_field",
                    "z field",
                    "1",
                    "mass",
                    vec![1, 1, 1, 3],
                    vec![1, 3],
                )
                .unwrap(),
                ForecastArraySpec::new(
                    "a_field",
                    "a field",
                    "m s-1",
                    "x face",
                    vec![1, 2],
                    vec![2],
                )
                .unwrap(),
            ],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let frame = ForecastFrame::new(
            ForecastClock::new(7, 3.5, 0.5).unwrap(),
            vec![
                ForecastArray::new("z_field", vec![1.0, -0.0, 3.0]),
                ForecastArray::new("a_field", vec![4.0, 5.0]),
            ],
            Vec::new(),
            Vec::new(),
        );
        let input = root.0.join("boundary-f01.cm1fhs");
        write_forecast_history_new(&contract, &frame, &input).unwrap();
        (input, contract)
    }

    #[test]
    fn validated_history_exports_sorted_exact_words_and_offsets() {
        let root = TestRoot::new("roundtrip");
        let (input, contract) = fixture(&root);
        let output = root.0.join("export");
        let receipt = export_forcing_history_new(&input, &contract.sha256(), &output).unwrap();
        assert_eq!(receipt.array_count, 2);
        assert_eq!(receipt.raw_nbytes, 20);
        let raw = fs::read(output.join(GM1_FORCING_HISTORY_RAW_FILE)).unwrap();
        let expected = [4.0f32, 5.0, 1.0, -0.0, 3.0]
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(raw, expected);
        assert_eq!(sha256::digest_hex(&raw), receipt.raw_sha256);
        let table = fs::read_to_string(output.join(GM1_FORCING_HISTORY_TABLE_FILE)).unwrap();
        assert!(table.starts_with("a_field\t<f4\tC\t2\t0\t8\t"));
        assert!(table.contains("\nz_field\t<f4\tC\t1,3\t8\t12\t"));
        let manifest = fs::read(output.join(GM1_FORCING_HISTORY_MANIFEST_FILE)).unwrap();
        assert_eq!(sha256::digest_hex(&manifest), receipt.manifest_sha256);
        assert_eq!(json::parse(&manifest).unwrap().canonical_bytes(), manifest);
    }

    #[test]
    fn wrong_contract_and_existing_output_refuse_without_replacement() {
        let root = TestRoot::new("refuse");
        let (input, contract) = fixture(&root);
        let output = root.0.join("export");
        assert!(export_forcing_history_new(&input, &"ff".repeat(32), &output).is_err());
        let first = export_forcing_history_new(&input, &contract.sha256(), &output).unwrap();
        let raw_before = fs::read(output.join(GM1_FORCING_HISTORY_RAW_FILE)).unwrap();
        assert!(export_forcing_history_new(&input, &contract.sha256(), &output).is_err());
        assert_eq!(
            fs::read(output.join(GM1_FORCING_HISTORY_RAW_FILE)).unwrap(),
            raw_before
        );
        assert_eq!(first.raw_sha256, sha256::digest_hex(&raw_before));
    }
}
