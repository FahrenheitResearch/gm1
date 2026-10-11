//! Strict, bounded classic-NetCDF ArWen/WRFOUT admission.
//!
//! The CDF header decoder is adapted from the Apache-2.0 Arwen volumetric
//! source by FahrenheitResearch. Unlike that reference reader,
//! this implementation never reads an entire multi-gigabyte WRFOUT file into
//! memory: it incrementally bounds the header, hashes the file by streaming,
//! and scans one contiguous variable record at a time. NetCDF-4/HDF5 is
//! rejected explicitly until a separately verified reader is bound.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::geofield::{
    ARWEN_WRF_ADAPTER_ID, GEO_FIELD_ABI, GeoFieldAdmission, GeoFieldDescriptor, GeoFieldGeometry,
    GeoGrid, GeoTime, GeoreferencedFrame, GeoreferencedWeatherAdapter,
};
use crate::input::{Gm1Dataset, VolumeField, f32_sha256};

const NC_DIMENSION: u32 = 10;
const NC_VARIABLE: u32 = 11;
const NC_ATTRIBUTE: u32 = 12;
const INITIAL_HEADER_BYTES: usize = 64 * 1024;
const MAX_HEADER_BYTES: usize = 16 * 1024 * 1024;
const IO_CHUNK_BYTES: usize = 4 * 1024 * 1024;
const WRF_GRAVITY_M_S2: f64 = 9.81;

pub const ARWEN_WRFOUT_ADMISSION_SCHEMA: &str = "gm1.view.arwen-strict-wrfout-admission/v1";
pub const ARWEN_WRFOUT_ADMISSION_STATUS: &str = "PASS_STRICT_REAL_CLASSIC_CDF_WRFOUT_GEOFRAME";
pub const ARWEN_WRFOUT_RENDER_PROFILE_ID: &str = "arwen.wrfout.classic-cdf-lambert-native/v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum CdfVersion {
    Classic,
    Offset64,
}

impl CdfVersion {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Classic => "CDF-1 classic",
            Self::Offset64 => "CDF-2 64-bit offset",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum CdfType {
    Byte,
    Char,
    Short,
    Int,
    Float,
    Double,
}

impl CdfType {
    fn from_code(code: u32) -> HeaderResult<Self> {
        match code {
            1 => Ok(Self::Byte),
            2 => Ok(Self::Char),
            3 => Ok(Self::Short),
            4 => Ok(Self::Int),
            5 => Ok(Self::Float),
            6 => Ok(Self::Double),
            other => Err(HeaderError::Unsupported(format!(
                "classic NetCDF type code {other}"
            ))),
        }
    }

    const fn size(self) -> usize {
        match self {
            Self::Byte | Self::Char => 1,
            Self::Short => 2,
            Self::Int | Self::Float => 4,
            Self::Double => 8,
        }
    }
}

#[derive(Clone, Debug)]
enum CdfAttributeValue {
    Text(String),
    Numeric(Vec<f64>),
}

impl CdfAttributeValue {
    fn scalar(&self) -> Option<f64> {
        match self {
            Self::Numeric(values) => values.first().copied(),
            Self::Text(_) => None,
        }
    }

    fn text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            Self::Numeric(_) => None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CdfDimension {
    pub name: String,
    pub len: usize,
    pub unlimited: bool,
}

#[derive(Clone, Debug)]
struct CdfVariable {
    name: String,
    dimensions: Vec<CdfDimension>,
    data_type: CdfType,
    attributes: BTreeMap<String, CdfAttributeValue>,
    begin: u64,
    is_record: bool,
    values_per_record: usize,
    bytes_per_record: u64,
}

impl CdfVariable {
    fn selected_shape(&self) -> Vec<usize> {
        self.dimensions
            .iter()
            .skip(usize::from(self.is_record))
            .map(|dimension| dimension.len)
            .collect()
    }

    fn dimension_names(&self) -> Vec<&str> {
        self.dimensions
            .iter()
            .map(|dimension| dimension.name.as_str())
            .collect()
    }

    fn attribute(&self, name: &str) -> Option<&CdfAttributeValue> {
        self.attributes.get(name)
    }
}

#[derive(Clone, Copy, Debug)]
struct Selection {
    offset: u64,
    count: usize,
}

#[derive(Debug)]
struct CdfFile {
    path: PathBuf,
    file_len: u64,
    version: CdfVersion,
    header_bytes: usize,
    num_records: usize,
    dimensions: Vec<CdfDimension>,
    global_attributes: BTreeMap<String, CdfAttributeValue>,
    variables: BTreeMap<String, CdfVariable>,
    record_stride: u64,
}

impl CdfFile {
    fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let mut file =
            File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?;
        let file_len = file
            .metadata()
            .map_err(|error| format!("stat {}: {error}", path.display()))?
            .len();
        if file_len < 8 {
            return Err(format!("{} is shorter than a CDF header", path.display()));
        }
        let max_header = usize::try_from(file_len.min(MAX_HEADER_BYTES as u64))
            .map_err(|_| "header bound does not fit usize".to_owned())?;
        let mut header_len = INITIAL_HEADER_BYTES.min(max_header);
        loop {
            file.seek(SeekFrom::Start(0))
                .map_err(|error| format!("seek {}: {error}", path.display()))?;
            let mut bytes = vec![0_u8; header_len];
            file.read_exact(&mut bytes)
                .map_err(|error| format!("read {} header: {error}", path.display()))?;
            match parse_header(&bytes) {
                Ok(parsed) => {
                    let record_variables = parsed
                        .variables
                        .values()
                        .filter(|variable| variable.is_record)
                        .count();
                    let record_stride = if record_variables == 1 {
                        parsed
                            .variables
                            .values()
                            .find(|variable| variable.is_record)
                            .map(|variable| {
                                (variable.values_per_record * variable.data_type.size()) as u64
                            })
                            .unwrap_or(0)
                    } else {
                        parsed
                            .variables
                            .values()
                            .filter(|variable| variable.is_record)
                            .try_fold(0_u64, |sum, variable| {
                                sum.checked_add(variable.bytes_per_record)
                                    .ok_or_else(|| "classic CDF record stride overflow".to_owned())
                            })?
                    };
                    let cdf = Self {
                        path,
                        file_len,
                        version: parsed.version,
                        header_bytes: parsed.header_bytes,
                        num_records: parsed.num_records,
                        dimensions: parsed.dimensions,
                        global_attributes: parsed.global_attributes,
                        variables: parsed.variables,
                        record_stride,
                    };
                    cdf.validate_ranges()?;
                    return Ok(cdf);
                }
                Err(HeaderError::Truncated) if header_len < max_header => {
                    header_len = header_len.saturating_mul(2).min(max_header);
                }
                Err(HeaderError::Truncated) => {
                    return Err(format!(
                        "{} CDF header exceeds the bounded {} byte admission limit",
                        path.display(),
                        MAX_HEADER_BYTES
                    ));
                }
                Err(HeaderError::Invalid(message)) => {
                    return Err(format!("invalid CDF {}: {message}", path.display()));
                }
                Err(HeaderError::Unsupported(message)) => {
                    return Err(format!("unsupported CDF {}: {message}", path.display()));
                }
            }
        }
    }

    fn validate_ranges(&self) -> Result<(), String> {
        for variable in self.variables.values() {
            let raw_bytes = variable
                .values_per_record
                .checked_mul(variable.data_type.size())
                .ok_or_else(|| format!("{} byte count overflow", variable.name))?;
            let last_record_offset = if variable.is_record && self.num_records > 0 {
                (self.num_records as u64 - 1)
                    .checked_mul(self.record_stride)
                    .ok_or_else(|| format!("{} last record offset overflow", variable.name))?
            } else {
                0
            };
            let end = variable
                .begin
                .checked_add(last_record_offset)
                .and_then(|value| value.checked_add(raw_bytes as u64))
                .ok_or_else(|| format!("{} byte range overflow", variable.name))?;
            if end > self.file_len {
                return Err(format!(
                    "{} byte range ends at {end}, beyond file length {}",
                    variable.name, self.file_len
                ));
            }
        }
        Ok(())
    }

    fn dimension(&self, name: &str) -> Result<&CdfDimension, String> {
        self.dimensions
            .iter()
            .find(|dimension| dimension.name == name)
            .ok_or_else(|| format!("required WRF dimension {name:?} is absent"))
    }

    fn variable(&self, name: &str) -> Result<&CdfVariable, String> {
        self.variables
            .get(name)
            .ok_or_else(|| format!("required WRF variable {name:?} is absent"))
    }

    fn global_scalar(&self, name: &str) -> Result<f64, String> {
        self.global_attributes
            .get(name)
            .and_then(CdfAttributeValue::scalar)
            .ok_or_else(|| format!("required numeric WRF global attribute {name:?} is absent"))
    }

    fn global_text(&self, name: &str) -> Result<&str, String> {
        self.global_attributes
            .get(name)
            .and_then(CdfAttributeValue::text)
            .map(str::trim)
            .ok_or_else(|| format!("required text WRF global attribute {name:?} is absent"))
    }

    fn selection(&self, variable: &CdfVariable, record: usize) -> Result<Selection, String> {
        if variable.is_record {
            if record >= self.num_records {
                return Err(format!(
                    "record {record} is outside 0..{}",
                    self.num_records
                ));
            }
            let offset = variable
                .begin
                .checked_add(
                    (record as u64)
                        .checked_mul(self.record_stride)
                        .ok_or_else(|| format!("{} record offset overflow", variable.name))?,
                )
                .ok_or_else(|| format!("{} record address overflow", variable.name))?;
            Ok(Selection {
                offset,
                count: variable.values_per_record,
            })
        } else {
            if record != 0 {
                return Err(format!(
                    "non-record variable {} cannot select record {record}",
                    variable.name
                ));
            }
            Ok(Selection {
                offset: variable.begin,
                count: variable.values_per_record,
            })
        }
    }

    fn read_text_record(&self, name: &str, record: usize) -> Result<String, String> {
        let variable = self.variable(name)?;
        if variable.data_type != CdfType::Char {
            return Err(format!("WRF variable {name} is not character data"));
        }
        let selection = self.selection(variable, record)?;
        let mut raw = vec![0_u8; selection.count];
        self.read_exact_at(selection.offset, &mut raw)?;
        let end = raw.iter().position(|byte| *byte == 0).unwrap_or(raw.len());
        String::from_utf8(raw[..end].to_vec())
            .map(|value| value.trim().to_owned())
            .map_err(|error| format!("WRF variable {name} is not UTF-8: {error}"))
    }

    fn read_scalar_f32(&self, name: &str, record: usize) -> Result<f32, String> {
        let variable = self.variable(name)?;
        if variable.data_type != CdfType::Float {
            return Err(format!("WRF scalar {name} is not exact f32 data"));
        }
        let selection = self.selection(variable, record)?;
        if selection.count != 1 {
            return Err(format!("WRF scalar {name} has {} values", selection.count));
        }
        self.read_f32_at(name, record, 0)
    }

    fn read_scalar_i32(&self, name: &str, record: usize) -> Result<i32, String> {
        let variable = self.variable(name)?;
        if variable.data_type != CdfType::Int {
            return Err(format!("WRF scalar {name} is not exact i32 data"));
        }
        let selection = self.selection(variable, record)?;
        if selection.count != 1 {
            return Err(format!("WRF scalar {name} has {} values", selection.count));
        }
        let mut raw = [0_u8; 4];
        self.read_exact_at(selection.offset, &mut raw)?;
        Ok(i32::from_be_bytes(raw))
    }

    fn read_f32_at(&self, name: &str, record: usize, index: usize) -> Result<f32, String> {
        let variable = self.variable(name)?;
        if variable.data_type != CdfType::Float {
            return Err(format!("WRF variable {name} is not exact f32 data"));
        }
        ensure_unpacked(variable)?;
        let selection = self.selection(variable, record)?;
        if index >= selection.count {
            return Err(format!(
                "WRF variable {name} index {index} is outside 0..{}",
                selection.count
            ));
        }
        let offset = selection
            .offset
            .checked_add((index as u64) * 4)
            .ok_or_else(|| format!("WRF variable {name} index offset overflow"))?;
        let mut raw = [0_u8; 4];
        self.read_exact_at(offset, &mut raw)?;
        Ok(f32::from_bits(u32::from_be_bytes(raw)))
    }

    fn read_exact_at(&self, offset: u64, output: &mut [u8]) -> Result<(), String> {
        let end = offset
            .checked_add(output.len() as u64)
            .ok_or_else(|| "CDF read range overflow".to_owned())?;
        if end > self.file_len {
            return Err(format!(
                "CDF read {offset}..{end} exceeds file length {}",
                self.file_len
            ));
        }
        let mut file = File::open(&self.path)
            .map_err(|error| format!("open {}: {error}", self.path.display()))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| format!("seek {}: {error}", self.path.display()))?;
        file.read_exact(output)
            .map_err(|error| format!("read {}: {error}", self.path.display()))
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct WrfProjection {
    pub map_proj: i32,
    pub map_proj_char: String,
    pub standard_longitude_deg: f64,
    pub true_latitude_1_deg: f64,
    pub true_latitude_2_deg: f64,
    pub mother_domain_center_latitude_deg: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArwenWrfoutMetadata {
    pub source_path: String,
    pub source_file_bytes: u64,
    pub source_sha256: String,
    pub container: &'static str,
    pub cdf_header_bytes: usize,
    pub record: usize,
    pub record_count: usize,
    pub model_title: String,
    pub valid_time: String,
    pub simulation_start_time: String,
    pub xtime_minutes: f32,
    pub itimestep: u64,
    pub dt_s: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx_m: f64,
    pub dy_m: f64,
    pub center_latitude_deg: f64,
    pub center_longitude_deg: f64,
    pub center_model_top_m: f64,
    pub projection: WrfProjection,
}

#[derive(Clone, Debug, Serialize)]
pub struct WrfFieldScan {
    pub variable: String,
    pub source_description: String,
    pub shape: Vec<usize>,
    pub units: String,
    pub stagger: String,
    pub decoded_f32_le_sha256: String,
    pub value_count: usize,
    pub finite_count: usize,
    pub invalid_count: usize,
    pub finite_min: Option<f32>,
    pub finite_max: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArwenWrfoutAdmission {
    pub schema: &'static str,
    pub status: &'static str,
    pub metadata: ArwenWrfoutMetadata,
    pub frame: GeoreferencedFrame,
    pub scans: BTreeMap<String, WrfFieldScan>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum ArwenResidencyKind {
    LatitudeSurface,
    LongitudeSurface,
    TerrainSurface,
    NativeXFaceWindVolume,
    NativeYFaceWindVolume,
    NativeZFaceWindVolume,
    PressurePerturbationVolume,
    PressureBaseStateVolume,
    GeopotentialPerturbationFaces,
    GeopotentialBaseStateFaces,
    ThetaPerturbationVolume,
    WaterVaporVolume,
    CloudWaterVolume,
    RainWaterVolume,
    CloudIceVolume,
    SnowVolume,
    GraupelVolume,
    HailVolume,
    SurfacePressure,
    NativeWrfReflectivityVolume,
    NativeWrfMaxReflectivitySurface,
}

#[derive(Clone, Debug)]
pub struct ArwenResidentField {
    pub descriptor: GeoFieldDescriptor,
    pub source_variable: String,
    pub stagger: String,
    pub residency: ArwenResidencyKind,
    pub values: Arc<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ArwenRenderBoundsKm {
    pub minimum_east_up_north_km: [f32; 3],
    pub size_east_up_north_km: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct ArwenRenderResidency {
    pub admission: ArwenWrfoutAdmission,
    pub residency_identity_sha256: String,
    pub bounds_km: ArwenRenderBoundsKm,
    pub fields: Vec<ArwenResidentField>,
}

impl ArwenRenderResidency {
    pub fn field(&self, id: &str) -> Option<&ArwenResidentField> {
        self.fields.iter().find(|field| field.descriptor.id == id)
    }

    pub fn cloud_dataset(&self) -> Result<Gm1Dataset, String> {
        let cloud = self.field("cloud_water").ok_or_else(|| {
            "ArWen rendering requires exact source QCLOUD; no fallback is permitted".to_owned()
        })?;
        if cloud.residency != ArwenResidencyKind::CloudWaterVolume {
            return Err("ArWen cloud-water residency kind changed".to_owned());
        }
        let shape: [usize; 3] = cloud
            .descriptor
            .shape
            .clone()
            .try_into()
            .map_err(|_| "ArWen QCLOUD is not rank-three [z,y,x]".to_owned())?;
        let scan = self
            .admission
            .scans
            .get("cloud_water")
            .ok_or_else(|| "ArWen QCLOUD scan is absent".to_owned())?;
        let finite_min = scan
            .finite_min
            .ok_or_else(|| "ArWen QCLOUD has no finite minimum".to_owned())?;
        let finite_max = scan
            .finite_max
            .ok_or_else(|| "ArWen QCLOUD has no finite maximum".to_owned())?;
        Ok(Gm1Dataset {
            config_sha256: String::new(),
            terrain_msl_m: None,
            face_height_m: None,
            tornado_surface_fields: Vec::new(),
            source_path: PathBuf::from(&self.admission.metadata.source_path),
            artifact_kind: "strict ArWen WRFOUT native QCLOUD".to_owned(),
            artifact_sha256: self.admission.metadata.source_sha256.clone(),
            payload_sha256: cloud.descriptor.values_sha256.clone(),
            contract_sha256: self.residency_identity_sha256.clone(),
            profile_id: ARWEN_WRFOUT_RENDER_PROFILE_ID.to_owned(),
            step: self.admission.frame.time.step,
            model_time_s: self.admission.frame.time.model_time_s,
            dt_s: self.admission.frame.time.dt_s,
            fields: vec![VolumeField {
                id: "qc".to_owned(),
                label: "ArWen/WRF native cloud-water mixing ratio QCLOUD".to_owned(),
                units: cloud.descriptor.units.clone(),
                meaning: "exact source QCLOUD only; not qc+qi, not remapped, normalized, clamped, or zero-filled".to_owned(),
                source: format!(
                    "wrfout_sha256={} record={} variable=QCLOUD values_sha256={}",
                    self.admission.metadata.source_sha256,
                    self.admission.metadata.record,
                    cloud.descriptor.values_sha256,
                ),
                shape,
                values: cloud.values.as_ref().clone(),
                values_sha256: cloud.descriptor.values_sha256.clone(),
                finite_min,
                finite_max,
            }],
        })
    }

    pub fn validate_source_native_words(&self) -> Result<(), String> {
        if self.fields.is_empty() {
            return Err("ArWen renderer residency has no available source fields".to_owned());
        }
        for field in &self.fields {
            if field.descriptor.admission != GeoFieldAdmission::Available
                || field.descriptor.values_sha256 != f32_sha256(&field.values)
                || field.descriptor.shape.iter().product::<usize>() != field.values.len()
            {
                return Err(format!(
                    "ArWen resident field {} changed after strict source admission",
                    field.descriptor.id
                ));
            }
        }
        let ph = self
            .field("geopotential_perturbation")
            .ok_or_else(|| "ArWen renderer lacks exact PH residency".to_owned())?;
        let phb = self
            .field("geopotential_base_state")
            .ok_or_else(|| "ArWen renderer lacks exact PHB residency".to_owned())?;
        if ph.descriptor.shape != phb.descriptor.shape
            || ph.residency != ArwenResidencyKind::GeopotentialPerturbationFaces
            || phb.residency != ArwenResidencyKind::GeopotentialBaseStateFaces
        {
            return Err("ArWen exact PH/PHB coordinate residency changed".to_owned());
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ArwenWrfoutSource {
    cdf: CdfFile,
    file_sha256: String,
    record: usize,
}

impl ArwenWrfoutSource {
    pub fn open(path: impl AsRef<Path>, record: usize) -> Result<Self, String> {
        let cdf = CdfFile::open(path)?;
        if record >= cdf.num_records {
            return Err(format!("record {record} is outside 0..{}", cdf.num_records));
        }
        let file_sha256 = sha256_file(&cdf.path)?;
        Ok(Self {
            cdf,
            file_sha256,
            record,
        })
    }

    pub fn open_expected(
        path: impl AsRef<Path>,
        record: usize,
        expected_sha256: &str,
    ) -> Result<Self, String> {
        require_sha256(expected_sha256, "expected WRFOUT SHA-256")?;
        let source = Self::open(path, record)?;
        if source.file_sha256 != expected_sha256 {
            return Err(format!(
                "WRFOUT SHA-256 mismatch: expected {expected_sha256}, got {}",
                source.file_sha256
            ));
        }
        Ok(source)
    }

    pub fn path(&self) -> &Path {
        &self.cdf.path
    }

    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }

    pub const fn record(&self) -> usize {
        self.record
    }

    pub fn variable_names(&self) -> impl Iterator<Item = &str> {
        self.cdf.variables.keys().map(String::as_str)
    }

    pub fn scan_field(&self, name: &str) -> Result<WrfFieldScan, String> {
        scan_exact_f32(&self.cdf, name, self.record)
    }

    /// Decode one exact source carrier without destaggering, normalization,
    /// interpolation, fill replacement, or unit conversion.
    pub fn read_exact_f32(
        &self,
        name: &str,
        maximum_values: usize,
    ) -> Result<(Vec<usize>, Vec<f32>), String> {
        let variable = self.cdf.variable(name)?;
        if variable.data_type != CdfType::Float {
            return Err(format!("WRF variable {name} is not exact f32 data"));
        }
        ensure_unpacked(variable)?;
        let selection = self.cdf.selection(variable, self.record)?;
        if selection.count > maximum_values {
            return Err(format!(
                "WRF variable {name} has {} values, above explicit bound {maximum_values}",
                selection.count
            ));
        }
        let bytes = selection
            .count
            .checked_mul(4)
            .ok_or_else(|| format!("WRF variable {name} byte count overflow"))?;
        let mut raw = vec![0_u8; bytes];
        self.cdf.read_exact_at(selection.offset, &mut raw)?;
        let mut values = Vec::with_capacity(selection.count);
        for word in raw.chunks_exact(4) {
            values.push(f32::from_bits(u32::from_be_bytes([
                word[0], word[1], word[2], word[3],
            ])));
        }
        Ok((variable.selected_shape(), values))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ArwenWrfoutAdapter;

impl ArwenWrfoutAdapter {
    pub fn admit(&self, source: &ArwenWrfoutSource) -> Result<ArwenWrfoutAdmission, String> {
        let metadata = admit_metadata(source)?;
        let mut descriptors = Vec::with_capacity(FIELD_SPECS.len());
        let mut scans = BTreeMap::new();
        for spec in FIELD_SPECS {
            match select_candidate(&source.cdf, spec.names) {
                Ok(Some(name)) => match validate_and_scan(source, spec, name) {
                    Ok(scan) if scan.invalid_count == 0 => {
                        descriptors.push(available_descriptor(source, spec, &scan));
                        scans.insert(spec.id.to_owned(), scan);
                    }
                    Ok(scan) if spec.required => {
                        return Err(format!(
                            "required WRF field {} contains {} fill/non-finite values",
                            name, scan.invalid_count
                        ));
                    }
                    Ok(scan) => {
                        descriptors.push(refused_descriptor(
                            source,
                            spec,
                            format!(
                                "source carrier {} contains {} fill/non-finite values",
                                name, scan.invalid_count
                            ),
                        ));
                        scans.insert(spec.id.to_owned(), scan);
                    }
                    Err(error) if spec.required => return Err(error),
                    Err(error) => descriptors.push(refused_descriptor(source, spec, error)),
                },
                Ok(None) if spec.required => {
                    return Err(format!(
                        "required WRF carrier {} is absent",
                        spec.names.join(" or ")
                    ));
                }
                Ok(None) => descriptors.push(missing_descriptor(source, spec)),
                Err(error) if spec.required => return Err(error),
                Err(error) => descriptors.push(refused_descriptor(source, spec, error)),
            }
        }

        let frame = GeoreferencedFrame {
            abi: GEO_FIELD_ABI,
            adapter_id: ARWEN_WRF_ADAPTER_ID,
            source_identity: format!(
                "sha256:{}#record={}#Times={}",
                source.file_sha256, source.record, metadata.valid_time
            ),
            grid: GeoGrid {
                center_latitude_deg: metadata.center_latitude_deg,
                center_longitude_deg: metadata.center_longitude_deg,
                nx: metadata.nx,
                ny: metadata.ny,
                nz: metadata.nz,
                dx_m: metadata.dx_m,
                dy_m: metadata.dy_m,
                model_top_m: metadata.center_model_top_m,
                horizontal_crs: "WRF Lambert conformal native C-grid; exact source projection attributes bound",
                vertical_coordinate: "WRF terrain-following eta; native PH/PHB Z-staggered interfaces bound",
            },
            time: GeoTime {
                step: metadata.itimestep,
                model_time_s: f64::from(metadata.xtime_minutes) * 60.0,
                dt_s: metadata.dt_s,
            },
            fields: descriptors,
        };
        Ok(ArwenWrfoutAdmission {
            schema: ARWEN_WRFOUT_ADMISSION_SCHEMA,
            status: ARWEN_WRFOUT_ADMISSION_STATUS,
            metadata,
            frame,
            scans,
        })
    }

    /// Materialize every admitted exact source carrier for renderer-side
    /// residency. This performs no destaggering, combination, normalization,
    /// clamping, quantization, or pressure/geopotential fallback.
    pub fn materialize_render_residency(
        &self,
        source: &ArwenWrfoutSource,
    ) -> Result<ArwenRenderResidency, String> {
        let admission = self.admit(source)?;
        let mut fields = Vec::new();
        for descriptor in admission
            .frame
            .fields
            .iter()
            .filter(|field| field.admission == GeoFieldAdmission::Available)
        {
            let scan = admission.scans.get(&descriptor.id).ok_or_else(|| {
                format!(
                    "available ArWen field {} lacks an exact scan",
                    descriptor.id
                )
            })?;
            let (shape, values) = source.read_exact_f32(&scan.variable, 100_000_000)?;
            if shape != descriptor.shape
                || f32_sha256(&values) != descriptor.values_sha256
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(format!(
                    "ArWen field {} changed between admission and residency",
                    descriptor.id
                ));
            }
            fields.push(ArwenResidentField {
                descriptor: descriptor.clone(),
                source_variable: scan.variable.clone(),
                stagger: scan.stagger.clone(),
                residency: arwen_residency_kind(&descriptor.id)?,
                values: Arc::new(values),
            });
        }

        let terrain_min_m = admission
            .scans
            .get("terrain_height")
            .and_then(|scan| scan.finite_min)
            .ok_or_else(|| "ArWen exact terrain minimum is absent".to_owned())?;
        let ph = fields
            .iter()
            .find(|field| field.descriptor.id == "geopotential_perturbation")
            .ok_or_else(|| "ArWen exact PH residency is absent".to_owned())?;
        let phb = fields
            .iter()
            .find(|field| field.descriptor.id == "geopotential_base_state")
            .ok_or_else(|| "ArWen exact PHB residency is absent".to_owned())?;
        let plane_words = admission
            .metadata
            .ny
            .checked_mul(admission.metadata.nx)
            .ok_or_else(|| "ArWen horizontal plane size overflow".to_owned())?;
        let top_start = admission
            .metadata
            .nz
            .checked_mul(plane_words)
            .ok_or_else(|| "ArWen top-face offset overflow".to_owned())?;
        let top_end = top_start
            .checked_add(plane_words)
            .ok_or_else(|| "ArWen top-face range overflow".to_owned())?;
        let ph_top = ph
            .values
            .get(top_start..top_end)
            .ok_or_else(|| "ArWen PH top face is truncated".to_owned())?;
        let phb_top = phb
            .values
            .get(top_start..top_end)
            .ok_or_else(|| "ArWen PHB top face is truncated".to_owned())?;
        let top_max_m =
            ph_top
                .iter()
                .zip(phb_top)
                .try_fold(f64::NEG_INFINITY, |maximum, (&ph, &phb)| {
                    let height = (f64::from(ph) + f64::from(phb)) / WRF_GRAVITY_M_S2;
                    if height.is_finite() {
                        Ok::<f64, String>(maximum.max(height))
                    } else {
                        Err("ArWen top PH+PHB height is non-finite".to_owned())
                    }
                })?;
        if top_max_m <= f64::from(terrain_min_m) {
            return Err("ArWen exact vertical bounds are inverted".to_owned());
        }
        let east_size_km = admission.metadata.nx as f64 * admission.metadata.dx_m / 1_000.0;
        let north_size_km = admission.metadata.ny as f64 * admission.metadata.dy_m / 1_000.0;
        let minimum_up_km = f64::from(terrain_min_m) / 1_000.0;
        let size_up_km = top_max_m / 1_000.0 - minimum_up_km;
        for (label, value) in [
            ("east extent", east_size_km),
            ("north extent", north_size_km),
            ("vertical extent", size_up_km),
        ] {
            if !value.is_finite() || value <= 0.0 || value > f64::from(f32::MAX) {
                return Err(format!("ArWen render {label} is invalid: {value}"));
            }
        }
        let bounds_km = ArwenRenderBoundsKm {
            minimum_east_up_north_km: [
                (-0.5 * east_size_km) as f32,
                minimum_up_km as f32,
                (-0.5 * north_size_km) as f32,
            ],
            size_east_up_north_km: [east_size_km as f32, size_up_km as f32, north_size_km as f32],
        };

        let mut identity = Sha256::new();
        identity.update(ARWEN_WRFOUT_ADMISSION_SCHEMA.as_bytes());
        identity.update(admission.metadata.source_sha256.as_bytes());
        identity.update((admission.metadata.record as u64).to_le_bytes());
        for field in &fields {
            identity.update(field.descriptor.id.as_bytes());
            identity.update([0]);
            identity.update(field.descriptor.values_sha256.as_bytes());
        }
        let residency = ArwenRenderResidency {
            admission,
            residency_identity_sha256: format!("{:x}", identity.finalize()),
            bounds_km,
            fields,
        };
        residency.validate_source_native_words()?;
        Ok(residency)
    }
}

impl GeoreferencedWeatherAdapter for ArwenWrfoutAdapter {
    type Source = ArwenWrfoutSource;

    fn adapter_id(&self) -> &'static str {
        ARWEN_WRF_ADAPTER_ID
    }

    fn adapt(&self, source: &Self::Source) -> Result<GeoreferencedFrame, String> {
        self.admit(source).map(|admission| admission.frame)
    }
}

#[derive(Clone, Copy)]
struct FieldSpec {
    names: &'static [&'static str],
    id: &'static str,
    label: &'static str,
    meaning: &'static str,
    expected_dimensions: &'static [&'static str],
    allowed_units: &'static [&'static str],
    expected_stagger: &'static str,
    geometry: GeoFieldGeometry,
    required: bool,
}

fn arwen_residency_kind(id: &str) -> Result<ArwenResidencyKind, String> {
    let kind = match id {
        "latitude" => ArwenResidencyKind::LatitudeSurface,
        "longitude" => ArwenResidencyKind::LongitudeSurface,
        "terrain_height" => ArwenResidencyKind::TerrainSurface,
        "u_staggered" => ArwenResidencyKind::NativeXFaceWindVolume,
        "v_staggered" => ArwenResidencyKind::NativeYFaceWindVolume,
        "w_staggered" => ArwenResidencyKind::NativeZFaceWindVolume,
        "pressure_perturbation" => ArwenResidencyKind::PressurePerturbationVolume,
        "pressure_base_state" => ArwenResidencyKind::PressureBaseStateVolume,
        "geopotential_perturbation" => ArwenResidencyKind::GeopotentialPerturbationFaces,
        "geopotential_base_state" => ArwenResidencyKind::GeopotentialBaseStateFaces,
        "theta_perturbation" => ArwenResidencyKind::ThetaPerturbationVolume,
        "water_vapor" => ArwenResidencyKind::WaterVaporVolume,
        "cloud_water" => ArwenResidencyKind::CloudWaterVolume,
        "rain_water" => ArwenResidencyKind::RainWaterVolume,
        "cloud_ice" => ArwenResidencyKind::CloudIceVolume,
        "snow_mixing_ratio" => ArwenResidencyKind::SnowVolume,
        "graupel_mixing_ratio" => ArwenResidencyKind::GraupelVolume,
        "hail_mixing_ratio" => ArwenResidencyKind::HailVolume,
        "surface_pressure" => ArwenResidencyKind::SurfacePressure,
        "wrf_10cm_reflectivity" => ArwenResidencyKind::NativeWrfReflectivityVolume,
        "wrf_max_reflectivity" => ArwenResidencyKind::NativeWrfMaxReflectivitySurface,
        other => {
            return Err(format!(
                "ArWen renderer has no source-native role for {other}"
            ));
        }
    };
    Ok(kind)
}

const XY: &[&str] = &["Time", "south_north", "west_east"];
const XYZ: &[&str] = &["Time", "bottom_top", "south_north", "west_east"];
const XYZI: &[&str] = &["Time", "bottom_top_stag", "south_north", "west_east"];
const U_DIMS: &[&str] = &["Time", "bottom_top", "south_north", "west_east_stag"];
const V_DIMS: &[&str] = &["Time", "bottom_top", "south_north_stag", "west_east"];

const FIELD_SPECS: &[FieldSpec] = &[
    FieldSpec {
        names: &["XLAT"],
        id: "latitude",
        label: "Latitude",
        meaning: "native WRF mass-grid latitude",
        expected_dimensions: XY,
        allowed_units: &["degree_north"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::RasterYx,
        required: true,
    },
    FieldSpec {
        names: &["XLONG"],
        id: "longitude",
        label: "Longitude",
        meaning: "native WRF mass-grid longitude",
        expected_dimensions: XY,
        allowed_units: &["degree_east"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::RasterYx,
        required: true,
    },
    FieldSpec {
        names: &["HGT", "HGT_M"],
        id: "terrain_height",
        label: "Terrain height",
        meaning: "native WRF terrain height; no terrain invention",
        expected_dimensions: XY,
        allowed_units: &["m"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::RasterYx,
        required: true,
    },
    FieldSpec {
        names: &["U"],
        id: "u_staggered",
        label: "U wind (native X stagger)",
        meaning: "native WRF x-wind; not destaggered",
        expected_dimensions: U_DIMS,
        allowed_units: &["m s-1"],
        expected_stagger: "X",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: true,
    },
    FieldSpec {
        names: &["V"],
        id: "v_staggered",
        label: "V wind (native Y stagger)",
        meaning: "native WRF y-wind; not destaggered",
        expected_dimensions: V_DIMS,
        allowed_units: &["m s-1"],
        expected_stagger: "Y",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: true,
    },
    FieldSpec {
        names: &["W"],
        id: "w_staggered",
        label: "W wind (native Z stagger)",
        meaning: "native WRF vertical wind; not destaggered",
        expected_dimensions: XYZI,
        allowed_units: &["m s-1"],
        expected_stagger: "Z",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: true,
    },
    FieldSpec {
        names: &["P"],
        id: "pressure_perturbation",
        label: "Perturbation pressure P",
        meaning: "native WRF perturbation-pressure component; not a fallback total pressure",
        expected_dimensions: XYZ,
        allowed_units: &["Pa"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: true,
    },
    FieldSpec {
        names: &["PB"],
        id: "pressure_base_state",
        label: "Base-state pressure PB",
        meaning: "native WRF base-state pressure component; not a fallback total pressure",
        expected_dimensions: XYZ,
        allowed_units: &["Pa"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: true,
    },
    FieldSpec {
        names: &["PH"],
        id: "geopotential_perturbation",
        label: "Perturbation geopotential PH",
        meaning: "native WRF perturbation-geopotential component",
        expected_dimensions: XYZI,
        allowed_units: &["m2 s-2"],
        expected_stagger: "Z",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: true,
    },
    FieldSpec {
        names: &["PHB"],
        id: "geopotential_base_state",
        label: "Base-state geopotential PHB",
        meaning: "native WRF base-state geopotential component",
        expected_dimensions: XYZI,
        allowed_units: &["m2 s-2"],
        expected_stagger: "Z",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: true,
    },
    FieldSpec {
        names: &["T"],
        id: "theta_perturbation",
        label: "Potential-temperature perturbation",
        meaning: "native WRF theta-minus-T0 carrier",
        expected_dimensions: XYZ,
        allowed_units: &["K"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: false,
    },
    FieldSpec {
        names: &["QVAPOR"],
        id: "water_vapor",
        label: "Water-vapor mixing ratio",
        meaning: "native WRF water-vapor carrier",
        expected_dimensions: XYZ,
        allowed_units: &["kg kg-1"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: false,
    },
    FieldSpec {
        names: &["QCLOUD"],
        id: "cloud_water",
        label: "Cloud-water mixing ratio",
        meaning: "native WRF cloud-water carrier; not combined or synthesized",
        expected_dimensions: XYZ,
        allowed_units: &["kg kg-1"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: false,
    },
    FieldSpec {
        names: &["QRAIN"],
        id: "rain_water",
        label: "Rain-water mixing ratio",
        meaning: "native WRF rain-water carrier; not combined or synthesized",
        expected_dimensions: XYZ,
        allowed_units: &["kg kg-1"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: false,
    },
    FieldSpec {
        names: &["QICE"],
        id: "cloud_ice",
        label: "Cloud-ice mixing ratio",
        meaning: "native WRF cloud-ice carrier; not combined or synthesized",
        expected_dimensions: XYZ,
        allowed_units: &["kg kg-1"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: false,
    },
    FieldSpec {
        names: &["QSNOW"],
        id: "snow_mixing_ratio",
        label: "Snow mixing ratio",
        meaning: "native WRF snow carrier when present; never zero-filled",
        expected_dimensions: XYZ,
        allowed_units: &["kg kg-1"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: false,
    },
    FieldSpec {
        names: &["QGRAUP", "QGRAUPEL"],
        id: "graupel_mixing_ratio",
        label: "Graupel mixing ratio",
        meaning: "native WRF graupel carrier when present; never zero-filled",
        expected_dimensions: XYZ,
        allowed_units: &["kg kg-1"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: false,
    },
    FieldSpec {
        names: &["QHAIL"],
        id: "hail_mixing_ratio",
        label: "Hail mixing ratio",
        meaning: "native WRF hail carrier when present; never zero-filled",
        expected_dimensions: XYZ,
        allowed_units: &["kg kg-1"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: false,
    },
    FieldSpec {
        names: &["PSFC"],
        id: "surface_pressure",
        label: "Surface pressure",
        meaning: "native WRF surface-pressure carrier",
        expected_dimensions: XY,
        allowed_units: &["Pa"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::RasterYx,
        required: false,
    },
    FieldSpec {
        names: &["REFL_10CM"],
        id: "wrf_10cm_reflectivity",
        label: "WRF 10 cm radar reflectivity",
        meaning: "native WRF model diagnostic; no viewer-side reflectivity derivation",
        expected_dimensions: XYZ,
        allowed_units: &["dBZ", "dbZ"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::VolumeZyx,
        required: false,
    },
    FieldSpec {
        names: &["REFD_MAX"],
        id: "wrf_max_reflectivity",
        label: "WRF maximum radar reflectivity",
        meaning: "native WRF model diagnostic; no viewer-side reflectivity derivation",
        expected_dimensions: XY,
        allowed_units: &["dBZ", "dbZ"],
        expected_stagger: "",
        geometry: GeoFieldGeometry::RasterYx,
        required: false,
    },
];

fn admit_metadata(source: &ArwenWrfoutSource) -> Result<ArwenWrfoutMetadata, String> {
    let cdf = &source.cdf;
    let nx = exact_dimension(cdf, "west_east")?;
    let ny = exact_dimension(cdf, "south_north")?;
    let nz = exact_dimension(cdf, "bottom_top")?;
    require_dimension_len(cdf, "west_east_stag", nx + 1)?;
    require_dimension_len(cdf, "south_north_stag", ny + 1)?;
    require_dimension_len(cdf, "bottom_top_stag", nz + 1)?;
    require_dimension_len(cdf, "DateStrLen", 19)?;
    let time_dimension = cdf.dimension("Time")?;
    if !time_dimension.unlimited || time_dimension.len != cdf.num_records {
        return Err("WRF Time must be the exact unlimited record dimension".to_owned());
    }

    let valid_time = cdf.read_text_record("Times", source.record)?;
    let start_time = cdf.global_text("SIMULATION_START_DATE")?.to_owned();
    let declared_start = cdf.global_text("START_DATE")?;
    if start_time != declared_start {
        return Err(format!(
            "WRF START_DATE {declared_start:?} differs from SIMULATION_START_DATE {start_time:?}"
        ));
    }
    let valid_seconds = parse_wrf_timestamp_seconds(&valid_time)?;
    let start_seconds = parse_wrf_timestamp_seconds(&start_time)?;
    let elapsed_seconds = valid_seconds
        .checked_sub(start_seconds)
        .ok_or_else(|| "WRF timestamp subtraction overflow".to_owned())?;
    if elapsed_seconds < 0 {
        return Err("WRF valid time precedes simulation start".to_owned());
    }
    let xtime_minutes = cdf.read_scalar_f32("XTIME", source.record)?;
    if !xtime_minutes.is_finite() || xtime_minutes < 0.0 {
        return Err("WRF XTIME is non-finite or negative".to_owned());
    }
    let xtime_seconds = f64::from(xtime_minutes) * 60.0;
    if (xtime_seconds - elapsed_seconds as f64).abs() > 0.01 {
        return Err(format!(
            "WRF Times/start elapsed {elapsed_seconds}s differs from XTIME {xtime_seconds}s"
        ));
    }
    let raw_step = cdf.read_scalar_i32("ITIMESTEP", source.record)?;
    let itimestep =
        u64::try_from(raw_step).map_err(|_| format!("WRF ITIMESTEP is negative: {raw_step}"))?;
    let dt_s = positive_global(cdf, "DT")?;
    if ((itimestep as f64) * dt_s - xtime_seconds).abs() > 0.05 {
        return Err(format!(
            "WRF ITIMESTEP*DT={} differs from XTIME seconds {xtime_seconds}",
            (itimestep as f64) * dt_s
        ));
    }

    let dx_m = positive_global(cdf, "DX")?;
    let dy_m = positive_global(cdf, "DY")?;
    let center_latitude_deg = finite_global(cdf, "CEN_LAT")?;
    let center_longitude_deg = finite_global(cdf, "CEN_LON")?;
    if !(-90.0..=90.0).contains(&center_latitude_deg)
        || !(-180.0..=180.0).contains(&center_longitude_deg)
    {
        return Err("WRF CEN_LAT/CEN_LON are outside Earth coordinate bounds".to_owned());
    }

    let map_proj = exact_global_i32(cdf, "MAP_PROJ")?;
    let map_proj_char = cdf.global_text("MAP_PROJ_CHAR")?.to_owned();
    if map_proj != 1 || map_proj_char != "Lambert Conformal" {
        return Err(format!(
            "only verified WRF Lambert conformal MAP_PROJ=1 is admitted, got {map_proj} {map_proj_char:?}"
        ));
    }
    let projection = WrfProjection {
        map_proj,
        map_proj_char,
        standard_longitude_deg: finite_global(cdf, "STAND_LON")?,
        true_latitude_1_deg: finite_global(cdf, "TRUELAT1")?,
        true_latitude_2_deg: finite_global(cdf, "TRUELAT2")?,
        mother_domain_center_latitude_deg: finite_global(cdf, "MOAD_CEN_LAT")?,
    };

    require_variable_layout(cdf, "XLAT", XY, &["degree_north"], "")?;
    require_variable_layout(cdf, "XLONG", XY, &["degree_east"], "")?;
    require_variable_layout(cdf, "PH", XYZI, &["m2 s-2"], "Z")?;
    require_variable_layout(cdf, "PHB", XYZI, &["m2 s-2"], "Z")?;
    let center_index = (ny / 2)
        .checked_mul(nx)
        .and_then(|value| value.checked_add(nx / 2))
        .ok_or_else(|| "WRF center index overflow".to_owned())?;
    let center_lat = cdf.read_f32_at("XLAT", source.record, center_index)?;
    let center_lon = cdf.read_f32_at("XLONG", source.record, center_index)?;
    if !center_lat.is_finite()
        || !center_lon.is_finite()
        || (f64::from(center_lat) - center_latitude_deg).abs() > 0.1
        || (f64::from(center_lon) - center_longitude_deg).abs() > 0.1
    {
        return Err(format!(
            "WRF mass-grid center ({center_lat},{center_lon}) is inconsistent with CEN_LAT/CEN_LON ({center_latitude_deg},{center_longitude_deg})"
        ));
    }
    let top_center_index = nz
        .checked_mul(ny)
        .and_then(|value| value.checked_mul(nx))
        .and_then(|value| value.checked_add(center_index))
        .ok_or_else(|| "WRF top-center index overflow".to_owned())?;
    let ph = cdf.read_f32_at("PH", source.record, top_center_index)?;
    let phb = cdf.read_f32_at("PHB", source.record, top_center_index)?;
    if !ph.is_finite() || !phb.is_finite() {
        return Err("WRF center top PH/PHB contain a non-finite value".to_owned());
    }
    let center_model_top_m = (f64::from(ph) + f64::from(phb)) / WRF_GRAVITY_M_S2;
    if !center_model_top_m.is_finite() || center_model_top_m <= 0.0 {
        return Err("WRF center model-top geopotential height is invalid".to_owned());
    }

    Ok(ArwenWrfoutMetadata {
        source_path: source.cdf.path.display().to_string(),
        source_file_bytes: source.cdf.file_len,
        source_sha256: source.file_sha256.clone(),
        container: source.cdf.version.name(),
        cdf_header_bytes: source.cdf.header_bytes,
        record: source.record,
        record_count: source.cdf.num_records,
        model_title: cdf.global_text("TITLE")?.to_owned(),
        valid_time,
        simulation_start_time: start_time,
        xtime_minutes,
        itimestep,
        dt_s,
        nx,
        ny,
        nz,
        dx_m,
        dy_m,
        center_latitude_deg,
        center_longitude_deg,
        center_model_top_m,
        projection,
    })
}

fn validate_and_scan(
    source: &ArwenWrfoutSource,
    spec: &FieldSpec,
    name: &str,
) -> Result<WrfFieldScan, String> {
    require_variable_layout(
        &source.cdf,
        name,
        spec.expected_dimensions,
        spec.allowed_units,
        spec.expected_stagger,
    )?;
    source.scan_field(name)
}

fn require_variable_layout(
    cdf: &CdfFile,
    name: &str,
    expected_dimensions: &[&str],
    allowed_units: &[&str],
    expected_stagger: &str,
) -> Result<(), String> {
    let variable = cdf.variable(name)?;
    if variable.data_type != CdfType::Float {
        return Err(format!("WRF field {name} is not exact f32 data"));
    }
    ensure_unpacked(variable)?;
    if variable.dimension_names() != expected_dimensions {
        return Err(format!(
            "WRF field {name} dimensions {:?} differ from exact {:?}",
            variable.dimension_names(),
            expected_dimensions
        ));
    }
    let units = variable
        .attribute("units")
        .and_then(CdfAttributeValue::text)
        .map(str::trim)
        .ok_or_else(|| format!("WRF field {name} has no units attribute"))?;
    if !allowed_units.contains(&units) {
        return Err(format!(
            "WRF field {name} units {units:?} are outside exact {:?}",
            allowed_units
        ));
    }
    let stagger = variable
        .attribute("stagger")
        .and_then(CdfAttributeValue::text)
        .map(str::trim)
        .ok_or_else(|| format!("WRF field {name} has no stagger attribute"))?;
    if stagger != expected_stagger {
        return Err(format!(
            "WRF field {name} stagger {stagger:?} differs from exact {expected_stagger:?}"
        ));
    }
    Ok(())
}

fn scan_exact_f32(cdf: &CdfFile, name: &str, record: usize) -> Result<WrfFieldScan, String> {
    let variable = cdf.variable(name)?;
    if variable.data_type != CdfType::Float {
        return Err(format!("WRF field {name} is not exact f32 data"));
    }
    ensure_unpacked(variable)?;
    let selection = cdf.selection(variable, record)?;
    if selection.count == 0 {
        return Err(format!("WRF field {name} is empty"));
    }
    let byte_count = selection
        .count
        .checked_mul(4)
        .ok_or_else(|| format!("WRF field {name} byte count overflow"))?;
    let end = selection
        .offset
        .checked_add(byte_count as u64)
        .ok_or_else(|| format!("WRF field {name} byte range overflow"))?;
    if end > cdf.file_len {
        return Err(format!("WRF field {name} exceeds source file"));
    }
    let sentinels = ["_FillValue", "missing_value"]
        .iter()
        .filter_map(|attribute| variable.attribute(attribute))
        .filter_map(CdfAttributeValue::scalar)
        .map(|value| (value as f32).to_bits())
        .collect::<Vec<_>>();
    let units = variable
        .attribute("units")
        .and_then(CdfAttributeValue::text)
        .unwrap_or("")
        .trim()
        .to_owned();
    let source_description = variable
        .attribute("description")
        .and_then(CdfAttributeValue::text)
        .unwrap_or("")
        .trim()
        .to_owned();
    let stagger = variable
        .attribute("stagger")
        .and_then(CdfAttributeValue::text)
        .unwrap_or("")
        .trim()
        .to_owned();

    let mut file =
        File::open(&cdf.path).map_err(|error| format!("open {}: {error}", cdf.path.display()))?;
    file.seek(SeekFrom::Start(selection.offset))
        .map_err(|error| format!("seek {} {name}: {error}", cdf.path.display()))?;
    let mut reader = BufReader::with_capacity(IO_CHUNK_BYTES, file);
    let mut remaining = byte_count;
    let mut buffer = vec![0_u8; IO_CHUNK_BYTES];
    let mut hasher = Sha256::new();
    let mut finite_count = 0_usize;
    let mut invalid_count = 0_usize;
    let mut finite_min = f32::INFINITY;
    let mut finite_max = f32::NEG_INFINITY;
    while remaining > 0 {
        let read_len = remaining.min(buffer.len());
        reader
            .read_exact(&mut buffer[..read_len])
            .map_err(|error| format!("read {} {name}: {error}", cdf.path.display()))?;
        for word in buffer[..read_len].chunks_exact(4) {
            let bits = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
            let value = f32::from_bits(bits);
            hasher.update(bits.to_le_bytes());
            if !value.is_finite() || sentinels.contains(&bits) {
                invalid_count += 1;
            } else {
                finite_count += 1;
                finite_min = finite_min.min(value);
                finite_max = finite_max.max(value);
            }
        }
        remaining -= read_len;
    }
    Ok(WrfFieldScan {
        variable: name.to_owned(),
        source_description,
        shape: variable.selected_shape(),
        units,
        stagger,
        decoded_f32_le_sha256: format!("{:x}", hasher.finalize()),
        value_count: selection.count,
        finite_count,
        invalid_count,
        finite_min: (finite_count > 0).then_some(finite_min),
        finite_max: (finite_count > 0).then_some(finite_max),
    })
}

fn available_descriptor(
    source: &ArwenWrfoutSource,
    spec: &FieldSpec,
    scan: &WrfFieldScan,
) -> GeoFieldDescriptor {
    GeoFieldDescriptor {
        id: spec.id.to_owned(),
        label: spec.label.to_owned(),
        units: scan.units.clone(),
        meaning: spec.meaning.to_owned(),
        geometry: spec.geometry,
        shape: scan.shape.clone(),
        values_sha256: scan.decoded_f32_le_sha256.clone(),
        admission: GeoFieldAdmission::Available,
        admission_reason:
            "exact classic-CDF f32 source carrier passed shape/units/stagger/finiteness admission"
                .to_owned(),
        provenance: format!(
            "wrfout_sha256={} record={} variable={} source_description={:?} decoded exact big-endian f32 to little-endian word hash; no remap/destagger/normalization",
            source.file_sha256, source.record, scan.variable, scan.source_description
        ),
    }
}

fn missing_descriptor(source: &ArwenWrfoutSource, spec: &FieldSpec) -> GeoFieldDescriptor {
    GeoFieldDescriptor {
        id: spec.id.to_owned(),
        label: spec.label.to_owned(),
        units: spec.allowed_units.first().copied().unwrap_or("").to_owned(),
        meaning: spec.meaning.to_owned(),
        geometry: spec.geometry,
        shape: expected_selected_shape(&source.cdf, spec.expected_dimensions).unwrap_or_default(),
        values_sha256: String::new(),
        admission: GeoFieldAdmission::Missing,
        admission_reason: format!(
            "exact WRFOUT has none of {}; no zero-fill or fallback is permitted",
            spec.names.join("/")
        ),
        provenance: format!(
            "wrfout_sha256={} record={}",
            source.file_sha256, source.record
        ),
    }
}

fn refused_descriptor(
    source: &ArwenWrfoutSource,
    spec: &FieldSpec,
    reason: String,
) -> GeoFieldDescriptor {
    GeoFieldDescriptor {
        id: spec.id.to_owned(),
        label: spec.label.to_owned(),
        units: spec.allowed_units.first().copied().unwrap_or("").to_owned(),
        meaning: spec.meaning.to_owned(),
        geometry: spec.geometry,
        shape: expected_selected_shape(&source.cdf, spec.expected_dimensions).unwrap_or_default(),
        values_sha256: String::new(),
        admission: GeoFieldAdmission::Refused,
        admission_reason: reason,
        provenance: format!(
            "wrfout_sha256={} record={}",
            source.file_sha256, source.record
        ),
    }
}

fn select_candidate<'a>(cdf: &CdfFile, names: &'a [&str]) -> Result<Option<&'a str>, String> {
    let found = names
        .iter()
        .copied()
        .filter(|name| cdf.variables.contains_key(*name))
        .collect::<Vec<_>>();
    match found.as_slice() {
        [] => Ok(None),
        [name] => Ok(Some(*name)),
        _ => Err(format!(
            "ambiguous WRF aliases are simultaneously present: {}",
            found.join(", ")
        )),
    }
}

fn expected_selected_shape(cdf: &CdfFile, dimensions: &[&str]) -> Result<Vec<usize>, String> {
    dimensions
        .iter()
        .skip(usize::from(dimensions.first() == Some(&"Time")))
        .map(|name| cdf.dimension(name).map(|dimension| dimension.len))
        .collect()
}

fn ensure_unpacked(variable: &CdfVariable) -> Result<(), String> {
    if variable.attribute("scale_factor").is_some() || variable.attribute("add_offset").is_some() {
        return Err(format!(
            "WRF field {} uses packed scale/offset; exact unpacking is not admitted",
            variable.name
        ));
    }
    Ok(())
}

fn exact_dimension(cdf: &CdfFile, name: &str) -> Result<usize, String> {
    let dimension = cdf.dimension(name)?;
    if dimension.unlimited || dimension.len == 0 {
        return Err(format!(
            "WRF dimension {name} is empty or unexpectedly unlimited"
        ));
    }
    Ok(dimension.len)
}

fn require_dimension_len(cdf: &CdfFile, name: &str, expected: usize) -> Result<(), String> {
    let actual = exact_dimension(cdf, name)?;
    if actual != expected {
        return Err(format!(
            "WRF dimension {name} length {actual} differs from exact {expected}"
        ));
    }
    Ok(())
}

fn finite_global(cdf: &CdfFile, name: &str) -> Result<f64, String> {
    let value = cdf.global_scalar(name)?;
    if !value.is_finite() {
        return Err(format!("WRF global attribute {name} is non-finite"));
    }
    Ok(value)
}

fn positive_global(cdf: &CdfFile, name: &str) -> Result<f64, String> {
    let value = finite_global(cdf, name)?;
    if value <= 0.0 {
        return Err(format!("WRF global attribute {name} is not positive"));
    }
    Ok(value)
}

fn exact_global_i32(cdf: &CdfFile, name: &str) -> Result<i32, String> {
    let value = finite_global(cdf, name)?;
    if value.fract() != 0.0 || value < i32::MIN as f64 || value > i32::MAX as f64 {
        return Err(format!("WRF global attribute {name} is not an exact i32"));
    }
    Ok(value as i32)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut reader = BufReader::with_capacity(IO_CHUNK_BYTES, file);
    let mut buffer = vec![0_u8; IO_CHUNK_BYTES];
    let mut hasher = Sha256::new();
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("hash {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} is not lowercase 64-hex SHA-256"));
    }
    Ok(())
}

fn parse_wrf_timestamp_seconds(value: &str) -> Result<i64, String> {
    let bytes = value.as_bytes();
    if bytes.len() != 19
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'_'
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return Err(format!(
            "WRF timestamp {value:?} is not YYYY-MM-DD_HH:MM:SS"
        ));
    }
    let year = parse_decimal(&bytes[0..4], "year")? as i32;
    let month = parse_decimal(&bytes[5..7], "month")?;
    let day = parse_decimal(&bytes[8..10], "day")?;
    let hour = parse_decimal(&bytes[11..13], "hour")?;
    let minute = parse_decimal(&bytes[14..16], "minute")?;
    let second = parse_decimal(&bytes[17..19], "second")?;
    if !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(format!(
            "WRF timestamp {value:?} has an out-of-range component"
        ));
    }
    let days = days_from_civil(year, month, day);
    Ok(days * 86_400 + (hour * 3_600 + minute * 60 + second) as i64)
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year % 400 == 0 || (year % 4 == 0 && year % 100 != 0) => 29,
        2 => 28,
        _ => 0,
    }
}

fn parse_decimal(bytes: &[u8], label: &str) -> Result<u32, String> {
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return Err(format!("WRF timestamp {label} is not decimal"));
    }
    bytes.iter().try_fold(0_u32, |value, byte| {
        value
            .checked_mul(10)
            .and_then(|value| value.checked_add(u32::from(*byte - b'0')))
            .ok_or_else(|| format!("WRF timestamp {label} overflow"))
    })
}

// Howard Hinnant's public-domain civil-date conversion, shifted to an
// arbitrary epoch because only exact differences are used.
fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let adjusted_year = year - i32::from(month <= 2);
    let era = if adjusted_year >= 0 {
        adjusted_year
    } else {
        adjusted_year - 399
    } / 400;
    let year_of_era = (adjusted_year - era * 400) as u32;
    let adjusted_month = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    i64::from(era) * 146_097 + i64::from(day_of_era)
}

#[derive(Debug)]
enum HeaderError {
    Truncated,
    Invalid(String),
    Unsupported(String),
}

type HeaderResult<T> = Result<T, HeaderError>;

struct ParsedHeader {
    version: CdfVersion,
    header_bytes: usize,
    num_records: usize,
    dimensions: Vec<CdfDimension>,
    global_attributes: BTreeMap<String, CdfAttributeValue>,
    variables: BTreeMap<String, CdfVariable>,
}

fn parse_header(bytes: &[u8]) -> HeaderResult<ParsedHeader> {
    if bytes.starts_with(b"\x89HDF\r\n\x1a\n") {
        return Err(HeaderError::Unsupported(
            "HDF5-backed NetCDF-4 is not admitted by the classic-CDF reader".to_owned(),
        ));
    }
    if bytes.len() < 8 || &bytes[0..3] != b"CDF" {
        return Err(HeaderError::Invalid("missing CDF magic".to_owned()));
    }
    let version = match bytes[3] {
        1 => CdfVersion::Classic,
        2 => CdfVersion::Offset64,
        5 => {
            return Err(HeaderError::Unsupported(
                "CDF-5 is not admitted by this verified reader".to_owned(),
            ));
        }
        other => {
            return Err(HeaderError::Invalid(format!(
                "unknown CDF version byte {other}"
            )));
        }
    };
    let mut reader = HeaderReader::new(bytes, 4);
    let num_records = usize_from_u32(reader.u32()?, "record count")?;
    let dimensions = reader.dimension_list(num_records)?;
    let global_attributes = reader.attribute_list()?;
    let variables = reader.variable_list(version, &dimensions)?;
    Ok(ParsedHeader {
        version,
        header_bytes: reader.position,
        num_records,
        dimensions,
        global_attributes,
        variables,
    })
}

struct HeaderReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> HeaderReader<'a> {
    const fn new(bytes: &'a [u8], position: usize) -> Self {
        Self { bytes, position }
    }

    fn take(&mut self, len: usize) -> HeaderResult<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| HeaderError::Invalid("header offset overflow".to_owned()))?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or(HeaderError::Truncated)?;
        self.position = end;
        Ok(slice)
    }

    fn u32(&mut self) -> HeaderResult<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> HeaderResult<u64> {
        let bytes = self.take(8)?;
        Ok(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn name(&mut self) -> HeaderResult<String> {
        let len = usize_from_u32(self.u32()?, "name length")?;
        let raw = self.take(len)?;
        let padding = align4(len) - len;
        let _ = self.take(padding)?;
        String::from_utf8(raw.to_vec())
            .map_err(|error| HeaderError::Invalid(format!("non-UTF8 CDF name: {error}")))
    }

    fn dimension_list(&mut self, num_records: usize) -> HeaderResult<Vec<CdfDimension>> {
        let count = self.list_count(NC_DIMENSION, "dimension")?;
        let mut dimensions = Vec::with_capacity(count);
        for _ in 0..count {
            let name = self.name()?;
            let raw_len = usize_from_u32(self.u32()?, "dimension length")?;
            let unlimited = raw_len == 0;
            dimensions.push(CdfDimension {
                name,
                len: if unlimited { num_records } else { raw_len },
                unlimited,
            });
        }
        Ok(dimensions)
    }

    fn attribute_list(&mut self) -> HeaderResult<BTreeMap<String, CdfAttributeValue>> {
        let count = self.list_count(NC_ATTRIBUTE, "attribute")?;
        let mut attributes = BTreeMap::new();
        for _ in 0..count {
            let name = self.name()?;
            let data_type = CdfType::from_code(self.u32()?)?;
            let count = usize_from_u32(self.u32()?, "attribute element count")?;
            let byte_len = count
                .checked_mul(data_type.size())
                .ok_or_else(|| HeaderError::Invalid("attribute byte count overflow".to_owned()))?;
            let raw = self.take(byte_len)?;
            let padding = align4(byte_len) - byte_len;
            let _ = self.take(padding)?;
            let value = if data_type == CdfType::Char {
                let end = raw.iter().position(|byte| *byte == 0).unwrap_or(raw.len());
                CdfAttributeValue::Text(String::from_utf8_lossy(&raw[..end]).to_string())
            } else {
                CdfAttributeValue::Numeric(decode_attribute_numeric(raw, data_type)?)
            };
            attributes.insert(name, value);
        }
        Ok(attributes)
    }

    fn variable_list(
        &mut self,
        version: CdfVersion,
        dimensions: &[CdfDimension],
    ) -> HeaderResult<BTreeMap<String, CdfVariable>> {
        let count = self.list_count(NC_VARIABLE, "variable")?;
        let mut variables = BTreeMap::new();
        for _ in 0..count {
            let name = self.name()?;
            let dimension_count = usize_from_u32(self.u32()?, "variable rank")?;
            let mut variable_dimensions = Vec::with_capacity(dimension_count);
            for _ in 0..dimension_count {
                let id = usize_from_u32(self.u32()?, "dimension id")?;
                let dimension = dimensions.get(id).ok_or_else(|| {
                    HeaderError::Invalid(format!(
                        "variable {name} refers to missing dimension id {id}"
                    ))
                })?;
                variable_dimensions.push(dimension.clone());
            }
            let attributes = self.attribute_list()?;
            let data_type = CdfType::from_code(self.u32()?)?;
            let _vsize = self.u32()?;
            let begin = match version {
                CdfVersion::Classic => u64::from(self.u32()?),
                CdfVersion::Offset64 => self.u64()?,
            };
            let is_record = variable_dimensions
                .first()
                .is_some_and(|dimension| dimension.unlimited);
            let values_per_record = variable_dimensions
                .iter()
                .skip(usize::from(is_record))
                .try_fold(1_usize, |total, dimension| {
                    total.checked_mul(dimension.len).ok_or_else(|| {
                        HeaderError::Invalid(format!("element count overflow for variable {name}"))
                    })
                })?;
            let raw_bytes = values_per_record
                .checked_mul(data_type.size())
                .ok_or_else(|| {
                    HeaderError::Invalid(format!("byte count overflow for variable {name}"))
                })?;
            let bytes_per_record = align4(raw_bytes) as u64;
            variables.insert(
                name.clone(),
                CdfVariable {
                    name,
                    dimensions: variable_dimensions,
                    data_type,
                    attributes,
                    begin,
                    is_record,
                    values_per_record,
                    bytes_per_record,
                },
            );
        }
        Ok(variables)
    }

    fn list_count(&mut self, expected_tag: u32, label: &str) -> HeaderResult<usize> {
        let tag = self.u32()?;
        let count = self.u32()?;
        if tag == 0 {
            if count != 0 {
                return Err(HeaderError::Invalid(format!(
                    "absent {label} list has non-zero count {count}"
                )));
            }
            return Ok(0);
        }
        if tag != expected_tag {
            return Err(HeaderError::Invalid(format!(
                "expected {label} list tag {expected_tag}, got {tag}"
            )));
        }
        usize_from_u32(count, "list count")
    }
}

fn decode_attribute_numeric(raw: &[u8], data_type: CdfType) -> HeaderResult<Vec<f64>> {
    let width = data_type.size();
    if !raw.len().is_multiple_of(width) {
        return Err(HeaderError::Invalid(
            "numeric attribute byte length is misaligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(raw.len() / width);
    match data_type {
        CdfType::Byte => values.extend(raw.iter().map(|value| f64::from(*value as i8))),
        CdfType::Char => {
            return Err(HeaderError::Invalid(
                "character data passed to numeric attribute decoder".to_owned(),
            ));
        }
        CdfType::Short => {
            for bytes in raw.chunks_exact(2) {
                values.push(f64::from(i16::from_be_bytes([bytes[0], bytes[1]])));
            }
        }
        CdfType::Int => {
            for bytes in raw.chunks_exact(4) {
                values.push(f64::from(i32::from_be_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3],
                ])));
            }
        }
        CdfType::Float => {
            for bytes in raw.chunks_exact(4) {
                values.push(f64::from(f32::from_bits(u32::from_be_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3],
                ]))));
            }
        }
        CdfType::Double => {
            for bytes in raw.chunks_exact(8) {
                values.push(f64::from_bits(u64::from_be_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ])));
            }
        }
    }
    Ok(values)
}

fn usize_from_u32(value: u32, label: &str) -> HeaderResult<usize> {
    usize::try_from(value)
        .map_err(|_| HeaderError::Invalid(format!("{label} does not fit usize: {value}")))
}

const fn align4(value: usize) -> usize {
    (value + 3) & !3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_wrf_timestamp_metadata_has_exact_four_hour_cadence() {
        let start = parse_wrf_timestamp_seconds("1974-04-03_12:00:00").unwrap();
        let valid = parse_wrf_timestamp_seconds("1974-04-03_16:00:00").unwrap();
        assert_eq!(valid - start, 14_400);
        assert!(parse_wrf_timestamp_seconds("1974-02-31_00:00:00").is_err());
    }

    #[test]
    fn hdf5_and_cdf5_are_explicitly_refused() {
        assert!(matches!(
            parse_header(b"\x89HDF\r\n\x1a\n"),
            Err(HeaderError::Unsupported(_))
        ));
        assert!(matches!(
            parse_header(b"CDF\x05\0\0\0\0"),
            Err(HeaderError::Unsupported(_))
        ));
    }

    #[test]
    fn source_hash_and_alias_inputs_are_strict() {
        assert!(require_sha256(&"a".repeat(64), "sha").is_ok());
        assert!(require_sha256(&"A".repeat(64), "sha").is_err());
        assert!(require_sha256("abc", "sha").is_err());
    }

    #[test]
    fn field_inventory_never_labels_a_vortex_or_derives_zeta() {
        let ids = FIELD_SPECS.iter().map(|spec| spec.id).collect::<Vec<_>>();
        assert!(!ids.contains(&"vortex"));
        assert!(!ids.contains(&"vertical_vorticity"));
        assert!(ids.contains(&"u_staggered"));
        assert!(ids.contains(&"v_staggered"));
        assert!(ids.contains(&"w_staggered"));
        for spec in FIELD_SPECS {
            arwen_residency_kind(spec.id).unwrap_or_else(|error| panic!("{}: {error}", spec.id));
        }
    }
}
