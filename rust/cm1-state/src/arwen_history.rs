//! One time record of a classic-format WRF/ArWen history file, decoded to raw
//! little-endian binary32 words for the ArWen -> GM1 chain.
//!
//! ArWen's default history writer is the Rust classic writer, CDF-2
//! (`NETCDF3_64BIT_OFFSET`, `gpuwm/io/wrfout.py`), so the container decoded
//! here is the netCDF classic format of the netCDF User's Guide, "The NetCDF
//! Classic Format Specification": CDF-1 (32-bit `begin`) and CDF-2 (64-bit
//! `begin`).  Stock WRF writes netCDF-4/HDF5 by default; that container is
//! refused by name with the lossless conversion that turns it into CDF-2
//! (`nccopy -k nc6`).  CDF-5 is refused because its counts are 64-bit and no
//! ArWen writer emits it.
//!
//! What this module does and does not do:
//!
//! * it selects the named variables of one record (record variables) and
//!   every named static variable, converts big-endian binary32 to
//!   little-endian binary32 word for word, and hashes the words it writes;
//! * it transcribes the global attributes as JSON so the Python admission can
//!   check projection, spacing, microphysics option and `USE_THETA_M`;
//! * it does **not** destagger, crop, interpolate, convert units or derive any
//!   field.  Every scientific decision about these words is made, and
//!   recorded, by `cm1core.arwen_chain`.
//!
//! Output directory (created new; an existing one is refused):
//!
//! ```text
//! fields.f32le   contiguous little-endian binary32 arrays, C order, the
//!                record dimension removed
//! frame.json     schema gm1.arwen-history-frame/v1: source identity, the
//!                record's Times string, dimensions, global attributes and
//!                one row per field (dims, shape, offset, nbytes, sha256)
//! ```

use crate::sha256::{hex, Sha256};
use crate::{Result, StateError};
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const GM1_ARWEN_HISTORY_FRAME_SCHEMA: &str = "gm1.arwen-history-frame/v1";
pub const GM1_ARWEN_HISTORY_EXTRACT_ABI: &str = "gm1.arwen-history-extract/v1";

/// The history variables the chain may read.  A variable absent from the
/// file is skipped here and named by the Python admission if it is needed.
pub const CHAIN_VARIABLES: &[&str] = &[
    "U", "V", "W", "T", "THM", "PH", "PHB", "P", "PB", "HGT", "MAPFAC_M", "XLAT", "XLONG",
    "PSFC", "TH2", "T2", "Q2", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE",
    "QNSNOW", "QNRAIN", "QNGRAUPEL", "QNCLOUD", "QNWFA", "QNIFA", "QNCCN",
];

const NC_DIMENSION: u32 = 0x0A;
const NC_VARIABLE: u32 = 0x0B;
const NC_ATTRIBUTE: u32 = 0x0C;
const NC_BYTE: u32 = 1;
const NC_CHAR: u32 = 2;
const NC_SHORT: u32 = 3;
const NC_INT: u32 = 4;
const NC_FLOAT: u32 = 5;
const NC_DOUBLE: u32 = 6;
const STREAMING_NUMRECS: u32 = u32::MAX;
const MAX_LIST: u32 = 1 << 20;
const MAX_NAME: u32 = 1 << 16;
const MAX_ATTRIBUTE_BYTES: u64 = 1 << 24;
const HDF5_MAGIC: &[u8; 8] = b"\x89HDF\r\n\x1a\n";

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
    Text(String),
    Numbers(Vec<f64>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dimension {
    pub name: String,
    /// Zero for the record dimension, as the classic header stores it.
    pub length: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Variable {
    pub name: String,
    pub dimids: Vec<usize>,
    pub attributes: Vec<(String, AttributeValue)>,
    pub nc_type: u32,
    pub vsize: u64,
    pub begin: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClassicHeader {
    pub version: u8,
    pub numrecs: u32,
    pub dimensions: Vec<Dimension>,
    pub attributes: Vec<(String, AttributeValue)>,
    pub variables: Vec<Variable>,
}

impl ClassicHeader {
    fn record_dimension(&self) -> Option<usize> {
        self.dimensions.iter().position(|d| d.length == 0)
    }

    fn is_record_variable(&self, variable: &Variable) -> bool {
        matches!((self.record_dimension(), variable.dimids.first()), (Some(r), Some(&d)) if r == d)
    }

    fn variable(&self, name: &str) -> Option<&Variable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// The per-record shape: every dimension but the record one.
    fn shape(&self, variable: &Variable) -> Result<Vec<(String, u64)>> {
        let skip = usize::from(self.is_record_variable(variable));
        variable.dimids[skip..]
            .iter()
            .map(|&id| {
                let d = self
                    .dimensions
                    .get(id)
                    .ok_or_else(|| invalid(format!("{} names dimension id {id}", variable.name)))?;
                if d.length == 0 {
                    return Err(invalid(format!(
                        "{} uses the record dimension other than first",
                        variable.name
                    )));
                }
                Ok((d.name.clone(), d.length))
            })
            .collect()
    }

    /// Bytes of one record: the sum of the record variables' vsize, except
    /// that a file with exactly one record variable stores it unpadded.
    fn record_size(&self) -> Result<u64> {
        let records: Vec<&Variable> =
            self.variables.iter().filter(|v| self.is_record_variable(v)).collect();
        if records.len() == 1 {
            let v = records[0];
            let count: u64 = self.shape(v)?.iter().map(|(_, n)| *n).product();
            return Ok(count * type_size(v.nc_type)?);
        }
        Ok(records.iter().map(|v| v.vsize).sum())
    }
}

fn type_size(nc_type: u32) -> Result<u64> {
    match nc_type {
        NC_BYTE | NC_CHAR => Ok(1),
        NC_SHORT => Ok(2),
        NC_INT | NC_FLOAT => Ok(4),
        NC_DOUBLE => Ok(8),
        other => Err(invalid(format!("classic nc_type {other} is not one of 1..6"))),
    }
}

fn padded(n: u64) -> u64 {
    (n + 3) & !3
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut word = [0_u8; 4];
    reader.read_exact(&mut word)?;
    Ok(u32::from_be_bytes(word))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut word = [0_u8; 8];
    reader.read_exact(&mut word)?;
    Ok(u64::from_be_bytes(word))
}

fn skip_padding<R: Read>(reader: &mut R, length: u64) -> Result<()> {
    let mut pad = [0_u8; 3];
    let extra = (padded(length) - length) as usize;
    reader.read_exact(&mut pad[..extra])?;
    Ok(())
}

fn read_name<R: Read>(reader: &mut R) -> Result<String> {
    let length = read_u32(reader)?;
    if length == 0 || length > MAX_NAME {
        return Err(invalid(format!("classic header name length {length} is out of range")));
    }
    let mut bytes = vec![0_u8; length as usize];
    reader.read_exact(&mut bytes)?;
    skip_padding(reader, u64::from(length))?;
    String::from_utf8(bytes).map_err(|_| invalid("classic header name is not UTF-8"))
}

fn read_list_header<R: Read>(reader: &mut R, tag: u32, label: &str) -> Result<u32> {
    let found = read_u32(reader)?;
    let count = read_u32(reader)?;
    if found == 0 && count == 0 {
        return Ok(0);
    }
    if found != tag {
        return Err(invalid(format!("classic {label} list tag is {found:#x}, not {tag:#x}")));
    }
    if count > MAX_LIST {
        return Err(invalid(format!("classic {label} list holds {count} entries")));
    }
    Ok(count)
}

fn read_attributes<R: Read>(reader: &mut R) -> Result<Vec<(String, AttributeValue)>> {
    let count = read_list_header(reader, NC_ATTRIBUTE, "attribute")?;
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let name = read_name(reader)?;
        let nc_type = read_u32(reader)?;
        let nelems = u64::from(read_u32(reader)?);
        let bytes = nelems * type_size(nc_type)?;
        if bytes > MAX_ATTRIBUTE_BYTES {
            return Err(invalid(format!("attribute {name} holds {bytes} bytes")));
        }
        let mut raw = vec![0_u8; bytes as usize];
        reader.read_exact(&mut raw)?;
        skip_padding(reader, bytes)?;
        let value = match nc_type {
            NC_CHAR => {
                let text = String::from_utf8_lossy(&raw);
                AttributeValue::Text(text.trim_end_matches('\0').to_string())
            }
            NC_BYTE => AttributeValue::Numbers(raw.iter().map(|&b| f64::from(b as i8)).collect()),
            NC_SHORT => AttributeValue::Numbers(
                raw.chunks_exact(2)
                    .map(|c| f64::from(i16::from_be_bytes([c[0], c[1]])))
                    .collect(),
            ),
            NC_INT => AttributeValue::Numbers(
                raw.chunks_exact(4)
                    .map(|c| f64::from(i32::from_be_bytes([c[0], c[1], c[2], c[3]])))
                    .collect(),
            ),
            NC_FLOAT => AttributeValue::Numbers(
                raw.chunks_exact(4)
                    .map(|c| f64::from(f32::from_be_bytes([c[0], c[1], c[2], c[3]])))
                    .collect(),
            ),
            _ => AttributeValue::Numbers(
                raw.chunks_exact(8)
                    .map(|c| f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            ),
        };
        out.push((name, value));
    }
    Ok(out)
}

/// Parse a classic header.  The reader must be at byte 0.
pub fn read_classic_header<R: Read>(reader: &mut R) -> Result<ClassicHeader> {
    let mut magic = [0_u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic == b"\x89HDF" {
        let mut rest = [0_u8; 4];
        reader.read_exact(&mut rest)?;
        let mut whole = [0_u8; 8];
        whole[..4].copy_from_slice(&magic);
        whole[4..].copy_from_slice(&rest);
        if &whole == HDF5_MAGIC {
            return Err(invalid(
                "this history file is netCDF-4/HDF5, which this decoder does not read (it would \
                 need an HDF5 implementation this dependency-free crate does not carry); convert it \
                 losslessly to the classic 64-bit-offset container with `nccopy -k nc6 IN OUT` or \
                 write ArWen history with its default Rust classic writer",
            ));
        }
    }
    if &magic[..3] != b"CDF" {
        return Err(invalid("the file is not a netCDF classic container (no CDF magic)"));
    }
    let version = magic[3];
    if version == 5 {
        return Err(invalid(
            "CDF-5 (64-bit data) is not decoded: its counts are 64-bit and no ArWen writer emits \
             it; convert with `nccopy -k nc6 IN OUT`",
        ));
    }
    if version != 1 && version != 2 {
        return Err(invalid(format!("classic version byte {version} is neither 1 nor 2")));
    }
    let numrecs = read_u32(reader)?;
    let dimension_count = read_list_header(reader, NC_DIMENSION, "dimension")?;
    let mut dimensions = Vec::with_capacity(dimension_count as usize);
    for _ in 0..dimension_count {
        let name = read_name(reader)?;
        let length = u64::from(read_u32(reader)?);
        dimensions.push(Dimension { name, length });
    }
    let attributes = read_attributes(reader)?;
    let variable_count = read_list_header(reader, NC_VARIABLE, "variable")?;
    let mut variables = Vec::with_capacity(variable_count as usize);
    for _ in 0..variable_count {
        let name = read_name(reader)?;
        let rank = read_u32(reader)?;
        if rank > 64 {
            return Err(invalid(format!("variable {name} has rank {rank}")));
        }
        let mut dimids = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            let id = read_u32(reader)? as usize;
            if id >= dimensions.len() {
                return Err(invalid(format!("variable {name} names dimension id {id}")));
            }
            dimids.push(id);
        }
        let attributes = read_attributes(reader)?;
        let nc_type = read_u32(reader)?;
        type_size(nc_type)?;
        let vsize = u64::from(read_u32(reader)?);
        let begin = if version == 1 { u64::from(read_u32(reader)?) } else { read_u64(reader)? };
        variables.push(Variable { name, dimids, attributes, nc_type, vsize, begin });
    }
    Ok(ClassicHeader { version, numrecs, dimensions, attributes, variables })
}

fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        // Rust's shortest round-trip spelling; integers print without a
        // fraction, which is valid JSON.
        format!("{value:?}").trim_end_matches(".0").to_string()
    } else {
        "null".to_string()
    }
}

fn json_attribute(value: &AttributeValue) -> String {
    match value {
        AttributeValue::Text(text) => json_string(text),
        AttributeValue::Numbers(values) if values.len() == 1 => json_number(values[0]),
        AttributeValue::Numbers(values) => format!(
            "[{}]",
            values.iter().map(|v| json_number(*v)).collect::<Vec<_>>().join(",")
        ),
    }
}

fn text_attribute(variable: &Variable, name: &str) -> Option<String> {
    variable.attributes.iter().find_map(|(key, value)| match value {
        AttributeValue::Text(text) if key == name => Some(text.clone()),
        _ => None,
    })
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut reader = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut hasher = Sha256::default();
    let mut buffer = vec![0_u8; 1 << 20];
    let mut total = 0_u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        total += read as u64;
    }
    Ok((hex(&hasher.finalize()), total))
}

/// Summary of one extraction, echoed by the binary.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtractedFrame {
    pub times: String,
    pub fields: Vec<String>,
    pub skipped: Vec<String>,
    pub header_json: String,
}

/// Decode `record` of `input` into the new directory `output`.
///
/// `wanted` names the variables to extract; absent ones are listed under
/// `absent` in `frame.json`.  Only binary32 (`NC_FLOAT`) variables are
/// extracted: every WRF state and diagnostic field the chain reads is one,
/// and silently narrowing a double would hand the chain words the model
/// never held.
pub fn extract_history_record(
    input: &Path,
    record: u32,
    output: &Path,
    wanted: &[&str],
) -> Result<ExtractedFrame> {
    let (source_sha256, source_bytes) = sha256_file(input)?;
    let mut file = BufReader::with_capacity(1 << 20, File::open(input)?);
    let header = read_classic_header(&mut file)?;
    let record_size = header.record_size()?;
    let numrecs = if header.numrecs == STREAMING_NUMRECS {
        // Streaming files store 0xFFFFFFFF; the count is what the data holds.
        let first_record = header
            .variables
            .iter()
            .filter(|v| header.is_record_variable(v))
            .map(|v| v.begin)
            .min()
            .unwrap_or(source_bytes);
        if record_size == 0 { 0 } else { ((source_bytes - first_record) / record_size) as u32 }
    } else {
        header.numrecs
    };
    if record >= numrecs {
        return Err(invalid(format!("record {record} requested; the file holds {numrecs}")));
    }
    let times = match header.variable("Times") {
        Some(v) if v.nc_type == NC_CHAR && header.is_record_variable(v) => {
            let count: u64 = header.shape(v)?.iter().map(|(_, n)| *n).product();
            let mut bytes = vec![0_u8; count as usize];
            file.seek(SeekFrom::Start(v.begin + u64::from(record) * record_size))?;
            file.read_exact(&mut bytes)?;
            String::from_utf8_lossy(&bytes).trim_end_matches('\0').to_string()
        }
        _ => return Err(invalid("the history file has no record CHAR variable Times")),
    };

    fs::create_dir(output).map_err(|error| {
        invalid(format!(
            "{} could not be created as a new directory ({error}); an extraction never writes \
             into an existing one, so two sources can never be mixed in one frame",
            output.display()
        ))
    })?;
    let mut arena = File::create(output.join("fields.f32le"))?;
    let mut offset = 0_u64;
    let mut rows = Vec::new();
    let mut fields = Vec::new();
    let mut skipped = Vec::new();
    for &name in wanted {
        let Some(variable) = header.variable(name) else {
            skipped.push(name.to_string());
            continue;
        };
        if variable.nc_type != NC_FLOAT {
            return Err(invalid(format!(
                "{name} has nc_type {}, not binary32; it would be narrowed or reinterpreted",
                variable.nc_type
            )));
        }
        let shape = header.shape(variable)?;
        let count: u64 = shape.iter().map(|(_, n)| *n).product();
        let nbytes = count * 4;
        let start = if header.is_record_variable(variable) {
            variable.begin + u64::from(record) * record_size
        } else {
            variable.begin
        };
        let mut words = vec![0_u8; nbytes as usize];
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut words)?;
        for chunk in words.chunks_exact_mut(4) {
            let value = f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        let mut hasher = Sha256::default();
        hasher.update(&words);
        let digest = hex(&hasher.finalize());
        arena.write_all(&words)?;
        rows.push(format!(
            "{{\"dims\":[{}],\"name\":{},\"nbytes\":{nbytes},\"offset\":{offset},\"sha256\":{},\"shape\":[{}],\"stagger\":{},\"units\":{}}}",
            shape.iter().map(|(d, _)| json_string(d)).collect::<Vec<_>>().join(","),
            json_string(name),
            json_string(&digest),
            shape.iter().map(|(_, n)| n.to_string()).collect::<Vec<_>>().join(","),
            json_string(&text_attribute(variable, "stagger").unwrap_or_default()),
            json_string(&text_attribute(variable, "units").unwrap_or_default()),
        ));
        fields.push(name.to_string());
        offset += nbytes;
    }
    arena.flush()?;
    drop(arena);

    let dimensions = header
        .dimensions
        .iter()
        .map(|d| {
            let length = if d.length == 0 { u64::from(numrecs) } else { d.length };
            format!("{}:{length}", json_string(&d.name))
        })
        .collect::<Vec<_>>()
        .join(",");
    let attributes = header
        .attributes
        .iter()
        .map(|(k, v)| format!("{}:{}", json_string(k), json_attribute(v)))
        .collect::<Vec<_>>()
        .join(",");
    let header_json = format!(
        "{{\"absent\":[{}],\"attributes\":{{{attributes}}},\"dimensions\":{{{dimensions}}},\"fields\":[{}],\"schema\":{},\"source\":{{\"bytes\":{source_bytes},\"format\":{},\"numrecs\":{numrecs},\"path\":{},\"record\":{record},\"sha256\":{}}},\"times\":{}}}",
        skipped.iter().map(|s| json_string(s)).collect::<Vec<_>>().join(","),
        rows.join(","),
        json_string(GM1_ARWEN_HISTORY_FRAME_SCHEMA),
        json_string(if header.version == 1 { "CDF-1" } else { "CDF-2" }),
        json_string(&input.to_string_lossy()),
        json_string(&source_sha256),
        json_string(&times),
    );
    fs::write(output.join("frame.json"), header_json.as_bytes())?;
    Ok(ExtractedFrame { times, fields, skipped, header_json })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn name(out: &mut Vec<u8>, text: &str) {
        out.extend_from_slice(&(text.len() as u32).to_be_bytes());
        out.extend_from_slice(text.as_bytes());
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }

    enum Att<'a> {
        Text(&'a str),
        Int(i32),
        Float(f32),
    }

    fn attributes(out: &mut Vec<u8>, atts: &[(&str, Att)]) {
        if atts.is_empty() {
            out.extend_from_slice(&[0; 8]);
            return;
        }
        out.extend_from_slice(&NC_ATTRIBUTE.to_be_bytes());
        out.extend_from_slice(&(atts.len() as u32).to_be_bytes());
        for (key, value) in atts {
            name(out, key);
            match value {
                Att::Text(t) => {
                    out.extend_from_slice(&NC_CHAR.to_be_bytes());
                    out.extend_from_slice(&(t.len() as u32).to_be_bytes());
                    out.extend_from_slice(t.as_bytes());
                    while out.len() % 4 != 0 {
                        out.push(0);
                    }
                }
                Att::Int(v) => {
                    out.extend_from_slice(&NC_INT.to_be_bytes());
                    out.extend_from_slice(&1_u32.to_be_bytes());
                    out.extend_from_slice(&v.to_be_bytes());
                }
                Att::Float(v) => {
                    out.extend_from_slice(&NC_FLOAT.to_be_bytes());
                    out.extend_from_slice(&1_u32.to_be_bytes());
                    out.extend_from_slice(&v.to_be_bytes());
                }
            }
        }
    }

    /// A two-record classic file: Times, record T (bt,sn,we), record U
    /// (bt,sn,we_stag) and static ZS (bt).  Values are `record*100 + index`.
    pub(crate) fn synthetic_file(version: u8) -> Vec<u8> {
        let dims = [("Time", 0_u32), ("DateStrLen", 19), ("west_east", 2), ("south_north", 1), ("bottom_top", 1), ("west_east_stag", 3)];
        // (name, dimids, nc_type, per-record element count, stagger)
        let vars: [(&str, Vec<u32>, u32, u64, &str); 4] = [
            ("Times", vec![0, 1], NC_CHAR, 19, ""),
            ("T", vec![0, 4, 3, 2], NC_FLOAT, 2, ""),
            ("U", vec![0, 4, 3, 5], NC_FLOAT, 3, "X"),
            ("ZS", vec![4], NC_FLOAT, 1, ""),
        ];
        let build = |begins: &[u64]| -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(b"CDF");
            out.push(version);
            out.extend_from_slice(&2_u32.to_be_bytes());
            out.extend_from_slice(&NC_DIMENSION.to_be_bytes());
            out.extend_from_slice(&(dims.len() as u32).to_be_bytes());
            for (n, l) in dims {
                name(&mut out, n);
                out.extend_from_slice(&l.to_be_bytes());
            }
            attributes(&mut out, &[("DX", Att::Float(3000.0)), ("MAP_PROJ", Att::Int(1)), ("TITLE", Att::Text("ArWen synthetic"))]);
            out.extend_from_slice(&NC_VARIABLE.to_be_bytes());
            out.extend_from_slice(&(vars.len() as u32).to_be_bytes());
            for (index, (n, ids, t, count, stagger)) in vars.iter().enumerate() {
                name(&mut out, n);
                out.extend_from_slice(&(ids.len() as u32).to_be_bytes());
                for id in ids {
                    out.extend_from_slice(&id.to_be_bytes());
                }
                if stagger.is_empty() {
                    attributes(&mut out, &[]);
                } else {
                    attributes(&mut out, &[("stagger", Att::Text(stagger))]);
                }
                out.extend_from_slice(&t.to_be_bytes());
                let vsize = padded(count * type_size(*t).unwrap());
                out.extend_from_slice(&(vsize as u32).to_be_bytes());
                if version == 1 {
                    out.extend_from_slice(&(begins[index] as u32).to_be_bytes());
                } else {
                    out.extend_from_slice(&begins[index].to_be_bytes());
                }
            }
            out
        };
        let header_len = build(&[0, 0, 0, 0]).len() as u64;
        // Static data first, then records (the classic layout).
        let zs_begin = header_len;
        let record_begin = zs_begin + 4;
        let times_vsize = padded(19);
        let t_vsize = 8;
        let begins = [record_begin, record_begin + times_vsize, record_begin + times_vsize + t_vsize, zs_begin];
        let mut out = build(&begins);
        out.extend_from_slice(&7.5_f32.to_be_bytes());
        for record in 0..2_u32 {
            let stamp = format!("2026-01-01_00:0{record}:00");
            out.extend_from_slice(stamp.as_bytes());
            while out.len() % 4 != 0 {
                out.push(0);
            }
            for i in 0..2 {
                out.extend_from_slice(&((record * 100 + i) as f32).to_be_bytes());
            }
            for i in 0..3 {
                out.extend_from_slice(&((record * 100 + 10 + i) as f32).to_be_bytes());
            }
        }
        out
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("gm1-arwen-history-{tag}-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn le_words(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn both_classic_versions_decode_record_and_static_words() {
        for version in [1_u8, 2] {
            let root = scratch(&format!("v{version}"));
            let input = root.join("wrfout");
            fs::write(&input, synthetic_file(version)).unwrap();
            let out = root.join("frame");
            let frame = extract_history_record(&input, 1, &out, &["T", "U", "ZS", "QVAPOR"]).unwrap();
            assert_eq!(frame.times, "2026-01-01_00:01:00");
            assert_eq!(frame.fields, vec!["T", "U", "ZS"]);
            assert_eq!(frame.skipped, vec!["QVAPOR"]);
            let words = le_words(&fs::read(out.join("fields.f32le")).unwrap());
            assert_eq!(words, vec![100.0, 101.0, 110.0, 111.0, 112.0, 7.5]);
            let header = crate::json::parse(&fs::read(out.join("frame.json")).unwrap()).unwrap();
            assert_eq!(header.get("schema").unwrap().as_str().unwrap(), GM1_ARWEN_HISTORY_FRAME_SCHEMA);
            assert_eq!(header.get("attributes").unwrap().get("MAP_PROJ").unwrap().as_i64().unwrap(), 1);
            assert_eq!(header.get("attributes").unwrap().get("DX").unwrap().as_f64().unwrap(), 3000.0);
            let fields = header.get("fields").unwrap().as_array().unwrap();
            let u = &fields[1];
            assert_eq!(u.get("stagger").unwrap().as_str().unwrap(), "X");
            assert_eq!(u.get("offset").unwrap().as_u64().unwrap(), 8);
            let shape: Vec<u64> = u.get("shape").unwrap().as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            assert_eq!(shape, vec![1, 1, 3]);
            fs::remove_dir_all(&root).unwrap();
        }
    }

    #[test]
    fn hdf5_cdf5_existing_output_and_missing_record_are_refused() {
        let root = scratch("refusals");
        let hdf = root.join("hdf");
        let mut bytes = HDF5_MAGIC.to_vec();
        bytes.extend_from_slice(&[0; 64]);
        fs::write(&hdf, bytes).unwrap();
        let error = extract_history_record(&hdf, 0, &root.join("a"), &["T"]).unwrap_err().to_string();
        assert!(error.contains("nccopy -k nc6"), "{error}");
        let mut cdf5 = synthetic_file(2);
        cdf5[3] = 5;
        let path5 = root.join("cdf5");
        fs::write(&path5, cdf5).unwrap();
        assert!(extract_history_record(&path5, 0, &root.join("b"), &["T"]).unwrap_err().to_string().contains("CDF-5"));
        let good = root.join("good");
        fs::write(&good, synthetic_file(2)).unwrap();
        assert!(extract_history_record(&good, 2, &root.join("c"), &["T"]).unwrap_err().to_string().contains("holds 2"));
        fs::create_dir(root.join("exists")).unwrap();
        assert!(extract_history_record(&good, 0, &root.join("exists"), &["T"]).unwrap_err().to_string().contains("new directory"));
        fs::remove_dir_all(&root).unwrap();
    }
}
