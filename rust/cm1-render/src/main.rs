//! Native Rust weather renderer for GM1 history tapes.
//!
//! The renderer deliberately has no image or NetCDF dependency. NetCDF bytes
//! are decoded by ArWen's `rw_netcdf` Rust executable; this process reads the
//! resulting little-endian f64 slab and owns every transform, reduction,
//! colour mapping, annotation, and PNG byte written afterwards. Python is
//! therefore only the process orchestrator, never a weather-field renderer.

use cm1_state::forecast::{read_forecast_checkpoint_embedded, read_forecast_history_embedded};
use cm1_state::nws_reflectivity::{NWS_DBZ_CLASSES, NWS_DBZ_FLOOR, nws_dbz_rgb};
use cm1_state::reduced_history::{
    is_reduced_history, reconstruct_reduced_history, ReconstructedCarrier, RECONSTRUCTED_CARRIERS,
};

use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const ABI: &str = "gpuwm-cm1-render/v1";
const VISUAL_RECONSTRUCTION: &str = "bilinear cell-center reconstruction with 2x2 supersampling; visualization-only; source arrays unchanged; finite samples remain within local source extrema";
const MATH_AUTHORITY: &str = "CM1 r21.1-derived mathematics";
/// The panel ground the NWS scale leaves below its 5 dBZ floor, where the
/// radar product shows no echo at all.
const NO_ECHO: [u8; 3] = [9, 13, 23];
/// One class of the published scale, in dBZ. Used only to place the legend
/// strip's ends outside the first and last class so both are fully drawn.
const NWS_DBZ_CLASS_STEP: f64 = 5.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rgb(u8, u8, u8);

#[derive(Debug)]
struct Args {
    input: PathBuf,
    decoder: Option<PathBuf>,
    contract_sha256: Option<String>,
    format: String,
    output: PathBuf,
    product: String,
    fields: Vec<String>,
    nx: usize,
    ny: usize,
    nz: usize,
    record: Option<usize>,
    level: usize,
    label: String,
    valid_seconds: Option<f64>,
}

fn usage() -> &'static str {
    "usage: cm1-render --input HISTORY --format netcdf|grads|cm1fhs|cm1fcp [--decoder RW_NETCDF] \
[--contract-sha256 SHA256] \
--output IMAGE.png --product PRODUCT --field NAME [--field NAME ...] \
--nx N --ny N [--nz N] [--record N] [--level K] [--label TEXT] \
[--valid-seconds S]\n\n\
products: reflectivity | reflectivity-level | updraft | vorticity | cold-pool | scalar | supercell | vortex-composite | storm-fusion\n\
        | updraft-helicity | swath-wind | swath-vorticity | swath-pressure | swath-helicity\n\
reflectivity is the column maximum (CM1 calccref, misclibs.F:2259-2292); reflectivity-level is --level K \
of the same volume; both are painted with the standard NWS dBZ scale\n\
supercell, vortex-composite, and storm-fusion require fields in this order: reflectivity, \
vertical velocity, vertical vorticity\n\
the updraft-helicity and swath-* products take one CM1 output_* surface field \
(uh, sws, svs, sps, shs) published with --nz 1 --level 0"
}

fn parse_usize(flag: &str, raw: OsString) -> Result<usize, String> {
    raw.to_string_lossy()
        .parse::<usize>()
        .map_err(|_| format!("{flag} needs a non-negative integer"))
}

fn parse_args<I>(values: I) -> Result<Args, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut it = values.into_iter();
    let _program = it.next();
    let mut input = None;
    let mut decoder = None;
    let mut contract_sha256 = None;
    let mut format = "netcdf".to_string();
    let mut output = None;
    let mut product = None;
    let mut fields = Vec::new();
    let mut nx = None;
    let mut ny = None;
    let mut nz = 1usize;
    let mut record = None;
    let mut level = 0usize;
    let mut label = String::new();
    let mut valid_seconds = None;

    while let Some(flag) = it.next() {
        let text = flag.to_string_lossy();
        if text == "--abi" {
            println!("{ABI}");
            std::process::exit(0);
        }
        if text == "--help" || text == "-h" {
            println!("{}", usage());
            std::process::exit(0);
        }
        let value = it
            .next()
            .ok_or_else(|| format!("{text} needs a value\n{}", usage()))?;
        match text.as_ref() {
            "--input" => input = Some(PathBuf::from(value)),
            "--decoder" => decoder = Some(PathBuf::from(value)),
            "--contract-sha256" => contract_sha256 = Some(value.to_string_lossy().to_string()),
            "--format" => format = value.to_string_lossy().to_ascii_lowercase(),
            "--output" => output = Some(PathBuf::from(value)),
            "--product" => product = Some(value.to_string_lossy().to_string()),
            "--field" => fields.push(value.to_string_lossy().to_string()),
            "--nx" => nx = Some(parse_usize("--nx", value)?),
            "--ny" => ny = Some(parse_usize("--ny", value)?),
            "--nz" => nz = parse_usize("--nz", value)?,
            "--record" => record = Some(parse_usize("--record", value)?),
            "--level" => level = parse_usize("--level", value)?,
            "--label" => label = value.to_string_lossy().to_string(),
            "--valid-seconds" => {
                let parsed = value
                    .to_string_lossy()
                    .parse::<f64>()
                    .map_err(|_| "--valid-seconds needs a finite number".to_string())?;
                if !parsed.is_finite() || parsed < 0.0 {
                    return Err("--valid-seconds needs a finite non-negative number".into());
                }
                valid_seconds = Some(parsed);
            }
            _ => return Err(format!("unknown argument {text:?}\n{}", usage())),
        }
    }

    let args = Args {
        input: input.ok_or_else(|| "missing --input".to_string())?,
        decoder,
        contract_sha256,
        format,
        output: output.ok_or_else(|| "missing --output".to_string())?,
        product: product.ok_or_else(|| "missing --product".to_string())?,
        fields,
        nx: nx.ok_or_else(|| "missing --nx".to_string())?,
        ny: ny.ok_or_else(|| "missing --ny".to_string())?,
        nz,
        record,
        level,
        label,
        valid_seconds,
    };
    if args.nx == 0 || args.ny == 0 || args.nz == 0 {
        return Err("--nx/--ny/--nz must each be at least 1".into());
    }
    match args.format.as_str() {
        "netcdf" if args.decoder.is_none() => {
            return Err("--format netcdf requires --decoder RW_NETCDF".into())
        }
        "netcdf" | "grads" => {
            if args.contract_sha256.is_some() {
                return Err(
                    "--contract-sha256 is accepted only with --format cm1fhs or cm1fcp".into(),
                );
            }
        }
        "cm1fhs" | "cm1fcp" => {
            if args.decoder.is_some() {
                return Err(
                    "--format cm1fhs/cm1fcp reads Rust state directly and refuses --decoder".into(),
                );
            }
            let digest = args.contract_sha256.as_deref().ok_or_else(|| {
                "--format cm1fhs/cm1fcp requires an independently sourced --contract-sha256"
                    .to_string()
            })?;
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err("--contract-sha256 must be 64 lowercase hexadecimal characters".into());
            }
        }
        other => {
            return Err(format!(
                "unknown --format {other:?}; expected netcdf, grads, cm1fhs, or cm1fcp"
            ))
        }
    }
    let required = if matches!(
        args.product.as_str(),
        "supercell" | "vortex-composite" | "storm-fusion"
    ) {
        3
    } else {
        1
    };
    if args.fields.len() != required {
        return Err(format!(
            "product {:?} needs exactly {required} --field value(s), got {}",
            args.product,
            args.fields.len()
        ));
    }
    match args.product.as_str() {
        "reflectivity" | "reflectivity-level" | "updraft" | "vorticity" | "cold-pool"
        | "scalar" | "supercell"
        | "vortex-composite" | "storm-fusion" | "cape" | "cin" | "lcl" | "lfc"
        | "updraft-helicity" | "swath-wind" | "swath-vorticity" | "swath-pressure"
        | "swath-helicity" => {}
        other => return Err(format!("unknown product {other:?}")),
    }
    if args.level >= args.nz {
        return Err(format!("--level {} is outside nz={}", args.level, args.nz));
    }
    Ok(args)
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn create() -> Result<Self, String> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("clock before epoch: {e}"))?
            .as_nanos();
        let path = env::temp_dir().join(format!("gpuwm-cm1-render-{}-{nonce}", std::process::id()));
        fs::create_dir(&path)
            .map_err(|e| format!("create temporary directory {}: {e}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let prefix = format!("gpuwm-cm1-render-{}-", std::process::id());
        let safe = self
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with(&prefix))
            .unwrap_or(false)
            && self.path.parent() == Some(env::temp_dir().as_path());
        if safe {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn decode_netcdf_field(args: &Args, name: &str) -> Result<Vec<f64>, String> {
    if !args.input.is_file() {
        return Err(format!("history does not exist: {}", args.input.display()));
    }
    let decoder = args
        .decoder
        .as_ref()
        .ok_or_else(|| "NetCDF rendering requires a Rust decoder".to_string())?;
    if !decoder.is_file() {
        return Err(format!(
            "Rust NetCDF decoder does not exist: {}",
            decoder.display()
        ));
    }
    let temporary = TempDir::create()?;
    let output = Command::new(decoder)
        .arg("dump")
        .arg(&args.input)
        .arg(&temporary.path)
        .arg(name)
        .output()
        .map_err(|e| format!("launch {}: {e}", decoder.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "Rust NetCDF decode of {name:?} failed: {}{}",
            stderr.trim(),
            stdout.trim()
        ));
    }
    let mut candidates = Vec::new();
    for entry in
        fs::read_dir(&temporary.path).map_err(|e| format!("inspect decoder output: {e}"))?
    {
        let path = entry
            .map_err(|e| format!("inspect decoder output: {e}"))?
            .path();
        if matches!(
            path.extension().and_then(|x| x.to_str()),
            Some("f64") | Some("f64le")
        ) {
            candidates.push(path);
        }
    }
    if candidates.len() != 1 {
        return Err(format!(
            "Rust decoder emitted {} f64 slab(s) for one variable; expected one",
            candidates.len()
        ));
    }
    let mut bytes = Vec::new();
    File::open(&candidates[0])
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|e| format!("read decoded slab {}: {e}", candidates[0].display()))?;
    if bytes.len() % 8 != 0 {
        return Err(format!(
            "decoded slab has {} bytes, not a whole number of f64 values",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|word| f64::from_le_bytes(word.try_into().unwrap()))
        .collect())
}

#[derive(Debug)]
struct GradsVariable {
    name: String,
    levels: usize,
    offset_planes: usize,
}

#[derive(Debug)]
struct GradsControl {
    data: PathBuf,
    nx: usize,
    ny: usize,
    declared_records: usize,
    big_endian: bool,
    variables: Vec<GradsVariable>,
    planes_per_record: usize,
}

fn grads_control(path: &PathBuf) -> Result<GradsControl, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("read GrADS control {}: {e}", path.display()))?;
    let mut dataset = None;
    let mut nx = None;
    let mut ny = None;
    let mut declared_records = None;
    let mut big_endian = false;
    let mut in_variables = false;
    let mut variables = Vec::new();
    let mut offset_planes = 0usize;
    for (line_number, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('*') {
            continue;
        }
        let words = line.split_whitespace().collect::<Vec<_>>();
        if words.is_empty() {
            continue;
        }
        let key = words[0].to_ascii_lowercase();
        if in_variables {
            if key == "endvars" {
                in_variables = false;
                continue;
            }
            if words.len() < 2 {
                return Err(format!(
                    "{}:{} malformed GrADS variable row",
                    path.display(),
                    line_number + 1
                ));
            }
            let raw_levels = words[1].parse::<usize>().map_err(|_| {
                format!(
                    "{}:{} variable level count is not an integer",
                    path.display(),
                    line_number + 1
                )
            })?;
            let levels = raw_levels.max(1);
            variables.push(GradsVariable {
                name: words[0].to_string(),
                levels,
                offset_planes,
            });
            offset_planes = offset_planes
                .checked_add(levels)
                .ok_or_else(|| "GrADS record plane count overflows usize".to_string())?;
            continue;
        }
        match key.as_str() {
            "dset" if words.len() >= 2 => dataset = Some(words[1].to_string()),
            "xdef" if words.len() >= 2 => {
                nx = Some(words[1].parse::<usize>().map_err(|_| {
                    format!(
                        "{}:{} xdef is not an integer",
                        path.display(),
                        line_number + 1
                    )
                })?)
            }
            "ydef" if words.len() >= 2 => {
                ny = Some(words[1].parse::<usize>().map_err(|_| {
                    format!(
                        "{}:{} ydef is not an integer",
                        path.display(),
                        line_number + 1
                    )
                })?)
            }
            "tdef" if words.len() >= 2 => {
                declared_records = Some(words[1].parse::<usize>().map_err(|_| {
                    format!(
                        "{}:{} tdef is not an integer",
                        path.display(),
                        line_number + 1
                    )
                })?)
            }
            "options" => {
                big_endian = words[1..]
                    .iter()
                    .any(|word| word.eq_ignore_ascii_case("big_endian"));
            }
            "vars" => in_variables = true,
            _ => {}
        }
    }
    if in_variables {
        return Err(format!("{} has no endvars", path.display()));
    }
    let token = dataset.ok_or_else(|| format!("{} has no dset", path.display()))?;
    let relative = token.strip_prefix('^').unwrap_or(&token);
    let data = if token.starts_with('^') {
        path.parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(relative)
    } else {
        PathBuf::from(relative)
    };
    if variables.is_empty() || offset_planes == 0 {
        return Err(format!("{} declares no GrADS variables", path.display()));
    }
    Ok(GradsControl {
        data,
        nx: nx.ok_or_else(|| format!("{} has no xdef", path.display()))?,
        ny: ny.ok_or_else(|| format!("{} has no ydef", path.display()))?,
        declared_records: declared_records
            .ok_or_else(|| format!("{} has no tdef", path.display()))?,
        big_endian,
        variables,
        planes_per_record: offset_planes,
    })
}

fn decode_grads_field(args: &Args, name: &str) -> Result<Vec<f64>, String> {
    let control = grads_control(&args.input)?;
    if control.nx != args.nx || control.ny != args.ny {
        return Err(format!(
            "GrADS control grid is {}x{}, CLI declared {}x{}",
            control.nx, control.ny, args.nx, args.ny
        ));
    }
    let variable = control
        .variables
        .iter()
        .find(|variable| variable.name.eq_ignore_ascii_case(name))
        .ok_or_else(|| format!("GrADS control declares no variable {name:?}"))?;
    if variable.levels != args.nz {
        return Err(format!(
            "GrADS variable {name:?} has {} level(s), CLI declared nz={}",
            variable.levels, args.nz
        ));
    }
    let plane_values = control
        .nx
        .checked_mul(control.ny)
        .ok_or_else(|| "GrADS plane size overflows usize".to_string())?;
    let record_values = plane_values
        .checked_mul(control.planes_per_record)
        .ok_or_else(|| "GrADS record size overflows usize".to_string())?;
    let record_bytes = record_values
        .checked_mul(4)
        .ok_or_else(|| "GrADS record byte size overflows usize".to_string())?;
    let metadata = fs::metadata(&control.data)
        .map_err(|e| format!("inspect GrADS data {}: {e}", control.data.display()))?;
    let complete_records = (metadata.len() as usize) / record_bytes;
    if complete_records == 0 {
        return Err(format!(
            "GrADS data {} has no complete record",
            control.data.display()
        ));
    }
    if complete_records > control.declared_records {
        return Err(format!(
            "GrADS data has {complete_records} complete records, exceeding tdef {}",
            control.declared_records
        ));
    }
    let field_values = plane_values
        .checked_mul(variable.levels)
        .ok_or_else(|| "GrADS field size overflows usize".to_string())?;
    let mut file = File::open(&control.data)
        .map_err(|e| format!("open GrADS data {}: {e}", control.data.display()))?;
    let mut output = Vec::with_capacity(complete_records * field_values);
    let mut bytes = vec![0u8; field_values * 4];
    for record in 0..complete_records {
        let byte_offset = record
            .checked_mul(record_bytes)
            .and_then(|value| value.checked_add(variable.offset_planes * plane_values * 4))
            .ok_or_else(|| "GrADS field offset overflows usize".to_string())?;
        file.seek(SeekFrom::Start(byte_offset as u64))
            .and_then(|_| file.read_exact(&mut bytes))
            .map_err(|e| format!("read GrADS {name:?} record {record}: {e}"))?;
        output.extend(bytes.chunks_exact(4).map(|word| {
            let raw: [u8; 4] = word.try_into().unwrap();
            if control.big_endian {
                f32::from_be_bytes(raw) as f64
            } else {
                f32::from_le_bytes(raw) as f64
            }
        }));
    }
    Ok(output)
}

#[derive(Debug)]
struct DecodedCm1Forecast {
    fields: Vec<Vec<f64>>,
    model_time_s: f64,
}

fn decode_cm1_forecast(args: &Args) -> Result<DecodedCm1Forecast, String> {
    let expected_contract = args
        .contract_sha256
        .as_deref()
        .expect("CM1 forecast argument validation requires a contract hash");
    let (contract, restored) = match args.format.as_str() {
        "cm1fhs" => read_forecast_history_embedded(&args.input, expected_contract),
        "cm1fcp" => read_forecast_checkpoint_embedded(&args.input, expected_contract),
        _ => unreachable!("CM1 forecast decoder called for another format"),
    }
    .map_err(|error| format!("strict CM1 forecast decode failed: {error}"))?;
    // A frame published under the reduced GM1 history profile omits prs and
    // zf and carries the columns their CM1 reconstruction reads instead.
    // Rebuild them here, once, so a requested field resolves the same way
    // whichever profile the producer chose.
    let reconstructed: Vec<ReconstructedCarrier> = if is_reduced_history(&contract) {
        reconstruct_reduced_history(&contract, restored.frame())
            .map_err(|error| format!("reduced CM1 history reconstruction failed: {error}"))?
    } else {
        Vec::new()
    };
    let mut fields = Vec::with_capacity(args.fields.len());
    for name in &args.fields {
        let carrier = contract
            .arrays()
            .iter()
            .find(|spec| spec.name() == name)
            .map(|spec| -> Result<(&[usize], Vec<f32>), String> {
                let array = restored
                    .frame()
                    .arrays()
                    .iter()
                    .find(|array| array.name() == name)
                    .ok_or_else(|| {
                        format!("restored CM1 forecast frame has no array {name:?}")
                    })?;
                Ok((spec.shape(), array.values().to_vec()))
            })
            .or_else(|| {
                reconstructed
                    .iter()
                    .find(|rebuilt| rebuilt.name() == name)
                    .map(|rebuilt| Ok((rebuilt.shape(), rebuilt.values().to_vec())))
            });
        let (shape, values) = match carrier {
            Some(result) => result?,
            None if is_reduced_history(&contract) => {
                return Err(format!(
                    "CM1 forecast contract declares no array {name:?}; this frame is the reduced GM1 history profile, which omits {RECONSTRUCTED_CARRIERS:?} and rebuilds only those"
                ));
            }
            None => {
                return Err(format!("CM1 forecast contract declares no array {name:?}"));
            }
        };
        let expected_shape = if shape.len() == 2 {
            vec![args.ny, args.nx]
        } else if shape.len() == 3 {
            vec![args.nz, args.ny, args.nx]
        } else {
            return Err(format!(
                "CM1 forecast array {name:?} has rank {}; renderer accepts only yx or zyx fields",
                shape.len()
            ));
        };
        if shape != expected_shape {
            return Err(format!(
                "CM1 forecast array {name:?} shape is {shape:?}; CLI rendering grid implies {expected_shape:?}"
            ));
        }
        if shape.len() == 2 && args.nz != 1 {
            return Err(format!(
                "CM1 forecast surface array {name:?} requires CLI --nz 1, got {}",
                args.nz
            ));
        }
        fields.push(values.iter().map(|value| f64::from(*value)).collect());
    }
    Ok(DecodedCm1Forecast {
        fields,
        model_time_s: restored.frame().clock().model_time_s(),
    })
}

fn decode_field(args: &Args, name: &str) -> Result<Vec<f64>, String> {
    match args.format.as_str() {
        "netcdf" => decode_netcdf_field(args, name),
        "grads" => decode_grads_field(args, name),
        "cm1fhs" | "cm1fcp" => Err("CM1 forecast fields must be decoded together".into()),
        _ => unreachable!("format validated during argument parsing"),
    }
}

fn select_record(values: &[f64], args: &Args) -> Result<(Vec<f64>, usize), String> {
    let cells = args
        .nx
        .checked_mul(args.ny)
        .and_then(|v| v.checked_mul(args.nz))
        .ok_or_else(|| "declared grid size overflows usize".to_string())?;
    if values.is_empty() || !values.len().is_multiple_of(cells) {
        return Err(format!(
            "decoded field has {} values, not an integer number of nz*ny*nx={} slabs",
            values.len(),
            cells
        ));
    }
    let records = values.len() / cells;
    let record = args.record.unwrap_or(records - 1);
    if record >= records {
        return Err(format!(
            "record {record} is outside {records} decoded record(s)"
        ));
    }
    Ok((
        values[record * cells..(record + 1) * cells].to_vec(),
        record,
    ))
}

fn plane_max(volume: &[f64], nx: usize, ny: usize, nz: usize) -> Vec<f64> {
    let cells = nx * ny;
    let mut plane = vec![f64::NEG_INFINITY; cells];
    for k in 0..nz {
        for p in 0..cells {
            plane[p] = plane[p].max(volume[k * cells + p]);
        }
    }
    plane
}

fn plane_level(volume: &[f64], nx: usize, ny: usize, level: usize) -> Vec<f64> {
    let cells = nx * ny;
    volume[level * cells..(level + 1) * cells].to_vec()
}

fn finite_range(values: &[f64]) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &value in values {
        if value.is_finite() {
            lo = lo.min(value);
            hi = hi.max(value);
        }
    }
    (lo <= hi).then_some((lo, hi))
}

#[derive(Clone, Copy)]
enum Palette {
    Reflectivity,
    ReflectivityLevel,
    Updraft,
    Vorticity,
    ColdPool,
    // CM1 r21.1 writeout.F:2937-3028: the four 2-D environment planes the
    // output_cape/output_cin/output_lcl/output_lfc switches publish.  They are
    // surface fields, one value per column, so the level reduction reads plane
    // zero of an nz=1 volume.
    Cape,
    Cin,
    Lcl,
    Lfc,
    Scalar,
    // CM1 r21.1 output_uh (src/misclibs.F:1417-1532) and the four surface
    // swaths a tornado run is judged on (src/solve3.F:855-1136).
    UpdraftHelicity,
    SwathWind,
    SwathVorticity,
    SwathPressure,
    SwathHelicity,
}

impl Palette {
    fn title(self) -> &'static str {
        match self {
            Self::Reflectivity => "COMPOSITE REFLECTIVITY",
            Self::ReflectivityLevel => "REFLECTIVITY AT LEVEL",
            Self::Updraft => "MAXIMUM UPDRAFT",
            Self::Vorticity => "LOW-LEVEL VERTICAL VORTICITY",
            Self::ColdPool => "LOW-LEVEL THETA PERTURBATION",
            Self::Cape => "CONVECTIVE AVAILABLE POTENTIAL ENERGY",
            Self::Cin => "CONVECTIVE INHIBITION",
            Self::Lcl => "LIFTED CONDENSATION LEVEL",
            Self::Lfc => "LEVEL OF FREE CONVECTION",
            Self::Scalar => "SCALAR FIELD",
            Self::UpdraftHelicity => "UPDRAFT HELICITY 2-5 KM AGL",
            Self::SwathWind => "SWATH: MAX SURFACE WIND",
            Self::SwathVorticity => "SWATH: MAX SURFACE VERTICAL VORTICITY",
            Self::SwathPressure => "SWATH: MIN SURFACE PRESSURE PERTURBATION",
            Self::SwathHelicity => "SWATH: MAX UPDRAFT HELICITY",
        }
    }

    fn units(self) -> &'static str {
        match self {
            Self::Reflectivity | Self::ReflectivityLevel => "DBZ",
            Self::Updraft => "M/S",
            Self::Vorticity => "1/S",
            Self::ColdPool => "K",
            Self::Cape | Self::Cin => "J/KG",
            Self::Lcl | Self::Lfc => "M AGL",
            Self::Scalar => "VALUE",
            Self::UpdraftHelicity | Self::SwathHelicity => "M2/S2",
            Self::SwathWind => "M/S",
            Self::SwathVorticity => "1/S",
            Self::SwathPressure => "PA",
        }
    }

    fn bounds(self, values: &[f64]) -> (f64, f64) {
        match self {
            // The NWS scale's own published span: its first class starts at
            // 5 dBZ and its last at 75. The legend strip is drawn across
            // these bounds, so it reads as the product's own legend.
            Self::Reflectivity | Self::ReflectivityLevel => (
                NWS_DBZ_FLOOR as f64 - NWS_DBZ_CLASS_STEP,
                NWS_DBZ_CLASSES[NWS_DBZ_CLASSES.len() - 1].0 as f64 + NWS_DBZ_CLASS_STEP,
            ),
            Self::Updraft => (0.0, 60.0),
            // Supercell-scale low-level mesocyclone/tornado diagnostics are
            // normally O(1e-2 s-1) on a 1 km grid. A +/-0.12 range hid the
            // entire native CM1 signal in near-white pixels.
            Self::Vorticity => (-0.025, 0.025),
            // Kain et al. 2008, Wea. Forecasting 23, 931-952 -- the paper
            // CM1 cites at src/misclibs.F:1423 -- flags a rotating updraft at
            // 25 m2/s2 and strong rotation at 100. The upper bound is set at
            // 300 so the operational range occupies the middle of the ramp
            // rather than the first eighth of it.
            Self::UpdraftHelicity | Self::SwathHelicity => (0.0, 300.0),
            // WSEC 2006 puts EF0 at 29 m/s and EF5 at 89 m/s; the bound is
            // the top of that scale, so a swath colour is readable against
            // the damage scale and not against this run's own maximum.
            Self::SwathWind => (0.0, 90.0),
            // 0.1 s-1 is the near-surface tornado-like-vortex threshold this
            // port reports against; 0.5 leaves four times it on the ramp.
            Self::SwathVorticity => (0.0, 0.5),
            // A tornado core's surface pressure deficit is tens of hPa.
            Self::SwathPressure => (-8000.0, 0.0),
            Self::ColdPool => (-12.0, 4.0),
            // Fixed physical display scales, not per-frame normalization, for
            // the same reason the storm palettes above are fixed: a weakening
            // inflow must look weaker between two frames, not the same.
            Self::Cape => (0.0, 6000.0),
            // getcape.F:475 accumulates CIN as a positive magnitude, so the
            // scale is positive and larger means more inhibition.
            Self::Cin => (0.0, 300.0),
            Self::Lcl | Self::Lfc => (0.0, 4000.0),
            Self::Scalar => {
                let (lo, hi) = finite_range(values).unwrap_or((0.0, 1.0));
                if (hi - lo).abs() < f64::EPSILON {
                    (lo - 0.5, hi + 0.5)
                } else {
                    (lo, hi)
                }
            }
        }
    }

    fn color(self, value: f64, bounds: (f64, f64)) -> Rgb {
        if !value.is_finite() {
            return Rgb(90, 90, 90);
        }
        // Reflectivity is not painted with a ramp. It is painted with the
        // sixteen-class NWS dBZ legend a forecaster already reads storms by,
        // so a 50 dBZ core looks like a 50 dBZ core and the class boundaries
        // stay where the product puts them. Below the scale's floor the
        // product shows no echo, which is the panel ground and not a colour.
        if matches!(self, Self::Reflectivity | Self::ReflectivityLevel) {
            let rgb = nws_dbz_rgb(value as f32, NO_ECHO);
            return Rgb(rgb[0], rgb[1], rgb[2]);
        }
        // Reflectivity is read on the operational NWS scale in raw dBZ, not
        // on a normalised ramp: the bin edges ARE the information.
        if matches!(self, Self::Reflectivity) {
            return nws_reflectivity_color(value);
        }
        let t = ((value - bounds.0) / (bounds.1 - bounds.0)).clamp(0.0, 1.0);
        let stops: &[(f64, Rgb)] = match self {
            Self::Reflectivity | Self::ReflectivityLevel => unreachable!(
                "reflectivity is painted with the NWS class table above"
            ),
            Self::Updraft => &[
                (0.00, Rgb(10, 14, 25)),
                (0.12, Rgb(23, 55, 138)),
                (0.30, Rgb(20, 178, 209)),
                (0.48, Rgb(55, 213, 121)),
                (0.64, Rgb(249, 222, 58)),
                (0.80, Rgb(238, 79, 43)),
                (1.00, Rgb(255, 250, 245)),
            ],
            Self::Vorticity => &[
                (0.00, Rgb(18, 43, 120)),
                (0.25, Rgb(64, 150, 225)),
                (0.49, Rgb(220, 235, 244)),
                (0.50, Rgb(246, 246, 241)),
                (0.75, Rgb(235, 105, 72)),
                (1.00, Rgb(130, 10, 38)),
            ],
            Self::ColdPool => &[
                (0.00, Rgb(28, 44, 133)),
                (0.30, Rgb(39, 152, 212)),
                (0.55, Rgb(126, 215, 193)),
                (0.75, Rgb(241, 241, 218)),
                (1.00, Rgb(210, 74, 36)),
            ],
            Self::Cape => &[
                (0.00, Rgb(14, 18, 30)),
                (0.20, Rgb(30, 70, 120)),
                (0.40, Rgb(36, 160, 150)),
                (0.60, Rgb(200, 205, 80)),
                (0.80, Rgb(238, 128, 36)),
                (1.00, Rgb(206, 32, 40)),
            ],
            Self::Cin => &[
                (0.00, Rgb(245, 245, 240)),
                (0.25, Rgb(196, 206, 224)),
                (0.55, Rgb(120, 128, 176)),
                (0.80, Rgb(66, 52, 124)),
                (1.00, Rgb(24, 12, 48)),
            ],
            // A low LCL or LFC is the interesting end, so the scale runs warm
            // to cool with height rather than the other way round.
            Self::Lcl | Self::Lfc => &[
                (0.00, Rgb(248, 226, 120)),
                (0.25, Rgb(232, 142, 62)),
                (0.50, Rgb(150, 96, 150)),
                (0.75, Rgb(52, 96, 176)),
                (1.00, Rgb(14, 22, 44)),
            ],
            Self::Scalar => &[
                (0.00, Rgb(17, 24, 60)),
                (0.25, Rgb(40, 116, 180)),
                (0.50, Rgb(43, 192, 166)),
                (0.75, Rgb(239, 211, 61)),
                (1.00, Rgb(181, 36, 50)),
            ],
            // The two literature thresholds are colour breaks, not tick
            // marks: below 25 m2/s2 the field stays dark, 25-100 is the
            // rotating-updraft band, and above 100 it saturates toward the
            // strong-rotation colour, so a reader sees the classification
            // without reading the legend.
            Self::UpdraftHelicity | Self::SwathHelicity => &[
                (0.0000, Rgb(10, 14, 25)),
                (0.0833, Rgb(28, 62, 110)),
                (0.3333, Rgb(56, 170, 206)),
                (0.5500, Rgb(235, 206, 66)),
                (0.8000, Rgb(228, 96, 44)),
                (1.0000, Rgb(255, 244, 238)),
            ],
            // The stops sit on the EF bounds scaled by the 90 m/s ceiling:
            // 29, 38, 50, 61, 74 and 89 m/s.
            Self::SwathWind => &[
                (0.0000, Rgb(10, 14, 25)),
                (0.3222, Rgb(37, 92, 165)),
                (0.4222, Rgb(48, 176, 190)),
                (0.5556, Rgb(226, 206, 72)),
                (0.6778, Rgb(233, 138, 44)),
                (0.8222, Rgb(214, 48, 44)),
                (0.9889, Rgb(151, 18, 92)),
                (1.0000, Rgb(245, 238, 250)),
            ],
            Self::SwathVorticity => &[
                (0.00, Rgb(10, 14, 25)),
                (0.20, Rgb(46, 74, 150)),
                (0.45, Rgb(72, 180, 190)),
                (0.70, Rgb(238, 200, 62)),
                (1.00, Rgb(150, 12, 40)),
            ],
            Self::SwathPressure => &[
                (0.00, Rgb(120, 6, 40)),
                (0.30, Rgb(216, 78, 48)),
                (0.60, Rgb(238, 196, 84)),
                (0.85, Rgb(96, 168, 204)),
                (1.00, Rgb(10, 14, 25)),
            ],
        };
        interpolate_stops(stops, t)
    }
}

/// The National Weather Service's operational NEXRAD reflectivity scale.
///
/// Every WSR-88D product a forecaster looks at uses this table: fourteen
/// 5 dBZ bins from 5 to 75 dBZ, each a flat colour, with everything below
/// 5 dBZ left as background.  The discreteness is the point -- the eye reads
/// the bin edges as contours, which is how "40 dBZ core", "hook echo" and
/// "bounded weak echo region" become visible at a glance.  A smooth gradient
/// over the same range is prettier and says less.
///
/// Colours are the published NWS/NEXRAD table values, not a re-derivation.
const NWS_REFLECTIVITY_BINS: [(f64, Rgb); 15] = [
    (5.0, Rgb(4, 233, 231)),
    (10.0, Rgb(1, 159, 244)),
    (15.0, Rgb(3, 0, 244)),
    (20.0, Rgb(2, 253, 2)),
    (25.0, Rgb(1, 197, 1)),
    (30.0, Rgb(0, 142, 0)),
    (35.0, Rgb(253, 248, 2)),
    (40.0, Rgb(229, 188, 0)),
    (45.0, Rgb(253, 149, 0)),
    (50.0, Rgb(253, 0, 0)),
    (55.0, Rgb(212, 0, 0)),
    (60.0, Rgb(188, 0, 0)),
    (65.0, Rgb(248, 0, 253)),
    (70.0, Rgb(152, 84, 198)),
    (75.0, Rgb(255, 255, 255)),
];

/// Background for anything below the 5 dBZ floor: CM1's calccref seeds an
/// echo-free column at -1000 dBZ, and painting that as "dark blue weak echo"
/// would put a radar return where the model has clear air.
const NWS_REFLECTIVITY_BELOW_FLOOR: Rgb = Rgb(12, 14, 23);

fn nws_reflectivity_color(dbz: f64) -> Rgb {
    if !dbz.is_finite() {
        return Rgb(90, 90, 90);
    }
    let mut chosen = NWS_REFLECTIVITY_BELOW_FLOOR;
    for (edge, colour) in NWS_REFLECTIVITY_BINS {
        if dbz >= edge {
            chosen = colour;
        }
    }
    chosen
}

fn interpolate_stops(stops: &[(f64, Rgb)], t: f64) -> Rgb {
    for pair in stops.windows(2) {
        if t <= pair[1].0 {
            let width = pair[1].0 - pair[0].0;
            let q = if width == 0.0 {
                0.0
            } else {
                (t - pair[0].0) / width
            };
            return mix(pair[0].1, pair[1].1, q);
        }
    }
    stops.last().unwrap().1
}

fn mix(a: Rgb, b: Rgb, t: f64) -> Rgb {
    let lerp = |x: u8, y: u8| ((x as f64) + ((y as f64) - (x as f64)) * t).round() as u8;
    Rgb(lerp(a.0, b.0), lerp(a.1, b.1), lerp(a.2, b.2))
}

struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl Canvas {
    fn new(width: usize, height: usize, color: Rgb) -> Self {
        let mut pixels = vec![0; width * height * 3];
        for chunk in pixels.chunks_exact_mut(3) {
            chunk.copy_from_slice(&[color.0, color.1, color.2]);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    fn set(&mut self, x: i32, y: i32, color: Rgb) {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return;
        }
        let offset = (y as usize * self.width + x as usize) * 3;
        self.pixels[offset..offset + 3].copy_from_slice(&[color.0, color.1, color.2]);
    }

    fn rect(&mut self, x: usize, y: usize, width: usize, height: usize, color: Rgb) {
        let xmax = (x + width).min(self.width);
        let ymax = (y + height).min(self.height);
        for py in y..ymax {
            for px in x..xmax {
                self.set(px as i32, py as i32, color);
            }
        }
    }

    fn line(&mut self, mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: Rgb) {
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            self.set(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let twice = 2 * error;
            if twice >= dy {
                error += dy;
                x0 += sx;
            }
            if twice <= dx {
                error += dx;
                y0 += sy;
            }
        }
    }

    fn text(&mut self, x: usize, y: usize, value: &str, scale: usize, color: Rgb) {
        let mut cursor = x;
        for ch in value.to_ascii_uppercase().chars() {
            let glyph = glyph(ch);
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        self.rect(cursor + col * scale, y + row * scale, scale, scale, color);
                    }
                }
            }
            cursor += 6 * scale;
        }
    }
}

fn glyph(ch: char) -> [u8; 7] {
    match ch {
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [31, 4, 4, 4, 4, 4, 31],
        'J' => [7, 2, 2, 2, 18, 18, 12],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 21, 19, 17, 17, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '+' => [0, 4, 4, 31, 4, 4, 0],
        '.' => [0, 0, 0, 0, 0, 12, 12],
        ':' => [0, 12, 12, 0, 12, 12, 0],
        '/' => [1, 1, 2, 4, 8, 16, 16],
        '(' => [2, 4, 8, 8, 8, 4, 2],
        ')' => [8, 4, 2, 2, 2, 4, 8],
        '|' => [4, 4, 4, 4, 4, 4, 4],
        '_' => [0, 0, 0, 0, 0, 0, 31],
        ' ' => [0; 7],
        _ => [31, 17, 2, 4, 0, 4, 0],
    }
}

fn bilinear_sample(plane: &[f64], nx: usize, ny: usize, x: f64, y: f64) -> Option<f64> {
    if nx == 0 || ny == 0 || plane.len() != nx.checked_mul(ny)? {
        return None;
    }
    let x = x.clamp(0.0, (nx - 1) as f64);
    let y = y.clamp(0.0, (ny - 1) as f64);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(nx - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;
    let corners = [
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x1, y0, tx * (1.0 - ty)),
        (x0, y1, (1.0 - tx) * ty),
        (x1, y1, tx * ty),
    ];
    let mut value = 0.0;
    for (gx, gy, weight) in corners {
        if weight > 0.0 {
            let corner = plane[gy * nx + gx];
            if !corner.is_finite() {
                return None;
            }
            value += weight * corner;
        }
    }
    Some(value)
}

fn reconstructed_pixel_value(
    plane: &[f64],
    nx: usize,
    ny: usize,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
) -> Option<f64> {
    // Two samples per axis provide antialiasing. Each sample is a convex
    // bilinear combination of the four neighbouring cell-centre values, and
    // their mean is another convex combination. This cannot create a finite
    // display value outside the local source extrema.
    let offsets = [0.25_f64, 0.75_f64];
    let mut sum = 0.0;
    for oy in offsets {
        let normalized_y = (py as f64 + oy) / height as f64;
        let grid_y = (1.0 - normalized_y) * ny as f64 - 0.5;
        for ox in offsets {
            let normalized_x = (px as f64 + ox) / width as f64;
            let grid_x = normalized_x * nx as f64 - 0.5;
            sum += bilinear_sample(plane, nx, ny, grid_x, grid_y)?;
        }
    }
    Some(sum * 0.25)
}

fn draw_panel_border(canvas: &mut Canvas, x0: usize, y0: usize, width: usize, height: usize) {
    let border = Rgb(200, 210, 226);
    canvas.line(
        x0 as i32,
        y0 as i32,
        (x0 + width - 1) as i32,
        y0 as i32,
        border,
    );
    canvas.line(
        x0 as i32,
        y0 as i32,
        x0 as i32,
        (y0 + height - 1) as i32,
        border,
    );
    canvas.line(
        (x0 + width - 1) as i32,
        y0 as i32,
        (x0 + width - 1) as i32,
        (y0 + height - 1) as i32,
        border,
    );
    canvas.line(
        x0 as i32,
        (y0 + height - 1) as i32,
        (x0 + width - 1) as i32,
        (y0 + height - 1) as i32,
        border,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_plane(
    canvas: &mut Canvas,
    plane: &[f64],
    nx: usize,
    ny: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    palette: Palette,
    title: &str,
) {
    let bounds = palette.bounds(plane);
    for py in 0..height {
        for px in 0..width {
            let color = reconstructed_pixel_value(plane, nx, ny, px, py, width, height)
                .map(|value| palette.color(value, bounds))
                .unwrap_or(Rgb(90, 90, 90));
            canvas.set((x0 + px) as i32, (y0 + py) as i32, color);
        }
    }
    draw_panel_border(canvas, x0, y0, width, height);
    canvas.rect(x0, y0.saturating_sub(28), width, 28, Rgb(16, 22, 36));
    canvas.text(x0 + 8, y0.saturating_sub(22), title, 2, Rgb(236, 240, 247));
    let legend_y = y0 + height + 12;
    for px in 0..width {
        let value = bounds.0 + (bounds.1 - bounds.0) * px as f64 / (width - 1) as f64;
        let color = palette.color(value, bounds);
        canvas.rect(x0 + px, legend_y, 1, 14, color);
    }
    let left = format!("{:.2}", bounds.0);
    let right = format!("{:.2} {}", bounds.1, palette.units());
    canvas.text(x0, legend_y + 20, &left, 1, Rgb(210, 217, 229));
    let right_width = right.chars().count() * 6;
    canvas.text(
        x0 + width.saturating_sub(right_width),
        legend_y + 20,
        &right,
        1,
        Rgb(210, 217, 229),
    );
}

fn product_palette(product: &str) -> Palette {
    match product {
        "reflectivity" => Palette::Reflectivity,
        "reflectivity-level" => Palette::ReflectivityLevel,
        "updraft" => Palette::Updraft,
        "vorticity" => Palette::Vorticity,
        "cold-pool" => Palette::ColdPool,
        "cape" => Palette::Cape,
        "cin" => Palette::Cin,
        "lcl" => Palette::Lcl,
        "lfc" => Palette::Lfc,
        "updraft-helicity" => Palette::UpdraftHelicity,
        "swath-wind" => Palette::SwathWind,
        "swath-vorticity" => Palette::SwathVorticity,
        "swath-pressure" => Palette::SwathPressure,
        "swath-helicity" => Palette::SwathHelicity,
        _ => Palette::Scalar,
    }
}

fn unit_interval(value: f64, lower: f64, upper: f64) -> f64 {
    ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
}

fn vortex_composite_color(reflectivity_dbz: f64, updraft_ms: f64, zeta_s: f64) -> Rgb {
    if !reflectivity_dbz.is_finite() || !updraft_ms.is_finite() || !zeta_s.is_finite() {
        return Rgb(90, 90, 90);
    }

    // These are fixed physical display scales, not per-frame normalization.
    // Consequently a weak initialized field remains weak; a red/maroon core
    // appears only when the supplied positive vertical vorticity supports it.
    let reflectivity = unit_interval(reflectivity_dbz, -10.0, 50.0);
    let updraft = unit_interval(updraft_ms, 0.0, 30.0);
    let cyclonic = unit_interval(zeta_s, 0.0, 0.025);
    let anticyclonic = unit_interval(-zeta_s, 0.0, 0.025);
    let reflectivity_color = interpolate_stops(
        &[
            (0.00, Rgb(13, 60, 55)),
            (0.35, Rgb(25, 157, 111)),
            (0.70, Rgb(153, 207, 78)),
            (1.00, Rgb(240, 224, 58)),
        ],
        reflectivity,
    );
    let ascending = mix(reflectivity_color, Rgb(38, 211, 230), updraft);
    let signed_rotation = mix(ascending, Rgb(30, 60, 156), anticyclonic);
    mix(signed_rotation, Rgb(177, 14, 43), cyclonic)
}

fn storm_fusion_color(reflectivity_dbz: f64, updraft_ms: f64, zeta_s: f64) -> Rgb {
    if !reflectivity_dbz.is_finite() || !updraft_ms.is_finite() || !zeta_s.is_finite() {
        return Rgb(90, 90, 90);
    }

    // A second fixed physical mapping exposes developing storm structure at
    // mesocyclone scales while remaining comparable between frames.  It is not
    // a per-frame contrast stretch and does not alter the source arrays.
    let reflectivity = unit_interval(reflectivity_dbz, -10.0, 35.0);
    let updraft = unit_interval(updraft_ms, 0.0, 8.0);
    let cyclonic = unit_interval(zeta_s, 0.0, 0.010);
    let anticyclonic = unit_interval(-zeta_s, 0.0, 0.010);
    let reflectivity_color = interpolate_stops(
        &[
            (0.00, Rgb(0, 51, 31)),
            (0.35, Rgb(0, 139, 69)),
            (0.70, Rgb(17, 190, 101)),
            (1.00, Rgb(137, 218, 105)),
        ],
        reflectivity,
    );
    let ascending = mix(reflectivity_color, Rgb(0, 224, 235), updraft);
    let signed_rotation = mix(ascending, Rgb(19, 86, 202), anticyclonic);
    mix(signed_rotation, Rgb(157, 0, 31), cyclonic)
}

fn time_caption(args: &Args, record: usize) -> String {
    args.valid_seconds
        .map(|seconds| format!("T+{seconds} S"))
        .unwrap_or_else(|| format!("RECORD {record}"))
}

fn is_initialization(args: &Args) -> bool {
    args.valid_seconds == Some(0.0)
}

fn standard_subtitle(args: &Args, record: usize) -> String {
    if is_initialization(args) {
        "GM1 INITIALIZATION | T+0 S | NOT TORNADO | NOT FORECAST".to_string()
    } else if args.label.is_empty() {
        time_caption(args, record)
    } else {
        format!("{} | {}", args.label, time_caption(args, record))
    }
}

fn reduce(product: &str, volume: &[f64], args: &Args) -> Vec<f64> {
    match product {
        "reflectivity" | "updraft" => plane_max(volume, args.nx, args.ny, args.nz),
        _ => plane_level(volume, args.nx, args.ny, args.level),
    }
}

fn render_single(args: &Args, plane: &[f64], palette: Palette, record: usize) -> Canvas {
    let size = 900usize;
    let mut canvas = Canvas::new(1020, 1100, Rgb(9, 13, 23));
    canvas.rect(0, 0, 1020, 92, Rgb(17, 25, 42));
    canvas.text(
        38,
        22,
        &format!("GM1 {}", palette.title()),
        4,
        Rgb(245, 247, 251),
    );
    let subtitle = standard_subtitle(args, record);
    canvas.text(40, 66, &subtitle, 2, Rgb(143, 205, 255));
    draw_plane(
        &mut canvas,
        plane,
        args.nx,
        args.ny,
        60,
        142,
        size,
        size,
        palette,
        &args.fields[0],
    );
    canvas
}

fn render_supercell(args: &Args, planes: &[Vec<f64>], record: usize) -> Canvas {
    let panel = 500usize;
    let margin = 35usize;
    let gap = 25usize;
    let width = margin * 2 + panel * 3 + gap * 2;
    let mut canvas = Canvas::new(width, 700, Rgb(9, 13, 23));
    canvas.rect(0, 0, width, 92, Rgb(17, 25, 42));
    canvas.text(
        32,
        20,
        "GM1 THREE-FIELD STORM DIAGNOSTICS",
        4,
        Rgb(245, 247, 251),
    );
    let subtitle = standard_subtitle(args, record);
    canvas.text(34, 66, &subtitle, 2, Rgb(143, 205, 255));
    let palettes = [Palette::Reflectivity, Palette::Updraft, Palette::Vorticity];
    for index in 0..3 {
        let x = margin + index * (panel + gap);
        draw_plane(
            &mut canvas,
            &planes[index],
            args.nx,
            args.ny,
            x,
            138,
            panel,
            panel,
            palettes[index],
            palettes[index].title(),
        );
    }
    canvas
}

fn render_vortex_composite(args: &Args, planes: &[Vec<f64>], record: usize) -> Canvas {
    let size = 900usize;
    let mut canvas = Canvas::new(1020, 1120, Rgb(9, 13, 23));
    canvas.rect(0, 0, 1020, 128, Rgb(17, 25, 42));
    canvas.text(38, 18, "GM1 VORTEX-FIELD COMPOSITE", 4, Rgb(245, 247, 251));
    let provenance = if args.label.is_empty() {
        format!("GM1 REAL STATE | {}", time_caption(args, record))
    } else {
        format!("{} | {}", args.label, time_caption(args, record))
    };
    canvas.text(40, 64, &provenance, 2, Rgb(143, 205, 255));
    let classification = if is_initialization(args) {
        "INITIALIZATION | NOT TORNADO | NOT FORECAST"
    } else {
        "EVOLVED SNAPSHOT | FIELD COMPOSITE | NO TORNADO DETECTION"
    };
    canvas.text(40, 92, classification, 2, Rgb(252, 190, 92));

    for py in 0..size {
        for px in 0..size {
            let reflectivity =
                reconstructed_pixel_value(&planes[0], args.nx, args.ny, px, py, size, size);
            let updraft =
                reconstructed_pixel_value(&planes[1], args.nx, args.ny, px, py, size, size);
            let zeta = reconstructed_pixel_value(&planes[2], args.nx, args.ny, px, py, size, size);
            let color = match (reflectivity, updraft, zeta) {
                (Some(reflectivity), Some(updraft), Some(zeta)) => {
                    vortex_composite_color(reflectivity, updraft, zeta)
                }
                _ => Rgb(90, 90, 90),
            };
            canvas.set((60 + px) as i32, (146 + py) as i32, color);
        }
    }
    draw_panel_border(&mut canvas, 60, 146, size, size);
    canvas.text(
        60,
        1063,
        "FIXED CHANNELS: REFL -10..50 DBZ | UPDRAFT 0..30 M/S | ZETA +/-0.025 1/S",
        1,
        Rgb(210, 217, 229),
    );
    canvas.text(
        60,
        1083,
        "BILINEAR 2X2 DISPLAY RECONSTRUCTION | SOURCE VALUES UNCHANGED",
        1,
        Rgb(143, 205, 255),
    );
    canvas
}

fn render_storm_fusion(args: &Args, planes: &[Vec<f64>], record: usize) -> Canvas {
    let size = 900usize;
    let mut canvas = Canvas::new(1020, 1160, Rgb(9, 13, 23));
    canvas.rect(0, 0, 1020, 128, Rgb(17, 25, 42));
    canvas.text(38, 18, "GM1 STORM-FIELD FUSION", 4, Rgb(245, 247, 251));
    let provenance = if args.label.is_empty() {
        format!("GM1 REAL STATE | {}", time_caption(args, record))
    } else {
        format!("{} | {}", args.label, time_caption(args, record))
    };
    canvas.text(40, 64, &provenance, 2, Rgb(143, 205, 255));
    let classification = if is_initialization(args) {
        "INITIALIZATION | NOT TORNADO | NOT FORECAST"
    } else {
        "EVOLVED SNAPSHOT | CONTINUOUS FIELD FUSION | NO TORNADO DETECTION"
    };
    canvas.text(40, 92, classification, 2, Rgb(252, 190, 92));

    for py in 0..size {
        for px in 0..size {
            let reflectivity =
                reconstructed_pixel_value(&planes[0], args.nx, args.ny, px, py, size, size);
            let updraft =
                reconstructed_pixel_value(&planes[1], args.nx, args.ny, px, py, size, size);
            let zeta = reconstructed_pixel_value(&planes[2], args.nx, args.ny, px, py, size, size);
            let color = match (reflectivity, updraft, zeta) {
                (Some(reflectivity), Some(updraft), Some(zeta)) => {
                    storm_fusion_color(reflectivity, updraft, zeta)
                }
                _ => Rgb(90, 90, 90),
            };
            canvas.set((60 + px) as i32, (146 + py) as i32, color);
        }
    }
    draw_panel_border(&mut canvas, 60, 146, size, size);
    canvas.rect(60, 1064, 18, 18, Rgb(17, 190, 101));
    canvas.text(86, 1068, "REFLECTIVITY -10..35 DBZ", 1, Rgb(210, 217, 229));
    canvas.rect(310, 1064, 18, 18, Rgb(0, 224, 235));
    canvas.text(336, 1068, "UPDRAFT 0..8 M/S", 1, Rgb(210, 217, 229));
    canvas.rect(540, 1064, 18, 18, Rgb(157, 0, 31));
    canvas.text(566, 1068, "CYCLONIC 0..0.010 1/S", 1, Rgb(210, 217, 229));
    canvas.rect(800, 1064, 18, 18, Rgb(19, 86, 202));
    canvas.text(826, 1068, "ANTI 0..0.010 1/S", 1, Rgb(210, 217, 229));
    canvas.text(
        60,
        1102,
        "FIXED PHYSICAL CHANNELS | NO PER-FRAME NORMALIZATION | SOURCE VALUES UNCHANGED",
        1,
        Rgb(210, 217, 229),
    );
    canvas.text(
        60,
        1122,
        "BILINEAR 2X2 DISPLAY RECONSTRUCTION | SINGLE CONTINUOUS CANVAS",
        1,
        Rgb(143, 205, 255),
    );
    canvas
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                0xedb8_8320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in bytes {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn push_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(data);
    png.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut offset = 0usize;
    while offset < raw.len() {
        let length = (raw.len() - offset).min(65_535);
        let final_block = offset + length == raw.len();
        out.push(if final_block { 1 } else { 0 });
        out.extend_from_slice(&(length as u16).to_le_bytes());
        out.extend_from_slice(&(!(length as u16)).to_le_bytes());
        out.extend_from_slice(&raw[offset..offset + length]);
        offset += length;
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

fn png_bytes(canvas: &Canvas, metadata: &[(&str, String)]) -> Vec<u8> {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(canvas.width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(canvas.height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    push_chunk(&mut png, b"IHDR", &ihdr);
    for (key, value) in metadata {
        let mut text = key.as_bytes().to_vec();
        text.push(0);
        text.extend(value.bytes().filter(|b| *b != 0));
        push_chunk(&mut png, b"tEXt", &text);
    }
    let mut scanlines = Vec::with_capacity(canvas.height * (1 + canvas.width * 3));
    for row in canvas.pixels.chunks_exact(canvas.width * 3) {
        scanlines.push(0);
        scanlines.extend_from_slice(row);
    }
    push_chunk(&mut png, b"IDAT", &zlib_stored(&scanlines));
    push_chunk(&mut png, b"IEND", &[]);
    png
}

fn render_metadata(
    args: &Args,
    record: usize,
    ranges: &[Option<(f64, f64)>],
) -> Vec<(&'static str, String)> {
    let mut metadata = vec![
        ("Title", format!("GM1 {}", args.product)),
        ("Model", "GM1".to_string()),
        ("MathAuthority", MATH_AUTHORITY.to_string()),
        ("Renderer", ABI.to_string()),
        ("Source", args.input.display().to_string()),
        ("Fields", args.fields.join(",")),
        ("Record", record.to_string()),
        ("Grid", format!("{}x{}x{}", args.nx, args.ny, args.nz)),
        ("FiniteRanges", format!("{ranges:?}")),
        ("Visualization", VISUAL_RECONSTRUCTION.to_string()),
        (
            "StateClassification",
            if is_initialization(args) {
                "initialization; not tornado; not forecast".to_string()
            } else {
                "evolved snapshot; diagnostic fields; no tornado detection".to_string()
            },
        ),
        (
            "Claim",
            "real GM1 state; no synthetic atmospheric input, vortex insertion, tornado detection, or source-array mutation; interpolation and colour compositing are visualization-only".to_string(),
        ),
    ];
    if args.product == "vortex-composite" {
        metadata.push((
            "CompositeMapping",
            "fixed physical channels: reflectivity -10..50 dBZ green/yellow; positive updraft 0..30 m/s cyan; signed vertical vorticity +/-0.025 s-1 blue/red; no per-frame normalization".to_string(),
        ));
    } else if args.product == "storm-fusion" {
        metadata.push((
            "FusionMapping",
            "fixed physical channels: reflectivity -10..35 dBZ green; positive updraft 0..8 m/s cyan; signed vertical vorticity +/-0.010 s-1 blue/red; no per-frame normalization".to_string(),
        ));
    } else if matches!(
        args.product.as_str(),
        "updraft-helicity" | "swath-wind" | "swath-vorticity" | "swath-pressure"
            | "swath-helicity"
    ) {
        metadata.push((
            "DiagnosticAuthority",
            "CM1 r21.1 src/misclibs.F:1417-1532 calcuh (2-5 km AGL updraft helicity) and src/solve3.F:855-1136 the lowest-model-level running swaths; colour bounds are fixed physical scales, not per-frame normalization".to_string(),
        ));
        metadata.push((
            "DiagnosticClaim",
            "a rendered CM1 diagnostic field; updraft helicity is a mesocyclone proxy, not a tornado detection, and a surface swath is a running extremum over the whole run whose wind colour is not an EF rating".to_string(),
        ));
    }
    if let Some(contract_sha256) = &args.contract_sha256 {
        metadata.push(("ForecastContractSHA256", contract_sha256.clone()));
    }
    metadata
}

fn run(mut args: Args) -> Result<(), String> {
    let decoded_fields = if matches!(args.format.as_str(), "cm1fhs" | "cm1fcp") {
        let decoded = decode_cm1_forecast(&args)?;
        if args.valid_seconds.is_none() {
            args.valid_seconds = Some(decoded.model_time_s);
        }
        decoded.fields
    } else {
        args.fields
            .iter()
            .map(|field| decode_field(&args, field))
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut volumes = Vec::new();
    let mut records = Vec::new();
    for decoded in decoded_fields {
        let (volume, record) = select_record(&decoded, &args)?;
        volumes.push(volume);
        records.push(record);
    }
    if records.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err("decoded fields selected different record numbers".into());
    }
    let record = records[0];
    let (canvas, ranges) = if matches!(
        args.product.as_str(),
        "supercell" | "vortex-composite" | "storm-fusion"
    ) {
        let planes = vec![
            plane_max(&volumes[0], args.nx, args.ny, args.nz),
            plane_max(&volumes[1], args.nx, args.ny, args.nz),
            plane_level(&volumes[2], args.nx, args.ny, args.level),
        ];
        let ranges = planes.iter().map(|p| finite_range(p)).collect::<Vec<_>>();
        let canvas = match args.product.as_str() {
            "supercell" => render_supercell(&args, &planes, record),
            "storm-fusion" => render_storm_fusion(&args, &planes, record),
            _ => render_vortex_composite(&args, &planes, record),
        };
        (canvas, ranges)
    } else {
        let palette = product_palette(&args.product);
        let plane = reduce(&args.product, &volumes[0], &args);
        let range = finite_range(&plane);
        (render_single(&args, &plane, palette, record), vec![range])
    };
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create output directory {}: {e}", parent.display()))?;
    }
    let temporary = args
        .output
        .with_extension(format!("png.partial-{}", std::process::id()));
    let metadata = render_metadata(&args, record, &ranges);
    let bytes = png_bytes(&canvas, &metadata);
    let mut file =
        File::create(&temporary).map_err(|e| format!("create {}: {e}", temporary.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| format!("write {}: {e}", temporary.display()))?;
    fs::rename(&temporary, &args.output).map_err(|e| {
        format!(
            "publish {} -> {}: {e}",
            temporary.display(),
            args.output.display()
        )
    })?;
    println!(
        "rendered {} bytes to {} ({ABI}, record {record})",
        bytes.len(),
        args.output.display()
    );
    Ok(())
}

fn main() {
    match parse_args(env::args_os()).and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("cm1-render: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cm1_state::forecast::{
        write_forecast_checkpoint_new, write_forecast_history_new, ForecastArray,
        ForecastArraySpec, ForecastClock, ForecastContract, ForecastFrame,
    };

    fn vortex_args(valid_seconds: &str) -> Args {
        parse_args(
            [
                "cm1-render",
                "--input",
                "real.cm1fhs",
                "--format",
                "cm1fhs",
                "--contract-sha256",
                "1111111111111111111111111111111111111111111111111111111111111111",
                "--output",
                "real.png",
                "--product",
                "vortex-composite",
                "--field",
                "reflectivity_dbz",
                "--field",
                "updraft_ms",
                "--field",
                "vertical_vorticity_s",
                "--nx",
                "2",
                "--ny",
                "2",
                "--nz",
                "1",
                "--label",
                "CORRECTED-REAL STATE",
                "--valid-seconds",
                valid_seconds,
            ]
            .into_iter()
            .map(OsString::from),
        )
        .unwrap()
    }

    fn environment_args(product: &str) -> Args {
        parse_args(
            [
                "cm1-render",
                "--input",
                "frame.cm1fhs",
                "--format",
                "cm1fhs",
                "--contract-sha256",
                "1111111111111111111111111111111111111111111111111111111111111111",
                "--output",
                "frame.png",
                "--product",
                product,
                "--field",
                product,
                "--nx",
                "4",
                "--ny",
                "3",
                "--nz",
                "1",
            ]
            .into_iter()
            .map(OsString::from),
        )
        .unwrap()
    }

    #[test]
    fn every_cm1_environment_plane_has_its_own_palette_and_fixed_scale() {
        // CM1 writeout.F:2937-3028 writes four 2-D planes.  Each has to reach a
        // palette of its own: falling through to Palette::Scalar would give it
        // a per-frame normalization, and a weakening inflow would then look
        // identical from one frame to the next, which is the one thing these
        // fields exist to show.
        for (product, units, upper) in [
            ("cape", "J/KG", 6000.0_f64),
            ("cin", "J/KG", 300.0),
            ("lcl", "M AGL", 4000.0),
            ("lfc", "M AGL", 4000.0),
        ] {
            let args = environment_args(product);
            assert_eq!(args.product, product);
            assert_eq!(args.nz, 1, "an environment plane is one value per column");
            let palette = product_palette(product);
            assert_ne!(
                std::mem::discriminant(&palette),
                std::mem::discriminant(&Palette::Scalar),
                "{product} fell through to the generic scalar palette"
            );
            assert_eq!(palette.units(), units);
            // The bounds must not depend on the frame's own values.
            let low = palette.bounds(&[0.0, 1.0]);
            let high = palette.bounds(&[1.0e6, -1.0e6]);
            assert_eq!(low, high, "{product} normalizes per frame");
            assert_eq!(low, (0.0, upper));
            // And the level reduction reads plane zero rather than a column max.
            let plane = reduce(product, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &Args {
                nx: 3,
                ny: 2,
                nz: 1,
                level: 0,
                ..environment_args(product)
            });
            assert_eq!(plane, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        }
    }

    #[test]
    fn an_unknown_product_is_still_refused_by_name() {
        let error = parse_args(
            [
                "cm1-render",
                "--input",
                "frame.cm1fhs",
                "--format",
                "cm1fhs",
                "--contract-sha256",
                "1111111111111111111111111111111111111111111111111111111111111111",
                "--output",
                "frame.png",
                "--product",
                "srh",
                "--field",
                "srh",
                "--nx",
                "4",
                "--ny",
                "3",
                "--nz",
                "1",
            ]
            .into_iter()
            .map(OsString::from),
        )
        .unwrap_err();
        assert!(error.contains("unknown product"), "{error}");
    }

    #[test]
    fn crc_matches_png_standard_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn adler_matches_standard_vector() {
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
    }

    #[test]
    fn max_reduction_respects_vertical_layout() {
        let volume = vec![1.0, 8.0, 3.0, 4.0, 9.0, 2.0, 7.0, 6.0];
        assert_eq!(plane_max(&volume, 2, 2, 2), vec![9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn encoded_png_has_signature_and_iend() {
        let canvas = Canvas::new(2, 2, Rgb(1, 2, 3));
        let png = png_bytes(&canvas, &[("Title", "test".into())]);
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert!(png.windows(4).any(|word| word == b"IEND"));
    }

    #[test]
    fn nws_reflectivity_scale_is_the_operational_table() {
        // The bin edges are what a forecaster reads: a value exactly on an
        // edge takes that bin's colour, and everything below 5 dBZ is clear
        // air, not a weak return.
        assert_eq!(nws_reflectivity_color(-1000.0), NWS_REFLECTIVITY_BELOW_FLOOR);
        assert_eq!(nws_reflectivity_color(4.999), NWS_REFLECTIVITY_BELOW_FLOOR);
        assert_eq!(nws_reflectivity_color(5.0), Rgb(4, 233, 231));
        assert_eq!(nws_reflectivity_color(34.999), Rgb(0, 142, 0));
        assert_eq!(nws_reflectivity_color(35.0), Rgb(253, 248, 2));
        assert_eq!(nws_reflectivity_color(50.0), Rgb(253, 0, 0));
        assert_eq!(nws_reflectivity_color(74.0), Rgb(152, 84, 198));
        assert_eq!(nws_reflectivity_color(120.0), Rgb(255, 255, 255));
        assert_eq!(nws_reflectivity_color(f64::NAN), Rgb(90, 90, 90));
        // And the palette itself uses it for reflectivity, in raw dBZ.
        let bounds = Palette::Reflectivity.bounds(&[]);
        assert_eq!(
            Palette::Reflectivity.color(47.0, bounds),
            nws_reflectivity_color(47.0)
        );
        // A 0.5 dBZ step across an edge changes colour; inside a bin it does not.
        assert_ne!(
            Palette::Reflectivity.color(39.9, bounds),
            Palette::Reflectivity.color(40.1, bounds)
        );
        assert_eq!(
            Palette::Reflectivity.color(41.0, bounds),
            Palette::Reflectivity.color(44.0, bounds)
        );
    }

    #[test]
    fn nonfinite_values_get_explicit_colour() {
        assert_eq!(Palette::Scalar.color(f64::NAN, (0.0, 1.0)), Rgb(90, 90, 90));
    }

    #[test]
    fn reflectivity_is_painted_with_the_published_nws_class_table() {
        // The scale is the product: a forecaster reads the class boundary,
        // not a gradient. Every class edge takes its own published colour,
        // the whole class holds it, and nothing between two classes blends.
        let bounds = Palette::Reflectivity.bounds(&[]);
        for (dbz, expected) in NWS_DBZ_CLASSES {
            let at_edge = Palette::Reflectivity.color(dbz as f64, bounds);
            assert_eq!(at_edge, Rgb(expected[0], expected[1], expected[2]));
            let inside = Palette::Reflectivity.color(dbz as f64 + 4.9, bounds);
            assert_eq!(inside, at_edge, "class starting at {dbz} dBZ is not flat");
            assert_eq!(Palette::ReflectivityLevel.color(dbz as f64, bounds), at_edge);
        }
        // 20 dBZ green, 40 dBZ amber, 50 dBZ red, 65 dBZ magenta: the four a
        // reader identifies a convective core by.
        assert_eq!(Palette::Reflectivity.color(22.0, bounds), Rgb(0x02, 0xFD, 0x02));
        assert_eq!(Palette::Reflectivity.color(41.0, bounds), Rgb(0xE5, 0xBC, 0x00));
        assert_eq!(Palette::Reflectivity.color(52.0, bounds), Rgb(0xFD, 0x00, 0x00));
        assert_eq!(Palette::Reflectivity.color(66.0, bounds), Rgb(0xF8, 0x00, 0xFD));
        // The top class holds above its edge instead of running off the table.
        assert_eq!(
            Palette::Reflectivity.color(120.0, bounds),
            Rgb(0xFD, 0xFD, 0xFD)
        );
    }

    #[test]
    fn weak_reflectivity_is_no_echo_and_nonfinite_is_not() {
        let bounds = Palette::Reflectivity.bounds(&[]);
        // Below the published floor the product shows no echo. Painting the
        // weakest class there would put a light-rain colour over clear air.
        for dbz in [-35.0, -1.0, 0.0, 4.999] {
            assert_eq!(
                Palette::Reflectivity.color(dbz, bounds),
                Rgb(NO_ECHO[0], NO_ECHO[1], NO_ECHO[2]),
                "{dbz} dBZ should read as no echo"
            );
        }
        // A non-finite word is a defect, not clear air, and must stay visible.
        assert_eq!(Palette::Reflectivity.color(f64::NAN, bounds), Rgb(90, 90, 90));
    }

    #[test]
    fn reflectivity_products_reduce_the_volume_the_way_cm1_writes_it() {
        // CM1 writes the composite as the column maximum (misclibs.F:2259-2292)
        // and the volume whole (writeout.F:4089-4107) for a user to slice.
        let args = Args {
            input: PathBuf::from("x"),
            decoder: None,
            contract_sha256: None,
            format: "netcdf".to_string(),
            output: PathBuf::from("y"),
            product: "reflectivity".to_string(),
            fields: vec!["dbz".to_string()],
            nx: 2,
            ny: 1,
            nz: 3,
            record: None,
            level: 1,
            label: String::new(),
            valid_seconds: None,
        };
        let volume = vec![10.0, 60.0, 45.0, 5.0, 20.0, 15.0];
        assert_eq!(reduce("reflectivity", &volume, &args), vec![45.0, 60.0]);
        assert_eq!(reduce("reflectivity-level", &volume, &args), vec![45.0, 5.0]);
        assert!(matches!(
            product_palette("reflectivity-level"),
            Palette::ReflectivityLevel
        ));
    }

    #[test]
    fn bilinear_reconstruction_is_smooth_bounded_and_does_not_mutate_source() {
        let source = vec![0.0, 10.0, 20.0, 30.0];
        let preserved = source.clone();
        assert_eq!(bilinear_sample(&source, 2, 2, 0.5, 0.5), Some(15.0));
        let mut distinct = std::collections::BTreeSet::new();
        for py in 0..8 {
            for px in 0..8 {
                let value = reconstructed_pixel_value(&source, 2, 2, px, py, 8, 8).unwrap();
                assert!((0.0..=30.0).contains(&value));
                distinct.insert(value.to_bits());
            }
        }
        assert!(
            distinct.len() > source.len(),
            "reconstruction stayed blockwise constant"
        );
        assert_eq!(
            source, preserved,
            "visual reconstruction changed source values"
        );
    }

    #[test]
    fn bilinear_reconstruction_propagates_nonfinite_source_corners() {
        let source = vec![0.0, f64::NAN, 20.0, 30.0];
        assert_eq!(bilinear_sample(&source, 2, 2, 0.0, 0.0), Some(0.0));
        assert_eq!(bilinear_sample(&source, 2, 2, 0.5, 0.5), None);
    }

    #[test]
    fn vortex_composite_requires_actual_rotation_for_maroon_core() {
        let weak = vortex_composite_color(10.0, 0.05, 0.0005);
        let strong = vortex_composite_color(10.0, 0.05, 0.0200);
        assert!(
            weak.1 > weak.0,
            "weak initialized state was made red: {weak:?}"
        );
        assert!(
            strong.0 > strong.1,
            "resolved positive rotation was not red: {strong:?}"
        );
        assert!(strong.0 > weak.0 && strong.1 < weak.1);
    }

    #[test]
    fn storm_fusion_is_continuous_fixed_scale_and_signed() {
        let neutral = storm_fusion_color(10.0, 1.0, 0.0);
        let cyclonic = storm_fusion_color(10.0, 1.0, 0.008);
        let anticyclonic = storm_fusion_color(10.0, 1.0, -0.008);
        assert!(
            neutral.1 > neutral.0,
            "neutral storm field is not green/cyan"
        );
        assert!(cyclonic.0 > cyclonic.1, "cyclonic rotation is not red");
        assert!(
            anticyclonic.2 > anticyclonic.0,
            "anticyclonic rotation is not blue"
        );

        let mut args = vortex_args("600");
        args.product = "storm-fusion".to_string();
        let metadata = render_metadata(
            &args,
            1200,
            &[Some((-10.0, 20.0)), Some((-1.0, 3.0)), Some((-0.01, 0.01))],
        );
        let mapping = metadata
            .iter()
            .find(|(key, _)| *key == "FusionMapping")
            .map(|(_, value)| value.as_str())
            .unwrap();
        assert!(mapping.contains("no per-frame normalization"));
    }

    #[test]
    fn vortex_composite_contract_and_t0_label_are_explicit() {
        let args = vortex_args("0");
        assert_eq!(args.fields.len(), 3);
        assert_eq!(
            standard_subtitle(&args, 0),
            "GM1 INITIALIZATION | T+0 S | NOT TORNADO | NOT FORECAST"
        );
        let metadata = render_metadata(
            &args,
            0,
            &[
                Some((-35.0, 10.0)),
                Some((-0.08, 0.06)),
                Some((-0.0004, 0.0005)),
            ],
        );
        let value = |key: &str| {
            metadata
                .iter()
                .find(|(candidate, _)| *candidate == key)
                .map(|(_, value)| value.as_str())
                .unwrap()
        };
        assert_eq!(value("Title"), "GM1 vortex-composite");
        assert_eq!(value("MathAuthority"), MATH_AUTHORITY);
        assert!(value("Visualization").contains("visualization-only"));
        assert_eq!(
            value("StateClassification"),
            "initialization; not tornado; not forecast"
        );
        assert!(value("CompositeMapping").contains("no per-frame normalization"));
    }

    fn swath_args(product: &str, field: &str) -> Args {
        parse_args(
            [
                "cm1-render",
                "--input",
                "history.cm1fhs",
                "--format",
                "cm1fhs",
                "--contract-sha256",
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "--output",
                "swath.png",
                "--product",
                product,
                "--field",
                field,
                "--nx",
                "60",
                "--ny",
                "60",
                "--nz",
                "1",
                "--valid-seconds",
                "3600",
            ]
            .into_iter()
            .map(std::ffi::OsString::from),
        )
        .unwrap()
    }

    #[test]
    fn tornado_diagnostic_products_take_one_surface_field() {
        for (product, field) in [
            ("updraft-helicity", "uh"),
            ("swath-wind", "sws"),
            ("swath-vorticity", "svs"),
            ("swath-pressure", "sps"),
            ("swath-helicity", "shs"),
        ] {
            let args = swath_args(product, field);
            assert_eq!(args.fields, vec![field.to_string()]);
            assert_eq!(args.nz, 1);
            assert_eq!(args.level, 0);
        }
    }

    #[test]
    fn tornado_diagnostic_palettes_are_fixed_physical_scales() {
        // The bounds must not depend on the plane, or two frames of one run
        // would colour the same swath value differently and a reader would
        // see a track grow where only the maximum changed.
        let weak = vec![0.0f64; 16];
        let strong = vec![250.0f64; 16];
        assert_eq!(
            Palette::UpdraftHelicity.bounds(&weak),
            Palette::UpdraftHelicity.bounds(&strong)
        );
        assert_eq!(Palette::UpdraftHelicity.bounds(&weak), (0.0, 300.0));
        assert_eq!(Palette::SwathWind.bounds(&weak), (0.0, 90.0));
        assert_eq!(Palette::SwathVorticity.bounds(&weak), (0.0, 0.5));
        assert_eq!(Palette::SwathPressure.bounds(&weak), (-8000.0, 0.0));
        // A rotating-updraft value and a strong-rotation value must not map
        // to the same colour, or the classification is invisible.
        let bounds = Palette::UpdraftHelicity.bounds(&weak);
        assert_ne!(
            Palette::UpdraftHelicity.color(25.0, bounds),
            Palette::UpdraftHelicity.color(100.0, bounds)
        );
    }

    #[test]
    fn tornado_diagnostic_metadata_refuses_a_tornado_claim() {
        let args = swath_args("swath-vorticity", "svs");
        let metadata = render_metadata(&args, 0, &[Some((0.0, 0.4))]);
        let value = |key: &str| {
            metadata
                .iter()
                .find(|(candidate, _)| *candidate == key)
                .map(|(_, value)| value.as_str())
                .unwrap()
        };
        assert!(value("DiagnosticAuthority").contains("src/misclibs.F:1417-1532"));
        assert!(value("DiagnosticAuthority").contains("src/solve3.F:855-1136"));
        assert!(value("DiagnosticClaim").contains("not a tornado detection"));
        assert!(value("DiagnosticClaim").contains("not an EF rating"));
    }

    #[test]
    fn subsecond_model_time_is_not_rounded_to_initialization() {
        let args = vortex_args("0.5");
        assert_eq!(time_caption(&args, 0), "T+0.5 S");
        assert_eq!(
            standard_subtitle(&args, 0),
            "CORRECTED-REAL STATE | T+0.5 S"
        );
        assert!(!is_initialization(&args));
    }

    #[test]
    fn scalar_grads_diagnostics_accept_one_by_one_grids() {
        let args = parse_args(
            [
                "cm1-render",
                "--input",
                "stats.ctl",
                "--format",
                "grads",
                "--output",
                "stats.png",
                "--product",
                "scalar",
                "--field",
                "wmax",
                "--nx",
                "1",
                "--ny",
                "1",
                "--nz",
                "1",
            ]
            .into_iter()
            .map(OsString::from),
        )
        .expect("one-cell scalar diagnostics are valid");
        assert_eq!((args.nx, args.ny, args.nz), (1, 1, 1));
    }

    #[test]
    fn cm1_forecast_decode_requires_and_checks_external_contract_hash() {
        let temporary = TempDir::create().unwrap();
        let history = temporary.path.join("real.cm1fhs");
        let checkpoint = temporary.path.join("real.cm1fcp");
        let contract = ForecastContract::new(
            "renderer-history-test/v1",
            "11".repeat(32),
            "22".repeat(32),
            vec![ForecastArraySpec::new(
                "refc_dbz",
                "direct composite reflectivity",
                "dBZ",
                "scalar-surface-yx",
                vec![1, 2, 1, 2],
                vec![2, 2],
            )
            .unwrap()],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let frame = ForecastFrame::new(
            ForecastClock::new(7200, 3600.0, 0.5).unwrap(),
            vec![ForecastArray::new("refc_dbz", vec![10.0, 20.0, 30.0, 40.0])],
            Vec::new(),
            Vec::new(),
        );
        write_forecast_history_new(&contract, &frame, &history).unwrap();
        write_forecast_checkpoint_new(&contract, &frame, &checkpoint).unwrap();

        let make_args = |input: &std::path::Path, format: &str, digest: &str| {
            parse_args([
                OsString::from("cm1-render"),
                OsString::from("--input"),
                input.as_os_str().to_owned(),
                OsString::from("--format"),
                OsString::from(format),
                OsString::from("--contract-sha256"),
                OsString::from(digest),
                OsString::from("--output"),
                temporary.path.join("out.png").as_os_str().to_owned(),
                OsString::from("--product"),
                OsString::from("reflectivity"),
                OsString::from("--field"),
                OsString::from("refc_dbz"),
                OsString::from("--nx"),
                OsString::from("2"),
                OsString::from("--ny"),
                OsString::from("2"),
                OsString::from("--nz"),
                OsString::from("1"),
            ])
            .unwrap()
        };
        for (input, format) in [
            (history.as_path(), "cm1fhs"),
            (checkpoint.as_path(), "cm1fcp"),
        ] {
            let decoded =
                decode_cm1_forecast(&make_args(input, format, &contract.sha256())).unwrap();
            assert_eq!(decoded.model_time_s, 3600.0);
            assert_eq!(decoded.fields, vec![vec![10.0, 20.0, 30.0, 40.0]]);
        }

        let error =
            decode_cm1_forecast(&make_args(&history, "cm1fhs", &"ee".repeat(32))).unwrap_err();
        assert!(error.contains("independently expected"), "{error}");
    }

    #[test]
    fn a_reduced_gm1_history_renders_the_carriers_it_omits() {
        // The reduced GM1 history profile publishes base.pi0 and
        // metrics.zf_profile instead of prs and zf.  The renderer must open
        // such a frame and hand back the same fields it would for a complete
        // one, rebuilt by the recorded CM1 expression -- otherwise the profile
        // is a format nothing in this distribution can read.
        use cm1_state::reduced_history::{CPDRD, P00_PA, REDUCED_HISTORY_PROFILE_ID};

        let temporary = TempDir::create().unwrap();
        let history = temporary.path.join("reduced.cm1fhs");
        let (nz, ny, nx) = (2usize, 2usize, 2usize);
        let column = |name: &str, units: &str, stagger: &str, levels: usize| {
            ForecastArraySpec::new(
                name,
                name,
                units,
                stagger,
                vec![1, i32::try_from(levels).unwrap()],
                vec![levels],
            )
            .unwrap()
        };
        let pi0 = vec![0.95_f32, 0.85];
        let zf_profile = vec![0.0_f32, 250.0, 500.0];
        let pp3d = vec![0.01_f32, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08];
        let contract = ForecastContract::new(
            REDUCED_HISTORY_PROFILE_ID,
            "11".repeat(32),
            "22".repeat(32),
            vec![
                column("base.pi0", "nondimensional", "mass-column", nz),
                column("metrics.zf_profile", "m", "w-column", nz + 1),
                ForecastArraySpec::new(
                    "pp3d",
                    "current perturbation nondimensional pressure",
                    "nondimensional",
                    "mass",
                    vec![1, 2, 1, 2, 1, 2],
                    vec![nz, ny, nx],
                )
                .unwrap(),
            ],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let frame = ForecastFrame::new(
            ForecastClock::new(7200, 3600.0, 0.5).unwrap(),
            vec![
                ForecastArray::new("base.pi0", pi0.clone()),
                ForecastArray::new("metrics.zf_profile", zf_profile.clone()),
                ForecastArray::new("pp3d", pp3d.clone()),
            ],
            Vec::new(),
            Vec::new(),
        );
        write_forecast_history_new(&contract, &frame, &history).unwrap();

        let render_args = |field: &str, levels: usize| {
            parse_args([
                OsString::from("cm1-render"),
                OsString::from("--input"),
                history.as_os_str().to_owned(),
                OsString::from("--format"),
                OsString::from("cm1fhs"),
                OsString::from("--contract-sha256"),
                OsString::from(contract.sha256()),
                OsString::from("--output"),
                temporary.path.join(format!("{field}.png")).as_os_str().to_owned(),
                OsString::from("--product"),
                OsString::from("reflectivity"),
                OsString::from("--field"),
                OsString::from(field),
                OsString::from("--nx"),
                OsString::from("2"),
                OsString::from("--ny"),
                OsString::from("2"),
                OsString::from("--nz"),
                OsString::from(levels.to_string()),
            ])
            .unwrap()
        };

        let decoded = decode_cm1_forecast(&render_args("prs", nz)).unwrap();
        let expected = (0..nz * ny * nx)
            .map(|word| {
                let pi = pi0[word / (ny * nx)] + pp3d[word];
                f64::from(P00_PA * pi.powf(CPDRD))
            })
            .collect::<Vec<f64>>();
        assert_eq!(decoded.fields, vec![expected]);
        assert_eq!(decoded.model_time_s, 3600.0);

        let decoded = decode_cm1_forecast(&render_args("zf", nz + 1)).unwrap();
        let expected = zf_profile
            .iter()
            .flat_map(|height| std::iter::repeat_n(f64::from(*height), ny * nx))
            .collect::<Vec<f64>>();
        assert_eq!(decoded.fields, vec![expected]);

        // A carrier the reduced profile neither publishes nor rebuilds still
        // refuses, and the refusal says which ones it does rebuild.
        let error = decode_cm1_forecast(&render_args("rf", nz + 1)).unwrap_err();
        assert!(error.contains("reduced GM1 history profile"), "{error}");
        assert!(error.contains("prs"), "{error}");
    }
}
