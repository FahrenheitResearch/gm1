//! Derive mass-point horizontal wind speed from a CM1 forecast checkpoint.
//!
//! The checkpoint stores the horizontal momentum carriers on the CM1 C-grid:
//! the x-component lives on x-faces with shape `(nz, ny, nx + 1)` and the
//! y-component lives on y-faces with shape `(nz, ny + 1, nx)`.  This tool
//! averages each component onto the mass point that the scalar carriers already
//! occupy, forms the horizontal speed there, and writes one little-endian
//! float32 volume plus a GrADS control file so the existing renderer can draw
//! it without any Python transform.
//!
//! No interpolation beyond the exact two-point C-grid average is performed and
//! no vertical or horizontal regridding happens here.

use cm1_state::forecast::read_forecast_checkpoint_embedded;

use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// The handshake every packaged native tool answers, so the wheel build can
/// record which executable it copied and a later run can check it.  Without
/// it this binary cannot be registered in build_rust.NATIVE_TOOLS, and an
/// unregistered binary is compiled but never copied into a wheel -- present
/// in the source tree and unreachable from an install.
const ABI: &str = "gpuwm-cm1-checkpoint-wind-grads/v1";

fn usage() -> &'static str {
    "usage: checkpoint-wind-grads --input CHECKPOINT --contract-sha256 SHA256 \
--u-name NAME --v-name NAME --nx N --ny N --nz N \
--output-bin FILE --output-ctl FILE [--variable NAME]\n\n\
Writes one mass-point horizontal wind-speed volume (nz, ny, nx) as little-endian \
float32 together with a GrADS control file naming it."
}

struct Args {
    input: PathBuf,
    contract_sha256: String,
    u_name: String,
    v_name: String,
    nx: usize,
    ny: usize,
    nz: usize,
    output_bin: PathBuf,
    output_ctl: PathBuf,
    variable: String,
}

fn parse_usize(flag: &str, raw: &OsString) -> Result<usize, String> {
    raw.to_string_lossy()
        .parse::<usize>()
        .map_err(|_| format!("{flag} needs a non-negative integer"))
}

fn parse_args<I: IntoIterator<Item = OsString>>(values: I) -> Result<Args, String> {
    let mut it = values.into_iter();
    let _program = it.next();
    let mut input = None;
    let mut contract_sha256 = None;
    let mut u_name = None;
    let mut v_name = None;
    let mut nx = None;
    let mut ny = None;
    let mut nz = None;
    let mut output_bin = None;
    let mut output_ctl = None;
    let mut variable = "wspd".to_string();
    while let Some(flag) = it.next() {
        let key = flag.to_string_lossy().to_string();
        // Answered before the value is taken: --abi and --help carry none, and
        // demanding one would make the handshake the wheel build performs fail
        // with "needs a value" instead of answering.
        if key == "--abi" {
            println!("{ABI}");
            std::process::exit(0);
        }
        if key == "--help" || key == "-h" {
            println!("{}", usage());
            std::process::exit(0);
        }
        let value = it
            .next()
            .ok_or_else(|| format!("{key} needs a value\n\n{}", usage()))?;
        match key.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--contract-sha256" => contract_sha256 = Some(value.to_string_lossy().to_string()),
            "--u-name" => u_name = Some(value.to_string_lossy().to_string()),
            "--v-name" => v_name = Some(value.to_string_lossy().to_string()),
            "--nx" => nx = Some(parse_usize("--nx", &value)?),
            "--ny" => ny = Some(parse_usize("--ny", &value)?),
            "--nz" => nz = Some(parse_usize("--nz", &value)?),
            "--output-bin" => output_bin = Some(PathBuf::from(value)),
            "--output-ctl" => output_ctl = Some(PathBuf::from(value)),
            "--variable" => variable = value.to_string_lossy().to_string(),
            other => return Err(format!("unknown flag {other:?}\n\n{}", usage())),
        }
    }
    let digest = contract_sha256.ok_or_else(|| "missing --contract-sha256".to_string())?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("--contract-sha256 must be 64 lowercase hexadecimal characters".into());
    }
    if variable.is_empty() || !variable.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return Err(
            "--variable must be a non-empty alphanumeric GrADS name; a name carrying \
spaces or punctuation would produce a control file whose variable row cannot be \
matched by the renderer"
                .into(),
        );
    }
    Ok(Args {
        input: input.ok_or_else(|| "missing --input".to_string())?,
        contract_sha256: digest,
        u_name: u_name.ok_or_else(|| "missing --u-name".to_string())?,
        v_name: v_name.ok_or_else(|| "missing --v-name".to_string())?,
        nx: nx.ok_or_else(|| "missing --nx".to_string())?,
        ny: ny.ok_or_else(|| "missing --ny".to_string())?,
        nz: nz.ok_or_else(|| "missing --nz".to_string())?,
        output_bin: output_bin.ok_or_else(|| "missing --output-bin".to_string())?,
        output_ctl: output_ctl.ok_or_else(|| "missing --output-ctl".to_string())?,
        variable,
    })
}

fn run(args: &Args) -> Result<String, String> {
    let (contract, restored) = read_forecast_checkpoint_embedded(&args.input, &args.contract_sha256)
        .map_err(|error| format!("strict CM1 checkpoint decode failed: {error}"))?;
    let shape_of = |name: &str| -> Result<Vec<usize>, String> {
        contract
            .arrays()
            .iter()
            .find(|spec| spec.name() == name)
            .map(|spec| spec.shape().to_vec())
            .ok_or_else(|| format!("checkpoint contract declares no array {name:?}"))
    };
    let values_of = |name: &str| -> Result<&[f32], String> {
        restored
            .frame()
            .arrays()
            .iter()
            .find(|array| array.name() == name)
            .map(|array| array.values())
            .ok_or_else(|| format!("checkpoint payload carries no array {name:?}"))
    };
    let (nx, ny, nz) = (args.nx, args.ny, args.nz);
    let u_shape = shape_of(&args.u_name)?;
    let v_shape = shape_of(&args.v_name)?;
    if u_shape != vec![nz, ny, nx + 1] {
        return Err(format!(
            "array {:?} has shape {u_shape:?}, not the C-grid x-face shape [{nz}, {ny}, {}]; \
averaging a differently staggered array would pair values from two different columns and \
yield a wind-speed field displaced by half a grid cell",
            args.u_name,
            nx + 1
        ));
    }
    if v_shape != vec![nz, ny + 1, nx] {
        return Err(format!(
            "array {:?} has shape {v_shape:?}, not the C-grid y-face shape [{nz}, {}, {nx}]; \
averaging a differently staggered array would pair values from two different rows and \
yield a wind-speed field displaced by half a grid cell",
            args.v_name,
            ny + 1
        ));
    }
    let u = values_of(&args.u_name)?;
    let v = values_of(&args.v_name)?;
    if u.len() != nz * ny * (nx + 1) || v.len() != nz * (ny + 1) * nx {
        return Err("checkpoint momentum payload length disagrees with its declared shape".into());
    }
    let cells = nx * ny;
    let mut speed = vec![0f32; cells * nz];
    let mut maximum = f32::NEG_INFINITY;
    let mut surface_maximum = f32::NEG_INFINITY;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let ua = u[(k * ny + j) * (nx + 1) + i];
                let ub = u[(k * ny + j) * (nx + 1) + i + 1];
                let va = v[(k * (ny + 1) + j) * nx + i];
                let vb = v[(k * (ny + 1) + j + 1) * nx + i];
                let um = 0.5f32 * (ua + ub);
                let vm = 0.5f32 * (va + vb);
                let value = (um * um + vm * vm).sqrt();
                if !value.is_finite() {
                    return Err(format!(
                        "mass-point wind speed at k={k}, j={j}, i={i} is not finite; \
rendering it would paint a non-physical plane as if it were model output"
                    ));
                }
                speed[(k * ny + j) * nx + i] = value;
                if value > maximum {
                    maximum = value;
                }
                if k == 0 && value > surface_maximum {
                    surface_maximum = value;
                }
            }
        }
    }
    let mut bytes = Vec::with_capacity(speed.len() * 4);
    for value in &speed {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&args.output_bin, &bytes)
        .map_err(|e| format!("write {}: {e}", args.output_bin.display()))?;
    let data_name = args
        .output_bin
        .file_name()
        .ok_or_else(|| "--output-bin has no file name".to_string())?
        .to_string_lossy()
        .to_string();
    let control = format!(
        "dset ^{data_name}\n\
options little_endian\n\
undef 9.999e20\n\
title mass-point horizontal wind speed derived from a CM1 forecast checkpoint\n\
xdef {nx} linear 0 1\n\
ydef {ny} linear 0 1\n\
zdef {nz} linear 0 1\n\
tdef 1 linear 00Z01JAN2000 1hr\n\
vars 1\n\
{} {nz} 99 mass-point horizontal wind speed (m s-1)\n\
endvars\n",
        args.variable
    );
    let mut handle = fs::File::create(&args.output_ctl)
        .map_err(|e| format!("create {}: {e}", args.output_ctl.display()))?;
    handle
        .write_all(control.as_bytes())
        .map_err(|e| format!("write {}: {e}", args.output_ctl.display()))?;
    Ok(format!(
        "wrote {} bytes to {} (variable {}, shape [{nz}, {ny}, {nx}], \
volume maximum {maximum} m s-1, level-0 maximum {surface_maximum} m s-1)",
        bytes.len(),
        args.output_bin.display(),
        args.variable
    ))
}

fn main() {
    let args = match parse_args(env::args_os()) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    match run(&args) {
        Ok(message) => println!("{message}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
