//! Strict Rust crop/downcast helper for decoded row-major weather fields.

use std::env;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

const ABI: &str = "gpuwm-cm1-f64-window/v1";

#[derive(Debug, PartialEq, Eq)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    nx: usize,
    ny: usize,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
}

fn usage() -> &'static str {
    "usage: cm1-f64-window INPUT.f64le OUTPUT.f32le NX NY I_START I_END J_START J_END"
}

fn parse_usize(raw: OsString, name: &str) -> Result<usize, String> {
    raw.to_string_lossy()
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a non-negative integer"))
}

fn parse_args<I>(values: I) -> Result<Args, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut values = values.into_iter();
    let _program = values.next();
    let raw = values.collect::<Vec<_>>();
    if raw.len() == 1 && raw[0] == "--abi" {
        println!("{ABI}");
        std::process::exit(0);
    }
    if raw.len() != 8 {
        return Err(usage().to_string());
    }
    let args = Args {
        input: PathBuf::from(&raw[0]),
        output: PathBuf::from(&raw[1]),
        nx: parse_usize(raw[2].clone(), "NX")?,
        ny: parse_usize(raw[3].clone(), "NY")?,
        i_start: parse_usize(raw[4].clone(), "I_START")?,
        i_end: parse_usize(raw[5].clone(), "I_END")?,
        j_start: parse_usize(raw[6].clone(), "J_START")?,
        j_end: parse_usize(raw[7].clone(), "J_END")?,
    };
    if args.nx == 0
        || args.ny == 0
        || args.i_start > args.i_end
        || args.j_start > args.j_end
        || args.i_end >= args.nx
        || args.j_end >= args.ny
    {
        return Err("window must be non-empty and inside NX by NY".to_string());
    }
    Ok(args)
}

fn run(args: Args) -> Result<(), String> {
    let metadata = fs::symlink_metadata(&args.input)
        .map_err(|error| format!("inspect {}: {error}", args.input.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("input must be a regular non-symlink file".to_string());
    }
    if args.output.exists() {
        return Err(format!("refusing to replace {}", args.output.display()));
    }
    let cells = args
        .nx
        .checked_mul(args.ny)
        .ok_or_else(|| "source cell count overflows usize".to_string())?;
    let expected_bytes = cells
        .checked_mul(8)
        .ok_or_else(|| "source byte count overflows usize".to_string())?;
    if metadata.len() != expected_bytes as u64 {
        return Err(format!(
            "input has {} bytes, expected {expected_bytes}",
            metadata.len()
        ));
    }
    let mut input = File::open(&args.input)
        .map_err(|error| format!("open {}: {error}", args.input.display()))?;
    let mut bytes = vec![0_u8; expected_bytes];
    input
        .read_exact(&mut bytes)
        .map_err(|error| format!("read {}: {error}", args.input.display()))?;
    let width = args.i_end - args.i_start + 1;
    let height = args.j_end - args.j_start + 1;
    let mut output = Vec::with_capacity(width * height * 4);
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for j in args.j_start..=args.j_end {
        for i in args.i_start..=args.i_end {
            let offset = (j * args.nx + i) * 8;
            let value = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            let narrowed = value as f32;
            if !value.is_finite() || !narrowed.is_finite() {
                return Err(format!(
                    "non-finite or FP32-overflowing value at i={i}, j={j}"
                ));
            }
            minimum = minimum.min(narrowed);
            maximum = maximum.max(narrowed);
            output.extend_from_slice(&narrowed.to_le_bytes());
        }
    }
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let temporary = args
        .output
        .with_extension(format!("f32le.partial-{}", std::process::id()));
    let result = (|| {
        let mut stream = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("create {}: {error}", temporary.display()))?;
        stream
            .write_all(&output)
            .and_then(|()| stream.sync_all())
            .map_err(|error| format!("write {}: {error}", temporary.display()))?;
        fs::hard_link(&temporary, &args.output)
            .map_err(|error| format!("publish {}: {error}", args.output.display()))?;
        Ok::<(), String>(())
    })();
    let _ = fs::remove_file(&temporary);
    result?;
    println!(
        "PASS {ABI} shape={}x{} range=[{minimum},{maximum}] output={}",
        width,
        height,
        args.output.display()
    );
    Ok(())
}

fn main() {
    match parse_args(env::args_os()).and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ERROR: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(values: &[&str]) -> Vec<OsString> {
        std::iter::once(OsString::from("cm1-f64-window"))
            .chain(values.iter().map(OsString::from))
            .collect()
    }

    #[test]
    fn parses_bounded_window() {
        let args = parse_args(words(&["in", "out", "10", "9", "2", "7", "3", "8"]))
            .expect("bounded window");
        assert_eq!(args.nx, 10);
        assert_eq!(args.j_end, 8);
    }

    #[test]
    fn refuses_out_of_bounds_and_extra_arguments() {
        assert!(parse_args(words(&["in", "out", "10", "9", "2", "10", "3", "8"])).is_err());
        assert!(parse_args(words(&["in", "out", "10", "9", "2", "7", "3", "8", "x"])).is_err());
    }
}
