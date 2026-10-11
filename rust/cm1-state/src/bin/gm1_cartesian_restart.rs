//! `gm1-cartesian-restart`: the durable restart artifact of GM1's Cartesian lane.
//!
//! CM1 writes its restart from `src/restart_write.F:10` after the large step
//! that reached `rsttim` (`src/cm1.F:3186,4246-4311`) and reads it back at
//! `src/restart_read.F:19` for `irst=1`/`rstnum` (`README.restart:28-49`).
//! This binary is the durable half for GM1: the caller transports one quiescent
//! little-endian binary32 arena in the declared inventory order, and Rust owns
//! the envelope, the identity binding and the atomic publication.

use cm1_state::cartesian_restart::{
    cartesian_restart_layout_json, read_cartesian_restart, restore_cartesian_restart_new,
    write_cartesian_restart_new, CartesianRestartInventory, CARTESIAN_RESTART_CARRIER_ABI,
};
use cm1_state::forecast::ForecastClock;
use cm1_state::{Result, StateError};
use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

const ABI: &str = "gm1.cartesian-restart-cli/v1";
const USAGE: &str = concat!(
    "usage:\n",
    "  gm1-cartesian-restart --abi\n",
    "  gm1-cartesian-restart layout INVENTORY.json\n",
    "  gm1-cartesian-restart build RAW_ARENA OUTPUT.cm1fcp RAW_SHA INVENTORY.json STEP \\\n",
    "      MODEL_TIME_F64_BITS DT_F64_BITS PHYSICS_STEP SOLVE3_STEP \\\n",
    "      DOMAINLOCX_F64_BITS DOMAINLOCY_F64_BITS MASS1_F64_BITS SCHEDULER.json\n",
    "  gm1-cartesian-restart inspect INPUT.cm1fcp CONTRACT_SHA ARTIFACT_SHA INVENTORY_SHA\n",
    "  gm1-cartesian-restart restore INPUT.cm1fcp OUTPUT_DIR CONTRACT_SHA ARTIFACT_SHA INVENTORY_SHA"
);

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn read_regular(path: &str, label: &str) -> Result<Vec<u8>> {
    let path = Path::new(path);
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
    fs::read(path).map_err(|error| {
        invalid(format!(
            "could not read {label} {}: {error}",
            path.display()
        ))
    })
}

fn inventory(path: &str) -> Result<CartesianRestartInventory> {
    CartesianRestartInventory::from_canonical_bytes(&read_regular(path, "restart inventory")?)
}

fn parse_u64(value: &str, label: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|_| invalid(format!("{label} must be an unsigned integer")))
}

fn parse_f64_bits(value: &str, label: &str) -> Result<f64> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{label} must be exactly 16 lower-case hexadecimal characters"
        )));
    }
    let bits = u64::from_str_radix(value, 16)
        .map_err(|_| invalid(format!("{label} is not a binary64 word")))?;
    let result = f64::from_bits(bits);
    if !result.is_finite() {
        return Err(invalid(format!(
            "{label} must encode a finite binary64 value"
        )));
    }
    Ok(result)
}

fn run(arguments: &[String]) -> Result<String> {
    match arguments {
        [flag] if flag == "--abi" => Ok(format!(
            "{ABI};carrier_abi={CARTESIAN_RESTART_CARRIER_ABI}"
        )),
        [command, manifest] if command == "layout" => {
            cartesian_restart_layout_json(&inventory(manifest)?)
        }
        [command, raw, output, raw_sha, manifest, step, model_time_bits, dt_bits, physics_step, solve3_step, domainlocx_bits, domainlocy_bits, mass1_bits, scheduler]
            if command == "build" =>
        {
            let step = parse_u64(step, "STEP")?;
            let model_time = parse_f64_bits(model_time_bits, "MODEL_TIME_F64_BITS")?;
            let dt = parse_f64_bits(dt_bits, "DT_F64_BITS")?;
            let physics = parse_u64(physics_step, "PHYSICS_STEP")?;
            let solve3 = parse_u64(solve3_step, "SOLVE3_STEP")?;
            let domainlocx = parse_f64_bits(domainlocx_bits, "DOMAINLOCX_F64_BITS")?;
            let domainlocy = parse_f64_bits(domainlocy_bits, "DOMAINLOCY_F64_BITS")?;
            let mass1 = parse_f64_bits(mass1_bits, "MASS1_F64_BITS")?;
            let scheduler = read_regular(scheduler, "scheduler blob")?;
            let receipt = write_cartesian_restart_new(
                raw,
                output,
                raw_sha,
                inventory(manifest)?,
                ForecastClock::new(step, model_time, dt)?,
                physics,
                solve3,
                domainlocx,
                domainlocy,
                mass1,
                &scheduler,
            )?;
            Ok(receipt.canonical_json())
        }
        [command, input, contract_sha, artifact_sha, inventory_sha] if command == "inspect" => {
            let restored =
                read_cartesian_restart(input, contract_sha, artifact_sha, inventory_sha)?;
            Ok(restored.forecast().receipt().canonical_json())
        }
        [command, input, output, contract_sha, artifact_sha, inventory_sha]
            if command == "restore" =>
        {
            let restored =
                read_cartesian_restart(input, contract_sha, artifact_sha, inventory_sha)?;
            restore_cartesian_restart_new(&restored, output)
        }
        _ => Err(invalid(USAGE)),
    }
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match run(&arguments) {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("gm1-cartesian-restart: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_abi_names_the_carrier_abi() {
        let output = run(&["--abi".to_string()]).unwrap();
        assert!(output.contains(ABI));
        assert!(output.contains(CARTESIAN_RESTART_CARRIER_ABI));
    }

    #[test]
    fn raw_f64_words_are_strict_and_exact() {
        assert_eq!(parse_f64_bits("3fe0000000000000", "dt").unwrap(), 0.5);
        assert!(parse_f64_bits("3FE0000000000000", "upper").is_err());
        assert!(parse_f64_bits("7ff8000000000000", "nan").is_err());
    }

    #[test]
    fn usage_is_reported_for_an_unknown_command() {
        assert!(run(&["unknown".to_string()]).is_err());
    }
}
