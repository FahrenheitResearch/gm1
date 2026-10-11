use cm1_state::hrrr::{Cm1TargetGrid, HrrrLambertGrid};
use cm1_state::hrrr_future::{
    publish_future_boundaries_new, FutureBoundarySeriesRequest, GM1_HRRR_FUTURE_BOUNDARY_ABI,
    GM1_HRRR_FUTURE_BOUNDARY_SERIES_SCHEMA,
};
use cm1_state::{Result, StateError};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = concat!(
    "usage:\n",
    "  gm1-hrrr-future-boundaries --abi\n",
    "  gm1-hrrr-future-boundaries RAW_ROOT COMPLETE_DECODED_ROOT STORM_SOURCE_ROOT ",
    "STORM_DUMP_F02 STORM_DUMP_F03 BASELINE_F01.cm1fhs NEW_OUTPUT_DIRECTORY ",
    "EXPECTED_CONTRACT_SHA256 EXPECTED_SHA256SUMS_SHA256 EXPECTED_STORM_RECEIPT_SHA256 ",
    "EXPECTED_STORM_F02_SHA256 EXPECTED_STORM_F03_SHA256 CENTER_LAT CENTER_LON ",
    "[NX NY NZ DX_M DY_M DZ_M MODEL_DT_S SOURCE_I SOURCE_J SOURCE_NX SOURCE_NY]"
);

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn parse<T: std::str::FromStr>(value: &str, label: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| invalid(format!("invalid {label} {value:?}: {error}")))
}

fn run(arguments: &[String]) -> Result<String> {
    if arguments == ["--abi"] {
        return Ok(format!(
            "{GM1_HRRR_FUTURE_BOUNDARY_ABI}\nreceipt_schema={GM1_HRRR_FUTURE_BOUNDARY_SERIES_SCHEMA}"
        ));
    }
    if arguments.len() != 14 && arguments.len() != 25 {
        return Err(invalid(USAGE));
    }
    let center_lat = parse::<f64>(&arguments[12], "CENTER_LAT")?;
    let center_lon = parse::<f64>(&arguments[13], "CENTER_LON")?;
    let (nx, ny, nz, dx, dy, dz, model_dt, source_i, source_j, source_nx, source_ny) =
        if arguments.len() == 25 {
            (
                parse(&arguments[14], "NX")?,
                parse(&arguments[15], "NY")?,
                parse(&arguments[16], "NZ")?,
                parse(&arguments[17], "DX_M")?,
                parse(&arguments[18], "DY_M")?,
                parse(&arguments[19], "DZ_M")?,
                parse(&arguments[20], "MODEL_DT_S")?,
                parse(&arguments[21], "SOURCE_I")?,
                parse(&arguments[22], "SOURCE_J")?,
                parse(&arguments[23], "SOURCE_NX")?,
                parse(&arguments[24], "SOURCE_NY")?,
            )
        } else {
            (300, 300, 75, 100.0, 100.0, 200.0, 0.5, 971, 622, 26, 26)
        };
    let request = FutureBoundarySeriesRequest {
        raw_source_root: PathBuf::from(&arguments[0]),
        decoded_root: PathBuf::from(&arguments[1]),
        storm_source_root: PathBuf::from(&arguments[2]),
        storm_dump_f02: PathBuf::from(&arguments[3]),
        storm_dump_f03: PathBuf::from(&arguments[4]),
        baseline_f01_history: PathBuf::from(&arguments[5]),
        output_directory: PathBuf::from(&arguments[6]),
        expected_contract_sha256: arguments[7].clone(),
        expected_sha256s_sha256: arguments[8].clone(),
        expected_storm_receipt_sha256: arguments[9].clone(),
        expected_storm_f02_sha256: arguments[10].clone(),
        expected_storm_f03_sha256: arguments[11].clone(),
        source_grid: HrrrLambertGrid::canonical_window(source_i, source_j, source_nx, source_ny)?,
        target: Cm1TargetGrid::new(nx, ny, nz, dx, dy, dz, center_lat, center_lon)?,
        model_dt_s: model_dt,
    };
    let receipt = publish_future_boundaries_new(&request)?;
    Ok(format!(
        "{}\nreceipt_sha256={}",
        receipt.canonical_json(),
        receipt.sha256()
    ))
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match run(&arguments) {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("gm1-hrrr-future-boundaries: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_and_usage_are_strict() {
        let abi = run(&["--abi".to_string()]).unwrap();
        assert!(abi.contains(GM1_HRRR_FUTURE_BOUNDARY_ABI));
        assert!(abi.contains(GM1_HRRR_FUTURE_BOUNDARY_SERIES_SCHEMA));
        assert!(run(&[]).unwrap_err().to_string().contains("usage:"));
    }
}
