use cm1_state::hrrr::{
    audit_horizontal_remap, Cm1TargetGrid, DecodedHrrrWindow, HrrrLambertGrid,
    HORIZONTAL_REMAP_AUDIT_SCHEMA,
};
use cm1_state::{Result, StateError};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

const ABI: &str = "gm1.hrrr-remap-audit-cli/v1";
const USAGE: &str = "usage:\n  gm1-hrrr-remap-audit --abi\n  gm1-hrrr-remap-audit F02_DECODED_DIRECTORY NEW_RECEIPT.json EXPECTED_SOURCE_FINGERPRINT_SHA256 CENTER_LAT CENTER_LON";

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn parse_f64(value: &str, label: &str) -> Result<f64> {
    let result = value
        .parse::<f64>()
        .map_err(|_| invalid(format!("{label} must be a finite binary64 value")))?;
    if !result.is_finite() {
        return Err(invalid(format!("{label} must be finite")));
    }
    Ok(result)
}

fn run(arguments: &[String]) -> Result<String> {
    if arguments == ["--abi"] {
        return Ok(format!(
            "{ABI};receipt_schema={HORIZONTAL_REMAP_AUDIT_SCHEMA}"
        ));
    }
    let [source, output, expected_source, center_lat, center_lon] = arguments else {
        return Err(invalid(USAGE));
    };
    if expected_source.len() != 64
        || !expected_source
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(
            "expected source fingerprint must be 64 lowercase hex characters",
        ));
    }
    let grid = HrrrLambertGrid::canonical_window(971, 622, 26, 26)?;
    let source = DecodedHrrrWindow::read_directory(PathBuf::from(source), grid, 2)?;
    if source.source_fingerprint_sha256() != expected_source {
        return Err(invalid(format!(
            "decoded f02 fingerprint is {}; expected {expected_source}",
            source.source_fingerprint_sha256()
        )));
    }
    let target = Cm1TargetGrid::new(
        300,
        300,
        75,
        100.0,
        100.0,
        200.0,
        parse_f64(center_lat, "CENTER_LAT")?,
        parse_f64(center_lon, "CENTER_LON")?,
    )?;
    let audit = audit_horizontal_remap(&source, target)?;
    audit.write_new(output)?;
    Ok(format!(
        "{}\nreceipt_sha256={}",
        audit.canonical_json(),
        audit.sha256()
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
            eprintln!("gm1-hrrr-remap-audit: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_and_usage_are_strict() {
        assert!(run(&["--abi".to_string()])
            .unwrap()
            .contains(HORIZONTAL_REMAP_AUDIT_SCHEMA));
        assert!(run(&[]).is_err());
    }
}
