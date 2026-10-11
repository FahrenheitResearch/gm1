use cm1_state::forecast::ForecastClock;
use cm1_state::gm1_checkpoint::{
    gm1_real_layout_json, read_gm1_checkpoint, restore_gm1_checkpoint_new,
    write_gm1_checkpoint_new, Gm1CheckpointConfig, Gm1OperatorManifest, Gm1SourceIdentities,
    GM1_CHECKPOINT_CARRIER_ABI,
};
use cm1_state::{Result, StateError};
use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

const ABI: &str = "gm1.forecast-checkpoint-cli/v1";
const USAGE: &str = concat!(
    "usage:\n",
    "  gm1-forecast-checkpoint --abi\n",
    "  gm1-forecast-checkpoint source-identities legacy|f02-roi\n",
    "  gm1-forecast-checkpoint layout OPERATOR_MANIFEST.json\n",
    "  gm1-forecast-checkpoint build RAW_ARENA OUTPUT.cm1fcp RAW_SHA OPERATOR_MANIFEST.json STEP MODEL_TIME_F64_BITS MASS1_F64_BITS COMPLETED_TRANSITIONS SCHEDULER_EXTENSION.json [SOURCE_IDENTITIES.json]\n",
    "  gm1-forecast-checkpoint inspect INPUT.cm1fcp CONTRACT_SHA ARTIFACT_SHA OPERATOR_MANIFEST_SHA [SOURCE_IDENTITIES.json]\n",
    "  gm1-forecast-checkpoint restore INPUT.cm1fcp OUTPUT_DIR CONTRACT_SHA ARTIFACT_SHA OPERATOR_MANIFEST_SHA [SOURCE_IDENTITIES.json]"
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

fn operator_manifest(path: &str) -> Result<Gm1OperatorManifest> {
    Gm1OperatorManifest::from_canonical_bytes(&read_regular(path, "operator manifest")?)
}

fn source_identities(path: Option<&String>) -> Result<Gm1SourceIdentities> {
    match path {
        Some(path) => {
            Gm1SourceIdentities::from_canonical_bytes(&read_regular(path, "source identities")?)
        }
        None => Ok(Gm1SourceIdentities::sealed()),
    }
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
        [flag] if flag == "--abi" => Ok(format!("{ABI};carrier_abi={GM1_CHECKPOINT_CARRIER_ABI}")),
        [command, kind] if command == "source-identities" => match kind.as_str() {
            "legacy" => String::from_utf8(Gm1SourceIdentities::sealed().canonical_bytes())
                .map_err(|_| invalid("canonical source identities are not UTF-8")),
            "f02-roi" => String::from_utf8(Gm1SourceIdentities::sealed_f02_roi().canonical_bytes())
                .map_err(|_| invalid("canonical source identities are not UTF-8")),
            _ => Err(invalid("source identity kind must be legacy or f02-roi")),
        },
        [command, operators] if command == "layout" => {
            gm1_real_layout_json(&operator_manifest(operators)?)
        }
        [command, raw, output, raw_sha, operators, step, model_time_bits, mass1_bits, completed, scheduler, tail @ ..]
            if command == "build" && tail.len() <= 1 =>
        {
            let step = parse_u64(step, "STEP")?;
            let model_time = parse_f64_bits(model_time_bits, "MODEL_TIME_F64_BITS")?;
            let mass1 = parse_f64_bits(mass1_bits, "MASS1_F64_BITS")?;
            let completed = parse_u64(completed, "COMPLETED_TRANSITIONS")?;
            let scheduler = read_regular(scheduler, "scheduler extension")?;
            let sources = source_identities(tail.first())?;
            let receipt = write_gm1_checkpoint_new(
                raw,
                output,
                raw_sha,
                Gm1CheckpointConfig::production(),
                sources,
                operator_manifest(operators)?,
                ForecastClock::new(step, model_time, 0.5)?,
                mass1,
                completed,
                &scheduler,
            )?;
            Ok(receipt.canonical_json())
        }
        [command, input, contract_sha, artifact_sha, operator_sha, tail @ ..]
            if command == "inspect" && tail.len() <= 1 =>
        {
            let sources = source_identities(tail.first())?;
            let restored = read_gm1_checkpoint(
                input,
                contract_sha,
                artifact_sha,
                &sources.product_manifest_sha256,
                operator_sha,
            )?;
            Ok(restored.forecast().receipt().canonical_json())
        }
        [command, input, output, contract_sha, artifact_sha, operator_sha, tail @ ..]
            if command == "restore" && tail.len() <= 1 =>
        {
            let sources = source_identities(tail.first())?;
            let restored = read_gm1_checkpoint(
                input,
                contract_sha,
                artifact_sha,
                &sources.product_manifest_sha256,
                operator_sha,
            )?;
            Ok(restore_gm1_checkpoint_new(&restored, output)?.canonical_json())
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
            eprintln!("gm1-forecast-checkpoint: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_f64_words_are_strict_and_exact() {
        assert_eq!(parse_f64_bits("3fe0000000000000", "dt").unwrap(), 0.5);
        assert_eq!(
            parse_f64_bits("8000000000000000", "signed zero")
                .unwrap()
                .to_bits(),
            1u64 << 63
        );
        assert!(parse_f64_bits("3FE0000000000000", "upper").is_err());
        assert!(parse_f64_bits("7ff8000000000000", "nan").is_err());
    }

    #[test]
    fn cli_abi_is_explicit() {
        let output = run(&["--abi".to_string()]).unwrap();
        assert!(output.contains(ABI));
        assert!(output.contains(GM1_CHECKPOINT_CARRIER_ABI));
    }

    #[test]
    fn cli_emits_exact_f02_roi_source_identity_input() {
        let output = run(&["source-identities".to_string(), "f02-roi".to_string()]).unwrap();
        let parsed = Gm1SourceIdentities::from_canonical_bytes(output.as_bytes()).unwrap();
        assert_eq!(parsed, Gm1SourceIdentities::sealed_f02_roi());
        assert!(run(&["source-identities".to_string(), "unknown".to_string()]).is_err());
    }
}
