use cm1_state::child_post_step::{
    child_post_step_layout_json, idealized_child_post_step_layout_json, inspect_child_post_step,
    inspect_idealized_child_post_step, publish_child_post_step_new,
    publish_idealized_child_post_step_new, restore_child_post_step_new,
    restore_idealized_child_post_step_new, ChildPostStepOperatorManifest,
    IdealizedPostStepAuthority, CHILD_POST_STEP_CHECKPOINT_ABI, CHILD_POST_STEP_HISTORY_ABI,
    IDEALIZED_POST_STEP_HISTORY_ABI,
};
use cm1_state::forecast::ForecastClock;
use cm1_state::{Result, StateError};
use std::env;
use std::fs;
use std::io::{Error, ErrorKind};
use std::path::Path;
use std::process::ExitCode;

const ABI: &str = "gm1.real-50m-child-post-step-cli/v1";
const IDEALIZED_ABI: &str = "gm1.idealized-isnd7-iinit1-post-step-cli/v1";
const USAGE: &str = concat!(
    "usage:\n",
    "  gm1-child-post-step --abi\n",
    "  gm1-child-post-step --abi-idealized\n",
    "  gm1-child-post-step layout OPERATOR.json OPERATOR_SHA256\n",
    "  gm1-child-post-step layout-idealized AUTHORITY.json AUTHORITY_SHA256 OPERATOR.json OPERATOR_SHA256\n",
    "  gm1-child-post-step publish CHILD_FULL_STATE CHECKPOINT_RAW CHECKPOINT_RAW_SHA256 HISTORY_RAW HISTORY_RAW_SHA256 OPERATOR.json OPERATOR_SHA256 SCHEDULER.json STEP MODEL_TIME_F64_BITS MASS1_F64_BITS NEW_OUTPUT_DIR\n",
    "  gm1-child-post-step inspect CHILD_FULL_STATE PAIR_DIR MANIFEST_SHA256 RECEIPT_SHA256 CHECKPOINT_ARTIFACT_SHA256 HISTORY_ARTIFACT_SHA256 OPERATOR_SHA256\n",
    "  gm1-child-post-step restore CHILD_FULL_STATE PAIR_DIR NEW_OUTPUT_DIR MANIFEST_SHA256 RECEIPT_SHA256 CHECKPOINT_ARTIFACT_SHA256 HISTORY_ARTIFACT_SHA256 OPERATOR_SHA256\n",
    "  gm1-child-post-step publish-idealized AUTHORITY.json AUTHORITY_SHA256 CHECKPOINT_RAW CHECKPOINT_RAW_SHA256 HISTORY_RAW HISTORY_RAW_SHA256 OPERATOR.json OPERATOR_SHA256 SCHEDULER.json STEP MODEL_TIME_F64_BITS MASS1_F64_BITS NEW_OUTPUT_DIR\n",
    "  gm1-child-post-step inspect-idealized AUTHORITY.json AUTHORITY_SHA256 PAIR_DIR MANIFEST_SHA256 RECEIPT_SHA256 CHECKPOINT_ARTIFACT_SHA256 HISTORY_ARTIFACT_SHA256 OPERATOR_SHA256\n",
    "  gm1-child-post-step restore-idealized AUTHORITY.json AUTHORITY_SHA256 PAIR_DIR NEW_OUTPUT_DIR MANIFEST_SHA256 RECEIPT_SHA256 CHECKPOINT_ARTIFACT_SHA256 HISTORY_ARTIFACT_SHA256 OPERATOR_SHA256"
);

fn invalid(message: impl Into<String>) -> StateError {
    Error::new(ErrorKind::InvalidInput, message.into()).into()
}

fn read_regular(path: &str, label: &str) -> Result<Vec<u8>> {
    let path = Path::new(path);
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect {label} {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > 128 * 1024 {
        return Err(invalid(format!(
            "{label} must be a bounded regular non-symlink file"
        )));
    }
    fs::read(path).map_err(Into::into)
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
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return Err(invalid(format!("{label} must be finite")));
    }
    Ok(value)
}

fn operator(path: &str, expected_sha256: &str) -> Result<ChildPostStepOperatorManifest> {
    ChildPostStepOperatorManifest::from_canonical_bytes(
        &read_regular(path, "child operator manifest")?,
        expected_sha256,
    )
}

fn idealized_authority(path: &str, expected_sha256: &str) -> Result<IdealizedPostStepAuthority> {
    IdealizedPostStepAuthority::from_file(path, expected_sha256)
}

fn run(arguments: &[String]) -> Result<String> {
    match arguments {
        [flag] if flag == "--abi" => Ok(format!(
            "{ABI};checkpoint_carrier_abi={CHILD_POST_STEP_CHECKPOINT_ABI};history_carrier_abi={CHILD_POST_STEP_HISTORY_ABI}"
        )),
        [flag] if flag == "--abi-idealized" => Ok(format!(
            "{IDEALIZED_ABI};checkpoint_carrier_abi={CHILD_POST_STEP_CHECKPOINT_ABI};history_carrier_abi={IDEALIZED_POST_STEP_HISTORY_ABI}"
        )),
        [command, operator_path, operator_sha] if command == "layout" => {
            child_post_step_layout_json(&operator(operator_path, operator_sha)?)
        }
        [command, authority_path, authority_sha, operator_path, operator_sha]
            if command == "layout-idealized" =>
        {
            idealized_child_post_step_layout_json(
                &operator(operator_path, operator_sha)?,
                &idealized_authority(authority_path, authority_sha)?,
            )
        }
        [
            command,
            full_state,
            checkpoint_raw,
            checkpoint_raw_sha,
            history_raw,
            history_raw_sha,
            operator_path,
            operator_sha,
            scheduler_path,
            step,
            model_time_bits,
            mass1_bits,
            output,
        ] if command == "publish" => {
            let step = parse_u64(step, "STEP")?;
            let model_time = parse_f64_bits(model_time_bits, "MODEL_TIME_F64_BITS")?;
            let mass1 = parse_f64_bits(mass1_bits, "MASS1_F64_BITS")?;
            let receipt = publish_child_post_step_new(
                full_state,
                checkpoint_raw,
                checkpoint_raw_sha,
                history_raw,
                history_raw_sha,
                operator(operator_path, operator_sha)?,
                &read_regular(scheduler_path, "child scheduler extension")?,
                ForecastClock::new(step, model_time, 0.5)?,
                mass1,
                output,
            )?;
            Ok(receipt.canonical_json())
        }
        [
            command,
            authority_path,
            authority_sha,
            checkpoint_raw,
            checkpoint_raw_sha,
            history_raw,
            history_raw_sha,
            operator_path,
            operator_sha,
            scheduler_path,
            step,
            model_time_bits,
            mass1_bits,
            output,
        ] if command == "publish-idealized" => {
            let step = parse_u64(step, "STEP")?;
            let model_time = parse_f64_bits(model_time_bits, "MODEL_TIME_F64_BITS")?;
            let mass1 = parse_f64_bits(mass1_bits, "MASS1_F64_BITS")?;
            let receipt = publish_idealized_child_post_step_new(
                idealized_authority(authority_path, authority_sha)?,
                checkpoint_raw,
                checkpoint_raw_sha,
                history_raw,
                history_raw_sha,
                operator(operator_path, operator_sha)?,
                &read_regular(scheduler_path, "child scheduler extension")?,
                ForecastClock::new(step, model_time, 0.5)?,
                mass1,
                output,
            )?;
            Ok(receipt.canonical_json())
        }
        [
            command,
            full_state,
            pair,
            manifest_sha,
            receipt_sha,
            checkpoint_artifact_sha,
            history_artifact_sha,
            operator_sha,
        ] if command == "inspect" => Ok(inspect_child_post_step(
            full_state,
            pair,
            manifest_sha,
            receipt_sha,
            checkpoint_artifact_sha,
            history_artifact_sha,
            operator_sha,
        )?
        .canonical_json()),
        [
            command,
            authority_path,
            authority_sha,
            pair,
            manifest_sha,
            receipt_sha,
            checkpoint_artifact_sha,
            history_artifact_sha,
            operator_sha,
        ] if command == "inspect-idealized" => Ok(inspect_idealized_child_post_step(
            idealized_authority(authority_path, authority_sha)?,
            pair,
            manifest_sha,
            receipt_sha,
            checkpoint_artifact_sha,
            history_artifact_sha,
            operator_sha,
        )?
        .canonical_json()),
        [
            command,
            full_state,
            pair,
            output,
            manifest_sha,
            receipt_sha,
            checkpoint_artifact_sha,
            history_artifact_sha,
            operator_sha,
        ] if command == "restore" => Ok(restore_child_post_step_new(
            full_state,
            pair,
            output,
            manifest_sha,
            receipt_sha,
            checkpoint_artifact_sha,
            history_artifact_sha,
            operator_sha,
        )?
        .canonical_json()),
        [
            command,
            authority_path,
            authority_sha,
            pair,
            output,
            manifest_sha,
            receipt_sha,
            checkpoint_artifact_sha,
            history_artifact_sha,
            operator_sha,
        ] if command == "restore-idealized" => Ok(restore_idealized_child_post_step_new(
            idealized_authority(authority_path, authority_sha)?,
            pair,
            output,
            manifest_sha,
            receipt_sha,
            checkpoint_artifact_sha,
            history_artifact_sha,
            operator_sha,
        )?
        .canonical_json()),
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
            eprintln!("gm1-child-post-step: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_words_are_strict() {
        assert_eq!(parse_f64_bits("3fe0000000000000", "dt").unwrap(), 0.5);
        assert!(parse_f64_bits("3FE0000000000000", "upper").is_err());
        assert!(parse_f64_bits("7ff8000000000000", "nan").is_err());
    }

    #[test]
    fn cli_abi_names_both_carrier_contracts() {
        let value = run(&["--abi".to_owned()]).unwrap();
        assert!(value.contains(CHILD_POST_STEP_CHECKPOINT_ABI));
        assert!(value.contains(CHILD_POST_STEP_HISTORY_ABI));
        let idealized = run(&["--abi-idealized".to_owned()]).unwrap();
        assert!(idealized.contains(IDEALIZED_ABI));
        assert!(idealized.contains(CHILD_POST_STEP_CHECKPOINT_ABI));
        assert!(idealized.contains(IDEALIZED_POST_STEP_HISTORY_ABI));
        assert_ne!(value, idealized);
    }
}
