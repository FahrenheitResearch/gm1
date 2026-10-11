use cm1_state::forecast::ForecastClock;
use cm1_state::live_full_state::{
    live_full_state_real_layout_json, read_live_full_state_checkpoint,
    restore_live_full_state_checkpoint_new, write_live_full_state_checkpoint_new,
    LiveFullStateConfig, LiveFullStateSourceIdentities, LIVE_FULL_STATE_CARRIER_ABI,
};
use cm1_state::{Result, StateError};
use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

const ABI: &str = "gpuwm-cm1.live-full-state-cli/v1";
const USAGE: &str = concat!(
    "usage:\n",
    "  cm1-live-full-state --abi\n",
    "  cm1-live-full-state layout\n",
    "  cm1-live-full-state build RAW_ARENA OUTPUT.cm1fcp RAW_SHA PRODUCT_RESTART_SHA PRODUCT_MANIFEST_SHA PRODUCT_PAYLOAD_SHA TERRAIN_RESTART_SHA WSM6_RESTART_SHA STEP MODEL_TIME_S DT_S COMPLETED_WSM6_TRANSITIONS SCHEDULER_EXTENSION.json\n",
    "  cm1-live-full-state inspect INPUT.cm1fcp CONTRACT_SHA ARTIFACT_SHA PRODUCT_RESTART_SHA\n",
    "  cm1-live-full-state restore INPUT.cm1fcp OUTPUT_DIR CONTRACT_SHA ARTIFACT_SHA PRODUCT_RESTART_SHA"
);

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn parse_u64(value: &str, label: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|_| invalid(format!("{label} must be an unsigned integer")))
}

fn parse_f64(value: &str, label: &str) -> Result<f64> {
    let result = value
        .parse::<f64>()
        .map_err(|_| invalid(format!("{label} must be a finite number")))?;
    if !result.is_finite() {
        return Err(invalid(format!("{label} must be finite")));
    }
    Ok(result)
}

fn sources(arguments: &[String], first: usize) -> Result<LiveFullStateSourceIdentities> {
    LiveFullStateSourceIdentities::new(
        &arguments[first],
        &arguments[first + 1],
        &arguments[first + 2],
        &arguments[first + 3],
        &arguments[first + 4],
    )
}

fn run(arguments: &[String]) -> Result<String> {
    match arguments {
        [flag] if flag == "--abi" => {
            Ok(format!("{ABI}\ncarrier_abi={LIVE_FULL_STATE_CARRIER_ABI}"))
        }
        [command] if command == "layout" => live_full_state_real_layout_json(),
        [command, raw, output, raw_sha, product_restart, product_manifest, product_payload, terrain_restart, wsm6_restart, step, model_time, dt, completed_wsm6, scheduler]
            if command == "build" =>
        {
            let source_arguments = vec![
                product_restart.clone(),
                product_manifest.clone(),
                product_payload.clone(),
                terrain_restart.clone(),
                wsm6_restart.clone(),
            ];
            let source = sources(&source_arguments, 0)?;
            let step = parse_u64(step, "STEP")?;
            let model_time = parse_f64(model_time, "MODEL_TIME_S")?;
            let dt = parse_f64(dt, "DT_S")?;
            let completed_wsm6 = parse_u64(completed_wsm6, "COMPLETED_WSM6_TRANSITIONS")?;
            let scheduler_bytes = fs::read(scheduler).map_err(|error| {
                invalid(format!(
                    "could not read scheduler extension {}: {error}",
                    Path::new(scheduler).display()
                ))
            })?;
            let receipt = write_live_full_state_checkpoint_new(
                raw,
                output,
                raw_sha,
                LiveFullStateConfig::real_product(),
                source,
                ForecastClock::new(step, model_time, dt)?,
                completed_wsm6,
                &scheduler_bytes,
            )?;
            Ok(receipt.canonical_json())
        }
        [command, input, contract_sha, artifact_sha, product_restart_sha]
            if command == "inspect" =>
        {
            let restored = read_live_full_state_checkpoint(
                input,
                contract_sha,
                artifact_sha,
                product_restart_sha,
            )?;
            Ok(restored.forecast().receipt().canonical_json())
        }
        [command, input, output, contract_sha, artifact_sha, product_restart_sha]
            if command == "restore" =>
        {
            let restored = read_live_full_state_checkpoint(
                input,
                contract_sha,
                artifact_sha,
                product_restart_sha,
            )?;
            Ok(restore_live_full_state_checkpoint_new(&restored, output)?.canonical_json())
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
            eprintln!("cm1-live-full-state: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_arguments_are_strict() {
        assert_eq!(parse_u64("7", "step").unwrap(), 7);
        assert!(parse_u64("-1", "step").is_err());
        assert_eq!(parse_f64("0.5", "dt").unwrap(), 0.5);
        assert!(parse_f64("NaN", "dt").is_err());
    }

    #[test]
    fn abi_is_explicit() {
        let output = run(&["--abi".to_string()]).unwrap();
        assert!(output.contains(ABI));
        assert!(output.contains(LIVE_FULL_STATE_CARRIER_ABI));
    }
}
