use cm1_state::child_full_state::{
    inspect_child_full_state, publish_child_full_state_new, ChildFullStateReceipt,
    CHILD_CHECKPOINT_ADMISSION_SCHEMA, CHILD_FIRST_STEP_SCHEMA, CHILD_FULL_STATE_SCHEMA,
};
use std::env;
use std::path::PathBuf;

fn usage() -> &'static str {
    "usage:\n  gm1-child-full-state --abi\n  gm1-child-full-state build <atmosphere-dir> <auxiliary-dir> <terrain-dir> <static-height-dir> <static-validation-receipt> <f02-product.cm1ipr> <closure-dir> <producer-source.rs> <output-dir>\n  gm1-child-full-state inspect <atmosphere-dir> <auxiliary-dir> <terrain-dir> <static-height-dir> <static-validation-receipt> <f02-product.cm1ipr> <closure-dir> <producer-source.rs> <artifact-dir> <manifest-sha256> <first-step-sha256> <checkpoint-admission-sha256> <receipt-sha256>"
}

fn main() {
    if let Err(error) = run() {
        eprintln!("gm1-child-full-state: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.len() == 1 && arguments[0] == "--abi" {
        println!(
            "{CHILD_FULL_STATE_SCHEMA};first_step={CHILD_FIRST_STEP_SCHEMA};checkpoint={CHILD_CHECKPOINT_ADMISSION_SCHEMA}"
        );
        return Ok(());
    }
    let command = arguments
        .first()
        .and_then(|value| value.to_str())
        .ok_or_else(|| usage().to_string())?;
    let executable = env::current_exe()?;
    match command {
        "build" if arguments.len() == 10 => {
            let receipt = publish_child_full_state_new(
                PathBuf::from(&arguments[1]),
                PathBuf::from(&arguments[2]),
                PathBuf::from(&arguments[3]),
                PathBuf::from(&arguments[4]),
                PathBuf::from(&arguments[5]),
                PathBuf::from(&arguments[6]),
                PathBuf::from(&arguments[7]),
                PathBuf::from(&arguments[8]),
                &executable,
                PathBuf::from(&arguments[9]),
            )?;
            print_receipt("built", &receipt);
        }
        "inspect" if arguments.len() == 14 => {
            let receipt = inspect_child_full_state(
                PathBuf::from(&arguments[1]),
                PathBuf::from(&arguments[2]),
                PathBuf::from(&arguments[3]),
                PathBuf::from(&arguments[4]),
                PathBuf::from(&arguments[5]),
                PathBuf::from(&arguments[6]),
                PathBuf::from(&arguments[7]),
                PathBuf::from(&arguments[8]),
                &executable,
                PathBuf::from(&arguments[9]),
                arguments[10].to_str().ok_or("manifest SHA must be UTF-8")?,
                arguments[11]
                    .to_str()
                    .ok_or("first-step SHA must be UTF-8")?,
                arguments[12]
                    .to_str()
                    .ok_or("checkpoint-admission SHA must be UTF-8")?,
                arguments[13].to_str().ok_or("receipt SHA must be UTF-8")?,
            )?;
            print_receipt("inspected", &receipt);
        }
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn print_receipt(action: &str, receipt: &ChildFullStateReceipt) {
    println!(
        "{{\"action\":\"{action}\",\"base_wind_sha256\":\"{}\",\"checkpoint_admission_sha256\":\"{}\",\"first_step_sha256\":\"{}\",\"manifest_sha256\":\"{}\",\"producer_executable_sha256\":\"{}\",\"producer_source_sha256\":\"{}\",\"receipt_sha256\":\"{}\"}}",
        receipt.base_wind_sha256,
        receipt.checkpoint_admission_sha256,
        receipt.first_step_sha256,
        receipt.manifest_sha256,
        receipt.producer_executable_sha256,
        receipt.producer_source_sha256,
        receipt.receipt_sha256,
    );
}
