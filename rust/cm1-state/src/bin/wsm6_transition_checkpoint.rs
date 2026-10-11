use cm1_state::transition_restart::{
    read_wsm6_transition_checkpoint, restore_wsm6_transition_checkpoint_new,
    write_wsm6_transition_checkpoint_new, TransitionCheckpointConfig, TransitionSourceIdentities,
};
use cm1_state::Result;
use std::env;
use std::io::{Error, ErrorKind};
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage:\n  cm1-wsm6-transition-checkpoint write CAPTURE_DIR OUTPUT.cm1w6r J_START I_START NY NX COMPLETED_BRIDGE_CALLS COMPLETED_COMPONENT_CALLS COMPLETED_TRANSITIONS PRODUCT_MANIFEST_SHA PRODUCT_PAYLOAD_SHA GEOPOTENTIAL_MANIFEST_SHA GEOPOTENTIAL_PAYLOAD_SHA SOURCE_ARTIFACT_SHA SOURCE_CONTRACT_SHA\n  cm1-wsm6-transition-checkpoint inspect INPUT.cm1w6r ARTIFACT_SHA J_START I_START NY NX COMPLETED_BRIDGE_CALLS COMPLETED_COMPONENT_CALLS COMPLETED_TRANSITIONS PRODUCT_MANIFEST_SHA PRODUCT_PAYLOAD_SHA GEOPOTENTIAL_MANIFEST_SHA GEOPOTENTIAL_PAYLOAD_SHA SOURCE_ARTIFACT_SHA SOURCE_CONTRACT_SHA\n  cm1-wsm6-transition-checkpoint restore INPUT.cm1w6r OUTPUT_DIR ARTIFACT_SHA J_START I_START NY NX COMPLETED_BRIDGE_CALLS COMPLETED_COMPONENT_CALLS COMPLETED_TRANSITIONS PRODUCT_MANIFEST_SHA PRODUCT_PAYLOAD_SHA GEOPOTENTIAL_MANIFEST_SHA GEOPOTENTIAL_PAYLOAD_SHA SOURCE_ARTIFACT_SHA SOURCE_CONTRACT_SHA"
    );
    std::process::exit(2)
}

fn usize_argument(value: &str, label: &str) -> Result<usize> {
    value.parse::<usize>().map_err(|_| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("{label} must be an unsigned integer"),
        )
        .into()
    })
}

fn u64_argument(value: &str, label: &str) -> Result<u64> {
    value.parse::<u64>().map_err(|_| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("{label} must be an unsigned integer"),
        )
        .into()
    })
}

fn config(arguments: &[String], first: usize) -> Result<TransitionCheckpointConfig> {
    TransitionCheckpointConfig::new(
        usize_argument(&arguments[first], "J_START")?,
        usize_argument(&arguments[first + 1], "I_START")?,
        usize_argument(&arguments[first + 2], "NY")?,
        usize_argument(&arguments[first + 3], "NX")?,
        u64_argument(&arguments[first + 4], "COMPLETED_BRIDGE_CALLS")?,
        u64_argument(&arguments[first + 5], "COMPLETED_COMPONENT_CALLS")?,
        u64_argument(&arguments[first + 6], "COMPLETED_TRANSITIONS")?,
    )
}

fn source(arguments: &[String], first: usize) -> Result<TransitionSourceIdentities> {
    TransitionSourceIdentities::new(
        &arguments[first],
        &arguments[first + 1],
        &arguments[first + 2],
        &arguments[first + 3],
        &arguments[first + 4],
        &arguments[first + 5],
    )
}

fn run() -> Result<String> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str) {
        Some("write") if arguments.len() == 16 => {
            let config = config(&arguments, 3)?;
            let source = source(&arguments, 10)?;
            let receipt =
                write_wsm6_transition_checkpoint_new(&arguments[1], &arguments[2], config, source)?;
            Ok(receipt.canonical_json())
        }
        Some("inspect") if arguments.len() == 16 => {
            let config = config(&arguments, 3)?;
            let source = source(&arguments, 10)?;
            let checkpoint =
                read_wsm6_transition_checkpoint(&arguments[1], &arguments[2], &config, &source)?;
            Ok(checkpoint.receipt().canonical_json())
        }
        Some("restore") if arguments.len() == 17 => {
            let config = config(&arguments, 4)?;
            let source = source(&arguments, 11)?;
            let checkpoint =
                read_wsm6_transition_checkpoint(&arguments[1], &arguments[3], &config, &source)?;
            let receipt = restore_wsm6_transition_checkpoint_new(&checkpoint, &arguments[2])?;
            Ok(receipt.canonical_json())
        }
        _ => usage(),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(receipt) => {
            println!("{receipt}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("cm1-wsm6-transition-checkpoint: {error}");
            ExitCode::from(1)
        }
    }
}
