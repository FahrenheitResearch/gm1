use cm1_state::geopotential::{export_wsm6_geopotential_new, read_wsm6_geopotential_auxiliary};
use cm1_state::product_restart::read_initialized_product_restart;
use std::env;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage:\n  cm1-wsm6-geopotential export INPUT.cm1ipr OUTPUT_DIR RESTART_SHA PRODUCT_MANIFEST_SHA PRODUCT_PAYLOAD_SHA PRODUCT_RECEIPT_SHA SOURCE_ARTIFACT_SHA SOURCE_CONTRACT_SHA\n  cm1-wsm6-geopotential inspect INPUT.cm1ipr AUX_DIR RESTART_SHA PRODUCT_MANIFEST_SHA PRODUCT_PAYLOAD_SHA PRODUCT_RECEIPT_SHA SOURCE_ARTIFACT_SHA SOURCE_CONTRACT_SHA AUX_MANIFEST_SHA AUX_PAYLOAD_SHA AUX_RECEIPT_SHA"
    );
    std::process::exit(2)
}

fn source(
    arguments: &[String],
) -> cm1_state::Result<cm1_state::product_restart::RestoredProductRestart> {
    read_initialized_product_restart(
        &arguments[1],
        &arguments[3],
        &arguments[4],
        &arguments[5],
        &arguments[6],
        &arguments[7],
        &arguments[8],
    )
}

fn run() -> cm1_state::Result<String> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str) {
        Some("export") if arguments.len() == 9 => {
            let source = source(&arguments)?;
            let receipt = export_wsm6_geopotential_new(&source, &arguments[2])?;
            Ok(receipt.canonical_json())
        }
        Some("inspect") if arguments.len() == 12 => {
            let source = source(&arguments)?;
            let auxiliary = read_wsm6_geopotential_auxiliary(
                &source,
                &arguments[2],
                &arguments[9],
                &arguments[10],
                &arguments[11],
            )?;
            Ok(auxiliary.receipt.canonical_json())
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
            eprintln!("cm1-wsm6-geopotential: {error}");
            ExitCode::from(1)
        }
    }
}
