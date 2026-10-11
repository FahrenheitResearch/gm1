use cm1_state::product_restart::{
    read_initialized_product_restart, restore_initialized_product_restart_new,
    write_initialized_product_restart_new,
};
use std::env;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage:\n  cm1-initialized-product-restart write PRODUCT_DIR OUTPUT.cm1ipr MANIFEST_SHA PAYLOAD_SHA RECEIPT_SHA SOURCE_ARTIFACT_SHA SOURCE_CONTRACT_SHA\n  cm1-initialized-product-restart inspect INPUT.cm1ipr ARTIFACT_SHA MANIFEST_SHA PAYLOAD_SHA RECEIPT_SHA SOURCE_ARTIFACT_SHA SOURCE_CONTRACT_SHA\n  cm1-initialized-product-restart restore INPUT.cm1ipr OUTPUT_DIR ARTIFACT_SHA MANIFEST_SHA PAYLOAD_SHA RECEIPT_SHA SOURCE_ARTIFACT_SHA SOURCE_CONTRACT_SHA"
    );
    std::process::exit(2)
}

fn run() -> cm1_state::Result<String> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let Some(command) = arguments.first().map(String::as_str) else {
        usage();
    };
    match command {
        "write" if arguments.len() == 8 => {
            let receipt = write_initialized_product_restart_new(
                &arguments[1],
                &arguments[2],
                &arguments[3],
                &arguments[4],
                &arguments[5],
                &arguments[6],
                &arguments[7],
            )?;
            Ok(receipt.canonical_json())
        }
        "inspect" if arguments.len() == 8 => {
            let restored = read_initialized_product_restart(
                &arguments[1],
                &arguments[2],
                &arguments[3],
                &arguments[4],
                &arguments[5],
                &arguments[6],
                &arguments[7],
            )?;
            Ok(restored.receipt().canonical_json())
        }
        "restore" if arguments.len() == 9 => {
            let restored = read_initialized_product_restart(
                &arguments[1],
                &arguments[3],
                &arguments[4],
                &arguments[5],
                &arguments[6],
                &arguments[7],
                &arguments[8],
            )?;
            let receipt = restore_initialized_product_restart_new(&restored, &arguments[2])?;
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
            eprintln!("cm1-initialized-product-restart: {error}");
            ExitCode::from(1)
        }
    }
}
