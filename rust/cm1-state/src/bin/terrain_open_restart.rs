use cm1_state::product_restart::read_initialized_product_restart;
use cm1_state::terrain_admission::{
    build_terrain_open_cold_start, diagnose_terrain_open, read_terrain_open_restart,
    write_terrain_open_restart_new, TerrainOpenAdmission,
};
use std::env;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "usage:\n  cm1-terrain-open-restart build INPUT.cm1ipr OUTPUT.cm1fcp RESTART_SHA MANIFEST_SHA PAYLOAD_SHA RECEIPT_SHA SOURCE_ARTIFACT_SHA SOURCE_CONTRACT_SHA\n  cm1-terrain-open-restart inspect INPUT.cm1fcp CONTRACT_SHA ARTIFACT_SHA SOURCE_RESTART_SHA\n  cm1-terrain-open-restart summary INPUT.cm1fcp CONTRACT_SHA ARTIFACT_SHA SOURCE_RESTART_SHA"
    );
    std::process::exit(2)
}

fn run(arguments: &[String]) -> cm1_state::Result<String> {
    match arguments.first().map(String::as_str) {
        Some("build") if arguments.len() == 9 => {
            let source = read_initialized_product_restart(
                &arguments[1],
                &arguments[3],
                &arguments[4],
                &arguments[5],
                &arguments[6],
                &arguments[7],
                &arguments[8],
            )?;
            let build = build_terrain_open_cold_start(
                &source,
                None,
                [100.0, 100.0, 200.0],
                0.5,
                TerrainOpenAdmission::RealProduct,
            )?;
            let receipt = write_terrain_open_restart_new(&build, &arguments[2])?;
            Ok(receipt.canonical_json())
        }
        Some("inspect") if arguments.len() == 5 => {
            let restored = read_terrain_open_restart(
                &arguments[1],
                &arguments[2],
                &arguments[3],
                &arguments[4],
            )?;
            Ok(restored.forecast().receipt().canonical_json())
        }
        Some("summary") if arguments.len() == 5 => {
            let restored = read_terrain_open_restart(
                &arguments[1],
                &arguments[2],
                &arguments[3],
                &arguments[4],
            )?;
            Ok(diagnose_terrain_open(&restored)?.canonical_json())
        }
        _ => usage(),
    }
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match run(&arguments) {
        Ok(receipt) => {
            println!("{receipt}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("cm1-terrain-open-restart: {error}");
            ExitCode::from(1)
        }
    }
}
