use cm1_state::live::{export_checkpoint_live_new, LiveAdmission, PhysicalCrop};
use std::env;
use std::error::Error;
use std::path::PathBuf;

fn parse(value: &str, name: &str) -> Result<usize, Box<dyn Error>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}").into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 && args.len() != 10 {
        return Err(concat!(
            "usage: cm1_live_export INPUT.cm1fcp NEW_OUTPUT_DIR ",
            "EXPECTED_CONTRACT_SHA256 EXPECTED_ARTIFACT_SHA256 ",
            "[K_START K_COUNT J_START J_COUNT I_START I_COUNT]"
        )
        .into());
    }
    let crop = if args.len() == 10 {
        Some(PhysicalCrop {
            k_start: parse(&args[4], "K_START")?,
            nz: parse(&args[5], "K_COUNT")?,
            j_start: parse(&args[6], "J_START")?,
            ny: parse(&args[7], "J_COUNT")?,
            i_start: parse(&args[8], "I_START")?,
            nx: parse(&args[9], "I_COUNT")?,
        })
    } else {
        None
    };
    let receipt = export_checkpoint_live_new(
        PathBuf::from(&args[0]),
        PathBuf::from(&args[1]),
        &args[2],
        &args[3],
        crop,
        LiveAdmission::RealProduct,
    )?;
    println!("{}", receipt.canonical_json());
    Ok(())
}
