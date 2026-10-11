use cm1_state::terrain::{
    export_checkpoint_terrain_static_new, HorizontalCrop, TerrainStaticAdmission,
    TerrainStaticConfig,
};
use std::env;
use std::error::Error;
use std::path::PathBuf;

fn parse_usize(value: &str, name: &str) -> Result<usize, Box<dyn Error>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}").into())
}

fn parse_u8(value: &str, name: &str) -> Result<u8, Box<dyn Error>> {
    value
        .parse::<u8>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}").into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 10 && args.len() != 14 {
        return Err(concat!(
            "usage: cm1_terrain_static_export INPUT.cm1fcp NEW_OUTPUT_DIR ",
            "EXPECTED_CONTRACT_SHA256 EXPECTED_ARTIFACT_SHA256 ",
            "HADVORDRS HADVORDRV WBC EBC SBC NBC ",
            "[J_START J_COUNT I_START I_COUNT]"
        )
        .into());
    }
    let config = TerrainStaticConfig::new(
        parse_u8(&args[4], "HADVORDRS")?,
        parse_u8(&args[5], "HADVORDRV")?,
        [
            parse_u8(&args[6], "WBC")?,
            parse_u8(&args[7], "EBC")?,
            parse_u8(&args[8], "SBC")?,
            parse_u8(&args[9], "NBC")?,
        ],
    )?;
    let crop = if args.len() == 14 {
        Some(HorizontalCrop {
            j_start: parse_usize(&args[10], "J_START")?,
            ny: parse_usize(&args[11], "J_COUNT")?,
            i_start: parse_usize(&args[12], "I_START")?,
            nx: parse_usize(&args[13], "I_COUNT")?,
        })
    } else {
        None
    };
    let receipt = export_checkpoint_terrain_static_new(
        PathBuf::from(&args[0]),
        PathBuf::from(&args[1]),
        &args[2],
        &args[3],
        crop,
        config,
        TerrainStaticAdmission::RealProduct,
    )?;
    println!("{}", receipt.canonical_json());
    Ok(())
}
