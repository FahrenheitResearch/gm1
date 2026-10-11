use cm1_state::initialization::MoistInitializationConfig;
use cm1_state::product::{
    export_initialized_product_f02_roi_history_new, export_initialized_product_history_new,
    InitializedProductAdmission,
};
use cm1_state::terrain::{HorizontalCrop, TerrainStaticConfig};
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

fn parse_f32_bits(value: &str, name: &str) -> Result<f32, Box<dyn Error>> {
    if value.len() != 8 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{name} must be exactly eight hexadecimal f32 bits").into());
    }
    Ok(f32::from_bits(u32::from_str_radix(value, 16)?))
}

fn parse_bool(value: &str, name: &str) -> Result<bool, Box<dyn Error>> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("{name} must be exactly true or false").into()),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let raw_args = env::args().skip(1).collect::<Vec<_>>();
    let (f02_roi, args): (bool, &[String]) = match raw_args.first().map(String::as_str) {
        Some("--f02-roi") => (true, &raw_args[1..]),
        _ => (false, &raw_args),
    };
    if args.len() != 19 && args.len() != 23 {
        return Err(concat!(
            "usage: cm1_initialized_history_product_export [--f02-roi] INPUT.cm1fhs NEW_OUTPUT_DIR ",
            "EXPECTED_CONTRACT_SHA256 EXPECTED_ARTIFACT_SHA256 ",
            "HADVORDRS HADVORDRV WBC EBC SBC NBC ",
            "G_F32_BITS CP_F32_BITS RD_F32_BITS RV_F32_BITS P00_F32_BITS ",
            "PSOLVER IBALANCE SETPPI IICE [J_START J_COUNT I_START I_COUNT]"
        )
        .into());
    }
    let terrain_config = TerrainStaticConfig::new(
        parse_u8(&args[4], "HADVORDRS")?,
        parse_u8(&args[5], "HADVORDRV")?,
        [
            parse_u8(&args[6], "WBC")?,
            parse_u8(&args[7], "EBC")?,
            parse_u8(&args[8], "SBC")?,
            parse_u8(&args[9], "NBC")?,
        ],
    )?;
    let moist_config = MoistInitializationConfig::new(
        parse_f32_bits(&args[10], "G_F32_BITS")?,
        parse_f32_bits(&args[11], "CP_F32_BITS")?,
        parse_f32_bits(&args[12], "RD_F32_BITS")?,
        parse_f32_bits(&args[13], "RV_F32_BITS")?,
        parse_f32_bits(&args[14], "P00_F32_BITS")?,
        parse_u8(&args[15], "PSOLVER")?,
        parse_u8(&args[16], "IBALANCE")?,
        parse_bool(&args[17], "SETPPI")?,
        parse_u8(&args[18], "IICE")?,
    )?;
    let crop = if args.len() == 23 {
        Some(HorizontalCrop {
            j_start: parse_usize(&args[19], "J_START")?,
            ny: parse_usize(&args[20], "J_COUNT")?,
            i_start: parse_usize(&args[21], "I_START")?,
            nx: parse_usize(&args[22], "I_COUNT")?,
        })
    } else {
        None
    };
    let receipt = if f02_roi {
        export_initialized_product_f02_roi_history_new(
            PathBuf::from(&args[0]),
            PathBuf::from(&args[1]),
            &args[2],
            &args[3],
            crop,
            terrain_config,
            moist_config,
            InitializedProductAdmission::RealProduct,
        )?
    } else {
        export_initialized_product_history_new(
            PathBuf::from(&args[0]),
            PathBuf::from(&args[1]),
            &args[2],
            &args[3],
            crop,
            terrain_config,
            moist_config,
            InitializedProductAdmission::RealProduct,
        )?
    };
    println!("{}", receipt.canonical_json());
    Ok(())
}
