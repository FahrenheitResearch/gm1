use cm1_state::child_auxiliary::{
    admit_child_auxiliary_base_capacity, child_auxiliary_base_layout_for_grid,
    inspect_child_auxiliary, publish_child_auxiliary_new, ChildAuxiliaryBaseCapacityGate,
    ChildAuxiliaryBaseLayout, CHILD_AUXILIARY_BASE_CAPACITY_SCHEMA,
    CHILD_AUXILIARY_BASE_LAYOUT_SCHEMA,
};
use cm1_state::child_auxiliary_stream::{
    inspect_child_auxiliary_stream, publish_child_auxiliary_stream_new, ChildAuxiliaryStreamReceipt,
};
use cm1_state::child_remap::{ChildGridSpec, ACCEPTED_50M_CHILD_GRID, HARD_5M_CHILD_GRID};
use std::env;
use std::path::PathBuf;

fn usage() -> &'static str {
    "usage:\n  gm1-child-auxiliary layout <50m|5m>\n  gm1-child-auxiliary capacity <50m|5m> <free-bytes> <reserve-bytes> <planned-payload-copies>\n  gm1-child-auxiliary stream-build <50m|5m> <free-bytes> <reserve-bytes> <atmosphere-dir> <terrain-dir> <static-height-dir> <static-validation-receipt> <f02-product.cm1ipr> <producer-source.rs> <output-dir>\n  gm1-child-auxiliary stream-inspect <50m|5m> <free-bytes> <reserve-bytes> <atmosphere-dir> <terrain-dir> <static-height-dir> <static-validation-receipt> <f02-product.cm1ipr> <producer-source.rs> <artifact-dir> <manifest-sha256> <terrain-open-sha256> <live-density-sha256> <geopotential-sha256> <base-wind-sha256> <receipt-sha256>\n  gm1-child-auxiliary build <atmosphere-dir> <terrain-dir> <static-height-dir> <static-validation-receipt> <f02-product.cm1ipr> <cm1r21.1-root> <producer-source.rs> <output-dir>\n  gm1-child-auxiliary inspect <atmosphere-dir> <terrain-dir> <static-height-dir> <static-validation-receipt> <f02-product.cm1ipr> <cm1r21.1-root> <producer-source.rs> <artifact-dir> <manifest-sha256> <terrain-open-sha256> <live-density-sha256> <geopotential-sha256> <receipt-sha256>"
}

fn main() {
    if let Err(error) = run() {
        eprintln!("gm1-child-auxiliary: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = env::args_os().skip(1).collect::<Vec<_>>();
    let command = arguments
        .first()
        .and_then(|value| value.to_str())
        .ok_or_else(|| usage().to_string())?;
    let executable = env::current_exe()?;
    match command {
        "layout" if arguments.len() == 2 => {
            let grid = parse_grid(&arguments[1])?;
            print_layout(&child_auxiliary_base_layout_for_grid(grid)?);
        }
        "capacity" if arguments.len() == 5 => {
            let grid = parse_grid(&arguments[1])?;
            let free_bytes = parse_u64(&arguments[2], "free bytes")?;
            let reserve_bytes = parse_u64(&arguments[3], "reserve bytes")?;
            let planned_payload_copies = parse_u64(&arguments[4], "planned payload copies")?;
            print_capacity(&admit_child_auxiliary_base_capacity(
                grid,
                free_bytes,
                reserve_bytes,
                planned_payload_copies,
            )?);
        }
        "stream-build" if arguments.len() == 11 => {
            let grid = parse_grid(&arguments[1])?;
            let free_bytes = parse_u64(&arguments[2], "free bytes")?;
            let reserve_bytes = parse_u64(&arguments[3], "reserve bytes")?;
            let receipt = publish_child_auxiliary_stream_new(
                grid,
                PathBuf::from(&arguments[4]),
                PathBuf::from(&arguments[5]),
                PathBuf::from(&arguments[6]),
                PathBuf::from(&arguments[7]),
                PathBuf::from(&arguments[8]),
                PathBuf::from(&arguments[9]),
                &executable,
                PathBuf::from(&arguments[10]),
                free_bytes,
                reserve_bytes,
            )?;
            print_stream_receipt("built", &receipt);
        }
        "stream-inspect" if arguments.len() == 17 => {
            let grid = parse_grid(&arguments[1])?;
            let free_bytes = parse_u64(&arguments[2], "free bytes")?;
            let reserve_bytes = parse_u64(&arguments[3], "reserve bytes")?;
            let receipt = inspect_child_auxiliary_stream(
                grid,
                PathBuf::from(&arguments[4]),
                PathBuf::from(&arguments[5]),
                PathBuf::from(&arguments[6]),
                PathBuf::from(&arguments[7]),
                PathBuf::from(&arguments[8]),
                PathBuf::from(&arguments[9]),
                &executable,
                PathBuf::from(&arguments[10]),
                free_bytes,
                reserve_bytes,
                arguments[11].to_str().ok_or("manifest SHA must be UTF-8")?,
                arguments[12]
                    .to_str()
                    .ok_or("terrain/open SHA must be UTF-8")?,
                arguments[13].to_str().ok_or("density SHA must be UTF-8")?,
                arguments[14]
                    .to_str()
                    .ok_or("geopotential SHA must be UTF-8")?,
                arguments[15]
                    .to_str()
                    .ok_or("base-wind SHA must be UTF-8")?,
                arguments[16].to_str().ok_or("receipt SHA must be UTF-8")?,
            )?;
            print_stream_receipt("inspected", &receipt);
        }
        "build" if arguments.len() == 9 => {
            let receipt = publish_child_auxiliary_new(
                PathBuf::from(&arguments[1]),
                PathBuf::from(&arguments[2]),
                PathBuf::from(&arguments[3]),
                PathBuf::from(&arguments[4]),
                PathBuf::from(&arguments[5]),
                PathBuf::from(&arguments[6]),
                PathBuf::from(&arguments[7]),
                &executable,
                PathBuf::from(&arguments[8]),
            )?;
            print_receipt("built", &receipt);
        }
        "inspect" if arguments.len() == 14 => {
            let receipt = inspect_child_auxiliary(
                PathBuf::from(&arguments[1]),
                PathBuf::from(&arguments[2]),
                PathBuf::from(&arguments[3]),
                PathBuf::from(&arguments[4]),
                PathBuf::from(&arguments[5]),
                PathBuf::from(&arguments[6]),
                PathBuf::from(&arguments[7]),
                &executable,
                PathBuf::from(&arguments[8]),
                arguments[9].to_str().ok_or("manifest SHA must be UTF-8")?,
                arguments[10]
                    .to_str()
                    .ok_or("terrain/open SHA must be UTF-8")?,
                arguments[11].to_str().ok_or("density SHA must be UTF-8")?,
                arguments[12]
                    .to_str()
                    .ok_or("geopotential SHA must be UTF-8")?,
                arguments[13].to_str().ok_or("receipt SHA must be UTF-8")?,
            )?;
            print_receipt("inspected", &receipt);
        }
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn parse_grid(value: &std::ffi::OsStr) -> Result<ChildGridSpec, Box<dyn std::error::Error>> {
    match value.to_str() {
        Some("50m") => Ok(ACCEPTED_50M_CHILD_GRID),
        Some("5m") => Ok(HARD_5M_CHILD_GRID),
        _ => Err("grid must be exactly 50m or 5m".into()),
    }
}

fn parse_u64(value: &std::ffi::OsStr, label: &str) -> Result<u64, Box<dyn std::error::Error>> {
    value
        .to_str()
        .ok_or_else(|| format!("{label} must be UTF-8"))?
        .parse::<u64>()
        .map_err(|_| format!("{label} must be an unsigned decimal integer").into())
}

fn print_layout(layout: &ChildAuxiliaryBaseLayout) {
    println!(
        "{{\"base_wind_nbytes\":{},\"dx_m\":{},\"dy_m\":{},\"geopotential_nbytes\":{},\"grid_nx\":{},\"grid_ny\":{},\"grid_nz\":{},\"live_density_carrier_count\":{},\"live_density_nbytes\":{},\"schema\":\"{}\",\"terrain_open_carrier_count\":{},\"terrain_open_nbytes\":{},\"total_nbytes\":{},\"u0_nbytes\":{},\"v0_nbytes\":{}}}",
        layout.base_wind_nbytes,
        layout.grid.dx_m(),
        layout.grid.dy_m(),
        layout.geopotential_nbytes,
        layout.grid.nx(),
        layout.grid.ny(),
        layout.grid.nz(),
        layout.live_density_carrier_count,
        layout.live_density_nbytes,
        CHILD_AUXILIARY_BASE_LAYOUT_SCHEMA,
        layout.terrain_open_carrier_count,
        layout.terrain_open_nbytes,
        layout.total_nbytes,
        layout.u0_nbytes,
        layout.v0_nbytes,
    );
}

fn print_capacity(gate: &ChildAuxiliaryBaseCapacityGate) {
    println!(
        "{{\"free_bytes\":{},\"planned_payload_copies\":{},\"post_publish_above_reserve_bytes\":{},\"post_publish_free_bytes\":{},\"reserve_bytes\":{},\"schema\":\"{}\",\"status\":\"PASS_SINGLE_COPY_STREAMED_CAPACITY\",\"total_nbytes\":{}}}",
        gate.free_bytes,
        gate.planned_payload_copies,
        gate.post_publish_above_reserve_bytes,
        gate.post_publish_free_bytes,
        gate.reserve_bytes,
        CHILD_AUXILIARY_BASE_CAPACITY_SCHEMA,
        gate.layout.total_nbytes,
    );
}

fn print_receipt(action: &str, receipt: &cm1_state::child_auxiliary::ChildAuxiliaryReceipt) {
    println!(
        "{{\"action\":\"{action}\",\"geopotential_sha256\":\"{}\",\"live_density_sha256\":\"{}\",\"manifest_sha256\":\"{}\",\"receipt_sha256\":\"{}\",\"terrain_open_sha256\":\"{}\"}}",
        receipt.geopotential_sha256,
        receipt.live_density_sha256,
        receipt.manifest_sha256,
        receipt.receipt_sha256,
        receipt.terrain_open_sha256,
    );
}

fn print_stream_receipt(action: &str, receipt: &ChildAuxiliaryStreamReceipt) {
    println!(
        "{{\"action\":\"{action}\",\"base_wind_sha256\":\"{}\",\"geopotential_sha256\":\"{}\",\"live_density_sha256\":\"{}\",\"manifest_sha256\":\"{}\",\"receipt_sha256\":\"{}\",\"terrain_open_sha256\":\"{}\"}}",
        receipt.base_wind_sha256,
        receipt.geopotential_sha256,
        receipt.live_density_sha256,
        receipt.manifest_sha256,
        receipt.receipt_sha256,
        receipt.terrain_open_sha256,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn grid_parser_exposes_only_the_two_admitted_presets() {
        assert_eq!(
            parse_grid(OsStr::new("50m")).unwrap(),
            ACCEPTED_50M_CHILD_GRID
        );
        assert_eq!(parse_grid(OsStr::new("5m")).unwrap(), HARD_5M_CHILD_GRID);
        assert!(parse_grid(OsStr::new("10m")).is_err());
    }

    #[test]
    fn decimal_capacity_parser_refuses_signed_or_nondecimal_values() {
        assert_eq!(
            parse_u64(OsStr::new("905475260416"), "free").unwrap(),
            905_475_260_416
        );
        assert!(parse_u64(OsStr::new("-1"), "free").is_err());
        assert!(parse_u64(OsStr::new("0x100"), "free").is_err());
    }
}
