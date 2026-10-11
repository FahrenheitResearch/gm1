use cm1_state::child_remap::ChildGridSpec;
use cm1_state::child_terrain::{
    publish_child_terrain_for_grid_new, publish_child_terrain_new, read_child_terrain_artifact,
    read_child_terrain_for_grid_artifact, CHILD_TERRAIN_ABI, CHILD_TERRAIN_GRID_ABI,
    CHILD_TERRAIN_GRID_MANIFEST_SCHEMA, CHILD_TERRAIN_GRID_SCHEMA, CHILD_TERRAIN_MANIFEST_SCHEMA,
    CHILD_TERRAIN_SCHEMA,
};
use cm1_state::{Result, StateError};
use std::env;
use std::process::ExitCode;

const USAGE: &str = "usage:\n  gm1-child-terrain --abi\n  gm1-child-terrain publish VALIDATED_PLAN.json SOURCE_PRODUCT.cm1ipr PRODUCER_SOURCE.rs NEW_DIRECTORY\n  gm1-child-terrain inspect VALIDATED_PLAN.json SOURCE_PRODUCT.cm1ipr PRODUCER_SOURCE.rs PRODUCER_EXE DIRECTORY MANIFEST_SHA PAYLOAD_SHA RECEIPT_SHA\n  gm1-child-terrain publish-grid NX NY NZ DX_M DY_M ZTOP_M VALIDATED_PLAN.json SOURCE_PRODUCT.cm1ipr PRODUCER_SOURCE.rs NEW_DIRECTORY\n  gm1-child-terrain inspect-grid NX NY NZ DX_M DY_M ZTOP_M VALIDATED_PLAN.json SOURCE_PRODUCT.cm1ipr PRODUCER_SOURCE.rs PRODUCER_EXE DIRECTORY MANIFEST_SHA PAYLOAD_SHA RECEIPT_SHA";

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn parse_grid(
    nx: &str,
    ny: &str,
    nz: &str,
    dx_m: &str,
    dy_m: &str,
    ztop_m: &str,
) -> Result<ChildGridSpec> {
    let nx = nx
        .parse::<usize>()
        .map_err(|error| invalid(format!("invalid NX: {error}")))?;
    let ny = ny
        .parse::<usize>()
        .map_err(|error| invalid(format!("invalid NY: {error}")))?;
    let nz = nz
        .parse::<usize>()
        .map_err(|error| invalid(format!("invalid NZ: {error}")))?;
    let dx_m = dx_m
        .parse::<f64>()
        .map_err(|error| invalid(format!("invalid DX_M: {error}")))?;
    let dy_m = dy_m
        .parse::<f64>()
        .map_err(|error| invalid(format!("invalid DY_M: {error}")))?;
    let ztop_m = ztop_m
        .parse::<f64>()
        .map_err(|error| invalid(format!("invalid ZTOP_M: {error}")))?;
    ChildGridSpec::new(nx, ny, nz, dx_m, dy_m, ztop_m)
}

fn run(arguments: &[String]) -> Result<String> {
    if arguments == ["--abi"] {
        return Ok(format!(
            "schema={CHILD_TERRAIN_SCHEMA};manifest={CHILD_TERRAIN_MANIFEST_SCHEMA};abi={CHILD_TERRAIN_ABI};grid_schema={CHILD_TERRAIN_GRID_SCHEMA};grid_manifest={CHILD_TERRAIN_GRID_MANIFEST_SCHEMA};grid_abi={CHILD_TERRAIN_GRID_ABI}"
        ));
    }
    match arguments {
        [command, plan, source_restart, producer_source, output] if command == "publish" => {
            let executable = env::current_exe()?;
            let receipt = publish_child_terrain_new(
                plan,
                source_restart,
                producer_source,
                &executable,
                output,
            )?;
            Ok(format!(
                "PASS_CHILD_TERRAIN manifest_sha256={} payload_sha256={} receipt_sha256={}",
                receipt.manifest_sha256, receipt.payload_sha256, receipt.receipt_sha256
            ))
        }
        [command, plan, source_restart, producer_source, producer_executable, directory, manifest, payload, receipt]
            if command == "inspect" =>
        {
            let restored = read_child_terrain_artifact(
                plan,
                source_restart,
                producer_source,
                producer_executable,
                directory,
                manifest,
                payload,
                receipt,
            )?;
            Ok(format!(
                "PASS_CHILD_TERRAIN_INSPECT manifest_sha256={} payload_sha256={} receipt_sha256={}",
                restored.manifest_sha256, restored.payload_sha256, restored.receipt_sha256
            ))
        }
        [command, nx, ny, nz, dx_m, dy_m, ztop_m, plan, source_restart, producer_source, output]
            if command == "publish-grid" =>
        {
            let grid = parse_grid(nx, ny, nz, dx_m, dy_m, ztop_m)?;
            let executable = env::current_exe()?;
            let receipt = publish_child_terrain_for_grid_new(
                grid,
                plan,
                source_restart,
                producer_source,
                &executable,
                output,
            )?;
            Ok(format!(
                "PASS_CHILD_TERRAIN_GRID manifest_sha256={} payload_sha256={} receipt_sha256={}",
                receipt.manifest_sha256, receipt.payload_sha256, receipt.receipt_sha256
            ))
        }
        [command, nx, ny, nz, dx_m, dy_m, ztop_m, plan, source_restart, producer_source, producer_executable, directory, manifest, payload, receipt]
            if command == "inspect-grid" =>
        {
            let grid = parse_grid(nx, ny, nz, dx_m, dy_m, ztop_m)?;
            let restored = read_child_terrain_for_grid_artifact(
                grid,
                plan,
                source_restart,
                producer_source,
                producer_executable,
                directory,
                manifest,
                payload,
                receipt,
            )?;
            Ok(format!(
                "PASS_CHILD_TERRAIN_GRID_INSPECT manifest_sha256={} payload_sha256={} receipt_sha256={}",
                restored.manifest_sha256,
                restored.payload_sha256,
                restored.receipt_sha256
            ))
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
            eprintln!("gm1-child-terrain: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_is_exact_and_usage_fails_closed() {
        let abi = run(&["--abi".to_owned()]).unwrap();
        assert!(abi.contains(CHILD_TERRAIN_SCHEMA));
        assert!(abi.contains(CHILD_TERRAIN_ABI));
        assert!(abi.contains(CHILD_TERRAIN_GRID_SCHEMA));
        assert!(abi.contains(CHILD_TERRAIN_GRID_ABI));
        assert!(run(&[]).is_err());
    }

    #[test]
    fn grid_parser_admits_exact_5m_domain_and_refuses_resizing() {
        let grid = parse_grid("3000", "3000", "75", "5", "5", "15000").unwrap();
        assert_eq!(grid.nx(), 3_000);
        assert_eq!(grid.ny(), 3_000);
        assert_eq!(grid.dx_m().to_bits(), 5.0f64.to_bits());
        assert!(parse_grid("3000", "3000", "75", "6", "5", "15000").is_err());
        assert!(parse_grid("3000", "3000", "76", "5", "5", "15000").is_err());
        assert!(parse_grid("bad", "3000", "75", "5", "5", "15000").is_err());
    }
}
