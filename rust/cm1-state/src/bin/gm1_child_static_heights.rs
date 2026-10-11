use cm1_state::child_remap::{ChildGridSpec, ACCEPTED_50M_CHILD_GRID, HARD_5M_CHILD_GRID};
use cm1_state::child_static_heights::{
    publish_child_static_heights_for_grid_new, read_child_static_heights_for_grid_artifact,
    ChildTerrainArtifactIdentity, CHILD_STATIC_HEIGHT_GRID_ABI,
    CHILD_STATIC_HEIGHT_GRID_MANIFEST_SCHEMA, CHILD_STATIC_HEIGHT_GRID_SCHEMA,
};
use cm1_state::{Result, StateError};
use std::env;
use std::process::ExitCode;

const USAGE: &str = "usage:\n  gm1-child-static-heights --abi\n  gm1-child-static-heights publish <50m|5m> TERRAIN_DIR TERRAIN_MANIFEST_SHA TERRAIN_PAYLOAD_SHA TERRAIN_RECEIPT_SHA init_terrain.F PRODUCER_SOURCE.rs NEW_DIRECTORY\n  gm1-child-static-heights inspect <50m|5m> TERRAIN_DIR TERRAIN_MANIFEST_SHA TERRAIN_PAYLOAD_SHA TERRAIN_RECEIPT_SHA init_terrain.F PRODUCER_SOURCE.rs PRODUCER_EXE DIRECTORY MANIFEST_SHA ZH_SHA ZF_SHA RECEIPT_SHA";

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn parse_grid(value: &str) -> Result<ChildGridSpec> {
    match value {
        "50m" => Ok(ACCEPTED_50M_CHILD_GRID),
        "5m" => Ok(HARD_5M_CHILD_GRID),
        _ => Err(invalid(
            "child static-height grid must be exactly 50m or 5m",
        )),
    }
}

fn run(arguments: &[String]) -> Result<String> {
    if arguments == ["--abi"] {
        return Ok(format!(
            "schema={CHILD_STATIC_HEIGHT_GRID_SCHEMA};manifest={CHILD_STATIC_HEIGHT_GRID_MANIFEST_SCHEMA};abi={CHILD_STATIC_HEIGHT_GRID_ABI};grids=50m,5m;payloads=zh,zf"
        ));
    }
    match arguments {
        [command, grid, terrain_directory, terrain_manifest, terrain_payload, terrain_receipt, init_terrain, producer_source, output]
            if command == "publish" =>
        {
            let grid = parse_grid(grid)?;
            let terrain = ChildTerrainArtifactIdentity::new(
                terrain_manifest,
                terrain_payload,
                terrain_receipt,
            )?;
            let executable = env::current_exe()?;
            let receipt = publish_child_static_heights_for_grid_new(
                grid,
                terrain_directory,
                terrain,
                init_terrain,
                producer_source,
                &executable,
                output,
            )?;
            Ok(format!(
                "PASS_CHILD_STATIC_HEIGHTS manifest_sha256={} zh_sha256={} zf_sha256={} receipt_sha256={}",
                receipt.manifest_sha256,
                receipt.zh_sha256,
                receipt.zf_sha256,
                receipt.receipt_sha256,
            ))
        }
        [command, grid, terrain_directory, terrain_manifest, terrain_payload, terrain_receipt, init_terrain, producer_source, producer_executable, directory, manifest, zh, zf, receipt]
            if command == "inspect" =>
        {
            let grid = parse_grid(grid)?;
            let terrain = ChildTerrainArtifactIdentity::new(
                terrain_manifest,
                terrain_payload,
                terrain_receipt,
            )?;
            let restored = read_child_static_heights_for_grid_artifact(
                grid,
                terrain_directory,
                terrain,
                init_terrain,
                producer_source,
                producer_executable,
                directory,
                manifest,
                zh,
                zf,
                receipt,
            )?;
            Ok(format!(
                "PASS_CHILD_STATIC_HEIGHTS_INSPECT manifest_sha256={} zh_sha256={} zf_sha256={} receipt_sha256={}",
                restored.manifest_sha256,
                restored.zh_sha256,
                restored.zf_sha256,
                restored.receipt_sha256,
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
            eprintln!("gm1-child-static-heights: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_and_grid_parser_are_strict() {
        let abi = run(&["--abi".to_owned()]).unwrap();
        assert!(abi.contains(CHILD_STATIC_HEIGHT_GRID_SCHEMA));
        assert!(abi.contains("payloads=zh,zf"));
        assert_eq!(parse_grid("50m").unwrap(), ACCEPTED_50M_CHILD_GRID);
        assert_eq!(parse_grid("5m").unwrap(), HARD_5M_CHILD_GRID);
        assert!(parse_grid("25m").is_err());
        assert!(run(&[]).is_err());
    }
}
