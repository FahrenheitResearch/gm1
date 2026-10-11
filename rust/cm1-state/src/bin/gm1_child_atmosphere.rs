use cm1_state::child_atmosphere::{
    child_static_height_identity_from_artifact, hard_5m_child_static_height_identity,
    publish_child_atmosphere_for_grid_new, publish_child_atmosphere_new,
    read_child_atmosphere_artifact, read_child_atmosphere_for_grid_artifact,
    replay_sealed_child_atmosphere_raw_streaming, ChildAtmosphereStaticHeightIdentity,
    CHILD_ATMOSPHERE_ABI, CHILD_ATMOSPHERE_GRID_ABI, CHILD_ATMOSPHERE_GRID_MANIFEST_SCHEMA,
    CHILD_ATMOSPHERE_GRID_SCHEMA, CHILD_ATMOSPHERE_MANIFEST_SCHEMA, CHILD_ATMOSPHERE_SCHEMA,
    CHILD_ATMOSPHERE_STREAM_REPLAY_STATUS,
};
use cm1_state::child_remap::{ChildGridSpec, HARD_5M_CHILD_GRID};
use cm1_state::{Result, StateError};
use std::env;
use std::process::ExitCode;

const USAGE: &str = "usage:\n  gm1-child-atmosphere --abi\n  gm1-child-atmosphere publish PLAN.json PARENT.cm1fcp STATIC_HEIGHT_DIR STATIC_VALIDATION.json restart_read.F init3d.F PRODUCER_SOURCE.rs NEW_DIRECTORY\n  gm1-child-atmosphere inspect PLAN.json PARENT.cm1fcp STATIC_HEIGHT_DIR STATIC_VALIDATION.json restart_read.F init3d.F PRODUCER_SOURCE.rs PRODUCER_EXE DIRECTORY MANIFEST_SHA ARENA_SHA RECEIPT_SHA\n  gm1-child-atmosphere replay-sealed-raw PLAN.json PARENT.cm1fcp DIRECTORY\n  gm1-child-atmosphere publish-grid (5m|from-static-heights) PLAN.json PARENT.cm1fcp STATIC_HEIGHT_DIR STATIC_VALIDATION.json restart_read.F init3d.F PRODUCER_SOURCE.rs NEW_DIRECTORY\n  gm1-child-atmosphere inspect-grid (5m|from-static-heights) PLAN.json PARENT.cm1fcp STATIC_HEIGHT_DIR STATIC_VALIDATION.json restart_read.F init3d.F PRODUCER_SOURCE.rs PRODUCER_EXE DIRECTORY MANIFEST_SHA ARENA_SHA RECEIPT_SHA";

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

/// Resolve the child grid and the static-height identity it is built on.
///
/// `from-static-heights` is the general form: the child grid is whatever the
/// admitted static-height artifact publishes, read back from that artifact's
/// own receipt, so a 50 m child - or any other admitted refinement - is built
/// through the same path as the 5 m one instead of being reachable only through
/// a frozen preset.  The two presets remain as shorthands for the grids whose
/// static-height identities are pinned in the library.
fn resolve_grid(
    value: &str,
    static_heights: &str,
    static_receipt: &str,
) -> Result<(ChildGridSpec, ChildAtmosphereStaticHeightIdentity)> {
    match value {
        "5m" => Ok((HARD_5M_CHILD_GRID, hard_5m_child_static_height_identity()?)),
        "from-static-heights" => {
            let identity =
                child_static_height_identity_from_artifact(static_heights, static_receipt)?;
            let grid = identity.grid();
            Ok((grid, identity))
        }
        other => Err(invalid(format!(
            "child grid selector {other:?} is unsupported; pass 5m for the pinned 3000x3000 deck or from-static-heights to build the grid the admitted static-height artifact publishes"
        ))),
    }
}

fn run(arguments: &[String]) -> Result<String> {
    if arguments == ["--abi"] {
        return Ok(format!(
            "schema={CHILD_ATMOSPHERE_SCHEMA};manifest={CHILD_ATMOSPHERE_MANIFEST_SCHEMA};abi={CHILD_ATMOSPHERE_ABI};grid_schema={CHILD_ATMOSPHERE_GRID_SCHEMA};grid_manifest={CHILD_ATMOSPHERE_GRID_MANIFEST_SCHEMA};grid_abi={CHILD_ATMOSPHERE_GRID_ABI};carriers=28;closure.tkea=REQUIRED_CHILD_CLOSURE_INITIALIZATION"
        ));
    }
    match arguments {
        [command, plan, checkpoint, static_heights, static_receipt, restart_read, init3d, producer_source, output]
            if command == "publish" =>
        {
            let executable = env::current_exe()?;
            let receipt = publish_child_atmosphere_new(
                plan,
                checkpoint,
                static_heights,
                static_receipt,
                restart_read,
                init3d,
                producer_source,
                &executable,
                output,
            )?;
            Ok(format!(
                "PASS_CHILD_ATMOSPHERE manifest_sha256={} arena_sha256={} receipt_sha256={} arena_nbytes={}",
                receipt.manifest_sha256,
                receipt.arena_sha256,
                receipt.receipt_sha256,
                receipt.arena_nbytes
            ))
        }
        [command, plan, checkpoint, static_heights, static_receipt, restart_read, init3d, producer_source, producer_executable, directory, manifest, arena, receipt]
            if command == "inspect" =>
        {
            let restored = read_child_atmosphere_artifact(
                plan,
                checkpoint,
                static_heights,
                static_receipt,
                restart_read,
                init3d,
                producer_source,
                producer_executable,
                directory,
                manifest,
                arena,
                receipt,
            )?;
            Ok(format!(
                "PASS_CHILD_ATMOSPHERE_INSPECT manifest_sha256={} arena_sha256={} receipt_sha256={} arena_nbytes={}",
                restored.manifest_sha256,
                restored.arena_sha256,
                restored.receipt_sha256,
                restored.arena_nbytes
            ))
        }
        [command, plan, checkpoint, directory] if command == "replay-sealed-raw" => {
            let replay = replay_sealed_child_atmosphere_raw_streaming(plan, checkpoint, directory)?;
            Ok(format!(
                "{CHILD_ATMOSPHERE_STREAM_REPLAY_STATUS} manifest_sha256={} arena_sha256={} receipt_sha256={} arena_nbytes={} carrier_count={} stream_chunk_bytes={}",
                replay.manifest_sha256,
                replay.arena_sha256,
                replay.receipt_sha256,
                replay.arena_nbytes,
                replay.carrier_count,
                replay.stream_chunk_bytes,
            ))
        }
        [command, grid, plan, checkpoint, static_heights, static_receipt, restart_read, init3d, producer_source, output]
            if command == "publish-grid" =>
        {
            let (grid, static_identity) = resolve_grid(grid, static_heights, static_receipt)?;
            let executable = env::current_exe()?;
            let receipt = publish_child_atmosphere_for_grid_new(
                grid,
                &static_identity,
                plan,
                checkpoint,
                static_heights,
                static_receipt,
                restart_read,
                init3d,
                producer_source,
                &executable,
                output,
            )?;
            Ok(format!(
                "PASS_CHILD_ATMOSPHERE_GRID manifest_sha256={} arena_sha256={} receipt_sha256={} arena_nbytes={}",
                receipt.manifest_sha256,
                receipt.arena_sha256,
                receipt.receipt_sha256,
                receipt.arena_nbytes
            ))
        }
        [command, grid, plan, checkpoint, static_heights, static_receipt, restart_read, init3d, producer_source, producer_executable, directory, manifest, arena, receipt]
            if command == "inspect-grid" =>
        {
            let (grid, static_identity) = resolve_grid(grid, static_heights, static_receipt)?;
            let restored = read_child_atmosphere_for_grid_artifact(
                grid,
                &static_identity,
                plan,
                checkpoint,
                static_heights,
                static_receipt,
                restart_read,
                init3d,
                producer_source,
                producer_executable,
                directory,
                manifest,
                arena,
                receipt,
            )?;
            Ok(format!(
                "PASS_CHILD_ATMOSPHERE_GRID_INSPECT manifest_sha256={} arena_sha256={} receipt_sha256={} arena_nbytes={}",
                restored.manifest_sha256,
                restored.arena_sha256,
                restored.receipt_sha256,
                restored.arena_nbytes
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
            eprintln!("gm1-child-atmosphere: {error}");
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
        assert!(abi.contains(CHILD_ATMOSPHERE_SCHEMA));
        assert!(abi.contains(CHILD_ATMOSPHERE_ABI));
        assert!(abi.contains(CHILD_ATMOSPHERE_GRID_ABI));
        assert!(abi.contains("carriers=28"));
        assert!(abi.contains("closure.tkea=REQUIRED_CHILD_CLOSURE_INITIALIZATION"));
        assert!(run(&[]).is_err());
        assert!(USAGE.contains("replay-sealed-raw"));
        assert!(USAGE.contains("publish-grid (5m|from-static-heights)"));
        // The pinned 5 m preset still resolves without touching any artifact.
        let (grid, identity) = resolve_grid("5m", "", "").unwrap();
        assert_eq!(grid, HARD_5M_CHILD_GRID);
        assert_eq!(identity.grid(), HARD_5M_CHILD_GRID);
        // An unknown selector is refused by value and names the general form.
        let error = resolve_grid("50m", "", "").unwrap_err().to_string();
        assert!(error.contains("from-static-heights"), "{error}");
        // The general form reads the grid from a published artifact, so it
        // fails on a missing one rather than refusing the selector itself.
        let error = resolve_grid("from-static-heights", "", "")
            .unwrap_err()
            .to_string();
        assert!(!error.contains("is unsupported"), "{error}");
    }
}
