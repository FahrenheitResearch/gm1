use cm1_state::forecast::{
    read_forecast_checkpoint, read_forecast_history, write_forecast_checkpoint_new,
    write_forecast_history_new,
};
use cm1_state::hrrr::{
    configuration_sha256, execution_identity_sha256, forecast_contract, map_to_cm1, Cm1TargetGrid,
    DecodedHrrrWindow, HrrrLambertGrid, InitializationReceipt,
};
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn parse<T: std::str::FromStr>(value: &str, name: &str) -> Result<T, Box<dyn Error>>
where
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}").into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 && args.len() != 11 {
        return Err(concat!(
            "usage: hrrr_cm1_init SOURCE_ROOT OUTPUT_DIR CENTER_LAT CENTER_LON ",
            "[NX NY NZ DX_M DY_M DZ_M MODEL_DT_S]"
        )
        .into());
    }
    let source_root = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);
    if output.exists() {
        return Err(format!("refusing to replace {}", output.display()).into());
    }
    let center_lat = parse::<f64>(&args[2], "CENTER_LAT")?;
    let center_lon = parse::<f64>(&args[3], "CENTER_LON")?;
    let (nx, ny, nz, dx, dy, dz, model_dt) = if args.len() == 11 {
        (
            parse(&args[4], "NX")?,
            parse(&args[5], "NY")?,
            parse(&args[6], "NZ")?,
            parse(&args[7], "DX_M")?,
            parse(&args[8], "DY_M")?,
            parse(&args[9], "DZ_M")?,
            parse(&args[10], "MODEL_DT_S")?,
        )
    } else {
        (300, 300, 75, 100.0, 100.0, 200.0, 0.5)
    };
    let source_grid = HrrrLambertGrid::canonical_window(971, 622, 26, 26)?;
    let target = Cm1TargetGrid::new(nx, ny, nz, dx, dy, dz, center_lat, center_lon)?;
    let started = Instant::now();
    let source0 =
        DecodedHrrrWindow::read_directory(source_root.join("atmosphere-f00"), source_grid, 0)?;
    let source1 =
        DecodedHrrrWindow::read_directory(source_root.join("atmosphere-f01"), source_grid, 1)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let map_started = Instant::now();
    let mapped0 = map_to_cm1(&source0, target)?;
    let mapped1 = map_to_cm1(&source1, target)?;
    let map_seconds = map_started.elapsed().as_secs_f64();
    let initial_stats = mapped0.stats().clone();
    let boundary_stats = mapped1.stats().clone();
    let initial_source_sha = source0.source_fingerprint_sha256().to_owned();
    let boundary_source_sha = source1.source_fingerprint_sha256().to_owned();
    let config_sha = configuration_sha256(source_grid, target, model_dt)?;
    let contract = forecast_contract(target, config_sha, execution_identity_sha256())?;
    fs::create_dir(&output)?;
    let write_started = Instant::now();
    let checkpoint_path = output.join("initial.cm1fcp");
    let boundary_path = output.join("boundary-f01.cm1fhs");
    let checkpoint_receipt = write_forecast_checkpoint_new(
        &contract,
        &mapped0.into_forecast_frame(model_dt)?,
        &checkpoint_path,
    )?;
    let boundary_receipt = write_forecast_history_new(
        &contract,
        &mapped1.into_forecast_frame(model_dt)?,
        &boundary_path,
    )?;
    let restored0 = read_forecast_checkpoint(&contract, &checkpoint_path)?;
    let restored1 = read_forecast_history(&contract, &boundary_path)?;
    if restored0.receipt() != &checkpoint_receipt || restored1.receipt() != &boundary_receipt {
        return Err("post-publication forecast restore receipt mismatch".into());
    }
    let receipt = InitializationReceipt::new(
        target,
        &contract,
        initial_source_sha,
        boundary_source_sha,
        checkpoint_receipt,
        boundary_receipt,
        initial_stats,
        boundary_stats,
    )?;
    receipt.write_new(output.join("initialization-receipt.json"))?;
    let write_seconds = write_started.elapsed().as_secs_f64();
    println!("{}", receipt.canonical_json());
    eprintln!(
        "PASS load_seconds={load_seconds:.6} map_seconds={map_seconds:.6} write_restore_seconds={write_seconds:.6} total_seconds={:.6}",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
