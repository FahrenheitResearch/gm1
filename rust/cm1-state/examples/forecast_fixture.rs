use cm1_state::forecast::{
    read_forecast_checkpoint, read_forecast_history, write_forecast_checkpoint_new,
    write_forecast_history_new, ComponentBlob, ComponentBlobSpec, ForecastArray, ForecastArraySpec,
    ForecastClock, ForecastContract, ForecastFrame, SchedulerScalar, SchedulerScalarSpec,
    SchedulerScalarType, SchedulerScalarValue,
};
use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

fn hash(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn contract() -> cm1_state::Result<ForecastContract> {
    ForecastContract::new(
        "cm1-forecast-api-evidence/v1",
        hash(0x33),
        hash(0x44),
        vec![
            ForecastArraySpec::new(
                "th3d",
                "current potential-temperature perturbation",
                "K",
                "scalar",
                vec![1, 2, 1, 2],
                vec![2, 2],
            )?,
            ForecastArraySpec::new(
                "pp3d",
                "current Exner pressure perturbation",
                "1",
                "scalar",
                vec![1, 2, 1, 2],
                vec![2, 2],
            )?,
        ],
        vec![
            SchedulerScalarSpec::new("history_record", SchedulerScalarType::U64)?,
            SchedulerScalarSpec::new("domainlocx", SchedulerScalarType::F64)?,
            SchedulerScalarSpec::new("moving_frame_active", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("restart_slot", SchedulerScalarType::I64)?,
        ],
        vec![ComponentBlobSpec::new(
            "arwen_wsm6",
            "arwen-wsm6-persistence/v1",
            64,
        )?],
    )
}

fn frame() -> cm1_state::Result<ForecastFrame> {
    Ok(ForecastFrame::new(
        ForecastClock::new(600, 3600.0, 6.0)?,
        vec![
            ForecastArray::new("th3d", vec![1.0, -0.0, f32::from_bits(1), 4.0]),
            ForecastArray::new("pp3d", vec![0.1, 0.2, 0.3, 0.4]),
        ],
        vec![
            SchedulerScalar::new("moving_frame_active", SchedulerScalarValue::Bool(true))?,
            SchedulerScalar::new("restart_slot", SchedulerScalarValue::I64(-1))?,
            SchedulerScalar::new("domainlocx", SchedulerScalarValue::F64(45_000.0))?,
            SchedulerScalar::new("history_record", SchedulerScalarValue::U64(4))?,
        ],
        vec![ComponentBlob::new(
            "arwen_wsm6",
            "arwen-wsm6-persistence/v1",
            vec![0, 1, 2, 3, 0xff],
        )],
    ))
}

fn run(arguments: Vec<OsString>) -> cm1_state::Result<()> {
    let [checkpoint, history] = arguments.as_slice() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "usage: forecast_fixture NEW_CHECKPOINT.cm1fcp NEW_HISTORY.cm1fhs",
        )
        .into());
    };
    let checkpoint = PathBuf::from(checkpoint);
    let history = PathBuf::from(history);
    let contract = contract()?;
    let frame = frame()?;
    let checkpoint_write = write_forecast_checkpoint_new(&contract, &frame, &checkpoint)?;
    let checkpoint_read = read_forecast_checkpoint(&contract, &checkpoint)?;
    if &checkpoint_write != checkpoint_read.receipt() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "forecast checkpoint write/read receipts differ",
        )
        .into());
    }
    let history_write = write_forecast_history_new(&contract, &frame, &history)?;
    let history_read = read_forecast_history(&contract, &history)?;
    if &history_write != history_read.receipt() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "forecast history write/read receipts differ",
        )
        .into());
    }
    println!("{}", checkpoint_write.canonical_json());
    println!("{}", history_write.canonical_json());
    Ok(())
}

fn main() -> ExitCode {
    match run(env::args_os().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("forecast_fixture: {error}");
            ExitCode::from(2)
        }
    }
}
