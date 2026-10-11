use std::{env, process::ExitCode, time::Instant};

use gm1_view::arwen_wrfout::{ArwenWrfoutAdapter, ArwenWrfoutSource};
use serde_json::json;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gm1-wrfout-probe refused: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or_else(|| {
        "usage: gm1_wrfout_probe <wrfout-path> <record> <expected-sha256>".to_owned()
    })?;
    let record = args
        .next()
        .ok_or_else(|| "missing record".to_owned())?
        .parse::<usize>()
        .map_err(|error| format!("invalid record: {error}"))?;
    let expected_sha256 = args
        .next()
        .ok_or_else(|| "missing expected SHA-256".to_owned())?;
    if args.next().is_some() {
        return Err("unexpected extra argument".to_owned());
    }

    let start = Instant::now();
    let source = ArwenWrfoutSource::open_expected(&path, record, &expected_sha256)?;
    let source_open_ms = start.elapsed().as_millis();
    let admit_start = Instant::now();
    let admission = ArwenWrfoutAdapter.admit(&source)?;
    let admission_ms = admit_start.elapsed().as_millis();
    let available = admission
        .frame
        .fields
        .iter()
        .filter(|field| field.admission == gm1_view::geofield::GeoFieldAdmission::Available)
        .count();
    let missing = admission
        .frame
        .fields
        .iter()
        .filter(|field| field.admission == gm1_view::geofield::GeoFieldAdmission::Missing)
        .count();
    let refused = admission
        .frame
        .fields
        .iter()
        .filter(|field| field.admission == gm1_view::geofield::GeoFieldAdmission::Refused)
        .count();
    let receipt = json!({
        "schema": admission.schema,
        "status": admission.status,
        "timings_ms": {
            "source_open_and_full_file_sha256": source_open_ms,
            "field_admission_and_exact_f32_hashes": admission_ms,
            "total": start.elapsed().as_millis(),
        },
        "field_counts": {
            "available": available,
            "missing": missing,
            "refused": refused,
        },
        "admission": admission,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&receipt)
            .map_err(|error| format!("serialize receipt: {error}"))?
    );
    Ok(())
}
