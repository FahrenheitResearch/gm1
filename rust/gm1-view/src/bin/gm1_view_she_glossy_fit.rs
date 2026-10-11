//! Deterministic CPU fit/held-direction receipt for the physical glossy SHE path.

use gm1_view::she_glossy::{
    SIMSAT_BUNDLE_PATCH_SHA256, SIMSAT_RADIANCE_SHA256, SIMSAT_RADIANCE_SOURCE_SHA256,
    SPECULAR_ALPHA_LEVELS, SPECULAR_HELD_GGX_SAMPLES, SPECULAR_HELD_RESOLUTION,
    SPECULAR_TRAIN_GGX_SAMPLES, SPECULAR_TRAIN_RESOLUTION, fit_t300_glossy_she,
    specular_coefficients_bytes,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf, time::Instant};

#[derive(Serialize)]
struct Receipt<'a> {
    schema: &'a str,
    status: &'a str,
    claim_limit: &'a str,
    upstream: serde_json::Value,
    physical_environment: serde_json::Value,
    fit: serde_json::Value,
    held: serde_json::Value,
    output: serde_json::Value,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("gm1-view glossy SHE fit: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let output = match (args.next().as_deref(), args.next(), args.next()) {
        (Some("--output-dir"), Some(path), None) => PathBuf::from(path),
        _ => return Err("usage: gm1_view_she_glossy_fit --output-dir <directory>".to_owned()),
    };
    let started = Instant::now();
    let fitted = fit_t300_glossy_she()?;
    let elapsed = started.elapsed().as_secs_f64();
    let bytes = specular_coefficients_bytes(&fitted.coefficients);
    fs::create_dir_all(&output).map_err(|error| error.to_string())?;
    fs::write(output.join("specular-she-fp32.f32le"), &bytes).map_err(|error| error.to_string())?;
    let receipt = Receipt {
        schema: "gm1.view-she-glossy-physical-fit/v1",
        status: "REFUSED_REDUCED_PREVIEW_FIT_ERROR",
        claim_limit: "Refused production fit: held RMS relative and luminance-relative error exceed admission. Coefficients may only drive an explicitly labeled diagnostic preview. This reduced CPU fit is not source-exact, not upstream-default, and not a production/SOTA-equivalence claim. Direct solar disk is separate; no meteorological mutation, cloud-scattering replacement, mapped terrain/road/vehicle geometry, FP16/packed, or architecture-independent claim.",
        upstream: serde_json::json!({
            "project": gm1_view::she::UPSTREAM_PROJECT,
            "commit": gm1_view::she::UPSTREAM_COMMIT,
            "license": gm1_view::she::UPSTREAM_LICENSE,
            "common_sh_sha256": gm1_view::she::UPSTREAM_COMMON_SH_SHA256,
            "source_exact": false,
            "production_default_required": {
                "collector_resolution": 256,
                "alpha_levels": 4,
                "ggx_samples": 4096,
                "solver": "unregularized FP32 Cholesky",
                "execution": "upstream WGSL-equivalent compute chain",
            },
        }),
        physical_environment: serde_json::json!({
            "kind": "SimSat Hillaire positive log-SH3 clear-sky radiance; direct sun excluded",
            "valid_time_utc": "2024-05-21T21:05:00Z",
            "center_latitude_deg": 41.39464041135_f64,
            "center_longitude_deg": -94.28268293687_f64,
            "radiance_payload_sha256": SIMSAT_RADIANCE_SHA256,
            "atmosphere_source_sha256": SIMSAT_RADIANCE_SOURCE_SHA256,
            "bundle_patch_sha256": SIMSAT_BUNDLE_PATCH_SHA256,
        }),
        fit: serde_json::json!({
            "training_resolution_nv": SPECULAR_TRAIN_RESOLUTION,
            "training_ggx_samples": SPECULAR_TRAIN_GGX_SAMPLES,
            "collector_resolution": SPECULAR_TRAIN_RESOLUTION,
            "solver": "regularized f64 CPU Cholesky (preview only)",
            "alpha_levels": SPECULAR_ALPHA_LEVELS,
            "training_cases": fitted.training_cases,
            "ridge": fitted.ridge,
            "wall_seconds": elapsed,
        }),
        held: serde_json::json!({
            "resolution_nv": SPECULAR_HELD_RESOLUTION,
            "ggx_samples": SPECULAR_HELD_GGX_SAMPLES,
            "compared_rgb_words": fitted.held.compared_rgb_words,
            "max_absolute_radiance": fitted.held.max_absolute_radiance,
            "rms_absolute_radiance": fitted.held.rms_absolute_radiance,
            "max_relative": fitted.held.max_relative,
            "rms_relative": fitted.held.rms_relative,
            "max_luminance_absolute": fitted.held.max_luminance_absolute,
            "rms_luminance_absolute": fitted.held.rms_luminance_absolute,
            "rms_luminance_relative": fitted.held.rms_luminance_relative,
            "reference_rms_radiance": fitted.held.reference_rms_radiance,
            "reference_rms_luminance": fitted.held.reference_rms_luminance,
            "normalized_rms_absolute_radiance": fitted.held.normalized_rms_absolute_radiance,
            "normalized_rms_luminance_absolute": fitted.held.normalized_rms_luminance_absolute,
        }),
        output: serde_json::json!({
            "file": "specular-she-fp32.f32le",
            "layout": "33 consecutive RGB little-endian f32 coefficients in ConstructSHVectorA order",
            "nbytes": bytes.len(),
            "sha256": fitted.coefficients_sha256,
        }),
    };
    let mut receipt_bytes = serde_json::to_vec(&receipt).map_err(|error| error.to_string())?;
    receipt_bytes.push(b'\n');
    fs::write(output.join("receipt.json"), &receipt_bytes).map_err(|error| error.to_string())?;
    println!(
        "REFUSED_REDUCED_PREVIEW_FIT_ERROR cases={} held_words={} max_abs={:.6e} rms_rel={:.6e} coeff={} receipt={:x}",
        fitted.training_cases,
        fitted.held.compared_rgb_words,
        fitted.held.max_absolute_radiance,
        fitted.held.rms_relative,
        fitted.coefficients_sha256,
        Sha256::digest(&receipt_bytes),
    );
    Ok(())
}
