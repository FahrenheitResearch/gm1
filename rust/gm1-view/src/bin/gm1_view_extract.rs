//! Independent raw-layout audit for WSM6 q3d forecast carriers.

use cm1_state::forecast::{
    RestoredForecast, read_forecast_checkpoint_embedded, read_forecast_history_embedded,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{env, path::Path};

const SPECIES: [&str; 6] = ["qv", "qc", "qr", "qi", "qs", "qg"];

#[derive(Serialize)]
struct SpeciesStats {
    species: &'static str,
    contiguous_offset_words: usize,
    words: usize,
    sha256_le_f32: String,
    minimum: f32,
    minimum_f32_bits: String,
    q01: f32,
    q01_f32_bits: String,
    q50: f32,
    q50_f32_bits: String,
    q99: f32,
    q99_f32_bits: String,
    maximum: f32,
    maximum_f32_bits: String,
    nonzero_words: usize,
    ge_1e4_words: usize,
    ge_1e3_words: usize,
}

#[derive(Serialize)]
struct StrideNegativeControl {
    lane: usize,
    sha256_le_f32: String,
    minimum: f32,
    maximum: f32,
    ge_1e3_words: usize,
}

#[derive(Serialize)]
struct Audit {
    schema: &'static str,
    input: String,
    artifact_kind: &'static str,
    artifact_sha256: String,
    contract_sha256: String,
    q3d_shape: Vec<usize>,
    q3d_words: usize,
    q3d_sha256_le_f32: String,
    admitted_layout: &'static str,
    admitted_species_order: [&'static str; 6],
    species: Vec<SpeciesStats>,
    derived_cloud_condensate_qc_plus_qi: SpeciesStats,
    derived_total_hydrometeor_qc_qr_qi_qs_qg: SpeciesStats,
    interleaved_stride6_negative_control: Vec<StrideNegativeControl>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("gm1-view-extract: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let input = args.next().ok_or_else(|| {
        "usage: gm1-view-extract INPUT.cm1fcp|INPUT.cm1fhs CONTRACT_SHA256 [OUTPUT.json]".to_owned()
    })?;
    let contract_sha256 = args
        .next()
        .ok_or_else(|| "missing CONTRACT_SHA256".to_owned())?;
    let output = args.next();
    if args.next().is_some() {
        return Err("unexpected extra arguments".to_owned());
    }
    if contract_sha256.len() != 64 || !contract_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("CONTRACT_SHA256 must be 64 hexadecimal digits".to_owned());
    }
    let path = Path::new(&input);
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "input extension is absent".to_owned())?;
    let (contract, restored, artifact_kind) = match extension.to_ascii_lowercase().as_str() {
        "cm1fcp" => {
            let (contract, restored) = read_forecast_checkpoint_embedded(path, &contract_sha256)
                .map_err(|error| error.to_string())?;
            (contract, restored, "CM1FCP1")
        }
        "cm1fhs" => {
            let (contract, restored) = read_forecast_history_embedded(path, &contract_sha256)
                .map_err(|error| error.to_string())?;
            (contract, restored, "CM1FHS1")
        }
        other => return Err(format!("unsupported extension {other:?}")),
    };
    let q3d = array(&restored, "q3d")?;
    let specification = contract
        .arrays()
        .iter()
        .find(|specification| specification.name() == "q3d")
        .ok_or_else(|| "q3d contract specification is absent".to_owned())?;
    let shape = specification.shape();
    if shape.len() != 4 || shape[0] != SPECIES.len() {
        return Err(format!("q3d shape {shape:?} is not [6,z,y,x]"));
    }
    let words = shape[1..].iter().product::<usize>();
    if q3d.len() != SPECIES.len() * words {
        return Err("q3d word length disagrees with [6,z,y,x] contract".to_owned());
    }

    let mut species = Vec::with_capacity(SPECIES.len());
    for (index, name) in SPECIES.iter().copied().enumerate() {
        species.push(stats(
            name,
            index * words,
            &q3d[index * words..(index + 1) * words],
        )?);
    }
    let mut negative = Vec::with_capacity(SPECIES.len());
    for lane in 0..SPECIES.len() {
        let values = q3d.iter().copied().skip(lane).step_by(SPECIES.len());
        let mut digest = Sha256::new();
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        let mut ge_1e3_words = 0usize;
        for value in values {
            if !value.is_finite() {
                return Err(format!(
                    "interleaved lane {lane} contains a non-finite word"
                ));
            }
            digest.update(value.to_bits().to_le_bytes());
            minimum = minimum.min(value);
            maximum = maximum.max(value);
            ge_1e3_words += usize::from(value >= 1.0e-3);
        }
        negative.push(StrideNegativeControl {
            lane,
            sha256_le_f32: format!("{:x}", digest.finalize()),
            minimum,
            maximum,
            ge_1e3_words,
        });
    }
    let mut cloud_condensate = Vec::with_capacity(words);
    let mut total_hydrometeor = Vec::with_capacity(words);
    for word in 0..words {
        let qc = q3d[words + word];
        let qr = q3d[2 * words + word];
        let qi = q3d[3 * words + word];
        let qs = q3d[4 * words + word];
        let qg = q3d[5 * words + word];
        cloud_condensate.push(qc + qi);
        total_hydrometeor.push((((qc + qr) + qi) + qs) + qg);
    }
    let audit = Audit {
        schema: "gm1.view-wsm6-q3d-layout-audit/v1",
        input,
        artifact_kind,
        artifact_sha256: restored.receipt().artifact_sha256.clone(),
        contract_sha256: contract.sha256(),
        q3d_shape: shape.to_vec(),
        q3d_words: q3d.len(),
        q3d_sha256_le_f32: hash(q3d.iter().copied()),
        admitted_layout: "C-contiguous [species,z,y,x], x-fast, six contiguous species slabs",
        admitted_species_order: SPECIES,
        species,
        derived_cloud_condensate_qc_plus_qi: stats("qc+qi", 0, &cloud_condensate)?,
        derived_total_hydrometeor_qc_qr_qi_qs_qg: stats("qc+qr+qi+qs+qg", 0, &total_hydrometeor)?,
        interleaved_stride6_negative_control: negative,
    };
    let mut json = serde_json::to_string_pretty(&audit).map_err(|error| error.to_string())?;
    json.push('\n');
    if let Some(output) = output {
        std::fs::write(&output, json.as_bytes())
            .map_err(|error| format!("could not write {output}: {error}"))?;
        eprintln!("gm1-view-extract: wrote {output}");
    } else {
        print!("{json}");
    }
    Ok(())
}

fn array<'a>(restored: &'a RestoredForecast, name: &str) -> Result<&'a [f32], String> {
    restored
        .frame()
        .arrays()
        .iter()
        .find(|array| array.name() == name)
        .map(|array| array.values())
        .ok_or_else(|| format!("{name} is absent"))
}

fn stats(
    species: &'static str,
    contiguous_offset_words: usize,
    values: &[f32],
) -> Result<SpeciesStats, String> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!("{species} word {index} is non-finite: {value:?}"));
    }
    let minimum = values.iter().copied().min_by(f32::total_cmp).unwrap();
    let maximum = values.iter().copied().max_by(f32::total_cmp).unwrap();
    let q01_index = (values.len() - 1) / 100;
    let q50_index = (values.len() - 1) * 50 / 100;
    let q99_index = (values.len() - 1) * 99 / 100;
    let mut partitioned = values.to_vec();
    partitioned.select_nth_unstable_by(q50_index, f32::total_cmp);
    let q50 = partitioned[q50_index];
    partitioned[..q50_index].select_nth_unstable_by(q01_index, f32::total_cmp);
    let q01 = partitioned[q01_index];
    let upper_index = q99_index - q50_index - 1;
    partitioned[q50_index + 1..].select_nth_unstable_by(upper_index, f32::total_cmp);
    let q99 = partitioned[q99_index];
    Ok(SpeciesStats {
        species,
        contiguous_offset_words,
        words: values.len(),
        sha256_le_f32: hash(values.iter().copied()),
        minimum,
        minimum_f32_bits: bits(minimum),
        q01,
        q01_f32_bits: bits(q01),
        q50,
        q50_f32_bits: bits(q50),
        q99,
        q99_f32_bits: bits(q99),
        maximum,
        maximum_f32_bits: bits(maximum),
        nonzero_words: values.iter().filter(|&&value| value != 0.0).count(),
        ge_1e4_words: values.iter().filter(|&&value| value >= 1.0e-4).count(),
        ge_1e3_words: values.iter().filter(|&&value| value >= 1.0e-3).count(),
    })
}

fn hash(values: impl Iterator<Item = f32>) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn bits(value: f32) -> String {
    format!("{:08x}", value.to_bits())
}
