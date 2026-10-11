//! Strict, receipt-anchored GM1 forecast ingestion.

use cm1_state::forecast::{
    ForecastArray, ForecastArraySpec, ForecastContract, RestoredForecast,
    read_forecast_checkpoint_embedded, read_forecast_history_embedded,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const WSM6_SPECIES: [&str; 6] = ["qv", "qc", "qr", "qi", "qs", "qg"];
pub const CONDENSATE_FORMULA: &str = "q3d[qc]+q3d[qr]+q3d[qi]+q3d[qs]+q3d[qg]";
pub const CLOUD_CONDENSATE_FORMULA: &str = "q3d[qc]+q3d[qi]";
pub const WSM6_ORDER_AUTHORITY: &str = "CM1 r21.1 ptype=5 six-species order qv,qc,qr,qi,qs,qg";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QualityProfile {
    #[default]
    Live,
    Cinema,
}

impl QualityProfile {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Live => "Scientific Live",
            Self::Cinema => "Scientific Cinema",
        }
    }

    pub const fn ray_steps(self) -> f32 {
        match self {
            Self::Live => 96.0,
            Self::Cinema => 256.0,
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "live" => Ok(Self::Live),
            "cinema" => Ok(Self::Cinema),
            _ => Err(format!(
                "invalid --quality {value:?}; expected live or cinema"
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cli {
    pub input: Option<PathBuf>,
    pub watch_dir: Option<PathBuf>,
    pub production_pack: Option<PathBuf>,
    pub watch_production_series: Option<PathBuf>,
    pub verify_production_pack: Option<PathBuf>,
    pub contract_sha256: Option<String>,
    pub screenshot: Option<PathBuf>,
    pub frames_dir: Option<PathBuf>,
    pub she_comparison: bool,
    pub she_glossy_comparison: bool,
    pub admitted_she_terrain: bool,
    /// Internal same-camera A/B oracle for the previously accepted v13
    /// lighting response. It is refused unless the admitted terrain scene is
    /// selected and is never a product render mode.
    pub lighting_reference_diagnostic: bool,
    /// Debug-build-only refused-frame readback for same-camera lighting
    /// diagnosis. The resulting file is not scientifically admitted.
    pub lighting_readback_diagnostic: bool,
    pub presentation: bool,
    pub drive: bool,
    pub quality: QualityProfile,
    pub science_dir: Option<PathBuf>,
    pub science_layer: Option<String>,
    pub profile_receipt: Option<PathBuf>,
    pub soak_seconds: u32,
}

impl Cli {
    pub fn from_env() -> Result<Self, String> {
        Self::parse(std::env::args().skip(1))
    }

    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut args = args.into_iter();
        let mut input = None;
        let mut watch_dir = None;
        let mut production_pack = None;
        let mut watch_production_series = None;
        let mut verify_production_pack = None;
        let mut contract_sha256 = None;
        let mut screenshot = None;
        let mut frames_dir = None;
        let mut she_comparison = false;
        let mut she_glossy_comparison = false;
        let mut admitted_she_terrain = false;
        let mut lighting_reference_diagnostic = false;
        let mut lighting_readback_diagnostic = false;
        let mut presentation = false;
        let mut drive = false;
        let mut quality = QualityProfile::Live;
        let mut quality_was_set = false;
        let mut science_dir = None;
        let mut science_layer = None;
        let mut profile_receipt = None;
        let mut soak_seconds = 60_u32;
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--input" => input = Some(PathBuf::from(required_value(&mut args, "--input")?)),
                "--watch-dir" => {
                    watch_dir = Some(PathBuf::from(required_value(&mut args, "--watch-dir")?))
                }
                "--production-pack" => {
                    production_pack = Some(PathBuf::from(required_value(
                        &mut args,
                        "--production-pack",
                    )?))
                }
                "--watch-production-series" => {
                    watch_production_series = Some(PathBuf::from(required_value(
                        &mut args,
                        "--watch-production-series",
                    )?))
                }
                "--verify-production-pack" => {
                    verify_production_pack = Some(PathBuf::from(required_value(
                        &mut args,
                        "--verify-production-pack",
                    )?))
                }
                "--contract-sha256" => {
                    contract_sha256 = Some(required_value(&mut args, "--contract-sha256")?)
                }
                "--screenshot" => {
                    screenshot = Some(PathBuf::from(required_value(&mut args, "--screenshot")?))
                }
                "--frames-dir" => {
                    frames_dir = Some(PathBuf::from(required_value(&mut args, "--frames-dir")?))
                }
                "--she-comparison" => she_comparison = true,
                "--she-glossy-comparison" => she_glossy_comparison = true,
                "--admitted-she-terrain" => admitted_she_terrain = true,
                "--lighting-reference-diagnostic" => lighting_reference_diagnostic = true,
                "--lighting-readback-diagnostic" => lighting_readback_diagnostic = true,
                "--presentation" => presentation = true,
                "--experimental-drive" => drive = true,
                "--quality" => {
                    quality_was_set = true;
                    quality = QualityProfile::parse(&required_value(&mut args, "--quality")?)?
                }
                "--science-dir" => {
                    science_dir = Some(PathBuf::from(required_value(&mut args, "--science-dir")?))
                }
                "--science-layer" => {
                    science_layer = Some(required_value(&mut args, "--science-layer")?)
                }
                "--profile-receipt" => {
                    profile_receipt = Some(PathBuf::from(required_value(
                        &mut args,
                        "--profile-receipt",
                    )?))
                }
                "--soak-seconds" => {
                    let value = required_value(&mut args, "--soak-seconds")?;
                    soak_seconds = value
                        .parse::<u32>()
                        .map_err(|_| format!("invalid --soak-seconds {value:?}"))?;
                }
                "--help" | "-h" => return Err(usage().to_owned()),
                other => return Err(format!("unknown argument {other:?}\n{}", usage())),
            }
        }
        if [
            input.is_some(),
            watch_dir.is_some(),
            production_pack.is_some(),
            watch_production_series.is_some(),
            verify_production_pack.is_some(),
        ]
        .into_iter()
        .filter(|selected| *selected)
        .count()
            != 1
        {
            return Err(format!(
                "supply exactly one of --input, --watch-dir, --production-pack, --watch-production-series, or --verify-production-pack\n{}",
                usage()
            ));
        }
        if [she_comparison, she_glossy_comparison, admitted_she_terrain]
            .into_iter()
            .filter(|selected| *selected)
            .count()
            > 1
        {
            return Err("choose at most one SHE render mode".to_owned());
        }
        if input.is_some() && contract_sha256.is_none() {
            return Err("--input requires an independent --contract-sha256".to_owned());
        }
        if production_pack.is_some() && contract_sha256.is_some() {
            return Err(
                "--production-pack carries its own strict history-contract identity; do not supply --contract-sha256"
                    .to_owned(),
            );
        }
        if watch_production_series.is_some() && contract_sha256.is_some() {
            return Err(
                "--watch-production-series admits every exact frame through its own strict history/operator authority; do not supply --contract-sha256"
                    .to_owned(),
            );
        }
        if verify_production_pack.is_some() && contract_sha256.is_some() {
            return Err(
                "--verify-production-pack carries its own strict history-contract identity; do not supply --contract-sha256"
                    .to_owned(),
            );
        }
        if verify_production_pack.is_some()
            && (screenshot.is_some()
                || frames_dir.is_some()
                || she_comparison
                || she_glossy_comparison
                || admitted_she_terrain
                || lighting_reference_diagnostic
                || lighting_readback_diagnostic
                || presentation
                || drive
                || quality_was_set
                || science_dir.is_some()
                || science_layer.is_some()
                || profile_receipt.is_some()
                || soak_seconds != 60)
        {
            return Err(
                "--verify-production-pack is renderer-free and exclusive; do not supply rendering, science-layer, or profiling options"
                    .to_owned(),
            );
        }
        if lighting_reference_diagnostic && !admitted_she_terrain {
            return Err(
                "--lighting-reference-diagnostic requires --admitted-she-terrain".to_owned(),
            );
        }
        if lighting_readback_diagnostic && !cfg!(debug_assertions) {
            return Err("--lighting-readback-diagnostic is disabled in release builds".to_owned());
        }
        if drive && !admitted_she_terrain {
            return Err("--experimental-drive requires --admitted-she-terrain".to_owned());
        }
        if !(10..=3_600).contains(&soak_seconds) {
            return Err("--soak-seconds must be in 10..=3600".to_owned());
        }
        if profile_receipt.is_none() && soak_seconds != 60 {
            return Err("--soak-seconds requires --profile-receipt".to_owned());
        }
        if science_layer.as_deref().is_some_and(|layer| {
            !matches!(
                layer,
                "simsat_top_view"
                    | "simsat_infrared"
                    | "simsat_water_vapor"
                    | "reflectivity_dbz"
                    | "updraft_ms"
                    | "vertical_vorticity_s"
            )
        }) {
            return Err("--science-layer is not a recognized strict layer id".to_owned());
        }
        if let Some(hash) = contract_sha256.as_mut() {
            validate_sha256(hash, "--contract-sha256")?;
            hash.make_ascii_lowercase();
        }
        Ok(Self {
            input,
            watch_dir,
            production_pack,
            watch_production_series,
            verify_production_pack,
            contract_sha256,
            screenshot,
            frames_dir,
            she_comparison,
            she_glossy_comparison,
            admitted_she_terrain,
            lighting_reference_diagnostic,
            lighting_readback_diagnostic,
            presentation,
            drive,
            quality,
            science_dir,
            science_layer,
            profile_receipt,
            soak_seconds,
        })
    }
}

#[derive(Clone, Debug)]
pub struct VolumeField {
    pub id: String,
    pub label: String,
    pub units: String,
    pub meaning: String,
    pub source: String,
    /// C-order `[z, y, x]`.
    pub shape: [usize; 3],
    pub values: Vec<f32>,
    pub values_sha256: String,
    pub finite_min: f32,
    pub finite_max: f32,
}

/// CM1's own diagnostic surface carriers, exactly as the frame published
/// them. The names are CM1 r21.1's: `uh` is `src/writeout.F:727`'s integrated
/// 2-5 km updraft helicity, `zvort_sfc` the lowest model level of
/// `src/writeout.F:4381`'s vertical vorticity, and `sws`/`svs`/`sps`/`shs`
/// the whole-run surface swaths of `src/solve3.F:855-1136`.
///
/// They are admitted by name. A frame from a run whose document selected none
/// of them carries none of these arrays and this list is empty, which is
/// exactly what every frame written before they existed looks like.
pub const TORNADO_SURFACE_CARRIERS: [&str; 6] =
    ["uh", "zvort_sfc", "sws", "svs", "sps", "shs"];

#[derive(Clone, Debug)]
pub struct SurfaceDiagnosticField {
    pub id: String,
    pub label: String,
    pub units: String,
    pub meaning: String,
    /// C-order `[y, x]`.
    pub shape: [usize; 2],
    pub values: Vec<f32>,
    pub values_sha256: String,
    pub finite_min: f32,
    pub finite_max: f32,
}

#[derive(Clone, Debug)]
pub struct Gm1Dataset {
    pub source_path: PathBuf,
    pub artifact_kind: String,
    pub artifact_sha256: String,
    pub payload_sha256: String,
    pub contract_sha256: String,
    /// Embedded-contract identity of the run configuration. Empty when the
    /// artifact's contract predates the key.
    pub config_sha256: String,
    pub profile_id: String,
    pub step: u64,
    pub model_time_s: f64,
    pub dt_s: f64,
    pub fields: Vec<VolumeField>,
    /// Exact surface height above MSL, C-order `[y, x]`, when the artifact
    /// publishes a terrain carrier.
    pub terrain_msl_m: Option<Vec<f32>>,
    /// Exact w-face height above MSL, C-order `[nz + 1, y, x]`, when the
    /// artifact publishes the model's vertical face carrier. These are the
    /// run's own level altitudes; nothing else in the envelope states them.
    pub face_height_m: Option<Vec<f32>>,
    /// CM1's diagnostic surface carriers, when the run's document selected
    /// any. Empty otherwise; never a zero-filled stand-in.
    pub tornado_surface_fields: Vec<SurfaceDiagnosticField>,
}

#[derive(Debug, Deserialize)]
struct ContractDocument {
    #[serde(rename = "array_specs")]
    arrays: Vec<ArrayMetadata>,
    /// Identity of the run configuration that produced this frame. The
    /// envelope carries no grid spacing, so this is the key a sibling
    /// geometry document must match before its domain size may be bound.
    #[serde(default)]
    config_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ArrayMetadata {
    name: String,
    meaning: String,
    units: String,
    stagger: String,
}

pub fn load_cli_dataset(cli: &Cli) -> Result<Gm1Dataset, String> {
    let input = cli
        .input
        .as_deref()
        .ok_or_else(|| "live-watch mode has no initial frame yet".to_owned())?;
    let contract = cli
        .contract_sha256
        .as_deref()
        .ok_or_else(|| "strict input contract pin is absent".to_owned())?;
    load_artifact(input, contract)
}

pub fn load_artifact(input: &Path, expected_contract_sha256: &str) -> Result<Gm1Dataset, String> {
    validate_sha256(expected_contract_sha256, "expected contract SHA-256")?;
    if !input.is_file() {
        return Err(format!("input is not a file: {}", input.display()));
    }
    let extension = input
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| "input has no recognized extension".to_owned())?;
    let (contract, restored, artifact_kind) = match extension.as_str() {
        "cm1fcp" => {
            let (contract, restored) =
                read_forecast_checkpoint_embedded(input, expected_contract_sha256)
                    .map_err(|error| format!("strict CM1FCP admission failed: {error}"))?;
            (contract, restored, "CM1FCP1 checkpoint")
        }
        "cm1fhs" => {
            let (contract, restored) =
                read_forecast_history_embedded(input, expected_contract_sha256)
                    .map_err(|error| format!("strict CM1FHS admission failed: {error}"))?;
            (contract, restored, "CM1FHS1 history")
        }
        _ => {
            return Err(format!(
                "unsupported input extension .{extension}; v0 strictly accepts .cm1fcp and .cm1fhs"
            ));
        }
    };
    if contract.sha256() != expected_contract_sha256.to_ascii_lowercase() {
        return Err("embedded contract identity changed after strict admission".to_owned());
    }
    build_dataset(input, artifact_kind, &contract, &restored)
}

fn build_dataset(
    source_path: &Path,
    artifact_kind: &str,
    contract: &ForecastContract,
    restored: &RestoredForecast,
) -> Result<Gm1Dataset, String> {
    let document: ContractDocument = serde_json::from_str(&contract.canonical_json())
        .map_err(|error| format!("embedded contract JSON was not understood: {error}"))?;
    let mut fields = if find_array(restored, "q3d").is_some() {
        checkpoint_fields(contract, &document, restored)?
    } else {
        scalar_fields(contract, &document, restored)?
    };
    if fields.is_empty() {
        return Err(
            "artifact exposes neither six-species q3d nor a scalar [z,y,x] field".to_owned(),
        );
    }
    fields.sort_by_key(|field| match field.id.as_str() {
        "cloud_condensate" => 0,
        "condensate" => 1,
        _ => 2,
    });
    let shape_zyx = fields[0].shape;
    if let Some(mismatched) = fields.iter().find(|field| field.shape != shape_zyx) {
        return Err(format!(
            "field {} has shape {:?} while the frame's first field has {shape_zyx:?}; \
             one frame cannot carry two grids, and indexing the two with a single \
             stride would place one field's words at another field's positions",
            mismatched.id, mismatched.shape
        ));
    }
    let shape_yx = [shape_zyx[1], shape_zyx[2]];
    let clock = restored.frame().clock();
    let receipt = restored.receipt();
    Ok(Gm1Dataset {
        source_path: source_path.to_path_buf(),
        artifact_kind: artifact_kind.to_owned(),
        artifact_sha256: receipt.artifact_sha256.clone(),
        payload_sha256: receipt.payload_sha256.clone(),
        contract_sha256: receipt.contract_sha256.clone(),
        config_sha256: document.config_sha256.clone(),
        profile_id: contract.profile_id().to_owned(),
        step: clock.step(),
        model_time_s: clock.model_time_s(),
        dt_s: clock.dt_s(),
        terrain_msl_m: surface_carrier(contract, restored, "terrain_msl", shape_yx)?,
        face_height_m: vertical_face_carrier(contract, restored, "zf", shape_zyx)?,
        tornado_surface_fields: tornado_surface_fields(contract, &document, restored, shape_yx)?,
        fields,
    })
}

/// Admit CM1's diagnostic surface carriers from this frame, by name.
///
/// A carrier whose extent is not the admitted field grid is refused rather
/// than indexed: a swath read with the wrong stride would put one column's
/// whole-run maximum at another column's position, and a swath is a damage
/// track, so that is not a display artefact but a wrong answer about where
/// the storm was.
fn tornado_surface_fields(
    contract: &ForecastContract,
    document: &ContractDocument,
    restored: &RestoredForecast,
    shape_yx: [usize; 2],
) -> Result<Vec<SurfaceDiagnosticField>, String> {
    let mut fields = Vec::new();
    for name in TORNADO_SURFACE_CARRIERS {
        let Some(spec) = find_spec(contract, name) else {
            continue;
        };
        let Some(metadata) = find_metadata(document, name) else {
            return Err(format!("array {name} lacks embedded metadata"));
        };
        if spec.shape() != shape_yx {
            return Err(format!(
                "CM1 diagnostic surface carrier {name} has shape {:?} against                  the admitted field grid {shape_yx:?} (ny,nx); reading it with                  the field's strides would report one column's swath at                  another column's position",
                spec.shape()
            ));
        }
        let Some(array) = find_array(restored, name) else {
            return Err(format!("array {name} has a contract but no payload"));
        };
        let values = array.values().to_vec();
        if values.len() != shape_yx[0] * shape_yx[1] {
            return Err(format!(
                "CM1 diagnostic surface carrier {name} carries {} words                  against the admitted {shape_yx:?} grid",
                values.len()
            ));
        }
        let mut finite_min = f32::INFINITY;
        let mut finite_max = f32::NEG_INFINITY;
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("{name} word {index} is non-finite"));
            }
            finite_min = finite_min.min(*value);
            finite_max = finite_max.max(*value);
        }
        fields.push(SurfaceDiagnosticField {
            id: name.to_owned(),
            label: metadata.meaning.clone(),
            units: metadata.units.clone(),
            meaning: metadata.meaning.clone(),
            shape: shape_yx,
            values_sha256: f32_sha256(&values),
            values,
            finite_min,
            finite_max,
        });
    }
    Ok(fields)
}

/// Exact `[y, x]` surface carrier, when the artifact publishes one. A carrier
/// whose shape disagrees with the admitted field grid is refused rather than
/// indexed: reading it with the wrong stride would attach one column's surface
/// height to another column's air.
fn surface_carrier(
    contract: &ForecastContract,
    restored: &RestoredForecast,
    name: &str,
    shape_yx: [usize; 2],
) -> Result<Option<Vec<f32>>, String> {
    let (Some(array), Some(spec)) = (find_array(restored, name), find_spec(contract, name)) else {
        return Ok(None);
    };
    if spec.shape() != shape_yx {
        return Err(format!(
            "{name} carrier shape {:?} disagrees with the admitted field grid \
             {shape_yx:?} (ny,nx); indexing it with the field's strides would \
             attach one column's surface height to another column's air",
            spec.shape()
        ));
    }
    let values = array.values();
    if values.len() != shape_yx[0] * shape_yx[1] || values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{name} carrier has {} words or a non-finite word against the \
             admitted {shape_yx:?} grid; a non-finite surface height would \
             place the whole column at an undefined altitude",
            values.len()
        ));
    }
    Ok(Some(values.to_vec()))
}

/// Exact `[nz + 1, y, x]` w-face height carrier, when the artifact publishes
/// one. These faces are the run's own level altitudes; the envelope states the
/// altitudes nowhere else, so a shape disagreement is refused rather than
/// reinterpreted.
fn vertical_face_carrier(
    contract: &ForecastContract,
    restored: &RestoredForecast,
    name: &str,
    shape_zyx: [usize; 3],
) -> Result<Option<Vec<f32>>, String> {
    let (Some(array), Some(spec)) = (find_array(restored, name), find_spec(contract, name)) else {
        return Ok(None);
    };
    let expected = [shape_zyx[0] + 1, shape_zyx[1], shape_zyx[2]];
    if spec.shape() != expected {
        return Err(format!(
            "{name} face carrier shape {:?} disagrees with the admitted field \
             grid {shape_zyx:?} plus one w face ({expected:?}); reading it with \
             the wrong stride would report every level at the wrong altitude",
            spec.shape()
        ));
    }
    let values = array.values();
    if values.len() != expected.iter().product::<usize>()
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(format!(
            "{name} face carrier has {} words or a non-finite word against \
             {expected:?}; a non-finite face height would place a level at an \
             undefined altitude",
            values.len()
        ));
    }
    Ok(Some(values.to_vec()))
}

fn checkpoint_fields(
    contract: &ForecastContract,
    document: &ContractDocument,
    restored: &RestoredForecast,
) -> Result<Vec<VolumeField>, String> {
    let q3d = find_array(restored, "q3d").ok_or_else(|| "q3d is absent".to_owned())?;
    let spec = find_spec(contract, "q3d").ok_or_else(|| "q3d lacks a contract".to_owned())?;
    if spec.shape().len() != 4 || spec.shape()[0] != WSM6_SPECIES.len() {
        return Err(format!(
            "q3d shape {:?} is not WSM6 [6,z,y,x]",
            spec.shape()
        ));
    }
    let shape = [spec.shape()[1], spec.shape()[2], spec.shape()[3]];
    let words = shape.iter().product::<usize>();
    if q3d.values().len() != words * WSM6_SPECIES.len() {
        return Err("q3d payload length disagrees with its contract".to_owned());
    }
    let cloud_condensate = derive_wsm6_species_sum(q3d.values(), words, &[1, 3])?;
    let condensate = derive_wsm6_condensate(q3d.values(), words)?;
    let mut fields = vec![
        make_field(
            "cloud_condensate",
            "Resolved cloud condensate",
            "kg kg-1",
            "WSM6 non-precipitating cloud water plus cloud ice",
            &format!("{CLOUD_CONDENSATE_FORMULA}; {WSM6_ORDER_AUTHORITY}"),
            shape,
            cloud_condensate,
        )?,
        make_field(
            "condensate",
            "Total resolved condensate",
            "kg kg-1",
            "WSM6 resolved non-vapor water mixing ratio",
            &format!("{CONDENSATE_FORMULA}; {WSM6_ORDER_AUTHORITY}"),
            shape,
            condensate,
        )?,
    ];
    fields.push(make_field(
        "qv",
        "Water-vapor mixing ratio",
        "kg kg-1",
        "WSM6 water-vapor mixing ratio",
        "exact q3d species 0 (qv)",
        shape,
        q3d.values()[..words].to_vec(),
    )?);
    let component_metadata = [
        (1usize, "qc", "Cloud-water mixing ratio"),
        (2usize, "qr", "Rainwater mixing ratio"),
        (3usize, "qi", "Cloud-ice mixing ratio"),
        (4usize, "qs", "Snow mixing ratio"),
        (5usize, "qg", "Graupel mixing ratio"),
    ];
    for (species, name, meaning) in component_metadata {
        fields.push(make_field(
            name,
            meaning,
            "kg kg-1",
            meaning,
            &format!("exact q3d species {species} ({name}); {WSM6_ORDER_AUTHORITY}"),
            shape,
            q3d.values()[species * words..(species + 1) * words].to_vec(),
        )?);
    }
    // ``dbz`` is CM1's simulated radar reflectivity carrier
    // (README.namelist:1366-1369 output_dbz; writeout.F:4089-4107). A frame
    // that publishes it is admitted here like any other exact scalar carrier,
    // so the viewer opens it from the history itself rather than needing a
    // separately produced layer pack.
    for name in ["th3d", "ppi", "dbz"] {
        let (Some(array), Some(array_spec), Some(metadata)) = (
            find_array(restored, name),
            find_spec(contract, name),
            find_metadata(document, name),
        ) else {
            continue;
        };
        if array_spec.shape() == shape {
            fields.push(make_field(
                name,
                &metadata.meaning,
                &metadata.units,
                &metadata.meaning,
                &format!("exact {name} forecast carrier"),
                shape,
                array.values().to_vec(),
            )?);
        }
    }
    Ok(fields)
}

fn scalar_fields(
    contract: &ForecastContract,
    document: &ContractDocument,
    restored: &RestoredForecast,
) -> Result<Vec<VolumeField>, String> {
    let mut fields = Vec::new();
    for array in restored.frame().arrays() {
        let Some(spec) = find_spec(contract, array.name()) else {
            return Err(format!("array {} lacks a contract", array.name()));
        };
        let Some(metadata) = find_metadata(document, array.name()) else {
            return Err(format!("array {} lacks embedded metadata", array.name()));
        };
        if spec.shape().len() != 3 || metadata.stagger != "scalar" {
            continue;
        }
        let shape = [spec.shape()[0], spec.shape()[1], spec.shape()[2]];
        fields.push(make_field(
            array.name(),
            &metadata.meaning,
            &metadata.units,
            &metadata.meaning,
            &format!("exact {} forecast carrier", array.name()),
            shape,
            array.values().to_vec(),
        )?);
    }
    Ok(fields)
}

fn make_field(
    id: &str,
    label: &str,
    units: &str,
    meaning: &str,
    source: &str,
    shape: [usize; 3],
    values: Vec<f32>,
) -> Result<VolumeField, String> {
    if values.len() != shape.iter().product::<usize>() {
        return Err(format!(
            "{id} payload length disagrees with shape {shape:?}"
        ));
    }
    let mut finite_min = f32::INFINITY;
    let mut finite_max = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{id} word {index} is non-finite"));
        }
        finite_min = finite_min.min(*value);
        finite_max = finite_max.max(*value);
    }
    Ok(VolumeField {
        id: id.to_owned(),
        label: label.to_owned(),
        units: units.to_owned(),
        meaning: meaning.to_owned(),
        source: source.to_owned(),
        shape,
        values_sha256: f32_sha256(&values),
        finite_min,
        finite_max,
        values,
    })
}

pub fn derive_wsm6_condensate(q3d: &[f32], words_per_species: usize) -> Result<Vec<f32>, String> {
    derive_wsm6_species_sum(q3d, words_per_species, &[1, 2, 3, 4, 5])
}

pub fn derive_wsm6_species_sum(
    q3d: &[f32],
    words_per_species: usize,
    species_indices: &[usize],
) -> Result<Vec<f32>, String> {
    if words_per_species == 0 || q3d.len() != WSM6_SPECIES.len() * words_per_species {
        return Err("q3d must contain six equally sized WSM6 species".to_owned());
    }
    if species_indices.is_empty()
        || species_indices
            .iter()
            .any(|&species| species == 0 || species >= WSM6_SPECIES.len())
    {
        return Err("WSM6 hydrometeor sum requires species indices 1..5".to_owned());
    }
    let mut condensate = vec![0.0_f32; words_per_species];
    for word in 0..words_per_species {
        let mut sum = 0.0_f32;
        for &species in species_indices {
            let value = q3d[species * words_per_species + word];
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "q3d {} word {word} is not finite nonnegative: {value:?}",
                    WSM6_SPECIES[species]
                ));
            }
            sum += value;
        }
        if !sum.is_finite() {
            return Err(format!("condensate sum at word {word} is non-finite"));
        }
        condensate[word] = sum;
    }
    Ok(condensate)
}

pub fn f32_sha256(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn find_array<'a>(restored: &'a RestoredForecast, name: &str) -> Option<&'a ForecastArray> {
    restored
        .frame()
        .arrays()
        .iter()
        .find(|array| array.name() == name)
}

fn find_spec<'a>(contract: &'a ForecastContract, name: &str) -> Option<&'a ForecastArraySpec> {
    contract
        .arrays()
        .iter()
        .find(|specification| specification.name() == name)
}

fn find_metadata<'a>(document: &'a ContractDocument, name: &str) -> Option<&'a ArrayMetadata> {
    document.arrays.iter().find(|array| array.name == name)
}

fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} must be exactly 64 hexadecimal digits"));
    }
    Ok(())
}

fn required_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn usage() -> &'static str {
    "usage: gm1-view --verify-production-pack <exact-v2-step-dir> | gm1-view (--input <real.cm1fcp|real.cm1fhs> --contract-sha256 <64hex> | --watch-dir <pairs> | --production-pack <exact-v2-step-dir> | --watch-production-series <exact-v2-series-dir>) [--screenshot <out.png>] [--frames-dir <dir>] [--presentation] [--quality live|cinema] [--science-dir <strict-layer-packs>] [--science-layer <simsat_top_view|simsat_infrared|simsat_water_vapor|reflectivity_dbz|updraft_ms|vertical_vorticity_s>] [--profile-receipt <json> [--soak-seconds <10..3600>]] [--she-comparison | --she-glossy-comparison | --admitted-she-terrain] [--lighting-reference-diagnostic] [--lighting-readback-diagnostic] [--experimental-drive]"
}

#[cfg(test)]
mod tests {
    use super::{Cli, QualityProfile, derive_wsm6_condensate, derive_wsm6_species_sum, f32_sha256};
    use std::path::PathBuf;

    #[test]
    fn cli_requires_independent_contract_identity() {
        let parsed = Cli::parse([
            "--input".to_owned(),
            "real.cm1fcp".to_owned(),
            "--contract-sha256".to_owned(),
            "A".repeat(64),
        ])
        .unwrap();
        assert_eq!(parsed.contract_sha256, Some("a".repeat(64)));
        assert!(parsed.production_pack.is_none());
        assert!(parsed.watch_production_series.is_none());
        assert!(parsed.verify_production_pack.is_none());
        assert!(!parsed.she_comparison);
        assert!(!parsed.she_glossy_comparison);
        assert!(!parsed.admitted_she_terrain);
        assert!(!parsed.lighting_reference_diagnostic);
        assert!(!parsed.lighting_readback_diagnostic);
        assert!(!parsed.presentation);
        assert!(!parsed.drive);
        assert_eq!(parsed.quality, QualityProfile::Live);
        assert!(parsed.science_dir.is_none());
        assert!(parsed.science_layer.is_none());
        assert!(parsed.profile_receipt.is_none());
        assert_eq!(parsed.soak_seconds, 60);
        assert!(
            Cli::parse([
                "--input".to_owned(),
                "real.cm1fcp".to_owned(),
                "--contract-sha256".to_owned(),
                "not-a-hash".to_owned(),
            ])
            .unwrap_err()
            .contains("64 hexadecimal")
        );
        assert!(Cli::parse(["--watch-dir".to_owned(), "frames".to_owned()]).is_ok());
        let production =
            Cli::parse(["--production-pack".to_owned(), "step-00000001".to_owned()]).unwrap();
        assert_eq!(
            production.production_pack,
            Some(PathBuf::from("step-00000001"))
        );
        assert!(production.input.is_none());
        assert!(production.watch_dir.is_none());
        assert!(production.watch_production_series.is_none());
        assert!(production.verify_production_pack.is_none());
        let production_series =
            Cli::parse(["--watch-production-series".to_owned(), "series".to_owned()]).unwrap();
        assert_eq!(
            production_series.watch_production_series,
            Some(PathBuf::from("series"))
        );
        assert!(production_series.production_pack.is_none());
        assert!(production_series.input.is_none());
        assert!(production_series.watch_dir.is_none());
        assert!(production_series.verify_production_pack.is_none());
        assert!(
            Cli::parse([
                "--watch-production-series".to_owned(),
                "series".to_owned(),
                "--contract-sha256".to_owned(),
                "a".repeat(64),
            ])
            .unwrap_err()
            .contains("own strict history/operator authority")
        );
        assert!(
            Cli::parse([
                "--watch-production-series".to_owned(),
                "series".to_owned(),
                "--production-pack".to_owned(),
                "step-00000001".to_owned(),
            ])
            .unwrap_err()
            .contains("exactly one")
        );
        assert!(
            Cli::parse([
                "--production-pack".to_owned(),
                "step-00000001".to_owned(),
                "--contract-sha256".to_owned(),
                "a".repeat(64),
            ])
            .unwrap_err()
            .contains("carries its own strict history-contract identity")
        );
        assert!(
            Cli::parse([
                "--production-pack".to_owned(),
                "step-00000001".to_owned(),
                "--watch-dir".to_owned(),
                "frames".to_owned(),
            ])
            .unwrap_err()
            .contains("exactly one")
        );
        assert!(
            Cli::parse([
                "--watch-dir".to_owned(),
                "frames".to_owned(),
                "--she-comparison".to_owned(),
            ])
            .unwrap()
            .she_comparison
        );
        assert!(
            Cli::parse([
                "--watch-dir".to_owned(),
                "frames".to_owned(),
                "--she-glossy-comparison".to_owned(),
            ])
            .unwrap()
            .she_glossy_comparison
        );
        assert!(
            Cli::parse([
                "--watch-dir".to_owned(),
                "frames".to_owned(),
                "--admitted-she-terrain".to_owned(),
            ])
            .unwrap()
            .admitted_she_terrain
        );
        assert!(
            Cli::parse([
                "--watch-dir".to_owned(),
                "frames".to_owned(),
                "--admitted-she-terrain".to_owned(),
                "--lighting-reference-diagnostic".to_owned(),
            ])
            .unwrap()
            .lighting_reference_diagnostic
        );
        assert!(
            Cli::parse([
                "--watch-dir".to_owned(),
                "frames".to_owned(),
                "--lighting-reference-diagnostic".to_owned(),
            ])
            .unwrap_err()
            .contains("requires --admitted-she-terrain")
        );
        // The readback diagnostic is a debug-only instrument; a release build
        // refuses it by name. Asserting only the debug outcome made this test
        // fail under `cargo test --release`, which reported a parser defect
        // that does not exist and hid real release regressions behind it.
        let readback = Cli::parse([
            "--watch-dir".to_owned(),
            "frames".to_owned(),
            "--admitted-she-terrain".to_owned(),
            "--lighting-readback-diagnostic".to_owned(),
        ]);
        if cfg!(debug_assertions) {
            assert!(readback.unwrap().lighting_readback_diagnostic);
        } else {
            assert!(
                readback
                    .unwrap_err()
                    .contains("disabled in release builds")
            );
        }
        assert!(
            Cli::parse([
                "--watch-dir".to_owned(),
                "frames".to_owned(),
                "--presentation".to_owned(),
            ])
            .unwrap()
            .presentation
        );
        assert!(
            Cli::parse([
                "--watch-dir".to_owned(),
                "frames".to_owned(),
                "--experimental-drive".to_owned(),
                "--admitted-she-terrain".to_owned(),
            ])
            .unwrap()
            .drive
        );
        assert!(
            Cli::parse([
                "--watch-dir".to_owned(),
                "frames".to_owned(),
                "--drive".to_owned(),
            ])
            .unwrap_err()
            .contains("unknown argument")
        );
        let profiled = Cli::parse([
            "--watch-dir".to_owned(),
            "frames".to_owned(),
            "--quality".to_owned(),
            "cinema".to_owned(),
            "--profile-receipt".to_owned(),
            "profile.json".to_owned(),
            "--soak-seconds".to_owned(),
            "75".to_owned(),
        ])
        .unwrap();
        assert_eq!(profiled.quality, QualityProfile::Cinema);
        assert_eq!(profiled.soak_seconds, 75);
        assert!(
            Cli::parse([
                "--watch-dir".to_owned(),
                "frames".to_owned(),
                "--admitted-she-terrain".to_owned(),
                "--she-comparison".to_owned(),
            ])
            .unwrap_err()
            .contains("at most one")
        );
    }

    #[test]
    fn production_pack_verification_is_renderer_free_and_exclusive() {
        let parsed = Cli::parse([
            "--verify-production-pack".to_owned(),
            "step-00000001".to_owned(),
        ])
        .unwrap();
        assert_eq!(
            parsed.verify_production_pack,
            Some(PathBuf::from("step-00000001"))
        );
        assert!(parsed.input.is_none());
        assert!(parsed.watch_dir.is_none());
        assert!(parsed.production_pack.is_none());
        assert!(parsed.watch_production_series.is_none());
        for extra in [
            vec!["--screenshot", "out.png"],
            vec!["--frames-dir", "frames"],
            vec!["--presentation"],
            vec!["--quality", "live"],
            vec!["--science-dir", "science"],
            vec!["--profile-receipt", "profile.json"],
        ] {
            let mut arguments = vec![
                "--verify-production-pack".to_owned(),
                "step-00000001".to_owned(),
            ];
            arguments.extend(extra.into_iter().map(str::to_owned));
            assert!(
                Cli::parse(arguments)
                    .unwrap_err()
                    .contains("renderer-free and exclusive")
            );
        }
        assert!(
            Cli::parse([
                "--verify-production-pack".to_owned(),
                "step-00000001".to_owned(),
                "--contract-sha256".to_owned(),
                "a".repeat(64),
            ])
            .unwrap_err()
            .contains("carries its own strict history-contract identity")
        );
        assert!(
            Cli::parse([
                "--verify-production-pack".to_owned(),
                "step-00000001".to_owned(),
                "--production-pack".to_owned(),
                "step-00000001".to_owned(),
            ])
            .unwrap_err()
            .contains("exactly one")
        );
    }

    #[test]
    fn condensate_is_source_ordered_real_species_sum() {
        let q3d = [
            9.0, 8.0, // qv excluded
            1.0, 2.0, // qc
            3.0, 4.0, // qr
            5.0, 6.0, // qi
            7.0, 8.0, // qs
            9.0, 10.0, // qg
        ];
        assert_eq!(derive_wsm6_condensate(&q3d, 2).unwrap(), [25.0, 30.0]);
        assert_eq!(
            derive_wsm6_species_sum(&q3d, 2, &[1, 3]).unwrap(),
            [6.0, 8.0]
        );
        assert!(derive_wsm6_species_sum(&q3d, 2, &[0]).is_err());
    }

    #[test]
    fn condensate_refuses_negative_or_nonfinite_species() {
        let mut q3d = vec![0.0_f32; 12];
        q3d[3] = -f32::EPSILON;
        assert!(derive_wsm6_condensate(&q3d, 2).is_err());
        q3d[3] = f32::NAN;
        assert!(derive_wsm6_condensate(&q3d, 2).is_err());
    }

    #[test]
    fn f32_hash_is_stable_raw_little_endian_bits() {
        let hash = f32_sha256(&[0.0, -0.0, 1.0]);
        assert_eq!(hash.len(), 64);
        assert_eq!(hash, f32_sha256(&[0.0, -0.0, 1.0]));
        assert_ne!(hash, f32_sha256(&[0.0, 0.0, 1.0]));
    }
}
