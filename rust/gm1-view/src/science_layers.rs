//! Strict auxiliary science-layer admission and source-backed top views.

use crate::{
    geofield::{GeoFieldAdmission, GeoFieldDescriptor, GeoFieldGeometry, GeoGrid},
    input::Gm1Dataset,
    lighting::{
        PositiveSky, SIMSAT_PROVENANCE, SIMSAT_SOURCE_REVISION, T300_DIRECT_IRRADIANCE_F32_BITS,
        T300_SKY_ROW_SHA256, T300_SUN_ENU_F32_BITS,
    },
    she_admitted_terrain::{
        ADMITTED_MODEL_TOP_M, ADMITTED_TERRAIN_SHA256, admitted_mass_height_km,
    },
    volume::{SIMSAT_GROUND_BOUNCE_ALBEDO, condensate_extinction_per_km, density_scale},
};
use cm1_state::nws_reflectivity::{NWS_DBZ_CLASSES, NWS_DBZ_FLOOR, nws_dbz_rgb};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

pub const SCIENCE_LAYER_PACK_SCHEMA: &str = "gm1.view.science-layer-pack/v1";
/// Number of scalar levels the sealed terrain-composed sigma profile states.
pub const SEALED_SCALAR_LEVEL_COUNT: usize = 75;
pub const SIMSAT_TOP_VIEW_PRODUCT: &str = "simsat.physical-cloud-top-view/v1";
pub const SIMSAT_TOP_VIEW_API_SOURCE_SHA256: &str =
    "feba678fe0489c6cf221b8077220ac4df3c67c5b3caa9fff410a65c994b54934";
pub const SIMSAT_TOP_VIEW_CLOUD_SOURCE_SHA256: &str =
    "0f4a259e31397f88c2d6beac8202e4aa74b6ef694b3869502827bf9070800d76";
pub const SIMSAT_TOP_VIEW_ATMOSPHERE_SOURCE_SHA256: &str =
    "6e9373ae7f8277cd62e9391c44bb1fc9cb8b3d72b92d61a3ecb25056ffd5d814";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ScienceLayerId {
    SimSatTopView,
    SimSatInfrared,
    SimSatWaterVapor,
    ReflectivityDbz,
    UpdraftMs,
    VerticalVorticityS,
    // CM1 r21.1's own diagnostic surface carriers. These do not come from a
    // science pack: they are published in the forecast frame itself by a run
    // whose document selected them, so they are read from the dataset and
    // refused by the pack loader below.
    UpdraftHelicityM2S2,
    SurfaceVerticalVorticityS,
    SwathMaxWindMs,
    SwathMaxVorticityS,
    SwathMinPressurePa,
    SwathMaxHelicityM2S2,
}

impl ScienceLayerId {
    pub const ALL: [Self; 12] = [
        Self::SimSatTopView,
        Self::SimSatInfrared,
        Self::SimSatWaterVapor,
        Self::ReflectivityDbz,
        Self::UpdraftMs,
        Self::VerticalVorticityS,
        Self::UpdraftHelicityM2S2,
        Self::SurfaceVerticalVorticityS,
        Self::SwathMaxWindMs,
        Self::SwathMaxVorticityS,
        Self::SwathMinPressurePa,
        Self::SwathMaxHelicityM2S2,
    ];

    /// The frame carrier this layer reads, for the layers that read one.
    ///
    /// The names are CM1 r21.1's own output names, so a viewer's layer list
    /// and a CM1 output file name the same field.
    pub const fn surface_carrier(self) -> Option<&'static str> {
        match self {
            Self::UpdraftHelicityM2S2 => Some("uh"),
            Self::SurfaceVerticalVorticityS => Some("zvort_sfc"),
            Self::SwathMaxWindMs => Some("sws"),
            Self::SwathMaxVorticityS => Some("svs"),
            Self::SwathMinPressurePa => Some("sps"),
            Self::SwathMaxHelicityM2S2 => Some("shs"),
            _ => None,
        }
    }

    /// The fixed physical colour range this layer is drawn against.
    ///
    /// Fixed, not per-frame: two frames of one run must colour the same swath
    /// value the same way, or a reader watching a playback sees a track grow
    /// where only the frame maximum changed. The numbers are the same ones
    /// `cm1-render` uses, and their sources are in
    /// `cm1core/tornado_diagnostics.py`.
    pub const fn surface_range(self) -> Option<[f32; 2]> {
        match self {
            // Kain et al. 2008, Wea. Forecasting 23, 931-952 -- the reference
            // CM1 cites at src/misclibs.F:1423 -- flags a rotating updraft at
            // 25 m2/s2 and strong rotation at 100.
            Self::UpdraftHelicityM2S2 | Self::SwathMaxHelicityM2S2 => Some([0.0, 300.0]),
            // Signed: the whole point of the lowest model level is which way
            // it turns. Drawn diverging about zero.
            Self::SurfaceVerticalVorticityS => Some([-0.5, 0.5]),
            // WSEC 2006 puts EF0 at 29 m/s and EF5 at 89 m/s.
            Self::SwathMaxWindMs => Some([0.0, 90.0]),
            Self::SwathMaxVorticityS => Some([0.0, 0.5]),
            Self::SwathMinPressurePa => Some([-8000.0, 0.0]),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SimSatTopView => "simsat_top_view",
            Self::SimSatInfrared => "simsat_infrared",
            Self::SimSatWaterVapor => "simsat_water_vapor",
            Self::ReflectivityDbz => "reflectivity_dbz",
            Self::UpdraftMs => "updraft_ms",
            Self::VerticalVorticityS => "vertical_vorticity_s",
            Self::UpdraftHelicityM2S2 => "uh",
            Self::SurfaceVerticalVorticityS => "zvort_sfc",
            Self::SwathMaxWindMs => "sws",
            Self::SwathMaxVorticityS => "svs",
            Self::SwathMinPressurePa => "sps",
            Self::SwathMaxHelicityM2S2 => "shs",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::SimSatTopView => "SimSat physical visible/true-color top view",
            Self::SimSatInfrared => "SimSat infrared brightness temperature",
            Self::SimSatWaterVapor => "SimSat water-vapor channel",
            Self::ReflectivityDbz => "10 cm reflectivity",
            Self::UpdraftMs => "Mass-centered vertical velocity",
            Self::VerticalVorticityS => "Terrain-coordinate vertical vorticity",
            Self::UpdraftHelicityM2S2 => "Updraft helicity 2-5 km AGL (CM1 uh)",
            Self::SurfaceVerticalVorticityS => {
                "Vertical vorticity at the lowest model level (CM1 zvort)"
            }
            Self::SwathMaxWindMs => "Swath: max surface wind (CM1 sws)",
            Self::SwathMaxVorticityS => "Swath: max surface vertical vorticity (CM1 svs)",
            Self::SwathMinPressurePa => {
                "Swath: min surface pressure perturbation (CM1 sps)"
            }
            Self::SwathMaxHelicityM2S2 => "Swath: max updraft helicity (CM1 shs)",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.as_str() == value)
    }
}

#[derive(Clone, Debug)]
pub enum ScienceLayerAvailability {
    Available(ScienceLayerData),
    Derived(GeoFieldDescriptor),
    Refused(GeoFieldDescriptor),
    Missing(GeoFieldDescriptor),
}

impl ScienceLayerAvailability {
    pub fn descriptor(&self) -> &GeoFieldDescriptor {
        match self {
            Self::Available(layer) => &layer.descriptor,
            Self::Derived(descriptor) => descriptor,
            Self::Refused(descriptor) | Self::Missing(descriptor) => descriptor,
        }
    }

    pub fn data(&self) -> Option<&ScienceLayerData> {
        match self {
            Self::Available(layer) => Some(layer),
            Self::Derived(_) | Self::Refused(_) | Self::Missing(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScienceLayerData {
    pub descriptor: GeoFieldDescriptor,
    pub values: Vec<f32>,
    pub finite_min: f32,
    pub finite_max: f32,
    pub projection: ScienceProjection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScienceProjection {
    ColumnMaximum,
    ColumnMaximumAbsoluteSigned,
}

#[derive(Clone, Debug, Default)]
pub struct ScienceLayerRegistry {
    frames: BTreeMap<u64, ScienceFrameLayers>,
    refusals: Vec<String>,
}

#[derive(Clone, Debug)]
struct ScienceFrameLayers {
    source_payload_sha256: String,
    layers: BTreeMap<ScienceLayerId, ScienceLayerAvailability>,
}

impl ScienceLayerRegistry {
    pub fn load(root: Option<&Path>, timeline: &[Gm1Dataset], grid: &GeoGrid) -> Self {
        let Some(root) = root else {
            return Self::default();
        };
        match load_registry(root, timeline, grid) {
            Ok(registry) => registry,
            Err(reason) => Self {
                frames: BTreeMap::new(),
                refusals: vec![reason],
            },
        }
    }

    pub fn layer(&self, dataset: &Gm1Dataset, id: ScienceLayerId) -> ScienceLayerAvailability {
        if id == ScienceLayerId::SimSatTopView {
            return ScienceLayerAvailability::Derived(simsat_descriptor(dataset));
        }
        self.frames
            .get(&dataset.step)
            .filter(|frame| frame.source_payload_sha256 == dataset.payload_sha256)
            .and_then(|frame| frame.layers.get(&id))
            .cloned()
            .unwrap_or_else(|| ScienceLayerAvailability::Missing(missing_descriptor(dataset, id)))
    }

    pub fn refusals(&self) -> &[String] {
        &self.refusals
    }
}

#[derive(Clone, Debug)]
pub struct TopViewProduct {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
    pub rgba_sha256: String,
    pub descriptor: GeoFieldDescriptor,
    pub optical_depth_min: f32,
    pub optical_depth_max: f32,
}

#[derive(Clone, Debug)]
pub struct ScalarTopView {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
    pub rgba: Vec<u8>,
    pub rgba_sha256: String,
    pub range: [f32; 2],
    pub descriptor: GeoFieldDescriptor,
}

/// How one column's scalar-level altitudes are known.
///
/// The optical path through a level is its physical thickness, so a projection
/// that assumes the wrong level altitudes reports the wrong optical depth for
/// the same words.
#[derive(Clone, Copy, Debug)]
pub enum ColumnLevelFaces<'a> {
    /// Sealed terrain-composed stretched sigma profile. Valid only for the
    /// sealed level count it was measured on.
    SealedSigma,
    /// The run's own w-face heights above MSL, C-order `[nz + 1, ny, nx]`.
    Source(&'a [f32]),
}

impl ColumnLevelFaces<'_> {
    /// Lower and upper face height, in kilometres, of scalar level `k` in the
    /// column at `(j, i)`.
    fn bounds_km(
        self,
        k: usize,
        j: usize,
        i: usize,
        shape_zyx: [usize; 3],
        terrain_m: f32,
    ) -> Result<(f32, f32), String> {
        let [nz, ny, nx] = shape_zyx;
        match self {
            Self::SealedSigma => {
                let level = |index: usize| admitted_mass_height_km(index, terrain_m);
                let lower = if k == 0 {
                    terrain_m * 1.0e-3
                } else {
                    0.5 * (level(k - 1)? + level(k)?)
                };
                let upper = if k + 1 == nz {
                    ADMITTED_MODEL_TOP_M * 1.0e-3
                } else {
                    0.5 * (level(k)? + level(k + 1)?)
                };
                Ok((lower, upper))
            }
            Self::Source(faces) => {
                let at = |face: usize| {
                    faces
                        .get((face * ny + j) * nx + i)
                        .copied()
                        .ok_or_else(|| {
                            format!(
                                "w-face carrier is short of face {face} in column ({j},{i}); \
                                 the level's thickness is unknown, so its optical path \
                                 would be invented rather than measured"
                            )
                        })
                };
                Ok((at(k)? * 1.0e-3, at(k + 1)? * 1.0e-3))
            }
        }
    }
}

/// Physical cloud-top view of one admitted frame, on that frame's own grid.
///
/// `terrain_msl_m` is the domain surface in C-order `[ny, nx]`, `faces` states
/// the scalar-level altitudes, and `dx_m`/`dy_m` are the frame's horizontal
/// spacing, which sets the scale of the surface-shading gradient.
pub fn simsat_physical_top_view(
    dataset: &Gm1Dataset,
    terrain_msl_m: &[f32],
    faces: ColumnLevelFaces<'_>,
    dx_m: f64,
    dy_m: f64,
) -> Result<TopViewProduct, String> {
    let field = dataset
        .fields
        .iter()
        .find(|field| field.id == "cloud_condensate")
        .ok_or_else(|| "SimSat top view requires exact cloud_condensate = qc+qi".to_owned())?;
    let [nz, ny, nx] = field.shape;
    if nz == 0 || ny == 0 || nx == 0 {
        return Err(format!(
            "SimSat top view refuses the empty qc+qi grid {:?} (nz,ny,nx); a \
             projection with no cells in one axis has no column to integrate",
            field.shape
        ));
    }
    if terrain_msl_m.len() != ny * nx {
        return Err(format!(
            "SimSat top view refuses a {}-word surface against the {ny}x{nx} qc+qi \
             grid; indexing the surface with the field's strides would place one \
             column's ground under another column's air, so every column's \
             optical path would start at the wrong altitude",
            terrain_msl_m.len()
        ));
    }
    if !(dx_m.is_finite() && dx_m > 0.0 && dy_m.is_finite() && dy_m > 0.0) {
        return Err(format!(
            "SimSat top view refuses spacing dx={dx_m} m dy={dy_m} m; the surface \
             shading gradient would be scaled by an undefined distance and would \
             report relief the domain does not have"
        ));
    }
    if matches!(faces, ColumnLevelFaces::SealedSigma) && nz != SEALED_SCALAR_LEVEL_COUNT {
        return Err(format!(
            "SimSat top view refuses the sealed sigma profile for a {nz}-level \
             frame; the sealed profile states altitudes for \
             {SEALED_SCALAR_LEVEL_COUNT} levels only, so this frame's levels \
             would be placed at altitudes the run never simulated"
        ));
    }
    let sky = PositiveSky::t300_physical();
    let sky_up = sky.irradiance([0.0, 0.0, 1.0]);
    let direct = T300_DIRECT_IRRADIANCE_F32_BITS.map(f32::from_bits);
    let sun = T300_SUN_ENU_F32_BITS.map(f32::from_bits);
    let sun_length = sun.iter().map(|value| value * value).sum::<f32>().sqrt();
    let sun = sun.map(|value| value / sun_length);
    let mu0 = sun[2].max(0.05);
    let dense_cutoff = density_scale(field);
    let onset = 1.0e-4_f32;
    let mut optical_depth = vec![0.0_f32; ny * nx];
    let mut rgba = vec![0_u8; ny * nx * 4];
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let max_light = (0..3)
        .map(|channel| sky_up[channel] as f32 + direct[channel] * mu0)
        .fold(0.0_f32, f32::max)
        .max(f32::MIN_POSITIVE);
    // Central difference across two cells, in metres, so the shading gradient
    // is the domain's real slope rather than a per-cell rise.
    let run_x_m = 2.0 * dx_m as f32;
    let run_y_m = 2.0 * dy_m as f32;
    for j in 0..ny {
        for i in 0..nx {
            let horizontal = j * nx + i;
            let terrain_m = terrain_msl_m[horizontal];
            let mut tau = 0.0_f32;
            for k in 0..nz {
                let (lower, upper) = faces.bounds_km(k, j, i, field.shape, terrain_m)?;
                let q = field.values[(k * ny + j) * nx + i];
                tau +=
                    condensate_extinction_per_km(q, onset, dense_cutoff) * (upper - lower).max(0.0);
            }
            optical_depth[horizontal] = tau;
            minimum = minimum.min(tau);
            maximum = maximum.max(tau);

            let left = terrain_msl_m[j * nx + i.saturating_sub(1)];
            let right = terrain_msl_m[j * nx + (i + 1).min(nx - 1)];
            let south = terrain_msl_m[j.saturating_sub(1) * nx + i];
            let north = terrain_msl_m[(j + 1).min(ny - 1) * nx + i];
            let normal = normalize3([
                -(right - left) / run_x_m,
                -(north - south) / run_y_m,
                1.0,
            ]);
            let ndotl = dot3(normal, sun).max(0.0);
            let cloud_reflectance = 1.0 - (-tau / mu0).exp();
            let surface_transmittance = (-tau / mu0).exp();
            // SimSat's sealed Lambertian ground-albedo default is a rendering
            // treatment, not a land-cover claim. Every chromatic ratio comes
            // from the sealed physical sky/direct irradiance row.
            let mut rgb = [0.0_f32; 3];
            for channel in 0..3 {
                let sky_channel = sky_up[channel] as f32;
                let cloud_light = (sky_channel + direct[channel] * mu0) / max_light;
                let ground_light = SIMSAT_GROUND_BOUNCE_ALBEDO
                    * (sky_channel + direct[channel] * ndotl)
                    / max_light;
                rgb[channel] =
                    cloud_reflectance * cloud_light + surface_transmittance * ground_light;
            }
            let output_j = ny - 1 - j; // north-up display
            let offset = (output_j * nx + i) * 4;
            for channel in 0..3 {
                rgba[offset + channel] = display_encode(rgb[channel]);
            }
            rgba[offset + 3] = 255;
        }
    }
    Ok(TopViewProduct {
        width: nx,
        height: ny,
        rgba_sha256: sha256_bytes(&rgba),
        rgba,
        descriptor: simsat_descriptor(dataset),
        optical_depth_min: minimum,
        optical_depth_max: maximum,
    })
}

pub fn scalar_top_view(layer: &ScienceLayerData) -> Result<ScalarTopView, String> {
    let [nz, ny, nx] = match layer.descriptor.shape.as_slice() {
        &[nz, ny, nx] if nz > 0 && ny > 0 && nx > 0 => [nz, ny, nx],
        other => {
            return Err(format!(
                "science layer projection refuses the shape {other:?}; a column \
                 projection needs a three-dimensional [z,y,x] volume with no \
                 empty axis, and anything else has no column to project"
            ));
        }
    };
    if layer.values.len() != nz * ny * nx {
        return Err(format!(
            "science layer {} carries {} words against its declared shape \
             {:?}; projecting it with that shape's strides would read one \
             column's words at another column's position",
            layer.descriptor.id,
            layer.values.len(),
            layer.descriptor.shape
        ));
    }
    let mut values = vec![0.0_f32; ny * nx];
    for j in 0..ny {
        for i in 0..nx {
            let output = (ny - 1 - j) * nx + i;
            values[output] = match layer.projection {
                ScienceProjection::ColumnMaximum => (0..nz)
                    .map(|k| layer.values[(k * ny + j) * nx + i])
                    .fold(f32::NEG_INFINITY, f32::max),
                ScienceProjection::ColumnMaximumAbsoluteSigned => (0..nz)
                    .map(|k| layer.values[(k * ny + j) * nx + i])
                    .max_by(|left, right| left.abs().total_cmp(&right.abs()))
                    .unwrap_or(0.0),
            };
        }
    }
    let reflectivity = layer.descriptor.id == "reflectivity_dbz";
    let range = match layer.projection {
        // The published NWS scale's own span: its classes run from 5 dBZ to
        // 75 dBZ. A per-frame quantile range would move the class boundaries
        // frame to frame, so the same storm would change colour while its
        // reflectivity did not.
        ScienceProjection::ColumnMaximum if reflectivity => [
            NWS_DBZ_FLOOR,
            NWS_DBZ_CLASSES[NWS_DBZ_CLASSES.len() - 1].0,
        ],
        ScienceProjection::ColumnMaximum => finite_quantile_range(&values, 0.01, 0.995),
        ScienceProjection::ColumnMaximumAbsoluteSigned => {
            let limit = finite_abs_quantile(&values, 0.995).max(1.0e-6);
            [-limit, limit]
        }
    };
    let rgba = if layer.descriptor.id == "reflectivity_dbz" {
        // The field a meteorologist actually reads.  On the operational NWS
        // scale a column maximum is a radar image; on a generic sequential
        // ramp it is a heat map that happens to have dBZ units.
        nws_reflectivity_rgba(&values)
    } else if layer.projection == ScienceProjection::ColumnMaximumAbsoluteSigned {
        diverging_rgba(&values, range[1])
    } else if reflectivity {
        // Column-maximum reflectivity is the plan-view radar product. It is
        // painted with the sixteen-class NWS dBZ legend rather than a
        // continuous scientific ramp, so it reads as radar: the greens,
        // yellows and reds mean what a forecaster already knows they mean,
        // and clear air stays clear air rather than becoming the ramp's
        // darkest colour.
        nws_dbz_rgba(&values)
    } else {
        sequential_rgba(&values, range)
    };
    Ok(ScalarTopView {
        width: nx,
        height: ny,
        rgba_sha256: sha256_bytes(&rgba),
        rgba,
        values,
        range,
        descriptor: layer.descriptor.clone(),
    })
}

fn load_registry(
    root: &Path,
    timeline: &[Gm1Dataset],
    grid: &GeoGrid,
) -> Result<ScienceLayerRegistry, String> {
    if !root.is_dir() {
        return Err(format!(
            "science-layer root is not a directory: {}",
            root.display()
        ));
    }
    let timeline = timeline
        .iter()
        .map(|dataset| (dataset.step, dataset))
        .collect::<BTreeMap<_, _>>();
    let mut registry = ScienceLayerRegistry::default();
    let mut seen_steps = BTreeSet::new();
    for entry in
        fs::read_dir(root).map_err(|error| format!("science-layer root read failed: {error}"))?
    {
        let entry =
            entry.map_err(|error| format!("science-layer directory entry failed: {error}"))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.starts_with('.')
            || !entry
                .file_type()
                .is_ok_and(|kind| kind.is_dir() && !kind.is_symlink())
        {
            continue;
        }
        let Some(step) = name
            .strip_prefix("step-")
            .and_then(|value| value.parse::<u64>().ok())
        else {
            continue;
        };
        if !seen_steps.insert(step) {
            return Err(format!("duplicate science-layer step {step}"));
        }
        let Some(dataset) = timeline.get(&step) else {
            registry
                .refusals
                .push(format!("step {step}: no strictly admitted timeline frame"));
            continue;
        };
        match load_frame_layers(&entry.path(), dataset, grid) {
            Ok(frame) => {
                registry.frames.insert(step, frame);
            }
            Err(reason) => registry.refusals.push(format!("step {step}: {reason}")),
        }
    }
    Ok(registry)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScienceManifest {
    schema: String,
    status: String,
    source: ManifestSource,
    grid: ManifestGrid,
    layers: BTreeMap<String, ManifestLayer>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestSource {
    step: u64,
    model_time_s: f64,
    cloud_pack_payload_sha256: String,
    diagnostic_receipt_sha256: String,
    diagnostic_receipt_file: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestGrid {
    shape_zyx: [usize; 3],
    dx_m: f64,
    dy_m: f64,
    center_latitude_deg: f64,
    center_longitude_deg: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestLayer {
    file: String,
    sha256: String,
    nbytes: u64,
    units: String,
    meaning: String,
    producer: String,
    status: String,
    admission_reason: String,
    finite_min: f32,
    finite_max: f32,
    projection: String,
}

fn load_frame_layers(
    directory: &Path,
    dataset: &Gm1Dataset,
    grid: &GeoGrid,
) -> Result<ScienceFrameLayers, String> {
    let manifest_path = directory.join("manifest.json");
    let manifest: ScienceManifest = serde_json::from_slice(
        &fs::read(&manifest_path).map_err(|error| format!("manifest read failed: {error}"))?,
    )
    .map_err(|error| format!("manifest parse failed: {error}"))?;
    if manifest.schema != SCIENCE_LAYER_PACK_SCHEMA
        || manifest.status != "PARTIAL_PASS_LAYER_SPECIFIC"
        || manifest.source.step != dataset.step
        || manifest.source.cloud_pack_payload_sha256 != dataset.payload_sha256
        // The pack must describe the grid this frame was admitted on, whatever
        // that grid is. A pack measured on another domain would be drawn over
        // this one cell-for-cell, reporting another run's reflectivity, updraft
        // and vorticity at this run's positions.
        || manifest.grid.shape_zyx != [grid.nz, grid.ny, grid.nx]
        || manifest.grid.dx_m.to_bits() != grid.dx_m.to_bits()
        || manifest.grid.dy_m.to_bits() != grid.dy_m.to_bits()
        || manifest.source.model_time_s.to_bits() != dataset.model_time_s.to_bits()
        || manifest.grid.center_latitude_deg.to_bits() != grid.center_latitude_deg.to_bits()
        || manifest.grid.center_longitude_deg.to_bits() != grid.center_longitude_deg.to_bits()
    {
        return Err(
            "science manifest identity/grid/time differs from admitted GM1 frame".to_owned(),
        );
    }
    validate_sha256(&manifest.source.diagnostic_receipt_sha256)?;
    if manifest
        .source
        .diagnostic_receipt_file
        .contains(['/', '\\'])
    {
        return Err("science manifest diagnostic receipt path is not canonical".to_owned());
    }
    let diagnostic_receipt_bytes =
        fs::read(directory.join(&manifest.source.diagnostic_receipt_file))
            .map_err(|error| format!("diagnostic receipt read failed: {error}"))?;
    if sha256_bytes(&diagnostic_receipt_bytes) != manifest.source.diagnostic_receipt_sha256 {
        return Err("diagnostic receipt identity changed".to_owned());
    }
    let diagnostic_receipt: serde_json::Value =
        serde_json::from_slice(&diagnostic_receipt_bytes)
            .map_err(|error| format!("diagnostic receipt parse failed: {error}"))?;
    if diagnostic_receipt["schema"] != "gm1.real-50m-child-step600-scientific-diagnostics/v1"
        || diagnostic_receipt["source"]["step"].as_u64() != Some(dataset.step)
        || diagnostic_receipt["source"]["model_time_s"]
            .as_f64()
            .map(f64::to_bits)
            != Some(dataset.model_time_s.to_bits())
    {
        return Err("diagnostic receipt source identity differs from admitted frame".to_owned());
    }
    let mut layers = BTreeMap::new();
    for (id_text, manifest_layer) in manifest.layers {
        let Some(id) = ScienceLayerId::parse(&id_text) else {
            return Err(format!("unknown science layer {id_text:?}"));
        };
        if id == ScienceLayerId::SimSatTopView || layers.contains_key(&id) {
            return Err(format!("invalid or duplicate science layer {id_text:?}"));
        }
        validate_sha256(&manifest_layer.sha256)?;
        let receipt_field = match id {
            ScienceLayerId::ReflectivityDbz => "reflectivity_dbz",
            ScienceLayerId::UpdraftMs => "updraft_ms",
            ScienceLayerId::VerticalVorticityS => "vertical_vorticity_s",
            ScienceLayerId::SimSatTopView
            | ScienceLayerId::SimSatInfrared
            | ScienceLayerId::SimSatWaterVapor => {
                return Err(format!("invalid derived SimSat payload layer {id_text:?}"));
            }
            other if other.surface_carrier().is_some() => {
                return Err(format!(
                    "science layer {id_text:?} is a CM1 diagnostic surface \
                     carrier published in the forecast frame itself; a pack \
                     copy of it could disagree with the frame it is drawn \
                     over, so it is read from the frame and refused here"
                ));
            }
            _ => return Err(format!("unknown science layer {id_text:?}")),
        };
        if diagnostic_receipt["diagnostic_fields"][receipt_field]["sha256"].as_str()
            != Some(manifest_layer.sha256.as_str())
        {
            return Err(format!(
                "science layer {id_text} differs from diagnostic receipt"
            ));
        }
        let layer_words = grid.nz * grid.ny * grid.nx;
        if manifest_layer.file.contains(['/', '\\'])
            || manifest_layer.nbytes != layer_words as u64 * 4
        {
            return Err(format!(
                "science layer {id_text} path is not canonical, or its \
                 {} bytes are not the {layer_words} binary32 words this \
                 frame's {}x{}x{} grid needs; reading a differently sized \
                 payload as this grid would project one column's words at \
                 another column's position",
                manifest_layer.nbytes, grid.nz, grid.ny, grid.nx
            ));
        }
        let path = directory.join(&manifest_layer.file);
        let bytes =
            fs::read(&path).map_err(|error| format!("{id_text} payload read failed: {error}"))?;
        if bytes.len() as u64 != manifest_layer.nbytes
            || sha256_bytes(&bytes) != manifest_layer.sha256
        {
            return Err(format!("science layer {id_text} payload identity changed"));
        }
        let mut values = Vec::with_capacity(layer_words);
        let mut finite_min = f32::INFINITY;
        let mut finite_max = f32::NEG_INFINITY;
        for word in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes(word.try_into().expect("four-byte chunk"));
            if !value.is_finite() {
                return Err(format!(
                    "science layer {id_text} contains non-finite values"
                ));
            }
            finite_min = finite_min.min(value);
            finite_max = finite_max.max(value);
            values.push(value);
        }
        if finite_min.to_bits() != manifest_layer.finite_min.to_bits()
            || finite_max.to_bits() != manifest_layer.finite_max.to_bits()
        {
            return Err(format!("science layer {id_text} finite range changed"));
        }
        let projection = match manifest_layer.projection.as_str() {
            "column_maximum" => ScienceProjection::ColumnMaximum,
            "column_maximum_absolute_signed" => ScienceProjection::ColumnMaximumAbsoluteSigned,
            other => {
                return Err(format!(
                    "science layer {id_text} has unsupported projection {other:?}"
                ));
            }
        };
        let admission = match manifest_layer.status.as_str() {
            "PASS" => GeoFieldAdmission::Available,
            status if status.starts_with("REFUSED_") => GeoFieldAdmission::Refused,
            other => {
                return Err(format!(
                    "science layer {id_text} has invalid status {other:?}"
                ));
            }
        };
        let continuity_key = match id {
            ScienceLayerId::ReflectivityDbz => "low_level_reflectivity_dbz",
            ScienceLayerId::UpdraftMs => "low_level_updraft_ms",
            ScienceLayerId::VerticalVorticityS => "low_level_abs_vertical_vorticity_s",
            _ => unreachable!("derived and frame-carried layers refused above"),
        };
        let continuity_passed = diagnostic_receipt["edge_continuity"]["fields"]
            [continuity_key]["outer_edge_below_or_equal_domain_p99_9"]
            .as_bool()
            .ok_or_else(|| format!("science layer {id_text} continuity receipt is missing"))?;
        if continuity_passed != (admission == GeoFieldAdmission::Available) {
            return Err(format!(
                "science layer {id_text} admission differs from boundary-continuity receipt"
            ));
        }
        let descriptor = GeoFieldDescriptor {
            id: id_text,
            label: id.label().to_owned(),
            units: manifest_layer.units,
            meaning: manifest_layer.meaning,
            geometry: GeoFieldGeometry::VolumeZyx,
            shape: vec![grid.nz, grid.ny, grid.nx],
            values_sha256: manifest_layer.sha256,
            admission,
            admission_reason: manifest_layer.admission_reason,
            provenance: manifest_layer.producer,
        };
        let availability = if admission == GeoFieldAdmission::Available {
            ScienceLayerAvailability::Available(ScienceLayerData {
                descriptor,
                values,
                finite_min,
                finite_max,
                projection,
            })
        } else {
            ScienceLayerAvailability::Refused(descriptor)
        };
        layers.insert(id, availability);
    }
    Ok(ScienceFrameLayers {
        source_payload_sha256: dataset.payload_sha256.clone(),
        layers,
    })
}

fn simsat_descriptor(dataset: &Gm1Dataset) -> GeoFieldDescriptor {
    GeoFieldDescriptor {
        id: ScienceLayerId::SimSatTopView.as_str().to_owned(),
        label: ScienceLayerId::SimSatTopView.label().to_owned(),
        units: "display RGB + exact-derived optical depth".to_owned(),
        meaning: "north-up physical cloud/surface top view from exact qc+qi and exact terrain"
            .to_owned(),
        geometry: GeoFieldGeometry::RgbaYx,
        shape: vec![300, 300, 4],
        values_sha256: dataset
            .fields
            .iter()
            .find(|field| field.id == "cloud_condensate")
            .map_or_else(|| "0".repeat(64), |field| field.values_sha256.clone()),
        admission: GeoFieldAdmission::Available,
        admission_reason: "source-backed display product; exact meteorological words unchanged"
            .to_owned(),
        provenance: format!(
            "{} @ {}; sources api={} clouds={} atmosphere={} sky-row={} terrain={}",
            SIMSAT_PROVENANCE,
            SIMSAT_SOURCE_REVISION,
            SIMSAT_TOP_VIEW_API_SOURCE_SHA256,
            SIMSAT_TOP_VIEW_CLOUD_SOURCE_SHA256,
            SIMSAT_TOP_VIEW_ATMOSPHERE_SOURCE_SHA256,
            T300_SKY_ROW_SHA256,
            ADMITTED_TERRAIN_SHA256,
        ),
    }
}

fn missing_descriptor(dataset: &Gm1Dataset, id: ScienceLayerId) -> GeoFieldDescriptor {
    GeoFieldDescriptor {
        id: id.as_str().to_owned(),
        label: id.label().to_owned(),
        units: science_layer_units(id).to_owned(),
        meaning: "optional science layer".to_owned(),
        geometry: if science_layer_is_raster(id) {
            GeoFieldGeometry::RasterYx
        } else {
            GeoFieldGeometry::VolumeZyx
        },
        shape: if science_layer_is_raster(id) {
            vec![300, 300]
        } else {
            vec![75, 300, 300]
        },
        values_sha256: "0".repeat(64),
        admission: GeoFieldAdmission::Missing,
        admission_reason: if id.surface_carrier().is_some() {
            format!(
                "MISSING_FAIL_CLOSED: this frame publishes no {} carrier at \
                 step {}; the run's document selected no such CM1 output_* \
                 switch, and drawing a zero plane in its place would show a \
                 quiet surface for a run that never measured one",
                id.as_str(),
                dataset.step
            )
        } else {
            format!(
                "MISSING_FAIL_CLOSED: exact carrier/receipt absent for step {}",
                dataset.step
            )
        },
        provenance: "no source admitted".to_owned(),
    }
}

pub const fn science_layer_units(id: ScienceLayerId) -> &'static str {
    match id {
        ScienceLayerId::ReflectivityDbz => "dBZ",
        ScienceLayerId::UpdraftMs => "m s-1",
        ScienceLayerId::VerticalVorticityS => "s-1",
        ScienceLayerId::SimSatTopView => "display RGB",
        ScienceLayerId::SimSatInfrared => "K",
        ScienceLayerId::SimSatWaterVapor => "K",
        ScienceLayerId::UpdraftHelicityM2S2 | ScienceLayerId::SwathMaxHelicityM2S2 => {
            "m2 s-2"
        }
        ScienceLayerId::SurfaceVerticalVorticityS | ScienceLayerId::SwathMaxVorticityS => {
            "s-1"
        }
        ScienceLayerId::SwathMaxWindMs => "m s-1",
        ScienceLayerId::SwathMinPressurePa => "Pa",
    }
}

const fn science_layer_is_raster(id: ScienceLayerId) -> bool {
    matches!(
        id,
        ScienceLayerId::SimSatTopView
            | ScienceLayerId::SimSatInfrared
            | ScienceLayerId::SimSatWaterVapor
    ) || id.surface_carrier().is_some()
}

/// Draw one CM1 diagnostic surface carrier published in this frame.
///
/// The frame owns the words; this owns only the north-up flip and the fixed
/// physical colour range. Nothing is normalized per frame, nothing is
/// extrapolated, and a carrier the frame does not publish is refused rather
/// than drawn as zero -- a swath of zeros and a swath nobody measured look
/// identical on screen, and only one of them is a statement about the run.
pub fn tornado_surface_top_view(
    dataset: &Gm1Dataset,
    id: ScienceLayerId,
) -> Result<ScalarTopView, String> {
    let Some(name) = id.surface_carrier() else {
        return Err(format!(
            "science layer {} is not a CM1 diagnostic surface carrier",
            id.as_str()
        ));
    };
    let Some(range) = id.surface_range() else {
        return Err(format!(
            "science layer {} declares no fixed physical colour range",
            id.as_str()
        ));
    };
    let Some(field) = dataset
        .tornado_surface_fields
        .iter()
        .find(|candidate| candidate.id == name)
    else {
        return Err(format!(
            "this frame publishes no {name} carrier; the run's document \
             selected no such CM1 output_* switch"
        ));
    };
    let [ny, nx] = field.shape;
    if ny == 0 || nx == 0 || field.values.len() != ny * nx {
        return Err(format!(
            "CM1 diagnostic surface carrier {name} carries {} words against \
             its declared shape {:?}",
            field.values.len(),
            field.shape
        ));
    }
    let mut values = vec![0.0_f32; ny * nx];
    for j in 0..ny {
        for i in 0..nx {
            values[(ny - 1 - j) * nx + i] = field.values[j * nx + i];
        }
    }
    let signed = range[0] < 0.0 && range[1] > 0.0;
    let rgba = if signed {
        diverging_rgba(&values, range[1])
    } else {
        sequential_rgba(&values, range)
    };
    Ok(ScalarTopView {
        width: nx,
        height: ny,
        rgba_sha256: sha256_bytes(&rgba),
        rgba,
        values,
        range,
        descriptor: GeoFieldDescriptor {
            id: field.id.clone(),
            label: id.label().to_owned(),
            units: field.units.clone(),
            meaning: field.meaning.clone(),
            geometry: GeoFieldGeometry::RasterYx,
            shape: vec![ny, nx],
            values_sha256: field.values_sha256.clone(),
            admission: GeoFieldAdmission::Available,
            admission_reason: format!(
                "exact CM1 {name} carrier published in this forecast frame; \
                 fixed physical colour range {range:?}, no per-frame \
                 normalization"
            ),
            provenance: format!(
                "CM1 r21.1 src/misclibs.F:1417-1532 calcuh, \
                 src/misclibs.F:1540-2174 calcvort and src/solve3.F:855-1136 \
                 the lowest-model-level running swaths"
            ),
        },
    })
}

fn normalize3(vector: [f32; 3]) -> [f32; 3] {
    let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    vector.map(|value| value / length.max(f32::MIN_POSITIVE))
}

fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn display_encode(value: f32) -> u8 {
    let exposed = (value.max(0.0) * 1.45).powf(1.0 / 2.2);
    (255.0 * exposed.clamp(0.0, 1.0)).round() as u8
}

/// The National Weather Service's operational NEXRAD reflectivity scale.
///
/// Every WSR-88D product a forecaster looks at uses this table: fourteen
/// 5 dBZ bins from 5 to 75 dBZ, each a flat colour, with everything below
/// 5 dBZ left as background.  The discreteness is the point -- the eye reads
/// the bin edges as contours, which is how "40 dBZ core", "hook echo" and
/// "bounded weak echo region" become visible at a glance.  A smooth gradient
/// over the same range is prettier and says less.
///
/// Colours are the published NWS/NEXRAD table values, not a re-derivation.
const NWS_REFLECTIVITY_BINS: [(f32, [u8; 3]); 15] = [
    (5.0, [4, 233, 231]),
    (10.0, [1, 159, 244]),
    (15.0, [3, 0, 244]),
    (20.0, [2, 253, 2]),
    (25.0, [1, 197, 1]),
    (30.0, [0, 142, 0]),
    (35.0, [253, 248, 2]),
    (40.0, [229, 188, 0]),
    (45.0, [253, 149, 0]),
    (50.0, [253, 0, 0]),
    (55.0, [212, 0, 0]),
    (60.0, [188, 0, 0]),
    (65.0, [248, 0, 253]),
    (70.0, [152, 84, 198]),
    (75.0, [255, 255, 255]),
];

/// Below the 5 dBZ floor there is no echo.  CM1's calccref seeds an
/// echo-free column at -1000 dBZ; painting that as a weak return would put
/// rain on the screen where the model has clear air.
const NWS_REFLECTIVITY_BELOW_FLOOR: [u8; 3] = [12, 14, 23];

pub fn nws_reflectivity_rgb(dbz: f32) -> [u8; 3] {
    if !dbz.is_finite() {
        return [90, 90, 90];
    }
    let mut chosen = NWS_REFLECTIVITY_BELOW_FLOOR;
    for (edge, colour) in NWS_REFLECTIVITY_BINS {
        if dbz >= edge {
            chosen = colour;
        }
    }
    chosen
}

fn nws_reflectivity_rgba(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let rgb = nws_reflectivity_rgb(*value);
            [rgb[0], rgb[1], rgb[2], 255]
        })
        .collect()
}

fn sequential_rgba(values: &[f32], range: [f32; 2]) -> Vec<u8> {
    let span = (range[1] - range[0]).max(f32::MIN_POSITIVE);
    values
        .iter()
        .flat_map(|value| {
            let t = ((*value - range[0]) / span).clamp(0.0, 1.0);
            let rgb = scientific_palette(t);
            [rgb[0], rgb[1], rgb[2], 255]
        })
        .collect()
}

/// The published NWS dBZ classes as RGBA, with no echo left fully
/// transparent so a viewer composites clear air against its own ground
/// instead of painting a colour over it.
fn nws_dbz_rgba(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            if !value.is_finite() || *value < NWS_DBZ_FLOOR {
                [0, 0, 0, 0]
            } else {
                let rgb = nws_dbz_rgb(*value, [0, 0, 0]);
                [rgb[0], rgb[1], rgb[2], 255]
            }
        })
        .collect()
}

fn diverging_rgba(values: &[f32], limit: f32) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let t = (*value / limit).clamp(-1.0, 1.0);
            let rgb = if t < 0.0 {
                lerp_rgb([44, 82, 130], [247, 246, 242], t + 1.0)
            } else {
                lerp_rgb([247, 246, 242], [163, 52, 54], t)
            };
            [rgb[0], rgb[1], rgb[2], 255]
        })
        .collect()
}

fn scientific_palette(t: f32) -> [u8; 3] {
    let stops = [
        [38, 59, 92],
        [35, 122, 142],
        [94, 173, 132],
        [224, 201, 110],
        [204, 82, 54],
    ];
    let x = t * (stops.len() - 1) as f32;
    let left = x.floor() as usize;
    let right = (left + 1).min(stops.len() - 1);
    lerp_rgb(stops[left], stops[right], x - left as f32)
}

fn lerp_rgb(left: [u8; 3], right: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    std::array::from_fn(|channel| {
        (f32::from(left[channel]) + (f32::from(right[channel]) - f32::from(left[channel])) * t)
            .round() as u8
    })
}

fn finite_quantile_range(values: &[f32], lo: f32, hi: f32) -> [f32; 2] {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    finite.sort_by(f32::total_cmp);
    if finite.is_empty() {
        return [0.0, 1.0];
    }
    let at = |quantile: f32| finite[((finite.len() - 1) as f32 * quantile).round() as usize];
    let lower = at(lo);
    let upper = at(hi);
    [
        lower,
        upper.max(lower + f32::EPSILON.max(lower.abs() * f32::EPSILON)),
    ]
}

fn finite_abs_quantile(values: &[f32], quantile: f32) -> f32 {
    let mut finite = values
        .iter()
        .filter_map(|value| value.is_finite().then_some(value.abs()))
        .collect::<Vec<_>>();
    finite.sort_by(f32::total_cmp);
    finite
        .get(((finite.len().saturating_sub(1)) as f32 * quantile).round() as usize)
        .copied()
        .unwrap_or(1.0)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err("science manifest contains an invalid SHA-256".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ColumnLevelFaces, NWS_DBZ_FLOOR, ScienceLayerAvailability, ScienceLayerId,
        ScienceLayerRegistry, ScienceProjection, missing_descriptor, scalar_top_view,
        simsat_physical_top_view, tornado_surface_top_view,
    };
    use crate::{
        geofield::{GeoFieldAdmission, GeoFieldDescriptor, GeoFieldGeometry},
        input::{Gm1Dataset, VolumeField},
        science_layers::ScienceLayerData,
    };
    use std::path::PathBuf;

    fn dataset_with_swath(name: &str, values: Vec<f32>, ny: usize, nx: usize) -> Gm1Dataset {
        let mut set = dataset(600);
        set.tornado_surface_fields = vec![crate::input::SurfaceDiagnosticField {
            id: name.to_owned(),
            label: "swath".to_owned(),
            units: "unit".to_owned(),
            meaning: "CM1 surface swath".to_owned(),
            shape: [ny, nx],
            values_sha256: "6".repeat(64),
            finite_min: values.iter().copied().fold(f32::INFINITY, f32::min),
            finite_max: values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            values,
        }];
        set
    }

    #[test]
    fn tornado_surface_layers_name_cm1_carriers_and_fixed_ranges() {
        for (id, carrier) in [
            (ScienceLayerId::UpdraftHelicityM2S2, "uh"),
            (ScienceLayerId::SurfaceVerticalVorticityS, "zvort_sfc"),
            (ScienceLayerId::SwathMaxWindMs, "sws"),
            (ScienceLayerId::SwathMaxVorticityS, "svs"),
            (ScienceLayerId::SwathMinPressurePa, "sps"),
            (ScienceLayerId::SwathMaxHelicityM2S2, "shs"),
        ] {
            assert_eq!(id.surface_carrier(), Some(carrier));
            assert_eq!(id.as_str(), carrier);
            assert!(id.surface_range().is_some());
            assert_eq!(ScienceLayerId::parse(carrier), Some(id));
        }
        // The pack layers must keep reporting no surface carrier, or the app
        // would read them from the frame and find nothing.
        for id in [
            ScienceLayerId::ReflectivityDbz,
            ScienceLayerId::UpdraftMs,
            ScienceLayerId::VerticalVorticityS,
            ScienceLayerId::SimSatTopView,
        ] {
            assert_eq!(id.surface_carrier(), None);
        }
    }

    #[test]
    fn tornado_surface_top_view_is_north_up_and_not_per_frame_normalized() {
        // Two frames of one run with different maxima must colour the same
        // value identically, or a playback shows a track that is not there.
        let weak = dataset_with_swath("sws", vec![10.0, 10.0, 10.0, 10.0], 2, 2);
        let strong = dataset_with_swath("sws", vec![10.0, 80.0, 10.0, 10.0], 2, 2);
        let a = tornado_surface_top_view(&weak, ScienceLayerId::SwathMaxWindMs).unwrap();
        let b = tornado_surface_top_view(&strong, ScienceLayerId::SwathMaxWindMs).unwrap();
        assert_eq!(a.range, [0.0, 90.0]);
        assert_eq!(a.range, b.range);
        assert_eq!(&a.rgba[0..4], &b.rgba[0..4]);
        // North-up: source row 0 is the south edge and must land on the last
        // output row.
        let flipped =
            tornado_surface_top_view(&dataset_with_swath("svs", vec![1.0, 0.0, 0.0, 0.0], 2, 2),
                                     ScienceLayerId::SwathMaxVorticityS)
                .unwrap();
        assert_eq!(flipped.values, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn a_frame_without_the_carrier_is_refused_not_drawn_as_zero() {
        let plain = dataset(600);
        let error = tornado_surface_top_view(&plain, ScienceLayerId::SwathMaxWindMs)
            .expect_err("a frame with no sws carrier must refuse");
        assert!(error.contains("publishes no sws carrier"));
        let descriptor = missing_descriptor(&plain, ScienceLayerId::SwathMaxWindMs);
        assert_eq!(descriptor.units, "m s-1");
        assert!(descriptor.admission_reason.contains("quiet surface"));
    }

    fn dataset(step: u64) -> Gm1Dataset {
        Gm1Dataset {
            source_path: PathBuf::from("exact-pack"),
            artifact_kind: "exact pack".to_owned(),
            artifact_sha256: "1".repeat(64),
            payload_sha256: "2".repeat(64),
            contract_sha256: "3".repeat(64),
            config_sha256: "5".repeat(64),
            profile_id: "gm1-real-50m-child-v1".to_owned(),
            step,
            model_time_s: step as f64 * 0.5,
            dt_s: 0.5,
            fields: vec![VolumeField {
                id: "cloud_condensate".to_owned(),
                label: "Cloud condensate".to_owned(),
                units: "kg kg-1".to_owned(),
                meaning: "qc+qi".to_owned(),
                source: "exact".to_owned(),
                shape: [75, 300, 300],
                values: Vec::new(),
                values_sha256: "4".repeat(64),
                finite_min: 0.0,
                finite_max: 0.0,
            }],
            terrain_msl_m: None,
            face_height_m: None,
            tornado_surface_fields: Vec::new(),
        }
    }

    #[test]
    fn absent_optional_science_layers_fail_closed() {
        let registry = ScienceLayerRegistry::default();
        let frame = dataset(43);
        match registry.layer(&frame, ScienceLayerId::ReflectivityDbz) {
            ScienceLayerAvailability::Missing(descriptor) => {
                assert_eq!(descriptor.admission, GeoFieldAdmission::Missing);
                assert!(descriptor.admission_reason.contains("MISSING_FAIL_CLOSED"));
            }
            other => panic!("unexpected optional-layer state {other:?}"),
        }
    }

    #[test]
    fn signed_vorticity_projection_preserves_sign_at_maximum_magnitude() {
        let mut values = vec![0.0_f32; 75 * 300 * 300];
        values[0] = 0.2;
        values[300 * 300] = -0.5;
        let layer = ScienceLayerData {
            descriptor: GeoFieldDescriptor {
                id: "vertical_vorticity_s".to_owned(),
                label: "Vertical vorticity".to_owned(),
                units: "s-1".to_owned(),
                meaning: "test".to_owned(),
                geometry: GeoFieldGeometry::VolumeZyx,
                shape: vec![75, 300, 300],
                values_sha256: "5".repeat(64),
                admission: GeoFieldAdmission::Available,
                admission_reason: "test".to_owned(),
                provenance: "test".to_owned(),
            },
            values,
            finite_min: -0.5,
            finite_max: 0.2,
            projection: ScienceProjection::ColumnMaximumAbsoluteSigned,
        };
        let raster = scalar_top_view(&layer).unwrap();
        assert_eq!(raster.values[299 * 300], -0.5);
        assert!(raster.range[0] < 0.0 && raster.range[1] > 0.0);
    }

    fn scalar_layer(shape: Vec<usize>, values: Vec<f32>) -> ScienceLayerData {
        ScienceLayerData {
            descriptor: GeoFieldDescriptor {
                id: "reflectivity_dbz".to_owned(),
                label: "Reflectivity".to_owned(),
                units: "dBZ".to_owned(),
                meaning: "test".to_owned(),
                geometry: GeoFieldGeometry::VolumeZyx,
                shape,
                values_sha256: "6".repeat(64),
                admission: GeoFieldAdmission::Available,
                admission_reason: "test".to_owned(),
                provenance: "test".to_owned(),
            },
            finite_min: values.iter().copied().fold(f32::INFINITY, f32::min),
            finite_max: values.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            values,
            projection: ScienceProjection::ColumnMaximum,
        }
    }

    #[test]
    fn a_column_projection_follows_the_layers_own_grid() {
        // A non-sealed grid, north-up, with the maximum in one known column.
        let (nz, ny, nx) = (5_usize, 7_usize, 11_usize);
        let mut values = vec![0.0_f32; nz * ny * nx];
        values[(3 * ny + 2) * nx + 4] = 42.0;
        let raster = scalar_top_view(&scalar_layer(vec![nz, ny, nx], values))
            .expect("an arbitrary [z,y,x] grid projects");
        assert_eq!(raster.width, nx);
        assert_eq!(raster.height, ny);
        assert_eq!(raster.values.len(), ny * nx);
        assert_eq!(raster.values[(ny - 1 - 2) * nx + 4], 42.0);
    }

    #[test]
    fn column_maximum_reflectivity_is_painted_with_the_nws_dbz_scale() {
        // The plan-view radar product: a fixed published scale, class
        // colours, and clear air left clear. A per-frame quantile range
        // would move the class boundaries frame to frame and the same core
        // would change colour while its reflectivity did not.
        let (nz, ny, nx) = (3_usize, 2_usize, 2_usize);
        let mut values = vec![-35.0_f32; nz * ny * nx];
        values[0] = 52.0; // column (0,0): a 50 dBZ class core
        values[3] = 22.0; // column (1,1): a 20 dBZ class edge
        let raster = scalar_top_view(&scalar_layer(vec![nz, ny, nx], values))
            .expect("a reflectivity volume projects");
        assert_eq!(raster.range, [NWS_DBZ_FLOOR, 75.0]);
        let pixel = |x: usize, y: usize| {
            let offset = ((ny - 1 - y) * nx + x) * 4;
            [
                raster.rgba[offset],
                raster.rgba[offset + 1],
                raster.rgba[offset + 2],
                raster.rgba[offset + 3],
            ]
        };
        assert_eq!(pixel(0, 0), [0xFD, 0x00, 0x00, 255]);
        assert_eq!(pixel(1, 1), [0x02, 0xFD, 0x02, 255]);
        // -35 dBZ is below the published floor: no echo, fully transparent.
        assert_eq!(pixel(1, 0), [0, 0, 0, 0]);
        assert_eq!(pixel(0, 1), [0, 0, 0, 0]);
    }

    #[test]
    fn a_word_count_that_disagrees_with_the_declared_grid_is_refused() {
        let error = scalar_top_view(&scalar_layer(vec![5, 7, 11], vec![0.0_f32; 5 * 7 * 10]))
            .expect_err("a short payload must not be projected with this grid's strides");
        assert!(
            error.contains("another column's position"),
            "{error}"
        );
    }

    #[test]
    fn a_non_volume_layer_shape_is_refused() {
        let error = scalar_top_view(&scalar_layer(vec![7, 11], vec![0.0_f32; 77]))
            .expect_err("a two-dimensional layer has no column to project");
        assert!(error.contains("no column to project"), "{error}");
    }

    #[test]
    fn the_sealed_sigma_profile_is_refused_for_a_frame_it_was_not_measured_on() {
        let [nz, ny, nx] = [4_usize, 6_usize, 6_usize];
        let mut dataset = dataset(11);
        let values = vec![0.0_f32; nz * ny * nx];
        dataset.fields = vec![VolumeField {
            id: "cloud_condensate".to_owned(),
            label: "Resolved cloud condensate".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "qc + qi".to_owned(),
            source: "test".to_owned(),
            shape: [nz, ny, nx],
            values_sha256: crate::input::f32_sha256(&values),
            finite_min: 0.0,
            finite_max: 0.0,
            values,
        }];
        let terrain = vec![0.0_f32; ny * nx];
        let error = simsat_physical_top_view(
            &dataset,
            &terrain,
            ColumnLevelFaces::SealedSigma,
            1_000.0,
            1_000.0,
        )
        .expect_err("the sealed profile states altitudes for its own level count only");
        assert!(
            error.contains("altitudes the run never simulated"),
            "{error}"
        );
    }

    #[test]
    fn a_frames_own_face_heights_project_its_own_grid() {
        let [nz, ny, nx] = [4_usize, 6_usize, 6_usize];
        let mut dataset = dataset(11);
        let mut values = vec![0.0_f32; nz * ny * nx];
        values[(2 * ny + 3) * nx + 3] = 3.0e-3;
        dataset.fields = vec![VolumeField {
            id: "cloud_condensate".to_owned(),
            label: "Resolved cloud condensate".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "qc + qi".to_owned(),
            source: "test".to_owned(),
            shape: [nz, ny, nx],
            values_sha256: crate::input::f32_sha256(&values),
            finite_min: 0.0,
            finite_max: 3.0e-3,
            values,
        }];
        let terrain = vec![0.0_f32; ny * nx];
        // Uniform 500 m faces, C-order [nz + 1, ny, nx].
        let faces = (0..=nz)
            .flat_map(|k| std::iter::repeat_n(k as f32 * 500.0, ny * nx))
            .collect::<Vec<_>>();
        let product = simsat_physical_top_view(
            &dataset,
            &terrain,
            ColumnLevelFaces::Source(&faces),
            1_000.0,
            1_000.0,
        )
        .expect("a frame's own faces project its own grid");
        assert_eq!(product.width, nx);
        assert_eq!(product.height, ny);
        assert_eq!(product.rgba.len(), ny * nx * 4);
        // Only the loaded column carries optical depth.
        assert!(product.optical_depth_max > 0.0);
        assert_eq!(product.optical_depth_min.to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn a_surface_that_does_not_match_the_field_grid_is_refused() {
        let dataset = dataset(11);
        let error = simsat_physical_top_view(
            &dataset,
            &[0.0_f32; 4],
            ColumnLevelFaces::SealedSigma,
            50.0,
            50.0,
        )
        .expect_err("a mismatched surface must not be indexed with the field's strides");
        assert!(
            error.contains("another column's air"),
            "{error}"
        );
    }
}
