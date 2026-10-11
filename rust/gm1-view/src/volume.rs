//! Bevy material and exact binary32-word storage-buffer preparation.

use crate::{
    input::{VolumeField, f32_sha256},
    lighting::{PositiveSky, T300_DIRECT_IRRADIANCE_F32_BITS, T300_SUN_ENU_F32_BITS},
    physical_atmosphere::PHYSICAL_ATMOSPHERE_EXPOSURE,
    production_playback::ProductionResidentField,
};
#[cfg(not(debug_assertions))]
use bevy::asset::{AssetPath, embedded_asset, embedded_path};
use bevy::{
    asset::RenderAssetUsages,
    mesh::MeshVertexBufferLayoutRef,
    pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin},
    prelude::*,
    reflect::TypePath,
    render::{
        render_resource::{
            AsBindGroup, Face, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
        },
        storage::ShaderBuffer,
    },
    shader::ShaderRef,
};

pub const VOLUME_SHADER: &str = "gm1_volume.wgsl";
pub const DEV_PROOF_SHADER: &str = "dev_proof.wgsl";
pub const MAX_RAY_STEPS: u32 = 256;
/// Display microphysics used only when the qc+qi transport does not carry
/// droplet/ice effective radii or air density. The 60 um onset radius is a
/// conservative mixed/ice-haze assumption; the 12 um dense radius represents
/// resolved cloud water. Both remain explicit and source words stay unchanged.
pub const ONSET_EFFECTIVE_RADIUS_M: f32 = 60.0e-6;
pub const DENSE_EFFECTIVE_RADIUS_M: f32 = 12.0e-6;
pub const ASSUMED_AIR_DENSITY_KG_M3: f32 = 0.8;
pub const EXTINCTION_EFFICIENCY: f32 = 2.0;
pub const CONDENSED_WATER_DENSITY_KG_M3: f32 = 1_000.0;
/// Minimum subgrid occupancy assigned to source words just above physical
/// onset. This distinguishes grid-mean haze from the fully occupied q99 core
/// while retaining every admitted condensate word in the integration.
pub const ONSET_GRID_OCCUPANCY: f32 = 0.025;
/// SimSat's physical Lambertian ground-albedo default, already sealed into
/// the admitted GM1 full-sphere environment receipt. This is source-backed
/// lower-hemisphere energy, not an authored terrain colour.
pub const SIMSAT_GROUND_BOUNCE_ALBEDO: f32 = 0.3;
/// Two energy-reduced same-field multiple-scatter octave weights. They change
/// neither the density carrier nor its sampling locations.
pub const MULTIPLE_SCATTER_OCTAVE_WEIGHTS: [f32; 2] = [0.5, 0.25];

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct VolumeUniform {
    pub world_min: Vec4,
    pub world_size: Vec4,
    /// threshold, opacity, ray steps, positive-sky toggle
    pub control: Vec4,
    pub sun_direction: Vec4,
    pub sun_color: Vec4,
    pub sky_amplitude: Vec4,
    /// Exact storage layout dimensions: nx, ny, nz; w toggles a display-only
    /// fade over the exact three-cell open lateral boundary band.
    pub grid_shape: UVec4,
    /// onset effective radius [m], dense effective radius [m], assumed air
    /// density [kg m-3], source-derived q99 connected-component cutoff.
    pub optics: Vec4,
    /// source-sky display exposure, SimSat ground albedo, then the two
    /// energy-reduced same-field multiple-scatter octave weights.
    pub scattering: Vec4,
    pub sky_log_coef: [Vec4; 16],
}

impl VolumeUniform {
    pub fn for_field(field: &VolumeField, world_min: Vec3, world_size: Vec3) -> Self {
        let threshold = visible_threshold(&field.values, field.finite_min, field.finite_max);
        let sky = PositiveSky::t300_physical();
        let sun_enu = T300_SUN_ENU_F32_BITS.map(f32::from_bits);
        let direct = T300_DIRECT_IRRADIANCE_F32_BITS.map(f32::from_bits);
        Self {
            world_min: world_min.extend(0.0),
            // The otherwise-unused homogeneous component carries the stable
            // display scale without perturbing the sealed WGSL uniform ABI.
            world_size: world_size.extend(density_scale(field)),
            control: Vec4::new(threshold, 1.1, 192.0, 1.0),
            // SimSat ENU -> Bevy world (east, up, north).
            sun_direction: Vec3::new(sun_enu[0], sun_enu[2], sun_enu[1])
                .normalize()
                .extend(0.0),
            // Physical direct-normal RGB W m-2 plus a display-only exposure scale.
            sun_color: Vec4::new(direct[0], direct[1], direct[2], 0.006),
            sky_amplitude: Vec4::new(
                sky.amplitude[0] as f32,
                sky.amplitude[1] as f32,
                sky.amplitude[2] as f32,
                0.0,
            ),
            grid_shape: UVec4::new(
                field.shape[2] as u32,
                field.shape[1] as u32,
                field.shape[0] as u32,
                1,
            ),
            optics: Vec4::new(
                ONSET_EFFECTIVE_RADIUS_M,
                DENSE_EFFECTIVE_RADIUS_M,
                ASSUMED_AIR_DENSITY_KG_M3,
                density_scale(field),
            ),
            scattering: Vec4::new(
                PHYSICAL_ATMOSPHERE_EXPOSURE,
                SIMSAT_GROUND_BOUNCE_ALBEDO,
                MULTIPLE_SCATTER_OCTAVE_WEIGHTS[0],
                MULTIPLE_SCATTER_OCTAVE_WEIGHTS[1],
            ),
            sky_log_coef: std::array::from_fn(|index| {
                Vec4::new(
                    sky.log_coef[index][0] as f32,
                    sky.log_coef[index][1] as f32,
                    sky.log_coef[index][2] as f32,
                    0.0,
                )
            }),
        }
    }

    pub fn threshold(&self) -> f32 {
        self.control.x
    }
}

/// Physical Beer–Lambert extinction per kilometre for the display-only
/// combined qc+qi microphysics assumption. Zero-at-onset prevents a long path
/// of threshold-equal words becoming an opaque box; above onset the formula is
/// `3 Qext rho_air (q-onset) / (4 rho_condensed r_eff)`.
pub fn condensate_extinction_per_km(q: f32, onset: f32, dense_cutoff: f32) -> f32 {
    let dense_weight = ((q - onset) / (dense_cutoff - onset).max(1.0e-8)).clamp(0.0, 1.0);
    let dense_weight = dense_weight * dense_weight * (3.0 - 2.0 * dense_weight);
    let effective_radius = ONSET_EFFECTIVE_RADIUS_M.lerp(DENSE_EFFECTIVE_RADIUS_M, dense_weight);
    let occupancy = ONSET_GRID_OCCUPANCY.lerp(1.0, dense_weight);
    let excess = (q - onset).max(0.0);
    1_000.0 * 3.0 * EXTINCTION_EFFICIENCY * ASSUMED_AIR_DENSITY_KG_M3 * excess * occupancy
        / (4.0 * CONDENSED_WATER_DENSITY_KG_M3 * effective_radius)
}

/// Display-only physical concentration scale. The uploaded R32 words and
/// fixed 0.1 g kg-1 onset are unchanged. Each WSM6 field keeps a stable scale
/// across exact snapshots so time comparisons do not auto-normalize.
pub fn density_scale(field: &VolumeField) -> f32 {
    match field.id.as_str() {
        "cloud_condensate" => 1.0e-3,
        "qc" => 8.0e-4,
        "qi" => 2.0e-4,
        "condensate" | "qr" | "qg" => 4.0e-3,
        "qs" => 2.0e-3,
        "qv" => 1.5e-2,
        _ => field.finite_max.max(1.0e-3),
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct Gm1VolumeMaterial {
    #[storage(0, read_only)]
    pub density: Handle<ShaderBuffer>,
    #[uniform(1)]
    pub uniform: VolumeUniform,
    /// Sealed coordinate words for the admitted terrain-following path. The
    /// ordinary level-space path binds a one-word sentinel and never reads it.
    #[storage(2, read_only)]
    pub coordinates: Handle<ShaderBuffer>,
}

impl Material for Gm1VolumeMaterial {
    fn fragment_shader() -> ShaderRef {
        #[cfg(debug_assertions)]
        {
            VOLUME_SHADER.into()
        }
        #[cfg(not(debug_assertions))]
        {
            AssetPath::from_path_buf(embedded_path!("gm1_volume.wgsl"))
                .with_source("embedded")
                .into()
        }
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None::<Face>;
        Ok(())
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }
}

pub struct Gm1VolumePlugin;

impl Plugin for Gm1VolumePlugin {
    fn build(&self, app: &mut App) {
        #[cfg(not(debug_assertions))]
        embedded_asset!(app, "gm1_volume.wgsl");
        app.add_plugins((
            MaterialPlugin::<Gm1VolumeMaterial>::default(),
            MaterialPlugin::<DevProofMaterial>::default(),
        ));
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct DevProofMaterial {
    #[storage(0, read_only)]
    pub density: Handle<ShaderBuffer>,
    #[uniform(1)]
    pub uniform: VolumeUniform,
}

impl Material for DevProofMaterial {
    fn fragment_shader() -> ShaderRef {
        DEV_PROOF_SHADER.into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

pub fn storage_from_field(field: &VolumeField) -> Result<ShaderBuffer, String> {
    let [nz, ny, nx] = field.shape;
    if nx > u32::MAX as usize || ny > u32::MAX as usize || nz > u32::MAX as usize {
        return Err("volume dimensions exceed the storage index ABI".to_owned());
    }
    if field.values.len() != nx * ny * nz {
        return Err("volume values disagree with [z,y,x] shape".to_owned());
    }
    let mut bytes = Vec::with_capacity(field.values.len() * 4);
    for value in &field.values {
        if !value.is_finite() {
            return Err("volume contains a non-finite value".to_owned());
        }
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(ShaderBuffer::new(&bytes, RenderAssetUsages::RENDER_WORLD))
}

/// Upload the already-admitted exact words of one production-v2 resident
/// field without changing its stagger, value range, or representation. This
/// is raw renderer residency only: it does not assert that a wind, pressure,
/// terrain, or coordinate field is an opacity/density field.
pub fn storage_from_production_resident(
    field: &ProductionResidentField,
) -> Result<ShaderBuffer, String> {
    if field.descriptor.admission != crate::geofield::GeoFieldAdmission::Available {
        return Err("production renderer storage requires an available field".to_owned());
    }
    let words = field
        .descriptor
        .shape
        .iter()
        .try_fold(1usize, |total, extent| total.checked_mul(*extent))
        .ok_or_else(|| "production renderer shape overflows usize".to_owned())?;
    if field.descriptor.shape.len() < 2
        || field.descriptor.shape.len() > 3
        || field.values.len() != words
    {
        return Err("production renderer values disagree with exact shape".to_owned());
    }
    if f32_sha256(&field.values) != field.descriptor.values_sha256 {
        return Err("production renderer in-memory values changed after admission".to_owned());
    }
    let mut bytes = Vec::with_capacity(field.values.len() * 4);
    for value in field.values.iter() {
        if !value.is_finite() {
            return Err("production renderer values contain a non-finite word".to_owned());
        }
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(ShaderBuffer::new(&bytes, RenderAssetUsages::RENDER_WORLD))
}

pub fn unused_coordinate_storage() -> ShaderBuffer {
    ShaderBuffer::new(&0_u32.to_le_bytes(), RenderAssetUsages::RENDER_WORLD)
}

pub fn visible_threshold(values: &[f32], minimum: f32, maximum: f32) -> f32 {
    let positive = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if positive.is_empty() {
        return minimum.max(0.0);
    }
    // 0.1 g kg-1 is a fixed condensate onset, so separate accepted forecast
    // steps use the same physically interpretable display threshold. Opacity
    // is handled by extinction in the shader rather than hiding weaker cloud.
    1.0e-4_f32.clamp(minimum, maximum)
}

#[cfg(test)]
mod tests {
    use super::{
        ASSUMED_AIR_DENSITY_KG_M3, CONDENSED_WATER_DENSITY_KG_M3, DENSE_EFFECTIVE_RADIUS_M,
        EXTINCTION_EFFICIENCY, MAX_RAY_STEPS, ONSET_EFFECTIVE_RADIUS_M, ONSET_GRID_OCCUPANCY,
        VolumeUniform, condensate_extinction_per_km, density_scale, storage_from_field,
        storage_from_production_resident, visible_threshold,
    };
    use crate::{
        geofield::{GeoFieldAdmission, GeoFieldDescriptor, GeoFieldGeometry},
        input::{VolumeField, f32_sha256},
        production_playback::{ProductionResidencyKind, ProductionResidentField},
    };
    use bevy::render::render_resource::ShaderType;
    use std::sync::Arc;

    #[test]
    fn volume_uniform_layout_matches_wgsl_storage_abi() {
        // Seven vec4<f32> fields, one vec4<u32>, then array<vec4<f32>, 16>. Both Rust's
        // host type and Bevy/encase must retain the WGSL-required 16-byte
        // alignment and 400-byte binding extent.
        assert_eq!(std::mem::align_of::<VolumeUniform>(), 16);
        assert_eq!(std::mem::size_of::<VolumeUniform>(), 400);
        assert_eq!(VolumeUniform::min_size().get(), 400);
    }

    #[test]
    fn volume_wgsl_parses_with_exact_mass_center_mapping() {
        let source = include_str!("gm1_volume.wgsl");
        let function = source
            .split_once("fn world_to_texture")
            .and_then(|(_, suffix)| suffix.split_once("struct DensitySample"))
            .map(|(body, _)| format!("fn world_to_texture{body}"))
            .expect("extract world_to_texture from GM1 volume WGSL");
        let probe = format!(
            "struct ProbeMaterial {{ world_min: vec4<f32>, world_size: vec4<f32>, grid_shape: vec4<u32>, }}\n\
             @group(0) @binding(0) var<uniform> material: ProbeMaterial;\n\
             fn admitted_vertical_coordinate(world_y: f32, horizontal: vec2<f32>) -> f32 {{ return world_y + horizontal.x * 0.0; }}\n\
             {function}"
        );
        naga::front::wgsl::parse_str(&probe).expect("parse exact mass-center mapping WGSL");
        assert!(source.contains("world_min.x, material.world_min.z) + 0.5 * cell_size"));
        assert!(source.contains("vec3<f32>(horizontal.x, horizontal.y, vertical)"));
    }

    #[test]
    fn exact_r32_volume_preserves_c_order_words() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let field = VolumeField {
            id: "real".to_owned(),
            label: "real".to_owned(),
            units: "1".to_owned(),
            meaning: "test operator fixture".to_owned(),
            source: "controlled analytic operator fixture".to_owned(),
            shape: [2, 2, 2],
            values_sha256: f32_sha256(&values),
            finite_min: 0.0,
            finite_max: 7.0,
            values,
        };
        let storage = storage_from_field(&field).unwrap();
        let bytes = storage.data.as_ref().unwrap();
        assert_eq!(bytes.len(), 8 * 4);
        for (word, value) in bytes.chunks_exact(4).zip(field.values) {
            assert_eq!(
                u32::from_le_bytes(word.try_into().unwrap()),
                value.to_bits()
            );
        }
        const { assert!(MAX_RAY_STEPS >= 128) };
    }

    #[test]
    fn production_residency_upload_preserves_words_and_refuses_mutation() {
        let values = vec![-2.0_f32, 0.0, 3.5, 9.0];
        let mut field = ProductionResidentField {
            descriptor: GeoFieldDescriptor {
                id: "u_staggered_ms".to_owned(),
                label: "Eastward wind (native x face)".to_owned(),
                units: "m s-1".to_owned(),
                meaning: "exact test words".to_owned(),
                geometry: GeoFieldGeometry::VolumeZyx,
                shape: vec![1, 2, 2],
                values_sha256: f32_sha256(&values),
                admission: GeoFieldAdmission::Available,
                admission_reason: "strict test admission".to_owned(),
                provenance: "exact test carrier".to_owned(),
            },
            stagger: "x-face C-grid".to_owned(),
            residency: ProductionResidencyKind::XFaceWindVolume,
            values: Arc::new(values.clone()),
        };
        let storage = storage_from_production_resident(&field).unwrap();
        let bytes = storage.data.as_ref().unwrap();
        for (word, value) in bytes.chunks_exact(4).zip(&values) {
            assert_eq!(
                f32::from_le_bytes(word.try_into().unwrap()).to_bits(),
                value.to_bits()
            );
        }
        Arc::make_mut(&mut field.values)[0] = 7.0;
        assert!(
            storage_from_production_resident(&field)
                .unwrap_err()
                .contains("changed after admission")
        );
    }

    #[test]
    fn threshold_uses_only_positive_admitted_words() {
        let values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09];
        let threshold = visible_threshold(&values, 0.0, 0.09);
        assert_eq!(threshold.to_bits(), 1.0e-4_f32.to_bits());
    }

    #[test]
    fn resolved_cloud_uses_fixed_physical_transfer_scale() {
        let field = VolumeField {
            id: "cloud_condensate".to_owned(),
            label: "resolved cloud condensate".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "test operator fixture".to_owned(),
            source: "controlled analytic operator fixture".to_owned(),
            shape: [1, 1, 1],
            values_sha256: String::new(),
            finite_min: 0.0,
            finite_max: 0.004,
            values: vec![0.001],
        };
        assert_eq!(density_scale(&field).to_bits(), 1.0e-3_f32.to_bits());
    }

    #[test]
    fn physical_extinction_is_zero_at_onset_and_dense_cloud_is_optically_thick() {
        let onset = 1.0e-4;
        let dense = 2.0e-3;
        assert_eq!(condensate_extinction_per_km(onset, onset, dense), 0.0);
        let onset_haze = condensate_extinction_per_km(2.0e-4, onset, dense);
        let dense_core = condensate_extinction_per_km(dense, onset, dense);
        assert!(onset_haze > 0.0 && onset_haze < dense_core);
        assert!(dense_core * 0.05 > 1.0, "50 m dense optical depth");
        assert_eq!(ONSET_EFFECTIVE_RADIUS_M.to_bits(), 60.0e-6_f32.to_bits());
        assert_eq!(DENSE_EFFECTIVE_RADIUS_M.to_bits(), 12.0e-6_f32.to_bits());
        assert_eq!(ASSUMED_AIR_DENSITY_KG_M3.to_bits(), 0.8_f32.to_bits());
        assert_eq!(EXTINCTION_EFFICIENCY.to_bits(), 2.0_f32.to_bits());
        assert_eq!(ONSET_GRID_OCCUPANCY.to_bits(), 0.025_f32.to_bits());
        assert_eq!(
            CONDENSED_WATER_DENSITY_KG_M3.to_bits(),
            1_000.0_f32.to_bits()
        );
    }
}
