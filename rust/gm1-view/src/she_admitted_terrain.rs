//! Exact GM1 terrain rendered with the source-specific admitted glossy SHE fit.

use crate::{
    lighting::{PositiveSky, T300_DIRECT_IRRADIANCE_F32_BITS, T300_SUN_ENU_F32_BITS},
    she::SheSpecularCoefficients,
};
#[cfg(not(debug_assertions))]
use bevy::asset::{AssetPath, embedded_asset, embedded_path};
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, MeshVertexBufferLayoutRef},
    pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin},
    prelude::*,
    reflect::TypePath,
    render::{
        render_resource::{
            AsBindGroup, Face, PrimitiveTopology, RenderPipelineDescriptor, ShaderType,
            SpecializedMeshPipelineError,
        },
        storage::ShaderBuffer,
    },
    shader::ShaderRef,
};
use sha2::{Digest, Sha256};

pub const ADMITTED_TERRAIN_SHADER: &str = "she_admitted_terrain.wgsl";
pub const ADMITTED_TERRAIN_NX: usize = 300;
pub const ADMITTED_TERRAIN_NY: usize = 300;
pub const ADMITTED_TERRAIN_DX_M: f32 = 50.0;
pub const ADMITTED_TERRAIN_DY_M: f32 = 50.0;
pub const ADMITTED_VERTICAL_GRID_SHA256: &str =
    "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56";
pub const ADMITTED_SIGMA_SHA256: &str =
    "e0c800cd95ac68f526efef46de91db5463784e91860f52ea90ad76c9e18037c3";
pub const ADMITTED_MODEL_TOP_M: f32 = 15_000.0;
pub const ADMITTED_TERRAIN_SHA256: &str =
    "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91";
pub const ADMITTED_SPECULAR_SHA256: &str =
    "1fe4c6059a43464af304adc0a8af41ec1791104e5d6da6b7af254a12688404a2";
pub const ADMITTED_RECEIPT_SHA256: &str =
    "b6a5fbf4bf784ff6494562092a76b2909d925121e86fca15907793bc78bfc9ea";
pub const BELFAST_ALPHA32_REFUSAL_RECEIPT_SHA256: &str =
    "f893e7768e1f90b37813eeabbdb86b909f9f7d9a73a0488d585bc311a9f7e463";
pub const ADMITTED_PROFILE: &str =
    "GM1 source-specific upstream-math 32-alpha quality extension; not upstream-default/global";
pub const ADMITTED_MICROFACET_ALPHA: f32 = 0.22;
pub const ADMITTED_PERCEPTUAL_ROUGHNESS: f32 = 0.469_041_6;

const TERRAIN_ASSET: &[u8; ADMITTED_TERRAIN_NX * ADMITTED_TERRAIN_NY * 4] =
    include_bytes!("../assets/gm1-t300-child-terrain-msl.f32le");
const SPECULAR_ASSET: &[u8; 33 * 3 * 4] =
    include_bytes!("../assets/gm1-she-gm1-alpha32-fp32.f32le");

/// Sealed CM1 r21.1 scalar-height profile, including the two halo values.
/// Physical mass level `k` uses `ADMITTED_SIGMA_M[k + 1]` before exact
/// terrain composition. The little-endian binary32 hash is
/// [`ADMITTED_SIGMA_SHA256`].
#[allow(clippy::excessive_precision)]
pub const ADMITTED_SIGMA_M: [f32; 77] = [
    -10.0,
    10.0,
    30.0,
    50.0,
    70.0,
    90.0,
    110.0,
    130.0,
    150.0,
    170.0,
    190.0,
    210.0,
    233.846_16,
    265.384_6,
    304.615_4,
    351.538_45,
    406.153_87,
    468.461_55,
    538.461_55,
    616.153_8,
    701.538_45,
    794.615_36,
    895.384_64,
    1_003.846_2,
    1_120.0,
    1_243.846_2,
    1_375.384_6,
    1_514.615_4,
    1_661.538_6,
    1_816.153_9,
    1_978.461_4,
    2_148.461_4,
    2_326.153_8,
    2_511.538_3,
    2_704.615_2,
    2_905.384_5,
    3_113.846_2,
    3_330.0,
    3_553.846,
    3_785.384_3,
    4_024.615_2,
    4_271.538,
    4_526.153_3,
    4_788.461_4,
    5_058.461_4,
    5_336.153_3,
    5_621.538,
    5_914.615_2,
    6_215.385,
    6_523.845_7,
    6_839.999_5,
    7_159.999_5,
    7_479.999_5,
    7_799.999_5,
    8_120.0,
    8_440.0,
    8_760.0,
    9_080.0,
    9_400.0,
    9_720.0,
    10_040.0,
    10_360.0,
    10_680.0,
    11_000.0,
    11_320.0,
    11_640.0,
    11_960.0,
    12_280.0,
    12_600.0,
    12_920.0,
    13_240.0,
    13_560.0,
    13_880.0,
    14_200.0,
    14_520.0,
    14_840.0,
    15_160.0,
];

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct AdmittedTerrainUniform {
    /// x: exact microfacet alpha; y: display exposure; z: direct-sun toggle.
    pub control: Vec4,
    /// xyz: explicitly authored damp-soil albedo; w: dielectric F0.
    pub surface: Vec4,
    pub sun_direction: Vec4,
    /// RGB direct-normal irradiance W m-2; w unused.
    pub direct_irradiance: Vec4,
    pub sky_amplitude: Vec4,
    pub sky_log_coefficients: [Vec4; 16],
    pub specular_coefficients: [Vec4; 33],
}

impl AdmittedTerrainUniform {
    fn t300() -> Result<Self, String> {
        let sky = PositiveSky::t300_physical();
        let coefficients = load_admitted_coefficients()?;
        let sun_enu = T300_SUN_ENU_F32_BITS.map(f32::from_bits);
        let direct = T300_DIRECT_IRRADIANCE_F32_BITS.map(f32::from_bits);
        Ok(Self {
            control: Vec4::new(ADMITTED_MICROFACET_ALPHA, 0.030, 1.0, 0.0),
            surface: Vec4::new(0.052, 0.042, 0.028, 0.04),
            sun_direction: Vec3::new(sun_enu[0], sun_enu[2], sun_enu[1])
                .normalize()
                .extend(0.0),
            direct_irradiance: Vec3::from_array(direct).extend(0.0),
            sky_amplitude: Vec3::new(
                sky.amplitude[0] as f32,
                sky.amplitude[1] as f32,
                sky.amplitude[2] as f32,
            )
            .extend(0.0),
            sky_log_coefficients: std::array::from_fn(|index| {
                Vec3::new(
                    sky.log_coef[index][0] as f32,
                    sky.log_coef[index][1] as f32,
                    sky.log_coef[index][2] as f32,
                )
                .extend(0.0)
            }),
            specular_coefficients: std::array::from_fn(|index| coefficients.0[index].extend(0.0)),
        })
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct AdmittedTerrainMaterial {
    #[uniform(0)]
    pub uniform: AdmittedTerrainUniform,
}

impl AdmittedTerrainMaterial {
    pub fn t300() -> Result<Self, String> {
        Ok(Self {
            uniform: AdmittedTerrainUniform::t300()?,
        })
    }
}

impl Material for AdmittedTerrainMaterial {
    fn fragment_shader() -> ShaderRef {
        #[cfg(debug_assertions)]
        {
            ADMITTED_TERRAIN_SHADER.into()
        }
        #[cfg(not(debug_assertions))]
        {
            AssetPath::from_path_buf(embedded_path!("she_admitted_terrain.wgsl"))
                .with_source("embedded")
                .into()
        }
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // The chase camera is only 75 m above the exact terrain. Disabling
        // face culling makes the source geometry robust at grazing incidence
        // without changing a vertex or normal.
        descriptor.primitive.cull_mode = None::<Face>;
        Ok(())
    }

    fn alpha_mode(&self) -> AlphaMode {
        // The exact terrain is a solid surface. Keeping it in the opaque pass
        // gives the transparent volume a correct depth boundary and avoids
        // centroid-sorted terrain compositing over clouds at chase altitude.
        AlphaMode::Opaque
    }
}

pub struct AdmittedTerrainMaterialPlugin;

impl Plugin for AdmittedTerrainMaterialPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(not(debug_assertions))]
        embedded_asset!(app, "she_admitted_terrain.wgsl");
        app.add_plugins(MaterialPlugin::<AdmittedTerrainMaterial>::default());
    }
}

pub fn load_admitted_coefficients() -> Result<SheSpecularCoefficients, String> {
    if sha256_bytes(SPECULAR_ASSET) != ADMITTED_SPECULAR_SHA256 {
        return Err("embedded admitted SHE coefficient identity changed".to_owned());
    }
    let mut words = SPECULAR_ASSET
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four coefficient bytes")));
    let coefficients = SheSpecularCoefficients(std::array::from_fn(|_| {
        Vec3::new(
            words.next().expect("33 RGB admitted coefficients"),
            words.next().expect("33 RGB admitted coefficients"),
            words.next().expect("33 RGB admitted coefficients"),
        )
    }));
    if words.next().is_some()
        || coefficients
            .0
            .iter()
            .any(|coefficient| !coefficient.is_finite())
    {
        return Err("embedded admitted SHE coefficients are malformed".to_owned());
    }
    Ok(coefficients)
}

pub fn admitted_terrain_values() -> Result<Vec<f32>, String> {
    if sha256_bytes(TERRAIN_ASSET) != ADMITTED_TERRAIN_SHA256 {
        return Err("embedded exact GM1 terrain identity changed".to_owned());
    }
    let terrain = TERRAIN_ASSET
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four terrain bytes")))
        .collect::<Vec<_>>();
    if terrain.len() != ADMITTED_TERRAIN_NX * ADMITTED_TERRAIN_NY
        || terrain.iter().any(|value| !value.is_finite())
    {
        return Err("embedded exact GM1 terrain has invalid shape or words".to_owned());
    }
    Ok(terrain)
}

/// Exact terrain-following physical height for one admitted scalar level.
pub fn admitted_mass_height_km(k: usize, terrain_msl_m: f32) -> Result<f32, String> {
    let sigma_m = *ADMITTED_SIGMA_M
        .get(k + 1)
        .ok_or_else(|| "scalar level is outside the sealed 75-level grid".to_owned())?;
    if !terrain_msl_m.is_finite() {
        return Err("terrain height is non-finite".to_owned());
    }
    Ok(
        (terrain_msl_m + sigma_m * (ADMITTED_MODEL_TOP_M - terrain_msl_m) / ADMITTED_MODEL_TOP_M)
            / 1_000.0,
    )
}

/// CPU mirror of the shader's terrain-aware inverse vertical transform.
/// Exact admitted mass heights map to `k / 74` for every terrain column.
pub fn admitted_vertical_fraction(world_y_km: f32, terrain_msl_m: f32) -> Result<f32, String> {
    if !world_y_km.is_finite() || !terrain_msl_m.is_finite() {
        return Err("vertical inverse received a non-finite coordinate".to_owned());
    }
    let sigma_m = (world_y_km * 1_000.0 - terrain_msl_m) * ADMITTED_MODEL_TOP_M
        / (ADMITTED_MODEL_TOP_M - terrain_msl_m);
    let first = ADMITTED_SIGMA_M[1];
    if sigma_m < first {
        return Ok((sigma_m - first) / (ADMITTED_SIGMA_M[2] - first) / 74.0);
    }
    for profile in 1..75 {
        let lower = ADMITTED_SIGMA_M[profile];
        let upper = ADMITTED_SIGMA_M[profile + 1];
        if sigma_m <= upper {
            return Ok((profile as f32 - 1.0 + (sigma_m - lower) / (upper - lower)) / 74.0);
        }
    }
    let last = ADMITTED_SIGMA_M[75];
    Ok(1.0 + (sigma_m - last) / (last - ADMITTED_SIGMA_M[74]) / 74.0)
}

/// Shader storage layout used only by the admitted physical-coordinate path:
/// 77 exact scalar sigma words followed by the exact 300x300 terrain carrier.
pub fn admitted_volume_coordinates() -> Result<ShaderBuffer, String> {
    let terrain = admitted_terrain_values()?;
    let mut bytes = Vec::with_capacity((ADMITTED_SIGMA_M.len() + terrain.len()) * 4);
    for value in ADMITTED_SIGMA_M {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    if sha256_bytes(&bytes) != ADMITTED_SIGMA_SHA256 {
        return Err("embedded exact scalar sigma profile identity changed".to_owned());
    }
    bytes.extend_from_slice(TERRAIN_ASSET);
    Ok(ShaderBuffer::new(&bytes, RenderAssetUsages::RENDER_WORLD))
}

pub fn admitted_terrain_mesh() -> Result<Mesh, String> {
    let terrain = admitted_terrain_values()?;
    let mut positions = Vec::with_capacity(terrain.len());
    let mut normals = Vec::with_capacity(terrain.len());
    let mut uvs = Vec::with_capacity(terrain.len());
    for j in 0..ADMITTED_TERRAIN_NY {
        for i in 0..ADMITTED_TERRAIN_NX {
            let i0 = i.saturating_sub(1);
            let i1 = (i + 1).min(ADMITTED_TERRAIN_NX - 1);
            let j0 = j.saturating_sub(1);
            let j1 = (j + 1).min(ADMITTED_TERRAIN_NY - 1);
            let dzdx = (terrain[j * ADMITTED_TERRAIN_NX + i1]
                - terrain[j * ADMITTED_TERRAIN_NX + i0])
                / ((i1 - i0) as f32 * ADMITTED_TERRAIN_DX_M);
            let dzdy = (terrain[j1 * ADMITTED_TERRAIN_NX + i]
                - terrain[j0 * ADMITTED_TERRAIN_NX + i])
                / ((j1 - j0) as f32 * ADMITTED_TERRAIN_DY_M);
            positions.push([
                -7.5 + (i as f32 + 0.5) * ADMITTED_TERRAIN_DX_M / 1000.0,
                terrain[j * ADMITTED_TERRAIN_NX + i] / 1000.0,
                -7.5 + (j as f32 + 0.5) * ADMITTED_TERRAIN_DY_M / 1000.0,
            ]);
            normals.push(Vec3::new(-dzdx, 1.0, -dzdy).normalize().to_array());
            uvs.push([
                i as f32 / (ADMITTED_TERRAIN_NX - 1) as f32,
                j as f32 / (ADMITTED_TERRAIN_NY - 1) as f32,
            ]);
        }
    }
    let mut indices = Vec::with_capacity((ADMITTED_TERRAIN_NX - 1) * (ADMITTED_TERRAIN_NY - 1) * 6);
    for j in 0..ADMITTED_TERRAIN_NY - 1 {
        for i in 0..ADMITTED_TERRAIN_NX - 1 {
            let a = (j * ADMITTED_TERRAIN_NX + i) as u32;
            let b = a + 1;
            let c = a + ADMITTED_TERRAIN_NX as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    Ok(mesh)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub const ADMITTED_SOURCE_SUMMARY: &str = concat!(
    "Avelina9X/SphericalHarmonicExponentials @ ",
    "3c14ddd5709705eebe00ec1305c891bc5f7d038b",
    " (MIT); GM1 source-specific alpha32; physical SimSat sky ",
    "0e29c568b6f2a22b6f79f072d991a9db60e86fb0a8584e3dde9e36be3c24d703",
);

#[cfg(test)]
mod tests {
    use super::{
        ADMITTED_MICROFACET_ALPHA, ADMITTED_PERCEPTUAL_ROUGHNESS, ADMITTED_RECEIPT_SHA256,
        ADMITTED_SIGMA_M, ADMITTED_SIGMA_SHA256, ADMITTED_SPECULAR_SHA256, ADMITTED_TERRAIN_NX,
        ADMITTED_TERRAIN_NY, ADMITTED_TERRAIN_SHA256, AdmittedTerrainUniform,
        admitted_mass_height_km, admitted_terrain_mesh, admitted_vertical_fraction,
        admitted_volume_coordinates, load_admitted_coefficients, sha256_bytes,
    };
    use crate::{
        lighting::T300_SKY_ROW_SHA256, she::SheSpecularCoefficients,
        she_glossy::specular_coefficients_sha256,
    };
    use bevy::render::render_resource::ShaderType;

    #[test]
    fn admitted_assets_and_uniform_are_exactly_bound() {
        let coefficients: SheSpecularCoefficients =
            load_admitted_coefficients().expect("admitted coefficients");
        assert_eq!(
            specular_coefficients_sha256(&coefficients),
            ADMITTED_SPECULAR_SHA256
        );
        assert_eq!(ADMITTED_TERRAIN_SHA256.len(), 64);
        assert_eq!(ADMITTED_RECEIPT_SHA256.len(), 64);
        assert_eq!(T300_SKY_ROW_SHA256.len(), 64);
        assert_eq!(ADMITTED_MICROFACET_ALPHA.to_bits(), 0.22_f32.to_bits());
        assert!((ADMITTED_PERCEPTUAL_ROUGHNESS.powi(2) - ADMITTED_MICROFACET_ALPHA).abs() < 1.0e-7);
        assert_eq!(std::mem::align_of::<AdmittedTerrainUniform>(), 16);
        assert_eq!(std::mem::size_of::<AdmittedTerrainUniform>(), 864);
        assert_eq!(AdmittedTerrainUniform::min_size().get(), 864);
    }

    #[test]
    fn exact_terrain_mesh_has_full_source_grid() {
        let mesh = admitted_terrain_mesh().expect("exact terrain mesh");
        assert_eq!(
            mesh.count_vertices(),
            ADMITTED_TERRAIN_NX * ADMITTED_TERRAIN_NY
        );
    }

    #[test]
    fn exact_vertical_profile_and_coordinate_storage_are_sealed() {
        let bytes = ADMITTED_SIGMA_M
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(sha256_bytes(&bytes), ADMITTED_SIGMA_SHA256);
        assert_eq!(
            admitted_mass_height_km(0, 0.0).unwrap().to_bits(),
            0.010_f32.to_bits()
        );
        assert_eq!(
            admitted_mass_height_km(74, 0.0).unwrap().to_bits(),
            14.840_f32.to_bits()
        );
        for terrain_msl_m in [306.059_57_f32, 336.0, 366.403_17] {
            for k in 0..75 {
                let world_y = admitted_mass_height_km(k, terrain_msl_m).unwrap();
                let fraction = admitted_vertical_fraction(world_y, terrain_msl_m).unwrap();
                let expected = k as f32 / 74.0;
                assert!(
                    (fraction - expected).abs() <= 2.0e-6,
                    "terrain={terrain_msl_m} k={k} fraction={fraction} expected={expected}",
                );
            }
        }
        let storage = admitted_volume_coordinates().expect("sealed coordinate storage");
        assert_eq!(
            storage.data.as_ref().unwrap().len(),
            (77 + ADMITTED_TERRAIN_NX * ADMITTED_TERRAIN_NY) * 4
        );
    }
}
