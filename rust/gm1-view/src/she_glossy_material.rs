//! Display-only wet-dielectric comparison for the physical SimSat radiance row.
//!
//! This deliberately labels the bundled 33-term fit as a reduced provisional
//! preview. Production admission still requires the pinned upstream-default
//! 256-collector, 4096-sample, unregularized FP32 compute chain.

use crate::{
    lighting::T300_SUN_ENU_F32_BITS,
    she::SheSpecularCoefficients,
    she_glossy::{SIMSAT_RADIANCE_SHA256, SimSatSkyRadiance, specular_coefficients_sha256},
};
#[cfg(not(debug_assertions))]
use bevy::asset::{AssetPath, embedded_asset, embedded_path};
use bevy::{
    pbr::{Material, MaterialPlugin},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};
use sha2::{Digest, Sha256};

pub const SHE_GLOSSY_MATERIAL_SHADER: &str = "she_glossy_material.wgsl";
pub const PROVISIONAL_SPECULAR_ASSET: &[u8; 396] =
    include_bytes!("../assets/t300-specular-she-provisional-fp32.f32le");
pub const PROVISIONAL_SPECULAR_SHA256: &str =
    "e416c591d8563468dd2ee762b2c343482fbc2d82b48a3d41996da629b7f1d9be";
pub const PROVISIONAL_STATUS: &str =
    "REFUSED_REDUCED_PREVIEW_FIT_ERROR - NOT UPSTREAM DEFAULT / NOT SOURCE-EXACT";
pub const PREVIEW_REFERENCE_GGX_SAMPLES: u32 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum GlossyComparisonMode {
    PhysicalReference = 0,
    ProvisionalShe33 = 1,
    AbsoluteError = 2,
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct SheGlossyMaterialUniform {
    /// x: comparison mode; y: roughness; z: display exposure; w: error gain.
    pub control: Vec4,
    /// xyz: wet-pavement display albedo; w: dielectric F0.
    pub surface: Vec4,
    pub sun_direction: Vec4,
    pub sky_amplitude: Vec4,
    pub sky_log_coefficients: [Vec4; 16],
    pub specular_coefficients: [Vec4; 33],
}

impl SheGlossyMaterialUniform {
    fn t300(mode: GlossyComparisonMode) -> Result<Self, String> {
        let sky = SimSatSkyRadiance::sealed_t300()?;
        let specular = load_provisional_specular()?;
        let sun_enu = T300_SUN_ENU_F32_BITS.map(f32::from_bits);
        Ok(Self {
            control: Vec4::new(mode as u32 as f32, 0.24, 0.055, 2.5),
            surface: Vec4::new(0.028, 0.038, 0.045, 0.04),
            sun_direction: Vec3::new(sun_enu[0], sun_enu[2], sun_enu[1])
                .normalize()
                .extend(0.0),
            sky_amplitude: Vec3::from_array(sky.amplitude).extend(0.0),
            sky_log_coefficients: std::array::from_fn(|index| {
                Vec3::from_array(sky.log_coefficients[index]).extend(0.0)
            }),
            specular_coefficients: std::array::from_fn(|index| specular.0[index].extend(0.0)),
        })
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct SheGlossyMaterial {
    #[uniform(0)]
    pub uniform: SheGlossyMaterialUniform,
}

impl SheGlossyMaterial {
    pub fn t300(mode: GlossyComparisonMode) -> Result<Self, String> {
        Ok(Self {
            uniform: SheGlossyMaterialUniform::t300(mode)?,
        })
    }
}

impl Material for SheGlossyMaterial {
    fn fragment_shader() -> ShaderRef {
        #[cfg(debug_assertions)]
        {
            SHE_GLOSSY_MATERIAL_SHADER.into()
        }
        #[cfg(not(debug_assertions))]
        {
            AssetPath::from_path_buf(embedded_path!("she_glossy_material.wgsl"))
                .with_source("embedded")
                .into()
        }
    }

    fn enable_shadows() -> bool {
        false
    }
}

pub struct SheGlossyMaterialPlugin;

impl Plugin for SheGlossyMaterialPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(not(debug_assertions))]
        embedded_asset!(app, "she_glossy_material.wgsl");
        app.add_plugins(MaterialPlugin::<SheGlossyMaterial>::default());
    }
}

pub fn load_provisional_specular() -> Result<SheSpecularCoefficients, String> {
    let payload_hash = format!("{:x}", Sha256::digest(PROVISIONAL_SPECULAR_ASSET));
    if payload_hash != PROVISIONAL_SPECULAR_SHA256 {
        return Err("provisional 33-term glossy coefficient asset hash changed".to_owned());
    }
    let mut words = PROVISIONAL_SPECULAR_ASSET
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte coefficient word")));
    let coefficients = SheSpecularCoefficients(std::array::from_fn(|_| {
        Vec3::new(
            words.next().expect("33 RGB coefficients"),
            words.next().expect("33 RGB coefficients"),
            words.next().expect("33 RGB coefficients"),
        )
    }));
    if words.next().is_some()
        || coefficients
            .0
            .iter()
            .any(|coefficient| !coefficient.is_finite())
    {
        return Err("provisional glossy coefficient asset is malformed".to_owned());
    }
    if specular_coefficients_sha256(&coefficients) != PROVISIONAL_SPECULAR_SHA256 {
        return Err("decoded glossy coefficients changed raw-word identity".to_owned());
    }
    Ok(coefficients)
}

pub const GLOSSY_RADIANCE_SOURCE_SHA256: &str = SIMSAT_RADIANCE_SHA256;

#[cfg(test)]
mod tests {
    use super::{PROVISIONAL_SPECULAR_SHA256, SheGlossyMaterialUniform, load_provisional_specular};
    use crate::she_glossy::specular_coefficients_sha256;
    use bevy::render::render_resource::ShaderType;

    #[test]
    fn provisional_coefficients_are_raw_word_pinned_and_explicitly_separate() {
        let coefficients = load_provisional_specular().expect("sealed preview coefficients");
        assert_eq!(
            specular_coefficients_sha256(&coefficients),
            PROVISIONAL_SPECULAR_SHA256
        );
    }

    #[test]
    fn glossy_uniform_layout_matches_wgsl() {
        assert_eq!(std::mem::align_of::<SheGlossyMaterialUniform>(), 16);
        assert_eq!(std::mem::size_of::<SheGlossyMaterialUniform>(), 848);
        assert_eq!(SheGlossyMaterialUniform::min_size().get(), 848);
    }
}
