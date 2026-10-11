//! Source-backed SimSat sky-radiance dome for the admitted cinematic scene.
//!
//! This is the sealed positive log-SH3 *radiance* row, not the diffuse
//! irradiance row used by cloud fill. It changes no meteorological carrier.

use crate::{
    lighting::T300_SUN_ENU_F32_BITS,
    she_glossy::{SIMSAT_RADIANCE_SHA256, SimSatSkyRadiance},
};
#[cfg(not(debug_assertions))]
use bevy::asset::{AssetPath, embedded_asset, embedded_path};
use bevy::{
    mesh::MeshVertexBufferLayoutRef,
    pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin},
    prelude::*,
    reflect::TypePath,
    render::render_resource::{
        AsBindGroup, Face, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    },
    shader::ShaderRef,
};

pub const PHYSICAL_ATMOSPHERE_SHADER: &str = "physical_atmosphere.wgsl";
pub const PHYSICAL_ATMOSPHERE_SOURCE_SHA256: &str = SIMSAT_RADIANCE_SHA256;
/// Existing sealed display exposure used by the physical-radiance reference.
pub const PHYSICAL_ATMOSPHERE_EXPOSURE: f32 = 0.055;

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct PhysicalAtmosphereUniform {
    /// x: display exposure. Remaining lanes are reserved and zero.
    pub control: Vec4,
    pub sun_direction: Vec4,
    pub amplitude: Vec4,
    pub log_coefficients: [Vec4; 16],
}

impl PhysicalAtmosphereUniform {
    fn sealed_t300() -> Result<Self, String> {
        let radiance = SimSatSkyRadiance::sealed_t300()?;
        let sun_enu = T300_SUN_ENU_F32_BITS.map(f32::from_bits);
        Ok(Self {
            control: Vec4::new(PHYSICAL_ATMOSPHERE_EXPOSURE, 0.0, 0.0, 0.0),
            sun_direction: Vec3::new(sun_enu[0], sun_enu[2], sun_enu[1])
                .normalize()
                .extend(0.0),
            amplitude: Vec3::from_array(radiance.amplitude).extend(0.0),
            log_coefficients: std::array::from_fn(|index| {
                Vec3::from_array(radiance.log_coefficients[index]).extend(0.0)
            }),
        })
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct PhysicalAtmosphereMaterial {
    #[uniform(0)]
    pub uniform: PhysicalAtmosphereUniform,
}

impl PhysicalAtmosphereMaterial {
    pub fn sealed_t300() -> Result<Self, String> {
        Ok(Self {
            uniform: PhysicalAtmosphereUniform::sealed_t300()?,
        })
    }
}

impl Material for PhysicalAtmosphereMaterial {
    fn fragment_shader() -> ShaderRef {
        #[cfg(debug_assertions)]
        {
            PHYSICAL_ATMOSPHERE_SHADER.into()
        }
        #[cfg(not(debug_assertions))]
        {
            AssetPath::from_path_buf(embedded_path!("physical_atmosphere.wgsl"))
                .with_source("embedded")
                .into()
        }
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

    fn enable_shadows() -> bool {
        false
    }
}

pub struct PhysicalAtmospherePlugin;

impl Plugin for PhysicalAtmospherePlugin {
    fn build(&self, app: &mut App) {
        #[cfg(not(debug_assertions))]
        embedded_asset!(app, "physical_atmosphere.wgsl");
        app.add_plugins(MaterialPlugin::<PhysicalAtmosphereMaterial>::default());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PHYSICAL_ATMOSPHERE_EXPOSURE, PHYSICAL_ATMOSPHERE_SOURCE_SHA256, PhysicalAtmosphereUniform,
    };
    use crate::she_glossy::SIMSAT_RADIANCE_SHA256;
    use bevy::render::render_resource::ShaderType;

    #[test]
    fn source_radiance_and_uniform_are_pinned() {
        assert_eq!(PHYSICAL_ATMOSPHERE_SOURCE_SHA256, SIMSAT_RADIANCE_SHA256);
        assert_eq!(PHYSICAL_ATMOSPHERE_EXPOSURE.to_bits(), 0.055_f32.to_bits());
        assert_eq!(std::mem::align_of::<PhysicalAtmosphereUniform>(), 16);
        assert_eq!(std::mem::size_of::<PhysicalAtmosphereUniform>(), 304);
        assert_eq!(PhysicalAtmosphereUniform::min_size().get(), 304);
    }
}
