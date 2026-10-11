#import bevy_pbr::forward_io::VertexOutput

struct VolumeUniform {
    world_min: vec4<f32>,
    world_size: vec4<f32>,
    control: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_amplitude: vec4<f32>,
    grid_shape: vec4<u32>,
    optics: vec4<f32>,
    scattering: vec4<f32>,
    sky_log_coef: array<vec4<f32>, 16>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<storage, read> proof_density: array<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> proof: VolumeUniform;

@fragment
fn fragment(_in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(proof.world_size.w * 1000.0, proof.control.x * 0.0, 0.72, 1.0);
}
