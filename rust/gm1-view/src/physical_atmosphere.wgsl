// Source-backed SimSat positive log-SH3 sky radiance.
// Asset SHA-256 12838043df7c505720b43addde2118aeac6aa550ace0cfb12e03afad09ced2d5.
// This background is separate from cloud-volume scattering and terrain SHE.

#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}

struct PhysicalAtmosphereUniform {
    control: vec4<f32>,
    sun_direction: vec4<f32>,
    amplitude: vec4<f32>,
    log_coefficients: array<vec4<f32>, 16>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: PhysicalAtmosphereUniform;

fn log_sh3_sum(direction: vec3<f32>) -> vec3<f32> {
    let n = normalize(direction);
    let x = n.x;
    let y = n.y;
    let z = n.z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    var value = material.log_coefficients[0].rgb * 0.282094791773878;
    value += material.log_coefficients[1].rgb * (0.488602511902920 * y);
    value += material.log_coefficients[2].rgb * (0.488602511902920 * z);
    value += material.log_coefficients[3].rgb * (0.488602511902920 * x);
    value += material.log_coefficients[4].rgb * (1.092548430592079 * x * y);
    value += material.log_coefficients[5].rgb * (1.092548430592079 * y * z);
    value += material.log_coefficients[6].rgb * (0.315391565252520 * (3.0 * z2 - 1.0));
    value += material.log_coefficients[7].rgb * (1.092548430592079 * x * z);
    value += material.log_coefficients[8].rgb * (0.546274215296040 * (x2 - y2));
    value += material.log_coefficients[9].rgb * (0.590043589926644 * y * (3.0 * x2 - y2));
    value += material.log_coefficients[10].rgb * (2.890611442640554 * x * y * z);
    value += material.log_coefficients[11].rgb * (0.457045799464466 * y * (5.0 * z2 - 1.0));
    value += material.log_coefficients[12].rgb * (0.373176332590115 * z * (5.0 * z2 - 3.0));
    value += material.log_coefficients[13].rgb * (0.457045799464466 * x * (5.0 * z2 - 1.0));
    value += material.log_coefficients[14].rgb * (1.445305721320277 * z * (x2 - y2));
    value += material.log_coefficients[15].rgb * (0.590043589926644 * x * (x2 - 3.0 * y2));
    return value;
}

fn sun_frame_direction(direction: vec3<f32>) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let horizontal = material.sun_direction.xyz - up * dot(material.sun_direction.xyz, up);
    let x_axis = normalize(horizontal);
    let y_axis = cross(up, x_axis);
    return vec3<f32>(dot(direction, x_axis), dot(direction, y_axis), dot(direction, up));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let view_ray_world = normalize(in.world_position.xyz - view.world_position);
    let direction = sun_frame_direction(view_ray_world);
    let log_value = clamp(log_sh3_sum(direction), vec3<f32>(-80.0), vec3<f32>(80.0));
    let radiance = material.amplitude.rgb * exp(log_value);
    return vec4<f32>(max(radiance * material.control.x, vec3<f32>(0.0)), 1.0);
}
