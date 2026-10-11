// Material A/B for the sealed SimSat physical sky versus its FP32 L2
// SphericalHarmonicExponentials projection. Cloud scattering is separate.
// Upstream SHE commit 3c14ddd5709705eebe00ec1305c891bc5f7d038b (MIT).

#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}

struct SheSkyMaterialUniform {
    control: vec4<f32>,
    albedo: vec4<f32>,
    sun_direction: vec4<f32>,
    sky_amplitude: vec4<f32>,
    sky_log_coef: array<vec4<f32>, 16>,
    diffuse_coefficients: array<vec4<f32>, 9>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: SheSkyMaterialUniform;

fn log_sh3_sum(direction: vec3<f32>) -> vec3<f32> {
    let n = normalize(direction);
    let x = n.x;
    let y = n.y;
    let z = n.z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    var value = material.sky_log_coef[0].rgb * 0.282094791773878;
    value += material.sky_log_coef[1].rgb * (0.488602511902920 * y);
    value += material.sky_log_coef[2].rgb * (0.488602511902920 * z);
    value += material.sky_log_coef[3].rgb * (0.488602511902920 * x);
    value += material.sky_log_coef[4].rgb * (1.092548430592079 * x * y);
    value += material.sky_log_coef[5].rgb * (1.092548430592079 * y * z);
    value += material.sky_log_coef[6].rgb * (0.315391565252520 * (3.0 * z2 - 1.0));
    value += material.sky_log_coef[7].rgb * (1.092548430592079 * x * z);
    value += material.sky_log_coef[8].rgb * (0.546274215296040 * (x2 - y2));
    value += material.sky_log_coef[9].rgb * (0.590043589926644 * y * (3.0 * x2 - y2));
    value += material.sky_log_coef[10].rgb * (2.890611442640554 * x * y * z);
    value += material.sky_log_coef[11].rgb * (0.457045799464466 * y * (5.0 * z2 - 1.0));
    value += material.sky_log_coef[12].rgb * (0.373176332590115 * z * (5.0 * z2 - 3.0));
    value += material.sky_log_coef[13].rgb * (0.457045799464466 * x * (5.0 * z2 - 1.0));
    value += material.sky_log_coef[14].rgb * (1.445305721320277 * z * (x2 - y2));
    value += material.sky_log_coef[15].rgb * (0.590043589926644 * x * (x2 - 3.0 * y2));
    return value;
}

fn sun_frame_direction(direction: vec3<f32>) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let horizontal = material.sun_direction.xyz - up * dot(material.sun_direction.xyz, up);
    let x_axis = normalize(horizontal);
    let y_axis = cross(up, x_axis);
    return vec3<f32>(dot(direction, x_axis), dot(direction, y_axis), dot(direction, up));
}

fn simsat_reference(normal: vec3<f32>) -> vec3<f32> {
    let log_value = clamp(log_sh3_sum(sun_frame_direction(normal)), vec3<f32>(-80.0), vec3<f32>(80.0));
    return material.sky_amplitude.rgb * exp(log_value);
}

// Exact FP32 basis/order from upstream CommonSH.hlsli.
fn she_diffuse(normal_world: vec3<f32>) -> vec3<f32> {
    let normal = normalize(vec3<f32>(normal_world.x, normal_world.z, normal_world.y));
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;
    let basis = array<f32, 9>(
        0.488603 * x,
        0.488603 * z,
        0.488603 * y,
        0.546274 * (x * x - y * y),
        1.092548 * x * z,
        0.946176 * z * z - 0.315392,
        1.092548 * y * z,
        1.092548 * x * y,
        0.282095,
    );
    var irradiance = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i += 1u) {
        irradiance += basis[i] * material.diffuse_coefficients[i].rgb;
    }
    return irradiance;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let reference = simsat_reference(normal);
    let projected = she_diffuse(normal);
    let irradiance = select(reference, projected, material.control.x > 0.5);
    let view_direction = normalize(view.world_position - in.world_position.xyz);
    let rim = pow(1.0 - max(dot(normal, view_direction), 0.0), 3.0);
    let color = max(irradiance, vec3<f32>(0.0)) * material.albedo.rgb * material.control.y
        + rim * vec3<f32>(0.025, 0.035, 0.05);
    return vec4<f32>(color, 1.0);
}
