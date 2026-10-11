// Source-specific admitted GM1 glossy SHE terrain material.
// Upstream math: Avelina9X/SphericalHarmonicExponentials 3c14ddd... (MIT).
// The 32-alpha extension is admitted only for the sealed GM1 full-sphere source.

#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}

const PI: f32 = 3.141592653589793;
const DEBUG_SOLID_TERRAIN: bool = false;

struct AdmittedTerrainUniform {
    control: vec4<f32>,
    surface: vec4<f32>,
    sun_direction: vec4<f32>,
    direct_irradiance: vec4<f32>,
    sky_amplitude: vec4<f32>,
    sky_log_coefficients: array<vec4<f32>, 16>,
    specular_coefficients: array<vec4<f32>, 33>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: AdmittedTerrainUniform;

fn sun_frame_direction(direction: vec3<f32>) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let horizontal = material.sun_direction.xyz - up * dot(material.sun_direction.xyz, up);
    let x_axis = normalize(horizontal);
    let y_axis = cross(up, x_axis);
    return vec3<f32>(dot(direction, x_axis), dot(direction, y_axis), dot(direction, up));
}

fn log_sh3_sum(direction: vec3<f32>) -> vec3<f32> {
    let n = normalize(direction);
    let x = n.x;
    let y = n.y;
    let z = n.z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    var value = material.sky_log_coefficients[0].rgb * 0.282094791773878;
    value += material.sky_log_coefficients[1].rgb * (0.488602511902920 * y);
    value += material.sky_log_coefficients[2].rgb * (0.488602511902920 * z);
    value += material.sky_log_coefficients[3].rgb * (0.488602511902920 * x);
    value += material.sky_log_coefficients[4].rgb * (1.092548430592079 * x * y);
    value += material.sky_log_coefficients[5].rgb * (1.092548430592079 * y * z);
    value += material.sky_log_coefficients[6].rgb * (0.315391565252520 * (3.0 * z2 - 1.0));
    value += material.sky_log_coefficients[7].rgb * (1.092548430592079 * x * z);
    value += material.sky_log_coefficients[8].rgb * (0.546274215296040 * (x2 - y2));
    value += material.sky_log_coefficients[9].rgb * (0.590043589926644 * y * (3.0 * x2 - y2));
    value += material.sky_log_coefficients[10].rgb * (2.890611442640554 * x * y * z);
    value += material.sky_log_coefficients[11].rgb * (0.457045799464466 * y * (5.0 * z2 - 1.0));
    value += material.sky_log_coefficients[12].rgb * (0.373176332590115 * z * (5.0 * z2 - 3.0));
    value += material.sky_log_coefficients[13].rgb * (0.457045799464466 * x * (5.0 * z2 - 1.0));
    value += material.sky_log_coefficients[14].rgb * (1.445305721320277 * z * (x2 - y2));
    value += material.sky_log_coefficients[15].rgb * (0.590043589926644 * x * (x2 - 3.0 * y2));
    return value;
}

fn sky_irradiance(normal_world: vec3<f32>) -> vec3<f32> {
    let direction = sun_frame_direction(normal_world);
    let log_value = clamp(log_sh3_sum(direction), vec3<f32>(-80.0), vec3<f32>(80.0));
    return material.sky_amplitude.rgb * exp(log_value);
}

fn compute_sh_basis_l2(normal: vec3<f32>) -> array<f32, 9> {
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    return array<f32, 9>(
        0.488603 * x, 0.488603 * z, 0.488603 * y,
        0.546274 * (x2 - y2), 1.092548 * x * z,
        0.946176 * z2 - 0.315392, 1.092548 * y * z,
        1.092548 * x * y, 0.282095,
    );
}

fn compute_sh_basis_l4(normal: vec3<f32>) -> array<f32, 25> {
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    return array<f32, 25>(
        0.488603*x, 0.488603*z, 0.488603*y, 0.546274*(x2-y2), 1.092548*x*z,
        0.946176*z2-0.315392, 1.092548*y*z, 1.092548*x*y,
        0.590044*x*(x2-3.0*y2), 1.445306*z*(x2-y2), 0.457046*x*(5.0*z2-1.0),
        0.373176*z*(5.0*z2-3.0), 0.457046*y*(5.0*z2-1.0), 2.890611*x*y*z,
        0.590044*y*(3.0*x2-y2), 0.625836*(x2*(x2-3.0*y2)-y2*(3.0*x2-y2)),
        1.770131*x*z*(x2-3.0*y2), 0.473087*(x2-y2)*(7.0*z2-1.0),
        0.669047*x*z*(7.0*z2-3.0), 0.105786*(35.0*z2*z2-30.0*z2+3.0),
        0.669047*y*z*(7.0*z2-3.0), 0.946174*x*y*(7.0*z2-1.0),
        1.770131*y*z*(3.0*x2-y2), 2.503343*x*y*(x2-y2), 0.282095,
    );
}

fn construct_sh_vector_a(normal: vec3<f32>, view_direction: vec3<f32>, alpha: f32) -> array<f32, 33> {
    let reflected = reflect(-view_direction, normal);
    let half_vector = normalize(normal + reflected);
    var p = compute_sh_basis_l4(reflected);
    var q = compute_sh_basis_l2(half_vector);
    let attenuation = exp(vec4<f32>(-1.0, -3.0, -6.0, -10.0) * alpha);
    for (var i = 0u; i < 3u; i += 1u) { p[i] *= attenuation.x; q[i] *= attenuation.x; }
    p[3] *= attenuation.y; q[3] *= attenuation.y;
    for (var i = 4u; i < 8u; i += 1u) { p[i] *= attenuation.y; q[i] *= attenuation.y; }
    for (var i = 8u; i < 15u; i += 1u) { p[i] *= attenuation.z; }
    for (var i = 15u; i < 24u; i += 1u) { p[i] *= attenuation.w; }
    var output: array<f32, 33>;
    for (var i = 0u; i < 24u; i += 1u) { output[i] = p[i]; }
    for (var i = 0u; i < 8u; i += 1u) { output[24u + i] = q[i]; }
    output[32] = p[24];
    return output;
}

fn admitted_environment_specular(
    normal: vec3<f32>,
    view_direction: vec3<f32>,
    alpha: f32,
) -> vec3<f32> {
    let a = construct_sh_vector_a(normal, view_direction, alpha);
    var log_radiance = vec3<f32>(0.0);
    for (var index = 0u; index < 33u; index += 1u) {
        log_radiance += a[index] * material.specular_coefficients[index].rgb;
    }
    return exp(clamp(log_radiance, vec3<f32>(-80.0), vec3<f32>(80.0)));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Same-camera terrain object-ID readback. The host hides the volume and
    // restores this otherwise-unused uniform lane before the combined frame.
    if (material.control.w > 0.5) {
        return vec4<f32>(1.0);
    }
    if (DEBUG_SOLID_TERRAIN) {
        return vec4<f32>(0.95, 0.05, 0.65, 1.0);
    }
    var normal_world = normalize(in.world_normal);
    let view_world = normalize(view.world_position - in.world_position.xyz);
    // The exact height mesh is treated as a two-sided solid at grazing chase
    // angles. Orient only the shading normal toward the view; geometry and
    // source normals remain unchanged.
    if (dot(normal_world, view_world) < 0.0) {
        normal_world = -normal_world;
    }
    let normal = sun_frame_direction(normal_world);
    let view_direction = sun_frame_direction(view_world);
    let alpha = material.control.x;
    let specular = admitted_environment_specular(normal, view_direction, alpha);
    let ndotv = max(dot(normal_world, view_world), 0.0);
    let fresnel = material.surface.w
        + (1.0 - material.surface.w) * pow(1.0 - ndotv, 5.0);
    let sky = sky_irradiance(normal_world);
    let direct = material.direct_irradiance.rgb
        * max(dot(normal_world, material.sun_direction.xyz), 0.0)
        * material.control.z;
    // This is a display-only tint derived solely from the exact terrain_msl
    // vertex height; it does not invent land cover or perturb geometry.
    let elevation_weight = smoothstep(0.300, 0.375, in.world_position.y);
    let exact_elevation_tint = mix(
        vec3<f32>(0.72, 0.82, 0.72),
        vec3<f32>(1.12, 0.98, 0.78),
        elevation_weight,
    );
    let diffuse = (sky + direct) * material.surface.rgb * exact_elevation_tint / PI;
    let color = max(diffuse + specular * fresnel, vec3<f32>(0.0)) * material.control.y;
    // Geometry ownership, rather than an authored opacity ramp, joins the
    // exact 50 m child to the clipped source-backed HRRR exterior. Both meshes
    // remain opaque, and neither source height carrier is changed.
    return vec4<f32>(color, 1.0);
}
