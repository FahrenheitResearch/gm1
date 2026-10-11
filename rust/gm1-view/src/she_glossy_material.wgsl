// Physical SimSat clear-sky radiance reference versus a REDUCED PROVISIONAL
// 33-term glossy SHE fit. The fit is not upstream-default/source-exact.
// Pinned upstream: Avelina9X/SphericalHarmonicExponentials 3c14ddd... (MIT).

#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}

struct SheGlossyMaterialUniform {
    control: vec4<f32>,
    surface: vec4<f32>,
    sun_direction: vec4<f32>,
    sky_amplitude: vec4<f32>,
    sky_log_coefficients: array<vec4<f32>, 16>,
    specular_coefficients: array<vec4<f32>, 33>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: SheGlossyMaterialUniform;

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

fn physical_sky_radiance(direction: vec3<f32>) -> vec3<f32> {
    if (direction.z <= 0.0) {
        return vec3<f32>(0.0);
    }
    let log_value = clamp(log_sh3_sum(direction), vec3<f32>(-80.0), vec3<f32>(80.0));
    return material.sky_amplitude.rgb * exp(log_value);
}

fn radical_inverse_vdc(input: u32) -> f32 {
    var bits = input;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn importance_sample_ggx(index: u32, samples: u32, normal: vec3<f32>, alpha: f32) -> vec3<f32> {
    let xi = vec2<f32>(f32(index) / f32(samples), radical_inverse_vdc(index));
    let phi = 6.283185307179586 * xi.x;
    let cosine = min(sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y)), 1.0);
    let sine = sqrt(max(1.0 - cosine * cosine, 0.0));
    let local = vec3<f32>(cos(phi) * sine, sin(phi) * sine, cosine);
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

fn reference_prefilter(normal: vec3<f32>, view_direction: vec3<f32>, alpha: f32) -> vec3<f32> {
    let ndotv = dot(normal, view_direction);
    if (ndotv <= 0.0) {
        return vec3<f32>(0.0);
    }
    const samples = 64u;
    var accumulator = vec3<f32>(0.0);
    for (var index = 0u; index < samples; index += 1u) {
        let half_vector = importance_sample_ggx(index, samples, normal, alpha);
        let vdoth = dot(view_direction, half_vector);
        if (vdoth <= 0.0) {
            continue;
        }
        let light = 2.0 * vdoth * half_vector - view_direction;
        let ndotl = dot(normal, light);
        if (ndotl <= 0.0) {
            continue;
        }
        let ndoth = dot(normal, half_vector);
        let geometry = min(min(2.0 * ndoth * ndotv / vdoth, 2.0 * ndoth * ndotl / vdoth), 1.0);
        let weight = geometry * vdoth / (ndotv * ndoth);
        accumulator += physical_sky_radiance(light) * weight;
    }
    return accumulator / f32(samples);
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

fn provisional_she(normal: vec3<f32>, view_direction: vec3<f32>, alpha: f32) -> vec3<f32> {
    let a = construct_sh_vector_a(normal, view_direction, alpha);
    var log_radiance = vec3<f32>(0.0);
    for (var index = 0u; index < 33u; index += 1u) {
        log_radiance += a[index] * material.specular_coefficients[index].rgb;
    }
    return exp(clamp(log_radiance, vec3<f32>(-80.0), vec3<f32>(80.0)));
}

fn heatmap(value: f32) -> vec3<f32> {
    let x = clamp(value, 0.0, 1.0);
    return clamp(vec3<f32>(1.5*x, 1.5 - abs(3.0*x - 1.5), 1.5*(1.0-x)), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal_world = normalize(in.world_normal);
    let view_world = normalize(view.world_position - in.world_position.xyz);
    let normal = sun_frame_direction(normal_world);
    let view_direction = sun_frame_direction(view_world);
    let alpha = material.control.y * material.control.y;
    let reference = reference_prefilter(normal, view_direction, alpha);
    let she = provisional_she(normal, view_direction, alpha);
    if (material.control.x > 1.5) {
        let luma_error = abs(dot(reference - she, vec3<f32>(0.2126, 0.7152, 0.0722)));
        return vec4<f32>(heatmap(luma_error * material.control.w), 1.0);
    }
    let radiance = select(reference, she, material.control.x > 0.5);
    let ndotv = max(dot(normal_world, view_world), 0.0);
    let fresnel = material.surface.w + (1.0 - material.surface.w) * pow(1.0 - ndotv, 5.0);
    let diffuse = material.surface.rgb * vec3<f32>(0.11, 0.16, 0.22);
    let color = max(radiance, vec3<f32>(0.0)) * fresnel * material.control.z + diffuse;
    return vec4<f32>(color, 1.0);
}
