// FP32 port of Avelina9X/SphericalHarmonicExponentials CommonSH.hlsli.
// Upstream commit: 3c14ddd5709705eebe00ec1305c891bc5f7d038b (MIT).
// CommonSH.hlsli SHA-256: 2ffeb4fcc001e57ba61b3f825177d70ed6283ebd5b710c39204acc16460aa56c.

struct SheCase {
    normal_alpha: vec4<f32>,
    view_pad: vec4<f32>,
}

struct SheOutput {
    blocks: array<vec4<f32>, 9>,
}

@group(0) @binding(0) var<storage, read> she_cases: array<SheCase>;
@group(0) @binding(1) var<storage, read_write> she_outputs: array<SheOutput>;

fn compute_sh_basis_l2(normal: vec3<f32>) -> array<f32, 9> {
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    return array<f32, 9>(
        0.488603 * x,
        0.488603 * z,
        0.488603 * y,
        0.546274 * (x2 - y2),
        1.092548 * x * z,
        0.946176 * z2 - 0.315392,
        1.092548 * y * z,
        1.092548 * x * y,
        0.282095,
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
        0.488603 * x,
        0.488603 * z,
        0.488603 * y,
        0.546274 * (x2 - y2),
        1.092548 * x * z,
        0.946176 * z2 - 0.315392,
        1.092548 * y * z,
        1.092548 * x * y,
        0.590044 * x * (x2 - 3.0 * y2),
        1.445306 * z * (x2 - y2),
        0.457046 * x * (5.0 * z2 - 1.0),
        0.373176 * z * (5.0 * z2 - 3.0),
        0.457046 * y * (5.0 * z2 - 1.0),
        2.890611 * x * y * z,
        0.590044 * y * (3.0 * x2 - y2),
        0.625836 * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2)),
        1.770131 * x * z * (x2 - 3.0 * y2),
        0.473087 * (x2 - y2) * (7.0 * z2 - 1.0),
        0.669047 * x * z * (7.0 * z2 - 3.0),
        0.105786 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0),
        0.669047 * y * z * (7.0 * z2 - 3.0),
        0.946174 * x * y * (7.0 * z2 - 1.0),
        1.770131 * y * z * (3.0 * x2 - y2),
        2.503343 * x * y * (x2 - y2),
        0.282095,
    );
}

fn construct_sh_vector_a(normal: vec3<f32>, view: vec3<f32>, alpha: f32) -> array<f32, 33> {
    let reflected = reflect(-view, normal);
    let half_vector = normalize(normal + reflected);
    var p = compute_sh_basis_l4(reflected);
    var q = compute_sh_basis_l2(half_vector);
    let attenuation = exp(vec4<f32>(-1.0, -3.0, -6.0, -10.0) * alpha);
    for (var i = 0u; i < 3u; i += 1u) {
        p[i] *= attenuation.x;
        q[i] *= attenuation.x;
    }
    p[3] *= attenuation.y;
    q[3] *= attenuation.y;
    for (var i = 4u; i < 8u; i += 1u) {
        p[i] *= attenuation.y;
        q[i] *= attenuation.y;
    }
    for (var i = 8u; i < 15u; i += 1u) {
        p[i] *= attenuation.z;
    }
    for (var i = 15u; i < 24u; i += 1u) {
        p[i] *= attenuation.w;
    }
    var output: array<f32, 33>;
    for (var i = 0u; i < 24u; i += 1u) {
        output[i] = p[i];
    }
    for (var i = 0u; i < 8u; i += 1u) {
        output[24u + i] = q[i];
    }
    output[32] = p[24];
    return output;
}

fn reconstruct_diffuse(normal: vec3<f32>, coefficients: ptr<function, array<vec3<f32>, 9>>) -> vec3<f32> {
    let basis = compute_sh_basis_l2(normal);
    var result = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i += 1u) {
        result += basis[i] * (*coefficients)[i];
    }
    return result;
}

fn reconstruct_specular(
    normal: vec3<f32>,
    view: vec3<f32>,
    roughness: f32,
    coefficients: ptr<function, array<vec3<f32>, 33>>,
) -> vec3<f32> {
    let a = construct_sh_vector_a(normal, view, roughness * roughness);
    var log_radiance = vec3<f32>(0.0);
    for (var i = 0u; i < 33u; i += 1u) {
        log_radiance += a[i] * (*coefficients)[i];
    }
    return exp(log_radiance);
}

@compute @workgroup_size(64)
fn she_fp32_oracle(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&she_cases)) {
        return;
    }
    let input = she_cases[index];
    let a = construct_sh_vector_a(input.normal_alpha.xyz, input.view_pad.xyz, input.normal_alpha.w);
    var output: SheOutput;
    for (var block = 0u; block < 9u; block += 1u) {
        var packed = vec4<f32>(0.0);
        for (var lane = 0u; lane < 4u; lane += 1u) {
            let coefficient = block * 4u + lane;
            if (coefficient < 33u) {
                packed[lane] = a[coefficient];
            }
        }
        output.blocks[block] = packed;
    }
    she_outputs[index] = output;
}
