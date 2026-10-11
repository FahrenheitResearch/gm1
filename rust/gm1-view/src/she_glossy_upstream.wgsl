// WGSL port of the upstream-default FP32 glossy Spherical Harmonic
// Exponentials compute chain. Source authority: Avelina9X/
// SphericalHarmonicExponentials commit 3c14ddd5709705eebe00ec1305c891bc5f7d038b (MIT).
// Collector: 256 x 256 N/V, four alpha levels [0.2, 1.0], 4096 GGX
// samples. Solver: unregularized FP32 Cholesky. The environment is the sealed
// SimSat physical positive log-SH3 sky-radiance row; below-horizon sky is zero.

const COLLECTOR_RESOLUTION: u32 = 256u;
const ALPHA_LEVELS: u32 = 4u;
const GGX_THREADS: u32 = 1024u;
const GGX_SAMPLES_PER_THREAD: u32 = 4u;
const GGX_SAMPLES: u32 = GGX_THREADS * GGX_SAMPLES_PER_THREAD;
const MATRIX_SIZE: u32 = 33u;
const MATRIX_COLUMNS: u32 = 36u;
const MATRIX_WORDS: u32 = MATRIX_SIZE * MATRIX_COLUMNS;
const MATRIX_VEC4S: u32 = MATRIX_WORDS / 4u;
const ACCUMULATOR_BLOCKS: u32 = COLLECTOR_RESOLUTION * 16u;
const PI: f32 = 3.141592654;
const TAU: f32 = 6.283185307;

struct CollectorParams {
    y_offset: u32,
    environment_mode: u32,
    environment_width: u32,
    environment_height: u32,
}

@group(0) @binding(0) var<uniform> collector_params: CollectorParams;
// 0..15 are RGB log-SH3 coefficients; 16 is RGB amplitude.
@group(0) @binding(1) var<storage, read> physical_sky: array<vec4<f32>, 17>;
@group(0) @binding(2) var<storage, read_write> collector: array<vec4<f32>>;
// Optional full-sphere environment. Mode 1 uses word zero as a constant
// operator control; mode 2 uses row-major equirectangular HDR pixels.
@group(0) @binding(3) var<storage, read> environment_pixels: array<vec4<f32>>;

var<workgroup> collector_shared: array<vec3<f32>, 1024>;

fn radical_inverse_vdc(input: u32) -> f32 {
    var bits = input;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn fibonacci_sphere(index: u32, count: u32) -> vec3<f32> {
    let phi = PI * (3.0 - sqrt(5.0));
    let y = 1.0 - 2.0 * f32(index) / (f32(count) - 1.0);
    let radius = sqrt(max(1.0 - clamp(y * y, 0.0, 1.0), 0.0));
    let theta = phi * f32(index);
    return vec3<f32>(radius * cos(theta), y, radius * sin(theta));
}

fn importance_sample_ggx(sample_index: u32, normal: vec3<f32>, alpha: f32) -> vec3<f32> {
    let xi = vec2<f32>(f32(sample_index) / f32(GGX_SAMPLES), radical_inverse_vdc(sample_index));
    let phi = TAU * xi.x;
    let cosine = min(1.0, sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y)));
    let sine = sqrt(max(1.0 - cosine * cosine, 0.0));
    let local = vec3<f32>(cos(phi) * sine, sin(phi) * sine, cosine);
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

fn log_sh3_sum(direction: vec3<f32>) -> vec3<f32> {
    let n = normalize(direction);
    let x = n.x;
    let y = n.y;
    let z = n.z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    var value = physical_sky[0].rgb * 0.282094791773878;
    value += physical_sky[1].rgb * (0.488602511902920 * y);
    value += physical_sky[2].rgb * (0.488602511902920 * z);
    value += physical_sky[3].rgb * (0.488602511902920 * x);
    value += physical_sky[4].rgb * (1.092548430592079 * x * y);
    value += physical_sky[5].rgb * (1.092548430592079 * y * z);
    value += physical_sky[6].rgb * (0.315391565252520 * (3.0 * z2 - 1.0));
    value += physical_sky[7].rgb * (1.092548430592079 * x * z);
    value += physical_sky[8].rgb * (0.546274215296040 * (x2 - y2));
    value += physical_sky[9].rgb * (0.590043589926644 * y * (3.0 * x2 - y2));
    value += physical_sky[10].rgb * (2.890611442640554 * x * y * z);
    value += physical_sky[11].rgb * (0.457045799464466 * y * (5.0 * z2 - 1.0));
    value += physical_sky[12].rgb * (0.373176332590115 * z * (5.0 * z2 - 3.0));
    value += physical_sky[13].rgb * (0.457045799464466 * x * (5.0 * z2 - 1.0));
    value += physical_sky[14].rgb * (1.445305721320277 * z * (x2 - y2));
    value += physical_sky[15].rgb * (0.590043589926644 * x * (x2 - 3.0 * y2));
    return value;
}

fn simsat_upper_sky_black_lower(direction: vec3<f32>) -> vec3<f32> {
    if (direction.z <= 0.0) {
        return vec3<f32>(0.0);
    }
    return physical_sky[16].rgb * exp(clamp(log_sh3_sum(direction), vec3<f32>(-80.0), vec3<f32>(80.0)));
}

fn wrap_x(value: i32, width: i32) -> u32 {
    return u32(((value % width) + width) % width);
}

fn environment_map_radiance(direction: vec3<f32>) -> vec3<f32> {
    let n = normalize(direction);
    let longitude = atan2(n.y, n.x);
    let u = fract(longitude / TAU + 0.5);
    let v = acos(clamp(n.z, -1.0, 1.0)) / PI;
    let width = i32(collector_params.environment_width);
    let height = i32(collector_params.environment_height);
    let fx = u * f32(width) - 0.5;
    let fy = v * f32(height) - 0.5;
    let x0 = i32(floor(fx));
    let y0 = clamp(i32(floor(fy)), 0, height - 1);
    let y1 = min(y0 + 1, height - 1);
    let tx = fract(fx);
    let ty = fract(fy);
    let x0w = wrap_x(x0, width);
    let x1w = wrap_x(x0 + 1, width);
    let row0 = u32(y0) * collector_params.environment_width;
    let row1 = u32(y1) * collector_params.environment_width;
    let a = mix(environment_pixels[row0 + x0w].rgb, environment_pixels[row0 + x1w].rgb, tx);
    let b = mix(environment_pixels[row1 + x0w].rgb, environment_pixels[row1 + x1w].rgb, tx);
    return mix(a, b, ty);
}

fn environment_radiance(direction: vec3<f32>) -> vec3<f32> {
    if (collector_params.environment_mode == 1u) {
        return environment_pixels[0].rgb;
    }
    if (collector_params.environment_mode == 2u) {
        return environment_map_radiance(direction);
    }
    if (collector_params.environment_mode == 3u) {
        if (direction.z > 0.0) {
            return simsat_upper_sky_black_lower(direction);
        }
        return environment_map_radiance(direction);
    }
    return simsat_upper_sky_black_lower(direction);
}

@compute @workgroup_size(1024, 1, 1)
fn collect_specular(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let x = group_id.x;
    let y = collector_params.y_offset + group_id.y;
    let alpha_index = group_id.z;
    let normal = fibonacci_sphere(x, COLLECTOR_RESOLUTION);
    let view_direction = fibonacci_sphere(y, COLLECTOR_RESOLUTION);
    let ndotv = dot(normal, view_direction);
    let alpha = mix(0.2, 1.0, f32(alpha_index) / f32(ALPHA_LEVELS - 1u));
    var local_sum = vec3<f32>(0.0);
    if (ndotv > 0.0) {
        for (var lane = 0u; lane < GGX_SAMPLES_PER_THREAD; lane += 1u) {
            let sample_index = lane + local_id.x * GGX_SAMPLES_PER_THREAD;
            let half_vector = importance_sample_ggx(sample_index, normal, alpha);
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
            let geometry = min(min(1.0, 2.0 * ndoth * ndotv / vdoth), 2.0 * ndoth * ndotl / vdoth);
            let weight = geometry * vdoth / (ndotv * ndoth);
            local_sum += environment_radiance(light) * weight;
        }
    }
    collector_shared[local_id.x] = local_sum;
    workgroupBarrier();
    if (local_id.x < 512u) { collector_shared[local_id.x] += collector_shared[local_id.x + 512u]; }
    workgroupBarrier();
    if (local_id.x < 256u) { collector_shared[local_id.x] += collector_shared[local_id.x + 256u]; }
    workgroupBarrier();
    if (local_id.x < 128u) { collector_shared[local_id.x] += collector_shared[local_id.x + 128u]; }
    workgroupBarrier();
    if (local_id.x < 64u) { collector_shared[local_id.x] += collector_shared[local_id.x + 64u]; }
    workgroupBarrier();
    if (local_id.x < 32u) { collector_shared[local_id.x] += collector_shared[local_id.x + 32u]; }
    workgroupBarrier();
    if (local_id.x < 16u) { collector_shared[local_id.x] += collector_shared[local_id.x + 16u]; }
    workgroupBarrier();
    if (local_id.x < 8u) { collector_shared[local_id.x] += collector_shared[local_id.x + 8u]; }
    workgroupBarrier();
    if (local_id.x < 4u) { collector_shared[local_id.x] += collector_shared[local_id.x + 4u]; }
    workgroupBarrier();
    if (local_id.x < 2u) { collector_shared[local_id.x] += collector_shared[local_id.x + 2u]; }
    workgroupBarrier();
    if (local_id.x < 1u) { collector_shared[0] += collector_shared[1]; }
    workgroupBarrier();
    if (local_id.x == 0u) {
        let index = (alpha_index * COLLECTOR_RESOLUTION + y) * COLLECTOR_RESOLUTION + x;
        collector[index] = select(
            vec4<f32>(collector_shared[0] / f32(GGX_SAMPLES), 1.0),
            vec4<f32>(0.0),
            ndotv <= 0.0,
        );
    }
}

fn compute_sh_basis_l2(normal: vec3<f32>) -> array<f32, 9> {
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    return array<f32, 9>(
        0.488603*x, 0.488603*z, 0.488603*y, 0.546274*(x2-y2), 1.092548*x*z,
        0.946176*z2-0.315392, 1.092548*y*z, 1.092548*x*y, 0.282095,
    );
}

fn compute_sh_basis_l4(normal: vec3<f32>) -> array<f32, 25> {
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;
    let x2 = x*x;
    let y2 = y*y;
    let z2 = z*z;
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
    for (var index = 0u; index < 3u; index += 1u) { p[index] *= attenuation.x; q[index] *= attenuation.x; }
    p[3] *= attenuation.y; q[3] *= attenuation.y;
    for (var index = 4u; index < 8u; index += 1u) { p[index] *= attenuation.y; q[index] *= attenuation.y; }
    for (var index = 8u; index < 15u; index += 1u) { p[index] *= attenuation.z; }
    for (var index = 15u; index < 24u; index += 1u) { p[index] *= attenuation.w; }
    var result: array<f32, 33>;
    for (var index = 0u; index < 24u; index += 1u) { result[index] = p[index]; }
    for (var index = 0u; index < 8u; index += 1u) { result[24u + index] = q[index]; }
    result[32] = p[24];
    return result;
}

@group(1) @binding(0) var<storage, read> collector_read: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> matrix_blocks: array<f32>;

@compute @workgroup_size(33, 1, 1)
fn accumulate_matrix(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let row = local_id.x;
    let x = group_id.x;
    let y_slot = group_id.y;
    var row_accumulator: array<f32, 36>;
    for (var column = 0u; column < MATRIX_COLUMNS; column += 1u) {
        row_accumulator[column] = 0.0;
    }
    let normal = fibonacci_sphere(x, COLLECTOR_RESOLUTION);
    for (var alpha_index = 0u; alpha_index < ALPHA_LEVELS; alpha_index += 1u) {
        let alpha = mix(0.2, 1.0, f32(alpha_index) / f32(ALPHA_LEVELS - 1u));
        for (var y = y_slot; y < COLLECTOR_RESOLUTION; y += 16u) {
            let collector_index = (alpha_index * COLLECTOR_RESOLUTION + y) * COLLECTOR_RESOLUTION + x;
            let target_sample = collector_read[collector_index];
            if (target_sample.a == 0.0 || any(target_sample.rgb <= vec3<f32>(0.0))) {
                continue;
            }
            let view_direction = fibonacci_sphere(y, COLLECTOR_RESOLUTION);
            let vector = construct_sh_vector_a(normal, view_direction, alpha);
            let row_value = vector[row];
            for (var column = 0u; column < MATRIX_SIZE; column += 1u) {
                row_accumulator[column] += row_value * vector[column];
            }
            let log_target = log(target_sample.rgb);
            row_accumulator[33] += row_value * log_target.r;
            row_accumulator[34] += row_value * log_target.g;
            row_accumulator[35] += row_value * log_target.b;
        }
    }
    let block = x + COLLECTOR_RESOLUTION * y_slot;
    let base = block * MATRIX_WORDS + row * MATRIX_COLUMNS;
    for (var column = 0u; column < MATRIX_COLUMNS; column += 1u) {
        matrix_blocks[base + column] = row_accumulator[column];
    }
}

@group(2) @binding(0) var<storage, read> matrix_blocks_read: array<f32>;
@group(2) @binding(1) var<storage, read_write> summed_matrix: array<f32>;

@compute @workgroup_size(297, 1, 1)
fn sum_matrix(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let vector_index = local_id.x;
    let word = vector_index * 4u;
    var accumulator = vec4<f32>(0.0);
    for (var block = 0u; block < ACCUMULATOR_BLOCKS; block += 1u) {
        let base = block * MATRIX_WORDS + word;
        accumulator += vec4<f32>(
            matrix_blocks_read[base],
            matrix_blocks_read[base + 1u],
            matrix_blocks_read[base + 2u],
            matrix_blocks_read[base + 3u],
        );
    }
    summed_matrix[word] = accumulator.x;
    summed_matrix[word + 1u] = accumulator.y;
    summed_matrix[word + 2u] = accumulator.z;
    summed_matrix[word + 3u] = accumulator.w;
}

@group(3) @binding(0) var<storage, read> solver_matrix: array<f32>;
@group(3) @binding(1) var<storage, read_write> solver_coefficients: array<vec4<f32>, 33>;
@group(3) @binding(2) var<storage, read_write> solver_status: array<u32, 4>;

@compute @workgroup_size(1, 1, 1)
fn solve_unregularized() {
    // Single-invocation FP32 formulation of upstream's unregularized Cholesky
    // arithmetic. This avoids backend-dependent barrier lowering while keeping
    // the exact matrix, precision, coefficient order, and absence of a ridge.
    var matrix: array<array<f32, 33>, 33>;
    var solution: array<vec3<f32>, 33>;
    var lower: array<array<f32, 33>, 33>;
    for (var row = 0u; row < MATRIX_SIZE; row += 1u) {
        let base = row * MATRIX_COLUMNS;
        for (var column = 0u; column < MATRIX_SIZE; column += 1u) {
            matrix[row][column] = solver_matrix[base + column];
            lower[row][column] = 0.0;
        }
        solution[row] = vec3<f32>(solver_matrix[base + 33u], solver_matrix[base + 34u], solver_matrix[base + 35u]);
    }
    for (var row = 0u; row < MATRIX_SIZE; row += 1u) {
        for (var column = 0u; column <= row; column += 1u) {
            var value = matrix[row][column];
            for (var k = 0u; k < column; k += 1u) { value -= lower[row][k] * lower[column][k]; }
            if (row == column) {
                lower[row][column] = sqrt(max(0.0, value));
            } else {
                lower[row][column] = value / lower[column][column];
            }
        }
    }
    for (var row = 0u; row < MATRIX_SIZE; row += 1u) {
        var value = solution[row];
        for (var k = 0u; k < row; k += 1u) { value -= lower[row][k] * solution[k]; }
        solution[row] = value / lower[row][row];
    }
    for (var reverse = 0u; reverse < MATRIX_SIZE; reverse += 1u) {
        let row = MATRIX_SIZE - 1u - reverse;
        var value = solution[row];
        for (var k = row + 1u; k < MATRIX_SIZE; k += 1u) { value -= lower[k][row] * solution[k]; }
        solution[row] = value / lower[row][row];
    }
    var valid = true;
    var positive_diagonal = true;
    for (var index = 0u; index < MATRIX_SIZE; index += 1u) {
        solver_coefficients[index] = vec4<f32>(solution[index], 0.0);
        valid = valid && all(abs(solution[index]) < vec3<f32>(3.0e38));
        positive_diagonal = positive_diagonal
            && abs(lower[index][index]) < 3.0e38
            && lower[index][index] > 0.0;
    }
    solver_status[0] = select(0u, 1u, valid && positive_diagonal);
    solver_status[1] = COLLECTOR_RESOLUTION;
    solver_status[2] = GGX_SAMPLES;
    solver_status[3] = ALPHA_LEVELS;
}
