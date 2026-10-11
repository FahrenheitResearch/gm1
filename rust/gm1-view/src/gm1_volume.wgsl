// GM1 View volume raymarch. Density words are exact admitted GM1 binary32 values.
// Positive log-SH3 basis/evaluation is adapted from the SimSat project's
// implementation (third_party/SimSat-PROVENANCE.md); direct sun remains a separate term.

#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}

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

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<storage, read> density_words: array<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> material: VolumeUniform;
// Admitted coordinate layout: exact sigma[77] followed by exact
// terrain_msl[300,300], all preserved as binary32 words.
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<storage, read> coordinate_words: array<u32>;

const LOG_SH_LIMIT: f32 = 80.0;
const MAX_STEPS: u32 = 256u;
const DEBUG_SOLID_AABB: bool = false;
// Incremental shader validation rung. Zero is the scientific raymarch.
const DEBUG_RUNG: u32 = 0u;

fn coordinate_word(index: u32) -> f32 {
    return bitcast<f32>(coordinate_words[index]);
}

fn admitted_terrain_m(horizontal: vec2<f32>) -> f32 {
    let dimensions = vec2<u32>(material.grid_shape.xy);
    let maximum = dimensions - vec2<u32>(1u);
    let grid = clamp(horizontal, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(maximum);
    let lower = vec2<u32>(floor(grid));
    let upper = min(lower + vec2<u32>(1u), maximum);
    let weight = fract(grid);
    let offset = 77u;
    let a = coordinate_word(offset + lower.y * dimensions.x + lower.x);
    let b = coordinate_word(offset + lower.y * dimensions.x + upper.x);
    let c = coordinate_word(offset + upper.y * dimensions.x + lower.x);
    let d = coordinate_word(offset + upper.y * dimensions.x + upper.x);
    return mix(mix(a, b, weight.x), mix(c, d, weight.x), weight.y);
}

fn admitted_vertical_coordinate(world_y_km: f32, horizontal: vec2<f32>) -> f32 {
    let terrain_m = admitted_terrain_m(horizontal);
    let model_top_m = 15000.0;
    // Invert CM1 init_terrain.F's exact composition:
    // z = terrain + sigma * (top - terrain) / top.
    let sigma_m = (world_y_km * 1000.0 - terrain_m)
        * model_top_m / (model_top_m - terrain_m);
    let first = coordinate_word(1u);
    if (sigma_m < first) {
        return (sigma_m - first) / (coordinate_word(2u) - first) / 74.0;
    }
    for (var profile = 1u; profile < 75u; profile += 1u) {
        let lower = coordinate_word(profile);
        let upper = coordinate_word(profile + 1u);
        if (sigma_m <= upper) {
            return (f32(profile - 1u) + clamp((sigma_m - lower) / (upper - lower), 0.0, 1.0)) / 74.0;
        }
    }
    let last = coordinate_word(75u);
    return 1.0 + (sigma_m - last) / (last - coordinate_word(74u)) / 74.0;
}

fn world_to_texture(world: vec3<f32>) -> vec3<f32> {
    // CM1 scalar mass points lie at half-cell centres, not on the outer
    // domain faces used by the raster AABB. Map the first/last physical mass
    // points exactly to source indices 0/(n-1), preserving source slices and
    // avoiding an implicit half-cell interpolation at every grid centre.
    let counts = vec2<f32>(material.grid_shape.xy);
    let cell_size = vec2<f32>(material.world_size.x, material.world_size.z) / counts;
    let grid_origin = vec2<f32>(material.world_min.x, material.world_min.z) + 0.5 * cell_size;
    let grid_span = cell_size * max(counts - vec2<f32>(1.0), vec2<f32>(1.0));
    let horizontal = (world.xz - grid_origin) / grid_span;
    // Scalar mass points lie at half-cell centres vertically as well, so the
    // linear level path uses the same first/last-mass-point convention as the
    // horizontal axes. Mapping the raw box fraction instead shifts every level
    // by half a cell and reports the whole column at the wrong altitude.
    let level_count = f32(material.grid_shape.z);
    let level_cell = material.world_size.y / level_count;
    let level_origin = material.world_min.y + 0.5 * level_cell;
    let level_span = level_cell * max(level_count - 1.0, 1.0);
    var vertical = (world.y - level_origin) / level_span;
    if (material.world_min.w > 0.5) {
        vertical = admitted_vertical_coordinate(world.y, horizontal);
    }
    // Texture order is model x, model y, model z; Bevy world y is vertical.
    return vec3<f32>(horizontal.x, horizontal.y, vertical);
}

struct DensitySample {
    value: f32,
    gradient: vec3<f32>,
}

// Load exact C-order [z,y,x] binary32 words, reconstruct trilinearly, and
// derive the interpolation gradient from those same source values. The GPU
// never filters or converts a source word before the explicit f32 bitcast.
fn load_density_word(index: vec3<i32>, dimensions: vec3<i32>) -> f32 {
    let flat = (u32(index.z) * u32(dimensions.y) + u32(index.y))
        * u32(dimensions.x) + u32(index.x);
    return bitcast<f32>(density_words[flat]);
}

fn sample_density(position: vec3<f32>) -> DensitySample {
    if (any(position < vec3<f32>(0.0)) || any(position > vec3<f32>(1.0))) {
        return DensitySample(0.0, vec3<f32>(0.0));
    }
    let dimensions = vec3<i32>(material.grid_shape.xyz);
    let maximum = dimensions - vec3<i32>(1);
    let grid = position * vec3<f32>(maximum);
    let lower = vec3<i32>(floor(grid));
    let upper = min(lower + vec3<i32>(1), maximum);
    let weight = fract(grid);

    let c000 = load_density_word(vec3<i32>(lower.x, lower.y, lower.z), dimensions);
    let c100 = load_density_word(vec3<i32>(upper.x, lower.y, lower.z), dimensions);
    let c010 = load_density_word(vec3<i32>(lower.x, upper.y, lower.z), dimensions);
    let c110 = load_density_word(vec3<i32>(upper.x, upper.y, lower.z), dimensions);
    let c001 = load_density_word(vec3<i32>(lower.x, lower.y, upper.z), dimensions);
    let c101 = load_density_word(vec3<i32>(upper.x, lower.y, upper.z), dimensions);
    let c011 = load_density_word(vec3<i32>(lower.x, upper.y, upper.z), dimensions);
    let c111 = load_density_word(vec3<i32>(upper.x, upper.y, upper.z), dimensions);

    let c00 = mix(c000, c100, weight.x);
    let c10 = mix(c010, c110, weight.x);
    let c01 = mix(c001, c101, weight.x);
    let c11 = mix(c011, c111, weight.x);
    let c0 = mix(c00, c10, weight.y);
    let c1 = mix(c01, c11, weight.y);
    let value = mix(c0, c1, weight.z);

    let derivative_x = mix(
        mix(c100 - c000, c110 - c010, weight.y),
        mix(c101 - c001, c111 - c011, weight.y),
        weight.z,
    ) * f32(maximum.x);
    let derivative_y = mix(
        mix(c010 - c000, c110 - c100, weight.x),
        mix(c011 - c001, c111 - c101, weight.x),
        weight.z,
    ) * f32(maximum.y);
    let derivative_z = (c1 - c0) * f32(maximum.z);
    let gradient = vec3<f32>(
        derivative_x / material.world_size.x,
        derivative_z / material.world_size.y,
        derivative_y / material.world_size.z,
    );
    return DensitySample(value, gradient);
}

// The full exact field remains resident. This display weight is one at and
// beyond the first interior cell, zero on the lateral edge, and affects only
// the exact three-cell open boundary band when the UI toggle is enabled.
fn open_boundary_display_weight(position: vec3<f32>) -> f32 {
    if (material.grid_shape.w == 0u) {
        return 1.0;
    }
    let maximum = vec2<f32>(material.grid_shape.xy - vec2<u32>(1u));
    let cell = position.xy * maximum;
    let edge_distance = min(
        min(cell.x, maximum.x - cell.x),
        min(cell.y, maximum.y - cell.y),
    );
    return smoothstep(0.0, 3.0, edge_distance);
}

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

fn positive_sky_irradiance(normal: vec3<f32>) -> vec3<f32> {
    let log_value = clamp(log_sh3_sum(sun_frame_direction(normal)), vec3<f32>(-LOG_SH_LIMIT), vec3<f32>(LOG_SH_LIMIT));
    return max(material.sky_amplitude.rgb, vec3<f32>(0.0)) * exp(log_value);
}

fn directional_transmittance_reference(position: vec3<f32>, threshold: f32, density_limit: f32) -> f32 {
    let sun_world = normalize(material.sun_direction.xyz);
    let sun_texture = normalize(vec3<f32>(
        sun_world.x / material.world_size.x,
        sun_world.z / material.world_size.z,
        sun_world.y / material.world_size.y,
    ));
    var optical_depth = 0.0;
    let offsets = array<f32, 4>(0.010, 0.025, 0.050, 0.085);
    for (var index = 0u; index < 4u; index += 1u) {
        let probe = position + sun_texture * offsets[index];
        let inside = all(probe >= vec3<f32>(0.0)) && all(probe <= vec3<f32>(1.0));
        if (inside) {
            let probe_density = sample_density(probe).value;
            let normalized = clamp(
                (probe_density - threshold) / max(density_limit - threshold, 1.0e-8),
                0.0,
                1.0,
            );
            optical_depth += pow(normalized, 1.35) * open_boundary_display_weight(probe);
        }
    }
    return exp(-0.46 * optical_depth);
}

fn condensate_extinction_per_km(density: f32, threshold: f32) -> f32 {
    // Combined qc+qi does not carry effective radius or air density. The host
    // seals a conservative 60 um onset/mixed-ice radius, 12 um dense-cloud
    // radius, and 0.8 kg m-3 air density. Qext=2 and rho_condensed=1000 kg m-3.
    let dense_weight = smoothstep(threshold, max(material.optics.w, threshold + 1.0e-8), density);
    let effective_radius_m = mix(material.optics.x, material.optics.y, dense_weight);
    // Grid-mean onset condensate is not treated as a fully occupied dense
    // parcel. This sealed occupancy response rises from 2.5% at onset to 100%
    // at the exact q99 component cutoff; no density word is changed or hidden.
    let occupancy = mix(0.025, 1.0, dense_weight);
    let q_excess = max(density - threshold, 0.0);
    let beta_per_m = 3.0 * 2.0 * material.optics.z * q_excess
        * occupancy / (4.0 * 1000.0 * effective_radius_m);
    return beta_per_m * 1000.0;
}

fn directional_transmittance_physical(position_world: vec3<f32>, threshold: f32) -> f32 {
    let sun_world = normalize(material.sun_direction.xyz);
    var optical_depth = 0.0;
    // Short deterministic 1.6 km sunward Beer-Lambert march. Distances are
    // physical kilometres; terrain/sigma mapping is recomputed per sample.
    let light_step_km = 0.16;
    for (var index = 0u; index < 10u; index += 1u) {
        let probe_world = position_world + sun_world * (f32(index) + 0.5) * light_step_km;
        if (inside_world_bounds(probe_world)) {
            let probe = world_to_texture(probe_world);
            let probe_density = sample_density(probe).value;
            optical_depth += condensate_extinction_per_km(probe_density, threshold)
                * light_step_km * open_boundary_display_weight(probe);
            if (optical_depth >= 12.0) {
                break;
            }
        }
    }
    return exp(-optical_depth);
}

fn henyey_greenstein_relative(cosine: f32, asymmetry: f32) -> f32 {
    let g2 = asymmetry * asymmetry;
    let denominator = pow(max(1.0 + g2 - 2.0 * asymmetry * cosine, 1.0e-4), 1.5);
    return (1.0 - g2) / (12.566370614359172 * denominator);
}

fn integration_jitter(pixel: vec2<f32>) -> f32 {
    // Interleaved-gradient integration jitter changes only the first sample
    // phase; it introduces no density or procedural cloud detail.
    return fract(52.9829189 * fract(dot(floor(pixel), vec2<f32>(0.06711056, 0.00583715))));
}

fn inside_world_bounds(position: vec3<f32>) -> bool {
    let world_max = material.world_min.xyz + material.world_size.xyz;
    return all(position >= material.world_min.xyz) && all(position <= world_max);
}

fn exit_distance_world(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    let safe_direction = select(
        direction,
        vec3<f32>(1.0e-7),
        abs(direction) < vec3<f32>(1.0e-7),
    );
    let t0 = (material.world_min.xyz - origin) / safe_direction;
    let t1 = (material.world_min.xyz + material.world_size.xyz - origin) / safe_direction;
    let far = max(t0, t1);
    return max(min(far.x, min(far.y, far.z)), 0.0);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if (DEBUG_SOLID_AABB) {
        return vec4<f32>(0.72, 0.10, 0.035, 1.0);
    }
    let camera_world = view.world_position;
    let surface_world = in.world_position.xyz;
    var entry_world = surface_world;
    let camera_inside = inside_world_bounds(camera_world);
    if (camera_inside) {
        // A camera inside the exact physical domain starts each ray at its own
        // position. The rasterized back face supplies only the exit direction,
        // keeping the model-terrain perimeter out of the presentation frustum.
        entry_world = camera_world;
    }
    let entry_texture = world_to_texture(entry_world);
    if (DEBUG_RUNG == 1u) {
        return vec4<f32>(clamp(entry_texture, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
    if (DEBUG_RUNG == 5u) {
        // Terrain-only composition probe: retain the exact volume mesh and
        // bindings while contributing no color or alpha.
        return vec4<f32>(0.0);
    }
    if (DEBUG_RUNG == 2u) {
        // CPU strict probe over field SHA a101f308... found the binary32
        // maximum at [k,j,i]=[45,56,231], bits 0x3b7409be.
        let known_max = bitcast<f32>(0x3b7409beu);
        let loaded_bits = density_words[4067031u];
        let loaded = bitcast<f32>(loaded_bits);
        let normalized = clamp(loaded / known_max, 0.0, 1.0);
        return select(
            vec4<f32>(1.0, 0.0, 0.7, 1.0),
            vec4<f32>(0.0, normalized, 0.2, 1.0),
            loaded_bits == 0x3b7409beu,
        );
    }
    if (DEBUG_RUNG == 20u) {
        let dimensions = vec3<f32>(material.grid_shape.xyz);
        return vec4<f32>(dimensions / vec3<f32>(300.0, 300.0, 75.0), 1.0);
    }
    if (DEBUG_RUNG == 21u) {
        let dimensions = vec3<i32>(material.grid_shape.xyz);
        let loaded = load_density_word(vec3<i32>(231, 56, 45), dimensions);
        return select(
            vec4<f32>(1.0, 0.0, 0.7, 1.0),
            vec4<f32>(0.0, 1.0, 0.2, 1.0),
            loaded >= 0.0,
        );
    }
    // March the physical camera ray in world space. The terrain-following
    // world-to-texture transform is nonlinear in x/y/z, so interpolating a
    // straight line in texture coordinates bends the actual view ray and can
    // create diagonal sheets or miss a low-AGL chase view entirely. Convert
    // every world-space sample independently instead.
    let direction_world = normalize(surface_world - camera_world);
    let distance_world = exit_distance_world(entry_world, direction_world);
    if (DEBUG_RUNG == 3u) {
        let inside = inside_world_bounds(entry_world);
        return vec4<f32>(
            clamp(distance_world / length(material.world_size.xyz), 0.0, 1.0),
            select(0.0, 0.86, inside),
            clamp(abs(direction_world.y), 0.0, 1.0),
            1.0,
        );
    }
    let requested_steps = clamp(u32(material.control.z), 24u, MAX_STEPS);
    let base_step_world = max(distance_world / f32(requested_steps), 1.0e-5);
    let threshold = material.control.x;
    let opacity = material.control.y;
    let jitter = integration_jitter(in.position.xy);
    var position_world = entry_world + direction_world * base_step_world * (0.15 + 0.70 * jitter);
    var travelled = 0.0;
    var accumulated = vec4<f32>(0.0);
    var step_size_world = base_step_world;
    var maximum_density_seen = 0.0;

    if (DEBUG_RUNG == 4u) {
        var maximum_density = 0.0;
        var debug_position_world = position_world;
        var debug_travelled = 0.0;
        for (var iteration = 0u; iteration < MAX_STEPS; iteration += 1u) {
            if (iteration >= requested_steps || debug_travelled >= distance_world) {
                break;
            }
            if (!inside_world_bounds(debug_position_world)) {
                break;
            }
            let debug_position_texture = world_to_texture(debug_position_world);
            maximum_density = max(maximum_density, sample_density(debug_position_texture).value);
            debug_position_world += direction_world * base_step_world;
            debug_travelled += base_step_world;
        }
        let density_limit = max(threshold + 1.0e-8, material.world_size.w);
        let projected = clamp(
            (maximum_density - threshold) / (density_limit - threshold),
            0.0,
            1.0,
        );
        if (maximum_density <= threshold) {
            return vec4<f32>(0.0);
        }
        return vec4<f32>(projected, 0.18 + 0.72 * projected, 1.0 - projected, 1.0);
    }

    for (var iteration = 0u; iteration < MAX_STEPS; iteration += 1u) {
        if (iteration >= requested_steps || travelled >= distance_world || accumulated.a > 0.992) {
            break;
        }
        if (!inside_world_bounds(position_world)) {
            break;
        }
        let position_texture = world_to_texture(position_world);
        let density_sample = sample_density(position_texture);
        let density = density_sample.value;
        maximum_density_seen = max(maximum_density_seen, density);
        // The exact combined qc+qi word drives a physical-km Beer-Lambert
        // extinction. The q99 weight changes scattering response only; it
        // does not threshold, crop, or perturb low-q condensate.
        let density_limit = max(threshold + 1.0e-8, material.world_size.w);
        let dense_response = smoothstep(
            threshold,
            max(material.optics.w, threshold + 1.0e-8),
            density,
        );
        let beta_per_km = condensate_extinction_per_km(density, threshold)
            * open_boundary_display_weight(position_texture);
        let extinction = beta_per_km;
        if (extinction > 0.0) {
            let gradient = density_sample.gradient;
            let normal = select(vec3<f32>(0.0, 1.0, 0.0), -normalize(gradient), length(gradient) > 1.0e-12);
            let sun_direction = normalize(material.sun_direction.xyz);
            let physical_lighting = material.sky_amplitude.w > 0.5;
            var shadow = directional_transmittance_reference(position_texture, threshold, density_limit);
            if (physical_lighting) {
                shadow = directional_transmittance_physical(position_world, threshold);
            }
            let sky = select(
                vec3<f32>(0.42, 0.5, 0.65),
                positive_sky_irradiance(vec3<f32>(0.0, 1.0, 0.0)),
                material.control.w > 0.5,
            );
            // `sun_direction` points sample->sun and `direction_world` points
            // camera->sample. Their dot is the scattering cosine because both
            // incident and outgoing photon directions reverse together.
            let phase_cosine = clamp(dot(sun_direction, direction_world), -1.0, 1.0);
            let phase = henyey_greenstein_relative(phase_cosine, 0.72);
            let direct_exposure = material.sun_color.a;
            // Keep direct-sun and sky display exposures distinct. The latter
            // exactly matches the sealed source-backed SimSat sky dome. The
            // lower hemisphere is the admitted SimSat Lambertian ground
            // bounce, Lground = Esky * albedo / pi; no authored gradient or
            // terrain colour enters this cloud source term.
            let upper_sky_radiance = sky * material.scattering.x / 3.141592653589793;
            let lower_ground_radiance = upper_sky_radiance * material.scattering.y;
            let full_sphere_mean_radiance =
                0.5 * (upper_sky_radiance + lower_ground_radiance);
            // Optically thick near-conservative cloud tends toward the sealed
            // full-sphere mean. Onset haze receives less because its sealed
            // occupancy is partial rather than a fully cloudy parcel.
            let ambient_single = full_sphere_mean_radiance
                * mix(0.30, 0.72, dense_response);
            // Two energy-reduced multiple-scatter octaves reuse the same
            // source field and optical-depth march. Their sealed 1/2 and 1/4
            // weights are normalized inside a bounded 0.28 dense energy share;
            // no procedural density/detail or energy gain is added.
            let octave_weight_sum = material.scattering.z + material.scattering.w;
            let multiple_scatter = full_sphere_mean_radiance
                * (1.0 - shadow)
                * mix(0.08, 0.28, dense_response)
                * clamp(octave_weight_sum / 0.75, 0.0, 1.0);
            let optical_depth = -log(max(shadow, 1.0e-6));
            let phase_octave_1 = henyey_greenstein_relative(phase_cosine, 0.36);
            let phase_octave_2 = henyey_greenstein_relative(phase_cosine, 0.18);
            let direct_scatter = material.sun_color.rgb * direct_exposure * (
                phase * exp(-optical_depth)
                + 0.32 * phase_octave_1 * exp(-0.50 * optical_depth)
                + 0.10 * phase_octave_2 * exp(-0.25 * optical_depth)
            );
            let gradient_rim = pow(max(dot(normal, -direction_world), 0.0), 6.0)
                * ambient_single * mix(0.05, 0.28, dense_response);
            let cloud_albedo = mix(
                vec3<f32>(0.74, 0.83, 0.92),
                vec3<f32>(1.00, 0.94, 0.84),
                dense_response,
            );
            let color = vec3<f32>(0.006, 0.014, 0.032)
                + cloud_albedo * (ambient_single + multiple_scatter + direct_scatter)
                + gradient_rim;
            let alpha = 1.0 - exp(-extinction * opacity * step_size_world);
            let premultiplied = color * alpha;
            let remaining = 1.0 - accumulated.a;
            accumulated = vec4<f32>(
                accumulated.rgb + remaining * premultiplied,
                accumulated.a + remaining * alpha,
            );
            step_size_world = base_step_world;
        } else {
            // Empty-space adaptation keeps the exact texture but skips clear cells faster.
            step_size_world = select(
                base_step_world,
                base_step_world * 2.0,
                density < threshold * 0.08,
            );
        }
        position_world += direction_world * step_size_world;
        travelled += step_size_world;
    }
    // Deterministic paired-readback object mask. The exact field, transfer,
    // world-space ray, camera, and sample cadence are unchanged; only the
    // final display target becomes a binary cloud-alpha identity. Terrain and
    // UI are disabled by the host for this pass. `sun_direction.w` is an
    // otherwise-unused uniform lane and is restored to zero before the
    // accepted combined presentation capture.
    if (material.sun_direction.w > 1.5) {
        if (accumulated.a >= 0.85 && maximum_density_seen >= material.optics.w) {
            return vec4<f32>(1.0);
        }
        return vec4<f32>(0.0);
    }
    if (material.sun_direction.w > 0.5) {
        if (accumulated.a >= 0.05) {
            return vec4<f32>(1.0);
        }
        return vec4<f32>(0.0);
    }
    return accumulated;
}
