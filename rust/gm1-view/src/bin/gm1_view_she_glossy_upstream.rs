//! Execute and seal the pinned upstream-math unregularized FP32 glossy SHE chain.

use bevy::math::Vec3;
use gm1_view::{
    lighting::{SIMSAT_ATMOSPHERE_SHA256, T300_SKY_ROW_SHA256},
    she::{
        SheSpecularCoefficients, UPSTREAM_COMMIT, UPSTREAM_LICENSE, UPSTREAM_PROJECT,
        reconstruct_specular,
    },
    she_glossy::{
        SIMSAT_BUNDLE_PATCH_SHA256, SIMSAT_RADIANCE_SHA256, SIMSAT_RADIANCE_SOURCE_SHA256,
        SimSatSkyRadiance, prefilter_radiance_ggx, specular_coefficients_bytes,
        specular_coefficients_sha256,
    },
    she_glossy_upstream::{
        ALPHA_LEVELS, COLLECTOR_RESOLUTION, GGX_SAMPLES, HELD_GGX_SAMPLES, HELD_RESOLUTION,
        UPSTREAM_ACCUMULATOR_SHA256, UPSTREAM_PREFILTER_SHA256, UPSTREAM_SOLVER_SHA256,
        UPSTREAM_SUM_SHA256, WGSL_SOURCE, held_error_for_radiance,
    },
    she_material::t300_reference_irradiance,
};
use image::Rgb32FImage;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    env, fs,
    path::{Path, PathBuf},
    sync::mpsc,
    time::Instant,
};
use wgpu::util::DeviceExt;

const Y_BATCH: u32 = 16;
const MATRIX_WORDS: usize = 33 * 36;
const MATRIX_BLOCKS: usize = 256 * 16;
const MATRIX_BLOCK_BYTES: u64 = (MATRIX_WORDS * MATRIX_BLOCKS * 4) as u64;
const SUMMED_MATRIX_BYTES: u64 = (MATRIX_WORDS * 4) as u64;
const COEFFICIENT_BYTES_PADDED: u64 = (33 * 16) as u64;
const STATUS_BYTES: u64 = 16;
const MAX_NORMALIZED_RMS_RADIANCE: f32 = 0.20;
const MAX_NORMALIZED_RMS_LUMINANCE: f32 = 0.20;
const CONSTANT_CONTROL_RADIANCE: [f32; 3] = [0.625, 1.0, 1.5];
const GM1_TERRAIN_SHA256: &str = "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91";
const HRRR_LANDSEA_SHA256: &str =
    "477d51178ab094a0f6cfac474918873d106954c7b2031665f920c2dc9d22eab8";
const HRRR_ZERO_SURFACE_SHA256: &str =
    "9dd75789e0f371d50efb6aaf2878d08ccf1374a4189a0941ae6057b11e487c02";
const HRRR_SKINTEMP_SHA256: &str =
    "f8f5457210df5842c089e42971a2d231a7b4d63857653b647e9aa176ecff49de";
const SIMSAT_GROUND_ALBEDO: f32 = 0.3;
const GM1_TERRAIN_NX: usize = 300;
const GM1_TERRAIN_NY: usize = 300;
const GM1_TERRAIN_DX_M: f32 = 50.0;
const GM1_TERRAIN_DY_M: f32 = 50.0;
const GM1_ENVIRONMENT_WIDTH: u32 = 512;
const GM1_ENVIRONMENT_HEIGHT: u32 = 256;
const MATERIAL_GRID_STRIDE: usize = 5;
const MATERIAL_REFERENCE_GGX_SAMPLES: u32 = 4096;
const MATERIAL_ALPHA: f32 = 0.22;
const MATERIAL_DIELECTRIC_F0: f32 = 0.04;
const MATERIAL_BASE_ALBEDO: [f32; 3] = [0.060, 0.050, 0.035];
const MATERIAL_VIEW_DIRECTION: [f32; 3] = [-0.42, -0.30, 0.856];

struct Arguments {
    output_dir: PathBuf,
    environment: EnvironmentRequest,
    training_alpha_levels: u32,
}

enum EnvironmentRequest {
    SimSatUpperBlack,
    ConstantFullSphere,
    UpstreamHdr(PathBuf),
    Gm1TerrainLower {
        terrain_file: PathBuf,
        surface_dir: PathBuf,
    },
}

enum EnvironmentData {
    SimSatUpperBlack {
        sky: SimSatSkyRadiance,
    },
    ConstantFullSphere {
        radiance: Vec3,
    },
    UpstreamHdr {
        path: PathBuf,
        width: u32,
        height: u32,
        pixels: Vec<[f32; 4]>,
        file_sha256: String,
        decoded_sha256: String,
    },
    Gm1TerrainLower {
        sky: SimSatSkyRadiance,
        terrain_file: PathBuf,
        terrain_manifest_sha256: String,
        terrain_receipt_sha256: String,
        pixels: Vec<[f32; 4]>,
        decoded_sha256: String,
        terrain_min_msl_m: f32,
        terrain_max_msl_m: f32,
        surface_dir: PathBuf,
        surface_skin_temperature_min_k: f32,
        surface_skin_temperature_max_k: f32,
    },
}

impl EnvironmentData {
    fn load(request: EnvironmentRequest) -> Result<Self, String> {
        match request {
            EnvironmentRequest::SimSatUpperBlack => Ok(Self::SimSatUpperBlack {
                sky: SimSatSkyRadiance::sealed_t300()?,
            }),
            EnvironmentRequest::ConstantFullSphere => Ok(Self::ConstantFullSphere {
                radiance: Vec3::from_array(CONSTANT_CONTROL_RADIANCE),
            }),
            EnvironmentRequest::UpstreamHdr(path) => {
                let file_sha256 = sha256_file(&path)?;
                let image = image::open(&path)
                    .map_err(|error| {
                        format!("cannot decode upstream HDR {}: {error}", path.display())
                    })?
                    .into_rgb32f();
                validate_hdr(&image)?;
                let (width, height) = image.dimensions();
                let pixels = image
                    .pixels()
                    .map(|pixel| [pixel[0], pixel[1], pixel[2], 0.0])
                    .collect::<Vec<_>>();
                let decoded_sha256 = sha256_bytes(bytemuck::cast_slice(&pixels));
                Ok(Self::UpstreamHdr {
                    path,
                    width,
                    height,
                    pixels,
                    file_sha256,
                    decoded_sha256,
                })
            }
            EnvironmentRequest::Gm1TerrainLower {
                terrain_file,
                surface_dir,
            } => {
                let terrain = read_pinned_f32(
                    &terrain_file,
                    GM1_TERRAIN_NX * GM1_TERRAIN_NY,
                    GM1_TERRAIN_SHA256,
                    "GM1 child terrain",
                )?;
                let terrain_min_msl_m = terrain.iter().copied().fold(f32::INFINITY, f32::min);
                let terrain_max_msl_m = terrain.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let landsea = read_pinned_f32(
                    &surface_dir.join("LANDSEA.f32le"),
                    26 * 26,
                    HRRR_LANDSEA_SHA256,
                    "HRRR LANDSEA",
                )?;
                let snow = read_pinned_f32(
                    &surface_dir.join("SNOWH.f32le"),
                    26 * 26,
                    HRRR_ZERO_SURFACE_SHA256,
                    "HRRR SNOWH",
                )?;
                let xice = read_pinned_f32(
                    &surface_dir.join("XICE.f32le"),
                    26 * 26,
                    HRRR_ZERO_SURFACE_SHA256,
                    "HRRR XICE",
                )?;
                let skin_temperature = read_pinned_f32(
                    &surface_dir.join("SKINTEMP.f32le"),
                    26 * 26,
                    HRRR_SKINTEMP_SHA256,
                    "HRRR SKINTEMP",
                )?;
                if landsea
                    .iter()
                    .any(|value| value.to_bits() != 1.0_f32.to_bits())
                    || snow.iter().any(|value| value.to_bits() != 0)
                    || xice.iter().any(|value| value.to_bits() != 0)
                {
                    return Err(
                        "pinned GM1 surface profile must be all land with no snow/ice".to_owned(),
                    );
                }
                let surface_skin_temperature_min_k = skin_temperature
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min);
                let surface_skin_temperature_max_k = skin_temperature
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let pixels = build_gm1_terrain_lower_environment(&terrain)?;
                let decoded_sha256 = sha256_bytes(bytemuck::cast_slice(&pixels));
                let terrain_root = terrain_file.parent().ok_or_else(|| {
                    "GM1 terrain payload has no parent directory for provenance".to_owned()
                })?;
                let terrain_manifest_sha256 = sha256_file(&terrain_root.join("manifest.json"))?;
                let terrain_receipt_sha256 = sha256_file(&terrain_root.join("receipt.json"))?;
                Ok(Self::Gm1TerrainLower {
                    sky: SimSatSkyRadiance::sealed_t300()?,
                    terrain_file,
                    terrain_manifest_sha256,
                    terrain_receipt_sha256,
                    pixels,
                    decoded_sha256,
                    terrain_min_msl_m,
                    terrain_max_msl_m,
                    surface_dir,
                    surface_skin_temperature_min_k,
                    surface_skin_temperature_max_k,
                })
            }
        }
    }

    fn mode(&self) -> u32 {
        match self {
            Self::SimSatUpperBlack { .. } => 0,
            Self::ConstantFullSphere { .. } => 1,
            Self::UpstreamHdr { .. } => 2,
            Self::Gm1TerrainLower { .. } => 3,
        }
    }

    fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::UpstreamHdr { width, height, .. } => (*width, *height),
            Self::Gm1TerrainLower { .. } => (GM1_ENVIRONMENT_WIDTH, GM1_ENVIRONMENT_HEIGHT),
            _ => (1, 1),
        }
    }

    fn gpu_pixels(&self) -> Cow<'_, [[f32; 4]]> {
        match self {
            Self::SimSatUpperBlack { .. } => Cow::Owned(vec![[0.0; 4]]),
            Self::ConstantFullSphere { radiance } => {
                Cow::Owned(vec![[radiance.x, radiance.y, radiance.z, 0.0]])
            }
            Self::UpstreamHdr { pixels, .. } => Cow::Borrowed(pixels),
            Self::Gm1TerrainLower { pixels, .. } => Cow::Borrowed(pixels),
        }
    }

    fn evaluate(&self, direction: Vec3) -> Vec3 {
        match self {
            Self::SimSatUpperBlack { sky } => {
                if direction.z > 0.0 {
                    sky.evaluate(direction)
                } else {
                    Vec3::ZERO
                }
            }
            Self::ConstantFullSphere { radiance } => *radiance,
            Self::UpstreamHdr {
                width,
                height,
                pixels,
                ..
            } => sample_equirectangular(*width, *height, pixels, direction),
            Self::Gm1TerrainLower { sky, pixels, .. } => {
                if direction.z > 0.0 {
                    sky.evaluate(direction)
                } else {
                    sample_equirectangular(
                        GM1_ENVIRONMENT_WIDTH,
                        GM1_ENVIRONMENT_HEIGHT,
                        pixels,
                        direction,
                    )
                }
            }
        }
    }

    fn slug(&self) -> &'static str {
        match self {
            Self::SimSatUpperBlack { .. } => "simsat-upper-black-lower",
            Self::ConstantFullSphere { .. } => "constant-full-sphere-control",
            Self::UpstreamHdr { .. } => "upstream-hdr-full-sphere",
            Self::Gm1TerrainLower { .. } => "gm1-terrain-lower-full-sphere",
        }
    }

    fn receipt(&self) -> serde_json::Value {
        match self {
            Self::SimSatUpperBlack { .. } => serde_json::json!({
                "kind": "SimSat Hillaire positive log-SH3 clear-sky radiance; direct sun excluded; below-horizon sky zero",
                "diagnostic_role": "physical upper-sky plus deliberately absent lower hemisphere source-isolation case",
                "valid_time_utc": "2024-05-21T21:05:00Z",
                "center_latitude_deg": 41.39464041135_f64,
                "center_longitude_deg": -94.28268293687_f64,
                "radiance_payload_sha256": SIMSAT_RADIANCE_SHA256,
                "atmosphere_source_sha256": SIMSAT_RADIANCE_SOURCE_SHA256,
                "bundle_patch_sha256": SIMSAT_BUNDLE_PATCH_SHA256,
            }),
            Self::ConstantFullSphere { radiance } => serde_json::json!({
                "kind": "constant RGB full-sphere analytic operator control",
                "diagnostic_role": "synthetic operator-only control; never a physical environment, meteorology, or product render source",
                "radiance_rgb": radiance.to_array(),
            }),
            Self::UpstreamHdr {
                path,
                width,
                height,
                file_sha256,
                decoded_sha256,
                ..
            } => serde_json::json!({
                "kind": "pinned upstream repository shipped full-sphere equirectangular HDR operator control",
                "diagnostic_role": "operator-only upstream source; not the GM1 production environment",
                "path": path.display().to_string(),
                "width": width,
                "height": height,
                "file_sha256": file_sha256,
                "decoded_rgba_f32le_sha256": decoded_sha256,
                "orientation": "row 0 north pole (+Z), longitude atan2(Y,X), horizontal wrap, manual bilinear",
            }),
            Self::Gm1TerrainLower {
                terrain_file,
                terrain_manifest_sha256,
                terrain_receipt_sha256,
                decoded_sha256,
                terrain_min_msl_m,
                terrain_max_msl_m,
                surface_dir,
                surface_skin_temperature_min_k,
                surface_skin_temperature_max_k,
                ..
            } => serde_json::json!({
                "kind": "physical SimSat upper sky plus source-backed GM1 terrain Lambertian lower hemisphere",
                "diagnostic_role": "candidate GM1 full-sphere display environment; direct sun remains separate",
                "upper_hemisphere": {
                    "radiance_payload_sha256": SIMSAT_RADIANCE_SHA256,
                    "atmosphere_source_sha256": SIMSAT_RADIANCE_SOURCE_SHA256,
                    "valid_time_utc": "2024-05-21T21:05:00Z",
                },
                "lower_hemisphere": {
                    "mapping": "orthographic hemisphere footprint: equirectangular direction XY maps to exact 300x300 terrain footprint; bilinear source normals; no procedural height/detail",
                    "resolution": [GM1_ENVIRONMENT_HEIGHT, GM1_ENVIRONMENT_WIDTH],
                    "decoded_rgba_f32le_sha256": decoded_sha256,
                    "terrain": {
                        "path": terrain_file.display().to_string(),
                        "payload_sha256": GM1_TERRAIN_SHA256,
                        "manifest_sha256": terrain_manifest_sha256,
                        "receipt_sha256": terrain_receipt_sha256,
                        "shape_yx": [GM1_TERRAIN_NY, GM1_TERRAIN_NX],
                        "dx_m": GM1_TERRAIN_DX_M,
                        "dy_m": GM1_TERRAIN_DY_M,
                        "minimum_msl_m": terrain_min_msl_m,
                        "maximum_msl_m": terrain_max_msl_m,
                    },
                    "land_surface": {
                        "path": surface_dir.display().to_string(),
                        "profile": "admitted HRRR f00 decoded 26x26 surface window; every cell land, zero snow depth, zero ice fraction",
                        "landsea_sha256": HRRR_LANDSEA_SHA256,
                        "snowh_sha256": HRRR_ZERO_SURFACE_SHA256,
                        "xice_sha256": HRRR_ZERO_SURFACE_SHA256,
                        "skintemp_sha256": HRRR_SKINTEMP_SHA256,
                        "skin_temperature_min_k": surface_skin_temperature_min_k,
                        "skin_temperature_max_k": surface_skin_temperature_max_k,
                        "skin_temperature_role": "provenance-only for visible-band environment; no invented visible thermal RGB",
                    },
                    "radiance_formula": "L_ground_rgb = SimSatPositiveSkyIrradiance(normal)_rgb * 0.3 / pi",
                    "albedo": SIMSAT_GROUND_ALBEDO,
                    "albedo_authority": "SimSat atmosphere.rs GROUND_ALBEDO physical multi-scatter ground-bounce default",
                    "diffuse_irradiance_sha256": T300_SKY_ROW_SHA256,
                    "diffuse_atmosphere_source_sha256": SIMSAT_ATMOSPHERE_SHA256,
                    "direct_sun": "excluded and kept separate",
                },
            }),
        }
    }
}

#[derive(Serialize)]
struct TimingReceipt {
    pipeline_compile_seconds: f64,
    collector_submit_wait_seconds: f64,
    matrix_accumulation_seconds: f64,
    sum_solve_copy_readback_seconds: f64,
    total_gpu_chain_wall_seconds: f64,
    collector_batches: u32,
}

#[derive(Serialize)]
struct PlatformReceipt {
    adapter_name: String,
    backend: String,
    driver: String,
    driver_info: String,
    max_compute_invocations_per_workgroup: u32,
    max_compute_workgroup_size_x: u32,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("gm1-view upstream-math glossy SHE: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let arguments = parse_arguments()?;
    let output_dir = arguments.output_dir;
    let training_alpha_levels = arguments.training_alpha_levels;
    fs::create_dir_all(&output_dir).map_err(|error| error.to_string())?;
    let total_started = Instant::now();
    let environment = EnvironmentData::load(arguments.environment)?;
    let physical_sky_words = physical_sky_words(SimSatSkyRadiance::sealed_t300()?);
    let environment_pixels = environment.gpu_pixels();
    let environment_dimensions = environment.dimensions();

    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    #[cfg(target_os = "windows")]
    {
        instance_descriptor.backends = wgpu::Backends::DX12;
    }
    #[cfg(not(target_os = "windows"))]
    {
        instance_descriptor.backends = wgpu::Backends::VULKAN;
    }
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .map_err(|error| format!("adapter request failed: {error}"))?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    if limits.max_compute_invocations_per_workgroup < 1024
        || limits.max_compute_workgroup_size_x < 1024
    {
        return Err(format!(
            "upstream 1024-thread collector unsupported: invocations={} size_x={}",
            limits.max_compute_invocations_per_workgroup, limits.max_compute_workgroup_size_x
        ));
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("GM1 View upstream-math glossy SHE"),
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .map_err(|error| format!("device request failed: {error}"))?;

    let compile_started = Instant::now();
    let shader_source = if training_alpha_levels == ALPHA_LEVELS.len() as u32 {
        Cow::Borrowed(WGSL_SOURCE)
    } else {
        Cow::Owned(WGSL_SOURCE.replacen(
            "const ALPHA_LEVELS: u32 = 4u;",
            &format!("const ALPHA_LEVELS: u32 = {training_alpha_levels}u;"),
            1,
        ))
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pinned upstream-math glossy SHE WGSL"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source.as_ref())),
    });
    let collector_pipeline = create_pipeline(&device, &shader, "collect_specular")?;
    let accumulator_pipeline = create_pipeline(&device, &shader, "accumulate_matrix")?;
    let sum_pipeline = create_pipeline(&device, &shader, "sum_matrix")?;
    let solver_pipeline = create_pipeline(&device, &shader, "solve_unregularized")?;
    let pipeline_compile_seconds = compile_started.elapsed().as_secs_f64();

    let physical_sky = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sealed SimSat physical sky radiance"),
        contents: bytemuck::cast_slice(&physical_sky_words),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let environment_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("full-sphere SHE source-isolation environment"),
        contents: bytemuck::cast_slice(environment_pixels.as_ref()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let collector_bytes = u64::from(COLLECTOR_RESOLUTION)
        * u64::from(COLLECTOR_RESOLUTION)
        * u64::from(training_alpha_levels)
        * 16;
    let collector = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("upstream-math 256x256 specular collector"),
        size: collector_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let collector_layout = collector_pipeline.get_bind_group_layout(0);
    let collector_batches = COLLECTOR_RESOLUTION / Y_BATCH;
    let collector_bindings = (0..collector_batches)
        .map(|batch| {
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("collector y batch"),
                contents: bytemuck::cast_slice(&[
                    batch * Y_BATCH,
                    environment.mode(),
                    environment_dimensions.0,
                    environment_dimensions.1,
                ]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("collector batch bindings"),
                layout: &collector_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: physical_sky.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: collector.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: environment_buffer.as_entire_binding(),
                    },
                ],
            });
            (params, bind_group)
        })
        .collect::<Vec<_>>();

    let collector_started = Instant::now();
    for (batch, (_, bind_group)) in collector_bindings.iter().enumerate() {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("upstream-math collector batch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("4096-sample GGX collector batch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&collector_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(COLLECTOR_RESOLUTION, Y_BATCH, training_alpha_levels);
        }
        queue.submit(Some(encoder.finish()));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| format!("collector batch {batch} wait failed: {error}"))?;
        eprintln!(
            "collector batch {}/{} complete ({:.3}s)",
            batch + 1,
            collector_batches,
            collector_started.elapsed().as_secs_f64(),
        );
    }
    let collector_seconds = collector_started.elapsed().as_secs_f64();

    let matrix_blocks = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("upstream-math AtA/Atb matrix blocks"),
        size: MATRIX_BLOCK_BYTES,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let accumulator_layout = accumulator_pipeline.get_bind_group_layout(1);
    let accumulator_bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("upstream-math matrix accumulator bindings"),
        layout: &accumulator_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: collector.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: matrix_blocks.as_entire_binding(),
            },
        ],
    });
    let matrix_started = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("upstream-math matrix accumulation"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("unregularized FP32 AtA/Atb accumulation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&accumulator_pipeline);
        pass.set_bind_group(1, &accumulator_bindings, &[]);
        pass.dispatch_workgroups(COLLECTOR_RESOLUTION, 16, 1);
    }
    queue.submit(Some(encoder.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| format!("matrix accumulation wait failed: {error}"))?;
    let matrix_accumulation_seconds = matrix_started.elapsed().as_secs_f64();

    let summed_matrix =
        storage_copy_buffer(&device, "summed 33x36 FP32 matrix", SUMMED_MATRIX_BYTES);
    let coefficients = storage_copy_buffer(
        &device,
        "33 RGB FP32 SHE coefficients",
        COEFFICIENT_BYTES_PADDED,
    );
    let solver_status = storage_copy_buffer(&device, "unregularized solver status", STATUS_BYTES);
    let sum_layout = sum_pipeline.get_bind_group_layout(2);
    let sum_bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("upstream-math final matrix sum bindings"),
        layout: &sum_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matrix_blocks.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: summed_matrix.as_entire_binding(),
            },
        ],
    });
    let solver_layout = solver_pipeline.get_bind_group_layout(3);
    let solver_bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("upstream-math unregularized solver bindings"),
        layout: &solver_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: summed_matrix.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: coefficients.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: solver_status.as_entire_binding(),
            },
        ],
    });
    let matrix_readback = readback_buffer(&device, "summed matrix readback", SUMMED_MATRIX_BYTES);
    let coefficient_readback =
        readback_buffer(&device, "coefficient readback", COEFFICIENT_BYTES_PADDED);
    let status_readback = readback_buffer(&device, "solver status readback", STATUS_BYTES);

    let solve_started = Instant::now();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("upstream-math sum, solve, and readback"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sum 4096 FP32 matrix blocks"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&sum_pipeline);
        pass.set_bind_group(2, &sum_bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("unregularized FP32 Cholesky solve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&solver_pipeline);
        pass.set_bind_group(3, &solver_bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&summed_matrix, 0, &matrix_readback, 0, SUMMED_MATRIX_BYTES);
    encoder.copy_buffer_to_buffer(
        &coefficients,
        0,
        &coefficient_readback,
        0,
        COEFFICIENT_BYTES_PADDED,
    );
    encoder.copy_buffer_to_buffer(&solver_status, 0, &status_readback, 0, STATUS_BYTES);
    queue.submit(Some(encoder.finish()));
    let matrix_bytes = map_readback(&device, &matrix_readback)?;
    let coefficient_bytes_padded = map_readback(&device, &coefficient_readback)?;
    let status_bytes = map_readback(&device, &status_readback)?;
    let solve_seconds = solve_started.elapsed().as_secs_f64();

    let solver_status_words = bytemuck::try_cast_slice::<u8, u32>(&status_bytes)
        .map_err(|error| format!("solver status layout mismatch: {error}"))?;
    if solver_status_words != [1, COLLECTOR_RESOLUTION, GGX_SAMPLES, training_alpha_levels] {
        return Err(format!(
            "unregularized WGSL solver refused status {solver_status_words:?}"
        ));
    }
    let gpu_coefficients = decode_coefficients(&coefficient_bytes_padded)?;
    let matrix = bytemuck::try_cast_slice::<u8, f32>(&matrix_bytes)
        .map_err(|error| format!("summed matrix layout mismatch: {error}"))?;
    let cpu_coefficients = solve_unregularized_fp32_cpu(matrix)?;
    let (solver_max_absolute, solver_max_ulp) =
        compare_coefficients(&cpu_coefficients, &gpu_coefficients)?;
    let held = held_error_for_radiance(
        &|direction| environment.evaluate(direction),
        &gpu_coefficients,
    )?;
    let admitted = held.normalized_rms_absolute_radiance <= MAX_NORMALIZED_RMS_RADIANCE
        && held.normalized_rms_luminance_absolute <= MAX_NORMALIZED_RMS_LUMINANCE;
    let coefficient_payload = specular_coefficients_bytes(&gpu_coefficients);
    let coefficient_sha256 = specular_coefficients_sha256(&gpu_coefficients);
    let matches_upstream_default = training_alpha_levels == ALPHA_LEVELS.len() as u32;
    let training_profile = if matches_upstream_default {
        "UPSTREAM_DEFAULT_4_ALPHA"
    } else {
        "GM1_QUALITY_EXTENSION_UPSTREAM_MATH"
    };
    let admission_status = match (admitted, matches_upstream_default) {
        (true, true) => "PASS_SOURCE_DIAGNOSTIC_UPSTREAM_DEFAULT_UNREGULARIZED_FP32",
        (false, true) => "REFUSED_SOURCE_DIAGNOSTIC_UPSTREAM_DEFAULT_HELD_ERROR",
        (true, false) => "PASS_UPSTREAM_MATH_ALPHA_EXTENSION_UNREGULARIZED_FP32",
        (false, false) => "REFUSED_UPSTREAM_MATH_ALPHA_EXTENSION_HELD_ERROR",
    };
    let coefficient_file = format!(
        "specular-she-upstream-math-{}-alpha{}-fp32.f32le",
        environment.slug(),
        training_alpha_levels,
    );
    let coefficient_path = output_dir.join(&coefficient_file);
    fs::write(&coefficient_path, &coefficient_payload).map_err(|error| error.to_string())?;
    let material_ab = if admitted {
        write_admitted_gm1_material_ab(&environment, &gpu_coefficients, &output_dir)?
    } else {
        serde_json::Value::Null
    };

    let receipt = serde_json::json!({
        "schema": "gm1.view-she-glossy-upstream-math-source-diagnostic/v1",
        "status": admission_status,
        "training_profile": training_profile,
        "claim_limit": "The four-alpha profile is the pinned upstream default. Any larger alpha count is an explicitly labeled GM1 quality extension that retains the pinned upstream math, 256x256 collector, 4096 GGX samples, and unregularized FP32 solve; it is not an upstream-default or paper/SOTA-equivalence claim. Direct sun, cloud-volume scattering, meteorological words, FP16/packed variants, and architecture-independent behavior are outside this receipt.",
        "admission": {
            "maximum_normalized_rms_radiance": MAX_NORMALIZED_RMS_RADIANCE,
            "maximum_normalized_rms_luminance": MAX_NORMALIZED_RMS_LUMINANCE,
            "old_gate_relaxed": false,
            "calibration_controls": {
                "constant_full_sphere": {
                    "receipt_sha256": "766b6ffa80578c42dc6dc70473ec90d3c43a746fea5b2a31b60a48c5887bdd8e",
                    "normalized_rms_radiance": 0.183954507112503_f64,
                    "normalized_rms_luminance": 0.183954298496246_f64,
                },
                "pinned_upstream_belfast_hdr": {
                    "receipt_sha256": "bffb018869648fce9f26b390939dc569a818286f3e5e894bf361d9cc6312ce32",
                    "normalized_rms_radiance": 0.203410595655441_f64,
                    "normalized_rms_luminance": 0.197730630636215_f64,
                },
            },
        },
        "upstream": {
            "project": UPSTREAM_PROJECT,
            "commit": UPSTREAM_COMMIT,
            "license": UPSTREAM_LICENSE,
            "collector_resolution": COLLECTOR_RESOLUTION,
            "ggx_samples": GGX_SAMPLES,
            "alpha_levels": training_alpha_values(training_alpha_levels),
            "default_alpha_levels": ALPHA_LEVELS,
            "matches_upstream_default": matches_upstream_default,
            "quality_extension": !matches_upstream_default,
            "quality_extension_dimension": if matches_upstream_default { serde_json::Value::Null } else { serde_json::json!("alpha training levels only") },
            "solver": "unregularized FP32 Cholesky",
            "prefilter_sha256": UPSTREAM_PREFILTER_SHA256,
            "accumulator_sha256": UPSTREAM_ACCUMULATOR_SHA256,
            "sum_sha256": UPSTREAM_SUM_SHA256,
            "solver_sha256": UPSTREAM_SOLVER_SHA256,
        },
        "environment": environment.receipt(),
        "platform": PlatformReceipt {
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            driver: info.driver,
            driver_info: info.driver_info,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
        },
        "implementation": {
            "wgsl_sha256": sha256_bytes(shader_source.as_bytes()),
            "executable_sha256": sha256_file(&env::current_exe().map_err(|error| error.to_string())?)?,
            "matrix_sha256": sha256_bytes(&matrix_bytes),
            "solver_status": solver_status_words,
            "solver_cpu_wgsl_max_absolute": solver_max_absolute,
            "solver_cpu_wgsl_max_ulp": solver_max_ulp,
        },
        "held": {
            "resolution_nv": HELD_RESOLUTION,
            "ggx_samples": HELD_GGX_SAMPLES,
            "alpha_levels": gm1_view::she_glossy_upstream::HELD_ALPHA_LEVELS,
            "compared_rgb_words": held.compared_rgb_words,
            "max_absolute_radiance": held.max_absolute_radiance,
            "rms_absolute_radiance": held.rms_absolute_radiance,
            "max_relative": held.max_relative,
            "rms_relative": held.rms_relative,
            "max_luminance_absolute": held.max_luminance_absolute,
            "rms_luminance_absolute": held.rms_luminance_absolute,
            "rms_luminance_relative": held.rms_luminance_relative,
            "reference_rms_radiance": held.reference_rms_radiance,
            "reference_rms_luminance": held.reference_rms_luminance,
            "normalized_rms_absolute_radiance": held.normalized_rms_absolute_radiance,
            "normalized_rms_luminance_absolute": held.normalized_rms_luminance_absolute,
            "admission_max_normalized_rms_radiance": MAX_NORMALIZED_RMS_RADIANCE,
            "admission_max_normalized_rms_luminance": MAX_NORMALIZED_RMS_LUMINANCE,
        },
        "output": {
            "file": coefficient_file,
            "layout": "33 consecutive RGB little-endian f32 coefficients in ConstructSHVectorA order",
            "nbytes": coefficient_payload.len(),
            "sha256": coefficient_sha256,
            "corruption_negative_control_detected": corruption_control(&coefficient_payload),
        },
        "material_ab": material_ab,
        "timing": TimingReceipt {
            pipeline_compile_seconds,
            collector_submit_wait_seconds: collector_seconds,
            matrix_accumulation_seconds,
            sum_solve_copy_readback_seconds: solve_seconds,
            total_gpu_chain_wall_seconds: total_started.elapsed().as_secs_f64(),
            collector_batches,
        },
    });
    let mut receipt_bytes = serde_json::to_vec(&receipt).map_err(|error| error.to_string())?;
    receipt_bytes.push(b'\n');
    let receipt_path = output_dir.join("receipt.json");
    fs::write(&receipt_path, &receipt_bytes).map_err(|error| error.to_string())?;
    println!(
        "{} coeff={} normalized_rms={:.6e} luminance={:.6e} gpu_chain={:.3}s receipt={} receipt_sha256={}",
        receipt["status"].as_str().unwrap_or("REFUSED"),
        coefficient_sha256,
        held.normalized_rms_absolute_radiance,
        held.normalized_rms_luminance_absolute,
        collector_seconds + matrix_accumulation_seconds + solve_seconds,
        receipt_path.display(),
        sha256_bytes(&receipt_bytes),
    );
    if admitted {
        Ok(())
    } else {
        Err(format!(
            "{training_profile} coefficients exceeded unchanged held normalized RMS admission"
        ))
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry: &str,
) -> Result<wgpu::ComputePipeline, String> {
    Ok(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        }),
    )
}

fn physical_sky_words(environment: SimSatSkyRadiance) -> [[f32; 4]; 17] {
    std::array::from_fn(|index| {
        let rgb = if index < 16 {
            environment.log_coefficients[index]
        } else {
            environment.amplitude
        };
        [rgb[0], rgb[1], rgb[2], 0.0]
    })
}

fn storage_copy_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn readback_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn map_readback(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, String> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| format!("device readback wait failed: {error}"))?;
    receiver
        .recv()
        .map_err(|error| format!("readback callback disappeared: {error}"))?
        .map_err(|error| format!("readback mapping failed: {error}"))?;
    let mapped = slice.get_mapped_range();
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(bytes)
}

fn decode_coefficients(bytes: &[u8]) -> Result<SheSpecularCoefficients, String> {
    let packed = bytemuck::try_cast_slice::<u8, [f32; 4]>(bytes)
        .map_err(|error| format!("coefficient output layout mismatch: {error}"))?;
    if packed.len() != 33 {
        return Err("coefficient output does not contain 33 padded RGB words".to_owned());
    }
    let coefficients = SheSpecularCoefficients(std::array::from_fn(|index| {
        Vec3::new(packed[index][0], packed[index][1], packed[index][2])
    }));
    if coefficients
        .0
        .iter()
        .any(|coefficient| !coefficient.is_finite())
        || packed.iter().any(|value| value[3].to_bits() != 0)
    {
        return Err("WGSL coefficient output is non-finite or padding changed".to_owned());
    }
    Ok(coefficients)
}

#[allow(clippy::needless_range_loop)] // fixed source-order Cholesky transcription
fn solve_unregularized_fp32_cpu(matrix: &[f32]) -> Result<SheSpecularCoefficients, String> {
    if matrix.len() != MATRIX_WORDS || matrix.iter().any(|value| !value.is_finite()) {
        return Err("summed FP32 matrix is malformed".to_owned());
    }
    let mut lower = [[0.0_f32; 33]; 33];
    let mut rhs = [[0.0_f32; 3]; 33];
    for row in 0..33 {
        rhs[row].copy_from_slice(&matrix[row * 36 + 33..row * 36 + 36]);
        for column in 0..=row {
            let mut value = matrix[row * 36 + column];
            for k in 0..column {
                value -= lower[row][k] * lower[column][k];
            }
            if row == column {
                lower[row][column] = value.max(0.0).sqrt();
            } else {
                lower[row][column] = value / lower[column][column];
            }
        }
    }
    for row in 0..33 {
        for channel in 0..3 {
            let mut value = rhs[row][channel];
            for k in 0..row {
                value -= lower[row][k] * rhs[k][channel];
            }
            rhs[row][channel] = value / lower[row][row];
        }
    }
    for row in (0..33).rev() {
        for channel in 0..3 {
            let mut value = rhs[row][channel];
            for k in row + 1..33 {
                value -= lower[k][row] * rhs[k][channel];
            }
            rhs[row][channel] = value / lower[row][row];
        }
    }
    let coefficients =
        SheSpecularCoefficients(std::array::from_fn(|index| Vec3::from_array(rhs[index])));
    if coefficients
        .0
        .iter()
        .any(|coefficient| !coefficient.is_finite())
    {
        return Err("independent CPU FP32 unregularized solve is non-finite".to_owned());
    }
    Ok(coefficients)
}

fn compare_coefficients(
    expected: &SheSpecularCoefficients,
    observed: &SheSpecularCoefficients,
) -> Result<(f32, u32), String> {
    let mut max_absolute = 0.0_f32;
    let mut max_ulp = 0_u32;
    for (a, b) in expected.0.iter().zip(&observed.0) {
        for channel in 0..3 {
            max_absolute = max_absolute.max((a[channel] - b[channel]).abs());
            max_ulp = max_ulp.max(ulp_distance(a[channel], b[channel]));
        }
    }
    if max_absolute > 5.0e-3 {
        return Err(format!(
            "independent CPU/WGSL solver coefficients diverged: max_abs={max_absolute:e} max_ulp={max_ulp}"
        ));
    }
    Ok((max_absolute, max_ulp))
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    fn ordered(value: f32) -> i32 {
        let bits = value.to_bits() as i32;
        if bits < 0 { i32::MIN - bits } else { bits }
    }
    ordered(a).abs_diff(ordered(b))
}

fn corruption_control(bytes: &[u8]) -> bool {
    let mut corrupt = bytes.to_vec();
    corrupt[0] ^= 1;
    sha256_bytes(&corrupt) != sha256_bytes(bytes)
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut args = env::args().skip(1);
    let mut output_dir = None;
    let mut environment_name = "simsat-upper-black".to_owned();
    let mut hdr_file = None;
    let mut terrain_file = None;
    let mut surface_dir = None;
    let mut training_alpha_levels = ALPHA_LEVELS.len() as u32;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--output-dir" => {
                output_dir = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--output-dir requires a path".to_owned())?,
                ));
            }
            "--environment" => {
                environment_name = args
                    .next()
                    .ok_or_else(|| "--environment requires a name".to_owned())?;
            }
            "--hdr-file" => {
                hdr_file = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--hdr-file requires a path".to_owned())?,
                ));
            }
            "--terrain-file" => {
                terrain_file =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--terrain-file requires a path".to_owned()
                    })?));
            }
            "--surface-dir" => {
                surface_dir = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--surface-dir requires a path".to_owned())?,
                ));
            }
            "--training-alpha-levels" => {
                training_alpha_levels = args
                    .next()
                    .ok_or_else(|| "--training-alpha-levels requires an integer".to_owned())?
                    .parse::<u32>()
                    .map_err(|error| format!("invalid --training-alpha-levels: {error}"))?;
            }
            _ => return Err(format!("unknown argument {argument}")),
        }
    }
    let output_dir = output_dir.ok_or_else(|| {
        "usage: gm1_view_she_glossy_upstream --output-dir <directory> [--environment simsat-upper-black|constant-full-sphere|upstream-hdr|gm1-terrain-lower] [--hdr-file <path>] [--terrain-file <path> --surface-dir <path>] [--training-alpha-levels 4..32]".to_owned()
    })?;
    if training_alpha_levels < ALPHA_LEVELS.len() as u32 || training_alpha_levels > 32 {
        return Err(format!(
            "--training-alpha-levels must preserve or exceed upstream default {} and be <=32",
            ALPHA_LEVELS.len()
        ));
    }
    let environment = match environment_name.as_str() {
        "simsat-upper-black" => {
            if hdr_file.is_some() || terrain_file.is_some() || surface_dir.is_some() {
                return Err("source file arguments are invalid for simsat-upper-black".to_owned());
            }
            EnvironmentRequest::SimSatUpperBlack
        }
        "constant-full-sphere" => {
            if hdr_file.is_some() || terrain_file.is_some() || surface_dir.is_some() {
                return Err("source file arguments are invalid for constant-full-sphere".to_owned());
            }
            EnvironmentRequest::ConstantFullSphere
        }
        "upstream-hdr" => {
            if terrain_file.is_some() || surface_dir.is_some() {
                return Err("terrain arguments are invalid for upstream-hdr".to_owned());
            }
            EnvironmentRequest::UpstreamHdr(
                hdr_file.ok_or_else(|| "upstream-hdr requires --hdr-file".to_owned())?,
            )
        }
        "gm1-terrain-lower" => {
            if hdr_file.is_some() {
                return Err("--hdr-file is invalid for gm1-terrain-lower".to_owned());
            }
            EnvironmentRequest::Gm1TerrainLower {
                terrain_file: terrain_file
                    .ok_or_else(|| "gm1-terrain-lower requires --terrain-file".to_owned())?,
                surface_dir: surface_dir
                    .ok_or_else(|| "gm1-terrain-lower requires --surface-dir".to_owned())?,
            }
        }
        _ => return Err(format!("unsupported --environment {environment_name}")),
    };
    Ok(Arguments {
        output_dir,
        environment,
        training_alpha_levels,
    })
}

fn training_alpha_values(levels: u32) -> Vec<f32> {
    (0..levels)
        .map(|index| 0.2 + 0.8 * index as f32 / (levels - 1) as f32)
        .collect()
}

fn read_pinned_f32(
    path: &PathBuf,
    expected_words: usize,
    expected_sha256: &str,
    label: &str,
) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    let sha256 = sha256_bytes(&bytes);
    if sha256 != expected_sha256 {
        return Err(format!(
            "{label} hash mismatch: expected {expected_sha256}, got {sha256}"
        ));
    }
    let words = bytemuck::try_cast_slice::<u8, f32>(&bytes)
        .map_err(|error| format!("{label} is not aligned little-endian f32: {error}"))?
        .to_vec();
    if words.len() != expected_words || words.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{label} must contain {expected_words} finite f32 words, got {}",
            words.len()
        ));
    }
    Ok(words)
}

fn build_gm1_terrain_lower_environment(terrain: &[f32]) -> Result<Vec<[f32; 4]>, String> {
    if terrain.len() != GM1_TERRAIN_NX * GM1_TERRAIN_NY {
        return Err("GM1 terrain shape changed before lower-environment build".to_owned());
    }
    let normals = build_gm1_terrain_normals(terrain)?;
    let mut pixels = vec![[0.0_f32; 4]; (GM1_ENVIRONMENT_WIDTH * GM1_ENVIRONMENT_HEIGHT) as usize];
    for y in 0..GM1_ENVIRONMENT_HEIGHT {
        let theta = std::f32::consts::PI * (y as f32 + 0.5) / GM1_ENVIRONMENT_HEIGHT as f32;
        let sin_theta = theta.sin();
        let z = theta.cos();
        if z > 0.0 {
            continue;
        }
        for x in 0..GM1_ENVIRONMENT_WIDTH {
            let longitude =
                std::f32::consts::TAU * ((x as f32 + 0.5) / GM1_ENVIRONMENT_WIDTH as f32 - 0.5);
            let direction = Vec3::new(sin_theta * longitude.cos(), sin_theta * longitude.sin(), z);
            let terrain_x = (direction.x * 0.5 + 0.5) * (GM1_TERRAIN_NX - 1) as f32;
            let terrain_y = (direction.y * 0.5 + 0.5) * (GM1_TERRAIN_NY - 1) as f32;
            let normal = sample_terrain_normal(&normals, terrain_x, terrain_y);
            // Terrain axes are east/north/up; the SimSat material helper takes
            // Bevy east/up/north and handles the sealed sun-relative frame.
            let irradiance = t300_reference_irradiance(Vec3::new(normal.x, normal.z, normal.y));
            let radiance = irradiance * (SIMSAT_GROUND_ALBEDO / std::f32::consts::PI);
            pixels[(y * GM1_ENVIRONMENT_WIDTH + x) as usize] =
                [radiance.x, radiance.y, radiance.z, 0.0];
        }
    }
    if pixels.iter().any(|pixel| {
        pixel[..3]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    }) || !pixels
        .iter()
        .any(|pixel| pixel[..3].iter().any(|value| *value > 0.0))
    {
        return Err("source-backed lower environment is non-finite, negative, or empty".to_owned());
    }
    Ok(pixels)
}

fn build_gm1_terrain_normals(terrain: &[f32]) -> Result<Vec<Vec3>, String> {
    if terrain.len() != GM1_TERRAIN_NX * GM1_TERRAIN_NY
        || terrain.iter().any(|value| !value.is_finite())
    {
        return Err(
            "GM1 terrain normal source must be the finite exact 300x300 carrier".to_owned(),
        );
    }
    let mut normals = vec![Vec3::ZERO; terrain.len()];
    for j in 0..GM1_TERRAIN_NY {
        for i in 0..GM1_TERRAIN_NX {
            let i0 = i.saturating_sub(1);
            let i1 = (i + 1).min(GM1_TERRAIN_NX - 1);
            let j0 = j.saturating_sub(1);
            let j1 = (j + 1).min(GM1_TERRAIN_NY - 1);
            let dzdx = (terrain[j * GM1_TERRAIN_NX + i1] - terrain[j * GM1_TERRAIN_NX + i0])
                / ((i1 - i0) as f32 * GM1_TERRAIN_DX_M);
            let dzdy = (terrain[j1 * GM1_TERRAIN_NX + i] - terrain[j0 * GM1_TERRAIN_NX + i])
                / ((j1 - j0) as f32 * GM1_TERRAIN_DY_M);
            normals[j * GM1_TERRAIN_NX + i] = Vec3::new(-dzdx, -dzdy, 1.0).normalize();
        }
    }
    Ok(normals)
}

fn sample_terrain_normal(normals: &[Vec3], x: f32, y: f32) -> Vec3 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(GM1_TERRAIN_NX - 1);
    let y1 = (y0 + 1).min(GM1_TERRAIN_NY - 1);
    let tx = x - x.floor();
    let ty = y - y.floor();
    let load = |i: usize, j: usize| normals[j * GM1_TERRAIN_NX + i];
    load(x0, y0)
        .lerp(load(x1, y0), tx)
        .lerp(load(x0, y1).lerp(load(x1, y1), tx), ty)
        .normalize()
}

fn write_admitted_gm1_material_ab(
    environment: &EnvironmentData,
    coefficients: &SheSpecularCoefficients,
    output_dir: &Path,
) -> Result<serde_json::Value, String> {
    let terrain_file = match environment {
        EnvironmentData::Gm1TerrainLower { terrain_file, .. } => terrain_file,
        _ => return Ok(serde_json::Value::Null),
    };
    let terrain = read_pinned_f32(
        terrain_file,
        GM1_TERRAIN_NX * GM1_TERRAIN_NY,
        GM1_TERRAIN_SHA256,
        "GM1 child terrain material source",
    )?;
    let normals = build_gm1_terrain_normals(&terrain)?;
    let width = GM1_TERRAIN_NX.div_ceil(MATERIAL_GRID_STRIDE);
    let height = GM1_TERRAIN_NY.div_ceil(MATERIAL_GRID_STRIDE);
    let mut sampled_height = Vec::with_capacity(width * height);
    let mut reference_rgb = Vec::with_capacity(width * height * 3);
    let mut she_rgb = Vec::with_capacity(width * height * 3);
    let mut absolute_error_rgb = Vec::with_capacity(width * height * 3);
    let view = Vec3::from_array(MATERIAL_VIEW_DIRECTION).normalize();
    let roughness = MATERIAL_ALPHA.sqrt();
    let base_albedo = Vec3::from_array(MATERIAL_BASE_ALBEDO);
    let mut sum_squared = 0.0_f64;
    let mut sum_squared_luminance = 0.0_f64;
    let mut max_absolute = 0.0_f32;
    let mut max_luminance_absolute = 0.0_f32;
    let luma = Vec3::new(0.2126, 0.7152, 0.0722);
    for j in (0..GM1_TERRAIN_NY).step_by(MATERIAL_GRID_STRIDE) {
        for i in (0..GM1_TERRAIN_NX).step_by(MATERIAL_GRID_STRIDE) {
            let index = j * GM1_TERRAIN_NX + i;
            let normal = normals[index];
            let ndotv = normal.dot(view);
            if ndotv <= 0.0 {
                return Err(format!(
                    "fixed material camera fell below exact terrain at ({i},{j})"
                ));
            }
            let reference_specular = prefilter_radiance_ggx(
                &|direction| environment.evaluate(direction),
                normal,
                view,
                MATERIAL_ALPHA,
                MATERIAL_REFERENCE_GGX_SAMPLES,
            );
            let she_specular = reconstruct_specular(normal, view, roughness, coefficients)?;
            let irradiance = t300_reference_irradiance(Vec3::new(normal.x, normal.z, normal.y));
            let diffuse = irradiance * base_albedo / std::f32::consts::PI;
            let fresnel =
                MATERIAL_DIELECTRIC_F0 + (1.0 - MATERIAL_DIELECTRIC_F0) * (1.0 - ndotv).powi(5);
            let reference = diffuse + reference_specular.max(Vec3::ZERO) * fresnel;
            let she = diffuse + she_specular.max(Vec3::ZERO) * fresnel;
            if !reference.is_finite() || !she.is_finite() || reference.min_element() < 0.0 {
                return Err(
                    "admitted terrain material response is non-finite or negative".to_owned(),
                );
            }
            let difference = (reference - she).abs();
            max_absolute = max_absolute.max(difference.max_element());
            let luminance_absolute = difference.dot(luma);
            max_luminance_absolute = max_luminance_absolute.max(luminance_absolute);
            sum_squared += f64::from(difference.length_squared());
            sum_squared_luminance += f64::from(luminance_absolute * luminance_absolute);
            sampled_height.push(terrain[index]);
            reference_rgb.extend_from_slice(&reference.to_array());
            she_rgb.extend_from_slice(&she.to_array());
            absolute_error_rgb.extend_from_slice(&difference.to_array());
        }
    }
    let compared_rgb_words = reference_rgb.len();
    let compared_pixels = sampled_height.len();
    let write = |name: &str, words: &[f32]| -> Result<serde_json::Value, String> {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let path = output_dir.join(name);
        fs::write(&path, &bytes)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        Ok(serde_json::json!({
            "file": name,
            "nbytes": bytes.len(),
            "sha256": sha256_bytes(&bytes),
        }))
    };
    Ok(serde_json::json!({
        "status": "PASS_ADMITTED_COEFFICIENTS_APPLIED_TO_EXACT_GM1_TERRAIN_NORMALS",
        "claim_limit": "Display-only damp/wet-soil dielectric treatment on exact GM1 terrain heights/normals. Surface reflectance is explicitly authored because the admitted source has no visible-band land-cover reflectance. It is not meteorological science, mapped road/water/vehicle geometry, or an upstream-default/SOTA claim.",
        "grid": {
            "source_shape_yx": [GM1_TERRAIN_NY, GM1_TERRAIN_NX],
            "source_stride": MATERIAL_GRID_STRIDE,
            "sampled_shape_yx": [height, width],
            "source_index_origin": [0, 0],
            "source_payload_sha256": GM1_TERRAIN_SHA256,
            "dx_m": GM1_TERRAIN_DX_M * MATERIAL_GRID_STRIDE as f32,
            "dy_m": GM1_TERRAIN_DY_M * MATERIAL_GRID_STRIDE as f32,
        },
        "material": {
            "class": "display-only damp/wet-soil dielectric",
            "base_albedo_rgb": MATERIAL_BASE_ALBEDO,
            "dielectric_f0": MATERIAL_DIELECTRIC_F0,
            "microfacet_alpha": MATERIAL_ALPHA,
            "perceptual_roughness": roughness,
            "view_direction_enu": view.to_array(),
            "reference_ggx_samples": MATERIAL_REFERENCE_GGX_SAMPLES,
            "reference_environment": "same sealed physical SimSat upper sky plus source-backed exact GM1 terrain lower hemisphere used by admission",
            "diffuse": "sealed SimSat positive log-SH3 sky irradiance times authored base albedo / pi",
            "direct_sun": "excluded from both sides",
            "display_transform": "none in payload; linear radiance",
        },
        "error": {
            "compared_pixels": compared_pixels,
            "compared_rgb_words": compared_rgb_words,
            "max_absolute_linear_radiance": max_absolute,
            "rms_absolute_linear_radiance": (sum_squared / compared_rgb_words as f64).sqrt(),
            "max_luminance_absolute_linear_radiance": max_luminance_absolute,
            "rms_luminance_absolute_linear_radiance": (sum_squared_luminance / compared_pixels as f64).sqrt(),
        },
        "payloads": {
            "terrain_msl_f32le": write("material-terrain-msl.f32le", &sampled_height)?,
            "reference_linear_rgb_f32le": write("material-reference-linear-rgb.f32le", &reference_rgb)?,
            "she_linear_rgb_f32le": write("material-she-linear-rgb.f32le", &she_rgb)?,
            "absolute_error_rgb_f32le": write("material-absolute-error-rgb.f32le", &absolute_error_rgb)?,
        },
    }))
}

fn validate_hdr(image: &Rgb32FImage) -> Result<(), String> {
    let (width, height) = image.dimensions();
    if width < 4 || height < 2 || width != 2 * height {
        return Err(format!(
            "upstream HDR must be 2:1 equirectangular, got {width}x{height}"
        ));
    }
    let mut positive = 0usize;
    for pixel in image.pixels() {
        for channel in pixel.0 {
            if !channel.is_finite() || channel < 0.0 {
                return Err("upstream HDR contains non-finite or negative radiance".to_owned());
            }
            positive += usize::from(channel > 0.0);
        }
    }
    if positive == 0 {
        return Err("upstream HDR contains no positive radiance".to_owned());
    }
    Ok(())
}

fn sample_equirectangular(width: u32, height: u32, pixels: &[[f32; 4]], direction: Vec3) -> Vec3 {
    let direction = direction.normalize();
    let longitude = direction.y.atan2(direction.x);
    let u = (longitude / std::f32::consts::TAU + 0.5).rem_euclid(1.0);
    let v = direction.z.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    let fx = u * width as f32 - 0.5;
    let fy = v * height as f32 - 0.5;
    let x0 = fx.floor() as i32;
    let y0 = (fy.floor() as i32).clamp(0, height as i32 - 1);
    let x1 = (x0 + 1).rem_euclid(width as i32) as u32;
    let x0 = x0.rem_euclid(width as i32) as u32;
    let y1 = (y0 + 1).min(height as i32 - 1) as u32;
    let y0 = y0 as u32;
    let tx = fx - fx.floor();
    let ty = fy - fy.floor();
    let load = |x: u32, y: u32| {
        Vec3::from_array(
            pixels[(y * width + x) as usize][0..3]
                .try_into()
                .expect("three RGB words"),
        )
    };
    let a = load(x0, y0).lerp(load(x1, y0), tx);
    let b = load(x0, y1).lerp(load(x1, y1), tx);
    a.lerp(b, ty)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &std::path::Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|error| format!("cannot hash {}: {error}", path.display()))
}
