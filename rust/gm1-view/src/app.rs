//! Bevy runtime for the real-data GM1 volume and scientific slice views.

use crate::{
    frame_geometry::load_frame_geometry,
    geofield::{
        GeoFieldDescriptor, GeoGrid, GeoreferencedWeatherAdapter, Gm1GeoreferencedAdapter,
    },
    hrrr_surround_terrain::{
        HRRR_SURROUND_ASSET_DIRECTORY, HRRR_SURROUND_NX, HRRR_SURROUND_NY,
        HRRR_SURROUND_WINDOW_I_START, HRRR_SURROUND_WINDOW_J_START, HRRR_WRFNAT_F02_SHA256,
        HrrrSurroundHeightSampler, load_source_backed_surround_assets,
        source_backed_surround_height_km_from_assets,
    },
    input::{Cli, Gm1Dataset, QualityProfile, f32_sha256},
    lighting::{
        T300_SKY_ROW_SHA256, T300_SOLAR_AZIMUTH_DEG, T300_SOLAR_ELEVATION_DEG,
        T300_SUN_ENU_F32_BITS, T300_VALID_TIME_UTC,
    },
    physical_atmosphere::{
        PHYSICAL_ATMOSPHERE_EXPOSURE, PHYSICAL_ATMOSPHERE_SOURCE_SHA256,
        PhysicalAtmosphereMaterial, PhysicalAtmospherePlugin,
    },
    production_playback::{
        EXACT_F32_BLEND_FORMULA, Gm1ViewSourceProfile, Gm1ViewTerrainClass, ProductionFramePair,
        ProductionRenderResidency, ProductionResidencyKind, ProductionSeriesPlayback,
        ProductionSeriesStartup, ProductionSeriesWatchEvent,
        cartesian_history_view_source_profile, gm1_view_source_profile_for_dataset_profile,
        validate_gm1_view_live_profile_transition,
    },
    science_layers::{
        ColumnLevelFaces, SEALED_SCALAR_LEVEL_COUNT, ScalarTopView, ScienceLayerAvailability,
        ScienceLayerId, ScienceLayerRegistry, TopViewProduct, scalar_top_view,
        tornado_surface_top_view,
        simsat_physical_top_view,
    },
    she::{UPSTREAM_COMMIT as SHE_UPSTREAM_COMMIT, UPSTREAM_PROJECT as SHE_UPSTREAM_PROJECT},
    she_admitted_terrain::{
        ADMITTED_MICROFACET_ALPHA, ADMITTED_PROFILE, ADMITTED_RECEIPT_SHA256,
        ADMITTED_SIGMA_SHA256, ADMITTED_SPECULAR_SHA256, ADMITTED_TERRAIN_NX, ADMITTED_TERRAIN_NY,
        ADMITTED_TERRAIN_SHA256, ADMITTED_TERRAIN_SHADER, ADMITTED_VERTICAL_GRID_SHA256,
        AdmittedTerrainMaterial, AdmittedTerrainMaterialPlugin,
        BELFAST_ALPHA32_REFUSAL_RECEIPT_SHA256, admitted_mass_height_km, admitted_terrain_mesh,
        admitted_terrain_values, admitted_volume_coordinates,
    },
    she_glossy::{SIMSAT_RADIANCE_SOURCE_SHA256, SPECULAR_HELD_GGX_SAMPLES},
    she_glossy_material::{
        GLOSSY_RADIANCE_SOURCE_SHA256, GlossyComparisonMode, PREVIEW_REFERENCE_GGX_SAMPLES,
        PROVISIONAL_SPECULAR_SHA256, PROVISIONAL_STATUS, SheGlossyMaterial,
        SheGlossyMaterialPlugin,
    },
    she_material::{
        SHE_DIFFUSE_FACE_RESOLUTION, SHE_SKY_SOURCE_SHA256, SheSkyMaterial, SheSkyMaterialPlugin,
        project_t300_simsat_diffuse_l2, t300_projection_error,
    },
    slice::{
        Segment, contours, is_reflectivity_units, nws_dbz_range, nws_dbz_rgba,
        quantile_range, scientific_rgba,
    },
    volume::{
        DevProofMaterial, Gm1VolumeMaterial, Gm1VolumePlugin, ONSET_GRID_OCCUPANCY, VolumeUniform,
        storage_from_field, storage_from_production_resident, unused_coordinate_storage,
    },
    watch::{WatchEvent, scan_pack_timeline, spawn_watch},
};
use bevy::{
    app::AppExit,
    asset::AssetPlugin,
    camera::{
        Exposure, Hdr,
        visibility::{NoCpuCulling, NoFrustumCulling},
    },
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    math::DVec3,
    prelude::*,
    render::{
        RenderPlugin,
        settings::{Backends, WgpuSettings},
        storage::ShaderBuffer,
        view::{
            NoIndirectDrawing,
            screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
        },
    },
    window::{PresentMode, WindowResolution},
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui, input::EguiWantsInput};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, Sender, channel},
    },
    time::Instant,
};

/// Scene box of the sealed terrain-following child: the exact model top and
/// the minimum sealed `terrain_msl` value as its conservative ray box.
/// Individual mass levels are placed by the exact terrain-composed sigma
/// profile in the shader. This is the box for that one source profile, not a
/// property of the program, and it is only used when no frame geometry has
/// been installed.
const SEALED_CHILD_WORLD_MIN: Vec3 = Vec3::new(-7.5, 0.306_059_57, -7.5);
const SEALED_CHILD_WORLD_SIZE: Vec3 = Vec3::new(15.0, 14.693_94, 15.0);

/// Physical extent of the rendered scene, in kilometres.
#[derive(Clone, Copy, Debug, PartialEq)]
struct WorldBox {
    min: Vec3,
    size: Vec3,
}

/// The box the current run draws into. One process renders one grid, so this
/// is installed once from that frame's admitted geometry before the render app
/// starts. A volume drawn at the wrong physical size makes every distance,
/// speed, height and camera framing the viewer reports wrong by that factor,
/// so the box is never inferred from a shape and never changed mid-run.
static WORLD_BOX: std::sync::OnceLock<WorldBox> = std::sync::OnceLock::new();

/// Scene box for an admitted georeferenced grid. `floor_km` is the lowest
/// surface in the domain; the vertical span runs from there to the model top.
fn world_box_for_grid(grid: &GeoGrid, floor_km: f32) -> WorldBox {
    let width_km = (grid.nx as f64 * grid.dx_m / 1_000.0) as f32;
    let depth_km = (grid.ny as f64 * grid.dy_m / 1_000.0) as f32;
    let top_km = (grid.model_top_m / 1_000.0) as f32;
    WorldBox {
        min: Vec3::new(-0.5 * width_km, floor_km, -0.5 * depth_km),
        size: Vec3::new(width_km, top_km - floor_km, depth_km),
    }
}

/// Install the scene box for this run. A second, different install is refused:
/// the already-built meshes, camera frames and volume uniforms carry the first
/// box, so accepting a second would draw one frame's words inside another
/// frame's physical bounds.
fn install_world_box(box_km: WorldBox) -> Result<(), String> {
    match WORLD_BOX.set(box_km) {
        Ok(()) => Ok(()),
        Err(_) => {
            let installed = *WORLD_BOX.get().expect("world box was just observed as set");
            if installed == box_km {
                Ok(())
            } else {
                Err(format!(
                    "scene box {box_km:?} cannot replace the installed {installed:?}; \
                     the meshes, camera and volume uniform already built from the \
                     first box would draw this frame's words inside another \
                     frame's physical bounds"
                ))
            }
        }
    }
}

/// Renderer metadata for the Cartesian history frame this run is drawing.
///
/// The static profile table can only describe grids known when the viewer was
/// built. A Cartesian history frame states its own geometry beside the
/// envelope, so its renderer metadata is resolved once at startup and held
/// here for the systems that read a profile per frame.
static RUNTIME_SOURCE_PROFILE: std::sync::OnceLock<Gm1ViewSourceProfile> =
    std::sync::OnceLock::new();

/// Install the run-time-resolved source profile. A second, different install
/// is refused: the scene is already built for the first profile's grid, so a
/// later one would label and place this frame by another frame's geometry.
fn install_runtime_source_profile(profile: Gm1ViewSourceProfile) -> Result<(), String> {
    match RUNTIME_SOURCE_PROFILE.set(profile) {
        Ok(()) => Ok(()),
        Err(_) => {
            let installed = *RUNTIME_SOURCE_PROFILE
                .get()
                .expect("runtime source profile was just observed as set");
            if installed == profile {
                Ok(())
            } else {
                Err(format!(
                    "source profile {} cannot replace the installed {}; the scene \
                     is already built for the first profile's grid, so the frame \
                     would be placed and labelled by another frame's geometry",
                    profile.domain_label(),
                    installed.domain_label()
                ))
            }
        }
    }
}

/// Renderer metadata for an admitted dataset profile: the sealed static table
/// first, then the geometry this run bound from the frame's own publication
/// document. Neither admits source bytes; the strict loaders already ran.
fn resolved_source_profile(profile_id: &str) -> Option<Gm1ViewSourceProfile> {
    if let Some(profile) = gm1_view_source_profile_for_dataset_profile(profile_id) {
        return Some(profile);
    }
    RUNTIME_SOURCE_PROFILE
        .get()
        .copied()
        .filter(|profile| profile_id.rsplit(';').next().unwrap_or(profile_id) == profile.id)
}

/// Lower corner of the scene box, in kilometres.
fn world_min() -> Vec3 {
    WORLD_BOX.get().map_or(SEALED_CHILD_WORLD_MIN, |value| value.min)
}

/// Extent of the scene box, in kilometres.
fn world_size() -> Vec3 {
    WORLD_BOX
        .get()
        .map_or(SEALED_CHILD_WORLD_SIZE, |value| value.size)
}
const DIAGNOSTIC_FIXED_CAMERA: bool = false;
const DEBUG_TERRAIN_GROUND_PROBE: bool = false;
// 1: standard magenta cube; 2: explicit AABB/grid; 3: minimal custom material;
// 4: exact raymarch.
const DEV_PROOF_STAGE: u8 = 4;

#[derive(Resource)]
struct ViewerData(Arc<Gm1Dataset>);

#[derive(Resource)]
struct ProductionViewData(Arc<ProductionRenderResidency>);

#[derive(Resource)]
struct ProductionSeriesLiveState {
    receiver: Mutex<Receiver<ProductionSeriesWatchEvent>>,
    root: PathBuf,
    playback: ProductionSeriesPlayback,
    status: String,
    last_refusal: Option<String>,
    rendered_alpha_f32_bits: u32,
}

#[derive(Clone)]
struct ProductionGpuField {
    descriptor: GeoFieldDescriptor,
    stagger: String,
    residency: ProductionResidencyKind,
    nbytes: usize,
    values: Arc<Vec<f32>>,
    statistics: ExactFieldStatistics,
    _buffer: Handle<ShaderBuffer>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ExactFieldStatistics {
    word_count: usize,
    nonzero_count: usize,
    minimum: f32,
    maximum: f32,
}

fn exact_field_statistics(values: &[f32]) -> Result<ExactFieldStatistics, String> {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut nonzero_count = 0usize;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "exact field inspector word {index} is non-finite: 0x{:08x}",
                value.to_bits(),
            ));
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        nonzero_count += usize::from(value.to_bits() & 0x7fff_ffff != 0);
    }
    if values.is_empty() {
        return Err("exact field inspector refuses an empty field".to_owned());
    }
    Ok(ExactFieldStatistics {
        word_count: values.len(),
        nonzero_count,
        minimum,
        maximum,
    })
}

fn upload_production_gpu_fields(
    production: &ProductionRenderResidency,
    buffers: &mut Assets<ShaderBuffer>,
) -> Result<Vec<ProductionGpuField>, String> {
    production
        .fields
        .iter()
        .map(|field| {
            let buffer = buffers.add(storage_from_production_resident(field)?);
            Ok(ProductionGpuField {
                descriptor: field.descriptor.clone(),
                stagger: field.stagger.clone(),
                residency: field.residency,
                nbytes: field.values.len() * std::mem::size_of::<f32>(),
                values: field.values.clone(),
                statistics: exact_field_statistics(&field.values)?,
                _buffer: buffer,
            })
        })
        .collect()
}

fn exact_field_level(field: &ProductionGpuField, level: usize) -> Result<&[f32], String> {
    match field.descriptor.shape.as_slice() {
        [ny, nx] => {
            if level != 0 || field.values.len() != ny * nx {
                return Err("2-D exact field inspector level/shape changed".to_owned());
            }
            Ok(field.values.as_slice())
        }
        [nz, ny, nx] => {
            if level >= *nz || field.values.len() != nz * ny * nx {
                return Err("3-D exact field inspector level/shape changed".to_owned());
            }
            let plane_words = ny * nx;
            Ok(&field.values[level * plane_words..(level + 1) * plane_words])
        }
        shape => Err(format!(
            "exact field inspector refuses unsupported shape {shape:?}"
        )),
    }
}

#[derive(Resource, Default)]
struct ProductionInspectorState {
    selected_field: usize,
    selected_level: usize,
}

impl ProductionInspectorState {
    fn clamp_to(&mut self, fields: &[ProductionGpuField]) {
        self.selected_field = self.selected_field.min(fields.len().saturating_sub(1));
        let levels = fields
            .get(self.selected_field)
            .and_then(|field| {
                (field.descriptor.shape.len() == 3).then_some(field.descriptor.shape[0])
            })
            .unwrap_or(1);
        self.selected_level = self.selected_level.min(levels.saturating_sub(1));
    }

    fn select_field(&mut self, field: usize, fields: &[ProductionGpuField]) {
        self.selected_field = field;
        self.selected_level = 0;
        self.clamp_to(fields);
    }
}

#[derive(Default)]
struct ProductionInspectorCache {
    field_sha256: String,
    level: usize,
    level_sha256: String,
    statistics: Option<ExactFieldStatistics>,
}

#[derive(Resource)]
struct ProductionGpuResidency {
    adapter_id: &'static str,
    source_profile: Gm1ViewSourceProfile,
    manifest_sha256: String,
    payload_set_sha256: String,
    fields: Vec<ProductionGpuField>,
}

#[derive(Resource)]
struct LiveWatchState {
    receiver: Mutex<Receiver<WatchEvent>>,
    root: PathBuf,
    status: String,
    last_refusal: Option<String>,
}

#[derive(Resource)]
struct ScienceViewState {
    selected: ScienceLayerId,
    show: bool,
    requested: Option<(u64, ScienceLayerId)>,
    rendered: Option<ScienceRaster>,
    status: String,
    sender: Sender<ScienceWorkerEvent>,
    receiver: Mutex<Receiver<ScienceWorkerEvent>>,
    root: Option<PathBuf>,
}

#[derive(Debug)]
struct ScienceWorkerEvent {
    step: u64,
    layer: ScienceLayerId,
    result: Result<ScienceRaster, String>,
}

#[derive(Clone, Debug)]
struct ScienceRaster {
    step: u64,
    layer: ScienceLayerId,
    width: usize,
    height: usize,
    rgba: Vec<u8>,
    rgba_sha256: String,
    descriptor: GeoFieldDescriptor,
    range: [f32; 2],
    summary: String,
}

#[derive(Default)]
struct ScienceTextureCache {
    key: Option<(u64, ScienceLayerId, String)>,
    texture: Option<egui::TextureHandle>,
}

#[derive(Resource)]
struct ViewerState {
    selected_field: usize,
    loaded_field: usize,
    selected_slice: usize,
    show_slice: bool,
    fly_mode: bool,
    material: Option<Handle<Gm1VolumeMaterial>>,
    camera_summary: String,
    build_summary: String,
    she_comparison: bool,
    she_glossy_comparison: bool,
    admitted_she_terrain: bool,
    lighting_reference_diagnostic: bool,
    lighting_readback_diagnostic: bool,
    presentation: bool,
    drive_mode: bool,
    quality: QualityProfile,
    she_summary: Option<String>,
}

fn interactive_input_allowed(capture_active: bool, egui_wants_input: bool) -> bool {
    !capture_active && !egui_wants_input
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SceneInputPermissions {
    pointer: bool,
    keyboard: bool,
}

fn scene_input_permissions(
    capture_active: bool,
    egui_wants_pointer: bool,
    egui_wants_keyboard: bool,
) -> SceneInputPermissions {
    // Treat any active UI interaction as ownership of the whole scene-input
    // chord. In particular, holding W while dragging an egui slider must not
    // move the camera, and timeline shortcuts must not fire under a click.
    let allowed =
        interactive_input_allowed(capture_active, egui_wants_pointer || egui_wants_keyboard);
    SceneInputPermissions {
        pointer: allowed,
        keyboard: allowed,
    }
}

fn offline_capture_requested(path: Option<&Path>) -> bool {
    path.is_some()
}

const FRAME_PROFILE_CAPACITY: usize = 16_384;
const PROFILE_WARMUP_SECONDS: f32 = 5.0;

#[derive(Resource)]
struct FrameProfiler {
    samples_ms: Box<[f32; FRAME_PROFILE_CAPACITY]>,
    sample_count: usize,
    next_sample: usize,
    elapsed_seconds: f32,
    ema_frame_ms: f32,
    receipt_path: Option<PathBuf>,
    soak_seconds: f32,
    written: bool,
    sampled_quality: Option<QualityProfile>,
    mixed_quality: bool,
    adapter: Option<ProfileAdapter>,
}

impl FrameProfiler {
    fn new(
        receipt_path: Option<PathBuf>,
        soak_seconds: u32,
        adapter: Option<ProfileAdapter>,
    ) -> Self {
        Self {
            samples_ms: Box::new([0.0; FRAME_PROFILE_CAPACITY]),
            sample_count: 0,
            next_sample: 0,
            elapsed_seconds: 0.0,
            ema_frame_ms: 0.0,
            receipt_path,
            soak_seconds: soak_seconds as f32,
            written: false,
            sampled_quality: None,
            mixed_quality: false,
            adapter,
        }
    }

    fn record(&mut self, delta_seconds: f32, quality: QualityProfile) {
        self.elapsed_seconds += delta_seconds;
        let frame_ms = delta_seconds * 1_000.0;
        self.ema_frame_ms = if self.ema_frame_ms == 0.0 {
            frame_ms
        } else {
            self.ema_frame_ms.lerp(frame_ms, 0.05)
        };
        if self.elapsed_seconds < PROFILE_WARMUP_SECONDS || !frame_ms.is_finite() {
            return;
        }
        match self.sampled_quality {
            None => self.sampled_quality = Some(quality),
            Some(sampled) if sampled != quality => self.mixed_quality = true,
            Some(_) => {}
        }
        self.samples_ms[self.next_sample] = frame_ms;
        self.next_sample = (self.next_sample + 1) % FRAME_PROFILE_CAPACITY;
        self.sample_count = (self.sample_count + 1).min(FRAME_PROFILE_CAPACITY);
    }

    fn fps(&self) -> f32 {
        if self.ema_frame_ms > 0.0 {
            1_000.0 / self.ema_frame_ms
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug)]
struct ProfileAdapter {
    name: String,
    backend: String,
    vendor: u32,
    device: u32,
    driver: String,
    driver_info: String,
}

fn probe_dx12_adapter() -> Result<ProfileAdapter, String> {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::DX12;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .map_err(|error| format!("DX12 profile adapter request failed: {error}"))?;
    let info = adapter.get_info();
    if info.backend != wgpu::Backend::Dx12 {
        return Err(format!(
            "DX12 profile adapter resolved unexpected backend {:?}",
            info.backend
        ));
    }
    Ok(ProfileAdapter {
        name: info.name,
        backend: format!("{:?}", info.backend),
        vendor: info.vendor,
        device: info.device,
        driver: info.driver,
        driver_info: info.driver_info,
    })
}

#[derive(Serialize)]
struct FrameProfileReceipt<'a> {
    schema: &'static str,
    status: &'a str,
    quality: &'a str,
    quality_stable: bool,
    backend_required: &'static str,
    backend_actual: &'a str,
    adapter_name: &'a str,
    adapter_vendor: u32,
    adapter_device: u32,
    adapter_driver: &'a str,
    adapter_driver_info: &'a str,
    warmup_seconds: f32,
    soak_seconds: f32,
    samples: usize,
    mean_frame_ms: f32,
    median_frame_ms: f32,
    p95_frame_ms: f32,
    p99_frame_ms: f32,
    worst_frame_ms: f32,
    mean_fps: f32,
    exact_step: u64,
    exact_payload_sha256: &'a str,
    target: &'static str,
    scientific_interpolation: bool,
    per_frame_readback: bool,
}

const TIMELINE_CAPACITY: usize = 12;
const TIMELINE_PERIOD_SECONDS: f32 = 1.25;
const DEFAULT_FLY_MODE: bool = true;
const DEFAULT_TIMELINE_PLAYING: bool = false;
const DEFAULT_TIMELINE_LOOPING: bool = false;
const DEFAULT_FOLLOW_LIVE: bool = false;
const SCIENCE_PANEL_ID: &str = "gm1-science-panel";
const SCIENCE_TOP_VIEW_ID: &str = "gm1-science-top-view";
const SCIENCE_SLICE_WINDOW_ID: &str = "gm1-scientific-horizontal-slice";
const SHELL_FRAME_SECTION_ID: &str = "gm1-shell-frame";
const SHELL_RENDER_SECTION_ID: &str = "gm1-shell-render";
const SHELL_FIELDS_SECTION_ID: &str = "gm1-shell-fields";
const SHELL_SCIENCE_SECTION_ID: &str = "gm1-shell-science";
const SHELL_VEHICLE_SECTION_ID: &str = "gm1-shell-experimental-vehicle";
const SHELL_PROVENANCE_SECTION_ID: &str = "gm1-shell-provenance";
const SCIENCE_PANEL_WIDTH: f32 = 352.0;
const SCIENCE_PANEL_MARGIN: f32 = 8.0;
const SCIENCE_PANEL_MAX_HEIGHT: f32 = 660.0;
const FLY_SPEED_KM_S: f32 = 1.5;

#[derive(Default)]
struct ScientificShellUiCache {
    visuals_initialized: bool,
}

fn install_scientific_shell_visuals_once(
    ctx: &egui::Context,
    cache: &mut ScientificShellUiCache,
) -> bool {
    if cache.visuals_initialized {
        return false;
    }
    let mut visuals = egui::Visuals::light();
    visuals.panel_fill = egui::Color32::from_rgb(247, 246, 242);
    visuals.window_fill = egui::Color32::from_rgba_unmultiplied(250, 249, 246, 242);
    visuals.override_text_color = Some(egui::Color32::from_rgb(30, 36, 43));
    ctx.set_visuals(visuals);
    cache.visuals_initialized = true;
    true
}

fn science_texture_cache_requires_update(
    cached: Option<&(u64, ScienceLayerId, String)>,
    step: u64,
    layer: ScienceLayerId,
    rgba_sha256: &str,
) -> bool {
    !matches!(
        cached,
        Some((cached_step, cached_layer, cached_sha256))
            if *cached_step == step
                && *cached_layer == layer
                && cached_sha256 == rgba_sha256
    )
}

fn science_panel_height(content_height: f32) -> f32 {
    (content_height - 2.0 * SCIENCE_PANEL_MARGIN).clamp(180.0, SCIENCE_PANEL_MAX_HEIGHT)
}

fn timeline_controls_available(frame_count: usize) -> bool {
    frame_count > 1
}

fn live_controls_available(frame_count: usize, has_live_watcher: bool) -> bool {
    timeline_controls_available(frame_count) && has_live_watcher
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InteractiveDefaults {
    fly_mode: bool,
    timeline_playing: bool,
    timeline_looping: bool,
    follow_live: bool,
}

impl Default for InteractiveDefaults {
    fn default() -> Self {
        Self {
            fly_mode: DEFAULT_FLY_MODE,
            timeline_playing: DEFAULT_TIMELINE_PLAYING,
            timeline_looping: DEFAULT_TIMELINE_LOOPING,
            follow_live: DEFAULT_FOLLOW_LIVE,
        }
    }
}

#[derive(Clone, Debug)]
struct TimelineFrame {
    dataset: Arc<Gm1Dataset>,
    source_gate: String,
}

#[derive(Resource)]
struct TimelineState {
    frames: VecDeque<TimelineFrame>,
    current: usize,
    requested: Option<usize>,
    playing: bool,
    looping: bool,
    follow_live: bool,
    elapsed: f32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum VehicleCameraMode {
    Cockpit,
    #[default]
    Chase,
}

#[derive(Component, Debug)]
struct ChaseVehicle {
    east_north_km: Vec2,
    yaw: f32,
    speed_km_s: f32,
    reset_east_north_km: Vec2,
    reset_yaw: f32,
    camera_mode: VehicleCameraMode,
    grounded: bool,
    terrain_status: String,
}

#[derive(Component)]
struct ChaseVehicleVisual;

#[derive(Resource)]
struct DriveTerrain {
    child_msl_m: Vec<f32>,
    surround: HrrrSurroundHeightSampler,
}

#[derive(Resource)]
struct CaptureState {
    path: Option<PathBuf>,
    frames: u32,
    requested: bool,
    attempts: u32,
    phase: CapturePhase,
    phase_started: u32,
    dense_mask: Option<CloudMaskReadback>,
    cloud_mask: Option<CloudMaskReadback>,
    terrain_mask: Option<CloudMaskReadback>,
    dominant_projection: Option<DominantComponentProjection>,
}

#[derive(Clone, Copy, Debug, Default)]
struct DominantComponentProjection {
    mass_fraction_in_frame: f64,
    minimum_ndc: Vec2,
    maximum_ndc: Vec2,
    width_fraction: f32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum CapturePhase {
    #[default]
    InitialWarmup,
    DenseMaskWarmup,
    PrepareCloudMask,
    CloudMaskWarmup,
    PrepareTerrainMask,
    TerrainMaskWarmup,
    RestorePresentation,
    PresentationWarmup,
}

#[derive(Component)]
struct AdmittedTerrainSurface;

#[derive(Component)]
struct ExactVolumeSurface;

#[derive(Component)]
struct PhysicalAtmosphereSurface;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaptureRenderPass {
    DenseCloudMask,
    CloudMask,
    TerrainMask,
    Presentation,
}

#[derive(Component)]
struct ViewerCamera {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32,
}

#[derive(Clone, Debug)]
struct FieldFrame {
    focus: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32,
    cutoff: f32,
    /// Condensate value at which the display treats a sample as fully dense
    /// cloud rather than onset haze. Taken from the cells that actually carry
    /// cloud, not from a whole-domain quantile.
    dense_response_knee: f32,
    dense_count: usize,
    dense_min: Vec3,
    dense_max: Vec3,
    visible_count: usize,
    visible_min: Vec3,
    visible_max: Vec3,
    dominant_indices: Vec<usize>,
    dominant_mass: f64,
    connected_mass: f64,
    dominant_mass_fraction: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ComponentMorphology {
    grid_min: UVec3,
    grid_max: UVec3,
    bbox_cell_count: usize,
    voxel_compactness: f64,
    lateral_boundary_distance_cells: usize,
    vertical_extent_km: f32,
    maximum_horizontal_extent_km: f32,
    vertical_to_horizontal_ratio: f32,
}

const DENSE_FRAME_MIN_CUTOFF: f32 = 1.0e-3;
const PHYSICAL_CLOUD_ONSET: f32 = 1.0e-4;
// With the camera position reconstructed as focus - forward * radius, a
// negative Bevy X pitch places the eye above the focus and looks down into the
// volume. The previous positive sign put the eye below the cloud deck.
const CAMERA_PITCH: f32 = -0.55;
const CHASE_EYE_ABOVE_TERRAIN_KM: f32 = 0.100;
const CHASE_VERTICAL_FOV_DEGREES: f32 = 77.0;
const PRESENTATION_ASPECT: f32 = 16.0 / 9.0;
const CHASE_DENSE_CLEARANCE_KM: f32 = 0.50;
const CHASE_MIN_STANDOFF_KM: f32 = 4.0;
const CHASE_MAX_STANDOFF_KM: f32 = 18.0;

#[derive(Default)]
struct SliceCache {
    field_hash: String,
    level: usize,
    texture: Option<egui::TextureHandle>,
    segments: Vec<Segment>,
    range: (f32, f32),
}

pub fn run(cli: Cli, dataset: Gm1Dataset) -> Result<(), String> {
    run_internal(cli, dataset, None, None)
}

pub fn run_production(
    cli: Cli,
    dataset: Gm1Dataset,
    production: ProductionRenderResidency,
) -> Result<(), String> {
    validate_production_view_binding(&cli, &dataset, &production)?;
    run_internal(cli, dataset, Some(Arc::new(production)), None)
}

pub fn run_production_series(
    cli: Cli,
    dataset: Gm1Dataset,
    production: ProductionRenderResidency,
    series: ProductionSeriesStartup,
) -> Result<(), String> {
    validate_production_view_binding(&cli, &dataset, &production)?;
    if series.playback.current().step != dataset.step
        || series.playback.current().manifest_sha256 != dataset.artifact_sha256
        || series.root
            != cli
                .watch_production_series
                .as_deref()
                .unwrap_or(Path::new(""))
    {
        return Err(
            "production series startup and exact renderer endpoint are not identical".to_owned(),
        );
    }
    run_internal(cli, dataset, Some(Arc::new(production)), Some(series))
}

pub fn validate_production_view_binding(
    cli: &Cli,
    dataset: &Gm1Dataset,
    production: &ProductionRenderResidency,
) -> Result<(), String> {
    let production_sources = usize::from(cli.production_pack.is_some())
        + usize::from(cli.watch_production_series.is_some());
    if production_sources != 1
        || cli.input.is_some()
        || cli.watch_dir.is_some()
        || cli.verify_production_pack.is_some()
    {
        return Err(
            "production renderer requires one exclusive --production-pack or --watch-production-series source"
                .to_owned(),
        );
    }
    validate_production_renderer_envelope(dataset, production)
}

fn validate_production_renderer_envelope(
    dataset: &Gm1Dataset,
    production: &ProductionRenderResidency,
) -> Result<(), String> {
    if production.frame.time.step != dataset.step
        || production.frame.time.model_time_s.to_bits() != dataset.model_time_s.to_bits()
        || production.manifest_sha256 != dataset.artifact_sha256
        || production.payload_set_sha256 != dataset.payload_sha256
        || production.frame.source_identity != dataset.payload_sha256
        || production.frame.grid != production.source_profile.georeferenced_grid()
        || !dataset.profile_id.ends_with(production.source_profile.id)
        || production.frame.fields.len() != 8
        || production.fields.len() != 8
    {
        return Err(
            "production renderer frame/dataset envelope changed after admission".to_owned(),
        );
    }
    let cloud = production
        .field("cloud_condensate_kgkg")
        .ok_or_else(|| "production renderer lacks exact cloud residency".to_owned())?;
    let [dataset_cloud] = dataset.fields.as_slice() else {
        return Err("production renderer dataset must expose only exact qc+qi".to_owned());
    };
    if dataset_cloud.id != "cloud_condensate"
        || cloud.residency != ProductionResidencyKind::CloudScalarVolume
        || cloud.descriptor.shape != dataset_cloud.shape
        || cloud.descriptor.values_sha256 != dataset_cloud.values_sha256
        || cloud.values.as_slice() != dataset_cloud.values.as_slice()
    {
        return Err("production renderer cloud carrier changed after admission".to_owned());
    }
    if production.fields.iter().any(|field| {
        matches!(
            field.descriptor.id.as_str(),
            "reflectivity_dbz" | "vertical_vorticity_s"
        )
    }) {
        return Err("production renderer cannot bind an unadmitted derived carrier".to_owned());
    }
    Ok(())
}

fn run_internal(
    cli: Cli,
    dataset: Gm1Dataset,
    production: Option<Arc<ProductionRenderResidency>>,
    production_series: Option<ProductionSeriesStartup>,
) -> Result<(), String> {
    let screenshot = cli.screenshot.clone();
    let she_comparison = cli.she_comparison;
    let she_glossy_comparison = cli.she_glossy_comparison;
    let admitted_she_terrain = cli.admitted_she_terrain;
    let lighting_reference_diagnostic = cli.lighting_reference_diagnostic;
    let lighting_readback_diagnostic = cli.lighting_readback_diagnostic;
    let presentation = cli.presentation;
    let drive_mode = cli.drive;
    let quality = cli.quality;
    let profile_receipt = cli.profile_receipt.clone();
    // A `.cm1fhs` envelope states array shapes but no dx/dy/dz/ztop, so shapes
    // alone cannot tell a 220 km domain from a 15 km one. A frame whose profile
    // is not in the sealed table therefore has its physical geometry bound from
    // the publication document the publisher writes beside the envelope, keyed
    // on the contract's own configuration identity. Guessing instead would draw
    // the volume at the wrong physical size and make every distance, speed,
    // height and camera framing the viewer reports wrong by that factor.
    if production.is_none()
        && production_series.is_none()
        && gm1_view_source_profile_for_dataset_profile(&dataset.profile_id).is_none()
    {
        let geometry = load_frame_geometry(
            &dataset.source_path,
            &dataset.config_sha256,
            &dataset.profile_id,
            dataset.fields[0].shape,
        )?;
        let profile = cartesian_history_view_source_profile(geometry);
        install_runtime_source_profile(profile)?;
        // The scene floor is the domain's own lowest surface. A frame that
        // publishes no surface carrier has none, and sits on zero.
        let floor_km = dataset
            .terrain_msl_m
            .as_deref()
            .map(|terrain| terrain.iter().copied().fold(f32::INFINITY, f32::min) * 1.0e-3)
            .filter(|floor| floor.is_finite())
            .unwrap_or(0.0);
        install_world_box(world_box_for_grid(&profile.georeferenced_grid(), floor_km))?;
        eprintln!(
            "gm1-view: bound frame geometry {} from {} • scene {:.3} x {:.3} x {:.3} km",
            profile.domain_label(),
            crate::frame_geometry::sidecar_path_for(&dataset.source_path)?.display(),
            world_size().x,
            world_size().y,
            world_size().z,
        );
    }
    let source_profile = production
        .as_ref()
        .map(|production| production.source_profile)
        .or_else(|| resolved_source_profile(&dataset.profile_id));
    if admitted_she_terrain
        && source_profile
            .is_some_and(|profile| profile.terrain == Gm1ViewTerrainClass::FlatZeroTerrain)
    {
        return Err(
            "refusing real-child admitted terrain for a flat-zero source profile: the \
             sealed terrain-following scene composes every level against the sealed \
             300x300 relief, so a domain with no such relief would be drawn with \
             another domain's hills under it"
                .to_owned(),
        );
    }
    let interaction_defaults = InteractiveDefaults::default();
    let profile_adapter = if profile_receipt.is_some() {
        match probe_dx12_adapter() {
            Ok(adapter) => Some(adapter),
            Err(error) => return Err(error),
        }
    } else {
        None
    };
    if screenshot.as_deref().is_some_and(Path::exists) {
        return Err(
            "refusing to overwrite an existing screenshot path: the file already there \
             would be replaced by this run's frame and could not be recovered"
                .to_owned(),
        );
    }
    if profile_receipt.as_deref().is_some_and(Path::exists) {
        return Err(
            "refusing to overwrite an existing profile receipt: the measurements \
             already recorded there would be replaced by this run's and could not \
             be recovered"
                .to_owned(),
        );
    }
    if let Some(path) = screenshot.as_deref().and_then(Path::parent)
        && let Err(error) = std::fs::create_dir_all(path)
    {
        return Err(format!("cannot create screenshot directory: {error}"));
    }
    if let Some(path) = profile_receipt.as_deref().and_then(Path::parent)
        && let Err(error) = std::fs::create_dir_all(path)
    {
        return Err(format!("cannot create profile-receipt directory: {error}"));
    }
    let default_slice = dataset.fields[0].shape[0] / 3;
    let mut timeline_frames = cli
        .watch_dir
        .as_deref()
        .map(scan_pack_timeline)
        .map(|scan| {
            for refusal in scan.refusals {
                warn!(
                    "startup timeline refused {:?} at {}: {}",
                    refusal.step,
                    refusal.directory.display(),
                    refusal.reason
                );
            }
            scan.frames
        })
        .unwrap_or_default();
    if !timeline_frames
        .iter()
        .any(|frame| frame.step == dataset.step)
    {
        timeline_frames.push(dataset.clone());
    }
    timeline_frames.sort_by_key(|frame| frame.step);
    timeline_frames.dedup_by_key(|frame| frame.step);
    let adapter = Gm1GeoreferencedAdapter;
    for frame in &timeline_frames {
        let adapted = match resolved_source_profile(&frame.profile_id) {
            Some(profile) => adapter.adapt_with_grid(frame, profile.georeferenced_grid()),
            None => adapter.adapt(frame),
        };
        let georeferenced = match adapted {
            Ok(frame) => frame,
            Err(error) => {
                return Err(format!(
                    "georeferenced GM1 adapter refused step {}: {error}",
                    frame.step
                ));
            }
        };
        info!(
            "georeferenced frame admitted: abi={} adapter={} step={} fields={}",
            georeferenced.abi,
            georeferenced.adapter_id,
            georeferenced.time.step,
            georeferenced.fields.len(),
        );
    }
    let current_timeline_index = timeline_frames
        .iter()
        .position(|frame| frame.step == dataset.step)
        .unwrap_or_else(|| timeline_frames.len().saturating_sub(1));
    let timeline_frames = timeline_frames
        .into_iter()
        .map(|dataset| TimelineFrame {
            source_gate: source_gate_for_dataset(&dataset),
            dataset: Arc::new(dataset),
        })
        .collect::<VecDeque<_>>();
    let viewer_dataset = timeline_frames[current_timeline_index].dataset.clone();
    let selected_science_layer = cli
        .science_layer
        .as_deref()
        .and_then(ScienceLayerId::parse)
        .unwrap_or(ScienceLayerId::SimSatTopView);
    let show_science_layer = cli.science_layer.is_some() || cli.science_dir.is_some();
    let (science_sender, science_receiver) = channel();
    let live_watch = cli.watch_dir.clone().map(|root| LiveWatchState {
        receiver: Mutex::new(spawn_watch(root.clone(), dataset.step)),
        status: format!("LIVE accepted step {}", dataset.step),
        root,
        last_refusal: None,
    });
    let mut wgpu_settings = WgpuSettings::default();
    if profile_adapter.is_some() {
        wgpu_settings.backends = Some(Backends::DX12);
    }
    let plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: Some(Window {
                title: "GM1 View - exact admitted model state".to_owned(),
                resolution: WindowResolution::new(1920, 1080),
                present_mode: PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        })
        .set(dev_asset_plugin())
        .set(RenderPlugin {
            render_creation: wgpu_settings.into(),
            ..default()
        });
    let mut app = App::new();
    app.add_plugins(plugins)
        .add_plugins((
            EguiPlugin::default(),
            Gm1VolumePlugin,
            SheSkyMaterialPlugin,
            SheGlossyMaterialPlugin,
            AdmittedTerrainMaterialPlugin,
            PhysicalAtmospherePlugin,
        ))
        .insert_resource(ClearColor(if admitted_she_terrain {
            Color::srgb(0.20, 0.43, 0.68)
        } else {
            Color::srgb(0.34, 0.56, 0.79)
        }))
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.36, 0.43, 0.56),
            brightness: 95.0,
            affects_lightmapped_meshes: true,
        })
        .insert_resource(ViewerData(viewer_dataset))
        .insert_resource(TimelineState {
            frames: timeline_frames,
            current: current_timeline_index,
            requested: None,
            playing: interaction_defaults.timeline_playing,
            looping: interaction_defaults.timeline_looping,
            follow_live: interaction_defaults.follow_live,
            elapsed: 0.0,
        })
        .insert_resource(ProductionInspectorState::default())
        .insert_resource(ViewerState {
            selected_field: 0,
            loaded_field: usize::MAX,
            selected_slice: default_slice,
            show_slice: false,
            fly_mode: interaction_defaults.fly_mode,
            material: None,
            camera_summary: String::new(),
            build_summary: runtime_build_summary(),
            she_comparison,
            she_glossy_comparison,
            admitted_she_terrain,
            lighting_reference_diagnostic,
            lighting_readback_diagnostic,
            presentation,
            drive_mode,
            quality,
            she_summary: None,
        })
        .insert_resource(ScienceViewState {
            selected: selected_science_layer,
            show: show_science_layer,
            requested: None,
            rendered: None,
            status: "waiting for strict science-layer admission".to_owned(),
            sender: science_sender,
            receiver: Mutex::new(science_receiver),
            root: cli.science_dir.clone(),
        })
        .insert_resource(FrameProfiler::new(
            profile_receipt,
            cli.soak_seconds,
            profile_adapter,
        ))
        .insert_resource(CaptureState {
            path: screenshot,
            frames: 0,
            requested: false,
            attempts: 0,
            phase: CapturePhase::InitialWarmup,
            phase_started: 0,
            dense_mask: None,
            cloud_mask: None,
            terrain_mask: None,
            dominant_projection: None,
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                timeline_control,
                apply_watch_events,
                apply_production_series_events,
                apply_production_series_interpolation,
                refresh_selected_volume,
                science_layer_update,
                camera_control,
                chase_vehicle_control,
                capture_when_ready,
                frame_profile,
            )
                .chain(),
        )
        .add_systems(EguiPrimaryContextPass, scientific_overlay);
    if let Some(live_watch) = live_watch {
        app.insert_resource(live_watch);
    }
    if let Some(production) = production {
        app.insert_resource(ProductionViewData(production));
    }
    if let Some(series) = production_series {
        let dormant = series.playback.interpolation_dormant();
        let step = series.playback.current().step;
        app.insert_resource(ProductionSeriesLiveState {
            receiver: Mutex::new(series.receiver),
            root: series.root,
            playback: series.playback,
            status: if dormant {
                format!(
                    "LIVE exact step {step}; one frame admitted; temporal interpolation dormant"
                )
            } else {
                format!("LIVE exact A/B endpoints admitted through step {step}")
            },
            last_refusal: None,
            rendered_alpha_f32_bits: 1.0f32.to_bits(),
        });
    }
    // A refusal inside the render app must reach the caller as a failed run.
    // Exiting zero after refusing to write a frame tells a script, a batch
    // render or a CI step that an image was produced when none was.
    match app.run() {
        AppExit::Success => Ok(()),
        AppExit::Error(code) => Err(format!(
            "the render app refused this frame (exit code {code}); the refusal \
             is logged above and no image was written"
        )),
    }
}

fn dev_asset_plugin() -> AssetPlugin {
    #[cfg(debug_assertions)]
    {
        let packaged = std::env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(Path::to_path_buf))
            .map(|directory| directory.join("assets").join("shaders"))
            .filter(|directory| directory.join("gm1_volume.wgsl").is_file());
        AssetPlugin {
            file_path: packaged
                .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"))
                .to_string_lossy()
                .into_owned(),
            watch_for_changes_override: Some(true),
            ..default()
        }
    }
    #[cfg(not(debug_assertions))]
    {
        AssetPlugin::default()
    }
}

fn runtime_assets_directory() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .map(|directory| directory.join("assets"))
        .filter(|directory| directory.join(HRRR_SURROUND_ASSET_DIRECTORY).is_dir())
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets"))
}

// Bevy setup systems naturally receive one parameter per resource family.
#[allow(clippy::too_many_arguments)]
fn setup(
    mut commands: Commands,
    data: Res<ViewerData>,
    production: Option<Res<ProductionViewData>>,
    mut state: ResMut<ViewerState>,
    mut capture: ResMut<CaptureState>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut volume_materials: ResMut<Assets<Gm1VolumeMaterial>>,
    mut proof_materials: ResMut<Assets<DevProofMaterial>>,
    mut pbr_materials: ResMut<Assets<StandardMaterial>>,
    mut she_materials: ResMut<Assets<SheSkyMaterial>>,
    mut she_glossy_materials: ResMut<Assets<SheGlossyMaterial>>,
    mut admitted_terrain_materials: ResMut<Assets<AdmittedTerrainMaterial>>,
    mut physical_atmosphere_materials: ResMut<Assets<PhysicalAtmosphereMaterial>>,
) {
    let started = Instant::now();
    let field = &data.0.fields[state.selected_field];
    let frame = frame_dense_condensate_for_profile(
        field,
        resolved_source_profile(&data.0.profile_id),
    );
    let production_fields = production.as_ref().map(|production| {
        upload_production_gpu_fields(&production.0, &mut buffers)
            .expect("strict production fields were already admitted")
    });
    let density = match production_fields.as_ref() {
        Some(fields) => fields
            .iter()
            .find(|field| field.residency == ProductionResidencyKind::CloudScalarVolume)
            .expect("validated production residency has exact qc+qi")
            ._buffer
            .clone(),
        None => buffers.add(storage_from_field(field).expect("strict field was already admitted")),
    };
    if let (Some(production), Some(fields)) = (production.as_ref(), production_fields) {
        commands.insert_resource(ProductionGpuResidency {
            adapter_id: production.0.frame.adapter_id,
            source_profile: production.0.source_profile,
            manifest_sha256: production.0.manifest_sha256.clone(),
            payload_set_sha256: production.0.payload_set_sha256.clone(),
            fields,
        });
    }
    let proof_density = density.clone();
    let coordinates = buffers.add(if state.admitted_she_terrain {
        admitted_volume_coordinates().expect("sealed terrain-following coordinate storage")
    } else {
        unused_coordinate_storage()
    });
    let mut volume_uniform = VolumeUniform::for_field(field, world_min(), world_size());
    volume_uniform.control.z = state.quality.ray_steps();
    // Production sunward transmittance for every scene, not only the sealed
    // terrain one. `w=0` selects the frozen v13 reference march, which walks a
    // fixed distance in *texture* space: on an anisotropic domain that is a
    // different physical distance along each axis, so the cloud is shaded by a
    // sun ray the domain never had and renders darker than the sky behind it.
    // `w=1` marches the same sealed sun in world kilometres.
    volume_uniform.sky_amplitude.w = f32::from(!state.lighting_reference_diagnostic);
    // The dense-response knee follows the condensate this frame actually
    // carries, so a domain the cloud occupies a few percent of still separates
    // core from haze instead of drawing the whole storm as haze.
    volume_uniform.optics.w = frame.dense_response_knee;
    // Production integration opacity. The default saturates a ray inside the
    // cloud's outermost cell, where the grid-mean condensate is still low, so
    // the frame reports the whole storm at the brightness, colour and
    // scattering of its own onset skin and none of the structure behind it.
    volume_uniform.control.y = PRODUCTION_INTEGRATION_OPACITY;
    if state.admitted_she_terrain {
        // The homogeneous lane is otherwise unused. It selects a labeled
        // physical-lighting display treatment without changing any source word.
        volume_uniform.world_min.w = 1.0;
        info!(
            "volume-optics profile=SEALED_COMBINED_QC_QI onset_q={:.9e} dense_q99={:.9e} onset_effective_radius_um={:.3} dense_effective_radius_um={:.3} assumed_air_density_kg_m3={:.6} qext=2 onset_grid_occupancy={:.6} ray_steps={} sun_march_km=1.600 sky_display_exposure={:.6} ground_bounce_albedo={:.6} multiple_scatter_octaves=[{:.6},{:.6}] density_words_unchanged=true phase_sign=dot(sample_to_sun,camera_to_sample)",
            volume_uniform.control.x,
            volume_uniform.optics.w,
            volume_uniform.optics.x * 1.0e6,
            volume_uniform.optics.y * 1.0e6,
            volume_uniform.optics.z,
            ONSET_GRID_OCCUPANCY,
            volume_uniform.control.z as u32,
            volume_uniform.scattering.x,
            volume_uniform.scattering.y,
            volume_uniform.scattering.z,
            volume_uniform.scattering.w,
        );
        info!(
            "volume-lighting profile={} {}",
            if state.lighting_reference_diagnostic {
                "REFERENCE_V13_DIAGNOSTIC_ONLY"
            } else {
                "PHYSICAL_PHASE_SHADOW_CANDIDATE"
            },
            lighting_probe_summary(field, &volume_uniform),
        );
        if DEBUG_TERRAIN_GROUND_PROBE {
            let probe_material = pbr_materials.add(StandardMaterial {
                base_color: Color::srgb(1.0, 0.0, 0.72),
                emissive: LinearRgba::new(4.0, 0.0, 2.5, 1.0),
                unlit: true,
                ..default()
            });
            commands.spawn((
                Name::new("REFUSED diagnostic flat ground visibility probe"),
                Mesh3d(meshes.add(Cuboid::new(15.0, 0.02, 15.0))),
                MeshMaterial3d(probe_material),
                Transform::from_xyz(0.0, 0.34, 0.0),
                NoCpuCulling,
                NoFrustumCulling,
            ));
        }
    }
    let material = volume_materials.add(Gm1VolumeMaterial {
        density,
        uniform: volume_uniform,
        coordinates,
    });
    state.material = Some(material.clone());
    state.loaded_field = state.selected_field;

    commands.spawn((
        Name::new("Exact GM1 condensate volume"),
        ExactVolumeSurface,
        Mesh3d(meshes.add(Cuboid::new(world_size().x, world_size().y, world_size().z))),
        MeshMaterial3d(material),
        Transform::from_translation(world_min() + world_size() * 0.5),
        if DEV_PROOF_STAGE >= 4 && !state.she_comparison && !state.she_glossy_comparison {
            Visibility::Visible
        } else {
            Visibility::Hidden
        },
        NoCpuCulling,
        NoFrustumCulling,
    ));

    if state.admitted_she_terrain {
        let atmosphere = physical_atmosphere_materials.add(
            PhysicalAtmosphereMaterial::sealed_t300()
                .expect("sealed source-backed SimSat radiance sky"),
        );
        commands.spawn((
            Name::new("Source-backed SimSat T+300 physical sky radiance"),
            PhysicalAtmosphereSurface,
            Mesh3d(meshes.add(Sphere::new(140.0).mesh().uv(192, 96))),
            MeshMaterial3d(atmosphere),
            NoCpuCulling,
            NoFrustumCulling,
        ));
        let terrain = admitted_terrain_mesh().expect("embedded exact GM1 terrain identity");
        let material = admitted_terrain_materials
            .add(AdmittedTerrainMaterial::t300().expect("admitted SHE terrain inputs"));
        let surround_asset_directory =
            runtime_assets_directory().join(HRRR_SURROUND_ASSET_DIRECTORY);
        let (surround, surround_stats) =
            load_source_backed_surround_assets(&surround_asset_directory)
                .expect("sealed source-backed HRRR surrounding terrain");
        commands.spawn((
            Name::new("Source-backed HRRR curved surrounding terrain context"),
            AdmittedTerrainSurface,
            Mesh3d(meshes.add(surround)),
            MeshMaterial3d(material.clone()),
            NoCpuCulling,
            NoFrustumCulling,
        ));
        commands.spawn((
            Name::new("Exact GM1 terrain - admitted source-specific glossy SHE"),
            AdmittedTerrainSurface,
            Mesh3d(meshes.add(terrain)),
            MeshMaterial3d(material),
            NoCpuCulling,
            NoFrustumCulling,
        ));
        state.she_summary = Some(format!(
            "ADMITTED source-specific alpha={ADMITTED_MICROFACET_ALPHA:.2} • exact child {} • source-backed HRRR context {}x{} • coeff {} • receipt {} • Belfast alpha32 control REFUSED {}",
            &ADMITTED_TERRAIN_SHA256[..16],
            HRRR_SURROUND_NX,
            HRRR_SURROUND_NY,
            &ADMITTED_SPECULAR_SHA256[..16],
            &ADMITTED_RECEIPT_SHA256[..16],
            &BELFAST_ALPHA32_REFUSAL_RECEIPT_SHA256[..16],
        ));
        info!(
            "spawned exact GM1 terrain plus source-backed curved HRRR context with admitted SHE: profile={} terrain={} vertical_grid={} sigma={} coefficients={} receipt={} belfast_control_refusal={} hrrr_wrfnat_f02={} hrrr_window=i{}..{} j{}..{} bounds_east_km={:.3}..{:.3} bounds_north_km={:.3}..{:.3} bounds_up_km={:.3}..{:.3} context_margin_km={:.3}",
            ADMITTED_PROFILE,
            ADMITTED_TERRAIN_SHA256,
            ADMITTED_VERTICAL_GRID_SHA256,
            ADMITTED_SIGMA_SHA256,
            ADMITTED_SPECULAR_SHA256,
            ADMITTED_RECEIPT_SHA256,
            BELFAST_ALPHA32_REFUSAL_RECEIPT_SHA256,
            HRRR_WRFNAT_F02_SHA256,
            HRRR_SURROUND_WINDOW_I_START,
            HRRR_SURROUND_WINDOW_I_START + HRRR_SURROUND_NX - 1,
            HRRR_SURROUND_WINDOW_J_START,
            HRRR_SURROUND_WINDOW_J_START + HRRR_SURROUND_NY - 1,
            surround_stats.minimum_east_km,
            surround_stats.maximum_east_km,
            surround_stats.minimum_north_km,
            surround_stats.maximum_north_km,
            surround_stats.minimum_up_km,
            surround_stats.maximum_up_km,
            surround_stats.horizontal_margin_km(),
        );
        info!(
            "spawned source-backed SimSat physical sky radiance: source={} display_exposure={:.6} direct_sun_separate=true",
            PHYSICAL_ATMOSPHERE_SOURCE_SHA256, PHYSICAL_ATMOSPHERE_EXPOSURE,
        );
    } else {
        let grid_material = pbr_materials.add(StandardMaterial {
            base_color: Color::srgba(0.20, 0.27, 0.24, 0.22),
            perceptual_roughness: 0.96,
            metallic: 0.0,
            alpha_mode: AlphaMode::Blend,
            ..default()
        });
        let north_south = meshes.add(Cuboid::new(0.014, 0.018, 12.0));
        let east_west = meshes.add(Cuboid::new(12.0, 0.018, 0.014));
        for index in [-1, 0, 1] {
            let offset = index as f32 * 4.0;
            commands.spawn((
                Name::new("Reference ground grid - north/south"),
                Mesh3d(north_south.clone()),
                MeshMaterial3d(grid_material.clone()),
                Transform::from_xyz(offset, 0.0, 0.0),
            ));
            commands.spawn((
                Name::new("Reference ground grid - east/west"),
                Mesh3d(east_west.clone()),
                MeshMaterial3d(grid_material.clone()),
                Transform::from_xyz(0.0, 0.0, offset),
            ));
        }
    }

    if DEV_PROOF_STAGE == 2 {
        spawn_diagnostic_aabb(&mut commands, &mut meshes, &mut pbr_materials);
    }

    commands.spawn((
        Name::new("Physical T+300 sun"),
        DirectionalLight {
            illuminance: 18_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.82, -1.15, 0.0)),
    ));

    if state.she_comparison {
        let projection = project_t300_simsat_diffuse_l2(SHE_DIFFUSE_FACE_RESOLUTION);
        let error = t300_projection_error(&projection, 4096);
        state.she_summary = Some(format!(
            "SHE FP32 L2 • {}²×6 physical-sky quadrature • coeff {} • max/RMS abs {:.3e}/{:.3e} W m⁻² • max/RMS rel {:.3}%/{:.3}% • luminance RMS abs/rel {:.3e}/{:.3}%",
            projection.face_resolution,
            &projection.sha256[..16],
            error.max_absolute,
            error.rms_absolute,
            error.max_relative * 100.0,
            error.rms_relative * 100.0,
            error.rms_luminance_absolute,
            error.rms_luminance_relative * 100.0,
        ));
        let sphere = meshes.add(Sphere::new(1.65).mesh().uv(96, 64));
        let reference = she_materials.add(SheSkyMaterial::t300_reference(&projection));
        let projected = she_materials.add(SheSkyMaterial::t300_projected(&projection));
        commands.spawn((
            Name::new("Sealed SimSat positive log-SH3 material reference"),
            Mesh3d(sphere.clone()),
            MeshMaterial3d(reference),
            Transform::from_xyz(-2.15, 2.7, 9.0),
            NoFrustumCulling,
        ));
        commands.spawn((
            Name::new("Pinned FP32 SHE L2 projected material"),
            Mesh3d(sphere),
            MeshMaterial3d(projected),
            Transform::from_xyz(2.15, 2.7, 9.0),
            NoFrustumCulling,
        ));
        info!(
            "SHE material comparison source_sky={} coefficients={} face_resolution={} max_relative_error={:.9e} rms_relative_error={:.9e}",
            SHE_SKY_SOURCE_SHA256,
            projection.sha256,
            projection.face_resolution,
            error.max_relative,
            error.rms_relative,
        );
    }

    if state.she_glossy_comparison {
        state.she_summary = Some(format!(
            "{} • physical radiance {} • refused coeff {} • WGSL reference {} GGX spp (held CPU {} spp)",
            PROVISIONAL_STATUS,
            &GLOSSY_RADIANCE_SOURCE_SHA256[..16],
            &PROVISIONAL_SPECULAR_SHA256[..16],
            PREVIEW_REFERENCE_GGX_SAMPLES,
            SPECULAR_HELD_GGX_SAMPLES,
        ));
        let coupon = meshes.add(Sphere::new(1.0).mesh().uv(96, 64));
        let modes = [
            (
                "Physical SimSat-radiance wet-dielectric reference",
                GlossyComparisonMode::PhysicalReference,
                -4.2,
            ),
            (
                "REFUSED reduced 33-term SHE diagnostic preview",
                GlossyComparisonMode::ProvisionalShe33,
                0.0,
            ),
            (
                "Absolute luminance-error diagnostic",
                GlossyComparisonMode::AbsoluteError,
                4.2,
            ),
        ];
        for (name, mode, x) in modes {
            let material = she_glossy_materials
                .add(SheGlossyMaterial::t300(mode).expect("sealed glossy diagnostic inputs"));
            commands.spawn((
                Name::new(name),
                Mesh3d(coupon.clone()),
                MeshMaterial3d(material),
                Transform::from_xyz(x, 3.0, 9.0).with_scale(Vec3::new(1.72, 2.55, 0.64)),
                NoFrustumCulling,
            ));
        }
        warn!(
            "glossy diagnostic preview only: status={} radiance={} atmosphere_source={} coeff={} upstream_default_required=256x256/4096/unregularized-fp32",
            PROVISIONAL_STATUS,
            GLOSSY_RADIANCE_SOURCE_SHA256,
            SIMSAT_RADIANCE_SOURCE_SHA256,
            PROVISIONAL_SPECULAR_SHA256,
        );
    }

    state.camera_summary = format!(
        "PCA q≥{:.3} g kg⁻¹ (q99+) • {} cells • yaw {:.1}° • pitch {:.1}° • r {:.1}",
        frame.cutoff * 1.0e3,
        frame.dense_count,
        frame.yaw.to_degrees(),
        frame.pitch.to_degrees(),
        frame.radius,
    );
    // Reproduce the last independently visible fixed view while the adaptive
    // framing path is audited. This is a display-only camera, not model data.
    let controller = if state.she_glossy_comparison {
        ViewerCamera {
            focus: Vec3::new(0.0, 3.0, 9.0),
            yaw: 0.0,
            pitch: -0.035,
            radius: 13.2,
        }
    } else if state.she_comparison {
        ViewerCamera {
            focus: Vec3::new(0.0, 2.7, 9.0),
            yaw: 0.0,
            pitch: -0.06,
            radius: 12.0,
        }
    } else if state.admitted_she_terrain {
        admitted_chase_camera(field, &frame)
    } else if DIAGNOSTIC_FIXED_CAMERA {
        ViewerCamera {
            focus: Vec3::new(3.078_735_8, 3.136_666_8, -1.198_130_4),
            yaw: 1.079_897,
            pitch: -0.24,
            radius: 22.345,
        }
    } else {
        ViewerCamera {
            focus: frame.focus,
            yaw: frame.yaw,
            pitch: frame.pitch,
            radius: frame.radius,
        }
    };
    let rotation = Quat::from_euler(EulerRot::YXZ, controller.yaw, controller.pitch, 0.0);
    let forward = rotation * Vec3::NEG_Z;
    let camera_position = controller.focus - forward * controller.radius;
    let camera_focus = controller.focus;
    let center_direction = (controller.focus - camera_position).normalize();
    let center_interval = ray_aabb_interval(
        camera_position,
        center_direction,
        world_min(),
        world_min() + world_size(),
    );
    if state.admitted_she_terrain
        && !center_interval.is_some_and(|(entry, exit)| exit > entry.max(0.0))
    {
        panic!("admitted terrain presentation camera misses the exact volume AABB");
    }
    if state.admitted_she_terrain {
        state.camera_summary = format!(
            "exact q≥{:.3} g kg⁻¹ dominant six-neighbour component {:.5}% mass • centroid {:?} • bbox {:?}…{:?} • physical-onset field bbox {:?}…{:?} • PCA terrain-relative chase yaw {:.1}° pitch {:.1}° r {:.1} km • ray {:?}",
            frame.cutoff * 1.0e3,
            frame.dominant_mass_fraction * 100.0,
            frame.focus,
            frame.dense_min,
            frame.dense_max,
            frame.visible_min,
            frame.visible_max,
            controller.yaw.to_degrees(),
            controller.pitch.to_degrees(),
            controller.radius,
            center_interval,
        );
    }
    if state.admitted_she_terrain {
        let sun_enu = T300_SUN_ENU_F32_BITS.map(f32::from_bits);
        let sun_world = Vec3::new(sun_enu[0], sun_enu[2], sun_enu[1]).normalize();
        info!(
            "volume-lighting directions sun_world={:?} camera_ray={:?} phase_cosine={:.9}",
            sun_world,
            center_direction,
            sun_world.dot(center_direction),
        );
    }
    if DEV_PROOF_STAGE == 1 {
        let proof_material = pbr_materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.0, 0.78),
            emissive: LinearRgba::new(8.0, 0.0, 4.8, 1.0),
            unlit: true,
            ..default()
        });
        commands.spawn((
            Name::new("GM1 View dev proof magenta cube"),
            Mesh3d(meshes.add(Cuboid::new(3.2, 3.2, 3.2))),
            MeshMaterial3d(proof_material),
            Transform::from_translation(camera_position + center_direction * 10.0),
        ));
    }
    if DEV_PROOF_STAGE == 3 {
        let proof_material = proof_materials.add(DevProofMaterial {
            density: proof_density,
            uniform: VolumeUniform::for_field(field, world_min(), world_size()),
        });
        commands.spawn((
            Name::new("GM1 View minimal custom-material proof"),
            Mesh3d(meshes.add(Cuboid::new(3.2, 3.2, 3.2))),
            MeshMaterial3d(proof_material),
            Transform::from_translation(camera_position + center_direction * 10.0),
            NoCpuCulling,
            NoFrustumCulling,
        ));
    }
    let camera_transform =
        Transform::from_translation(camera_position).looking_at(controller.focus, Vec3::Y);
    if state.admitted_she_terrain {
        log_admitted_camera_projection(&camera_transform, controller.focus);
        let (projected_min, projected_max) = projected_dense_bbox(
            camera_position,
            controller.focus,
            frame.dense_min,
            frame.dense_max,
        )
        .expect("admitted chase search already proved dense bbox projection");
        info!(
            "admitted exact dense-bbox projection ndc={projected_min:?}..{projected_max:?} fov_y_deg={CHASE_VERTICAL_FOV_DEGREES:.3}"
        );
        let dominant = project_dominant_component(field, &frame, camera_position, controller.focus);
        info!(
            "dominant-component projection mass_in_frame={:.9} ndc={:?}..{:?} width_fraction={:.6}",
            dominant.mass_fraction_in_frame,
            dominant.minimum_ndc,
            dominant.maximum_ndc,
            dominant.width_fraction,
        );
        capture.dominant_projection = Some(dominant);
    }
    let initial_vehicle_yaw = controller.yaw;
    commands.spawn((
        Name::new("GM1 orbit/fly camera"),
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: if state.admitted_she_terrain {
                CHASE_VERTICAL_FOV_DEGREES.to_radians()
            } else {
                45.0_f32.to_radians()
            },
            // World units are kilometres; the chase eye is only 0.100 km
            // above the exact terrain. A 1 m near plane preserves the
            // source-backed foreground without sacrificing this 15 km scene.
            near: if state.admitted_she_terrain {
                0.001
            } else {
                0.1
            },
            ..default()
        }),
        NoCpuCulling,
        NoIndirectDrawing,
        Hdr,
        Tonemapping::AgX,
        Exposure { ev100: 9.5 },
        camera_transform,
        controller,
    ));
    if state.drive_mode {
        let surround_directory = runtime_assets_directory().join(HRRR_SURROUND_ASSET_DIRECTORY);
        commands.insert_resource(DriveTerrain {
            child_msl_m: admitted_terrain_values().expect("sealed drive terrain identity"),
            surround: HrrrSurroundHeightSampler::load(&surround_directory)
                .expect("sealed drive HRRR context sampler"),
        });
        spawn_chase_vehicle(
            &mut commands,
            &mut meshes,
            &mut pbr_materials,
            camera_position.xz(),
            initial_vehicle_yaw,
        );
    }
    info!(
        "uploaded exact {} binary32 words ({}) through a read-only storage buffer in {:.3}s; dominant connected q99 component count={} q_mass={:.9e}/{:.9e} fraction={:.9} bbox={:?}..{:?} physical-onset count={} bbox={:?}..{:?} focus={:?} yaw={:.6} pitch={:.6} radius={:.3}",
        field.values.len(),
        field.values_sha256,
        started.elapsed().as_secs_f64(),
        frame.dense_count,
        frame.dominant_mass,
        frame.connected_mass,
        frame.dominant_mass_fraction,
        frame.dense_min,
        frame.dense_max,
        frame.visible_count,
        frame.visible_min,
        frame.visible_max,
        frame.focus,
        frame.yaw,
        frame.pitch,
        frame.radius,
    );
    if let Some(morphology) = component_morphology(field, &frame) {
        info!(
            "source-morphology q99_grid_bbox={:?}..{:?} bbox_cells={} voxel_compactness={:.9} lateral_boundary_distance_cells={} vertical_extent_km={:.6} max_horizontal_extent_km={:.6} vertical_to_horizontal_ratio={:.9} auxiliary_w_or_reflectivity_in_pack=false",
            morphology.grid_min,
            morphology.grid_max,
            morphology.bbox_cell_count,
            morphology.voxel_compactness,
            morphology.lateral_boundary_distance_cells,
            morphology.vertical_extent_km,
            morphology.maximum_horizontal_extent_km,
            morphology.vertical_to_horizontal_ratio,
        );
    }
    info!(
        "diagnostic camera position={:?} focus={:?} center-ray interval={:?}",
        camera_position, camera_focus, center_interval,
    );
    info!("{}", state.build_summary);
}

fn runtime_build_summary() -> String {
    let executable = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("<unresolved>"));
    let executable_hash = hash_file(&executable).unwrap_or_else(|error| format!("ERROR:{error}"));
    let shader = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/gm1_volume.wgsl");
    let shader_hash = hash_file(&shader).unwrap_or_else(|error| format!("ERROR:{error}"));
    let she_shader = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/she_material.wgsl");
    let she_shader_hash = hash_file(&she_shader).unwrap_or_else(|error| format!("ERROR:{error}"));
    let glossy_shader =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/she_glossy_material.wgsl");
    let glossy_shader_hash =
        hash_file(&glossy_shader).unwrap_or_else(|error| format!("ERROR:{error}"));
    let admitted_shader =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("src/{ADMITTED_TERRAIN_SHADER}"));
    let admitted_shader_hash =
        hash_file(&admitted_shader).unwrap_or_else(|error| format!("ERROR:{error}"));
    format!(
        "dev-stage={DEV_PROOF_STAGE} exe={} exe-sha256={} shader={} shader-sha256={} she-shader={} she-shader-sha256={} glossy-shader={} glossy-shader-sha256={} admitted-terrain-shader={} admitted-terrain-shader-sha256={}",
        executable.display(),
        executable_hash,
        shader.display(),
        shader_hash,
        she_shader.display(),
        she_shader_hash,
        glossy_shader.display(),
        glossy_shader_hash,
        admitted_shader.display(),
        admitted_shader_hash,
    )
}

fn hash_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn spawn_diagnostic_aabb(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.95, 0.44, 0.12, 0.58),
        emissive: LinearRgba::new(0.7, 0.16, 0.025, 1.0),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    let center = world_min() + world_size() * 0.5;
    let thickness = 0.035;
    for &y in &[world_min().y, world_min().y + world_size().y] {
        for &z in &[world_min().z, world_min().z + world_size().z] {
            commands.spawn((
                Name::new("Diagnostic volume AABB x edge"),
                Mesh3d(meshes.add(Cuboid::new(world_size().x, thickness, thickness))),
                MeshMaterial3d(material.clone()),
                Transform::from_xyz(center.x, y, z),
            ));
        }
        for &x in &[world_min().x, world_min().x + world_size().x] {
            commands.spawn((
                Name::new("Diagnostic volume AABB z edge"),
                Mesh3d(meshes.add(Cuboid::new(thickness, thickness, world_size().z))),
                MeshMaterial3d(material.clone()),
                Transform::from_xyz(x, y, center.z),
            ));
        }
    }
    for &x in &[world_min().x, world_min().x + world_size().x] {
        for &z in &[world_min().z, world_min().z + world_size().z] {
            commands.spawn((
                Name::new("Diagnostic volume AABB y edge"),
                Mesh3d(meshes.add(Cuboid::new(thickness, world_size().y, thickness))),
                MeshMaterial3d(material.clone()),
                Transform::from_xyz(x, center.y, z),
            ));
        }
    }
}

fn ray_aabb_interval(
    origin: Vec3,
    direction: Vec3,
    minimum: Vec3,
    maximum: Vec3,
) -> Option<(f32, f32)> {
    let inverse = Vec3::new(1.0 / direction.x, 1.0 / direction.y, 1.0 / direction.z);
    let first = (minimum - origin) * inverse;
    let second = (maximum - origin) * inverse;
    let near = first.min(second).max_element();
    let far = first.max(second).min_element();
    (far >= near.max(0.0)).then_some((near, far))
}

fn admitted_chase_camera(field: &crate::input::VolumeField, frame: &FieldFrame) -> ViewerCamera {
    let (camera_position, focus) = admitted_chase_solution(field, frame);
    let delta = focus - camera_position;
    let forward = delta.normalize();
    ViewerCamera {
        focus,
        yaw: (-forward.x).atan2(-forward.z),
        pitch: forward.y.asin(),
        radius: delta.length(),
    }
}

#[cfg(test)]
fn admitted_chase_position(field: &crate::input::VolumeField, frame: &FieldFrame) -> Vec3 {
    admitted_chase_solution(field, frame).0
}

fn admitted_chase_solution(field: &crate::input::VolumeField, frame: &FieldFrame) -> (Vec3, Vec3) {
    let dense_center = (frame.dense_min + frame.dense_max) * 0.5;
    let horizontal_forward = (Quat::from_rotation_y(frame.yaw) * Vec3::NEG_Z)
        .xz()
        .normalize();
    let horizontal_focus = Vec2::new(frame.focus.x, frame.focus.z);
    let target_ground = admitted_terrain_height_km(frame.focus.x, frame.focus.z)
        .expect("dominant-component focus is inside exact child terrain");
    let mut best: Option<(f32, Vec2, Vec3)> = None;

    // The q99 threshold is source-derived and the camera sees only the
    // largest six-neighbour connected component for composition. All cloud
    // words remain rendered; disconnected fragments are not deleted or
    // thresholded. The component PCA supplies azimuth, while distance and
    // optical-axis lift are solved for a 45–65% component width and a lower
    // 20–25% terrain horizon.
    let distance_steps = ((CHASE_MAX_STANDOFF_KM - CHASE_MIN_STANDOFF_KM) / 0.125).round() as usize;
    for raise_index in 0..=32 {
        let raise = 0.50 + raise_index as f32 * 0.125;
        let focus = Vec3::new(frame.focus.x, dense_center.y + raise, frame.focus.z);
        for distance_index in 0..=distance_steps {
            let distance = CHASE_MIN_STANDOFF_KM + distance_index as f32 * 0.125;
            let horizontal = horizontal_focus - horizontal_forward * distance;
            let eye = Vec3::new(horizontal.x, 0.4, horizontal.y);
            let outside_dense = eye.x < frame.dense_min.x - CHASE_DENSE_CLEARANCE_KM
                || eye.x > frame.dense_max.x + CHASE_DENSE_CLEARANCE_KM
                || eye.z < frame.dense_min.z - CHASE_DENSE_CLEARANCE_KM
                || eye.z > frame.dense_max.z + CHASE_DENSE_CLEARANCE_KM;
            if !outside_dense {
                continue;
            }
            let Some((projected_min, projected_max)) =
                projected_dense_bbox(eye, focus, frame.dense_min, frame.dense_max)
            else {
                continue;
            };
            let width_fraction = (projected_max.x - projected_min.x) * 0.5;
            let component_margin = (1.0 - projected_max.x)
                .min(projected_min.x + 1.0)
                .min(1.0 - projected_max.y)
                .min(projected_min.y + 1.0);
            if !(0.45..=0.65).contains(&width_fraction) || component_margin < 0.06 {
                continue;
            }
            let Some(terrain_ndc) = projected_point_ndc(
                eye,
                focus,
                Vec3::new(frame.focus.x, target_ground, frame.focus.z),
            ) else {
                continue;
            };
            let score = -(width_fraction - 0.55).abs() * 4.0 - (terrain_ndc.y + 0.55).abs()
                + 0.04 * component_margin;
            if best.is_none_or(|(best_score, _, _)| score > best_score) {
                best = Some((score, horizontal, focus));
            }
        }
    }
    let (_, _, bbox_focus) =
        best.expect("dominant connected component has no admitted chase composition");
    // A terrain-relative chase view aims at the source-derived dense-core
    // midpoint, not above it. The prior bbox-only lift pushed the real horizon
    // below the admitted 20–25% context band after exact-cell stand-off
    // refinement.
    let focus = Vec3::new(bbox_focus.x, dense_center.y, bbox_focus.z);
    // The axis-aligned dense bbox is deliberately conservative. Search the
    // source-derived stand-off against the exact connected-component cells so
    // sparse geometry cannot make the primary storm too small or push mass
    // outside the frame. The exact field is never cropped or modified by this
    // camera-only solve.
    let mut exact_best: Option<(f32, Vec2, DominantComponentProjection)> = None;
    for distance_index in 0..=distance_steps {
        let distance = CHASE_MIN_STANDOFF_KM + distance_index as f32 * 0.125;
        let horizontal = horizontal_focus - horizontal_forward * distance;
        let terrain_km = camera_terrain_height_km(horizontal.x, horizontal.y)
            .expect("selected chase eye is outside both exact child and HRRR context terrain");
        let eye = Vec3::new(
            horizontal.x,
            terrain_km + CHASE_EYE_ABOVE_TERRAIN_KM,
            horizontal.y,
        );
        let projection = project_dominant_component(field, frame, eye, focus);
        let margin = (1.0 - projection.maximum_ndc.x)
            .min(projection.minimum_ndc.x + 1.0)
            .min(1.0 - projection.maximum_ndc.y)
            .min(projection.minimum_ndc.y + 1.0);
        if !(0.45..=0.65).contains(&projection.width_fraction)
            || projection.mass_fraction_in_frame < 0.995
            || margin < 0.02
        {
            continue;
        }
        let center = (projection.minimum_ndc + projection.maximum_ndc) * 0.5;
        let score =
            -(projection.width_fraction - 0.55).abs() * 4.0 - center.x.abs() * 0.20 + margin * 0.04;
        if exact_best.is_none_or(|(best_score, _, _)| score > best_score) {
            exact_best = Some((score, horizontal, projection));
        }
    }
    let (_, horizontal, searched_projection) =
        exact_best.expect("exact dominant connected component has no admitted chase composition");
    let terrain_km = camera_terrain_height_km(horizontal.x, horizontal.y)
        .expect("selected chase eye is outside both exact child and HRRR context terrain");
    let eye = Vec3::new(
        horizontal.x,
        terrain_km + CHASE_EYE_ABOVE_TERRAIN_KM,
        horizontal.y,
    );
    let exact_projection = project_dominant_component(field, frame, eye, focus);
    assert!(
        (0.45..=0.65).contains(&exact_projection.width_fraction)
            && exact_projection.mass_fraction_in_frame >= 0.995,
        "exact dominant-component refinement failed: searched={searched_projection:?} final={exact_projection:?}"
    );
    (eye, focus)
}

fn camera_terrain_height_km(east_km: f32, north_km: f32) -> Result<f32, String> {
    if east_km.abs() <= 7.475 && north_km.abs() <= 7.475 {
        admitted_terrain_height_km(east_km, north_km)
    } else {
        let asset_directory = runtime_assets_directory().join(HRRR_SURROUND_ASSET_DIRECTORY);
        source_backed_surround_height_km_from_assets(&asset_directory, east_km, north_km)
    }
}

fn projected_point_ndc(eye: Vec3, focus: Vec3, point: Vec3) -> Option<Vec2> {
    let camera = Transform::from_translation(eye).looking_at(focus, Vec3::Y);
    let delta = point - eye;
    let depth = delta.dot(*camera.forward());
    if depth <= 0.0 {
        return None;
    }
    let tan_half_vertical = (CHASE_VERTICAL_FOV_DEGREES.to_radians() * 0.5).tan();
    let tan_half_horizontal = tan_half_vertical * PRESENTATION_ASPECT;
    Some(Vec2::new(
        delta.dot(*camera.right()) / (depth * tan_half_horizontal),
        delta.dot(*camera.up()) / (depth * tan_half_vertical),
    ))
}

fn project_dominant_component(
    field: &crate::input::VolumeField,
    frame: &FieldFrame,
    eye: Vec3,
    focus: Vec3,
) -> DominantComponentProjection {
    let [nz, ny, nx] = field.shape;
    let terrain = (nz == 75 && ny == ADMITTED_TERRAIN_NY && nx == ADMITTED_TERRAIN_NX)
        .then(|| admitted_terrain_values().ok())
        .flatten();
    let world_position = |index: usize| {
        let k = index / (ny * nx);
        let remainder = index % (ny * nx);
        let j = remainder / nx;
        let i = remainder % nx;
        let horizontal = world_min()
            + Vec3::new(
                (i as f32 + 0.5) / nx as f32,
                0.0,
                (j as f32 + 0.5) / ny as f32,
            ) * world_size();
        let y = terrain
            .as_ref()
            .and_then(|values| admitted_mass_height_km(k, values[j * nx + i]).ok())
            .unwrap_or(world_min().y + (k as f32 + 0.5) / nz as f32 * world_size().y);
        Vec3::new(horizontal.x, y, horizontal.z)
    };
    let mut in_frame_mass = 0.0_f64;
    let mut minimum_ndc = Vec2::splat(f32::INFINITY);
    let mut maximum_ndc = Vec2::splat(f32::NEG_INFINITY);
    for &index in &frame.dominant_indices {
        let Some(ndc) = projected_point_ndc(eye, focus, world_position(index)) else {
            continue;
        };
        minimum_ndc = minimum_ndc.min(ndc);
        maximum_ndc = maximum_ndc.max(ndc);
        if ndc.cmpge(Vec2::splat(-1.0)).all() && ndc.cmple(Vec2::splat(1.0)).all() {
            in_frame_mass += f64::from(field.values[index]);
        }
    }
    DominantComponentProjection {
        mass_fraction_in_frame: if frame.dominant_mass > 0.0 {
            in_frame_mass / frame.dominant_mass
        } else {
            0.0
        },
        minimum_ndc,
        maximum_ndc,
        width_fraction: (maximum_ndc.x - minimum_ndc.x) * 0.5,
    }
}

fn projected_dense_bbox(
    eye: Vec3,
    focus: Vec3,
    dense_min: Vec3,
    dense_max: Vec3,
) -> Option<(Vec2, Vec2)> {
    let camera = Transform::from_translation(eye).looking_at(focus, Vec3::Y);
    let forward = *camera.forward();
    let right = *camera.right();
    let up = *camera.up();
    let tan_half_vertical = (CHASE_VERTICAL_FOV_DEGREES.to_radians() * 0.5).tan();
    let tan_half_horizontal = tan_half_vertical * PRESENTATION_ASPECT;
    let mut projected_min = Vec2::splat(f32::INFINITY);
    let mut projected_max = Vec2::splat(f32::NEG_INFINITY);
    for x in [dense_min.x, dense_max.x] {
        for y in [dense_min.y, dense_max.y] {
            for z in [dense_min.z, dense_max.z] {
                let delta = Vec3::new(x, y, z) - eye;
                let depth = delta.dot(forward);
                if depth <= 0.0 {
                    return None;
                }
                let projected = Vec2::new(
                    delta.dot(right) / (depth * tan_half_horizontal),
                    delta.dot(up) / (depth * tan_half_vertical),
                );
                projected_min = projected_min.min(projected);
                projected_max = projected_max.max(projected);
            }
        }
    }
    Some((projected_min, projected_max))
}

fn admitted_terrain_height_km(world_x: f32, world_z: f32) -> Result<f32, String> {
    let terrain = admitted_terrain_values()?;
    Ok(admitted_terrain_height_km_from_values(
        &terrain, world_x, world_z,
    ))
}

fn admitted_terrain_height_km_from_values(terrain: &[f32], world_x: f32, world_z: f32) -> f32 {
    assert_eq!(
        terrain.len(),
        ADMITTED_TERRAIN_NX * ADMITTED_TERRAIN_NY,
        "sealed terrain shape"
    );
    let maximum = Vec2::new(
        (ADMITTED_TERRAIN_NX - 1) as f32,
        (ADMITTED_TERRAIN_NY - 1) as f32,
    );
    let cell_size = Vec2::new(
        world_size().x / ADMITTED_TERRAIN_NX as f32,
        world_size().z / ADMITTED_TERRAIN_NY as f32,
    );
    let grid_origin = Vec2::new(world_min().x, world_min().z) + 0.5 * cell_size;
    let grid = ((Vec2::new(world_x, world_z) - grid_origin) / cell_size).clamp(Vec2::ZERO, maximum);
    let lower = grid.floor().as_uvec2();
    let upper = (lower + UVec2::ONE).min(UVec2::new(
        (ADMITTED_TERRAIN_NX - 1) as u32,
        (ADMITTED_TERRAIN_NY - 1) as u32,
    ));
    let weight = grid.fract();
    let load = |point: UVec2| terrain[point.y as usize * ADMITTED_TERRAIN_NX + point.x as usize];
    let lower_value = load(lower).lerp(load(UVec2::new(upper.x, lower.y)), weight.x);
    let upper_value = load(UVec2::new(lower.x, upper.y)).lerp(load(upper), weight.x);
    lower_value.lerp(upper_value, weight.y) * 1.0e-3
}

fn log_admitted_camera_projection(camera: &Transform, focus: Vec3) {
    let eye = camera.translation;
    let forward = *camera.forward();
    let right = *camera.right();
    let up = *camera.up();
    let tan_half_vertical = (CHASE_VERTICAL_FOV_DEGREES.to_radians() * 0.5).tan();
    let tan_half_horizontal = tan_half_vertical * PRESENTATION_ASPECT;
    info!(
        "admitted chase camera basis eye={eye:?} forward={forward:?} right={right:?} up={up:?} handedness={:.9}",
        right.cross(up).dot(forward),
    );
    let probes = [
        ("eye_xz", eye.x, eye.z),
        ("focus_xz", focus.x, focus.z),
        ("domain_center", 0.0, 0.0),
        ("far_upwind", -6.0, -6.0),
        ("far_downwind", 6.0, 6.0),
    ];
    for (label, x, z) in probes {
        let Ok(terrain_y) = admitted_terrain_height_km(x, z) else {
            warn!("terrain projection probe {label} refused at ({x:.3},{z:.3})");
            continue;
        };
        let point = Vec3::new(x, terrain_y, z);
        let delta = point - eye;
        let depth = delta.dot(forward);
        let ndc = if depth > 0.0 {
            Vec2::new(
                delta.dot(right) / (depth * tan_half_horizontal),
                delta.dot(up) / (depth * tan_half_vertical),
            )
        } else {
            Vec2::splat(f32::NAN)
        };
        info!(
            "terrain projection probe={label} world={point:?} camera_depth={depth:.6} ndc_approx={ndc:?}"
        );
    }
}

/// Cells the renderer fades at the open lateral boundary. A subject inside
/// that band is drawn through the fade, not at its own density.
const OPEN_LATERAL_BOUNDARY_BAND_CELLS: usize = 3;

/// Integration opacity the accepted presentation pass uses.
const PRODUCTION_INTEGRATION_OPACITY: f32 = 0.48;

/// Share of the frame the bulk of the subject is allowed to span. The rest is
/// the margin the capture gate requires the subject to keep from the edge.
const SUBJECT_FRAME_FILL: f32 = 0.82;

/// Physical clearance the hero subject must keep from the lateral boundary.
/// At 50 m spacing this is the twelve cells the criterion was first measured
/// at; stating it in metres keeps that intent at any spacing instead of
/// demanding twelve kilometres of clearance on a one-kilometre grid.
const LATERAL_BOUNDARY_CLEARANCE_M: f64 = 600.0;

/// Least share of its own bounding box the dominant component may fill.
///
/// This screens out a "component" that is scattered speckle rather than a
/// coherent body: framing such a subject aims the camera at mostly empty air.
/// It is deliberately far below the fill fraction of a compact cloud. The
/// earlier 0.08 floor described one dataset's morphology - a single fat
/// warm-bubble blob nearly filling a small domain - and is simply false for a
/// mesoscale storm, whose dense component is a narrow updraft column under a
/// wide thin anvil sheet and whose honest bounding-box fill is a few percent.
/// Bounding-box fill cannot be made resolution-invariant (the cell volume
/// cancels), so the floor is set where it still catches speckle and nothing
/// else.
const MINIMUM_DOMINANT_BBOX_FILL: f64 = 0.005;

fn source_gate_for_dataset(dataset: &Gm1Dataset) -> String {
    let Some(field) = dataset
        .fields
        .iter()
        .find(|field| field.id == "cloud_condensate")
    else {
        return "REFUSED_NO_EXACT_CLOUD_CONDENSATE".to_owned();
    };
    let profile = resolved_source_profile(&dataset.profile_id);
    let frame = frame_dense_condensate_for_profile(field, profile);
    let Some(morphology) = component_morphology(field, &frame) else {
        return "REFUSED_NO_DOMINANT_Q99_COMPONENT".to_owned();
    };
    let Some(cell_m) = profile.map(|profile| profile.dx_m.min(profile.dy_m)) else {
        return "REFUSED_NO_ADMITTED_GRID_SPACING".to_owned();
    };
    let clearance_cells = morphology.lateral_boundary_distance_cells;
    let clearance_m = clearance_cells as f64 * cell_m;
    let mut reasons = Vec::new();
    // A component clipped by the lateral boundary is drawn with a flat
    // artificial face that is a boundary artefact, not storm structure, and
    // one inside the faded open band is drawn below its own density.
    if clearance_cells <= OPEN_LATERAL_BOUNDARY_BAND_CELLS
        || clearance_m < LATERAL_BOUNDARY_CLEARANCE_M
    {
        reasons.push("BOUNDARY_ADJACENT");
    }
    if morphology.voxel_compactness < MINIMUM_DOMINANT_BBOX_FILL {
        reasons.push("LOW_COMPACTNESS");
    }
    // Physical aspect, in kilometres of the admitted scene: a subject with no
    // vertical development is a fog deck, and a hero volume render of one
    // shows a flat grey sheet rather than cloud structure.
    if morphology.vertical_to_horizontal_ratio < 0.30 {
        reasons.push("FLAT");
    }
    if frame.dominant_mass_fraction < 0.50 {
        reasons.push("FRAGMENTED_MASS");
    }
    let measurements = format!(
        "ratio={:.3} compactness={:.4} boundary={}cells={:.0}m mass={:.3} \
         vertical_km={:.3} horizontal_km={:.3} cells={}",
        morphology.vertical_to_horizontal_ratio,
        morphology.voxel_compactness,
        clearance_cells,
        clearance_m,
        frame.dominant_mass_fraction,
        morphology.vertical_extent_km,
        morphology.maximum_horizontal_extent_km,
        frame.dominant_indices.len(),
    );
    if reasons.is_empty() {
        format!("CANDIDATE_SOURCE_MORPHOLOGY_ONLY {measurements}")
    } else {
        format!("REFUSED_{} {measurements}", reasons.join("_"))
    }
}

fn source_gate_allows_offline_capture(source_gate: &str) -> bool {
    source_gate.starts_with("CANDIDATE_SOURCE_MORPHOLOGY_ONLY ")
}

fn activate_timeline_frame(
    data: &mut ViewerData,
    state: &mut ViewerState,
    timeline: &mut TimelineState,
    index: usize,
) {
    let index = index.min(timeline.frames.len().saturating_sub(1));
    let selected_id = data.0.fields[state.selected_field].id.clone();
    data.0 = timeline.frames[index].dataset.clone();
    timeline.current = index;
    state.selected_field = data
        .0
        .fields
        .iter()
        .position(|field| field.id == selected_id)
        .unwrap_or(0);
    state.loaded_field = usize::MAX;
    state.selected_slice = data.0.fields[state.selected_field].shape[0] / 3;
    info!(
        "exact snapshot timeline selected step {} T+{:.3}s; interpolation=false source_gate={}",
        data.0.step, data.0.model_time_s, timeline.frames[index].source_gate,
    );
}

fn timeline_control(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    capture: Res<CaptureState>,
    egui_input: Res<EguiWantsInput>,
    mut timeline: ResMut<TimelineState>,
    mut data: ResMut<ViewerData>,
    mut state: ResMut<ViewerState>,
) {
    if capture.path.is_some() || !timeline_controls_available(timeline.frames.len()) {
        return;
    }
    let input_permissions = scene_input_permissions(
        offline_capture_requested(capture.path.as_deref()),
        egui_input.wants_any_pointer_input(),
        egui_input.wants_any_keyboard_input(),
    );
    if input_permissions.keyboard {
        if keys.just_pressed(KeyCode::KeyP) {
            timeline.playing = !timeline.playing;
            timeline.follow_live = false;
            timeline.elapsed = 0.0;
        }
        if keys.just_pressed(KeyCode::KeyL) {
            timeline.looping = !timeline.looping;
        }
        if keys.just_pressed(KeyCode::KeyM) {
            timeline.follow_live = !timeline.follow_live;
            if timeline.follow_live {
                timeline.requested = Some(timeline.frames.len() - 1);
            }
        }
        if keys.just_pressed(KeyCode::Comma) {
            timeline.playing = false;
            timeline.follow_live = false;
            timeline.requested = Some(timeline.current.saturating_sub(1));
        }
        if keys.just_pressed(KeyCode::Period) {
            timeline.playing = false;
            timeline.follow_live = false;
            timeline.requested = Some((timeline.current + 1).min(timeline.frames.len() - 1));
        }
    }
    if timeline.playing {
        timeline.elapsed += time.delta_secs();
        if timeline.elapsed >= TIMELINE_PERIOD_SECONDS {
            timeline.elapsed %= TIMELINE_PERIOD_SECONDS;
            let next = timeline.current + 1;
            if next < timeline.frames.len() {
                timeline.requested = Some(next);
            } else if timeline.looping {
                timeline.requested = Some(0);
            } else {
                timeline.playing = false;
            }
        }
    }
    if let Some(index) = timeline.requested.take()
        && index != timeline.current
    {
        activate_timeline_frame(&mut data, &mut state, &mut timeline, index);
    }
}

fn refresh_selected_volume(
    data: Res<ViewerData>,
    mut state: ResMut<ViewerState>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut materials: ResMut<Assets<Gm1VolumeMaterial>>,
    camera: Single<(&mut Transform, &mut ViewerCamera), With<Camera3d>>,
) {
    if state.selected_field == state.loaded_field {
        return;
    }
    let field = &data.0.fields[state.selected_field];
    let Ok(buffer) = storage_from_field(field) else {
        return;
    };
    let buffer = buffers.add(buffer);
    let Some(material_handle) = state.material.as_ref() else {
        return;
    };
    let Some(mut material) = materials.get_mut(material_handle) else {
        return;
    };
    material.density = buffer;
    material.uniform = VolumeUniform::for_field(field, world_min(), world_size());
    material.uniform.sky_amplitude.w = f32::from(!state.lighting_reference_diagnostic);
    // Production integration opacity for every scene, not only the sealed
    // terrain one. The default is high enough that a ray saturates inside the
    // cloud's outermost cell, where the grid-mean condensate is still low: the
    // frame then reports the whole storm at the brightness, colour and
    // scattering of its own onset skin, and none of the structure behind it.
    material.uniform.control.y = PRODUCTION_INTEGRATION_OPACITY;
    if state.admitted_she_terrain {
        material.uniform.world_min.w = 1.0;
    }
    state.loaded_field = state.selected_field;
    state.selected_slice = field.shape[0] / 3;

    let frame = frame_dense_condensate_for_profile(
        field,
        resolved_source_profile(&data.0.profile_id),
    );
    material.uniform.control.z = state.quality.ray_steps();
    material.uniform.optics.w = frame.dense_response_knee;

    state.camera_summary = format!(
        "PCA q≥{:.3} g kg⁻¹ (q99+) • {} cells • yaw {:.1}° • pitch {:.1}° • r {:.1}",
        frame.cutoff * 1.0e3,
        frame.dense_count,
        frame.yaw.to_degrees(),
        frame.pitch.to_degrees(),
        frame.radius,
    );
    let (mut transform, mut controller) = camera.into_inner();
    let (focus, yaw, pitch, radius) = if state.admitted_she_terrain {
        let admitted = admitted_chase_camera(field, &frame);
        (
            admitted.focus,
            admitted.yaw,
            admitted.pitch,
            admitted.radius,
        )
    } else {
        (frame.focus, frame.yaw, frame.pitch, frame.radius)
    };
    controller.focus = focus;
    controller.yaw = yaw;
    controller.pitch = pitch;
    controller.radius = radius;
    let rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
    let forward = rotation * Vec3::NEG_Z;
    transform.translation = focus - forward * radius;
    transform.look_at(focus, Vec3::Y);
}

fn apply_watch_events(
    mut data: ResMut<ViewerData>,
    mut state: ResMut<ViewerState>,
    mut timeline: ResMut<TimelineState>,
    capture: Res<CaptureState>,
    watch: Option<ResMut<LiveWatchState>>,
) {
    // A paired cloud-mask/presentation readback must preserve one exact field
    // and one camera matrix. New monotonic frames remain queued and are
    // admitted by the next process/run after this deterministic capture exits.
    if capture.path.is_some() {
        return;
    }
    let Some(mut watch) = watch else {
        return;
    };
    let events_result: Result<Vec<_>, ()> = {
        match watch.receiver.lock() {
            Ok(receiver) => Ok(receiver.try_iter().collect()),
            Err(_) => Err(()),
        }
    };
    let events = match events_result {
        Ok(events) => events,
        Err(()) => {
            watch.status = "ERROR watch channel mutex poisoned; last good retained".to_owned();
            return;
        }
    };
    for event in events {
        match event {
            WatchEvent::Accepted(dataset)
                if dataset.step
                    > timeline
                        .frames
                        .back()
                        .map(|frame| frame.dataset.step)
                        .unwrap_or(0) =>
            {
                if let Err(reason) = validate_gm1_view_live_profile_transition(
                    &data.0.profile_id,
                    &dataset.profile_id,
                ) {
                    let step = dataset.step;
                    watch.status =
                        "STALE - cross-profile frame refused; last good retained".to_owned();
                    watch.last_refusal = Some(format!(
                        "REFUSED step {step}: {reason} - last good step {} retained",
                        data.0.step,
                    ));
                    warn!("atomic watch refused cross-profile step {step}: {reason}");
                    continue;
                }
                let step = dataset.step;
                let artifact = dataset.artifact_sha256.clone();
                timeline.frames.push_back(TimelineFrame {
                    source_gate: source_gate_for_dataset(&dataset),
                    dataset: Arc::new(dataset),
                });
                while timeline.frames.len() > TIMELINE_CAPACITY {
                    timeline.frames.pop_front();
                    timeline.current = timeline.current.saturating_sub(1);
                }
                if timeline.follow_live {
                    let latest = timeline.frames.len() - 1;
                    activate_timeline_frame(&mut data, &mut state, &mut timeline, latest);
                }
                watch.status = format!("LIVE accepted step {step} • history {}", &artifact[..16]);
                watch.last_refusal = None;
                info!("atomic watch accepted monotonic step {step} ({artifact})");
            }
            WatchEvent::Accepted(_) => {}
            WatchEvent::Refused(refusal) => {
                let step = refusal
                    .step
                    .map(|step| step.to_string())
                    .unwrap_or_else(|| "?".to_owned());
                let summary = format!(
                    "REFUSED step {step}: {} - last good step {} retained",
                    refusal.reason, data.0.step
                );
                warn!("{} ({})", summary, refusal.directory.display());
                watch.status = "STALE - newer frame refused; last good retained".to_owned();
                watch.last_refusal = Some(summary);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_production_series_events(
    mut data: ResMut<ViewerData>,
    mut state: ResMut<ViewerState>,
    mut timeline: ResMut<TimelineState>,
    capture: Res<CaptureState>,
    series: Option<ResMut<ProductionSeriesLiveState>>,
    production_gpu: Option<ResMut<ProductionGpuResidency>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    if capture.path.is_some() {
        return;
    }
    let (Some(mut series), Some(mut production_gpu)) = (series, production_gpu) else {
        return;
    };
    let events_result: Result<Vec<_>, ()> = match series.receiver.lock() {
        Ok(receiver) => Ok(receiver.try_iter().collect::<Vec<_>>()),
        Err(_) => Err(()),
    };
    let events = match events_result {
        Ok(events) => events,
        Err(_) => {
            series.status =
                "STALE - production-series channel poisoned; last good retained".to_owned();
            return;
        }
    };
    for event in events {
        match event {
            ProductionSeriesWatchEvent::Refused(refusal) => {
                let step = refusal
                    .step
                    .map(|step| step.to_string())
                    .unwrap_or_else(|| "?".to_owned());
                let summary = format!(
                    "REFUSED step {step}: {} - last good exact step {} retained",
                    refusal.reason,
                    series.playback.current().step,
                );
                warn!("{} ({})", summary, refusal.path.display());
                series.status = "STALE - production-series update refused".to_owned();
                series.last_refusal = Some(summary);
            }
            ProductionSeriesWatchEvent::Accepted(pair) => {
                let mut candidate = series.playback.clone();
                if let Err(reason) = candidate.accept_pair(pair) {
                    series.status =
                        "STALE - production-series transition refused; last good retained"
                            .to_owned();
                    series.last_refusal = Some(reason);
                    continue;
                }
                let candidate_frame = candidate.current();
                let residency = match candidate_frame.renderer_residency() {
                    Ok(residency) => residency,
                    Err(reason) => {
                        series.status =
                            "STALE - renderer residency refused; last good retained".to_owned();
                        series.last_refusal = Some(reason);
                        continue;
                    }
                };
                let dataset = match candidate_frame.cloud_dataset() {
                    Ok(dataset) => dataset,
                    Err(reason) => {
                        series.status =
                            "STALE - cloud endpoint refused; last good retained".to_owned();
                        series.last_refusal = Some(reason);
                        continue;
                    }
                };
                if let Err(reason) = validate_production_renderer_envelope(&dataset, &residency) {
                    series.status =
                        "STALE - renderer envelope refused; last good retained".to_owned();
                    series.last_refusal = Some(reason);
                    continue;
                }
                let gpu_fields = match upload_production_gpu_fields(&residency, &mut buffers) {
                    Ok(fields) => fields,
                    Err(reason) => {
                        series.status =
                            "STALE - exact GPU residency refused; last good retained".to_owned();
                        series.last_refusal = Some(reason);
                        continue;
                    }
                };
                let step = dataset.step;
                let manifest_sha256 = residency.manifest_sha256.clone();
                let payload_set_sha256 = residency.payload_set_sha256.clone();
                production_gpu.adapter_id = residency.frame.adapter_id;
                production_gpu.source_profile = residency.source_profile;
                production_gpu.manifest_sha256 = manifest_sha256.clone();
                production_gpu.payload_set_sha256 = payload_set_sha256;
                production_gpu.fields = gpu_fields;
                timeline.frames.push_back(TimelineFrame {
                    source_gate: source_gate_for_dataset(&dataset),
                    dataset: Arc::new(dataset),
                });
                while timeline.frames.len() > TIMELINE_CAPACITY {
                    timeline.frames.pop_front();
                    timeline.current = timeline.current.saturating_sub(1);
                }
                let latest = timeline.frames.len() - 1;
                activate_timeline_frame(&mut data, &mut state, &mut timeline, latest);
                series.playback = candidate;
                series.rendered_alpha_f32_bits = 1.0f32.to_bits();
                series.status = format!(
                    "LIVE exact A/B endpoints admitted through step {step}; manifest {}",
                    &manifest_sha256[..16],
                );
                series.last_refusal = None;
                info!(
                    "strict production series accepted exact A/B generation {step}; newest endpoint displayed"
                );
            }
        }
    }
}

fn production_temporal_cloud_field(
    pair: &ProductionFramePair,
    alpha: f32,
) -> Result<crate::input::VolumeField, String> {
    let cloud = pair.blend_field_exact_f32("cloud_condensate_kgkg", alpha)?;
    let shape: [usize; 3] = cloud
        .shape
        .clone()
        .try_into()
        .map_err(|_| "production temporal cloud is not rank three".to_owned())?;
    let finite_min = cloud.values.iter().copied().fold(f32::INFINITY, f32::min);
    let finite_max = cloud
        .values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    Ok(crate::input::VolumeField {
        id: "cloud_condensate".to_owned(),
        label: "Resolved cloud condensate temporal A/B display".to_owned(),
        units: cloud.units,
        meaning: format!(
            "{}; source-word A/B blend only; {}",
            cloud.role, EXACT_F32_BLEND_FORMULA
        ),
        source: format!(
            "strict production A/B endpoints {:?}; alpha_f32_bits={:08x}",
            [pair.a.step, pair.b.step],
            alpha.to_bits(),
        ),
        shape,
        values: cloud.values.as_ref().clone(),
        values_sha256: cloud.values_sha256,
        finite_min,
        finite_max,
    })
}

fn apply_production_series_interpolation(
    series: Option<ResMut<ProductionSeriesLiveState>>,
    state: Res<ViewerState>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut materials: ResMut<Assets<Gm1VolumeMaterial>>,
) {
    let Some(mut series) = series else {
        return;
    };
    let alpha = series.playback.alpha();
    if series.rendered_alpha_f32_bits == alpha.to_bits() {
        return;
    }
    let Some(pair) = series.playback.pair().cloned() else {
        return;
    };
    let field = match production_temporal_cloud_field(&pair, alpha) {
        Ok(field) => field,
        Err(reason) => {
            series.status = "STALE - temporal field refused; exact endpoint retained".to_owned();
            series.last_refusal = Some(reason);
            return;
        }
    };
    let storage = match storage_from_field(&field) {
        Ok(storage) => storage,
        Err(reason) => {
            series.status = "STALE - temporal upload refused; exact endpoint retained".to_owned();
            series.last_refusal = Some(reason);
            return;
        }
    };
    let Some(mut material) = state
        .material
        .as_ref()
        .and_then(|handle| materials.get_mut(handle))
    else {
        return;
    };
    material.density = buffers.add(storage);
    series.rendered_alpha_f32_bits = alpha.to_bits();
    info!(
        "production temporal cloud displayed: steps={:?} alpha_f32_bits={:08x} values_sha256={}",
        [pair.a.step, pair.b.step],
        alpha.to_bits(),
        field.values_sha256,
    );
}

fn lighting_probe_summary(field: &crate::input::VolumeField, uniform: &VolumeUniform) -> String {
    let [nz, ny, nx] = field.shape;
    let Some((maximum_index, maximum)) = field
        .values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.total_cmp(&right.1))
    else {
        return "shadow_probe=REFUSED_NO_FINITE_FIELD".to_owned();
    };
    let k = maximum_index / (ny * nx);
    let remainder = maximum_index % (ny * nx);
    let j = remainder / nx;
    let i = remainder % nx;
    let position = Vec3::new(
        i as f32 / (nx - 1) as f32,
        j as f32 / (ny - 1) as f32,
        k as f32 / (nz - 1) as f32,
    );
    let sun_world = uniform.sun_direction.xyz().normalize();
    let sun_texture = Vec3::new(
        sun_world.x / uniform.world_size.x,
        sun_world.z / uniform.world_size.z,
        sun_world.y / uniform.world_size.y,
    )
    .normalize();
    let offsets = [0.008_f32, 0.018, 0.034, 0.056, 0.082, 0.115];
    let threshold = uniform.threshold();
    let density_limit = uniform.world_size.w.max(threshold + 1.0e-8);
    let mut normalized = [0.0_f32; 6];
    let mut optical = [0.0_f32; 6];
    for (slot, offset) in offsets.into_iter().enumerate() {
        let probe = position + sun_texture * offset;
        if probe.cmple(Vec3::ONE).all() && probe.cmpge(Vec3::ZERO).all() {
            let density = trilinear_field_sample(field, probe);
            normalized[slot] =
                ((density - threshold) / (density_limit - threshold)).clamp(0.0, 1.0);
            optical[slot] = normalized[slot].powf(1.35) * open_boundary_probe_weight(probe, nx, ny);
        }
    }
    let optical4 = optical[..4].iter().sum::<f32>();
    let optical6 = optical.iter().sum::<f32>();
    format!(
        "shadow_probe=max[k,j,i]=[{k},{j},{i}] bits=0x{:08x} value={maximum:.9e} texture={position:?} sun_texture={sun_texture:?} normalized={normalized:?} optical4={optical4:.9} transmittance_v13={:.9} optical6={optical6:.9} transmittance_candidate={:.9}",
        maximum.to_bits(),
        (-0.46 * optical4).exp(),
        (-0.42 * optical6).exp(),
    )
}

fn trilinear_field_sample(field: &crate::input::VolumeField, position: Vec3) -> f32 {
    let [nz, ny, nx] = field.shape;
    let maximum = Vec3::new((nx - 1) as f32, (ny - 1) as f32, (nz - 1) as f32);
    let grid = position.clamp(Vec3::ZERO, Vec3::ONE) * maximum;
    let lower = grid.floor().as_uvec3();
    let upper = (lower + UVec3::ONE).min(UVec3::new(
        (nx - 1) as u32,
        (ny - 1) as u32,
        (nz - 1) as u32,
    ));
    let weight = grid.fract();
    let load = |point: UVec3| {
        field.values[(point.z as usize * ny + point.y as usize) * nx + point.x as usize]
    };
    let c000 = load(UVec3::new(lower.x, lower.y, lower.z));
    let c100 = load(UVec3::new(upper.x, lower.y, lower.z));
    let c010 = load(UVec3::new(lower.x, upper.y, lower.z));
    let c110 = load(UVec3::new(upper.x, upper.y, lower.z));
    let c001 = load(UVec3::new(lower.x, lower.y, upper.z));
    let c101 = load(UVec3::new(upper.x, lower.y, upper.z));
    let c011 = load(UVec3::new(lower.x, upper.y, upper.z));
    let c111 = load(UVec3::new(upper.x, upper.y, upper.z));
    let c00 = c000.lerp(c100, weight.x);
    let c10 = c010.lerp(c110, weight.x);
    let c01 = c001.lerp(c101, weight.x);
    let c11 = c011.lerp(c111, weight.x);
    c00.lerp(c10, weight.y)
        .lerp(c01.lerp(c11, weight.y), weight.z)
}

fn open_boundary_probe_weight(position: Vec3, nx: usize, ny: usize) -> f32 {
    let cell = position.xy() * Vec2::new((nx - 1) as f32, (ny - 1) as f32);
    let maximum = Vec2::new((nx - 1) as f32, (ny - 1) as f32);
    let edge_distance = cell
        .x
        .min(maximum.x - cell.x)
        .min(cell.y.min(maximum.y - cell.y));
    let t = (edge_distance / 3.0).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
fn frame_dense_condensate(field: &crate::input::VolumeField) -> FieldFrame {
    frame_dense_condensate_for_profile(field, None)
}

fn frame_dense_condensate_for_profile(
    field: &crate::input::VolumeField,
    source_profile: Option<Gm1ViewSourceProfile>,
) -> FieldFrame {
    let [nz, ny, nx] = field.shape;
    let terrain = source_profile
        .is_none_or(|profile| profile.terrain == Gm1ViewTerrainClass::TerrainFollowingSource)
        .then_some(())
        .filter(|()| nz == 75 && ny == ADMITTED_TERRAIN_NY && nx == ADMITTED_TERRAIN_NX)
        .and_then(|()| admitted_terrain_values().ok());
    let mut visible_count = 0usize;
    let mut visible_min = Vec3::splat(f32::INFINITY);
    let mut visible_max = Vec3::splat(f32::NEG_INFINITY);
    // Running first and second moments of the visible cloud's positions. The
    // raw bounding box is set by the furthest stray wisp, which on a mesoscale
    // domain can sit tens of kilometres from the storm; framing on it shrinks
    // the storm to a speck. These give an outlier-resistant extent instead.
    let mut visible_sum = [0.0_f64; 3];
    let mut visible_square_sum = [0.0_f64; 3];

    let world_position = |index: usize| {
        let k = index / (ny * nx);
        let remainder = index % (ny * nx);
        let j = remainder / nx;
        let i = remainder % nx;
        let horizontal = world_min()
            + Vec3::new(
                (i as f32 + 0.5) / nx as f32,
                0.0,
                (j as f32 + 0.5) / ny as f32,
            ) * world_size();
        let y = terrain
            .as_ref()
            .and_then(|values| admitted_mass_height_km(k, values[j * nx + i]).ok())
            .unwrap_or(world_min().y + (k as f32 + 0.5) / nz as f32 * world_size().y);
        Vec3::new(horizontal.x, y, horizontal.z)
    };

    let mut ordered = field
        .values
        .iter()
        .enumerate()
        .filter(|(index, _)| interior_mass_cell(*index, ny, nx))
        .map(|(_, value)| value)
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if ordered.is_empty() {
        return FieldFrame {
            focus: world_min() + world_size() * 0.5,
            yaw: -0.62,
            pitch: CAMERA_PITCH,
            radius: 24.0,
            cutoff: DENSE_FRAME_MIN_CUTOFF,
            dense_response_knee: DENSE_FRAME_MIN_CUTOFF,
            dense_count: 0,
            dense_min: world_min(),
            dense_max: world_min() + world_size(),
            visible_count: 0,
            visible_min: world_min(),
            visible_max: world_min() + world_size(),
            dominant_indices: Vec::new(),
            dominant_mass: 0.0,
            connected_mass: 0.0,
            dominant_mass_fraction: 0.0,
        };
    }
    let q99_index = (ordered.len() - 1) * 99 / 100;
    ordered.select_nth_unstable_by(q99_index, f32::total_cmp);
    let dense_cutoff = ordered[q99_index].max(DENSE_FRAME_MIN_CUTOFF);
    // Where the display stops treating a sample as onset haze and starts
    // treating it as fully dense cloud.
    //
    // A quantile over every interior cell answers "how much of the domain is
    // clear air", not "how dense is the cloud". On a domain a cloud nearly
    // fills, the two coincide. On one where cloud occupies a few percent of
    // cells the whole-domain quantile lands three orders of magnitude below any
    // condensate and collapses onto its floor, so the storm's own body is
    // classified as haze: it is then drawn at haze brightness, haze colour and
    // haze scattering, and the render reports a thin veil where the model has a
    // resolved cloud. Taking the quantile over the cells that carry cloud is
    // the same measurement on either domain.
    let mut cloudy = ordered
        .iter()
        .copied()
        .filter(|value| *value >= PHYSICAL_CLOUD_ONSET)
        .collect::<Vec<_>>();
    let dense_response_knee = if cloudy.is_empty() {
        dense_cutoff
    } else {
        let knee_index = (cloudy.len() - 1) * 3 / 4;
        cloudy.select_nth_unstable_by(knee_index, f32::total_cmp);
        cloudy[knee_index].max(PHYSICAL_CLOUD_ONSET * 2.0)
    };

    for (index, &value) in field.values.iter().enumerate() {
        if !interior_mass_cell(index, ny, nx) {
            continue;
        }
        if value >= PHYSICAL_CLOUD_ONSET {
            let position = world_position(index);
            visible_count += 1;
            visible_min = visible_min.min(position);
            visible_max = visible_max.max(position);
            for axis in 0..3 {
                let value = f64::from(position[axis]);
                visible_sum[axis] += value;
                visible_square_sum[axis] += value * value;
            }
        }
    }

    let (dominant_indices, dominant_mass, connected_mass) =
        dominant_connected_component(field, dense_cutoff);
    let dense_count = dominant_indices.len();
    let mut weight_sum = 0.0_f64;
    let mut weighted_x = 0.0_f64;
    let mut weighted_z = 0.0_f64;
    let mut dense_min = Vec3::splat(f32::INFINITY);
    let mut dense_max = Vec3::splat(f32::NEG_INFINITY);
    for &index in &dominant_indices {
        let position = world_position(index);
        let weight = f64::from(field.values[index]).powi(3);
        weight_sum += weight;
        weighted_x += weight * f64::from(position.x);
        weighted_z += weight * f64::from(position.z);
        dense_min = dense_min.min(position);
        dense_max = dense_max.max(position);
    }

    if dense_count == 0 || weight_sum == 0.0 {
        return FieldFrame {
            focus: world_min() + world_size() * 0.5,
            yaw: -0.62,
            pitch: CAMERA_PITCH,
            radius: 24.0,
            cutoff: dense_cutoff,
            dense_response_knee,
            dense_count: 0,
            dense_min: world_min(),
            dense_max: world_min() + world_size(),
            visible_count,
            visible_min: if visible_count == 0 {
                world_min()
            } else {
                visible_min
            },
            visible_max: if visible_count == 0 {
                world_min() + world_size()
            } else {
                visible_max
            },
            dominant_indices,
            dominant_mass,
            connected_mass,
            dominant_mass_fraction: 0.0,
        };
    }

    let mean_x = weighted_x / weight_sum;
    let mean_z = weighted_z / weight_sum;
    let mut covariance_xx = 0.0_f64;
    let mut covariance_xz = 0.0_f64;
    let mut covariance_zz = 0.0_f64;
    for &index in &dominant_indices {
        let value = field.values[index];
        let position = world_position(index);
        let weight = f64::from(value).powi(3);
        let dx = f64::from(position.x) - mean_x;
        let dz = f64::from(position.z) - mean_z;
        covariance_xx += weight * dx * dx;
        covariance_xz += weight * dx * dz;
        covariance_zz += weight * dz * dz;
    }
    let major_angle = 0.5 * (2.0 * covariance_xz).atan2(covariance_xx - covariance_zz);
    // View across the minor axis with a small oblique rotation, then choose
    // the sign that places the sealed physical sun behind the dense cloud for
    // an informative forward-scattering edge.
    let view_angle = major_angle + std::f64::consts::FRAC_PI_2 + 0.28;
    let mut forward = Vec3::new(view_angle.cos() as f32, 0.0, view_angle.sin() as f32);
    let sun_enu = T300_SUN_ENU_F32_BITS.map(f32::from_bits);
    let sun_horizontal = Vec3::new(sun_enu[0], 0.0, sun_enu[1]).normalize_or_zero();
    if forward.dot(sun_horizontal) < 0.0 {
        forward = -forward;
    }
    let yaw = (-forward.x).atan2(-forward.z);
    // Frame the dense subject the render actually resolves, and stand far
    // enough back to clear the cloud around it.
    //
    // Three scales are in play and only one of them is the subject. The q99
    // connected core is what the volume render draws solidly. Around it sits
    // the rest of the cloud, most of which is thin enough to read as haze. Its
    // raw bounding box is set by the furthest stray cell, which on a mesoscale
    // domain can sit tens of kilometres downshear.
    //
    // A standoff measured in absolute kilometres for a 15 km domain leaves the
    // camera inside the cloud on a wider one, so every ray starts in cloud and
    // the frame fills with its interior at one uniform luminance. A standoff
    // measured from the raw bounding box pushes the camera so far back that the
    // storm covers a few percent of the frame. So: frame the core, and take as
    // a floor the distance that clears the bulk of the surrounding cloud -
    // two standard deviations of its own position about its mean, which the
    // stray cells cannot move.
    let (bulk_min, bulk_max) = if visible_count > 0 {
        let count = visible_count as f64;
        let mut low = [0.0_f32; 3];
        let mut high = [0.0_f32; 3];
        for axis in 0..3 {
            let mean = visible_sum[axis] / count;
            let variance = (visible_square_sum[axis] / count - mean * mean).max(0.0);
            let spread = 2.0 * variance.sqrt();
            low[axis] = (mean - spread) as f32;
            high[axis] = (mean + spread) as f32;
        }
        (
            Vec3::from_array(low).max(visible_min),
            Vec3::from_array(high).min(visible_max),
        )
    } else {
        (dense_min, dense_max)
    };
    let core_extent = dense_max - dense_min;
    let horizontal_span = core_extent.x.max(core_extent.z);
    let core_span = core_extent.max_element().max(1.0e-3);
    let bulk_clearance = 0.5 * (bulk_max - bulk_min).length();
    // Finally, stand back far enough that the bulk of the cloud fits inside
    // the frame at this camera's own field of view. Without this the standoff
    // is chosen entirely from the subject's size and says nothing about how
    // much of it the lens can see: a storm whose core is small but whose body
    // is broad is framed from inside its own body, and the capture gate
    // rightly refuses a portrait that is really a crop.
    // The thing that has to fit is the whole visible cloud, not just its bulk:
    // the capture gate measures what was drawn, and a veil that leaves the
    // frame is a crop however little of the cloud's mass it carries.
    let (fit_min, fit_max) = if visible_count > 0 {
        (visible_min.min(dense_min), visible_max.max(dense_max))
    } else {
        (bulk_min, bulk_max)
    };
    let fit_extent = fit_max - fit_min;
    let bulk_horizontal = fit_extent.x.max(fit_extent.z);
    let tan_half_vertical = (CHASE_VERTICAL_FOV_DEGREES.to_radians() * 0.5).tan();
    let tan_half_horizontal = tan_half_vertical * PRESENTATION_ASPECT;
    // Looking down at the scene, the subject's own depth projects onto the
    // frame's vertical axis alongside its height.
    let projected_half_height = 0.5
        * (fit_extent.y * CAMERA_PITCH.cos().abs()
            + bulk_horizontal * CAMERA_PITCH.sin().abs());
    let fit_radius = (projected_half_height / tan_half_vertical)
        .max(0.5 * bulk_horizontal / tan_half_horizontal)
        / SUBJECT_FRAME_FILL;
    let mut radius = (horizontal_span * 1.35 + core_span * 0.17)
        .clamp(core_span, core_span * 1.8)
        .max(bulk_clearance)
        .max(fit_radius);
    let rotation = Quat::from_euler(EulerRot::YXZ, yaw, CAMERA_PITCH, 0.0);
    let screen_right = rotation * Vec3::X;
    // Aim at the middle of what has to fit, biased a little toward the dense
    // core so the storm's own body still holds the centre of the frame. Aiming
    // at the core alone leaves an anvil that streams downshear from it hanging
    // off one edge, which is the same crop the capture gate refuses.
    let fit_center = (fit_min + fit_max) * 0.5;
    let centroid = Vec3::new(mean_x as f32, fit_center.y, mean_z as f32);
    let focus = fit_center.lerp(centroid, 0.30) + screen_right * (horizontal_span * 0.09);
    // Then grow the standoff until the corners of the fitting box actually
    // project inside the frame.
    //
    // Measuring the fit at the focus plane is not enough: a cloud tens of
    // kilometres deep has a near face much closer to the camera than its
    // centre, and that face projects correspondingly larger. On a domain wide
    // enough for the storm to be deep in view, the arithmetic estimate says it
    // fits while the render shows it running off both edges, which is the crop
    // the capture gate refuses.
    let screen_up = rotation * Vec3::Y;
    let corners = [
        Vec3::new(fit_min.x, fit_min.y, fit_min.z),
        Vec3::new(fit_max.x, fit_min.y, fit_min.z),
        Vec3::new(fit_min.x, fit_max.y, fit_min.z),
        Vec3::new(fit_max.x, fit_max.y, fit_min.z),
        Vec3::new(fit_min.x, fit_min.y, fit_max.z),
        Vec3::new(fit_max.x, fit_min.y, fit_max.z),
        Vec3::new(fit_min.x, fit_max.y, fit_max.z),
        Vec3::new(fit_max.x, fit_max.y, fit_max.z),
    ];
    for _ in 0..64 {
        let eye = focus - forward * radius;
        let escapes = corners.iter().any(|corner| {
            let delta = *corner - eye;
            let depth = delta.dot(forward);
            if depth <= 1.0e-3 {
                return true;
            }
            let ndc_x = delta.dot(screen_right) / (depth * tan_half_horizontal);
            let ndc_y = delta.dot(screen_up) / (depth * tan_half_vertical);
            ndc_x.abs() > SUBJECT_FRAME_FILL || ndc_y.abs() > SUBJECT_FRAME_FILL
        });
        if !escapes {
            break;
        }
        radius *= 1.06;
    }
    FieldFrame {
        focus,
        yaw,
        pitch: CAMERA_PITCH,
        radius,
        cutoff: dense_cutoff,
        dense_response_knee,
        dense_count,
        dense_min,
        dense_max,
        visible_count,
        visible_min,
        visible_max,
        dominant_indices,
        dominant_mass,
        connected_mass,
        dominant_mass_fraction: dominant_mass / connected_mass,
    }
}

fn dominant_connected_component(
    field: &crate::input::VolumeField,
    cutoff: f32,
) -> (Vec<usize>, f64, f64) {
    let [nz, ny, nx] = field.shape;
    let layer = ny * nx;
    let active = |index: usize| {
        interior_mass_cell(index, ny, nx)
            && field.values[index].is_finite()
            && field.values[index] >= cutoff
    };
    let mut visited = vec![false; field.values.len()];
    let mut queue = VecDeque::new();
    let mut dominant = Vec::new();
    let mut dominant_mass = 0.0_f64;
    let mut connected_mass = 0.0_f64;

    for seed in 0..field.values.len() {
        if visited[seed] || !active(seed) {
            continue;
        }
        visited[seed] = true;
        queue.push_back(seed);
        let mut component = Vec::new();
        let mut component_mass = 0.0_f64;
        while let Some(index) = queue.pop_front() {
            component.push(index);
            component_mass += f64::from(field.values[index]);
            let k = index / layer;
            let remainder = index % layer;
            let j = remainder / nx;
            let i = remainder % nx;
            let neighbours = [
                (i > 0).then(|| index - 1),
                (i + 1 < nx).then(|| index + 1),
                (j > 0).then(|| index - nx),
                (j + 1 < ny).then(|| index + nx),
                (k > 0).then(|| index - layer),
                (k + 1 < nz).then(|| index + layer),
            ];
            for neighbour in neighbours.into_iter().flatten() {
                if !visited[neighbour] && active(neighbour) {
                    visited[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        connected_mass += component_mass;
        if component_mass > dominant_mass {
            dominant_mass = component_mass;
            dominant = component;
        }
    }
    (dominant, dominant_mass, connected_mass)
}

fn component_morphology(
    field: &crate::input::VolumeField,
    frame: &FieldFrame,
) -> Option<ComponentMorphology> {
    let [nz, ny, nx] = field.shape;
    if frame.dominant_indices.is_empty() || nx == 0 || ny == 0 || nz == 0 {
        return None;
    }
    let layer = ny * nx;
    let mut grid_min = UVec3::splat(u32::MAX);
    let mut grid_max = UVec3::ZERO;
    for &index in &frame.dominant_indices {
        let k = index / layer;
        let remainder = index % layer;
        let j = remainder / nx;
        let i = remainder % nx;
        let point = UVec3::new(i as u32, j as u32, k as u32);
        grid_min = grid_min.min(point);
        grid_max = grid_max.max(point);
    }
    let extent = grid_max - grid_min + UVec3::ONE;
    let bbox_cell_count = extent.x as usize * extent.y as usize * extent.z as usize;
    let voxel_compactness = frame.dominant_indices.len() as f64 / bbox_cell_count as f64;
    let lateral_boundary_distance_cells = [
        grid_min.x as usize,
        grid_min.y as usize,
        nx - 1 - grid_max.x as usize,
        ny - 1 - grid_max.y as usize,
    ]
    .into_iter()
    .min()
    .unwrap_or(0);
    let vertical_extent_km = frame.dense_max.y - frame.dense_min.y;
    let maximum_horizontal_extent_km =
        (frame.dense_max.x - frame.dense_min.x).max(frame.dense_max.z - frame.dense_min.z);
    Some(ComponentMorphology {
        grid_min,
        grid_max,
        bbox_cell_count,
        voxel_compactness,
        lateral_boundary_distance_cells,
        vertical_extent_km,
        maximum_horizontal_extent_km,
        vertical_to_horizontal_ratio: vertical_extent_km
            / maximum_horizontal_extent_km.max(f32::MIN_POSITIVE),
    })
}

fn interior_mass_cell(index: usize, ny: usize, nx: usize) -> bool {
    if nx <= 6 || ny <= 6 {
        return false;
    }
    let remainder = index % (ny * nx);
    let j = remainder / nx;
    let i = remainder % nx;
    (3..nx - 3).contains(&i) && (3..ny - 3).contains(&j)
}

fn child_enu_to_wgs84(east_km: f32, north_km: f32, up_km: f32) -> (f64, f64) {
    const EARTH_RADIUS_KM: f64 = 6_371.0;
    const CENTER_LAT_DEG: f64 = 41.394_640_411_35;
    const CENTER_LON_DEG: f64 = -94.282_682_936_87;
    let latitude = CENTER_LAT_DEG.to_radians();
    let longitude = CENTER_LON_DEG.to_radians();
    let (sin_latitude, cos_latitude) = latitude.sin_cos();
    let (sin_longitude, cos_longitude) = longitude.sin_cos();
    let up = DVec3::new(
        cos_latitude * cos_longitude,
        cos_latitude * sin_longitude,
        sin_latitude,
    );
    let east = DVec3::new(-sin_longitude, cos_longitude, 0.0);
    let north = DVec3::new(
        -sin_latitude * cos_longitude,
        -sin_latitude * sin_longitude,
        cos_latitude,
    );
    let ecef = up * EARTH_RADIUS_KM
        + east * f64::from(east_km)
        + north * f64::from(north_km)
        + up * f64::from(up_km);
    let geocentric_latitude = (ecef.z / ecef.length()).asin().to_degrees();
    let geocentric_longitude = ecef.y.atan2(ecef.x).to_degrees();
    (geocentric_latitude, geocentric_longitude)
}

fn spawn_chase_vehicle(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    east_north_km: Vec2,
    yaw: f32,
) {
    let terrain_km = camera_terrain_height_km(east_north_km.x, east_north_km.y)
        .expect("initial chase vehicle lies on admitted terrain");
    let paint = materials.add(StandardMaterial {
        base_color: Color::srgb(0.055, 0.13, 0.21),
        metallic: 0.82,
        perceptual_roughness: 0.24,
        ..default()
    });
    let glass = materials.add(StandardMaterial {
        base_color: Color::srgba(0.20, 0.34, 0.43, 0.72),
        metallic: 0.15,
        perceptual_roughness: 0.10,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let tire = materials.add(StandardMaterial {
        base_color: Color::srgb(0.018, 0.021, 0.024),
        metallic: 0.02,
        perceptual_roughness: 0.90,
        ..default()
    });
    let lamp = materials.add(StandardMaterial {
        base_color: Color::srgb(0.98, 0.82, 0.52),
        emissive: LinearRgba::new(5.0, 3.7, 1.7, 1.0),
        ..default()
    });
    commands
        .spawn((
            Name::new("GM1 View source-terrain chase vehicle (display geometry)"),
            ChaseVehicleVisual,
            ChaseVehicle {
                east_north_km,
                yaw,
                speed_km_s: 0.0,
                reset_east_north_km: east_north_km,
                reset_yaw: yaw,
                camera_mode: VehicleCameraMode::Chase,
                grounded: true,
                terrain_status: "EXACT TERRAIN CONTACT".to_owned(),
            },
            Transform::from_xyz(east_north_km.x, terrain_km, east_north_km.y)
                .with_rotation(Quat::from_rotation_y(yaw)),
            Visibility::Visible,
            NoFrustumCulling,
        ))
        .with_children(|vehicle| {
            vehicle.spawn((
                Name::new("chase vehicle chassis"),
                Mesh3d(meshes.add(Cuboid::new(0.00205, 0.00078, 0.00455))),
                MeshMaterial3d(paint.clone()),
                Transform::from_xyz(0.0, 0.00102, 0.0),
            ));
            vehicle.spawn((
                Name::new("chase vehicle cockpit"),
                Mesh3d(meshes.add(Cuboid::new(0.00172, 0.00078, 0.00195))),
                MeshMaterial3d(glass),
                Transform::from_xyz(0.0, 0.00172, 0.00018),
            ));
            let wheel_mesh = meshes.add(Cuboid::new(0.00042, 0.00066, 0.00088));
            for x in [-0.00112, 0.00112] {
                for z in [-0.00148, 0.00148] {
                    vehicle.spawn((
                        Name::new("chase vehicle terrain contact wheel"),
                        Mesh3d(wheel_mesh.clone()),
                        MeshMaterial3d(tire.clone()),
                        Transform::from_xyz(x, 0.00048, z),
                    ));
                }
            }
            let lamp_mesh = meshes.add(Cuboid::new(0.00043, 0.00024, 0.00012));
            for x in [-0.00062, 0.00062] {
                vehicle.spawn((
                    Name::new("chase vehicle headlamp"),
                    Mesh3d(lamp_mesh.clone()),
                    MeshMaterial3d(lamp.clone()),
                    Transform::from_xyz(x, 0.00108, -0.00232),
                ));
            }
        });
    info!(
        "spawned deterministic chase vehicle at east/north={east_north_km:?} terrain_km={terrain_km:.6}; geometry=display-only terrain=science-source-exact"
    );
}

fn drive_terrain_height_km(
    terrain: &DriveTerrain,
    east_km: f32,
    north_km: f32,
) -> Result<f32, String> {
    if east_km.abs() <= 7.475 && north_km.abs() <= 7.475 {
        Ok(admitted_terrain_height_km_from_values(
            &terrain.child_msl_m,
            east_km,
            north_km,
        ))
    } else {
        terrain.surround.sample_height_km(east_km, north_km)
    }
}

fn drive_ground_pose(
    terrain: &DriveTerrain,
    east_north_km: Vec2,
    yaw: f32,
) -> Result<(Vec3, Quat), String> {
    const HALF_WHEELBASE_KM: f32 = 0.00155;
    const HALF_TRACK_KM: f32 = 0.00092;
    const CHASSIS_CLEARANCE_KM: f32 = 0.00034;
    let yaw_rotation = Quat::from_rotation_y(yaw);
    let forward_xz = (yaw_rotation * Vec3::NEG_Z).xz().normalize();
    let right_xz = (yaw_rotation * Vec3::X).xz().normalize();
    let center = drive_terrain_height_km(terrain, east_north_km.x, east_north_km.y)?;
    let front = drive_terrain_height_km(
        terrain,
        east_north_km.x + forward_xz.x * HALF_WHEELBASE_KM,
        east_north_km.y + forward_xz.y * HALF_WHEELBASE_KM,
    )?;
    let rear = drive_terrain_height_km(
        terrain,
        east_north_km.x - forward_xz.x * HALF_WHEELBASE_KM,
        east_north_km.y - forward_xz.y * HALF_WHEELBASE_KM,
    )?;
    let right_height = drive_terrain_height_km(
        terrain,
        east_north_km.x + right_xz.x * HALF_TRACK_KM,
        east_north_km.y + right_xz.y * HALF_TRACK_KM,
    )?;
    let left_height = drive_terrain_height_km(
        terrain,
        east_north_km.x - right_xz.x * HALF_TRACK_KM,
        east_north_km.y - right_xz.y * HALF_TRACK_KM,
    )?;
    let forward = Vec3::new(
        forward_xz.x,
        (front - rear) / (2.0 * HALF_WHEELBASE_KM),
        forward_xz.y,
    )
    .normalize();
    let approximate_right = Vec3::new(
        right_xz.x,
        (right_height - left_height) / (2.0 * HALF_TRACK_KM),
        right_xz.y,
    )
    .normalize();
    let up = approximate_right.cross(forward).normalize();
    let right = forward.cross(up).normalize();
    let rotation = Quat::from_mat3(&Mat3::from_cols(right, up, -forward));
    Ok((
        Vec3::new(
            east_north_km.x,
            center + CHASSIS_CLEARANCE_KM,
            east_north_km.y,
        ),
        rotation,
    ))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn chase_vehicle_control(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    capture: Res<CaptureState>,
    egui_input: Res<EguiWantsInput>,
    state: Res<ViewerState>,
    terrain: Option<Res<DriveTerrain>>,
    mut vehicles: Query<(&mut Transform, &mut ChaseVehicle), With<ChaseVehicleVisual>>,
    camera: Single<
        (&mut Transform, &mut ViewerCamera),
        (With<Camera3d>, Without<ChaseVehicleVisual>),
    >,
) {
    if !state.drive_mode {
        return;
    }
    let Some(terrain) = terrain else {
        return;
    };
    let Ok((mut transform, mut vehicle)) = vehicles.single_mut() else {
        return;
    };
    let dt = time.delta_secs().min(0.05);
    let input_permissions = scene_input_permissions(
        offline_capture_requested(capture.path.as_deref()),
        egui_input.wants_any_pointer_input(),
        egui_input.wants_any_keyboard_input(),
    );
    if input_permissions.keyboard {
        if keys.just_pressed(KeyCode::KeyC) {
            vehicle.camera_mode = match vehicle.camera_mode {
                VehicleCameraMode::Cockpit => VehicleCameraMode::Chase,
                VehicleCameraMode::Chase => VehicleCameraMode::Cockpit,
            };
        }
        if keys.just_pressed(KeyCode::KeyR) {
            vehicle.east_north_km = vehicle.reset_east_north_km;
            vehicle.yaw = vehicle.reset_yaw;
            vehicle.speed_km_s = 0.0;
        }
        let throttle = f32::from(keys.pressed(KeyCode::KeyW));
        let brake_or_reverse = f32::from(keys.pressed(KeyCode::KeyS));
        if throttle > 0.0 {
            vehicle.speed_km_s += 0.010 * dt;
        }
        if brake_or_reverse > 0.0 {
            vehicle.speed_km_s -= if vehicle.speed_km_s > 0.0 {
                0.022
            } else {
                0.007
            } * dt;
        }
        if keys.pressed(KeyCode::Space) {
            vehicle.speed_km_s *= (-8.0 * dt).exp();
        } else if throttle == 0.0 && brake_or_reverse == 0.0 {
            vehicle.speed_km_s *= (-0.42 * dt).exp();
        }
        vehicle.speed_km_s = vehicle.speed_km_s.clamp(-0.012, 0.042);
        let steer = f32::from(keys.pressed(KeyCode::KeyA)) - f32::from(keys.pressed(KeyCode::KeyD));
        if steer != 0.0 && vehicle.speed_km_s.abs() > 0.0002 {
            let direction = vehicle.speed_km_s.signum();
            let response = (vehicle.speed_km_s.abs() / 0.012).clamp(0.15, 1.0);
            vehicle.yaw += steer * direction * response * 1.30 * dt;
        }
        let forward = Quat::from_rotation_y(vehicle.yaw) * Vec3::NEG_Z;
        let candidate = vehicle.east_north_km + forward.xz() * vehicle.speed_km_s * dt;
        match drive_ground_pose(&terrain, candidate, vehicle.yaw) {
            Ok((translation, rotation)) => {
                vehicle.east_north_km = candidate;
                vehicle.grounded = true;
                vehicle.terrain_status = if candidate.x.abs() <= 7.475 && candidate.y.abs() <= 7.475
                {
                    "EXACT 50 m CHILD TERRAIN CONTACT".to_owned()
                } else {
                    "SOURCE-BACKED CURVED 3 km HRRR CONTACT".to_owned()
                };
                transform.translation = translation;
                transform.rotation = rotation;
            }
            Err(error) => {
                vehicle.speed_km_s = 0.0;
                vehicle.grounded = false;
                vehicle.terrain_status = format!("BOUNDARY COLLISION: {error}");
            }
        }
    }
    if vehicle.grounded
        && let Ok((translation, rotation)) =
            drive_ground_pose(&terrain, vehicle.east_north_km, vehicle.yaw)
    {
        transform.translation = translation;
        transform.rotation = rotation;
    }
    let forward = transform.rotation * Vec3::NEG_Z;
    let up = transform.rotation * Vec3::Y;
    let (eye, focus) = match vehicle.camera_mode {
        VehicleCameraMode::Cockpit => {
            let eye = transform.translation + up * 0.00162 + forward * 0.00055;
            (eye, eye + forward * 0.090 + up * 0.004)
        }
        VehicleCameraMode::Chase => (
            transform.translation + up * 0.0048 - forward * 0.014,
            transform.translation + up * 0.0014 + forward * 0.010,
        ),
    };
    let (mut camera_transform, mut controller) = camera.into_inner();
    *camera_transform = Transform::from_translation(eye).looking_at(focus, up);
    let direction = (focus - eye).normalize();
    controller.focus = focus;
    controller.yaw = (-direction.x).atan2(-direction.z);
    controller.pitch = direction.y.asin();
    controller.radius = eye.distance(focus);
}

#[allow(clippy::too_many_arguments)]
fn camera_control(
    camera: Single<(&mut Transform, &mut ViewerCamera), With<Camera3d>>,
    state: Res<ViewerState>,
    capture: Res<CaptureState>,
    egui_input: Res<EguiWantsInput>,
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
) {
    // Pin the exact camera across the cloud-object mask and the combined
    // presentation pass. No input event may move it between readbacks.
    if offline_capture_requested(capture.path.as_deref()) {
        return;
    }
    if state.drive_mode {
        return;
    }
    let (mut transform, mut controller) = camera.into_inner();
    let input_permissions = scene_input_permissions(
        false,
        egui_input.wants_any_pointer_input(),
        egui_input.wants_any_keyboard_input(),
    );
    if input_permissions.pointer && buttons.pressed(MouseButton::Right) {
        controller.yaw -= motion.delta.x * 0.0045;
        controller.pitch = (controller.pitch - motion.delta.y * 0.0045).clamp(-1.45, 1.45);
    }
    let rotation = Quat::from_euler(EulerRot::YXZ, controller.yaw, controller.pitch, 0.0);
    let forward = rotation * Vec3::NEG_Z;
    if state.fly_mode {
        let mut movement = Vec3::ZERO;
        if input_permissions.keyboard && keys.pressed(KeyCode::KeyW) {
            movement += forward;
        }
        if input_permissions.keyboard && keys.pressed(KeyCode::KeyS) {
            movement -= forward;
        }
        if input_permissions.keyboard && keys.pressed(KeyCode::KeyD) {
            movement += rotation * Vec3::X;
        }
        if input_permissions.keyboard && keys.pressed(KeyCode::KeyA) {
            movement -= rotation * Vec3::X;
        }
        if input_permissions.keyboard && keys.pressed(KeyCode::KeyE) {
            movement += Vec3::Y;
        }
        if input_permissions.keyboard && keys.pressed(KeyCode::KeyQ) {
            movement -= Vec3::Y;
        }
        if movement.length_squared() > 0.0 {
            transform.translation += movement.normalize() * FLY_SPEED_KM_S * time.delta_secs();
        }
        transform.rotation = rotation;
        controller.focus = transform.translation + forward * controller.radius;
    } else {
        if input_permissions.pointer {
            // Zoom limits are fractions of the scene, so a domain an order of
            // magnitude wider can still be backed out of. Absolute limits
            // measured for one domain leave a larger one impossible to see
            // whole and a smaller one impossible to approach.
            let scene_span = world_size().x.max(world_size().z);
            controller.radius = (controller.radius * (-scroll.delta.y * 0.08).exp())
                .clamp(scene_span / 32.0, scene_span * 4.0);
        }
        transform.translation = controller.focus - forward * controller.radius;
        transform.look_at(controller.focus, Vec3::Y);
    }
}

fn frame_profile(
    time: Res<Time>,
    data: Res<ViewerData>,
    viewer: Res<ViewerState>,
    mut profiler: ResMut<FrameProfiler>,
    mut exit: MessageWriter<AppExit>,
) {
    profiler.record(time.delta_secs(), viewer.quality);
    let Some(path) = profiler.receipt_path.clone() else {
        return;
    };
    if profiler.written || profiler.elapsed_seconds < PROFILE_WARMUP_SECONDS + profiler.soak_seconds
    {
        return;
    }
    profiler.written = true;
    let mut samples = profiler.samples_ms[..profiler.sample_count].to_vec();
    samples.sort_by(f32::total_cmp);
    if samples.is_empty() {
        error!("frame profile produced no post-warmup samples");
        exit.write(AppExit::error());
        return;
    }
    let percentile =
        |fraction: f32| samples[((samples.len() - 1) as f32 * fraction).round() as usize];
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    let median = percentile(0.5);
    let p95 = percentile(0.95);
    let p99 = percentile(0.99);
    let worst = *samples.last().expect("non-empty profile samples");
    let minimum_samples = (profiler.soak_seconds * 15.0) as usize;
    let latency_passed =
        samples.len() >= minimum_samples && mean <= 33.334 && p95 <= 50.0 && worst <= 250.0;
    let quality = profiler.sampled_quality.unwrap_or(viewer.quality);
    let quality_stable = !profiler.mixed_quality;
    let passed = latency_passed && quality_stable;
    let status = if !quality_stable {
        "REFUSED_MIXED_QUALITY_PROFILE"
    } else if !latency_passed {
        "REFUSED_INTERACTIVE_FRAME_LATENCY"
    } else {
        match quality {
            QualityProfile::Live => "PASS_RESPONSIVE_LIVE_PROFILE",
            QualityProfile::Cinema => "PASS_RESPONSIVE_CINEMA_PROFILE",
        }
    };
    let Some(adapter) = profiler.adapter.as_ref() else {
        error!("profile receipt has no independently probed DX12 adapter identity");
        exit.write(AppExit::error());
        return;
    };
    let receipt = FrameProfileReceipt {
        schema: "gm1.view.interactive-frame-profile/v1",
        status,
        quality: quality.label(),
        quality_stable,
        backend_required: "DX12 on the recorded NVIDIA adapter",
        backend_actual: &adapter.backend,
        adapter_name: &adapter.name,
        adapter_vendor: adapter.vendor,
        adapter_device: adapter.device,
        adapter_driver: &adapter.driver,
        adapter_driver_info: &adapter.driver_info,
        warmup_seconds: PROFILE_WARMUP_SECONDS,
        soak_seconds: profiler.soak_seconds,
        samples: samples.len(),
        mean_frame_ms: mean,
        median_frame_ms: median,
        p95_frame_ms: p95,
        p99_frame_ms: p99,
        worst_frame_ms: worst,
        mean_fps: 1_000.0 / mean,
        exact_step: data.0.step,
        exact_payload_sha256: &data.0.payload_sha256,
        target: "mean <= 33.334 ms, p95 <= 50 ms, worst <= 250 ms after 5 s warmup",
        scientific_interpolation: false,
        per_frame_readback: false,
    };
    let result = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("profile receipt serialization failed: {error}"))
        .and_then(|bytes| {
            fs::write(&path, bytes).map_err(|error| {
                format!(
                    "profile receipt write failed at {}: {error}",
                    path.display()
                )
            })
        });
    match result {
        Ok(()) => {
            info!(
                "{} profile wrote {} samples to {}: mean={mean:.3}ms p95={p95:.3}ms p99={p99:.3}ms worst={worst:.3}ms",
                receipt.status,
                samples.len(),
                path.display(),
            );
            if passed {
                exit.write(AppExit::Success);
            } else {
                exit.write(AppExit::error());
            }
        }
        Err(error) => {
            error!("{error}");
            exit.write(AppExit::error());
        }
    }
}

fn science_layer_update(data: Res<ViewerData>, mut science: ResMut<ScienceViewState>) {
    if !science.show {
        return;
    }
    let event = science
        .receiver
        .lock()
        .ok()
        .and_then(|receiver| receiver.try_recv().ok());
    if let Some(event) = event {
        let key = (event.step, event.layer);
        if science.requested == Some(key) {
            match event.result {
                Ok(raster) => {
                    science.status = format!(
                        "PASS exact step {} • {} • RGBA {}",
                        raster.step,
                        raster.layer.label(),
                        &raster.rgba_sha256[..16],
                    );
                    science.rendered = Some(raster);
                }
                Err(reason) => {
                    science.status = reason;
                    science.rendered = None;
                }
            }
        }
    }

    let key = (data.0.step, science.selected);
    if science.requested == Some(key) {
        return;
    }
    science.requested = Some(key);
    science.rendered = None;
    science.status = format!(
        "validating exact step {} • {} off-thread",
        key.0,
        key.1.label()
    );
    let dataset = data.0.clone();
    let layer = science.selected;
    let root = science.root.clone();
    let sender = science.sender.clone();
    let spawn = std::thread::Builder::new()
        .name(format!("gm1-science-{}-{}", dataset.step, layer.as_str()))
        .spawn(move || {
            let result = build_science_raster(dataset.as_ref(), root.as_deref(), layer);
            let _ = sender.send(ScienceWorkerEvent {
                step: dataset.step,
                layer,
                result,
            });
        });
    if let Err(error) = spawn {
        science.status = format!("REFUSED science worker launch failed: {error}");
        science.requested = None;
    }
}

/// Grid the science layers project on: the resolved source profile's, or the
/// sealed table's when the frame is one the sealed table describes.
fn science_projection_grid(dataset: &Gm1Dataset) -> Result<GeoGrid, String> {
    resolved_source_profile(&dataset.profile_id)
        .map(Gm1ViewSourceProfile::georeferenced_grid)
        .ok_or_else(|| {
            format!(
                "no admitted grid for source profile {:?}; a science layer \
                 projected without one would report optical path, spacing and \
                 extent from a domain this frame was not simulated on",
                dataset.profile_id
            )
        })
}

/// Surface and scalar-level altitudes for this frame's own domain.
///
/// A frame that publishes its `terrain_msl`/`zf` carriers states its own
/// geometry; only a frame that publishes neither falls back to the sealed
/// terrain-following child assets, and then only when its grid is that child's.
fn frame_surface_and_levels(dataset: &Gm1Dataset) -> Result<(Vec<f32>, Option<Vec<f32>>), String> {
    let [nz, ny, nx] = dataset.fields[0].shape;
    if let Some(terrain) = dataset.terrain_msl_m.as_ref() {
        return Ok((terrain.clone(), dataset.face_height_m.clone()));
    }
    if nz == SEALED_SCALAR_LEVEL_COUNT && ny == ADMITTED_TERRAIN_NY && nx == ADMITTED_TERRAIN_NX {
        return Ok((admitted_terrain_values()?, None));
    }
    Err(format!(
        "frame grid {:?} (nz,ny,nx) publishes no surface carrier and is not the \
         sealed {SEALED_SCALAR_LEVEL_COUNT}x{ADMITTED_TERRAIN_NY}x{ADMITTED_TERRAIN_NX} \
         child the sealed relief was measured on; projecting it against that \
         relief would put another domain's hills under this domain's air",
        dataset.fields[0].shape
    ))
}

fn build_science_raster(
    dataset: &Gm1Dataset,
    science_root: Option<&Path>,
    layer: ScienceLayerId,
) -> Result<ScienceRaster, String> {
    let grid = science_projection_grid(dataset)?;
    if layer == ScienceLayerId::SimSatTopView {
        let (terrain, faces) = frame_surface_and_levels(dataset)?;
        let levels = match faces.as_deref() {
            Some(faces) => ColumnLevelFaces::Source(faces),
            None => ColumnLevelFaces::SealedSigma,
        };
        let product =
            simsat_physical_top_view(dataset, &terrain, levels, grid.dx_m, grid.dy_m)?;
        return Ok(science_raster_from_simsat(dataset.step, product));
    }

    if layer.surface_carrier().is_some() {
        // CM1's diagnostic surface carriers travel in the forecast frame
        // itself, not in a science pack: the run that computed them published
        // them beside the state they describe, and a pack copy could disagree
        // with the frame it is drawn over.
        let product = tornado_surface_top_view(dataset, layer)?;
        return Ok(science_raster_from_scalar(dataset.step, layer, product));
    }

    let registry = ScienceLayerRegistry::load(science_root, std::slice::from_ref(dataset), &grid);
    let availability = registry.layer(dataset, layer);
    match availability {
        ScienceLayerAvailability::Available(science) => {
            let product = scalar_top_view(&science)?;
            Ok(science_raster_from_scalar(dataset.step, layer, product))
        }
        ScienceLayerAvailability::Derived(descriptor) => Err(format!(
            "REFUSED unexpected derived layer {}: {}",
            descriptor.id, descriptor.admission_reason
        )),
        ScienceLayerAvailability::Refused(descriptor)
        | ScienceLayerAvailability::Missing(descriptor) => Err(format!(
            "{} • {}",
            descriptor.admission_reason,
            if registry.refusals().is_empty() {
                descriptor.provenance
            } else {
                registry.refusals().join(" | ")
            }
        )),
    }
}

fn science_raster_from_simsat(step: u64, product: TopViewProduct) -> ScienceRaster {
    ScienceRaster {
        step,
        layer: ScienceLayerId::SimSatTopView,
        width: product.width,
        height: product.height,
        rgba: product.rgba,
        rgba_sha256: product.rgba_sha256,
        descriptor: product.descriptor,
        range: [product.optical_depth_min, product.optical_depth_max],
        summary: format!(
            "column optical depth {:.4} … {:.4}",
            product.optical_depth_min, product.optical_depth_max
        ),
    }
}

fn science_raster_from_scalar(
    step: u64,
    layer: ScienceLayerId,
    product: ScalarTopView,
) -> ScienceRaster {
    ScienceRaster {
        step,
        layer,
        width: product.width,
        height: product.height,
        rgba: product.rgba,
        rgba_sha256: product.rgba_sha256,
        descriptor: product.descriptor,
        range: product.range,
        summary: format!(
            "north-up exact column projection {:.5e} … {:.5e}",
            product.range[0], product.range[1]
        ),
    }
}

fn production_field_inspector_ui(
    ui: &mut egui::Ui,
    production: &ProductionGpuResidency,
    inspector: &mut ProductionInspectorState,
    cache: &mut ProductionInspectorCache,
    step: u64,
    model_time_s: f64,
) {
    if production.fields.is_empty() {
        ui.colored_label(
            egui::Color32::from_rgb(164, 66, 42),
            "REFUSED • no exact resident fields",
        );
        return;
    }
    inspector.clamp_to(&production.fields);
    let mut selected_field = inspector.selected_field;
    egui::ComboBox::from_label("Resident source field")
        .selected_text(&production.fields[selected_field].descriptor.label)
        .show_ui(ui, |ui| {
            for (index, field) in production.fields.iter().enumerate() {
                ui.selectable_value(
                    &mut selected_field,
                    index,
                    format!(
                        "{} [{}] • {}",
                        field.descriptor.label, field.descriptor.units, field.stagger,
                    ),
                );
            }
        });
    if selected_field != inspector.selected_field {
        inspector.select_field(selected_field, &production.fields);
    }
    inspector.clamp_to(&production.fields);
    let field = &production.fields[inspector.selected_field];
    match field.descriptor.shape.as_slice() {
        [nz, _, _] => {
            ui.add(
                egui::Slider::new(&mut inspector.selected_level, 0..=nz.saturating_sub(1))
                    .text("native vertical index k"),
            );
        }
        [_, _] => {
            inspector.selected_level = 0;
            ui.small("2-D source surface • no vertical index");
        }
        _ => {}
    }

    ui.add_enabled_ui(false, |ui| {
        egui::ComboBox::from_label("Exact time sample")
            .selected_text(format!("step {step} • T+{model_time_s:.3} s"))
            .show_ui(ui, |ui| {
                ui.label(format!("step {step} • T+{model_time_s:.3} s"));
            });
    });
    ui.small("One admitted whole snapshot only • no temporal interpolation or generated frame.");
    ui.strong(&field.descriptor.label);
    ui.monospace(format!(
        "{} [{}] • {} • {:?}",
        field.descriptor.id, field.descriptor.units, field.stagger, field.residency,
    ));
    ui.label(&field.descriptor.meaning);
    ui.monospace(format!("native shape {:?}", field.descriptor.shape));
    ui.monospace(format!("field SHA-256 {}", field.descriptor.values_sha256));
    ui.monospace(format!(
        "whole field words {} • nonzero {} • min {:.8e} (0x{:08x}) • max {:.8e} (0x{:08x})",
        field.statistics.word_count,
        field.statistics.nonzero_count,
        field.statistics.minimum,
        field.statistics.minimum.to_bits(),
        field.statistics.maximum,
        field.statistics.maximum.to_bits(),
    ));

    match exact_field_level(field, inspector.selected_level) {
        Ok(level) => {
            if cache.field_sha256 != field.descriptor.values_sha256
                || cache.level != inspector.selected_level
            {
                cache
                    .field_sha256
                    .clone_from(&field.descriptor.values_sha256);
                cache.level = inspector.selected_level;
                cache.level_sha256 = f32_sha256(level);
                cache.statistics = exact_field_statistics(level).ok();
            }
            if let Some(statistics) = cache.statistics {
                ui.monospace(format!(
                    "selected plane k={} • words {} • nonzero {} • min {:.8e} • max {:.8e}",
                    inspector.selected_level,
                    statistics.word_count,
                    statistics.nonzero_count,
                    statistics.minimum,
                    statistics.maximum,
                ));
                ui.monospace(format!("plane SHA-256 {}", cache.level_sha256));
            }
        }
        Err(reason) => {
            ui.colored_label(egui::Color32::from_rgb(164, 66, 42), reason);
        }
    }
    ui.small(format!(
        "{} • {}",
        production.source_profile.vertical_grid_label(),
        production.source_profile.terrain_label(),
    ));
    ui.small(
        "Read-only exact-field inspector. Its selection never changes the cloud-density buffer or opacity transfer function; no zeta, reflectivity, remap, normalization, clamping, or quantization is performed.",
    );
}

#[allow(clippy::too_many_arguments)]
fn scientific_overlay(
    mut contexts: EguiContexts,
    data: Res<ViewerData>,
    mut state: ResMut<ViewerState>,
    mut science: ResMut<ScienceViewState>,
    mut timeline: ResMut<TimelineState>,
    mut materials: ResMut<Assets<Gm1VolumeMaterial>>,
    live_watch: Option<Res<LiveWatchState>>,
    production: Option<Res<ProductionGpuResidency>>,
    mut production_series: Option<ResMut<ProductionSeriesLiveState>>,
    mut production_inspector: ResMut<ProductionInspectorState>,
    vehicles: Query<(&ChaseVehicle, &Transform), With<ChaseVehicleVisual>>,
    profiler: Res<FrameProfiler>,
    mut cache: Local<SliceCache>,
    mut science_cache: Local<ScienceTextureCache>,
    mut production_cache: Local<ProductionInspectorCache>,
    mut shell_cache: Local<ScientificShellUiCache>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    install_scientific_shell_visuals_once(ctx, &mut shell_cache);

    let field = &data.0.fields[state.selected_field];
    if state.presentation {
        // Presentation captures intentionally hide the interactive science
        // panel. Exact field/terrain/SHE identities and all display treatment
        // labels remain in the machine-readable receipt and startup log.
        return Ok(());
    }
    let panel_height = science_panel_height(ctx.content_rect().height());
    egui::Window::new(SCIENCE_PANEL_ID)
        .id(egui::Id::new(SCIENCE_PANEL_ID))
        .anchor(
            egui::Align2::RIGHT_TOP,
            [-SCIENCE_PANEL_MARGIN, SCIENCE_PANEL_MARGIN],
        )
        .fixed_size([SCIENCE_PANEL_WIDTH, panel_height])
        .title_bar(false)
        .resizable(false)
        .collapsible(false)
        .movable(false)
        .vscroll(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("GM1 View");
                ui.monospace(format!("{:4.0} FPS", profiler.fps()));
            });
            ui.strong(
                resolved_source_profile(&data.0.profile_id)
                    .map(|profile| profile.scientific_label)
                    .unwrap_or("Strict admitted GM1 state"),
            );
            ui.monospace(format!(
                "step {}  •  T+{:.1} s  •  {}",
                data.0.step,
                data.0.model_time_s,
                state.quality.label(),
            ));
            ui.small("RMB look  •  WASD move  •  Q/E down/up  •  fly camera");

            egui::CollapsingHeader::new("Frame")
                .id_salt(SHELL_FRAME_SECTION_ID)
                .default_open(true)
                .show(ui, |ui| {
                    if let Some(production) = production.as_ref() {
                        let total_nbytes = production
                            .fields
                            .iter()
                            .map(|field| field.nbytes)
                            .sum::<usize>();
                        ui.colored_label(
                            egui::Color32::from_rgb(20, 105, 74),
                            format!(
                                "PASS • {} native fields • {:.3} MiB exact-f32 resident",
                                production.fields.len(),
                                total_nbytes as f64 / (1024.0 * 1024.0),
                            ),
                        );
                        if let Some(series) = production_series.as_mut() {
                            ui.label(&series.status);
                            if let Some(refusal) = series.last_refusal.as_deref() {
                                ui.colored_label(
                                    egui::Color32::from_rgb(164, 66, 42),
                                    refusal,
                                );
                            }
                            let endpoints = series
                                .playback
                                .pair()
                                .map(|pair| [pair.a.step, pair.b.step]);
                            if let Some(endpoints) = endpoints {
                                let mut alpha = series.playback.alpha();
                                ui.monospace(format!(
                                    "exact resident endpoints {} → {}",
                                    endpoints[0], endpoints[1]
                                ));
                                if ui
                                    .add(
                                        egui::Slider::new(&mut alpha, 0.0..=1.0)
                                            .text("Temporal A/B alpha"),
                                    )
                                    .changed()
                                    && let Err(reason) = series.playback.set_alpha(alpha)
                                {
                                    series.status =
                                        "STALE - invalid temporal control; last good retained"
                                            .to_owned();
                                    series.last_refusal = Some(reason);
                                }
                            }
                        }
                    }
                    if let Some(watch) = live_watch.as_ref() {
                        ui.colored_label(
                            egui::Color32::from_rgb(20, 105, 74),
                            &watch.status,
                        );
                        if let Some(refusal) = watch.last_refusal.as_ref() {
                            ui.colored_label(
                                egui::Color32::from_rgb(164, 66, 42),
                                refusal,
                            );
                        }
                    }
                    if timeline_controls_available(timeline.frames.len()) {
                        ui.horizontal(|ui| {
                            if ui
                                .button("◀")
                                .on_hover_text("previous exact snapshot")
                                .clicked()
                            {
                                timeline.playing = false;
                                timeline.follow_live = false;
                                timeline.requested =
                                    Some(timeline.current.saturating_sub(1));
                            }
                            if ui
                                .button(if timeline.playing { "Pause" } else { "Play" })
                                .clicked()
                            {
                                timeline.playing = !timeline.playing;
                                timeline.follow_live = false;
                                timeline.elapsed = 0.0;
                            }
                            if ui
                                .button("▶")
                                .on_hover_text("next exact snapshot")
                                .clicked()
                            {
                                timeline.playing = false;
                                timeline.follow_live = false;
                                timeline.requested = Some(
                                    (timeline.current + 1).min(timeline.frames.len() - 1),
                                );
                            }
                            ui.checkbox(&mut timeline.looping, "Loop");
                            if live_controls_available(
                                timeline.frames.len(),
                                live_watch.is_some() || production_series.is_some(),
                            ) && ui
                                .checkbox(&mut timeline.follow_live, "Live")
                                .changed()
                                && timeline.follow_live
                            {
                                timeline.requested = Some(timeline.frames.len() - 1);
                            }
                        });
                        let mut selected = timeline.current;
                        if ui
                            .add(
                                egui::Slider::new(
                                    &mut selected,
                                    0..=timeline.frames.len() - 1,
                                )
                                .integer()
                                .text("snapshot"),
                            )
                            .changed()
                        {
                            timeline.playing = false;
                            timeline.follow_live = false;
                            timeline.requested = Some(selected);
                        }
                        ui.small(
                            "Whole verified snapshots only • no generated frames or scientific interpolation.",
                        );
                    } else {
                        ui.colored_label(
                            egui::Color32::from_rgb(94, 88, 72),
                            "INITIALIZED SINGLE FRAME • playback and live controls unavailable",
                        );
                    }
                });

            egui::CollapsingHeader::new("Render")
                .id_salt(SHELL_RENDER_SECTION_ID)
                .default_open(true)
                .show(ui, |ui| {
                    let previous_quality = state.quality;
                    egui::ComboBox::from_label("Quality")
                        .selected_text(state.quality.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut state.quality,
                                QualityProfile::Live,
                                "Scientific Live",
                            );
                            ui.selectable_value(
                                &mut state.quality,
                                QualityProfile::Cinema,
                                "Scientific Cinema",
                            );
                        });
                    if state.quality != previous_quality
                        && let Some(mut material) = state
                            .material
                            .as_ref()
                            .and_then(|handle| materials.get_mut(handle))
                    {
                        material.uniform.control.z = state.quality.ray_steps();
                    }

                    let render_controls = state.material.as_ref().and_then(|handle| {
                        materials.get(handle).map(|material| {
                            (
                                material.uniform.control.x,
                                material.uniform.control.y,
                                material.uniform.optics.w,
                                material.uniform.sun_color.w,
                                material.uniform.scattering.y,
                                material.uniform.grid_shape.w != 0,
                            )
                        })
                    });
                    if !state.she_comparison
                        && !state.she_glossy_comparison
                        && let Some((
                            mut threshold,
                            mut opacity,
                            _q99,
                            _direct_exposure,
                            _ground_albedo,
                            mut fade_open_boundary,
                        )) = render_controls
                    {
                        let upper = field.finite_max.max(f32::MIN_POSITIVE);
                        let mut controls_changed = ui
                            .add(
                                egui::Slider::new(&mut threshold, 0.0..=upper)
                                    .logarithmic(true)
                                    .text("cloud threshold"),
                            )
                            .changed();
                        controls_changed |= ui
                            .add(egui::Slider::new(&mut opacity, 0.15..=4.0).text("opacity"))
                            .changed();
                        ui.checkbox(&mut state.show_slice, "2D slice + contour");
                        controls_changed |= ui
                            .checkbox(
                                &mut fade_open_boundary,
                                "Fade exact open 3-cell boundary band",
                            )
                            .changed();
                        if controls_changed
                            && let Some(mut material) = state
                                .material
                                .as_ref()
                                .and_then(|handle| materials.get_mut(handle))
                        {
                            material.uniform.control.x = threshold;
                            material.uniform.control.y = opacity;
                            material.uniform.grid_shape.w = u32::from(fade_open_boundary);
                        }
                    }
                    ui.small(format!(
                        "{} exact-field ray steps • display controls never alter source words",
                        state.quality.ray_steps() as u32,
                    ));
                });

            egui::CollapsingHeader::new("Fields")
                .id_salt(SHELL_FIELDS_SECTION_ID)
                .default_open(false)
                .show(ui, |ui| {
                    egui::ComboBox::from_label("Variable")
                        .selected_text(&data.0.fields[state.selected_field].label)
                        .show_ui(ui, |ui| {
                            for (index, candidate) in data.0.fields.iter().enumerate() {
                                ui.selectable_value(
                                    &mut state.selected_field,
                                    index,
                                    format!("{} [{}]", candidate.label, candidate.units),
                                );
                            }
                        });
                    ui.strong(&field.label);
                    ui.monospace(format!("{} [{}]", field.id, field.units));
                    ui.label(&field.meaning);
                    ui.monospace(format!("shape [z,y,x] = {:?}", field.shape));
                    ui.monospace(format!(
                        "range {:.5e} … {:.5e}",
                        field.finite_min, field.finite_max
                    ));
                    if let Some(production) = production.as_ref() {
                        egui::CollapsingHeader::new("Native v2 field inspector")
                            .id_salt("gm1-shell-native-v2-field-inspector")
                            .default_open(false)
                            .show(ui, |ui| {
                                production_field_inspector_ui(
                                    ui,
                                    production,
                                    &mut production_inspector,
                                    &mut production_cache,
                                    data.0.step,
                                    data.0.model_time_s,
                                );
                            });
                    }
                });

            egui::CollapsingHeader::new("Science")
                .id_salt(SHELL_SCIENCE_SECTION_ID)
                .default_open(false)
                .show(ui, |ui| {
                    ui.checkbox(&mut science.show, "Show north-up science view");
                    egui::ComboBox::from_label("Science layer")
                        .selected_text(science.selected.label())
                        .show_ui(ui, |ui| {
                            for candidate in ScienceLayerId::ALL {
                                ui.selectable_value(
                                    &mut science.selected,
                                    candidate,
                                    candidate.label(),
                                );
                            }
                        });
                    let status_color = if science.status.starts_with("PASS") {
                        egui::Color32::from_rgb(20, 105, 74)
                    } else if science.status.starts_with("validating")
                        || science.status.starts_with("waiting")
                    {
                        egui::Color32::from_rgb(82, 91, 101)
                    } else {
                        egui::Color32::from_rgb(164, 66, 42)
                    };
                    ui.colored_label(status_color, &science.status);
                    ui.small(
                        "Unavailable carriers remain refused. No tornado classification or invented diagnostics.",
                    );
                });

            if state.drive_mode
                && let Ok((vehicle, vehicle_transform)) = vehicles.single()
            {
                egui::CollapsingHeader::new("Experimental vehicle")
                    .id_salt(SHELL_VEHICLE_SECTION_ID)
                    .default_open(false)
                    .show(ui, |ui| {
                        let (latitude, longitude) = child_enu_to_wgs84(
                            vehicle.east_north_km.x,
                            vehicle.east_north_km.y,
                            vehicle_transform.translation.y,
                        );
                        ui.strong(format!(
                            "{:?} • {:5.1} km h⁻¹",
                            vehicle.camera_mode,
                            vehicle.speed_km_s * 3600.0,
                        ));
                        ui.monospace(format!(
                            "ENU E{:+.3} N{:+.3} km • {:.6}°, {:.6}°",
                            vehicle.east_north_km.x,
                            vehicle.east_north_km.y,
                            latitude,
                            longitude,
                        ));
                        ui.colored_label(
                            if vehicle.grounded {
                                egui::Color32::from_rgb(20, 105, 74)
                            } else {
                                egui::Color32::from_rgb(164, 66, 42)
                            },
                            &vehicle.terrain_status,
                        );
                        ui.small(
                            "W/S throttle • A/D steer • Space brake • C camera • R reset",
                        );
                    });
            }

            egui::CollapsingHeader::new("Provenance")
                .id_salt(SHELL_PROVENANCE_SECTION_ID)
                .default_open(false)
                .show(ui, |ui| {
                    ui.label(format!(
                        "step {} • T+{:.1} s • Δt {:.3} s",
                        data.0.step, data.0.model_time_s, data.0.dt_s
                    ));
                    ui.label(format!("{} • {}", data.0.artifact_kind, data.0.profile_id));
                    if let Some(profile) =
                        resolved_source_profile(&data.0.profile_id)
                    {
                        ui.strong(profile.scientific_label);
                        ui.monospace(format!("profile {}", profile.id));
                        ui.monospace(format!("domain {}", profile.domain_label()));
                        ui.monospace(format!("vertical {}", profile.vertical_grid_label()));
                        ui.monospace(format!("terrain {}", profile.terrain_label()));
                        if profile.terrain == Gm1ViewTerrainClass::FlatZeroTerrain {
                            ui.label("Idealized model clock • no UTC or solar geolocation");
                        } else {
                            ui.label(format!(
                                "{} • sun {:.2}° / az {:.2}°",
                                T300_VALID_TIME_UTC,
                                T300_SOLAR_ELEVATION_DEG,
                                T300_SOLAR_AZIMUTH_DEG,
                            ));
                        }
                    }
                    if let Some(production) = production.as_ref() {
                        ui.monospace(format!("adapter {}", production.adapter_id));
                        ui.monospace(format!(
                            "manifest {} • payload set {}",
                            &production.manifest_sha256[..16],
                            &production.payload_set_sha256[..16],
                        ));
                    }
                    if let Some(series) = production_series.as_ref() {
                        ui.monospace(format!("series {}", series.root.display()));
                    }
                    if let Some(watch) = live_watch.as_ref() {
                        ui.monospace(format!("watch {}", watch.root.display()));
                    }
                    ui.monospace(format!("field {}", &field.values_sha256[..16]));
                    ui.monospace(format!("artifact {}", &data.0.artifact_sha256[..16]));
                    ui.monospace(format!("contract {}", &data.0.contract_sha256[..16]));
                    ui.monospace(format!("sky {}", &T300_SKY_ROW_SHA256[..16]));
                    ui.small(&state.camera_summary);
                    if !state.admitted_she_terrain {
                        ui.small(&state.build_summary);
                    }
                    if let Some(summary) = state.she_summary.as_ref() {
                        ui.monospace(summary);
                    }
                    ui.small(if state.admitted_she_terrain {
                        "Exact source-specific terrain display • physical SimSat sky/direct sun • raw meteorological words unchanged."
                    } else {
                        "Exact binary32 storage • Beer–Lambert display transfer • no procedural cloud detail or source-word changes."
                    });
                    ui.small(
                        "INITIALIZED DATA VIEW • no tornado classification or generated meteorology.",
                    );
                    if state.she_comparison || state.she_glossy_comparison || state.admitted_she_terrain {
                        ui.small(format!(
                            "{} @ {} (MIT)",
                            SHE_UPSTREAM_PROJECT,
                            &SHE_UPSTREAM_COMMIT[..12],
                        ));
                    }
                });
        });

    if science.show {
        let current_key = (data.0.step, science.selected);
        let current_raster = science
            .rendered
            .as_ref()
            .filter(|raster| (raster.step, raster.layer) == current_key);
        if let Some(raster) = current_raster
            && science_texture_cache_requires_update(
                science_cache.key.as_ref(),
                raster.step,
                raster.layer,
                &raster.rgba_sha256,
            )
        {
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [raster.width, raster.height],
                &raster.rgba,
            );
            match science_cache.texture.as_mut() {
                Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                None => {
                    science_cache.texture = Some(ctx.load_texture(
                        "gm1-strict-science-layer",
                        image,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }
            science_cache.key = Some((raster.step, raster.layer, raster.rgba_sha256.clone()));
        }
        egui::Window::new("North-up exact-data science view")
            .id(egui::Id::new(SCIENCE_TOP_VIEW_ID))
            .anchor(egui::Align2::LEFT_TOP, [SCIENCE_PANEL_MARGIN, SCIENCE_PANEL_MARGIN])
            .fixed_size([540.0, 650.0])
            .title_bar(false)
            .resizable(false)
            .collapsible(false)
            .movable(false)
            .vscroll(true)
            .show(ctx, |ui| {
                ui.heading(science.selected.label());
                ui.monospace(format!(
                    "exact step {} • T+{:.1} s • north up",
                    data.0.step, data.0.model_time_s
                ));
                if let (Some(raster), Some(texture)) =
                    (current_raster, science_cache.texture.as_ref())
                {
                    let side = ui.available_width().min(500.0);
                    ui.add(
                        egui::Image::new(texture)
                            .fit_to_exact_size(egui::vec2(side, side)),
                    );
                    ui.monospace(&raster.summary);
                    ui.monospace(format!(
                        "display range {:.5e} … {:.5e}",
                        raster.range[0], raster.range[1]
                    ));
                    ui.monospace(format!("RGBA {}", &raster.rgba_sha256[..16]));
                    ui.small(format!(
                        "{} [{}] • {}",
                        raster.descriptor.label,
                        raster.descriptor.units,
                        raster.descriptor.admission_reason,
                    ));
                    ui.small(&raster.descriptor.provenance);
                    ui.small(
                        "Source-backed display transform of exact admitted carriers; no meteorological interpolation, generated detail, or synthetic weather.",
                    );
                } else {
                    ui.spinner();
                    ui.colored_label(
                        egui::Color32::from_rgb(164, 66, 42),
                        &science.status,
                    );
                }
            });
    }

    if state.show_slice {
        let [nz, ny, nx] = field.shape;
        state.selected_slice = state.selected_slice.min(nz - 1);
        egui::Window::new("Scientific horizontal slice")
            .id(egui::Id::new(SCIENCE_SLICE_WINDOW_ID))
            .default_pos([18.0, 70.0])
            .default_size([440.0, 520.0])
            .show(ctx, |ui| {
                ui.add(egui::Slider::new(&mut state.selected_slice, 0..=nz - 1).text("model level k"));
                let slice = &field.values[state.selected_slice * nx * ny..(state.selected_slice + 1) * nx * ny];
                if cache.field_hash != field.values_sha256 || cache.level != state.selected_slice {
                    cache.field_hash.clone_from(&field.values_sha256);
                    cache.level = state.selected_slice;
                    // A reflectivity slice is a radar product: fixed published
                    // classes, not this frame's own quantiles.
                    let reflectivity = is_reflectivity_units(&field.units);
                    cache.range = if reflectivity {
                        nws_dbz_range()
                    } else {
                        quantile_range(slice).unwrap_or((field.finite_min, field.finite_max))
                    };
                    let rgba = if reflectivity {
                        nws_dbz_rgba(slice)
                    } else {
                        scientific_rgba(slice, cache.range.0, cache.range.1)
                    };
                    let image = egui::ColorImage::from_rgba_unmultiplied([nx, ny], &rgba);
                    match cache.texture.as_mut() {
                        Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                        None => {
                            cache.texture = Some(ctx.load_texture(
                                "gm1-exact-slice",
                                image,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                    }
                    let level = cache.range.0 + 0.72 * (cache.range.1 - cache.range.0);
                    cache.segments = contours(slice, nx, ny, level).unwrap_or_default();
                }
                if let Some(texture) = cache.texture.as_ref() {
                    let side = ui.available_width().min(420.0);
                    let response = ui.add(
                        egui::Image::new(texture).fit_to_exact_size(egui::vec2(side, side)),
                    );
                    let rect = response.rect;
                    for segment in &cache.segments {
                        let map = |point: [f32; 2]| {
                            egui::pos2(
                                rect.left() + point[0] / (nx - 1) as f32 * rect.width(),
                                rect.top() + point[1] / (ny - 1) as f32 * rect.height(),
                            )
                        };
                        ui.painter().line_segment(
                            [map(segment.from), map(segment.to)],
                            egui::Stroke::new(0.8, egui::Color32::WHITE),
                        );
                    }
                }
                ui.monospace(format!("display q01–q99 {:.4e} … {:.4e}", cache.range.0, cache.range.1));
                ui.small("Pseudocolor and contour are display transforms of the exact selected horizontal level.");
            });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn capture_when_ready(
    mut commands: Commands,
    mut state: ResMut<CaptureState>,
    viewer: Res<ViewerState>,
    data: Res<ViewerData>,
    science: Res<ScienceViewState>,
    mut materials: ResMut<Assets<Gm1VolumeMaterial>>,
    mut terrain_materials: ResMut<Assets<AdmittedTerrainMaterial>>,
    mut terrain: Query<
        &mut Visibility,
        (With<AdmittedTerrainSurface>, Without<ExactVolumeSurface>),
    >,
    mut volume: Query<&mut Visibility, (With<ExactVolumeSurface>, Without<AdmittedTerrainSurface>)>,
    mut atmosphere: Query<
        &mut Visibility,
        (
            With<PhysicalAtmosphereSurface>,
            Without<ExactVolumeSurface>,
            Without<AdmittedTerrainSurface>,
        ),
    >,
    mut clear_color: ResMut<ClearColor>,
    mut exit: MessageWriter<AppExit>,
) {
    state.frames += 1;
    if !offline_capture_requested(state.path.as_deref()) {
        return;
    }
    if state.frames >= 1_200 {
        error!("render-ready screenshot timed out after 1200 frames; no output was accepted");
        state.path = None;
        exit.write(AppExit::error());
        return;
    }

    if !viewer.admitted_she_terrain {
        let source_gate = source_gate_for_dataset(&data.0);
        if !source_gate_allows_offline_capture(&source_gate) {
            error!(
                "offline presentation capture refused by exact source morphology: {source_gate}"
            );
            state.path = None;
            exit.write(AppExit::error());
            return;
        }
        let science_ready = science
            .rendered
            .as_ref()
            .is_some_and(|raster| raster.step == data.0.step && raster.layer == science.selected);
        if !state.requested && science_ready && state.frames >= 60 + state.attempts * 60 {
            let path = state.path.clone().expect("capture path checked above");
            commands
                .spawn(Screenshot::primary_window())
                .observe(validate_and_save_screenshot(
                    path,
                    true,
                    viewer.lighting_readback_diagnostic,
                ));
            state.requested = true;
        }
        return;
    }

    match state.phase {
        CapturePhase::InitialWarmup if state.frames >= 90 => {
            set_admitted_capture_pass(
                CaptureRenderPass::DenseCloudMask,
                &viewer,
                &mut materials,
                &mut terrain_materials,
                &mut terrain,
                &mut volume,
                &mut atmosphere,
                &mut clear_color,
            );
            state.phase = CapturePhase::DenseMaskWarmup;
            state.phase_started = state.frames;
            state.requested = false;
            info!(
                "paired capture entered deterministic q99 dense-cloud mask pass at frame {}",
                state.frames
            );
        }
        CapturePhase::DenseMaskWarmup
            if !state.requested && state.frames >= state.phase_started + 12 =>
        {
            commands
                .spawn(Screenshot::primary_window())
                .observe(record_dense_cloud_mask_readback());
            state.requested = true;
        }
        CapturePhase::PrepareCloudMask => {
            set_admitted_capture_pass(
                CaptureRenderPass::CloudMask,
                &viewer,
                &mut materials,
                &mut terrain_materials,
                &mut terrain,
                &mut volume,
                &mut atmosphere,
                &mut clear_color,
            );
            state.phase = CapturePhase::CloudMaskWarmup;
            state.phase_started = state.frames;
            state.requested = false;
            info!(
                "paired capture entered deterministic full-onset cloud mask pass at frame {}",
                state.frames
            );
        }
        CapturePhase::CloudMaskWarmup
            if !state.requested && state.frames >= state.phase_started + 12 =>
        {
            commands
                .spawn(Screenshot::primary_window())
                .observe(record_cloud_mask_readback());
            state.requested = true;
        }
        CapturePhase::PrepareTerrainMask => {
            set_admitted_capture_pass(
                CaptureRenderPass::TerrainMask,
                &viewer,
                &mut materials,
                &mut terrain_materials,
                &mut terrain,
                &mut volume,
                &mut atmosphere,
                &mut clear_color,
            );
            state.phase = CapturePhase::TerrainMaskWarmup;
            state.phase_started = state.frames;
            state.requested = false;
            info!(
                "paired capture entered deterministic terrain-object mask pass at frame {}",
                state.frames
            );
        }
        CapturePhase::TerrainMaskWarmup
            if !state.requested && state.frames >= state.phase_started + 12 =>
        {
            commands
                .spawn(Screenshot::primary_window())
                .observe(record_terrain_mask_readback());
            state.requested = true;
        }
        CapturePhase::RestorePresentation => {
            set_admitted_capture_pass(
                CaptureRenderPass::Presentation,
                &viewer,
                &mut materials,
                &mut terrain_materials,
                &mut terrain,
                &mut volume,
                &mut atmosphere,
                &mut clear_color,
            );
            state.phase = CapturePhase::PresentationWarmup;
            state.phase_started = state.frames;
            state.requested = false;
            info!(
                "paired capture restored exact terrain and combined presentation at frame {}",
                state.frames
            );
        }
        CapturePhase::PresentationWarmup
            if !state.requested && state.frames >= state.phase_started + 45 =>
        {
            let path = state.path.clone().expect("capture path checked above");
            commands.spawn(Screenshot::primary_window()).observe(
                validate_paired_and_save_screenshot(path, viewer.lighting_readback_diagnostic),
            );
            state.requested = true;
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn set_admitted_capture_pass(
    pass: CaptureRenderPass,
    viewer: &ViewerState,
    materials: &mut Assets<Gm1VolumeMaterial>,
    terrain_materials: &mut Assets<AdmittedTerrainMaterial>,
    terrain: &mut Query<
        &mut Visibility,
        (With<AdmittedTerrainSurface>, Without<ExactVolumeSurface>),
    >,
    volume: &mut Query<
        &mut Visibility,
        (With<ExactVolumeSurface>, Without<AdmittedTerrainSurface>),
    >,
    atmosphere: &mut Query<
        &mut Visibility,
        (
            With<PhysicalAtmosphereSurface>,
            Without<ExactVolumeSurface>,
            Without<AdmittedTerrainSurface>,
        ),
    >,
    clear_color: &mut ClearColor,
) {
    let mut material = viewer
        .material
        .as_ref()
        .and_then(|handle| materials.get_mut(handle))
        .expect("admitted capture has a resident volume material");
    // The otherwise-unused w lane is an object-mask display switch. Every
    // scientific carrier, transfer coefficient, ray, and camera matrix stays
    // pinned across the paired readbacks.
    material.uniform.sun_direction.w = match pass {
        CaptureRenderPass::DenseCloudMask => 2.0,
        CaptureRenderPass::CloudMask => 1.0,
        CaptureRenderPass::TerrainMask | CaptureRenderPass::Presentation => 0.0,
    };
    for (_, terrain_material) in terrain_materials.iter_mut() {
        terrain_material.uniform.control.w = f32::from(pass == CaptureRenderPass::TerrainMask);
    }
    for mut visibility in terrain.iter_mut() {
        *visibility = if matches!(
            pass,
            CaptureRenderPass::CloudMask | CaptureRenderPass::DenseCloudMask
        ) {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    }
    for mut visibility in volume.iter_mut() {
        *visibility = if pass == CaptureRenderPass::TerrainMask {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    }
    for mut visibility in atmosphere.iter_mut() {
        *visibility = if pass == CaptureRenderPass::Presentation {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    clear_color.0 = if pass == CaptureRenderPass::Presentation {
        Color::srgb(0.20, 0.43, 0.68)
    } else {
        Color::BLACK
    };
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ObjectMaskStatistics {
    count: usize,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CloudMaskReadback {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
    statistics: ObjectMaskStatistics,
}

impl CloudMaskReadback {
    fn from_rgb(bytes: &[u8], width: usize, height: usize) -> Self {
        if width == 0 || height == 0 || bytes.len() != width * height * 3 {
            return Self::default();
        }
        let mut pixels = vec![0_u8; width * height];
        let mut statistics = ObjectMaskStatistics {
            min_x: width,
            min_y: height,
            ..default()
        };
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 3;
                let present =
                    bytes[offset] >= 128 && bytes[offset + 1] >= 128 && bytes[offset + 2] >= 128;
                if present {
                    pixels[y * width + x] = 1;
                    statistics.count += 1;
                    statistics.min_x = statistics.min_x.min(x);
                    statistics.min_y = statistics.min_y.min(y);
                    statistics.max_x = statistics.max_x.max(x);
                    statistics.max_y = statistics.max_y.max(y);
                }
            }
        }
        if statistics.count == 0 {
            statistics = ObjectMaskStatistics::default();
        }
        Self {
            width,
            height,
            pixels,
            statistics,
        }
    }
}

fn cloud_statistics_from_mask(bytes: &[u8], mask: &CloudMaskReadback) -> CloudReadbackStatistics {
    if mask.width == 0
        || mask.height == 0
        || bytes.len() != mask.width * mask.height * 3
        || mask.pixels.len() != mask.width * mask.height
    {
        return CloudReadbackStatistics::default();
    }
    let mut luminances = Vec::with_capacity(mask.statistics.count);
    for (index, &present) in mask.pixels.iter().enumerate() {
        if present != 0 {
            let offset = index * 3;
            let luma = (54 * usize::from(bytes[offset])
                + 183 * usize::from(bytes[offset + 1])
                + 19 * usize::from(bytes[offset + 2]))
                / 256;
            luminances.push(luma as u8);
        }
    }
    cloud_statistics_from_luminances(luminances)
}

fn cloud_statistics_from_luminances(mut luminances: Vec<u8>) -> CloudReadbackStatistics {
    if luminances.is_empty() {
        return CloudReadbackStatistics::default();
    }
    luminances.sort_unstable();
    let p10 = luminances[(luminances.len() - 1) / 10];
    let p50 = luminances[(luminances.len() - 1) / 2];
    let p90 = luminances[(luminances.len() - 1) * 9 / 10];
    let mut histogram = [0usize; 16];
    for luminance in &luminances {
        histogram[(usize::from(*luminance) * 16 / 256).min(15)] += 1;
    }
    CloudReadbackStatistics {
        count: luminances.len(),
        min_x: 0,
        min_y: 0,
        max_x: 0,
        max_y: 0,
        body_min_x: 0,
        body_min_y: 0,
        body_max_x: 0,
        body_max_y: 0,
        luma_p10: p10,
        luma_p50: p50,
        luma_p90: p90,
        luma_p90_minus_p10: p90.saturating_sub(p10),
        // A mask-derived statistic carries no clear-sky reference to measure a
        // signed offset against; the paired path reads the within-subject
        // spread instead.
        luma_offset_p50: 0,
        luminance_histogram16: histogram,
    }
}

fn terrain_mask_is_ready(mask: &CloudMaskReadback) -> bool {
    let pixels = mask.width * mask.height;
    mask.width > 0
        && mask.height > 0
        && mask.statistics.count >= 10_000
        && mask.statistics.min_y >= mask.height * 72 / 100
        && mask.statistics.min_y <= mask.height * 84 / 100
        && mask.statistics.max_y >= mask.height * 90 / 100
        && mask.statistics.count >= pixels * 18 / 100
        && mask.statistics.count <= pixels * 28 / 100
}

fn mask_difference(
    complete: &CloudMaskReadback,
    subtract: &CloudMaskReadback,
) -> CloudMaskReadback {
    if complete.width != subtract.width
        || complete.height != subtract.height
        || complete.pixels.len() != subtract.pixels.len()
    {
        return CloudMaskReadback::default();
    }
    let mut rgb = vec![0_u8; complete.width * complete.height * 3];
    for (index, (&present, &removed)) in complete.pixels.iter().zip(&subtract.pixels).enumerate() {
        if present != 0 && removed == 0 {
            rgb[index * 3..index * 3 + 3].fill(255);
        }
    }
    CloudMaskReadback::from_rgb(&rgb, complete.width, complete.height)
}

fn record_dense_cloud_mask_readback() -> impl FnMut(On<ScreenshotCaptured>, ResMut<CaptureState>) {
    move |captured: On<ScreenshotCaptured>, mut state: ResMut<CaptureState>| {
        let image = match captured.image.clone().try_into_dynamic() {
            Ok(image) => image.to_rgb8(),
            Err(error) => {
                warn!("dense-cloud mask readback conversion refused: {error}");
                state.requested = false;
                state.attempts += 1;
                state.phase_started = state.frames;
                return;
            }
        };
        let mask = CloudMaskReadback::from_rgb(
            image.as_raw(),
            image.width() as usize,
            image.height() as usize,
        );
        if mask.statistics.count < 10_000 {
            warn!(
                "q99 dense-cloud mask refused: only {} optically-thick pixels; retrying without changing camera/field",
                mask.statistics.count,
            );
            state.requested = false;
            state.attempts += 1;
            state.phase_started = state.frames;
            return;
        }
        info!(
            "q99 dense-cloud mask readback: pixels={} bbox=({}, {})..({}, {}) dimensions={}x{} alpha_min=0.85",
            mask.statistics.count,
            mask.statistics.min_x,
            mask.statistics.min_y,
            mask.statistics.max_x,
            mask.statistics.max_y,
            mask.width,
            mask.height,
        );
        state.dense_mask = Some(mask);
        state.phase = CapturePhase::PrepareCloudMask;
        state.requested = false;
        state.phase_started = state.frames;
    }
}

fn record_cloud_mask_readback() -> impl FnMut(On<ScreenshotCaptured>, ResMut<CaptureState>) {
    move |captured: On<ScreenshotCaptured>, mut state: ResMut<CaptureState>| {
        let image = match captured.image.clone().try_into_dynamic() {
            Ok(image) => image.to_rgb8(),
            Err(error) => {
                warn!("cloud-object mask readback conversion refused: {error}");
                state.requested = false;
                state.attempts += 1;
                state.phase_started = state.frames;
                return;
            }
        };
        let mask = CloudMaskReadback::from_rgb(
            image.as_raw(),
            image.width() as usize,
            image.height() as usize,
        );
        if mask.statistics.count < 10_000 {
            warn!(
                "cloud-object mask refused: only {} alpha-qualified pixels; retrying mask without changing camera/field",
                mask.statistics.count,
            );
            state.requested = false;
            state.attempts += 1;
            state.phase_started = state.frames;
            return;
        }
        info!(
            "cloud-object mask readback: pixels={} bbox=({}, {})..({}, {}) dimensions={}x{}",
            mask.statistics.count,
            mask.statistics.min_x,
            mask.statistics.min_y,
            mask.statistics.max_x,
            mask.statistics.max_y,
            mask.width,
            mask.height,
        );
        state.cloud_mask = Some(mask);
        state.phase = CapturePhase::PrepareTerrainMask;
        state.requested = false;
        state.phase_started = state.frames;
    }
}

fn record_terrain_mask_readback() -> impl FnMut(On<ScreenshotCaptured>, ResMut<CaptureState>) {
    move |captured: On<ScreenshotCaptured>, mut state: ResMut<CaptureState>| {
        let image = match captured.image.clone().try_into_dynamic() {
            Ok(image) => image.to_rgb8(),
            Err(error) => {
                warn!("terrain-object mask readback conversion refused: {error}");
                state.requested = false;
                state.attempts += 1;
                state.phase_started = state.frames;
                return;
            }
        };
        let mask = CloudMaskReadback::from_rgb(
            image.as_raw(),
            image.width() as usize,
            image.height() as usize,
        );
        if mask.statistics.count < 10_000 {
            warn!(
                "terrain-object mask refused: only {} exact-geometry pixels; retrying terrain mask without changing camera/field",
                mask.statistics.count,
            );
            state.requested = false;
            state.attempts += 1;
            state.phase_started = state.frames;
            return;
        }
        info!(
            "terrain-object mask readback: pixels={} bbox=({}, {})..({}, {}) dimensions={}x{}",
            mask.statistics.count,
            mask.statistics.min_x,
            mask.statistics.min_y,
            mask.statistics.max_x,
            mask.statistics.max_y,
            mask.width,
            mask.height,
        );
        state.terrain_mask = Some(mask);
        state.phase = CapturePhase::RestorePresentation;
        state.requested = false;
        state.phase_started = state.frames;
    }
}

fn validate_paired_and_save_screenshot(
    path: PathBuf,
    save_refused_diagnostic: bool,
) -> impl FnMut(On<ScreenshotCaptured>, ResMut<CaptureState>, MessageWriter<AppExit>) {
    move |captured: On<ScreenshotCaptured>,
          mut state: ResMut<CaptureState>,
          mut exit: MessageWriter<AppExit>| {
        let image = match captured.image.clone().try_into_dynamic() {
            Ok(image) => image.to_rgb8(),
            Err(error) => {
                warn!("paired presentation readback conversion refused: {error}");
                state.requested = false;
                state.attempts += 1;
                state.phase_started = state.frames;
                return;
            }
        };
        let Some(mask) = state.cloud_mask.as_ref() else {
            error!("paired presentation capture refused without a cloud-object mask");
            state.path = None;
            exit.write(AppExit::error());
            return;
        };
        let Some(dense_mask) = state.dense_mask.as_ref() else {
            error!("paired presentation capture refused without a q99 dense-cloud mask");
            state.path = None;
            exit.write(AppExit::error());
            return;
        };
        let Some(terrain_mask) = state.terrain_mask.as_ref() else {
            error!("paired presentation capture refused without a terrain-object mask");
            state.path = None;
            exit.write(AppExit::error());
            return;
        };
        if mask.width != image.width() as usize
            || mask.height != image.height() as usize
            || dense_mask.width != mask.width
            || dense_mask.height != mask.height
            || terrain_mask.width != mask.width
            || terrain_mask.height != mask.height
        {
            error!(
                "paired capture dimensions changed: dense-mask={}x{} cloud-mask={}x{} terrain-mask={}x{} presentation={}x{}",
                dense_mask.width,
                dense_mask.height,
                mask.width,
                mask.height,
                terrain_mask.width,
                terrain_mask.height,
                image.width(),
                image.height(),
            );
            state.path = None;
            exit.write(AppExit::error());
            return;
        }
        let width = mask.width;
        let height = mask.height;
        let non_background = count_scene_non_background_rgb(image.as_raw(), width, height);
        let cloud = cloud_statistics_from_mask(image.as_raw(), mask);
        let dense_visual = cloud_statistics_from_mask(image.as_raw(), dense_mask);
        let haze_mask = mask_difference(mask, dense_mask);
        let haze_visual = cloud_statistics_from_mask(image.as_raw(), &haze_mask);
        let terrain_visual = cloud_statistics_from_mask(image.as_raw(), terrain_mask);
        let dominant = state.dominant_projection.unwrap_or_default();
        let dominant_margin = (1.0 - dominant.maximum_ndc.x)
            .min(dominant.minimum_ndc.x + 1.0)
            .min(1.0 - dominant.maximum_ndc.y)
            .min(dominant.minimum_ndc.y + 1.0);
        let dominant_ready = dominant.mass_fraction_in_frame >= 0.995
            && (0.45..=0.65).contains(&dominant.width_fraction)
            && dominant_margin >= 0.02;
        let terrain_ready = terrain_mask_is_ready(terrain_mask);
        let rim_body_contrast = cloud.luma_p90.saturating_sub(dense_visual.luma_p50);
        let dense_ready = dense_mask.statistics.count >= 10_000
            && dense_visual.luma_p10 >= 45
            && dense_visual.luma_p50 >= 64
            && rim_body_contrast >= 18;
        let haze_ready = haze_mask.statistics.count >= 1_000 && haze_visual.luma_p50 >= 60;
        let terrain_coverage_permille = terrain_mask.statistics.count * 1_000 / (width * height);
        let admitted = non_background >= 10_000
            && mask.statistics.count >= 10_000
            && dominant_ready
            && cloud.luma_p90_minus_p10 >= 24
            && dense_ready
            && haze_ready
            && terrain_ready
            && terrain_visual.luma_p90_minus_p10 >= 12;
        if !admitted {
            warn!(
                "paired screenshot refused: scene_pixels={non_background} cloud_mask_pixels={} cloud_mask_bbox=({}, {})..({}, {}) cloud_luma_p10/p50/p90={}/{}/{} spread={} dense_mask_pixels={} dense_luma_p10/p50/p90={}/{}/{} rim_body={} onset_haze_pixels={} haze_luma_p10/p50/p90={}/{}/{} dominant_mass_in_frame={:.9} dominant_ndc={:?}..{:?} dominant_width_fraction={:.6} dominant_margin={:.6} terrain_mask_pixels={} terrain_mask_bbox=({}, {})..({}, {}) terrain_coverage_permille={} terrain_luma_p10/p50/p90={}/{}/{} terrain_spread={}; retrying without changing camera/field",
                mask.statistics.count,
                mask.statistics.min_x,
                mask.statistics.min_y,
                mask.statistics.max_x,
                mask.statistics.max_y,
                cloud.luma_p10,
                cloud.luma_p50,
                cloud.luma_p90,
                cloud.luma_p90_minus_p10,
                dense_mask.statistics.count,
                dense_visual.luma_p10,
                dense_visual.luma_p50,
                dense_visual.luma_p90,
                rim_body_contrast,
                haze_mask.statistics.count,
                haze_visual.luma_p10,
                haze_visual.luma_p50,
                haze_visual.luma_p90,
                dominant.mass_fraction_in_frame,
                dominant.minimum_ndc,
                dominant.maximum_ndc,
                dominant.width_fraction,
                dominant_margin,
                terrain_mask.statistics.count,
                terrain_mask.statistics.min_x,
                terrain_mask.statistics.min_y,
                terrain_mask.statistics.max_x,
                terrain_mask.statistics.max_y,
                terrain_coverage_permille,
                terrain_visual.luma_p10,
                terrain_visual.luma_p50,
                terrain_visual.luma_p90,
                terrain_visual.luma_p90_minus_p10,
            );
            if save_refused_diagnostic {
                warn!(
                    "REFUSED_LIGHTING_READBACK_DIAGNOSTIC_ONLY: saving paired non-admitted combined frame to {}",
                    path.display(),
                );
                save_to_disk(&path)(captured);
                state.path = None;
                exit.write(AppExit::Success);
                return;
            }
            state.requested = false;
            state.attempts += 1;
            state.phase_started = state.frames;
            return;
        }
        info!(
            "paired screenshot PASSED: scene_pixels={non_background} cloud_mask_pixels={} cloud_bbox=({}, {})..({}, {}) cloud_luma_p10/p50/p90={}/{}/{} spread={} dense_mask_pixels={} dense_luma_p10/p50/p90={}/{}/{} rim_body={} onset_haze_pixels={} haze_luma_p10/p50/p90={}/{}/{} dominant_mass_in_frame={:.9} dominant_ndc={:?}..{:?} dominant_width_fraction={:.6} terrain_mask_pixels={} terrain_bbox=({}, {})..({}, {}) terrain_coverage_permille={} terrain_luma_p10/p50/p90={}/{}/{} terrain_spread={} camera_field_pair=PINNED",
            mask.statistics.count,
            mask.statistics.min_x,
            mask.statistics.min_y,
            mask.statistics.max_x,
            mask.statistics.max_y,
            cloud.luma_p10,
            cloud.luma_p50,
            cloud.luma_p90,
            cloud.luma_p90_minus_p10,
            dense_mask.statistics.count,
            dense_visual.luma_p10,
            dense_visual.luma_p50,
            dense_visual.luma_p90,
            rim_body_contrast,
            haze_mask.statistics.count,
            haze_visual.luma_p10,
            haze_visual.luma_p50,
            haze_visual.luma_p90,
            dominant.mass_fraction_in_frame,
            dominant.minimum_ndc,
            dominant.maximum_ndc,
            dominant.width_fraction,
            terrain_mask.statistics.count,
            terrain_mask.statistics.min_x,
            terrain_mask.statistics.min_y,
            terrain_mask.statistics.max_x,
            terrain_mask.statistics.max_y,
            terrain_coverage_permille,
            terrain_visual.luma_p10,
            terrain_visual.luma_p50,
            terrain_visual.luma_p90,
            terrain_visual.luma_p90_minus_p10,
        );
        save_to_disk(&path)(captured);
        state.path = None;
        exit.write(AppExit::Success);
    }
}

fn validate_and_save_screenshot(
    path: PathBuf,
    require_cloud: bool,
    save_refused_diagnostic: bool,
) -> impl FnMut(On<ScreenshotCaptured>, ResMut<CaptureState>, MessageWriter<AppExit>) {
    move |captured: On<ScreenshotCaptured>,
          mut state: ResMut<CaptureState>,
          mut exit: MessageWriter<AppExit>| {
        let image = match captured.image.clone().try_into_dynamic() {
            Ok(image) => image.to_rgb8(),
            Err(error) => {
                warn!("screenshot readback conversion refused: {error}");
                state.requested = false;
                state.attempts += 1;
                return;
            }
        };
        let non_background = count_scene_non_background_rgb(
            image.as_raw(),
            image.width() as usize,
            image.height() as usize,
        );
        if non_background < 10_000 {
            warn!(
                "screenshot readback has only {non_background} non-background scene pixels; retrying without writing"
            );
            if save_refused_diagnostic && state.attempts >= 5 {
                warn!(
                    "REFUSED_LIGHTING_READBACK_DIAGNOSTIC_ONLY: saving blank/terrain diagnostic to {}",
                    path.display(),
                );
                save_to_disk(&path)(captured);
                state.path = None;
                exit.write(AppExit::Success);
                return;
            }
            state.requested = false;
            state.attempts += 1;
            return;
        }
        let cloud = cloud_candidate_statistics_rgb(
            image.as_raw(),
            image.width() as usize,
            image.height() as usize,
        );
        let margin_x = (image.width() as usize / 50).max(1);
        let margin_y = (image.height() as usize / 50).max(1);
        let silhouette_fits = cloud.count >= 10_000
            && cloud.body_min_x >= margin_x
            && cloud.body_max_x + margin_x < image.width() as usize
            && cloud.body_min_y >= margin_y
            && cloud.body_max_y + margin_y < image.height() as usize;
        // Rendered contrast, counted either way it can appear. The spread
        // within the response is the contrast of a subject lit toward the
        // camera. A subject lit from behind is tonally flat within itself and
        // carries its contrast against the sky instead; measuring only the
        // first calls such a frame flat when it is not. A frame that really is
        // flat has neither.
        let rendered_contrast = cloud.luma_p90_minus_p10.max(cloud.luma_offset_p50);
        if require_cloud && (!silhouette_fits || rendered_contrast < 24) {
            let scene = scene_readback_summary(
                image.as_raw(),
                image.width() as usize,
                image.height() as usize,
            );
            warn!(
                "screenshot cloud silhouette/contrast refused: scene_pixels={non_background} \
                 scene_background_rgb={:?} scene_luma_p10/p50/p90={}/{}/{} scene_luma_max={} \
                 scene_chroma_max={} • candidate count={} bbox=({}, {})..({}, {}) \
                 luma-p10/p50/p90={}/{}/{} spread={} histogram16={:?}; retrying without writing",
                scene.background,
                scene.luma_p10,
                scene.luma_p50,
                scene.luma_p90,
                scene.maximum_luma,
                scene.maximum_chroma,
                cloud.count,
                cloud.min_x,
                cloud.min_y,
                cloud.max_x,
                cloud.max_y,
                cloud.luma_p10,
                cloud.luma_p50,
                cloud.luma_p90,
                cloud.luma_p90_minus_p10,
                cloud.luminance_histogram16,
            );
            if save_refused_diagnostic {
                warn!(
                    "REFUSED_LIGHTING_READBACK_DIAGNOSTIC_ONLY: saving non-admitted frame to {}",
                    path.display(),
                );
                save_to_disk(&path)(captured);
                state.path = None;
                exit.write(AppExit::Success);
                return;
            }
            state.requested = false;
            state.attempts += 1;
            return;
        }
        info!(
            "render-ready screenshot readback passed: scene_pixels={non_background} cloud_pixels={} cloud_bbox=({}, {})..({}, {}) cloud_luma_p10/p50/p90={}/{}/{} cloud_luma_p90_minus_p10={} cloud_histogram16={:?}",
            cloud.count,
            cloud.min_x,
            cloud.min_y,
            cloud.max_x,
            cloud.max_y,
            cloud.luma_p10,
            cloud.luma_p50,
            cloud.luma_p90,
            cloud.luma_p90_minus_p10,
            cloud.luminance_histogram16,
        );
        save_to_disk(&path)(captured);
        state.path = None;
        exit.write(AppExit::Success);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CloudReadbackStatistics {
    count: usize,
    min_x: usize,
    min_y: usize,
    max_x: usize,
    max_y: usize,
    /// Box holding all but the outermost [`SILHOUETTE_TAIL_PERMILLE`] of the
    /// response on each axis. The full box is set by the single faintest pixel
    /// anywhere in the frame; this one is set by the subject's body.
    body_min_x: usize,
    body_min_y: usize,
    body_max_x: usize,
    body_max_y: usize,
    luma_p10: u8,
    luma_p50: u8,
    luma_p90: u8,
    luma_p90_minus_p10: u8,
    /// Median distance of the response from the clear-sky background. This is
    /// the contrast the frame actually shows when the subject is lit from
    /// behind and so is tonally flat within itself.
    luma_offset_p50: u8,
    luminance_histogram16: [usize; 16],
}

/// Share of the condensate response, per thousand and per axis, that the
/// silhouette-fit test lets fall outside the body box.
///
/// The test exists to refuse a portrait that is really a crop: a cropped
/// subject puts a large part of its response hard against the frame edge. A
/// volume render of an atmosphere legitimately trails thin haze past the edge,
/// and requiring literally every responding pixel to clear the margin refuses a
/// correctly framed storm because its outermost veil grazes a corner.
const SILHOUETTE_TAIL_PERMILLE: usize = 15;

fn cloud_candidate_statistics_rgb(
    bytes: &[u8],
    width: usize,
    height: usize,
) -> CloudReadbackStatistics {
    if width == 0 || height == 0 || bytes.len() != width * height * 3 {
        return CloudReadbackStatistics::default();
    }
    let background = [bytes[0], bytes[1], bytes[2]];
    let background_luma = (54 * usize::from(background[0])
        + 183 * usize::from(background[1])
        + 19 * usize::from(background[2]))
        / 256;
    let mut luminances = Vec::new();
    let mut luma_offsets = Vec::new();
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut column_counts = vec![0_usize; width];
    let mut row_counts = vec![0_usize; height];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            let rgb = [bytes[offset], bytes[offset + 1], bytes[offset + 2]];
            let luma =
                (54 * usize::from(rgb[0]) + 183 * usize::from(rgb[1]) + 19 * usize::from(rgb[2]))
                    / 256;
            let maximum = *rgb.iter().max().expect("three channels");
            let minimum = *rgb.iter().min().expect("three channels");
            // This is a render-readiness gate, not a meteorological
            // classifier. It rejects terrain-only/blank captures by requiring
            // a sizeable bright, low-chroma condensate response distinct from
            // the clear-sky background. Exact field admission happened before
            // rendering and remains the scientific authority.
            //
            // Distinct in either direction. The earlier form required the
            // response to be *brighter* than the background, which holds only
            // while the subject is lit toward the camera. The camera is placed
            // with the sealed sun behind the dense cloud, so once the subject
            // is optically thick enough that little transmits - which any
            // resolved storm core is at kilometre spacing - it renders as a
            // dark, strongly contrasting mass against a bright sky. A one-sided
            // test counts none of those pixels and refuses a fully rendered
            // frame as blank. A large signed difference keeps every defect the
            // gate was written for: a blank or sky-only capture has no such
            // pixel at all, and a chromatic terrain-only capture is still
            // excluded by chroma.
            let luma_offset = luma.abs_diff(background_luma);
            let red_offset = rgb[0].abs_diff(background[0]);
            if luma_offset > 18 && red_offset > 20 && maximum - minimum < 75 {
                luminances.push(luma as u8);
                luma_offsets.push(luma_offset as u8);
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                column_counts[x] += 1;
                row_counts[y] += 1;
            }
        }
    }
    if luminances.is_empty() {
        return CloudReadbackStatistics::default();
    }
    luminances.sort_unstable();
    luma_offsets.sort_unstable();
    let offset_p50 = luma_offsets[(luma_offsets.len() - 1) / 2];
    let p10 = luminances[(luminances.len() - 1) / 10];
    let p50 = luminances[(luminances.len() - 1) / 2];
    let p90 = luminances[(luminances.len() - 1) * 9 / 10];
    let mut histogram = [0usize; 16];
    for luminance in &luminances {
        histogram[(usize::from(*luminance) * 16 / 256).min(15)] += 1;
    }
    // Trim the same small share of the response off each end of both axes, so
    // the body box is set by where the subject actually is rather than by its
    // faintest outlying pixel.
    let tail = (luminances.len() * SILHOUETTE_TAIL_PERMILLE / 1_000).max(1);
    let trim_low = |counts: &[usize]| {
        let mut seen = 0usize;
        for (index, count) in counts.iter().enumerate() {
            seen += count;
            if seen > tail {
                return index;
            }
        }
        counts.len().saturating_sub(1)
    };
    let trim_high = |counts: &[usize]| {
        let mut seen = 0usize;
        for (index, count) in counts.iter().enumerate().rev() {
            seen += count;
            if seen > tail {
                return index;
            }
        }
        0
    };
    let body_min_x = trim_low(&column_counts).max(min_x);
    let body_max_x = trim_high(&column_counts).min(max_x);
    let body_min_y = trim_low(&row_counts).max(min_y);
    let body_max_y = trim_high(&row_counts).min(max_y);
    CloudReadbackStatistics {
        count: luminances.len(),
        min_x,
        min_y,
        max_x,
        max_y,
        body_min_x,
        body_min_y,
        body_max_x: body_max_x.max(body_min_x),
        body_max_y: body_max_y.max(body_min_y),
        luma_p10: p10,
        luma_p50: p50,
        luma_p90: p90,
        luma_p90_minus_p10: p90.saturating_sub(p10),
        luma_offset_p50: offset_p50,
        luminance_histogram16: histogram,
    }
}

#[cfg(test)]
fn count_cloud_candidate_pixels_rgb(bytes: &[u8], width: usize, height: usize) -> usize {
    cloud_candidate_statistics_rgb(bytes, width, height).count
}

/// What a refused capture actually contained, so a refusal says what was seen
/// rather than only that the cloud criterion was unmet. Without this the only
/// evidence of a scene that rendered dark, flat or empty is the absence of
/// candidate pixels, which cannot distinguish "no cloud drawn" from "cloud
/// drawn too dark to pass".
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SceneReadbackSummary {
    background: [u8; 3],
    luma_p10: u8,
    luma_p50: u8,
    luma_p90: u8,
    maximum_luma: u8,
    maximum_chroma: u8,
}

fn scene_readback_summary(bytes: &[u8], width: usize, height: usize) -> SceneReadbackSummary {
    if width == 0 || height == 0 || bytes.len() != width * height * 3 {
        return SceneReadbackSummary::default();
    }
    let mut luminances = Vec::with_capacity(width * height);
    let mut maximum_chroma = 0_u8;
    for pixel in bytes.chunks_exact(3) {
        let luma = (54 * usize::from(pixel[0])
            + 183 * usize::from(pixel[1])
            + 19 * usize::from(pixel[2]))
            / 256;
        luminances.push(luma as u8);
        let maximum = *pixel.iter().max().expect("three channels");
        let minimum = *pixel.iter().min().expect("three channels");
        maximum_chroma = maximum_chroma.max(maximum - minimum);
    }
    luminances.sort_unstable();
    let at = |numerator: usize, denominator: usize| {
        luminances[(luminances.len() - 1) * numerator / denominator]
    };
    SceneReadbackSummary {
        background: [bytes[0], bytes[1], bytes[2]],
        luma_p10: at(1, 10),
        luma_p50: at(1, 2),
        luma_p90: at(9, 10),
        maximum_luma: *luminances.last().expect("at least one pixel"),
        maximum_chroma,
    }
}

fn count_scene_non_background_rgb(bytes: &[u8], width: usize, height: usize) -> usize {
    if width == 0 || height == 0 || bytes.len() != width * height * 3 {
        return 0;
    }
    let background = [bytes[0], bytes[1], bytes[2]];
    let scene_width = width * 3 / 4;
    let mut count = 0usize;
    for y in 0..height {
        for x in 0..scene_width {
            let offset = (y * width + x) * 3;
            let distance = (i16::from(bytes[offset]) - i16::from(background[0])).unsigned_abs()
                + (i16::from(bytes[offset + 1]) - i16::from(background[1])).unsigned_abs()
                + (i16::from(bytes[offset + 2]) - i16::from(background[2])).unsigned_abs();
            count += usize::from(distance > 30);
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::{
        CAMERA_PITCH, CHASE_EYE_ABOVE_TERRAIN_KM, CloudMaskReadback, DENSE_FRAME_MIN_CUTOFF,
        DriveTerrain, FRAME_PROFILE_CAPACITY, FrameProfiler, InteractiveDefaults,
        ProductionGpuField, ProductionInspectorState, SCIENCE_PANEL_ID, SCIENCE_PANEL_MARGIN,
        SCIENCE_PANEL_MAX_HEIGHT, SCIENCE_PANEL_WIDTH, SCIENCE_SLICE_WINDOW_ID,
        SCIENCE_TOP_VIEW_ID, SHELL_FIELDS_SECTION_ID, SHELL_FRAME_SECTION_ID,
        SHELL_PROVENANCE_SECTION_ID, SHELL_RENDER_SECTION_ID, SHELL_SCIENCE_SECTION_ID,
        CHASE_VERTICAL_FOV_DEGREES, PHYSICAL_CLOUD_ONSET, PRESENTATION_ASPECT,
        SHELL_VEHICLE_SECTION_ID, ScienceLayerId, ScientificShellUiCache,
        world_box_for_grid, world_min, world_size,
        admitted_chase_camera, admitted_chase_position, admitted_terrain_height_km_from_values,
        camera_terrain_height_km, child_enu_to_wgs84, cloud_candidate_statistics_rgb,
        cloud_statistics_from_mask, component_morphology, count_cloud_candidate_pixels_rgb,
        count_scene_non_background_rgb, dominant_connected_component, drive_ground_pose,
        exact_field_level, exact_field_statistics, frame_dense_condensate,
        install_scientific_shell_visuals_once, interactive_input_allowed, interior_mass_cell,
        live_controls_available, mask_difference, offline_capture_requested,
        project_dominant_component, scene_input_permissions, science_panel_height,
        science_texture_cache_requires_update, source_gate_allows_offline_capture,
        terrain_mask_is_ready, timeline_controls_available, validate_production_view_binding,
    };
    use crate::geofield::{
        GEO_FIELD_ABI, GeoFieldAdmission, GeoFieldDescriptor, GeoFieldGeometry, GeoGrid, GeoTime,
        GeoreferencedFrame,
    };
    use crate::hrrr_surround_terrain::{HRRR_SURROUND_ASSET_DIRECTORY, HrrrSurroundHeightSampler};
    use crate::input::{Cli, Gm1Dataset, QualityProfile, VolumeField, f32_sha256};
    use crate::production_playback::{
        IDEALIZED_VIEW_SOURCE_PROFILE, PRODUCTION_RENDER_ADAPTER_ID, ProductionRenderResidency,
        ProductionResidencyKind, ProductionResidentField, REAL_CHILD_VIEW_SOURCE_PROFILE,
    };
    use crate::she_admitted_terrain::{
        ADMITTED_TERRAIN_NX, ADMITTED_TERRAIN_NY, admitted_terrain_values,
    };
    use bevy::prelude::{Vec2, Vec3};
    use bevy_egui::egui;
    use std::{
        path::{Path, PathBuf},
        sync::Arc,
    };

    fn production_binding_fixture() -> (Cli, Gm1Dataset, ProductionRenderResidency) {
        let values = vec![1.25_f32];
        let values_sha256 = f32_sha256(&values);
        let descriptor = GeoFieldDescriptor {
            id: "cloud_condensate_kgkg".to_owned(),
            label: "Resolved cloud condensate (qc + qi)".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "exact qc plus qi cloud condensate".to_owned(),
            geometry: GeoFieldGeometry::VolumeZyx,
            shape: vec![1, 1, 1],
            values_sha256: values_sha256.clone(),
            admission: GeoFieldAdmission::Available,
            admission_reason: "test strict admission".to_owned(),
            provenance: "test exact words".to_owned(),
        };
        let fields = (0..8)
            .map(|index| {
                let mut field_descriptor = descriptor.clone();
                if index != 0 {
                    field_descriptor.id = format!("diagnostic_{index}");
                }
                ProductionResidentField {
                    descriptor: field_descriptor,
                    stagger: "mass".to_owned(),
                    residency: ProductionResidencyKind::CloudScalarVolume,
                    values: Arc::new(values.clone()),
                }
            })
            .collect::<Vec<_>>();
        let payload = "b".repeat(64);
        let manifest = "a".repeat(64);
        let production = ProductionRenderResidency {
            frame: GeoreferencedFrame {
                abi: GEO_FIELD_ABI,
                adapter_id: PRODUCTION_RENDER_ADAPTER_ID,
                source_identity: payload.clone(),
                grid: REAL_CHILD_VIEW_SOURCE_PROFILE.georeferenced_grid(),
                time: GeoTime {
                    step: 1,
                    model_time_s: 0.5,
                    dt_s: 0.5,
                },
                fields: fields
                    .iter()
                    .map(|field| field.descriptor.clone())
                    .collect(),
            },
            source_profile: REAL_CHILD_VIEW_SOURCE_PROFILE,
            manifest_sha256: manifest.clone(),
            payload_set_sha256: payload.clone(),
            fields,
        };
        let dataset = Gm1Dataset {
            config_sha256: String::new(),
            terrain_msl_m: None,
            face_height_m: None,
            tornado_surface_fields: Vec::new(),
            source_path: PathBuf::from("step-00000001"),
            artifact_kind: "production-v2 test".to_owned(),
            artifact_sha256: manifest,
            payload_sha256: payload,
            contract_sha256: "c".repeat(64),
            profile_id: format!("production-v2 test;{}", REAL_CHILD_VIEW_SOURCE_PROFILE.id),
            step: 1,
            model_time_s: 0.5,
            dt_s: 0.5,
            fields: vec![VolumeField {
                id: "cloud_condensate".to_owned(),
                label: "Resolved cloud condensate".to_owned(),
                units: "kg kg-1".to_owned(),
                meaning: "exact qc plus qi cloud condensate".to_owned(),
                source: "test exact words".to_owned(),
                shape: [1, 1, 1],
                values,
                values_sha256,
                finite_min: 1.25,
                finite_max: 1.25,
            }],
        };
        let cli = Cli::parse(["--production-pack".to_owned(), "step-00000001".to_owned()]).unwrap();
        (cli, dataset, production)
    }

    #[test]
    fn production_view_binding_refuses_crossed_or_derived_residency() {
        let (cli, dataset, production) = production_binding_fixture();
        validate_production_view_binding(&cli, &dataset, &production).unwrap();

        let mut changed_dataset = dataset.clone();
        changed_dataset.fields[0].values[0] = 2.0;
        assert!(
            validate_production_view_binding(&cli, &changed_dataset, &production)
                .unwrap_err()
                .contains("cloud carrier changed")
        );

        let mut derived = production.clone();
        derived.fields[7].descriptor.id = "vertical_vorticity_s".to_owned();
        assert!(
            validate_production_view_binding(&cli, &dataset, &derived)
                .unwrap_err()
                .contains("unadmitted derived carrier")
        );

        let mut crossed_profile = production.clone();
        crossed_profile.source_profile = IDEALIZED_VIEW_SOURCE_PROFILE;
        crossed_profile.frame.grid = IDEALIZED_VIEW_SOURCE_PROFILE.georeferenced_grid();
        assert!(
            validate_production_view_binding(&cli, &dataset, &crossed_profile)
                .unwrap_err()
                .contains("envelope changed")
        );

        let watch_cli = Cli::parse(["--watch-dir".to_owned(), "pairs".to_owned()]).unwrap();
        assert!(
            validate_production_view_binding(&watch_cli, &dataset, &production)
                .unwrap_err()
                .contains("exclusive --production-pack or --watch-production-series")
        );
        let series_cli =
            Cli::parse(["--watch-production-series".to_owned(), "series".to_owned()]).unwrap();
        validate_production_view_binding(&series_cli, &dataset, &production).unwrap();
    }

    #[test]
    fn production_inspector_selects_only_exact_resident_words() {
        let values = Arc::new(vec![0.0_f32, -0.0, 1.0, -2.0]);
        let statistics = exact_field_statistics(&values).unwrap();
        assert_eq!(statistics.word_count, 4);
        assert_eq!(statistics.nonzero_count, 2);
        assert_eq!(statistics.minimum.to_bits(), (-2.0_f32).to_bits());
        assert_eq!(statistics.maximum.to_bits(), 1.0_f32.to_bits());
        assert!(exact_field_statistics(&[f32::NAN]).is_err());

        let volume = ProductionGpuField {
            descriptor: GeoFieldDescriptor {
                id: "w_staggered_ms".to_owned(),
                label: "Native vertical velocity".to_owned(),
                units: "m s-1".to_owned(),
                meaning: "exact z-face words".to_owned(),
                geometry: GeoFieldGeometry::VolumeZyx,
                shape: vec![2, 1, 2],
                values_sha256: f32_sha256(&values),
                admission: GeoFieldAdmission::Available,
                admission_reason: "exact test words".to_owned(),
                provenance: "test".to_owned(),
            },
            stagger: "z-face C-grid".to_owned(),
            residency: ProductionResidencyKind::ZFaceWindVolume,
            nbytes: values.len() * 4,
            values: values.clone(),
            statistics,
            _buffer: Default::default(),
        };
        assert_eq!(exact_field_level(&volume, 1).unwrap(), &[1.0, -2.0]);
        assert!(exact_field_level(&volume, 2).is_err());

        let surface_values = Arc::new(vec![4.0_f32, 5.0]);
        let surface = ProductionGpuField {
            descriptor: GeoFieldDescriptor {
                id: "terrain_msl_m".to_owned(),
                label: "Exact terrain".to_owned(),
                units: "m".to_owned(),
                meaning: "source surface".to_owned(),
                geometry: GeoFieldGeometry::RasterYx,
                shape: vec![1, 2],
                values_sha256: f32_sha256(&surface_values),
                admission: GeoFieldAdmission::Available,
                admission_reason: "exact test words".to_owned(),
                provenance: "test".to_owned(),
            },
            stagger: "horizontal mass".to_owned(),
            residency: ProductionResidencyKind::TerrainSurface,
            nbytes: surface_values.len() * 4,
            values: surface_values.clone(),
            statistics: exact_field_statistics(&surface_values).unwrap(),
            _buffer: Default::default(),
        };
        let fields = vec![volume, surface];
        let opacity_field_index = 0usize;
        let mut inspector = ProductionInspectorState {
            selected_field: 0,
            selected_level: 1,
        };
        inspector.select_field(1, &fields);
        assert_eq!(inspector.selected_field, 1);
        assert_eq!(inspector.selected_level, 0);
        assert_eq!(exact_field_level(&fields[1], 0).unwrap(), &[4.0, 5.0]);
        assert_eq!(opacity_field_index, 0);
        assert_eq!(fields[0].values.as_slice(), values.as_slice());
    }

    #[test]
    fn interactive_defaults_are_fly_first_and_quiescent() {
        let defaults = InteractiveDefaults::default();
        assert!(defaults.fly_mode);
        assert!(!defaults.timeline_playing);
        assert!(!defaults.timeline_looping);
        assert!(!defaults.follow_live);
    }

    #[test]
    fn single_frame_has_no_timeline_or_live_controls() {
        assert!(!timeline_controls_available(0));
        assert!(!timeline_controls_available(1));
        assert!(!live_controls_available(1, true));
        assert!(timeline_controls_available(2));
        assert!(!live_controls_available(2, false));
        assert!(live_controls_available(2, true));
    }

    #[test]
    fn egui_or_offline_capture_owns_interactive_input() {
        assert!(interactive_input_allowed(false, false));
        assert!(!interactive_input_allowed(false, true));
        assert!(!interactive_input_allowed(true, false));
        assert!(!interactive_input_allowed(true, true));
        assert!(!offline_capture_requested(None));
        assert!(offline_capture_requested(Some(Path::new("sealed.png"))));
    }

    #[test]
    fn any_egui_interaction_quiesces_the_whole_camera_chord() {
        assert_eq!(
            scene_input_permissions(false, false, false),
            super::SceneInputPermissions {
                pointer: true,
                keyboard: true,
            }
        );
        for permissions in [
            scene_input_permissions(false, true, false),
            scene_input_permissions(false, false, true),
            scene_input_permissions(false, true, true),
            scene_input_permissions(true, false, false),
        ] {
            assert!(!permissions.pointer);
            assert!(!permissions.keyboard);
        }
    }

    #[test]
    fn scientific_shell_visuals_are_installed_once() {
        let ctx = egui::Context::default();
        let mut cache = ScientificShellUiCache::default();
        assert!(install_scientific_shell_visuals_once(&ctx, &mut cache));
        let first_style = ctx.style_of(ctx.theme());
        assert_eq!(
            first_style.visuals.panel_fill,
            egui::Color32::from_rgb(247, 246, 242)
        );
        assert!(!install_scientific_shell_visuals_once(&ctx, &mut cache));
        let second_style = ctx.style_of(ctx.theme());
        assert!(Arc::ptr_eq(&first_style, &second_style));
    }

    #[test]
    fn science_texture_identity_check_does_not_rebuild_a_stable_key() {
        let key = (7, ScienceLayerId::SimSatTopView, "a".repeat(64));
        assert!(!science_texture_cache_requires_update(
            Some(&key),
            7,
            ScienceLayerId::SimSatTopView,
            &"a".repeat(64),
        ));
        assert!(science_texture_cache_requires_update(
            Some(&key),
            8,
            ScienceLayerId::SimSatTopView,
            &"a".repeat(64),
        ));
        assert!(science_texture_cache_requires_update(
            Some(&key),
            7,
            ScienceLayerId::ReflectivityDbz,
            &"a".repeat(64),
        ));
        assert!(science_texture_cache_requires_update(
            Some(&key),
            7,
            ScienceLayerId::SimSatTopView,
            &"b".repeat(64),
        ));
    }

    #[test]
    fn shell_and_auxiliary_window_ids_are_unique_and_stable() {
        let ids = [
            SCIENCE_PANEL_ID,
            SCIENCE_TOP_VIEW_ID,
            SCIENCE_SLICE_WINDOW_ID,
            SHELL_FRAME_SECTION_ID,
            SHELL_RENDER_SECTION_ID,
            SHELL_FIELDS_SECTION_ID,
            SHELL_SCIENCE_SECTION_ID,
            SHELL_VEHICLE_SECTION_ID,
            SHELL_PROVENANCE_SECTION_ID,
        ];
        for (index, id) in ids.iter().enumerate() {
            assert!(id.starts_with("gm1-"));
            assert!(!ids[index + 1..].contains(id));
        }
    }

    #[test]
    fn science_panel_geometry_is_stable_and_viewport_bounded() {
        assert_eq!(SCIENCE_PANEL_ID, "gm1-science-panel");
        assert_eq!(SCIENCE_PANEL_WIDTH.to_bits(), 352.0_f32.to_bits());
        assert_eq!(SCIENCE_PANEL_MARGIN.to_bits(), 8.0_f32.to_bits());
        assert_eq!(SCIENCE_PANEL_MAX_HEIGHT.to_bits(), 660.0_f32.to_bits());
        assert_eq!(science_panel_height(1_080.0), 660.0);
        assert_eq!(science_panel_height(200.0), 184.0);
        assert_eq!(science_panel_height(100.0), 180.0);
        assert_eq!(science_panel_height(1_080.0), science_panel_height(1_080.0));
    }

    #[test]
    fn live_frame_profiler_has_one_fixed_allocation_and_bounded_ring() {
        let mut profiler = FrameProfiler::new(None, 60, None);
        let storage = profiler.samples_ms.as_ptr();
        for _ in 0..(FRAME_PROFILE_CAPACITY + 2_000) {
            profiler.record(1.0 / 120.0, QualityProfile::Live);
        }
        assert_eq!(profiler.samples_ms.as_ptr(), storage);
        assert_eq!(profiler.sample_count, FRAME_PROFILE_CAPACITY);
        assert!(profiler.next_sample < FRAME_PROFILE_CAPACITY);
        assert!(profiler.fps().is_finite());
        assert!(profiler.fps() > 100.0);
    }

    #[test]
    fn profile_refuses_a_quality_switch_after_warmup() {
        let mut profiler = FrameProfiler::new(None, 60, None);
        for _ in 0..601 {
            profiler.record(1.0 / 120.0, QualityProfile::Live);
        }
        assert_eq!(profiler.sampled_quality, Some(QualityProfile::Live));
        assert!(!profiler.mixed_quality);
        profiler.record(1.0 / 120.0, QualityProfile::Cinema);
        assert!(profiler.mixed_quality);
    }

    #[test]
    fn spherical_enu_overlay_preserves_sealed_child_center() {
        let (latitude, longitude) = child_enu_to_wgs84(0.0, 0.0, 0.0);
        assert!((latitude - 41.394_640_411_35).abs() < 1.0e-12);
        assert!((longitude - -94.282_682_936_87).abs() < 1.0e-12);
        let (north_latitude, east_longitude) = {
            let east = child_enu_to_wgs84(1.0, 0.0, 0.0).1;
            let north = child_enu_to_wgs84(0.0, 1.0, 0.0).0;
            (north, east)
        };
        assert!(north_latitude > latitude);
        assert!(east_longitude > longitude);
    }

    #[test]
    fn chase_vehicle_four_contact_pose_uses_sealed_child_and_hrrr_surfaces() {
        let assets = super::runtime_assets_directory().join(HRRR_SURROUND_ASSET_DIRECTORY);
        let terrain = DriveTerrain {
            child_msl_m: admitted_terrain_values().unwrap(),
            surround: HrrrSurroundHeightSampler::load(&assets).unwrap(),
        };
        let (child_position, child_rotation) =
            drive_ground_pose(&terrain, Vec2::ZERO, 0.37).unwrap();
        assert!(child_position.is_finite());
        assert!(child_rotation.is_finite());
        assert!(child_position.y > 0.30 && child_position.y < 0.50);
        assert!((child_rotation * Vec3::Y).dot(Vec3::Y) > 0.99);

        let (hrrr_position, hrrr_rotation) =
            drive_ground_pose(&terrain, Vec2::new(12.0, 2.0), -0.42).unwrap();
        assert!(hrrr_position.is_finite());
        assert!(hrrr_rotation.is_finite());
        assert!(hrrr_position.y > -1.2 && hrrr_position.y < 1.0);
    }

    #[test]
    fn exact_mass_point_horizontal_mapping_reads_source_cells_without_half_cell_blend() {
        let terrain = (0..ADMITTED_TERRAIN_NX * ADMITTED_TERRAIN_NY)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        for (i, j) in [(0, 0), (1, 2), (149, 149), (298, 297), (299, 299)] {
            let world_x =
                world_min().x + (i as f32 + 0.5) * world_size().x / ADMITTED_TERRAIN_NX as f32;
            let world_z =
                world_min().z + (j as f32 + 0.5) * world_size().z / ADMITTED_TERRAIN_NY as f32;
            let sampled =
                admitted_terrain_height_km_from_values(&terrain, world_x, world_z) * 1_000.0;
            let expected = terrain[j * ADMITTED_TERRAIN_NX + i];
            assert!(
                (sampled - expected).abs() <= 1.0e-2,
                "i={i} j={j} sampled={sampled} expected={expected}",
            );
        }
    }

    #[test]
    fn dense_frame_uses_only_physical_dense_condensate() {
        let shape = [4, 8, 10];
        let mut values = vec![DENSE_FRAME_MIN_CUTOFF * 0.5; shape.iter().product()];
        for i in 2..8 {
            values[2 * shape[1] * shape[2] + 4 * shape[2] + i] = 4.0e-3;
        }
        let field = VolumeField {
            id: "analytic-camera-operator".to_owned(),
            label: "analytic camera operator".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "controlled analytic camera fixture".to_owned(),
            source: "controlled analytic operator fixture".to_owned(),
            shape,
            values_sha256: f32_sha256(&values),
            finite_min: 0.0,
            finite_max: 4.0e-3,
            values,
        };
        let frame = frame_dense_condensate(&field);
        // The dense fixture spans i=2..7; exact three-cell open-band
        // exclusion leaves only interior i=3..6 for display-camera statistics.
        assert_eq!(frame.dense_count, 4);
        assert!(frame.focus.is_finite());
        assert!(frame.yaw.is_finite());
        assert_eq!(frame.pitch.to_bits(), CAMERA_PITCH.to_bits());
        // The standoff is measured against the subject, not in kilometres tuned
        // for one domain, so the assertion is scale-free: whatever the spacing,
        // the camera ends up outside the dense component.
        let core_span = (frame.dense_max - frame.dense_min).max_element();
        assert!(
            frame.radius >= core_span,
            "standoff {} must clear the dense core ({core_span})",
            frame.radius
        );
        assert!(frame.radius.is_finite());
        assert!(frame.dense_max.x > frame.dense_min.x);
        assert_eq!(frame.dominant_indices.len(), 4);
        assert_eq!(frame.dominant_mass_fraction.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn a_field_with_no_condensate_falls_back_to_the_fixed_camera() {
        // The solved view needs a subject. With nothing above the cutoff there
        // is no storm to stand in front of, so the frame takes the fixed
        // overview pitch instead of solving a ground observer against an empty
        // domain -- which would aim at the floor and divide by a zero extent.
        let shape = [4, 8, 10];
        let values = vec![0.0_f32; shape.iter().product()];
        let field = VolumeField {
            id: "analytic-empty-operator".to_owned(),
            label: "analytic empty operator".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "controlled analytic empty fixture".to_owned(),
            source: "controlled analytic operator fixture".to_owned(),
            shape,
            values_sha256: f32_sha256(&values),
            finite_min: 0.0,
            finite_max: 0.0,
            values,
        };
        let frame = frame_dense_condensate(&field);
        assert_eq!(frame.dense_count, 0);
        assert_eq!(frame.pitch.to_bits(), CAMERA_PITCH.to_bits());
        assert!(frame.pitch < 0.0);
        assert!(frame.focus.is_finite());
        assert!(frame.radius.is_finite());
    }

    #[test]
    fn scene_box_is_derived_from_the_admitted_grid_not_a_fixed_domain() {
        let mesoscale = world_box_for_grid(
            &GeoGrid {
                center_latitude_deg: 0.0,
                center_longitude_deg: 0.0,
                nx: 220,
                ny: 220,
                nz: 40,
                dx_m: 1_000.0,
                dy_m: 1_000.0,
                model_top_m: 20_000.0,
                horizontal_crs: "idealized Cartesian x/y; no Earth geolocation",
                vertical_coordinate: "uniform dz from admitted profile",
            },
            0.0,
        );
        assert_eq!(mesoscale.size, Vec3::new(220.0, 20.0, 220.0));
        assert_eq!(mesoscale.min, Vec3::new(-110.0, 0.0, -110.0));

        // A different spacing on the same shape is a different physical
        // domain, and the scene box must say so.
        let fine = world_box_for_grid(
            &GeoGrid {
                nx: 512,
                ny: 512,
                nz: 120,
                dx_m: 125.0,
                dy_m: 125.0,
                model_top_m: 15_000.0,
                ..REAL_CHILD_VIEW_SOURCE_PROFILE.georeferenced_grid()
            },
            0.0,
        );
        assert_eq!(fine.size, Vec3::new(64.0, 15.0, 64.0));
        assert_eq!(fine.min, Vec3::new(-32.0, 0.0, -32.0));

        // A domain whose lowest surface is above sea level starts there, and
        // its vertical span is shortened by exactly that floor.
        let raised = world_box_for_grid(&REAL_CHILD_VIEW_SOURCE_PROFILE.georeferenced_grid(), 0.5);
        assert_eq!(raised.min.y.to_bits(), 0.5_f32.to_bits());
        assert_eq!(raised.size, Vec3::new(15.0, 14.5, 15.0));
    }

    #[test]
    fn the_standoff_clears_a_cloud_far_wider_than_its_dense_core() {
        // A narrow dense core under a wide thin veil: the mesoscale storm case.
        // A standoff measured from the core alone leaves the camera inside the
        // veil, and every ray then starts in cloud.
        let shape = [8, 60, 60];
        let mut values = vec![0.0_f32; shape.iter().product::<usize>()];
        let index = |k: usize, j: usize, i: usize| (k * shape[1] + j) * shape[2] + i;
        for j in 5..56 {
            for i in 5..56 {
                values[index(4, j, i)] = PHYSICAL_CLOUD_ONSET * 2.0;
            }
        }
        for k in 3..6 {
            for j in 29..32 {
                for i in 29..32 {
                    values[index(k, j, i)] = 4.0e-3;
                }
            }
        }
        let field = VolumeField {
            id: "analytic-camera-operator".to_owned(),
            label: "analytic camera operator".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "controlled analytic camera fixture".to_owned(),
            source: "controlled analytic operator fixture".to_owned(),
            shape,
            values_sha256: f32_sha256(&values),
            finite_min: 0.0,
            finite_max: 4.0e-3,
            values,
        };
        let frame = frame_dense_condensate(&field);
        let core_span = (frame.dense_max - frame.dense_min).max_element();
        let visible_span = (frame.visible_max - frame.visible_min).max_element();
        assert!(
            visible_span > core_span,
            "fixture must have a veil wider than its core: {visible_span} vs {core_span}"
        );
        // The core alone would place the camera at most 1.8 of its own span
        // away. The veil around it pushes the standoff further out, which is
        // the whole point: at the core-only distance the camera sits inside it,
        // and the veil does not fit the frame.
        assert!(
            frame.radius > core_span * 1.8,
            "standoff {} must be widened past the core-only bound ({})",
            frame.radius,
            core_span * 1.8
        );
        // And the veil itself fits: at this standoff the camera's own field of
        // view spans more than the visible cloud.
        let tan_half_horizontal =
            (CHASE_VERTICAL_FOV_DEGREES.to_radians() * 0.5).tan() * PRESENTATION_ASPECT;
        assert!(
            2.0 * frame.radius * tan_half_horizontal > visible_span,
            "at standoff {} the frame spans {} but the visible cloud is {visible_span}",
            frame.radius,
            2.0 * frame.radius * tan_half_horizontal
        );
    }

    #[test]
    fn dominant_component_is_selected_by_exact_condensate_mass() {
        let shape = [2, 10, 12];
        let mut values = vec![0.0_f32; shape.iter().product()];
        let index = |k: usize, j: usize, i: usize| (k * shape[1] + j) * shape[2] + i;
        let expected = [index(0, 4, 4), index(0, 4, 5), index(1, 4, 4)];
        for &cell in &expected {
            values[cell] = 3.0e-3;
        }
        values[index(0, 6, 8)] = 4.0e-3;
        let field = VolumeField {
            id: "analytic-disconnected-components".to_owned(),
            label: "analytic disconnected components".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "controlled connected-component fixture".to_owned(),
            source: "controlled analytic operator fixture".to_owned(),
            shape,
            values_sha256: f32_sha256(&values),
            finite_min: 0.0,
            finite_max: 4.0e-3,
            values,
        };
        let (mut dominant, dominant_mass, connected_mass) =
            dominant_connected_component(&field, 2.0e-3);
        dominant.sort_unstable();
        assert_eq!(dominant, expected);
        assert!((dominant_mass - 9.0e-3).abs() < 1.0e-9);
        assert!((connected_mass - 13.0e-3).abs() < 1.0e-9);
        assert!((dominant_mass / connected_mass - 9.0 / 13.0).abs() < 1.0e-8);
        let frame = frame_dense_condensate(&field);
        let morphology = component_morphology(&field, &frame).expect("dominant morphology");
        assert_eq!(morphology.grid_min, bevy::prelude::UVec3::new(4, 4, 0));
        assert_eq!(morphology.grid_max, bevy::prelude::UVec3::new(5, 4, 1));
        assert_eq!(morphology.bbox_cell_count, 4);
        assert_eq!(morphology.voxel_compactness.to_bits(), 0.75_f64.to_bits());
        assert_eq!(morphology.lateral_boundary_distance_cells, 4);
        assert!(morphology.vertical_extent_km > morphology.maximum_horizontal_extent_km);
    }

    #[test]
    fn screenshot_gate_excludes_ui_quarter_and_requires_scene_pixels() {
        let width = 8;
        let height = 4;
        let mut pixels = vec![120_u8; width * height * 3];
        for x in 6..8 {
            let offset = x * 3;
            pixels[offset..offset + 3].copy_from_slice(&[255, 0, 0]);
        }
        assert_eq!(count_scene_non_background_rgb(&pixels, width, height), 0);
        for x in 1..5 {
            let offset = (2 * width + x) * 3;
            pixels[offset..offset + 3].copy_from_slice(&[180, 180, 180]);
        }
        assert_eq!(count_scene_non_background_rgb(&pixels, width, height), 4);
    }

    #[test]
    fn screenshot_cloud_gate_rejects_sky_and_accepts_bright_condensate() {
        let width = 8;
        let height = 4;
        let mut pixels = vec![0_u8; width * height * 3];
        for pixel in pixels.chunks_exact_mut(3) {
            pixel.copy_from_slice(&[89, 126, 163]);
        }
        assert_eq!(count_cloud_candidate_pixels_rgb(&pixels, width, height), 0);
        for x in 2..6 {
            let offset = (2 * width + x) * 3;
            let value = if x < 4 { 160 } else { 198 };
            pixels[offset..offset + 3].copy_from_slice(&[value, value, value]);
        }
        assert_eq!(count_cloud_candidate_pixels_rgb(&pixels, width, height), 4);
        let statistics = cloud_candidate_statistics_rgb(&pixels, width, height);
        assert_eq!((statistics.min_x, statistics.max_x), (2, 5));
        assert_eq!((statistics.min_y, statistics.max_y), (2, 2));
        assert_eq!(statistics.luma_p90_minus_p10, 38);
    }

    #[test]
    fn screenshot_cloud_gate_accepts_a_back_lit_subject_darker_than_the_sky() {
        // The camera is placed with the sealed sun behind the dense cloud, so a
        // subject thick enough to transmit little renders darker than the sky.
        // A one-sided brightness test counts none of it and calls a fully
        // rendered frame blank.
        let width = 8;
        let height = 4;
        let mut pixels = vec![0_u8; width * height * 3];
        for pixel in pixels.chunks_exact_mut(3) {
            pixel.copy_from_slice(&[120, 150, 177]);
        }
        assert_eq!(count_cloud_candidate_pixels_rgb(&pixels, width, height), 0);
        for x in 2..6 {
            let offset = (2 * width + x) * 3;
            let value = if x < 4 { 62_u8 } else { 90 };
            pixels[offset..offset + 3].copy_from_slice(&[value, value + 23, value + 50]);
        }
        assert_eq!(count_cloud_candidate_pixels_rgb(&pixels, width, height), 4);
        let statistics = cloud_candidate_statistics_rgb(&pixels, width, height);
        assert_eq!((statistics.min_x, statistics.max_x), (2, 5));
        assert!(statistics.luma_p90_minus_p10 >= 24);
    }

    #[test]
    fn the_dense_response_knee_follows_the_cloud_not_the_clear_air() {
        // A domain that is almost all clear air: a whole-domain quantile lands
        // in the clear air and pins the knee to its floor, so the cloud that is
        // there is displayed as onset haze.
        let shape = [4, 24, 24];
        let mut values = vec![0.0_f32; shape.iter().product::<usize>()];
        let index = |k: usize, j: usize, i: usize| (k * shape[1] + j) * shape[2] + i;
        for k in 1..3 {
            for j in 10..14 {
                for i in 10..14 {
                    values[index(k, j, i)] = 6.0e-4;
                }
            }
        }
        let field = VolumeField {
            id: "analytic-camera-operator".to_owned(),
            label: "analytic camera operator".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "controlled analytic camera fixture".to_owned(),
            source: "controlled analytic operator fixture".to_owned(),
            shape,
            values_sha256: f32_sha256(&values),
            finite_min: 0.0,
            finite_max: 6.0e-4,
            values,
        };
        let frame = frame_dense_condensate(&field);
        assert_eq!(frame.cutoff.to_bits(), DENSE_FRAME_MIN_CUTOFF.to_bits());
        // The knee sits on the condensate the frame actually carries, so the
        // cloud reads as dense rather than as haze.
        assert!(
            frame.dense_response_knee <= 6.0e-4,
            "knee {} must follow the cloud, not the clear air",
            frame.dense_response_knee
        );
        assert!(frame.dense_response_knee >= PHYSICAL_CLOUD_ONSET);
    }

    #[test]
    fn offline_capture_requires_candidate_source_morphology() {
        assert!(source_gate_allows_offline_capture(
            "CANDIDATE_SOURCE_MORPHOLOGY_ONLY ratio=0.600 compactness=0.500 boundary=30cells"
        ));
        for refused in [
            "REFUSED_NO_EXACT_CLOUD_CONDENSATE",
            "REFUSED_NO_DOMINANT_Q99_COMPONENT",
            "REFUSED_LOW_COMPACTNESS ratio=0.465 compactness=0.071 boundary=13cells mass=1.000",
            "REFUSED_BOUNDARY_ADJACENT_FLAT ratio=0.200 compactness=0.400 boundary=2cells mass=0.800",
        ] {
            assert!(!source_gate_allows_offline_capture(refused));
        }
    }

    #[test]
    fn paired_cloud_mask_owns_identity_and_combined_frame_owns_contrast() {
        let width = 20;
        let height = 10;
        let mut mask_rgb = vec![0_u8; width * height * 3];
        for y in 2..8 {
            for x in 3..15 {
                let offset = (y * width + x) * 3;
                mask_rgb[offset..offset + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        // A colored terrain-like pixel cannot enter the binary cloud mask.
        mask_rgb[(9 * width + 18) * 3..(9 * width + 18) * 3 + 3].copy_from_slice(&[110, 70, 30]);
        let mask = CloudMaskReadback::from_rgb(&mask_rgb, width, height);
        assert_eq!(mask.statistics.count, 72);
        assert_eq!(
            (
                mask.statistics.min_x,
                mask.statistics.min_y,
                mask.statistics.max_x,
                mask.statistics.max_y,
            ),
            (3, 2, 14, 7)
        );

        let mut combined = vec![0_u8; width * height * 3];
        for pixel in combined.chunks_exact_mut(3) {
            pixel.copy_from_slice(&[80, 120, 170]);
        }
        for (index, &present) in mask.pixels.iter().enumerate() {
            if present != 0 {
                let value = if index % 2 == 0 { 120 } else { 205 };
                combined[index * 3..index * 3 + 3].copy_from_slice(&[value, value, value]);
            }
        }
        let cloud = cloud_statistics_from_mask(&combined, &mask);
        assert_eq!(cloud.count, 72);
        assert_eq!(cloud.luma_p90_minus_p10, 85);

        let mut dense_rgb = vec![0_u8; width * height * 3];
        for y in 3..7 {
            for x in 5..12 {
                dense_rgb[(y * width + x) * 3..(y * width + x) * 3 + 3].fill(255);
            }
        }
        let dense = CloudMaskReadback::from_rgb(&dense_rgb, width, height);
        let haze = mask_difference(&mask, &dense);
        assert_eq!(dense.statistics.count, 28);
        assert_eq!(haze.statistics.count, 44);
        assert!(
            haze.pixels
                .iter()
                .zip(&dense.pixels)
                .all(|(&haze, &dense)| haze == 0 || dense == 0)
        );
    }

    #[test]
    fn terrain_object_mask_is_separate_and_lower_frame_bounded() {
        let width = 400;
        let height = 200;
        let mut mask_rgb = vec![0_u8; width * height * 3];
        for y in 150..200 {
            for x in 0..width {
                let offset = (y * width + x) * 3;
                mask_rgb[offset..offset + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        let terrain = CloudMaskReadback::from_rgb(&mask_rgb, width, height);
        assert_eq!(terrain.statistics.count, 20_000);
        assert_eq!(
            (terrain.statistics.min_y, terrain.statistics.max_y),
            (150, 199)
        );
        assert!(terrain_mask_is_ready(&terrain));

        for y in 0..40 {
            for x in 0..width {
                let offset = (y * width + x) * 3;
                mask_rgb[offset..offset + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        assert!(!terrain_mask_is_ready(&CloudMaskReadback::from_rgb(
            &mask_rgb, width, height,
        )));
    }

    #[test]
    fn camera_statistics_exclude_exact_three_cell_lateral_open_band() {
        let ny = 10;
        let nx = 10;
        assert!(!interior_mass_cell(2 * nx + 5, ny, nx));
        assert!(!interior_mass_cell(5 * nx + 2, ny, nx));
        assert!(interior_mass_cell(3 * nx + 3, ny, nx));
        assert!(interior_mass_cell(6 * nx + 6, ny, nx));
        assert!(!interior_mass_cell(7 * nx + 6, ny, nx));
    }

    #[test]
    fn chase_camera_is_exact_terrain_relative_and_fits_dominant_component() {
        let shape = [8, 20, 20];
        let mut values = vec![0.0_f32; shape.iter().product()];
        for k in 2..7 {
            for j in 6..15 {
                for i in 6..15 {
                    values[(k * shape[1] + j) * shape[2] + i] = 3.0e-3;
                }
            }
        }
        let field = VolumeField {
            id: "analytic-chase-camera".to_owned(),
            label: "analytic chase camera".to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "controlled connected camera fixture".to_owned(),
            source: "controlled analytic operator fixture".to_owned(),
            shape,
            values_sha256: f32_sha256(&values),
            finite_min: 0.0,
            finite_max: 3.0e-3,
            values,
        };
        let frame = frame_dense_condensate(&field);
        let position = admitted_chase_position(&field, &frame);
        let terrain = camera_terrain_height_km(position.x, position.z).unwrap();
        assert!(
            (position.y - terrain - CHASE_EYE_ABOVE_TERRAIN_KM).abs() < 1.0e-6,
            "position={position:?} terrain={terrain}"
        );
        let outside_x = position.x < frame.dense_min.x - super::CHASE_DENSE_CLEARANCE_KM
            || position.x > frame.dense_max.x + super::CHASE_DENSE_CLEARANCE_KM;
        let outside_z = position.z < frame.dense_min.z - super::CHASE_DENSE_CLEARANCE_KM
            || position.z > frame.dense_max.z + super::CHASE_DENSE_CLEARANCE_KM;
        assert!(
            outside_x || outside_z,
            "position={position:?} frame={frame:?}"
        );
        let camera = admitted_chase_camera(&field, &frame);
        let rotation = bevy::prelude::Quat::from_euler(
            bevy::prelude::EulerRot::YXZ,
            camera.yaw,
            camera.pitch,
            0.0,
        );
        let reconstructed = camera.focus - rotation * Vec3::NEG_Z * camera.radius;
        assert!((reconstructed - position).length() < 1.0e-5);
        let projection = project_dominant_component(&field, &frame, position, camera.focus);
        assert!((0.45..=0.65).contains(&projection.width_fraction));
        assert!(projection.mass_fraction_in_frame >= 0.995);
        assert!(
            projection
                .minimum_ndc
                .cmpge(Vec3::splat(-0.98).truncate())
                .all()
        );
        assert!(
            projection
                .maximum_ndc
                .cmple(Vec3::splat(0.98).truncate())
                .all()
        );
        assert!(
            position.y < frame.dense_min.y && camera.pitch > 0.0,
            "terrain-relative chase position={:?} frame={:?}",
            position,
            frame
        );
    }
}
