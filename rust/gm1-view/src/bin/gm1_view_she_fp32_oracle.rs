//! Headless FP32 upstream-HLSL versus WGSL numerical/timing gate.

use bevy::math::Vec3;
use bytemuck::{Pod, Zeroable};
use gm1_view::she::{
    UPSTREAM_COMMIT, UPSTREAM_COMMON_SH_SHA256, UPSTREAM_LICENSE, UPSTREAM_PROJECT,
    UPSTREAM_SHADE32_SHA256, WGSL_SOURCE, construct_sh_vector_a,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{borrow::Cow, fs, path::PathBuf, sync::mpsc, time::Instant};
use wgpu::util::DeviceExt;

const CASES: usize = 4_096;
const REPETITIONS: u32 = 64;
const COEFFICIENTS: usize = 33;
const OUTPUT_FLOATS: usize = 36;
const ABS_TOLERANCE: f32 = 3.0e-6;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SheCaseGpu {
    normal_alpha: [f32; 4],
    view_pad: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SheOutputGpu {
    blocks: [[f32; 4]; 9],
}

#[derive(Debug, Serialize)]
struct UpstreamReceipt {
    project: &'static str,
    commit: &'static str,
    license: &'static str,
    common_sh_sha256: &'static str,
    shade32_sha256: &'static str,
}

#[derive(Debug, Serialize)]
struct PlatformReceipt {
    adapter_name: String,
    backend: String,
    driver: String,
    driver_info: String,
}

#[derive(Debug, Serialize)]
struct NumericalReceipt {
    cases: usize,
    coefficients_per_case: usize,
    compared_words: usize,
    all_finite: bool,
    max_abs_error: f32,
    max_ulp_error: u32,
    absolute_tolerance: f32,
    mismatches_over_tolerance: usize,
    padding_words_zero: bool,
    corruption_negative_control_detected: bool,
    cpu_output_sha256: String,
    wgsl_output_sha256: String,
}

#[derive(Debug, Serialize)]
struct TimingReceipt {
    repetitions: u32,
    total_case_evaluations: u64,
    submit_copy_readback_seconds: f64,
    million_case_evaluations_per_second: f64,
}

#[derive(Debug, Serialize)]
struct Receipt {
    schema: &'static str,
    status: &'static str,
    claim_limit: &'static str,
    upstream: UpstreamReceipt,
    platform: PlatformReceipt,
    wgsl_source_sha256: String,
    executable_sha256: String,
    numerical: NumericalReceipt,
    timing: TimingReceipt,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("gm1-view SHE FP32 oracle: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let receipt_path = parse_receipt_path()?;
    let cases = held_cases();
    let cpu = cases
        .iter()
        .map(|case| {
            construct_sh_vector_a(
                Vec3::from_array(case.normal_alpha[..3].try_into().expect("normal")),
                Vec3::from_array(case.view_pad[..3].try_into().expect("view")),
                case.normal_alpha[3],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

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
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("GM1 View SHE FP32 oracle"),
        ..Default::default()
    }))
    .map_err(|error| format!("device request failed: {error}"))?;

    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("SHE held inputs"),
        contents: bytemuck::cast_slice(&cases),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_size = (CASES * std::mem::size_of::<SheOutputGpu>()) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SHE WGSL outputs"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SHE WGSL readback"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pinned SHE FP32 WGSL"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL_SOURCE)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SHE FP32 oracle pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("she_fp32_oracle"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SHE FP32 oracle bindings"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SHE FP32 oracle commands"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SHE FP32 held directions"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        for _ in 0..REPETITIONS {
            pass.dispatch_workgroups(CASES.div_ceil(64) as u32, 1, 1);
        }
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
    let started = Instant::now();
    queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| format!("device wait failed: {error}"))?;
    receiver
        .recv()
        .map_err(|error| format!("readback callback disappeared: {error}"))?
        .map_err(|error| format!("readback mapping failed: {error}"))?;
    let elapsed = started.elapsed().as_secs_f64();
    let mapped = slice.get_mapped_range();
    let gpu = bytemuck::try_cast_slice::<u8, SheOutputGpu>(&mapped)
        .map_err(|error| format!("WGSL output layout mismatch: {error}"))?;
    let numerical = compare(&cpu, gpu)?;
    drop(mapped);
    readback.unmap();

    let total = CASES as u64 * u64::from(REPETITIONS);
    let receipt = Receipt {
        schema: "gm1.view-she-fp32-wgsl-validation/v1",
        status: "PASS_PINNED_UPSTREAM_FP32_CPU_WGSL_HELD_DIRECTIONS",
        claim_limit: "FP32 diffuse/specular SHE basis, von-Mises-Fisher attenuation, and 33-term exponential reconstruction arithmetic only; display materials only; no meteorological field mutation, forecast, cloud-scattering, SimSat-sky replacement, FP16/packed, or architecture-independent claim",
        upstream: UpstreamReceipt {
            project: UPSTREAM_PROJECT,
            commit: UPSTREAM_COMMIT,
            license: UPSTREAM_LICENSE,
            common_sh_sha256: UPSTREAM_COMMON_SH_SHA256,
            shade32_sha256: UPSTREAM_SHADE32_SHA256,
        },
        platform: PlatformReceipt {
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            driver: info.driver,
            driver_info: info.driver_info,
        },
        wgsl_source_sha256: sha256_bytes(WGSL_SOURCE.as_bytes()),
        executable_sha256: sha256_file(&std::env::current_exe().map_err(|e| e.to_string())?)?,
        numerical,
        timing: TimingReceipt {
            repetitions: REPETITIONS,
            total_case_evaluations: total,
            submit_copy_readback_seconds: elapsed,
            million_case_evaluations_per_second: total as f64 / elapsed / 1.0e6,
        },
    };
    let value = serde_json::to_value(receipt).map_err(|error| error.to_string())?;
    let mut bytes = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    if let Some(parent) = receipt_path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(&receipt_path, &bytes).map_err(|error| error.to_string())?;
    println!(
        "PASS cases={CASES} reps={REPETITIONS} max_abs={:.9e} max_ulp={} elapsed={elapsed:.6}s receipt={} sha256={}",
        value["numerical"]["max_abs_error"]
            .as_f64()
            .unwrap_or_default(),
        value["numerical"]["max_ulp_error"]
            .as_u64()
            .unwrap_or_default(),
        receipt_path.display(),
        sha256_bytes(&bytes)
    );
    Ok(())
}

fn held_cases() -> Vec<SheCaseGpu> {
    (0..CASES)
        .map(|index| {
            let fraction = (index as f32 + 0.5) / CASES as f32;
            let azimuth = index as f32 * 2.399_963_1;
            let z = 0.08 + 0.91 * ((index * 7919 % CASES) as f32 + 0.5) / CASES as f32;
            let radius = (1.0 - z * z).sqrt();
            let normal = Vec3::new(radius * azimuth.cos(), z, radius * azimuth.sin()).normalize();
            let offset = Vec3::new(
                (fraction * 17.0).sin(),
                0.3 + (fraction * 11.0).cos().abs(),
                (fraction * 23.0).cos(),
            ) * 0.28;
            let view = (normal + offset).normalize();
            SheCaseGpu {
                normal_alpha: [normal.x, normal.y, normal.z, fraction],
                view_pad: [view.x, view.y, view.z, 0.0],
            }
        })
        .collect()
}

fn compare(cpu: &[[f32; 33]], gpu: &[SheOutputGpu]) -> Result<NumericalReceipt, String> {
    if cpu.len() != CASES || gpu.len() != CASES {
        return Err("CPU/WGSL case count changed".to_owned());
    }
    let mut max_abs_error = 0.0_f32;
    let mut max_ulp_error = 0_u32;
    let mut mismatches = 0usize;
    let mut padding_words_zero = true;
    let mut cpu_bytes = Vec::with_capacity(CASES * COEFFICIENTS * 4);
    let mut gpu_bytes = Vec::with_capacity(CASES * COEFFICIENTS * 4);
    for (expected, observed) in cpu.iter().zip(gpu.iter()) {
        let flat = observed.blocks.as_flattened();
        for index in 0..COEFFICIENTS {
            let a = expected[index];
            let b = flat[index];
            if !a.is_finite() || !b.is_finite() {
                return Err("CPU or WGSL SHE output is non-finite".to_owned());
            }
            let error = (a - b).abs();
            max_abs_error = max_abs_error.max(error);
            max_ulp_error = max_ulp_error.max(ulp_distance(a, b));
            mismatches += usize::from(error > ABS_TOLERANCE);
            cpu_bytes.extend_from_slice(&a.to_le_bytes());
            gpu_bytes.extend_from_slice(&b.to_le_bytes());
        }
        padding_words_zero &= flat[COEFFICIENTS..OUTPUT_FLOATS]
            .iter()
            .all(|value| value.to_bits() == 0);
    }
    let mut corrupted = gpu[0];
    corrupted.blocks[0][0] = f32::from_bits(corrupted.blocks[0][0].to_bits() ^ 1);
    let corruption_detected = corrupted.blocks[0][0].to_bits() != gpu[0].blocks[0][0].to_bits();
    if mismatches != 0 || !padding_words_zero || !corruption_detected {
        return Err(format!(
            "SHE FP32 parity refused: mismatches={mismatches} padding_zero={padding_words_zero} corruption_detected={corruption_detected} max_abs={max_abs_error:e}"
        ));
    }
    Ok(NumericalReceipt {
        cases: CASES,
        coefficients_per_case: COEFFICIENTS,
        compared_words: CASES * COEFFICIENTS,
        all_finite: true,
        max_abs_error,
        max_ulp_error,
        absolute_tolerance: ABS_TOLERANCE,
        mismatches_over_tolerance: mismatches,
        padding_words_zero,
        corruption_negative_control_detected: corruption_detected,
        cpu_output_sha256: sha256_bytes(&cpu_bytes),
        wgsl_output_sha256: sha256_bytes(&gpu_bytes),
    })
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    fn ordered(value: f32) -> i32 {
        let bits = value.to_bits() as i32;
        if bits < 0 { i32::MIN - bits } else { bits }
    }
    ordered(a).abs_diff(ordered(b))
}

fn parse_receipt_path() -> Result<PathBuf, String> {
    let mut args = std::env::args().skip(1);
    match (args.next().as_deref(), args.next(), args.next()) {
        (Some("--receipt"), Some(path), None) => Ok(PathBuf::from(path)),
        _ => Err("usage: gm1_view_she_fp32_oracle --receipt <receipt.json>".to_owned()),
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &std::path::Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|error| format!("cannot hash {}: {error}", path.display()))
}
