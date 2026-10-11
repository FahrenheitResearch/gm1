//! Source-backed HRRR terrain context around the exact GM1 child domain.
//!
//! The child terrain remains the authoritative 50 m model surface.  This
//! module only builds a display-context mesh from an independently sealed
//! native HRRR `SOILHGT` window.  Source grid points are transformed from the
//! canonical HRRR Lambert grid through latitude/longitude and ECEF into the
//! local east/up/north frame at the exact child center.  The ECEF transform
//! retains physical Earth curvature; no height extrapolation or procedural
//! detail is added.

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::render_resource::PrimitiveTopology,
};
use cm1_state::hrrr::{Cm1TargetGrid, HrrrLambertGrid};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub const HRRR_EARTH_RADIUS_M: f64 = 6_371_229.0;
pub const HRRR_FULL_NX: usize = 1_799;
pub const HRRR_FULL_NY: usize = 1_059;
pub const HRRR_SURROUND_WINDOW_I_START: usize = 959;
pub const HRRR_SURROUND_WINDOW_J_START: usize = 609;
pub const HRRR_SURROUND_NX: usize = 64;
pub const HRRR_SURROUND_NY: usize = 64;
pub const HRRR_SURROUND_DX_M: f64 = 3_000.0;
pub const HRRR_SURROUND_DY_M: f64 = 3_000.0;
pub const HRRR_WRFNAT_F02_SHA256: &str =
    "63c75fba7f25f619e219d9a23b3e0c07cfb8e362b4e2f23ee54937be2f63d218";
pub const HRRR_SOIL_F02_SHA256: &str =
    "703945dc2a10e2416ae5773444aa0483cd5e645c2ca116f554f101ebd29c1fbf";
pub const HRRR_WRFNAT_F03_SHA256: &str =
    "acec58d144f6a3398159af604f8f6436e277de74a2baf5cdcb5c6574b44ab8f0";
pub const HRRR_SOIL_F03_SHA256: &str =
    "8e6fcc42a50dd26bd349475383480578d25a861bb740f67cf039bf1514c9efbd";
pub const HRRR_DECODED_WINDOW_SCHEMA: &str = "gpuwm-cm1.hrrr-decoded-f32-window/v1";
pub const HRRR_SOURCE_F02_FORECAST_HOUR: u32 = 2;
pub const HRRR_SOURCE_VALID_TIME_UTC: &str = "2024-05-21T21:00:00Z";
pub const HRRR_SURROUND_MANIFEST_SCHEMA: &str = "gm1.view.hrrr-source-backed-surround-manifest/v1";
pub const HRRR_SURROUND_RECEIPT_SCHEMA: &str = "gm1.view.hrrr-source-backed-surround-receipt/v1";
pub const HRRR_SURROUND_STATUS: &str = "PASS_SOURCE_BACKED_HRRR_SURROUND_TERRAIN";
pub const HRRR_BRIDGE_SHA256: &str =
    "c45e31cb323850a13985b9a83266e69d6b00a3f5827c6103787653a6cce60183";
pub const HRRR_BRIDGE_SOURCE_COMMIT: &str = "09e69271017d13632846dec2a858e3f2d4e5c923";
pub const HRRR_BRIDGE_BUILD_MANIFEST_SHA256: &str =
    "b440948e2a238210fd9377496bb2f885f4fe27f74a7fc5d957fd8af93b4c1a0b";
pub const HRRR_FETCH_MANIFEST_SHA256: &str =
    "d53e5b659dcb47b4a0e0a9d8a4bb6d9c31db839b3460d1f0ebe6f29b65fc77c0";
pub const EXACT_CHILD_TERRAIN_SHA256: &str =
    "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91";
pub const HRRR_SURROUND_CLAIM_LIMIT: &str = "source-backed HRRR f02 SOILHGT display context with spherical ECEF-to-child-ENU curvature and exact-child ownership; not model terrain, evolved meteorology, sub-grid topography, roads, buildings, or procedural fill";
pub const HRRR_SURROUND_OWNERSHIP: &str = "the exact 300x300 50m child terrain is authoritative inside its +/-7.475km mass-point footprint; clipped HRRR triangles own only the exterior";
pub const HRRR_SURROUND_HEIGHT_TREATMENT: &str = "native decoded binary32 SOILHGT heights; triangle-edge interpolation only; no extrapolation, displacement, noise, or authored extension";
pub const HRRR_SURROUND_PAYLOAD_NAME: &str = "gm1-hrrr-f02-soilhgt-64x64.f32le";
pub const HRRR_SURROUND_ASSET_DIRECTORY: &str = "hrrr-surround-v1";
pub const HRRR_SURROUND_MANIFEST_NAME: &str = "gm1-hrrr-f02-soilhgt-64x64.manifest.json";
pub const HRRR_SURROUND_RECEIPT_NAME: &str = "gm1-hrrr-f02-soilhgt-64x64.receipt.json";

const PARENT_NX: usize = 300;
const PARENT_NY: usize = 300;
const PARENT_NZ: usize = 75;
const PARENT_DX_M: f64 = 100.0;
const PARENT_DY_M: f64 = 100.0;
const PARENT_DZ_M: f64 = 200.0;
const PARENT_CENTER_LAT_DEG: f64 = f64::from_bits(0x4044_b283_93b6_36d0);
const PARENT_CENTER_LON_DEG: f64 = f64::from_bits(0xc057_9217_7a2c_3f9c);
const CHILD_PARENT_CENTER_I: usize = 224;
const CHILD_PARENT_CENTER_J: usize = 224;
/// Exact outer mass-point footprint of the 300 x 300, 50 m child mesh.
const CHILD_SURFACE_HALF_EXTENT_KM: f32 = 7.475;
/// A 100 m AGL observer on a 6,371,229 m sphere has a 35.696 km horizon.
/// Requiring 36 km beyond every child edge keeps the decoded perimeter below
/// that horizon for every horizontally in-domain chase-camera position.
const MINIMUM_CONTEXT_MARGIN_KM: f32 = 36.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrrrSurroundSpec {
    pub window_i_start: usize,
    pub window_j_start: usize,
    pub nx: usize,
    pub ny: usize,
}

impl HrrrSurroundSpec {
    pub const fn admitted() -> Self {
        Self {
            window_i_start: HRRR_SURROUND_WINDOW_I_START,
            window_j_start: HRRR_SURROUND_WINDOW_J_START,
            nx: HRRR_SURROUND_NX,
            ny: HRRR_SURROUND_NY,
        }
    }

    fn validate(self) -> Result<(), String> {
        if self.nx < 2
            || self.ny < 2
            || self.window_i_start + self.nx > HRRR_FULL_NX
            || self.window_j_start + self.ny > HRRR_FULL_NY
        {
            return Err("HRRR surrounding-terrain window is outside the canonical grid".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HrrrSurroundManifest {
    schema: String,
    status: String,
    source: HrrrSurroundSource,
    decoder: HrrrSurroundDecoder,
    payload: HrrrSurroundPayload,
    grid: HrrrSurroundGrid,
    transform: HrrrSurroundTransform,
    overlay: HrrrSurroundOverlay,
    claim_limit: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HrrrSurroundSource {
    cycle_utc: String,
    valid_time_utc: String,
    forecast_hour: u32,
    fetch_manifest_sha256: String,
    files: Vec<HrrrRawFile>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HrrrRawFile {
    name: String,
    nbytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HrrrSurroundDecoder {
    name: String,
    source_commit: String,
    executable_sha256: String,
    build_manifest_sha256: String,
    output_schema: String,
    gate_sha256: String,
    inventory_sha256: String,
    atmosphere_f02_fingerprint_sha256: String,
    atmosphere_f03_fingerprint_sha256: String,
    decoded_tree_sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HrrrSurroundPayload {
    name: String,
    nbytes: u64,
    sha256: String,
    dtype: String,
    order: String,
    shape: [usize; 2],
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HrrrSurroundGrid {
    name: String,
    full_shape_xy: [usize; 2],
    window_origin_ij: [usize; 2],
    window_shape_xy: [usize; 2],
    spacing_m_xy: [u64; 2],
    scan_mode: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HrrrSurroundTransform {
    method: String,
    earth_radius_m_f64_bits: String,
    child_center_lat_f64_bits: String,
    child_center_lon_f64_bits: String,
    output_axes_units: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HrrrSurroundOverlay {
    exact_child_terrain_sha256: String,
    exact_child_half_extent_km_f32_bits: String,
    ownership: String,
    height_treatment: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct HrrrSurroundReceipt {
    schema: String,
    status: String,
    manifest_sha256: String,
    payload_sha256: String,
    exact_child_terrain_sha256: String,
    no_extrapolation: bool,
    earth_curvature_retained: bool,
    claim_limit: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrrrSurroundStats {
    pub child_center_lat_deg: f64,
    pub child_center_lon_deg: f64,
    pub minimum_terrain_m: f32,
    pub maximum_terrain_m: f32,
    pub minimum_east_km: f32,
    pub maximum_east_km: f32,
    pub minimum_north_km: f32,
    pub maximum_north_km: f32,
    pub minimum_up_km: f32,
    pub maximum_up_km: f32,
    pub indexed_triangle_count: usize,
}

impl HrrrSurroundStats {
    pub fn horizontal_margin_km(self) -> f32 {
        [
            -self.minimum_east_km - CHILD_SURFACE_HALF_EXTENT_KM,
            self.maximum_east_km - CHILD_SURFACE_HALF_EXTENT_KM,
            -self.minimum_north_km - CHILD_SURFACE_HALF_EXTENT_KM,
            self.maximum_north_km - CHILD_SURFACE_HALF_EXTENT_KM,
        ]
        .into_iter()
        .fold(f32::INFINITY, f32::min)
    }
}

/// Strictly admit the immutable presentation envelope and build its curved
/// exterior mesh. Receipt and manifest hashes cover their exact bytes; the
/// payload is independently hashed before any float is decoded.
pub fn source_backed_surround_mesh_from_envelope(
    manifest_bytes: &[u8],
    receipt_bytes: &[u8],
    payload_bytes: &[u8],
) -> Result<(Mesh, HrrrSurroundStats), String> {
    let manifest: HrrrSurroundManifest = serde_json::from_slice(manifest_bytes)
        .map_err(|error| format!("HRRR surrounding-terrain manifest refused: {error}"))?;
    let receipt: HrrrSurroundReceipt = serde_json::from_slice(receipt_bytes)
        .map_err(|error| format!("HRRR surrounding-terrain receipt refused: {error}"))?;
    validate_manifest(&manifest)?;
    validate_receipt(&receipt, manifest_bytes, &manifest.payload)?;
    if payload_bytes.len() as u64 != manifest.payload.nbytes
        || sha256_bytes(payload_bytes) != manifest.payload.sha256
    {
        return Err("HRRR surrounding-terrain payload bytes changed".into());
    }
    source_backed_surround_mesh(
        payload_bytes,
        &manifest.payload.sha256,
        HrrrSurroundSpec::admitted(),
    )
}

/// Load the three immutable presentation assets without accepting a partial
/// directory. The manifest and receipt still bind the decoder and raw GRIB
/// sources; this helper only supplies their bytes to the strict envelope gate.
pub fn load_source_backed_surround_assets(
    directory: &Path,
) -> Result<(Mesh, HrrrSurroundStats), String> {
    let read = |name: &str| {
        fs::read(directory.join(name)).map_err(|error| {
            format!(
                "failed to read HRRR surrounding-terrain asset {}: {error}",
                directory.join(name).display()
            )
        })
    };
    let manifest = read(HRRR_SURROUND_MANIFEST_NAME)?;
    let receipt = read(HRRR_SURROUND_RECEIPT_NAME)?;
    let payload = read(HRRR_SURROUND_PAYLOAD_NAME)?;
    source_backed_surround_mesh_from_envelope(&manifest, &receipt, &payload)
}

/// Sample the admitted curved HRRR surface at one child-ENU horizontal point.
///
/// The returned height is barycentric interpolation on the exact clipped
/// source triangle that is also rendered. Points in the authoritative child
/// footprint (the intentional hole) or outside the decoded window refuse.
pub fn source_backed_surround_height_km_from_assets(
    directory: &Path,
    east_km: f32,
    north_km: f32,
) -> Result<f32, String> {
    if !east_km.is_finite() || !north_km.is_finite() {
        return Err("HRRR surrounding-terrain sample coordinate is non-finite".into());
    }
    let (mesh, _) = load_source_backed_surround_assets(directory)?;
    sample_mesh_height_km(&mesh, east_km, north_km)
}

#[derive(Clone, Debug)]
pub struct HrrrSurroundHeightSampler {
    positions: Vec<[f32; 3]>,
    indices: Vec<usize>,
}

impl HrrrSurroundHeightSampler {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let (mesh, _) = load_source_backed_surround_assets(directory)?;
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(values)) => values.clone(),
            _ => return Err("HRRR surrounding-terrain mesh has no binary32 positions".into()),
        };
        let indices = match mesh.indices() {
            Some(Indices::U16(values)) => values.iter().map(|&value| value as usize).collect(),
            Some(Indices::U32(values)) => values.iter().map(|&value| value as usize).collect(),
            None => return Err("HRRR surrounding-terrain mesh has no triangle indices".into()),
        };
        Ok(Self { positions, indices })
    }

    pub fn sample_height_km(&self, east_km: f32, north_km: f32) -> Result<f32, String> {
        if !east_km.is_finite() || !north_km.is_finite() {
            return Err("HRRR surrounding-terrain sample coordinate is non-finite".into());
        }
        sample_triangles_height_km(&self.positions, &self.indices, east_km, north_km)
    }
}

fn sample_mesh_height_km(mesh: &Mesh, east_km: f32, north_km: f32) -> Result<f32, String> {
    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(values)) => values,
        _ => return Err("HRRR surrounding-terrain mesh has no binary32 positions".into()),
    };
    let indices: Vec<usize> = match mesh.indices() {
        Some(Indices::U16(values)) => values.iter().map(|&value| value as usize).collect(),
        Some(Indices::U32(values)) => values.iter().map(|&value| value as usize).collect(),
        None => return Err("HRRR surrounding-terrain mesh has no triangle indices".into()),
    };
    sample_triangles_height_km(positions, &indices, east_km, north_km)
}

fn sample_triangles_height_km(
    positions: &[[f32; 3]],
    indices: &[usize],
    east_km: f32,
    north_km: f32,
) -> Result<f32, String> {
    let point = Vec2::new(east_km, north_km);
    for triangle in indices.chunks_exact(3) {
        let a = Vec3::from_array(positions[triangle[0]]);
        let b = Vec3::from_array(positions[triangle[1]]);
        let c = Vec3::from_array(positions[triangle[2]]);
        let a_xz = Vec2::new(a.x, a.z);
        let b_xz = Vec2::new(b.x, b.z);
        let c_xz = Vec2::new(c.x, c.z);
        let denominator =
            (b_xz.y - c_xz.y) * (a_xz.x - c_xz.x) + (c_xz.x - b_xz.x) * (a_xz.y - c_xz.y);
        if denominator.abs() <= 1.0e-12 {
            continue;
        }
        let wa = ((b_xz.y - c_xz.y) * (point.x - c_xz.x) + (c_xz.x - b_xz.x) * (point.y - c_xz.y))
            / denominator;
        let wb = ((c_xz.y - a_xz.y) * (point.x - c_xz.x) + (a_xz.x - c_xz.x) * (point.y - c_xz.y))
            / denominator;
        let wc = 1.0 - wa - wb;
        let tolerance = -2.0e-5;
        if wa >= tolerance && wb >= tolerance && wc >= tolerance {
            return Ok(wa * a.y + wb * b.y + wc * c.y);
        }
    }
    Err(format!(
        "HRRR surrounding-terrain point ({east_km:.6}, {north_km:.6}) is outside the admitted exterior mesh"
    ))
}

fn validate_manifest(manifest: &HrrrSurroundManifest) -> Result<(), String> {
    let expected_files = [
        HrrrRawFile {
            name: "hrrr.t19z.soilf02.grib2".into(),
            nbytes: 431_574_022,
            sha256: HRRR_SOIL_F02_SHA256.into(),
        },
        HrrrRawFile {
            name: "hrrr.t19z.soilf03.grib2".into(),
            nbytes: 434_331_894,
            sha256: HRRR_SOIL_F03_SHA256.into(),
        },
        HrrrRawFile {
            name: "hrrr.t19z.wrfnatf02.grib2".into(),
            nbytes: 763_838_413,
            sha256: HRRR_WRFNAT_F02_SHA256.into(),
        },
        HrrrRawFile {
            name: "hrrr.t19z.wrfnatf03.grib2".into(),
            nbytes: 766_890_103,
            sha256: HRRR_WRFNAT_F03_SHA256.into(),
        },
    ];
    if manifest.schema != HRRR_SURROUND_MANIFEST_SCHEMA
        || manifest.status != HRRR_SURROUND_STATUS
        || manifest.source.cycle_utc != "2024-05-21T19:00:00Z"
        || manifest.source.valid_time_utc != HRRR_SOURCE_VALID_TIME_UTC
        || manifest.source.forecast_hour != HRRR_SOURCE_F02_FORECAST_HOUR
        || manifest.source.fetch_manifest_sha256 != HRRR_FETCH_MANIFEST_SHA256
        || manifest.source.files != expected_files
        || manifest.decoder.name
            != "ArWen RW-WPS hrrr_grib2_bridge @ 09e69271017d13632846dec2a858e3f2d4e5c923"
        || manifest.decoder.source_commit != HRRR_BRIDGE_SOURCE_COMMIT
        || manifest.decoder.executable_sha256 != HRRR_BRIDGE_SHA256
        || manifest.decoder.build_manifest_sha256 != HRRR_BRIDGE_BUILD_MANIFEST_SHA256
        || manifest.decoder.output_schema != HRRR_DECODED_WINDOW_SCHEMA
        || manifest.payload.name != "SOILHGT.f32le"
        || manifest.payload.nbytes != (HRRR_SURROUND_NX * HRRR_SURROUND_NY * 4) as u64
        || manifest.payload.dtype != "little-endian binary32"
        || manifest.payload.order != "south-to-north row-major"
        || manifest.payload.shape != [HRRR_SURROUND_NY, HRRR_SURROUND_NX]
        || manifest.grid.name != "HRRR Lambert GDT3.30"
        || manifest.grid.full_shape_xy != [HRRR_FULL_NX, HRRR_FULL_NY]
        || manifest.grid.window_origin_ij
            != [HRRR_SURROUND_WINDOW_I_START, HRRR_SURROUND_WINDOW_J_START]
        || manifest.grid.window_shape_xy != [HRRR_SURROUND_NX, HRRR_SURROUND_NY]
        || manifest.grid.spacing_m_xy != [3_000, 3_000]
        || manifest.grid.scan_mode != "0x40"
        || manifest.transform.method != "HRRR Lambert inverse -> spherical ECEF -> child ENU"
        || manifest.transform.earth_radius_m_f64_bits != "41584de740000000"
        || manifest.transform.child_center_lat_f64_bits != "4044bb15cda77904"
        || manifest.transform.child_center_lon_f64_bits != "c0578c5ebb59704f"
        || manifest.transform.output_axes_units != "east-km, up-km, north-km"
        || manifest.overlay.exact_child_terrain_sha256 != EXACT_CHILD_TERRAIN_SHA256
        || manifest.overlay.exact_child_half_extent_km_f32_bits
            != format!("{:08x}", CHILD_SURFACE_HALF_EXTENT_KM.to_bits())
        || manifest.overlay.ownership != HRRR_SURROUND_OWNERSHIP
        || manifest.overlay.height_treatment != HRRR_SURROUND_HEIGHT_TREATMENT
        || manifest.claim_limit != HRRR_SURROUND_CLAIM_LIMIT
    {
        return Err("HRRR surrounding-terrain manifest contract changed".into());
    }
    for hash in [
        &manifest.decoder.gate_sha256,
        &manifest.decoder.inventory_sha256,
        &manifest.decoder.atmosphere_f02_fingerprint_sha256,
        &manifest.decoder.atmosphere_f03_fingerprint_sha256,
        &manifest.decoder.decoded_tree_sha256,
        &manifest.payload.sha256,
    ] {
        validate_sha256(hash)?;
    }
    Ok(())
}

fn validate_receipt(
    receipt: &HrrrSurroundReceipt,
    manifest_bytes: &[u8],
    payload: &HrrrSurroundPayload,
) -> Result<(), String> {
    if receipt.schema != HRRR_SURROUND_RECEIPT_SCHEMA
        || receipt.status != HRRR_SURROUND_STATUS
        || receipt.manifest_sha256 != sha256_bytes(manifest_bytes)
        || receipt.payload_sha256 != payload.sha256
        || receipt.exact_child_terrain_sha256 != EXACT_CHILD_TERRAIN_SHA256
        || !receipt.no_extrapolation
        || !receipt.earth_curvature_retained
        || receipt.claim_limit != HRRR_SURROUND_CLAIM_LIMIT
    {
        return Err("HRRR surrounding-terrain receipt contract changed".into());
    }
    Ok(())
}

/// Build the admitted source-context mesh and return its measured bounds.
///
/// `expected_sha256` seals the exact decoded `SOILHGT.f32le` payload.  The
/// caller is responsible for pinning the decoder tree/receipt alongside it.
pub fn source_backed_surround_mesh(
    bytes: &[u8],
    expected_sha256: &str,
    spec: HrrrSurroundSpec,
) -> Result<(Mesh, HrrrSurroundStats), String> {
    spec.validate()?;
    validate_sha256(expected_sha256)?;
    if sha256_bytes(bytes) != expected_sha256 {
        return Err("decoded HRRR surrounding terrain identity changed".into());
    }
    let expected_bytes = spec
        .nx
        .checked_mul(spec.ny)
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| "HRRR surrounding terrain byte count overflowed".to_owned())?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "decoded HRRR surrounding terrain has {} bytes; expected {expected_bytes}",
            bytes.len()
        ));
    }
    let terrain = bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("four terrain bytes")))
        .collect::<Vec<_>>();
    if terrain.iter().any(|value| !value.is_finite()) {
        return Err("decoded HRRR surrounding terrain contains a non-finite word".into());
    }

    let grid = HrrrLambertGrid::canonical_window(
        spec.window_i_start,
        spec.window_j_start,
        spec.nx,
        spec.ny,
    )
    .map_err(|error| format!("canonical HRRR grid refused surround: {error}"))?;
    let (center_lat_deg, center_lon_deg) = child_center_lat_lon()?;
    let origin = ecef(center_lat_deg, center_lon_deg, 0.0);
    let basis = EnuBasis::at(center_lat_deg, center_lon_deg);

    let mut positions = Vec::with_capacity(terrain.len());
    for j in 0..spec.ny {
        for i in 0..spec.nx {
            let (lat_deg, lon_deg) = grid
                .grid_point_lat_lon(
                    (spec.window_i_start + i) as f64,
                    (spec.window_j_start + j) as f64,
                )
                .map_err(|error| format!("HRRR grid coordinate refused: {error}"))?;
            let point = ecef(lat_deg, lon_deg, f64::from(terrain[j * spec.nx + i]));
            let local = basis.project(subtract(point, origin));
            positions.push([
                (local[0] / 1_000.0) as f32,
                (local[1] / 1_000.0) as f32,
                (local[2] / 1_000.0) as f32,
            ]);
        }
    }

    let mut normals = Vec::with_capacity(positions.len());
    let mut uvs = Vec::with_capacity(positions.len());
    for j in 0..spec.ny {
        for i in 0..spec.nx {
            let i0 = i.saturating_sub(1);
            let i1 = (i + 1).min(spec.nx - 1);
            let j0 = j.saturating_sub(1);
            let j1 = (j + 1).min(spec.ny - 1);
            let east = Vec3::from_array(positions[j * spec.nx + i1])
                - Vec3::from_array(positions[j * spec.nx + i0]);
            let north = Vec3::from_array(positions[j1 * spec.nx + i])
                - Vec3::from_array(positions[j0 * spec.nx + i]);
            let normal = north.cross(east).normalize_or_zero();
            if normal.length_squared() == 0.0 || normal.y <= 0.0 {
                return Err(
                    "HRRR surrounding terrain produced a degenerate/upside-down normal".into(),
                );
            }
            normals.push(normal.to_array());
            uvs.push([
                i as f32 / (spec.nx - 1) as f32,
                j as f32 / (spec.ny - 1) as f32,
            ]);
        }
    }

    let vertices = positions
        .iter()
        .zip(&normals)
        .zip(&uvs)
        .map(|((position, normal), uv)| ContextVertex {
            position: *position,
            normal: *normal,
            uv: *uv,
        })
        .collect::<Vec<_>>();
    let mut clipped_positions = Vec::with_capacity(positions.len() * 2);
    let mut clipped_normals = Vec::with_capacity(normals.len() * 2);
    let mut clipped_uvs = Vec::with_capacity(uvs.len() * 2);
    let mut indices = Vec::with_capacity((spec.nx - 1) * (spec.ny - 1) * 6);
    for j in 0..spec.ny - 1 {
        for i in 0..spec.nx - 1 {
            let a = j * spec.nx + i;
            let b = a + 1;
            let c = a + spec.nx;
            let d = c + 1;
            emit_triangle_outside_child(
                [vertices[a], vertices[c], vertices[b]],
                &mut clipped_positions,
                &mut clipped_normals,
                &mut clipped_uvs,
                &mut indices,
            );
            emit_triangle_outside_child(
                [vertices[b], vertices[c], vertices[d]],
                &mut clipped_positions,
                &mut clipped_normals,
                &mut clipped_uvs,
                &mut indices,
            );
        }
    }
    if indices.is_empty() {
        return Err("HRRR surrounding terrain has no triangles outside the child".into());
    }

    let stats = measured_stats(
        &terrain,
        &positions,
        indices.len() / 3,
        center_lat_deg,
        center_lon_deg,
    );
    if stats.horizontal_margin_km() < MINIMUM_CONTEXT_MARGIN_KM {
        return Err(format!(
            "HRRR surrounding terrain has only {:.3} km minimum context beyond the child; require {MINIMUM_CONTEXT_MARGIN_KM} km",
            stats.horizontal_margin_km()
        ));
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, clipped_positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, clipped_normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, clipped_uvs);
    mesh.insert_indices(Indices::U32(indices));
    Ok((mesh, stats))
}

#[derive(Clone, Copy, Debug)]
struct ContextVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[derive(Clone, Copy)]
enum ClipBoundary {
    LessEqual { axis: usize, value: f32 },
    GreaterEqual { axis: usize, value: f32 },
}

impl ClipBoundary {
    fn signed_distance(self, vertex: ContextVertex) -> f32 {
        match self {
            Self::LessEqual { axis, value } => value - vertex.position[axis],
            Self::GreaterEqual { axis, value } => vertex.position[axis] - value,
        }
    }
}

/// Clip one source triangle into four disjoint convex regions surrounding the
/// exact child footprint. Intersections are linear interpolation on the native
/// HRRR triangle; this introduces neither extrapolated height nor new detail.
fn emit_triangle_outside_child(
    triangle: [ContextVertex; 3],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u32>,
) {
    use ClipBoundary::{GreaterEqual as Ge, LessEqual as Le};
    let half = CHILD_SURFACE_HALF_EXTENT_KM;
    let regions: [&[ClipBoundary]; 4] = [
        &[Le {
            axis: 0,
            value: -half,
        }],
        &[Ge {
            axis: 0,
            value: half,
        }],
        &[
            Ge {
                axis: 0,
                value: -half,
            },
            Le {
                axis: 0,
                value: half,
            },
            Le {
                axis: 2,
                value: -half,
            },
        ],
        &[
            Ge {
                axis: 0,
                value: -half,
            },
            Le {
                axis: 0,
                value: half,
            },
            Ge {
                axis: 2,
                value: half,
            },
        ],
    ];
    for region in regions {
        let mut polygon = triangle.to_vec();
        for boundary in region {
            polygon = clip_polygon(&polygon, *boundary);
            if polygon.len() < 3 {
                break;
            }
        }
        if polygon.len() < 3 {
            continue;
        }
        let base = positions.len() as u32;
        for vertex in &polygon {
            positions.push(vertex.position);
            normals.push(vertex.normal);
            uvs.push(vertex.uv);
        }
        for index in 1..polygon.len() - 1 {
            let a =
                Vec3::from_array(polygon[index].position) - Vec3::from_array(polygon[0].position);
            let b = Vec3::from_array(polygon[index + 1].position)
                - Vec3::from_array(polygon[0].position);
            if a.cross(b).length_squared() > 1.0e-12 {
                indices.extend_from_slice(&[base, base + index as u32, base + index as u32 + 1]);
            }
        }
    }
}

fn clip_polygon(polygon: &[ContextVertex], boundary: ClipBoundary) -> Vec<ContextVertex> {
    if polygon.is_empty() {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(polygon.len() + 1);
    let mut previous = *polygon.last().expect("nonempty polygon");
    let mut previous_distance = boundary.signed_distance(previous);
    for current in polygon {
        let current_distance = boundary.signed_distance(*current);
        let previous_inside = previous_distance >= 0.0;
        let current_inside = current_distance >= 0.0;
        if previous_inside != current_inside {
            let weight = previous_distance / (previous_distance - current_distance);
            output.push(interpolate_vertex(previous, *current, weight));
        }
        if current_inside {
            output.push(*current);
        }
        previous = *current;
        previous_distance = current_distance;
    }
    output
}

fn interpolate_vertex(left: ContextVertex, right: ContextVertex, weight: f32) -> ContextVertex {
    let position = Vec3::from_array(left.position)
        .lerp(Vec3::from_array(right.position), weight)
        .to_array();
    let normal = Vec3::from_array(left.normal)
        .lerp(Vec3::from_array(right.normal), weight)
        .normalize()
        .to_array();
    let uv = Vec2::from_array(left.uv)
        .lerp(Vec2::from_array(right.uv), weight)
        .to_array();
    ContextVertex {
        position,
        normal,
        uv,
    }
}

pub fn horizon_distance_km(observer_agl_m: f64) -> Result<f64, String> {
    if !observer_agl_m.is_finite() || !(0.0..=100.0).contains(&observer_agl_m) {
        return Err("horizon observer AGL must be finite and in 0..=100 m".into());
    }
    Ok(
        ((HRRR_EARTH_RADIUS_M + observer_agl_m).powi(2) - HRRR_EARTH_RADIUS_M.powi(2)).sqrt()
            / 1_000.0,
    )
}

pub fn child_center_lat_lon() -> Result<(f64, f64), String> {
    let parent = Cm1TargetGrid::new(
        PARENT_NX,
        PARENT_NY,
        PARENT_NZ,
        PARENT_DX_M,
        PARENT_DY_M,
        PARENT_DZ_M,
        PARENT_CENTER_LAT_DEG,
        PARENT_CENTER_LON_DEG,
    )
    .map_err(|error| format!("sealed GM1 parent grid refused: {error}"))?;
    parent
        .mass_point_lat_lon(CHILD_PARENT_CENTER_I, CHILD_PARENT_CENTER_J)
        .map_err(|error| format!("sealed GM1 child center refused: {error}"))
}

fn measured_stats(
    terrain: &[f32],
    positions: &[[f32; 3]],
    indexed_triangle_count: usize,
    child_center_lat_deg: f64,
    child_center_lon_deg: f64,
) -> HrrrSurroundStats {
    let terrain_bounds = terrain.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(*value), maximum.max(*value)),
    );
    let bounds = positions.iter().fold(
        ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
        |(mut minimum, mut maximum), position| {
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(position[axis]);
                maximum[axis] = maximum[axis].max(position[axis]);
            }
            (minimum, maximum)
        },
    );
    HrrrSurroundStats {
        child_center_lat_deg,
        child_center_lon_deg,
        minimum_terrain_m: terrain_bounds.0,
        maximum_terrain_m: terrain_bounds.1,
        minimum_east_km: bounds.0[0],
        maximum_east_km: bounds.1[0],
        minimum_north_km: bounds.0[2],
        maximum_north_km: bounds.1[2],
        minimum_up_km: bounds.0[1],
        maximum_up_km: bounds.1[1],
        indexed_triangle_count,
    }
}

fn ecef(lat_deg: f64, lon_deg: f64, height_m: f64) -> [f64; 3] {
    let latitude = lat_deg.to_radians();
    let longitude = lon_deg.to_radians();
    let radius = HRRR_EARTH_RADIUS_M + height_m;
    [
        radius * latitude.cos() * longitude.cos(),
        radius * latitude.cos() * longitude.sin(),
        radius * latitude.sin(),
    ]
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

#[derive(Clone, Copy)]
struct EnuBasis {
    east: [f64; 3],
    up: [f64; 3],
    north: [f64; 3],
}

impl EnuBasis {
    fn at(lat_deg: f64, lon_deg: f64) -> Self {
        let latitude = lat_deg.to_radians();
        let longitude = lon_deg.to_radians();
        Self {
            east: [-longitude.sin(), longitude.cos(), 0.0],
            up: [
                latitude.cos() * longitude.cos(),
                latitude.cos() * longitude.sin(),
                latitude.sin(),
            ],
            north: [
                -latitude.sin() * longitude.cos(),
                -latitude.sin() * longitude.sin(),
                latitude.cos(),
            ],
        }
    }

    fn project(self, value: [f64; 3]) -> [f64; 3] {
        [
            dot(self.east, value),
            dot(self.up, value),
            dot(self.north, value),
        ]
    }
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn validate_sha256(value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("expected HRRR surrounding-terrain SHA-256 is not lowercase hex".into());
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    fn flat_fixture(spec: HrrrSurroundSpec) -> (Vec<u8>, String) {
        let mut bytes = Vec::with_capacity(spec.nx * spec.ny * 4);
        for _ in 0..spec.nx * spec.ny {
            bytes.extend_from_slice(&340.0_f32.to_le_bytes());
        }
        let hash = sha256_bytes(&bytes);
        (bytes, hash)
    }

    fn envelope_fixture() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let spec = HrrrSurroundSpec::admitted();
        let (payload, payload_sha256) = flat_fixture(spec);
        let manifest = serde_json::json!({
            "schema": HRRR_SURROUND_MANIFEST_SCHEMA,
            "status": HRRR_SURROUND_STATUS,
            "source": {
                "cycle_utc": "2024-05-21T19:00:00Z",
                "valid_time_utc": HRRR_SOURCE_VALID_TIME_UTC,
                "forecast_hour": HRRR_SOURCE_F02_FORECAST_HOUR,
                "fetch_manifest_sha256": HRRR_FETCH_MANIFEST_SHA256,
                "files": [
                    {"name":"hrrr.t19z.soilf02.grib2","nbytes":431574022_u64,"sha256":HRRR_SOIL_F02_SHA256},
                    {"name":"hrrr.t19z.soilf03.grib2","nbytes":434331894_u64,"sha256":HRRR_SOIL_F03_SHA256},
                    {"name":"hrrr.t19z.wrfnatf02.grib2","nbytes":763838413_u64,"sha256":HRRR_WRFNAT_F02_SHA256},
                    {"name":"hrrr.t19z.wrfnatf03.grib2","nbytes":766890103_u64,"sha256":HRRR_WRFNAT_F03_SHA256}
                ]
            },
            "decoder": {
                "name": "ArWen RW-WPS hrrr_grib2_bridge @ 09e69271017d13632846dec2a858e3f2d4e5c923",
                "source_commit": HRRR_BRIDGE_SOURCE_COMMIT,
                "executable_sha256": HRRR_BRIDGE_SHA256,
                "build_manifest_sha256": HRRR_BRIDGE_BUILD_MANIFEST_SHA256,
                "output_schema": HRRR_DECODED_WINDOW_SCHEMA,
                "gate_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "inventory_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "atmosphere_f02_fingerprint_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                "atmosphere_f03_fingerprint_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                "decoded_tree_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            },
            "payload": {
                "name": "SOILHGT.f32le",
                "nbytes": payload.len() as u64,
                "sha256": payload_sha256,
                "dtype": "little-endian binary32",
                "order": "south-to-north row-major",
                "shape": [HRRR_SURROUND_NY, HRRR_SURROUND_NX]
            },
            "grid": {
                "name": "HRRR Lambert GDT3.30",
                "full_shape_xy": [HRRR_FULL_NX, HRRR_FULL_NY],
                "window_origin_ij": [HRRR_SURROUND_WINDOW_I_START, HRRR_SURROUND_WINDOW_J_START],
                "window_shape_xy": [HRRR_SURROUND_NX, HRRR_SURROUND_NY],
                "spacing_m_xy": [3000, 3000],
                "scan_mode": "0x40"
            },
            "transform": {
                "method": "HRRR Lambert inverse -> spherical ECEF -> child ENU",
                "earth_radius_m_f64_bits": "41584de740000000",
                "child_center_lat_f64_bits": "4044bb15cda77904",
                "child_center_lon_f64_bits": "c0578c5ebb59704f",
                "output_axes_units": "east-km, up-km, north-km"
            },
            "overlay": {
                "exact_child_terrain_sha256": EXACT_CHILD_TERRAIN_SHA256,
                "exact_child_half_extent_km_f32_bits": format!("{:08x}", CHILD_SURFACE_HALF_EXTENT_KM.to_bits()),
                "ownership": HRRR_SURROUND_OWNERSHIP,
                "height_treatment": HRRR_SURROUND_HEIGHT_TREATMENT
            },
            "claim_limit": HRRR_SURROUND_CLAIM_LIMIT
        });
        let manifest_bytes = serde_json::to_vec(&manifest).expect("manifest fixture");
        let receipt = serde_json::json!({
            "schema": HRRR_SURROUND_RECEIPT_SCHEMA,
            "status": HRRR_SURROUND_STATUS,
            "manifest_sha256": sha256_bytes(&manifest_bytes),
            "payload_sha256": manifest["payload"]["sha256"],
            "exact_child_terrain_sha256": EXACT_CHILD_TERRAIN_SHA256,
            "no_extrapolation": true,
            "earth_curvature_retained": true,
            "claim_limit": HRRR_SURROUND_CLAIM_LIMIT
        });
        (
            manifest_bytes,
            serde_json::to_vec(&receipt).expect("receipt fixture"),
            payload,
        )
    }

    #[test]
    fn child_center_is_source_order_exact() {
        let (latitude, longitude) = child_center_lat_lon().expect("child center");
        assert_eq!(latitude.to_bits(), 0x4044_bb15_cda7_7904);
        assert_eq!(longitude.to_bits(), 0xc057_8c5e_bb59_704f);
    }

    #[test]
    fn admitted_window_has_curved_source_backed_horizon_context() {
        let spec = HrrrSurroundSpec::admitted();
        let (bytes, hash) = flat_fixture(spec);
        let (mesh, stats) = source_backed_surround_mesh(&bytes, &hash, spec).expect("mesh");
        assert!(mesh.count_vertices() > HRRR_SURROUND_NX * HRRR_SURROUND_NY);
        assert!(stats.indexed_triangle_count > 0);
        let horizon = horizon_distance_km(100.0).expect("100 m AGL horizon");
        assert!((35.6..35.8).contains(&horizon));
        assert!(stats.horizontal_margin_km() > horizon as f32);
        assert!(
            stats.minimum_up_km < -0.5,
            "Earth curvature must be retained"
        );
        assert!(stats.maximum_up_km < 0.5);

        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(values)) => values,
            other => panic!("unexpected positions: {other:?}"),
        };
        let indices = match mesh.indices() {
            Some(Indices::U32(values)) => values,
            other => panic!("unexpected indices: {other:?}"),
        };
        for triangle in indices.chunks_exact(3) {
            let centroid = triangle.iter().fold(Vec3::ZERO, |sum, index| {
                sum + Vec3::from_array(positions[*index as usize])
            }) / 3.0;
            assert!(
                centroid.x.abs() >= CHILD_SURFACE_HALF_EXTENT_KM - 1.0e-5
                    || centroid.z.abs() >= CHILD_SURFACE_HALF_EXTENT_KM - 1.0e-5,
                "context triangle entered exact-child ownership at {centroid:?}"
            );
        }
        let sampled = sample_mesh_height_km(&mesh, -12.0, -12.0).expect("exterior sample");
        assert!(sampled.is_finite());
        assert!(sample_mesh_height_km(&mesh, 0.0, 0.0).is_err());
    }

    #[test]
    fn corrupt_or_asymmetric_context_refuses() {
        let spec = HrrrSurroundSpec::admitted();
        let (mut bytes, hash) = flat_fixture(spec);
        bytes[0] ^= 1;
        assert!(source_backed_surround_mesh(&bytes, &hash, spec).is_err());

        let narrow = HrrrSurroundSpec {
            window_i_start: 971,
            window_j_start: 622,
            nx: 26,
            ny: 26,
        };
        let (bytes, hash) = flat_fixture(narrow);
        let error = source_backed_surround_mesh(&bytes, &hash, narrow).unwrap_err();
        assert!(error.contains("minimum context"));
    }

    #[test]
    fn strict_envelope_accepts_only_the_sealed_contract() {
        let (manifest, receipt, payload) = envelope_fixture();
        source_backed_surround_mesh_from_envelope(&manifest, &receipt, &payload)
            .expect("strict fixture envelope");

        let mut corrupt_payload = payload.clone();
        corrupt_payload[0] ^= 1;
        assert!(
            source_backed_surround_mesh_from_envelope(&manifest, &receipt, &corrupt_payload)
                .is_err()
        );

        let mut changed_manifest: serde_json::Value =
            serde_json::from_slice(&manifest).expect("fixture manifest");
        changed_manifest["grid"]["scan_mode"] = "0x00".into();
        let changed_manifest = serde_json::to_vec(&changed_manifest).unwrap();
        assert!(
            source_backed_surround_mesh_from_envelope(&changed_manifest, &receipt, &payload)
                .is_err()
        );
    }
}
