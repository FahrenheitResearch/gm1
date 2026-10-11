//! Rust-owned source-coordinate terrain producer for the accepted GM1 child.
//!
//! This emits only the real 300x300 `terrain_msl` carrier selected by the
//! validated T+300 parent plan.  It does not construct atmospheric child
//! fields, terrain/stretch metrics, hydrostatic state, or boundary histories.

use crate::child_remap::{
    parent_fractional_xy_for_grid, read_validated_parent_plan, validate_bound_parent_identity,
    ChildGridSpec, HorizontalStagger, ParentHorizontalGrid, ACCEPTED_50M_CHILD_GRID, CHILD_NX,
    CHILD_NY, CHILD_REMAP_PLAN_SCHEMA, CHILD_ZTOP_M, CM1_R21_AUTHORITY_SHA256, PARENT_NX,
    PARENT_NY,
};
use crate::gm1_checkpoint::{
    GM1_F02_ROI_PRODUCT_MANIFEST_SHA256, GM1_F02_ROI_PRODUCT_PAYLOAD_SHA256,
    GM1_F02_ROI_PRODUCT_SOURCE_ARTIFACT_SHA256, GM1_F02_ROI_PRODUCT_SOURCE_CONTRACT_SHA256,
};
use crate::json::JsonValue;
use crate::product_restart::read_initialized_product_restart;
use crate::sha256;
use crate::{Result, StateError};
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHILD_TERRAIN_SCHEMA: &str = "gm1.source-coordinate-50m-child-terrain/v1";
pub const CHILD_TERRAIN_MANIFEST_SCHEMA: &str =
    "gm1.source-coordinate-50m-child-terrain-manifest/v1";
pub const CHILD_TERRAIN_ABI: &str = "gm1.source-coordinate-50m-child-terrain-f32/v1";
pub const CHILD_TERRAIN_PAYLOAD_FILE: &str = "terrain-msl.f32le";
pub const CHILD_TERRAIN_MANIFEST_FILE: &str = "manifest.json";
pub const CHILD_TERRAIN_RECEIPT_FILE: &str = "receipt.json";
pub const CHILD_TERRAIN_STATUS: &str = "PASS_REAL_SOURCE_COORDINATE_TERRAIN";
pub const CHILD_TERRAIN_CLAIM_LIMIT: &str = "one Rust-owned, source-coordinate bilinear remap of the exact real f02 ROI terrain_msl carrier onto the accepted fixed 300x300 50 m child mass grid; the payload is bound to the pdefq-clean T+300 parent plan and exact CM1IPR1/product/source identities, with no synthetic translation; terrain/stretch composition, atmospheric remap, hydrostatic/base reconstruction, child boundary initialization, restart identity, trajectory, forecast skill, tornado, and performance remain unclaimed";
pub const CHILD_TERRAIN_GRID_SCHEMA: &str = "gm1.source-coordinate-child-terrain/v2";
pub const CHILD_TERRAIN_GRID_MANIFEST_SCHEMA: &str =
    "gm1.source-coordinate-child-terrain-manifest/v2";
pub const CHILD_TERRAIN_GRID_ABI: &str = "gm1.source-coordinate-child-terrain-f32/v2";
pub const CHILD_TERRAIN_GRID_STATUS: &str = "PASS_REAL_SOURCE_COORDINATE_CHILD_TERRAIN";
pub const CHILD_TERRAIN_GRID_CLAIM_LIMIT: &str = "one Rust-owned, source-coordinate bilinear remap of the exact real f02 ROI terrain_msl carrier onto an explicitly sealed 15 km child mass grid; every output word is provenance-backed interpolation of the admitted source carrier, with no synthetic detail or translation; this terrain-only artifact does not admit static metrics, atmospheric remap, hydrostatic/base reconstruction, child boundary initialization, closure, checkpoint state, trajectory, forecast skill, tornado, or performance";

pub const F02_ROI_PRODUCT_RESTART_SHA256: &str =
    "9efe09771763a3f727af715f7dd41a2c6256cd5613aa2f31637cc08c9721ed6e";
pub const F02_ROI_PRODUCT_RECEIPT_SHA256: &str =
    "379b989884e773a942ade62069e99ce5e032374c2ae2ef13679f32a524fbbdcd";
pub const F02_ROI_SOURCE_TERRAIN_SHA256: &str =
    "80a2ce41cc02e7f247a1852ebbeba50a8058e71d9b2426e0b3312a416bc59a19";

const PAYLOAD_NBYTES: usize = CHILD_NX * CHILD_NY * 4;
const MAX_JSON_BYTES: u64 = 256 * 1024;
const MAX_SOURCE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_EXECUTABLE_BYTES: u64 = 64 * 1024 * 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildTerrainReceipt {
    pub parent_checkpoint_sha256: String,
    pub parent_plan_sha256: String,
    pub manifest_sha256: String,
    pub payload_sha256: String,
    pub receipt_sha256: String,
    pub producer_source_sha256: String,
    pub producer_executable_sha256: String,
    pub minimum_f32_bits: u32,
    pub maximum_f32_bits: u32,
    pub nonzero_count: usize,
}

impl ChildTerrainReceipt {
    pub fn canonical_json(&self) -> String {
        String::from_utf8(receipt_value(self).canonical_bytes())
            .expect("canonical child-terrain receipt is UTF-8")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChildTerrainGridReceipt {
    pub grid: ChildGridSpec,
    pub parent_checkpoint_sha256: String,
    pub parent_plan_sha256: String,
    pub manifest_sha256: String,
    pub payload_sha256: String,
    pub receipt_sha256: String,
    pub producer_source_sha256: String,
    pub producer_executable_sha256: String,
    pub minimum_f32_bits: u32,
    pub maximum_f32_bits: u32,
    pub nonzero_count: usize,
}

impl ChildTerrainGridReceipt {
    pub fn canonical_json(&self) -> String {
        String::from_utf8(grid_receipt_value(self).canonical_bytes())
            .expect("canonical parameterized child-terrain receipt is UTF-8")
    }
}

#[derive(Clone, Debug)]
struct TerrainTransform {
    payload: Vec<u8>,
    payload_sha256: String,
    minimum_f32_bits: u32,
    maximum_f32_bits: u32,
    nonzero_count: usize,
}

/// Publish the exact real child terrain as one atomic three-file directory.
pub fn publish_child_terrain_new(
    validated_parent_plan: impl AsRef<Path>,
    source_product_restart: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
) -> Result<ChildTerrainReceipt> {
    let (binding, parent_plan_sha256) = read_validated_parent_plan(validated_parent_plan.as_ref())?;
    validate_bound_parent_identity(
        &binding.checkpoint_artifact_sha256,
        &parent_plan_sha256,
        "child terrain",
    )?;
    let producer_source_sha256 = hash_regular_bounded(
        producer_source.as_ref(),
        MAX_SOURCE_BYTES,
        "child-terrain producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable.as_ref(),
        MAX_EXECUTABLE_BYTES,
        "child-terrain producer executable",
    )?;
    let transform = build_from_source_restart(source_product_restart.as_ref())?;
    let manifest = manifest_value(
        &transform,
        &binding.checkpoint_artifact_sha256,
        &parent_plan_sha256,
        &producer_source_sha256,
        &producer_executable_sha256,
    );
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let mut receipt = ChildTerrainReceipt {
        parent_checkpoint_sha256: binding.checkpoint_artifact_sha256.clone(),
        parent_plan_sha256: parent_plan_sha256.clone(),
        manifest_sha256,
        payload_sha256: transform.payload_sha256.clone(),
        receipt_sha256: String::new(),
        producer_source_sha256,
        producer_executable_sha256,
        minimum_f32_bits: transform.minimum_f32_bits,
        maximum_f32_bits: transform.maximum_f32_bits,
        nonzero_count: transform.nonzero_count,
    };
    let receipt_bytes = receipt_value(&receipt).canonical_bytes();
    receipt.receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    publish_directory(
        output_directory.as_ref(),
        &transform.payload,
        &manifest_bytes,
        &receipt_bytes,
    )?;
    Ok(receipt)
}

/// Publish one exact real child terrain for an explicitly validated 15 km grid.
///
/// The historical [`publish_child_terrain_new`] entry point and its v1 JSON
/// bytes remain fixed to the accepted 300x300/50 m artifact.  This v2 entry
/// point carries the grid identity in every manifest and receipt so a 5 m
/// terrain can never be mistaken for the historical 50 m product.
pub fn publish_child_terrain_for_grid_new(
    grid: ChildGridSpec,
    validated_parent_plan: impl AsRef<Path>,
    source_product_restart: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
) -> Result<ChildTerrainGridReceipt> {
    let (binding, parent_plan_sha256) = read_validated_parent_plan(validated_parent_plan.as_ref())?;
    validate_bound_parent_identity(
        &binding.checkpoint_artifact_sha256,
        &parent_plan_sha256,
        "child terrain",
    )?;
    let producer_source_sha256 = hash_regular_bounded(
        producer_source.as_ref(),
        MAX_SOURCE_BYTES,
        "parameterized child-terrain producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable.as_ref(),
        MAX_EXECUTABLE_BYTES,
        "parameterized child-terrain producer executable",
    )?;
    let transform = build_from_source_restart_for_grid(source_product_restart.as_ref(), grid)?;
    let manifest = grid_manifest_value(
        grid,
        &transform,
        &binding.checkpoint_artifact_sha256,
        &parent_plan_sha256,
        &producer_source_sha256,
        &producer_executable_sha256,
    );
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let mut receipt = ChildTerrainGridReceipt {
        grid,
        parent_checkpoint_sha256: binding.checkpoint_artifact_sha256.clone(),
        parent_plan_sha256: parent_plan_sha256.clone(),
        manifest_sha256,
        payload_sha256: transform.payload_sha256.clone(),
        receipt_sha256: String::new(),
        producer_source_sha256,
        producer_executable_sha256,
        minimum_f32_bits: transform.minimum_f32_bits,
        maximum_f32_bits: transform.maximum_f32_bits,
        nonzero_count: transform.nonzero_count,
    };
    let receipt_bytes = grid_receipt_value(&receipt).canonical_bytes();
    receipt.receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    publish_directory(
        output_directory.as_ref(),
        &transform.payload,
        &manifest_bytes,
        &receipt_bytes,
    )?;
    Ok(receipt)
}

fn build_from_source_restart(path: &Path) -> Result<TerrainTransform> {
    build_from_source_restart_for_grid(path, ACCEPTED_50M_CHILD_GRID)
}

fn build_from_source_restart_for_grid(
    path: &Path,
    grid: ChildGridSpec,
) -> Result<TerrainTransform> {
    let source = read_initialized_product_restart(
        path,
        F02_ROI_PRODUCT_RESTART_SHA256,
        GM1_F02_ROI_PRODUCT_MANIFEST_SHA256,
        GM1_F02_ROI_PRODUCT_PAYLOAD_SHA256,
        F02_ROI_PRODUCT_RECEIPT_SHA256,
        GM1_F02_ROI_PRODUCT_SOURCE_ARTIFACT_SHA256,
        GM1_F02_ROI_PRODUCT_SOURCE_CONTRACT_SHA256,
    )?;
    // The parent this child window is placed against is read out of the
    // product itself - its recorded source extent and spacing, cross-checked
    // against the shape of the terrain carrier that is actually present - and
    // the window is then required to have been placed against exactly that
    // grid.  Without this the window's index mapping could divide by one
    // parent's spacing while striding another parent's rows.
    let declared = source.source_grid()?;
    let parent = ParentHorizontalGrid::new(
        declared.shape_zyx[2],
        declared.shape_zyx[1],
        declared.dx_m,
        declared.dy_m,
    )?;
    let carrier = source
        .carrier("terrain_msl")
        .ok_or_else(|| invalid("source initialized product lacks terrain_msl"))?;
    let parent_words = parent
        .nx()
        .checked_mul(parent.ny())
        .ok_or_else(|| invalid("parent surface word count overflows"))?;
    if carrier.shape != [parent.ny(), parent.nx()] || carrier.nbytes != (parent_words * 4) as u64 {
        return Err(invalid(format!(
            "source terrain carrier is {:?}, but the product records a {}x{} source domain; the surface and the domain it is supposed to cover are different shapes",
            carrier.shape,
            parent.nx(),
            parent.ny()
        )));
    }
    if grid.parent_grid() != parent {
        return Err(invalid(format!(
            "this child window was placed against a {}x{} parent at {} m x {} m, but the source product records a {}x{} parent at {} m x {} m; every child column would be interpolated from the wrong place in the parent domain",
            grid.parent_nx(),
            grid.parent_ny(),
            grid.parent_dx_m(),
            grid.parent_dy_m(),
            parent.nx(),
            parent.ny(),
            parent.dx_m(),
            parent.dy_m()
        )));
    }
    let source_bytes = source.read_carrier_f32le_exact("terrain_msl")?;
    if sha256::digest_hex(&source_bytes) != carrier.sha256 {
        return Err(invalid(
            "source terrain bytes differ from the digest the product's own manifest records",
        ));
    }
    let mut source_words = Vec::with_capacity(parent_words);
    for word in source_bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(word.try_into().expect("four-byte f32 chunk"));
        if !value.is_finite() || f64::from(value) >= CHILD_ZTOP_M {
            return Err(invalid(
                "source terrain contains a non-finite value or reaches child model top",
            ));
        }
        source_words.push(value);
    }
    transform_source_terrain_for_grid(&source_words, grid)
}

#[cfg(test)]
fn transform_source_terrain(source: &[f32]) -> Result<TerrainTransform> {
    transform_source_terrain_for_grid(source, ACCEPTED_50M_CHILD_GRID)
}

fn transform_source_terrain_for_grid(
    source: &[f32],
    grid: ChildGridSpec,
) -> Result<TerrainTransform> {
    // The child window carries the parent grid it was placed against, so the
    // surface it is remapped from must be that parent's surface.  A source of
    // another extent would be indexed with this parent's row stride and every
    // child point would sample a column from somewhere else in the domain.
    let expected = grid
        .parent_nx()
        .checked_mul(grid.parent_ny())
        .ok_or_else(|| invalid("parent surface word count overflows"))?;
    if source.len() != expected {
        return Err(invalid(format!(
            "source terrain has {} words, but this child window was placed against a {}x{} parent and would index it with that row stride; every child column would be interpolated from a different place in the domain",
            source.len(),
            grid.parent_nx(),
            grid.parent_ny()
        )));
    }
    let payload_nbytes = grid
        .nx()
        .checked_mul(grid.ny())
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| invalid("child terrain payload byte count overflows usize"))?;
    let mut payload = Vec::with_capacity(payload_nbytes);
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut nonzero_count = 0usize;
    for child_j in 0..grid.ny() {
        for child_i in 0..grid.nx() {
            let value = interpolate_source_terrain_point(source, grid, child_i, child_j)?;
            if !value.is_finite() || f64::from(value) >= CHILD_ZTOP_M {
                return Err(invalid(
                    "child terrain remap produced a non-finite value or reached model top",
                ));
            }
            minimum = minimum.min(value);
            maximum = maximum.max(value);
            nonzero_count += usize::from(value.to_bits() & 0x7fff_ffff != 0);
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    if payload.len() != payload_nbytes {
        return Err(invalid("child terrain payload byte count drifted"));
    }
    Ok(TerrainTransform {
        payload_sha256: sha256::digest_hex(&payload),
        payload,
        minimum_f32_bits: minimum.to_bits(),
        maximum_f32_bits: maximum.to_bits(),
        nonzero_count,
    })
}

fn interpolate_source_terrain_point(
    source: &[f32],
    grid: ChildGridSpec,
    child_i: usize,
    child_j: usize,
) -> Result<f32> {
    let parent_nx = grid.parent_nx();
    let parent_ny = grid.parent_ny();
    if source.len()
        != parent_nx
            .checked_mul(parent_ny)
            .ok_or_else(|| invalid("parent surface word count overflows"))?
    {
        return Err(invalid(format!(
            "source terrain is not the {parent_nx}x{parent_ny} surface this child window was placed against"
        )));
    }
    let (source_i, source_j) =
        parent_fractional_xy_for_grid(grid, HorizontalStagger::Surface, child_i, child_j)?;
    if !source_i.is_finite() || !source_j.is_finite() || source_i < 0.0 || source_j < 0.0 {
        return Err(invalid(
            "child terrain mapping escaped the source-coordinate bilinear stencil",
        ));
    }
    let i0 = source_i.floor() as usize;
    let j0 = source_j.floor() as usize;
    let fx = source_i - i0 as f64;
    let fy = source_j - j0 as f64;
    if i0 + 1 >= parent_nx
        || j0 + 1 >= parent_ny
        || !(0.0..1.0).contains(&fx)
        || !(0.0..1.0).contains(&fy)
    {
        return Err(invalid(
            "child terrain mapping escaped the source-coordinate bilinear stencil",
        ));
    }
    let v00 = f64::from(source[j0 * parent_nx + i0]);
    let v10 = f64::from(source[j0 * parent_nx + i0 + 1]);
    let v01 = f64::from(source[(j0 + 1) * parent_nx + i0]);
    let v11 = f64::from(source[(j0 + 1) * parent_nx + i0 + 1]);
    let row0 = v00 * (1.0 - fx) + v10 * fx;
    let row1 = v01 * (1.0 - fx) + v11 * fx;
    Ok((row0 * (1.0 - fy) + row1 * fy) as f32)
}

fn manifest_value(
    transform: &TerrainTransform,
    parent_checkpoint_sha256: &str,
    parent_plan_sha256: &str,
    producer_source_sha256: &str,
    producer_executable_sha256: &str,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_TERRAIN_ABI)),
        ("artifact_schema", string(CHILD_TERRAIN_SCHEMA)),
        ("authority", authority_value()),
        ("claim_limit", string(CHILD_TERRAIN_CLAIM_LIMIT)),
        ("dtype", string("little-endian-f32")),
        ("itemsize_bytes", number(4)),
        ("model", string("GM1")),
        ("order", string("C")),
        (
            "parent",
            object(vec![
                ("checkpoint_sha256", string(parent_checkpoint_sha256)),
                ("plan_schema", string(CHILD_REMAP_PLAN_SCHEMA)),
                ("plan_sha256", string(parent_plan_sha256)),
            ]),
        ),
        (
            "payload",
            object(vec![
                ("file", string(CHILD_TERRAIN_PAYLOAD_FILE)),
                ("nbytes", number(PAYLOAD_NBYTES)),
                ("sha256", string(&transform.payload_sha256)),
                ("shape_yx", number_array(&[CHILD_NY, CHILD_NX])),
            ]),
        ),
        (
            "producer",
            object(vec![
                ("executable_sha256", string(producer_executable_sha256)),
                ("source_sha256", string(producer_source_sha256)),
            ]),
        ),
        ("schema", string(CHILD_TERRAIN_MANIFEST_SCHEMA)),
        ("source", source_identity_value()),
        (
            "statistics",
            object(vec![
                ("all_finite", JsonValue::Bool(true)),
                (
                    "maximum_f32_bits",
                    string(&format!("{:08x}", transform.maximum_f32_bits)),
                ),
                (
                    "minimum_f32_bits",
                    string(&format!("{:08x}", transform.minimum_f32_bits)),
                ),
                ("nonzero_count", number(transform.nonzero_count)),
            ]),
        ),
        ("status", string(CHILD_TERRAIN_STATUS)),
        ("synthetic_translation", JsonValue::Bool(false)),
        ("transform", transform_value()),
    ])
}

fn receipt_value(receipt: &ChildTerrainReceipt) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_TERRAIN_ABI)),
        ("all_finite", JsonValue::Bool(true)),
        ("authority", authority_value()),
        ("claim_limit", string(CHILD_TERRAIN_CLAIM_LIMIT)),
        ("dtype", string("little-endian-f32")),
        ("manifest_sha256", string(&receipt.manifest_sha256)),
        (
            "maximum_f32_bits",
            string(&format!("{:08x}", receipt.maximum_f32_bits)),
        ),
        (
            "minimum_f32_bits",
            string(&format!("{:08x}", receipt.minimum_f32_bits)),
        ),
        ("model", string("GM1")),
        ("nonzero_count", number(receipt.nonzero_count)),
        ("order", string("C")),
        (
            "parent_checkpoint_sha256",
            string(&receipt.parent_checkpoint_sha256),
        ),
        ("parent_plan_sha256", string(&receipt.parent_plan_sha256)),
        ("payload_nbytes", number(PAYLOAD_NBYTES)),
        ("payload_sha256", string(&receipt.payload_sha256)),
        (
            "producer_executable_sha256",
            string(&receipt.producer_executable_sha256),
        ),
        (
            "producer_source_sha256",
            string(&receipt.producer_source_sha256),
        ),
        ("schema", string(CHILD_TERRAIN_SCHEMA)),
        ("shape_yx", number_array(&[CHILD_NY, CHILD_NX])),
        ("source", source_identity_value()),
        (
            "source_coordinate_child_terrain_sha256",
            string(&receipt.payload_sha256),
        ),
        ("status", string(CHILD_TERRAIN_STATUS)),
        ("synthetic_translation", JsonValue::Bool(false)),
        ("transform", transform_value()),
    ])
}

fn grid_manifest_value(
    grid: ChildGridSpec,
    transform: &TerrainTransform,
    parent_checkpoint_sha256: &str,
    parent_plan_sha256: &str,
    producer_source_sha256: &str,
    producer_executable_sha256: &str,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_TERRAIN_GRID_ABI)),
        ("artifact_schema", string(CHILD_TERRAIN_GRID_SCHEMA)),
        ("authority", authority_value()),
        ("claim_limit", string(CHILD_TERRAIN_GRID_CLAIM_LIMIT)),
        ("dtype", string("little-endian-f32")),
        ("grid", grid_identity_value(grid)),
        ("itemsize_bytes", number(4)),
        ("model", string("GM1")),
        ("order", string("C")),
        (
            "parent_source_binding",
            object(vec![
                ("checkpoint_sha256", string(parent_checkpoint_sha256)),
                ("plan_schema", string(CHILD_REMAP_PLAN_SCHEMA)),
                ("plan_sha256", string(parent_plan_sha256)),
                (
                    "scope",
                    string(
                        "the accepted plan binds source state and georeferencing; the v2 grid block independently binds output geometry",
                    ),
                ),
            ]),
        ),
        (
            "payload",
            object(vec![
                ("file", string(CHILD_TERRAIN_PAYLOAD_FILE)),
                ("nbytes", number(transform.payload.len())),
                ("sha256", string(&transform.payload_sha256)),
                ("shape_yx", number_array(&[grid.ny(), grid.nx()])),
            ]),
        ),
        (
            "producer",
            object(vec![
                ("executable_sha256", string(producer_executable_sha256)),
                ("source_sha256", string(producer_source_sha256)),
            ]),
        ),
        ("schema", string(CHILD_TERRAIN_GRID_MANIFEST_SCHEMA)),
        ("source", source_identity_value()),
        ("source_backed_interpolation", JsonValue::Bool(true)),
        (
            "statistics",
            object(vec![
                ("all_finite", JsonValue::Bool(true)),
                (
                    "maximum_f32_bits",
                    string(&format!("{:08x}", transform.maximum_f32_bits)),
                ),
                (
                    "minimum_f32_bits",
                    string(&format!("{:08x}", transform.minimum_f32_bits)),
                ),
                ("nonzero_count", number(transform.nonzero_count)),
            ]),
        ),
        ("status", string(CHILD_TERRAIN_GRID_STATUS)),
        ("synthetic_detail", JsonValue::Bool(false)),
        ("synthetic_translation", JsonValue::Bool(false)),
        ("transform", grid_transform_value(grid)),
    ])
}

fn grid_receipt_value(receipt: &ChildTerrainGridReceipt) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_TERRAIN_GRID_ABI)),
        ("all_finite", JsonValue::Bool(true)),
        ("authority", authority_value()),
        ("claim_limit", string(CHILD_TERRAIN_GRID_CLAIM_LIMIT)),
        ("dtype", string("little-endian-f32")),
        ("grid", grid_identity_value(receipt.grid)),
        ("manifest_sha256", string(&receipt.manifest_sha256)),
        (
            "maximum_f32_bits",
            string(&format!("{:08x}", receipt.maximum_f32_bits)),
        ),
        (
            "minimum_f32_bits",
            string(&format!("{:08x}", receipt.minimum_f32_bits)),
        ),
        ("model", string("GM1")),
        ("nonzero_count", number(receipt.nonzero_count)),
        ("order", string("C")),
        (
            "parent_checkpoint_sha256",
            string(&receipt.parent_checkpoint_sha256),
        ),
        ("parent_plan_sha256", string(&receipt.parent_plan_sha256)),
        (
            "payload_nbytes",
            number(receipt.grid.nx() * receipt.grid.ny() * 4),
        ),
        ("payload_sha256", string(&receipt.payload_sha256)),
        (
            "producer_executable_sha256",
            string(&receipt.producer_executable_sha256),
        ),
        (
            "producer_source_sha256",
            string(&receipt.producer_source_sha256),
        ),
        ("schema", string(CHILD_TERRAIN_GRID_SCHEMA)),
        (
            "shape_yx",
            number_array(&[receipt.grid.ny(), receipt.grid.nx()]),
        ),
        ("source", source_identity_value()),
        ("source_backed_interpolation", JsonValue::Bool(true)),
        (
            "source_coordinate_child_terrain_sha256",
            string(&receipt.payload_sha256),
        ),
        ("status", string(CHILD_TERRAIN_GRID_STATUS)),
        ("synthetic_detail", JsonValue::Bool(false)),
        ("synthetic_translation", JsonValue::Bool(false)),
        ("transform", grid_transform_value(receipt.grid)),
    ])
}

fn grid_identity_value(grid: ChildGridSpec) -> JsonValue {
    object(vec![
        ("abi", string(crate::child_remap::CHILD_GRID_SPEC_ABI)),
        ("domain_x_m_f64_bits", f64_bits(grid.extent_x_m())),
        ("domain_y_m_f64_bits", f64_bits(grid.extent_y_m())),
        ("dx_m_f64_bits", f64_bits(grid.dx_m())),
        ("dy_m_f64_bits", f64_bits(grid.dy_m())),
        ("nx", number(grid.nx())),
        ("ny", number(grid.ny())),
        ("nz", number(grid.nz())),
        ("ztop_m_f64_bits", f64_bits(grid.ztop_m())),
    ])
}

fn grid_transform_value(grid: ChildGridSpec) -> JsonValue {
    let (first_i, first_j) = parent_fractional_xy_for_grid(grid, HorizontalStagger::Surface, 0, 0)
        .expect("validated child grid has a first source coordinate");
    let (last_i, last_j) = parent_fractional_xy_for_grid(
        grid,
        HorizontalStagger::Surface,
        grid.nx() - 1,
        grid.ny() - 1,
    )
    .expect("validated child grid has a last source coordinate");
    object(vec![
        (
            "formula",
            string("row0=f64(v00)*(1-fx)+f64(v10)*fx; row1=f64(v01)*(1-fx)+f64(v11)*fx; child=f32(row0*(1-fy)+row1*fy)"),
        ),
        (
            "fractional_coordinates",
            string("parent_i=224+(child_i+0.5-0.5*nx)*dx_m/100; parent_j=224+(child_j+0.5-0.5*ny)*dy_m/100"),
        ),
        ("interpolation", string("coordinate-aware-bilinear")),
        (
            "source_coordinate_extent_f64_bits",
            object(vec![
                ("first_i", f64_bits(first_i)),
                ("first_j", f64_bits(first_j)),
                ("last_i", f64_bits(last_i)),
                ("last_j", f64_bits(last_j)),
            ]),
        ),
        (
            "rounding",
            string("binary64 weighted sums in the written statement order, then one round-to-nearest-ties-to-even binary32 store"),
        ),
    ])
}

fn authority_value() -> JsonValue {
    object(vec![
        ("model", string("GM1")),
        ("model_source", string("CM1 r21.1")),
        ("tarball_sha256", string(CM1_R21_AUTHORITY_SHA256)),
    ])
}

fn source_identity_value() -> JsonValue {
    object(vec![
        (
            "initialized_product_manifest_sha256",
            string(GM1_F02_ROI_PRODUCT_MANIFEST_SHA256),
        ),
        (
            "initialized_product_payload_sha256",
            string(GM1_F02_ROI_PRODUCT_PAYLOAD_SHA256),
        ),
        (
            "initialized_product_receipt_sha256",
            string(F02_ROI_PRODUCT_RECEIPT_SHA256),
        ),
        (
            "initialized_product_restart_sha256",
            string(F02_ROI_PRODUCT_RESTART_SHA256),
        ),
        (
            "source_artifact_sha256",
            string(GM1_F02_ROI_PRODUCT_SOURCE_ARTIFACT_SHA256),
        ),
        (
            "source_contract_sha256",
            string(GM1_F02_ROI_PRODUCT_SOURCE_CONTRACT_SHA256),
        ),
        ("source_kind", string("boundary-analysis-history-f02-roi")),
        (
            "terrain_carrier",
            object(vec![
                ("name", string("terrain_msl")),
                ("sha256", string(F02_ROI_SOURCE_TERRAIN_SHA256)),
                ("shape_yx", number_array(&[PARENT_NY, PARENT_NX])),
            ]),
        ),
    ])
}

fn transform_value() -> JsonValue {
    object(vec![
        (
            "formula",
            string("row0=f64(v00)*(1-fx)+f64(v10)*fx; row1=f64(v01)*(1-fx)+f64(v11)*fx; child=f32(row0*(1-fy)+row1*fy)"),
        ),
        (
            "fractional_coordinates",
            string("parent_i=149.25+0.5*child_i; parent_j=149.25+0.5*child_j; fx and fy are exactly 0.25 or 0.75"),
        ),
        ("interpolation", string("coordinate-aware-bilinear")),
        (
            "required_parent_i",
            string("149..299 inclusive bilinear stencil"),
        ),
        (
            "required_parent_j",
            string("149..299 inclusive bilinear stencil"),
        ),
        (
            "rounding",
            string("binary64 weighted sums in the written statement order, then one round-to-nearest-ties-to-even binary32 store"),
        ),
    ])
}

/// Recompute and verify every byte of a published child-terrain artifact.
#[allow(clippy::too_many_arguments)]
pub fn read_child_terrain_artifact(
    validated_parent_plan: impl AsRef<Path>,
    source_product_restart: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_payload_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<ChildTerrainReceipt> {
    for (value, label) in [
        (
            expected_manifest_sha256,
            "expected child-terrain manifest SHA-256",
        ),
        (
            expected_payload_sha256,
            "expected child-terrain payload SHA-256",
        ),
        (
            expected_receipt_sha256,
            "expected child-terrain receipt SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    let (binding, parent_plan_sha256) = read_validated_parent_plan(validated_parent_plan.as_ref())?;
    validate_bound_parent_identity(
        &binding.checkpoint_artifact_sha256,
        &parent_plan_sha256,
        "child terrain",
    )?;
    let producer_source_sha256 = hash_regular_bounded(
        producer_source.as_ref(),
        MAX_SOURCE_BYTES,
        "child-terrain producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable.as_ref(),
        MAX_EXECUTABLE_BYTES,
        "child-terrain producer executable",
    )?;
    let transform = build_from_source_restart(source_product_restart.as_ref())?;
    let directory = directory.as_ref();
    verify_directory_inventory(directory)?;
    let manifest_bytes = read_regular_bounded(
        &directory.join(CHILD_TERRAIN_MANIFEST_FILE),
        MAX_JSON_BYTES,
        "child-terrain manifest",
    )?;
    let payload_bytes = read_regular_bounded(
        &directory.join(CHILD_TERRAIN_PAYLOAD_FILE),
        PAYLOAD_NBYTES as u64,
        "child-terrain payload",
    )?;
    let receipt_bytes = read_regular_bounded(
        &directory.join(CHILD_TERRAIN_RECEIPT_FILE),
        MAX_JSON_BYTES,
        "child-terrain receipt",
    )?;
    let actual_manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let actual_payload_sha256 = sha256::digest_hex(&payload_bytes);
    let actual_receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    for (actual, expected, label) in [
        (
            actual_manifest_sha256.as_str(),
            expected_manifest_sha256,
            "child-terrain manifest",
        ),
        (
            actual_payload_sha256.as_str(),
            expected_payload_sha256,
            "child-terrain payload",
        ),
        (
            actual_receipt_sha256.as_str(),
            expected_receipt_sha256,
            "child-terrain receipt",
        ),
    ] {
        if actual != expected {
            return Err(invalid(format!(
                "{label} SHA-256 is {actual}; independently expected {expected}"
            )));
        }
    }
    if payload_bytes != transform.payload {
        return Err(invalid(
            "published child terrain differs from the exact Rust source-coordinate transform",
        ));
    }
    let expected_manifest = manifest_value(
        &transform,
        &binding.checkpoint_artifact_sha256,
        &parent_plan_sha256,
        &producer_source_sha256,
        &producer_executable_sha256,
    )
    .canonical_bytes();
    if manifest_bytes != expected_manifest {
        return Err(invalid(
            "child-terrain manifest differs from the exact source-bound contract",
        ));
    }
    let mut receipt = ChildTerrainReceipt {
        parent_checkpoint_sha256: binding.checkpoint_artifact_sha256.clone(),
        parent_plan_sha256: parent_plan_sha256.clone(),
        manifest_sha256: actual_manifest_sha256,
        payload_sha256: actual_payload_sha256,
        receipt_sha256: String::new(),
        producer_source_sha256,
        producer_executable_sha256,
        minimum_f32_bits: transform.minimum_f32_bits,
        maximum_f32_bits: transform.maximum_f32_bits,
        nonzero_count: transform.nonzero_count,
    };
    if receipt_bytes != receipt_value(&receipt).canonical_bytes() {
        return Err(invalid(
            "child-terrain receipt differs from the exact source-bound contract",
        ));
    }
    receipt.receipt_sha256 = actual_receipt_sha256;
    Ok(receipt)
}

/// Recompute and verify every byte of a parameterized child-terrain artifact.
#[allow(clippy::too_many_arguments)]
pub fn read_child_terrain_for_grid_artifact(
    grid: ChildGridSpec,
    validated_parent_plan: impl AsRef<Path>,
    source_product_restart: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_payload_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<ChildTerrainGridReceipt> {
    for (value, label) in [
        (
            expected_manifest_sha256,
            "expected parameterized child-terrain manifest SHA-256",
        ),
        (
            expected_payload_sha256,
            "expected parameterized child-terrain payload SHA-256",
        ),
        (
            expected_receipt_sha256,
            "expected parameterized child-terrain receipt SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    let (binding, parent_plan_sha256) = read_validated_parent_plan(validated_parent_plan.as_ref())?;
    validate_bound_parent_identity(
        &binding.checkpoint_artifact_sha256,
        &parent_plan_sha256,
        "child terrain",
    )?;
    let producer_source_sha256 = hash_regular_bounded(
        producer_source.as_ref(),
        MAX_SOURCE_BYTES,
        "parameterized child-terrain producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable.as_ref(),
        MAX_EXECUTABLE_BYTES,
        "parameterized child-terrain producer executable",
    )?;
    let transform = build_from_source_restart_for_grid(source_product_restart.as_ref(), grid)?;
    let directory = directory.as_ref();
    verify_directory_inventory(directory)?;
    let manifest_bytes = read_regular_bounded(
        &directory.join(CHILD_TERRAIN_MANIFEST_FILE),
        MAX_JSON_BYTES,
        "parameterized child-terrain manifest",
    )?;
    let payload_bytes = read_regular_bounded(
        &directory.join(CHILD_TERRAIN_PAYLOAD_FILE),
        transform.payload.len() as u64,
        "parameterized child-terrain payload",
    )?;
    let receipt_bytes = read_regular_bounded(
        &directory.join(CHILD_TERRAIN_RECEIPT_FILE),
        MAX_JSON_BYTES,
        "parameterized child-terrain receipt",
    )?;
    let actual_manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let actual_payload_sha256 = sha256::digest_hex(&payload_bytes);
    let actual_receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    for (actual, expected, label) in [
        (
            actual_manifest_sha256.as_str(),
            expected_manifest_sha256,
            "parameterized child-terrain manifest",
        ),
        (
            actual_payload_sha256.as_str(),
            expected_payload_sha256,
            "parameterized child-terrain payload",
        ),
        (
            actual_receipt_sha256.as_str(),
            expected_receipt_sha256,
            "parameterized child-terrain receipt",
        ),
    ] {
        if actual != expected {
            return Err(invalid(format!(
                "{label} SHA-256 is {actual}; independently expected {expected}"
            )));
        }
    }
    if payload_bytes != transform.payload {
        return Err(invalid(
            "published parameterized child terrain differs from the exact Rust source-coordinate transform",
        ));
    }
    let expected_manifest = grid_manifest_value(
        grid,
        &transform,
        &binding.checkpoint_artifact_sha256,
        &parent_plan_sha256,
        &producer_source_sha256,
        &producer_executable_sha256,
    )
    .canonical_bytes();
    if manifest_bytes != expected_manifest {
        return Err(invalid(
            "parameterized child-terrain manifest differs from the exact source-bound contract",
        ));
    }
    let mut receipt = ChildTerrainGridReceipt {
        grid,
        parent_checkpoint_sha256: binding.checkpoint_artifact_sha256.clone(),
        parent_plan_sha256: parent_plan_sha256.clone(),
        manifest_sha256: actual_manifest_sha256,
        payload_sha256: actual_payload_sha256,
        receipt_sha256: String::new(),
        producer_source_sha256,
        producer_executable_sha256,
        minimum_f32_bits: transform.minimum_f32_bits,
        maximum_f32_bits: transform.maximum_f32_bits,
        nonzero_count: transform.nonzero_count,
    };
    if receipt_bytes != grid_receipt_value(&receipt).canonical_bytes() {
        return Err(invalid(
            "parameterized child-terrain receipt differs from the exact source-bound contract",
        ));
    }
    receipt.receipt_sha256 = actual_receipt_sha256;
    Ok(receipt)
}

fn publish_directory(output: &Path, payload: &[u8], manifest: &[u8], receipt: &[u8]) -> Result<()> {
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace child-terrain destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "child-terrain destination parent does not exist: {}",
            parent.display()
        )));
    }
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("child-terrain destination needs a UTF-8 directory name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let staging =
        output.with_file_name(format!(".{name}.partial-{}-{sequence}", std::process::id()));
    fs::create_dir(&staging)?;
    let result = (|| -> Result<()> {
        write_new_synced(&staging.join(CHILD_TERRAIN_PAYLOAD_FILE), payload)?;
        write_new_synced(&staging.join(CHILD_TERRAIN_MANIFEST_FILE), manifest)?;
        write_new_synced(&staging.join(CHILD_TERRAIN_RECEIPT_FILE), receipt)?;
        verify_directory_inventory(&staging)?;
        fs::rename(&staging, output)?;
        Ok(())
    })();
    if result.is_err() {
        for file in [
            CHILD_TERRAIN_PAYLOAD_FILE,
            CHILD_TERRAIN_MANIFEST_FILE,
            CHILD_TERRAIN_RECEIPT_FILE,
        ] {
            let _ = fs::remove_file(staging.join(file));
        }
        let _ = fs::remove_dir(&staging);
    }
    result
}

fn verify_directory_inventory(directory: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(directory).map_err(|error| {
        invalid(format!(
            "could not inspect child-terrain directory {}: {error}",
            directory.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!(
            "child-terrain input is not a real directory: {}",
            directory.display()
        )));
    }
    let expected = BTreeSet::from([
        CHILD_TERRAIN_MANIFEST_FILE.to_owned(),
        CHILD_TERRAIN_PAYLOAD_FILE.to_owned(),
        CHILD_TERRAIN_RECEIPT_FILE.to_owned(),
    ]);
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(invalid(
                "child-terrain directory contains a non-regular entry",
            ));
        }
        actual.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| invalid("child-terrain directory has a non-UTF-8 name"))?,
        );
    }
    if actual != expected {
        return Err(invalid(format!(
            "child-terrain directory inventory is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn read_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("could not inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        return Err(invalid(format!(
            "{label} is not a bounded real regular file"
        )));
    }
    let mut file = File::open(path)?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)?;
    if bytes.len() as u64 != metadata.len() {
        return Err(invalid(format!("{label} size changed while reading")));
    }
    Ok(bytes)
}

fn hash_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<String> {
    Ok(sha256::digest_hex(&read_regular_bounded(
        path, maximum, label,
    )?))
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(format!("{label} is not a SHA-256 digest")));
    }
    Ok(())
}

fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

fn string(value: &str) -> JsonValue {
    JsonValue::String(value.to_owned())
}

fn number(value: usize) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn number_array(values: &[usize]) -> JsonValue {
    JsonValue::Array(values.iter().map(|value| number(*value)).collect())
}

fn f64_bits(value: f64) -> JsonValue {
    string(&format!("{:016x}", value.to_bits()))
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child_remap::{
        HARD_5M_CHILD_GRID, NOMINAL_PARENT_ORIGIN_I, NOMINAL_PARENT_ORIGIN_J,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn legacy_accepted_transform(source: &[f32]) -> Vec<u8> {
        let mut payload = Vec::with_capacity(PAYLOAD_NBYTES);
        for child_j in 0..CHILD_NY {
            for child_i in 0..CHILD_NX {
                let source_i = 149.25 + 0.5 * child_i as f64;
                let source_j = 149.25 + 0.5 * child_j as f64;
                let i0 = NOMINAL_PARENT_ORIGIN_I + child_i / 2;
                let j0 = NOMINAL_PARENT_ORIGIN_J + child_j / 2;
                let fx = source_i - i0 as f64;
                let fy = source_j - j0 as f64;
                let v00 = f64::from(source[j0 * PARENT_NX + i0]);
                let v10 = f64::from(source[j0 * PARENT_NX + i0 + 1]);
                let v01 = f64::from(source[(j0 + 1) * PARENT_NX + i0]);
                let v11 = f64::from(source[(j0 + 1) * PARENT_NX + i0 + 1]);
                let row0 = v00 * (1.0 - fx) + v10 * fx;
                let row1 = v01 * (1.0 - fx) + v11 * fx;
                let value = (row0 * (1.0 - fy) + row1 * fy) as f32;
                payload.extend_from_slice(&value.to_le_bytes());
            }
        }
        payload
    }

    #[test]
    fn source_coordinate_bilinear_plane_is_continuous_and_not_nearest_cell() {
        let mut source = vec![0.0f32; PARENT_NX * PARENT_NY];
        for j in 0..PARENT_NY {
            for i in 0..PARENT_NX {
                source[j * PARENT_NX + i] = i as f32 * 0.1 + j as f32 * 0.2;
            }
        }
        let transform = transform_source_terrain(&source).unwrap();
        assert_eq!(transform.payload.len(), PAYLOAD_NBYTES);
        let first = f32::from_le_bytes(transform.payload[0..4].try_into().unwrap());
        let second = f32::from_le_bytes(transform.payload[4..8].try_into().unwrap());
        let last = f32::from_le_bytes(
            transform.payload[PAYLOAD_NBYTES - 4..PAYLOAD_NBYTES]
                .try_into()
                .unwrap(),
        );
        let expected_first = (149.25f64 * 0.1 + 149.25f64 * 0.2) as f32;
        let expected_second = (149.75f64 * 0.1 + 149.25f64 * 0.2) as f32;
        let expected_last = (298.75f64 * 0.1 + 298.75f64 * 0.2) as f32;
        assert_eq!(first.to_bits(), expected_first.to_bits());
        assert_eq!(second.to_bits(), expected_second.to_bits());
        assert_eq!(last.to_bits(), expected_last.to_bits());
        assert_ne!(first.to_bits(), source[149 * PARENT_NX + 149].to_bits());
    }

    #[test]
    fn accepted_300x300_50m_transform_is_raw_bit_identical_to_legacy() {
        let source = (0..PARENT_NX * PARENT_NY)
            .map(|index| {
                let mixed = (index as u32)
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                50.0 + (mixed % 1_000_003) as f32 * 0.000_1
            })
            .collect::<Vec<_>>();
        let accepted = transform_source_terrain_for_grid(&source, ACCEPTED_50M_CHILD_GRID)
            .unwrap()
            .payload;
        let legacy = legacy_accepted_transform(&source);
        assert_eq!(accepted.len(), PAYLOAD_NBYTES);
        assert_eq!(accepted, legacy);
    }

    #[test]
    fn the_child_window_is_remapped_from_the_parent_grid_it_was_placed_against() {
        // A 600x600 parent at 50 m, with a child window placed inside it.
        // Nothing about this window is the sealed 300x300 parent, and the
        // remap must follow it rather than reindexing the same child into a
        // foreign row stride.
        let parent = ParentHorizontalGrid::new(600, 600, 50.0, 50.0).unwrap();
        let mut source = vec![0.0f32; parent.nx() * parent.ny()];
        for j in 0..parent.ny() {
            for i in 0..parent.nx() {
                source[j * parent.nx() + i] = i as f32 * 0.1 + j as f32 * 0.2;
            }
        }
        let grid =
            ChildGridSpec::new_in_parent(parent, 40, 40, 75, 25.0, 25.0, CHILD_ZTOP_M, 300, 300)
                .unwrap();
        assert_eq!(grid.parent_grid(), parent);
        let transformed = transform_source_terrain_for_grid(&source, grid).unwrap();
        assert_eq!(transformed.payload.len(), 40 * 40 * 4);

        for (child_i, child_j) in [(0usize, 0usize), (39, 0), (0, 39), (39, 39)] {
            let (source_i, source_j) =
                parent_fractional_xy_for_grid(grid, HorizontalStagger::Surface, child_i, child_j)
                    .unwrap();
            let i0 = source_i.floor() as usize;
            let j0 = source_j.floor() as usize;
            let fx = source_i - i0 as f64;
            let fy = source_j - j0 as f64;
            let v00 = f64::from(source[j0 * parent.nx() + i0]);
            let v10 = f64::from(source[j0 * parent.nx() + i0 + 1]);
            let v01 = f64::from(source[(j0 + 1) * parent.nx() + i0]);
            let v11 = f64::from(source[(j0 + 1) * parent.nx() + i0 + 1]);
            let row0 = v00 * (1.0 - fx) + v10 * fx;
            let row1 = v01 * (1.0 - fx) + v11 * fx;
            let expected = (row0 * (1.0 - fy) + row1 * fy) as f32;
            assert_eq!(
                interpolate_source_terrain_point(&source, grid, child_i, child_j)
                    .unwrap()
                    .to_bits(),
                expected.to_bits()
            );
        }

        // The sealed 300x300 surface offered to that window is refused where
        // the mismatch is, not later by a message about a stencil.
        let sealed_surface = vec![0.0f32; PARENT_NX * PARENT_NY];
        let error = transform_source_terrain_for_grid(&sealed_surface, grid)
            .unwrap_err()
            .to_string();
        assert!(error.contains("placed against a 600x600 parent"), "{error}");
    }

    #[test]
    fn source_coordinate_transform_is_grid_parameterized_and_5m_contained() {
        let mut source = vec![0.0f32; PARENT_NX * PARENT_NY];
        for j in 0..PARENT_NY {
            for i in 0..PARENT_NX {
                source[j * PARENT_NX + i] = i as f32 * 0.1 + j as f32 * 0.2;
            }
        }
        let coarse = ChildGridSpec::new(30, 30, 75, 500.0, 500.0, 15_000.0).unwrap();
        let transformed = transform_source_terrain_for_grid(&source, coarse).unwrap();
        assert_eq!(transformed.payload.len(), 30 * 30 * 4);

        for (child_i, child_j) in [(0, 0), (2_999, 0), (0, 2_999), (2_999, 2_999)] {
            let (source_i, source_j) = parent_fractional_xy_for_grid(
                HARD_5M_CHILD_GRID,
                HorizontalStagger::Surface,
                child_i,
                child_j,
            )
            .unwrap();
            let value =
                interpolate_source_terrain_point(&source, HARD_5M_CHILD_GRID, child_i, child_j)
                    .unwrap();
            let i0 = source_i.floor() as usize;
            let j0 = source_j.floor() as usize;
            let fx = source_i - i0 as f64;
            let fy = source_j - j0 as f64;
            let v00 = f64::from(source[j0 * PARENT_NX + i0]);
            let v10 = f64::from(source[j0 * PARENT_NX + i0 + 1]);
            let v01 = f64::from(source[(j0 + 1) * PARENT_NX + i0]);
            let v11 = f64::from(source[(j0 + 1) * PARENT_NX + i0 + 1]);
            let row0 = v00 * (1.0 - fx) + v10 * fx;
            let row1 = v01 * (1.0 - fx) + v11 * fx;
            let expected = (row0 * (1.0 - fy) + row1 * fy) as f32;
            assert_eq!(value.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn terrain_transform_refuses_nonfinite_selected_source() {
        let mut source = vec![100.0f32; PARENT_NX * PARENT_NY];
        source[NOMINAL_PARENT_ORIGIN_J * PARENT_NX + NOMINAL_PARENT_ORIGIN_I] = f32::NAN;
        assert!(transform_source_terrain(&source).is_err());
    }

    #[test]
    fn atomic_three_file_directory_refuses_replacement() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("gm1-child-terrain-{nonce}"));
        fs::create_dir(&root).unwrap();
        let output = root.join("artifact");
        publish_directory(&output, &[0u8; 4], b"{}", b"{}").unwrap();
        verify_directory_inventory(&output).unwrap();
        assert!(publish_directory(&output, &[0u8; 4], b"{}", b"{}").is_err());
        for file in [
            CHILD_TERRAIN_PAYLOAD_FILE,
            CHILD_TERRAIN_MANIFEST_FILE,
            CHILD_TERRAIN_RECEIPT_FILE,
        ] {
            fs::remove_file(output.join(file)).unwrap();
        }
        fs::remove_dir(output).unwrap();
        fs::remove_dir(root).unwrap();
    }

    #[test]
    fn receipt_contract_binds_payload_to_source_and_parent() {
        let receipt = ChildTerrainReceipt {
            parent_checkpoint_sha256: "aa".repeat(32),
            parent_plan_sha256: "bb".repeat(32),
            manifest_sha256: "11".repeat(32),
            payload_sha256: "22".repeat(32),
            receipt_sha256: String::new(),
            producer_source_sha256: "33".repeat(32),
            producer_executable_sha256: "44".repeat(32),
            minimum_f32_bits: 1.0f32.to_bits(),
            maximum_f32_bits: 2.0f32.to_bits(),
            nonzero_count: CHILD_NX * CHILD_NY,
        };
        let value = receipt_value(&receipt);
        assert_eq!(
            value.get("schema").unwrap().as_str().unwrap(),
            CHILD_TERRAIN_SCHEMA
        );
        assert_eq!(
            value
                .get("source_coordinate_child_terrain_sha256")
                .unwrap()
                .as_str()
                .unwrap(),
            receipt.payload_sha256
        );
        assert_eq!(
            value.get("synthetic_translation").unwrap(),
            &JsonValue::Bool(false)
        );
    }

    #[test]
    fn parameterized_receipt_seals_hard_5m_grid_without_synthetic_detail() {
        let receipt = ChildTerrainGridReceipt {
            grid: HARD_5M_CHILD_GRID,
            parent_checkpoint_sha256: "aa".repeat(32),
            parent_plan_sha256: "bb".repeat(32),
            manifest_sha256: "11".repeat(32),
            payload_sha256: "22".repeat(32),
            receipt_sha256: String::new(),
            producer_source_sha256: "33".repeat(32),
            producer_executable_sha256: "44".repeat(32),
            minimum_f32_bits: 1.0f32.to_bits(),
            maximum_f32_bits: 2.0f32.to_bits(),
            nonzero_count: HARD_5M_CHILD_GRID.nx() * HARD_5M_CHILD_GRID.ny(),
        };
        let value = grid_receipt_value(&receipt);
        assert_eq!(
            value.get("schema").unwrap().as_str().unwrap(),
            CHILD_TERRAIN_GRID_SCHEMA
        );
        assert_eq!(
            value.get("payload_nbytes").unwrap().as_u64().unwrap(),
            36_000_000
        );
        assert_eq!(
            value.get("synthetic_detail").unwrap(),
            &JsonValue::Bool(false)
        );
        assert_eq!(
            value.get("source_backed_interpolation").unwrap(),
            &JsonValue::Bool(true)
        );
        let grid = value.get("grid").unwrap();
        assert_eq!(grid.get("nx").unwrap().as_u64().unwrap(), 3_000);
        assert_eq!(grid.get("ny").unwrap().as_u64().unwrap(), 3_000);
        assert_eq!(
            grid.get("dx_m_f64_bits").unwrap().as_str().unwrap(),
            &format!("{:016x}", 5.0f64.to_bits())
        );
        let transform = value.get("transform").unwrap();
        let extent = transform.get("source_coordinate_extent_f64_bits").unwrap();
        assert_eq!(
            extent.get("first_i").unwrap().as_str().unwrap(),
            &format!("{:016x}", 149.025f64.to_bits())
        );
        assert_eq!(
            extent.get("last_i").unwrap().as_str().unwrap(),
            &format!("{:016x}", 298.975f64.to_bits())
        );
    }
}
