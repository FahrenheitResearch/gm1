//! Source-sealed child base, live-density, geopotential, and open-boundary publication.
//!
//! The input atmosphere is the committed 28-carrier CM1 restart-grid artifact.
//! This producer reconstructs a terrain-varying base from the exact f02 source
//! profiles, preserves the interpolated perturbation carriers, rebuilds the
//! selected `psolver=3` density family, and initializes all four radiation
//! histories to fresh positive zero.  TKE is deliberately not synthesized.

use crate::child_artifact::{json_sha256_at, read_canonical_json};
use crate::child_atmosphere::open_child_atmosphere_artifact;
use crate::child_remap::{
    sealed_child_stretch_grid, validate_bound_parent_identity, ChildGridSpec, CHILD_NX, CHILD_NY,
    CHILD_NZ, CHILD_STRETCH_GRID_SHA256,
};
use crate::json::JsonValue;
use crate::product_restart::{
    read_initialized_product_restart_self_anchored, RestoredProductRestart,
};
use crate::sha256::{self, Sha256};
use crate::terrain::HorizontalCrop;
use crate::terrain_admission::{
    terrain_open_carrier_shapes, terrain_open_carrier_specs, TERRAIN_OPEN_CARRIER_ABI,
    TERRAIN_OPEN_CARRIER_NAMES, TERRAIN_OPEN_IDENTITY_SCHEMA, TERRAIN_OPEN_RECEIPT_SCHEMA,
};
use crate::terrain_admission_science::{
    compute_child_live_density, compute_stretched_terrain_base, compute_surface_halos,
    pack_cold_start_mutable, LiveDensityFrame, MutableCarrier, PerturbationPolicy,
    SurfaceHaloFrame, TerrainBaseFrame, G,
};
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHILD_AUXILIARY_SCHEMA: &str = "gm1.source-coordinate-50m-child-auxiliary/v1";
pub const CHILD_AUXILIARY_MANIFEST_SCHEMA: &str =
    "gm1.source-coordinate-50m-child-auxiliary-manifest/v1";
pub const CHILD_AUXILIARY_RECEIPT_SCHEMA: &str =
    "gm1.source-coordinate-50m-child-auxiliary-receipt/v1";
pub const CHILD_LIVE_DENSITY_ABI: &str = "gm1.source-coordinate-50m-child-live-density-f32/v1";
pub const CHILD_AUXILIARY_TERRAIN_OPEN_FILE: &str = "terrain-open.f32le";
pub const CHILD_AUXILIARY_LIVE_DENSITY_FILE: &str = "live-density.f32le";
pub const CHILD_AUXILIARY_GEOPOTENTIAL_FILE: &str = "total-geopotential-zf.f32le";
pub const CHILD_AUXILIARY_MANIFEST_FILE: &str = "manifest.json";
pub const CHILD_AUXILIARY_RECEIPT_FILE: &str = "receipt.json";
pub const CHILD_AUXILIARY_STATUS: &str =
    "PASS_REAL_CHILD_BASE_DENSITY_GEOPOTENTIAL_FRESH_OPEN_BOUNDARY";
pub const CHILD_AUXILIARY_CLAIM_LIMIT: &str = "source-backed CM1 r21.1 base.F terrain-column hydrostatic reconstruction on the exact real f02 50 m child, exact preservation and source-family halo packing of the committed restart-grid perturbation carriers, selected psolver=3 getset pressure/density publication, binary32 zf geopotential, and fresh positive-zero four-open radiation histories; TKE remains fail-closed pending admitted child calcnm/calcdef inputs, moisture halos remain consumer-owned, and no child time integration, trajectory, forecast-skill, tornado, or performance claim is made";
pub const CHILD_LIVE_DENSITY_CARRIER_NAMES: [&str; 4] = ["prs", "rf", "rho", "rr"];
pub const CHILD_AUXILIARY_BASE_LAYOUT_SCHEMA: &str =
    "gm1.source-coordinate-child-auxiliary-base-layout/v1";
pub const CHILD_AUXILIARY_BASE_CAPACITY_SCHEMA: &str =
    "gm1.source-coordinate-child-auxiliary-base-capacity/v1";
pub const CHILD_AUXILIARY_BASE_STREAM_CHUNK_BYTES: u64 = 1024 * 1024;
pub const HARD_5M_CHILD_TERRAIN_OPEN_NBYTES: u64 = 92_416_017_536;
pub const HARD_5M_CHILD_LIVE_DENSITY_NBYTES: u64 = 10_836_000_000;
pub const HARD_5M_CHILD_GEOPOTENTIAL_NBYTES: u64 = 2_736_000_000;
pub const HARD_5M_CHILD_U0_NBYTES: u64 = 2_700_900_000;
pub const HARD_5M_CHILD_V0_NBYTES: u64 = 2_700_900_000;
pub const HARD_5M_CHILD_BASE_WIND_NBYTES: u64 = 5_401_800_000;
pub const HARD_5M_CHILD_AUXILIARY_BASE_NBYTES: u64 = 111_389_817_536;

/// The first initialized product this chain's base states were read from.
///
/// These are lineage evidence, not the admission gate for the 50 m child: the
/// 50 m auxiliary and full-state producers read the product by the identity it
/// records itself and then require it to be the product the atmosphere's own
/// parent record names.  The hard 5 m streaming producer in
/// `child_auxiliary_stream` still pins them, because that producer publishes
/// against a compile-time preset atmosphere pinned to the same run.
pub const F02_PRODUCT_RESTART_SHA256: &str =
    "9efe09771763a3f727af715f7dd41a2c6256cd5613aa2f31637cc08c9721ed6e";
pub const F02_PRODUCT_MANIFEST_SHA256: &str =
    "b25de4008d17ddad46b8057f022b50a76e6bff61b537fa9952f68e638adf2567";
pub const F02_PRODUCT_PAYLOAD_SHA256: &str =
    "ddafeaa8ca523e81925cb4b53e0577963a3e57e06114b6b7df8c6d436e0d68b1";
pub const F02_PRODUCT_RECEIPT_SHA256: &str =
    "379b989884e773a942ade62069e99ce5e032374c2ae2ef13679f32a524fbbdcd";
pub const F02_SOURCE_ARTIFACT_SHA256: &str =
    "356fe3022ce83909f5c14e7ce9a3639811d9f6dfc5ed9b33a0782cfffb7cd79d";
pub const F02_SOURCE_CONTRACT_SHA256: &str =
    "a9ccd6f0c6650fd89cd42f16977209f57695464181bcb62beff14e9a9f61d896";

pub const CHILD_TERRAIN_MANIFEST_SHA256: &str =
    "bc6006491174c4b7594f4c12582749f3d3494015bba2eb692106005d3aeb391a";
pub const CHILD_TERRAIN_PAYLOAD_SHA256: &str =
    "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91";
pub const CHILD_TERRAIN_RECEIPT_SHA256: &str =
    "0c50ea83658b4ea47a711580200fe7280face00baba7b089d3e5572280004bbf";
pub const STATIC_HEIGHT_MANIFEST_SHA256: &str =
    "5459be4acf6ba3f8f83e6ff59bdfec11946d2c4846f60556bf02a1892289b9d6";
pub const STATIC_ZH_SHA256: &str =
    "7fe66bd1639fb0cb4ad171be87b77484703a4d92ce868524e796e888a52a5159";
pub const STATIC_ZF_SHA256: &str =
    "df5aef17965915406fe334600afe2f8f2e1ebb5d07665f71820eaac9bde4128f";
pub const STATIC_VALIDATION_RECEIPT_SHA256: &str =
    "f90fcadbe9f75eacc66a96d56dc9584166b2e50641a80e311f358c4bb49a27a8";
pub const STATIC_HEIGHT_ABI: &str = "gm1.cuda-source-coordinate-50m-child-static-heights-f32/v1";
pub const GEOPOTENTIAL_ABI: &str = "gpuwm-cm1.arwen-wsm6-total-geopotential/v1";
pub const SOURCE_MASS_Z_SHA256: &str =
    "458b7ecc933a04b929e2d59e2a1b0cde856e583189d9690d5c481fed4e4f9419";
pub const FRESH_TKE_COMMIT: &str = "d32af2cb8fd343e551a7911aedd2e057c827961c";
pub const FRESH_TKE_SCHEMA: &str = "gm1.cuda-child-turbsmag-init/v1";
pub const FRESH_TKE_ABI: &str = "gm1.cuda-child-turbsmag-init-f32/v1";
pub const FRESH_TKE_KERNEL_SHA256: &str =
    "8937435b68d61aa6c602b5f5c37796216e15a61e725b83cc0e4aab3af7eb5ac5";
pub const FRESH_TKE_RECEIPT_SHA256: &str =
    "d0eba61ee44091b30dbf4e0ab1fd28e366c8f984d7840a68e1dc8ea3a52aacfe";

const AUTHORITY_TARBALL_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
const AUTHORITY_SOURCE: [(&str, &str, &str); 6] = [
    (
        "init_terrain.F",
        "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c",
        "175-297",
    ),
    (
        "param.F",
        "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
        "7789-7812,8162-8207,8464-8492,8988-9062",
    ),
    (
        "base.F",
        "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef",
        "625-749,2044-2243",
    ),
    (
        "init3d.F",
        "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
        "2280-2467",
    ),
    (
        "bc.F",
        "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
        "12-172,180-1206,1948-2172",
    ),
    (
        "cm1.F",
        "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
        "1559-1657",
    ),
];
const JSON_LIMIT: u64 = 2 * 1024 * 1024;
const SOURCE_LIMIT: u64 = 16 * 1024 * 1024;
const OUTPUT_LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const CHILD_TOP_M: f32 = 15_000.0;
const SOURCE_DZ_M: f32 = 200.0;
const TERRAIN_WORDS: usize = CHILD_NY * CHILD_NX;
const MASS_WORDS: usize = CHILD_NZ * CHILD_NY * CHILD_NX;
const FACE_WORDS: usize = (CHILD_NZ + 1) * CHILD_NY * CHILD_NX;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
struct Record {
    name: String,
    shape: Vec<usize>,
    offset_nbytes: u64,
    nbytes: u64,
    sha256: String,
    finite_count: usize,
    nonzero_count: usize,
    minimum_f32_bits: u32,
    maximum_f32_bits: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChildAuxiliaryReceipt {
    pub manifest_sha256: String,
    pub terrain_open_sha256: String,
    pub live_density_sha256: String,
    pub geopotential_sha256: String,
    pub receipt_sha256: String,
    pub producer_source_sha256: String,
    pub producer_executable_sha256: String,
}

/// Allocation-free byte layout for the source-backed auxiliary and terrain-
/// aware base-wind estate of one admitted child grid.
///
/// The terrain/open bytes include the exact 50-carrier CM1 halo allocation.
/// `total_nbytes` deliberately excludes the already immutable atmosphere,
/// terrain, and static-height artifacts so a publisher can bind those by hash
/// without silently duplicating them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChildAuxiliaryBaseLayout {
    pub grid: ChildGridSpec,
    pub terrain_open_carrier_count: usize,
    pub terrain_open_nbytes: u64,
    pub live_density_carrier_count: usize,
    pub live_density_nbytes: u64,
    pub geopotential_nbytes: u64,
    pub u0_nbytes: u64,
    pub v0_nbytes: u64,
    pub base_wind_nbytes: u64,
    pub total_nbytes: u64,
}

/// Admission result for a same-volume, single-copy streamed publication.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChildAuxiliaryBaseCapacityGate {
    pub layout: ChildAuxiliaryBaseLayout,
    pub free_bytes: u64,
    pub reserve_bytes: u64,
    pub planned_payload_copies: u64,
    pub post_publish_free_bytes: u64,
    pub post_publish_above_reserve_bytes: u64,
}

/// Compute the exact auxiliary/base-state payload sizes without allocating any
/// carrier data.
pub fn child_auxiliary_base_layout_for_grid(
    grid: ChildGridSpec,
) -> Result<ChildAuxiliaryBaseLayout> {
    let terrain_open_shapes = terrain_open_carrier_shapes(grid.nz(), grid.ny(), grid.nx())?;
    let terrain_open_nbytes =
        terrain_open_shapes
            .iter()
            .try_fold(0u64, |total, (_, shape)| {
                let words = shape.iter().try_fold(1u64, |words, extent| {
                    words
                        .checked_mul(*extent as u64)
                        .ok_or_else(|| invalid("child terrain/open carrier word count overflows"))
                })?;
                total
                    .checked_add(words.checked_mul(4).ok_or_else(|| {
                        invalid("child terrain/open carrier byte count overflows")
                    })?)
                    .ok_or_else(|| invalid("child terrain/open arena byte count overflows"))
            })?;
    let mass_words = checked_grid_words(grid.nz(), grid.ny(), grid.nx(), "mass")?;
    let face_words = checked_grid_words(grid.nz() + 1, grid.ny(), grid.nx(), "z face")?;
    let live_density_nbytes = mass_words
        .checked_mul(3)
        .and_then(|words| words.checked_add(face_words))
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| invalid("child live-density byte count overflows"))?;
    let geopotential_nbytes = face_words
        .checked_mul(4)
        .ok_or_else(|| invalid("child geopotential byte count overflows"))?;
    let u0_nbytes = checked_grid_words(grid.nz(), grid.ny(), grid.nx() + 1, "u0")?
        .checked_mul(4)
        .ok_or_else(|| invalid("child u0 byte count overflows"))?;
    let v0_nbytes = checked_grid_words(grid.nz(), grid.ny() + 1, grid.nx(), "v0")?
        .checked_mul(4)
        .ok_or_else(|| invalid("child v0 byte count overflows"))?;
    let base_wind_nbytes = u0_nbytes
        .checked_add(v0_nbytes)
        .ok_or_else(|| invalid("child base-wind byte count overflows"))?;
    let total_nbytes = terrain_open_nbytes
        .checked_add(live_density_nbytes)
        .and_then(|value| value.checked_add(geopotential_nbytes))
        .and_then(|value| value.checked_add(base_wind_nbytes))
        .ok_or_else(|| invalid("child auxiliary/base total byte count overflows"))?;
    Ok(ChildAuxiliaryBaseLayout {
        grid,
        terrain_open_carrier_count: terrain_open_shapes.len(),
        terrain_open_nbytes,
        live_density_carrier_count: CHILD_LIVE_DENSITY_CARRIER_NAMES.len(),
        live_density_nbytes,
        geopotential_nbytes,
        u0_nbytes,
        v0_nbytes,
        base_wind_nbytes,
        total_nbytes,
    })
}

/// Fail closed unless the exact payload can be staged once on its destination
/// volume while preserving the requested free-space reserve.
pub fn admit_child_auxiliary_base_capacity(
    grid: ChildGridSpec,
    free_bytes: u64,
    reserve_bytes: u64,
    planned_payload_copies: u64,
) -> Result<ChildAuxiliaryBaseCapacityGate> {
    if planned_payload_copies != 1 {
        return Err(invalid(
            "child auxiliary/base publication requires exactly one same-volume staged payload copy",
        ));
    }
    let layout = child_auxiliary_base_layout_for_grid(grid)?;
    let required = layout
        .total_nbytes
        .checked_add(reserve_bytes)
        .ok_or_else(|| invalid("child auxiliary/base capacity requirement overflows"))?;
    if free_bytes < required {
        return Err(invalid(format!(
            "child auxiliary/base publication needs {} payload bytes plus {} reserve bytes, but destination volume has only {} free bytes",
            layout.total_nbytes, reserve_bytes, free_bytes
        )));
    }
    let post_publish_free_bytes = free_bytes - layout.total_nbytes;
    Ok(ChildAuxiliaryBaseCapacityGate {
        layout,
        free_bytes,
        reserve_bytes,
        planned_payload_copies,
        post_publish_free_bytes,
        post_publish_above_reserve_bytes: post_publish_free_bytes - reserve_bytes,
    })
}

fn checked_grid_words(z: usize, y: usize, x: usize, label: &str) -> Result<u64> {
    (z as u64)
        .checked_mul(y as u64)
        .and_then(|words| words.checked_mul(x as u64))
        .ok_or_else(|| invalid(format!("child {label} word count overflows")))
}

struct Prepared {
    /// The initialization product the base-state sounding was read from,
    /// recorded rather than assumed.
    source_product_manifest_sha256: String,
    source_product_payload_sha256: String,
    source_product_receipt_sha256: String,
    source_product_restart_sha256: String,
    source_artifact_sha256: String,
    source_contract_sha256: String,
    /// The atmosphere this auxiliary estate was actually derived from, and
    /// the parent that atmosphere was built from - recorded, not assumed.
    atmosphere_manifest_sha256: String,
    atmosphere_arena_sha256: String,
    atmosphere_receipt_sha256: String,
    parent_checkpoint_sha256: String,
    parent_plan_sha256: String,
    terrain_manifest_sha256: String,
    terrain_payload_sha256: String,
    terrain_receipt_sha256: String,
    static_manifest_sha256: String,
    static_validation_sha256: String,
    static_zh_sha256: String,
    static_zf_sha256: String,
    terrain_open: Vec<MutableCarrier>,
    live_density: Vec<MutableCarrier>,
    geopotential: Vec<f32>,
    source_profile_hashes: BTreeMap<String, String>,
    source_z_sha256: String,
    child_bottom: [f32; 3],
    child_top: [f32; 3],
    producer_source_sha256: String,
    producer_executable_sha256: String,
}

type SourceProfileValues = BTreeMap<String, Vec<f32>>;
type SourceProfileHashes = BTreeMap<String, String>;

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[allow(clippy::too_many_arguments)]
pub fn publish_child_auxiliary_new(
    atmosphere_directory: impl AsRef<Path>,
    terrain_directory: impl AsRef<Path>,
    static_height_directory: impl AsRef<Path>,
    static_validation_receipt: impl AsRef<Path>,
    source_product_restart: impl AsRef<Path>,
    authority_directory: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
) -> Result<ChildAuxiliaryReceipt> {
    let prepared = prepare(
        atmosphere_directory.as_ref(),
        terrain_directory.as_ref(),
        static_height_directory.as_ref(),
        static_validation_receipt.as_ref(),
        source_product_restart.as_ref(),
        authority_directory.as_ref(),
        producer_source.as_ref(),
        producer_executable.as_ref(),
    )?;
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace child auxiliary destination {}",
            output.display()
        )));
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid("child auxiliary destination parent does not exist"));
    }
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("child auxiliary destination needs a UTF-8 name"))?;
    let staging = output.with_file_name(format!(
        ".{name}.partial-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&staging)?;
    let result = (|| -> Result<ChildAuxiliaryReceipt> {
        let (terrain_open_records, terrain_open_sha256, terrain_open_nbytes) = write_arena_new(
            &staging.join(CHILD_AUXILIARY_TERRAIN_OPEN_FILE),
            &prepared.terrain_open,
        )?;
        let (live_density_records, live_density_sha256, live_density_nbytes) = write_arena_new(
            &staging.join(CHILD_AUXILIARY_LIVE_DENSITY_FILE),
            &prepared.live_density,
        )?;
        let geopotential_carrier = MutableCarrier {
            name: "total_geopotential_zf",
            role: "binary32 child total geopotential on z faces",
            shape: vec![CHILD_NZ + 1, CHILD_NY, CHILD_NX],
            live_region: "physical z faces; php is fresh positive zero downstream",
            values: prepared.geopotential.clone(),
        };
        let (geopotential_records, geopotential_sha256, geopotential_nbytes) = write_arena_new(
            &staging.join(CHILD_AUXILIARY_GEOPOTENTIAL_FILE),
            &[geopotential_carrier],
        )?;
        let manifest = manifest_value(
            &prepared,
            &terrain_open_records,
            &terrain_open_sha256,
            terrain_open_nbytes,
            &live_density_records,
            &live_density_sha256,
            live_density_nbytes,
            &geopotential_records,
            &geopotential_sha256,
            geopotential_nbytes,
        );
        let manifest_bytes = manifest.canonical_bytes();
        let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
        let template = ChildAuxiliaryReceipt {
            manifest_sha256,
            terrain_open_sha256,
            live_density_sha256,
            geopotential_sha256,
            receipt_sha256: String::new(),
            producer_source_sha256: prepared.producer_source_sha256.clone(),
            producer_executable_sha256: prepared.producer_executable_sha256.clone(),
        };
        let receipt_bytes = receipt_value(&template).canonical_bytes();
        let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
        write_new_synced(
            &staging.join(CHILD_AUXILIARY_MANIFEST_FILE),
            &manifest_bytes,
        )?;
        write_new_synced(&staging.join(CHILD_AUXILIARY_RECEIPT_FILE), &receipt_bytes)?;
        verify_inventory(&staging)?;
        fs::rename(&staging, output)?;
        Ok(ChildAuxiliaryReceipt {
            receipt_sha256,
            ..template
        })
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub fn inspect_child_auxiliary(
    atmosphere_directory: impl AsRef<Path>,
    terrain_directory: impl AsRef<Path>,
    static_height_directory: impl AsRef<Path>,
    static_validation_receipt: impl AsRef<Path>,
    source_product_restart: impl AsRef<Path>,
    authority_directory: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    artifact_directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_terrain_open_sha256: &str,
    expected_live_density_sha256: &str,
    expected_geopotential_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<ChildAuxiliaryReceipt> {
    for (value, label) in [
        (expected_manifest_sha256, "child auxiliary manifest"),
        (expected_terrain_open_sha256, "child terrain/open arena"),
        (expected_live_density_sha256, "child live-density arena"),
        (expected_geopotential_sha256, "child geopotential payload"),
        (expected_receipt_sha256, "child auxiliary receipt"),
    ] {
        validate_sha256(value, label)?;
    }
    let prepared = prepare(
        atmosphere_directory.as_ref(),
        terrain_directory.as_ref(),
        static_height_directory.as_ref(),
        static_validation_receipt.as_ref(),
        source_product_restart.as_ref(),
        authority_directory.as_ref(),
        producer_source.as_ref(),
        producer_executable.as_ref(),
    )?;
    let directory = artifact_directory.as_ref();
    verify_inventory(directory)?;
    let (terrain_records, terrain_hash, terrain_nbytes) = compare_arena(
        &directory.join(CHILD_AUXILIARY_TERRAIN_OPEN_FILE),
        &prepared.terrain_open,
    )?;
    let (density_records, density_hash, density_nbytes) = compare_arena(
        &directory.join(CHILD_AUXILIARY_LIVE_DENSITY_FILE),
        &prepared.live_density,
    )?;
    let geopotential_carrier = MutableCarrier {
        name: "total_geopotential_zf",
        role: "binary32 child total geopotential on z faces",
        shape: vec![CHILD_NZ + 1, CHILD_NY, CHILD_NX],
        live_region: "physical z faces; php is fresh positive zero downstream",
        values: prepared.geopotential.clone(),
    };
    let (geopototential_records, geopotential_hash, geopotential_nbytes) = compare_arena(
        &directory.join(CHILD_AUXILIARY_GEOPOTENTIAL_FILE),
        &[geopotential_carrier],
    )?;
    if terrain_hash != expected_terrain_open_sha256
        || density_hash != expected_live_density_sha256
        || geopotential_hash != expected_geopotential_sha256
    {
        return Err(invalid(
            "child auxiliary payload hash differs from its independent pin",
        ));
    }
    let manifest = manifest_value(
        &prepared,
        &terrain_records,
        &terrain_hash,
        terrain_nbytes,
        &density_records,
        &density_hash,
        density_nbytes,
        &geopototential_records,
        &geopotential_hash,
        geopotential_nbytes,
    );
    let expected_manifest = manifest.canonical_bytes();
    let observed_manifest = read_regular_bounded(
        &directory.join(CHILD_AUXILIARY_MANIFEST_FILE),
        JSON_LIMIT,
        "child auxiliary manifest",
    )?;
    if sha256::digest_hex(&observed_manifest) != expected_manifest_sha256
        || observed_manifest != expected_manifest
    {
        return Err(invalid(
            "child auxiliary manifest differs from exact recomputation",
        ));
    }
    let template = ChildAuxiliaryReceipt {
        manifest_sha256: expected_manifest_sha256.to_string(),
        terrain_open_sha256: terrain_hash,
        live_density_sha256: density_hash,
        geopotential_sha256: geopotential_hash,
        receipt_sha256: String::new(),
        producer_source_sha256: prepared.producer_source_sha256,
        producer_executable_sha256: prepared.producer_executable_sha256,
    };
    let expected_receipt = receipt_value(&template).canonical_bytes();
    let observed_receipt = read_regular_bounded(
        &directory.join(CHILD_AUXILIARY_RECEIPT_FILE),
        JSON_LIMIT,
        "child auxiliary receipt",
    )?;
    if sha256::digest_hex(&observed_receipt) != expected_receipt_sha256
        || observed_receipt != expected_receipt
    {
        return Err(invalid(
            "child auxiliary receipt differs from exact recomputation",
        ));
    }
    Ok(ChildAuxiliaryReceipt {
        receipt_sha256: expected_receipt_sha256.to_string(),
        ..template
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    atmosphere_directory: &Path,
    terrain_directory: &Path,
    static_height_directory: &Path,
    static_validation_receipt: &Path,
    source_product_restart: &Path,
    authority_directory: &Path,
    producer_source: &Path,
    producer_executable: &Path,
) -> Result<Prepared> {
    validate_authority(authority_directory)?;
    let producer_source_sha256 = hash_regular_bounded(
        producer_source,
        SOURCE_LIMIT,
        "child auxiliary producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable,
        SOURCE_LIMIT * 16,
        "child auxiliary producer executable",
    )?;

    let atmosphere = open_child_atmosphere_artifact(atmosphere_directory)?;
    validate_bound_parent_identity(
        &atmosphere.receipt().parent_checkpoint_sha256,
        &atmosphere.receipt().parent_plan_sha256,
        "child auxiliary source atmosphere",
    )?;
    let atmosphere_manifest_sha256 = atmosphere.receipt().manifest_sha256.clone();
    let atmosphere_arena_sha256 = atmosphere.receipt().arena_sha256.clone();
    let atmosphere_receipt_sha256 = atmosphere.receipt().receipt_sha256.clone();
    let parent_checkpoint_sha256 = atmosphere.receipt().parent_checkpoint_sha256.clone();
    let parent_plan_sha256 = atmosphere.receipt().parent_plan_sha256.clone();
    let mut atmosphere_bytes =
        atmosphere.read_carriers_f32le_exact(&["ppi", "qa", "tha", "ua", "va", "wa"])?;
    let ppi = take_f32(&mut atmosphere_bytes, "ppi", MASS_WORDS)?;
    let qa = take_f32(&mut atmosphere_bytes, "qa", 6 * MASS_WORDS)?;
    let tha = take_f32(&mut atmosphere_bytes, "tha", MASS_WORDS)?;
    let ua = take_f32(
        &mut atmosphere_bytes,
        "ua",
        CHILD_NZ * CHILD_NY * (CHILD_NX + 1),
    )?;
    let va = take_f32(
        &mut atmosphere_bytes,
        "va",
        CHILD_NZ * (CHILD_NY + 1) * CHILD_NX,
    )?;
    let wa = take_f32(&mut atmosphere_bytes, "wa", FACE_WORDS)?;
    if !atmosphere_bytes.is_empty() {
        return Err(invalid(
            "child atmosphere extraction left unexpected carriers",
        ));
    }
    if qa.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(invalid(
            "pdefq-clean child qa contains a non-finite or negative word",
        ));
    }

    // Terrain and static heights are admitted by the identities they record
    // themselves, not by a digest frozen to one historical publication.  The
    // terrain receipt names the parent it was built from, the terrain payload
    // digest, and its own manifest digest; the static-height validation receipt
    // names the terrain it was derived from and both height payload digests.
    // Requiring those to agree is what actually prevents the wrong result - a
    // child whose open-boundary and base-state carriers were reconstructed over
    // one surface while its atmosphere and metrics came from another.
    verify_exact_directory(
        terrain_directory,
        &["manifest.json", "receipt.json", "terrain-msl.f32le"],
        "child terrain",
    )?;
    let terrain_manifest_sha256 = hash_regular_bounded(
        &terrain_directory.join("manifest.json"),
        JSON_LIMIT,
        "child terrain manifest",
    )?;
    let (terrain_receipt_bytes, terrain_receipt) = read_canonical_json(
        &terrain_directory.join("receipt.json"),
        JSON_LIMIT,
        "child terrain receipt",
    )?;
    let terrain_receipt_sha256 = sha256::digest_hex(&terrain_receipt_bytes);
    if json_sha256(&terrain_receipt, "manifest_sha256", "child terrain receipt")?
        != terrain_manifest_sha256
    {
        return Err(invalid(
            "child terrain manifest bytes are not the manifest its own receipt anchors; the published transform and the published receipt describe different terrain",
        ));
    }
    let terrain_payload_sha256 =
        json_sha256(&terrain_receipt, "payload_sha256", "child terrain receipt")?;
    let terrain_parent_checkpoint_sha256 = json_sha256(
        &terrain_receipt,
        "parent_checkpoint_sha256",
        "child terrain receipt",
    )?;
    let terrain_parent_plan_sha256 = json_sha256(
        &terrain_receipt,
        "parent_plan_sha256",
        "child terrain receipt",
    )?;
    if terrain_parent_checkpoint_sha256 != parent_checkpoint_sha256
        || terrain_parent_plan_sha256 != parent_plan_sha256
    {
        return Err(invalid(format!(
            "child terrain was published from parent checkpoint {terrain_parent_checkpoint_sha256} / plan {terrain_parent_plan_sha256}, but the atmosphere this auxiliary estate is derived from names checkpoint {parent_checkpoint_sha256} / plan {parent_plan_sha256}; the child would stand on a surface from one parent run with perturbation carriers from another"
        )));
    }
    let terrain = read_f32_file(
        &terrain_directory.join("terrain-msl.f32le"),
        TERRAIN_WORDS,
        &terrain_payload_sha256,
        "child terrain payload",
    )?;

    verify_exact_directory(
        static_height_directory,
        &["manifest.json", "zf.f32le", "zh.f32le"],
        "child static heights",
    )?;
    let static_manifest_sha256 = hash_regular_bounded(
        &static_height_directory.join("manifest.json"),
        JSON_LIMIT,
        "child static-height manifest",
    )?;
    let (static_receipt_bytes, static_receipt) = read_canonical_json(
        static_validation_receipt,
        JSON_LIMIT,
        "child static-height validation receipt",
    )?;
    let static_validation_sha256 = sha256::digest_hex(&static_receipt_bytes);
    let export = member(
        &static_receipt,
        "static_height_export",
        "child static-height validation receipt",
    )?;
    if json_sha256(export, "manifest_sha256", "child static-height export")?
        != static_manifest_sha256
    {
        return Err(invalid(
            "child static-height manifest bytes are not the manifest its validation receipt anchors; the published heights and the validated heights are different artifacts",
        ));
    }
    let source_terrain = member(
        &static_receipt,
        "source_terrain",
        "child static-height validation receipt",
    )?;
    for (key, expected, what) in [
        (
            "manifest_sha256",
            terrain_manifest_sha256.as_str(),
            "manifest",
        ),
        ("payload_sha256", terrain_payload_sha256.as_str(), "payload"),
        ("receipt_sha256", terrain_receipt_sha256.as_str(), "receipt"),
    ] {
        let recorded = json_sha256(source_terrain, key, "child static-height source terrain")?;
        if recorded != expected {
            return Err(invalid(format!(
                "child static heights were derived from terrain {what} {recorded}, but the terrain admitted here is {expected}; every vertical metric and face height would describe a surface the child does not have"
            )));
        }
    }
    let payloads = member(export, "payloads", "child static-height export")?;
    let static_zh_sha256 = json_sha256(
        member(payloads, "zh", "child static-height export")?,
        "sha256",
        "child static-height zh",
    )?;
    let static_zf_sha256 = json_sha256(
        member(payloads, "zf", "child static-height export")?,
        "sha256",
        "child static-height zf",
    )?;
    let zh = read_f32_file(
        &static_height_directory.join("zh.f32le"),
        MASS_WORDS,
        &static_zh_sha256,
        "child static zh",
    )?;
    let zf = read_f32_file(
        &static_height_directory.join("zf.f32le"),
        FACE_WORDS,
        &static_zf_sha256,
        "child static zf",
    )?;

    // The base-state sounding comes from the product the parent run was itself
    // initialized from.  The product is admitted by the identity it records
    // itself, and is then required to be the product the atmosphere this
    // estate is derived from names as its parent's initialization: pi0/th0/qv0
    // reconstructed from another run's sounding would put this child's base
    // state under perturbation carriers that never stood on it.
    let source = read_initialized_product_restart_self_anchored(source_product_restart)?;
    let parent_source_product_manifest_sha256 = json_sha256_at(
        atmosphere.manifest(),
        &["parent", "source_product_manifest_sha256"],
        "child atmosphere manifest",
    )?;
    if source.receipt().product_manifest_sha256 != parent_source_product_manifest_sha256 {
        return Err(invalid(format!(
            "child auxiliary base-state source is initialized product {}, but the atmosphere it is derived from was built from a parent initialized by product {}; the child's pi0/th0/qv0 base state would come from a different run than its perturbation carriers",
            source.receipt().product_manifest_sha256,
            parent_source_product_manifest_sha256
        )));
    }
    let source_product_manifest_sha256 = source.receipt().product_manifest_sha256.clone();
    let source_product_payload_sha256 = source.receipt().payload_sha256.clone();
    let source_product_receipt_sha256 = source.receipt().product_receipt_sha256.clone();
    let source_product_restart_sha256 = source.receipt().artifact_sha256.clone();
    let source_artifact_sha256 = source.receipt().source_artifact_sha256.clone();
    let source_contract_sha256 = source.receipt().source_contract_sha256.clone();
    let (mut profiles, source_profile_hashes) = read_source_profiles(&source)?;
    let pi_profile = profiles
        .remove("pi0")
        .ok_or_else(|| invalid("source profile extraction lost pi0"))?;
    let theta_profile = profiles
        .remove("th0")
        .ok_or_else(|| invalid("source profile extraction lost th0"))?;
    let qv_profile = profiles
        .remove("qv0")
        .ok_or_else(|| invalid("source profile extraction lost qv0"))?;
    drop(profiles);
    let source_z = (0..CHILD_NZ)
        .map(|k| (k as f32 + 0.5) * SOURCE_DZ_M)
        .collect::<Vec<_>>();
    let source_z_sha256 = hash_f32(&source_z);
    if source_z_sha256 != SOURCE_MASS_Z_SHA256 {
        return Err(invalid("source f32 mass-height profile identity drifted"));
    }

    let grid = sealed_child_stretch_grid()?;
    if grid.combined_sha256() != CHILD_STRETCH_GRID_SHA256 {
        return Err(invalid("sealed child stretch-grid identity drifted"));
    }
    let sigma_mass = grid.sigma[1..=CHILD_NZ].to_vec();
    let sigma_face = grid.sigmaf[1..=CHILD_NZ + 1].to_vec();
    let child_bottom = extrapolation_weights(&sigma_mass, &sigma_face, false)?;
    let child_top = extrapolation_weights(&sigma_mass, &sigma_face, true)?;
    let (c1, c2) = derive_and_validate_child_metrics(&terrain, &zh, &zf, &sigma_mass, &sigma_face)?;

    let base = compute_stretched_terrain_base(
        &terrain,
        [CHILD_NZ, CHILD_NY, CHILD_NX],
        &zh,
        &sigma_mass,
        CHILD_TOP_M,
        SOURCE_DZ_M,
        &pi_profile,
        &theta_profile,
        &qv_profile,
        &c1,
        &c2,
        child_bottom,
        child_top,
    )?;
    let surface = compute_surface_halos(
        &terrain,
        [CHILD_NY, CHILD_NX],
        HorizontalCrop::full(CHILD_NY, CHILD_NX),
        [50.0, 50.0, 200.0],
        CHILD_NZ,
    )?;
    let mutable = pack_cold_start_mutable(
        [CHILD_NZ, CHILD_NY, CHILD_NX],
        HorizontalCrop::full(CHILD_NY, CHILD_NX),
        &pi_profile,
        &theta_profile,
        &ua,
        &va,
        &wa,
        &ppi,
        None,
        &tha,
        &qa,
        &base,
        &surface,
        PerturbationPolicy::PreserveRestartGrid,
    )?;
    for name in ["radbcw", "radbce", "radbcs", "radbcn"] {
        let carrier = mutable
            .carriers
            .iter()
            .find(|carrier| carrier.name == name)
            .ok_or_else(|| invalid(format!("fresh boundary pack lacks {name}")))?;
        if carrier.values.iter().any(|value| value.to_bits() != 0) {
            return Err(invalid(format!(
                "fresh radiation history {name} is not all positive zero"
            )));
        }
    }
    let density = compute_child_live_density(
        [CHILD_NZ, CHILD_NY, CHILD_NX],
        &ppi,
        &tha,
        &qa,
        &base,
        &c1,
        &c2,
        child_bottom,
        child_top,
    )?;
    let terrain_open = assemble_terrain_open(base, surface, mutable.carriers)?;
    validate_live_density(&density)?;
    let geopotential = zf.iter().map(|height| *height * G).collect::<Vec<_>>();
    if geopotential
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(invalid("child zf geopotential is non-finite or negative"));
    }
    Ok(Prepared {
        source_product_manifest_sha256,
        source_product_payload_sha256,
        source_product_receipt_sha256,
        source_product_restart_sha256,
        source_artifact_sha256,
        source_contract_sha256,
        atmosphere_manifest_sha256,
        atmosphere_arena_sha256,
        atmosphere_receipt_sha256,
        parent_checkpoint_sha256,
        parent_plan_sha256,
        terrain_manifest_sha256,
        terrain_payload_sha256,
        terrain_receipt_sha256,
        static_manifest_sha256,
        static_validation_sha256,
        static_zh_sha256,
        static_zf_sha256,
        terrain_open,
        live_density: density.carriers,
        geopotential,
        source_profile_hashes,
        source_z_sha256,
        child_bottom,
        child_top,
        producer_source_sha256,
        producer_executable_sha256,
    })
}

fn read_source_profiles(
    source: &RestoredProductRestart,
) -> Result<(SourceProfileValues, SourceProfileHashes)> {
    let names = ["pi0", "prs0", "qv0", "rho0", "th0"];
    let mut bytes = source.read_carriers_f32le_exact(&names)?;
    let mut values = BTreeMap::new();
    let mut hashes = BTreeMap::new();
    for name in names {
        let carrier = source
            .carrier(name)
            .ok_or_else(|| invalid(format!("source product lacks {name}")))?;
        if carrier.shape != [CHILD_NZ] || carrier.nbytes != (CHILD_NZ * 4) as u64 {
            return Err(invalid(format!("source base profile {name} shape drifted")));
        }
        hashes.insert(name.to_string(), carrier.sha256.clone());
        let raw = bytes
            .remove(name)
            .ok_or_else(|| invalid(format!("source profile extraction lost {name}")))?;
        values.insert(name.to_string(), decode_f32(raw, CHILD_NZ, name)?);
    }
    Ok((values, hashes))
}

fn derive_and_validate_child_metrics(
    terrain: &[f32],
    zh: &[f32],
    zf: &[f32],
    sigma: &[f32],
    sigmaf: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut c1 = vec![0.0_f32; FACE_WORDS];
    let mut c2 = vec![0.0_f32; FACE_WORDS];
    for j in 0..CHILD_NY {
        for i in 0..CHILD_NX {
            let column = j * CHILD_NX + i;
            let surface = terrain[column];
            let metrics = derive_stretched_column_metrics(surface, sigma, sigmaf, CHILD_TOP_M)?;
            for k in 0..CHILD_NZ {
                if metrics.zh[k].to_bits() != zh[k * TERRAIN_WORDS + column].to_bits() {
                    return Err(invalid(
                        "child zh does not match the sealed terrain/stretch transform",
                    ));
                }
            }
            for face in 0..=CHILD_NZ {
                if metrics.zf[face].to_bits() != zf[face * TERRAIN_WORDS + column].to_bits() {
                    return Err(invalid(
                        "child zf does not match the sealed terrain/stretch transform",
                    ));
                }
                let p = face * TERRAIN_WORDS + column;
                c1[p] = metrics.c1[face];
                c2[p] = metrics.c2[face];
            }
        }
    }
    Ok((c1, c2))
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StretchedColumnMetrics {
    pub zh: Vec<f32>,
    pub zf: Vec<f32>,
    pub c1: Vec<f32>,
    pub c2: Vec<f32>,
}

pub(crate) fn derive_stretched_column_metrics(
    surface: f32,
    sigma: &[f32],
    sigmaf: &[f32],
    top_height: f32,
) -> Result<StretchedColumnMetrics> {
    let nz = sigma.len();
    let mut zh = vec![0.0_f32; nz];
    let mut zf = vec![0.0_f32; nz + 1];
    let mut c1 = vec![0.0_f32; nz + 1];
    let mut c2 = vec![0.0_f32; nz + 1];
    fill_stretched_column_metrics(
        surface, sigma, sigmaf, top_height, &mut zh, &mut zf, &mut c1, &mut c2,
    )?;
    Ok(StretchedColumnMetrics { zh, zf, c1, c2 })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_stretched_column_metrics(
    surface: f32,
    sigma: &[f32],
    sigmaf: &[f32],
    top_height: f32,
    zh: &mut [f32],
    zf: &mut [f32],
    c1: &mut [f32],
    c2: &mut [f32],
) -> Result<()> {
    let nz = sigma.len();
    if nz < 4
        || sigmaf.len() != nz + 1
        || zh.len() != nz
        || zf.len() != nz + 1
        || c1.len() != nz + 1
        || c2.len() != nz + 1
        || !surface.is_finite()
        || !top_height.is_finite()
        || top_height <= surface
        || sigma.iter().any(|value| !value.is_finite())
        || sigmaf.iter().any(|value| !value.is_finite())
    {
        return Err(invalid("stretched child column metric inputs are invalid"));
    }
    for (target, value) in zh.iter_mut().zip(sigma) {
        *target = surface + *value * (top_height - surface) / top_height;
    }
    for (target, value) in zf.iter_mut().zip(sigmaf) {
        *target = surface + *value * (top_height - surface) / top_height;
    }
    let below = zf[0] - (zf[1] - zf[0]);
    let above = zf[nz] + (zf[nz] - zf[nz - 1]);
    let bottom_mass = 0.5 * (below + zf[0]);
    let top_mass = 0.5 * (zf[nz] + above);
    for face in 0..=nz {
        let low = if face == 0 { bottom_mass } else { zh[face - 1] };
        let high = if face == nz { top_mass } else { zh[face] };
        let span = high - low;
        c2[face] = (zf[face] - low) / span;
        c1[face] = 1.0 - c2[face];
        if !c1[face].is_finite() || !c2[face].is_finite() || span <= 0.0 {
            return Err(invalid("child c1/c2 metric reconstruction is invalid"));
        }
    }
    Ok(())
}

fn assemble_terrain_open(
    base: TerrainBaseFrame,
    surface: SurfaceHaloFrame,
    mutable: Vec<MutableCarrier>,
) -> Result<Vec<MutableCarrier>> {
    let mass_shape = vec![CHILD_NZ + 2, CHILD_NY + 6, CHILD_NX + 6];
    let surface_shape = vec![CHILD_NY + 6, CHILD_NX + 6];
    let TerrainBaseFrame {
        pi0,
        prs0,
        th0,
        thv0,
        rth0,
        qv0,
        rho0,
        rr0,
        rf0,
        rrf0,
        rho0s,
        pi0s,
        prs0s,
        rth0s,
    } = base;
    let SurfaceHaloFrame {
        terrain_msl_halo,
        gz_halo,
        rgz_halo,
        gzu_halo,
        rgzu_halo,
        gzv_halo,
        rgzv_halo,
        dzdx_halo,
        dzdy_halo,
    } = surface;
    let mut by_name = BTreeMap::new();
    for (name, values) in [
        ("pi0", pi0),
        ("prs0", prs0),
        ("th0", th0),
        ("thv0", thv0),
        ("rth0", rth0),
        ("qv0", qv0),
        ("rho0", rho0),
        ("rr0", rr0),
        ("rf0", rf0),
        ("rrf0", rrf0),
    ] {
        insert_unique(
            &mut by_name,
            carrier(
                name,
                "source-backed stretched terrain base",
                mass_shape.clone(),
                values,
            ),
        )?;
    }
    for (name, values) in [
        ("rho0s", rho0s),
        ("pi0s", pi0s),
        ("prs0s", prs0s),
        ("rth0s", rth0s),
        ("terrain_msl_halo", terrain_msl_halo),
        ("gz", gz_halo),
        ("rgz", rgz_halo),
        ("gzu", gzu_halo),
        ("rgzu", rgzu_halo),
        ("gzv", gzv_halo),
        ("rgzv", rgzv_halo),
        ("dzdx", dzdx_halo),
        ("dzdy", dzdy_halo),
    ] {
        insert_unique(
            &mut by_name,
            carrier(
                name,
                "source-valid fixed-child surface/metric halo",
                surface_shape.clone(),
                values,
            ),
        )?;
    }
    for value in mutable {
        insert_unique(&mut by_name, value)?;
    }
    let specs = terrain_open_carrier_specs(CHILD_NZ, CHILD_NY, CHILD_NX)?;
    let spec_by_name = specs
        .iter()
        .map(|spec| (spec.name(), spec.shape()))
        .collect::<BTreeMap<_, _>>();
    let mut ordered = Vec::with_capacity(TERRAIN_OPEN_CARRIER_NAMES.len());
    for name in TERRAIN_OPEN_CARRIER_NAMES {
        let value = by_name
            .remove(name)
            .ok_or_else(|| invalid(format!("child terrain/open pack lacks {name}")))?;
        let expected = spec_by_name
            .get(name)
            .ok_or_else(|| invalid(format!("terrain/open ABI lacks {name}")))?;
        if value.shape.as_slice() != *expected || value.values.len() != product(expected)? {
            return Err(invalid(format!(
                "child terrain/open carrier {name} shape differs from the v1 ABI"
            )));
        }
        ordered.push(value);
    }
    if !by_name.is_empty() || ordered.len() != 50 {
        return Err(invalid("child terrain/open carrier inventory drifted"));
    }
    Ok(ordered)
}

fn validate_live_density(density: &LiveDensityFrame) -> Result<()> {
    if density.carriers.len() != CHILD_LIVE_DENSITY_CARRIER_NAMES.len() {
        return Err(invalid("child live-density carrier count drifted"));
    }
    for (carrier, expected_name) in density
        .carriers
        .iter()
        .zip(CHILD_LIVE_DENSITY_CARRIER_NAMES)
    {
        let expected_shape = if expected_name == "rf" {
            vec![CHILD_NZ + 1, CHILD_NY, CHILD_NX]
        } else {
            vec![CHILD_NZ, CHILD_NY, CHILD_NX]
        };
        if carrier.name != expected_name
            || carrier.shape != expected_shape
            || carrier.values.len() != product(&expected_shape)?
            || carrier
                .values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(invalid(format!(
                "child live-density carrier {expected_name} is invalid"
            )));
        }
    }
    Ok(())
}

fn carrier(
    name: &'static str,
    role: &'static str,
    shape: Vec<usize>,
    values: Vec<f32>,
) -> MutableCarrier {
    MutableCarrier {
        name,
        role,
        shape,
        live_region: "complete source allocation",
        values,
    }
}

fn insert_unique(
    values: &mut BTreeMap<&'static str, MutableCarrier>,
    carrier: MutableCarrier,
) -> Result<()> {
    if values.insert(carrier.name, carrier).is_some() {
        return Err(invalid("child auxiliary carrier name is duplicated"));
    }
    Ok(())
}

pub(crate) fn extrapolation_weights(sigma: &[f32], sigmaf: &[f32], top: bool) -> Result<[f32; 3]> {
    let nz = sigma.len();
    if nz < 4 || sigmaf.len() != nz + 1 {
        return Err(invalid("child Fornberg profile geometry is invalid"));
    }
    let denominator = if top {
        sigmaf[nz] - sigmaf[nz - 1]
    } else {
        sigmaf[1] - sigmaf[0]
    };
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(invalid("child Fornberg endpoint spacing is invalid"));
    }
    let x0 = f64::from(if top {
        sigmaf[nz] / denominator
    } else {
        sigmaf[0] / denominator
    });
    let mut alpha = [0.0_f64; 4];
    for n in 0..4 {
        alpha[n] = f64::from(if top {
            sigma[nz - 1 - n] / denominator
        } else {
            sigma[n] / denominator
        });
    }
    let delta = fornberg_delta(alpha, x0)?;
    Ok([
        delta[2][1][0] as f32,
        delta[2][1][1] as f32,
        delta[2][1][2] as f32,
    ])
}

fn fornberg_delta(alpha: [f64; 4], x0: f64) -> Result<[[[f64; 4]; 6]; 4]> {
    let mut delta = [[[0.0_f64; 4]; 6]; 4];
    delta[0][1][0] = 1.0;
    let mut b1 = 1.0;
    for n in 1..=3 {
        let mut b2 = 1.0;
        for nu in 0..n {
            let b3 = alpha[n] - alpha[nu];
            if !b3.is_finite() || b3 == 0.0 {
                return Err(invalid("child Fornberg nodes are duplicated"));
            }
            b2 *= b3;
            delta[n - 1][n + 1][nu] = 0.0;
            for m in 0..=n.min(4) {
                delta[n][m + 1][nu] = ((alpha[n] - x0) * delta[n - 1][m + 1][nu]
                    - m as f64 * delta[n - 1][m][nu])
                    / b3;
            }
        }
        for m in 0..=n.min(4) {
            delta[n][m + 1][n] = (b1 / b2)
                * (m as f64 * delta[n - 1][m][n - 1]
                    - (alpha[n - 1] - x0) * delta[n - 1][m + 1][n - 1]);
        }
        b1 = b2;
    }
    Ok(delta)
}

fn validate_authority(directory: &Path) -> Result<()> {
    if !directory.is_dir() || fs::symlink_metadata(directory)?.file_type().is_symlink() {
        return Err(invalid("CM1 authority root must be a real directory"));
    }
    for (name, expected, _) in AUTHORITY_SOURCE {
        require_hash(
            &directory.join("src").join(name),
            SOURCE_LIMIT,
            expected,
            &format!("CM1 authority src/{name}"),
        )?;
    }
    Ok(())
}

fn verify_inventory(directory: &Path) -> Result<()> {
    verify_exact_directory(
        directory,
        &[
            CHILD_AUXILIARY_GEOPOTENTIAL_FILE,
            CHILD_AUXILIARY_LIVE_DENSITY_FILE,
            CHILD_AUXILIARY_MANIFEST_FILE,
            CHILD_AUXILIARY_RECEIPT_FILE,
            CHILD_AUXILIARY_TERRAIN_OPEN_FILE,
        ],
        "child auxiliary",
    )
}

fn verify_exact_directory(directory: &Path, expected: &[&str], label: &str) -> Result<()> {
    if !directory.is_dir() || fs::symlink_metadata(directory)?.file_type().is_symlink() {
        return Err(invalid(format!("{label} must be a real directory")));
    }
    let expected = expected
        .iter()
        .map(|value| (*value).to_string())
        .collect::<BTreeSet<_>>();
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_name = entry
            .file_name()
            .into_string()
            .map_err(|_| invalid(format!("{label} contains a non-UTF-8 name")))?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(invalid(format!("{label} contains a non-regular entry")));
        }
        observed.insert(file_name);
    }
    if observed != expected {
        return Err(invalid(format!("{label} directory inventory differs")));
    }
    Ok(())
}

fn member<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a JsonValue> {
    value
        .get(key)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn json_sha256(value: &JsonValue, key: &str, label: &str) -> Result<String> {
    json_sha256_at(value, &[key], label)
}

fn require_hash(path: &Path, maximum: u64, expected: &str, label: &str) -> Result<()> {
    validate_sha256(expected, label)?;
    let actual = hash_regular_bounded(path, maximum, label)?;
    if actual != expected {
        return Err(invalid(format!(
            "{label} SHA-256 is {actual}; independently expected {expected}"
        )));
    }
    Ok(())
}

fn hash_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<String> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > maximum {
        return Err(invalid(format!("{label} is not a bounded regular file")));
    }
    let mut reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut hasher = Sha256::default();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut total = 0u64;
    loop {
        let amount = reader.read(&mut buffer)?;
        if amount == 0 {
            break;
        }
        total += amount as u64;
        if total > maximum {
            return Err(invalid(format!("{label} exceeds its byte bound")));
        }
        hasher.update(&buffer[..amount]);
    }
    if total != metadata.len() {
        return Err(invalid(format!("{label} size changed while hashing")));
    }
    Ok(sha256::hex(&hasher.finalize()))
}

fn read_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > maximum {
        return Err(invalid(format!("{label} is not a bounded regular file")));
    }
    let length = usize::try_from(metadata.len())
        .map_err(|_| invalid(format!("{label} size does not fit this platform")))?;
    let mut bytes = Vec::with_capacity(length);
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() != length {
        return Err(invalid(format!("{label} size changed while reading")));
    }
    Ok(bytes)
}

fn read_f32_file(path: &Path, words: usize, expected: &str, label: &str) -> Result<Vec<f32>> {
    let nbytes = words
        .checked_mul(4)
        .ok_or_else(|| invalid(format!("{label} byte count overflows")))?;
    let bytes = read_regular_bounded(path, nbytes as u64, label)?;
    if bytes.len() != nbytes || sha256::digest_hex(&bytes) != expected {
        return Err(invalid(format!(
            "{label} differs from its exact SHA-256/shape pin"
        )));
    }
    decode_f32(bytes, words, label)
}

fn take_f32(values: &mut BTreeMap<String, Vec<u8>>, name: &str, words: usize) -> Result<Vec<f32>> {
    let bytes = values
        .remove(name)
        .ok_or_else(|| invalid(format!("child atmosphere extraction lost {name}")))?;
    decode_f32(bytes, words, name)
}

fn decode_f32(bytes: Vec<u8>, words: usize, label: &str) -> Result<Vec<f32>> {
    if bytes.len()
        != words
            .checked_mul(4)
            .ok_or_else(|| invalid("f32 byte count overflows"))?
    {
        return Err(invalid(format!("{label} f32 byte count differs")));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("{label} contains a non-finite f32 word")));
    }
    Ok(values)
}

fn product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid("child auxiliary shape product overflows"))
    })
}

fn hash_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::default();
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    sha256::hex(&hasher.finalize())
}

fn write_arena_new(path: &Path, carriers: &[MutableCarrier]) -> Result<(Vec<Record>, String, u64)> {
    if fs::symlink_metadata(path).is_ok() {
        return Err(invalid(format!("refusing to replace {}", path.display())));
    }
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);
    let result = assemble_arena(carriers, &mut writer)?;
    writer.flush()?;
    let file = writer
        .into_inner()
        .map_err(|error| invalid(format!("could not finalize child auxiliary arena: {error}")))?;
    file.sync_all()?;
    if file.metadata()?.len() != result.2 {
        return Err(invalid("child auxiliary arena size changed after write"));
    }
    Ok(result)
}

fn compare_arena(path: &Path, carriers: &[MutableCarrier]) -> Result<(Vec<Record>, String, u64)> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > OUTPUT_LIMIT {
        return Err(invalid(
            "child auxiliary payload is not a bounded regular file",
        ));
    }
    let reader = BufReader::with_capacity(1024 * 1024, File::open(path)?);
    let mut comparator = CompareWriter::new(reader);
    let result = assemble_arena(carriers, &mut comparator)?;
    comparator.finish()?;
    if result.2 != metadata.len() {
        return Err(invalid(
            "child auxiliary payload size differs from recomputation",
        ));
    }
    Ok(result)
}

fn assemble_arena<W: Write>(
    carriers: &[MutableCarrier],
    writer: &mut W,
) -> Result<(Vec<Record>, String, u64)> {
    let mut records = Vec::with_capacity(carriers.len());
    let mut aggregate = Sha256::default();
    let mut offset = 0u64;
    let mut chunk = Vec::with_capacity(1024 * 1024);
    for carrier in carriers {
        if carrier.values.len() != product(&carrier.shape)? {
            return Err(invalid(format!(
                "carrier {} shape differs from its words",
                carrier.name
            )));
        }
        if carrier.values.is_empty() || carrier.values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "carrier {} is empty or non-finite",
                carrier.name
            )));
        }
        let mut carrier_hasher = Sha256::default();
        let mut minimum = carrier.values[0];
        let mut maximum = carrier.values[0];
        let mut nonzero = 0usize;
        chunk.clear();
        for value in &carrier.values {
            minimum = minimum.min(*value);
            maximum = maximum.max(*value);
            nonzero += usize::from(value.to_bits() != 0);
            let bytes = value.to_bits().to_le_bytes();
            carrier_hasher.update(&bytes);
            aggregate.update(&bytes);
            chunk.extend_from_slice(&bytes);
            if chunk.len() >= 1024 * 1024 {
                writer.write_all(&chunk)?;
                chunk.clear();
            }
        }
        if !chunk.is_empty() {
            writer.write_all(&chunk)?;
        }
        let nbytes = (carrier.values.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid("carrier byte count overflowed"))?;
        records.push(Record {
            name: carrier.name.to_string(),
            shape: carrier.shape.clone(),
            offset_nbytes: offset,
            nbytes,
            sha256: sha256::hex(&carrier_hasher.finalize()),
            finite_count: carrier.values.len(),
            nonzero_count: nonzero,
            minimum_f32_bits: minimum.to_bits(),
            maximum_f32_bits: maximum.to_bits(),
        });
        offset = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("arena byte count overflowed"))?;
    }
    if offset > OUTPUT_LIMIT {
        return Err(invalid("child auxiliary arena exceeds its byte limit"));
    }
    Ok((records, sha256::hex(&aggregate.finalize()), offset))
}

struct CompareWriter<R> {
    reader: R,
    offset: u64,
}

impl<R: Read> CompareWriter<R> {
    fn new(reader: R) -> Self {
        Self { reader, offset: 0 }
    }

    fn finish(mut self) -> Result<()> {
        let mut extra = [0u8; 1];
        if self.reader.read(&mut extra)? != 0 {
            return Err(invalid("child auxiliary payload has trailing bytes"));
        }
        Ok(())
    }
}

impl<R: Read> Write for CompareWriter<R> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let mut observed = vec![0u8; bytes.len()];
        self.reader.read_exact(&mut observed)?;
        if observed != bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "child auxiliary payload differs at or before byte {}",
                    self.offset
                ),
            ));
        }
        self.offset += bytes.len() as u64;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!("{label} is not lowercase SHA-256")));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn manifest_value(
    prepared: &Prepared,
    terrain_records: &[Record],
    terrain_sha256: &str,
    terrain_nbytes: u64,
    density_records: &[Record],
    density_sha256: &str,
    density_nbytes: u64,
    geopotential_records: &[Record],
    geopotential_sha256: &str,
    geopotential_nbytes: u64,
) -> JsonValue {
    object(vec![
        ("artifact_schema", string(CHILD_AUXILIARY_SCHEMA)),
        (
            "authority",
            object(vec![
                (
                    "sources",
                    JsonValue::Array(
                        AUTHORITY_SOURCE
                            .iter()
                            .map(|(name, sha, lines)| {
                                object(vec![
                                    ("anchors", string(lines)),
                                    ("path", string(&format!("src/{name}"))),
                                    ("sha256", string(sha)),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
            ]),
        ),
        (
            "boundary_initialization",
            object(vec![
                ("all_four_boundaries_open", JsonValue::Bool(true)),
                ("parent_history_reused", JsonValue::Bool(false)),
                (
                    "radiation_history",
                    string("radbcw/radbce/radbcs/radbcn all exact positive-zero fresh child state"),
                ),
                ("source_valid_family_halos", JsonValue::Bool(true)),
            ]),
        ),
        (
            "child_grid",
            object(vec![
                ("dx_m_f32_bits", f32_bits(50.0)),
                ("dy_m_f32_bits", f32_bits(50.0)),
                ("nx", number(CHILD_NX)),
                ("ny", number(CHILD_NY)),
                ("nz", number(CHILD_NZ)),
                ("stretch_grid_sha256", string(CHILD_STRETCH_GRID_SHA256)),
                ("top_m_f32_bits", f32_bits(CHILD_TOP_M)),
            ]),
        ),
        ("claim_limit", string(CHILD_AUXILIARY_CLAIM_LIMIT)),
        (
            "closure_disposition",
            object(vec![
                ("action", string("REQUIRED_FRESH_CHILD_CLOSURE_INITIALIZATION")),
                (
                    "blocked_on",
                    string_array(&["nm", "defv", "defh", "rmf", "ruh", "rvh"]),
                ),
                (
                    "contract",
                    string("restart_read.F:1206-1211 excludes old TKE; init3d.F:2688-2756 diagnoses fresh tkea then copies tke3d and zeros tketen"),
                ),
                ("consumer_abi", string(FRESH_TKE_ABI)),
                ("consumer_commit", string(FRESH_TKE_COMMIT)),
                ("consumer_kernel_sha256", string(FRESH_TKE_KERNEL_SHA256)),
                ("consumer_receipt_sha256", string(FRESH_TKE_RECEIPT_SHA256)),
                ("consumer_schema", string(FRESH_TKE_SCHEMA)),
                ("status", string("REQUIRED_DOWNSTREAM_BLOCKED_ON_NM_DEFV_DEFH")),
                ("synthesized_or_zeroed_tkea", JsonValue::Bool(false)),
            ]),
        ),
        (
            "outputs",
            object(vec![
                (
                    "geopotential",
                    output_value(
                        CHILD_AUXILIARY_GEOPOTENTIAL_FILE,
                        GEOPOTENTIAL_ABI,
                        geopotential_records,
                        geopotential_sha256,
                        geopotential_nbytes,
                    ),
                ),
                (
                    "live_density",
                    output_value(
                        CHILD_AUXILIARY_LIVE_DENSITY_FILE,
                        CHILD_LIVE_DENSITY_ABI,
                        density_records,
                        density_sha256,
                        density_nbytes,
                    ),
                ),
                (
                    "terrain_open",
                    object(vec![
                        ("file", string(CHILD_AUXILIARY_TERRAIN_OPEN_FILE)),
                        ("abi", string(TERRAIN_OPEN_CARRIER_ABI)),
                        ("identity_schema", string(TERRAIN_OPEN_IDENTITY_SCHEMA)),
                        ("receipt_schema", string(TERRAIN_OPEN_RECEIPT_SCHEMA)),
                        ("nbytes", u64_number(terrain_nbytes)),
                        ("sha256", string(terrain_sha256)),
                        ("carriers", record_array(terrain_records)),
                    ]),
                ),
            ]),
        ),
        (
            "parent",
            object(vec![
                (
                    "checkpoint_sha256",
                    string(&prepared.parent_checkpoint_sha256),
                ),
                ("plan_sha256", string(&prepared.parent_plan_sha256)),
            ]),
        ),
        (
            "producer",
            object(vec![
                ("executable_sha256", string(&prepared.producer_executable_sha256)),
                ("source_sha256", string(&prepared.producer_source_sha256)),
            ]),
        ),
        (
            "source_atmosphere",
            object(vec![
                ("arena_sha256", string(&prepared.atmosphere_arena_sha256)),
                (
                    "manifest_sha256",
                    string(&prepared.atmosphere_manifest_sha256),
                ),
                (
                    "receipt_sha256",
                    string(&prepared.atmosphere_receipt_sha256),
                ),
                ("qa_nonnegative_required", JsonValue::Bool(true)),
            ]),
        ),
        (
            "source_base_profiles",
            object(vec![
                (
                    "carrier_sha256",
                    JsonValue::Object(
                        prepared
                            .source_profile_hashes
                            .iter()
                            .map(|(name, value)| (name.clone(), string(value)))
                            .collect(),
                    ),
                ),
                ("mass_z_f32_sha256", string(&prepared.source_z_sha256)),
                ("mass_z_rule", string("f32((k+0.5)*200 m), k=0..74")),
                (
                    "product_manifest_sha256",
                    string(&prepared.source_product_manifest_sha256),
                ),
                (
                    "product_payload_sha256",
                    string(&prepared.source_product_payload_sha256),
                ),
                (
                    "product_receipt_sha256",
                    string(&prepared.source_product_receipt_sha256),
                ),
                (
                    "product_restart_sha256",
                    string(&prepared.source_product_restart_sha256),
                ),
                (
                    "source_artifact_sha256",
                    string(&prepared.source_artifact_sha256),
                ),
                (
                    "source_contract_sha256",
                    string(&prepared.source_contract_sha256),
                ),
                (
                    "transformation",
                    string("base.F:625-749 sounding thv/pisnd/psnd/temperature/RH reconstruction, interpolation to each child terrain-following zh, qv recovery from interpolated RH, then child-column hydrostatic pi0"),
                ),
            ]),
        ),
        (
            "source_sequence",
            string_array(&[
                "base.F terrain sounding and hydrostatic base",
                "preserve restart_read interpolated ppi/tha/qa and source A-to-current copies",
                "getset psolver3 overwrites ppx with ppi, then applies bcs(ppx)",
                "apply source-family fixed four-open halos and terrain lower-w boundary",
                "init3d getset psolver3 pressure/density then rr/rf",
                "publish fresh positive-zero radiation histories",
            ]),
        ),
        (
            "static_inputs",
            object(vec![
                ("abi", string(STATIC_HEIGHT_ABI)),
                ("manifest_sha256", string(&prepared.static_manifest_sha256)),
                (
                    "terrain_manifest_sha256",
                    string(&prepared.terrain_manifest_sha256),
                ),
                (
                    "terrain_payload_sha256",
                    string(&prepared.terrain_payload_sha256),
                ),
                (
                    "terrain_receipt_sha256",
                    string(&prepared.terrain_receipt_sha256),
                ),
                (
                    "validation_receipt_sha256",
                    string(&prepared.static_validation_sha256),
                ),
                ("zf_sha256", string(&prepared.static_zf_sha256)),
                ("zh_sha256", string(&prepared.static_zh_sha256)),
            ]),
        ),
        (
            "stretched_endpoint_weights",
            object(vec![
                ("bottom_f32_bits", f32_bits_array(prepared.child_bottom)),
                ("top_f32_bits", f32_bits_array(prepared.child_top)),
            ]),
        ),
        ("model", string("GM1")),
        ("schema", string(CHILD_AUXILIARY_MANIFEST_SCHEMA)),
        ("status", string(CHILD_AUXILIARY_STATUS)),
    ])
}

fn receipt_value(receipt: &ChildAuxiliaryReceipt) -> JsonValue {
    object(vec![
        ("artifact_schema", string(CHILD_AUXILIARY_SCHEMA)),
        ("claim_limit", string(CHILD_AUXILIARY_CLAIM_LIMIT)),
        ("geopotential_sha256", string(&receipt.geopotential_sha256)),
        ("live_density_sha256", string(&receipt.live_density_sha256)),
        ("manifest_sha256", string(&receipt.manifest_sha256)),
        ("model", string("GM1")),
        (
            "producer_executable_sha256",
            string(&receipt.producer_executable_sha256),
        ),
        (
            "producer_source_sha256",
            string(&receipt.producer_source_sha256),
        ),
        ("schema", string(CHILD_AUXILIARY_RECEIPT_SCHEMA)),
        ("status", string(CHILD_AUXILIARY_STATUS)),
        ("terrain_open_sha256", string(&receipt.terrain_open_sha256)),
    ])
}

fn output_value(file: &str, abi: &str, records: &[Record], sha256: &str, nbytes: u64) -> JsonValue {
    object(vec![
        ("abi", string(abi)),
        ("carriers", record_array(records)),
        ("file", string(file)),
        ("nbytes", u64_number(nbytes)),
        ("sha256", string(sha256)),
    ])
}

fn record_array(records: &[Record]) -> JsonValue {
    JsonValue::Array(
        records
            .iter()
            .map(|record| {
                object(vec![
                    ("dtype", string("little-endian-f32")),
                    ("finite_count", number(record.finite_count)),
                    (
                        "maximum_f32_bits",
                        string(&format!("{:08x}", record.maximum_f32_bits)),
                    ),
                    (
                        "minimum_f32_bits",
                        string(&format!("{:08x}", record.minimum_f32_bits)),
                    ),
                    ("name", string(&record.name)),
                    ("nbytes", u64_number(record.nbytes)),
                    ("nonzero_count", number(record.nonzero_count)),
                    ("offset_nbytes", u64_number(record.offset_nbytes)),
                    ("order", string("C")),
                    ("sha256", string(&record.sha256)),
                    (
                        "shape",
                        JsonValue::Array(record.shape.iter().map(|value| number(*value)).collect()),
                    ),
                ])
            })
            .collect(),
    )
}

fn f32_bits_array(values: [f32; 3]) -> JsonValue {
    JsonValue::Array(values.into_iter().map(f32_bits).collect())
}

fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    )
}

fn string(value: &str) -> JsonValue {
    JsonValue::String(value.to_string())
}

fn number(value: usize) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn u64_number(value: u64) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn f32_bits(value: f32) -> JsonValue {
    string(&format!("{:08x}", value.to_bits()))
}

fn string_array(values: &[&str]) -> JsonValue {
    JsonValue::Array(values.iter().map(|value| string(value)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child_remap::{ACCEPTED_50M_CHILD_GRID, HARD_5M_CHILD_GRID};

    #[test]
    fn parameterized_auxiliary_layout_preserves_the_historical_50m_bytes() {
        let layout = child_auxiliary_base_layout_for_grid(ACCEPTED_50M_CHILD_GRID).unwrap();
        assert_eq!(layout.terrain_open_carrier_count, 50);
        assert_eq!(layout.terrain_open_nbytes, 953_971_136);
        assert_eq!(layout.live_density_carrier_count, 4);
        assert_eq!(layout.live_density_nbytes, 108_360_000);
        assert_eq!(layout.geopotential_nbytes, 27_360_000);
        assert_eq!(layout.u0_nbytes, 27_090_000);
        assert_eq!(layout.v0_nbytes, 27_090_000);
        assert_eq!(layout.base_wind_nbytes, 54_180_000);
        assert_eq!(layout.total_nbytes, 1_143_871_136);
    }

    #[test]
    fn hard_5m_auxiliary_layout_matches_the_frozen_capacity_ledger() {
        let layout = child_auxiliary_base_layout_for_grid(HARD_5M_CHILD_GRID).unwrap();
        assert_eq!(layout.terrain_open_carrier_count, 50);
        assert_eq!(
            layout.terrain_open_nbytes,
            HARD_5M_CHILD_TERRAIN_OPEN_NBYTES
        );
        assert_eq!(
            layout.live_density_nbytes,
            HARD_5M_CHILD_LIVE_DENSITY_NBYTES
        );
        assert_eq!(
            layout.geopotential_nbytes,
            HARD_5M_CHILD_GEOPOTENTIAL_NBYTES
        );
        assert_eq!(layout.u0_nbytes, HARD_5M_CHILD_U0_NBYTES);
        assert_eq!(layout.v0_nbytes, HARD_5M_CHILD_V0_NBYTES);
        assert_eq!(layout.base_wind_nbytes, HARD_5M_CHILD_BASE_WIND_NBYTES);
        assert_eq!(layout.total_nbytes, HARD_5M_CHILD_AUXILIARY_BASE_NBYTES);
    }

    #[test]
    fn capacity_gate_requires_one_copy_and_preserves_the_requested_reserve() {
        let free_bytes = 905_475_260_416;
        let reserve_bytes = 8 * 1024 * 1024 * 1024;
        let gate =
            admit_child_auxiliary_base_capacity(HARD_5M_CHILD_GRID, free_bytes, reserve_bytes, 1)
                .unwrap();
        assert_eq!(gate.post_publish_free_bytes, 794_085_442_880);
        assert_eq!(gate.post_publish_above_reserve_bytes, 785_495_508_288);

        let duplicate =
            admit_child_auxiliary_base_capacity(HARD_5M_CHILD_GRID, free_bytes, reserve_bytes, 2)
                .unwrap_err();
        assert!(duplicate.to_string().contains("exactly one"));

        let insufficient = admit_child_auxiliary_base_capacity(
            HARD_5M_CHILD_GRID,
            HARD_5M_CHILD_AUXILIARY_BASE_NBYTES + reserve_bytes - 1,
            reserve_bytes,
            1,
        )
        .unwrap_err();
        assert!(insufficient.to_string().contains("destination volume"));
    }

    #[test]
    fn source_mass_height_profile_matches_cross_language_pin() {
        let values = (0..CHILD_NZ)
            .map(|k| (k as f32 + 0.5) * SOURCE_DZ_M)
            .collect::<Vec<_>>();
        assert_eq!(hash_f32(&values), SOURCE_MASS_Z_SHA256);
    }

    #[test]
    fn uniform_grid_fornberg_endpoints_are_exact() {
        let mass = (0..4).map(|k| 100.0 + 200.0 * k as f32).collect::<Vec<_>>();
        let faces = (0..=4).map(|k| 200.0 * k as f32).collect::<Vec<_>>();
        assert_eq!(
            extrapolation_weights(&mass, &faces, false)
                .unwrap()
                .map(f32::to_bits),
            [1.875_f32, -1.25_f32, 0.375_f32].map(f32::to_bits)
        );
        assert_eq!(
            extrapolation_weights(&mass, &faces, true)
                .unwrap()
                .map(f32::to_bits),
            [1.875_f32, -1.25_f32, 0.375_f32].map(f32::to_bits)
        );
    }

    #[test]
    fn selected_restart_pack_overwrites_ppx_with_ppi_and_zeros_histories() {
        let (nk, ny, nx) = (4usize, 2usize, 2usize);
        let padded_words = (nk + 2) * (ny + 6) * (nx + 6);
        let surface_words = (ny + 6) * (nx + 6);
        let base = TerrainBaseFrame {
            pi0: vec![0.9; padded_words],
            prs0: vec![90_000.0; padded_words],
            th0: vec![300.0; padded_words],
            thv0: vec![300.0; padded_words],
            rth0: vec![1.0 / 300.0; padded_words],
            qv0: vec![0.0; padded_words],
            rho0: vec![1.0; padded_words],
            rr0: vec![1.0; padded_words],
            rf0: vec![1.0; padded_words],
            rrf0: vec![1.0; padded_words],
            rho0s: vec![1.0; surface_words],
            pi0s: vec![0.9; surface_words],
            prs0s: vec![90_000.0; surface_words],
            rth0s: vec![1.0 / 300.0; surface_words],
        };
        let surface = SurfaceHaloFrame {
            terrain_msl_halo: vec![0.0; surface_words],
            gz_halo: vec![1.0; surface_words],
            rgz_halo: vec![1.0; surface_words],
            gzu_halo: vec![1.0; surface_words],
            rgzu_halo: vec![1.0; surface_words],
            gzv_halo: vec![1.0; surface_words],
            rgzv_halo: vec![1.0; surface_words],
            dzdx_halo: vec![0.0; surface_words],
            dzdy_halo: vec![0.0; surface_words],
        };
        let mass = nk * ny * nx;
        let ppi = vec![0.01; mass];
        let frame = pack_cold_start_mutable(
            [nk, ny, nx],
            HorizontalCrop::full(ny, nx),
            &vec![0.9; nk],
            &vec![300.0; nk],
            &vec![0.0; nk * ny * (nx + 1)],
            &vec![0.0; nk * (ny + 1) * nx],
            &vec![0.0; (nk + 1) * ny * nx],
            &ppi,
            None,
            &vec![0.0; mass],
            &vec![0.0; 6 * mass],
            &base,
            &surface,
            PerturbationPolicy::PreserveRestartGrid,
        )
        .unwrap();
        let ppi_out = frame
            .carriers
            .iter()
            .find(|value| value.name == "ppi")
            .unwrap();
        let ppx_out = frame
            .carriers
            .iter()
            .find(|value| value.name == "ppx")
            .unwrap();
        assert_eq!(ppi_out.values, ppx_out.values);
        for name in ["radbcw", "radbce", "radbcs", "radbcn"] {
            assert!(frame
                .carriers
                .iter()
                .find(|value| value.name == name)
                .unwrap()
                .values
                .iter()
                .all(|value| value.to_bits() == 0));
        }
    }
}
