//! Rust-owned restart-grid interpolation for the accepted GM1 50 m child.
//!
//! The numerical rule is the `interp_on_restart` branch in CM1 r21.1.  It
//! interpolates in the normalized `sigma`/`sigmaf` coordinates, evaluates the
//! eight weights as default REAL before widening them to double precision,
//! and performs the eight-term double-precision sum in source order.  The
//! official below-support behavior duplicates the first old plane.  This is a
//! new-child initialization artifact, not a raw-bit restart of the parent.

use crate::child_remap::{
    parent_fractional_xy_for_grid, read_validated_parent_plan, sealed_child_stretch_grid,
    validate_bound_parent_identity, ChildGridSpec, HorizontalStagger, ACCEPTED_50M_CHILD_GRID,
    CHILD_NX, CHILD_NY, CHILD_NZ, CHILD_REMAP_PLAN_SCHEMA, CHILD_STRETCH_ABI,
    CHILD_STRETCH_GRID_SHA256, CHILD_STRETCH_SCHEMA, CM1_R21_AUTHORITY_SHA256,
    FIRST_ADMITTED_PARENT_CHECKPOINT_SHA256, FIRST_ADMITTED_PARENT_PLAN_SHA256, HARD_5M_CHILD_GRID,
    PARENT_NX, PARENT_NY, PARENT_NZ,
};
use crate::child_static_heights::{
    child_static_height_nbytes_for_grid, ChildStaticHeightGridReceipt,
    ChildTerrainArtifactIdentity, CHILD_STATIC_HEIGHT_GRID_ABI,
};
use crate::forecast::ForecastArray;
use crate::gm1_checkpoint::{
    read_gm1_checkpoint, Gm1CheckpointConfig, Gm1SourceIdentities, RestoredGm1Checkpoint,
    GM1_F02_ROI_PRODUCT_MANIFEST_SHA256,
};
use crate::json::JsonValue;
use crate::sha256::{self, Sha256};
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const CHILD_ATMOSPHERE_SCHEMA: &str = "gm1.source-coordinate-50m-child-atmosphere/v1";
pub const CHILD_ATMOSPHERE_MANIFEST_SCHEMA: &str =
    "gm1.source-coordinate-50m-child-atmosphere-manifest/v1";
pub const CHILD_ATMOSPHERE_ABI: &str = "gm1.source-coordinate-50m-child-atmosphere-arena-f32/v1";
pub const CHILD_ATMOSPHERE_ARENA_FILE: &str = "arena.f32le";
pub const CHILD_ATMOSPHERE_MANIFEST_FILE: &str = "manifest.json";
pub const CHILD_ATMOSPHERE_RECEIPT_FILE: &str = "receipt.json";
pub const CHILD_ATMOSPHERE_STATUS: &str = "PASS_REAL_CM1_RESTART_GRID_CHILD_ATMOSPHERE";
pub const CHILD_ATMOSPHERE_STREAM_REPLAY_STATUS: &str = "PASS_CHILD_ATMOSPHERE_STREAM_RAW_REPLAY";
pub const CHILD_ATMOSPHERE_CLAIM_LIMIT: &str = "exact Rust-owned application of the CM1 r21.1 restart-grid normalized-coordinate interpolation and source-defined copy/zero dispositions to the accepted real pdefq-clean T+300 GM1 parent, plus an explicitly identified ArWen persistence-policy seam for eleven WSM6 component carriers; closure TKE, terrain-varying hydrostatic/base-state reconstruction, child open-boundary/radiation-history initialization, and time integration remain required downstream; this is not parent-to-child raw-bit restart identity and makes no forecast-skill, tornado, or performance claim";

pub const CHILD_ATMOSPHERE_GRID_SCHEMA: &str = "gm1.source-coordinate-child-atmosphere/v2";
pub const CHILD_ATMOSPHERE_GRID_MANIFEST_SCHEMA: &str =
    "gm1.source-coordinate-child-atmosphere-manifest/v2";
pub const CHILD_ATMOSPHERE_GRID_ABI: &str = "gm1.source-coordinate-child-atmosphere-arena-f32/v2";
pub const CHILD_ATMOSPHERE_GRID_STATUS: &str = "PASS_REAL_SOURCE_COORDINATE_CHILD_ATMOSPHERE";
pub const CHILD_ATMOSPHERE_GRID_CLAIM_LIMIT: &str = "exact bounded-memory Rust application of the CM1 r21.1 restart-grid normalized-coordinate interpolation and source-defined copy/positive-zero dispositions to one explicitly admitted source-coordinate child grid from the accepted real pdefq-clean T+300 GM1 parent, with eleven ArWen WSM6 persistence carriers separately identified; every atmospheric output derives from the sealed real parent with no synthetic detail or translation; closure TKE, terrain-varying hydrostatic/base-state reconstruction, child open-boundary/radiation-history initialization, and time integration remain required downstream; this is not parent-to-child raw-bit restart identity and makes no forecast-skill, storm, tornado, or performance claim";

pub const CM1_RESTART_READ_SHA256: &str =
    "72a041737a4025ce445cd303cd169eb6ff4c001a1140f9fca9e9cdb918884e2a";
pub const CM1_INIT3D_SHA256: &str =
    "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28";
pub const STATIC_HEIGHT_MANIFEST_SHA256: &str =
    "5459be4acf6ba3f8f83e6ff59bdfec11946d2c4846f60556bf02a1892289b9d6";
pub const STATIC_HEIGHT_VALIDATION_RECEIPT_SHA256: &str =
    "f90fcadbe9f75eacc66a96d56dc9584166b2e50641a80e311f358c4bb49a27a8";
pub const STATIC_HEIGHT_ABI: &str = "gm1.cuda-source-coordinate-50m-child-static-heights-f32/v1";
pub const STATIC_HEIGHT_ZH_SHA256: &str =
    "7fe66bd1639fb0cb4ad171be87b77484703a4d92ce868524e796e888a52a5159";
pub const STATIC_HEIGHT_ZF_SHA256: &str =
    "df5aef17965915406fe334600afe2f8f2e1ebb5d07665f71820eaac9bde4128f";
pub const CHILD_TERRAIN_MANIFEST_SHA256: &str =
    "bc6006491174c4b7594f4c12582749f3d3494015bba2eb692106005d3aeb391a";
pub const CHILD_TERRAIN_PAYLOAD_SHA256: &str =
    "afbec69ee1f23c881f5337f1eb7828eeb29ab7b34f7236a284980dc29681ba91";
pub const CHILD_TERRAIN_RECEIPT_SHA256: &str =
    "0c50ea83658b4ea47a711580200fe7280face00baba7b089d3e5572280004bbf";
pub const STATIC_HEIGHT_OPERATOR_COMMIT: &str = "77ca2d6073ee7b74b29748477f6ac48394cac049";
pub const SEALED_CHILD_ATMOSPHERE_MANIFEST_SHA256: &str =
    "88798478087b2c3d8d5a66a956e891ad5936741c3c47285bb243f709a90e30b8";
pub const SEALED_CHILD_ATMOSPHERE_ARENA_SHA256: &str =
    "59aa935cee8f7a69187af97ead3e2b29fd50e6ef665ca6b22ccb07bc9a34c3b7";
pub const SEALED_CHILD_ATMOSPHERE_RECEIPT_SHA256: &str =
    "17ca4f48a353d13ec69b6e45ca2e6180e7c8c125c60cf69a4b9d7fa702580c6c";
pub const SEALED_CHILD_ATMOSPHERE_PRODUCER_SOURCE_SHA256: &str =
    "5999371aba40bf1b3aaf1af00230f0a883bf7d9d51ff681455db5ae666a3e55b";
pub const SEALED_CHILD_ATMOSPHERE_PRODUCER_EXECUTABLE_SHA256: &str =
    "79c7305db869443e51eeaa4c2240224346dc4a4d39308ce4c5ac2e7a692f16b5";
pub const SEALED_CHILD_ATMOSPHERE_ARENA_NBYTES: u64 = 840_600_000;
pub const HARD_5M_CHILD_ATMOSPHERE_ARENA_NBYTES: u64 = 84_027_600_000;
pub const HARD_5M_CHILD_ATMOSPHERE_MANIFEST_SHA256: &str =
    "ba1391a3d9f4914f4b9a23f0d5021994d7587786829f4b85cb6eee0db5047447";
pub const HARD_5M_CHILD_ATMOSPHERE_ARENA_SHA256: &str =
    "36c945e488307900b30cf809ffd8c777d84f6cd5e4ecaa318dcecdc95fa23ce5";
pub const HARD_5M_CHILD_ATMOSPHERE_RECEIPT_SHA256: &str =
    "24790ddca9403c5066c69517e51442ff24d19d7f6dc38f999cabe96116983483";
pub const HARD_5M_CHILD_ATMOSPHERE_PRODUCER_SOURCE_SHA256: &str =
    "67eb3242df305af18f2bda818e5317d1503ff85b48449f22897b20350c98136e";
pub const HARD_5M_CHILD_ATMOSPHERE_PRODUCER_EXECUTABLE_SHA256: &str =
    "f0cd46d1da25b3fb09173ac7d8e72fa313487182a99d7d1a688a645810aea141";

pub const HARD_5M_STATIC_HEIGHT_MANIFEST_SHA256: &str =
    "e991bf9b85ca71d835b5d34c16e4c877a97af3e36dd12e69a44a930bc96e8104";
pub const HARD_5M_STATIC_HEIGHT_ZH_SHA256: &str =
    "7aa08247f64d6c05fce30b12f7bb8ef8cd050637b4c3a69feea873f20b7d6864";
pub const HARD_5M_STATIC_HEIGHT_ZF_SHA256: &str =
    "8ff218d8cd8f1f37b797d373b127b4f5b98097385f3cccdab4450e6a4bec449d";
pub const HARD_5M_STATIC_HEIGHT_RECEIPT_SHA256: &str =
    "1501521cbf0dd4a025d842916bf2cd0a5939f7c7150484751b52dd64ce920155";
pub const HARD_5M_STATIC_HEIGHT_VALIDATION_SHA256: &str =
    "cd3032b2a515cf40200532b84780e8b1a1d6a1b92b645d67c8df3333176e3766";
pub const HARD_5M_STATIC_HEIGHT_PRODUCER_SOURCE_SHA256: &str =
    "ee21c57d62ac335e57c3a2683becfbe2397747f281929dd1537c88eb6452dd48";
pub const HARD_5M_STATIC_HEIGHT_PRODUCER_EXECUTABLE_SHA256: &str =
    "a41167b66d75d8fd0013e94bc1a70bffcb32d9206b2875000193f888ecbabf3c";
pub const HARD_5M_TERRAIN_MANIFEST_SHA256: &str =
    "7a6b5139168af3384c58998bff8df1dd61f102c2f611ac163251425e607ca45e";
pub const HARD_5M_TERRAIN_PAYLOAD_SHA256: &str =
    "2a444d93e482cc159a2b2b83392d41105a5587261cef801f623a2d4d9d51d72d";
pub const HARD_5M_TERRAIN_RECEIPT_SHA256: &str =
    "553b27baa2f0fda542703a09ba924284f78e9c251e14cd56309958d39c151152";

const ARENA_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const JSON_MAX_BYTES: u64 = 1024 * 1024;
const SOURCE_MAX_BYTES: u64 = 4 * 1024 * 1024;
const EXECUTABLE_MAX_BYTES: u64 = 64 * 1024 * 1024;
const STREAM_CHUNK_BYTES: usize = 1024 * 1024;
const STATIC_ZH_BYTES: u64 = (CHILD_NZ * CHILD_NY * CHILD_NX * 4) as u64;
const STATIC_ZF_BYTES: u64 = ((CHILD_NZ + 1) * CHILD_NY * CHILD_NX * 4) as u64;
const OUTPUT_CARRIER_COUNT: usize = 28;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShapeKind {
    Mass,
    Packed,
    XFace,
    YFace,
    ZFace,
    Surface,
}

impl ShapeKind {
    fn shape(self) -> Vec<usize> {
        self.shape_for_grid(ACCEPTED_50M_CHILD_GRID)
    }

    fn shape_for_grid(self, grid: ChildGridSpec) -> Vec<usize> {
        match self {
            Self::Mass => vec![grid.nz(), grid.ny(), grid.nx()],
            Self::Packed => vec![6, grid.nz(), grid.ny(), grid.nx()],
            Self::XFace => vec![grid.nz(), grid.ny(), grid.nx() + 1],
            Self::YFace => vec![grid.nz(), grid.ny() + 1, grid.nx()],
            Self::ZFace => vec![grid.nz() + 1, grid.ny(), grid.nx()],
            Self::Surface => vec![grid.ny(), grid.nx()],
        }
    }

    const fn stagger(self) -> HorizontalStagger {
        match self {
            Self::XFace => HorizontalStagger::XFace,
            Self::YFace => HorizontalStagger::YFace,
            Self::ZFace => HorizontalStagger::ZFace,
            Self::Surface => HorizontalStagger::Surface,
            Self::Mass | Self::Packed => HorizontalStagger::Scalar,
        }
    }

    const fn source_shape(self) -> [usize; 4] {
        match self {
            Self::Mass => [1, PARENT_NZ, PARENT_NY, PARENT_NX],
            Self::Packed => [6, PARENT_NZ, PARENT_NY, PARENT_NX],
            Self::XFace => [1, PARENT_NZ, PARENT_NY, PARENT_NX + 1],
            Self::YFace => [1, PARENT_NZ, PARENT_NY + 1, PARENT_NX],
            Self::ZFace => [1, PARENT_NZ + 1, PARENT_NY, PARENT_NX],
            Self::Surface => [1, 1, PARENT_NY, PARENT_NX],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Disposition {
    Cm1Interpolate,
    ArwenInterpolate,
    Copy,
    Zero,
}

impl Disposition {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Cm1Interpolate => "CM1_RESTART_GRID_INTERPOLATE",
            Self::ArwenInterpolate => "ARWEN_PERSISTENCE_POLICY_CM1_ARITHMETIC",
            Self::Copy => "CM1_A_TO_CURRENT_RAW_COPY",
            Self::Zero => "CM1_INTERP_ON_RESTART_POSITIVE_ZERO",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CarrierPlan {
    name: &'static str,
    source: Option<&'static str>,
    shape: ShapeKind,
    disposition: Disposition,
}

const CARRIER_PLANS: [CarrierPlan; OUTPUT_CARRIER_COUNT] = [
    direct("component.effc", ShapeKind::Mass, true),
    direct("component.effi", ShapeKind::Mass, true),
    direct("component.effs", ShapeKind::Mass, true),
    direct("component.h_diabatic", ShapeKind::Mass, true),
    direct("component.mp_graupelnc", ShapeKind::Surface, true),
    direct("component.mp_graupelncv", ShapeKind::Surface, true),
    direct("component.mp_rainnc", ShapeKind::Surface, true),
    direct("component.mp_rainncv", ShapeKind::Surface, true),
    direct("component.mp_snownc", ShapeKind::Surface, true),
    direct("component.mp_snowncv", ShapeKind::Surface, true),
    direct("component.mp_sr", ShapeKind::Surface, true),
    zero("held.qcten"),
    zero("held.qpten"),
    zero("held.qtten"),
    zero("held.qvten"),
    copy("pp3d", "ppi", ShapeKind::Mass),
    direct("ppi", ShapeKind::Mass, false),
    direct("ppx", ShapeKind::Mass, false),
    copy("q3d", "qa", ShapeKind::Packed),
    direct("qa", ShapeKind::Packed, false),
    copy("th3d", "tha", ShapeKind::Mass),
    direct("tha", ShapeKind::Mass, false),
    copy("u3d", "ua", ShapeKind::XFace),
    direct("ua", ShapeKind::XFace, false),
    copy("v3d", "va", ShapeKind::YFace),
    direct("va", ShapeKind::YFace, false),
    copy("w3d", "wa", ShapeKind::ZFace),
    direct("wa", ShapeKind::ZFace, false),
];

const fn direct(name: &'static str, shape: ShapeKind, arwen: bool) -> CarrierPlan {
    CarrierPlan {
        name,
        source: Some(name),
        shape,
        disposition: if arwen {
            Disposition::ArwenInterpolate
        } else {
            Disposition::Cm1Interpolate
        },
    }
}

const fn copy(name: &'static str, source: &'static str, shape: ShapeKind) -> CarrierPlan {
    CarrierPlan {
        name,
        source: Some(source),
        shape,
        disposition: Disposition::Copy,
    }
}

const fn zero(name: &'static str) -> CarrierPlan {
    CarrierPlan {
        name,
        source: None,
        shape: ShapeKind::Mass,
        disposition: Disposition::Zero,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AxisPoint {
    lower: usize,
    upper: usize,
    fraction: f32,
    below_support: bool,
}

#[derive(Clone, Debug)]
struct GeneratedCarrier {
    bytes: Vec<u8>,
    sha256: String,
    below_support_words: usize,
    source_first_level_sha256: String,
}

#[derive(Clone, Debug)]
struct StreamedCarrier {
    sha256: String,
    nbytes: u64,
    below_support_words: usize,
    source_first_level_sha256: Option<String>,
}

#[derive(Clone, Debug)]
struct CarrierRecord {
    plan: CarrierPlan,
    shape: Vec<usize>,
    offset_bytes: u64,
    nbytes: u64,
    sha256: String,
    below_support_words: usize,
    source_first_level_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ChildAtmosphereCarrierLayout {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub offset_nbytes: u64,
    pub nbytes: u64,
}

#[derive(Clone, Debug)]
pub struct ChildAtmosphereReceipt {
    pub manifest_sha256: String,
    pub arena_sha256: String,
    pub receipt_sha256: String,
    pub arena_nbytes: u64,
    pub producer_source_sha256: String,
    pub producer_executable_sha256: String,
    /// The parent checkpoint and validated plan this atmosphere was built
    /// from, recorded rather than assumed.  The composing full-state gate
    /// requires every child producer to name the same pair.
    pub parent_checkpoint_sha256: String,
    pub parent_plan_sha256: String,
}

#[derive(Clone, Debug)]
pub struct ChildAtmosphereStreamReplay {
    pub manifest_sha256: String,
    pub arena_sha256: String,
    pub receipt_sha256: String,
    pub arena_nbytes: u64,
    pub carrier_count: usize,
    pub stream_chunk_bytes: usize,
}

/// Independently pinned static-height artifact admitted by the grid-generic
/// atmosphere publisher. The embedded receipt must come from the strict
/// `child_static_heights` producer/reader; this wrapper additionally binds the
/// separately committed validation receipt used at the product boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct ChildAtmosphereStaticHeightIdentity {
    static_heights: ChildStaticHeightGridReceipt,
    validation_receipt_sha256: String,
}

impl ChildAtmosphereStaticHeightIdentity {
    pub fn new(
        static_heights: ChildStaticHeightGridReceipt,
        validation_receipt_sha256: impl Into<String>,
    ) -> Result<Self> {
        let validation_receipt_sha256 = validation_receipt_sha256.into();
        for (label, value) in [
            (
                "static-height manifest",
                static_heights.manifest_sha256.as_str(),
            ),
            ("static-height zh", static_heights.zh_sha256.as_str()),
            ("static-height zf", static_heights.zf_sha256.as_str()),
            (
                "static-height receipt",
                static_heights.receipt_sha256.as_str(),
            ),
            (
                "static-height producer source",
                static_heights.producer_source_sha256.as_str(),
            ),
            (
                "static-height producer executable",
                static_heights.producer_executable_sha256.as_str(),
            ),
            (
                "static-height terrain manifest",
                static_heights.terrain.manifest_sha256.as_str(),
            ),
            (
                "static-height terrain payload",
                static_heights.terrain.payload_sha256.as_str(),
            ),
            (
                "static-height terrain receipt",
                static_heights.terrain.receipt_sha256.as_str(),
            ),
            (
                "static-height validation receipt",
                validation_receipt_sha256.as_str(),
            ),
        ] {
            validate_sha256(value, label)?;
        }
        Ok(Self {
            static_heights,
            validation_receipt_sha256,
        })
    }

    pub fn static_heights(&self) -> &ChildStaticHeightGridReceipt {
        &self.static_heights
    }

    /// The child grid these heights were published for.
    pub fn grid(&self) -> ChildGridSpec {
        self.static_heights.grid
    }

    pub fn validation_receipt_sha256(&self) -> &str {
        &self.validation_receipt_sha256
    }
}

/// Build the static-height identity from a published artifact's own receipt.
///
/// This is the general form of the two frozen preset identities below: the
/// child grid, both payload digests, the producer identities and the terrain
/// the heights were computed from are all read back from the artifact, so a
/// newly published static-height export at any admitted child grid can drive
/// the atmosphere producer.  The separately committed CUDA validation receipt
/// is hashed and bound alongside it, so the heights still cannot be composed
/// without the record that their metric construction passed.
pub fn child_static_height_identity_from_artifact(
    static_height_directory: impl AsRef<Path>,
    validation_receipt: impl AsRef<Path>,
) -> Result<ChildAtmosphereStaticHeightIdentity> {
    let receipt = crate::child_static_heights::read_child_static_height_identity(
        static_height_directory.as_ref(),
    )?;
    let validation_sha256 = hash_regular_bounded(
        validation_receipt.as_ref(),
        JSON_MAX_BYTES,
        "child static-height validation receipt",
    )?;
    ChildAtmosphereStaticHeightIdentity::new(receipt, validation_sha256)
}

/// Frozen production identity for the real-source 3000x3000, 5 m static
/// terrain-following heights. The payload hashes are independent of the
/// atmosphere artifact and were already fully recomputed by the Rust reader.
pub fn hard_5m_child_static_height_identity() -> Result<ChildAtmosphereStaticHeightIdentity> {
    ChildAtmosphereStaticHeightIdentity::new(
        ChildStaticHeightGridReceipt {
            grid: HARD_5M_CHILD_GRID,
            manifest_sha256: HARD_5M_STATIC_HEIGHT_MANIFEST_SHA256.to_owned(),
            zh_sha256: HARD_5M_STATIC_HEIGHT_ZH_SHA256.to_owned(),
            zf_sha256: HARD_5M_STATIC_HEIGHT_ZF_SHA256.to_owned(),
            receipt_sha256: HARD_5M_STATIC_HEIGHT_RECEIPT_SHA256.to_owned(),
            producer_source_sha256: HARD_5M_STATIC_HEIGHT_PRODUCER_SOURCE_SHA256.to_owned(),
            producer_executable_sha256: HARD_5M_STATIC_HEIGHT_PRODUCER_EXECUTABLE_SHA256.to_owned(),
            terrain: ChildTerrainArtifactIdentity::new(
                HARD_5M_TERRAIN_MANIFEST_SHA256,
                HARD_5M_TERRAIN_PAYLOAD_SHA256,
                HARD_5M_TERRAIN_RECEIPT_SHA256,
            )?,
        },
        HARD_5M_STATIC_HEIGHT_VALIDATION_SHA256,
    )
}

/// Exact reader for the committed real 28-carrier child atmosphere.  The
/// fixed manifest/receipt/arena hashes bind every offset and disposition, so
/// downstream producers cannot silently consume a different candidate.
#[derive(Clone, Debug)]
pub struct SealedChildAtmosphere {
    directory: PathBuf,
    manifest: JsonValue,
    receipt: ChildAtmosphereReceipt,
}

impl SealedChildAtmosphere {
    pub fn receipt(&self) -> &ChildAtmosphereReceipt {
        &self.receipt
    }

    /// The manifest bytes this artifact's own receipt anchors, already parsed.
    ///
    /// Consumers read the identities the producer recorded here - the parent
    /// plan and checkpoint, and the product that parent was initialized from -
    /// rather than assuming one historical lineage.
    pub(crate) fn manifest(&self) -> &JsonValue {
        &self.manifest
    }

    pub fn carrier_shape(&self, name: &str) -> Option<Vec<usize>> {
        CARRIER_PLANS
            .iter()
            .find(|plan| plan.name == name)
            .map(|plan| plan.shape.shape())
    }

    /// Scan and hash the complete arena once while capturing the requested
    /// slices.  A valid slice is never returned from a corrupted remainder.
    pub fn read_carriers_f32le_exact(&self, names: &[&str]) -> Result<BTreeMap<String, Vec<u8>>> {
        if names.is_empty() {
            return Err(invalid(
                "sealed child-atmosphere extraction requires at least one carrier",
            ));
        }
        let mut requested = BTreeSet::new();
        for name in names {
            if !requested.insert(*name) {
                return Err(invalid(format!(
                    "sealed child-atmosphere extraction duplicates {name:?}"
                )));
            }
            if !CARRIER_PLANS.iter().any(|plan| plan.name == *name) {
                return Err(invalid(format!(
                    "sealed child atmosphere has no carrier {name:?}"
                )));
            }
        }
        let arena_path = self.directory.join(CHILD_ATMOSPHERE_ARENA_FILE);
        let metadata = fs::metadata(&arena_path)?;
        if !metadata.is_file() || metadata.len() != self.receipt.arena_nbytes {
            return Err(invalid(
                "admitted child-atmosphere arena size or file type changed since its receipt was read",
            ));
        }
        let mut reader = BufReader::with_capacity(1024 * 1024, File::open(&arena_path)?);
        let mut aggregate = Sha256::default();
        let mut captured = BTreeMap::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut total = 0u64;
        for plan in CARRIER_PLANS {
            let nbytes = plan
                .shape
                .shape()
                .iter()
                .try_fold(4u64, |value, extent| value.checked_mul(*extent as u64))
                .ok_or_else(|| invalid("sealed child-atmosphere carrier size overflowed"))?;
            let mut remaining = nbytes;
            let mut selected = requested.contains(plan.name).then(|| {
                Vec::with_capacity(usize::try_from(nbytes).expect("sealed carrier fits usize"))
            });
            while remaining != 0 {
                let amount = usize::try_from(remaining.min(buffer.len() as u64))
                    .expect("bounded scan amount fits usize");
                reader.read_exact(&mut buffer[..amount])?;
                aggregate.update(&buffer[..amount]);
                if let Some(bytes) = selected.as_mut() {
                    bytes.extend_from_slice(&buffer[..amount]);
                }
                remaining -= amount as u64;
                total += amount as u64;
            }
            if let Some(bytes) = selected {
                captured.insert(plan.name.to_string(), bytes);
            }
        }
        let mut extra = [0u8; 1];
        if reader.read(&mut extra)? != 0
            || total != self.receipt.arena_nbytes
            || sha256::hex(&aggregate.finalize()) != self.receipt.arena_sha256
        {
            return Err(invalid(
                "admitted child-atmosphere arena differs from the identity its own receipt records; a consumer would read carriers that no published manifest describes",
            ));
        }
        Ok(captured)
    }
}

/// Open any published child atmosphere by the identity it records itself.
///
/// This replaces a fixed-hash pin with the property a consumer actually needs:
/// the artifact's own receipt must be canonical JSON of an admitted atmosphere
/// schema, its recorded manifest digest must be the manifest bytes that are
/// present, and its recorded arena digest and byte count must be the arena
/// that is present.  A directory that fails any of those would hand a consumer
/// carriers no published manifest describes - offsets, shapes and dispositions
/// from one run applied to another run's bytes.
///
/// The parent the atmosphere was built from is read out of the receipt and
/// carried, not assumed, so the composing gate can require every child
/// producer to name the same parent.
pub fn open_child_atmosphere_artifact(
    directory: impl AsRef<Path>,
) -> Result<SealedChildAtmosphere> {
    let directory = directory.as_ref();
    verify_directory_inventory(
        directory,
        &[
            CHILD_ATMOSPHERE_ARENA_FILE,
            CHILD_ATMOSPHERE_MANIFEST_FILE,
            CHILD_ATMOSPHERE_RECEIPT_FILE,
        ],
        "child-atmosphere",
    )?;
    let manifest = read_regular_bounded(
        &directory.join(CHILD_ATMOSPHERE_MANIFEST_FILE),
        JSON_MAX_BYTES,
        "child-atmosphere manifest",
    )?;
    let receipt_bytes = read_regular_bounded(
        &directory.join(CHILD_ATMOSPHERE_RECEIPT_FILE),
        JSON_MAX_BYTES,
        "child-atmosphere receipt",
    )?;
    let receipt = parse_canonical_json(&receipt_bytes, "child-atmosphere receipt")?;
    let schema = json_string(&receipt, "schema", "child-atmosphere receipt")?;
    if schema != CHILD_ATMOSPHERE_SCHEMA && schema != CHILD_ATMOSPHERE_GRID_SCHEMA {
        return Err(invalid(format!(
            "child-atmosphere receipt schema {schema:?} is not an admitted child-atmosphere schema; its carrier layout is undescribed and a consumer would read the arena at the wrong offsets"
        )));
    }
    let status = json_string(&receipt, "status", "child-atmosphere receipt")?;
    if status != CHILD_ATMOSPHERE_STATUS && status != CHILD_ATMOSPHERE_GRID_STATUS {
        return Err(invalid(format!(
            "child-atmosphere receipt status {status:?} does not record a passed publication; the arena may be a partial or refused candidate"
        )));
    }
    let manifest_sha256 = json_sha256(&receipt, "manifest_sha256", "child-atmosphere receipt")?;
    if sha256::digest_hex(&manifest) != manifest_sha256 {
        return Err(invalid(
            "child-atmosphere manifest bytes are not the manifest its own receipt anchors; the published layout and the published receipt describe different artifacts",
        ));
    }
    let arena_sha256 = json_sha256(&receipt, "arena_sha256", "child-atmosphere receipt")?;
    let arena_nbytes = receipt
        .get("arena_nbytes")
        .and_then(JsonValue::as_u64)
        .map_err(|error| invalid(format!("child-atmosphere receipt arena_nbytes: {error}")))?;
    let arena_path = directory.join(CHILD_ATMOSPHERE_ARENA_FILE);
    let metadata = fs::symlink_metadata(&arena_path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != arena_nbytes {
        return Err(invalid(
            "child-atmosphere arena is not a regular file of the byte count its receipt records",
        ));
    }
    if hash_regular_bounded(&arena_path, ARENA_MAX_BYTES, "child-atmosphere arena")? != arena_sha256
    {
        return Err(invalid(
            "child-atmosphere arena differs from the digest its own receipt records",
        ));
    }
    let parent_checkpoint_sha256 = json_sha256(
        &receipt,
        "parent_checkpoint_sha256",
        "child-atmosphere receipt",
    )?;
    let parent_plan_sha256 =
        json_sha256(&receipt, "parent_plan_sha256", "child-atmosphere receipt")?;
    validate_bound_parent_identity(
        &parent_checkpoint_sha256,
        &parent_plan_sha256,
        "child-atmosphere artifact",
    )?;
    Ok(SealedChildAtmosphere {
        directory: directory.to_path_buf(),
        manifest: parse_canonical_json(&manifest, "child-atmosphere manifest")?,
        receipt: ChildAtmosphereReceipt {
            manifest_sha256,
            arena_sha256,
            receipt_sha256: sha256::digest_hex(&receipt_bytes),
            arena_nbytes,
            producer_source_sha256: json_sha256(
                &receipt,
                "producer_source_sha256",
                "child-atmosphere receipt",
            )?,
            producer_executable_sha256: json_sha256(
                &receipt,
                "producer_executable_sha256",
                "child-atmosphere receipt",
            )?,
            parent_checkpoint_sha256,
            parent_plan_sha256,
        },
    })
}

/// Open the one immutable committed 300/50 atmosphere by its historical bytes.
///
/// Raw-bit replay of those exact bytes is the only thing this is for; every
/// production consumer uses [`open_child_atmosphere_artifact`], which admits
/// any atmosphere that is internally consistent with its own receipt.
pub fn open_sealed_child_atmosphere(directory: impl AsRef<Path>) -> Result<SealedChildAtmosphere> {
    let opened = open_child_atmosphere_artifact(directory)?;
    if opened.receipt.manifest_sha256 != SEALED_CHILD_ATMOSPHERE_MANIFEST_SHA256
        || opened.receipt.receipt_sha256 != SEALED_CHILD_ATMOSPHERE_RECEIPT_SHA256
        || opened.receipt.arena_sha256 != SEALED_CHILD_ATMOSPHERE_ARENA_SHA256
        || opened.receipt.arena_nbytes != SEALED_CHILD_ATMOSPHERE_ARENA_NBYTES
    {
        return Err(invalid(
            "child-atmosphere artifact is not the committed historical bytes this raw-bit replay compares against; replaying a different atmosphere would compare two unrelated arenas and report a false mismatch",
        ));
    }
    Ok(opened)
}

fn parse_canonical_json(bytes: &[u8], label: &str) -> Result<JsonValue> {
    let value = crate::json::parse(bytes)
        .map_err(|error| invalid(format!("{label} is invalid JSON: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid(format!(
            "{label} is not canonical sorted-key compact JSON; two byte spellings of one document would hash differently and no digest could bind it"
        )));
    }
    Ok(value)
}

fn json_string(value: &JsonValue, key: &str, label: &str) -> Result<String> {
    crate::child_artifact::json_string_at(value, &[key], label)
}

fn json_sha256(value: &JsonValue, key: &str, label: &str) -> Result<String> {
    crate::child_artifact::json_sha256_at(value, &[key], label)
}

/// Recompute the complete accepted 300/50 atmosphere payload with bounded
/// carrier streaming and compare every byte against the immutable arena.
/// This gate proves the numerical remap/copy/zero payload only; the sealed
/// manifest and receipt remain the independent envelope authority.
pub fn replay_sealed_child_atmosphere_raw_streaming(
    validated_parent_plan: impl AsRef<Path>,
    parent_checkpoint: impl AsRef<Path>,
    directory: impl AsRef<Path>,
) -> Result<ChildAtmosphereStreamReplay> {
    let (binding, plan_sha256) = read_validated_parent_plan(validated_parent_plan.as_ref())?;
    // This entry point replays the one immutable committed atmosphere arena
    // whose bytes are pinned below, so it is bound to the parent that produced
    // those bytes.  The production publishers admit any validated parent.
    if plan_sha256 != FIRST_ADMITTED_PARENT_PLAN_SHA256
        || binding.checkpoint_artifact_sha256 != FIRST_ADMITTED_PARENT_CHECKPOINT_SHA256
    {
        return Err(invalid(
            "raw streaming replay of the committed child-atmosphere arena requires the parent plan and checkpoint those exact bytes were published from; another parent would recompute different bytes and the comparison would be meaningless",
        ));
    }
    let restored = read_gm1_checkpoint(
        parent_checkpoint.as_ref(),
        &binding.checkpoint_contract_sha256,
        &binding.checkpoint_artifact_sha256,
        GM1_F02_ROI_PRODUCT_MANIFEST_SHA256,
        &binding.operator_manifest_sha256,
    )?;
    if restored.config() != Gm1CheckpointConfig::production()
        || restored.source() != &Gm1SourceIdentities::sealed_f02_roi()
        || restored.raw_arena_sha256() != binding.checkpoint_raw_arena_sha256
    {
        return Err(invalid(
            "streamed child-atmosphere parent checkpoint configuration, source, or arena drifted",
        ));
    }
    validate_parent_carriers(&restored)?;
    let sealed = open_sealed_child_atmosphere(directory)?;
    let arena_path = sealed.directory.join(CHILD_ATMOSPHERE_ARENA_FILE);
    let arena_file = File::open(&arena_path)?;
    let mut comparator =
        CompareWriter::new(BufReader::with_capacity(STREAM_CHUNK_BYTES, arena_file));
    let (records, arena_sha256, arena_nbytes) =
        assemble_arena_streaming(&restored, ACCEPTED_50M_CHILD_GRID, &mut comparator)?;
    comparator.finish()?;
    if records.len() != OUTPUT_CARRIER_COUNT
        || arena_nbytes != SEALED_CHILD_ATMOSPHERE_ARENA_NBYTES
        || arena_sha256 != SEALED_CHILD_ATMOSPHERE_ARENA_SHA256
    {
        return Err(invalid(
            "streamed child-atmosphere raw replay differs from the sealed payload",
        ));
    }
    Ok(ChildAtmosphereStreamReplay {
        manifest_sha256: sealed.receipt.manifest_sha256.clone(),
        arena_sha256,
        receipt_sha256: sealed.receipt.receipt_sha256.clone(),
        arena_nbytes,
        carrier_count: records.len(),
        stream_chunk_bytes: STREAM_CHUNK_BYTES,
    })
}

struct PreparedInputs {
    restored: RestoredGm1Checkpoint,
    plan_sha256: String,
    checkpoint_sha256: String,
    producer_source_sha256: String,
    producer_executable_sha256: String,
}

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

fn sub32(left: f32, right: f32) -> f32 {
    left - right
}

fn mul32(left: f32, right: f32) -> f32 {
    left * right
}

fn div32(left: f32, right: f32) -> f32 {
    left / right
}

fn horizontal_axis(values: &[f64], source_extent: usize) -> Result<Vec<AxisPoint>> {
    values
        .iter()
        .map(|value| horizontal_point(*value, source_extent))
        .collect()
}

fn horizontal_point(value: f64, source_extent: usize) -> Result<AxisPoint> {
    if !value.is_finite() || source_extent < 2 || value < 0.0 {
        return Err(invalid("restart-grid horizontal coordinate is invalid"));
    }
    let floor = value.floor();
    let (lower, fraction) = if value.to_bits() == floor.to_bits() {
        let index = floor as usize;
        if index == 0 {
            (0, 0.0f32)
        } else {
            (index - 1, 1.0f32)
        }
    } else {
        (floor as usize, (value - floor) as f32)
    };
    if lower + 1 >= source_extent || !(0.0..=1.0).contains(&fraction) {
        return Err(invalid(
            "restart-grid horizontal interpolation would extrapolate",
        ));
    }
    Ok(AxisPoint {
        lower,
        upper: lower + 1,
        fraction,
        below_support: false,
    })
}

fn vertical_axis(old: &[f32], targets: &[f32], source_levels: usize) -> Result<Vec<AxisPoint>> {
    if old.len() != source_levels + 2 || source_levels < 2 {
        return Err(invalid(
            "restart-grid old vertical coordinate shape drifted",
        ));
    }
    if old.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(invalid(
            "restart-grid old vertical coordinates are not increasing",
        ));
    }
    targets
        .iter()
        .map(|target| {
            if !target.is_finite() {
                return Err(invalid(
                    "restart-grid target vertical coordinate is not finite",
                ));
            }
            let mut old_index = 0usize;
            while old_index + 1 < old.len() && old[old_index + 1] < *target {
                old_index += 1;
            }
            if old_index + 1 >= old.len() {
                return Err(invalid(
                    "restart-grid target exceeds the sealed old-grid ghost support",
                ));
            }
            let fraction = div32(
                sub32(*target, old[old_index]),
                sub32(old[old_index + 1], old[old_index]),
            );
            if !fraction.is_finite() {
                return Err(invalid("restart-grid vertical fraction is not finite"));
            }
            let (lower, upper) = if old_index == 0 {
                (0, 0)
            } else if old_index >= source_levels {
                (source_levels - 1, source_levels - 1)
            } else {
                (old_index - 1, old_index)
            };
            Ok(AxisPoint {
                lower,
                upper,
                fraction,
                below_support: *target < old[1],
            })
        })
        .collect()
}

fn old_mass_sigma() -> Vec<f32> {
    (0..=(PARENT_NZ + 1))
        .map(|index| index as f32 * 200.0 - 100.0)
        .collect()
}

fn old_face_sigma() -> Vec<f32> {
    (0..=(PARENT_NZ + 2))
        .map(|index| index as f32 * 200.0 - 200.0)
        .collect()
}

fn child_horizontal_maps(
    kind: ShapeKind,
) -> Result<(Vec<AxisPoint>, Vec<AxisPoint>, usize, usize)> {
    child_horizontal_maps_for_grid(kind, ACCEPTED_50M_CHILD_GRID)
}

fn child_horizontal_maps_for_grid(
    kind: ShapeKind,
    grid: ChildGridSpec,
) -> Result<(Vec<AxisPoint>, Vec<AxisPoint>, usize, usize)> {
    let shape = kind.shape_for_grid(grid);
    let (target_ny, target_nx) = match kind {
        ShapeKind::Packed => (shape[2], shape[3]),
        ShapeKind::Mass | ShapeKind::XFace | ShapeKind::YFace | ShapeKind::ZFace => {
            (shape[shape.len() - 2], shape[shape.len() - 1])
        }
        ShapeKind::Surface => (shape[0], shape[1]),
    };
    let source = kind.source_shape();
    let x_values = (0..target_nx)
        .map(|i| parent_fractional_xy_for_grid(grid, kind.stagger(), i, 0).map(|point| point.0))
        .collect::<Result<Vec<_>>>()?;
    let y_values = (0..target_ny)
        .map(|j| parent_fractional_xy_for_grid(grid, kind.stagger(), 0, j).map(|point| point.1))
        .collect::<Result<Vec<_>>>()?;
    Ok((
        horizontal_axis(&x_values, source[3])?,
        horizontal_axis(&y_values, source[2])?,
        target_nx,
        target_ny,
    ))
}

fn carrier_nbytes_for_grid(plan: CarrierPlan, grid: ChildGridSpec) -> Result<u64> {
    plan.shape
        .shape_for_grid(grid)
        .iter()
        .try_fold(4u64, |count, extent| count.checked_mul(*extent as u64))
        .ok_or_else(|| invalid("child atmosphere carrier byte count overflowed"))
}

/// Exact byte count of the canonical 28-carrier atmosphere arena for an
/// admitted child grid. This is layout-only and performs no allocation or
/// meteorological interpolation.
pub fn child_atmosphere_arena_nbytes_for_grid(grid: ChildGridSpec) -> Result<u64> {
    CARRIER_PLANS.iter().try_fold(0u64, |total, plan| {
        total
            .checked_add(carrier_nbytes_for_grid(*plan, grid)?)
            .ok_or_else(|| invalid("child atmosphere arena byte count overflowed"))
    })
}

/// Allocation-free carrier offsets for strict streamed consumers of one
/// grid-generic atmosphere arena.
pub(crate) fn child_atmosphere_carrier_layout_for_grid(
    grid: ChildGridSpec,
) -> Result<Vec<ChildAtmosphereCarrierLayout>> {
    let mut offset_nbytes = 0u64;
    let mut layout = Vec::with_capacity(CARRIER_PLANS.len());
    for plan in CARRIER_PLANS {
        let nbytes = carrier_nbytes_for_grid(plan, grid)?;
        layout.push(ChildAtmosphereCarrierLayout {
            name: plan.name,
            shape: plan.shape.shape_for_grid(grid),
            offset_nbytes,
            nbytes,
        });
        offset_nbytes = offset_nbytes
            .checked_add(nbytes)
            .ok_or_else(|| invalid("child atmosphere carrier offset overflows"))?;
    }
    if offset_nbytes != child_atmosphere_arena_nbytes_for_grid(grid)? {
        return Err(invalid(
            "child atmosphere carrier layout does not exhaust its arena",
        ));
    }
    Ok(layout)
}

fn weights(rx: f32, ry: f32, rz: f32) -> [f64; 8] {
    let one_x = sub32(1.0, rx);
    let one_y = sub32(1.0, ry);
    let one_z = sub32(1.0, rz);
    let xy00 = mul32(one_x, one_y);
    let xy10 = mul32(rx, one_y);
    let xy01 = mul32(one_x, ry);
    let xy11 = mul32(rx, ry);
    [
        f64::from(mul32(xy00, one_z)),
        f64::from(mul32(xy10, one_z)),
        f64::from(mul32(xy01, one_z)),
        f64::from(mul32(xy00, rz)),
        f64::from(mul32(xy10, rz)),
        f64::from(mul32(xy01, rz)),
        f64::from(mul32(xy11, one_z)),
        f64::from(mul32(xy11, rz)),
    ]
}

fn source_order_interp(values: [f32; 8], weights: [f64; 8]) -> f32 {
    let mut result = f64::from(values[0]) * weights[0];
    result += f64::from(values[1]) * weights[1];
    result += f64::from(values[2]) * weights[2];
    result += f64::from(values[3]) * weights[3];
    result += f64::from(values[4]) * weights[4];
    result += f64::from(values[5]) * weights[5];
    result += f64::from(values[6]) * weights[6];
    result += f64::from(values[7]) * weights[7];
    result as f32
}

fn first_level_sha256(source: &[f32], shape: [usize; 4]) -> Result<String> {
    let [outer, levels, ny, nx] = shape;
    let expected = outer
        .checked_mul(levels)
        .and_then(|value| value.checked_mul(ny))
        .and_then(|value| value.checked_mul(nx))
        .ok_or_else(|| invalid("source carrier shape product overflowed"))?;
    if source.len() != expected {
        return Err(invalid(
            "source carrier length differs from its sealed shape",
        ));
    }
    let plane = ny * nx;
    let mut hasher = Sha256::default();
    for outer_index in 0..outer {
        let start = outer_index * levels * plane;
        for value in &source[start..start + plane] {
            hasher.update(&value.to_le_bytes());
        }
    }
    Ok(sha256::hex(&hasher.finalize()))
}

fn generate_direct(source: &[f32], kind: ShapeKind) -> Result<GeneratedCarrier> {
    let [outer, source_nz, source_ny, source_nx] = kind.source_shape();
    let expected_source = outer * source_nz * source_ny * source_nx;
    if source.len() != expected_source {
        return Err(invalid(format!(
            "source carrier has {} words; expected {expected_source}",
            source.len()
        )));
    }
    if source.iter().any(|value| !value.is_finite()) {
        return Err(invalid("source carrier contains a non-finite word"));
    }
    let (x_map, y_map, target_nx, target_ny) = child_horizontal_maps(kind)?;
    let grid = sealed_child_stretch_grid()?;
    let (z_map, target_nz) = match kind {
        ShapeKind::Mass | ShapeKind::Packed | ShapeKind::XFace | ShapeKind::YFace => (
            vertical_axis(&old_mass_sigma(), &grid.sigma[1..=CHILD_NZ], PARENT_NZ)?,
            CHILD_NZ,
        ),
        ShapeKind::ZFace => (
            vertical_axis(
                &old_face_sigma(),
                &grid.sigmaf[1..=CHILD_NZ + 1],
                PARENT_NZ + 1,
            )?,
            CHILD_NZ + 1,
        ),
        ShapeKind::Surface => {
            let mass = vertical_axis(&old_mass_sigma(), &grid.sigma[1..=CHILD_NZ], PARENT_NZ)?;
            (
                vec![AxisPoint {
                    lower: 0,
                    upper: 0,
                    fraction: mass[0].fraction,
                    below_support: false,
                }],
                1,
            )
        }
    };
    let target_words = outer * target_nz * target_ny * target_nx;
    let mut bytes = Vec::with_capacity(target_words * 4);
    for outer_index in 0..outer {
        for z in &z_map {
            for y in &y_map {
                for x in &x_map {
                    let index = |k: usize, j: usize, i: usize| {
                        (((outer_index * source_nz + k) * source_ny + j) * source_nx) + i
                    };
                    let value = source_order_interp(
                        [
                            source[index(z.lower, y.lower, x.lower)],
                            source[index(z.lower, y.lower, x.upper)],
                            source[index(z.lower, y.upper, x.lower)],
                            source[index(z.upper, y.lower, x.lower)],
                            source[index(z.upper, y.lower, x.upper)],
                            source[index(z.upper, y.upper, x.lower)],
                            source[index(z.lower, y.upper, x.upper)],
                            source[index(z.upper, y.upper, x.upper)],
                        ],
                        weights(x.fraction, y.fraction, z.fraction),
                    );
                    if !value.is_finite() {
                        return Err(invalid(
                            "restart-grid interpolation produced non-finite data",
                        ));
                    }
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
    }
    if bytes.len() != target_words * 4 {
        return Err(invalid("restart-grid output byte count drifted"));
    }
    let below_levels = z_map.iter().filter(|point| point.below_support).count();
    Ok(GeneratedCarrier {
        sha256: sha256::digest_hex(&bytes),
        bytes,
        below_support_words: outer * below_levels * target_ny * target_nx,
        source_first_level_sha256: first_level_sha256(source, kind.source_shape())?,
    })
}

fn write_hashed_chunk(
    writer: &mut dyn Write,
    bytes: &[u8],
    carrier_hasher: &mut Sha256,
    arena_hasher: &mut Sha256,
) -> Result<()> {
    writer.write_all(bytes)?;
    carrier_hasher.update(bytes);
    arena_hasher.update(bytes);
    Ok(())
}

fn stream_zero_carrier(
    nbytes: u64,
    writer: &mut dyn Write,
    arena_hasher: &mut Sha256,
) -> Result<StreamedCarrier> {
    let zero_chunk = vec![0u8; STREAM_CHUNK_BYTES];
    let mut carrier_hasher = Sha256::default();
    let mut remaining = nbytes;
    while remaining != 0 {
        let amount = usize::try_from(remaining.min(STREAM_CHUNK_BYTES as u64))
            .expect("bounded zero-stream chunk fits usize");
        write_hashed_chunk(
            writer,
            &zero_chunk[..amount],
            &mut carrier_hasher,
            arena_hasher,
        )?;
        remaining -= amount as u64;
    }
    Ok(StreamedCarrier {
        sha256: sha256::hex(&carrier_hasher.finalize()),
        nbytes,
        below_support_words: 0,
        source_first_level_sha256: None,
    })
}

fn stream_direct_carrier(
    source: &[f32],
    kind: ShapeKind,
    grid_spec: ChildGridSpec,
    reject_negative: bool,
    writer: &mut dyn Write,
    arena_hasher: &mut Sha256,
) -> Result<StreamedCarrier> {
    let [outer, source_nz, source_ny, source_nx] = kind.source_shape();
    let expected_source = outer
        .checked_mul(source_nz)
        .and_then(|value| value.checked_mul(source_ny))
        .and_then(|value| value.checked_mul(source_nx))
        .ok_or_else(|| invalid("source carrier shape product overflowed"))?;
    if source.len() != expected_source {
        return Err(invalid(format!(
            "source carrier has {} words; expected {expected_source}",
            source.len()
        )));
    }
    if source.iter().any(|value| !value.is_finite()) {
        return Err(invalid("source carrier contains a non-finite word"));
    }
    let (x_map, y_map, target_nx, target_ny) = child_horizontal_maps_for_grid(kind, grid_spec)?;
    let grid = sealed_child_stretch_grid()?;
    let (z_map, target_nz) = match kind {
        ShapeKind::Mass | ShapeKind::Packed | ShapeKind::XFace | ShapeKind::YFace => (
            vertical_axis(
                &old_mass_sigma(),
                &grid.sigma[1..=grid_spec.nz()],
                PARENT_NZ,
            )?,
            grid_spec.nz(),
        ),
        ShapeKind::ZFace => (
            vertical_axis(
                &old_face_sigma(),
                &grid.sigmaf[1..=grid_spec.nz() + 1],
                PARENT_NZ + 1,
            )?,
            grid_spec.nz() + 1,
        ),
        ShapeKind::Surface => {
            let mass = vertical_axis(
                &old_mass_sigma(),
                &grid.sigma[1..=grid_spec.nz()],
                PARENT_NZ,
            )?;
            (
                vec![AxisPoint {
                    lower: 0,
                    upper: 0,
                    fraction: mass[0].fraction,
                    below_support: false,
                }],
                1,
            )
        }
    };
    let target_words = outer
        .checked_mul(target_nz)
        .and_then(|value| value.checked_mul(target_ny))
        .and_then(|value| value.checked_mul(target_nx))
        .ok_or_else(|| invalid("target carrier shape product overflowed"))?;
    let expected_nbytes = u64::try_from(target_words)
        .ok()
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| invalid("target carrier byte count overflowed"))?;
    let mut carrier_hasher = Sha256::default();
    let mut buffer = Vec::with_capacity(STREAM_CHUNK_BYTES);
    let mut nbytes = 0u64;
    for outer_index in 0..outer {
        for z in &z_map {
            for y in &y_map {
                for x in &x_map {
                    let index = |k: usize, j: usize, i: usize| {
                        (((outer_index * source_nz + k) * source_ny + j) * source_nx) + i
                    };
                    let value = source_order_interp(
                        [
                            source[index(z.lower, y.lower, x.lower)],
                            source[index(z.lower, y.lower, x.upper)],
                            source[index(z.lower, y.upper, x.lower)],
                            source[index(z.upper, y.lower, x.lower)],
                            source[index(z.upper, y.lower, x.upper)],
                            source[index(z.upper, y.upper, x.lower)],
                            source[index(z.lower, y.upper, x.upper)],
                            source[index(z.upper, y.upper, x.upper)],
                        ],
                        weights(x.fraction, y.fraction, z.fraction),
                    );
                    if !value.is_finite() {
                        return Err(invalid(
                            "restart-grid interpolation produced non-finite data",
                        ));
                    }
                    if reject_negative && value < 0.0 {
                        return Err(invalid(
                            "child qa became negative under restart-grid interpolation",
                        ));
                    }
                    buffer.extend_from_slice(&value.to_le_bytes());
                    if buffer.len() == STREAM_CHUNK_BYTES {
                        write_hashed_chunk(writer, &buffer, &mut carrier_hasher, arena_hasher)?;
                        nbytes += buffer.len() as u64;
                        buffer.clear();
                    }
                }
            }
        }
    }
    if !buffer.is_empty() {
        write_hashed_chunk(writer, &buffer, &mut carrier_hasher, arena_hasher)?;
        nbytes += buffer.len() as u64;
    }
    if nbytes != expected_nbytes {
        return Err(invalid("streamed restart-grid output byte count drifted"));
    }
    let below_levels = z_map.iter().filter(|point| point.below_support).count();
    let below_support_words = outer
        .checked_mul(below_levels)
        .and_then(|value| value.checked_mul(target_ny))
        .and_then(|value| value.checked_mul(target_nx))
        .ok_or_else(|| invalid("below-support word count overflowed"))?;
    Ok(StreamedCarrier {
        sha256: sha256::hex(&carrier_hasher.finalize()),
        nbytes,
        below_support_words,
        source_first_level_sha256: Some(first_level_sha256(source, kind.source_shape())?),
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_inputs(
    validated_parent_plan: &Path,
    parent_checkpoint: &Path,
    static_height_directory: &Path,
    static_height_validation_receipt: &Path,
    restart_read_source: &Path,
    init3d_source: &Path,
    producer_source: &Path,
    producer_executable: &Path,
) -> Result<PreparedInputs> {
    let (binding, plan_sha256) = read_validated_parent_plan(validated_parent_plan)?;
    validate_bound_parent_identity(
        &binding.checkpoint_artifact_sha256,
        &plan_sha256,
        "child atmosphere",
    )?;
    verify_static_heights(static_height_directory, static_height_validation_receipt)?;
    verify_exact_hash(
        restart_read_source,
        SOURCE_MAX_BYTES,
        CM1_RESTART_READ_SHA256,
        "CM1 restart_read.F authority",
    )?;
    verify_exact_hash(
        init3d_source,
        SOURCE_MAX_BYTES,
        CM1_INIT3D_SHA256,
        "CM1 init3d.F authority",
    )?;
    let producer_source_sha256 = hash_regular_bounded(
        producer_source,
        SOURCE_MAX_BYTES,
        "child-atmosphere producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable,
        EXECUTABLE_MAX_BYTES,
        "child-atmosphere producer executable",
    )?;
    let restored = read_gm1_checkpoint(
        parent_checkpoint,
        &binding.checkpoint_contract_sha256,
        &binding.checkpoint_artifact_sha256,
        GM1_F02_ROI_PRODUCT_MANIFEST_SHA256,
        &binding.operator_manifest_sha256,
    )?;
    if restored.config() != Gm1CheckpointConfig::production()
        || restored.source() != &Gm1SourceIdentities::sealed_f02_roi()
        || restored.raw_arena_sha256() != binding.checkpoint_raw_arena_sha256
    {
        return Err(invalid(
            "child atmosphere parent checkpoint configuration, source, or arena drifted",
        ));
    }
    validate_parent_carriers(&restored)?;
    Ok(PreparedInputs {
        restored,
        plan_sha256,
        checkpoint_sha256: binding.checkpoint_artifact_sha256.clone(),
        producer_source_sha256,
        producer_executable_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_inputs_for_grid(
    grid: ChildGridSpec,
    static_identity: &ChildAtmosphereStaticHeightIdentity,
    validated_parent_plan: &Path,
    parent_checkpoint: &Path,
    static_height_directory: &Path,
    static_height_validation_receipt: &Path,
    restart_read_source: &Path,
    init3d_source: &Path,
    producer_source: &Path,
    producer_executable: &Path,
) -> Result<PreparedInputs> {
    if static_identity.static_heights.grid != grid {
        return Err(invalid(
            "child atmosphere grid differs from the admitted static-height grid",
        ));
    }
    let (binding, plan_sha256) = read_validated_parent_plan(validated_parent_plan)?;
    validate_bound_parent_identity(
        &binding.checkpoint_artifact_sha256,
        &plan_sha256,
        "grid child atmosphere",
    )?;
    verify_static_heights_for_grid(
        grid,
        static_identity,
        static_height_directory,
        static_height_validation_receipt,
    )?;
    verify_exact_hash(
        restart_read_source,
        SOURCE_MAX_BYTES,
        CM1_RESTART_READ_SHA256,
        "CM1 restart_read.F authority",
    )?;
    verify_exact_hash(
        init3d_source,
        SOURCE_MAX_BYTES,
        CM1_INIT3D_SHA256,
        "CM1 init3d.F authority",
    )?;
    let producer_source_sha256 = hash_regular_bounded(
        producer_source,
        SOURCE_MAX_BYTES,
        "child-atmosphere producer source",
    )?;
    let producer_executable_sha256 = hash_regular_bounded(
        producer_executable,
        EXECUTABLE_MAX_BYTES,
        "child-atmosphere producer executable",
    )?;
    let restored = read_gm1_checkpoint(
        parent_checkpoint,
        &binding.checkpoint_contract_sha256,
        &binding.checkpoint_artifact_sha256,
        GM1_F02_ROI_PRODUCT_MANIFEST_SHA256,
        &binding.operator_manifest_sha256,
    )?;
    if restored.config() != Gm1CheckpointConfig::production()
        || restored.source() != &Gm1SourceIdentities::sealed_f02_roi()
        || restored.raw_arena_sha256() != binding.checkpoint_raw_arena_sha256
    {
        return Err(invalid(
            "child atmosphere parent checkpoint configuration, source, or arena drifted",
        ));
    }
    validate_parent_carriers(&restored)?;
    Ok(PreparedInputs {
        restored,
        plan_sha256,
        checkpoint_sha256: binding.checkpoint_artifact_sha256.clone(),
        producer_source_sha256,
        producer_executable_sha256,
    })
}

fn verify_static_heights(directory: &Path, validation_receipt: &Path) -> Result<()> {
    verify_directory_inventory(
        directory,
        &["manifest.json", "zf.f32le", "zh.f32le"],
        "child static-height",
    )?;
    verify_exact_hash(
        &directory.join("manifest.json"),
        JSON_MAX_BYTES,
        STATIC_HEIGHT_MANIFEST_SHA256,
        "child static-height manifest",
    )?;
    verify_exact_file(
        &directory.join("zh.f32le"),
        STATIC_ZH_BYTES,
        STATIC_HEIGHT_ZH_SHA256,
        "child static zh",
    )?;
    verify_exact_file(
        &directory.join("zf.f32le"),
        STATIC_ZF_BYTES,
        STATIC_HEIGHT_ZF_SHA256,
        "child static zf",
    )?;
    verify_exact_hash(
        validation_receipt,
        JSON_MAX_BYTES,
        STATIC_HEIGHT_VALIDATION_RECEIPT_SHA256,
        "child static-height validation receipt",
    )?;
    Ok(())
}

fn verify_static_heights_for_grid(
    grid: ChildGridSpec,
    identity: &ChildAtmosphereStaticHeightIdentity,
    directory: &Path,
    validation_receipt: &Path,
) -> Result<()> {
    if identity.static_heights.grid != grid {
        return Err(invalid(
            "static-height identity is for a different child grid",
        ));
    }
    verify_directory_inventory(
        directory,
        &["manifest.json", "receipt.json", "zf.f32le", "zh.f32le"],
        "grid child static-height",
    )?;
    verify_exact_hash(
        &directory.join("manifest.json"),
        JSON_MAX_BYTES,
        &identity.static_heights.manifest_sha256,
        "grid child static-height manifest",
    )?;
    verify_exact_hash(
        &directory.join("receipt.json"),
        JSON_MAX_BYTES,
        &identity.static_heights.receipt_sha256,
        "grid child static-height receipt",
    )?;
    let (zh_nbytes, zf_nbytes) = child_static_height_nbytes_for_grid(grid)?;
    verify_exact_file(
        &directory.join("zh.f32le"),
        zh_nbytes,
        &identity.static_heights.zh_sha256,
        "grid child static zh",
    )?;
    verify_exact_file(
        &directory.join("zf.f32le"),
        zf_nbytes,
        &identity.static_heights.zf_sha256,
        "grid child static zf",
    )?;
    verify_exact_hash(
        validation_receipt,
        JSON_MAX_BYTES,
        &identity.validation_receipt_sha256,
        "grid child static-height validation receipt",
    )?;
    Ok(())
}

fn source_arrays(restored: &RestoredGm1Checkpoint) -> BTreeMap<&str, &ForecastArray> {
    restored
        .forecast()
        .frame()
        .arrays()
        .iter()
        .map(|array| (array.name(), array))
        .collect()
}

fn validate_parent_carriers(restored: &RestoredGm1Checkpoint) -> Result<()> {
    let arrays = source_arrays(restored);
    let specs = restored
        .contract()
        .arrays()
        .iter()
        .map(|spec| (spec.name(), spec.shape()))
        .collect::<BTreeMap<_, _>>();
    for plan in CARRIER_PLANS {
        if let Some(source_name) = plan.source {
            let array = arrays.get(source_name).ok_or_else(|| {
                invalid(format!(
                    "parent checkpoint lacks source carrier {source_name}"
                ))
            })?;
            let expected = plan.shape.source_shape();
            let expected_shape = match plan.shape {
                ShapeKind::Packed => expected.to_vec(),
                ShapeKind::Surface => expected[2..].to_vec(),
                _ => expected[1..].to_vec(),
            };
            if specs.get(source_name).copied() != Some(expected_shape.as_slice())
                || array.values().len() != expected.iter().product::<usize>()
            {
                return Err(invalid(format!(
                    "parent carrier {source_name} shape differs from the source-grid ABI"
                )));
            }
        }
    }
    let tkea = arrays
        .get("closure.tkea")
        .ok_or_else(|| invalid("parent checkpoint lacks closure.tkea disposition source"))?;
    if specs.get("closure.tkea").copied() != Some(&[PARENT_NZ + 1, PARENT_NY, PARENT_NX][..])
        || tkea.values().len() != (PARENT_NZ + 1) * PARENT_NY * PARENT_NX
    {
        return Err(invalid("parent closure.tkea shape drifted"));
    }
    for (current, a_state) in [
        ("u3d", "ua"),
        ("v3d", "va"),
        ("w3d", "wa"),
        ("pp3d", "ppi"),
        ("th3d", "tha"),
        ("q3d", "qa"),
    ] {
        let current_values = arrays
            .get(current)
            .ok_or_else(|| invalid(format!("parent checkpoint lacks {current}")))?
            .values();
        let a_values = arrays
            .get(a_state)
            .ok_or_else(|| invalid(format!("parent checkpoint lacks {a_state}")))?
            .values();
        if current_values.len() != a_values.len()
            || !current_values
                .iter()
                .zip(a_values)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        {
            return Err(invalid(format!(
                "completed parent boundary violates required {a_state} to {current} raw-copy identity"
            )));
        }
    }
    Ok(())
}

fn generate_direct_cache(
    restored: &RestoredGm1Checkpoint,
) -> Result<BTreeMap<&'static str, GeneratedCarrier>> {
    let arrays = source_arrays(restored);
    let mut cache = BTreeMap::new();
    for plan in CARRIER_PLANS {
        if !matches!(
            plan.disposition,
            Disposition::Cm1Interpolate | Disposition::ArwenInterpolate
        ) {
            continue;
        }
        let source_name = plan.source.expect("direct plan has a source");
        let values = arrays
            .get(source_name)
            .ok_or_else(|| invalid(format!("parent checkpoint lacks {source_name}")))?
            .values();
        let generated = generate_direct(values, plan.shape)?;
        if source_name == "qa"
            && generated
                .bytes
                .chunks_exact(4)
                .any(|word| f32::from_le_bytes(word.try_into().expect("four-byte f32 word")) < 0.0)
        {
            return Err(invalid(
                "child qa became negative under restart-grid interpolation",
            ));
        }
        cache.insert(source_name, generated);
    }
    if cache.len() != 18 {
        return Err(invalid("direct child carrier inventory drifted"));
    }
    Ok(cache)
}

fn assemble_arena(
    cache: &BTreeMap<&'static str, GeneratedCarrier>,
    writer: &mut dyn Write,
) -> Result<(Vec<CarrierRecord>, String, u64)> {
    let zero_words = CHILD_NZ * CHILD_NY * CHILD_NX;
    let zero_bytes = vec![0u8; zero_words * 4];
    let zero_sha256 = sha256::digest_hex(&zero_bytes);
    let mut arena_hasher = Sha256::default();
    let mut offset = 0u64;
    let mut records = Vec::with_capacity(OUTPUT_CARRIER_COUNT);
    for plan in CARRIER_PLANS {
        let (bytes, carrier_sha256, below_support_words, source_first_level_sha256) =
            match plan.disposition {
                Disposition::Cm1Interpolate | Disposition::ArwenInterpolate | Disposition::Copy => {
                    let source_name = plan.source.expect("nonzero plan has a source");
                    let generated = cache.get(source_name).ok_or_else(|| {
                        invalid(format!("generated source carrier {source_name} is absent"))
                    })?;
                    (
                        generated.bytes.as_slice(),
                        generated.sha256.clone(),
                        generated.below_support_words,
                        Some(generated.source_first_level_sha256.clone()),
                    )
                }
                Disposition::Zero => (zero_bytes.as_slice(), zero_sha256.clone(), 0, None),
            };
        let expected_nbytes = plan
            .shape
            .shape()
            .iter()
            .try_fold(4usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| invalid("child carrier byte count overflowed"))?;
        if bytes.len() != expected_nbytes {
            return Err(invalid(format!(
                "child carrier {} has {} bytes; expected {expected_nbytes}",
                plan.name,
                bytes.len()
            )));
        }
        writer.write_all(bytes)?;
        arena_hasher.update(bytes);
        records.push(CarrierRecord {
            plan,
            shape: plan.shape.shape(),
            offset_bytes: offset,
            nbytes: bytes.len() as u64,
            sha256: carrier_sha256,
            below_support_words,
            source_first_level_sha256,
        });
        offset = offset
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("child atmosphere arena byte count overflowed"))?;
    }
    if records.len() != OUTPUT_CARRIER_COUNT
        || records
            .windows(2)
            .any(|pair| pair[0].plan.name >= pair[1].plan.name)
        || offset > ARENA_MAX_BYTES
    {
        return Err(invalid(
            "child atmosphere carrier order, count, or total size drifted",
        ));
    }
    Ok((records, sha256::hex(&arena_hasher.finalize()), offset))
}

fn assemble_arena_streaming(
    restored: &RestoredGm1Checkpoint,
    grid: ChildGridSpec,
    writer: &mut dyn Write,
) -> Result<(Vec<CarrierRecord>, String, u64)> {
    let arrays = source_arrays(restored);
    let mut arena_hasher = Sha256::default();
    let mut offset = 0u64;
    let mut records = Vec::with_capacity(OUTPUT_CARRIER_COUNT);
    for plan in CARRIER_PLANS {
        let expected_nbytes = carrier_nbytes_for_grid(plan, grid)?;
        let streamed = match plan.disposition {
            Disposition::Cm1Interpolate | Disposition::ArwenInterpolate | Disposition::Copy => {
                let source_name = plan.source.expect("nonzero plan has a source");
                let source = arrays
                    .get(source_name)
                    .ok_or_else(|| invalid(format!("parent checkpoint lacks {source_name}")))?
                    .values();
                stream_direct_carrier(
                    source,
                    plan.shape,
                    grid,
                    source_name == "qa",
                    writer,
                    &mut arena_hasher,
                )?
            }
            Disposition::Zero => stream_zero_carrier(expected_nbytes, writer, &mut arena_hasher)?,
        };
        if streamed.nbytes != expected_nbytes {
            return Err(invalid(format!(
                "streamed child carrier {} has {} bytes; expected {expected_nbytes}",
                plan.name, streamed.nbytes
            )));
        }
        records.push(CarrierRecord {
            plan,
            shape: plan.shape.shape_for_grid(grid),
            offset_bytes: offset,
            nbytes: streamed.nbytes,
            sha256: streamed.sha256,
            below_support_words: streamed.below_support_words,
            source_first_level_sha256: streamed.source_first_level_sha256,
        });
        offset = offset
            .checked_add(streamed.nbytes)
            .ok_or_else(|| invalid("streamed child atmosphere arena byte count overflowed"))?;
    }
    let expected_arena_nbytes = child_atmosphere_arena_nbytes_for_grid(grid)?;
    if records.len() != OUTPUT_CARRIER_COUNT
        || records
            .windows(2)
            .any(|pair| pair[0].plan.name >= pair[1].plan.name)
        || offset != expected_arena_nbytes
    {
        return Err(invalid(
            "streamed child atmosphere carrier order, count, or total size drifted",
        ));
    }
    Ok((records, sha256::hex(&arena_hasher.finalize()), offset))
}

/// Publish the child atmosphere against the original 300/50 static-height
/// export, whose zh/zf bytes predate the in-artifact receipt and are therefore
/// still admitted by their committed digests.
///
/// Every other child grid - including a newly published 300/50 export - goes
/// through [`publish_child_atmosphere_for_grid_new`], which takes the
/// static-height identity read back from that artifact's own receipt.
#[allow(clippy::too_many_arguments)]
pub fn publish_child_atmosphere_new(
    validated_parent_plan: impl AsRef<Path>,
    parent_checkpoint: impl AsRef<Path>,
    static_height_directory: impl AsRef<Path>,
    static_height_validation_receipt: impl AsRef<Path>,
    restart_read_source: impl AsRef<Path>,
    init3d_source: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
) -> Result<ChildAtmosphereReceipt> {
    let prepared = prepare_inputs(
        validated_parent_plan.as_ref(),
        parent_checkpoint.as_ref(),
        static_height_directory.as_ref(),
        static_height_validation_receipt.as_ref(),
        restart_read_source.as_ref(),
        init3d_source.as_ref(),
        producer_source.as_ref(),
        producer_executable.as_ref(),
    )?;
    let cache = generate_direct_cache(&prepared.restored)?;
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace child-atmosphere destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "child-atmosphere destination parent does not exist: {}",
            parent.display()
        )));
    }
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("child-atmosphere destination needs a UTF-8 name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let staging =
        output.with_file_name(format!(".{name}.partial-{}-{sequence}", std::process::id()));
    fs::create_dir(&staging)?;
    let result = (|| -> Result<ChildAtmosphereReceipt> {
        let arena_path = staging.join(CHILD_ATMOSPHERE_ARENA_FILE);
        let arena_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&arena_path)?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, arena_file);
        let (records, arena_sha256, arena_nbytes) = assemble_arena(&cache, &mut writer)?;
        writer.flush()?;
        let arena_file = writer
            .into_inner()
            .map_err(|error| invalid(format!("could not finalize child arena: {error}")))?;
        arena_file.sync_all()?;
        drop(arena_file);
        let manifest = manifest_value(&prepared, &records, &arena_sha256, arena_nbytes);
        let manifest_bytes = manifest.canonical_bytes();
        let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
        let receipt_template = ChildAtmosphereReceipt {
            manifest_sha256,
            arena_sha256,
            receipt_sha256: String::new(),
            arena_nbytes,
            producer_source_sha256: prepared.producer_source_sha256.clone(),
            producer_executable_sha256: prepared.producer_executable_sha256.clone(),
            parent_checkpoint_sha256: prepared.checkpoint_sha256.clone(),
            parent_plan_sha256: prepared.plan_sha256.clone(),
        };
        let receipt_bytes = receipt_value(&receipt_template, &records).canonical_bytes();
        let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
        write_new_synced(
            &staging.join(CHILD_ATMOSPHERE_MANIFEST_FILE),
            &manifest_bytes,
        )?;
        write_new_synced(&staging.join(CHILD_ATMOSPHERE_RECEIPT_FILE), &receipt_bytes)?;
        verify_directory_inventory(
            &staging,
            &[
                CHILD_ATMOSPHERE_ARENA_FILE,
                CHILD_ATMOSPHERE_MANIFEST_FILE,
                CHILD_ATMOSPHERE_RECEIPT_FILE,
            ],
            "child-atmosphere",
        )?;
        fs::rename(&staging, output)?;
        Ok(ChildAtmosphereReceipt {
            receipt_sha256,
            ..receipt_template
        })
    })();
    if result.is_err() {
        for file in [
            CHILD_ATMOSPHERE_ARENA_FILE,
            CHILD_ATMOSPHERE_MANIFEST_FILE,
            CHILD_ATMOSPHERE_RECEIPT_FILE,
        ] {
            let _ = fs::remove_file(staging.join(file));
        }
        let _ = fs::remove_dir(&staging);
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub fn read_child_atmosphere_artifact(
    validated_parent_plan: impl AsRef<Path>,
    parent_checkpoint: impl AsRef<Path>,
    static_height_directory: impl AsRef<Path>,
    static_height_validation_receipt: impl AsRef<Path>,
    restart_read_source: impl AsRef<Path>,
    init3d_source: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_arena_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<ChildAtmosphereReceipt> {
    for (value, label) in [
        (
            expected_manifest_sha256,
            "expected child-atmosphere manifest",
        ),
        (expected_arena_sha256, "expected child-atmosphere arena"),
        (expected_receipt_sha256, "expected child-atmosphere receipt"),
    ] {
        validate_sha256(value, label)?;
    }
    let prepared = prepare_inputs(
        validated_parent_plan.as_ref(),
        parent_checkpoint.as_ref(),
        static_height_directory.as_ref(),
        static_height_validation_receipt.as_ref(),
        restart_read_source.as_ref(),
        init3d_source.as_ref(),
        producer_source.as_ref(),
        producer_executable.as_ref(),
    )?;
    let directory = directory.as_ref();
    verify_directory_inventory(
        directory,
        &[
            CHILD_ATMOSPHERE_ARENA_FILE,
            CHILD_ATMOSPHERE_MANIFEST_FILE,
            CHILD_ATMOSPHERE_RECEIPT_FILE,
        ],
        "child-atmosphere",
    )?;
    let manifest_bytes = read_regular_bounded(
        &directory.join(CHILD_ATMOSPHERE_MANIFEST_FILE),
        JSON_MAX_BYTES,
        "child-atmosphere manifest",
    )?;
    let receipt_bytes = read_regular_bounded(
        &directory.join(CHILD_ATMOSPHERE_RECEIPT_FILE),
        JSON_MAX_BYTES,
        "child-atmosphere receipt",
    )?;
    let actual_manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let actual_receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    if actual_manifest_sha256 != expected_manifest_sha256
        || actual_receipt_sha256 != expected_receipt_sha256
    {
        return Err(invalid(
            "child-atmosphere manifest or receipt differs from its independent hash",
        ));
    }
    let arena_path = directory.join(CHILD_ATMOSPHERE_ARENA_FILE);
    let actual_arena_sha256 =
        hash_regular_bounded(&arena_path, ARENA_MAX_BYTES, "child-atmosphere arena")?;
    if actual_arena_sha256 != expected_arena_sha256 {
        return Err(invalid(
            "child-atmosphere arena differs from its independent hash",
        ));
    }
    let arena_file = File::open(&arena_path)?;
    let mut comparator = CompareWriter::new(BufReader::with_capacity(1024 * 1024, arena_file));
    let (records, recomputed_arena_sha256, arena_nbytes) =
        assemble_arena_streaming(&prepared.restored, ACCEPTED_50M_CHILD_GRID, &mut comparator)?;
    comparator.finish()?;
    if recomputed_arena_sha256 != actual_arena_sha256 {
        return Err(invalid(
            "published child atmosphere differs from the Rust source recomputation",
        ));
    }
    let expected_manifest =
        manifest_value(&prepared, &records, &recomputed_arena_sha256, arena_nbytes)
            .canonical_bytes();
    if manifest_bytes != expected_manifest {
        return Err(invalid(
            "child-atmosphere manifest differs from the exact source-bound contract",
        ));
    }
    let mut receipt = ChildAtmosphereReceipt {
        manifest_sha256: actual_manifest_sha256,
        arena_sha256: actual_arena_sha256,
        receipt_sha256: String::new(),
        arena_nbytes,
        producer_source_sha256: prepared.producer_source_sha256.clone(),
        producer_executable_sha256: prepared.producer_executable_sha256.clone(),
        parent_checkpoint_sha256: prepared.checkpoint_sha256.clone(),
        parent_plan_sha256: prepared.plan_sha256.clone(),
    };
    if receipt_bytes != receipt_value(&receipt, &records).canonical_bytes() {
        return Err(invalid(
            "child-atmosphere receipt differs from the exact source-bound contract",
        ));
    }
    receipt.receipt_sha256 = actual_receipt_sha256;
    Ok(receipt)
}

fn validate_completed_grid_records(records: &[CarrierRecord]) -> Result<()> {
    let by_name = records
        .iter()
        .map(|record| (record.plan.name, record))
        .collect::<BTreeMap<_, _>>();
    if by_name.len() != OUTPUT_CARRIER_COUNT {
        return Err(invalid(
            "grid child-atmosphere carrier inventory is not exact",
        ));
    }
    for (current, a_state) in [
        ("u3d", "ua"),
        ("v3d", "va"),
        ("w3d", "wa"),
        ("pp3d", "ppi"),
        ("th3d", "tha"),
        ("q3d", "qa"),
    ] {
        let current = by_name
            .get(current)
            .ok_or_else(|| invalid("grid child-atmosphere current carrier is absent"))?;
        let a_state = by_name
            .get(a_state)
            .ok_or_else(|| invalid("grid child-atmosphere A-state carrier is absent"))?;
        if current.sha256 != a_state.sha256
            || current.shape != a_state.shape
            || current.nbytes != a_state.nbytes
        {
            return Err(invalid(
                "grid child-atmosphere current/A output pair is not raw-bit equal",
            ));
        }
    }
    Ok(())
}

/// Stream one shape-parameterized, source-backed child-atmosphere artifact
/// directly to its final arena. Only the immutable parent checkpoint and a
/// separately admitted static-height identity are accepted; no generated
/// carrier is cached at full child-grid size.
#[allow(clippy::too_many_arguments)]
pub fn publish_child_atmosphere_for_grid_new(
    grid: ChildGridSpec,
    static_identity: &ChildAtmosphereStaticHeightIdentity,
    validated_parent_plan: impl AsRef<Path>,
    parent_checkpoint: impl AsRef<Path>,
    static_height_directory: impl AsRef<Path>,
    static_height_validation_receipt: impl AsRef<Path>,
    restart_read_source: impl AsRef<Path>,
    init3d_source: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
) -> Result<ChildAtmosphereReceipt> {
    let prepared = prepare_inputs_for_grid(
        grid,
        static_identity,
        validated_parent_plan.as_ref(),
        parent_checkpoint.as_ref(),
        static_height_directory.as_ref(),
        static_height_validation_receipt.as_ref(),
        restart_read_source.as_ref(),
        init3d_source.as_ref(),
        producer_source.as_ref(),
        producer_executable.as_ref(),
    )?;
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace grid child-atmosphere destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "grid child-atmosphere destination parent does not exist: {}",
            parent.display()
        )));
    }
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("grid child-atmosphere destination needs a UTF-8 name"))?;
    let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let staging =
        output.with_file_name(format!(".{name}.partial-{}-{sequence}", std::process::id()));
    fs::create_dir(&staging)?;
    let result = (|| -> Result<ChildAtmosphereReceipt> {
        let arena_path = staging.join(CHILD_ATMOSPHERE_ARENA_FILE);
        let arena_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&arena_path)?;
        let mut writer = BufWriter::with_capacity(STREAM_CHUNK_BYTES, arena_file);
        let (records, arena_sha256, arena_nbytes) =
            assemble_arena_streaming(&prepared.restored, grid, &mut writer)?;
        validate_completed_grid_records(&records)?;
        writer.flush()?;
        let arena_file = writer.into_inner().map_err(|error| {
            invalid(format!(
                "could not finalize grid child-atmosphere arena: {error}"
            ))
        })?;
        arena_file.sync_all()?;
        drop(arena_file);
        let manifest = manifest_value_for_grid(
            &prepared,
            static_identity,
            grid,
            &records,
            &arena_sha256,
            arena_nbytes,
        );
        let manifest_bytes = manifest.canonical_bytes();
        let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
        let receipt_template = ChildAtmosphereReceipt {
            manifest_sha256,
            arena_sha256,
            receipt_sha256: String::new(),
            arena_nbytes,
            producer_source_sha256: prepared.producer_source_sha256.clone(),
            producer_executable_sha256: prepared.producer_executable_sha256.clone(),
            parent_checkpoint_sha256: prepared.checkpoint_sha256.clone(),
            parent_plan_sha256: prepared.plan_sha256.clone(),
        };
        let receipt_bytes =
            receipt_value_for_grid(&receipt_template, static_identity, grid, &records)
                .canonical_bytes();
        let receipt_sha256 = sha256::digest_hex(&receipt_bytes);
        write_new_synced(
            &staging.join(CHILD_ATMOSPHERE_MANIFEST_FILE),
            &manifest_bytes,
        )?;
        write_new_synced(&staging.join(CHILD_ATMOSPHERE_RECEIPT_FILE), &receipt_bytes)?;
        verify_directory_inventory(
            &staging,
            &[
                CHILD_ATMOSPHERE_ARENA_FILE,
                CHILD_ATMOSPHERE_MANIFEST_FILE,
                CHILD_ATMOSPHERE_RECEIPT_FILE,
            ],
            "grid child-atmosphere",
        )?;
        fs::rename(&staging, output)?;
        Ok(ChildAtmosphereReceipt {
            receipt_sha256,
            ..receipt_template
        })
    })();
    if result.is_err() {
        for file in [
            CHILD_ATMOSPHERE_ARENA_FILE,
            CHILD_ATMOSPHERE_MANIFEST_FILE,
            CHILD_ATMOSPHERE_RECEIPT_FILE,
        ] {
            let _ = fs::remove_file(staging.join(file));
        }
        let _ = fs::remove_dir(&staging);
    }
    result
}

/// Independently recompute and compare every byte of a grid-generic child
/// atmosphere. The expected hashes are external pins, and the complete arena
/// is consumed exactly once by the comparator while the Rust source transform
/// regenerates it with bounded memory.
#[allow(clippy::too_many_arguments)]
pub fn read_child_atmosphere_for_grid_artifact(
    grid: ChildGridSpec,
    static_identity: &ChildAtmosphereStaticHeightIdentity,
    validated_parent_plan: impl AsRef<Path>,
    parent_checkpoint: impl AsRef<Path>,
    static_height_directory: impl AsRef<Path>,
    static_height_validation_receipt: impl AsRef<Path>,
    restart_read_source: impl AsRef<Path>,
    init3d_source: impl AsRef<Path>,
    producer_source: impl AsRef<Path>,
    producer_executable: impl AsRef<Path>,
    directory: impl AsRef<Path>,
    expected_manifest_sha256: &str,
    expected_arena_sha256: &str,
    expected_receipt_sha256: &str,
) -> Result<ChildAtmosphereReceipt> {
    for (value, label) in [
        (
            expected_manifest_sha256,
            "expected grid child-atmosphere manifest",
        ),
        (
            expected_arena_sha256,
            "expected grid child-atmosphere arena",
        ),
        (
            expected_receipt_sha256,
            "expected grid child-atmosphere receipt",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    let prepared = prepare_inputs_for_grid(
        grid,
        static_identity,
        validated_parent_plan.as_ref(),
        parent_checkpoint.as_ref(),
        static_height_directory.as_ref(),
        static_height_validation_receipt.as_ref(),
        restart_read_source.as_ref(),
        init3d_source.as_ref(),
        producer_source.as_ref(),
        producer_executable.as_ref(),
    )?;
    let directory = directory.as_ref();
    verify_directory_inventory(
        directory,
        &[
            CHILD_ATMOSPHERE_ARENA_FILE,
            CHILD_ATMOSPHERE_MANIFEST_FILE,
            CHILD_ATMOSPHERE_RECEIPT_FILE,
        ],
        "grid child-atmosphere",
    )?;
    let manifest_bytes = read_regular_bounded(
        &directory.join(CHILD_ATMOSPHERE_MANIFEST_FILE),
        JSON_MAX_BYTES,
        "grid child-atmosphere manifest",
    )?;
    let receipt_bytes = read_regular_bounded(
        &directory.join(CHILD_ATMOSPHERE_RECEIPT_FILE),
        JSON_MAX_BYTES,
        "grid child-atmosphere receipt",
    )?;
    let actual_manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    let actual_receipt_sha256 = sha256::digest_hex(&receipt_bytes);
    if actual_manifest_sha256 != expected_manifest_sha256
        || actual_receipt_sha256 != expected_receipt_sha256
    {
        return Err(invalid(
            "grid child-atmosphere manifest or receipt differs from its independent hash",
        ));
    }
    let arena_path = directory.join(CHILD_ATMOSPHERE_ARENA_FILE);
    let expected_arena_nbytes = child_atmosphere_arena_nbytes_for_grid(grid)?;
    let arena_metadata = fs::symlink_metadata(&arena_path)?;
    if arena_metadata.file_type().is_symlink()
        || !arena_metadata.is_file()
        || arena_metadata.len() != expected_arena_nbytes
    {
        return Err(invalid(
            "grid child-atmosphere arena is not an exact regular file",
        ));
    }
    let arena_file = File::open(&arena_path)?;
    let mut comparator =
        CompareWriter::new(BufReader::with_capacity(STREAM_CHUNK_BYTES, arena_file));
    let (records, recomputed_arena_sha256, arena_nbytes) =
        assemble_arena_streaming(&prepared.restored, grid, &mut comparator)?;
    comparator.finish()?;
    validate_completed_grid_records(&records)?;
    if arena_nbytes != expected_arena_nbytes || recomputed_arena_sha256 != expected_arena_sha256 {
        return Err(invalid(
            "published grid child atmosphere differs from the Rust source recomputation or external arena pin",
        ));
    }
    let expected_manifest = manifest_value_for_grid(
        &prepared,
        static_identity,
        grid,
        &records,
        &recomputed_arena_sha256,
        arena_nbytes,
    )
    .canonical_bytes();
    if manifest_bytes != expected_manifest {
        return Err(invalid(
            "grid child-atmosphere manifest differs from the exact source-bound contract",
        ));
    }
    let mut receipt = ChildAtmosphereReceipt {
        manifest_sha256: actual_manifest_sha256,
        arena_sha256: recomputed_arena_sha256,
        receipt_sha256: String::new(),
        arena_nbytes,
        producer_source_sha256: prepared.producer_source_sha256.clone(),
        producer_executable_sha256: prepared.producer_executable_sha256.clone(),
        parent_checkpoint_sha256: prepared.checkpoint_sha256.clone(),
        parent_plan_sha256: prepared.plan_sha256.clone(),
    };
    if receipt_bytes
        != receipt_value_for_grid(&receipt, static_identity, grid, &records).canonical_bytes()
    {
        return Err(invalid(
            "grid child-atmosphere receipt differs from the exact source-bound contract",
        ));
    }
    receipt.receipt_sha256 = actual_receipt_sha256;
    Ok(receipt)
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
            return Err(invalid("child-atmosphere arena has trailing bytes"));
        }
        Ok(())
    }
}

impl<R: Read> Write for CompareWriter<R> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let mut consumed = 0usize;
        let mut observed = [0u8; 64 * 1024];
        while consumed < bytes.len() {
            let amount = observed.len().min(bytes.len() - consumed);
            self.reader.read_exact(&mut observed[..amount])?;
            if observed[..amount] != bytes[consumed..consumed + amount] {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "child-atmosphere arena differs at or before byte {}",
                        self.offset + consumed as u64
                    ),
                ));
            }
            consumed += amount;
        }
        self.offset += bytes.len() as u64;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn manifest_value(
    prepared: &PreparedInputs,
    records: &[CarrierRecord],
    arena_sha256: &str,
    arena_nbytes: u64,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_ATMOSPHERE_ABI)),
        ("artifact_schema", string(CHILD_ATMOSPHERE_SCHEMA)),
        ("authority", authority_value()),
        (
            "arena",
            object(vec![
                ("carrier_count", number(OUTPUT_CARRIER_COUNT)),
                ("dtype", string("little-endian-f32")),
                ("file", string(CHILD_ATMOSPHERE_ARENA_FILE)),
                ("nbytes", u64_number(arena_nbytes)),
                ("order", string("C; carrier-major in manifest order")),
                ("sha256", string(arena_sha256)),
            ]),
        ),
        (
            "carriers",
            JsonValue::Array(records.iter().map(carrier_record_value).collect()),
        ),
        ("claim_limit", string(CHILD_ATMOSPHERE_CLAIM_LIMIT)),
        ("dispositions", disposition_value()),
        ("grid", grid_value()),
        ("model", string("GM1")),
        ("parent", parent_value(prepared)),
        (
            "producer",
            object(vec![
                (
                    "executable_sha256",
                    string(&prepared.producer_executable_sha256),
                ),
                ("source_sha256", string(&prepared.producer_source_sha256)),
            ]),
        ),
        ("schema", string(CHILD_ATMOSPHERE_MANIFEST_SCHEMA)),
        ("static_heights", static_height_value()),
        ("status", string(CHILD_ATMOSPHERE_STATUS)),
        ("synthetic_translation", JsonValue::Bool(false)),
        ("transform", transform_value()),
    ])
}

fn receipt_value(receipt: &ChildAtmosphereReceipt, records: &[CarrierRecord]) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_ATMOSPHERE_ABI)),
        ("all_finite", JsonValue::Bool(true)),
        ("arena_nbytes", u64_number(receipt.arena_nbytes)),
        ("arena_sha256", string(&receipt.arena_sha256)),
        ("authority", authority_value()),
        ("carrier_count", number(records.len())),
        (
            "carrier_sha256",
            JsonValue::Array(
                records
                    .iter()
                    .map(|record| {
                        object(vec![
                            (
                                "below_source_support_words",
                                number(record.below_support_words),
                            ),
                            ("name", string(record.plan.name)),
                            ("sha256", string(&record.sha256)),
                            (
                                "source_first_level_sha256",
                                optional_string(record.source_first_level_sha256.as_deref()),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("claim_limit", string(CHILD_ATMOSPHERE_CLAIM_LIMIT)),
        ("manifest_sha256", string(&receipt.manifest_sha256)),
        ("model", string("GM1")),
        (
            "parent_checkpoint_sha256",
            string(&receipt.parent_checkpoint_sha256),
        ),
        ("parent_plan_sha256", string(&receipt.parent_plan_sha256)),
        (
            "producer_executable_sha256",
            string(&receipt.producer_executable_sha256),
        ),
        (
            "producer_source_sha256",
            string(&receipt.producer_source_sha256),
        ),
        ("schema", string(CHILD_ATMOSPHERE_SCHEMA)),
        (
            "static_heights_manifest_sha256",
            string(STATIC_HEIGHT_MANIFEST_SHA256),
        ),
        ("status", string(CHILD_ATMOSPHERE_STATUS)),
        ("synthetic_translation", JsonValue::Bool(false)),
    ])
}

fn manifest_value_for_grid(
    prepared: &PreparedInputs,
    static_identity: &ChildAtmosphereStaticHeightIdentity,
    grid: ChildGridSpec,
    records: &[CarrierRecord],
    arena_sha256: &str,
    arena_nbytes: u64,
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_ATMOSPHERE_GRID_ABI)),
        ("artifact_schema", string(CHILD_ATMOSPHERE_GRID_SCHEMA)),
        ("authority", authority_value()),
        (
            "arena",
            object(vec![
                ("carrier_count", number(OUTPUT_CARRIER_COUNT)),
                ("dtype", string("little-endian-f32")),
                ("file", string(CHILD_ATMOSPHERE_ARENA_FILE)),
                ("nbytes", u64_number(arena_nbytes)),
                ("order", string("C; carrier-major in manifest order")),
                ("sha256", string(arena_sha256)),
            ]),
        ),
        (
            "carriers",
            JsonValue::Array(records.iter().map(carrier_record_value).collect()),
        ),
        ("claim_limit", string(CHILD_ATMOSPHERE_GRID_CLAIM_LIMIT)),
        (
            "completed_boundary_invariants",
            object(vec![
                ("current_a_pairs_raw_bit_equal", JsonValue::Bool(true)),
                ("qa_nonnegative", JsonValue::Bool(true)),
                ("tkea_excluded_for_fresh_child", JsonValue::Bool(true)),
            ]),
        ),
        ("dispositions", disposition_value()),
        ("grid", grid_value_for_grid(grid)),
        ("model", string("GM1")),
        ("parent", parent_value(prepared)),
        (
            "producer",
            object(vec![
                (
                    "executable_sha256",
                    string(&prepared.producer_executable_sha256),
                ),
                ("source_sha256", string(&prepared.producer_source_sha256)),
                ("stream_chunk_bytes", number(STREAM_CHUNK_BYTES)),
            ]),
        ),
        ("schema", string(CHILD_ATMOSPHERE_GRID_MANIFEST_SCHEMA)),
        (
            "static_heights",
            static_height_value_for_grid(static_identity),
        ),
        ("status", string(CHILD_ATMOSPHERE_GRID_STATUS)),
        ("synthetic_detail", JsonValue::Bool(false)),
        ("synthetic_translation", JsonValue::Bool(false)),
        ("transform", transform_value_for_grid(grid)),
    ])
}

fn receipt_value_for_grid(
    receipt: &ChildAtmosphereReceipt,
    static_identity: &ChildAtmosphereStaticHeightIdentity,
    grid: ChildGridSpec,
    records: &[CarrierRecord],
) -> JsonValue {
    object(vec![
        ("abi", string(CHILD_ATMOSPHERE_GRID_ABI)),
        ("all_finite", JsonValue::Bool(true)),
        ("arena_nbytes", u64_number(receipt.arena_nbytes)),
        ("arena_sha256", string(&receipt.arena_sha256)),
        ("authority", authority_value()),
        ("carrier_count", number(records.len())),
        (
            "carrier_sha256",
            JsonValue::Array(
                records
                    .iter()
                    .map(|record| {
                        object(vec![
                            (
                                "below_source_support_words",
                                number(record.below_support_words),
                            ),
                            ("name", string(record.plan.name)),
                            ("sha256", string(&record.sha256)),
                            (
                                "source_first_level_sha256",
                                optional_string(record.source_first_level_sha256.as_deref()),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("claim_limit", string(CHILD_ATMOSPHERE_GRID_CLAIM_LIMIT)),
        ("current_a_pairs_raw_bit_equal", JsonValue::Bool(true)),
        ("grid", grid_value_for_grid(grid)),
        ("manifest_sha256", string(&receipt.manifest_sha256)),
        ("model", string("GM1")),
        (
            "parent_checkpoint_sha256",
            string(&receipt.parent_checkpoint_sha256),
        ),
        ("parent_plan_sha256", string(&receipt.parent_plan_sha256)),
        (
            "producer_executable_sha256",
            string(&receipt.producer_executable_sha256),
        ),
        (
            "producer_source_sha256",
            string(&receipt.producer_source_sha256),
        ),
        ("qa_nonnegative", JsonValue::Bool(true)),
        ("schema", string(CHILD_ATMOSPHERE_GRID_SCHEMA)),
        (
            "static_heights_manifest_sha256",
            string(&static_identity.static_heights.manifest_sha256),
        ),
        (
            "static_heights_receipt_sha256",
            string(&static_identity.static_heights.receipt_sha256),
        ),
        (
            "static_heights_validation_sha256",
            string(&static_identity.validation_receipt_sha256),
        ),
        ("status", string(CHILD_ATMOSPHERE_GRID_STATUS)),
        ("stream_chunk_bytes", number(STREAM_CHUNK_BYTES)),
        ("synthetic_detail", JsonValue::Bool(false)),
        ("synthetic_translation", JsonValue::Bool(false)),
        ("tkea_excluded_for_fresh_child", JsonValue::Bool(true)),
    ])
}

fn carrier_record_value(record: &CarrierRecord) -> JsonValue {
    object(vec![
        (
            "authority",
            string(match record.plan.disposition {
                Disposition::Cm1Interpolate | Disposition::Zero => "CM1 r21.1 restart_read.F",
                Disposition::Copy => "CM1 r21.1 init3d.F",
                Disposition::ArwenInterpolate => {
                    "ArWen WSM6 persistence seam using CM1 r21.1 restart-grid arithmetic"
                }
            }),
        ),
        (
            "below_source_support_words",
            number(record.below_support_words),
        ),
        ("disposition", string(record.plan.disposition.as_str())),
        ("dtype", string("little-endian-f32")),
        ("name", string(record.plan.name)),
        ("nbytes", u64_number(record.nbytes)),
        ("offset_bytes", u64_number(record.offset_bytes)),
        ("order", string("C")),
        ("sha256", string(&record.sha256)),
        ("shape", number_array(&record.shape)),
        ("source_carrier", optional_string(record.plan.source)),
        (
            "source_first_level_sha256",
            optional_string(record.source_first_level_sha256.as_deref()),
        ),
        ("stagger", string(record.plan.shape.stagger().as_str())),
    ])
}

fn authority_value() -> JsonValue {
    object(vec![
        (
            "init3d",
            object(vec![
                ("copy_lines", string("2280-2333")),
                ("sha256", string(CM1_INIT3D_SHA256)),
            ]),
        ),
        ("model_source", string("CM1 r21.1")),
        (
            "restart_read",
            object(vec![
                ("bracket_lines", string("539-609")),
                ("held_zero_lines", string("2657-2664")),
                ("interpolation_lines", string("3302-3465")),
                ("sha256", string(CM1_RESTART_READ_SHA256)),
                ("tke_exclusion_lines", string("1206-1211")),
            ]),
        ),
        ("tarball_sha256", string(CM1_R21_AUTHORITY_SHA256)),
    ])
}

fn parent_value(prepared: &PreparedInputs) -> JsonValue {
    object(vec![
        (
            "checkpoint_artifact_sha256",
            string(&prepared.checkpoint_sha256),
        ),
        (
            "checkpoint_contract_sha256",
            string(
                prepared
                    .restored
                    .forecast()
                    .receipt()
                    .contract_sha256
                    .as_str(),
            ),
        ),
        (
            "checkpoint_raw_arena_sha256",
            string(prepared.restored.raw_arena_sha256()),
        ),
        (
            "operator_manifest_sha256",
            string(prepared.restored.operator_manifest().sha256()),
        ),
        ("plan_schema", string(CHILD_REMAP_PLAN_SCHEMA)),
        ("plan_sha256", string(&prepared.plan_sha256)),
        (
            "source_product_manifest_sha256",
            string(GM1_F02_ROI_PRODUCT_MANIFEST_SHA256),
        ),
    ])
}

fn grid_value() -> JsonValue {
    object(vec![
        (
            "dx_m_f64_bits",
            string(&format!("{:016x}", 50.0f64.to_bits())),
        ),
        (
            "dy_m_f64_bits",
            string(&format!("{:016x}", 50.0f64.to_bits())),
        ),
        ("nx", number(CHILD_NX)),
        ("ny", number(CHILD_NY)),
        ("nz", number(CHILD_NZ)),
        ("stretch_abi", string(CHILD_STRETCH_ABI)),
        ("stretch_grid_sha256", string(CHILD_STRETCH_GRID_SHA256)),
        ("stretch_schema", string(CHILD_STRETCH_SCHEMA)),
        ("terrain_fixed_domain", JsonValue::Bool(true)),
        (
            "ztop_m_f64_bits",
            string(&format!("{:016x}", 15_000.0f64.to_bits())),
        ),
    ])
}

fn grid_value_for_grid(grid: ChildGridSpec) -> JsonValue {
    object(vec![
        (
            "domain_x_m_f64_bits",
            string(&format!(
                "{:016x}",
                (grid.nx() as f64 * grid.dx_m()).to_bits()
            )),
        ),
        (
            "domain_y_m_f64_bits",
            string(&format!(
                "{:016x}",
                (grid.ny() as f64 * grid.dy_m()).to_bits()
            )),
        ),
        (
            "dx_m_f64_bits",
            string(&format!("{:016x}", grid.dx_m().to_bits())),
        ),
        (
            "dy_m_f64_bits",
            string(&format!("{:016x}", grid.dy_m().to_bits())),
        ),
        (
            "grid_spec_abi",
            string("gm1.source-coordinate-child-grid-spec/v1"),
        ),
        ("nx", number(grid.nx())),
        ("ny", number(grid.ny())),
        ("nz", number(grid.nz())),
        ("stretch_abi", string(CHILD_STRETCH_ABI)),
        ("stretch_grid_sha256", string(CHILD_STRETCH_GRID_SHA256)),
        ("stretch_schema", string(CHILD_STRETCH_SCHEMA)),
        ("terrain_fixed_domain", JsonValue::Bool(true)),
        (
            "ztop_m_f64_bits",
            string(&format!("{:016x}", grid.ztop_m().to_bits())),
        ),
    ])
}

fn static_height_value() -> JsonValue {
    object(vec![
        ("abi", string(STATIC_HEIGHT_ABI)),
        ("manifest_sha256", string(STATIC_HEIGHT_MANIFEST_SHA256)),
        ("operator_commit", string(STATIC_HEIGHT_OPERATOR_COMMIT)),
        (
            "source_terrain_manifest_sha256",
            string(CHILD_TERRAIN_MANIFEST_SHA256),
        ),
        (
            "source_terrain_payload_sha256",
            string(CHILD_TERRAIN_PAYLOAD_SHA256),
        ),
        (
            "source_terrain_receipt_sha256",
            string(CHILD_TERRAIN_RECEIPT_SHA256),
        ),
        (
            "validation_receipt_sha256",
            string(STATIC_HEIGHT_VALIDATION_RECEIPT_SHA256),
        ),
        ("zf_sha256", string(STATIC_HEIGHT_ZF_SHA256)),
        ("zh_sha256", string(STATIC_HEIGHT_ZH_SHA256)),
    ])
}

fn static_height_value_for_grid(identity: &ChildAtmosphereStaticHeightIdentity) -> JsonValue {
    let static_heights = &identity.static_heights;
    object(vec![
        ("abi", string(CHILD_STATIC_HEIGHT_GRID_ABI)),
        ("manifest_sha256", string(&static_heights.manifest_sha256)),
        (
            "producer_executable_sha256",
            string(&static_heights.producer_executable_sha256),
        ),
        (
            "producer_source_sha256",
            string(&static_heights.producer_source_sha256),
        ),
        ("receipt_sha256", string(&static_heights.receipt_sha256)),
        (
            "source_terrain_manifest_sha256",
            string(&static_heights.terrain.manifest_sha256),
        ),
        (
            "source_terrain_payload_sha256",
            string(&static_heights.terrain.payload_sha256),
        ),
        (
            "source_terrain_receipt_sha256",
            string(&static_heights.terrain.receipt_sha256),
        ),
        (
            "validation_receipt_sha256",
            string(&identity.validation_receipt_sha256),
        ),
        ("zf_sha256", string(&static_heights.zf_sha256)),
        ("zh_sha256", string(&static_heights.zh_sha256)),
    ])
}

fn disposition_value() -> JsonValue {
    object(vec![
        (
            "child_boundary_initialize",
            string_array(&["radbce", "radbcn", "radbcs", "radbcw"]),
        ),
        (
            "child_closure_initialize",
            string_array(&["closure.tkea", "closure.tke3d", "closure.tketen"]),
        ),
        (
            "child_hydrostatic_reconstruct",
            string_array(&["prs", "rho", "rr", "rf"]),
        ),
        (
            "closure_tkea_reason",
            string("restart_read.F:1206-1211 suppresses TKE read when interp_on_restart"),
        ),
        (
            "held_tendency_reason",
            string("restart_read.F:2657-2664 assigns positive 0.0 after interpolation"),
        ),
    ])
}

fn transform_value() -> JsonValue {
    object(vec![
        (
            "below_support",
            string("when the new sigma lies below old physical level 1, prep_for_interp duplicates old level 1 into both data brackets; no lapse-rate or linear extrapolation"),
        ),
        (
            "face_array_coordinates",
            string("u x-face parent-array coordinate=149.5+0.5*i (physical parent mass-coordinate=149.0+0.5*i); v y-face analog"),
        ),
        (
            "horizontal_mass_coordinates",
            string("parent_i=149.25+0.5*child_i; parent_j=149.25+0.5*child_j"),
        ),
        (
            "interpolation",
            string("CM1 interp_on_restart normalized sigma/sigmaf trilinear"),
        ),
        (
            "rounding",
            string("rx/ry/rz and each parenthesized w1..w8 RHS are binary32; weights widen to binary64; eight data*weight terms sum in restart_read.F order; one binary32 store"),
        ),
        (
            "strict_search",
            string("advance while old[next] < new; an exact old coordinate uses the preceding bracket with fraction 1.0"),
        ),
        ("vertical_mass_coordinate", string("sigma")),
        ("vertical_w_coordinate", string("sigmaf")),
    ])
}

fn transform_value_for_grid(grid: ChildGridSpec) -> JsonValue {
    let scalar_first = parent_fractional_xy_for_grid(grid, HorizontalStagger::Scalar, 0, 0)
        .expect("admitted grid has a scalar origin");
    let scalar_last = parent_fractional_xy_for_grid(
        grid,
        HorizontalStagger::Scalar,
        grid.nx() - 1,
        grid.ny() - 1,
    )
    .expect("admitted grid has a scalar endpoint");
    object(vec![
        (
            "below_support",
            string("when the new sigma lies below old physical level 1, prep_for_interp duplicates old level 1 into both data brackets; no lapse-rate or linear extrapolation"),
        ),
        (
            "horizontal_mapping",
            string("source-coordinate child centers and native x/y faces map through the sealed child grid to the accepted parent array coordinates; no 50 m byte resizing or synthetic subgrid detail"),
        ),
        (
            "interpolation",
            string("CM1 interp_on_restart normalized sigma/sigmaf trilinear"),
        ),
        (
            "rounding",
            string("rx/ry/rz and each parenthesized w1..w8 RHS are binary32; weights widen to binary64; eight data*weight terms sum in restart_read.F order; one binary32 store"),
        ),
        (
            "scalar_source_first_i_f64_bits",
            string(&format!("{:016x}", scalar_first.0.to_bits())),
        ),
        (
            "scalar_source_first_j_f64_bits",
            string(&format!("{:016x}", scalar_first.1.to_bits())),
        ),
        (
            "scalar_source_last_i_f64_bits",
            string(&format!("{:016x}", scalar_last.0.to_bits())),
        ),
        (
            "scalar_source_last_j_f64_bits",
            string(&format!("{:016x}", scalar_last.1.to_bits())),
        ),
        (
            "strict_search",
            string("advance while old[next] < new; an exact old coordinate uses the preceding bracket with fraction 1.0"),
        ),
        ("vertical_mass_coordinate", string("sigma")),
        ("vertical_w_coordinate", string("sigmaf")),
    ])
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

fn optional_string(value: Option<&str>) -> JsonValue {
    value.map_or(JsonValue::Null, string)
}

fn number(value: usize) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn u64_number(value: u64) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn number_array(values: &[usize]) -> JsonValue {
    JsonValue::Array(values.iter().map(|value| number(*value)).collect())
}

fn string_array(values: &[&str]) -> JsonValue {
    JsonValue::Array(values.iter().map(|value| string(value)).collect())
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn verify_directory_inventory(directory: &Path, expected: &[&str], label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(directory)
        .map_err(|error| invalid(format!("could not inspect {label} directory: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!("{label} input is not a real directory")));
    }
    let expected = expected
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(invalid(format!(
                "{label} directory contains a non-regular entry"
            )));
        }
        actual.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| invalid(format!("{label} directory has a non-UTF-8 name")))?,
        );
    }
    if actual != expected {
        return Err(invalid(format!(
            "{label} directory inventory is {actual:?}; expected {expected:?}"
        )));
    }
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
    let mut file = File::open(path).map_err(|error| {
        invalid(format!(
            "could not open {label} {}: {error}",
            path.display()
        ))
    })?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)?;
    if bytes.len() as u64 != metadata.len() {
        return Err(invalid(format!("{label} size changed while reading")));
    }
    Ok(bytes)
}

fn hash_regular_bounded(path: &Path, maximum: u64, label: &str) -> Result<String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("could not inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        return Err(invalid(format!(
            "{label} is not a bounded real regular file"
        )));
    }
    let file = File::open(path).map_err(|error| {
        invalid(format!(
            "could not open {label} {}: {error}",
            path.display()
        ))
    })?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hasher = Sha256::default();
    let mut total = 0u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| invalid(format!("{label} byte count overflowed")))?;
        hasher.update(&buffer[..count]);
    }
    if total != metadata.len() {
        return Err(invalid(format!("{label} size changed while hashing")));
    }
    Ok(sha256::hex(&hasher.finalize()))
}

fn verify_exact_hash(path: &Path, maximum: u64, expected_sha256: &str, label: &str) -> Result<()> {
    let actual = hash_regular_bounded(path, maximum, label)?;
    if actual != expected_sha256 {
        return Err(invalid(format!(
            "{label} SHA-256 is {actual}; expected {expected_sha256}"
        )));
    }
    Ok(())
}

fn verify_exact_file(path: &Path, nbytes: u64, expected_sha256: &str, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("could not inspect {label}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() != nbytes {
        return Err(invalid(format!(
            "{label} is not an exact {nbytes}-byte regular file"
        )));
    }
    verify_exact_hash(path, nbytes, expected_sha256, label)
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(format!("{label} is not a SHA-256 digest")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child_remap::{parent_fractional_xy, HARD_5M_CHILD_GRID};

    #[test]
    fn carrier_inventory_is_sorted_and_source_exact() {
        assert_eq!(CARRIER_PLANS.len(), 28);
        assert!(CARRIER_PLANS
            .windows(2)
            .all(|pair| pair[0].name < pair[1].name));
        assert!(!CARRIER_PLANS.iter().any(|plan| plan.name.contains("tke")));
        assert_eq!(
            CARRIER_PLANS
                .iter()
                .filter(|plan| matches!(plan.disposition, Disposition::Cm1Interpolate))
                .count(),
            7
        );
        assert_eq!(
            CARRIER_PLANS
                .iter()
                .filter(|plan| matches!(plan.disposition, Disposition::ArwenInterpolate))
                .count(),
            11
        );
        assert_eq!(
            CARRIER_PLANS
                .iter()
                .filter(|plan| matches!(plan.disposition, Disposition::Copy))
                .count(),
            6
        );
        assert_eq!(
            CARRIER_PLANS
                .iter()
                .filter(|plan| matches!(plan.disposition, Disposition::Zero))
                .count(),
            4
        );
    }

    #[test]
    fn accepted_grid_layout_is_the_historical_raw_byte_layout() {
        for plan in CARRIER_PLANS {
            assert_eq!(
                plan.shape.shape_for_grid(ACCEPTED_50M_CHILD_GRID),
                plan.shape.shape()
            );
        }
        assert_eq!(
            child_atmosphere_arena_nbytes_for_grid(ACCEPTED_50M_CHILD_GRID).unwrap(),
            SEALED_CHILD_ATMOSPHERE_ARENA_NBYTES
        );
        assert_eq!(
            child_atmosphere_arena_nbytes_for_grid(HARD_5M_CHILD_GRID).unwrap(),
            HARD_5M_CHILD_ATMOSPHERE_ARENA_NBYTES
        );
        assert_eq!(HARD_5M_CHILD_ATMOSPHERE_ARENA_NBYTES, 84_027_600_000);
        let layout = child_atmosphere_carrier_layout_for_grid(HARD_5M_CHILD_GRID).unwrap();
        assert_eq!(layout.len(), 28);
        assert_eq!(layout[0].name, "component.effc");
        assert_eq!(layout[0].offset_nbytes, 0);
        assert_eq!(layout[0].nbytes, 2_700_000_000);
        let qa = layout.iter().find(|value| value.name == "qa").unwrap();
        assert_eq!(qa.offset_nbytes, 46_152_000_000);
        assert_eq!(qa.shape, [6, 75, 3_000, 3_000]);
        let wa = layout.iter().find(|value| value.name == "wa").unwrap();
        assert_eq!(wa.offset_nbytes, 81_291_600_000);
        assert_eq!(wa.nbytes, 2_736_000_000);
        assert_eq!(
            wa.offset_nbytes + wa.nbytes,
            HARD_5M_CHILD_ATMOSPHERE_ARENA_NBYTES
        );
    }

    #[test]
    fn accepted_grid_horizontal_maps_match_the_historical_wrapper_raw_bits() {
        for kind in [
            ShapeKind::Mass,
            ShapeKind::Packed,
            ShapeKind::XFace,
            ShapeKind::YFace,
            ShapeKind::ZFace,
            ShapeKind::Surface,
        ] {
            let (x_map, y_map, target_nx, target_ny) =
                child_horizontal_maps_for_grid(kind, ACCEPTED_50M_CHILD_GRID).unwrap();
            let (legacy_x, legacy_y, legacy_nx, legacy_ny) = child_horizontal_maps(kind).unwrap();
            assert_eq!((target_nx, target_ny), (legacy_nx, legacy_ny));
            assert_eq!(x_map, legacy_x);
            assert_eq!(y_map, legacy_y);
            for (i, actual) in x_map.iter().enumerate() {
                let source_coordinate = parent_fractional_xy(kind.stagger(), i, 0).unwrap().0;
                assert_eq!(
                    *actual,
                    horizontal_point(source_coordinate, kind.source_shape()[3]).unwrap()
                );
            }
            for (j, actual) in y_map.iter().enumerate() {
                let source_coordinate = parent_fractional_xy(kind.stagger(), 0, j).unwrap().1;
                assert_eq!(
                    *actual,
                    horizontal_point(source_coordinate, kind.source_shape()[2]).unwrap()
                );
            }
        }
    }

    #[test]
    fn hard_5m_atmosphere_staggers_fit_the_sealed_parent_support() {
        for kind in [
            ShapeKind::Mass,
            ShapeKind::Packed,
            ShapeKind::XFace,
            ShapeKind::YFace,
            ShapeKind::ZFace,
            ShapeKind::Surface,
        ] {
            let (x_map, y_map, target_nx, target_ny) =
                child_horizontal_maps_for_grid(kind, HARD_5M_CHILD_GRID).unwrap();
            let shape = kind.shape_for_grid(HARD_5M_CHILD_GRID);
            let expected_xy = match kind {
                ShapeKind::Packed => (shape[3], shape[2]),
                ShapeKind::Surface => (shape[1], shape[0]),
                ShapeKind::Mass | ShapeKind::XFace | ShapeKind::YFace | ShapeKind::ZFace => {
                    (shape[shape.len() - 1], shape[shape.len() - 2])
                }
            };
            assert_eq!((target_nx, target_ny), expected_xy);
            assert_eq!((x_map.len(), y_map.len()), expected_xy);
            assert!(x_map.iter().all(|point| {
                point.upper < kind.source_shape()[3]
                    && point.fraction.is_finite()
                    && (0.0..=1.0).contains(&point.fraction)
            }));
            assert!(y_map.iter().all(|point| {
                point.upper < kind.source_shape()[2]
                    && point.fraction.is_finite()
                    && (0.0..=1.0).contains(&point.fraction)
            }));
        }
    }

    #[test]
    fn hard_5m_static_identity_is_independent_and_exact() {
        let identity = hard_5m_child_static_height_identity().unwrap();
        assert_eq!(identity.static_heights().grid, HARD_5M_CHILD_GRID);
        assert_eq!(
            identity.static_heights().manifest_sha256,
            HARD_5M_STATIC_HEIGHT_MANIFEST_SHA256
        );
        assert_eq!(
            identity.validation_receipt_sha256(),
            HARD_5M_STATIC_HEIGHT_VALIDATION_SHA256
        );
        assert_eq!(
            identity.static_heights().terrain.payload_sha256,
            HARD_5M_TERRAIN_PAYLOAD_SHA256
        );
        assert_ne!(CHILD_ATMOSPHERE_GRID_SCHEMA, CHILD_ATMOSPHERE_SCHEMA);
        assert_ne!(CHILD_ATMOSPHERE_GRID_ABI, CHILD_ATMOSPHERE_ABI);
    }

    #[test]
    fn hard_5m_grid_envelope_binds_exact_extent_and_source_coordinates() {
        let grid = grid_value_for_grid(HARD_5M_CHILD_GRID).canonical_bytes();
        let grid = std::str::from_utf8(&grid).unwrap();
        assert!(grid.contains("\"nx\":3000"));
        assert!(grid.contains("\"ny\":3000"));
        assert!(grid.contains("\"dx_m_f64_bits\":\"4014000000000000\""));
        assert!(grid.contains("\"domain_x_m_f64_bits\":\"40cd4c0000000000\""));
        let transform = transform_value_for_grid(HARD_5M_CHILD_GRID).canonical_bytes();
        let transform = std::str::from_utf8(&transform).unwrap();
        assert!(transform.contains("\"scalar_source_first_i_f64_bits\":\"4062a0cccccccccd\""));
        assert!(transform.contains("\"scalar_source_last_i_f64_bits\":\"4072af999999999a\""));
        assert!(transform.contains("no 50 m byte resizing or synthetic subgrid detail"));
    }

    #[test]
    fn strict_less_exact_coordinate_uses_preceding_bracket() {
        let point = horizontal_point(150.0, 301).unwrap();
        assert_eq!(point.lower, 149);
        assert_eq!(point.upper, 150);
        assert_eq!(point.fraction.to_bits(), 1.0f32.to_bits());
        let fractional = horizontal_point(149.25, 300).unwrap();
        assert_eq!((fractional.lower, fractional.upper), (149, 150));
        assert_eq!(fractional.fraction.to_bits(), 0.25f32.to_bits());
        assert!(horizontal_point(299.5, 300).is_err());
    }

    #[test]
    fn official_lower_support_rule_duplicates_first_plane() {
        let mapping = vertical_axis(&old_mass_sigma(), &[10.0, 100.0, 300.0], PARENT_NZ).unwrap();
        assert_eq!((mapping[0].lower, mapping[0].upper), (0, 0));
        assert!(mapping[0].below_support);
        assert_eq!(mapping[0].fraction.to_bits(), 0.55f32.to_bits());
        assert_eq!((mapping[1].lower, mapping[1].upper), (0, 0));
        assert!(!mapping[1].below_support);
        assert_eq!(mapping[1].fraction.to_bits(), 1.0f32.to_bits());
        assert_eq!((mapping[2].lower, mapping[2].upper), (0, 1));
        assert_eq!(mapping[2].fraction.to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn source_order_arithmetic_uses_default_real_weights() {
        let values = [
            f32::from_bits(0x48ee_29c7),
            f32::from_bits(0xc809_0cd7),
            f32::from_bits(0xc8c0_9378),
            f32::from_bits(0x48fa_77be),
            f32::from_bits(0xc94e_4301),
            f32::from_bits(0xc69c_6fd9),
            f32::from_bits(0xc971_43bf),
            f32::from_bits(0xc8bf_0dba),
        ];
        let result = source_order_interp(values, weights(0.840_822_16, 0.950_475_9, 0.318_874_57));
        assert_eq!(result.to_bits(), 0xc928_96dc);
    }

    #[test]
    fn sealed_child_grid_has_five_below_parent_mass_levels_and_no_face_levels() {
        let grid = sealed_child_stretch_grid().unwrap();
        let mass = vertical_axis(&old_mass_sigma(), &grid.sigma[1..=CHILD_NZ], PARENT_NZ).unwrap();
        let face = vertical_axis(
            &old_face_sigma(),
            &grid.sigmaf[1..=CHILD_NZ + 1],
            PARENT_NZ + 1,
        )
        .unwrap();
        assert_eq!(mass.iter().filter(|point| point.below_support).count(), 5);
        assert_eq!(face.iter().filter(|point| point.below_support).count(), 0);
    }
}
