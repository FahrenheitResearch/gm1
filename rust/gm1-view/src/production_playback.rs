//! Strict multi-field v2 admission, paired exact-f32 blending, and coalesced loading.
//!
//! This module is deliberately display-only. It never normalizes, clamps,
//! substitutes, quantizes, or invents a carrier. All eight payloads must pass
//! their exact v2 envelope before a frame can enter playback.

use crate::{
    frame_geometry::FrameGeometry,
    geofield::{
        GEO_FIELD_ABI, GeoFieldAdmission, GeoFieldDescriptor, GeoFieldGeometry, GeoGrid, GeoTime,
        GeoreferencedFrame,
    },
    input::{Gm1Dataset, VolumeField},
    lighting::{T300_CENTER_LATITUDE_DEG, T300_CENTER_LONGITUDE_DEG},
};
use cm1_state::child_post_step::{
    CHILD_POST_STEP_HISTORY_PROFILE, IDEALIZED_POST_STEP_HISTORY_PROFILE,
};
use cm1_state::child_view_pack::{
    CHILD_PRODUCTION_VIEW_PACK_ABI, CHILD_PRODUCTION_VIEW_PACK_MANIFEST_SCHEMA,
    CHILD_PRODUCTION_VIEW_PACK_RECEIPT_SCHEMA,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

pub const EXACT_F32_BLEND_FORMULA: &str = "alpha=0 -> A word exact; alpha=1 -> B word exact; otherwise f32::mul_add(B-A,alpha,A) in source word order";
pub const ADMITTED_PRODUCTION_VIEW_PRODUCER_COMMIT: &str =
    "3d047cded68bb67f9913e5e107252eaa38e791ee";
pub const ADMITTED_PRODUCTION_VIEW_SOURCE_SHA256: &str =
    "f1572d7a4aa4521de071c29818af7c1f9213d8e6e1757b2c4b49c9431f9d373e";
pub const ADMITTED_PRODUCTION_VIEW_EXECUTABLE_SHA256: &str =
    "17757a02a81a3d3e67aafbd795cfb1345306a2dd59cbd9ac01780148621d63cb";
pub const PRODUCTION_RENDER_ADAPTER_ID: &str = "gm1.strict-production-view-v2-residency/v1";
pub const REAL_CHILD_VIEW_SOURCE_PROFILE_ID: &str = "gm1.real-child-50m-source-authority/v1";
pub const IDEALIZED_VIEW_SOURCE_PROFILE_ID: &str = "gm1.idealized-isnd7-iinit1-source-authority/v1";
pub const CARTESIAN_HISTORY_VIEW_SOURCE_PROFILE_ID: &str = "gm1.cartesian-history/v1";
pub const IDEALIZED_VIEW_SOURCE_AUTHORITY_SHA256: &str =
    "d4857df7e121f48a27cc15b3ff8845b0f064a790f4954a08d24588af0d5f6366";
pub const IDEALIZED_VIEW_SOURCE_PROFILE_STATUS: &str =
    "REFUSED_NO_ADMITTED_IDEALIZED_HISTORY_OPERATOR_AUTHORITY";

const STATUS: &str = "PASS_EXACT_MULTI_FIELD_VIEW_TRANSPORT_NO_REFLECTIVITY";
const CLAIM_LIMIT: &str = "one-pass additive, non-quantized binary32 view transport from one independently authenticated GM1 CM1FHS1 history: exact qc+qi plus unmodified u3d/v3d/w3d/prs/terrain_msl/zf/total_geopotential_zf source words; raw staggered winds enable a separately pinned exact updraft/vorticity derivation, but this pack contains no reflectivity carrier, invented boundary values, checkpoint, restart, independent meteorological product, forecast-skill, storm, tornado, or performance claim";
const RAW_COPY_FORMULA: &str = "out[word]=source_carrier[word] (IEEE-754 binary32 word exact)";
const QC_QI_FORMULA: &str = "out[word]=f32(q3d[1*words+word]+q3d[3*words+word])";
const REFLECTIVITY_REASON: &str = "the admitted 13-carrier history contains no reflectivity carrier and no source-exact reflectivity diagnostic is bound";
const UPDRAFT_REASON: &str = "exact native w3d z-face words are present; mass-center derivation is deferred and must pin its formula";
const VORTICITY_REASON: &str = "exact native u3d/v3d face words are present; C-grid vorticity and boundary disposition are deferred and must pin their formula";
const MANIFEST_FILE: &str = "manifest.json";
const RECEIPT_FILE: &str = "receipt.json";
const MAX_JSON_BYTES: u64 = 128 * 1024;

const LIVE_HISTORY_CONTRACT: &str =
    "a5d3f1c784544cf0398a0499d51fc77792b74fcab8129755b9e43b872c9dbb90";
const LIVE_OPERATOR_MANIFEST: &str =
    "6112fb9302f57f82b09118b3f38c6ae8dd3d82c59eb14fd01940f2dc43428c09";
const RESTORED_HISTORY_CONTRACT: &str =
    "d2be76466ff9848abc2239edfb9df03013b1ccd08ee061142f05e1bd96458894";
const RESTORED_OPERATOR_MANIFEST: &str =
    "a52400db650092944f152276ed764ad3cb852473f9b8b9b805b7ed68b4c1f3e7";
const PRODUCTION_HISTORY_CONTRACT: &str =
    "2d2df2c9c4fb2df6b383ed3a09be61a4e420212e888ad6136f40e716acc88cdd";
const PRODUCTION_OPERATOR_MANIFEST: &str =
    "0e834606732821d6f54e770f3c41f01ae6fe546b5a5b04048091fa67bb58a5fe";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gm1ViewProfileClass {
    RealChildProduction,
    IdealizedSingleSoundingWarmBubble,
    /// Uniform-spacing Cartesian history frame whose grid geometry is bound at
    /// run time from the publisher's sibling geometry document, because the
    /// envelope itself states no spacing.
    CartesianHistory,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gm1ViewTerrainClass {
    TerrainFollowingSource,
    FlatZeroTerrain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gm1ViewVerticalGridClass {
    SourceStretched,
    Uniform { dz_m: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gm1ViewSourceProfile {
    pub id: &'static str,
    pub scientific_label: &'static str,
    pub class: Gm1ViewProfileClass,
    pub center_latitude_deg: f64,
    pub center_longitude_deg: f64,
    pub horizontal_crs: &'static str,
    pub shape_zyx: [usize; 3],
    pub dx_m: f64,
    pub dy_m: f64,
    pub model_top_m: f64,
    pub vertical_grid: Gm1ViewVerticalGridClass,
    pub terrain: Gm1ViewTerrainClass,
}

impl Gm1ViewSourceProfile {
    pub fn domain_label(self) -> String {
        format!(
            "{}x{}x{} • dx={:.0} m • dy={:.0} m • top={:.0} m",
            self.shape_zyx[2],
            self.shape_zyx[1],
            self.shape_zyx[0],
            self.dx_m,
            self.dy_m,
            self.model_top_m,
        )
    }

    pub const fn vertical_grid_label(self) -> &'static str {
        match self.vertical_grid {
            Gm1ViewVerticalGridClass::SourceStretched => {
                "source-stretched; exact static and terrain-following zf faces"
            }
            Gm1ViewVerticalGridClass::Uniform { dz_m: 200 } => "uniform dz=200 m",
            Gm1ViewVerticalGridClass::Uniform { .. } => "uniform dz from admitted profile",
        }
    }

    pub const fn terrain_label(self) -> &'static str {
        match self.terrain {
            Gm1ViewTerrainClass::TerrainFollowingSource => {
                "source terrain-following; exact terrain_msl"
            }
            Gm1ViewTerrainClass::FlatZeroTerrain => "flat zero terrain",
        }
    }

    pub fn georeferenced_grid(self) -> GeoGrid {
        GeoGrid {
            center_latitude_deg: self.center_latitude_deg,
            center_longitude_deg: self.center_longitude_deg,
            nx: self.shape_zyx[2],
            ny: self.shape_zyx[1],
            nz: self.shape_zyx[0],
            dx_m: self.dx_m,
            dy_m: self.dy_m,
            model_top_m: self.model_top_m,
            horizontal_crs: self.horizontal_crs,
            vertical_coordinate: self.vertical_grid_label(),
        }
    }
}

pub const REAL_CHILD_VIEW_SOURCE_PROFILE: Gm1ViewSourceProfile = Gm1ViewSourceProfile {
    id: REAL_CHILD_VIEW_SOURCE_PROFILE_ID,
    scientific_label: "Real 50 m terrain-following GM1 child",
    class: Gm1ViewProfileClass::RealChildProduction,
    center_latitude_deg: T300_CENTER_LATITUDE_DEG,
    center_longitude_deg: T300_CENTER_LONGITUDE_DEG,
    horizontal_crs: "local ENU centered on sealed GM1 child center",
    shape_zyx: [75, 300, 300],
    dx_m: 50.0,
    dy_m: 50.0,
    model_top_m: 15_000.0,
    vertical_grid: Gm1ViewVerticalGridClass::SourceStretched,
    terrain: Gm1ViewTerrainClass::TerrainFollowingSource,
};

/// Renderer metadata only. No bytes can resolve to this profile until a
/// first-class idealized history/operator authority is sealed by persistence.
pub const IDEALIZED_VIEW_SOURCE_PROFILE: Gm1ViewSourceProfile = Gm1ViewSourceProfile {
    id: IDEALIZED_VIEW_SOURCE_PROFILE_ID,
    scientific_label: "CM1 isnd=7 / iinit=1 single-sounding warm bubble",
    class: Gm1ViewProfileClass::IdealizedSingleSoundingWarmBubble,
    center_latitude_deg: 0.0,
    center_longitude_deg: 0.0,
    horizontal_crs: "idealized Cartesian x/y; no Earth geolocation",
    shape_zyx: [75, 300, 300],
    dx_m: 50.0,
    dy_m: 50.0,
    model_top_m: 15_000.0,
    vertical_grid: Gm1ViewVerticalGridClass::Uniform { dz_m: 200 },
    terrain: Gm1ViewTerrainClass::FlatZeroTerrain,
};

/// Renderer metadata for one Cartesian history frame, sized by the geometry
/// the publisher recorded beside the envelope rather than by a pinned shape.
/// Nothing here admits source bytes; the strict envelope gate already ran.
pub fn cartesian_history_view_source_profile(geometry: FrameGeometry) -> Gm1ViewSourceProfile {
    Gm1ViewSourceProfile {
        id: CARTESIAN_HISTORY_VIEW_SOURCE_PROFILE_ID,
        scientific_label: "Uniform-spacing Cartesian history frame",
        class: Gm1ViewProfileClass::CartesianHistory,
        // A Cartesian run carries no map projection, so the viewer states no
        // Earth location for it rather than borrowing another domain's.
        center_latitude_deg: 0.0,
        center_longitude_deg: 0.0,
        horizontal_crs: "idealized Cartesian x/y; no Earth geolocation",
        shape_zyx: geometry.shape_zyx(),
        dx_m: geometry.dx_m,
        dy_m: geometry.dy_m,
        model_top_m: geometry.model_top_m,
        vertical_grid: Gm1ViewVerticalGridClass::Uniform {
            dz_m: geometry.dz_m as u32,
        },
        terrain: Gm1ViewTerrainClass::FlatZeroTerrain,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gm1ViewSourceAuthority {
    pub history_contract_sha256: &'static str,
    pub operator_manifest_sha256: &'static str,
    pub profile: Gm1ViewSourceProfile,
}

pub const ADMITTED_GM1_VIEW_SOURCE_AUTHORITIES: [Gm1ViewSourceAuthority; 3] = [
    Gm1ViewSourceAuthority {
        history_contract_sha256: LIVE_HISTORY_CONTRACT,
        operator_manifest_sha256: LIVE_OPERATOR_MANIFEST,
        profile: REAL_CHILD_VIEW_SOURCE_PROFILE,
    },
    Gm1ViewSourceAuthority {
        history_contract_sha256: RESTORED_HISTORY_CONTRACT,
        operator_manifest_sha256: RESTORED_OPERATOR_MANIFEST,
        profile: REAL_CHILD_VIEW_SOURCE_PROFILE,
    },
    Gm1ViewSourceAuthority {
        history_contract_sha256: PRODUCTION_HISTORY_CONTRACT,
        operator_manifest_sha256: PRODUCTION_OPERATOR_MANIFEST,
        profile: REAL_CHILD_VIEW_SOURCE_PROFILE,
    },
];

/// Resolve renderer metadata from an already admitted dataset profile. This
/// does not admit source bytes: pair/pack loaders remain responsible for the
/// exact history/operator authority gate before returning a dataset.
pub fn gm1_view_source_profile_for_dataset_profile(
    profile_id: &str,
) -> Option<Gm1ViewSourceProfile> {
    let suffix = profile_id.rsplit(';').next().unwrap_or(profile_id);
    if profile_id == CHILD_POST_STEP_HISTORY_PROFILE || suffix == REAL_CHILD_VIEW_SOURCE_PROFILE_ID
    {
        return Some(REAL_CHILD_VIEW_SOURCE_PROFILE);
    }
    if profile_id == IDEALIZED_POST_STEP_HISTORY_PROFILE
        || suffix == IDEALIZED_VIEW_SOURCE_PROFILE_ID
    {
        return Some(IDEALIZED_VIEW_SOURCE_PROFILE);
    }
    None
}

/// A live task may advance within one admitted science profile, but it may not
/// silently turn a real-child timeline into an idealized experiment (or vice
/// versa). Unknown legacy profiles remain compatible only with an exact label.
pub fn validate_gm1_view_live_profile_transition(
    current_profile_id: &str,
    next_profile_id: &str,
) -> Result<(), String> {
    let current = gm1_view_source_profile_for_dataset_profile(current_profile_id);
    let next = gm1_view_source_profile_for_dataset_profile(next_profile_id);
    match (current, next) {
        (Some(current), Some(next)) if current == next => Ok(()),
        (None, None) if current_profile_id == next_profile_id => Ok(()),
        _ => Err(format!(
            "live GM1 View source profile changed from {current_profile_id:?} to {next_profile_id:?}"
        )),
    }
}

#[derive(Clone, Copy, Debug)]
struct ChannelContract {
    name: &'static str,
    file: &'static str,
    shape: &'static [usize],
    source_carriers: &'static [&'static str],
    role: &'static str,
    units: &'static str,
    stagger: &'static str,
    formula: &'static str,
    nonnegative: Option<bool>,
}

const CHANNELS: [ChannelContract; 8] = [
    ChannelContract {
        name: "cloud_condensate_kgkg",
        file: "cloud-condensate.f32le",
        shape: &[75, 300, 300],
        source_carriers: &["q3d.qc", "q3d.qi"],
        role: "exact qc plus qi cloud condensate",
        units: "kg kg-1",
        stagger: "mass",
        formula: QC_QI_FORMULA,
        nonnegative: Some(true),
    },
    ChannelContract {
        name: "u_staggered_ms",
        file: "u3d.f32le",
        shape: &[75, 300, 301],
        source_carriers: &["u3d"],
        role: "eastward wind on native x faces",
        units: "m s-1",
        stagger: "x-face C-grid",
        formula: RAW_COPY_FORMULA,
        nonnegative: None,
    },
    ChannelContract {
        name: "v_staggered_ms",
        file: "v3d.f32le",
        shape: &[75, 301, 300],
        source_carriers: &["v3d"],
        role: "northward wind on native y faces",
        units: "m s-1",
        stagger: "y-face C-grid",
        formula: RAW_COPY_FORMULA,
        nonnegative: None,
    },
    ChannelContract {
        name: "w_staggered_ms",
        file: "w3d.f32le",
        shape: &[76, 300, 300],
        source_carriers: &["w3d"],
        role: "vertical velocity on native z faces",
        units: "m s-1",
        stagger: "z-face C-grid",
        formula: RAW_COPY_FORMULA,
        nonnegative: None,
    },
    ChannelContract {
        name: "pressure_pa",
        file: "prs.f32le",
        shape: &[75, 300, 300],
        source_carriers: &["prs"],
        role: "physical pressure",
        units: "Pa",
        stagger: "mass",
        formula: RAW_COPY_FORMULA,
        nonnegative: Some(true),
    },
    ChannelContract {
        name: "terrain_msl_m",
        file: "terrain_msl.f32le",
        shape: &[300, 300],
        source_carriers: &["terrain_msl"],
        role: "terrain height above mean sea level",
        units: "m",
        stagger: "horizontal mass",
        formula: RAW_COPY_FORMULA,
        nonnegative: Some(true),
    },
    ChannelContract {
        name: "static_zf_m",
        file: "zf.f32le",
        shape: &[76, 300, 300],
        source_carriers: &["zf"],
        role: "static geometric height on vertical faces",
        units: "m",
        stagger: "z-face mass column",
        formula: RAW_COPY_FORMULA,
        nonnegative: Some(true),
    },
    ChannelContract {
        name: "total_geopotential_zf_m",
        file: "total-geopotential-zf.f32le",
        shape: &[76, 300, 300],
        source_carriers: &["total_geopotential_zf"],
        role: "terrain-following total geopotential height on vertical faces",
        units: "m",
        stagger: "z-face mass column",
        formula: RAW_COPY_FORMULA,
        nonnegative: Some(true),
    },
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ProducerIdentity {
    commit: String,
    executable_sha256: String,
    source_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct HistoryIdentity {
    artifact_sha256: String,
    contract_sha256: String,
    payload_sha256: String,
    raw_arena_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Q3dIdentity {
    nbytes: usize,
    qc_sha256: String,
    qi_sha256: String,
    sha256: String,
    shape_species_kji: [usize; 4],
    species_order: [String; 6],
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct SourceIdentity {
    history: HistoryIdentity,
    model_time_f64_bits: String,
    operator_manifest_sha256: String,
    pair_manifest_sha256: String,
    pair_receipt_sha256: String,
    q3d: Q3dIdentity,
    step: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ChannelStatistics {
    all_finite: bool,
    maximum_f32_bits: String,
    minimum_f32_bits: String,
    nonnegative: bool,
    nonzero_count: usize,
    word_count: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ChannelManifest {
    dtype: String,
    file: String,
    formula: String,
    name: String,
    nbytes: usize,
    order: String,
    payload_sha256: String,
    role: String,
    shape: Vec<usize>,
    source_carriers: Vec<String>,
    source_sha256: Vec<String>,
    stagger: String,
    statistics: ChannelStatistics,
    units: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ChannelReceipt {
    file: String,
    name: String,
    nbytes: usize,
    payload_sha256: String,
    shape: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct OptionalDisposition {
    reason: String,
    status: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct OptionalChannels {
    reflectivity_dbz: OptionalDisposition,
    updraft_mass_center_ms: OptionalDisposition,
    vertical_vorticity_s: OptionalDisposition,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackManifest {
    abi: String,
    channels: Vec<ChannelManifest>,
    claim_limit: String,
    model: String,
    optional_channels: OptionalChannels,
    payload_set_sha256: String,
    producer: ProducerIdentity,
    qc_plus_qi_v1_payload_sha256: String,
    schema: String,
    source: SourceIdentity,
    status: String,
    synthetic_detail: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackReceipt {
    abi: String,
    channels: Vec<ChannelReceipt>,
    claim_limit: String,
    manifest_sha256: String,
    model: String,
    payload_set_sha256: String,
    producer: ProducerIdentity,
    schema: String,
    source: SourceIdentity,
    status: String,
    synthetic_detail: bool,
}

#[derive(Clone, Debug)]
pub struct ExactProductionField {
    pub name: String,
    pub role: String,
    pub units: String,
    pub stagger: String,
    pub shape: Vec<usize>,
    pub values: Arc<Vec<f32>>,
    pub values_sha256: String,
}

#[derive(Clone, Debug)]
pub struct ExactProductionFrame {
    pub source_path: PathBuf,
    pub source_profile: Gm1ViewSourceProfile,
    pub step: u64,
    pub model_time_s: f64,
    pub manifest_sha256: String,
    pub payload_set_sha256: String,
    pub history_contract_sha256: String,
    pub operator_manifest_sha256: String,
    pub fields: Vec<ExactProductionField>,
    producer: ProducerIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductionResidencyKind {
    CloudScalarVolume,
    XFaceWindVolume,
    YFaceWindVolume,
    ZFaceWindVolume,
    PressureScalarVolume,
    TerrainSurface,
    StaticVerticalFaces,
    TerrainFollowingVerticalFaces,
}

#[derive(Clone, Debug)]
pub struct ProductionResidentField {
    pub descriptor: GeoFieldDescriptor,
    pub stagger: String,
    pub residency: ProductionResidencyKind,
    pub values: Arc<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct ProductionRenderResidency {
    pub frame: GeoreferencedFrame,
    pub source_profile: Gm1ViewSourceProfile,
    pub manifest_sha256: String,
    pub payload_set_sha256: String,
    pub fields: Vec<ProductionResidentField>,
}

impl ProductionRenderResidency {
    pub fn field(&self, id: &str) -> Option<&ProductionResidentField> {
        self.fields.iter().find(|field| field.descriptor.id == id)
    }
}

impl ExactProductionFrame {
    pub fn field(&self, name: &str) -> Option<&ExactProductionField> {
        self.fields.iter().find(|field| field.name == name)
    }

    pub fn cloud_dataset(&self) -> Result<Gm1Dataset, String> {
        let cloud = self
            .field("cloud_condensate_kgkg")
            .ok_or_else(|| "production frame lacks exact cloud condensate".to_owned())?;
        let shape: [usize; 3] = cloud
            .shape
            .clone()
            .try_into()
            .map_err(|_| "production cloud field is not rank three".to_owned())?;
        let finite_min = cloud.values.iter().copied().fold(f32::INFINITY, f32::min);
        let finite_max = cloud
            .values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        Ok(Gm1Dataset {
            config_sha256: String::new(),
            terrain_msl_m: None,
            face_height_m: None,
            tornado_surface_fields: Vec::new(),
            source_path: self.source_path.clone(),
            artifact_kind: "atomic exact production-view v2 transport".to_owned(),
            artifact_sha256: self.manifest_sha256.clone(),
            payload_sha256: self.payload_set_sha256.clone(),
            contract_sha256: self.history_contract_sha256.clone(),
            profile_id: format!(
                "{};{}",
                CHILD_PRODUCTION_VIEW_PACK_ABI, self.source_profile.id
            ),
            step: self.step,
            model_time_s: self.model_time_s,
            dt_s: 0.5,
            fields: vec![VolumeField {
                id: "cloud_condensate".to_owned(),
                label: "Resolved cloud condensate".to_owned(),
                units: cloud.units.clone(),
                meaning: cloud.role.clone(),
                source: format!("{QC_QI_FORMULA}; exact v2 payload {}", cloud.values_sha256),
                shape,
                values: cloud.values.as_ref().clone(),
                values_sha256: cloud.values_sha256.clone(),
                finite_min,
                finite_max,
            }],
        })
    }

    pub fn renderer_residency(&self) -> Result<ProductionRenderResidency, String> {
        if self.fields.len() != CHANNELS.len() {
            return Err("production renderer requires the exact eight-channel v2 frame".to_owned());
        }
        let mut descriptors = Vec::with_capacity(CHANNELS.len());
        let mut resident = Vec::with_capacity(CHANNELS.len());
        for (field, contract) in self.fields.iter().zip(CHANNELS) {
            if field.name != contract.name
                || field.shape != contract.shape
                || field.values.len() != contract.shape.iter().product::<usize>()
                || field.values_sha256 != hash_f32(&field.values)
            {
                return Err(format!(
                    "production renderer field {} changed after strict admission",
                    contract.name
                ));
            }
            let display = production_display_contract(&field.name)
                .ok_or_else(|| format!("production renderer has no contract for {}", field.name))?;
            let descriptor = GeoFieldDescriptor {
                id: field.name.clone(),
                label: display.label.to_owned(),
                units: field.units.clone(),
                meaning: field.role.clone(),
                geometry: display.geometry,
                shape: field.shape.clone(),
                values_sha256: field.values_sha256.clone(),
                admission: GeoFieldAdmission::Available,
                admission_reason:
                    "exact production-view v2 manifest, receipt, payload, and in-memory hash passed"
                        .to_owned(),
                provenance: format!(
                    "{}; native stagger={}; payload={}; no remap, normalization, clamping, quantization, or derived meteorology",
                    CHILD_PRODUCTION_VIEW_PACK_ABI, field.stagger, field.values_sha256
                ),
            };
            descriptors.push(descriptor.clone());
            resident.push(ProductionResidentField {
                descriptor,
                stagger: field.stagger.clone(),
                residency: display.residency,
                values: field.values.clone(),
            });
        }
        if resident.iter().any(|field| {
            matches!(
                field.descriptor.id.as_str(),
                "reflectivity_dbz" | "vertical_vorticity_s"
            )
        }) {
            return Err(
                "production renderer cannot admit unbound derived science fields".to_owned(),
            );
        }
        Ok(ProductionRenderResidency {
            frame: GeoreferencedFrame {
                abi: GEO_FIELD_ABI,
                adapter_id: PRODUCTION_RENDER_ADAPTER_ID,
                source_identity: self.payload_set_sha256.clone(),
                grid: self.source_profile.georeferenced_grid(),
                time: GeoTime {
                    step: self.step,
                    model_time_s: self.model_time_s,
                    dt_s: 0.5,
                },
                fields: descriptors,
            },
            source_profile: self.source_profile,
            manifest_sha256: self.manifest_sha256.clone(),
            payload_set_sha256: self.payload_set_sha256.clone(),
            fields: resident,
        })
    }
}

#[derive(Clone, Copy)]
struct ProductionDisplayContract {
    label: &'static str,
    geometry: GeoFieldGeometry,
    residency: ProductionResidencyKind,
}

fn production_display_contract(name: &str) -> Option<ProductionDisplayContract> {
    let contract = match name {
        "cloud_condensate_kgkg" => ProductionDisplayContract {
            label: "Resolved cloud condensate (qc + qi)",
            geometry: GeoFieldGeometry::VolumeZyx,
            residency: ProductionResidencyKind::CloudScalarVolume,
        },
        "u_staggered_ms" => ProductionDisplayContract {
            label: "Eastward wind (native x face)",
            geometry: GeoFieldGeometry::VolumeZyx,
            residency: ProductionResidencyKind::XFaceWindVolume,
        },
        "v_staggered_ms" => ProductionDisplayContract {
            label: "Northward wind (native y face)",
            geometry: GeoFieldGeometry::VolumeZyx,
            residency: ProductionResidencyKind::YFaceWindVolume,
        },
        "w_staggered_ms" => ProductionDisplayContract {
            label: "Vertical velocity (native z face)",
            geometry: GeoFieldGeometry::VolumeZyx,
            residency: ProductionResidencyKind::ZFaceWindVolume,
        },
        "pressure_pa" => ProductionDisplayContract {
            label: "Physical pressure (mass points)",
            geometry: GeoFieldGeometry::VolumeZyx,
            residency: ProductionResidencyKind::PressureScalarVolume,
        },
        "terrain_msl_m" => ProductionDisplayContract {
            label: "Terrain height above mean sea level",
            geometry: GeoFieldGeometry::RasterYx,
            residency: ProductionResidencyKind::TerrainSurface,
        },
        "static_zf_m" => ProductionDisplayContract {
            label: "Static geometric height (vertical faces)",
            geometry: GeoFieldGeometry::VolumeZyx,
            residency: ProductionResidencyKind::StaticVerticalFaces,
        },
        "total_geopotential_zf_m" => ProductionDisplayContract {
            label: "Terrain-following total geopotential height (vertical faces)",
            geometry: GeoFieldGeometry::VolumeZyx,
            residency: ProductionResidencyKind::TerrainFollowingVerticalFaces,
        },
        _ => return None,
    };
    Some(contract)
}

#[derive(Clone, Debug)]
pub struct ProductionFramePair {
    pub a: Arc<ExactProductionFrame>,
    pub b: Arc<ExactProductionFrame>,
    pub generation: u64,
}

impl ProductionFramePair {
    pub fn new(
        a: ExactProductionFrame,
        b: ExactProductionFrame,
        generation: u64,
    ) -> Result<Self, String> {
        validate_pair(&a, &b)?;
        Ok(Self {
            a: Arc::new(a),
            b: Arc::new(b),
            generation,
        })
    }

    pub fn blend_exact_f32(&self, alpha: f32) -> Result<BlendedProductionFrame, String> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err("production playback alpha must be finite in [0,1]".to_owned());
        }
        let fields = self
            .a
            .fields
            .iter()
            .zip(&self.b.fields)
            .map(|(left, right)| {
                let values = if alpha.to_bits() == 0.0f32.to_bits() {
                    left.values.as_ref().clone()
                } else if alpha.to_bits() == 1.0f32.to_bits() {
                    right.values.as_ref().clone()
                } else {
                    left.values
                        .iter()
                        .zip(right.values.iter())
                        .enumerate()
                        .map(|(index, (a, b))| {
                            let value = (*b - *a).mul_add(alpha, *a);
                            if value.is_finite() {
                                Ok(value)
                            } else {
                                Err(format!(
                                    "blended field {} word {index} became non-finite",
                                    left.name
                                ))
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?
                };
                let values_sha256 = hash_f32(&values);
                Ok(ExactProductionField {
                    name: left.name.clone(),
                    role: left.role.clone(),
                    units: left.units.clone(),
                    stagger: left.stagger.clone(),
                    shape: left.shape.clone(),
                    values: Arc::new(values),
                    values_sha256,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let model_time_s = if alpha.to_bits() == 0.0f32.to_bits() {
            self.a.model_time_s
        } else if alpha.to_bits() == 1.0f32.to_bits() {
            self.b.model_time_s
        } else {
            (self.b.model_time_s - self.a.model_time_s)
                .mul_add(f64::from(alpha), self.a.model_time_s)
        };
        Ok(BlendedProductionFrame {
            source_steps: [self.a.step, self.b.step],
            source_manifest_sha256: [
                self.a.manifest_sha256.clone(),
                self.b.manifest_sha256.clone(),
            ],
            alpha_f32_bits: format!("{:08x}", alpha.to_bits()),
            model_time_s,
            formula: EXACT_F32_BLEND_FORMULA,
            fields,
        })
    }

    /// Blend one named field without materializing the other seven payloads.
    /// Both endpoints remain the independently admitted binary32 arrays; this
    /// is the same formula used by `blend_exact_f32` in the same word order.
    pub fn blend_field_exact_f32(
        &self,
        name: &str,
        alpha: f32,
    ) -> Result<ExactProductionField, String> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err("production playback alpha must be finite in [0,1]".to_owned());
        }
        let index = self
            .a
            .fields
            .iter()
            .position(|field| field.name == name)
            .ok_or_else(|| format!("production A/B pair lacks field {name}"))?;
        let left = &self.a.fields[index];
        let right = &self.b.fields[index];
        let values = if alpha.to_bits() == 0.0f32.to_bits() {
            left.values.as_ref().clone()
        } else if alpha.to_bits() == 1.0f32.to_bits() {
            right.values.as_ref().clone()
        } else {
            left.values
                .iter()
                .zip(right.values.iter())
                .enumerate()
                .map(|(word, (a, b))| {
                    let value = (*b - *a).mul_add(alpha, *a);
                    value.is_finite().then_some(value).ok_or_else(|| {
                        format!("blended field {name} word {word} became non-finite")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(ExactProductionField {
            name: left.name.clone(),
            role: left.role.clone(),
            units: left.units.clone(),
            stagger: left.stagger.clone(),
            shape: left.shape.clone(),
            values_sha256: hash_f32(&values),
            values: Arc::new(values),
        })
    }
}

#[derive(Clone, Debug)]
pub struct BlendedProductionFrame {
    pub source_steps: [u64; 2],
    pub source_manifest_sha256: [String; 2],
    pub alpha_f32_bits: String,
    pub model_time_s: f64,
    pub formula: &'static str,
    pub fields: Vec<ExactProductionField>,
}

pub fn is_production_pack_directory(directory: &Path) -> bool {
    directory.join(MANIFEST_FILE).is_file()
        && directory.join("u3d.f32le").is_file()
        && directory.join("total-geopotential-zf.f32le").is_file()
}

pub fn load_production_pack(directory: &Path) -> Result<ExactProductionFrame, String> {
    verify_inventory(directory)?;
    let manifest_bytes = read_bounded(&directory.join(MANIFEST_FILE), MAX_JSON_BYTES, "manifest")?;
    let receipt_bytes = read_bounded(&directory.join(RECEIPT_FILE), MAX_JSON_BYTES, "receipt")?;
    let manifest: PackManifest = parse_canonical(&manifest_bytes, "production manifest")?;
    let receipt: PackReceipt = parse_canonical(&receipt_bytes, "production receipt")?;
    let source_profile = validate_envelopes(directory, &manifest_bytes, &manifest, &receipt)?;

    let mut fields = Vec::with_capacity(CHANNELS.len());
    for (channel, contract) in manifest.channels.iter().zip(CHANNELS) {
        let payload = read_bounded(
            &directory.join(&channel.file),
            channel.nbytes as u64,
            &format!("channel {}", channel.name),
        )?;
        if payload.len() != channel.nbytes || sha256_bytes(&payload) != channel.payload_sha256 {
            return Err(format!("channel {} payload identity changed", channel.name));
        }
        let (values, statistics) = decode_and_measure(&payload)?;
        if statistics != channel.statistics {
            return Err(format!(
                "channel {} payload statistics changed",
                channel.name
            ));
        }
        if let Some(expected) = contract.nonnegative
            && statistics.nonnegative != expected
        {
            return Err(format!("channel {} sign disposition changed", channel.name));
        }
        if contract.formula == RAW_COPY_FORMULA
            && channel.source_sha256.as_slice() != [channel.payload_sha256.as_str()]
        {
            return Err(format!(
                "raw channel {} is not source-word identical",
                channel.name
            ));
        }
        fields.push(ExactProductionField {
            name: channel.name.clone(),
            role: channel.role.clone(),
            units: channel.units.clone(),
            stagger: channel.stagger.clone(),
            shape: channel.shape.clone(),
            values: Arc::new(values),
            values_sha256: channel.payload_sha256.clone(),
        });
    }
    let model_time_s = f64::from_bits(parse_hex_u64(
        &manifest.source.model_time_f64_bits,
        "model time",
    )?);
    Ok(ExactProductionFrame {
        source_path: directory.to_path_buf(),
        source_profile,
        step: manifest.source.step,
        model_time_s,
        manifest_sha256: sha256_bytes(&manifest_bytes),
        payload_set_sha256: manifest.payload_set_sha256,
        history_contract_sha256: manifest.source.history.contract_sha256,
        operator_manifest_sha256: manifest.source.operator_manifest_sha256,
        fields,
        producer: manifest.producer,
    })
}

pub fn load_production_frame_pair(
    a: &Path,
    b: &Path,
    generation: u64,
) -> Result<ProductionFramePair, String> {
    ProductionFramePair::new(
        load_production_pack(a)?,
        load_production_pack(b)?,
        generation,
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionSeriesRefusal {
    pub step: Option<u64>,
    pub path: PathBuf,
    pub reason: String,
}

#[derive(Clone, Debug, Default)]
pub struct ProductionSeriesScan {
    pub request: Option<ProductionLoadRequest>,
    /// Newest visible exact pack. A lone frame is a valid renderer startup,
    /// but it is never enough to enable temporal interpolation.
    pub latest_frame: Option<PathBuf>,
    pub visible_steps: Vec<u64>,
    pub refusals: Vec<ProductionSeriesRefusal>,
}

/// Discover the newest real, consecutive pair without reading or deriving any
/// payload. The returned request must still pass `ProductionPairLoader`, which
/// independently admits every byte of both v2 packs before playback.
pub fn scan_production_series(root: &Path, after_step: Option<u64>) -> ProductionSeriesScan {
    let mut scan = ProductionSeriesScan::default();
    match fs::symlink_metadata(root) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
        Ok(_) => {
            scan.refusals.push(ProductionSeriesRefusal {
                step: None,
                path: root.to_path_buf(),
                reason: "production series root is not a real directory".to_owned(),
            });
            return scan;
        }
        Err(error) => {
            scan.refusals.push(ProductionSeriesRefusal {
                step: None,
                path: root.to_path_buf(),
                reason: format!("production series root is unreadable: {error}"),
            });
            return scan;
        }
    }

    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) => {
            scan.refusals.push(ProductionSeriesRefusal {
                step: None,
                path: root.to_path_buf(),
                reason: format!("production series root could not be listed: {error}"),
            });
            return scan;
        }
    };
    let mut candidates = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                scan.refusals.push(ProductionSeriesRefusal {
                    step: None,
                    path: root.to_path_buf(),
                    reason: format!("production series entry is unreadable: {error}"),
                });
                continue;
            }
        };
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            scan.refusals.push(ProductionSeriesRefusal {
                step: None,
                path: entry.path(),
                reason: "production series entry name is not UTF-8".to_owned(),
            });
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let Some(step) = parse_step_directory(&name) else {
            if name.starts_with("step-") {
                scan.refusals.push(ProductionSeriesRefusal {
                    step: None,
                    path: entry.path(),
                    reason: "production series step name is not exact step-NNNNNNNN".to_owned(),
                });
            }
            continue;
        };
        match entry.file_type() {
            Ok(kind) if kind.is_dir() && !kind.is_symlink() => {
                candidates.push((step, entry.path()));
            }
            Ok(_) => scan.refusals.push(ProductionSeriesRefusal {
                step: Some(step),
                path: entry.path(),
                reason: "production series step is not a real directory".to_owned(),
            }),
            Err(error) => scan.refusals.push(ProductionSeriesRefusal {
                step: Some(step),
                path: entry.path(),
                reason: format!("production series step type is unreadable: {error}"),
            }),
        }
    }
    candidates.sort_by_key(|(step, _)| *step);
    scan.visible_steps = candidates.iter().map(|(step, _)| *step).collect();
    if scan.visible_steps.windows(2).any(|pair| pair[0] == pair[1]) {
        scan.refusals.push(ProductionSeriesRefusal {
            step: None,
            path: root.to_path_buf(),
            reason: "production series contains a duplicate numeric step".to_owned(),
        });
    }
    if let Some(pair) = scan
        .visible_steps
        .windows(2)
        .find(|pair| pair[1] != pair[0] + 1)
    {
        scan.refusals.push(ProductionSeriesRefusal {
            step: Some(pair[1]),
            path: root.to_path_buf(),
            reason: format!(
                "production series contains a step gap: {} -> {}",
                pair[0], pair[1]
            ),
        });
    }
    let maximum = scan.visible_steps.last().copied();
    if let Some(cursor) = after_step
        && maximum.is_none_or(|step| step < cursor)
    {
        scan.refusals.push(ProductionSeriesRefusal {
            step: maximum,
            path: root.to_path_buf(),
            reason: format!(
                "production series rolled back below admitted step {cursor}; maximum visible is {maximum:?}"
            ),
        });
    }
    if let (Some(cursor), Some(maximum)) = (after_step, maximum)
        && maximum > cursor
    {
        let first_new = scan
            .visible_steps
            .iter()
            .copied()
            .find(|step| *step > cursor);
        if first_new != Some(cursor + 1) {
            scan.refusals.push(ProductionSeriesRefusal {
                step: first_new,
                path: root.to_path_buf(),
                reason: format!(
                    "production series has a gap after admitted step {cursor}; first newer step is {first_new:?}"
                ),
            });
        }
    }
    if !scan.refusals.is_empty() || maximum.is_none_or(|step| after_step.is_some_and(|v| step <= v))
    {
        return scan;
    }
    scan.latest_frame = candidates.last().map(|(_, path)| path.clone());
    if candidates.len() == 1 && after_step.is_none() {
        return scan;
    }
    let [.., (step_a, frame_a), (step_b, frame_b)] = candidates.as_slice() else {
        scan.refusals.push(ProductionSeriesRefusal {
            step: maximum,
            path: root.to_path_buf(),
            reason:
                "production series advance lacks two exact consecutive endpoints for A/B playback"
                    .to_owned(),
        });
        return scan;
    };
    if *step_b != *step_a + 1 {
        scan.refusals.push(ProductionSeriesRefusal {
            step: Some(*step_b),
            path: root.to_path_buf(),
            reason: format!(
                "production series newest frames are not consecutive: {step_a} -> {step_b}"
            ),
        });
        return scan;
    }
    scan.request = Some(ProductionLoadRequest {
        generation: *step_b,
        frame_a: frame_a.clone(),
        frame_b: frame_b.clone(),
    });
    scan
}

fn parse_step_directory(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("step-")?;
    if digits.len() != 8 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let step = digits.parse::<u64>().ok()?;
    (step > 0).then_some(step)
}

fn validate_envelopes(
    directory: &Path,
    manifest_bytes: &[u8],
    manifest: &PackManifest,
    receipt: &PackReceipt,
) -> Result<Gm1ViewSourceProfile, String> {
    if manifest.schema != CHILD_PRODUCTION_VIEW_PACK_MANIFEST_SCHEMA
        || receipt.schema != CHILD_PRODUCTION_VIEW_PACK_RECEIPT_SCHEMA
        || manifest.abi != CHILD_PRODUCTION_VIEW_PACK_ABI
        || receipt.abi != CHILD_PRODUCTION_VIEW_PACK_ABI
        || manifest.status != STATUS
        || receipt.status != STATUS
        || manifest.claim_limit != CLAIM_LIMIT
        || receipt.claim_limit != CLAIM_LIMIT
        || manifest.model != "GM1"
        || receipt.model != "GM1"
        || manifest.synthetic_detail
        || receipt.synthetic_detail
    {
        return Err("production pack schema/ABI/status/scientific disposition changed".to_owned());
    }
    if manifest.producer
        != (ProducerIdentity {
            commit: ADMITTED_PRODUCTION_VIEW_PRODUCER_COMMIT.to_owned(),
            executable_sha256: ADMITTED_PRODUCTION_VIEW_EXECUTABLE_SHA256.to_owned(),
            source_sha256: ADMITTED_PRODUCTION_VIEW_SOURCE_SHA256.to_owned(),
        })
    {
        return Err("production pack producer component identity is not admitted".to_owned());
    }
    if receipt.producer != manifest.producer
        || receipt.source != manifest.source
        || receipt.abi != manifest.abi
        || receipt.claim_limit != manifest.claim_limit
        || receipt.model != manifest.model
        || receipt.status != manifest.status
        || receipt.payload_set_sha256 != manifest.payload_set_sha256
        || receipt.manifest_sha256 != sha256_bytes(manifest_bytes)
    {
        return Err("production receipt does not bind the exact manifest/source".to_owned());
    }
    let source_profile = validate_source(&manifest.source)?;
    validate_channels(&manifest.channels, &receipt.channels)?;
    if manifest.qc_plus_qi_v1_payload_sha256 != manifest.channels[0].payload_sha256 {
        return Err("v2 cloud channel is not explicitly v1-byte compatible".to_owned());
    }
    if manifest.channels[0].source_sha256
        != [
            manifest.source.q3d.qc_sha256.clone(),
            manifest.source.q3d.qi_sha256.clone(),
        ]
    {
        return Err("v2 cloud channel is not bound to the exact qc/qi source words".to_owned());
    }
    let payload_set = serde_json::to_vec(&receipt.channels)
        .map_err(|error| format!("could not canonicalize payload inventory: {error}"))?;
    if sha256_bytes(&payload_set) != manifest.payload_set_sha256 {
        return Err("production payload-set SHA-256 changed".to_owned());
    }
    if manifest.optional_channels.reflectivity_dbz.status != "MISSING_SOURCE_CARRIER"
        || manifest.optional_channels.reflectivity_dbz.reason != REFLECTIVITY_REASON
        || manifest.optional_channels.updraft_mass_center_ms.status != "DERIVABLE_FROM_EXACT_W3D"
        || manifest.optional_channels.updraft_mass_center_ms.reason != UPDRAFT_REASON
        || manifest.optional_channels.vertical_vorticity_s.status != "DERIVABLE_FROM_EXACT_U3D_V3D"
        || manifest.optional_channels.vertical_vorticity_s.reason != VORTICITY_REASON
    {
        return Err("production optional-channel fail-closed disposition changed".to_owned());
    }
    let step_name = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "production pack directory lacks a UTF-8 name".to_owned())?;
    if step_name != format!("step-{:08}", manifest.source.step) {
        return Err("production pack directory and source step disagree".to_owned());
    }
    Ok(source_profile)
}

fn validate_source(source: &SourceIdentity) -> Result<Gm1ViewSourceProfile, String> {
    for (value, label) in [
        (&source.history.artifact_sha256, "history artifact"),
        (&source.history.contract_sha256, "history contract"),
        (&source.history.payload_sha256, "history payload"),
        (&source.history.raw_arena_sha256, "history raw arena"),
        (&source.operator_manifest_sha256, "operator manifest"),
        (&source.pair_manifest_sha256, "pair manifest"),
        (&source.pair_receipt_sha256, "pair receipt"),
        (&source.q3d.sha256, "q3d"),
        (&source.q3d.qc_sha256, "qc"),
        (&source.q3d.qi_sha256, "qi"),
    ] {
        validate_sha256(value, label)?;
    }
    let source_profile = gm1_view_source_profile_for_authority(
        &source.history.contract_sha256,
        &source.operator_manifest_sha256,
    )?;
    if source.step == 0
        || source.q3d.nbytes != 162_000_000
        || source.q3d.shape_species_kji != [6, 75, 300, 300]
        || source.q3d.species_order != ["qv", "qc", "qr", "qi", "qs", "qg"]
    {
        return Err("production source q3d/step geometry changed".to_owned());
    }
    let model_time = f64::from_bits(parse_hex_u64(&source.model_time_f64_bits, "model time")?);
    if !model_time.is_finite() || model_time.to_bits() != (source.step as f64 * 0.5).to_bits() {
        return Err("production source clock differs from exact step*0.5 cadence".to_owned());
    }
    Ok(source_profile)
}

pub fn gm1_view_source_profile_for_authority(
    history_contract_sha256: &str,
    operator_manifest_sha256: &str,
) -> Result<Gm1ViewSourceProfile, String> {
    validate_sha256(history_contract_sha256, "history contract")?;
    validate_sha256(operator_manifest_sha256, "operator manifest")?;
    if let Some(authority) = ADMITTED_GM1_VIEW_SOURCE_AUTHORITIES
        .iter()
        .find(|authority| {
            authority.history_contract_sha256 == history_contract_sha256
                && authority.operator_manifest_sha256 == operator_manifest_sha256
        })
    {
        return Ok(authority.profile);
    }
    Err(format!(
        "production pack history/operator science envelope is not admitted; {} remains {}",
        IDEALIZED_VIEW_SOURCE_PROFILE_ID, IDEALIZED_VIEW_SOURCE_PROFILE_STATUS,
    ))
}

fn validate_channels(
    manifest: &[ChannelManifest],
    receipt: &[ChannelReceipt],
) -> Result<(), String> {
    if manifest.len() != CHANNELS.len() || receipt.len() != CHANNELS.len() {
        return Err("production pack channel count changed".to_owned());
    }
    for ((channel, sealed), expected) in manifest.iter().zip(receipt).zip(CHANNELS) {
        let expected_words = expected.shape.iter().product::<usize>();
        let expected_receipt = ChannelReceipt {
            file: channel.file.clone(),
            name: channel.name.clone(),
            nbytes: channel.nbytes,
            payload_sha256: channel.payload_sha256.clone(),
            shape: channel.shape.clone(),
        };
        if channel.name != expected.name
            || channel.file != expected.file
            || channel.shape != expected.shape
            || channel.source_carriers
                != expected
                    .source_carriers
                    .iter()
                    .map(|value| (*value).to_owned())
                    .collect::<Vec<_>>()
            || channel.role != expected.role
            || channel.units != expected.units
            || channel.stagger != expected.stagger
            || channel.formula != expected.formula
            || channel.dtype != "little-endian-f32"
            || channel.order != "C"
            || channel.nbytes != expected_words * 4
            || channel.statistics.word_count != expected_words
            || !channel.statistics.all_finite
            || channel.source_sha256.len() != expected.source_carriers.len()
            || *sealed != expected_receipt
        {
            return Err(format!(
                "production channel {} contract changed",
                expected.name
            ));
        }
        validate_sha256(&channel.payload_sha256, "channel payload")?;
        for value in &channel.source_sha256 {
            validate_sha256(value, "channel source")?;
        }
    }
    Ok(())
}

fn validate_pair(a: &ExactProductionFrame, b: &ExactProductionFrame) -> Result<(), String> {
    if b.step != a.step + 1
        || b.model_time_s.to_bits() != (a.model_time_s + 0.5).to_bits()
        || a.source_profile != b.source_profile
        || a.history_contract_sha256 != b.history_contract_sha256
        || a.operator_manifest_sha256 != b.operator_manifest_sha256
        || a.producer != b.producer
        || a.fields.len() != b.fields.len()
    {
        return Err("production A/B frames are not one compatible monotonic cadence".to_owned());
    }
    for (left, right) in a.fields.iter().zip(&b.fields) {
        if left.name != right.name
            || left.role != right.role
            || left.units != right.units
            || left.stagger != right.stagger
            || left.shape != right.shape
            || left.values.len() != right.values.len()
        {
            return Err(format!(
                "production A/B field {} contract changed",
                left.name
            ));
        }
        if matches!(
            left.name.as_str(),
            "terrain_msl_m" | "static_zf_m" | "total_geopotential_zf_m"
        ) && left.values_sha256 != right.values_sha256
        {
            return Err(format!(
                "production A/B static field {} changed bytes",
                left.name
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ProductionLoadRequest {
    pub generation: u64,
    pub frame_a: PathBuf,
    pub frame_b: PathBuf,
}

#[derive(Debug)]
pub enum ProductionLoadEvent {
    Ready(ProductionFramePair),
    Refused { generation: u64, reason: String },
}

impl ProductionLoadEvent {
    pub fn generation(&self) -> u64 {
        match self {
            Self::Ready(pair) => pair.generation,
            Self::Refused { generation, .. } => *generation,
        }
    }
}

enum LoaderCommand {
    Load(ProductionLoadRequest),
    Shutdown,
}

pub struct ProductionPairLoader {
    sender: Sender<LoaderCommand>,
    receiver: Receiver<ProductionLoadEvent>,
    latest_generation: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
}

impl ProductionPairLoader {
    pub fn new() -> Self {
        Self::spawn_with(load_production_frame_pair)
    }

    fn spawn_with<F>(loader: F) -> Self
    where
        F: Fn(&Path, &Path, u64) -> Result<ProductionFramePair, String> + Send + 'static,
    {
        let (sender, commands) = mpsc::channel();
        let (events, receiver) = mpsc::channel();
        let latest_generation = Arc::new(AtomicU64::new(0));
        let worker_latest_generation = latest_generation.clone();
        let worker = thread::Builder::new()
            .name("gm1-production-pair-loader".to_owned())
            .spawn(move || loader_worker(commands, events, worker_latest_generation, loader))
            .expect("production pair loader thread must start");
        Self {
            sender,
            receiver,
            latest_generation,
            worker: Some(worker),
        }
    }

    pub fn request(&self, request: ProductionLoadRequest) -> Result<(), String> {
        let previous = self
            .latest_generation
            .fetch_max(request.generation, Ordering::SeqCst);
        if request.generation <= previous {
            return Err(format!(
                "production pair generation {} is not newer than {previous}",
                request.generation
            ));
        }
        self.sender
            .send(LoaderCommand::Load(request))
            .map_err(|_| "production pair loader stopped".to_owned())
    }

    pub fn try_receive(&self) -> Option<ProductionLoadEvent> {
        while let Ok(event) = self.receiver.try_recv() {
            if self.is_current_generation(event.generation()) {
                return Some(event);
            }
        }
        None
    }

    pub fn receive_timeout(
        &self,
        timeout: Duration,
    ) -> Result<ProductionLoadEvent, RecvTimeoutError> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let event = self.receiver.recv_timeout(remaining)?;
            if self.is_current_generation(event.generation()) {
                return Ok(event);
            }
        }
    }

    pub fn is_current_generation(&self, generation: u64) -> bool {
        generation == self.latest_generation.load(Ordering::SeqCst)
    }
}

impl Default for ProductionPairLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ProductionPairLoader {
    fn drop(&mut self) {
        let _ = self.sender.send(LoaderCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Source-word-resident temporal state used by the normal application path.
/// A lone admitted frame is useful and visible, but cannot interpolate. A
/// refused update is validated before mutation, so `current` is always the
/// last good exact endpoint.
#[derive(Clone, Debug)]
pub struct ProductionSeriesPlayback {
    current: Arc<ExactProductionFrame>,
    pair: Option<ProductionFramePair>,
    alpha: f32,
}

impl ProductionSeriesPlayback {
    pub fn from_single(frame: ExactProductionFrame) -> Self {
        Self {
            current: Arc::new(frame),
            pair: None,
            alpha: 1.0,
        }
    }

    pub fn from_pair(pair: ProductionFramePair) -> Self {
        Self {
            current: pair.b.clone(),
            pair: Some(pair),
            alpha: 1.0,
        }
    }

    pub fn current(&self) -> &Arc<ExactProductionFrame> {
        &self.current
    }

    pub fn pair(&self) -> Option<&ProductionFramePair> {
        self.pair.as_ref()
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn interpolation_dormant(&self) -> bool {
        self.pair.is_none()
    }

    pub fn set_alpha(&mut self, alpha: f32) -> Result<(), String> {
        if self.pair.is_none() {
            return Err(
                "production interpolation is dormant until two exact frames are admitted"
                    .to_owned(),
            );
        }
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err("production playback alpha must be finite in [0,1]".to_owned());
        }
        self.alpha = alpha;
        Ok(())
    }

    pub fn accept_pair(&mut self, pair: ProductionFramePair) -> Result<(), String> {
        validate_series_transition(&self.current, &pair)?;
        self.current = pair.b.clone();
        self.pair = Some(pair);
        // Every newly admitted generation first displays its exact newest
        // endpoint. Temporal inspection is an explicit user action.
        self.alpha = 1.0;
        Ok(())
    }
}

fn validate_series_transition(
    current: &ExactProductionFrame,
    pair: &ProductionFramePair,
) -> Result<(), String> {
    if pair.b.step <= current.step
        || pair.a.source_profile != current.source_profile
        || pair.a.history_contract_sha256 != current.history_contract_sha256
        || pair.a.operator_manifest_sha256 != current.operator_manifest_sha256
        || pair.a.producer != current.producer
    {
        return Err(
            "production series update is rollback, cross-profile, or cross-authority; last good retained"
                .to_owned(),
        );
    }
    if pair.a.step == current.step
        && (pair.a.manifest_sha256 != current.manifest_sha256
            || pair.a.payload_set_sha256 != current.payload_set_sha256)
    {
        return Err(
            "production series rewrote the current exact endpoint; last good retained".to_owned(),
        );
    }
    for name in ["terrain_msl_m", "static_zf_m", "total_geopotential_zf_m"] {
        let current_hash = current
            .field(name)
            .map(|field| field.values_sha256.as_str());
        let next_hash = pair.a.field(name).map(|field| field.values_sha256.as_str());
        if current_hash != next_hash {
            return Err(format!(
                "production series static field {name} changed authority; last good retained"
            ));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum ProductionSeriesWatchEvent {
    Accepted(ProductionFramePair),
    Refused(ProductionSeriesRefusal),
}

pub struct ProductionSeriesStartup {
    pub root: PathBuf,
    pub playback: ProductionSeriesPlayback,
    pub receiver: Receiver<ProductionSeriesWatchEvent>,
}

/// Strictly admit the newest visible v2 endpoint and start the bounded
/// coalescing watcher. One visible frame is sufficient for startup; A/B state
/// appears only after a second consecutive frame passes the full byte gate.
pub fn load_initial_production_series(root: &Path) -> Result<ProductionSeriesStartup, String> {
    let scan = scan_production_series(root, None);
    if !scan.refusals.is_empty() {
        return Err(format_series_refusals(root, &scan));
    }
    let playback = if let Some(request) = scan.request {
        ProductionSeriesPlayback::from_pair(load_production_frame_pair(
            &request.frame_a,
            &request.frame_b,
            request.generation,
        )?)
    } else {
        let latest = scan.latest_frame.as_deref().ok_or_else(|| {
            format!(
                "production series {} has no visible exact v2 frame",
                root.display()
            )
        })?;
        ProductionSeriesPlayback::from_single(load_production_pack(latest)?)
    };
    let receiver = spawn_production_series_watch(root.to_path_buf(), playback.clone());
    Ok(ProductionSeriesStartup {
        root: root.to_path_buf(),
        playback,
        receiver,
    })
}

fn format_series_refusals(root: &Path, scan: &ProductionSeriesScan) -> String {
    let reasons = scan
        .refusals
        .iter()
        .map(|refusal| {
            format!(
                "step {:?} at {}: {}",
                refusal.step,
                refusal.path.display(),
                refusal.reason
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "production series {} refused; visible_steps={:?}; {reasons}",
        root.display(),
        scan.visible_steps
    )
}

fn spawn_production_series_watch(
    root: PathBuf,
    mut playback: ProductionSeriesPlayback,
) -> Receiver<ProductionSeriesWatchEvent> {
    const POLL_INTERVAL: Duration = Duration::from_millis(650);
    let (sender, receiver) = mpsc::channel();
    thread::Builder::new()
        .name("gm1-production-series-watch".to_owned())
        .spawn(move || {
            let loader = ProductionPairLoader::new();
            let mut cursor = playback.current().step;
            let mut requested = cursor;
            let mut last_refusal = None::<String>;
            loop {
                let scan = scan_production_series(&root, Some(cursor));
                if let Some(refusal) = scan.refusals.into_iter().last() {
                    let fingerprint = format!(
                        "{:?}|{}|{}",
                        refusal.step,
                        refusal.path.display(),
                        refusal.reason
                    );
                    if last_refusal.as_deref() != Some(&fingerprint) {
                        last_refusal = Some(fingerprint);
                        if sender
                            .send(ProductionSeriesWatchEvent::Refused(refusal))
                            .is_err()
                        {
                            return;
                        }
                    }
                } else if let Some(request) = scan.request
                    && request.generation > requested
                {
                    requested = request.generation;
                    if let Err(reason) = loader.request(request) {
                        let refusal = ProductionSeriesRefusal {
                            step: Some(requested),
                            path: root.clone(),
                            reason,
                        };
                        if sender
                            .send(ProductionSeriesWatchEvent::Refused(refusal))
                            .is_err()
                        {
                            return;
                        }
                    }
                }
                if let Some(event) = loader.try_receive() {
                    match event {
                        ProductionLoadEvent::Ready(pair) => {
                            let generation = pair.generation;
                            match playback.accept_pair(pair) {
                                Ok(()) => {
                                    cursor = playback.current().step;
                                    last_refusal = None;
                                    let accepted = playback
                                        .pair()
                                        .expect("accepted playback retains its pair")
                                        .clone();
                                    if sender
                                        .send(ProductionSeriesWatchEvent::Accepted(accepted))
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                                Err(reason) => {
                                    let refusal = ProductionSeriesRefusal {
                                        step: Some(generation),
                                        path: root.clone(),
                                        reason,
                                    };
                                    if sender
                                        .send(ProductionSeriesWatchEvent::Refused(refusal))
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                            }
                        }
                        ProductionLoadEvent::Refused { generation, reason } => {
                            let refusal = ProductionSeriesRefusal {
                                step: Some(generation),
                                path: root.clone(),
                                reason: format!(
                                    "strict production A/B payload admission refused: {reason}"
                                ),
                            };
                            if sender
                                .send(ProductionSeriesWatchEvent::Refused(refusal))
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                }
                thread::sleep(POLL_INTERVAL);
            }
        })
        .expect("GM1 production series watch thread must start");
    receiver
}

fn loader_worker<F>(
    commands: Receiver<LoaderCommand>,
    events: Sender<ProductionLoadEvent>,
    latest_generation: Arc<AtomicU64>,
    loader: F,
) where
    F: Fn(&Path, &Path, u64) -> Result<ProductionFramePair, String>,
{
    let mut pending = None;
    loop {
        let mut request = match pending.take() {
            Some(request) => request,
            None => match commands.recv() {
                Ok(LoaderCommand::Load(request)) => request,
                Ok(LoaderCommand::Shutdown) | Err(_) => return,
            },
        };
        loop {
            match commands.try_recv() {
                Ok(LoaderCommand::Load(newer)) => request = newer,
                Ok(LoaderCommand::Shutdown) => return,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        let event = match loader(&request.frame_a, &request.frame_b, request.generation) {
            Ok(pair) => ProductionLoadEvent::Ready(pair),
            Err(reason) => ProductionLoadEvent::Refused {
                generation: request.generation,
                reason,
            },
        };
        loop {
            match commands.try_recv() {
                Ok(LoaderCommand::Load(newer)) => pending = Some(newer),
                Ok(LoaderCommand::Shutdown) => return,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        if pending.is_none()
            && event.generation() == latest_generation.load(Ordering::SeqCst)
            && events.send(event).is_err()
        {
            return;
        }
    }
}

fn read_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("could not inspect {label}: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum {
        return Err(format!("{label} is not a bounded regular file"));
    }
    let bytes = fs::read(path).map_err(|error| format!("could not read {label}: {error}"))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{label} size changed while reading"));
    }
    Ok(bytes)
}

fn parse_canonical<T: DeserializeOwned>(bytes: &[u8], label: &str) -> Result<T, String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("{label}: {error}"))?;
    let canonical = serde_json::to_vec(&value)
        .map_err(|error| format!("could not canonicalize {label}: {error}"))?;
    if canonical != bytes {
        return Err(format!("{label} is not canonical compact JSON"));
    }
    serde_json::from_value(value).map_err(|error| format!("{label}: {error}"))
}

fn verify_inventory(directory: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(directory)
        .map_err(|error| format!("could not inspect production directory: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("production pack is not a real directory".to_owned());
    }
    let mut expected = CHANNELS
        .iter()
        .map(|channel| channel.file.to_owned())
        .collect::<BTreeSet<_>>();
    expected.insert(MANIFEST_FILE.to_owned());
    expected.insert(RECEIPT_FILE.to_owned());
    let mut observed = BTreeSet::new();
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let metadata = fs::symlink_metadata(entry.path()).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err("production pack contains a symlink or non-file".to_owned());
        }
        observed.insert(
            entry
                .file_name()
                .into_string()
                .map_err(|_| "production pack contains a non-UTF-8 name".to_owned())?,
        );
    }
    if observed != expected {
        return Err(format!(
            "production pack inventory is {observed:?}; expected {expected:?}"
        ));
    }
    Ok(())
}

fn decode_and_measure(payload: &[u8]) -> Result<(Vec<f32>, ChannelStatistics), String> {
    let chunks = payload.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err("production payload is not whole f32 words".to_owned());
    }
    let mut values = Vec::with_capacity(payload.len() / 4);
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut nonzero_count = 0usize;
    let mut nonnegative = true;
    for (index, bytes) in chunks.enumerate() {
        let value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
        if !value.is_finite() {
            return Err(format!("production payload word {index} is non-finite"));
        }
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        nonzero_count += usize::from(value.to_bits() & 0x7fff_ffff != 0);
        nonnegative &= value >= 0.0;
        values.push(value);
    }
    Ok((
        values,
        ChannelStatistics {
            all_finite: true,
            maximum_f32_bits: format!("{:08x}", maximum.to_bits()),
            minimum_f32_bits: format!("{:08x}", minimum.to_bits()),
            nonnegative,
            nonzero_count,
            word_count: payload.len() / 4,
        },
    ))
}

fn parse_hex_u64(value: &str, label: &str) -> Result<u64, String> {
    if value.len() != 16 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} is not exact 16-hex f64 bits"));
    }
    u64::from_str_radix(value, 16).map_err(|error| format!("{label}: {error}"))
}

fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{label} is not exact lowercase SHA-256"));
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc, Mutex},
        time::{Instant, SystemTime, UNIX_EPOCH},
    };

    fn scratch_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gm1-production-playback-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        root
    }

    fn visible_step(root: &Path, step: u64) -> PathBuf {
        let path = root.join(format!("step-{step:08}"));
        fs::create_dir(&path).unwrap();
        path
    }

    fn identity() -> ProducerIdentity {
        ProducerIdentity {
            commit: ADMITTED_PRODUCTION_VIEW_PRODUCER_COMMIT.to_owned(),
            executable_sha256: ADMITTED_PRODUCTION_VIEW_EXECUTABLE_SHA256.to_owned(),
            source_sha256: ADMITTED_PRODUCTION_VIEW_SOURCE_SHA256.to_owned(),
        }
    }

    fn frame(step: u64, value: f32) -> ExactProductionFrame {
        let dynamic = vec![value, value + 1.0];
        let static_values = vec![12.0, 13.0];
        ExactProductionFrame {
            source_path: PathBuf::from(format!("step-{step:08}")),
            source_profile: REAL_CHILD_VIEW_SOURCE_PROFILE,
            step,
            model_time_s: step as f64 * 0.5,
            manifest_sha256: format!("{step:064x}"),
            payload_set_sha256: format!("{:064x}", step + 100),
            history_contract_sha256: LIVE_HISTORY_CONTRACT.to_owned(),
            operator_manifest_sha256: LIVE_OPERATOR_MANIFEST.to_owned(),
            fields: vec![
                ExactProductionField {
                    name: "u_staggered_ms".to_owned(),
                    role: "wind".to_owned(),
                    units: "m s-1".to_owned(),
                    stagger: "x-face C-grid".to_owned(),
                    shape: vec![2],
                    values_sha256: hash_f32(&dynamic),
                    values: Arc::new(dynamic),
                },
                ExactProductionField {
                    name: "terrain_msl_m".to_owned(),
                    role: "terrain".to_owned(),
                    units: "m".to_owned(),
                    stagger: "horizontal mass".to_owned(),
                    shape: vec![2],
                    values_sha256: hash_f32(&static_values),
                    values: Arc::new(static_values),
                },
            ],
            producer: identity(),
        }
    }

    #[test]
    fn exact_f32_blend_preserves_endpoints_and_pins_fma_interior() {
        let pair = ProductionFramePair::new(frame(1, 2.0), frame(2, 6.0), 9).unwrap();
        let a = pair.blend_exact_f32(0.0).unwrap();
        let b = pair.blend_exact_f32(1.0).unwrap();
        let middle = pair.blend_exact_f32(0.25).unwrap();
        assert_eq!(a.fields[0].values.as_slice(), [2.0, 3.0]);
        assert_eq!(b.fields[0].values.as_slice(), [6.0, 7.0]);
        assert_eq!(middle.fields[0].values.as_slice(), [3.0, 4.0]);
        assert_eq!(middle.fields[1].values.as_slice(), [12.0, 13.0]);
        assert_eq!(middle.alpha_f32_bits, "3e800000");
        assert_eq!(middle.formula, EXACT_F32_BLEND_FORMULA);
        let one_field = pair.blend_field_exact_f32("u_staggered_ms", 0.25).unwrap();
        assert_eq!(one_field.values.as_slice(), [3.0, 4.0]);
        assert_eq!(one_field.values_sha256, hash_f32(&[3.0, 4.0]));
        assert!(pair.blend_field_exact_f32("reflectivity_dbz", 0.5).is_err());
        assert!(pair.blend_exact_f32(f32::NAN).is_err());
        assert!(pair.blend_exact_f32(1.01).is_err());
    }

    #[test]
    fn series_playback_keeps_single_frame_dormant_and_refusals_nonmutating() {
        let mut playback = ProductionSeriesPlayback::from_single(frame(1, 2.0));
        assert_eq!(playback.current().step, 1);
        assert!(playback.interpolation_dormant());
        assert!(playback.set_alpha(0.5).unwrap_err().contains("dormant"));

        let pair = ProductionFramePair::new(frame(1, 2.0), frame(2, 6.0), 2).unwrap();
        playback.accept_pair(pair).unwrap();
        assert_eq!(playback.current().step, 2);
        assert!(!playback.interpolation_dormant());
        playback.set_alpha(0.25).unwrap();
        assert_eq!(playback.alpha().to_bits(), 0.25f32.to_bits());

        let retained_manifest = playback.current().manifest_sha256.clone();
        let mut foreign_a = frame(3, 8.0);
        let mut foreign_b = frame(4, 9.0);
        foreign_a.source_profile = IDEALIZED_VIEW_SOURCE_PROFILE;
        foreign_b.source_profile = IDEALIZED_VIEW_SOURCE_PROFILE;
        let foreign = ProductionFramePair::new(foreign_a, foreign_b, 4).unwrap();
        assert!(
            playback
                .accept_pair(foreign)
                .unwrap_err()
                .contains("last good")
        );
        assert_eq!(playback.current().step, 2);
        assert_eq!(playback.current().manifest_sha256, retained_manifest);
        assert_eq!(playback.alpha().to_bits(), 0.25f32.to_bits());
    }

    #[test]
    fn pair_refuses_nonconsecutive_and_changed_static_bytes() {
        assert!(ProductionFramePair::new(frame(1, 0.0), frame(3, 1.0), 1).is_err());
        let mut changed = frame(2, 1.0);
        Arc::make_mut(&mut changed.fields[1].values)[0] = 999.0;
        changed.fields[1].values_sha256 = hash_f32(&changed.fields[1].values);
        assert!(ProductionFramePair::new(frame(1, 0.0), changed, 1).is_err());

        let mut crossed_profile = frame(2, 1.0);
        crossed_profile.source_profile = IDEALIZED_VIEW_SOURCE_PROFILE;
        assert!(ProductionFramePair::new(frame(1, 0.0), crossed_profile, 1).is_err());
    }

    #[test]
    fn source_profiles_separate_science_authority_from_render_geometry() {
        assert_eq!(ADMITTED_GM1_VIEW_SOURCE_AUTHORITIES.len(), 3);
        assert!(
            ADMITTED_GM1_VIEW_SOURCE_AUTHORITIES
                .iter()
                .all(|authority| {
                    authority.profile == REAL_CHILD_VIEW_SOURCE_PROFILE
                        && authority.profile != IDEALIZED_VIEW_SOURCE_PROFILE
                })
        );
        assert_eq!(
            gm1_view_source_profile_for_authority(LIVE_HISTORY_CONTRACT, LIVE_OPERATOR_MANIFEST)
                .unwrap(),
            REAL_CHILD_VIEW_SOURCE_PROFILE,
        );
        assert_eq!(
            REAL_CHILD_VIEW_SOURCE_PROFILE
                .georeferenced_grid()
                .vertical_coordinate,
            "source-stretched; exact static and terrain-following zf faces",
        );
        assert!(
            gm1_view_source_profile_for_authority(
                LIVE_HISTORY_CONTRACT,
                RESTORED_OPERATOR_MANIFEST,
            )
            .is_err()
        );
        assert_eq!(
            IDEALIZED_VIEW_SOURCE_PROFILE.vertical_grid,
            Gm1ViewVerticalGridClass::Uniform { dz_m: 200 },
        );
        assert_eq!(
            IDEALIZED_VIEW_SOURCE_PROFILE.terrain,
            Gm1ViewTerrainClass::FlatZeroTerrain,
        );
        assert_eq!(IDEALIZED_VIEW_SOURCE_AUTHORITY_SHA256.len(), 64);
        assert_eq!(
            IDEALIZED_VIEW_SOURCE_PROFILE
                .georeferenced_grid()
                .horizontal_crs,
            "idealized Cartesian x/y; no Earth geolocation",
        );
        let refusal =
            gm1_view_source_profile_for_authority(&"d".repeat(64), &"e".repeat(64)).unwrap_err();
        assert!(refusal.contains(IDEALIZED_VIEW_SOURCE_PROFILE_ID));
        assert!(refusal.contains(IDEALIZED_VIEW_SOURCE_PROFILE_STATUS));

        assert_eq!(
            gm1_view_source_profile_for_dataset_profile(CHILD_POST_STEP_HISTORY_PROFILE),
            Some(REAL_CHILD_VIEW_SOURCE_PROFILE),
        );
        assert_eq!(
            gm1_view_source_profile_for_dataset_profile(IDEALIZED_POST_STEP_HISTORY_PROFILE),
            Some(IDEALIZED_VIEW_SOURCE_PROFILE),
        );
        assert_eq!(
            gm1_view_source_profile_for_dataset_profile(&format!(
                "{};{}",
                CHILD_PRODUCTION_VIEW_PACK_ABI, REAL_CHILD_VIEW_SOURCE_PROFILE_ID,
            )),
            Some(REAL_CHILD_VIEW_SOURCE_PROFILE),
        );
        assert!(
            gm1_view_source_profile_for_dataset_profile("foreign-science-profile/v1").is_none()
        );
        validate_gm1_view_live_profile_transition(
            CHILD_POST_STEP_HISTORY_PROFILE,
            REAL_CHILD_VIEW_SOURCE_PROFILE_ID,
        )
        .unwrap();
        assert!(
            validate_gm1_view_live_profile_transition(
                CHILD_POST_STEP_HISTORY_PROFILE,
                IDEALIZED_POST_STEP_HISTORY_PROFILE,
            )
            .unwrap_err()
            .contains("source profile changed")
        );
        assert!(
            validate_gm1_view_live_profile_transition(
                "foreign-science-profile/v1",
                "different-foreign-profile/v1",
            )
            .is_err()
        );
    }

    #[test]
    fn newest_request_coalescing_discards_completed_superseded_work() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let observed = calls.clone();
        let loader = ProductionPairLoader::spawn_with(move |_a, _b, generation| {
            observed.lock().unwrap().push(generation);
            thread::sleep(Duration::from_millis(25));
            ProductionFramePair::new(
                frame(generation * 2 + 1, generation as f32),
                frame(generation * 2 + 2, generation as f32 + 1.0),
                generation,
            )
        });
        loader
            .request(ProductionLoadRequest {
                generation: 1,
                frame_a: PathBuf::from("a1"),
                frame_b: PathBuf::from("b1"),
            })
            .unwrap();
        let started = Instant::now();
        while calls.lock().unwrap().is_empty() {
            assert!(started.elapsed() < Duration::from_secs(1));
            thread::yield_now();
        }
        for generation in [2, 3] {
            loader
                .request(ProductionLoadRequest {
                    generation,
                    frame_a: PathBuf::from(format!("a{generation}")),
                    frame_b: PathBuf::from(format!("b{generation}")),
                })
                .unwrap();
        }
        assert!(loader.is_current_generation(3));
        assert!(
            loader
                .request(ProductionLoadRequest {
                    generation: 3,
                    frame_a: PathBuf::from("duplicate-a"),
                    frame_b: PathBuf::from("duplicate-b"),
                })
                .is_err()
        );
        assert!(
            loader
                .request(ProductionLoadRequest {
                    generation: 2,
                    frame_a: PathBuf::from("rollback-a"),
                    frame_b: PathBuf::from("rollback-b"),
                })
                .is_err()
        );
        let event = loader.receive_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(event.generation(), 3);
        assert!(loader.try_receive().is_none());
        assert_eq!(*calls.lock().unwrap(), [1, 3]);
    }

    #[test]
    fn channel_contract_retains_raw_stagger_and_no_reflectivity() {
        assert_eq!(CHANNELS.len(), 8);
        assert_eq!(CHANNELS[1].shape, [75, 300, 301]);
        assert_eq!(CHANNELS[2].shape, [75, 301, 300]);
        assert_eq!(CHANNELS[3].shape, [76, 300, 300]);
        assert!(
            !CHANNELS
                .iter()
                .any(|channel| channel.name.contains("reflect"))
        );
        assert!(
            CHANNELS[1..=3]
                .iter()
                .all(|channel| channel.formula == RAW_COPY_FORMULA)
        );
        assert!(
            CHANNELS
                .iter()
                .all(|channel| production_display_contract(channel.name).is_some())
        );
        assert!(production_display_contract("reflectivity_dbz").is_none());
        assert!(production_display_contract("vertical_vorticity_s").is_none());
    }

    #[test]
    fn series_scanner_emits_only_newest_real_consecutive_paths() {
        let root = scratch_root("consecutive");
        let first = visible_step(&root, 1);
        let waiting = scan_production_series(&root, None);
        assert!(waiting.request.is_none());
        assert_eq!(waiting.latest_frame, Some(first.clone()));
        assert_eq!(waiting.visible_steps, [1]);
        assert!(waiting.refusals.is_empty());

        let second = visible_step(&root, 2);
        let ready = scan_production_series(&root, None);
        let request = ready.request.unwrap();
        assert!(ready.refusals.is_empty());
        assert_eq!(request.generation, 2);
        assert_eq!(request.frame_a, first);
        assert_eq!(request.frame_b, second);

        let unchanged = scan_production_series(&root, Some(2));
        assert!(unchanged.request.is_none());
        assert!(unchanged.refusals.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn series_scanner_refuses_gap_rollback_and_malformed_step() {
        let gap_root = scratch_root("gap");
        visible_step(&gap_root, 1);
        visible_step(&gap_root, 3);
        let gap = scan_production_series(&gap_root, None);
        assert!(gap.request.is_none());
        assert!(gap.refusals[0].reason.contains("step gap"));
        fs::remove_dir_all(gap_root).unwrap();

        let rollback_root = scratch_root("rollback");
        visible_step(&rollback_root, 1);
        let rollback = scan_production_series(&rollback_root, Some(2));
        assert!(rollback.request.is_none());
        assert!(rollback.refusals[0].reason.contains("rolled back"));
        fs::remove_dir_all(rollback_root).unwrap();

        let malformed_root = scratch_root("malformed");
        visible_step(&malformed_root, 1);
        fs::create_dir(malformed_root.join("step-latest")).unwrap();
        let malformed = scan_production_series(&malformed_root, None);
        assert!(malformed.request.is_none());
        assert!(
            malformed.refusals[0]
                .reason
                .contains("not exact step-NNNNNNNN")
        );
        fs::remove_dir_all(malformed_root).unwrap();
    }
}
