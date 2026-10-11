//! Durable GM1 forecast checkpoint at the safe post-large-step boundary.
//!
//! The caller transports one quiescent binary32 arena in the Rust-published
//! order. Rust owns the carrier inventory, WSM6 alias map, cadence, source and
//! operator identities, the strict `CM1FCP1` envelope, and exact restore.
//! Mid-stage capture is deliberately not representable by this ABI.

use crate::forecast::{
    read_forecast_checkpoint_embedded, write_forecast_checkpoint_new, ComponentBlob,
    ComponentBlobSpec, ForecastArray, ForecastArraySpec, ForecastClock, ForecastContract,
    ForecastFrame, ForecastReceipt, RestoredForecast, SchedulerScalar, SchedulerScalarSpec,
    SchedulerScalarType, SchedulerScalarValue,
};
use crate::geopotential::WSM6_GEOPOTENTIAL_CARRIER_ABI;
use crate::json::{self, JsonValue};
use crate::product::INITIALIZED_PRODUCT_CARRIER_ABI;
use crate::sha256::{self, Sha256};
use crate::transition_restart::{
    WSM6_TRANSITION_ABI, WSM6_TRANSITION_BRIDGE_ABI, WSM6_TRANSITION_CARRIER_ABI,
    WSM6_TRANSITION_COMPONENT_ABI, WSM6_TRANSITION_IN_MEMORY_RESTART_SCHEMA,
};
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const GM1_CHECKPOINT_MODEL: &str = "GM1";
pub const GM1_CHECKPOINT_PROFILE_ID: &str = "gm1-real-100m-forecast/v1";
pub const GM1_CHECKPOINT_CARRIER_ABI: &str = "gm1.forecast-checkpoint-carriers/v1";
pub const GM1_CHECKPOINT_IDENTITY_SCHEMA: &str = "gm1.forecast-checkpoint-identity/v1";
pub const GM1_SOURCE_IDENTITIES_SCHEMA: &str = "gm1.forecast-source-identities/v1";
pub const GM1_OPERATOR_MANIFEST_SCHEMA: &str = "gm1.operator-source-manifest/v1";
pub const GM1_SCHEDULER_EXTENSION_SCHEMA: &str = "gm1.scheduler-extension/v1";
pub const GM1_CHECKPOINT_LAYOUT_SCHEMA: &str = "gm1.forecast-checkpoint-layout/v1";
pub const GM1_CHECKPOINT_RECEIPT_SCHEMA: &str = "gm1.forecast-checkpoint-receipt/v1";
pub const GM1_RESTORE_MANIFEST_SCHEMA: &str = "gm1.forecast-restore-directory/v1";
pub const GM1_RESTORE_RECEIPT_SCHEMA: &str = "gm1.forecast-restore-receipt/v1";
pub const GM1_CHECKPOINT_CAPTURE_PHASE: &str = "post-large-step";
pub const GM1_CHECKPOINT_RESTORE_PHASE: &str = "before-next-large-step";
pub const GM1_CHECKPOINT_IDENTITY_BLOB: &str = "gm1_identity";
pub const GM1_OPERATOR_MANIFEST_BLOB: &str = "gm1_operator_manifest";
pub const GM1_SCHEDULER_EXTENSION_BLOB: &str = "gm1_scheduler_extension";
pub const GM1_RESTORE_ARENA_FILE: &str = "gm1-carriers.f32le";
pub const GM1_RESTORE_MANIFEST_FILE: &str = "manifest.json";
pub const GM1_RESTORE_RECEIPT_FILE: &str = "receipt.json";
/// The WSM6 (CM1 `ptype=7`) inventory size.  A component's own count is
/// `Gm1CheckpointComponent::array_count`; this constant is the WSM6 answer
/// and is kept because the published v1 profile is sealed to it.
pub const GM1_CHECKPOINT_ARRAY_COUNT: usize = 37;
pub const GM1_CHECKPOINT_REAL_PAYLOAD_NBYTES: u64 = 976_696_992;
pub const GM1_CHECKPOINT_BLOB_MAX_NBYTES: usize = 64 * 1024;
pub const GM1_CHECKPOINT_PUBLICATION_CONTRACT: &str = "Rust validates one complete canonical GM1 arena and all metadata, publishes one new synchronized CM1FCP1 artifact atomically, and never replaces an existing target";
pub const GM1_RESTORE_PUBLICATION_CONTRACT: &str = "Rust writes one exact arena plus canonical manifest and receipt into a new sibling staging directory, synchronizes every file, and renames without replacing an existing destination";
pub const GM1_CHECKPOINT_CLAIM_LIMIT: &str = "exact post-large-step GM1 carrier/cadence persistence and raw-bit restore only; padded prognostic halos, stage-held tendencies, acoustic/advection/closure scratch, tke3d/tketen, live scheduler wiring, uninterrupted/resumed trajectory identity, forecast skill, tornado, and model validation remain unclaimed";

pub const GM1_REAL_NZ: usize = 75;
pub const GM1_REAL_NY: usize = 300;
pub const GM1_REAL_NX: usize = 300;
pub const GM1_REAL_SPACING_M: [f64; 3] = [100.0, 100.0, 200.0];
pub const GM1_REAL_DT_S: f64 = 0.5;

pub const GM1_PRODUCT_MANIFEST_SHA256: &str =
    "7fd218a005b1c9f01d37b57452b281c2df6c797973268e8fe86b996eb6b61e2e";
pub const GM1_PRODUCT_PAYLOAD_SHA256: &str =
    "8f589cee64b044afcfef34dca6b638c3829946a999537885b1db3576d4ec7981";
pub const GM1_PRODUCT_SOURCE_ARTIFACT_SHA256: &str =
    "d98951632ec2c3cf411f74aa35e1e2ab09296aa22badb432976a6a6f9b0f0a7f";
pub const GM1_PRODUCT_SOURCE_CONTRACT_SHA256: &str =
    "d1a3ca0764fd0e15aa07fbf9b613abb756ae1222fdf0baf63fb35d5c1392dd12";
pub const GM1_GEOPOTENTIAL_MANIFEST_SHA256: &str =
    "01c028338ddbfaa001048c2fb230d90585542f4fea798f3a5c4ae1a4eb5bba4b";
pub const GM1_GEOPOTENTIAL_PAYLOAD_SHA256: &str =
    "deec02bec84da5136eaa23f2fc7780dba04e1140068eede39cd540a04ed83a4b";
pub const GM1_GEOPOTENTIAL_RECEIPT_SHA256: &str =
    "80f3a80632c834d7889ae29a5fcb6e4609bfea9cb329a8702471a042621a04dd";
pub const GM1_GEOPOTENTIAL_SOURCE_RESTART_SHA256: &str =
    "dfb80b614daf24f547d124d5590d17069f8001d3470e0474a998649e21a8db4a";
pub const GM1_F02_ROI_PRODUCT_MANIFEST_SHA256: &str =
    "b25de4008d17ddad46b8057f022b50a76e6bff61b537fa9952f68e638adf2567";
pub const GM1_F02_ROI_PRODUCT_PAYLOAD_SHA256: &str =
    "ddafeaa8ca523e81925cb4b53e0577963a3e57e06114b6b7df8c6d436e0d68b1";
pub const GM1_F02_ROI_PRODUCT_SOURCE_ARTIFACT_SHA256: &str =
    "356fe3022ce83909f5c14e7ce9a3639811d9f6dfc5ed9b33a0782cfffb7cd79d";
pub const GM1_F02_ROI_PRODUCT_SOURCE_CONTRACT_SHA256: &str =
    "a9ccd6f0c6650fd89cd42f16977209f57695464181bcb62beff14e9a9f61d896";
pub const GM1_F02_ROI_GEOPOTENTIAL_MANIFEST_SHA256: &str =
    "da37e1a22f7ef93a4abdd98fa8373fc3d50f26091aeac4812a9c216970433d1a";
pub const GM1_F02_ROI_GEOPOTENTIAL_PAYLOAD_SHA256: &str =
    "74625673f6a3aba254b90eff65dedda0688dd64d62f94eeec8709b365748ae3e";
pub const GM1_F02_ROI_GEOPOTENTIAL_RECEIPT_SHA256: &str =
    "09eb31934987b30ff953fb9846a2ab10bf73c8f13452e2612b26a0b2306b64b4";
pub const GM1_F02_ROI_GEOPOTENTIAL_SOURCE_RESTART_SHA256: &str =
    "9efe09771763a3f727af715f7dd41a2c6256cd5613aa2f31637cc08c9721ed6e";

const AUTHORITY_TARBALL_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
const AUTHORITY_SOURCES: [(&str, &str); 4] = [
    (
        "src/cm1.F",
        "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
    ),
    (
        "src/solve1.F",
        "515346f18268aea39e7a919536ee6187386e5db6d184e6e4c597509c701e2076",
    ),
    (
        "src/solve2.F",
        "a853067a5d3a943aed099dd85a8335fae9c16035a4fe18bed41ae9b05eecf734",
    ),
    (
        "src/solve3.F",
        "686f89fa8c58fbf19d8fa2f5954e930eb2ec869ce60586fd763416f69c6a3393",
    ),
];

pub const GM1_OPERATOR_SOURCE_KEYS: [&str; 22] = [
    "closure.coordinator",
    "closure.rayleigh-held",
    "closure.scalar-turbulence",
    "closure.sgs-local",
    "closure.sgs-stress",
    "closure.terrain-deformation",
    "closure.tke-evolution",
    "dynamics.apmasscon",
    "dynamics.lateral-closure",
    "dynamics.moist-stage-bridge",
    "dynamics.momentum-advection5",
    "dynamics.momentum-rk",
    "dynamics.psolver3-terrain-open",
    "dynamics.scalar-advection5",
    "dynamics.scalar-moist-rk",
    "dynamics.sound3-open",
    "dynamics.terrain-open-boundary",
    "forecast.coordinator",
    "physics.arwen-wsm6-estate",
    "physics.transition",
    "physics.wsm6-pdefq",
    "state.initialized-product-terrain-open",
];

const CORE_SCALAR_NAMES: [&str; 16] = [
    "completed_large_steps",
    "completed_wsm6_bridge_calls",
    "completed_wsm6_component_calls",
    "completed_wsm6_transitions",
    "large_step_boundary",
    "mass1_f64",
    "moist_buoyancy_applied",
    "moist_post_acoustic_ready",
    "moist_terms_ready",
    "next_stage",
    "post_stage_hook_active",
    "quiescent_capture",
    "radiation_history_initialized",
    "scalar_completed",
    "scalar_next_nrk",
    "solve1_ready",
];

pub const GM1_WSM6_ALIASES: [(&str, &str); 6] = [
    ("state.pp3d", "pp3d"),
    ("state.prs", "prs"),
    ("state.q3d", "q3d"),
    ("state.rho", "rho"),
    ("state.rr", "rr"),
    ("state.th3d", "th3d"),
];

const PHYSICAL_NAMES: [&str; 17] = [
    "ua", "u3d", "va", "v3d", "wa", "w3d", "ppi", "pp3d", "ppx", "tha", "th3d", "qa", "q3d", "prs",
    "rho", "rr", "rf",
];
const RADIATION_NAMES: [&str; 4] = ["radbcw", "radbce", "radbcs", "radbcn"];
const WSM6_UNIQUE_NAMES: [&str; 15] = [
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "held.qcten",
    "component.h_diabatic",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_sr",
];
const SPECIES: [&str; 6] = ["qv", "qc", "qr", "qi", "qs", "qg"];

/// CM1 `src/param.F:3441-3449`, the `ptype=5` `inum=1` table: the same six
/// mass mixing ratios followed by four number concentrations, in CM1's own
/// order `nci, ncs, ncr, ncg`.  The order is CM1's, not ArWen's launcher's.
const MORRISON_SPECIES: [&str; 10] = [
    "qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncs", "ncr", "ncg",
];

/// The Morrison component's held tendencies and retained carriers.  It is the
/// WSM6 list plus `component.effr` (Morrison diagnoses a rain effective
/// radius; WSM6 does not) and `component.nc`, the constant cloud-droplet
/// number CM1 reassigns from `ndcnst` because `src/morrison.F:325` fixes
/// `INUM = 1` and the concentration is therefore not on the transported axis.
const MORRISON_UNIQUE_NAMES: [&str; 17] = [
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "held.qcten",
    "component.h_diabatic",
    "component.effc",
    "component.effr",
    "component.effi",
    "component.effs",
    "component.nc",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_sr",
];

/// The Morrison profile identity.  It is a different artifact contract, not a
/// variant of the WSM6 one: a reader that assumed six species would decode
/// four kg-1 rows as mixing ratios.
pub const GM1_MORRISON_CHECKPOINT_PROFILE_ID: &str = "gm1-real-100m-forecast-morrison/v1";

/// CM1 `src/param.F:3346-3354`, the `ptype=3` table: the same six mass mixing
/// ratios followed by cloud-ice and rain number, in CM1's own order.
const THOMPSON_SPECIES: [&str; 8] = [
    "qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncr",
];

/// Aerosol-aware Thompson has no CM1 `ptype`; the axis is ArWen's
/// (`gpuwm/core/microphysics_aerosol.py`): the classic eight scalars plus
/// cloud-droplet number and the two aerosol numbers the scheme activates
/// droplets and nucleates ice from.
const THOMPSON_AEROSOL_SPECIES: [&str; 11] = [
    "qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncr", "ncc", "nwfa", "nifa",
];

/// Aerosol-aware Thompson retains WSM6's carriers plus WRF's two surface
/// aerosol emission rates, which are two-dimensional fields the scheme reads
/// and CM1 has no equivalent for.
const THOMPSON_AEROSOL_UNIQUE_NAMES: [&str; 17] = [
    "held.qpten",
    "held.qtten",
    "held.qvten",
    "held.qcten",
    "component.h_diabatic",
    "component.effc",
    "component.effi",
    "component.effs",
    "component.nwfa2d",
    "component.nifa2d",
    "component.mp_rainnc",
    "component.mp_rainncv",
    "component.mp_snownc",
    "component.mp_snowncv",
    "component.mp_graupelnc",
    "component.mp_graupelncv",
    "component.mp_sr",
];

pub const GM1_THOMPSON_AEROSOL_CHECKPOINT_PROFILE_ID: &str =
    "gm1-real-100m-forecast-thompson-aerosol/v1";

/// WDM6 has no CM1 `ptype`; the axis is ArWen's (`gpuwm/core/wdm6.py:151`),
/// written in CM1's own naming convention: the six mass rows then the CCN,
/// cloud-droplet and rain number concentrations.
const WDM6_SPECIES: [&str; 9] = [
    "qv", "qc", "qr", "qi", "qs", "qg", "ncn", "ncc", "ncr",
];

/// Thompson and WDM6 retain exactly the carriers WSM6 retains: one latent
/// heating rate, three effective radii and seven surface accumulators.  Their
/// checkpoints are still separate contracts, because the packed scalar axis
/// beside those carriers is eight and nine rows rather than six, and a reader
/// that assumed six would decode number concentrations as mixing ratios.
pub const GM1_THOMPSON_CHECKPOINT_PROFILE_ID: &str =
    "gm1-real-100m-forecast-thompson/v1";
pub const GM1_WDM6_CHECKPOINT_PROFILE_ID: &str = "gm1-real-100m-forecast-wdm6/v1";

/// Which CM1 `ptype` a checkpoint's carrier inventory belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gm1CheckpointComponent {
    /// CM1 `ptype=7`, ArWen `mp_physics=6`.
    Wsm6,
    /// CM1 `ptype=5`, ArWen `mp_physics=10`.
    Morrison,
    /// CM1 `ptype=3`, ArWen `mp_physics=8`.
    Thompson,
    /// ArWen `mp_physics=16`.  CM1 r21.1 has no WDM6 `ptype`.
    Wdm6,
    /// ArWen `mp_physics=28`.  CM1 r21.1 has no aerosol-aware Thompson `ptype`.
    ThompsonAerosol,
}

impl Gm1CheckpointComponent {
    pub const fn component_id(self) -> &'static str {
        match self {
            Self::Wsm6 => "wsm6-mp6",
            Self::Morrison => "morrison-mp10",
            Self::Thompson => "thompson-mp8",
            Self::Wdm6 => "wdm6-mp16",
            Self::ThompsonAerosol => "thompson-aerosol-mp28",
        }
    }

    /// CM1's `ptype`, or `None` where CM1 r21.1 numbers no such scheme.
    pub const fn cm1_ptype(self) -> u32 {
        match self {
            Self::Wsm6 => 7,
            Self::Morrison => 5,
            Self::Thompson => 3,
            // WDM6 is absent from src/param.F:3141-4509 entirely.  Zero is not
            // a ptype: `cm1_scheme_present` is the question a reader asks.
            Self::Wdm6 | Self::ThompsonAerosol => 0,
        }
    }

    /// Whether CM1 r21.1 has a scheme of its own at this component's identity.
    pub const fn cm1_scheme_present(self) -> bool {
        !matches!(self, Self::Wdm6 | Self::ThompsonAerosol)
    }

    pub const fn profile_id(self) -> &'static str {
        match self {
            Self::Wsm6 => GM1_CHECKPOINT_PROFILE_ID,
            Self::Morrison => GM1_MORRISON_CHECKPOINT_PROFILE_ID,
            Self::Thompson => GM1_THOMPSON_CHECKPOINT_PROFILE_ID,
            Self::Wdm6 => GM1_WDM6_CHECKPOINT_PROFILE_ID,
            Self::ThompsonAerosol => GM1_THOMPSON_AEROSOL_CHECKPOINT_PROFILE_ID,
        }
    }

    pub fn species(self) -> &'static [&'static str] {
        match self {
            Self::Wsm6 => &SPECIES,
            Self::Morrison => &MORRISON_SPECIES,
            Self::Thompson => &THOMPSON_SPECIES,
            Self::Wdm6 => &WDM6_SPECIES,
            Self::ThompsonAerosol => &THOMPSON_AEROSOL_SPECIES,
        }
    }

    pub fn unique_names(self) -> &'static [&'static str] {
        match self {
            // Thompson and WDM6 retain exactly WSM6's carriers; only the
            // packed scalar axis beside them differs, and `species` carries
            // that.
            Self::Wsm6 | Self::Thompson | Self::Wdm6 => &WSM6_UNIQUE_NAMES,
            Self::Morrison => &MORRISON_UNIQUE_NAMES,
            Self::ThompsonAerosol => &THOMPSON_AEROSOL_UNIQUE_NAMES,
        }
    }

    /// Exactly the carriers `gm1_array_specs` emits for this component.
    pub fn array_count(self) -> usize {
        PHYSICAL_NAMES.len() + RADIATION_NAMES.len() + self.unique_names().len() + 1
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "wsm6-mp6" => Ok(Self::Wsm6),
            "morrison-mp10" => Ok(Self::Morrison),
            "thompson-mp8" => Ok(Self::Thompson),
            "wdm6-mp16" => Ok(Self::Wdm6),
            "thompson-aerosol-mp28" => Ok(Self::ThompsonAerosol),
            other => Err(invalid(format!(
                "GM1 checkpoint column component {other:?} has no declared carrier \
                 inventory; restoring it would decode the packed scalar axis with the \
                 wrong number of rows and the wrong units"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gm1CheckpointAdmission {
    Production,
    CorrectedRealCropProof,
}

impl Gm1CheckpointAdmission {
    fn as_str(self) -> &'static str {
        match self {
            Self::Production => "production-full-domain",
            Self::CorrectedRealCropProof => "bounded-corrected-real-proof",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "production-full-domain" => Ok(Self::Production),
            "bounded-corrected-real-proof" => Ok(Self::CorrectedRealCropProof),
            _ => Err(invalid(format!(
                "GM1 checkpoint admission {value:?} is unknown"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gm1CheckpointConfig {
    pub j_start: usize,
    pub i_start: usize,
    pub ny: usize,
    pub nx: usize,
    pub admission: Gm1CheckpointAdmission,
    /// Which CM1 `ptype`'s carrier inventory this checkpoint declares.  The
    /// published v1 profile is WSM6's, so `production()` selects it and every
    /// existing caller keeps exactly the artifact it wrote before.
    pub component: Gm1CheckpointComponent,
}

impl Gm1CheckpointConfig {
    pub const fn production() -> Self {
        Self {
            j_start: 0,
            i_start: 0,
            ny: GM1_REAL_NY,
            nx: GM1_REAL_NX,
            admission: Gm1CheckpointAdmission::Production,
            component: Gm1CheckpointComponent::Wsm6,
        }
    }

    /// The same geometry and admission with a different column component.
    pub const fn with_component(self, component: Gm1CheckpointComponent) -> Self {
        Self { component, ..self }
    }

    pub fn corrected_real_crop(
        j_start: usize,
        i_start: usize,
        ny: usize,
        nx: usize,
    ) -> Result<Self> {
        let value = Self {
            j_start,
            i_start,
            ny,
            nx,
            admission: Gm1CheckpointAdmission::CorrectedRealCropProof,
            component: Gm1CheckpointComponent::Wsm6,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(self) -> Result<()> {
        if self.ny == 0
            || self.nx == 0
            || self.j_start.checked_add(self.ny).is_none()
            || self.i_start.checked_add(self.nx).is_none()
            || self.j_start + self.ny > GM1_REAL_NY
            || self.i_start + self.nx > GM1_REAL_NX
        {
            return Err(invalid(
                "GM1 checkpoint crop is empty, overflows, or exceeds the sealed 300x300 source domain",
            ));
        }
        // The production admission is about the captured DOMAIN, not about
        // which column component wrote it: comparing the whole value would
        // refuse a full-domain capture solely for carrying another CM1 ptype.
        if self.admission == Gm1CheckpointAdmission::Production
            && (self.j_start, self.i_start, self.ny, self.nx) != (0, 0, GM1_REAL_NY, GM1_REAL_NX)
        {
            return Err(invalid(
                "GM1 production checkpoint admission requires the full 300x300 domain at origin 0,0",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gm1SourceIdentities {
    pub product_manifest_sha256: String,
    pub product_payload_sha256: String,
    pub product_source_artifact_sha256: String,
    pub product_source_contract_sha256: String,
    pub geopotential_manifest_sha256: String,
    pub geopotential_payload_sha256: String,
    pub geopotential_receipt_sha256: String,
    pub geopotential_source_restart_sha256: String,
}

impl Gm1SourceIdentities {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        product_manifest_sha256: impl Into<String>,
        product_payload_sha256: impl Into<String>,
        product_source_artifact_sha256: impl Into<String>,
        product_source_contract_sha256: impl Into<String>,
        geopotential_manifest_sha256: impl Into<String>,
        geopotential_payload_sha256: impl Into<String>,
        geopotential_receipt_sha256: impl Into<String>,
        geopotential_source_restart_sha256: impl Into<String>,
    ) -> Result<Self> {
        let result = Self {
            product_manifest_sha256: product_manifest_sha256.into(),
            product_payload_sha256: product_payload_sha256.into(),
            product_source_artifact_sha256: product_source_artifact_sha256.into(),
            product_source_contract_sha256: product_source_contract_sha256.into(),
            geopotential_manifest_sha256: geopotential_manifest_sha256.into(),
            geopotential_payload_sha256: geopotential_payload_sha256.into(),
            geopotential_receipt_sha256: geopotential_receipt_sha256.into(),
            geopotential_source_restart_sha256: geopotential_source_restart_sha256.into(),
        };
        for (label, actual) in [
            ("product manifest SHA-256", &result.product_manifest_sha256),
            ("product payload SHA-256", &result.product_payload_sha256),
            (
                "product source artifact SHA-256",
                &result.product_source_artifact_sha256,
            ),
            (
                "product source contract SHA-256",
                &result.product_source_contract_sha256,
            ),
            (
                "geopotential manifest SHA-256",
                &result.geopotential_manifest_sha256,
            ),
            (
                "geopotential payload SHA-256",
                &result.geopotential_payload_sha256,
            ),
            (
                "geopotential receipt SHA-256",
                &result.geopotential_receipt_sha256,
            ),
            (
                "geopotential source restart SHA-256",
                &result.geopotential_source_restart_sha256,
            ),
        ] {
            validate_sha256(actual, label)?;
        }
        if result != Self::legacy_values() && result != Self::f02_roi_values() {
            return Err(invalid(
                "GM1 source identity tuple is neither the sealed legacy product nor the sealed f02 interior-ROI product",
            ));
        }
        Ok(result)
    }

    fn legacy_values() -> Self {
        Self {
            product_manifest_sha256: GM1_PRODUCT_MANIFEST_SHA256.to_owned(),
            product_payload_sha256: GM1_PRODUCT_PAYLOAD_SHA256.to_owned(),
            product_source_artifact_sha256: GM1_PRODUCT_SOURCE_ARTIFACT_SHA256.to_owned(),
            product_source_contract_sha256: GM1_PRODUCT_SOURCE_CONTRACT_SHA256.to_owned(),
            geopotential_manifest_sha256: GM1_GEOPOTENTIAL_MANIFEST_SHA256.to_owned(),
            geopotential_payload_sha256: GM1_GEOPOTENTIAL_PAYLOAD_SHA256.to_owned(),
            geopotential_receipt_sha256: GM1_GEOPOTENTIAL_RECEIPT_SHA256.to_owned(),
            geopotential_source_restart_sha256: GM1_GEOPOTENTIAL_SOURCE_RESTART_SHA256.to_owned(),
        }
    }

    fn f02_roi_values() -> Self {
        Self {
            product_manifest_sha256: GM1_F02_ROI_PRODUCT_MANIFEST_SHA256.to_owned(),
            product_payload_sha256: GM1_F02_ROI_PRODUCT_PAYLOAD_SHA256.to_owned(),
            product_source_artifact_sha256: GM1_F02_ROI_PRODUCT_SOURCE_ARTIFACT_SHA256.to_owned(),
            product_source_contract_sha256: GM1_F02_ROI_PRODUCT_SOURCE_CONTRACT_SHA256.to_owned(),
            geopotential_manifest_sha256: GM1_F02_ROI_GEOPOTENTIAL_MANIFEST_SHA256.to_owned(),
            geopotential_payload_sha256: GM1_F02_ROI_GEOPOTENTIAL_PAYLOAD_SHA256.to_owned(),
            geopotential_receipt_sha256: GM1_F02_ROI_GEOPOTENTIAL_RECEIPT_SHA256.to_owned(),
            geopotential_source_restart_sha256: GM1_F02_ROI_GEOPOTENTIAL_SOURCE_RESTART_SHA256
                .to_owned(),
        }
    }

    pub fn sealed() -> Self {
        Self::legacy_values()
    }

    pub fn sealed_f02_roi() -> Self {
        Self::f02_roi_values()
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        object(vec![
            ("schema", string(GM1_SOURCE_IDENTITIES_SCHEMA)),
            ("source", source_json(self)),
        ])
        .canonical_bytes()
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self> {
        let value = json::parse(bytes)
            .map_err(|error| invalid(format!("GM1 source identities: {error}")))?;
        exact_keys(&value, &["schema", "source"], "GM1 source identities")?;
        fixed_string(
            &value,
            "schema",
            GM1_SOURCE_IDENTITIES_SCHEMA,
            "GM1 source identities",
        )?;
        let result = parse_source(
            value
                .get("source")
                .map_err(|error| invalid(format!("GM1 source identities.source: {error}")))?,
        )?;
        if value.canonical_bytes() != bytes {
            return Err(invalid("GM1 source identities are not canonical JSON"));
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gm1OperatorManifest {
    bytes: Vec<u8>,
    sha256: String,
}

impl Gm1OperatorManifest {
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.is_empty() || bytes.len() > GM1_CHECKPOINT_BLOB_MAX_NBYTES {
            return Err(invalid(
                "GM1 operator manifest length is outside its v1 cap",
            ));
        }
        let value = json::parse(bytes)
            .map_err(|error| invalid(format!("GM1 operator manifest is invalid JSON: {error}")))?;
        if value.canonical_bytes() != bytes {
            return Err(invalid("GM1 operator manifest is not canonical JSON"));
        }
        exact_keys(
            &value,
            &["model", "operators", "schema"],
            "GM1 operator manifest",
        )?;
        fixed_string(
            &value,
            "schema",
            GM1_OPERATOR_MANIFEST_SCHEMA,
            "GM1 operator manifest",
        )?;
        fixed_string(
            &value,
            "model",
            GM1_CHECKPOINT_MODEL,
            "GM1 operator manifest",
        )?;
        let operators = value
            .get("operators")
            .and_then(JsonValue::as_object)
            .map_err(|error| invalid(format!("GM1 operator manifest.operators: {error}")))?;
        let actual = operators
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>();
        let expected = GM1_OPERATOR_SOURCE_KEYS
            .into_iter()
            .collect::<BTreeSet<_>>();
        if actual != expected {
            return Err(invalid(format!(
                "GM1 operator source inventory is {actual:?}; expected {expected:?}"
            )));
        }
        for (name, value) in operators {
            let hash = value.as_str().map_err(|error| {
                invalid(format!("GM1 operator source {name} identity: {error}"))
            })?;
            validate_sha256(hash, &format!("GM1 operator source {name} SHA-256"))?;
        }
        Ok(Self {
            bytes: bytes.to_vec(),
            sha256: sha256::digest_hex(bytes),
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gm1CheckpointReceipt {
    pub forecast: ForecastReceipt,
    pub raw_arena_sha256: String,
    pub raw_arena_nbytes: u64,
    pub operator_manifest_sha256: String,
    pub completed_transitions: u64,
    pub source: Gm1SourceIdentities,
}

impl Gm1CheckpointReceipt {
    pub fn canonical_json(&self) -> String {
        canonical_string(object(vec![
            ("artifact_nbytes", number(self.forecast.artifact_nbytes)),
            ("artifact_sha256", string(&self.forecast.artifact_sha256)),
            ("carrier_abi", string(GM1_CHECKPOINT_CARRIER_ABI)),
            ("carrier_count", number(self.forecast.arrays)),
            ("capture_phase", string(GM1_CHECKPOINT_CAPTURE_PHASE)),
            ("claim_limit", string(GM1_CHECKPOINT_CLAIM_LIMIT)),
            (
                "completed_transitions",
                u64_number(self.completed_transitions),
            ),
            ("contract_sha256", string(&self.forecast.contract_sha256)),
            ("model", string(GM1_CHECKPOINT_MODEL)),
            (
                "operator_manifest_sha256",
                string(&self.operator_manifest_sha256),
            ),
            ("payload_sha256", string(&self.forecast.payload_sha256)),
            (
                "publication_contract",
                string(GM1_CHECKPOINT_PUBLICATION_CONTRACT),
            ),
            ("raw_arena_nbytes", u64_number(self.raw_arena_nbytes)),
            ("raw_arena_sha256", string(&self.raw_arena_sha256)),
            ("schema", string(GM1_CHECKPOINT_RECEIPT_SCHEMA)),
            ("source", source_json(&self.source)),
            ("step", u64_number(self.forecast.step)),
        ]))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gm1RestoreReceipt {
    pub checkpoint_artifact_sha256: String,
    pub checkpoint_contract_sha256: String,
    pub manifest_sha256: String,
    pub manifest_nbytes: usize,
    pub payload_sha256: String,
    pub payload_nbytes: u64,
    pub carrier_count: usize,
    pub step: u64,
}

impl Gm1RestoreReceipt {
    pub fn canonical_json(&self) -> String {
        canonical_string(object(vec![
            ("carrier_abi", string(GM1_CHECKPOINT_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            (
                "checkpoint_artifact_sha256",
                string(&self.checkpoint_artifact_sha256),
            ),
            (
                "checkpoint_contract_sha256",
                string(&self.checkpoint_contract_sha256),
            ),
            ("claim_limit", string(GM1_CHECKPOINT_CLAIM_LIMIT)),
            ("manifest_nbytes", number(self.manifest_nbytes)),
            ("manifest_sha256", string(&self.manifest_sha256)),
            ("model", string(GM1_CHECKPOINT_MODEL)),
            ("payload_nbytes", u64_number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            (
                "publication_contract",
                string(GM1_RESTORE_PUBLICATION_CONTRACT),
            ),
            ("schema", string(GM1_RESTORE_RECEIPT_SCHEMA)),
            ("step", u64_number(self.step)),
        ]))
    }
}

#[derive(Clone, Debug)]
pub struct RestoredGm1Checkpoint {
    contract: ForecastContract,
    forecast: RestoredForecast,
    config: Gm1CheckpointConfig,
    source: Gm1SourceIdentities,
    operator_manifest: Gm1OperatorManifest,
    scheduler_extension: Vec<u8>,
    raw_arena_sha256: String,
    mass1_f64: f64,
    completed_transitions: u64,
}

impl RestoredGm1Checkpoint {
    pub fn contract(&self) -> &ForecastContract {
        &self.contract
    }

    pub fn forecast(&self) -> &RestoredForecast {
        &self.forecast
    }

    pub fn config(&self) -> Gm1CheckpointConfig {
        self.config
    }

    pub fn source(&self) -> &Gm1SourceIdentities {
        &self.source
    }

    pub fn operator_manifest(&self) -> &Gm1OperatorManifest {
        &self.operator_manifest
    }

    pub fn scheduler_extension(&self) -> &[u8] {
        &self.scheduler_extension
    }

    pub fn raw_arena_sha256(&self) -> &str {
        &self.raw_arena_sha256
    }

    pub fn mass1_f64(&self) -> f64 {
        self.mass1_f64
    }

    pub fn completed_transitions(&self) -> u64 {
        self.completed_transitions
    }
}

#[derive(Clone, Debug)]
struct ParsedIdentity {
    config: Gm1CheckpointConfig,
    source: Gm1SourceIdentities,
    raw_arena_sha256: String,
    operator_manifest_sha256: String,
    scheduler_extension_sha256: String,
}

#[derive(Clone, Debug)]
struct RestoreCarrier {
    name: String,
    shape: Vec<usize>,
    offset_nbytes: u64,
    nbytes: u64,
    sha256: String,
}

impl RestoreCarrier {
    fn json(&self) -> JsonValue {
        object(vec![
            ("dtype", string("<f4")),
            ("itemsize", number(4)),
            ("name", string(&self.name)),
            ("nbytes", u64_number(self.nbytes)),
            ("offset_nbytes", u64_number(self.offset_nbytes)),
            ("order", string("C")),
            (
                "shape",
                JsonValue::Array(self.shape.iter().copied().map(number).collect()),
            ),
            ("sha256", string(&self.sha256)),
        ])
    }
}

pub fn gm1_real_layout_json(operator_manifest: &Gm1OperatorManifest) -> Result<String> {
    gm1_layout_json(Gm1CheckpointConfig::production(), operator_manifest)
}

pub fn gm1_layout_json(
    config: Gm1CheckpointConfig,
    operator_manifest: &Gm1OperatorManifest,
) -> Result<String> {
    config.validate()?;
    let contract = gm1_contract(config, operator_manifest)?;
    let mut offset = 0u64;
    let mut carriers = Vec::with_capacity(contract.arrays().len());
    for spec in contract.arrays() {
        let nbytes = spec_nbytes(spec)?;
        carriers.push(object(vec![
            ("name", string(spec.name())),
            ("nbytes", u64_number(nbytes)),
            ("offset_nbytes", u64_number(offset)),
            (
                "shape",
                JsonValue::Array(spec.shape().iter().copied().map(number).collect()),
            ),
        ]));
        offset = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("GM1 layout byte count overflows"))?;
    }
    // The sealed byte count belongs to the WSM6 v1 profile.  A component with
    // a different scalar axis has a different arena by construction, so the
    // seal is checked against the profile that declares it rather than
    // against whichever component happens to be writing.
    if config.admission == Gm1CheckpointAdmission::Production
        && config.component == Gm1CheckpointComponent::Wsm6
        && offset != GM1_CHECKPOINT_REAL_PAYLOAD_NBYTES
    {
        return Err(invalid(format!(
            "GM1 production arena is {offset} bytes; v1 requires {GM1_CHECKPOINT_REAL_PAYLOAD_NBYTES}"
        )));
    }
    Ok(canonical_string(object(vec![
        ("aliases", aliases_json()),
        ("carrier_abi", string(GM1_CHECKPOINT_CARRIER_ABI)),
        ("carrier_count", number(contract.arrays().len())),
        ("carriers", JsonValue::Array(carriers)),
        ("capture_phase", string(GM1_CHECKPOINT_CAPTURE_PHASE)),
        ("claim_limit", string(GM1_CHECKPOINT_CLAIM_LIMIT)),
        ("configuration", configuration_json(config)),
        ("contract_sha256", string(&contract.sha256())),
        ("dtype", string("<f4")),
        ("itemsize", number(4)),
        ("model", string(GM1_CHECKPOINT_MODEL)),
        (
            "operator_manifest_sha256",
            string(operator_manifest.sha256()),
        ),
        ("order", string("C")),
        ("payload_nbytes", u64_number(offset)),
        ("profile_id", string(config.component.profile_id())),
        ("schema", string(GM1_CHECKPOINT_LAYOUT_SCHEMA)),
        ("species_order", string_array_json(config.component.species())),
    ])))
}

#[allow(clippy::too_many_arguments)]
pub fn write_gm1_checkpoint_new(
    raw_arena: impl AsRef<Path>,
    output: impl AsRef<Path>,
    expected_raw_arena_sha256: &str,
    config: Gm1CheckpointConfig,
    source: Gm1SourceIdentities,
    operator_manifest: Gm1OperatorManifest,
    clock: ForecastClock,
    mass1_f64: f64,
    completed_transitions: u64,
    scheduler_extension: &[u8],
) -> Result<Gm1CheckpointReceipt> {
    config.validate()?;
    validate_sha256(expected_raw_arena_sha256, "expected GM1 arena SHA-256")?;
    validate_boundary_clock(clock, completed_transitions)?;
    if !mass1_f64.is_finite() || mass1_f64 <= 0.0 {
        return Err(invalid("GM1 mass1 ledger must be finite and positive"));
    }
    let scheduler_extension = validate_scheduler_extension(scheduler_extension)?;
    let scheduler_sha256 = sha256::digest_hex(&scheduler_extension);
    let identity = identity_json(
        config,
        &source,
        expected_raw_arena_sha256,
        operator_manifest.sha256(),
        &scheduler_sha256,
    )
    .canonical_bytes();
    let contract = gm1_contract(config, &operator_manifest)?;
    let arrays = read_raw_arena(raw_arena.as_ref(), &contract, expected_raw_arena_sha256)?;
    validate_array_semantics(&arrays)?;
    let frame = ForecastFrame::new(
        clock,
        arrays,
        boundary_scalars(clock.step(), completed_transitions, mass1_f64)?,
        vec![
            ComponentBlob::new(
                GM1_CHECKPOINT_IDENTITY_BLOB,
                GM1_CHECKPOINT_IDENTITY_SCHEMA,
                identity,
            ),
            ComponentBlob::new(
                GM1_OPERATOR_MANIFEST_BLOB,
                GM1_OPERATOR_MANIFEST_SCHEMA,
                operator_manifest.bytes().to_vec(),
            ),
            ComponentBlob::new(
                GM1_SCHEDULER_EXTENSION_BLOB,
                GM1_SCHEDULER_EXTENSION_SCHEMA,
                scheduler_extension,
            ),
        ],
    );
    let forecast = write_forecast_checkpoint_new(&contract, &frame, output)?;
    Ok(Gm1CheckpointReceipt {
        raw_arena_sha256: expected_raw_arena_sha256.to_string(),
        raw_arena_nbytes: array_bytes(&contract)?,
        operator_manifest_sha256: operator_manifest.sha256().to_string(),
        completed_transitions,
        source,
        forecast,
    })
}

pub fn read_gm1_checkpoint(
    input: impl AsRef<Path>,
    expected_contract_sha256: &str,
    expected_artifact_sha256: &str,
    expected_product_manifest_sha256: &str,
    expected_operator_manifest_sha256: &str,
) -> Result<RestoredGm1Checkpoint> {
    validate_sha256(expected_artifact_sha256, "expected GM1 artifact SHA-256")?;
    validate_sha256(
        expected_product_manifest_sha256,
        "expected GM1 product manifest SHA-256",
    )?;
    validate_sha256(
        expected_operator_manifest_sha256,
        "expected GM1 operator manifest SHA-256",
    )?;
    let (contract, forecast) =
        read_forecast_checkpoint_embedded(input.as_ref(), expected_contract_sha256)?;
    if forecast.receipt().artifact_sha256 != expected_artifact_sha256 {
        return Err(invalid(
            "GM1 checkpoint artifact differs from the independent hash",
        ));
    }
    let blobs = forecast
        .frame()
        .component_blobs()
        .iter()
        .map(|blob| (blob.name(), blob))
        .collect::<BTreeMap<_, _>>();
    if blobs.len() != 3 {
        return Err(invalid("GM1 checkpoint component blob inventory changed"));
    }
    let identity_blob = required_blob(
        &blobs,
        GM1_CHECKPOINT_IDENTITY_BLOB,
        GM1_CHECKPOINT_IDENTITY_SCHEMA,
    )?;
    let operator_blob = required_blob(
        &blobs,
        GM1_OPERATOR_MANIFEST_BLOB,
        GM1_OPERATOR_MANIFEST_SCHEMA,
    )?;
    let scheduler_blob = required_blob(
        &blobs,
        GM1_SCHEDULER_EXTENSION_BLOB,
        GM1_SCHEDULER_EXTENSION_SCHEMA,
    )?;
    let parsed = parse_identity(identity_blob.bytes())?;
    if parsed.source.product_manifest_sha256 != expected_product_manifest_sha256 {
        return Err(invalid(
            "GM1 checkpoint product manifest differs from the independent expectation",
        ));
    }
    let operator_manifest = Gm1OperatorManifest::from_canonical_bytes(operator_blob.bytes())?;
    if operator_manifest.sha256() != expected_operator_manifest_sha256
        || operator_manifest.sha256() != parsed.operator_manifest_sha256
    {
        return Err(invalid(
            "GM1 checkpoint operator manifest differs from its independent or embedded identity",
        ));
    }
    let scheduler_extension = validate_scheduler_extension(scheduler_blob.bytes())?;
    if sha256::digest_hex(&scheduler_extension) != parsed.scheduler_extension_sha256 {
        return Err(invalid(
            "GM1 scheduler extension differs from its embedded identity",
        ));
    }
    let expected_identity = identity_json(
        parsed.config,
        &parsed.source,
        &parsed.raw_arena_sha256,
        &parsed.operator_manifest_sha256,
        &parsed.scheduler_extension_sha256,
    )
    .canonical_bytes();
    if expected_identity != identity_blob.bytes() {
        return Err(invalid("GM1 checkpoint identity semantics changed"));
    }
    let expected_contract = gm1_contract(parsed.config, &operator_manifest)?;
    if contract != expected_contract {
        return Err(invalid(
            "GM1 checkpoint embedded contract differs from the Rust-owned inventory",
        ));
    }
    let (completed_transitions, mass1_f64) =
        validate_boundary_scalars(forecast.frame(), forecast.frame().clock().step())?;
    validate_boundary_clock(forecast.frame().clock(), completed_transitions)?;
    validate_array_semantics(forecast.frame().arrays())?;
    let raw_arena_sha256 = arrays_sha256(forecast.frame().arrays());
    if raw_arena_sha256 != parsed.raw_arena_sha256 {
        return Err(invalid(
            "GM1 restored arena differs from its embedded raw hash",
        ));
    }
    Ok(RestoredGm1Checkpoint {
        contract,
        forecast,
        config: parsed.config,
        source: parsed.source,
        operator_manifest,
        scheduler_extension,
        raw_arena_sha256,
        mass1_f64,
        completed_transitions,
    })
}

pub fn restore_gm1_checkpoint_new(
    restored: &RestoredGm1Checkpoint,
    output_directory: impl AsRef<Path>,
) -> Result<Gm1RestoreReceipt> {
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace GM1 restore destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "GM1 restore parent does not exist: {}",
            parent.display()
        )));
    }
    let staging = staging_path(output)?;
    fs::create_dir(&staging)?;
    match restore_into_directory(restored, &staging) {
        Ok(receipt) => {
            if let Err(error) = fs::rename(&staging, output) {
                let _ = fs::remove_dir_all(&staging);
                return Err(error.into());
            }
            Ok(receipt)
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

fn gm1_contract(
    config: Gm1CheckpointConfig,
    operator_manifest: &Gm1OperatorManifest,
) -> Result<ForecastContract> {
    ForecastContract::new(
        // The profile id belongs to the component's carrier inventory: a
        // ten-scalar arena must not be published under the six-species v1
        // identity, because a reader would decode four kg-1 rows as mixing
        // ratios.  Wsm6 returns exactly GM1_CHECKPOINT_PROFILE_ID.
        config.component.profile_id(),
        sha256::digest_hex(&configuration_json(config).canonical_bytes()),
        sha256::digest_hex(&execution_identity_json(operator_manifest.sha256()).canonical_bytes()),
        gm1_array_specs(config)?,
        vec![
            SchedulerScalarSpec::new("completed_large_steps", SchedulerScalarType::U64)?,
            SchedulerScalarSpec::new("completed_wsm6_bridge_calls", SchedulerScalarType::U64)?,
            SchedulerScalarSpec::new("completed_wsm6_component_calls", SchedulerScalarType::U64)?,
            SchedulerScalarSpec::new("completed_wsm6_transitions", SchedulerScalarType::U64)?,
            SchedulerScalarSpec::new("large_step_boundary", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("mass1_f64", SchedulerScalarType::F64)?,
            SchedulerScalarSpec::new("moist_buoyancy_applied", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("moist_post_acoustic_ready", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("moist_terms_ready", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("next_stage", SchedulerScalarType::I64)?,
            SchedulerScalarSpec::new("post_stage_hook_active", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("quiescent_capture", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("radiation_history_initialized", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("scalar_completed", SchedulerScalarType::Bool)?,
            SchedulerScalarSpec::new("scalar_next_nrk", SchedulerScalarType::I64)?,
            SchedulerScalarSpec::new("solve1_ready", SchedulerScalarType::Bool)?,
        ],
        vec![
            ComponentBlobSpec::new(
                GM1_CHECKPOINT_IDENTITY_BLOB,
                GM1_CHECKPOINT_IDENTITY_SCHEMA,
                GM1_CHECKPOINT_BLOB_MAX_NBYTES,
            )?,
            ComponentBlobSpec::new(
                GM1_OPERATOR_MANIFEST_BLOB,
                GM1_OPERATOR_MANIFEST_SCHEMA,
                GM1_CHECKPOINT_BLOB_MAX_NBYTES,
            )?,
            ComponentBlobSpec::new(
                GM1_SCHEDULER_EXTENSION_BLOB,
                GM1_SCHEDULER_EXTENSION_SCHEMA,
                GM1_CHECKPOINT_BLOB_MAX_NBYTES,
            )?,
        ],
    )
}

fn gm1_array_specs(config: Gm1CheckpointConfig) -> Result<Vec<ForecastArraySpec>> {
    config.validate()?;
    let nz = GM1_REAL_NZ;
    let ny = config.ny;
    let nx = config.nx;
    let volume = vec![nz, ny, nx];
    let uface = vec![nz, ny, nx + 1];
    let vface = vec![nz, ny + 1, nx];
    let zface = vec![nz + 1, ny, nx];
    let component = config.component;
    let packed = vec![component.species().len(), nz, ny, nx];
    let surface = vec![ny, nx];
    let mut specs = Vec::with_capacity(component.array_count());
    for name in PHYSICAL_NAMES {
        let (shape, meaning, units, stagger) = match name {
            "ua" => (uface.clone(), "zonal velocity A carrier", "m s-1", "x face"),
            "u3d" => (uface.clone(), "current zonal velocity", "m s-1", "x face"),
            "va" => (
                vface.clone(),
                "meridional velocity A carrier",
                "m s-1",
                "y face",
            ),
            "v3d" => (
                vface.clone(),
                "current meridional velocity",
                "m s-1",
                "y face",
            ),
            "wa" => (
                zface.clone(),
                "vertical velocity A carrier",
                "m s-1",
                "z face",
            ),
            "w3d" => (
                zface.clone(),
                "current vertical velocity",
                "m s-1",
                "z face",
            ),
            // The packed axis carries one unit per row and a double-moment
            // table does not have one unit: the meaning names the scalars and
            // the units field says so rather than labelling four kg-1 rows as
            // mixing ratios.  The layout's species_order is the authority for
            // which row is which.
            "qa" => (
                packed.clone(),
                if component == Gm1CheckpointComponent::Wsm6 {
                    "packed six-species A carrier"
                } else {
                    "packed ten-scalar A carrier: six mass mixing ratios then four number concentrations"
                },
                if component == Gm1CheckpointComponent::Wsm6 {
                    "kg kg-1"
                } else {
                    "kg kg-1 (rows 1-6), kg-1 (rows 7-10)"
                },
                "species,scalar",
            ),
            "q3d" => (
                packed.clone(),
                if component == Gm1CheckpointComponent::Wsm6 {
                    "packed six-species current carrier"
                } else {
                    "packed ten-scalar current carrier: six mass mixing ratios then four number concentrations"
                },
                if component == Gm1CheckpointComponent::Wsm6 {
                    "kg kg-1"
                } else {
                    "kg kg-1 (rows 1-6), kg-1 (rows 7-10)"
                },
                "species,scalar",
            ),
            "prs" => (volume.clone(), "diagnostic pressure", "Pa", "scalar"),
            "rho" => (volume.clone(), "diagnostic density", "kg m-3", "scalar"),
            "rr" => (volume.clone(), "reciprocal density", "m3 kg-1", "scalar"),
            "rf" => (
                zface.clone(),
                "interface density factor",
                "model-specific",
                "z face",
            ),
            "ppi" => (
                volume.clone(),
                "Exner perturbation A carrier",
                "1",
                "scalar",
            ),
            "pp3d" => (volume.clone(), "current Exner perturbation", "1", "scalar"),
            "ppx" => (
                volume.clone(),
                "held pressure helper",
                "model-specific",
                "scalar",
            ),
            "tha" => (
                volume.clone(),
                "theta perturbation A carrier",
                "K",
                "scalar",
            ),
            "th3d" => (volume.clone(), "current theta perturbation", "K", "scalar"),
            _ => unreachable!(),
        };
        push_spec(&mut specs, name, meaning, units, stagger, shape)?;
    }
    for name in RADIATION_NAMES {
        let shape = if name == "radbcw" || name == "radbce" {
            vec![nz + 2, ny + 6]
        } else {
            vec![nz + 2, nx + 6]
        };
        push_spec(
            &mut specs,
            name,
            "padded all-open radiation history",
            "varies",
            "lateral boundary plane with source-valid halo",
            shape,
        )?;
    }
    // The published v1 profile spells these meanings with the scheme's own
    // name, so the WSM6 strings are exactly the bytes they were; a component
    // added later spells its own.
    let scheme = match component {
        Gm1CheckpointComponent::Wsm6 => "WSM6",
        Gm1CheckpointComponent::Morrison => "Morrison",
        Gm1CheckpointComponent::Thompson => "Thompson",
        Gm1CheckpointComponent::Wdm6 => "WDM6",
        Gm1CheckpointComponent::ThompsonAerosol => "Thompson aerosol-aware",
    };
    let held_meaning = format!("{scheme} recaptured held tendency");
    let surface_meaning = format!("{scheme} persistent surface carrier");
    let volume_meaning = format!("{scheme} persistent volume carrier");
    for &name in component.unique_names() {
        let (shape, meaning, units, stagger) = if name.starts_with("held.") {
            let units = match name {
                "held.qpten" => "s-1",
                "held.qtten" => "K s-1",
                _ => "kg kg-1 s-1",
            };
            (
                volume.clone(),
                held_meaning.as_str(),
                units,
                "scalar",
            )
        } else if name.starts_with("component.mp_") {
            let units = if name == "component.mp_sr" { "1" } else { "mm" };
            (
                surface.clone(),
                surface_meaning.as_str(),
                units,
                "surface",
            )
        } else {
            let units = match name {
                "component.h_diabatic" => "K s-1",
                // src/morrison.F:1832 carries the constant droplet
                // concentration in kg-1; the effective radii are metres.
                "component.nc" => "kg-1",
                _ => "m",
            };
            (
                volume.clone(),
                volume_meaning.as_str(),
                units,
                "scalar",
            )
        };
        push_spec(&mut specs, name, meaning, units, stagger, shape)?;
    }
    push_spec(
        &mut specs,
        "closure.tkea",
        "prognostic Deardorff TKE A carrier",
        "m2 s-2",
        "z face",
        zface,
    )?;
    specs.sort_by(|left, right| left.name().cmp(right.name()));
    if specs.len() != component.array_count()
        || specs
            .iter()
            .map(ForecastArraySpec::name)
            .collect::<BTreeSet<_>>()
            .len()
            != specs.len()
    {
        return Err(invalid("GM1 checkpoint carrier inventory drifted"));
    }
    Ok(specs)
}

fn push_spec(
    specs: &mut Vec<ForecastArraySpec>,
    name: &str,
    meaning: &str,
    units: &str,
    stagger: &str,
    shape: Vec<usize>,
) -> Result<()> {
    specs.push(ForecastArraySpec::new(
        name,
        meaning,
        units,
        stagger,
        one_bounds(&shape)?,
        shape,
    )?);
    Ok(())
}

fn boundary_scalars(
    step: u64,
    completed_transitions: u64,
    mass1_f64: f64,
) -> Result<Vec<SchedulerScalar>> {
    Ok(vec![
        SchedulerScalar::new("completed_large_steps", SchedulerScalarValue::U64(step))?,
        SchedulerScalar::new(
            "completed_wsm6_bridge_calls",
            SchedulerScalarValue::U64(completed_transitions),
        )?,
        SchedulerScalar::new(
            "completed_wsm6_component_calls",
            SchedulerScalarValue::U64(completed_transitions),
        )?,
        SchedulerScalar::new(
            "completed_wsm6_transitions",
            SchedulerScalarValue::U64(completed_transitions),
        )?,
        SchedulerScalar::new("large_step_boundary", SchedulerScalarValue::Bool(true))?,
        SchedulerScalar::new("mass1_f64", SchedulerScalarValue::F64(mass1_f64))?,
        SchedulerScalar::new("moist_buoyancy_applied", SchedulerScalarValue::Bool(false))?,
        SchedulerScalar::new(
            "moist_post_acoustic_ready",
            SchedulerScalarValue::Bool(false),
        )?,
        SchedulerScalar::new("moist_terms_ready", SchedulerScalarValue::Bool(false))?,
        SchedulerScalar::new("next_stage", SchedulerScalarValue::I64(1))?,
        SchedulerScalar::new("post_stage_hook_active", SchedulerScalarValue::Bool(false))?,
        SchedulerScalar::new("quiescent_capture", SchedulerScalarValue::Bool(true))?,
        SchedulerScalar::new(
            "radiation_history_initialized",
            SchedulerScalarValue::Bool(true),
        )?,
        SchedulerScalar::new("scalar_completed", SchedulerScalarValue::Bool(false))?,
        SchedulerScalar::new("scalar_next_nrk", SchedulerScalarValue::I64(1))?,
        SchedulerScalar::new("solve1_ready", SchedulerScalarValue::Bool(false))?,
    ])
}

fn validate_boundary_scalars(frame: &ForecastFrame, step: u64) -> Result<(u64, f64)> {
    let map = frame
        .scheduler_scalars()
        .iter()
        .map(|scalar| (scalar.name(), scalar.value()))
        .collect::<BTreeMap<_, _>>();
    if map.len() != CORE_SCALAR_NAMES.len()
        || map.keys().copied().collect::<BTreeSet<_>>()
            != CORE_SCALAR_NAMES.into_iter().collect::<BTreeSet<_>>()
    {
        return Err(invalid("GM1 scheduler scalar inventory changed"));
    }
    for (name, expected) in [
        ("completed_large_steps", SchedulerScalarValue::U64(step)),
        ("large_step_boundary", SchedulerScalarValue::Bool(true)),
        ("moist_buoyancy_applied", SchedulerScalarValue::Bool(false)),
        (
            "moist_post_acoustic_ready",
            SchedulerScalarValue::Bool(false),
        ),
        ("moist_terms_ready", SchedulerScalarValue::Bool(false)),
        ("next_stage", SchedulerScalarValue::I64(1)),
        ("post_stage_hook_active", SchedulerScalarValue::Bool(false)),
        ("quiescent_capture", SchedulerScalarValue::Bool(true)),
        (
            "radiation_history_initialized",
            SchedulerScalarValue::Bool(true),
        ),
        ("scalar_completed", SchedulerScalarValue::Bool(false)),
        ("scalar_next_nrk", SchedulerScalarValue::I64(1)),
        ("solve1_ready", SchedulerScalarValue::Bool(false)),
    ] {
        if map.get(name).copied() != Some(expected) {
            return Err(invalid(format!(
                "GM1 scheduler scalar {name} is not at its safe-boundary value"
            )));
        }
    }
    let completed = match map.get("completed_wsm6_transitions").copied() {
        Some(SchedulerScalarValue::U64(value)) => value,
        _ => return Err(invalid("GM1 completed transition cadence is missing")),
    };
    for name in [
        "completed_wsm6_bridge_calls",
        "completed_wsm6_component_calls",
    ] {
        if map.get(name).copied() != Some(SchedulerScalarValue::U64(completed)) {
            return Err(invalid("GM1 WSM6 bridge/component cadence diverged"));
        }
    }
    if completed == 0 || completed != step {
        return Err(invalid(
            "GM1 post-large-step checkpoint requires positive equal step/WSM6 cadence",
        ));
    }
    let mass1 = match map.get("mass1_f64").copied() {
        Some(SchedulerScalarValue::F64(value)) if value.is_finite() && value > 0.0 => value,
        _ => return Err(invalid("GM1 mass1 ledger is missing or invalid")),
    };
    Ok((completed, mass1))
}

fn validate_boundary_clock(clock: ForecastClock, completed: u64) -> Result<()> {
    if clock.dt_s().to_bits() != GM1_REAL_DT_S.to_bits() {
        return Err(invalid(
            "GM1 checkpoint clock dt differs from the sealed selection",
        ));
    }
    if clock.step() == 0 || clock.step() != completed {
        return Err(invalid(
            "GM1 checkpoint requires one or more completed equal large-step/physics transitions",
        ));
    }
    let expected_time = (clock.step() as f64) * GM1_REAL_DT_S;
    if clock.model_time_s().to_bits() != expected_time.to_bits() {
        return Err(invalid(
            "GM1 model time does not equal completed_large_steps * dt at the safe boundary",
        ));
    }
    Ok(())
}

fn read_raw_arena(
    path: &Path,
    contract: &ForecastContract,
    expected_sha256: &str,
) -> Result<Vec<ForecastArray>> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect GM1 raw arena {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid("GM1 raw arena must be a regular non-symlink file"));
    }
    let expected_nbytes = array_bytes(contract)?;
    if metadata.len() != expected_nbytes {
        return Err(invalid(format!(
            "GM1 raw arena has {} bytes; contract requires {expected_nbytes}",
            metadata.len()
        )));
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut aggregate = Sha256::default();
    let mut arrays = Vec::with_capacity(contract.arrays().len());
    let mut raw = vec![0u8; 1024 * 1024];
    for spec in contract.arrays() {
        let words = spec_words(spec)?;
        let mut values = Vec::with_capacity(words);
        let mut remaining = words;
        while remaining != 0 {
            let take_words = remaining.min(raw.len() / 4);
            let bytes = &mut raw[..take_words * 4];
            reader.read_exact(bytes)?;
            aggregate.update(bytes);
            for word in bytes.chunks_exact(4) {
                values.push(f32::from_bits(u32::from_le_bytes(
                    word.try_into().expect("four-byte word"),
                )));
            }
            remaining -= take_words;
        }
        arrays.push(ForecastArray::new(spec.name(), values));
    }
    let actual = sha256::hex(&aggregate.finalize());
    if actual != expected_sha256 {
        return Err(invalid(format!(
            "GM1 raw arena SHA-256 is {actual}; independently expected {expected_sha256}"
        )));
    }
    Ok(arrays)
}

fn validate_array_semantics(arrays: &[ForecastArray]) -> Result<()> {
    let map = arrays
        .iter()
        .map(|array| (array.name(), array.values()))
        .collect::<BTreeMap<_, _>>();
    if map.len() != GM1_CHECKPOINT_ARRAY_COUNT {
        return Err(invalid("GM1 array inventory changed"));
    }
    for array in arrays {
        if let Some(index) = array.values().iter().position(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "GM1 carrier {} contains non-finite bits {:08x} at word {index}",
                array.name(),
                array.values()[index].to_bits()
            )));
        }
    }
    for name in ["prs", "rho", "rr", "rf"] {
        require_positive(array(&map, name)?, name)?;
    }
    for name in [
        "qa",
        "q3d",
        "closure.tkea",
        "component.effc",
        "component.effi",
        "component.effs",
        "component.mp_rainnc",
        "component.mp_rainncv",
        "component.mp_snownc",
        "component.mp_snowncv",
        "component.mp_graupelnc",
        "component.mp_graupelncv",
        "component.mp_sr",
    ] {
        require_nonnegative(array(&map, name)?, name)?;
    }
    for (left, right) in [
        ("ua", "u3d"),
        ("va", "v3d"),
        ("wa", "w3d"),
        ("ppi", "pp3d"),
        ("tha", "th3d"),
        ("qa", "q3d"),
    ] {
        require_bits_equal(array(&map, left)?, array(&map, right)?, left, right)?;
    }
    Ok(())
}

fn validate_scheduler_extension(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.is_empty() || bytes.len() > GM1_CHECKPOINT_BLOB_MAX_NBYTES {
        return Err(invalid(
            "GM1 scheduler extension length is outside its v1 cap",
        ));
    }
    let value = json::parse(bytes)
        .map_err(|error| invalid(format!("GM1 scheduler extension is invalid JSON: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid("GM1 scheduler extension is not canonical JSON"));
    }
    exact_keys(&value, &["entries", "schema"], "GM1 scheduler extension")?;
    fixed_string(
        &value,
        "schema",
        GM1_SCHEDULER_EXTENSION_SCHEMA,
        "GM1 scheduler extension",
    )?;
    let entries = value
        .get("entries")
        .and_then(JsonValue::as_object)
        .map_err(|error| invalid(format!("GM1 scheduler extension.entries: {error}")))?;
    for (name, _) in entries {
        validate_extension_name(name)?;
        if CORE_SCALAR_NAMES.contains(&name.as_str()) {
            return Err(invalid(format!(
                "GM1 scheduler extension key {name:?} collides with typed cadence"
            )));
        }
    }
    Ok(bytes.to_vec())
}

fn parse_identity(bytes: &[u8]) -> Result<ParsedIdentity> {
    let value = json::parse(bytes)
        .map_err(|error| invalid(format!("GM1 identity is invalid JSON: {error}")))?;
    if value.canonical_bytes() != bytes {
        return Err(invalid("GM1 identity is not canonical JSON"));
    }
    exact_keys(
        &value,
        &[
            "aliases",
            "authority",
            "capture_phase",
            "carrier_abi",
            "claim_limit",
            "configuration",
            "model",
            "operator_manifest_sha256",
            "raw_arena_sha256",
            "recomputable",
            "restore_phase",
            "scheduler_extension_sha256",
            "schema",
            "source",
            "transition_contracts",
        ],
        "GM1 identity",
    )?;
    fixed_string(
        &value,
        "schema",
        GM1_CHECKPOINT_IDENTITY_SCHEMA,
        "GM1 identity",
    )?;
    fixed_string(&value, "model", GM1_CHECKPOINT_MODEL, "GM1 identity")?;
    let config_value = value
        .get("configuration")
        .map_err(|error| invalid(format!("GM1 identity configuration: {error}")))?;
    let config = parse_configuration(config_value)?;
    if config_value.canonical_bytes() != configuration_json(config).canonical_bytes() {
        return Err(invalid("GM1 checkpoint selection or crop drifted"));
    }
    let source_value = value
        .get("source")
        .map_err(|error| invalid(format!("GM1 identity source: {error}")))?;
    let source = parse_source(source_value)?;
    let raw_arena_sha256 = required_sha(&value, "raw_arena_sha256", "GM1 identity")?;
    let operator_manifest_sha256 =
        required_sha(&value, "operator_manifest_sha256", "GM1 identity")?;
    let scheduler_extension_sha256 =
        required_sha(&value, "scheduler_extension_sha256", "GM1 identity")?;
    Ok(ParsedIdentity {
        config,
        source,
        raw_arena_sha256,
        operator_manifest_sha256,
        scheduler_extension_sha256,
    })
}

fn identity_json(
    config: Gm1CheckpointConfig,
    source: &Gm1SourceIdentities,
    raw_arena_sha256: &str,
    operator_manifest_sha256: &str,
    scheduler_extension_sha256: &str,
) -> JsonValue {
    object(vec![
        ("aliases", aliases_json()),
        ("authority", authority_json()),
        ("capture_phase", string(GM1_CHECKPOINT_CAPTURE_PHASE)),
        ("carrier_abi", string(GM1_CHECKPOINT_CARRIER_ABI)),
        ("claim_limit", string(GM1_CHECKPOINT_CLAIM_LIMIT)),
        ("configuration", configuration_json(config)),
        ("model", string(GM1_CHECKPOINT_MODEL)),
        ("operator_manifest_sha256", string(operator_manifest_sha256)),
        ("raw_arena_sha256", string(raw_arena_sha256)),
        (
            "recomputable",
            JsonValue::Array(
                [
                    "padded prognostic/A halos from physical state plus boundary pack",
                    "momentum/scalar held and stage tendencies",
                    "acoustic, advection, moist-stage and closure scratch",
                    "tke3d exact copy of closure.tkea; tketen reset before next solve",
                    "Rayleigh coefficients from hash-bound static geometry",
                    "compiled images, launch geometry, events and transient reflectivity",
                ]
                .into_iter()
                .map(string)
                .collect(),
            ),
        ),
        ("restore_phase", string(GM1_CHECKPOINT_RESTORE_PHASE)),
        (
            "scheduler_extension_sha256",
            string(scheduler_extension_sha256),
        ),
        ("schema", string(GM1_CHECKPOINT_IDENTITY_SCHEMA)),
        ("source", source_json(source)),
        ("transition_contracts", transition_contracts_json()),
    ])
}

fn configuration_json(config: Gm1CheckpointConfig) -> JsonValue {
    object(vec![
        ("admission", string(config.admission.as_str())),
        (
            "capture_domain",
            object(vec![
                (
                    "crop_origin_yx",
                    JsonValue::Array([config.j_start, config.i_start].map(number).to_vec()),
                ),
                (
                    "crop_shape_zyx",
                    JsonValue::Array([GM1_REAL_NZ, config.ny, config.nx].map(number).to_vec()),
                ),
                (
                    "source_shape_zyx",
                    JsonValue::Array([GM1_REAL_NZ, GM1_REAL_NY, GM1_REAL_NX].map(number).to_vec()),
                ),
            ]),
        ),
        ("selection", selection_json(config)),
    ])
}

fn selection_json(config: Gm1CheckpointConfig) -> JsonValue {
    object(vec![
        ("acoustic_substeps", number(6)),
        ("coriolis", number(0)),
        ("dry_mass_correction", JsonValue::Bool(true)),
        ("dt_f64_bits", f64_bits(GM1_REAL_DT_S)),
        (
            "grid_spacing_m_f64_bits",
            JsonValue::Array(GM1_REAL_SPACING_M.map(f64_bits).to_vec()),
        ),
        (
            "grid_shape_zyx",
            JsonValue::Array([GM1_REAL_NZ, GM1_REAL_NY, GM1_REAL_NX].map(number).to_vec()),
        ),
        (
            "lateral_boundaries",
            JsonValue::Array([2usize, 2, 2, 2].map(number).to_vec()),
        ),
        ("microphysics", string(config.component.component_id())),
        ("moving_frame", JsonValue::Bool(false)),
        ("psolver", number(3)),
        ("equation_set", number(2)),
        ("terrain", JsonValue::Bool(true)),
    ])
}

fn parse_configuration(value: &JsonValue) -> Result<Gm1CheckpointConfig> {
    exact_keys(
        value,
        &["admission", "capture_domain", "selection"],
        "GM1 configuration",
    )?;
    let admission =
        Gm1CheckpointAdmission::parse(required_string(value, "admission", "GM1 configuration")?)?;
    let domain = value
        .get("capture_domain")
        .map_err(|error| invalid(format!("GM1 capture domain: {error}")))?;
    exact_keys(
        domain,
        &["crop_origin_yx", "crop_shape_zyx", "source_shape_zyx"],
        "GM1 capture domain",
    )?;
    let origin = usize_array::<2>(domain, "crop_origin_yx", "GM1 capture domain")?;
    let shape = usize_array::<3>(domain, "crop_shape_zyx", "GM1 capture domain")?;
    if shape[0] != GM1_REAL_NZ {
        return Err(invalid("GM1 checkpoint vertical extent changed"));
    }
    // The selection block already names the column component, so a restored
    // checkpoint recovers its own scalar axis from the artifact rather than
    // assuming one; the whole block is then re-derived and compared byte for
    // byte by the caller, so a mismatched pair cannot pass.
    let selection = value
        .get("selection")
        .map_err(|error| invalid(format!("GM1 selection: {error}")))?;
    let component =
        Gm1CheckpointComponent::parse(required_string(selection, "microphysics", "GM1 selection")?)?;
    let config = Gm1CheckpointConfig {
        j_start: origin[0],
        i_start: origin[1],
        ny: shape[1],
        nx: shape[2],
        admission,
        component,
    };
    config.validate()?;
    Ok(config)
}

fn source_json(source: &Gm1SourceIdentities) -> JsonValue {
    object(vec![
        (
            "geopotential_manifest_sha256",
            string(&source.geopotential_manifest_sha256),
        ),
        (
            "geopotential_payload_sha256",
            string(&source.geopotential_payload_sha256),
        ),
        (
            "geopotential_receipt_sha256",
            string(&source.geopotential_receipt_sha256),
        ),
        (
            "geopotential_source_restart_sha256",
            string(&source.geopotential_source_restart_sha256),
        ),
        (
            "product_manifest_sha256",
            string(&source.product_manifest_sha256),
        ),
        (
            "product_payload_sha256",
            string(&source.product_payload_sha256),
        ),
        (
            "product_source_artifact_sha256",
            string(&source.product_source_artifact_sha256),
        ),
        (
            "product_source_contract_sha256",
            string(&source.product_source_contract_sha256),
        ),
    ])
}

fn parse_source(value: &JsonValue) -> Result<Gm1SourceIdentities> {
    exact_keys(
        value,
        &[
            "geopotential_manifest_sha256",
            "geopotential_payload_sha256",
            "geopotential_receipt_sha256",
            "geopotential_source_restart_sha256",
            "product_manifest_sha256",
            "product_payload_sha256",
            "product_source_artifact_sha256",
            "product_source_contract_sha256",
        ],
        "GM1 source",
    )?;
    Gm1SourceIdentities::new(
        required_string(value, "product_manifest_sha256", "GM1 source")?,
        required_string(value, "product_payload_sha256", "GM1 source")?,
        required_string(value, "product_source_artifact_sha256", "GM1 source")?,
        required_string(value, "product_source_contract_sha256", "GM1 source")?,
        required_string(value, "geopotential_manifest_sha256", "GM1 source")?,
        required_string(value, "geopotential_payload_sha256", "GM1 source")?,
        required_string(value, "geopotential_receipt_sha256", "GM1 source")?,
        required_string(value, "geopotential_source_restart_sha256", "GM1 source")?,
    )
}

fn authority_json() -> JsonValue {
    object(vec![
        ("model", string("CM1")),
        ("release", string("r21.1")),
        (
            "source_sha256",
            JsonValue::Object(
                AUTHORITY_SOURCES
                    .into_iter()
                    .map(|(name, hash)| (name.to_string(), string(hash)))
                    .collect(),
            ),
        ),
        ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
    ])
}

fn execution_identity_json(operator_manifest_sha256: &str) -> JsonValue {
    object(vec![
        ("authority", authority_json()),
        ("carrier_abi", string(GM1_CHECKPOINT_CARRIER_ABI)),
        ("model", string(GM1_CHECKPOINT_MODEL)),
        ("operator_manifest_sha256", string(operator_manifest_sha256)),
        ("transition_contracts", transition_contracts_json()),
    ])
}

fn transition_contracts_json() -> JsonValue {
    object(vec![
        (
            "initialized_product_carrier_abi",
            string(INITIALIZED_PRODUCT_CARRIER_ABI),
        ),
        ("wsm6_bridge_abi", string(WSM6_TRANSITION_BRIDGE_ABI)),
        ("wsm6_carrier_abi", string(WSM6_TRANSITION_CARRIER_ABI)),
        ("wsm6_component_abi", string(WSM6_TRANSITION_COMPONENT_ABI)),
        (
            "wsm6_geopotential_carrier_abi",
            string(WSM6_GEOPOTENTIAL_CARRIER_ABI),
        ),
        (
            "wsm6_in_memory_restart_schema",
            string(WSM6_TRANSITION_IN_MEMORY_RESTART_SCHEMA),
        ),
        ("wsm6_transition_abi", string(WSM6_TRANSITION_ABI)),
    ])
}

fn aliases_json() -> JsonValue {
    JsonValue::Array(
        GM1_WSM6_ALIASES
            .into_iter()
            .map(|(alias, canonical)| {
                object(vec![
                    ("alias", string(alias)),
                    ("canonical", string(canonical)),
                    (
                        "relation",
                        string("same live allocation and exact byte span"),
                    ),
                ])
            })
            .collect(),
    )
}

fn restore_into_directory(
    restored: &RestoredGm1Checkpoint,
    directory: &Path,
) -> Result<Gm1RestoreReceipt> {
    let payload_path = directory.join(GM1_RESTORE_ARENA_FILE);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&payload_path)?;
    let mut writer = BufWriter::new(file);
    let mut aggregate = Sha256::default();
    let mut offset = 0u64;
    let mut carriers = Vec::new();
    for (spec, array) in restored
        .contract
        .arrays()
        .iter()
        .zip(restored.forecast.frame().arrays())
    {
        if spec.name() != array.name() {
            return Err(invalid("GM1 restore array order changed"));
        }
        let mut field = Sha256::default();
        let mut bytes = Vec::with_capacity(1024 * 1024);
        for chunk in array.values().chunks((1024 * 1024) / 4) {
            bytes.clear();
            for value in chunk {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            writer.write_all(&bytes)?;
            field.update(&bytes);
            aggregate.update(&bytes);
        }
        let nbytes = spec_nbytes(spec)?;
        carriers.push(RestoreCarrier {
            name: spec.name().to_string(),
            shape: spec.shape().to_vec(),
            offset_nbytes: offset,
            nbytes,
            sha256: sha256::hex(&field.finalize()),
        });
        offset = offset
            .checked_add(nbytes)
            .ok_or_else(|| invalid("GM1 restore offset overflows"))?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    let payload_sha256 = sha256::hex(&aggregate.finalize());
    if payload_sha256 != restored.raw_arena_sha256 || offset != array_bytes(&restored.contract)? {
        return Err(invalid(
            "GM1 restored arena differs from checkpoint identity",
        ));
    }
    let manifest = restore_manifest_json(restored, &carriers, offset, &payload_sha256);
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    write_new_synced(&directory.join(GM1_RESTORE_MANIFEST_FILE), &manifest_bytes)?;
    let receipt = Gm1RestoreReceipt {
        checkpoint_artifact_sha256: restored.forecast.receipt().artifact_sha256.clone(),
        checkpoint_contract_sha256: restored.contract.sha256(),
        manifest_sha256,
        manifest_nbytes: manifest_bytes.len(),
        payload_sha256,
        payload_nbytes: offset,
        carrier_count: carriers.len(),
        step: restored.forecast.frame().clock().step(),
    };
    write_new_synced(
        &directory.join(GM1_RESTORE_RECEIPT_FILE),
        receipt.canonical_json().as_bytes(),
    )?;
    let actual = fs::read_dir(directory)?
        .map(|entry| {
            entry
                .map(|value| value.file_name().to_string_lossy().into_owned())
                .map_err(StateError::from)
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let expected = [
        GM1_RESTORE_ARENA_FILE.to_string(),
        GM1_RESTORE_MANIFEST_FILE.to_string(),
        GM1_RESTORE_RECEIPT_FILE.to_string(),
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid("GM1 restore directory inventory changed"));
    }
    Ok(receipt)
}

fn restore_manifest_json(
    restored: &RestoredGm1Checkpoint,
    carriers: &[RestoreCarrier],
    payload_nbytes: u64,
    payload_sha256: &str,
) -> JsonValue {
    let scalars = restored
        .forecast
        .frame()
        .scheduler_scalars()
        .iter()
        .map(|scalar| {
            let value = match scalar.value() {
                SchedulerScalarValue::I64(value) => JsonValue::Number(value.to_string()),
                SchedulerScalarValue::U64(value) => JsonValue::Number(value.to_string()),
                SchedulerScalarValue::F64(value) => f64_bits(value),
                SchedulerScalarValue::Bool(value) => JsonValue::Bool(value),
            };
            (scalar.name().to_string(), value)
        })
        .collect();
    let extension = json::parse(&restored.scheduler_extension)
        .expect("validated GM1 scheduler extension parses");
    object(vec![
        ("aliases", aliases_json()),
        ("capture_phase", string(GM1_CHECKPOINT_CAPTURE_PHASE)),
        ("carrier_abi", string(GM1_CHECKPOINT_CARRIER_ABI)),
        (
            "carriers",
            JsonValue::Array(carriers.iter().map(RestoreCarrier::json).collect()),
        ),
        ("claim_limit", string(GM1_CHECKPOINT_CLAIM_LIMIT)),
        ("configuration", configuration_json(restored.config)),
        ("dtype", string("<f4")),
        ("itemsize", number(4)),
        ("model", string(GM1_CHECKPOINT_MODEL)),
        (
            "operator_manifest",
            json::parse(restored.operator_manifest.bytes()).expect("validated manifest"),
        ),
        ("order", string("C")),
        (
            "payload",
            object(vec![
                ("file", string(GM1_RESTORE_ARENA_FILE)),
                ("nbytes", u64_number(payload_nbytes)),
                ("sha256", string(payload_sha256)),
            ]),
        ),
        ("restore_phase", string(GM1_CHECKPOINT_RESTORE_PHASE)),
        ("scheduler_extension", extension),
        ("scheduler_scalars", JsonValue::Object(scalars)),
        ("schema", string(GM1_RESTORE_MANIFEST_SCHEMA)),
        ("source", source_json(&restored.source)),
        (
            "source_checkpoint",
            object(vec![
                (
                    "artifact_sha256",
                    string(&restored.forecast.receipt().artifact_sha256),
                ),
                ("contract_sha256", string(&restored.contract.sha256())),
                (
                    "dt_f64_bits",
                    f64_bits(restored.forecast.frame().clock().dt_s()),
                ),
                (
                    "model_time_f64_bits",
                    f64_bits(restored.forecast.frame().clock().model_time_s()),
                ),
                ("step", u64_number(restored.forecast.frame().clock().step())),
            ]),
        ),
        ("species_order", string_array_json(restored.config.component.species())),
    ])
}

fn required_blob<'a>(
    blobs: &BTreeMap<&str, &'a ComponentBlob>,
    name: &str,
    schema: &str,
) -> Result<&'a ComponentBlob> {
    let blob = blobs
        .get(name)
        .copied()
        .ok_or_else(|| invalid(format!("GM1 component blob {name} is missing")))?;
    if blob.schema() != schema {
        return Err(invalid(format!("GM1 component blob {name} schema changed")));
    }
    Ok(blob)
}

fn require_bits_equal(
    left: &[f32],
    right: &[f32],
    left_name: &str,
    right_name: &str,
) -> Result<()> {
    if left.len() != right.len() {
        return Err(invalid(format!(
            "GM1 safe-boundary pair {left_name}/{right_name} lengths differ"
        )));
    }
    if let Some(index) = left
        .iter()
        .zip(right)
        .position(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err(invalid(format!(
            "GM1 safe-boundary pair {left_name}/{right_name} differs at word {index}"
        )));
    }
    Ok(())
}

fn require_positive(values: &[f32], name: &str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| *value <= 0.0) {
        return Err(invalid(format!(
            "GM1 carrier {name} must be positive; word {index} is {:08x}",
            values[index].to_bits()
        )));
    }
    Ok(())
}

fn require_nonnegative(values: &[f32], name: &str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| *value < 0.0) {
        return Err(invalid(format!(
            "GM1 carrier {name} must be nonnegative; word {index} is {:08x}",
            values[index].to_bits()
        )));
    }
    Ok(())
}

fn array<'a>(map: &BTreeMap<&str, &'a [f32]>, name: &str) -> Result<&'a [f32]> {
    map.get(name)
        .copied()
        .ok_or_else(|| invalid(format!("GM1 carrier {name} is missing")))
}

fn arrays_sha256(arrays: &[ForecastArray]) -> String {
    let mut hasher = Sha256::default();
    let mut bytes = Vec::with_capacity(1024 * 1024);
    for array in arrays {
        for chunk in array.values().chunks((1024 * 1024) / 4) {
            bytes.clear();
            for value in chunk {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            hasher.update(&bytes);
        }
    }
    sha256::hex(&hasher.finalize())
}

fn array_bytes(contract: &ForecastContract) -> Result<u64> {
    contract.arrays().iter().try_fold(0u64, |total, spec| {
        total
            .checked_add(spec_nbytes(spec)?)
            .ok_or_else(|| invalid("GM1 arena byte total overflows"))
    })
}

fn spec_words(spec: &ForecastArraySpec) -> Result<usize> {
    spec.shape().iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("GM1 carrier {} shape overflows", spec.name())))
    })
}

fn spec_nbytes(spec: &ForecastArraySpec) -> Result<u64> {
    let bytes = spec_words(spec)?
        .checked_mul(4)
        .ok_or_else(|| invalid("GM1 carrier byte count overflows"))?;
    u64::try_from(bytes).map_err(|_| invalid("GM1 carrier byte count exceeds u64"))
}

fn one_bounds(shape: &[usize]) -> Result<Vec<i32>> {
    let mut bounds = Vec::with_capacity(shape.len() * 2);
    for extent in shape {
        bounds.push(1);
        bounds.push(i32::try_from(*extent).map_err(|_| invalid("GM1 extent does not fit i32"))?);
    }
    Ok(bounds)
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("GM1 restore destination needs a UTF-8 file name"))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| invalid(format!("system clock precedes epoch: {error}")))?
        .as_nanos();
    Ok(output.with_file_name(format!(".{name}.partial-{}-{nonce}", std::process::id())))
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) -> Result<()> {
    let actual = value
        .as_object()
        .map_err(|error| invalid(format!("{label}: {error}")))?
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(invalid(format!(
            "{label} keys are {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn fixed_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = required_string(value, key, label)?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn required_string<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn required_sha(value: &JsonValue, key: &str, label: &str) -> Result<String> {
    let result = required_string(value, key, label)?.to_string();
    validate_sha256(&result, &format!("{label}.{key}"))?;
    Ok(result)
}

fn usize_array<const N: usize>(value: &JsonValue, key: &str, label: &str) -> Result<[usize; N]> {
    let values = value
        .get(key)
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
    if values.len() != N {
        return Err(invalid(format!("{label}.{key} must contain {N} values")));
    }
    values
        .iter()
        .map(|item| {
            let value = item
                .as_u64()
                .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
            usize::try_from(value).map_err(|_| invalid(format!("{label}.{key} exceeds usize")))
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| invalid(format!("{label}.{key} length changed")))
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{label} must be 64 lower-case hexadecimal characters"
        )));
    }
    Ok(())
}

fn validate_extension_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        })
    {
        return Err(invalid(format!(
            "GM1 scheduler extension key {value:?} must use 1..128 lower-case ASCII [a-z0-9._-] characters"
        )));
    }
    Ok(())
}

fn string_array_json(values: &[&str]) -> JsonValue {
    JsonValue::Array(values.iter().copied().map(string).collect())
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
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

fn f64_bits(value: f64) -> JsonValue {
    string(&format!("{:016x}", value.to_bits()))
}

fn canonical_string(value: JsonValue) -> String {
    String::from_utf8(value.canonical_bytes()).expect("canonical GM1 JSON is UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "gm1-checkpoint-{label}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn operator_manifest() -> Gm1OperatorManifest {
        let operators = GM1_OPERATOR_SOURCE_KEYS
            .into_iter()
            .enumerate()
            .map(|(index, name)| (name.to_string(), string(&format!("{:064x}", index + 1))))
            .collect();
        let bytes = object(vec![
            ("model", string(GM1_CHECKPOINT_MODEL)),
            ("operators", JsonValue::Object(operators)),
            ("schema", string(GM1_OPERATOR_MANIFEST_SCHEMA)),
        ])
        .canonical_bytes();
        Gm1OperatorManifest::from_canonical_bytes(&bytes).unwrap()
    }

    fn extension() -> Vec<u8> {
        br#"{"entries":{"history.schedule_index":3},"schema":"gm1.scheduler-extension/v1"}"#
            .to_vec()
    }

    fn fixture(config: Gm1CheckpointConfig) -> Vec<ForecastArray> {
        let specs = gm1_array_specs(config).unwrap();
        let mut arrays = specs
            .iter()
            .map(|spec| {
                let fill = if [
                    "qa",
                    "q3d",
                    "closure.tkea",
                    "component.effc",
                    "component.effi",
                    "component.effs",
                    "component.mp_rainnc",
                    "component.mp_rainncv",
                    "component.mp_snownc",
                    "component.mp_snowncv",
                    "component.mp_graupelnc",
                    "component.mp_graupelncv",
                    "component.mp_sr",
                ]
                .contains(&spec.name())
                {
                    0.001
                } else {
                    1.0
                };
                ForecastArray::new(spec.name(), vec![fill; spec_words(spec).unwrap()])
            })
            .collect::<Vec<_>>();
        let qpten = arrays
            .iter_mut()
            .find(|array| array.name() == "held.qpten")
            .unwrap();
        qpten.values_mut()[0] = f32::from_bits(0x8000_0000);
        qpten.values_mut()[1] = f32::from_bits(0x0000_0001);
        arrays
    }

    fn arena_bytes(arrays: &[ForecastArray]) -> Vec<u8> {
        arrays
            .iter()
            .flat_map(|array| array.values().iter())
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn production_layout_has_canonical_deduplicated_inventory() {
        let operator = operator_manifest();
        assert_eq!(GM1_OPERATOR_SOURCE_KEYS.len(), 22);
        assert!(GM1_OPERATOR_SOURCE_KEYS.contains(&"physics.wsm6-pdefq"));
        let contract = gm1_contract(Gm1CheckpointConfig::production(), &operator).unwrap();
        assert_eq!(contract.arrays().len(), GM1_CHECKPOINT_ARRAY_COUNT);
        assert_eq!(
            array_bytes(&contract).unwrap(),
            GM1_CHECKPOINT_REAL_PAYLOAD_NBYTES
        );
        let names = contract
            .arrays()
            .iter()
            .map(ForecastArraySpec::name)
            .collect::<BTreeSet<_>>();
        assert!(GM1_WSM6_ALIASES
            .iter()
            .all(|(alias, canonical)| !names.contains(alias) && names.contains(canonical)));
        assert!(names.contains("closure.tkea"));
        assert!(!names.contains("closure.tke3d"));
        assert!(!names.contains("closure.tketen"));
        // The v1 profile is WSM6's: six kg/kg species and no rain effective
        // radius or cloud-droplet number carrier.
        assert_eq!(Gm1CheckpointComponent::Wsm6.profile_id(), GM1_CHECKPOINT_PROFILE_ID);
        assert_eq!(Gm1CheckpointComponent::Wsm6.species(), &SPECIES);
        assert_eq!(Gm1CheckpointComponent::Wsm6.array_count(), GM1_CHECKPOINT_ARRAY_COUNT);
        assert!(!names.contains("component.effr"));
        assert!(!names.contains("component.nc"));
    }

    #[test]
    fn morrison_inventory_carries_ten_scalars_and_its_own_profile() {
        let operator = operator_manifest();
        let config = Gm1CheckpointConfig::production()
            .with_component(Gm1CheckpointComponent::Morrison);
        let contract = gm1_contract(config, &operator).unwrap();
        // 17 physical + 4 radiation + 17 held/component + closure.tkea.
        assert_eq!(contract.arrays().len(), 39);
        assert_eq!(Gm1CheckpointComponent::Morrison.array_count(), 39);
        assert_eq!(Gm1CheckpointComponent::Morrison.cm1_ptype(), 5);
        assert_ne!(
            Gm1CheckpointComponent::Morrison.profile_id(),
            GM1_CHECKPOINT_PROFILE_ID
        );
        // src/param.F:3446-3449: CM1's own number-concentration order.
        assert_eq!(
            Gm1CheckpointComponent::Morrison.species(),
            &["qv", "qc", "qr", "qi", "qs", "qg", "nci", "ncs", "ncr", "ncg"]
        );
        let names = contract
            .arrays()
            .iter()
            .map(ForecastArraySpec::name)
            .collect::<BTreeSet<_>>();
        // Every WSM6 carrier is still here; the double-moment set only adds.
        for name in WSM6_UNIQUE_NAMES {
            assert!(names.contains(name), "{name} missing from the Morrison set");
        }
        assert!(names.contains("component.effr"));
        assert!(names.contains("component.nc"));
        for name in ["qa", "q3d"] {
            let spec = contract
                .arrays()
                .iter()
                .find(|spec| spec.name() == name)
                .unwrap();
            assert_eq!(spec.shape()[0], 10, "{name} must carry CM1's numq=10 axis");
        }
        // A double-moment arena is larger than the sealed WSM6 v1 arena, and
        // the seal is not applied to a profile that never declared it.
        assert!(array_bytes(&contract).unwrap() > GM1_CHECKPOINT_REAL_PAYLOAD_NBYTES);
    }

    #[test]
    fn bounded_roundtrip_preserves_every_raw_word_and_metadata() {
        let root = TestRoot::new("roundtrip");
        let config = Gm1CheckpointConfig::corrected_real_crop(140, 141, 2, 3).unwrap();
        let arrays = fixture(config);
        validate_array_semantics(&arrays).unwrap();
        let raw = arena_bytes(&arrays);
        let raw_sha = sha256::digest_hex(&raw);
        let raw_path = root.0.join("arena.f32le");
        fs::write(&raw_path, &raw).unwrap();
        let output = root.0.join("state.cm1fcp");
        let operator = operator_manifest();
        let operator_sha = operator.sha256().to_string();
        let receipt = write_gm1_checkpoint_new(
            &raw_path,
            &output,
            &raw_sha,
            config,
            Gm1SourceIdentities::sealed(),
            operator,
            ForecastClock::new(3, 1.5, 0.5).unwrap(),
            12345.25,
            3,
            &extension(),
        )
        .unwrap();
        let restored = read_gm1_checkpoint(
            &output,
            &receipt.forecast.contract_sha256,
            &receipt.forecast.artifact_sha256,
            GM1_PRODUCT_MANIFEST_SHA256,
            &operator_sha,
        )
        .unwrap();
        assert_eq!(restored.raw_arena_sha256(), raw_sha);
        assert_eq!(restored.mass1_f64().to_bits(), 12345.25f64.to_bits());
        assert_eq!(restored.completed_transitions(), 3);
        assert_eq!(restored.scheduler_extension(), extension());
        let restore = root.0.join("restored");
        let restore_receipt = restore_gm1_checkpoint_new(&restored, &restore).unwrap();
        assert_eq!(restore_receipt.payload_sha256, raw_sha);
        assert_eq!(fs::read(restore.join(GM1_RESTORE_ARENA_FILE)).unwrap(), raw);
    }

    #[test]
    fn refuses_safe_boundary_pair_drift_before_publication() {
        let config = Gm1CheckpointConfig::corrected_real_crop(0, 0, 1, 1).unwrap();
        let mut arrays = fixture(config);
        arrays
            .iter_mut()
            .find(|array| array.name() == "u3d")
            .unwrap()
            .values_mut()[0] = 2.0;
        let error = validate_array_semantics(&arrays).unwrap_err().to_string();
        assert!(error.contains("ua/u3d"), "{error}");
    }

    #[test]
    fn refuses_operator_inventory_and_scheduler_collision() {
        let bad = br#"{"model":"GM1","operators":{},"schema":"gm1.operator-source-manifest/v1"}"#;
        assert!(Gm1OperatorManifest::from_canonical_bytes(bad)
            .unwrap_err()
            .to_string()
            .contains("inventory"));
        let without_pdefq = GM1_OPERATOR_SOURCE_KEYS
            .into_iter()
            .filter(|name| *name != "physics.wsm6-pdefq")
            .enumerate()
            .map(|(index, name)| (name.to_owned(), string(&format!("{:064x}", index + 1))))
            .collect();
        let without_pdefq = object(vec![
            ("model", string(GM1_CHECKPOINT_MODEL)),
            ("operators", JsonValue::Object(without_pdefq)),
            ("schema", string(GM1_OPERATOR_MANIFEST_SCHEMA)),
        ])
        .canonical_bytes();
        assert!(Gm1OperatorManifest::from_canonical_bytes(&without_pdefq)
            .unwrap_err()
            .to_string()
            .contains("physics.wsm6-pdefq"));
        let collision = br#"{"entries":{"next_stage":1},"schema":"gm1.scheduler-extension/v1"}"#;
        assert!(validate_scheduler_extension(collision)
            .unwrap_err()
            .to_string()
            .contains("collides"));
    }

    #[test]
    fn refuses_clock_and_source_drift() {
        assert!(
            validate_boundary_clock(ForecastClock::new(2, 0.5, 0.5).unwrap(), 2)
                .unwrap_err()
                .to_string()
                .contains("model time")
        );
        assert!(Gm1SourceIdentities::new(
            "11".repeat(32),
            GM1_PRODUCT_PAYLOAD_SHA256,
            GM1_PRODUCT_SOURCE_ARTIFACT_SHA256,
            GM1_PRODUCT_SOURCE_CONTRACT_SHA256,
            GM1_GEOPOTENTIAL_MANIFEST_SHA256,
            GM1_GEOPOTENTIAL_PAYLOAD_SHA256,
            GM1_GEOPOTENTIAL_RECEIPT_SHA256,
            GM1_GEOPOTENTIAL_SOURCE_RESTART_SHA256,
        )
        .unwrap_err()
        .to_string()
        .contains("sealed"));
    }

    #[test]
    fn f02_roi_source_identity_roundtrips_and_mixed_tuple_refuses() {
        let f02 = Gm1SourceIdentities::sealed_f02_roi();
        let bytes = f02.canonical_bytes();
        assert_eq!(
            Gm1SourceIdentities::from_canonical_bytes(&bytes).unwrap(),
            f02
        );

        let mut mixed = Gm1SourceIdentities::sealed_f02_roi();
        mixed.geopotential_source_restart_sha256 =
            GM1_GEOPOTENTIAL_SOURCE_RESTART_SHA256.to_owned();
        assert!(Gm1SourceIdentities::new(
            mixed.product_manifest_sha256,
            mixed.product_payload_sha256,
            mixed.product_source_artifact_sha256,
            mixed.product_source_contract_sha256,
            mixed.geopotential_manifest_sha256,
            mixed.geopotential_payload_sha256,
            mixed.geopotential_receipt_sha256,
            mixed.geopotential_source_restart_sha256,
        )
        .unwrap_err()
        .to_string()
        .contains("neither"));
    }

    #[test]
    fn f02_roi_checkpoint_outer_source_seal_is_restorable() {
        let root = TestRoot::new("f02-roi-source");
        let config = Gm1CheckpointConfig::corrected_real_crop(140, 141, 1, 2).unwrap();
        let raw = arena_bytes(&fixture(config));
        let raw_sha = sha256::digest_hex(&raw);
        let raw_path = root.0.join("arena.f32le");
        fs::write(&raw_path, &raw).unwrap();
        let output = root.0.join("state.cm1fcp");
        let operator = operator_manifest();
        let operator_sha = operator.sha256().to_owned();
        let receipt = write_gm1_checkpoint_new(
            &raw_path,
            &output,
            &raw_sha,
            config,
            Gm1SourceIdentities::sealed_f02_roi(),
            operator,
            ForecastClock::new(120, 60.0, 0.5).unwrap(),
            12_345.25,
            120,
            &extension(),
        )
        .unwrap();
        let restored = read_gm1_checkpoint(
            &output,
            &receipt.forecast.contract_sha256,
            &receipt.forecast.artifact_sha256,
            GM1_F02_ROI_PRODUCT_MANIFEST_SHA256,
            &operator_sha,
        )
        .unwrap();
        assert_eq!(restored.source(), &Gm1SourceIdentities::sealed_f02_roi());
    }
}
