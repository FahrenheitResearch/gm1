//! Strict handoff contract for a real evolved parent to a refined GM1 child.
//!
//! This is a new-child initialization, not a restart.  Horizontal locations
//! are selected in the parent's source coordinate system and each native
//! stagger is retained.
//!
//! The parent is admitted by the properties a child handoff actually needs,
//! not by the identity of one historical run:
//!
//!  * the checkpoint sits on a completed large-step boundary, so the child
//!    inherits state some whole `nstep` produced rather than a mid-RK stage
//!    (CM1 r21.1 `src/cm1.F:3171` increments `nstep` at the top of the step
//!    and `src/cm1.F:3510` advances `mtime` by one `dbldt` at the bottom,
//!    with the source comment "do not change mtime anywhere else");
//!  * the recorded clock satisfies that same relation, so the step and the
//!    model time written by `src/restart_write.F:364` and
//!    `src/restart_write.F:403` cannot disagree; and
//!  * the moisture state is pdefq-clean before the child copies it.
//!
//! Any model time that satisfies those properties is admitted.  The first
//! admitted parent (step 600, 300.0 s) is retained below as lineage evidence
//! only; it is not a gate.

use crate::domain_grid_checkpoint::{build_stretch_z1, StretchZ1Grid, StretchZ1Parameters};
use crate::geopotential::{
    WSM6_GEOPOTENTIAL_CARRIER_ABI, WSM6_GEOPOTENTIAL_CARRIER_NAME,
    WSM6_GEOPOTENTIAL_RECEIPT_SCHEMA, WSM6_GEOPOTENTIAL_SCHEMA,
};
use crate::gm1_checkpoint::{
    read_gm1_checkpoint, Gm1CheckpointConfig, Gm1SourceIdentities,
    GM1_F02_ROI_PRODUCT_MANIFEST_SHA256,
};
use crate::hrrr::Cm1TargetGrid;
use crate::json::{self, JsonValue};
use crate::sha256;
use crate::snapshot::atomic_write_new;
use crate::terrain_admission::{
    TERRAIN_OPEN_CARRIER_ABI, TERRAIN_OPEN_CARRIER_NAMES, TERRAIN_OPEN_IDENTITY_SCHEMA,
    TERRAIN_OPEN_RECEIPT_SCHEMA,
};
use crate::{Result, StateError};
use std::fs;
use std::path::Path;

pub const T300_PARENT_BINDING_SCHEMA: &str = "gm1.pdefq-clean-t300-parent-binding/v1";
pub const CHILD_REMAP_PREFLIGHT_SCHEMA: &str = "gm1.source-coordinate-50m-child-preflight/v1";
pub const CHILD_REMAP_PLAN_SCHEMA: &str = "gm1.source-coordinate-50m-child-remap-plan/v1";
pub const CHILD_REMAP_BUILDER_ABI: &str = "gm1.t300-child-remap-builder/v1";
pub const CHILD_PRODUCER_REQUIREMENTS_SCHEMA: &str =
    "gm1.source-coordinate-50m-child-producer-requirements/v1";
pub const CHILD_PRODUCER_GATE_ABI: &str = "gm1.t300-child-producer-gate/v1";
pub const CHILD_TERRAIN_STRETCH_SCHEMA: &str = "gpuwm-cm1.cuda-stretch-z1-terrain/v1";
pub const CHILD_TERRAIN_STRETCH_ABI: &str =
    "gpuwm-cm1.cuda-stretch-z1-terrain-physical-carriers-f32/v1";
/// The first parent plan and checkpoint this chain admitted.
///
/// These are lineage evidence for the historical 50 m child.  No production
/// publisher gates on them: every producer records the plan and checkpoint
/// identity it was actually bound to, and the composing full-state gate
/// requires those recorded identities to agree with one another.
///
/// They remain a gate in exactly one place, and only there:
/// [`crate::child_atmosphere::replay_sealed_child_atmosphere_raw_streaming`]
/// recomputes the one committed atmosphere arena byte for byte and compares it
/// against those committed bytes, so it must be handed the parent that
/// produced them.  Any other parent would recompute a different arena and the
/// comparison would report a mismatch that means nothing.
pub const FIRST_ADMITTED_PARENT_PLAN_SHA256: &str =
    "14609494f7c87937fcb5ec0db3f214809ed514380560c43beb18183e6eebce17";
pub const FIRST_ADMITTED_PARENT_CHECKPOINT_SHA256: &str =
    "4f3a00ae66e0b68f0f05f4eff54a3625e846c0efee31b3ecbdee01c3c87ac1b4";
pub const CM1_R21_AUTHORITY_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
pub const CHILD_REMAP_CLAIM_LIMIT: &str = "strict source-coordinate geometry, native-stagger support stencils, real pdefq-clean T+300 parent checkpoint admission, exact flat-only native/CPU/CUDA pins for the official 75-level/15-km stretch_z=1 deck, and explicit carrier dispositions for a fixed-terrain 50 m child initialization; numerical atmospheric remap remains blocked until terrain/stretch composition, hydrostatic rebalance, and child open-boundary initialization are sealed; this is not a restart and makes no trajectory, forecast-skill, tornado, or performance claim";
pub const CHILD_PRODUCER_REQUIREMENTS_CLAIM_LIMIT: &str = "exact source/configuration/ABI and carrier requirements for independently sealed terrain+stretch composition, terrain-varying hydrostatic/base-state reconstruction, fresh all-open child halo/radiation-history initialization, and terrain-following WSM6 geopotential; validates the accepted pdefq-clean T+300 parent plan but emits no numerical child carrier, admits no terrain+stretch composition, and makes no restart-identity, trajectory, forecast-skill, tornado, or performance claim";
pub const CHILD_STRETCH_SCHEMA: &str = "gpuwm-cm1.stretch-z1-child-15km/v1";
pub const CHILD_STRETCH_ABI: &str = "gpuwm-cm1.stretch-z1-child-15km-profiles-f32/v1";
pub const CHILD_STRETCH_GRID_SHA256: &str =
    "f66aa7937e0f4942dfbc55001641f221b8a84352961058e81247ee4e3ca8db56";
pub const CHILD_STRETCH_PROOF_SHA256: &str =
    "277f2b8831f316b2e99a81a903f5ac20ee5d781e2c2b058963539f648f9ec02e";
pub const CHILD_STRETCH_CUDA_RECEIPT_SHA256: &str =
    "ce75a4b2258b0b05b56cbc7834f22489c26defab285c29e08281994c6f453c0b";
pub const CHILD_STRETCH_SOURCE_SHA256: &str =
    "bf5f1a4f96333e5e68eaba6847841d035bec515db71db78e5228108b5676d3d0";
pub const CHILD_STRETCH_COMMIT: &str = "78fde76b95b8ca980947640e8bcd64a67ff84126";
pub const CHILD_STRETCH_ZF_SHA256: &str =
    "ceb1624e4dc10eb31562c7e8003f939d6d157e1ecd73d1dee479de5459f2a90e";
pub const CHILD_STRETCH_ZH_SHA256: &str =
    "e0c800cd95ac68f526efef46de91db5463784e91860f52ea90ad76c9e18037c3";
pub const CHILD_STRETCH_MH_SHA256: &str =
    "bc520e91ecab72cd55662dbb301c29f78cc04864ec371b3abba5dca8d8c6cbcc";
pub const CHILD_STRETCH_MF_SHA256: &str =
    "e686a95701fb4694da25878934e7da4713b512831a06adf549ef61e1704e245f";
pub const CHILD_STRETCH_RMH_SHA256: &str =
    "54833ee529867d7c005060a52c3d9e883bfe34731fd5fb74cdba15afbd039b35";
pub const CHILD_STRETCH_RMF_SHA256: &str =
    "ab5f395ab5c7726528eb7f7f4dd1946911d03537eb94ea9b38e1ed1dc8551515";

pub const PARENT_NX: usize = 300;
pub const PARENT_NY: usize = 300;
pub const PARENT_NZ: usize = 75;
pub const PARENT_DX_M: f64 = 100.0;
pub const PARENT_DY_M: f64 = 100.0;
pub const PARENT_ZTOP_M: f64 = 15_000.0;
pub const PARENT_CENTER_I: usize = 224;
pub const PARENT_CENTER_J: usize = 224;
pub const CHILD_NX: usize = 300;
pub const CHILD_NY: usize = 300;
pub const CHILD_NZ: usize = 75;
pub const CHILD_DX_M: f64 = 50.0;
pub const CHILD_DY_M: f64 = 50.0;
pub const CHILD_DOMAIN_X_M: f64 = 15_000.0;
pub const CHILD_DOMAIN_Y_M: f64 = 15_000.0;
pub const CHILD_ZTOP_M: f64 = 15_000.0;
pub const CHILD_DZ_BOT_M: f64 = 20.0;
pub const NOMINAL_PARENT_ORIGIN_I: usize = 149;
pub const NOMINAL_PARENT_ORIGIN_J: usize = 149;
pub const NOMINAL_PARENT_COUNT_X: usize = 150;
pub const NOMINAL_PARENT_COUNT_Y: usize = 150;
/// The step and model time of the first admitted parent, kept as lineage
/// evidence.  `ParentClock` admits any completed large-step boundary.
pub const FIRST_ADMITTED_PARENT_STEP: u64 = 600;
pub const FIRST_ADMITTED_PARENT_MODEL_TIME_S: f64 = 300.0;
pub const PARENT_DT_S: f64 = 0.5;
pub const CHILD_GRID_SPEC_ABI: &str = "gm1.source-coordinate-child-grid-spec/v1";

/// One validated horizontal child geometry inside the parent source domain.
///
/// The fields stay private so downstream producers can only receive one of the
/// audited constants or a value that passed [`ChildGridSpec::new`].
///
/// The horizontal extent, spacing and the parent cell the child is centred on
/// are free, and so is the parent grid the window is placed against.  What is
/// checked is the property that actually matters: every staggered child corner
/// must land strictly inside the parent's bilinear interpolation support, so no
/// child point is filled by extrapolating off the parent domain.
///
/// How far that genericity reaches today, stated exactly so no reader has to
/// find out by running into a refusal:
///
///  * the terrain producer follows it end to end - it reads the parent extent
///    and spacing out of the source product's own manifest, requires the window
///    to have been placed against that parent, and strides the surface by that
///    parent's row length (`child_terrain::transform_source_terrain_for_grid`);
///  * the atmospheric remap in `child_atmosphere` still indexes the parent
///    checkpoint's carriers with the compile-time `PARENT_NX`/`PARENT_NY`/
///    `PARENT_NZ` of the admitted real domain, and `product_restart` refuses a
///    real product published for any other source domain by value, naming the
///    resident arena sized for it.  A window placed against a different parent
///    is therefore constructible and usable for terrain, and is refused - with
///    its reason - before any atmospheric carrier is read.
///
/// The child's vertical deck is separately fixed at `CHILD_NZ` levels by the
/// refusal below, because every child carrier layout indexes that deck.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChildGridSpec {
    nx: usize,
    ny: usize,
    nz: usize,
    dx_m: f64,
    dy_m: f64,
    ztop_m: f64,
    parent_nx: usize,
    parent_ny: usize,
    parent_dx_m: f64,
    parent_dy_m: f64,
    parent_center_i: usize,
    parent_center_j: usize,
}

impl ChildGridSpec {
    /// Build a child grid centred on the default anchor of the sealed parent.
    pub fn new(nx: usize, ny: usize, nz: usize, dx_m: f64, dy_m: f64, ztop_m: f64) -> Result<Self> {
        Self::new_centered(
            nx,
            ny,
            nz,
            dx_m,
            dy_m,
            ztop_m,
            PARENT_CENTER_I,
            PARENT_CENTER_J,
        )
    }

    /// Build a child grid centred on an explicit mass point of the sealed
    /// parent geometry.
    #[allow(clippy::too_many_arguments)]
    pub fn new_centered(
        nx: usize,
        ny: usize,
        nz: usize,
        dx_m: f64,
        dy_m: f64,
        ztop_m: f64,
        parent_center_i: usize,
        parent_center_j: usize,
    ) -> Result<Self> {
        Self::new_in_parent(
            ParentHorizontalGrid::sealed(),
            nx,
            ny,
            nz,
            dx_m,
            dy_m,
            ztop_m,
            parent_center_i,
            parent_center_j,
        )
    }

    /// Build a child grid inside the horizontal grid the parent actually ran.
    ///
    /// Nothing here is frozen to one historical child: the parent extent and
    /// spacing come from the parent's own recorded geometry, and the child
    /// count, spacing and centre are free.  What is refused is a window whose
    /// staggered corners fall outside the parent's interpolation support,
    /// because every such child point would be filled by extrapolating off the
    /// parent domain instead of interpolating parent state.
    #[allow(clippy::too_many_arguments)]
    pub fn new_in_parent(
        parent: ParentHorizontalGrid,
        nx: usize,
        ny: usize,
        nz: usize,
        dx_m: f64,
        dy_m: f64,
        ztop_m: f64,
        parent_center_i: usize,
        parent_center_j: usize,
    ) -> Result<Self> {
        if nx < 2 || ny < 2 {
            return Err(invalid(
                "child grid requires at least 2x2 horizontal points: a single-column child has no horizontal stencil to fill",
            ));
        }
        if nz != CHILD_NZ {
            return Err(invalid(format!(
                "child grid level count must be the admitted {CHILD_NZ}-level stretched deck; the child atmosphere, auxiliary, terrain-open and post-step carrier layouts index that deck directly, so another level count would read and write past their arenas"
            )));
        }
        if !dx_m.is_finite()
            || !dy_m.is_finite()
            || dx_m <= 0.0
            || dy_m <= 0.0
            || ztop_m.to_bits() != CHILD_ZTOP_M.to_bits()
        {
            return Err(invalid(
                "child grid horizontal spacing must be finite and positive and the model top must be the admitted deck top",
            ));
        }
        if parent_center_i >= parent.nx || parent_center_j >= parent.ny {
            return Err(invalid(format!(
                "child grid centre ({parent_center_i},{parent_center_j}) is not a mass point of the {}x{} parent the checkpoint ran on: the child window would be anchored outside the parent domain and every child column would be filled from parent memory that does not exist",
                parent.nx, parent.ny
            )));
        }
        let extent_x = nx as f64 * dx_m;
        let extent_y = ny as f64 * dy_m;
        if !extent_x.is_finite() || !extent_y.is_finite() {
            return Err(invalid("child grid horizontal extent is not finite"));
        }
        let grid = Self {
            nx,
            ny,
            nz,
            dx_m,
            dy_m,
            ztop_m,
            parent_nx: parent.nx,
            parent_ny: parent.ny,
            parent_dx_m: parent.dx_m,
            parent_dy_m: parent.dy_m,
            parent_center_i,
            parent_center_j,
        };
        for stagger in [
            HorizontalStagger::Scalar,
            HorizontalStagger::XFace,
            HorizontalStagger::YFace,
            HorizontalStagger::ZFace,
            HorizontalStagger::Surface,
        ] {
            let (count_x, count_y) = stagger.child_shape_xy_for(grid);
            for (child_i, child_j) in [
                (0, 0),
                (count_x - 1, 0),
                (0, count_y - 1),
                (count_x - 1, count_y - 1),
            ] {
                let (parent_i, parent_j) =
                    parent_fractional_xy_validated(grid, stagger, child_i, child_j);
                let parent_x_extent = if stagger == HorizontalStagger::XFace {
                    parent.nx + 1
                } else {
                    parent.nx
                };
                let parent_y_extent = if stagger == HorizontalStagger::YFace {
                    parent.ny + 1
                } else {
                    parent.ny
                };
                // Every consumer fills a child point by bilinear interpolation
                // from parent indices i0 = floor(parent_i) and i0 + 1, so the
                // last parent point is not itself an admissible corner: a
                // corner there has no i0 + 1 to interpolate towards.  The guard
                // therefore requires the corner to be strictly inside the
                // support, which is exactly the property its message claims -
                // and is the same bound `child_terrain::
                // interpolate_source_terrain_point` enforces, so the refusal
                // fires here, by name, instead of deep inside the remap.
                if !parent_i.is_finite()
                    || !parent_j.is_finite()
                    || parent_i < 0.0
                    || parent_j < 0.0
                    || parent_i >= (parent_x_extent - 1) as f64
                    || parent_j >= (parent_y_extent - 1) as f64
                {
                    return Err(invalid(format!(
                        "child {} corner ({child_i},{child_j}) maps to parent fractional ({parent_i},{parent_j}), which is not strictly inside the {parent_x_extent}x{parent_y_extent} parent support: bilinear interpolation of that child point needs both the parent index below it and the one above, so it would be extrapolated off the parent domain rather than interpolated from it",
                        stagger.as_str()
                    )));
                }
            }
        }
        Ok(grid)
    }

    pub const fn nx(self) -> usize {
        self.nx
    }

    pub const fn ny(self) -> usize {
        self.ny
    }

    pub const fn nz(self) -> usize {
        self.nz
    }

    pub const fn dx_m(self) -> f64 {
        self.dx_m
    }

    pub const fn dy_m(self) -> f64 {
        self.dy_m
    }

    pub const fn ztop_m(self) -> f64 {
        self.ztop_m
    }

    pub const fn parent_nx(self) -> usize {
        self.parent_nx
    }

    pub const fn parent_ny(self) -> usize {
        self.parent_ny
    }

    pub const fn parent_dx_m(self) -> f64 {
        self.parent_dx_m
    }

    /// The horizontal grid this child window was placed against.
    pub fn parent_grid(self) -> ParentHorizontalGrid {
        ParentHorizontalGrid {
            nx: self.parent_nx,
            ny: self.parent_ny,
            dx_m: self.parent_dx_m,
            dy_m: self.parent_dy_m,
        }
    }

    pub const fn parent_dy_m(self) -> f64 {
        self.parent_dy_m
    }

    pub const fn parent_center_i(self) -> usize {
        self.parent_center_i
    }

    pub const fn parent_center_j(self) -> usize {
        self.parent_center_j
    }

    pub fn extent_x_m(self) -> f64 {
        self.nx as f64 * self.dx_m
    }

    pub fn extent_y_m(self) -> f64 {
        self.ny as f64 * self.dy_m
    }
}

/// The horizontal grid a parent checkpoint actually ran on.
///
/// The child is placed against this, not against a frozen constant, so a
/// parent published at a different extent or spacing moves the child window
/// with it instead of silently reindexing the same child into foreign memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParentHorizontalGrid {
    nx: usize,
    ny: usize,
    dx_m: f64,
    dy_m: f64,
}

impl ParentHorizontalGrid {
    pub const fn sealed() -> Self {
        Self {
            nx: PARENT_NX,
            ny: PARENT_NY,
            dx_m: PARENT_DX_M,
            dy_m: PARENT_DY_M,
        }
    }

    pub fn new(nx: usize, ny: usize, dx_m: f64, dy_m: f64) -> Result<Self> {
        if nx < 2 || ny < 2 {
            return Err(invalid(
                "parent horizontal grid needs at least 2x2 mass points: a child cannot interpolate from a parent with no horizontal stencil",
            ));
        }
        if !dx_m.is_finite() || !dy_m.is_finite() || dx_m <= 0.0 || dy_m <= 0.0 {
            return Err(invalid(
                "parent horizontal spacing must be finite and positive; a zero or non-finite spacing would divide every child-to-parent index mapping by zero",
            ));
        }
        Ok(Self { nx, ny, dx_m, dy_m })
    }

    pub const fn nx(self) -> usize {
        self.nx
    }

    pub const fn ny(self) -> usize {
        self.ny
    }

    pub const fn dx_m(self) -> f64 {
        self.dx_m
    }

    pub const fn dy_m(self) -> f64 {
        self.dy_m
    }
}

pub const ACCEPTED_50M_CHILD_GRID: ChildGridSpec = ChildGridSpec {
    nx: CHILD_NX,
    ny: CHILD_NY,
    nz: CHILD_NZ,
    dx_m: CHILD_DX_M,
    dy_m: CHILD_DY_M,
    ztop_m: CHILD_ZTOP_M,
    parent_nx: PARENT_NX,
    parent_ny: PARENT_NY,
    parent_dx_m: PARENT_DX_M,
    parent_dy_m: PARENT_DY_M,
    parent_center_i: PARENT_CENTER_I,
    parent_center_j: PARENT_CENTER_J,
};

pub const HARD_5M_CHILD_GRID: ChildGridSpec = ChildGridSpec {
    nx: 3_000,
    ny: 3_000,
    nz: CHILD_NZ,
    dx_m: 5.0,
    dy_m: 5.0,
    ztop_m: CHILD_ZTOP_M,
    parent_nx: PARENT_NX,
    parent_ny: PARENT_NY,
    parent_dx_m: PARENT_DX_M,
    parent_dy_m: PARENT_DY_M,
    parent_center_i: PARENT_CENTER_I,
    parent_center_j: PARENT_CENTER_J,
};

const PARENT_GRID_CENTER_LAT_BITS: u64 = 0x4044_b283_93b6_36d0;
const PARENT_GRID_CENTER_LON_BITS: u64 = 0xc057_9217_7a2c_3f9c;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HorizontalStagger {
    Scalar,
    XFace,
    YFace,
    ZFace,
    Surface,
}

impl HorizontalStagger {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::XFace => "x-face",
            Self::YFace => "y-face",
            Self::ZFace => "z-face",
            Self::Surface => "surface",
        }
    }

    const fn child_shape_xy_for(self, grid: ChildGridSpec) -> (usize, usize) {
        match self {
            Self::XFace => (grid.nx + 1, grid.ny),
            Self::YFace => (grid.nx, grid.ny + 1),
            Self::Scalar | Self::ZFace | Self::Surface => (grid.nx, grid.ny),
        }
    }
}

fn parent_fractional_xy_validated(
    grid: ChildGridSpec,
    stagger: HorizontalStagger,
    child_i: usize,
    child_j: usize,
) -> (f64, f64) {
    let mass_x = grid.parent_center_i as f64
        + (child_i as f64 + 0.5 - 0.5 * grid.nx as f64) * grid.dx_m / grid.parent_dx_m;
    let mass_y = grid.parent_center_j as f64
        + (child_j as f64 + 0.5 - 0.5 * grid.ny as f64) * grid.dy_m / grid.parent_dy_m;
    let xface_x = grid.parent_center_i as f64
        + 0.5
        + (child_i as f64 - 0.5 * grid.nx as f64) * grid.dx_m / grid.parent_dx_m;
    let yface_y = grid.parent_center_j as f64
        + 0.5
        + (child_j as f64 - 0.5 * grid.ny as f64) * grid.dy_m / grid.parent_dy_m;
    match stagger {
        HorizontalStagger::XFace => (xface_x, mass_y),
        HorizontalStagger::YFace => (mass_x, yface_y),
        HorizontalStagger::Scalar | HorizontalStagger::ZFace | HorizontalStagger::Surface => {
            (mass_x, mass_y)
        }
    }
}

/// Return parent fractional indices for one independently validated grid.
pub fn parent_fractional_xy_for_grid(
    grid: ChildGridSpec,
    stagger: HorizontalStagger,
    child_i: usize,
    child_j: usize,
) -> Result<(f64, f64)> {
    let (nx, ny) = stagger.child_shape_xy_for(grid);
    if child_i >= nx || child_j >= ny {
        return Err(invalid(format!(
            "child {} index ({child_i},{child_j}) exceeds ({nx},{ny})",
            stagger.as_str()
        )));
    }
    Ok(parent_fractional_xy_validated(
        grid, stagger, child_i, child_j,
    ))
}

/// Return the parent-grid fractional indices of a child point.
///
/// The child geometric center is exactly parent mass point (224,224).  This
/// keeps the 15 km child source-contained while retaining each native stagger.
pub fn parent_fractional_xy(
    stagger: HorizontalStagger,
    child_i: usize,
    child_j: usize,
) -> Result<(f64, f64)> {
    parent_fractional_xy_for_grid(ACCEPTED_50M_CHILD_GRID, stagger, child_i, child_j)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParentAnchor {
    pub i: usize,
    pub j: usize,
    pub value: f32,
}

/// The largest completed-step count this admission will replay.
///
/// At the sealed half-second real timestep this is more than nineteen model
/// days, far past any parent a child is spawned from.  The bound exists so a
/// corrupt `nstep` word cannot turn admission into an unbounded loop.
pub const MAX_ADMITTED_PARENT_STEP: u64 = 3_400_000;

/// Replay CM1's own model clock for `step` completed large steps.
///
/// CM1 r21.1 advances the model clock by exactly one `dbldt` per completed
/// large step (`src/cm1.F:3510`, whose own comment is "do not change mtime
/// anywhere else"), starting from zero for a fresh start.  The clock a
/// completed-step checkpoint must carry is therefore that binary64
/// accumulation, which is what this reproduces - not `step * dt`, which
/// differs from the accumulation whenever `dt` is not binary exact.
///
/// When `dt` is a power of two the accumulation is provably equal to the
/// product for every step count that stays exactly representable, so that case
/// is short-circuited instead of looped.
fn replay_model_time_s(step: u64, dt_s: f64) -> Result<f64> {
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(invalid(format!(
            "parent timestep {dt_s:?} is not finite and positive: a child cannot be dated against a clock that never advances"
        )));
    }
    if step > MAX_ADMITTED_PARENT_STEP {
        return Err(invalid(format!(
            "parent completed-step count {step} exceeds the admitted {MAX_ADMITTED_PARENT_STEP}: a step word that large is corrupt, and replaying CM1's model clock for it would not terminate in bounded time"
        )));
    }
    let mantissa = dt_s.to_bits() & 0x000f_ffff_ffff_ffff;
    let exponent = (dt_s.to_bits() >> 52) & 0x7ff;
    let power_of_two = mantissa == 0 && exponent != 0;
    let model_time_s = if power_of_two {
        step as f64 * dt_s
    } else {
        let mut accumulated = 0.0f64;
        for _ in 0..step {
            accumulated += dt_s;
        }
        accumulated
    };
    if !model_time_s.is_finite() {
        return Err(invalid(
            "parent model clock overflows binary64 before the recorded completed step",
        ));
    }
    Ok(model_time_s)
}

/// One completed large-step boundary of the parent run.
///
/// CM1 r21.1 increments `nstep` once per large step (`src/cm1.F:3171`) and
/// advances `mtime` by exactly one `dbldt` at the bottom of the same step
/// (`src/cm1.F:3510`, "do not change mtime anywhere else"); the restart writer
/// persists `nstep`, `dt` and `mtime` as three independent records
/// (`src/restart_write.F:364`, `src/restart_write.F:389` and
/// `src/restart_write.F:403`).  Those three must agree with CM1's own clock
/// accumulation, so a checkpoint whose `mtime` is not the replay of `nstep`
/// additions of `dt` was not taken at a completed large-step boundary; a child
/// built from it would inherit a mid-RK stage and be dated by a clock that
/// never produced it.
///
/// Every step that satisfies that relation is admitted.  No particular step,
/// model time or historical run is required.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParentClock {
    step: u64,
    model_time_s: f64,
    dt_s: f64,
}

impl ParentClock {
    /// Admit any completed large-step boundary at the sealed real timestep.
    pub fn completed_large_step(step: u64) -> Result<Self> {
        Self::completed_large_step_at(step, PARENT_DT_S)
    }

    /// Admit any completed large-step boundary at an explicit fixed timestep.
    pub fn completed_large_step_at(step: u64, dt_s: f64) -> Result<Self> {
        if step == 0 {
            return Err(invalid(
                "a child cannot be initialized from a step-zero parent: the handoff would copy the analysis the parent started from, not an evolved parent state",
            ));
        }
        let model_time_s = replay_model_time_s(step, dt_s)?;
        Ok(Self {
            step,
            model_time_s,
            dt_s,
        })
    }

    /// Admit a step/model-time/dt triple exactly as a checkpoint recorded it.
    pub fn from_recorded(step: u64, model_time_s: f64, dt_s: f64) -> Result<Self> {
        let derived = Self::completed_large_step_at(step, dt_s)?;
        if derived.model_time_s.to_bits() != model_time_s.to_bits() {
            return Err(invalid(format!(
                "parent clock records step {step} at dt {dt_s:?} with model time {model_time_s:?}; replaying CM1's own per-step clock advance gives {:?}, so this checkpoint was not written at a completed large-step boundary and the child would inherit a mid-RK stage carrying a time that never produced it",
                derived.model_time_s
            )));
        }
        Ok(derived)
    }

    pub const fn step(self) -> u64 {
        self.step
    }

    pub const fn model_time_s(self) -> f64 {
        self.model_time_s
    }

    pub const fn dt_s(self) -> f64 {
        self.dt_s
    }
}

/// Admit the parent identity a child producer is actually bound to.
///
/// The plan digest and the checkpoint digest are re-derived by
/// [`read_validated_parent_plan`] from the binding the plan itself carries, so
/// they are self-consistent by construction.  What this adds is that both are
/// well formed and therefore recordable: every child producer writes the pair
/// into its published manifest and receipt, and the composing full-state gate
/// refuses a child whose producers name different parents.  That mismatch is
/// the concrete wrong result this replaces a fixed-hash pin with - a child
/// whose atmosphere came from one parent run and whose terrain, auxiliary or
/// closure came from another, blended into one initialization that never
/// existed.
pub fn validate_bound_parent_identity(
    checkpoint_sha256: &str,
    plan_sha256: &str,
    label: &str,
) -> Result<()> {
    for (what, value) in [
        ("parent checkpoint", checkpoint_sha256),
        ("parent plan", plan_sha256),
    ] {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(invalid(format!(
                "{label} {what} identity is not a lower-case SHA-256; the published artifact could not be re-bound to the parent it was built from"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParentBinding {
    pub checkpoint_artifact_sha256: String,
    pub checkpoint_contract_sha256: String,
    pub checkpoint_raw_arena_sha256: String,
    pub operator_manifest_sha256: String,
    pub diagnostics_receipt_sha256: String,
    pub pdefq_receipt_sha256: String,
    pub clock: ParentClock,
    pub low_level_reflectivity: ParentAnchor,
    pub column_updraft: ParentAnchor,
    pub low_level_abs_vorticity: ParentAnchor,
}

impl ParentBinding {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        checkpoint_artifact_sha256: impl Into<String>,
        checkpoint_contract_sha256: impl Into<String>,
        checkpoint_raw_arena_sha256: impl Into<String>,
        operator_manifest_sha256: impl Into<String>,
        diagnostics_receipt_sha256: impl Into<String>,
        pdefq_receipt_sha256: impl Into<String>,
        clock: ParentClock,
        low_level_reflectivity: ParentAnchor,
        column_updraft: ParentAnchor,
        low_level_abs_vorticity: ParentAnchor,
    ) -> Result<Self> {
        let binding = Self {
            checkpoint_artifact_sha256: checkpoint_artifact_sha256.into(),
            checkpoint_contract_sha256: checkpoint_contract_sha256.into(),
            checkpoint_raw_arena_sha256: checkpoint_raw_arena_sha256.into(),
            operator_manifest_sha256: operator_manifest_sha256.into(),
            diagnostics_receipt_sha256: diagnostics_receipt_sha256.into(),
            pdefq_receipt_sha256: pdefq_receipt_sha256.into(),
            clock,
            low_level_reflectivity,
            column_updraft,
            low_level_abs_vorticity,
        };
        binding.validate()?;
        Ok(binding)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        parent_binding_json(self).canonical_bytes()
    }

    pub fn write_new(&self, path: impl AsRef<Path>) -> Result<()> {
        write_newline_json(path.as_ref(), &self.canonical_bytes())
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self> {
        let value =
            json::parse(bytes).map_err(|error| invalid(format!("T300 parent binding: {error}")))?;
        let binding = parse_parent_binding(&value)?;
        if value.canonical_bytes() != bytes {
            return Err(invalid("T300 parent binding is not canonical JSON"));
        }
        Ok(binding)
    }

    fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("checkpoint artifact", &self.checkpoint_artifact_sha256),
            ("checkpoint contract", &self.checkpoint_contract_sha256),
            ("checkpoint raw arena", &self.checkpoint_raw_arena_sha256),
            ("operator manifest", &self.operator_manifest_sha256),
            ("diagnostics receipt", &self.diagnostics_receipt_sha256),
            ("pdefq receipt", &self.pdefq_receipt_sha256),
        ] {
            validate_sha256(value, label)?;
        }
        for (label, anchor) in [
            ("low-level reflectivity", self.low_level_reflectivity),
            ("column updraft", self.column_updraft),
            ("low-level absolute vorticity", self.low_level_abs_vorticity),
        ] {
            validate_anchor(label, anchor)?;
        }
        Ok(())
    }
}

pub fn encode_geometry_preflight() -> Result<Vec<u8>> {
    let value = preflight_json(None)?;
    Ok(value.canonical_bytes())
}

pub fn write_geometry_preflight_new(path: impl AsRef<Path>) -> Result<()> {
    write_newline_json(path.as_ref(), &encode_geometry_preflight()?)
}

/// Validate an exact accepted-parent plan and emit the producer requirements
/// that must be satisfied before any numerical 50 m child is constructed.
///
/// This deliberately emits metadata only.  A PASS here does not admit the
/// terrain/stretch composition or any remapped atmospheric carrier.
pub fn encode_child_producer_requirements(
    validated_parent_plan_path: impl AsRef<Path>,
) -> Result<Vec<u8>> {
    let (binding, plan_sha256) = read_validated_parent_plan(validated_parent_plan_path.as_ref())?;
    Ok(child_producer_requirements_json(&binding, &plan_sha256).canonical_bytes())
}

pub fn write_child_producer_requirements_new(
    validated_parent_plan_path: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<()> {
    write_newline_json(
        output.as_ref(),
        &encode_child_producer_requirements(validated_parent_plan_path)?,
    )
}

pub fn sealed_child_stretch_grid() -> Result<StretchZ1Grid> {
    let grid = build_stretch_z1(StretchZ1Parameters {
        nk: CHILD_NZ,
        dz: 100.0,
        ztop: CHILD_ZTOP_M as f32,
        str_bot: 200.0,
        str_top: 7_000.0,
        dz_bot: CHILD_DZ_BOT_M as f32,
        dz_top: 320.0,
    })?;
    for (name, expected) in [
        ("zf", CHILD_STRETCH_ZF_SHA256),
        ("sigmaf", CHILD_STRETCH_ZF_SHA256),
        ("zh", CHILD_STRETCH_ZH_SHA256),
        ("sigma", CHILD_STRETCH_ZH_SHA256),
        ("mh", CHILD_STRETCH_MH_SHA256),
        ("mf", CHILD_STRETCH_MF_SHA256),
        ("rmh", CHILD_STRETCH_RMH_SHA256),
        ("rmf", CHILD_STRETCH_RMF_SHA256),
    ] {
        if grid.profile_sha256(name)? != expected {
            return Err(invalid(format!(
                "15 km child stretch profile {name} differs from its native/CPU pin"
            )));
        }
    }
    if grid.combined_sha256() != CHILD_STRETCH_GRID_SHA256
        || (grid.derived.nk1, grid.derived.nk2, grid.derived.nk3) != (10, 40, 25)
    {
        return Err(invalid(
            "15 km child stretch grid or segment counts differ from the native/CPU pin",
        ));
    }
    Ok(grid)
}

pub fn validate_parent_and_encode_plan(
    binding: &ParentBinding,
    checkpoint_path: impl AsRef<Path>,
    diagnostics_receipt_path: impl AsRef<Path>,
    pdefq_receipt_path: impl AsRef<Path>,
) -> Result<Vec<u8>> {
    binding.validate()?;
    let diagnostics_receipt = read_bound_json_receipt(
        diagnostics_receipt_path.as_ref(),
        &binding.diagnostics_receipt_sha256,
        "T300 diagnostics receipt",
    )?;
    let pdefq_receipt = read_bound_json_receipt(
        pdefq_receipt_path.as_ref(),
        &binding.pdefq_receipt_sha256,
        "T300 pdefq receipt",
    )?;
    validate_t300_diagnostics_receipt(&diagnostics_receipt, binding)?;
    validate_pdefq_receipt(&pdefq_receipt)?;
    let restored = read_gm1_checkpoint(
        checkpoint_path,
        &binding.checkpoint_contract_sha256,
        &binding.checkpoint_artifact_sha256,
        GM1_F02_ROI_PRODUCT_MANIFEST_SHA256,
        &binding.operator_manifest_sha256,
    )?;
    if restored.config() != Gm1CheckpointConfig::production() {
        return Err(invalid(
            "T300 child source is not the full 300x300 parent checkpoint",
        ));
    }
    if restored.source() != &Gm1SourceIdentities::sealed_f02_roi() {
        return Err(invalid(
            "T300 child source is not sealed to the real f02 ROI product",
        ));
    }
    if restored.raw_arena_sha256() != binding.checkpoint_raw_arena_sha256 {
        return Err(invalid(
            "T300 parent raw arena differs from its independent binding",
        ));
    }
    let clock = restored.forecast().frame().clock();
    // Admit the checkpoint by the property a child handoff needs - that the
    // state was produced by some whole number of completed large steps - and
    // then require the independent binding to name that same boundary.  A
    // binding that names a different step would attach this checkpoint's
    // carriers to another run's diagnostics, pdefq receipt and anchors.
    if clock.dt_s().to_bits() != PARENT_DT_S.to_bits() {
        return Err(invalid(format!(
            "parent checkpoint advances at dt {:?}, not the sealed GM1 real timestep {PARENT_DT_S:?}; the child's own acoustic and RK cadence is derived from that timestep, so it would integrate state advanced by a step this chain never admitted",
            clock.dt_s()
        )));
    }
    let recorded = ParentClock::from_recorded(clock.step(), clock.model_time_s(), clock.dt_s())?;
    if recorded != binding.clock {
        return Err(invalid(format!(
            "parent checkpoint is at completed step {} (model time {} s) but the independent binding names step {} (model time {} s)",
            recorded.step(),
            recorded.model_time_s(),
            binding.clock.step(),
            binding.clock.model_time_s()
        )));
    }
    Ok(preflight_json(Some(binding))?.canonical_bytes())
}

pub fn write_validated_parent_plan_new(
    binding: &ParentBinding,
    checkpoint_path: impl AsRef<Path>,
    diagnostics_receipt_path: impl AsRef<Path>,
    pdefq_receipt_path: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<()> {
    let bytes = validate_parent_and_encode_plan(
        binding,
        checkpoint_path,
        diagnostics_receipt_path,
        pdefq_receipt_path,
    )?;
    write_newline_json(output.as_ref(), &bytes)
}

fn preflight_json(binding: Option<&ParentBinding>) -> Result<JsonValue> {
    let stretch = sealed_child_stretch_grid()?;
    let parent_grid = Cm1TargetGrid::new(
        PARENT_NX,
        PARENT_NY,
        PARENT_NZ,
        PARENT_DX_M,
        PARENT_DY_M,
        200.0,
        f64::from_bits(PARENT_GRID_CENTER_LAT_BITS),
        f64::from_bits(PARENT_GRID_CENTER_LON_BITS),
    )?;
    let (child_center_lat, child_center_lon) =
        parent_grid.mass_point_lat_lon(PARENT_CENTER_I, PARENT_CENTER_J)?;
    let child_grid = Cm1TargetGrid::new(
        CHILD_NX,
        CHILD_NY,
        CHILD_NZ,
        CHILD_DX_M,
        CHILD_DY_M,
        200.0,
        child_center_lat,
        child_center_lon,
    )?;
    let (southwest_lat, southwest_lon) = child_grid.mass_point_lat_lon(0, 0)?;
    let (northeast_lat, northeast_lon) =
        child_grid.mass_point_lat_lon(CHILD_NX - 1, CHILD_NY - 1)?;
    let (schema, status, parent) = if let Some(binding) = binding {
        (
            CHILD_REMAP_PLAN_SCHEMA,
            "PASS_PARENT_CHECKPOINT_AWAITING_CHILD_PRODUCERS",
            parent_binding_json(binding),
        )
    } else {
        (
            CHILD_REMAP_PREFLIGHT_SCHEMA,
            "PREPARED_AWAITING_PDEFQ_CLEAN_T300_AND_TERRAIN_STRETCH_ADMISSION",
            object(vec![
                ("required_binding_schema", string(T300_PARENT_BINDING_SCHEMA)),
                ("required_dt_s_f64_bits", f64_bits(PARENT_DT_S)),
                ("required_minimum_step", u64_number(1)),
                (
                    "required_step_time_relation",
                    string("model_time_s == the binary64 replay of CM1's own clock: mtime advanced by exactly one dbldt per completed large step, nstep incremented once per step (CM1 r21.1 src/cm1.F:3171 and src/cm1.F:3510, \"do not change mtime anywhere else\"), persisted as three independent restart records (src/restart_write.F:364, :389, :403). That accumulation equals step * dt_s exactly when dt_s is a power of two and differs from it otherwise, so the product is not the contract."),
                ),
            ]),
        )
    };
    Ok(object(vec![
        (
            "authority",
            object(vec![
                ("model", string("GM1")),
                ("model_source", string("CM1 r21.1")),
                (
                    "source_selection",
                    string("real f02 ROI evolved parent; no synthetic data or field translation"),
                ),
            ]),
        ),
        ("builder_abi", string(CHILD_REMAP_BUILDER_ABI)),
        (
            "carrier_disposition",
            object(vec![
                (
                    "child_boundary_initialize",
                    string_array(&["radbce", "radbcn", "radbcs", "radbcw"]),
                ),
                (
                    "horizontal_and_physical_height_remap_after_vertical_admission",
                    string_array(&[
                        "closure.tkea",
                        "component.effc",
                        "component.effi",
                        "component.effs",
                        "component.h_diabatic",
                        "held.qcten",
                        "held.qpten",
                        "held.qtten",
                        "held.qvten",
                        "pp3d",
                        "ppi",
                        "ppx",
                        "q3d",
                        "qa",
                        "th3d",
                        "tha",
                        "u3d",
                        "ua",
                        "v3d",
                        "va",
                        "w3d",
                        "wa",
                    ]),
                ),
                (
                    "horizontal_surface_remap",
                    string_array(&[
                        "component.mp_graupelnc",
                        "component.mp_graupelncv",
                        "component.mp_rainnc",
                        "component.mp_rainncv",
                        "component.mp_snownc",
                        "component.mp_snowncv",
                        "component.mp_sr",
                    ]),
                ),
                (
                    "recompute_after_remap",
                    string_array(&["prs", "rf", "rho", "rr"]),
                ),
                ("source_carrier_count", number(37)),
            ]),
        ),
        ("claim_limit", string(CHILD_REMAP_CLAIM_LIMIT)),
        (
            "geometry",
            object(vec![
                (
                    "child",
                    object(vec![
                        ("center_lat_f64_bits", f64_bits(child_center_lat)),
                        ("center_lon_f64_bits", f64_bits(child_center_lon)),
                        ("dx_m_f64_bits", f64_bits(CHILD_DX_M)),
                        ("dy_m_f64_bits", f64_bits(CHILD_DY_M)),
                        ("northeast_mass_lat_f64_bits", f64_bits(northeast_lat)),
                        ("northeast_mass_lon_f64_bits", f64_bits(northeast_lon)),
                        ("nx", number(CHILD_NX)),
                        ("ny", number(CHILD_NY)),
                        ("nz", number(CHILD_NZ)),
                        ("southwest_mass_lat_f64_bits", f64_bits(southwest_lat)),
                        ("southwest_mass_lon_f64_bits", f64_bits(southwest_lon)),
                    ]),
                ),
                (
                    "parent",
                    object(vec![
                        ("center_lat_f64_bits", f64_bits(parent_grid.center_lat_deg)),
                        ("center_lon_f64_bits", f64_bits(parent_grid.center_lon_deg)),
                        ("dx_m_f64_bits", f64_bits(PARENT_DX_M)),
                        ("dy_m_f64_bits", f64_bits(PARENT_DY_M)),
                        ("nx", number(PARENT_NX)),
                        ("ny", number(PARENT_NY)),
                        ("nz", number(PARENT_NZ)),
                    ]),
                ),
                (
                    "selection",
                    object(vec![
                        ("child_center_parent_mass_i", number(PARENT_CENTER_I)),
                        ("child_center_parent_mass_j", number(PARENT_CENTER_J)),
                        ("nominal_parent_count_x", number(NOMINAL_PARENT_COUNT_X)),
                        ("nominal_parent_count_y", number(NOMINAL_PARENT_COUNT_Y)),
                        ("nominal_parent_origin_i", number(NOMINAL_PARENT_ORIGIN_I)),
                        ("nominal_parent_origin_j", number(NOMINAL_PARENT_ORIGIN_J)),
                        ("synthetic_translation", JsonValue::Bool(false)),
                    ]),
                ),
            ]),
        ),
        (
            "horizontal_mapping",
            object(vec![
                (
                    "scalar_surface_zface",
                    object(vec![
                        ("formula_i", string("149.25 + 0.5*child_i")),
                        ("formula_j", string("149.25 + 0.5*child_j")),
                        ("required_parent_i", string("149..299")),
                        ("required_parent_j", string("149..299")),
                    ]),
                ),
                (
                    "x_face",
                    object(vec![
                        ("formula_i", string("149.5 + 0.5*child_i")),
                        ("formula_j", string("149.25 + 0.5*child_j")),
                        ("required_parent_i", string("149..300")),
                        ("required_parent_j", string("149..299")),
                    ]),
                ),
                (
                    "y_face",
                    object(vec![
                        ("formula_i", string("149.25 + 0.5*child_i")),
                        ("formula_j", string("149.5 + 0.5*child_j")),
                        ("required_parent_i", string("149..299")),
                        ("required_parent_j", string("149..300")),
                    ]),
                ),
            ]),
        ),
        ("model", string("GM1")),
        ("parent", parent),
        ("schema", string(schema)),
        ("status", string(status)),
        (
            "terrain_domain",
            object(vec![
                ("imove", number(0)),
                ("mode", string("fixed")),
                ("terrain", JsonValue::Bool(true)),
                (
                    "terrain_static_policy",
                    string("coordinate-remap source terrain then rebuild child terrain metrics"),
                ),
            ]),
        ),
        (
            "vertical_gate",
            object(vec![
                ("abi", string(CHILD_STRETCH_ABI)),
                (
                    "derived",
                    object(vec![
                        ("nk1", number(stretch.derived.nk1)),
                        ("nk2", number(stretch.derived.nk2)),
                        ("nk3", number(stretch.derived.nk3)),
                    ]),
                ),
                ("extrapolation_allowed", JsonValue::Bool(false)),
                (
                    "hashes",
                    object(vec![
                        ("combined_grid_sha256", string(CHILD_STRETCH_GRID_SHA256)),
                        ("mf_sha256", string(CHILD_STRETCH_MF_SHA256)),
                        ("mh_sha256", string(CHILD_STRETCH_MH_SHA256)),
                        ("rmf_sha256", string(CHILD_STRETCH_RMF_SHA256)),
                        ("rmh_sha256", string(CHILD_STRETCH_RMH_SHA256)),
                        ("sigma_sha256", string(CHILD_STRETCH_ZH_SHA256)),
                        ("sigmaf_sha256", string(CHILD_STRETCH_ZF_SHA256)),
                        ("zf_sha256", string(CHILD_STRETCH_ZF_SHA256)),
                        ("zh_sha256", string(CHILD_STRETCH_ZH_SHA256)),
                    ]),
                ),
                (
                    "parameters",
                    object(vec![
                        ("dz_bot_f32_bits", f32_bits(CHILD_DZ_BOT_M as f32)),
                        ("dz_f32_bits", f32_bits(100.0)),
                        ("dz_top_f32_bits", f32_bits(320.0)),
                        ("nk", number(CHILD_NZ)),
                        ("str_bot_f32_bits", f32_bits(200.0)),
                        ("str_top_f32_bits", f32_bits(7_000.0)),
                        ("ztop_f32_bits", f32_bits(CHILD_ZTOP_M as f32)),
                    ]),
                ),
                (
                    "proof",
                    object(vec![
                        ("commit", string(CHILD_STRETCH_COMMIT)),
                        (
                            "cuda_receipt_sha256",
                            string(CHILD_STRETCH_CUDA_RECEIPT_SHA256),
                        ),
                        ("schema", string("gpuwm-cm1.stretch-z1-metric-proof/v1")),
                        ("sha256", string(CHILD_STRETCH_PROOF_SHA256)),
                        ("source_sha256", string(CHILD_STRETCH_SOURCE_SHA256)),
                        ("status", string("PASS_NATIVE_CPU_CUDA_RAW_BIT_EXACT")),
                    ]),
                ),
                ("schema", string(CHILD_STRETCH_SCHEMA)),
                (
                    "status",
                    string("PINNED_FLAT_CPU_CUDA_BLOCKED_ON_TERRAIN_COMPOSITION"),
                ),
            ]),
        ),
    ]))
}

fn child_producer_requirements_json(
    binding: &ParentBinding,
    validated_parent_plan_sha256: &str,
) -> JsonValue {
    object(vec![
        (
            "authority",
            object(vec![
                ("model", string("GM1")),
                ("model_source", string("CM1 r21.1")),
                ("tarball_sha256", string(CM1_R21_AUTHORITY_SHA256)),
                (
                    "files",
                    JsonValue::Array(vec![
                        authority_file(
                            "src/init_terrain.F",
                            "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c",
                            "175-297",
                        ),
                        authority_file(
                            "src/param.F",
                            "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
                            "7789-7812,8162-8207,8464-8492,8988-9062",
                        ),
                        authority_file(
                            "src/base.F",
                            "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef",
                            "625-749,2060-2243,3209",
                        ),
                        authority_file(
                            "src/init3d.F",
                            "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28",
                            "1733-1785,2280-2467",
                        ),
                        authority_file(
                            "src/bc.F",
                            "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
                            "12-172,180-1206,1948-2172",
                        ),
                        authority_file(
                            "src/cm1.F",
                            "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4",
                            "747-775,884-923,1381-1411,1559-1657",
                        ),
                    ]),
                ),
            ]),
        ),
        ("claim_limit", string(CHILD_PRODUCER_REQUIREMENTS_CLAIM_LIMIT)),
        (
            "child_identity",
            object(vec![
                ("dx_m_f64_bits", f64_bits(CHILD_DX_M)),
                ("dy_m_f64_bits", f64_bits(CHILD_DY_M)),
                ("imove", number(0)),
                ("nx", number(CHILD_NX)),
                ("ny", number(CHILD_NY)),
                ("nz", number(CHILD_NZ)),
                ("synthetic_translation", JsonValue::Bool(false)),
                ("terrain", JsonValue::Bool(true)),
                ("ztop_m_f64_bits", f64_bits(CHILD_ZTOP_M)),
            ]),
        ),
        ("gate_abi", string(CHILD_PRODUCER_GATE_ABI)),
        ("model", string("GM1")),
        ("numerical_child_output_allowed", JsonValue::Bool(false)),
        (
            "parent_plan",
            object(vec![
                (
                    "checkpoint_artifact_sha256",
                    string(&binding.checkpoint_artifact_sha256),
                ),
                (
                    "diagnostics_receipt_sha256",
                    string(&binding.diagnostics_receipt_sha256),
                ),
                ("pdefq_receipt_sha256", string(&binding.pdefq_receipt_sha256)),
                ("schema", string(CHILD_REMAP_PLAN_SCHEMA)),
                ("sha256", string(validated_parent_plan_sha256)),
            ]),
        ),
        (
            "producer_requirements",
            object(vec![
                (
                    "geopotential",
                    object(vec![
                        ("carrier_abi", string(WSM6_GEOPOTENTIAL_CARRIER_ABI)),
                        ("carrier_name", string(WSM6_GEOPOTENTIAL_CARRIER_NAME)),
                        (
                            "must_bind",
                            string_array(&[
                                "validated_parent_plan_sha256",
                                "child_initialized_product_manifest_sha256",
                                "child_initialized_product_payload_sha256",
                                "terrain_stretch_manifest_sha256",
                                "terrain_stretch_payload_sha256",
                            ]),
                        ),
                        (
                            "requirements",
                            string_array(&[
                                "rebuild from exact three-dimensional child terrain-following zf",
                                "one binary32 multiplication by the sealed consumer gravity per C-order zf word",
                                "do not reuse the parent geopotential artifact",
                            ]),
                        ),
                        ("receipt_schema", string(WSM6_GEOPOTENTIAL_RECEIPT_SCHEMA)),
                        ("schema", string(WSM6_GEOPOTENTIAL_SCHEMA)),
                        ("status", string("REQUIRED_NOT_YET_BOUND")),
                    ]),
                ),
                (
                    "hydrostatic_and_open_boundary",
                    object(vec![
                        ("carrier_abi", string(TERRAIN_OPEN_CARRIER_ABI)),
                        ("identity_schema", string(TERRAIN_OPEN_IDENTITY_SCHEMA)),
                        (
                            "must_bind",
                            string_array(&[
                                "validated_parent_plan_sha256",
                                "child_initialized_product_restart_artifact_sha256",
                                "terrain_stretch_manifest_sha256",
                                "terrain_stretch_payload_sha256",
                                "terrain_stretch_receipt_sha256",
                            ]),
                        ),
                        (
                            "open_boundary_invariants",
                            object(vec![
                                ("all_four_boundaries_open", JsonValue::Bool(true)),
                                ("cold_start_history_nonzero_words", number(0)),
                                (
                                    "parent_radiation_history_reuse_allowed",
                                    JsonValue::Bool(false),
                                ),
                                (
                                    "source_valid_child_halos_required",
                                    JsonValue::Bool(true),
                                ),
                            ]),
                        ),
                        (
                            "required_carriers",
                            string_array(&TERRAIN_OPEN_CARRIER_NAMES),
                        ),
                        (
                            "required_diagnostics",
                            string_array(&[
                                "all_carriers_finite",
                                "pi_recovery",
                                "theta_recovery",
                                "hydrostatic_recurrence",
                                "radiative_history_nonzero_words",
                            ]),
                        ),
                        ("receipt_schema", string(TERRAIN_OPEN_RECEIPT_SCHEMA)),
                        ("status", string("REQUIRED_NOT_YET_BOUND")),
                    ]),
                ),
                (
                    "terrain_stretch_composition",
                    object(vec![
                        ("abi", string(CHILD_TERRAIN_STRETCH_ABI)),
                        (
                            "configuration",
                            object(vec![
                                ("hadvordrs", number(5)),
                                ("hadvordrv", number(5)),
                                ("imove", number(0)),
                                ("serial_iopen_e", number(2)),
                                ("serial_iopen_n", number(2)),
                                ("serial_iopen_s", number(2)),
                                ("serial_iopen_w", number(2)),
                            ]),
                        ),
                        ("flat_grid_sha256", string(CHILD_STRETCH_GRID_SHA256)),
                        (
                            "must_bind",
                            string_array(&[
                                "validated_parent_plan_sha256",
                                "source_coordinate_child_terrain_sha256",
                                "flat_stretch_grid_sha256",
                                "producer_source_sha256",
                                "manifest_sha256",
                                "payload_sha256",
                                "receipt_sha256",
                            ]),
                        ),
                        (
                            "required_carriers",
                            string_array(&[
                                "terrain_msl",
                                "zh",
                                "zf",
                                "mh",
                                "rmh",
                                "mf",
                                "rmf",
                                "c1",
                                "c2",
                                "gz",
                                "rgz",
                                "gzu",
                                "rgzu",
                                "gzv",
                                "rgzv",
                                "dzdx",
                                "dzdy",
                                "gx",
                                "gxu",
                                "gy",
                                "gyv",
                                "rds",
                                "rdsf",
                            ]),
                        ),
                        ("schema", string(CHILD_TERRAIN_STRETCH_SCHEMA)),
                        ("status", string("REQUIRED_NOT_YET_BOUND")),
                    ]),
                ),
            ]),
        ),
        ("schema", string(CHILD_PRODUCER_REQUIREMENTS_SCHEMA)),
        (
            "status",
            string("PASS_PARENT_AND_REQUIREMENTS_BLOCKED_ON_NUMERICAL_PRODUCERS"),
        ),
    ])
}

fn authority_file(path: &str, sha256_value: &str, anchors: &str) -> JsonValue {
    object(vec![
        ("anchors", string(anchors)),
        ("path", string(path)),
        ("sha256", string(sha256_value)),
    ])
}

fn parent_binding_json(binding: &ParentBinding) -> JsonValue {
    object(vec![
        (
            "checkpoint",
            object(vec![
                (
                    "artifact_sha256",
                    string(&binding.checkpoint_artifact_sha256),
                ),
                (
                    "contract_sha256",
                    string(&binding.checkpoint_contract_sha256),
                ),
                (
                    "operator_manifest_sha256",
                    string(&binding.operator_manifest_sha256),
                ),
                (
                    "product_manifest_sha256",
                    string(GM1_F02_ROI_PRODUCT_MANIFEST_SHA256),
                ),
                (
                    "raw_arena_sha256",
                    string(&binding.checkpoint_raw_arena_sha256),
                ),
            ]),
        ),
        (
            "clock",
            object(vec![
                ("dt_f64_bits", f64_bits(binding.clock.dt_s())),
                (
                    "model_time_f64_bits",
                    f64_bits(binding.clock.model_time_s()),
                ),
                ("step", u64_number(binding.clock.step())),
            ]),
        ),
        (
            "diagnostics",
            object(vec![
                ("column_updraft", anchor_json(binding.column_updraft)),
                (
                    "low_level_abs_vorticity",
                    anchor_json(binding.low_level_abs_vorticity),
                ),
                (
                    "low_level_reflectivity",
                    anchor_json(binding.low_level_reflectivity),
                ),
                (
                    "receipt_sha256",
                    string(&binding.diagnostics_receipt_sha256),
                ),
            ]),
        ),
        ("model", string("GM1")),
        (
            "pdefq",
            object(vec![
                ("qa_q3d_nonnegative", JsonValue::Bool(true)),
                ("qa_q3d_raw_bit_equal", JsonValue::Bool(true)),
                ("receipt_sha256", string(&binding.pdefq_receipt_sha256)),
                (
                    "timing",
                    string("q3d repaired before WSM6; solve3 copies q3d to qa"),
                ),
            ]),
        ),
        ("schema", string(T300_PARENT_BINDING_SCHEMA)),
        ("status", string("PASS_PDEFQ_CLEAN_T300")),
    ])
}

/// Map a parent mass point into the child's fractional mass-point indices.
///
/// This is the exact inverse of [`parent_fractional_xy_validated`] for the
/// scalar stagger, so a diagnostic anchor recorded on the parent and the
/// support stencil the child remap will read are the same mapping.
pub fn child_fractional_xy_for_grid(
    grid: ChildGridSpec,
    parent_i: usize,
    parent_j: usize,
) -> (f64, f64) {
    let child_i = (parent_i as f64 - grid.parent_center_i() as f64) * grid.parent_dx_m()
        / grid.dx_m()
        + 0.5 * grid.nx() as f64
        - 0.5;
    let child_j = (parent_j as f64 - grid.parent_center_j() as f64) * grid.parent_dy_m()
        / grid.dy_m()
        + 0.5 * grid.ny() as f64
        - 0.5;
    (child_i, child_j)
}

fn anchor_json_for_grid(grid: ChildGridSpec, anchor: ParentAnchor) -> JsonValue {
    let (child_i, child_j) = child_fractional_xy_for_grid(grid, anchor.i, anchor.j);
    let nearest_i = child_i.round() as usize;
    let nearest_j = child_j.round() as usize;
    let edge = nearest_i
        .min(nearest_j)
        .min(grid.nx() - 1 - nearest_i)
        .min(grid.ny() - 1 - nearest_j);
    object(vec![
        ("child_fractional_i_f64_bits", f64_bits(child_i)),
        ("child_fractional_j_f64_bits", f64_bits(child_j)),
        ("child_nearest_edge_cells", number(edge)),
        ("child_nearest_i", number(nearest_i)),
        ("child_nearest_j", number(nearest_j)),
        ("parent_i", number(anchor.i)),
        ("parent_j", number(anchor.j)),
        ("value_f32_bits", f32_bits(anchor.value)),
    ])
}

fn anchor_json(anchor: ParentAnchor) -> JsonValue {
    anchor_json_for_grid(ACCEPTED_50M_CHILD_GRID, anchor)
}

pub fn read_parent_binding(path: impl AsRef<Path>) -> Result<ParentBinding> {
    const MAX_BINDING_BYTES: u64 = 64 * 1024;
    let path = path.as_ref();
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_BINDING_BYTES {
        return Err(invalid("T300 parent binding is not a bounded regular file"));
    }
    let bytes = fs::read(path)?;
    let canonical = bytes
        .strip_suffix(b"\n")
        .ok_or_else(|| invalid("T300 parent binding file must end in one LF"))?;
    if canonical.ends_with(b"\r") || canonical.ends_with(b"\n") {
        return Err(invalid("T300 parent binding file is not canonical LF JSON"));
    }
    ParentBinding::from_canonical_bytes(canonical)
}

pub(crate) fn read_validated_parent_plan(path: &Path) -> Result<(ParentBinding, String)> {
    const MAX_PLAN_BYTES: u64 = 128 * 1024;
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_PLAN_BYTES {
        return Err(invalid(
            "validated T300 parent plan is not a bounded regular file",
        ));
    }
    let bytes = fs::read(path)?;
    let canonical = bytes
        .strip_suffix(b"\n")
        .ok_or_else(|| invalid("validated T300 parent plan must end in one LF"))?;
    if canonical.ends_with(b"\r") || canonical.ends_with(b"\n") {
        return Err(invalid(
            "validated T300 parent plan is not canonical LF JSON",
        ));
    }
    let value = json::parse(canonical)
        .map_err(|error| invalid(format!("validated T300 parent plan: {error}")))?;
    if value.canonical_bytes() != canonical {
        return Err(invalid("validated T300 parent plan is not canonical JSON"));
    }
    exact_keys(
        &value,
        &[
            "authority",
            "builder_abi",
            "carrier_disposition",
            "claim_limit",
            "geometry",
            "horizontal_mapping",
            "model",
            "parent",
            "schema",
            "status",
            "terrain_domain",
            "vertical_gate",
        ],
        "validated T300 parent plan",
    )?;
    fixed_string(
        &value,
        "schema",
        CHILD_REMAP_PLAN_SCHEMA,
        "validated T300 parent plan",
    )?;
    fixed_string(
        &value,
        "builder_abi",
        CHILD_REMAP_BUILDER_ABI,
        "validated T300 parent plan",
    )?;
    fixed_string(&value, "model", "GM1", "validated T300 parent plan")?;
    fixed_string(
        &value,
        "status",
        "PASS_PARENT_CHECKPOINT_AWAITING_CHILD_PRODUCERS",
        "validated T300 parent plan",
    )?;
    let binding = parse_parent_binding(member(&value, "parent", "validated T300 parent plan")?)?;
    if value != preflight_json(Some(&binding))? {
        return Err(invalid(
            "validated T300 parent plan differs from the exact Rust geometry, grid, or carrier contract",
        ));
    }
    Ok((binding, sha256::digest_hex(&bytes)))
}

fn parse_parent_binding(value: &JsonValue) -> Result<ParentBinding> {
    exact_keys(
        value,
        &[
            "checkpoint",
            "clock",
            "diagnostics",
            "model",
            "pdefq",
            "schema",
            "status",
        ],
        "T300 parent binding",
    )?;
    fixed_string(
        value,
        "schema",
        T300_PARENT_BINDING_SCHEMA,
        "T300 parent binding",
    )?;
    fixed_string(value, "model", "GM1", "T300 parent binding")?;
    fixed_string(
        value,
        "status",
        "PASS_PDEFQ_CLEAN_T300",
        "T300 parent binding",
    )?;
    let checkpoint = member(value, "checkpoint", "T300 parent binding")?;
    exact_keys(
        checkpoint,
        &[
            "artifact_sha256",
            "contract_sha256",
            "operator_manifest_sha256",
            "product_manifest_sha256",
            "raw_arena_sha256",
        ],
        "T300 parent checkpoint",
    )?;
    fixed_string(
        checkpoint,
        "product_manifest_sha256",
        GM1_F02_ROI_PRODUCT_MANIFEST_SHA256,
        "T300 parent checkpoint",
    )?;
    let clock = member(value, "clock", "T300 parent binding")?;
    exact_keys(
        clock,
        &["dt_f64_bits", "model_time_f64_bits", "step"],
        "T300 parent clock",
    )?;
    fixed_string(
        clock,
        "dt_f64_bits",
        &format!("{:016x}", PARENT_DT_S.to_bits()),
        "T300 parent clock",
    )?;
    // The clock is replayed against the timestep this document itself records,
    // not against a constant: the check above already requires the two to
    // agree, so reading the recorded value keeps the replay and the published
    // document describing the same run rather than two things that happen to
    // match today.
    let parent_clock = ParentClock::from_recorded(
        u64_member(clock, "step", "parent clock")?,
        f64_member(clock, "model_time_f64_bits", "parent clock")?,
        f64_member(clock, "dt_f64_bits", "parent clock")?,
    )?;
    let diagnostics = member(value, "diagnostics", "T300 parent binding")?;
    exact_keys(
        diagnostics,
        &[
            "column_updraft",
            "low_level_abs_vorticity",
            "low_level_reflectivity",
            "receipt_sha256",
        ],
        "T300 diagnostics",
    )?;
    let pdefq = member(value, "pdefq", "T300 parent binding")?;
    exact_keys(
        pdefq,
        &[
            "qa_q3d_nonnegative",
            "qa_q3d_raw_bit_equal",
            "receipt_sha256",
            "timing",
        ],
        "T300 pdefq",
    )?;
    if !bool_member(pdefq, "qa_q3d_nonnegative", "T300 pdefq")?
        || !bool_member(pdefq, "qa_q3d_raw_bit_equal", "T300 pdefq")?
    {
        return Err(invalid(
            "T300 pdefq binding does not assert the safe-boundary moisture state",
        ));
    }
    fixed_string(
        pdefq,
        "timing",
        "q3d repaired before WSM6; solve3 copies q3d to qa",
        "T300 pdefq",
    )?;
    ParentBinding::new(
        string_member(checkpoint, "artifact_sha256", "T300 parent checkpoint")?,
        string_member(checkpoint, "contract_sha256", "T300 parent checkpoint")?,
        string_member(checkpoint, "raw_arena_sha256", "T300 parent checkpoint")?,
        string_member(
            checkpoint,
            "operator_manifest_sha256",
            "T300 parent checkpoint",
        )?,
        string_member(diagnostics, "receipt_sha256", "T300 diagnostics")?,
        string_member(pdefq, "receipt_sha256", "T300 pdefq")?,
        parent_clock,
        parse_anchor(member(
            diagnostics,
            "low_level_reflectivity",
            "T300 diagnostics",
        )?)?,
        parse_anchor(member(diagnostics, "column_updraft", "T300 diagnostics")?)?,
        parse_anchor(member(
            diagnostics,
            "low_level_abs_vorticity",
            "T300 diagnostics",
        )?)?,
    )
}

fn parse_anchor(value: &JsonValue) -> Result<ParentAnchor> {
    exact_keys(
        value,
        &[
            "child_fractional_i_f64_bits",
            "child_fractional_j_f64_bits",
            "child_nearest_edge_cells",
            "child_nearest_i",
            "child_nearest_j",
            "parent_i",
            "parent_j",
            "value_f32_bits",
        ],
        "T300 diagnostic anchor",
    )?;
    let anchor = ParentAnchor {
        i: usize_member(value, "parent_i", "T300 diagnostic anchor")?,
        j: usize_member(value, "parent_j", "T300 diagnostic anchor")?,
        value: f32_member(value, "value_f32_bits", "T300 diagnostic anchor")?,
    };
    if value != &anchor_json(anchor) {
        return Err(invalid("T300 diagnostic child-coordinate mapping drifted"));
    }
    Ok(anchor)
}

fn validate_anchor(label: &str, anchor: ParentAnchor) -> Result<()> {
    if anchor.i >= PARENT_NX || anchor.j >= PARENT_NY || !anchor.value.is_finite() {
        return Err(invalid(format!(
            "{label} anchor is outside the finite parent domain"
        )));
    }
    if (label == "low-level reflectivity" || label == "low-level absolute vorticity")
        && anchor.value.is_sign_negative()
    {
        return Err(invalid(format!("{label} diagnostic must be nonnegative")));
    }
    if label == "column updraft" && anchor.value <= 0.0 {
        return Err(invalid("column updraft diagnostic must be positive"));
    }
    let grid = ACCEPTED_50M_CHILD_GRID;
    let (child_i, child_j) = child_fractional_xy_for_grid(grid, anchor.i, anchor.j);
    if !(0.0..=(grid.nx() - 1) as f64).contains(&child_i)
        || !(0.0..=(grid.ny() - 1) as f64).contains(&child_j)
    {
        return Err(invalid(format!(
            "{label} anchor is outside the nominal child window"
        )));
    }
    let nearest_i = child_i.round() as usize;
    let nearest_j = child_j.round() as usize;
    let edge = nearest_i
        .min(nearest_j)
        .min(grid.nx() - 1 - nearest_i)
        .min(grid.ny() - 1 - nearest_j);
    if edge < 32 {
        return Err(invalid(format!(
            "{label} anchor is not safely interior to the child"
        )));
    }
    Ok(())
}

fn write_newline_json(path: &Path, canonical: &[u8]) -> Result<()> {
    let mut bytes = canonical.to_vec();
    bytes.push(b'\n');
    atomic_write_new(path, &bytes)
}

fn read_bound_json_receipt(path: &Path, expected_sha256: &str, label: &str) -> Result<JsonValue> {
    const MAX_RECEIPT_BYTES: u64 = 4 * 1024 * 1024;
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_RECEIPT_BYTES {
        return Err(invalid(format!("{label} is not a bounded regular file")));
    }
    let bytes = fs::read(path)?;
    let actual = sha256::digest_hex(&bytes);
    if actual != expected_sha256 {
        return Err(invalid(format!(
            "{label} SHA-256 is {actual}; independently bound {expected_sha256}"
        )));
    }
    json::parse(&bytes).map_err(|error| invalid(format!("{label} is not strict JSON: {error}")))
}

fn validate_t300_diagnostics_receipt(receipt: &JsonValue, binding: &ParentBinding) -> Result<()> {
    fixed_string(
        receipt,
        "schema",
        "gm1.real-f02-roi-parent-gate/v1",
        "T300 diagnostics receipt",
    )?;
    fixed_string(
        receipt,
        "status",
        "PASS_REAL_F02_ROI_PARENT",
        "T300 diagnostics receipt",
    )?;
    if !bool_member(receipt, "moisture_nonnegative", "T300 diagnostics receipt")? {
        return Err(invalid(
            "T300 diagnostics receipt does not assert nonnegative moisture",
        ));
    }
    let candidate = member(receipt, "child_50m_candidate", "T300 diagnostics receipt")?;
    fixed_string(
        candidate,
        "status",
        "CANDIDATE_REQUIRES_RUST_SOURCE_COORDINATE_REMAP",
        "T300 child candidate",
    )?;
    if bool_member(candidate, "native_moving_domain", "T300 child candidate")? {
        return Err(invalid(
            "T300 child candidate unexpectedly asserts a moving domain",
        ));
    }
    require_usize_array(
        member(candidate, "child_shape_zyx", "T300 child candidate")?,
        &[PARENT_NZ, CHILD_NY, CHILD_NX],
        "T300 child shape",
    )?;
    require_f64_bits_array(
        member(candidate, "child_spacing_m", "T300 child candidate")?,
        &[CHILD_DX_M, CHILD_DY_M, 200.0],
        "T300 child spacing",
    )?;
    require_usize_array(
        member(candidate, "parent_crop_origin_ij", "T300 child candidate")?,
        &[NOMINAL_PARENT_ORIGIN_I, NOMINAL_PARENT_ORIGIN_J],
        "T300 child parent-crop origin",
    )?;
    require_usize_array(
        member(candidate, "parent_crop_count_ij", "T300 child candidate")?,
        &[NOMINAL_PARENT_COUNT_X, NOMINAL_PARENT_COUNT_Y],
        "T300 child parent-crop count",
    )?;
    let anchors = member(candidate, "anchors", "T300 child candidate")?;
    validate_receipt_anchor(
        member(anchors, "low_level_reflectivity_dbz", "T300 child anchors")?,
        binding.low_level_reflectivity,
        "T300 low-level reflectivity anchor",
    )?;
    validate_receipt_anchor(
        member(anchors, "column_updraft_ms", "T300 child anchors")?,
        binding.column_updraft,
        "T300 column updraft anchor",
    )?;
    validate_receipt_anchor(
        member(
            anchors,
            "low_level_abs_vertical_vorticity_s",
            "T300 child anchors",
        )?,
        binding.low_level_abs_vorticity,
        "T300 low-level absolute vorticity anchor",
    )?;
    let pdefq = member(receipt, "pre_wsm6_pdefq", "T300 diagnostics receipt")?;
    if !bool_member(pdefq, "cadence_admitted", "T300 pdefq cadence")?
        || !bool_member(
            pdefq,
            "qa_q3d_raw_bit_equal_after_solve3",
            "T300 pdefq cadence",
        )?
    {
        return Err(invalid(
            "T300 diagnostics receipt does not seal pdefq cadence and solve3 qa/q3d identity",
        ));
    }
    fixed_string(
        pdefq,
        "independent_full_gate_sha256",
        &binding.pdefq_receipt_sha256,
        "T300 pdefq cadence",
    )?;
    let step_key = binding.clock.step().to_string();
    let checkpoint = member(
        member(receipt, "restart_checkpoints", "T300 diagnostics receipt")?,
        &step_key,
        "parent restart checkpoints",
    )?;
    fixed_string(
        checkpoint,
        "sha256",
        &binding.checkpoint_artifact_sha256,
        "T300 restart checkpoint",
    )?;
    if usize_member(
        checkpoint,
        "persisted_carrier_count",
        "T300 restart checkpoint",
    )? != 37
        || usize_member(
            checkpoint,
            "operator_source_hash_input_count",
            "T300 restart checkpoint",
        )? != 22
    {
        return Err(invalid(
            "T300 diagnostics receipt checkpoint inventory is not 37 carriers / 22 operators",
        ));
    }
    let transport = member(checkpoint, "transport", "T300 restart checkpoint")?;
    fixed_string(
        transport,
        "raw_arena_sha256",
        &binding.checkpoint_raw_arena_sha256,
        "T300 checkpoint transport",
    )?;
    if u64_member(transport, "model_step", "parent checkpoint transport")? != binding.clock.step() {
        return Err(invalid(
            "parent diagnostics receipt transport step differs from the bound completed large step",
        ));
    }
    fixed_string(
        transport,
        "model_time_f64_bits",
        &format!("{:016x}", binding.clock.model_time_s().to_bits()),
        "parent checkpoint transport",
    )?;
    Ok(())
}

fn validate_pdefq_receipt(receipt: &JsonValue) -> Result<()> {
    fixed_string(
        receipt,
        "schema",
        "gpuwm-cm1.wsm6-pdefq-full-gate/v1",
        "T300 pdefq receipt",
    )?;
    fixed_string(
        receipt,
        "status",
        "PASS_EXACT_COMPACT_AND_FULL_REAL_CARRIER",
        "T300 pdefq receipt",
    )?;
    let full = member(receipt, "full_real_carrier", "T300 pdefq receipt")?;
    if !bool_member(full, "all_nonnegative", "T300 pdefq full carrier")?
        || bool_member(full, "synthetic_product", "T300 pdefq full carrier")?
    {
        return Err(invalid(
            "T300 pdefq full-carrier gate is synthetic or leaves negative moisture",
        ));
    }
    require_usize_array(
        member(full, "geometry", "T300 pdefq full carrier")?,
        &[PARENT_NX, PARENT_NY, PARENT_NZ],
        "T300 pdefq geometry",
    )?;
    fixed_string(
        full,
        "product_manifest_sha256",
        GM1_F02_ROI_PRODUCT_MANIFEST_SHA256,
        "T300 pdefq full carrier",
    )?;
    Ok(())
}

fn validate_receipt_anchor(value: &JsonValue, anchor: ParentAnchor, label: &str) -> Result<()> {
    let mapped = anchor_json(anchor);
    for (actual_key, expected_key) in [
        ("parent_i", "parent_i"),
        ("parent_j", "parent_j"),
        ("candidate_child_i", "child_nearest_i"),
        ("candidate_child_j", "child_nearest_j"),
        (
            "candidate_child_edge_distance_points",
            "child_nearest_edge_cells",
        ),
    ] {
        if usize_member(value, actual_key, label)? != usize_member(&mapped, expected_key, label)? {
            return Err(invalid(format!(
                "{label}.{actual_key} differs from the Rust binding"
            )));
        }
    }
    let maximum = member(value, "maximum", label)?
        .as_f64()
        .map_err(|error| invalid(format!("{label}.maximum: {error}")))?;
    if !maximum.is_finite() || (maximum as f32).to_bits() != anchor.value.to_bits() {
        return Err(invalid(format!(
            "{label}.maximum differs from the bound binary32 diagnostic"
        )));
    }
    Ok(())
}

fn require_usize_array(value: &JsonValue, expected: &[usize], label: &str) -> Result<()> {
    let values = value
        .as_array()
        .map_err(|error| invalid(format!("{label}: {error}")))?;
    if values.len() != expected.len()
        || values.iter().zip(expected).any(|(value, expected)| {
            value
                .as_u64()
                .ok()
                .and_then(|raw| usize::try_from(raw).ok())
                != Some(*expected)
        })
    {
        return Err(invalid(format!("{label} differs from the sealed values")));
    }
    Ok(())
}

fn require_f64_bits_array(value: &JsonValue, expected: &[f64], label: &str) -> Result<()> {
    let values = value
        .as_array()
        .map_err(|error| invalid(format!("{label}: {error}")))?;
    if values.len() != expected.len()
        || values.iter().zip(expected).any(|(value, expected)| {
            value.as_f64().ok().map(f64::to_bits) != Some(expected.to_bits())
        })
    {
        return Err(invalid(format!("{label} differs from the sealed values")));
    }
    Ok(())
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

fn string_array(values: &[&str]) -> JsonValue {
    JsonValue::Array(values.iter().map(|value| string(value)).collect())
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

fn f64_bits(value: f64) -> JsonValue {
    string(&format!("{:016x}", value.to_bits()))
}

fn member<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a JsonValue> {
    value
        .get(key)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn string_member<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    member(value, key, label)?
        .as_str()
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn usize_member(value: &JsonValue, key: &str, label: &str) -> Result<usize> {
    let raw = member(value, key, label)?
        .as_u64()
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
    usize::try_from(raw).map_err(|_| invalid(format!("{label}.{key} does not fit usize")))
}

fn bool_member(value: &JsonValue, key: &str, label: &str) -> Result<bool> {
    match member(value, key, label)? {
        JsonValue::Bool(result) => Ok(*result),
        _ => Err(invalid(format!("{label}.{key} must be a JSON boolean"))),
    }
}

fn f32_member(value: &JsonValue, key: &str, label: &str) -> Result<f32> {
    let word = string_member(value, key, label)?;
    if word.len() != 8 || !word.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(format!("{label}.{key} is not eight hex digits")));
    }
    let result = f32::from_bits(
        u32::from_str_radix(word, 16)
            .map_err(|_| invalid(format!("{label}.{key} does not parse")))?,
    );
    if !result.is_finite() {
        return Err(invalid(format!("{label}.{key} is not finite")));
    }
    Ok(result)
}

fn f64_member(value: &JsonValue, key: &str, label: &str) -> Result<f64> {
    let word = string_member(value, key, label)?;
    if word.len() != 16
        || !word
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{label}.{key} is not sixteen lower-case hex digits"
        )));
    }
    let result = f64::from_bits(
        u64::from_str_radix(word, 16)
            .map_err(|_| invalid(format!("{label}.{key} does not parse")))?,
    );
    if !result.is_finite() {
        return Err(invalid(format!("{label}.{key} is not finite")));
    }
    Ok(result)
}

fn u64_member(value: &JsonValue, key: &str, label: &str) -> Result<u64> {
    member(value, key, label)?
        .as_u64()
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn fixed_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    if string_member(value, key, label)? != expected {
        return Err(invalid(format!(
            "{label}.{key} differs from the sealed value"
        )));
    }
    Ok(())
}

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) -> Result<()> {
    let mut actual = value
        .as_object()
        .map_err(|error| invalid(format!("{label}: {error}")))?
        .iter()
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err(invalid(format!(
            "{label} keys are {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    // ---- the parent clock is a property, not one historical step ----------

    /// CM1 r21.1 advances the model clock by exactly one `dbldt` per completed
    /// large step (`src/cm1.F:3510`); `nstep` is incremented once per step
    /// (`src/cm1.F:3171`) and both are persisted by the restart writer
    /// (`src/restart_write.F:364` and `src/restart_write.F:403`).
    #[test]
    fn every_completed_large_step_is_admitted_not_only_the_first_one() {
        for step in [1u64, 2, 599, 600, 601, 1_200, 7_200, 86_400, 3_400_000] {
            let clock = ParentClock::completed_large_step(step).unwrap();
            assert_eq!(clock.step(), step);
            assert_eq!(clock.dt_s().to_bits(), PARENT_DT_S.to_bits());
            // The clock CM1 itself would carry: `step` accumulations of dt.
            let mut replay = 0.0f64;
            for _ in 0..step.min(200_000) {
                replay += PARENT_DT_S;
            }
            if step <= 200_000 {
                assert_eq!(clock.model_time_s().to_bits(), replay.to_bits());
            }
            assert_eq!(
                ParentClock::from_recorded(step, clock.model_time_s(), PARENT_DT_S).unwrap(),
                clock
            );
        }
        // The first admitted parent is still admitted, and is no longer special.
        let first = ParentClock::completed_large_step(FIRST_ADMITTED_PARENT_STEP).unwrap();
        assert_eq!(
            first.model_time_s().to_bits(),
            FIRST_ADMITTED_PARENT_MODEL_TIME_S.to_bits()
        );
    }

    #[test]
    fn a_step_zero_or_mid_step_parent_clock_is_refused_with_its_reason() {
        let error = ParentClock::completed_large_step(0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("step-zero parent"), "{error}");

        // A model time that is not the replay of `nstep` advances: the
        // checkpoint was not written at a large-step boundary.
        let error = ParentClock::from_recorded(600, 300.25, PARENT_DT_S)
            .unwrap_err()
            .to_string();
        assert!(error.contains("mid-RK stage"), "{error}");
        assert!(error.contains("completed large-step boundary"), "{error}");

        let error = ParentClock::from_recorded(600, 300.0, 0.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("never advances"), "{error}");

        let error = ParentClock::completed_large_step(MAX_ADMITTED_PARENT_STEP + 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("corrupt"), "{error}");
    }

    #[test]
    fn a_non_binary_exact_timestep_replays_cm1s_accumulation_not_the_product() {
        // 0.1 s is not representable in binary64, so CM1's repeated addition
        // drifts from `step * dt`.  Admission must follow the model, or a
        // perfectly good checkpoint would be refused for a rounding artefact.
        let mut accumulated = 0.0f64;
        for _ in 0..1000 {
            accumulated += 0.1f64;
        }
        assert_ne!(accumulated.to_bits(), (1000.0f64 * 0.1).to_bits());
        let clock = ParentClock::completed_large_step_at(1000, 0.1).unwrap();
        assert_eq!(clock.model_time_s().to_bits(), accumulated.to_bits());
        assert!(ParentClock::from_recorded(1000, 1000.0 * 0.1, 0.1).is_err());
        assert!(ParentClock::from_recorded(1000, accumulated, 0.1).is_ok());
    }

    // ---- the child window is placed in the parent that actually ran -------

    #[test]
    fn a_child_window_may_be_centred_anywhere_its_stencil_stays_inside_the_parent() {
        // Same 300x300 50 m child, moved off the historical anchor.
        let moved = ChildGridSpec::new_centered(
            300,
            300,
            CHILD_NZ,
            CHILD_DX_M,
            CHILD_DY_M,
            CHILD_ZTOP_M,
            100,
            180,
        )
        .unwrap();
        assert_eq!(moved.parent_center_i(), 100);
        assert_eq!(moved.parent_center_j(), 180);
        assert_eq!(moved.extent_x_m().to_bits(), 15_000.0f64.to_bits());

        // A smaller, finer child is admitted too: the frozen 15 km extent is
        // no longer a requirement.
        let fine = ChildGridSpec::new_centered(
            256,
            256,
            CHILD_NZ,
            25.0,
            25.0,
            CHILD_ZTOP_M,
            PARENT_CENTER_I,
            PARENT_CENTER_J,
        )
        .unwrap();
        assert_eq!(fine.nx(), 256);
        assert_eq!(fine.extent_x_m().to_bits(), 6_400.0f64.to_bits());
    }

    #[test]
    fn a_corner_on_the_last_parent_point_is_refused_where_it_is_placed() {
        // Scalar corner lands on exactly parent mass point 299 of a 300-point
        // parent: inside the domain, but with no i0 + 1 to interpolate
        // towards.  Every consumer fills that point bilinearly, so the window
        // must be refused where it is constructed, by the message that names
        // the reason - not later, inside the terrain remap, by a message about
        // a stencil the caller never mentioned.
        let error = ChildGridSpec::new_centered(
            3,
            3,
            CHILD_NZ,
            PARENT_DX_M,
            PARENT_DY_M,
            CHILD_ZTOP_M,
            298,
            PARENT_CENTER_J,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("not strictly inside") && error.contains("scalar"),
            "{error}"
        );

        // One parent cell further in is admitted, and its far corner is
        // strictly inside the support.
        let grid = ChildGridSpec::new_centered(
            3,
            3,
            CHILD_NZ,
            PARENT_DX_M,
            PARENT_DY_M,
            CHILD_ZTOP_M,
            297,
            PARENT_CENTER_J,
        )
        .unwrap();
        let (parent_i, _) =
            parent_fractional_xy_for_grid(grid, HorizontalStagger::Scalar, 2, 2).unwrap();
        assert!(parent_i < (PARENT_NX - 1) as f64, "{parent_i}");
    }

    #[test]
    fn a_child_window_that_would_extrapolate_off_the_parent_is_refused_by_name() {
        let error = ChildGridSpec::new_centered(
            300,
            300,
            CHILD_NZ,
            CHILD_DX_M,
            CHILD_DY_M,
            CHILD_ZTOP_M,
            10,
            10,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("extrapolated off the parent domain"),
            "{error}"
        );

        let error = ChildGridSpec::new_centered(
            300,
            300,
            CHILD_NZ,
            CHILD_DX_M,
            CHILD_DY_M,
            CHILD_ZTOP_M,
            PARENT_NX,
            PARENT_CENTER_J,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("anchored outside the parent domain"),
            "{error}"
        );
    }

    #[test]
    fn the_child_follows_the_parent_grid_the_checkpoint_actually_ran_on() {
        // A parent published at a different extent and spacing moves the child
        // window with it instead of reindexing the same child into memory the
        // parent does not have.
        let parent = ParentHorizontalGrid::new(600, 600, 50.0, 50.0).unwrap();
        let child = ChildGridSpec::new_in_parent(
            parent,
            300,
            300,
            CHILD_NZ,
            25.0,
            25.0,
            CHILD_ZTOP_M,
            300,
            300,
        )
        .unwrap();
        assert_eq!(child.parent_nx(), 600);
        assert_eq!(child.parent_dx_m().to_bits(), 50.0f64.to_bits());
        // Halfway across the child is halfway between parent cells 299 and 300.
        let (x, _) =
            parent_fractional_xy_for_grid(child, HorizontalStagger::Scalar, 150, 150).unwrap();
        assert_eq!(x.to_bits(), 300.25f64.to_bits());
        // The same child geometry in the sealed 100 m parent maps elsewhere.
        let sealed =
            ChildGridSpec::new_centered(300, 300, CHILD_NZ, 25.0, 25.0, CHILD_ZTOP_M, 224, 224)
                .unwrap();
        let (sealed_x, _) =
            parent_fractional_xy_for_grid(sealed, HorizontalStagger::Scalar, 150, 150).unwrap();
        assert_ne!(sealed_x.to_bits(), x.to_bits());

        assert!(ParentHorizontalGrid::new(600, 600, 0.0, 50.0).is_err());
        assert!(ParentHorizontalGrid::new(1, 600, 50.0, 50.0).is_err());
    }

    #[test]
    fn a_bound_parent_identity_must_be_recordable() {
        validate_bound_parent_identity(&"ab".repeat(32), &"cd".repeat(32), "test").unwrap();
        let error = validate_bound_parent_identity("short", &"cd".repeat(32), "test")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("could not be re-bound to the parent"),
            "{error}"
        );
        let error = validate_bound_parent_identity(&"AB".repeat(32), &"cd".repeat(32), "test")
            .unwrap_err()
            .to_string();
        assert!(error.contains("lower-case SHA-256"), "{error}");
    }

    fn real_preliminary_binding() -> ParentBinding {
        ParentBinding::new(
            "11".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
            "44".repeat(32),
            "55".repeat(32),
            "66".repeat(32),
            ParentClock::completed_large_step(600).unwrap(),
            ParentAnchor {
                i: 243,
                j: 217,
                value: 63.182,
            },
            ParentAnchor {
                i: 264,
                j: 199,
                value: 37.312,
            },
            ParentAnchor {
                i: 247,
                j: 258,
                value: 0.04631,
            },
        )
        .unwrap()
    }

    fn legacy_parent_fractional_reference(
        stagger: HorizontalStagger,
        child_i: usize,
        child_j: usize,
    ) -> (f64, f64) {
        let mass_x = PARENT_CENTER_I as f64
            + (child_i as f64 + 0.5 - 0.5 * CHILD_NX as f64) * CHILD_DX_M / PARENT_DX_M;
        let mass_y = PARENT_CENTER_J as f64
            + (child_j as f64 + 0.5 - 0.5 * CHILD_NY as f64) * CHILD_DY_M / PARENT_DY_M;
        let xface_x = PARENT_CENTER_I as f64
            + 0.5
            + (child_i as f64 - 0.5 * CHILD_NX as f64) * CHILD_DX_M / PARENT_DX_M;
        let yface_y = PARENT_CENTER_J as f64
            + 0.5
            + (child_j as f64 - 0.5 * CHILD_NY as f64) * CHILD_DY_M / PARENT_DY_M;
        match stagger {
            HorizontalStagger::XFace => (xface_x, mass_y),
            HorizontalStagger::YFace => (mass_x, yface_y),
            HorizontalStagger::Scalar | HorizontalStagger::ZFace | HorizontalStagger::Surface => {
                (mass_x, mass_y)
            }
        }
    }

    #[test]
    fn accepted_50m_grid_parameterization_is_raw_bit_identical() {
        assert_eq!(
            ChildGridSpec::new(300, 300, 75, 50.0, 50.0, 15_000.0).unwrap(),
            ACCEPTED_50M_CHILD_GRID
        );
        for stagger in [
            HorizontalStagger::Scalar,
            HorizontalStagger::XFace,
            HorizontalStagger::YFace,
            HorizontalStagger::ZFace,
            HorizontalStagger::Surface,
        ] {
            let (nx, ny) = stagger.child_shape_xy_for(ACCEPTED_50M_CHILD_GRID);
            for child_j in 0..ny {
                for child_i in 0..nx {
                    let actual = parent_fractional_xy_for_grid(
                        ACCEPTED_50M_CHILD_GRID,
                        stagger,
                        child_i,
                        child_j,
                    )
                    .unwrap();
                    let legacy = legacy_parent_fractional_reference(stagger, child_i, child_j);
                    assert_eq!(actual.0.to_bits(), legacy.0.to_bits());
                    assert_eq!(actual.1.to_bits(), legacy.1.to_bits());
                    assert_eq!(
                        parent_fractional_xy(stagger, child_i, child_j).unwrap(),
                        actual
                    );
                }
            }
        }
        // The published preflight document changed deliberately: it no longer
        // demands one historical step/model-time pair, and it now states the
        // step-to-clock relation this code actually implements - CM1's own
        // per-step accumulation, which is the product only when dt is a power
        // of two.  The digest is repinned to that document.
        let document = encode_geometry_preflight().unwrap();
        let relation = crate::json::parse(&document)
            .unwrap()
            .get("parent")
            .unwrap()
            .get("required_step_time_relation")
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        assert!(
            relation.contains("binary64 replay of CM1's own clock")
                && relation.contains("the product is not the contract"),
            "{relation}"
        );
        assert_eq!(
            sha256::digest_hex(&document),
            "a1ffb10f5bd4213fc4e48037d4ee52313963c4784765968f962ff4b3119a1962"
        );
    }

    #[test]
    fn hard_5m_grid_is_exact_source_contained_same_domain() {
        assert_eq!(
            ChildGridSpec::new(3_000, 3_000, 75, 5.0, 5.0, 15_000.0).unwrap(),
            HARD_5M_CHILD_GRID
        );
        assert_eq!(HARD_5M_CHILD_GRID.nx(), 3_000);
        assert_eq!(HARD_5M_CHILD_GRID.ny(), 3_000);
        assert_eq!(HARD_5M_CHILD_GRID.nz(), 75);
        assert_eq!(HARD_5M_CHILD_GRID.dx_m().to_bits(), 5.0f64.to_bits());
        assert_eq!(HARD_5M_CHILD_GRID.dy_m().to_bits(), 5.0f64.to_bits());
        assert_eq!(HARD_5M_CHILD_GRID.ztop_m().to_bits(), 15_000.0f64.to_bits());
        assert_eq!(
            parent_fractional_xy_for_grid(HARD_5M_CHILD_GRID, HorizontalStagger::Scalar, 0, 0,)
                .unwrap(),
            (149.025, 149.025)
        );
        assert_eq!(
            parent_fractional_xy_for_grid(
                HARD_5M_CHILD_GRID,
                HorizontalStagger::Scalar,
                2_999,
                2_999,
            )
            .unwrap(),
            (298.975, 298.975)
        );
        assert_eq!(
            parent_fractional_xy_for_grid(
                HARD_5M_CHILD_GRID,
                HorizontalStagger::XFace,
                3_000,
                2_999,
            )
            .unwrap(),
            (299.5, 298.975)
        );
    }

    #[test]
    fn child_grid_spec_refuses_extent_vertical_and_index_drift() {
        assert!(ChildGridSpec::new(3_000, 3_000, 75, 5.1, 5.0, 15_000.0).is_err());
        assert!(ChildGridSpec::new(3_000, 3_000, 74, 5.0, 5.0, 15_000.0).is_err());
        assert!(ChildGridSpec::new(3_000, 3_000, 75, 5.0, 5.0, 15_001.0).is_err());
        assert!(parent_fractional_xy_for_grid(
            HARD_5M_CHILD_GRID,
            HorizontalStagger::Scalar,
            3_000,
            0,
        )
        .is_err());
    }

    #[test]
    fn selected_horizontal_mapping_retains_native_staggers() {
        assert_eq!(
            parent_fractional_xy(HorizontalStagger::Scalar, 0, 0).unwrap(),
            (149.25, 149.25)
        );
        assert_eq!(
            parent_fractional_xy(HorizontalStagger::Scalar, 299, 299).unwrap(),
            (298.75, 298.75)
        );
        assert_eq!(
            parent_fractional_xy(HorizontalStagger::XFace, 0, 0).unwrap(),
            (149.5, 149.25)
        );
        assert_eq!(
            parent_fractional_xy(HorizontalStagger::XFace, 300, 299).unwrap(),
            (299.5, 298.75)
        );
        assert_eq!(
            parent_fractional_xy(HorizontalStagger::YFace, 299, 300).unwrap(),
            (298.75, 299.5)
        );
        assert!(parent_fractional_xy(HorizontalStagger::Scalar, 300, 0).is_err());
    }

    #[test]
    fn real_preliminary_anchors_are_safely_interior() {
        let binding = real_preliminary_binding();
        binding.validate().unwrap();
        let value = parent_binding_json(&binding);
        let diagnostics = value.get("diagnostics").unwrap();
        assert_eq!(
            diagnostics
                .get("low_level_reflectivity")
                .unwrap()
                .get("child_nearest_edge_cells")
                .unwrap()
                .as_u64()
                .unwrap(),
            111
        );
        assert_eq!(
            diagnostics
                .get("column_updraft")
                .unwrap()
                .get("child_nearest_edge_cells")
                .unwrap()
                .as_u64()
                .unwrap(),
            69
        );
    }

    #[test]
    fn parent_binding_roundtrips_and_refuses_drift() {
        let binding = real_preliminary_binding();
        let bytes = binding.canonical_bytes();
        assert_eq!(
            ParentBinding::from_canonical_bytes(&bytes).unwrap(),
            binding
        );
        let mut drift = bytes;
        let offset = drift
            .windows("PASS_PDEFQ_CLEAN_T300".len())
            .position(|word| word == b"PASS_PDEFQ_CLEAN_T300")
            .unwrap();
        drift[offset] = b'F';
        assert!(ParentBinding::from_canonical_bytes(&drift).is_err());
    }

    #[test]
    fn binding_file_roundtrip_requires_one_lf_and_refuses_replacement() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("gm1-child-binding-{nonce}"));
        fs::create_dir(&root).unwrap();
        let path = root.join("binding.json");
        let binding = real_preliminary_binding();
        binding.write_new(&path).unwrap();
        assert_eq!(read_parent_binding(&path).unwrap(), binding);
        assert!(binding.write_new(&path).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn preflight_is_explicitly_blocked_on_vertical_and_boundary_producers() {
        let bytes = encode_geometry_preflight().unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("PINNED_FLAT_CPU_CUDA_BLOCKED_ON_TERRAIN_COMPOSITION"));
        assert!(text.contains(CHILD_STRETCH_CUDA_RECEIPT_SHA256));
        assert!(text.contains(CHILD_STRETCH_GRID_SHA256));
        assert!(text.contains("child_boundary_initialize"));
        assert!(text.contains("synthetic_translation\":false"));
        assert!(!text.contains("17650"));
        assert_eq!(crate::sha256::digest_hex(&bytes).len(), 64);
    }

    #[test]
    fn child_stretch_grid_matches_every_native_cpu_pin_without_upper_fill() {
        let grid = sealed_child_stretch_grid().unwrap();
        assert_eq!(grid.zf[CHILD_NZ + 1].to_bits(), (15_000.0f32).to_bits());
        assert_eq!(grid.zh[1].to_bits(), 10.0f32.to_bits());
        assert_eq!(grid.combined_sha256(), CHILD_STRETCH_GRID_SHA256);
        assert_eq!(
            (grid.derived.nk1, grid.derived.nk2, grid.derived.nk3),
            (10, 40, 25)
        );
    }

    #[test]
    fn outside_or_edge_pinned_storm_anchor_refuses() {
        let mut binding = real_preliminary_binding();
        binding.low_level_reflectivity.i = 299;
        assert!(binding.validate().is_err());
    }

    #[test]
    fn producer_requirements_bind_exact_parent_and_emit_no_child_carriers() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("gm1-child-producers-{nonce}"));
        fs::create_dir(&root).unwrap();
        let plan_path = root.join("plan.json");
        let binding = real_preliminary_binding();
        write_newline_json(
            &plan_path,
            &preflight_json(Some(&binding)).unwrap().canonical_bytes(),
        )
        .unwrap();
        let plan_sha256 = sha256::digest_hex(&fs::read(&plan_path).unwrap());
        let requirements = encode_child_producer_requirements(&plan_path).unwrap();
        let value = json::parse(&requirements).unwrap();
        assert_eq!(
            value.get("schema").unwrap().as_str().unwrap(),
            CHILD_PRODUCER_REQUIREMENTS_SCHEMA
        );
        assert_eq!(
            value
                .get("parent_plan")
                .unwrap()
                .get("sha256")
                .unwrap()
                .as_str()
                .unwrap(),
            plan_sha256
        );
        assert_eq!(
            value.get("numerical_child_output_allowed").unwrap(),
            &JsonValue::Bool(false)
        );
        let text = std::str::from_utf8(&requirements).unwrap();
        assert!(text.contains(CHILD_TERRAIN_STRETCH_SCHEMA));
        assert!(text.contains(TERRAIN_OPEN_RECEIPT_SCHEMA));
        assert!(text.contains(WSM6_GEOPOTENTIAL_RECEIPT_SCHEMA));
        assert!(text.contains("parent_radiation_history_reuse_allowed\":false"));
        assert!(text.contains("cold_start_history_nonzero_words\":0"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn producer_requirements_refuse_plan_status_or_geometry_drift() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("gm1-child-plan-drift-{nonce}"));
        fs::create_dir(&root).unwrap();
        let plan_path = root.join("plan.json");
        let binding = real_preliminary_binding();
        let mut bytes = preflight_json(Some(&binding)).unwrap().canonical_bytes();
        let offset = bytes
            .windows("PASS_PARENT_CHECKPOINT_AWAITING_CHILD_PRODUCERS".len())
            .position(|word| word == b"PASS_PARENT_CHECKPOINT_AWAITING_CHILD_PRODUCERS")
            .unwrap();
        bytes[offset] = b'F';
        bytes.push(b'\n');
        fs::write(&plan_path, bytes).unwrap();
        assert!(encode_child_producer_requirements(&plan_path).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
