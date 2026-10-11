//! CM1 r21.1 analytic base-state lanes: `isnd=5` and `iwnd=2`.
//!
//! This module is the second admitted idealized initialization lane beside
//! the external-sounding `isnd=7` lane in
//! [`crate::idealized_initialization`].  Both lanes share one admission type,
//! one physical-domain frame, and one padded-carrier adapter; only the
//! base-state construction differs.  Here the thermodynamic profile is the
//! closed form of `base.F:363-450` (Weisman and Klemp, 1982, MWR, 110,
//! 504-520) and the base winds are the closed form of `base.F:2321-2360`.
//! Nothing on this path reads an external file, so there is no byte stream to
//! preserve and no sounding provenance hash to emit.
//!
//! Its outputs are idealized analytic initial conditions, never observations,
//! forecasts, detections, or validated simulations of any weather event.

use crate::base_state_wind::{base_wind_u, base_wind_v};
use crate::terrain::DeclaredVerticalCoordinate;
use crate::idealized_initialization::{
    adapt_flat_idealized_frame_to_padded_state, adapt_stretched_idealized_frame_to_padded_state,
    apply_translating_frame, build_initial_perturbation, domain_span,
    finish_base_dependent_perturbation,
    checked_product, index3, invalid, require_finite, rslf, IdealizedBaseStateLane,
    IdealizedInitializationFrame, IdealizedInitializationSelection, IdealizedPaddedState,
    TranslatingFrame, WarmBubbleControls, CP, G, P00, RD, RV,
};
use crate::Result;

pub const ANALYTIC_INITIALIZATION_SCHEMA: &str = "gm1.cm1-r21.1-isnd5-iwnd2-initialization/v1";
pub const ANALYTIC_INITIALIZATION_STATUS: &str = "ADMITTED_IDEALIZED_CM1_ISND5_IWND2";
pub const ANALYTIC_CLASSIFICATION: &str =
    "idealized analytic initial condition; never an observed or forecast meteorological product";
pub const ANALYTIC_CLAIM_LIMIT: &str = "CM1 r21.1 isnd=5 analytic sounding, iwnd=2 analytic wind profile, and selectable iinit=1 theta perturbation only; no dynamics, physics trajectory, forecast, or model-validation claim";

/// `constants.F:14` is `3.1415926535897932384626433` declared as a default
/// REAL, whose binary32 rounding is exactly `f32::consts::PI`.  CM1 then
/// evaluates `pi/180.0` in default REAL, which this port also does.
const PI: f32 = std::f32::consts::PI;

// base.F:369-374, the named Weisman-Klemp sounding parameters.
pub const WK_Z_TROP_M: f32 = 12_000.0_f32;
pub const WK_TH_TROP_K: f32 = 343.0_f32;
pub const WK_T_TROP_K: f32 = 213.0_f32;
pub const WK_TH_SFC_K: f32 = 300.0_f32;
pub const WK_PRS_SFC_PA: f32 = 100_000.0_f32;
pub const WK_QV_PBL: f32 = 0.014_f32;

/// base.F:409 (`do n=1,20`).  The hydrostatic fixed point is run exactly this
/// many times; the count is part of the answer, not a tolerance.
pub const WK_HYDROSTATIC_ITERATIONS: usize = 20;

// base.F:2323-2326 and base.F:2344, the named analytic wind parameters.
pub const ANALYTIC_SHEAR_UDEP1_M: f32 = 2000.0_f32;
pub const ANALYTIC_SHEAR_UDEP2_M: f32 = 6000.0_f32;
pub const ANALYTIC_SHEAR_UMAX1_M_PER_S: f32 = 7.0_f32;
pub const ANALYTIC_SHEAR_UMAX2_M_PER_S: f32 = 31.0_f32;

/// The three surface words of `base.F:378-380`.
///
/// `prs_sfc` is exactly `p00`, so `pi_sfc` is exactly one; the port still
/// evaluates the source expression instead of folding it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeismanKlempSurface {
    pi: f32,
    qv_kg_per_kg: f32,
    theta_v_k: f32,
}

impl WeismanKlempSurface {
    pub fn official() -> Self {
        let reps = RV / RD;
        let pi = (WK_PRS_SFC_PA / P00).powf(RD / CP);
        let qv_kg_per_kg = rslf(WK_PRS_SFC_PA, WK_TH_SFC_K * pi);
        let theta_v_k = WK_TH_SFC_K * (1.0_f32 + qv_kg_per_kg * reps) / (1.0_f32 + qv_kg_per_kg);
        Self {
            pi,
            qv_kg_per_kg,
            theta_v_k,
        }
    }

    pub fn pi(self) -> f32 {
        self.pi
    }

    pub fn qv_kg_per_kg(self) -> f32 {
        self.qv_kg_per_kg
    }

    pub fn theta_v_k(self) -> f32 {
        self.theta_v_k
    }

    /// `base.F:440`.
    pub fn pressure_pa(self) -> f32 {
        WK_PRS_SFC_PA
    }

    /// `base.F:442`: `th_sfc` times `psurf*rp00` raised to `rovcp`.
    pub fn temperature_k(self) -> f32 {
        let rp00 = 1.0_f32 / P00;
        let rovcp = RD / CP;
        WK_TH_SFC_K * (WK_PRS_SFC_PA * rp00).powf(rovcp)
    }
}

/// One analytic base-state column on the caller's mass heights.
#[derive(Clone, Debug)]
pub struct WeismanKlempColumn {
    heights_m: Vec<f32>,
    theta_k: Vec<f32>,
    relative_humidity: Vec<f32>,
    qv_kg_per_kg: Vec<f32>,
    theta_v_k: Vec<f32>,
    pi: Vec<f32>,
    pressure_pa: Vec<f32>,
    temperature_k: Vec<f32>,
    iterations: usize,
    imoist: i32,
    final_qv_update_linf: f32,
    final_pi_update_linf: f32,
}

impl WeismanKlempColumn {
    pub fn heights_m(&self) -> &[f32] {
        &self.heights_m
    }
    pub fn theta_k(&self) -> &[f32] {
        &self.theta_k
    }
    pub fn relative_humidity(&self) -> &[f32] {
        &self.relative_humidity
    }
    pub fn qv_kg_per_kg(&self) -> &[f32] {
        &self.qv_kg_per_kg
    }
    pub fn theta_v_k(&self) -> &[f32] {
        &self.theta_v_k
    }
    pub fn pi(&self) -> &[f32] {
        &self.pi
    }
    pub fn pressure_pa(&self) -> &[f32] {
        &self.pressure_pa
    }
    pub fn temperature_k(&self) -> &[f32] {
        &self.temperature_k
    }
    pub fn iterations(&self) -> usize {
        self.iterations
    }
    pub fn imoist(&self) -> i32 {
        self.imoist
    }

    /// Largest `|qv0|` change made by the final hydrostatic iteration.
    pub fn final_qv_update_linf(&self) -> f32 {
        self.final_qv_update_linf
    }

    /// Largest `|pi0|` change made by the final hydrostatic iteration.
    pub fn final_pi_update_linf(&self) -> f32 {
        self.final_pi_update_linf
    }

    /// `base.F:442-450`.  `weights` are the CM1 `cgs1,cgs2,cgs3` bottom
    /// extrapolation coefficients for this grid.
    pub fn surface_qv_kg_per_kg(&self, weights: [f32; 3]) -> Result<f32> {
        if self.imoist != 1 {
            return Ok(0.0_f32);
        }
        if self.qv_kg_per_kg.len() < 3 {
            return Err(invalid(
                "analytic surface qv needs the lowest three mass levels (base.F:444)",
            ));
        }
        Ok(if self.qv_kg_per_kg[0] < WK_QV_PBL {
            weights[0] * self.qv_kg_per_kg[0]
                + weights[1] * self.qv_kg_per_kg[1]
                + weights[2] * self.qv_kg_per_kg[2]
        } else {
            WK_QV_PBL
        })
    }
}

/// Build the `isnd=5` column with CM1's exact 20 hydrostatic iterations.
pub fn weisman_klemp_column(heights_m: &[f32], imoist: i32) -> Result<WeismanKlempColumn> {
    weisman_klemp_column_with_iterations(heights_m, imoist, WK_HYDROSTATIC_ITERATIONS)
}

/// `base.F:363-450` transcribed in source order.
///
/// `iterations` exists only so a test can show the fixed point is already
/// settled at CM1's count; every admitted lane uses
/// [`WK_HYDROSTATIC_ITERATIONS`].
#[allow(clippy::needless_range_loop)] // Keep the source k traversal explicit.
pub fn weisman_klemp_column_with_iterations(
    heights_m: &[f32],
    imoist: i32,
    iterations: usize,
) -> Result<WeismanKlempColumn> {
    let nk = heights_m.len();
    if nk == 0 {
        return Err(invalid("analytic sounding column requires nk>=1"));
    }
    if !matches!(imoist, 0 | 1) {
        return Err(invalid(format!(
            "base.F:389-397 branches on imoist=0 or 1 only; got imoist={imoist}. Any other value \
             would leave rh0 undefined and emit a base state with no defined moisture"
        )));
    }
    if iterations == 0 {
        return Err(invalid(
            "the analytic hydrostatic fixed point requires at least one iteration; zero would \
             emit qv0=0 and pi0=0, which is not a base state",
        ));
    }
    for k in 0..nk {
        require_finite("analytic sounding mass height", heights_m[k])?;
        let previous = if k == 0 { 0.0_f32 } else { heights_m[k - 1] };
        if heights_m[k] <= previous {
            return Err(invalid(format!(
                "analytic sounding mass heights must be strictly increasing above z=0; level {k} \
                 is {} m after {previous} m. A non-increasing height would invert the sign of the \
                 base.F:415-418 hydrostatic step and emit pressure that rises with height",
                heights_m[k]
            )));
        }
    }

    let reps = RV / RD;
    let surface = WeismanKlempSurface::official();

    // base.F:385-387 then base.F:389-397.
    let mut theta_k = vec![0.0_f32; nk];
    let mut relative_humidity = vec![0.0_f32; nk];
    for k in 0..nk {
        if heights_m[k] < WK_Z_TROP_M {
            theta_k[k] = WK_TH_SFC_K
                + (WK_TH_TROP_K - WK_TH_SFC_K) * ((heights_m[k] / WK_Z_TROP_M).powf(1.25_f32));
            if imoist == 1 {
                relative_humidity[k] =
                    1.0_f32 - 0.75_f32 * ((heights_m[k] / WK_Z_TROP_M).powf(1.25_f32));
            }
        } else {
            theta_k[k] =
                WK_TH_TROP_K * ((G / (WK_T_TROP_K * CP)) * (heights_m[k] - WK_Z_TROP_M)).exp();
            if imoist == 1 {
                relative_humidity[k] = 0.25_f32;
            }
        }
    }
    // base.F:399-400 sets the th0 ghost levels by copy.  They are never read
    // by the pressure integration below, so this port carries the interior.

    // base.F:405-409: qv0 starts at exact zero, then the fixed point runs.
    let mut qv_kg_per_kg = vec![0.0_f32; nk];
    let mut theta_v_k = vec![0.0_f32; nk];
    let mut pi = vec![0.0_f32; nk];
    let mut pressure_pa = vec![0.0_f32; nk];
    let mut final_qv_update_linf = 0.0_f32;
    let mut final_pi_update_linf = 0.0_f32;

    for _ in 0..iterations {
        let previous_qv = qv_kg_per_kg.clone();
        let previous_pi = pi.clone();
        // virtual potential temperature (base.F:411-413)
        for k in 0..nk {
            theta_v_k[k] =
                theta_k[k] * (1.0_f32 + reps * qv_kg_per_kg[k]) / (1.0_f32 + qv_kg_per_kg[k]);
        }
        // base.F:415-418
        pi[0] = surface.pi - G * heights_m[0] / (CP * 0.5_f32 * (surface.theta_v_k + theta_v_k[0]));
        for k in 1..nk {
            pi[k] = pi[k - 1]
                - G * (heights_m[k] - heights_m[k - 1])
                    / (CP * 0.5_f32 * (theta_v_k[k] + theta_v_k[k - 1]));
        }
        // pressure (base.F:421-423)
        for k in 0..nk {
            pressure_pa[k] = P00 * pi[k].powf(CP / RD);
        }
        // mixing ratio (base.F:426-429)
        for k in 0..nk {
            qv_kg_per_kg[k] = relative_humidity[k] * rslf(pressure_pa[k], theta_k[k] * pi[k]);
            if qv_kg_per_kg[k] > WK_QV_PBL {
                qv_kg_per_kg[k] = WK_QV_PBL;
            }
        }
        final_qv_update_linf = 0.0_f32;
        final_pi_update_linf = 0.0_f32;
        for k in 0..nk {
            final_qv_update_linf =
                final_qv_update_linf.max((qv_kg_per_kg[k] - previous_qv[k]).abs());
            final_pi_update_linf = final_pi_update_linf.max((pi[k] - previous_pi[k]).abs());
        }
    }

    // base.F:433-435
    let mut temperature_k = vec![0.0_f32; nk];
    for k in 0..nk {
        relative_humidity[k] = qv_kg_per_kg[k] / rslf(pressure_pa[k], theta_k[k] * pi[k]);
        temperature_k[k] = theta_k[k] * pi[k];
    }

    for k in 0..nk {
        for (label, value) in [
            ("th0", theta_k[k]),
            ("qv0", qv_kg_per_kg[k]),
            ("thv0", theta_v_k[k]),
            ("pi0", pi[k]),
            ("prs0", pressure_pa[k]),
            ("t0", temperature_k[k]),
            ("rh0", relative_humidity[k]),
        ] {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "analytic sounding {label} is not finite at level {k}"
                )));
            }
        }
        if theta_k[k] <= 0.0
            || pi[k] <= 0.0
            || pressure_pa[k] <= 0.0
            || temperature_k[k] <= 0.0
            || qv_kg_per_kg[k] < 0.0
            || relative_humidity[k] < 0.0
        {
            return Err(invalid(format!(
                "analytic sounding column is non-physical at level {k}: the requested mass \
                 heights leave the interval where base.F:363-450 produces a positive \
                 hydrostatic base state"
            )));
        }
    }

    Ok(WeismanKlempColumn {
        heights_m: heights_m.to_vec(),
        theta_k,
        relative_humidity,
        qv_kg_per_kg,
        theta_v_k,
        pi,
        pressure_pa,
        temperature_k,
        iterations,
        imoist,
        final_qv_update_linf,
        final_pi_update_linf,
    })
}

/// `base.F:2331-2339`, evaluated on one staggered u-face height.
pub fn analytic_shear_u(zu_m: f32) -> f32 {
    if zu_m <= ANALYTIC_SHEAR_UDEP1_M {
        let angle = 90.0_f32 * (zu_m / ANALYTIC_SHEAR_UDEP1_M) * (PI / 180.0_f32);
        ANALYTIC_SHEAR_UMAX1_M_PER_S - ANALYTIC_SHEAR_UMAX1_M_PER_S * angle.cos()
    } else if zu_m > ANALYTIC_SHEAR_UDEP1_M && zu_m <= ANALYTIC_SHEAR_UDEP2_M {
        ANALYTIC_SHEAR_UMAX1_M_PER_S
            + (zu_m - ANALYTIC_SHEAR_UDEP1_M)
                * (ANALYTIC_SHEAR_UMAX2_M_PER_S - ANALYTIC_SHEAR_UMAX1_M_PER_S)
                / (ANALYTIC_SHEAR_UDEP2_M - ANALYTIC_SHEAR_UDEP1_M)
    } else {
        ANALYTIC_SHEAR_UMAX2_M_PER_S
    }
}

/// `base.F:2344-2357`, evaluated on one staggered v-face height.
///
/// `vmax1=umax1`, and both branches above `udep1` assign `vmax1`; the source
/// spells them out separately and so does this transcription.
#[allow(clippy::if_same_then_else)] // base.F:2353-2356 writes both branches.
pub fn analytic_shear_v(zv_m: f32) -> f32 {
    let vmax1 = ANALYTIC_SHEAR_UMAX1_M_PER_S;
    if zv_m <= ANALYTIC_SHEAR_UDEP1_M {
        let angle = 90.0_f32 * (zv_m / ANALYTIC_SHEAR_UDEP1_M) * (PI / 180.0_f32);
        vmax1 * angle.sin()
    } else if zv_m > ANALYTIC_SHEAR_UDEP1_M && zv_m <= ANALYTIC_SHEAR_UDEP2_M {
        vmax1
    } else {
        vmax1
    }
}

/// The staggered face heights `iwnd=2` is evaluated on.
///
/// `base.F:2331` averages `zh(i-1,j,k)` and `zh(i,j,k)` over `i=1..ni+1`, and
/// `base.F:2349` averages `zh(i,j-1,k)` and `zh(i,j,k)` over `j=1..nj+1`.  The
/// `i-1` read at the first u-face and the `i` read at the last one are CM1
/// ghost columns; this port fills them with the zero-gradient copy of the
/// nearest interior column, which is CM1's open edge law and is exact on the
/// flat grid this lane admits.  Returned shapes are `[nk,nj,ni+1]` for `zu`
/// and `[nk,nj+1,ni]` for `zv`.
pub fn analytic_shear_face_heights(
    mass_heights_m: &[f32],
    shape_zyx: [usize; 3],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let [nz, ny, nx] = shape_zyx;
    if nz == 0 || ny == 0 || nx == 0 {
        return Err(invalid("analytic face heights require a positive shape"));
    }
    let words = checked_product(shape_zyx, "analytic face height")?;
    if mass_heights_m.len() != words {
        return Err(invalid(format!(
            "analytic face heights need {words} mass-height words for shape {shape_zyx:?}; got {}",
            mass_heights_m.len()
        )));
    }
    let mut zu = vec![0.0_f32; nz * ny * (nx + 1)];
    let mut zv = vec![0.0_f32; nz * (ny + 1) * nx];
    for k in 0..nz {
        for j in 0..ny {
            for face in 0..=nx {
                let west = face.saturating_sub(1);
                let east = face.min(nx - 1);
                zu[index3(k, j, face, ny, nx + 1)] = 0.5_f32
                    * (mass_heights_m[index3(k, j, west, ny, nx)]
                        + mass_heights_m[index3(k, j, east, ny, nx)]);
            }
        }
        for face in 0..=ny {
            let south = face.saturating_sub(1);
            let north = face.min(ny - 1);
            for i in 0..nx {
                zv[index3(k, face, i, ny + 1, nx)] = 0.5_f32
                    * (mass_heights_m[index3(k, south, i, ny, nx)]
                        + mass_heights_m[index3(k, north, i, ny, nx)]);
            }
        }
    }
    Ok((zu, zv))
}

/// `base.F:2321-2360` on the native staggered arrays: `u0` is `[nk,nj,ni+1]`
/// and `v0` is `[nk,nj+1,ni]`.
pub fn build_analytic_shear_faces(
    mass_heights_m: &[f32],
    shape_zyx: [usize; 3],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (zu, zv) = analytic_shear_face_heights(mass_heights_m, shape_zyx)?;
    let u0 = zu.iter().copied().map(analytic_shear_u).collect::<Vec<_>>();
    let v0 = zv.iter().copied().map(analytic_shear_v).collect::<Vec<_>>();
    Ok((u0, v0))
}

/// One analytic column's words, from whichever `base.F` arm built it.
struct AnalyticColumnWords {
    theta_k: Vec<f32>,
    qv_kg_per_kg: Vec<f32>,
    pi: Vec<f32>,
    pressure_pa: Vec<f32>,
    /// Empty: `t0 = th0*pi0` (base.F:2106).
    temperature_k: Vec<f32>,
    relative_humidity: Vec<f32>,
}

/// Build one physical-domain `isnd=5` / `iwnd=2` base plus its selected
/// `iinit` perturbation.
///
/// The height arrays use C-order `[k,j,i]` with the same physical mass shape,
/// exactly as the `isnd=7` entry does.  `u_heights_m` and `v_heights_m` carry
/// the CM1 face-averaged heights the analytic wind profile is evaluated on.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn initialize_analytic_idealized_domain(
    selection: IdealizedInitializationSelection,
    controls: WarmBubbleControls,
    xh_m: &[f32],
    yh_m: &[f32],
    mass_heights_m: &[f32],
    u_heights_m: &[f32],
    v_heights_m: &[f32],
    shape_zyx: [usize; 3],
    moving_frame: TranslatingFrame,
) -> Result<IdealizedInitializationFrame> {
    if !matches!(
        selection.lane(),
        IdealizedBaseStateLane::Isnd5WeismanKlempIwnd2 | IdealizedBaseStateLane::AnalyticBaseF
    ) {
        return Err(invalid(format!(
            "the analytic base-state entry admits base.F's analytic soundings only (isnd=5 and \
             crate::analytic_soundings' arms); got isnd={}, iwnd={}. Running a file lane's \
             selection through this entry would label an analytic base state with a \
             construction that did not produce it",
            selection.requested_isnd(),
            selection.requested_iwnd()
        )));
    }
    let [nz, ny, nx] = shape_zyx;
    if nz == 0 || ny == 0 || nx == 0 || xh_m.len() != nx || yh_m.len() != ny {
        return Err(invalid(
            "analytic domain shape must be positive and match xh/yh coordinate lengths",
        ));
    }
    let words = checked_product(shape_zyx, "analytic domain")?;
    for (label, values) in [
        ("mass heights", mass_heights_m),
        ("u heights", u_heights_m),
        ("v heights", v_heights_m),
    ] {
        if values.len() != words {
            return Err(invalid(format!(
                "analytic domain {label} length {} does not match shape {shape_zyx:?} ({words} words)",
                values.len()
            )));
        }
    }

    let span = domain_span(xh_m, yh_m);
    let mut warm_bubble = build_initial_perturbation(
        selection,
        controls,
        xh_m,
        yh_m,
        mass_heights_m,
        shape_zyx,
        span,
    )?;
    let mut theta_base_k = vec![0.0_f32; words];
    let mut qv_base_kg_per_kg = vec![0.0_f32; words];
    let mut u_base_m_per_s = vec![0.0_f32; words];
    let mut v_base_m_per_s = vec![0.0_f32; words];
    let mut pi_base = vec![0.0_f32; words];
    let mut pressure_base_pa = vec![0.0_f32; words];
    let mut temperature_base_k = vec![0.0_f32; words];
    let mut relative_humidity_base = vec![0.0_f32; words];
    let mut qc_base_kg_per_kg = vec![0.0_f32; words];
    let mut qi_base_kg_per_kg = vec![0.0_f32; words];
    let mut analytic_surface = None;
    let mut mass_column = vec![0.0_f32; nz];

    // base.F:382-383 loops j then i over the whole allocated domain and
    // rebuilds the column from that column's own zh.
    for j in 0..ny {
        for i in 0..nx {
            for k in 0..nz {
                mass_column[k] = mass_heights_m[index3(k, j, i, ny, nx)];
            }
            let column = if selection.lane() == IdealizedBaseStateLane::AnalyticBaseF {
                let words = selection.analytic_words();
                let controls = crate::analytic_soundings::AnalyticSoundingControls {
                    imoist: selection.imoist(),
                    testcase: selection.testcase(),
                    iice: selection.vortex_environment().iice,
                    tsk0_k: words.tsk0_k,
                    thec_mb_k: words.thec_mb_k,
                    qt_mb: words.qt_mb,
                    eqtset: words.eqtset,
                };
                // zh(nk+1): the ghost level above the top, one more layer of
                // the top layer's thickness (param.F's sigma ghost).
                let above = if nz >= 2 {
                    mass_column[nz - 1] + (mass_column[nz - 1] - mass_column[nz - 2])
                } else {
                    mass_column[nz - 1]
                };
                let arm = crate::analytic_soundings::analytic_sounding_column(
                    selection.requested_isnd(),
                    &mass_column,
                    above,
                    controls,
                )?;
                analytic_surface = Some(arm.analytic_surface());
                for k in 0..nz {
                    let index = index3(k, j, i, ny, nx);
                    qc_base_kg_per_kg[index] = arm.qc0[k];
                    qi_base_kg_per_kg[index] = arm.qi0[k];
                }
                // base.F:2104 recomputes prs0 = p00*(pi0**cpdrd) after every
                // arm; the arm's own prs0 (built first, pi0 from it, for
                // isnd = 2, 3 and 21) is not what CM1 keeps.
                let pressure_pa = arm.pi0.iter().map(|pi| P00 * pi.powf(CP / RD)).collect();
                AnalyticColumnWords {
                    theta_k: arm.th0,
                    qv_kg_per_kg: arm.qv0,
                    pi: arm.pi0,
                    pressure_pa,
                    // base.F:2106 recomputes t0 = th0*pi0 for every arm.
                    temperature_k: Vec::new(),
                    relative_humidity: arm.rh0,
                }
            } else {
                let column = weisman_klemp_column(&mass_column, selection.imoist())?;
                AnalyticColumnWords {
                    theta_k: column.theta_k,
                    qv_kg_per_kg: column.qv_kg_per_kg,
                    pi: column.pi,
                    pressure_pa: column.pressure_pa,
                    temperature_k: column.temperature_k,
                    relative_humidity: column.relative_humidity,
                }
            };
            for k in 0..nz {
                let index = index3(k, j, i, ny, nx);
                theta_base_k[index] = column.theta_k[k];
                qv_base_kg_per_kg[index] = column.qv_kg_per_kg[k];
                pi_base[index] = column.pi[k];
                pressure_base_pa[index] = column.pressure_pa[k];
                temperature_base_k[index] = if column.temperature_k.is_empty() {
                    column.theta_k[k] * column.pi[k]
                } else {
                    column.temperature_k[k]
                };
                relative_humidity_base[index] = column.relative_humidity[k];
                // base.F:2286-2640, the selected iwnd branch.
                u_base_m_per_s[index] = base_wind_u(
                    selection.effective_iwnd(),
                    selection.testcase(),
                    u_heights_m[index],
                    selection.hurricane(),
                );
                v_base_m_per_s[index] = base_wind_v(
                    selection.effective_iwnd(),
                    selection.testcase(),
                    v_heights_m[index],
                    selection.hurricane(),
                );
            }
        }
    }

    // base.F:2661-2674.  The source reaches this block after the iwnd=2
    // branch at base.F:2321-2360, so the analytic hodograph is built first
    // and the frame is then subtracted from it exactly once.
    apply_translating_frame(&mut u_base_m_per_s, &mut v_base_m_per_s, moving_frame);

    let initial_qv_kg_per_kg = finish_base_dependent_perturbation(
        selection,
        controls,
        xh_m,
        mass_heights_m,
        &pi_base,
        &pressure_base_pa,
        &theta_base_k,
        &qv_base_kg_per_kg,
        &mut warm_bubble.theta_perturbation_k,
        shape_zyx,
    )?;

    let surface = WeismanKlempSurface::official();
    Ok(IdealizedInitializationFrame {
        selection,
        input_sounding: None,
        // The analytic base.F arms publish their surface through
        // `analytic_surface` (base.F:2212-2246); these two words are the
        // isnd=5 lane's and are not read when that is set.
        surface_pressure_pa: surface.pressure_pa(),
        surface_theta_k: WK_TH_SFC_K,
        shape_zyx,
        mass_heights_m: mass_heights_m.to_vec(),
        theta_base_k,
        qv_base_kg_per_kg,
        u_base_m_per_s,
        v_base_m_per_s,
        pi_base,
        pressure_base_pa,
        temperature_base_k,
        relative_humidity_base,
        warm_bubble,
        initial_qv_kg_per_kg,
        vortex: None,
        moving_frame,
        qc_base_kg_per_kg,
        qi_base_kg_per_kg,
        analytic_surface,
        initial_qc_kg_per_kg: None,
    })
}

/// First-class selectable flat `isnd=5` / `iwnd=2` entry.
///
/// Grid coordinates are the official unstretched horizontal mass centers and
/// source-ordered vertical face midpoints from `param.F:7789-7811`, matching
/// the `isnd=7` flat entry word for word.  The emitted `u0`/`v0` carriers are
/// checked against the native staggered `base.F:2321-2360` arrays before the
/// state is returned.
pub fn initialize_flat_analytic_padded_state(
    selection: IdealizedInitializationSelection,
    controls: WarmBubbleControls,
    shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    origin_xy_m: [f32; 2],
    moving_frame: TranslatingFrame,
) -> Result<IdealizedPaddedState> {
    initialize_analytic_padded_state(
        selection,
        controls,
        shape_zyx,
        spacing_m,
        origin_xy_m,
        moving_frame,
        None,
    )
}

/// The same `isnd=5` / `iwnd=2` entry over a declared vertical coordinate.
///
/// This is the lane a tornado-grade deck runs: the analytic Weisman-Klemp
/// sounding and shear evaluated on `sigma(1..nk)` from `vertical` instead of
/// `param.F:7792,7801`'s uniform midpoints, so the hodograph is sampled at the
/// levels the run actually carries.  The `u0`/`v0` check against the native
/// staggered `base.F:2321-2360` arrays is unchanged and now runs on those
/// levels too.
#[allow(clippy::too_many_arguments)]
pub fn initialize_stretched_analytic_padded_state(
    selection: IdealizedInitializationSelection,
    controls: WarmBubbleControls,
    shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    origin_xy_m: [f32; 2],
    moving_frame: TranslatingFrame,
    vertical: &DeclaredVerticalCoordinate,
) -> Result<IdealizedPaddedState> {
    initialize_analytic_padded_state(
        selection,
        controls,
        shape_zyx,
        spacing_m,
        origin_xy_m,
        moving_frame,
        Some(vertical),
    )
}

#[allow(clippy::too_many_arguments)]
fn initialize_analytic_padded_state(
    selection: IdealizedInitializationSelection,
    controls: WarmBubbleControls,
    shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    origin_xy_m: [f32; 2],
    moving_frame: TranslatingFrame,
    vertical: Option<&DeclaredVerticalCoordinate>,
) -> Result<IdealizedPaddedState> {
    let [nz, ny, nx] = shape_zyx;
    if nz < 4 || ny == 0 || nx == 0 {
        return Err(invalid(
            "flat analytic padded entry requires nz>=4 and positive horizontal extents",
        ));
    }
    for value in origin_xy_m {
        require_finite("analytic domain origin", value)?;
    }
    for (axis, value) in ["dx", "dy", "dz"].into_iter().zip(spacing_m) {
        if !value.is_finite() || value <= 0.0 || (value as f32) as f64 != value {
            return Err(invalid(format!(
                "flat analytic padded {axis} must be positive finite and exactly representable in default REAL"
            )));
        }
    }
    let dx = spacing_m[0] as f32;
    let dy = spacing_m[1] as f32;
    let dz = spacing_m[2] as f32;
    let xh = (0..nx)
        .map(|i| origin_xy_m[0] + (i as f32 + 0.5_f32) * dx)
        .collect::<Vec<_>>();
    let yh = (0..ny)
        .map(|j| origin_xy_m[1] + (j as f32 + 0.5_f32) * dy)
        .collect::<Vec<_>>();
    if let Some(coordinate) = vertical {
        if coordinate.sigma.len() != nz {
            return Err(invalid(format!(
                "declared vertical coordinate has {} scalar levels; this grid has {nz}",
                coordinate.sigma.len()
            )));
        }
        if coordinate.dz_reference_m.to_bits() != dz.to_bits() {
            return Err(invalid(
                "declared vertical dz reference differs from spacing_m[2]; param.F:8170 divides every mh word by exactly one number",
            ));
        }
    }
    let mut heights = vec![0.0_f32; nz * ny * nx];
    for k in 0..nz {
        // param.F:7792,7801 midpoints, or sigma(1..nk) when declared.
        let height = match vertical {
            None => 0.5_f32 * (dz * k as f32 + dz * (k + 1) as f32),
            Some(coordinate) => coordinate.sigma[k],
        };
        for j in 0..ny {
            for i in 0..nx {
                heights[index3(k, j, i, ny, nx)] = height;
            }
        }
    }
    let (zu, zv) = analytic_shear_face_heights(&heights, shape_zyx)?;
    let mut frame = initialize_analytic_idealized_domain(
        selection,
        controls,
        &xh,
        &yh,
        &heights,
        &heights,
        &heights,
        shape_zyx,
        moving_frame,
    )?;
    crate::idealized_initialization::apply_tc_vortex_arm(
        &mut frame, &xh, &yh, [dx, dy, dz], origin_xy_m, vertical,
    )?;
    crate::idealized_initialization::apply_moist_benchmark_arm(&mut frame, controls, &xh)?;
    let state = match vertical {
        None => adapt_flat_idealized_frame_to_padded_state(frame, spacing_m)?,
        Some(coordinate) => {
            adapt_stretched_idealized_frame_to_padded_state(frame, spacing_m, coordinate)?
        }
    };

    // The padded adapter broadcasts one mass-level base wind onto every face.
    // On this flat grid that must equal the native staggered construction; if
    // it ever does not, the emitted hodograph is not the source hodograph.
    let (mut expected_u, mut expected_v) = crate::base_state_wind::build_base_wind_faces(
        selection.effective_iwnd(),
        selection.testcase(),
        selection.hurricane(),
        &zu,
        &zv,
    )?;
    // base.F:2663-2669 subtracts the frame from the native staggered u0/v0
    // too, so the check has to be against the translated hodograph.  Against
    // the untranslated one a moving-frame run would refuse itself.
    apply_translating_frame(&mut expected_u, &mut expected_v, moving_frame);
    for (name, expected, shape) in [
        ("u0", &expected_u, vec![nz, ny, nx + 1]),
        ("v0", &expected_v, vec![nz, ny + 1, nx]),
    ] {
        let carrier = state
            .carrier(name)
            .ok_or_else(|| invalid(format!("analytic padded state lacks {name}")))?;
        if carrier.shape() != shape.as_slice() {
            return Err(invalid(format!(
                "analytic padded {name} shape {:?} is not the native staggered {shape:?}",
                carrier.shape()
            )));
        }
        for (index, (actual, wanted)) in carrier.values().iter().zip(expected).enumerate() {
            if actual.to_bits() != wanted.to_bits() {
                return Err(invalid(format!(
                    "analytic padded {name} word {index} is not the base.F:2286-2640 staggered \
                     value: expected bits {:08x}, got {:08x}",
                    wanted.to_bits(),
                    actual.to_bits()
                )));
            }
        }
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idealized_initialization::{
        initialize_flat_idealized_padded_state, WarmBubbleDisposition,
    };
    use crate::sha256;

    /// The CM1 supercell deck vertical grid: nz=40, dz=500.
    fn deck_heights(nz: usize, dz: f32) -> Vec<f32> {
        (0..nz)
            .map(|k| 0.5_f32 * (dz * k as f32 + dz * (k + 1) as f32))
            .collect()
    }

    fn selection() -> IdealizedInitializationSelection {
        IdealizedInitializationSelection::new(5, 2, 1, 0, 1).unwrap()
    }

    fn raw_sha256(values: &[f32]) -> String {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        sha256::digest_hex(&bytes)
    }

    #[test]
    fn analytic_sounding_profile_matches_hand_evaluated_source_at_named_heights() {
        // Each expected word below is the base.F:389-397 expression written
        // out again from the Fortran, not a value read back from this port.
        let heights = [
            250.0_f32,
            3000.0_f32,
            6000.0_f32,
            11_999.0_f32,
            12_000.0_f32,
            18_000.0_f32,
        ];
        let column = weisman_klemp_column(&heights, 1).unwrap();

        for (k, z) in heights.into_iter().enumerate() {
            let expected_theta = if z < 12_000.0_f32 {
                300.0_f32 + (343.0_f32 - 300.0_f32) * ((z / 12_000.0_f32).powf(1.25_f32))
            } else {
                343.0_f32 * ((9.81_f32 / (213.0_f32 * 1005.7_f32)) * (z - 12_000.0_f32)).exp()
            };
            assert_eq!(
                column.theta_k()[k].to_bits(),
                expected_theta.to_bits(),
                "th0 at z={z}"
            );
        }

        // Independent numeric checks, so a wrong exponent or constant cannot
        // hide behind an identically wrong expression above.
        //   z=6000: 300 + 43*2^-1.25            = 318.07928
        //   z=12000: 343*exp(0)                 = 343
        //   z=18000: 343*exp(9.81*6000/214214.1)= 451.467
        assert!(
            (column.theta_k()[2] - 318.079_28_f32).abs() < 1.0e-3,
            "{:?}",
            column.theta_k()
        );
        assert_eq!(column.theta_k()[4].to_bits(), 343.0_f32.to_bits());
        assert!(
            (column.theta_k()[5] - 451.467_f32).abs() < 1.0e-2,
            "{:?}",
            column.theta_k()
        );

        // rh0 is overwritten at base.F:433-435 by qv0/rslf, so the analytic
        // 1-0.75*(z/z_trop)**1.25 law is checked through qv0 instead.  Below
        // the PBL cap the source law survives; the cap is exact.
        assert_eq!(column.qv_kg_per_kg()[0].to_bits(), WK_QV_PBL.to_bits());
        let saturation = rslf(column.pressure_pa()[2], column.temperature_k()[2]);
        let expected_rh = 1.0_f32 - 0.75_f32 * ((6000.0_f32 / 12_000.0_f32).powf(1.25_f32));
        assert_eq!(
            column.qv_kg_per_kg()[2].to_bits(),
            (expected_rh * saturation).to_bits()
        );
        // z >= z_trop uses the flat rh0 = 0.25 branch.
        let saturation_trop = rslf(column.pressure_pa()[4], column.temperature_k()[4]);
        assert_eq!(
            column.qv_kg_per_kg()[4].to_bits(),
            (0.25_f32 * saturation_trop).to_bits()
        );
        // The branch is strict: 11999 m is still the troposphere law.
        assert!(column.theta_k()[3] < 343.0_f32);
    }

    #[test]
    fn analytic_surface_words_match_the_source() {
        let surface = WeismanKlempSurface::official();
        // prs_sfc == p00, so pi_sfc is exactly one (base.F:378).
        assert_eq!(surface.pi().to_bits(), 1.0_f32.to_bits());
        assert_eq!(
            surface.qv_kg_per_kg().to_bits(),
            rslf(100_000.0_f32, 300.0_f32 * 1.0_f32).to_bits()
        );
        let reps = RV / RD;
        let qv = surface.qv_kg_per_kg();
        assert_eq!(
            surface.theta_v_k().to_bits(),
            (300.0_f32 * (1.0_f32 + qv * reps) / (1.0_f32 + qv)).to_bits()
        );
        assert_eq!(surface.pressure_pa().to_bits(), 100_000.0_f32.to_bits());
        assert_eq!(surface.temperature_k().to_bits(), 300.0_f32.to_bits());

        // base.F:442-450: below the PBL cap the surface value extrapolates,
        // at or above it the cap is used verbatim.
        let column = weisman_klemp_column(&deck_heights(40, 500.0_f32), 1).unwrap();
        assert_eq!(column.qv_kg_per_kg()[0].to_bits(), WK_QV_PBL.to_bits());
        assert_eq!(
            column
                .surface_qv_kg_per_kg([1.875, -1.25, 0.375])
                .unwrap()
                .to_bits(),
            WK_QV_PBL.to_bits()
        );
        let dry = weisman_klemp_column(&deck_heights(40, 500.0_f32), 0).unwrap();
        assert_eq!(
            dry.surface_qv_kg_per_kg([1.875, -1.25, 0.375]).unwrap(),
            0.0_f32
        );
        assert!(dry.qv_kg_per_kg().iter().all(|value| *value == 0.0_f32));
    }

    #[test]
    fn hydrostatic_fixed_point_is_settled_at_the_source_iteration_count() {
        let heights = deck_heights(40, 500.0_f32);
        let twenty = weisman_klemp_column(&heights, 1).unwrap();
        assert_eq!(twenty.iterations(), WK_HYDROSTATIC_ITERATIONS);
        assert_eq!(WK_HYDROSTATIC_ITERATIONS, 20);
        // The 20th iteration changes nothing at all in binary32.
        assert_eq!(twenty.final_qv_update_linf(), 0.0_f32);
        assert_eq!(twenty.final_pi_update_linf(), 0.0_f32);

        // Re-running the fixed point past CM1's count reproduces it bit for bit.
        for extra in [21_usize, 40, 60] {
            let more = weisman_klemp_column_with_iterations(&heights, 1, extra).unwrap();
            for (name, a, b) in [
                ("qv0", twenty.qv_kg_per_kg(), more.qv_kg_per_kg()),
                ("pi0", twenty.pi(), more.pi()),
                ("prs0", twenty.pressure_pa(), more.pressure_pa()),
                ("thv0", twenty.theta_v_k(), more.theta_v_k()),
                ("rh0", twenty.relative_humidity(), more.relative_humidity()),
            ] {
                assert!(
                    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{name} moved between 20 and {extra} iterations"
                );
            }
        }
        // ...and the count is load bearing: one pass is a different state.
        let one = weisman_klemp_column_with_iterations(&heights, 1, 1).unwrap();
        assert!(one
            .qv_kg_per_kg()
            .iter()
            .zip(twenty.qv_kg_per_kg())
            .any(|(x, y)| x.to_bits() != y.to_bits()));
        assert!(weisman_klemp_column_with_iterations(&heights, 1, 0).is_err());

        // The emitted column is a monotone hydrostatic base state.
        for k in 1..heights.len() {
            assert!(twenty.pi()[k] < twenty.pi()[k - 1], "pi0 rose at k={k}");
            assert!(
                twenty.pressure_pa()[k] < twenty.pressure_pa()[k - 1],
                "prs0 rose at k={k}"
            );
        }
        assert!(twenty.pi()[0] < WeismanKlempSurface::official().pi());
        assert!((twenty.pressure_pa()[0] - 97_212.0_f32).abs() < 5.0_f32);

        assert_eq!(
            raw_sha256(twenty.qv_kg_per_kg()),
            "b0da826d09cc5ebb5acdea6bd8d69368ccb151db6ede8c2ec56dd12a1f1ac0d9",
            "sealed base.F:363-450 qv0 words changed"
        );
        assert_eq!(
            raw_sha256(twenty.pi()),
            "add8a472815e4688c9f3e4fea10ea1a2ea5e54269b7878704eedfd9f3041f0de",
            "sealed base.F:363-450 pi0 words changed"
        );
    }

    #[test]
    fn analytic_wind_profile_matches_hand_evaluated_source_at_named_heights() {
        // base.F:2332-2339 and base.F:2350-2357, written out from the Fortran.
        for z in [0.0_f32, 500.0, 1000.0, 1999.0, 2000.0] {
            let angle = 90.0_f32 * (z / 2000.0_f32) * (PI / 180.0_f32);
            assert_eq!(
                analytic_shear_u(z).to_bits(),
                (7.0_f32 - 7.0_f32 * angle.cos()).to_bits(),
                "u0 at z={z}"
            );
            assert_eq!(
                analytic_shear_v(z).to_bits(),
                (7.0_f32 * angle.sin()).to_bits(),
                "v0 at z={z}"
            );
        }
        for z in [2001.0_f32, 3000.0, 4000.0, 6000.0] {
            assert_eq!(
                analytic_shear_u(z).to_bits(),
                (7.0_f32 + (z - 2000.0_f32) * (31.0_f32 - 7.0_f32) / (6000.0_f32 - 2000.0_f32))
                    .to_bits(),
                "u0 at z={z}"
            );
            assert_eq!(analytic_shear_v(z).to_bits(), 7.0_f32.to_bits());
        }

        // Independent numeric checks of the named turning points.
        assert_eq!(analytic_shear_u(0.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(analytic_shear_v(0.0).to_bits(), 0.0_f32.to_bits());
        // z=1000: u = 7-7*cos(45 deg) = 2.0502524, v = 7*sin(45 deg) = 4.9497476
        assert!((analytic_shear_u(1000.0) - 2.050_252_4_f32).abs() < 1.0e-5);
        assert!((analytic_shear_v(1000.0) - 4.949_747_6_f32).abs() < 1.0e-5);
        // z=2000: the quarter cosine has turned through 90 deg.
        assert!((analytic_shear_u(2000.0) - 7.0_f32).abs() < 1.0e-5);
        assert_eq!(analytic_shear_v(2000.0).to_bits(), 7.0_f32.to_bits());
        // z=4000: the midpoint of the linear ramp.
        assert_eq!(analytic_shear_u(4000.0).to_bits(), 19.0_f32.to_bits());
        // z=6000 and above: constant umax2, and v holds vmax1 forever.
        assert_eq!(analytic_shear_u(6000.0).to_bits(), 31.0_f32.to_bits());
        for z in [6000.5_f32, 10_000.0, 20_000.0] {
            assert_eq!(analytic_shear_u(z).to_bits(), 31.0_f32.to_bits());
            assert_eq!(analytic_shear_v(z).to_bits(), 7.0_f32.to_bits());
        }
        // 0-6 km shear vector, from the emitted profile alone.
        assert!((analytic_shear_u(6000.0) - analytic_shear_u(0.0) - 31.0_f32).abs() < 1.0e-5);
        assert!((analytic_shear_v(6000.0) - analytic_shear_v(0.0) - 7.0_f32).abs() < 1.0e-5);
    }

    #[test]
    fn shear_face_heights_use_the_source_u_and_v_stagger() {
        // A height field that varies in i and j, so the two stagger axes
        // cannot be confused for one another.
        let shape = [2_usize, 3, 4];
        let [nz, ny, nx] = shape;
        let mut zh = vec![0.0_f32; nz * ny * nx];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    zh[index3(k, j, i, ny, nx)] =
                        1000.0_f32 * (k + 1) as f32 + 10.0_f32 * j as f32 + i as f32;
                }
            }
        }
        let (zu, zv) = analytic_shear_face_heights(&zh, shape).unwrap();
        assert_eq!(zu.len(), nz * ny * (nx + 1));
        assert_eq!(zv.len(), nz * (ny + 1) * nx);

        for k in 0..nz {
            for j in 0..ny {
                let base = 1000.0_f32 * (k + 1) as f32 + 10.0_f32 * j as f32;
                // i=1 in Fortran: 0.5*(zh(0)+zh(1)), the ghost column copy.
                assert_eq!(zu[index3(k, j, 0, ny, nx + 1)].to_bits(), base.to_bits());
                for face in 1..nx {
                    let expected = base + face as f32 - 0.5_f32;
                    assert_eq!(
                        zu[index3(k, j, face, ny, nx + 1)].to_bits(),
                        expected.to_bits(),
                        "zu at k={k},j={j},face={face}"
                    );
                }
                // i=ni+1 in Fortran: 0.5*(zh(ni)+zh(ni+1)), the top ghost.
                assert_eq!(
                    zu[index3(k, j, nx, ny, nx + 1)].to_bits(),
                    (base + (nx - 1) as f32).to_bits()
                );
            }
            for i in 0..nx {
                let base = 1000.0_f32 * (k + 1) as f32 + i as f32;
                assert_eq!(zv[index3(k, 0, i, ny + 1, nx)].to_bits(), base.to_bits());
                for face in 1..ny {
                    let expected = base + 10.0_f32 * face as f32 - 5.0_f32;
                    assert_eq!(
                        zv[index3(k, face, i, ny + 1, nx)].to_bits(),
                        expected.to_bits(),
                        "zv at k={k},face={face},i={i}"
                    );
                }
                assert_eq!(
                    zv[index3(k, ny, i, ny + 1, nx)].to_bits(),
                    (base + 10.0_f32 * (ny - 1) as f32).to_bits()
                );
            }
        }

        let (u0, v0) = build_analytic_shear_faces(&zh, shape).unwrap();
        assert_eq!(u0.len(), zu.len());
        assert_eq!(v0.len(), zv.len());
        for (value, height) in u0.iter().zip(&zu) {
            assert_eq!(value.to_bits(), analytic_shear_u(*height).to_bits());
        }
        for (value, height) in v0.iter().zip(&zv) {
            assert_eq!(value.to_bits(), analytic_shear_v(*height).to_bits());
        }

        // On the flat grid the admitted lane uses, both stagger averages
        // collapse onto the mass height, which is why the padded adapter's
        // broadcast is exact there.
        let flat = {
            let mut flat = vec![0.0_f32; nz * ny * nx];
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        flat[index3(k, j, i, ny, nx)] = 250.0_f32 + 500.0_f32 * k as f32;
                    }
                }
            }
            flat
        };
        let (flat_zu, flat_zv) = analytic_shear_face_heights(&flat, shape).unwrap();
        for k in 0..nz {
            let height = 250.0_f32 + 500.0_f32 * k as f32;
            assert!(flat_zu[k * ny * (nx + 1)..(k + 1) * ny * (nx + 1)]
                .iter()
                .all(|value| value.to_bits() == height.to_bits()));
            assert!(flat_zv[k * (ny + 1) * nx..(k + 1) * (ny + 1) * nx]
                .iter()
                .all(|value| value.to_bits() == height.to_bits()));
        }
    }

    #[test]
    fn unadmitted_isnd_and_iwnd_pairs_are_refused_by_value() {
        assert_eq!(
            IdealizedInitializationSelection::new(5, 2, 1, 0, 1)
                .unwrap()
                .lane(),
            IdealizedBaseStateLane::Isnd5WeismanKlempIwnd2
        );
        assert_eq!(
            IdealizedInitializationSelection::new(5, 2, 1, 0, 1)
                .unwrap()
                .effective_iwnd(),
            2
        );
        // base.F:2286-2640's eleven branches are all selectable under isnd=5;
        // the one number the chain has no arm for is refused by value.
        for iwnd in [0_i32, 1, 3, 4, 5, 6, 8, 9, 10, 11] {
            let selection = IdealizedInitializationSelection::new(5, iwnd, 1, 0, 1).unwrap();
            assert_eq!(selection.effective_iwnd(), iwnd);
        }
        for iwnd in [-1_i32, 7, 12] {
            let error = IdealizedInitializationSelection::new(5, iwnd, 1, 0, 1)
                .unwrap_err()
                .to_string();
            assert!(error.contains("motionless environment"), "{error}");
        }
        for (isnd, iwnd) in [(6_i32, 2_i32), (15, 2), (0, 0), (-1, 2), (16, 2)] {
            let error = IdealizedInitializationSelection::new(isnd, iwnd, 1, 0, 1)
                .unwrap_err()
                .to_string();
            assert!(error.contains(&format!("isnd={isnd}")), "{error}");
        }

        // param.F:836-852 is preserved exactly: isnd=7 still forces iwnd=0,
        // for every nonnegative iwnd, and still refuses a negative one.
        for iwnd in [0_i32, 1, 2, 4, 9] {
            let selection = IdealizedInitializationSelection::new(7, iwnd, 1, 0, 1).unwrap();
            assert_eq!(
                selection.lane(),
                IdealizedBaseStateLane::Isnd7ExternalSounding
            );
            assert_eq!(selection.requested_iwnd(), iwnd);
            assert_eq!(selection.effective_iwnd(), 0);
        }
        assert!(IdealizedInitializationSelection::new(7, -1, 1, 0, 1).is_err());

        // The analytic entry refuses another lane's selection.
        let error = initialize_analytic_idealized_domain(
            IdealizedInitializationSelection::new(7, 0, 1, 0, 1).unwrap(),
            WarmBubbleControls::official(0.0, 0.0).unwrap(),
            &[0.0],
            &[0.0],
            &[250.0],
            &[250.0],
            &[250.0],
            [1, 1, 1],
            TranslatingFrame::fixed(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("analytic soundings only"), "{error}");

        // Non-increasing heights would invert the hydrostatic step.
        let error = weisman_klemp_column(&[250.0, 250.0], 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("strictly increasing"), "{error}");
        assert!(weisman_klemp_column(&[0.0, 500.0], 1).is_err());
        let error = weisman_klemp_column(&[250.0, 750.0], 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("imoist=0 or 1 only"), "{error}");
    }

    /// CM1's own supercell deck: `run/namelist.input:83,101,102`.
    const DECK_UMOVE: f32 = 12.5_f32;
    const DECK_VMOVE: f32 = 3.0_f32;

    fn analytic_state(frame: TranslatingFrame) -> IdealizedPaddedState {
        initialize_flat_analytic_padded_state(
            selection(),
            WarmBubbleControls::official(2000.0, 1500.0).unwrap(),
            [40_usize, 3, 4],
            [1000.0, 1000.0, 500.0],
            [0.0, 0.0],
            frame,
        )
        .unwrap()
    }

    #[test]
    fn a_fixed_frame_leaves_the_analytic_lane_raw_bit_identical() {
        // The whole reason zero is a default: it must not be a code path.
        let fixed = analytic_state(TranslatingFrame::fixed());
        let zero = analytic_state(TranslatingFrame::new(0.0, 0.0).unwrap());
        assert_eq!(fixed.carrier_digest_sha256(), zero.carrier_digest_sha256());
        assert!(!TranslatingFrame::fixed().is_active());
        assert_eq!(TranslatingFrame::fixed().imove(), 0);
        assert_eq!(
            TranslatingFrame::new(DECK_UMOVE, DECK_VMOVE)
                .unwrap()
                .imove(),
            1
        );
        // A signed-zero speed must skip the subtraction rather than flip a
        // -0.0 base-state word to +0.0, which `x - (-0.0)` would do.
        let negative_zero = analytic_state(TranslatingFrame::new(-0.0, -0.0).unwrap());
        assert_eq!(
            fixed.carrier_digest_sha256(),
            negative_zero.carrier_digest_sha256()
        );
    }

    #[test]
    fn the_analytic_lane_offsets_every_staggered_face_by_exactly_the_frame() {
        // base.F:2663-2669 on the iwnd=2 hodograph of base.F:2321-2360.
        let fixed = analytic_state(TranslatingFrame::fixed());
        let moving = analytic_state(TranslatingFrame::new(DECK_UMOVE, DECK_VMOVE).unwrap());
        assert_eq!(moving.moving_frame().u_move_m_per_s(), DECK_UMOVE);
        assert_eq!(moving.moving_frame().v_move_m_per_s(), DECK_VMOVE);

        // u lives on nx+1 faces and v on ny+1; both must move, every word.
        let u_still = fixed.carrier("u0").unwrap();
        let u_moved = moving.carrier("u0").unwrap();
        assert_eq!(u_still.shape(), &[40, 3, 5]);
        assert_eq!(u_moved.shape(), u_still.shape());
        for (index, (still, moved)) in u_still.values().iter().zip(u_moved.values()).enumerate() {
            assert_eq!(
                moved.to_bits(),
                (still - DECK_UMOVE).to_bits(),
                "u0 word {index}"
            );
        }
        let v_still = fixed.carrier("v0").unwrap();
        let v_moved = moving.carrier("v0").unwrap();
        assert_eq!(v_still.shape(), &[40, 4, 4]);
        for (index, (still, moved)) in v_still.values().iter().zip(v_moved.values()).enumerate() {
            assert_eq!(
                moved.to_bits(),
                (still - DECK_VMOVE).to_bits(),
                "v0 word {index}"
            );
        }

        // init3d.F:210-219 copies u0/v0 into ua/u3d and va/v3d, so the
        // initial condition carries the same offset from the same
        // subtraction.  A base state that moved while ua did not would be a
        // sheared reference profile that is not the declared hodograph.
        //
        // These carriers are padded.  The physical rectangle is written from
        // the base wind and the lateral halo is then filled by the bc.F
        // wbc=2 zero-gradient copy, so both must carry the offset; only the
        // allocation-zero remainder may stay put.  Every analytic wind word
        // on this grid is strictly positive, so "nonzero in the fixed run"
        // is exactly "written", and the assertion is that every written word
        // moved by exactly the constant and nothing else moved at all.
        for (name, offset, physical_words) in [
            ("ua", DECK_UMOVE, 40 * 3 * 5_usize),
            ("u3d", DECK_UMOVE, 40 * 3 * 5),
            ("va", DECK_VMOVE, 40 * 4 * 4),
            ("v3d", DECK_VMOVE, 40 * 4 * 4),
        ] {
            let still = fixed.carrier(name).unwrap().values();
            let moved = moving.carrier(name).unwrap().values();
            assert_eq!(still.len(), moved.len(), "{name} length");
            let mut offset_words = 0_usize;
            for (index, (a, b)) in still.iter().zip(moved).enumerate() {
                if *a == 0.0_f32 {
                    assert_eq!(b.to_bits(), a.to_bits(), "{name} unwritten word {index}");
                    continue;
                }
                assert_eq!(b.to_bits(), (a - offset).to_bits(), "{name} word {index}");
                offset_words += 1;
            }
            // Strictly more than the physical faces: the open-boundary halo
            // moved too, which is what keeps the lateral condition in the
            // same frame as the interior.
            assert!(
                offset_words > physical_words,
                "{name} moved {offset_words} words, not more than the {physical_words} physical faces"
            );
        }
    }

    #[test]
    fn the_frame_velocity_is_refused_by_name_and_value() {
        for (name, u, v) in [
            ("umove", f32::NAN, 0.0_f32),
            ("umove", f32::INFINITY, 0.0_f32),
            ("vmove", 0.0_f32, f32::NEG_INFINITY),
        ] {
            let error = TranslatingFrame::new(u, v).unwrap_err().to_string();
            assert!(error.contains(name), "{error}");
            assert!(error.contains("not finite"), "{error}");
            assert!(error.contains("base.F:2667"), "{error}");
        }
        for (name, u, v) in [
            ("umove", 151.0_f32, 0.0_f32),
            ("vmove", 0.0_f32, -1.0e6_f32),
        ] {
            let error = TranslatingFrame::new(u, v).unwrap_err().to_string();
            assert!(error.contains(name), "{error}");
            assert!(error.contains("150"), "{error}");
        }
        // The bound itself is admitted, not refused.
        assert!(TranslatingFrame::new(150.0, -150.0).is_ok());
    }

    #[test]
    fn flat_analytic_padded_state_is_selectable_and_sealed() {
        // The CM1 supercell deck's vertical grid at a small horizontal size.
        let shape = [40_usize, 3, 4];
        let state = initialize_flat_analytic_padded_state(
            selection(),
            WarmBubbleControls::official(2000.0, 1500.0).unwrap(),
            shape,
            [1000.0, 1000.0, 500.0],
            [0.0, 0.0],
            TranslatingFrame::fixed(),
        )
        .unwrap();
        assert_eq!(
            state.selection().lane(),
            IdealizedBaseStateLane::Isnd5WeismanKlempIwnd2
        );
        // No external byte stream exists on this lane, so none is claimed.
        assert_eq!(state.input_sounding_sha256(), None);
        assert_eq!(
            state.warm_bubble_disposition(),
            WarmBubbleDisposition::AppliedIinit1
        );
        assert_eq!(state.pi_recovery_mismatch_count(), 0);
        assert_eq!(state.theta_recovery_mismatch_count(), 0);
        assert_eq!(state.terrain_open_carriers().len(), 54);
        assert_eq!(state.base_wind_carriers().len(), 2);
        assert_eq!(state.moist_carriers().len(), 4);
        assert_eq!(state.carrier("u0").unwrap().shape(), &[40, 3, 5]);
        assert_eq!(state.carrier("v0").unwrap().shape(), &[40, 4, 4]);

        // u0/v0 carry the analytic profile on the source stagger; these are
        // the words initialize_flat_analytic_padded_state re-derives and
        // compares before returning, asserted again here from the outside.
        let u0 = state.carrier("u0").unwrap().values();
        let v0 = state.carrier("v0").unwrap().values();
        for k in 0..40 {
            let height = 250.0_f32 + 500.0_f32 * k as f32;
            for word in &u0[k * 3 * 5..(k + 1) * 3 * 5] {
                assert_eq!(word.to_bits(), analytic_shear_u(height).to_bits());
            }
            for word in &v0[k * 4 * 4..(k + 1) * 4 * 4] {
                assert_eq!(word.to_bits(), analytic_shear_v(height).to_bits());
            }
        }
        // k=0 is 250 m, k=11 is 5750 m, k=12 is 6250 m -> the constant layer.
        assert!((u0[0] - analytic_shear_u(250.0)).abs() < 1.0e-6);
        assert_eq!(u0[12 * 3 * 5].to_bits(), 31.0_f32.to_bits());

        // th0/qv0/pi0 in the padded carriers are the analytic column.
        let column = weisman_klemp_column(&deck_heights(40, 500.0_f32), 1).unwrap();
        let halo_ny = 3 + 6;
        let halo_nx = 4 + 6;
        for (name, expected) in [
            ("th0", column.theta_k()),
            ("qv0", column.qv_kg_per_kg()),
            ("pi0", column.pi()),
            ("prs0", column.pressure_pa()),
        ] {
            let values = state.carrier(name).unwrap().values();
            for (k, word) in expected.iter().enumerate() {
                let index = index3(k + 1, 3, 3, halo_ny, halo_nx);
                assert_eq!(values[index].to_bits(), word.to_bits(), "{name} at k={k}");
            }
        }

        assert_eq!(
            state.carrier_digest_sha256(),
            "af4bae4444512d21d2a87c3125fdd6708bdc6af778571fcba27902d827bc64ad",
            "sealed isnd=5/iwnd=2 padded carrier bytes changed"
        );
    }

    #[test]
    fn the_analytic_lane_leaves_the_external_sounding_lane_untouched() {
        const SOUNDING: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../upstream/cm1-21.1/pristine/run/config_files/sea_breeze/input_sounding"
        ));
        let external = initialize_flat_idealized_padded_state(
            SOUNDING,
            IdealizedInitializationSelection::new(7, 4, 1, 0, 1).unwrap(),
            WarmBubbleControls::official(1500.0, 1000.0).unwrap(),
            [6, 3, 4],
            [1000.0, 1000.0, 500.0],
            [0.0, 0.0],
            TranslatingFrame::fixed(),
        )
        .unwrap();
        // The same fence the isnd=7 module seals, asserted from this module
        // so adding the analytic lane cannot move it unnoticed.
        assert_eq!(
            external.carrier_digest_sha256(),
            "dcfda0d231693937e2a8874bfd51d78f598255e3cc62a5b9cbd70277c54d75a3"
        );
        assert_eq!(
            external.selection().lane(),
            IdealizedBaseStateLane::Isnd7ExternalSounding
        );
        assert!(external.input_sounding_sha256().is_some());

        let analytic = initialize_flat_analytic_padded_state(
            selection(),
            WarmBubbleControls::official(1500.0, 1000.0).unwrap(),
            [6, 3, 4],
            [1000.0, 1000.0, 500.0],
            [0.0, 0.0],
            TranslatingFrame::fixed(),
        )
        .unwrap();
        assert_ne!(
            analytic.carrier_digest_sha256(),
            external.carrier_digest_sha256()
        );
        // Same carrier names and shapes, different words: the two lanes fill
        // one layout rather than defining two.
        for (a, b) in analytic
            .terrain_open_carriers()
            .iter()
            .zip(external.terrain_open_carriers())
        {
            assert_eq!(a.name(), b.name());
            assert_eq!(a.shape(), b.shape());
        }
        // The external sea-breeze sounding is windless; the analytic lane is
        // not.  That difference is the point of this lane.
        assert!(external
            .carrier("u0")
            .unwrap()
            .values()
            .iter()
            .all(|value| *value == 0.0_f32));
        assert!(analytic
            .carrier("u0")
            .unwrap()
            .values()
            .iter()
            .any(|value| *value != 0.0_f32));
    }
}
