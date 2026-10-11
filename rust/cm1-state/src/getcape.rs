//! Faithful transcription of CM1 r21.1 `src/getcape.F`, plus the hodograph
//! parameters CM1 does not compute.
//!
//! `getcape.F` is George Bryan's parcel integrator, called from two places in
//! CM1 r21.1:
//!
//! * `src/base.F:3165-3167` runs it three times over the base-state sounding
//!   at `i=1,j=1` -- once per parcel source -- and prints the source parcel,
//!   LCL/LFC/EL and CAPE/CIN (`base.F:3175-3180`).  That is the t=0 report.
//! * `src/writeout.F:2921-2922` runs it per column with `source=3` and writes
//!   the four 2-D fields the `output_cape`/`output_cin`/`output_lcl`/
//!   `output_lfc` switches select (`writeout.F:2937-3028`).
//!
//! Everything in [`getcape`] below is `getcape.F` line for line in binary32,
//! with the source line cited at each step.  The saturation formula is
//! `getcape.F:526-538` (`getqvl`), which is character for character the
//! Bolton branch GM1 already uses for its microphysics: see
//! `idealized_initialization::rslf`, including the `min(esl, p*0.5)` clamp
//! added to CM1 on 171023 (`getcape.F:535`).
//!
//! The hodograph half of this module is *not* CM1.  CM1 r21.1 computes no
//! storm motion, no bulk shear and no storm-relative helicity anywhere in its
//! source tree, so those are implemented here from the published method and
//! named as such in every record: Bunkers et al. (2000, Wea. Forecasting 15,
//! 61-79) internal-dynamics right and left movers, with the 7.5 m/s deviation
//! and the 0-6 km mean wind / (5.5-6 km minus 0-0.5 km) shear vector the paper
//! specifies.  A reader of a receipt is told the method by name rather than
//! left to guess which storm motion the helicity was taken against.

use crate::{Result, StateError};
use std::fmt::Write as _;

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

/// `getcape.F:100-114`.  `getcape` declares its own constants rather than
/// using `constants.F`; they are transcribed here with the same values and the
/// same spelling so the comparison with the source is mechanical.
pub const G: f32 = 9.81;
pub const P00: f32 = 100_000.0;
pub const CP: f32 = 1005.7;
pub const RD: f32 = 287.04;
pub const RV: f32 = 461.5;
pub const XLV: f32 = 2_501_000.0;
pub const XLS: f32 = 2_836_017.0;
pub const T0: f32 = 273.15;
pub const CPV: f32 = 1875.0;
pub const CPL: f32 = 4190.0;
pub const CPI: f32 = 2118.636;
pub const LV1: f32 = XLV + (CPL - CPV) * T0;
pub const LV2: f32 = CPL - CPV;
pub const LS1: f32 = XLS + (CPI - CPV) * T0;
pub const LS2: f32 = CPI - CPV;

/// `getcape.F:116-121`.
pub const RP00: f32 = 1.0 / P00;
pub const EPS: f32 = RD / RV;
pub const REPS: f32 = RV / RD;
pub const RDDCP: f32 = RD / CP;
pub const CPDG: f32 = CP / G;

/// `getcape.F:123`.
pub const CONVERGE: f32 = 0.0002;
/// `getcape.F:60`.  The default pressure increment of the parcel ascent.
pub const DEFAULT_PRESSURE_INCREMENT_PA: f32 = 200.0;
/// `getcape.F:70`.  The default mixed-layer depth for `source=3`.
pub const DEFAULT_MIXED_LAYER_DEPTH_M: f32 = 500.0;
/// `getcape.F:383`.  The iteration count at which `getcape.F` abandons the
/// moist-adiabat fixed point and returns zeros.
pub const MAX_ADIABAT_ITERATIONS: u32 = 100;

/// Bunkers et al. (2000) internal-dynamics deviation magnitude, m/s.
pub const BUNKERS_DEVIATION_M_PER_S: f32 = 7.5;
/// Bunkers et al. (2000) mean-wind layer top, m AGL.
pub const BUNKERS_MEAN_WIND_TOP_M: f32 = 6000.0;
/// Bunkers et al. (2000) shear-vector lower layer, m AGL.
pub const BUNKERS_SHEAR_LOWER_TOP_M: f32 = 500.0;
/// Bunkers et al. (2000) shear-vector upper layer, m AGL.
pub const BUNKERS_SHEAR_UPPER_BASE_M: f32 = 5500.0;

/// `getcape.F:65-68`.  The three parcel definitions CM1 r21.1 offers, and the
/// only three: `getcape.F:242-250` stops the model on any other value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParcelSource {
    /// `getcape.F:157-160`.  The lowest level of the supplied column.
    Surface = 1,
    /// `getcape.F:161-181`.  Maximum `getthx` theta-e at or below 500 mb.
    MostUnstable = 2,
    /// `getcape.F:183-241`.  Depth-weighted mean theta and qv over
    /// `ml_depth` metres, lifted from the lowest level.
    MixedLayer = 3,
}

impl ParcelSource {
    pub fn from_i32(value: i32) -> Result<Self> {
        match value {
            1 => Ok(Self::Surface),
            2 => Ok(Self::MostUnstable),
            3 => Ok(Self::MixedLayer),
            other => Err(invalid(format!(
                "getcape.F:242-250 stops CM1 on source={other}; the three \
                 transcribed parcel definitions are 1=surface, 2=most-unstable \
                 (max theta-e below 500 mb), 3=mixed-layer"
            ))),
        }
    }

    pub fn as_i32(self) -> i32 {
        self as i32
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::MostUnstable => "most-unstable",
            Self::MixedLayer => "mixed-layer",
        }
    }
}

/// `getcape.F:73-77`.  The four moist-adiabat formulations.  `getcape.F:73`
/// ships `adiabat = 1` -- pseudoadiabatic, liquid only -- and that is what
/// both CM1 call sites therefore use; the other three are transcribed because
/// `getcape.F` implements them, and `getcape.F:419-423` stops the model on
/// anything outside 1..4.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Adiabat {
    /// `getcape.F:74`.
    PseudoadiabaticLiquid = 1,
    /// `getcape.F:75`.
    ReversibleLiquid = 2,
    /// `getcape.F:76`.
    PseudoadiabaticIce = 3,
    /// `getcape.F:77`.
    ReversibleIce = 4,
}

impl Adiabat {
    pub fn from_i32(value: i32) -> Result<Self> {
        match value {
            1 => Ok(Self::PseudoadiabaticLiquid),
            2 => Ok(Self::ReversibleLiquid),
            3 => Ok(Self::PseudoadiabaticIce),
            4 => Ok(Self::ReversibleIce),
            other => Err(invalid(format!(
                "getcape.F:419-423 stops CM1 on adiabat={other}; the four \
                 transcribed formulations are 1=pseudoadiabatic liquid, \
                 2=reversible liquid, 3=pseudoadiabatic with ice, \
                 4=reversible with ice"
            ))),
        }
    }

    pub fn as_i32(self) -> i32 {
        self as i32
    }

    /// `getcape.F:290-294`.
    pub fn ice(self) -> bool {
        matches!(self, Self::PseudoadiabaticIce | Self::ReversibleIce)
    }

    /// `getcape.F:414-418`.
    pub fn pseudoadiabatic(self) -> bool {
        matches!(
            self,
            Self::PseudoadiabaticLiquid | Self::PseudoadiabaticIce
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::PseudoadiabaticLiquid => "pseudoadiabatic-liquid",
            Self::ReversibleLiquid => "reversible-liquid",
            Self::PseudoadiabaticIce => "pseudoadiabatic-ice",
            Self::ReversibleIce => "reversible-ice",
        }
    }
}

/// `getcape.F:526-538`.  Saturation mixing ratio over liquid, Bolton (1980)
/// with the 171023 `min(es, p*0.5)` clamp.  Identical in form and constants to
/// `idealized_initialization::rslf`, which is what GM1's WSM6 seam already
/// uses, so the parcel integration and the microphysics saturate on the same
/// curve.
pub fn getqvl(p: f32, t: f32) -> f32 {
    let es = 611.2_f32 * (17.67_f32 * (t - 273.15_f32) / (t - 29.65_f32)).exp();
    let es = es.min(p * 0.5_f32);
    EPS * es / (p - es)
}

/// `getcape.F:544-556`.  Saturation mixing ratio over ice.
pub fn getqvi(p: f32, t: f32) -> f32 {
    let es = 611.2_f32 * (21.874_5584_f32 * (t - 273.15_f32) / (t - 7.66_f32)).exp();
    let es = es.min(p * 0.5_f32);
    EPS * es / (p - es)
}

/// `getcape.F:562-577`.  Bolton (1980) equivalent potential temperature.
pub fn getthx(p: f32, t: f32, td: f32, q: f32) -> f32 {
    let tlcl = if (td - t) >= -0.1_f32 {
        t
    } else {
        56.0_f32 + (1.0_f32 / (td - 56.0_f32) + 0.00125_f32 * (t / td).ln()).recip()
    };
    t * (100_000.0_f32 / p).powf(0.2854_f32 * (1.0_f32 - 0.28_f32 * q))
        * (((3376.0_f32 / tlcl) - 2.54_f32) * q * (1.0_f32 + 0.81_f32 * q)).exp()
}

/// `getcape.F:583-594`.  Dewpoint from pressure and mixing ratio.
pub fn gettd(p: f32, _t: f32, q: f32) -> f32 {
    let el = ((q / EPS) * p / 100.0_f32 / (1.0_f32 + (q / EPS))).ln();
    273.15_f32 + (243.5_f32 * el - 440.8_f32) / (19.48_f32 - el)
}

/// The controls `getcape.F` exposes as source parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GetCapeControls {
    pub source: ParcelSource,
    pub adiabat: Adiabat,
    /// `getcape.F:70`.
    pub mixed_layer_depth_m: f32,
    /// `getcape.F:60`.
    pub pressure_increment_pa: f32,
}

impl Default for GetCapeControls {
    /// `writeout.F:2921` fixes `source=3`; `getcape.F:70,73` fix the mixed
    /// layer depth and the adiabat; `getcape.F:60` fixes the increment.
    fn default() -> Self {
        Self {
            source: ParcelSource::MixedLayer,
            adiabat: Adiabat::PseudoadiabaticLiquid,
            mixed_layer_depth_m: DEFAULT_MIXED_LAYER_DEPTH_M,
            pressure_increment_pa: DEFAULT_PRESSURE_INCREMENT_PA,
        }
    }
}

impl GetCapeControls {
    pub fn validate(&self) -> Result<()> {
        if !self.mixed_layer_depth_m.is_finite() || self.mixed_layer_depth_m <= 0.0 {
            return Err(invalid(format!(
                "getcape.F:70 divides the depth-weighted mixed-layer sums by \
                 ml_depth at getcape.F:233-234, so ml_depth={} would make the \
                 source parcel theta and qv infinite or NaN and report a CAPE \
                 that describes no atmosphere",
                self.mixed_layer_depth_m
            )));
        }
        if !self.pressure_increment_pa.is_finite() || self.pressure_increment_pa <= 0.0 {
            return Err(invalid(format!(
                "getcape.F:328-333 divides each layer's pressure drop by pinc \
                 to choose the sub-step count, so pinc={} would make the ascent \
                 take zero or a non-finite number of sub-steps and the parcel \
                 would never leave its source level",
                self.pressure_increment_pa
            )));
        }
        Ok(())
    }
}

/// One `getcape.F` result: the five output words plus the source parcel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GetCape {
    /// J/kg.  `getcape.F:500`.
    pub cape: f32,
    /// J/kg, signed the way `getcape.F:475` accumulates it: CIN is reported
    /// as a positive magnitude by `getcape.F` itself.
    pub cin: f32,
    /// m AGL, or -1.0 when the parcel never saturated (`getcape.F:310`).
    pub zlcl: f32,
    /// m AGL, or -1.0 (`getcape.F:311`).
    pub zlfc: f32,
    /// m AGL, or -1.0 (`getcape.F:312`).
    pub zel: f32,
    /// Pa.  `getcape.F:276`.
    pub psource: f32,
    /// K.  `getcape.F:277`.
    pub tsource: f32,
    /// kg/kg.  `getcape.F:278`.
    pub qvsource: f32,
    /// False when `getcape.F:383-396` abandoned the moist-adiabat fixed point
    /// and returned zeros.  `getcape.F` itself leaves the caller no way to
    /// tell that apart from a genuinely zero-CAPE column; this flag does.
    pub converged: bool,
}

/// `getcape.F:14-520`.  `p_in` is mb, `t_in` is degrees Celsius and `q_in` is
/// kg/kg, exactly as `getcape.F:47-51` documents and as both CM1 call sites
/// supply (`base.F:3149-3157`, `writeout.F:2907-2910`).
pub fn getcape(controls: GetCapeControls, p_in: &[f32], t_in: &[f32], q_in: &[f32]) -> Result<GetCape> {
    controls.validate()?;
    let nk = p_in.len();
    if nk < 3 || t_in.len() != nk || q_in.len() != nk {
        return Err(invalid(format!(
            "getcape.F:321 ascends while k<nk over three equal-length \
             profiles; got p={}, t={}, q={} levels, and fewer than three \
             levels leaves the parcel at its source with CAPE identically \
             zero for any atmosphere",
            p_in.len(),
            t_in.len(),
            q_in.len()
        )));
    }

    // getcape.F:131-143.
    let mut p = vec![0.0_f32; nk];
    let mut t = vec![0.0_f32; nk];
    let mut q = vec![0.0_f32; nk];
    let mut td = vec![0.0_f32; nk];
    let mut pi = vec![0.0_f32; nk];
    let mut th = vec![0.0_f32; nk];
    let mut thv = vec![0.0_f32; nk];
    for k in 0..nk {
        if !p_in[k].is_finite() || !t_in[k].is_finite() || !q_in[k].is_finite() {
            return Err(invalid(format!(
                "getcape.F:131-143 converts level {k} of the supplied column; a \
                 non-finite p/T/qv there propagates into thv and every buoyancy \
                 above it, so the reported CAPE would be NaN rather than a number"
            )));
        }
        p[k] = 100.0_f32 * p_in[k];
        t[k] = 273.15_f32 + t_in[k];
        q[k] = q_in[k].max(1.0e-10_f32);
        if p[k] <= 0.0 {
            return Err(invalid(format!(
                "getcape.F:138 raises p*rp00 to rd/cp at level {k}; p={} mb is \
                 not a pressure and the Exner function there is not real",
                p_in[k]
            )));
        }
        let ee = ((q[k] / EPS) * p[k] / 100.0_f32 / (1.0_f32 + (q[k] / EPS))).ln();
        td[k] = 273.15_f32 + (243.5_f32 * ee - 440.8_f32) / (19.48_f32 - ee);
        pi[k] = (p[k] * RP00).powf(RDDCP);
        th[k] = t[k] / pi[k];
        thv[k] = th[k] * (1.0_f32 + REPS * q[k]) / (1.0_f32 + q[k]);
    }

    // getcape.F:147-151.
    let mut z = vec![0.0_f32; nk];
    for k in 1..nk {
        let dz = -CPDG * 0.5_f32 * (thv[k] + thv[k - 1]) * (pi[k] - pi[k - 1]);
        z[k] = z[k - 1] + dz;
    }

    // getcape.F:155-251.
    let mut kmax = 0usize;
    let mut avgth = 0.0_f64;
    let mut avgqv = 0.0_f64;
    match controls.source {
        ParcelSource::Surface => {
            // getcape.F:157-160.
            kmax = 0;
        }
        ParcelSource::MostUnstable => {
            // getcape.F:161-181.
            if p[0] < 50_000.0_f32 {
                kmax = 0;
            } else {
                let mut maxthe = 0.0_f32;
                for k in 0..nk {
                    if p[k] >= 50_000.0_f32 {
                        let the = getthx(p[k], t[k], td[k], q[k]);
                        if the > maxthe {
                            maxthe = the;
                            kmax = k;
                        }
                    }
                }
            }
        }
        ParcelSource::MixedLayer => {
            // getcape.F:183-241.
            let ml_depth = controls.mixed_layer_depth_m;
            if (z[1] - z[0]) > ml_depth {
                // getcape.F:186-192.
                avgth = f64::from(th[0]);
                avgqv = f64::from(q[0]);
                kmax = 0;
            } else if z[nk - 1] < ml_depth {
                // getcape.F:194-200.
                avgth = f64::from(th[nk - 1]);
                avgqv = f64::from(q[nk - 1]);
                kmax = nk - 1;
            } else {
                // getcape.F:202-237.
                let mut k = 1usize;
                while k < nk && z[k] <= ml_depth {
                    avgth += f64::from(0.5_f32 * (z[k] - z[k - 1]) * (th[k] + th[k - 1]));
                    avgqv += f64::from(0.5_f32 * (z[k] - z[k - 1]) * (q[k] + q[k - 1]));
                    k += 1;
                }
                if k >= nk {
                    // Unreachable for an admitted column: getcape.F:194 has
                    // already established z(nk) >= ml_depth and z is strictly
                    // increasing, so the loop above exits on z(k) > ml_depth
                    // with k <= nk.  Reported rather than indexed past the end.
                    return Err(invalid(
                        "getcape.F:212-224 left the mixed-layer scan past the top \
                         of the column; the supplied heights are not strictly \
                         increasing, so the depth-weighted source parcel would be \
                         formed from levels the column does not contain",
                    ));
                }
                let th2 = th[k - 1]
                    + (th[k] - th[k - 1]) * (ml_depth - z[k - 1]) / (z[k] - z[k - 1]);
                let qv2 =
                    q[k - 1] + (q[k] - q[k - 1]) * (ml_depth - z[k - 1]) / (z[k] - z[k - 1]);
                avgth += f64::from(0.5_f32 * (ml_depth - z[k - 1]) * (th2 + th[k - 1]));
                avgqv += f64::from(0.5_f32 * (ml_depth - z[k - 1]) * (qv2 + q[k - 1]));
                avgth /= f64::from(ml_depth);
                avgqv /= f64::from(ml_depth);
                kmax = 0;
            }
        }
    }

    // getcape.F:253-312.
    let mut narea = 0.0_f32;
    let mut k = kmax;
    let mut th2;
    let mut pi2 = pi[kmax];
    let mut p2 = p[kmax];
    let mut t2;
    let mut qv2;
    let mut b2;
    // getcape.F:262 sets thv2 = thv(kmax) on the source=1/2 branch and
    // getcape.F:269 recomputes it from the mixed-layer average on source=3.
    // Only the source=3 value is read before getcape.F:428 overwrites it, at
    // getcape.F:273, so only that one is formed here.
    match controls.source {
        ParcelSource::Surface | ParcelSource::MostUnstable => {
            th2 = th[kmax];
            t2 = t[kmax];
            qv2 = q[kmax];
            b2 = 0.0_f32;
        }
        ParcelSource::MixedLayer => {
            th2 = avgth as f32;
            qv2 = avgqv as f32;
            let thv_source = th2 * (1.0_f32 + REPS * qv2) / (1.0_f32 + qv2);
            t2 = th2 * pi2;
            b2 = G * (thv_source - thv[kmax]) / thv[kmax];
        }
    }
    let psource = p2;
    let tsource = t2;
    let qvsource = qv2;

    let mut ql2 = 0.0_f32;
    let mut qi2 = 0.0_f32;
    let mut qt = qv2;

    let mut cape = 0.0_f32;
    let mut cin = 0.0_f32;

    let mut doit = true;
    let mut cloud = false;
    let ice = controls.adiabat.ice();

    let mut zlcl = -1.0_f32;
    let mut zlfc = -1.0_f32;
    let mut zel = -1.0_f32;

    // getcape.F:321-513.
    while doit && k + 1 < nk {
        k += 1;
        let b1 = b2;

        // getcape.F:326-333.
        let mut dp = p[k - 1] - p[k];
        let nloop = if dp < controls.pressure_increment_pa {
            1i32
        } else {
            let count = 1 + (dp / controls.pressure_increment_pa) as i32;
            dp /= count as f32;
            count
        };

        for n in 1..=nloop {
            // getcape.F:337-344.
            let p1 = p2;
            let t1 = t2;
            let th1 = th2;
            let qv1 = qv2;
            let ql1 = ql2;
            let qi1 = qi2;

            // getcape.F:346-347.
            p2 -= dp;
            pi2 = (p2 * RP00).powf(RDDCP);

            // getcape.F:349-403.
            let mut thlast = th1;
            let mut iteration = 0u32;
            loop {
                iteration += 1;
                t2 = thlast * pi2;
                let (fliq, fice) = if ice {
                    let fliq = ((t2 - 233.15_f32) / (273.15_f32 - 233.15_f32))
                        .min(1.0_f32)
                        .max(0.0_f32);
                    (fliq, 1.0_f32 - fliq)
                } else {
                    (1.0_f32, 0.0_f32)
                };
                qv2 = qt.min(fliq * getqvl(p2, t2) + fice * getqvi(p2, t2));
                qi2 = (fice * (qt - qv2)).max(0.0_f32);
                ql2 = (qt - qv2 - qi2).max(0.0_f32);

                let tbar = 0.5_f32 * (t1 + t2);
                let qvbar = 0.5_f32 * (qv1 + qv2);
                let qlbar = 0.5_f32 * (ql1 + ql2);
                let qibar = 0.5_f32 * (qi1 + qi2);

                let lhv = LV1 - LV2 * tbar;
                let lhs = LS1 - LS2 * tbar;

                let rm = RD + RV * qvbar;
                let cpm = CP + CPV * qvbar + CPL * qlbar + CPI * qibar;
                th2 = th1
                    * (lhv * (ql2 - ql1) / (cpm * tbar)
                        + lhs * (qi2 - qi1) / (cpm * tbar)
                        + (rm / cpm - RD / CP) * (p2 / p1).ln())
                    .exp();

                if iteration > MAX_ADIABAT_ITERATIONS {
                    // getcape.F:383-396.  CAPE, CIN and the three source words
                    // are zeroed and the routine returns; zlcl, zlfc and zel are
                    // left exactly as the ascent had them, which is what makes
                    // the `converged` flag necessary -- without it a caller
                    // cannot tell this from a genuinely zero-CAPE column that
                    // happened to reach its LCL.
                    return Ok(GetCape {
                        cape: 0.0,
                        cin: 0.0,
                        zlcl,
                        zlfc,
                        zel,
                        psource: 0.0,
                        tsource: 0.0,
                        qvsource: 0.0,
                        converged: false,
                    });
                }
                if (th2 - thlast).abs() > CONVERGE {
                    thlast += 0.3_f32 * (th2 - thlast);
                } else {
                    break;
                }
            }

            // getcape.F:408-412.
            if ql2 >= 1.0e-10_f32 {
                cloud = true;
            }
            if cloud && zlcl < 0.0 {
                zlcl = z[k - 1] + (z[k] - z[k - 1]) * (n as f32) / (nloop as f32);
            }

            // getcape.F:414-424.
            if controls.adiabat.pseudoadiabatic() {
                qt = qv2;
                ql2 = 0.0;
                qi2 = 0.0;
            }
        }

        // getcape.F:428-430.
        let thv2 = th2 * (1.0_f32 + REPS * qv2) / (1.0_f32 + qv2 + ql2 + qi2);
        b2 = G * (thv2 - thv[k]) / thv[k];
        let dz = -CPDG * 0.5_f32 * (thv[k] + thv[k - 1]) * (pi[k] - pi[k - 1]);

        // getcape.F:432-440.
        if zlcl > 0.0 && zlfc < 0.0 && b2 > 0.0 {
            if b1 > 0.0 {
                zlfc = zlcl;
            } else {
                zlfc = z[k - 1] + (z[k] - z[k - 1]) * (0.0_f32 - b1) / (b2 - b1);
            }
        }

        // getcape.F:442-445.
        if zlfc > 0.0 && zel < 0.0 && b2 < 0.0 {
            zel = z[k - 1] + (z[k] - z[k - 1]) * (0.0_f32 - b1) / (b2 - b1);
        }

        // getcape.F:460-498.
        let parea;
        if b2 >= 0.0 && b1 < 0.0 {
            let frac = b2 / (b2 - b1);
            parea = 0.5_f32 * b2 * dz * frac;
            narea -= 0.5_f32 * b1 * dz * (1.0_f32 - frac);
            cin += narea;
            narea = 0.0;
        } else if b2 < 0.0 && b1 > 0.0 {
            let frac = b1 / (b1 - b2);
            parea = 0.5_f32 * b1 * dz * frac;
            narea = -0.5_f32 * b2 * dz * (1.0_f32 - frac);
        } else if b2 < 0.0 {
            parea = 0.0;
            narea -= 0.5_f32 * dz * (b1 + b2);
        } else {
            parea = 0.5_f32 * dz * (b1 + b2);
            narea = 0.0;
        }

        // getcape.F:500.
        cape += parea.max(0.0_f32);

        // getcape.F:508-511.
        if p[k] <= 10_000.0_f32 && b2 < 0.0 {
            doit = false;
        }
    }

    Ok(GetCape {
        cape,
        cin,
        zlcl,
        zlfc,
        zel,
        psource,
        tsource,
        qvsource,
        converged: true,
    })
}

// ---------------------------------------------------------------------------
// Hodograph parameters.  Not CM1: see the module header.
// ---------------------------------------------------------------------------

fn interp_at(z: &[f32], value: &[f32], height: f32) -> f32 {
    // Linear in height, clamped at both ends, which is what a hodograph
    // sampled outside its own reported depth can honestly be.
    if height <= z[0] {
        return value[0];
    }
    let last = z.len() - 1;
    if height >= z[last] {
        return value[last];
    }
    let mut upper = 1usize;
    while z[upper] < height {
        upper += 1;
    }
    let lower = upper - 1;
    let span = z[upper] - z[lower];
    if span <= 0.0 {
        return value[lower];
    }
    value[lower] + (value[upper] - value[lower]) * (height - z[lower]) / span
}

/// Mean wind over `[bottom, top]`, sampled at `samples` evenly spaced heights.
fn layer_mean(z: &[f32], u: &[f32], v: &[f32], bottom: f32, top: f32, samples: usize) -> (f32, f32) {
    let mut su = 0.0_f64;
    let mut sv = 0.0_f64;
    for index in 0..samples {
        let fraction = index as f32 / (samples - 1) as f32;
        let height = bottom + (top - bottom) * fraction;
        su += f64::from(interp_at(z, u, height));
        sv += f64::from(interp_at(z, v, height));
    }
    ((su / samples as f64) as f32, (sv / samples as f64) as f32)
}

/// One Bunkers et al. (2000) internal-dynamics storm motion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StormMotion {
    pub u_m_per_s: f32,
    pub v_m_per_s: f32,
}

/// Which Bunkers mover the storm-relative helicity is taken against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StormMotionChoice {
    BunkersRight,
    BunkersLeft,
}

impl StormMotionChoice {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "bunkers-right" => Ok(Self::BunkersRight),
            "bunkers-left" => Ok(Self::BunkersLeft),
            other => Err(invalid(format!(
                "storm motion {other:?} is not a transcribed method; storm-relative \
                 helicity is an integral taken against one storm motion, and an \
                 unnamed motion would publish a helicity that cannot be reproduced \
                 or compared with any other estimate.  Use \"bunkers-right\" or \
                 \"bunkers-left\" (Bunkers et al. 2000 internal dynamics)"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::BunkersRight => "bunkers-right",
            Self::BunkersLeft => "bunkers-left",
        }
    }
}

/// The complete Bunkers et al. (2000) internal-dynamics solution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BunkersMotion {
    pub mean_wind: StormMotion,
    pub shear_u_m_per_s: f32,
    pub shear_v_m_per_s: f32,
    pub right: StormMotion,
    pub left: StormMotion,
    /// `true` when the shear vector is exactly zero (a calm or uniform
    /// hodograph, CM1's iwnd=0 among them): Bunkers' deviation direction is
    /// then undefined and both movers are the 0-6 km mean wind.
    pub deviation_undefined: bool,
}

/// Bunkers et al. (2000, Wea. Forecasting 15, 61-79) internal-dynamics method.
///
/// `V_mean` is the 0-6 km AGL mean wind, `V_shear` the (5.5-6 km mean) minus
/// (0-0.5 km mean) vector, and the right and left movers deviate from the mean
/// wind by 7.5 m/s orthogonally to `V_shear`.
pub fn bunkers_motion(z: &[f32], u: &[f32], v: &[f32]) -> Result<BunkersMotion> {
    if z.len() < 2 || u.len() != z.len() || v.len() != z.len() {
        return Err(invalid(
            "the Bunkers internal-dynamics motion needs at least two hodograph \
             levels with matching u and v; with fewer, the 0-6 km mean wind and \
             the shear vector are undefined and the reported storm motion would \
             be an invented number",
        ));
    }
    let samples = 201usize;
    let mean = layer_mean(z, u, v, 0.0, BUNKERS_MEAN_WIND_TOP_M, samples);
    let lower = layer_mean(z, u, v, 0.0, BUNKERS_SHEAR_LOWER_TOP_M, samples);
    let upper = layer_mean(
        z,
        u,
        v,
        BUNKERS_SHEAR_UPPER_BASE_M,
        BUNKERS_MEAN_WIND_TOP_M,
        samples,
    );
    let shear_u = upper.0 - lower.0;
    let shear_v = upper.1 - lower.1;
    let magnitude = (shear_u * shear_u + shear_v * shear_v).sqrt();
    if !magnitude.is_finite() {
        return Err(invalid(
            "the 0-0.5 km to 5.5-6 km shear vector is not finite, so the Bunkers \
             deviation is undefined and every storm-relative helicity taken \
             against it would be NaN",
        ));
    }
    if magnitude <= 0.0 {
        // A calm or uniform hodograph (CM1's iwnd=0, for one) has no shear to
        // be orthogonal to: there is no deviate motion to report, and a storm
        // in it moves with the mean wind.  Both movers are that wind, and the
        // report says so instead of dividing by zero.
        let mean_wind = StormMotion {
            u_m_per_s: mean.0,
            v_m_per_s: mean.1,
        };
        return Ok(BunkersMotion {
            mean_wind,
            shear_u_m_per_s: shear_u,
            shear_v_m_per_s: shear_v,
            right: mean_wind,
            left: mean_wind,
            deviation_undefined: true,
        });
    }
    let deviation_u = BUNKERS_DEVIATION_M_PER_S * shear_v / magnitude;
    let deviation_v = BUNKERS_DEVIATION_M_PER_S * shear_u / magnitude;
    Ok(BunkersMotion {
        mean_wind: StormMotion {
            u_m_per_s: mean.0,
            v_m_per_s: mean.1,
        },
        shear_u_m_per_s: shear_u,
        shear_v_m_per_s: shear_v,
        right: StormMotion {
            u_m_per_s: mean.0 + deviation_u,
            v_m_per_s: mean.1 - deviation_v,
        },
        left: StormMotion {
            u_m_per_s: mean.0 - deviation_u,
            v_m_per_s: mean.1 + deviation_v,
        },
        deviation_undefined: false,
    })
}

/// Bulk wind difference |V(top) - V(bottom)| across one layer, m/s.
pub fn bulk_shear(z: &[f32], u: &[f32], v: &[f32], bottom: f32, top: f32) -> f32 {
    let du = interp_at(z, u, top) - interp_at(z, u, bottom);
    let dv = interp_at(z, v, top) - interp_at(z, v, bottom);
    (du * du + dv * dv).sqrt()
}

/// Storm-relative helicity over `0..top` m AGL against one storm motion.
///
/// The discrete form is the standard one, positive for a clockwise-turning
/// (veering) hodograph:
/// `SRH = sum[(u(k+1)-c_u)(v(k)-c_v) - (u(k)-c_u)(v(k+1)-c_v)]`.
pub fn storm_relative_helicity(
    z: &[f32],
    u: &[f32],
    v: &[f32],
    motion: StormMotion,
    top: f32,
    samples: usize,
) -> f32 {
    let mut total = 0.0_f64;
    let mut previous_u = interp_at(z, u, 0.0);
    let mut previous_v = interp_at(z, v, 0.0);
    for index in 1..samples {
        let height = top * (index as f32) / ((samples - 1) as f32);
        let next_u = interp_at(z, u, height);
        let next_v = interp_at(z, v, height);
        total += f64::from(
            (next_u - motion.u_m_per_s) * (previous_v - motion.v_m_per_s)
                - (previous_u - motion.u_m_per_s) * (next_v - motion.v_m_per_s),
        );
        previous_u = next_u;
        previous_v = next_v;
    }
    total as f32
}

/// The number of hodograph samples the helicity integral uses.
pub const HELICITY_SAMPLES: usize = 501;
/// Storm-relative helicity layer tops, m AGL.
pub const HELICITY_DEPTHS_M: [f32; 2] = [1000.0, 3000.0];
/// Bulk shear layer tops, m AGL.
pub const SHEAR_DEPTHS_M: [f32; 2] = [1000.0, 6000.0];

/// One column's complete environment report.
#[derive(Clone, Debug, PartialEq)]
pub struct SoundingParameters {
    pub levels: usize,
    pub surface_pressure_pa: f32,
    pub surface_temperature_k: f32,
    pub surface_qv_kg_per_kg: f32,
    pub lowest_mass_level_m: f32,
    pub column_top_m: f32,
    pub surface_based: GetCape,
    pub most_unstable: GetCape,
    pub mixed_layer: GetCape,
    pub mixed_layer_depth_m: f32,
    pub adiabat: Adiabat,
    pub pressure_increment_pa: f32,
    pub bunkers: BunkersMotion,
    pub storm_motion_choice: StormMotionChoice,
    pub shear_0_1km_m_per_s: f32,
    pub shear_0_6km_m_per_s: f32,
    pub srh_0_1km_m2_per_s2: f32,
    pub srh_0_3km_m2_per_s2: f32,
}

/// Build the complete environment report for one column.
///
/// `z/p/t/qv/u/v` are the model's own mass levels in ascending height, and the
/// three `surface_*` words are CM1's `psurf/tsurf/qsurf` for this base state
/// (`base.F:600-603` for `isnd=7`, `base.F:440-449` for `isnd=5`).  The report
/// column is those levels with the surface prepended at z=0, which is exactly
/// the column `base.F:3149-3157` hands `getcape`.
#[allow(clippy::too_many_arguments)]
pub fn sounding_parameters(
    z_m: &[f32],
    pressure_pa: &[f32],
    temperature_k: &[f32],
    qv_kg_per_kg: &[f32],
    u_m_per_s: &[f32],
    v_m_per_s: &[f32],
    surface_pressure_pa: f32,
    surface_temperature_k: f32,
    surface_qv_kg_per_kg: f32,
    surface_u_m_per_s: f32,
    surface_v_m_per_s: f32,
    mixed_layer_depth_m: f32,
    adiabat: Adiabat,
    pressure_increment_pa: f32,
    storm_motion_choice: StormMotionChoice,
) -> Result<SoundingParameters> {
    let nz = z_m.len();
    if nz < 3 {
        return Err(invalid(
            "a sounding report needs at least three mass levels; with fewer, \
             getcape.F:321's ascent never leaves the source parcel and every \
             reported parameter would be a constant, not a measurement of this \
             atmosphere",
        ));
    }
    for (label, values) in [
        ("pressure", pressure_pa),
        ("temperature", temperature_k),
        ("qv", qv_kg_per_kg),
        ("u", u_m_per_s),
        ("v", v_m_per_s),
    ] {
        if values.len() != nz {
            return Err(invalid(format!(
                "the sounding report column carries {nz} heights but {} {label} \
                 words; a mismatched profile would pair each height with another \
                 level's thermodynamics and report an atmosphere the model never \
                 held",
                values.len()
            )));
        }
    }

    // base.F:3149-3157.  Surface first, then the model's own mass levels.
    let mut p_mb = Vec::with_capacity(nz + 1);
    let mut t_c = Vec::with_capacity(nz + 1);
    let mut q = Vec::with_capacity(nz + 1);
    p_mb.push(0.01_f32 * surface_pressure_pa);
    t_c.push(surface_temperature_k - 273.15_f32);
    q.push(surface_qv_kg_per_kg.max(1.0e-20_f32));
    for k in 0..nz {
        p_mb.push(0.01_f32 * pressure_pa[k]);
        t_c.push(temperature_k[k] - 273.15_f32);
        q.push(qv_kg_per_kg[k].max(1.0e-20_f32));
    }

    let mut report_z = Vec::with_capacity(nz + 1);
    let mut report_u = Vec::with_capacity(nz + 1);
    let mut report_v = Vec::with_capacity(nz + 1);
    report_z.push(0.0_f32);
    report_u.push(surface_u_m_per_s);
    report_v.push(surface_v_m_per_s);
    for k in 0..nz {
        report_z.push(z_m[k]);
        report_u.push(u_m_per_s[k]);
        report_v.push(v_m_per_s[k]);
    }

    let mut results = Vec::with_capacity(3);
    for source in [
        ParcelSource::Surface,
        ParcelSource::MostUnstable,
        ParcelSource::MixedLayer,
    ] {
        results.push(getcape(
            GetCapeControls {
                source,
                adiabat,
                mixed_layer_depth_m,
                pressure_increment_pa,
            },
            &p_mb,
            &t_c,
            &q,
        )?);
    }

    let bunkers = bunkers_motion(&report_z, &report_u, &report_v)?;
    let motion = match storm_motion_choice {
        StormMotionChoice::BunkersRight => bunkers.right,
        StormMotionChoice::BunkersLeft => bunkers.left,
    };

    Ok(SoundingParameters {
        levels: nz,
        surface_pressure_pa,
        surface_temperature_k,
        surface_qv_kg_per_kg,
        lowest_mass_level_m: z_m[0],
        column_top_m: z_m[nz - 1],
        surface_based: results[0],
        most_unstable: results[1],
        mixed_layer: results[2],
        mixed_layer_depth_m,
        adiabat,
        pressure_increment_pa,
        bunkers,
        storm_motion_choice,
        shear_0_1km_m_per_s: bulk_shear(&report_z, &report_u, &report_v, 0.0, SHEAR_DEPTHS_M[0]),
        shear_0_6km_m_per_s: bulk_shear(&report_z, &report_u, &report_v, 0.0, SHEAR_DEPTHS_M[1]),
        srh_0_1km_m2_per_s2: storm_relative_helicity(
            &report_z,
            &report_u,
            &report_v,
            motion,
            HELICITY_DEPTHS_M[0],
            HELICITY_SAMPLES,
        ),
        srh_0_3km_m2_per_s2: storm_relative_helicity(
            &report_z,
            &report_u,
            &report_v,
            motion,
            HELICITY_DEPTHS_M[1],
            HELICITY_SAMPLES,
        ),
    })
}

/// The method citations this report carries so a reader never has to guess.
pub const CM1_PARCEL_AUTHORITY: &str =
    "CM1 r21.1 src/getcape.F:14-520 (getcape), src/getcape.F:526-538 (getqvl, \
     Bolton 1980 with the 171023 min(es,p/2) clamp), src/base.F:3143-3189 (the \
     t=0 three-parcel report), src/writeout.F:2895-3030 (the output_cape / \
     output_cin / output_lcl / output_lfc 2-D fields, source=3)";
pub const STORM_MOTION_AUTHORITY: &str =
    "Bunkers, Klimowski, Zeitler, Thompson and Weisman (2000, Wea. Forecasting \
     15, 61-79) internal-dynamics method: V_mean is the 0-6 km AGL mean wind, \
     V_shear the (5.5-6 km mean) minus (0-0.5 km mean) vector, and the right \
     and left movers deviate 7.5 m/s orthogonally to V_shear.  CM1 r21.1 \
     computes no storm motion, bulk shear or storm-relative helicity anywhere \
     in its source tree, so these three are not transcribed from it";
pub const REPORT_CLAIM_LIMIT: &str =
    "the parcel and hodograph parameters of one model column, integrated on \
     that column's own mass levels with CM1's surface level prepended; a \
     sounding file integrated at 1 hPa resolution reports a larger CAPE \
     because the model's lowest mass level sits half a grid cell above the \
     ground and the layer below it is never integrated; not a forecast, a \
     severe-weather outlook, or a claim about any observed storm";

fn json_f32(out: &mut String, value: f32) {
    if value.is_finite() {
        let _ = write!(out, "{}", f64::from(value));
    } else {
        out.push_str("null");
    }
}

fn json_getcape(out: &mut String, label: &str, source: ParcelSource, value: &GetCape) {
    let _ = write!(out, "\"{label}\":{{\"parcel_source\":\"{}\",", source.name());
    let _ = write!(out, "\"parcel_source_selector\":{},", source.as_i32());
    out.push_str("\"cape_j_per_kg\":");
    json_f32(out, value.cape);
    out.push_str(",\"cin_j_per_kg\":");
    json_f32(out, value.cin);
    out.push_str(",\"lcl_m_agl\":");
    json_f32(out, value.zlcl);
    out.push_str(",\"lfc_m_agl\":");
    json_f32(out, value.zlfc);
    out.push_str(",\"el_m_agl\":");
    json_f32(out, value.zel);
    out.push_str(",\"source_pressure_pa\":");
    json_f32(out, value.psource);
    out.push_str(",\"source_temperature_k\":");
    json_f32(out, value.tsource);
    out.push_str(",\"source_qv_kg_per_kg\":");
    json_f32(out, value.qvsource);
    let _ = write!(out, ",\"converged\":{}}}", value.converged);
}

impl SoundingParameters {
    /// Canonical JSON for the run receipt.  Keys are emitted in a fixed order.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push_str("{\"schema\":\"gm1.sounding-parameters/v1\",");
        let _ = write!(
            out,
            "\"cm1_parcel_authority\":{:?},\"storm_motion_authority\":{:?},\"claim_limit\":{:?},",
            CM1_PARCEL_AUTHORITY, STORM_MOTION_AUTHORITY, REPORT_CLAIM_LIMIT
        );
        let _ = write!(out, "\"mass_levels\":{},", self.levels);
        out.push_str("\"lowest_mass_level_m_agl\":");
        json_f32(&mut out, self.lowest_mass_level_m);
        out.push_str(",\"column_top_m_agl\":");
        json_f32(&mut out, self.column_top_m);
        out.push_str(",\"surface_pressure_pa\":");
        json_f32(&mut out, self.surface_pressure_pa);
        out.push_str(",\"surface_temperature_k\":");
        json_f32(&mut out, self.surface_temperature_k);
        out.push_str(",\"surface_qv_kg_per_kg\":");
        json_f32(&mut out, self.surface_qv_kg_per_kg);
        let _ = write!(
            out,
            ",\"adiabat\":\"{}\",\"adiabat_selector\":{},",
            self.adiabat.name(),
            self.adiabat.as_i32()
        );
        out.push_str("\"mixed_layer_depth_m\":");
        json_f32(&mut out, self.mixed_layer_depth_m);
        out.push_str(",\"pressure_increment_pa\":");
        json_f32(&mut out, self.pressure_increment_pa);
        out.push(',');
        json_getcape(&mut out, "surface_based", ParcelSource::Surface, &self.surface_based);
        out.push(',');
        json_getcape(&mut out, "most_unstable", ParcelSource::MostUnstable, &self.most_unstable);
        out.push(',');
        json_getcape(&mut out, "mixed_layer", ParcelSource::MixedLayer, &self.mixed_layer);
        let _ = write!(
            out,
            ",\"storm_motion\":\"{}\",",
            self.storm_motion_choice.name()
        );
        out.push_str("\"bunkers_mean_wind_u_m_per_s\":");
        json_f32(&mut out, self.bunkers.mean_wind.u_m_per_s);
        out.push_str(",\"bunkers_mean_wind_v_m_per_s\":");
        json_f32(&mut out, self.bunkers.mean_wind.v_m_per_s);
        out.push_str(",\"bunkers_shear_u_m_per_s\":");
        json_f32(&mut out, self.bunkers.shear_u_m_per_s);
        out.push_str(",\"bunkers_shear_v_m_per_s\":");
        json_f32(&mut out, self.bunkers.shear_v_m_per_s);
        out.push_str(",\"bunkers_right_u_m_per_s\":");
        json_f32(&mut out, self.bunkers.right.u_m_per_s);
        out.push_str(",\"bunkers_right_v_m_per_s\":");
        json_f32(&mut out, self.bunkers.right.v_m_per_s);
        out.push_str(",\"bunkers_left_u_m_per_s\":");
        json_f32(&mut out, self.bunkers.left.u_m_per_s);
        out.push_str(",\"bunkers_left_v_m_per_s\":");
        json_f32(&mut out, self.bunkers.left.v_m_per_s);
        out.push_str(",\"shear_0_1km_m_per_s\":");
        json_f32(&mut out, self.shear_0_1km_m_per_s);
        out.push_str(",\"shear_0_6km_m_per_s\":");
        json_f32(&mut out, self.shear_0_6km_m_per_s);
        out.push_str(",\"srh_0_1km_m2_per_s2\":");
        json_f32(&mut out, self.srh_0_1km_m2_per_s2);
        out.push_str(",\"srh_0_3km_m2_per_s2\":");
        json_f32(&mut out, self.srh_0_3km_m2_per_s2);
        let _ = write!(out, ",\"helicity_samples\":{}", HELICITY_SAMPLES);
        // Only a zero-shear hodograph carries the word, so every report of a
        // sheared sounding keeps its bytes.
        if self.bunkers.deviation_undefined {
            out.push_str(",\"bunkers_deviation_undefined\":true");
        }
        out.push('}');
        out
    }

    /// The console block `base.F:3175-3180` prints, in the same order.
    pub fn to_text(&self) -> String {
        let mut out = String::with_capacity(1024);
        out.push_str("  Thermodynamic properties of the initial sounding (i=1,j=1):\n");
        for (label, source, value) in [
            ("surface parcel", ParcelSource::Surface, &self.surface_based),
            ("most-unstable parcel", ParcelSource::MostUnstable, &self.most_unstable),
            ("mixed-layer parcel", ParcelSource::MixedLayer, &self.mixed_layer),
        ] {
            let _ = source;
            let _ = writeln!(out, "    for {label}:");
            let _ = writeln!(
                out,
                "        source p(mb),T(K),qv(g/kg) = {:>10.1}{:>10.1}{:>10.1}",
                0.01 * value.psource,
                value.tsource,
                1000.0 * value.qvsource
            );
            let _ = writeln!(
                out,
                "        LCL,LFC,EL (m AGL)         = {:>10.1}{:>10.1}{:>10.1}",
                value.zlcl, value.zlfc, value.zel
            );
            let _ = writeln!(
                out,
                "        CAPE,CIN (J/kg)            = {:>10.1}{:>10.1}",
                value.cape, value.cin
            );
            if !value.converged {
                out.push_str(
                    "        (getcape.F:383-396 abandoned the moist adiabat here; \
                     the zeros above are not a measurement)\n",
                );
            }
        }
        let _ = writeln!(
            out,
            "    hodograph ({} mover, Bunkers et al. 2000 internal dynamics):",
            self.storm_motion_choice.name()
        );
        let _ = writeln!(
            out,
            "        storm motion u,v (m/s)     = {:>10.1}{:>10.1}",
            match self.storm_motion_choice {
                StormMotionChoice::BunkersRight => self.bunkers.right.u_m_per_s,
                StormMotionChoice::BunkersLeft => self.bunkers.left.u_m_per_s,
            },
            match self.storm_motion_choice {
                StormMotionChoice::BunkersRight => self.bunkers.right.v_m_per_s,
                StormMotionChoice::BunkersLeft => self.bunkers.left.v_m_per_s,
            }
        );
        if self.bunkers.deviation_undefined {
            out.push_str(
                "        (no 0-0.5 to 5.5-6 km shear: the Bunkers deviation is \
                 undefined, so the motion is the 0-6 km mean wind)\n",
            );
        }
        let _ = writeln!(
            out,
            "        0-1,0-6 km shear (m/s)     = {:>10.1}{:>10.1}",
            self.shear_0_1km_m_per_s, self.shear_0_6km_m_per_s
        );
        let _ = writeln!(
            out,
            "        0-1,0-3 km SRH (m2/s2)     = {:>10.1}{:>10.1}",
            self.srh_0_1km_m2_per_s2, self.srh_0_3km_m2_per_s2
        );
        let _ = writeln!(
            out,
            "        lowest mass level (m AGL)  = {:>10.1}",
            self.lowest_mass_level_m
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idealized_initialization::rslf;

    /// One analytic convective column: a constant lapse rate to the tropopause,
    /// an isothermal layer above it, and exponentially decaying moisture.  It
    /// carries no fixture, place or date; its only job is to reach an LCL, an
    /// LFC and an EL so every branch of the ascent runs.
    fn column(levels: usize, dz: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut p = Vec::with_capacity(levels);
        let mut t = Vec::with_capacity(levels);
        let mut q = Vec::with_capacity(levels);
        for k in 0..levels {
            let z = k as f32 * dz;
            p.push(1000.0_f32 * (-z / 8500.0_f32).exp());
            t.push(if z < 12000.0 {
                30.0_f32 - 8.0e-3_f32 * z
            } else {
                30.0_f32 - 8.0e-3_f32 * 12000.0_f32
            });
            q.push((0.017_f32 * (-z / 2500.0_f32).exp()).max(1.0e-8_f32));
        }
        (p, t, q)
    }

    fn hodograph(levels: usize, dz: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // A quarter-circle turning clockwise with height then a straight
        // westerly jet: a veering hodograph, so the right-mover helicity is
        // positive by construction.
        let mut z = Vec::with_capacity(levels);
        let mut u = Vec::with_capacity(levels);
        let mut v = Vec::with_capacity(levels);
        for k in 0..levels {
            let height = k as f32 * dz;
            let fraction = (height / 3000.0_f32).min(1.0_f32);
            let angle = fraction * std::f32::consts::FRAC_PI_2;
            u.push(20.0_f32 * angle.sin() + 0.002_f32 * height);
            v.push(10.0_f32 * angle.cos() - 10.0_f32);
            z.push(height);
        }
        (z, u, v)
    }

    #[test]
    fn saturation_formula_is_the_one_the_microphysics_already_uses() {
        // getcape.F:526-538 and idealized_initialization::rslf are the same
        // Bolton branch with the same 171023 min(es, p/2) clamp.  If they ever
        // diverge, the parcel and the condensation the model actually performs
        // saturate on two different curves.
        for &pressure in &[100_000.0_f32, 85_000.0, 50_000.0, 20_000.0, 5_000.0, 500.0] {
            for &celsius in &[35.0_f32, 20.0, 0.0, -20.0, -40.0, -60.0, -80.0] {
                let temperature = 273.15_f32 + celsius;
                assert_eq!(
                    getqvl(pressure, temperature).to_bits(),
                    rslf(pressure, temperature).to_bits(),
                    "getqvl and rslf differ at p={pressure} T={temperature}"
                );
            }
        }
    }

    #[test]
    fn every_parcel_and_adiabat_getcape_implements_is_admitted_and_no_other() {
        for value in 1..=3 {
            assert_eq!(ParcelSource::from_i32(value).unwrap().as_i32(), value);
        }
        for value in [-1, 0, 4, 7] {
            let message = ParcelSource::from_i32(value).unwrap_err().to_string();
            assert!(message.contains("getcape.F:242-250"), "{message}");
        }
        for value in 1..=4 {
            assert_eq!(Adiabat::from_i32(value).unwrap().as_i32(), value);
        }
        for value in [-1, 0, 5, 9] {
            let message = Adiabat::from_i32(value).unwrap_err().to_string();
            assert!(message.contains("getcape.F:419-423"), "{message}");
        }
    }

    #[test]
    fn the_transcription_reaches_lcl_lfc_and_el_and_orders_them() {
        let (p, t, q) = column(41, 500.0);
        let result = getcape(GetCapeControls::default(), &p, &t, &q).unwrap();
        assert!(result.converged);
        assert!(result.cape > 1000.0, "cape={}", result.cape);
        assert!(result.zlcl > 0.0 && result.zlfc > 0.0 && result.zel > 0.0);
        assert!(result.zel > result.zlcl, "EL below LCL");
        assert!(result.cin >= 0.0, "getcape.F:475 reports CIN as a magnitude");
    }

    #[test]
    fn the_pseudoadiabat_and_the_reversible_adiabat_are_different_ascents() {
        // getcape.F:414-418 removes the condensate on adiabat 1 and 3 and
        // carries it on 2 and 4; carrying it loads the parcel and lowers CAPE.
        let (p, t, q) = column(41, 500.0);
        let pseudo = getcape(
            GetCapeControls {
                adiabat: Adiabat::PseudoadiabaticLiquid,
                ..GetCapeControls::default()
            },
            &p,
            &t,
            &q,
        )
        .unwrap();
        let reversible = getcape(
            GetCapeControls {
                adiabat: Adiabat::ReversibleLiquid,
                ..GetCapeControls::default()
            },
            &p,
            &t,
            &q,
        )
        .unwrap();
        assert!(
            reversible.cape < pseudo.cape,
            "reversible {} should be below pseudoadiabatic {}",
            reversible.cape,
            pseudo.cape
        );
    }

    #[test]
    fn the_mixed_layer_parcel_differs_from_the_surface_parcel_when_the_layer_is_resolved() {
        // With 100 m levels the 500 m mixed layer spans several of them, so
        // getcape.F:202-237's depth-weighted average is a different parcel
        // from getcape.F:157-160's lowest level.
        let (p, t, q) = column(120, 100.0);
        let surface = getcape(
            GetCapeControls {
                source: ParcelSource::Surface,
                ..GetCapeControls::default()
            },
            &p,
            &t,
            &q,
        )
        .unwrap();
        let mixed = getcape(GetCapeControls::default(), &p, &t, &q).unwrap();
        assert_ne!(surface.qvsource.to_bits(), mixed.qvsource.to_bits());
        assert_ne!(surface.cape.to_bits(), mixed.cape.to_bits());
    }

    #[test]
    fn a_column_of_two_levels_is_refused_by_name() {
        let message = getcape(GetCapeControls::default(), &[1000.0, 950.0], &[30.0, 25.0], &[0.01, 0.01])
            .unwrap_err()
            .to_string();
        assert!(message.contains("getcape.F:321"), "{message}");
        assert!(message.contains("CAPE identically"), "{message}");
    }

    #[test]
    fn a_zero_mixed_layer_depth_and_a_zero_increment_are_refused_by_name() {
        let (p, t, q) = column(41, 500.0);
        for (controls, anchor) in [
            (
                GetCapeControls {
                    mixed_layer_depth_m: 0.0,
                    ..GetCapeControls::default()
                },
                "getcape.F:233-234",
            ),
            (
                GetCapeControls {
                    pressure_increment_pa: 0.0,
                    ..GetCapeControls::default()
                },
                "getcape.F:328-333",
            ),
        ] {
            let message = getcape(controls, &p, &t, &q).unwrap_err().to_string();
            assert!(message.contains(anchor), "{message}");
        }
    }

    #[test]
    fn bunkers_places_the_movers_seven_and_a_half_metres_per_second_off_the_mean_wind() {
        let (z, u, v) = hodograph(61, 250.0);
        let motion = bunkers_motion(&z, &u, &v).unwrap();
        for mover in [motion.right, motion.left] {
            let du = mover.u_m_per_s - motion.mean_wind.u_m_per_s;
            let dv = mover.v_m_per_s - motion.mean_wind.v_m_per_s;
            let magnitude = (du * du + dv * dv).sqrt();
            assert!(
                (magnitude - BUNKERS_DEVIATION_M_PER_S).abs() < 1.0e-3,
                "deviation magnitude {magnitude}"
            );
            // Orthogonal to the shear vector, which is what "internal
            // dynamics" means: the movers straddle the mean wind.
            let dot = du * motion.shear_u_m_per_s + dv * motion.shear_v_m_per_s;
            assert!(dot.abs() < 1.0e-2, "deviation is not orthogonal: {dot}");
        }
        assert!(
            (motion.right.u_m_per_s - motion.left.u_m_per_s).abs()
                + (motion.right.v_m_per_s - motion.left.v_m_per_s).abs()
                > 1.0,
            "the right and left movers coincide"
        );
    }

    #[test]
    fn a_veering_hodograph_gives_the_right_mover_more_helicity_than_the_left() {
        let (z, u, v) = hodograph(61, 250.0);
        let motion = bunkers_motion(&z, &u, &v).unwrap();
        let right =
            storm_relative_helicity(&z, &u, &v, motion.right, 3000.0, HELICITY_SAMPLES);
        let left = storm_relative_helicity(&z, &u, &v, motion.left, 3000.0, HELICITY_SAMPLES);
        assert!(right > 0.0, "clockwise hodograph gives negative SRH: {right}");
        assert!(right > left, "right {right} should exceed left {left}");
    }

    #[test]
    fn a_hodograph_with_no_shear_moves_with_the_mean_wind_rather_than_a_nan() {
        let z: Vec<f32> = (0..40).map(|k| k as f32 * 500.0).collect();
        let u = vec![7.0_f32; 40];
        let v = vec![-3.0_f32; 40];
        let motion = bunkers_motion(&z, &u, &v).unwrap();
        assert!(motion.deviation_undefined);
        for mover in [motion.right, motion.left, motion.mean_wind] {
            assert!((mover.u_m_per_s - 7.0).abs() < 1e-5 && (mover.v_m_per_s + 3.0).abs() < 1e-5);
        }
        let nan = vec![f32::NAN; 40];
        assert!(bunkers_motion(&z, &nan, &v).is_err());
    }

    #[test]
    fn bulk_shear_is_the_magnitude_of_the_wind_difference_across_the_layer() {
        let z = vec![0.0_f32, 1000.0, 6000.0, 12000.0];
        let u = vec![0.0_f32, 3.0, 20.0, 30.0];
        let v = vec![0.0_f32, 4.0, 0.0, 0.0];
        assert!((bulk_shear(&z, &u, &v, 0.0, 1000.0) - 5.0).abs() < 1.0e-4);
        assert!((bulk_shear(&z, &u, &v, 0.0, 6000.0) - 20.0).abs() < 1.0e-4);
    }

    #[test]
    fn an_unnamed_storm_motion_is_refused_by_value() {
        assert_eq!(
            StormMotionChoice::parse("bunkers-right").unwrap(),
            StormMotionChoice::BunkersRight
        );
        assert_eq!(
            StormMotionChoice::parse("bunkers-left").unwrap(),
            StormMotionChoice::BunkersLeft
        );
        let message = StormMotionChoice::parse("mean-wind").unwrap_err().to_string();
        assert!(message.contains("Bunkers"), "{message}");
        assert!(message.contains("reproduced"), "{message}");
    }

    #[test]
    fn the_report_json_is_parseable_and_names_both_authorities() {
        let (p, t, q) = column(41, 500.0);
        let (z, u, v) = hodograph(41, 500.0);
        // The report column is the mass levels; the surface words are supplied
        // separately, as base.F:3149-3157 does.
        let mass: Vec<f32> = z[1..].to_vec();
        let pressure: Vec<f32> = p[1..].iter().map(|value| value * 100.0).collect();
        let temperature: Vec<f32> = t[1..].iter().map(|value| value + 273.15).collect();
        let report = sounding_parameters(
            &mass,
            &pressure,
            &temperature,
            &q[1..],
            &u[1..],
            &v[1..],
            100_000.0,
            303.15,
            0.017,
            u[0],
            v[0],
            DEFAULT_MIXED_LAYER_DEPTH_M,
            Adiabat::PseudoadiabaticLiquid,
            DEFAULT_PRESSURE_INCREMENT_PA,
            StormMotionChoice::BunkersRight,
        )
        .unwrap();
        let json = report.to_json();
        crate::json::parse(json.as_bytes()).expect("the report is not parseable JSON");
        assert!(json.contains("getcape.F:14-520"), "{json}");
        assert!(json.contains("Bunkers"), "{json}");
        assert!(json.contains("\"storm_motion\":\"bunkers-right\""), "{json}");
        assert!(report.surface_based.cape > 0.0);
        let text = report.to_text();
        assert!(text.contains("surface parcel"), "{text}");
        assert!(text.contains("0-1,0-3 km SRH"), "{text}");
    }
}
