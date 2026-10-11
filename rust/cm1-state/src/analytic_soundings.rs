//! CM1 r21.1 `base.F`'s analytic sounding arms beside `isnd = 5`.
//!
//! Every branch of `base.F:134-2036` that builds its base state from
//! constants (and not from `input_sounding`) is transcribed here statement by
//! statement, in default REAL, in CM1's own evaluation order:
//!
//! | `isnd` | `base.F` | profile |
//! |---|---|---|
//! | 1 | 134-162 | dry adiabatic, theta = 300 K |
//! | 2 | 169-206 | dry isothermal, T = 250 K |
//! | 3 | 215-253 | dry, constant dT/dz = 6.5 K/km |
//! | 4 | 263-355 | saturated, neutrally stable (Bryan and Fritsch 2002) |
//! | 8 | 787-835 | dry (or `testcase`-moist), constant d(theta)/dz |
//! | 9 | 842-905 | constant Brunt-Vaisala frequency |
//! | 10 | 913-1181 | saturated, constant moist Brunt-Vaisala frequency |
//! | 11 | 1185-1340 | saturated, constant theta-e (Bryan and Rotunno 2009) |
//! | 12 | 1345-1371 | dry, adiabatic below 960 m, 10 K/km above |
//! | 13 | 1376-1443 | dry, three constant-N layers (Klemp 2011) |
//! | 14 | 1448-1478 | dry convective boundary layer (Sullivan and Patton 2011) |
//! | 18 | 1621-1655 | dry sheared boundary layer (Moeng and Sullivan 1994) |
//! | 19 | 1660-1762 | BOMEX shallow cumulus (Siebesma et al. 2003), `testcase = 11` variant |
//! | 20 | 1767-1837 | RICO precipitating shallow cumulus (vanZanten et al. 2011) |
//! | 21 | 1842-1891 | RCEMIP (Wing et al. 2018), reads `tsk0` |
//! | 22 | 1896-1921 | stable boundary layer (Beare et al. 2006) |
//! | 23 | 1926-2034 | diurnal shallow cumulus over land (Brown et al. 2002) |
//!
//! `isnd = 5` stays in [`crate::storm_initialization`], `isnd = 7`/`17` in
//! [`crate::idealized_initialization`].  Two branches are refused by value:
//! `isnd = 6` (`base.F:455-460` stops CM1: "code has been removed") and
//! `isnd = 15` (`base.F:1491-1510` requires `testcase = 4` or `5`, and
//! `constants.F:78-98` then changes `rd`, `cp` and `xlv` for the whole model;
//! GM1's kernels and ArWen's physics are built with the default constants).
//!
//! Each arm is evaluated on one flat column: CM1 loops the same statements over
//! every `(i,j)` with the same `zh`, so every column is the same words.  Only
//! the interior levels `k = 1..nk` are returned; `base.F:2044-2077`'s
//! `bcs`/`extrapbcs` overwrite whatever an arm left in the ghost levels.
//!
//! Where an arm reads a level CM1 never set, the grid is refused instead of
//! inventing that word: the inversion arms (14, 18, 22) read `pi0(k-1)` at
//! `k = 1` when the first level is above the inversion (the ghost `pi0(0)` is
//! still `cm1.F:1379`'s allocation zero), and the tabulated arms (19, 20, 23)
//! leave `th0`/`qv0` unset above their last node.

use crate::idealized_initialization::{invalid, rslf, require_finite, CP, G, P00, RD, RV};
use crate::Result;

// constants.F:100-120, the default (testcase not 4 or 5) branch.
const CPV: f32 = 1870.0_f32;
const CPL: f32 = 4190.0_f32;
const CPI: f32 = 2106.0_f32;
const XLV: f32 = 2_501_000.0_f32;
const XLS: f32 = 2_834_000.0_f32;
/// constants.F:18 `to = 273.15`.
const TO: f32 = 273.15_f32;
/// cm1.F:312 in the single-precision build.
pub const TSMALL: f32 = 0.0001_f32;

/// Every `base.F` branch this module builds.
pub const TRANSCRIBED_ANALYTIC_ISND: [i32; 17] =
    [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23];

/// The `isnd` values `init3d.F:295-306` copies `qc0` into `qa(nqc)` for.
pub const ISND_WITH_INITIAL_CLOUD: [i32; 5] = [4, 9, 10, 11, 15];
/// The `isnd` values `init3d.F:308-317` copies `qi0` into `qa(nqi)` for (with `iice = 1`).
pub const ISND_WITH_INITIAL_ICE: [i32; 3] = [4, 9, 10];

/// `base.F:267-268`: the moist benchmark's two words.  `input.F:181` declares
/// them in the `input` module, `param.F:6335-6336` zeroes them, and
/// `base.F:267-268` sets them in source; `init3d.F:625-645` reads `qt_mb`
/// again for the `iinit = 4` bubble.
pub const CM1_THEC_MB_K: f32 = 320.0_f32;
pub const CM1_QT_MB: f32 = 0.020_f32;
/// `run/namelist.input`'s `tsk0`, read by `isnd = 11` and `21`.
pub const CM1_TSK0_K: f32 = 299.28_f32;

/// CM1's derived constants, evaluated at run time in default REAL exactly as
/// `constants.F:124-158` evaluates them.
#[derive(Clone, Copy, Debug)]
struct Constants {
    reps: f32,
    eps: f32,
    cpdrd: f32,
    rovcp: f32,
    rp00: f32,
    lv1: f32,
    lv2: f32,
    ls1: f32,
    ls2: f32,
}

impl Constants {
    fn cm1() -> Self {
        Self {
            reps: RV / RD,
            eps: RD / RV,
            cpdrd: CP / RD,
            rovcp: RD / CP,
            // constants.F:18: `real, parameter :: rp00 = 1.0/p00`.
            rp00: 1.0_f32 / P00,
            lv1: XLV + (CPL - CPV) * TO,
            lv2: CPL - CPV,
            ls1: XLS + (CPI - CPV) * TO,
            ls2: CPI - CPV,
        }
    }
}

/// `cm1libs.F:69-118`, the Tao et al. (1989) branch both of GM1's schemes
/// (CM1 `ptype = 5` and `7`) take.
fn rsif(p: f32, t: f32) -> f32 {
    let eps = RD / RV;
    let esi = (611.2_f32 * (21.874_558_4_f32 * (t - 273.15_f32) / (t - 7.66_f32)).exp())
        .min(p * 0.5_f32);
    eps * esi / (p - esi)
}

/// What an arm leaves for `base.F:2212-2246`'s surface words.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArmSurface {
    /// `psurf` and `tsurf` both set (so `base.F:2212`'s branch is taken).
    PsurfTsurf { psurf: f32, tsurf: f32 },
    /// `isnd = 4`: `psurf` set and `tsurf = cgs1*t0(1)+cgs2*t0(2)+cgs3*t0(3)`
    /// over the arm's own `t0`, which needs the grid's `cgs` weights.
    PsurfTsurfFromT0 { psurf: f32, t0: [f32; 3] },
    /// The arm leaves `psurf = tsurf = 0` (`base.F:127-129`), so
    /// `base.F:2227-2239` extrapolates hydrostatically from level 1.
    Unset,
}

/// What `base.F:2212-2246` needs from an analytic arm: its surface rule and,
/// for the tabulated arms (19, 20, 23), the `qsurf` it caps at the first
/// node's specific humidity (`base.F:1753-1759`).  Every other arm's `qsurf`
/// is `cgs1*qv0(1)+cgs2*qv0(2)+cgs3*qv0(3)`, either assigned in the arm or
/// by `base.F:2245`'s fallback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticSurface {
    pub surface: ArmSurface,
    pub qsurf_override: Option<f32>,
}

/// Everything besides the grid an analytic arm reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticSoundingControls {
    pub imoist: i32,
    pub testcase: i32,
    /// param.F's `iice`: 1 for every ice-bearing scheme (both of GM1's).
    pub iice: bool,
    /// `tsk0` (namelist), read by `isnd = 11` and `21`.
    pub tsk0_k: f32,
    /// `base.F:267-268`, read by `isnd = 4` (and `init3d.F`'s `iinit = 4`).
    pub thec_mb_k: f32,
    pub qt_mb: f32,
    /// CM1's `eqtset` after `param.F:936-950` (which forces 1 for `ptype` 4,
    /// 7, 50-55), read by `isnd = 10`'s moist lapse rate.
    pub eqtset: i32,
}

impl AnalyticSoundingControls {
    pub fn cm1(imoist: i32, testcase: i32) -> Self {
        Self {
            imoist,
            testcase,
            iice: true,
            tsk0_k: CM1_TSK0_K,
            thec_mb_k: CM1_THEC_MB_K,
            qt_mb: CM1_QT_MB,
            eqtset: 2,
        }
    }
}

/// One arm's column on `k = 1..nk`.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalyticSoundingColumn {
    pub isnd: i32,
    pub th0: Vec<f32>,
    pub qv0: Vec<f32>,
    pub qc0: Vec<f32>,
    pub qi0: Vec<f32>,
    pub pi0: Vec<f32>,
    pub prs0: Vec<f32>,
    pub t0: Vec<f32>,
    pub rh0: Vec<f32>,
    pub surface: ArmSurface,
    pub qsurf_override: Option<f32>,
}

impl AnalyticSoundingColumn {
    pub fn analytic_surface(&self) -> AnalyticSurface {
        AnalyticSurface { surface: self.surface, qsurf_override: self.qsurf_override }
    }
}

fn arm_refusal(isnd: i32) -> Option<String> {
    match isnd {
        6 => Some(
            "base.F:455-460: isnd=6 stops CM1 r21.1 ('isnd = 6 code has been removed'); there is \
             no base state to transcribe"
                .to_owned(),
        ),
        15 => Some(
            "base.F:1491-1510 runs isnd=15 (DYCOMS-II) only with testcase=4 or 5, and \
             constants.F:78-98 then sets rd=287.0, cp=1015.0 or 1004.0 and xlv=2.47e6 or 2.50e6 \
             for the whole model; GM1's dynamics kernels and ArWen's physics are built with \
             constants.F's default rd=287.04, cp=1005.7, xlv=2.501e6, so this base state would \
             be integrated by different thermodynamics than the one that built it"
                .to_owned(),
        ),
        _ => None,
    }
}

/// Build one `base.F` analytic column on the mass heights `zh(1..nk)`.
///
/// `zh_above` is `zh(nk+1)`, the ghost height `isnd = 9` and `13` compare
/// against when deciding whether the next level starts a new layer; it never
/// changes a returned word (the switch it triggers applies to level `nk+1`),
/// but it is read so the traversal is CM1's.
pub fn analytic_sounding_column(
    isnd: i32,
    zh: &[f32],
    zh_above: f32,
    controls: AnalyticSoundingControls,
) -> Result<AnalyticSoundingColumn> {
    if let Some(reason) = arm_refusal(isnd) {
        return Err(invalid(reason));
    }
    if !TRANSCRIBED_ANALYTIC_ISND.contains(&isnd) {
        return Err(invalid(format!(
            "base.F has no analytic isnd={isnd} arm transcribed here; the analytic arms are \
             {TRANSCRIBED_ANALYTIC_ISND:?} (isnd=5 is the Weisman-Klemp lane, 7 and 17 read \
             input_sounding)"
        )));
    }
    if !matches!(controls.imoist, 0 | 1) {
        return Err(invalid(format!(
            "README.namelist:308 admits imoist=0 or 1; got {}",
            controls.imoist
        )));
    }
    let nk = zh.len();
    if nk < 3 {
        return Err(invalid(
            "an analytic base state needs at least three mass levels (base.F's cgs extrapolation reads k=1..3)",
        ));
    }
    for k in 0..nk {
        require_finite("analytic sounding mass height", zh[k])?;
        let previous = if k == 0 { 0.0_f32 } else { zh[k - 1] };
        if zh[k] <= previous {
            return Err(invalid(format!(
                "analytic sounding mass heights must be strictly increasing above z=0; level {k} \
                 is {} m after {previous} m",
                zh[k]
            )));
        }
    }
    require_finite("analytic sounding ghost height zh(nk+1)", zh_above)?;
    let mut column = AnalyticSoundingColumn {
        isnd,
        th0: vec![0.0; nk],
        qv0: vec![0.0; nk],
        qc0: vec![0.0; nk],
        qi0: vec![0.0; nk],
        pi0: vec![0.0; nk],
        prs0: vec![0.0; nk],
        t0: vec![0.0; nk],
        rh0: vec![0.0; nk],
        surface: ArmSurface::Unset,
        qsurf_override: None,
    };
    let c = Constants::cm1();
    match isnd {
        1 => isnd1(&mut column, zh, c),
        2 => isnd2(&mut column, zh),
        3 => isnd3(&mut column, zh, c),
        4 => isnd4(&mut column, zh, c, controls)?,
        8 => isnd8(&mut column, zh, c, controls.testcase),
        9 => isnd9(&mut column, zh, zh_above, c),
        10 => isnd10(&mut column, zh, c, controls)?,
        11 => isnd11(&mut column, zh, c, controls)?,
        12 => isnd12(&mut column, zh),
        13 => isnd13(&mut column, zh, zh_above, c),
        14 => isnd_inversion(&mut column, zh, 974.0, 1074.0, 0.08, 0.003, "isnd=14")?,
        18 => {
            // base.F:1625-1633, all in default REAL.
            let delz = 60.0_f32;
            let lr1 = 8.0_f32 / delz;
            let zl1 = 465.0_f32 - delz / 2.0_f32;
            let zl2 = 465.0_f32 + delz / 2.0_f32;
            isnd_inversion(&mut column, zh, zl1, zl2, lr1, 0.003, "isnd=18")?
        }
        19 => isnd_tabulated(&mut column, zh, c, controls, 19)?,
        20 => isnd_tabulated(&mut column, zh, c, controls, 20)?,
        21 => isnd21(&mut column, zh, c, controls),
        22 => isnd22(&mut column, zh)?,
        23 => isnd_tabulated(&mut column, zh, c, controls, 23)?,
        _ => unreachable!("guarded by TRANSCRIBED_ANALYTIC_ISND"),
    }
    // base.F:2088-2100: imoist=0 zeroes qv0 and rh0 (thv0 = th0 follows in
    // the padded builder).  qc0/qi0 are not read by a dry run.
    if controls.imoist == 0 {
        column.qv0.iter_mut().for_each(|value| *value = 0.0);
        column.rh0.iter_mut().for_each(|value| *value = 0.0);
        column.qc0.iter_mut().for_each(|value| *value = 0.0);
        column.qi0.iter_mut().for_each(|value| *value = 0.0);
    }
    for k in 0..nk {
        for (label, value) in [
            ("th0", column.th0[k]),
            ("qv0", column.qv0[k]),
            ("qc0", column.qc0[k]),
            ("qi0", column.qi0[k]),
            ("pi0", column.pi0[k]),
            ("prs0", column.prs0[k]),
        ] {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "base.F isnd={isnd} leaves {label} non-finite at level {}: the grid reaches \
                     outside the heights this analytic profile is defined for",
                    k + 1
                )));
            }
        }
        if column.th0[k] <= 0.0 || column.pi0[k] <= 0.0 || column.prs0[k] <= 0.0 {
            return Err(invalid(format!(
                "base.F isnd={isnd} gives a non-positive th0/pi0/prs0 at level {} (z = {} m): \
                 the column is taller than this analytic atmosphere",
                k + 1,
                zh[k]
            )));
        }
        if column.qv0[k] < 0.0 || column.qc0[k] < 0.0 || column.qi0[k] < 0.0 {
            return Err(invalid(format!(
                "base.F isnd={isnd} gives negative moisture at level {}",
                k + 1
            )));
        }
    }
    Ok(column)
}

/// base.F:134-162.
fn isnd1(col: &mut AnalyticSoundingColumn, zh: &[f32], c: Constants) {
    let th_sfc = 300.0_f32;
    let pi_sfc = 1.0_f32;
    for k in 0..zh.len() {
        col.th0[k] = th_sfc;
        col.pi0[k] = pi_sfc - G * zh[k] / (CP * th_sfc);
        col.prs0[k] = P00 * col.pi0[k].powf(c.cpdrd);
    }
    let thsurf = th_sfc;
    let psurf = P00 * pi_sfc.powf(c.cpdrd);
    let tsurf = thsurf * (psurf * c.rp00).powf(c.rovcp);
    col.surface = ArmSurface::PsurfTsurf { psurf, tsurf };
}

/// base.F:169-206.
fn isnd2(col: &mut AnalyticSoundingColumn, zh: &[f32]) {
    let t_sfc = 250.0_f32;
    let prs_sfc = P00;
    let hs = RD * t_sfc / G;
    for k in 0..zh.len() {
        col.prs0[k] = prs_sfc * (-zh[k] / hs).exp();
        col.pi0[k] = (col.prs0[k] / P00).powf(RD / CP);
        col.th0[k] = t_sfc / col.pi0[k];
    }
    col.surface = ArmSurface::PsurfTsurf { psurf: prs_sfc, tsurf: t_sfc };
}

/// base.F:215-253.
fn isnd3(col: &mut AnalyticSoundingColumn, zh: &[f32], c: Constants) {
    let th_sfc = 300.0_f32;
    let prs_sfc = P00;
    let lapse = 0.0065_f32;
    for k in 0..zh.len() {
        col.t0[k] = th_sfc - lapse * zh[k];
        col.prs0[k] = P00 * (col.t0[k] / th_sfc).powf(G / (lapse * RD));
        col.pi0[k] = (col.prs0[k] / P00).powf(RD / CP);
        // base.F:236: `1-g/(lapse*cp)` mixes the INTEGER 1 into a REAL
        // expression, which Fortran converts to 1.0 before subtracting.
        col.th0[k] = th_sfc * (col.t0[k] / th_sfc).powf(1.0_f32 - G / (lapse * CP));
    }
    let psurf = prs_sfc;
    let tsurf = th_sfc * (psurf * c.rp00).powf(c.rovcp);
    col.surface = ArmSurface::PsurfTsurf { psurf, tsurf };
}

/// base.F:263-355: the Bryan and Fritsch (2002) saturated, neutrally stable
/// sounding.  Both fixed points are CM1's: 20 relaxed passes at level 1 and,
/// above it, relaxation until two passes agree to 1e-4 K (CM1 stops after 100).
fn isnd4(
    col: &mut AnalyticSoundingColumn,
    zh: &[f32],
    c: Constants,
    controls: AnalyticSoundingControls,
) -> Result<()> {
    let thec_mb = controls.thec_mb_k;
    let qt_mb = controls.qt_mb;
    require_finite("isnd=4 thec_mb", thec_mb)?;
    require_finite("isnd=4 qt_mb", qt_mb)?;
    if thec_mb <= 0.0 || !(0.0..1.0).contains(&qt_mb) || qt_mb == 0.0 {
        return Err(invalid(format!(
            "base.F:267-268's thec_mb must be a positive temperature and qt_mb a total water \
             mixing ratio in (0, 1); got thec_mb={thec_mb}, qt_mb={qt_mb}"
        )));
    }
    let nk = zh.len();
    let prs_sfc = 100_000.0_f32;
    // rd/(cp+cpl*qt_mb): the same REAL expression each time it is written.
    let cpml = CP + CPL * qt_mb;
    let exponent = RD / cpml;
    let reversible_t = |prs: f32, qv: f32, tmp: f32| -> f32 {
        // base.F:292-293 and :322-323.
        thec_mb * ((prs / (1.0_f32 + qv / c.eps) / P00).powf(exponent))
            / ((c.lv1 - c.lv2 * tmp) * qv / (tmp * cpml)).exp()
    };

    // base.F:280-296, lowest model level.
    col.prs0[0] = prs_sfc;
    let mut qv1 = qt_mb;
    let mut tmp = thec_mb;
    let mut tlast = tmp;
    for _ in 0..20 {
        let thbar = (tmp * (P00 / col.prs0[0]).powf(RD / CP)) * (1.0_f32 + qv1 * c.reps)
            / (1.0_f32 + qt_mb);
        col.pi0[0] = 1.0_f32 - G * zh[0] / (CP * thbar);
        col.prs0[0] = P00 * col.pi0[0].powf(c.cpdrd);
        qv1 = rslf(col.prs0[0], tmp);
        tmp = reversible_t(col.prs0[0], qv1, tmp);
        tmp = tlast + 0.3_f32 * (tmp - tlast);
        tlast = tmp;
    }
    col.t0[0] = tmp;
    col.th0[0] = tmp / col.pi0[0];
    col.rh0[0] = 1.0;
    col.qv0[0] = rslf(col.prs0[0], tmp);
    let mut t1 = tmp;
    qv1 = col.qv0[0];
    let mut th1 = col.th0[0];

    // base.F:307-344.
    for k in 1..nk {
        let mut tlast = th1;
        let mut t2;
        let mut qv2 = qv1;
        let mut th2 = th1;
        let mut n = 0;
        col.pi0[k] = col.pi0[k - 1] - G * (zh[k] - zh[k - 1]) / (CP * th2);
        loop {
            n += 1;
            th2 = tlast;
            t2 = th2 * col.pi0[k];
            let thbar = 0.5_f32
                * (th1 * (1.0_f32 + qv1 * c.reps) / (1.0_f32 + qt_mb)
                    + th2 * (1.0_f32 + qv2 * c.reps) / (1.0_f32 + qt_mb));
            col.pi0[k] = col.pi0[k - 1] - G * (zh[k] - zh[k - 1]) / (CP * thbar);
            col.prs0[k] = P00 * col.pi0[k].powf(c.cpdrd);
            qv2 = rslf(col.prs0[k], t2);
            t2 = reversible_t(col.prs0[k], qv2, t2);
            th2 = t2 / col.pi0[k];
            if (th2 - tlast).abs() > 0.0001_f32 && n < 100 {
                tlast = tlast + 0.3_f32 * (th2 - tlast);
                continue;
            } else if n >= 100 {
                return Err(invalid(format!(
                    "base.F:330-332: the isnd=4 fixed point at level {} did not settle in 100 \
                     passes, where CM1 prints 'stuck in loop!' and stops",
                    k + 1
                )));
            }
            break;
        }
        col.t0[k] = t2;
        col.th0[k] = th2;
        col.qv0[k] = rslf(col.prs0[k], col.th0[k] * col.pi0[k]);
        col.rh0[k] = 1.0;
        th1 = th2;
        qv1 = qv2;
        t1 = t2;
    }
    let _ = t1;
    // base.F:346-348.
    for k in 0..nk {
        col.qc0[k] = qt_mb - col.qv0[k];
    }
    col.surface = ArmSurface::PsurfTsurfFromT0 { psurf: prs_sfc, t0: [col.t0[0], col.t0[1], col.t0[2]] };
    Ok(())
}

/// base.F:787-835.
fn isnd8(col: &mut AnalyticSoundingColumn, zh: &[f32], c: Constants, testcase: i32) {
    let th_sfc = 300.0_f32;
    let pi_sfc = 1.0_f32;
    let mut lapse = -0.003_058_1_f32;
    let mut rhval = 0.0_f32;
    if testcase == 6 || testcase == 10 {
        lapse = 0.005;
    }
    if testcase == 10 {
        rhval = 0.90;
    }
    if testcase == 15 {
        lapse = 0.005;
        rhval = 0.80;
    }
    for k in 0..zh.len() {
        col.th0[k] = th_sfc + lapse * zh[k];
        col.pi0[k] = pi_sfc - (G / (CP * lapse)) * (col.th0[k] / th_sfc).ln();
        col.prs0[k] = P00 * col.pi0[k].powf(CP / RD);
        col.t0[k] = col.th0[k] * col.pi0[k];
        col.rh0[k] = rhval;
        col.qv0[k] = rhval * rslf(col.prs0[k], col.t0[k]);
    }
    let psurf = P00 * pi_sfc.powf(CP / RD);
    let tsurf = th_sfc * pi_sfc;
    let _ = c;
    col.surface = ArmSurface::PsurfTsurf { psurf, tsurf };
}

/// `base.F:861-869` and `:1400-1408`: the surface words when the first level
/// is already above the first layer top.
fn shift_constant_n_layer(pi_sfc: &mut f32, th_sfc: &mut f32, ns1: f32, zl1: f32) {
    if ns1.abs() < 1.0e-6_f32 {
        *pi_sfc = *pi_sfc - G * zl1 / (CP * *th_sfc);
    } else {
        *pi_sfc = *pi_sfc + G * G / (CP * ns1 * *th_sfc) * ((-ns1 * zl1 / G).exp() - 1.0_f32);
        *th_sfc = *th_sfc * (ns1 * zl1 / G).exp();
    }
}

/// base.F:842-905.
fn isnd9(col: &mut AnalyticSoundingColumn, zh: &[f32], zh_above: f32, c: Constants) {
    let ns1 = 0.0001_f32;
    let ns2 = 0.0000_f32;
    let zl1 = 40_000.0_f32;
    let mut th_sfc = 288.00_f32;
    let mut pi_sfc = 1.0_f32;
    let psurf = P00 * pi_sfc.powf(CP / RD);
    let tsurf = th_sfc * pi_sfc;
    let (mut zsfc, mut ns);
    if zh[0] < zl1 {
        zsfc = 0.0_f32;
        ns = ns1;
    } else {
        shift_constant_n_layer(&mut pi_sfc, &mut th_sfc, ns1, zl1);
        zsfc = zl1;
        ns = ns2;
    }
    let nk = zh.len();
    for k in 0..nk {
        col.qv0[k] = 0.0;
        col.qc0[k] = 0.0;
        col.rh0[k] = 0.0;
        let thv0;
        if ns.abs() < 1.0e-6_f32 {
            thv0 = th_sfc;
            col.pi0[k] = pi_sfc - G * (zh[k] - zsfc) / (CP * th_sfc);
        } else {
            thv0 = th_sfc * (ns * (zh[k] - zsfc) / G).exp();
            col.pi0[k] = pi_sfc
                + G * G / (CP * ns * th_sfc) * ((-ns * (zh[k] - zsfc) / G).exp() - 1.0_f32);
        }
        col.prs0[k] = P00 * col.pi0[k].powf(c.cpdrd);
        col.th0[k] = thv0;
        for _ in 0..20 {
            col.t0[k] = col.th0[k] * col.pi0[k];
            col.qv0[k] = col.rh0[k] * rslf(col.prs0[k], col.t0[k]);
            col.th0[k] = thv0 * (1.0_f32 + col.qv0[k] + col.qc0[k])
                / (1.0_f32 + c.reps * col.qv0[k]);
        }
        let z_next = if k + 1 < nk { zh[k + 1] } else { zh_above };
        if z_next > zl1 && zsfc < 1.0_f32 {
            shift_constant_n_layer(&mut pi_sfc, &mut th_sfc, ns1, zl1);
            zsfc = zl1;
            ns = ns2;
        }
    }
    col.surface = ArmSurface::PsurfTsurf { psurf, tsurf };
}

/// base.F:913-1181: saturated, constant moist N (reversible), with CM1's
/// `qcval = 0.005` cloud water.  `iice = 1` takes base.F:1094-1123's
/// liquid/ice partition through `goddard_module`'s `T0K`, `T00K` and `RT0`,
/// which only `goddard.F:1177-1178,1347` (`consat`, called for `ptype = 2`)
/// ever sets; with both of GM1's schemes (`ptype = 5`, `7`) they are still
/// their static-storage zero, so `fliq = 0` and CM1 partitions every level to
/// ice.  That is what CM1 r21.1 does with these schemes, so it is what this
/// port does.
fn isnd10(
    col: &mut AnalyticSoundingColumn,
    zh: &[f32],
    c: Constants,
    controls: AnalyticSoundingControls,
) -> Result<()> {
    // goddard_module's T0K/T00K/RT0 as ptype 5 and 7 leave them.
    let t0k = 0.0_f32;
    let t00k = 0.0_f32;
    let rt0 = 0.0_f32;
    let qcval = 0.005_f32;
    let ns1 = 0.0_f32;
    let ns2 = 4.0e-4_f32;
    let zl1 = 111_700.0_f32;
    let nk = zh.len();
    for k in 0..nk {
        col.rh0[k] = 1.0;
        col.qc0[k] = qcval;
    }
    let th_sfc = 288.0_f32;
    let pi_sfc = 1.0_f32;
    let psurf = P00 * pi_sfc.powf(CP / RD);
    let tsurf = th_sfc * pi_sfc;
    let mut ns = ns1;
    let p_sfc = P00 * pi_sfc.powf(CP / RD);
    let t_sfc = th_sfc * pi_sfc;
    let qv_sfc = col.rh0[0] * rslf(p_sfc, t_sfc);
    let ql_sfc = qcval;
    let thv_sfc = th_sfc * (1.0_f32 + c.reps * qv_sfc) / (1.0_f32 + qv_sfc + ql_sfc);
    let mut thn = th_sfc;
    let mut thlast = thn;
    // ---- k = 1 (base.F:955-1016) ----
    let mut thv0 = vec![0.0_f32; nk];
    // base.F:983 sets desdt at level 1 only; the eqtset=1 branch above it
    // (base.F:1122-1123) reads whatever the last level-1 pass left.
    let mut desdt = 0.0_f32;
    let mut n = 1;
    col.qv0[0] = qv_sfc;
    col.qc0[0] = ql_sfc;
    loop {
        col.th0[0] = thn;
        thv0[0] = col.th0[0] * (1.0_f32 + c.reps * col.qv0[0]) / (1.0_f32 + col.qv0[0] + col.qc0[0]);
        col.pi0[0] = pi_sfc - G * zh[0] / (CP * 0.5_f32 * (thv0[0] + thv_sfc));
        col.prs0[0] = P00 * col.pi0[0].powf(CP / RD);
        col.t0[0] = col.th0[0] * col.pi0[0];
        let qvs = col.rh0[0] * rslf(col.prs0[0], col.t0[0]);
        col.qv0[0] = qvs;
        col.qc0[0] = qcval;
        let qtm = qv_sfc + ql_sfc;
        let qtp = col.qv0[0] + col.qc0[0];
        let tavg = 0.5_f32 * (col.t0[0] + t_sfc);
        let lhv = c.lv1 - c.lv2 * tavg;
        let qvavg = 0.5_f32 * (qv_sfc + col.qv0[0]);
        let qlavg = 0.5_f32 * (ql_sfc + col.qc0[0]);
        let qtavg = qvavg + qlavg;
        let tm29 = tavg - 29.65_f32;
        desdt = 17.67_f32 * (273.15_f32 - 29.65_f32) / (tm29 * tm29);
        let drdt = 17.67_f32 * (273.15_f32 - 29.65_f32) * qvavg / (tm29 * tm29);
        let gamma = if controls.eqtset == 2 {
            G * (1.0_f32 + qtavg) * (1.0_f32 + lhv * qvavg / RD / tavg)
                / (CP + CPV * qvavg + CPL * qlavg + lhv * drdt)
        } else {
            G * (1.0_f32 + qtavg) * (RD / (RD + RV * qvavg) + lhv * qvavg / RD / tavg)
                / (CP + lhv * (1.0_f32 + qvavg * c.reps) * qvavg * desdt)
        };
        let tp = t_sfc
            * (zh[0]
                * (((ns / G) + ((1.0_f32 + qtp) / (1.0_f32 + qtm)).ln() / zh[0])
                    / (1.0_f32 + tavg * drdt / (c.eps + qvavg))
                    - gamma / tavg))
                .exp();
        thn = tp / col.pi0[0];
        n += 1;
        if (thn - thlast).abs() > 0.0001_f32 && n < 200 {
            thn = thlast + 0.3_f32 * (thn - thlast);
            thlast = thn;
            continue;
        } else if n >= 200 {
            return Err(invalid(
                "base.F:1000-1003: the isnd=10 level-1 fixed point did not settle in 200 passes, \
                 where CM1 prints 'stuck in loop (111)!' and stops",
            ));
        }
        break;
    }
    col.th0[0] = thn;
    col.prs0[0] = P00 * col.pi0[0].powf(CP / RD);
    col.t0[0] = col.th0[0] * col.pi0[0];
    col.qv0[0] = col.rh0[0] * rslf(col.prs0[0], col.t0[0]);
    col.qc0[0] = qcval;
    // ---- k = 2..nk (base.F:1018-1172) ----
    ns = ns1;
    for k in 1..nk {
        let mut tflag = 0;
        for _niter in 0..2 {
            if tflag >= 2 {
                continue;
            }
            let (delz, pim, thm, thvm, qtm, qvm, qlm, qim, tm);
            if zh[k] > zl1 && ns == ns1 && tflag == 0 && zh[k - 1] < zl1 {
                tflag = 1;
                delz = zl1 - zh[k - 1];
                pim = col.pi0[k - 1];
                thm = col.th0[k - 1];
                thvm = thv0[k - 1];
                qtm = col.qv0[k - 1] + col.qc0[k - 1] + col.qi0[k - 1];
                qvm = col.qv0[k - 1];
                qlm = col.qc0[k - 1];
                qim = col.qi0[k - 1];
                tm = col.t0[k - 1];
            } else if tflag == 1 {
                tflag = 0;
                ns = ns2;
                delz = zh[k] - zl1;
                pim = col.pi0[k];
                thm = col.th0[k];
                thvm = thv0[k];
                qtm = col.qv0[k] + col.qc0[k] + col.qi0[k];
                qvm = col.qv0[k];
                qlm = col.qc0[k];
                qim = col.qi0[k];
                tm = col.t0[k];
            } else {
                tflag = 2;
                delz = zh[k] - zh[k - 1];
                pim = col.pi0[k - 1];
                thm = col.th0[k - 1];
                thvm = thv0[k - 1];
                qtm = col.qv0[k - 1] + col.qc0[k - 1] + col.qi0[k - 1];
                qvm = col.qv0[k - 1];
                qlm = col.qc0[k - 1];
                qim = col.qi0[k - 1];
                tm = col.t0[k - 1];
            }
            col.th0[k] = thm;
            col.qv0[k] = qvm;
            col.qc0[k] = qlm;
            col.qi0[k] = qim;
            thv0[k] = thvm;
            let mut thlast = thm;
            let mut n = 0;
            let mut thn = thlast;
            thlast = thn;
            let mut tp;
            loop {
                col.th0[k] = thn;
                thv0[k] = col.th0[k] * (1.0_f32 + c.reps * col.qv0[k])
                    / (1.0_f32 + col.qv0[k] + col.qc0[k] + col.qi0[k]);
                col.pi0[k] = pim - G * delz / (CP * 0.5_f32 * (thv0[k] + thvm));
                col.prs0[k] = P00 * col.pi0[k].powf(CP / RD);
                col.t0[k] = col.th0[k] * col.pi0[k];
                let (qvavg, qlavg, qtavg, drdt, lhv, cpml, tavg, qtp);
                if !controls.iice {
                    let qvs = col.rh0[k] * rslf(col.prs0[k], col.t0[k]);
                    col.qv0[k] = qvs;
                    col.qc0[k] = qcval;
                    qtp = col.qv0[k] + col.qc0[k];
                    let prsavg = 0.5_f32 * (col.prs0[k - 1] + col.prs0[k]);
                    tavg = 0.5_f32 * (col.t0[k] + tm);
                    qvavg = col.rh0[k] * rslf(prsavg, tavg);
                    qlavg = 0.5_f32 * (qlm + col.qc0[k]);
                    qtavg = 0.5_f32 * (qtm + qtp);
                    let tm29 = tavg - 29.65_f32;
                    drdt = 17.67_f32 * (273.15_f32 - 29.65_f32) * qvavg / (tm29 * tm29);
                    lhv = c.lv1 - c.lv2 * tavg;
                    cpml = CP + CPV * qvavg + CPL * qlavg;
                } else {
                    let qvl = col.rh0[k] * rslf(col.prs0[k], col.t0[k]);
                    let qvi = col.rh0[k] * rsif(col.prs0[k], col.t0[k]);
                    let fliq = ((col.t0[k] - t00k) * rt0).min(1.0_f32).max(0.0_f32);
                    let fice = 1.0_f32 - fliq;
                    let qvs = fliq * qvl + fice * qvi;
                    col.qv0[k] = qvs;
                    col.qc0[k] = fliq * qcval;
                    col.qi0[k] = fice * qcval;
                    qtp = col.qv0[k] + col.qc0[k] + col.qi0[k];
                    let prsavg = 0.5_f32 * (col.prs0[k - 1] + col.prs0[k]);
                    tavg = 0.5_f32 * (col.t0[k] + tm);
                    let qvl = col.rh0[k] * rslf(prsavg, tavg);
                    let qvi = col.rh0[k] * rsif(prsavg, tavg);
                    let fliq = ((tavg - t00k) * rt0).min(1.0_f32).max(0.0_f32);
                    let fice = 1.0_f32 - fliq;
                    qvavg = fliq * qvl + fice * qvi;
                    qlavg = 0.5_f32 * (qlm + col.qc0[k]);
                    qtavg = 0.5_f32 * (qtm + qtp);
                    let tm29 = tavg - 29.65_f32;
                    let tm7 = tavg - 7.66_f32;
                    let mut d = fliq * 17.67_f32 * (273.15_f32 - 29.65_f32) * qvl / (tm29 * tm29)
                        + fice * 21.874_558_4_f32 * (273.15_f32 - 7.66_f32) * qvi / (tm7 * tm7);
                    if tavg > t00k && tavg < t0k {
                        d = d + (qvl - qvi) * rt0;
                    }
                    drdt = d;
                    lhv = fliq * (c.lv1 - c.lv2 * tavg) + fice * (c.ls1 - c.ls2 * tavg);
                    cpml = CP + CPV * qvavg + CPL * qlavg + CPI * (qtavg - qlavg - qvavg);
                }
                let gamma = if controls.eqtset == 2 {
                    G * (1.0_f32 + qtavg) * (1.0_f32 + lhv * qvavg / RD / tavg)
                        / (cpml + lhv * drdt)
                } else {
                    G * (1.0_f32 + qtavg) * (RD / (RD + RV * qvavg) + lhv * qvavg / RD / tavg)
                        / (CP + lhv * (1.0_f32 + qvavg * c.reps) * qvavg * desdt)
                };
                tp = tm
                    * (delz
                        * (((ns / G) + ((1.0_f32 + qtp) / (1.0_f32 + qtm)).ln() / delz)
                            / (1.0_f32 + tavg * drdt / (c.eps + qvavg))
                            - gamma / tavg))
                        .exp();
                thn = tp / col.pi0[k];
                n += 1;
                if (thn - thlast).abs() > 0.0001_f32 && n < 200 {
                    thn = thlast + 0.3_f32 * (thn - thlast);
                    thlast = thn;
                    continue;
                } else if n >= 200 {
                    return Err(invalid(format!(
                        "base.F:1133-1136: the isnd=10 fixed point at level {} did not settle in \
                         200 passes, where CM1 prints 'stuck in loop (333)!' and stops",
                        k + 1
                    )));
                }
                break;
            }
            col.t0[k] = tp;
            col.th0[k] = thn;
            col.prs0[k] = P00 * col.pi0[k].powf(CP / RD);
            col.t0[k] = col.th0[k] * col.pi0[k];
            if !controls.iice {
                col.qv0[k] = col.rh0[k] * rslf(col.prs0[k], col.t0[k]);
            } else {
                let qvl = col.rh0[k] * rslf(col.prs0[k], col.t0[k]);
                let qvi = col.rh0[k] * rsif(col.prs0[k], col.t0[k]);
                let fliq = ((col.t0[k] - t00k) * rt0).min(1.0_f32).max(0.0_f32);
                let fice = 1.0_f32 - fliq;
                col.qv0[k] = fliq * qvl + fice * qvi;
                col.qc0[k] = fliq * qcval;
                col.qi0[k] = fice * qcval;
            }
        }
    }
    col.surface = ArmSurface::PsurfTsurf { psurf, tsurf };
    Ok(())
}

/// base.F:3354-3375, `getthe`.
fn getthe(ttype: i32, p: f32, t: f32, qv: f32, qt: f32, c: Constants) -> f32 {
    if ttype == 1 {
        let pd = p / (1.0_f32 + c.reps * qv);
        let cpm = CP + CPL * qt;
        let lhv = c.lv1 - c.lv2 * t;
        t * (P00 / pd).powf(RD / cpm) * (lhv * qv / (cpm * t)).exp()
    } else {
        t * (P00 / p).powf(0.2854_f32 * (1.0_f32 - 0.28_f32 * qv))
            * (((3376.0_f32 / t) - 2.54_f32) * qv * (1.0_f32 + 0.81_f32 * qv)).exp()
    }
}

/// base.F:3381-3426, `revthe`: `t` is the first guess on input.
fn revthe(ttype: i32, the: f32, p: f32, qt: f32, t: &mut f32, qv: &mut f32, c: Constants) -> Result<()> {
    let mut n = 1;
    let mut tlast = *t;
    let mut tinc = 0.0_f32;
    loop {
        *t = tlast + tinc;
        *qv = rslf(p, *t);
        let thx = getthe(ttype, p, *t, *qv, qt, c);
        let diff = the - thx;
        if diff.abs() > 0.0001_f32 && n < 50 {
            n += 1;
            tinc = 0.30_f32 * diff;
            tlast = *t;
            if tinc.abs() >= 0.0001_f32 {
                continue;
            }
        } else if n >= 50 {
            return Err(invalid(
                "base.F:3415-3421: revthe did not settle in 50 passes, where CM1 prints 'n \
                 exceeded 50!' and stops",
            ));
        }
        return Ok(());
    }
}

/// base.F:1185-1340: saturated, constant theta-e (pseudoadiabatic, `ttype = 2`).
fn isnd11(
    col: &mut AnalyticSoundingColumn,
    zh: &[f32],
    c: Constants,
    controls: AnalyticSoundingControls,
) -> Result<()> {
    if controls.imoist == 0 {
        return Err(invalid(
            "base.F:1187-1195 stops CM1 for isnd=11 with imoist=0 ('isnd=11 requires imoist=1')",
        ));
    }
    let zl1 = 15_000.0_f32;
    let ttype = 2;
    let psurf = 101_510.0_f32;
    let pi_sfc = (psurf * c.rp00).powf(c.rovcp);
    let mut th_sfc = (controls.tsk0_k - 3.0_f32) / pi_sfc;
    let tsurf = th_sfc * pi_sfc;
    let t_sfc0 = th_sfc * pi_sfc;
    let p_sfc = P00 * pi_sfc.powf(CP / RD);
    let qv_sfc0 = rslf(p_sfc, t_sfc0);
    let qt_sfc = qv_sfc0;
    let thv_sfc = th_sfc * (1.0_f32 + c.reps * qv_sfc0) / (1.0_f32 + qv_sfc0);
    let the_sfc = getthe(ttype, p_sfc, t_sfc0, qv_sfc0, qt_sfc, c);
    th_sfc = 0.0;
    let mut t_sfc = t_sfc0;
    let mut qv_sfc = 0.0_f32;
    revthe(ttype, the_sfc, p_sfc, qt_sfc, &mut t_sfc, &mut qv_sfc, c)?;
    let mut pi1 = pi_sfc;
    let mut z1 = 0.0_f32;
    let mut t1 = t_sfc;
    let mut thv1 = thv_sfc;
    let mut thv2 = thv1;
    let mut flag = 0;
    let mut pi_layer = pi_sfc;
    let mut zsfc = 0.0_f32;
    for k in 0..zh.len() {
        let mut z2 = zh[k];
        let mut t2 = t1;
        let mut tlast = 0.0_f32;
        let (mut pi2, mut p2, mut qv2, mut th2, mut ql2) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        if z2 <= zl1 {
            let theq = the_sfc;
            while (t2 - tlast).abs() > 0.0001_f32 {
                tlast = t2;
                pi2 = pi1 - G * (z2 - z1) / (CP * 0.5_f32 * (thv1 + thv2));
                p2 = P00 * pi2.powf(CP / RD);
                revthe(ttype, theq, p2, qt_sfc, &mut t2, &mut qv2, c)?;
                th2 = t2 / pi2;
                ql2 = if ttype == 1 { qt_sfc - qv2 } else { 0.0 };
                thv2 = th2 * (1.0_f32 + c.reps * qv2) / (1.0_f32 + qv2 + ql2);
            }
        } else {
            let ns = 4.0e-4_f32;
            if flag == 0 {
                z2 = zl1;
                t2 = t1;
                tlast = 0.0;
                let theq = the_sfc;
                while (t2 - tlast).abs() > 0.0001_f32 {
                    tlast = t2;
                    pi2 = pi1 - G * (z2 - z1) / (CP * 0.5_f32 * (thv1 + thv2));
                    p2 = P00 * pi2.powf(CP / RD);
                    revthe(ttype, theq, p2, qt_sfc, &mut t2, &mut qv2, c)?;
                    th2 = t2 / pi2;
                    ql2 = if ttype == 1 { qt_sfc - qv2 } else { 0.0 };
                    thv2 = th2 * (1.0_f32 + c.reps * qv2) / (1.0_f32 + qv2 + ql2);
                }
                flag = 1;
                th_sfc = thv2;
                pi_layer = pi2;
                zsfc = z2;
                z2 = zh[k];
            }
            if ns.abs() < 1.0e-6_f32 {
                thv2 = th_sfc;
                pi2 = pi_layer - G * (z2 - zsfc) / (CP * th_sfc);
            } else {
                thv2 = th_sfc * (ns * (z2 - zsfc) / G).exp();
                pi2 = pi_layer
                    + G * G / (CP * ns * th_sfc) * ((-ns * (z2 - zsfc) / G).exp() - 1.0_f32);
            }
            p2 = P00 * pi2.powf(c.cpdrd);
            th2 = thv2;
            for _ in 0..20 {
                t2 = th2 * pi2;
                qv2 = rslf(p2, t2);
                ql2 = if ttype == 1 { qt_sfc - qv2 } else { 0.0 };
                th2 = thv2 * (1.0_f32 + qv2 + ql2) / (1.0_f32 + c.reps * qv2);
            }
        }
        t1 = t2;
        thv1 = thv2;
        z1 = z2;
        pi1 = pi2;
        col.pi0[k] = pi2;
        col.prs0[k] = p2;
        col.t0[k] = t2;
        col.th0[k] = th2;
        col.rh0[k] = 1.0;
        col.qv0[k] = qv2;
        col.qc0[k] = ql2;
    }
    col.surface = ArmSurface::PsurfTsurf { psurf, tsurf };
    Ok(())
}

/// base.F:1345-1371.
fn isnd12(col: &mut AnalyticSoundingColumn, zh: &[f32]) {
    let pi_sfc = 1.0_f32;
    let th_sfc = 300.0_f32;
    let zl1 = 960.0_f32;
    let lapse = 0.010_f32;
    let pisfc = pi_sfc - G * zl1 / (CP * th_sfc);
    for k in 0..zh.len() {
        if zh[k] <= zl1 {
            col.th0[k] = th_sfc;
            col.pi0[k] = pi_sfc - G * zh[k] / (CP * th_sfc);
        } else {
            col.th0[k] = th_sfc + lapse * (zh[k] - zl1);
            col.pi0[k] = pisfc - (G / (CP * lapse)) * (col.th0[k] / th_sfc).ln();
        }
        col.prs0[k] = P00 * col.pi0[k].powf(CP / RD);
        col.t0[k] = col.th0[k] * col.pi0[k];
    }
    col.surface = ArmSurface::Unset;
}

/// base.F:1376-1443.
fn isnd13(col: &mut AnalyticSoundingColumn, zh: &[f32], zh_above: f32, c: Constants) {
    let ns1 = 0.0001_f32;
    let ns2 = 0.0004_f32;
    let ns3 = 0.0001_f32;
    let zl1 = 2000.0_f32;
    let zl2 = 3000.0_f32;
    let mut th_sfc = 288.00_f32;
    let mut pi_sfc = 1.0_f32;
    let psurf = P00 * pi_sfc.powf(CP / RD);
    let tsurf = th_sfc * pi_sfc;
    let (mut zsfc, mut ns);
    if zh[0] < zl1 {
        zsfc = 0.0_f32;
        ns = ns1;
    } else {
        shift_constant_n_layer(&mut pi_sfc, &mut th_sfc, ns1, zl1);
        zsfc = zl1;
        ns = ns2;
    }
    let nk = zh.len();
    for k in 0..nk {
        if ns.abs() < 1.0e-6_f32 {
            col.th0[k] = th_sfc;
            col.pi0[k] = pi_sfc - G * (zh[k] - zsfc) / (CP * th_sfc);
        } else {
            col.th0[k] = th_sfc * (ns * (zh[k] - zsfc) / G).exp();
            col.pi0[k] = pi_sfc
                + G * G / (CP * ns * th_sfc) * ((-ns * (zh[k] - zsfc) / G).exp() - 1.0_f32);
        }
        col.prs0[k] = P00 * col.pi0[k].powf(c.cpdrd);
        col.t0[k] = col.th0[k] * col.pi0[k];
        let z_next = if k + 1 < nk { zh[k + 1] } else { zh_above };
        if z_next > zl1 && zsfc < 1.0_f32 {
            shift_constant_n_layer(&mut pi_sfc, &mut th_sfc, ns1, zl1);
            zsfc = zl1;
            ns = ns2;
        } else if z_next > zl2 && zsfc < zl2 {
            // base.F:1431-1437: note the ns1 test on the ns2 layer, as written.
            if ns1.abs() < 1.0e-6_f32 {
                pi_sfc = pi_sfc - G * (zl2 - zl1) / (CP * th_sfc);
            } else {
                pi_sfc = pi_sfc
                    + G * G / (CP * ns2 * th_sfc) * ((-ns2 * (zl2 - zl1) / G).exp() - 1.0_f32);
                th_sfc = th_sfc * (ns2 * (zl2 - zl1) / G).exp();
            }
            zsfc = zl2;
            ns = ns3;
        }
    }
    col.surface = ArmSurface::PsurfTsurf { psurf, tsurf };
}

/// base.F:1448-1478 (isnd=14) and base.F:1621-1655 (isnd=18): adiabatic
/// below `zl1`, `lr1` through the inversion to `zl2`, `lr2` above.
fn isnd_inversion(
    col: &mut AnalyticSoundingColumn,
    zh: &[f32],
    zl1: f32,
    zl2: f32,
    lr1: f32,
    lr2: f32,
    label: &str,
) -> Result<()> {
    if zh[0] > zl1 {
        return Err(invalid(format!(
            "base.F's {label} integrates pi0(k) from pi0(k-1) above {zl1} m, and this grid's first \
             level ({} m) is already above it, so CM1 would read the ghost pi0(0), still \
             cm1.F's allocation zero; put the first level at or below {zl1} m",
            zh[0]
        )));
    }
    let pi_sfc = 1.0_f32;
    let th_sfc = 300.0_f32;
    for k in 0..zh.len() {
        if zh[k] <= zl1 {
            col.th0[k] = th_sfc;
            col.pi0[k] = pi_sfc - G * zh[k] / (CP * th_sfc);
        } else if zh[k] <= zl2 {
            col.th0[k] = th_sfc + (zh[k] - zl1) * lr1;
            col.pi0[k] = col.pi0[k - 1]
                - G * (zh[k] - zh[k - 1]) / (CP * 0.5_f32 * (col.th0[k] + col.th0[k - 1]));
        } else {
            col.th0[k] = th_sfc + (zl2 - zl1) * lr1 + (zh[k] - zl2) * lr2;
            col.pi0[k] = col.pi0[k - 1]
                - G * (zh[k] - zh[k - 1]) / (CP * 0.5_f32 * (col.th0[k] + col.th0[k - 1]));
        }
        col.prs0[k] = P00 * col.pi0[k].powf(CP / RD);
        col.t0[k] = col.th0[k] * col.pi0[k];
    }
    col.surface = ArmSurface::Unset;
    Ok(())
}

/// base.F:1896-1921.
fn isnd22(col: &mut AnalyticSoundingColumn, zh: &[f32]) -> Result<()> {
    let pi_sfc = 1.0_f32;
    let th_sfc = 265.0_f32;
    let zl1 = 100.0_f32;
    let lr1 = 0.01_f32;
    if zh[0] > zl1 {
        return Err(invalid(format!(
            "base.F:1911 integrates pi0(k) from pi0(k-1) above {zl1} m, and this grid's first \
             level ({} m) is already above it, so CM1 would read the ghost pi0(0), still \
             cm1.F's allocation zero; put the first level at or below {zl1} m (the stable \
             boundary-layer case runs dz = 6.25 m)",
            zh[0]
        )));
    }
    for k in 0..zh.len() {
        if zh[k] <= zl1 {
            col.th0[k] = th_sfc;
            col.pi0[k] = pi_sfc - G * zh[k] / (CP * th_sfc);
        } else {
            col.th0[k] = th_sfc + (zh[k] - zl1) * lr1;
            col.pi0[k] = col.pi0[k - 1]
                - G * (zh[k] - zh[k - 1]) / (CP * 0.5_f32 * (col.th0[k] + col.th0[k - 1]));
        }
        col.prs0[k] = P00 * col.pi0[k].powf(CP / RD);
        col.t0[k] = col.th0[k] * col.pi0[k];
        col.rh0[k] = 0.0;
        col.qv0[k] = col.rh0[k] * rslf(col.prs0[k], col.t0[k]);
    }
    col.surface = ArmSurface::Unset;
    Ok(())
}

/// base.F:1660-1762 (19), 1767-1837 (20) and 1926-2034 (23): piecewise
/// linear theta and specific humidity between tabulated nodes, converted to
/// mixing ratio, then integrated hydrostatically from the surface.
fn isnd_tabulated(
    col: &mut AnalyticSoundingColumn,
    zh: &[f32],
    c: Constants,
    controls: AnalyticSoundingControls,
    isnd: i32,
) -> Result<()> {
    let (z, th, qv, prs_sfc): (Vec<f32>, Vec<f32>, Vec<f32>, f32) = match isnd {
        19 if controls.testcase == 11 => {
            let z = vec![0.0_f32, 352.5, 442.5, 3000.0, 20_000.0];
            let th3 = 298.1_f32;
            let qv3 = 7.8e-3_f32;
            let th = vec![296.6_f32, 296.6, th3, th3 + 0.003_f32 * (z[3] - z[2]), th3 + 0.003_f32 * (z[4] - z[2])];
            let qv = vec![
                11.8e-3_f32,
                11.8e-3,
                qv3,
                qv3 - 0.004e-3_f32 * (z[3] - z[2]),
                qv3 - 0.004e-3_f32 * (z[4] - z[2]),
            ];
            (z, th, qv, 100_000.0)
        }
        19 => (
            vec![0.0_f32, 520.0, 1480.0, 2000.0, 3000.0],
            vec![298.7_f32, 298.7, 302.4, 308.2, 311.85],
            vec![0.0170_f32, 0.0163, 0.0107, 0.0042, 0.0030],
            101_500.0,
        ),
        20 => (
            vec![0.0_f32, 740.0, 3260.0, 4000.0],
            vec![297.9_f32, 297.9, 312.6644, 317.0],
            vec![0.0160_f32, 0.0138, 0.0024, 0.0018],
            101_540.0,
        ),
        23 => {
            let mut z = vec![0.0_f32, 50.0, 350.0, 650.0, 700.0, 1300.0, 2500.0, 5500.0];
            let mut th = vec![299.0_f32, 301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2];
            let mut qv = vec![0.0152_f32, 0.01517, 0.01498, 0.0148, 0.0147, 0.0135, 0.003, 0.003];
            // base.F:1976-1978.
            let z9 = 10_000.0_f32;
            let th9 = th[7] + (z9 - z[7]) * (th[7] - th[6]) / (z[7] - z[6]);
            let qv9 = qv[7] + (z9 - z[7]) * (qv[7] - qv[6]) / (z[7] - z[6]);
            z.push(z9);
            th.push(th9);
            qv.push(qv9);
            (z, th, qv, 97_000.0)
        }
        _ => unreachable!(),
    };
    let top = *z.last().expect("nodes");
    let nk = zh.len();
    if zh[nk - 1] > top {
        return Err(invalid(format!(
            "base.F's isnd={isnd} defines theta and humidity only up to {top} m and leaves \
             th0/qv0 unset above it; this grid's top level is {} m",
            zh[nk - 1]
        )));
    }
    let pi_sfc = (prs_sfc * c.rp00).powf(c.rovcp);
    let thv_sfc = th[0] * (1.0_f32 + c.reps * qv[0]) / (1.0_f32 + qv[0]);
    let mut thv0 = vec![0.0_f32; nk];
    for k in 0..nk {
        let mut value = 0.0_f32;
        for m in 1..z.len() {
            if zh[k] <= z[m] {
                col.th0[k] = th[m - 1] + (th[m] - th[m - 1]) * (zh[k] - z[m - 1]) / (z[m] - z[m - 1]);
                value = qv[m - 1] + (qv[m] - qv[m - 1]) * (zh[k] - z[m - 1]) / (z[m] - z[m - 1]);
                break;
            }
        }
        // base.F:1728 (19) keeps a max(0, ...); 20 and 23 convert directly.
        col.qv0[k] = if isnd == 19 {
            (value / (1.0_f32 - value)).max(0.0_f32)
        } else {
            value / (1.0_f32 - value)
        };
        thv0[k] = col.th0[k] * (1.0_f32 + c.reps * col.qv0[k]) / (1.0_f32 + col.qv0[k]);
    }
    col.pi0[0] = pi_sfc - G * zh[0] / (CP * 0.5_f32 * (thv_sfc + thv0[0]));
    for k in 1..nk {
        col.pi0[k] = col.pi0[k - 1]
            - G * (zh[k] - zh[k - 1]) / (CP * 0.5_f32 * (thv0[k] + thv0[k - 1]));
    }
    for k in 0..nk {
        col.prs0[k] = P00 * col.pi0[k].powf(CP / RD);
        col.rh0[k] = col.qv0[k] / rslf(col.prs0[k], col.th0[k] * col.pi0[k]);
        col.t0[k] = col.th0[k] * col.pi0[k];
    }
    let psurf = prs_sfc;
    let tsurf = th[0] * (psurf * c.rp00).powf(c.rovcp);
    col.surface = ArmSurface::PsurfTsurf { psurf, tsurf };
    // base.F:1753-1759: qsurf is the first node's (specific) humidity when
    // the lowest level is not drier than it.
    if controls.imoist == 1 && col.qv0[0] >= qv[0] {
        col.qsurf_override = Some(qv[0]);
    }
    Ok(())
}

/// base.F:1842-1891: RCEMIP.  `psurf`/`tsurf` are left unset (base.F:127-129).
fn isnd21(col: &mut AnalyticSoundingColumn, zh: &[f32], c: Constants, controls: AnalyticSoundingControls) {
    let ztrop = 15_000.0_f32;
    let qsfc = 0.01865_f32;
    // base.F:1847: `10.0**(-11)`, a constant expression gfortran folds.
    let qt = 1.0e-11_f32;
    let zq1 = 4000.0_f32;
    let zq2 = 7500.0_f32;
    let tsfc = controls.tsk0_k;
    let tvsfc = tsfc * (1.0_f32 + 0.608_f32 * qsfc);
    let psfc = 101_480.0_f32;
    let _pisfc = (psfc / P00).powf(RD / CP);
    let lr = 0.0067_f32;
    let tvtrop = tvsfc - lr * ztrop;
    let ptrop = psfc * (tvtrop / tvsfc).powf(G / (RD * lr));
    for k in 0..zh.len() {
        let tv0;
        if zh[k] < ztrop {
            let a = zh[k] / zq2;
            col.qv0[k] = qsfc * (-zh[k] / zq1).exp() * (-(a * a)).exp();
            tv0 = tvsfc - lr * zh[k];
            col.prs0[k] = psfc * ((tvsfc - lr * zh[k]) / tvsfc).powf(G / (RD * lr));
        } else {
            col.qv0[k] = qt;
            tv0 = tvtrop;
            col.prs0[k] = ptrop * (-(G * (zh[k] - ztrop) / (RD * tvtrop))).exp();
        }
        col.t0[k] = tv0 / (1.0_f32 + 0.608_f32 * col.qv0[k]);
        col.pi0[k] = (col.prs0[k] * c.rp00).powf(c.rovcp);
        col.th0[k] = col.t0[k] / col.pi0[k];
        col.qv0[k] = col.qv0[k] / (1.0_f32 - col.qv0[k]);
    }
    col.surface = ArmSurface::Unset;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heights(nk: usize, dz: f32) -> Vec<f32> {
        (0..nk).map(|k| 0.5_f32 * (dz * k as f32 + dz * (k + 1) as f32)).collect()
    }

    fn column(isnd: i32, nk: usize, dz: f32) -> Result<AnalyticSoundingColumn> {
        let zh = heights(nk, dz);
        let above = 0.5_f32 * (dz * nk as f32 + dz * (nk + 1) as f32);
        analytic_sounding_column(isnd, &zh, above, AnalyticSoundingControls::cm1(1, 0))
    }

    #[test]
    fn every_transcribed_arm_builds_a_positive_column_on_a_grid_it_is_defined_for() {
        for isnd in TRANSCRIBED_ANALYTIC_ISND {
            let (nk, dz) = match isnd {
                19 => (60, 50.0),
                20 => (80, 50.0),
                22 => (60, 6.25),
                14 | 18 => (60, 20.0),
                _ => (40, 250.0),
            };
            let col = column(isnd, nk, dz).unwrap_or_else(|e| panic!("isnd={isnd}: {e}"));
            assert!(col.th0.iter().all(|v| *v > 150.0 && *v < 1000.0), "isnd={isnd}");
            assert!(col.pi0.windows(2).all(|w| w[1] < w[0]), "isnd={isnd} pi0 must fall with height");
        }
    }

    #[test]
    fn the_moist_benchmark_is_saturated_with_cm1s_total_water() {
        let col = column(4, 40, 250.0).unwrap();
        for k in 0..40 {
            assert_eq!(col.rh0[k], 1.0);
            // base.F:347: qc0 = qt_mb - qv0, in REAL.
            assert_eq!(col.qc0[k].to_bits(), (CM1_QT_MB - col.qv0[k]).to_bits());
            assert!(col.qc0[k] > 0.0);
        }
        // Wet-equivalent potential temperature is constant: theta falls with
        // height only as fast as condensate is released, so theta rises.
        assert!(col.th0[39] > col.th0[0]);
        assert!(matches!(col.surface, ArmSurface::PsurfTsurfFromT0 { psurf, .. } if psurf == 100_000.0));
    }

    #[test]
    fn isnd10_partitions_every_level_to_ice_as_cm1_does_without_goddard() {
        let col = column(10, 40, 250.0).unwrap();
        assert_eq!(col.qc0[0], 0.005);
        for k in 1..40 {
            assert_eq!(col.qc0[k], 0.0, "k={k}");
            assert_eq!(col.qi0[k], 0.005, "k={k}");
        }
    }

    #[test]
    fn refused_arms_name_their_cm1_reason() {
        for isnd in [6, 15] {
            let error = column(isnd, 40, 250.0).unwrap_err().to_string();
            assert!(error.contains("base.F"), "{error}");
        }
        assert!(column(16, 40, 250.0).is_err());
        assert!(column(22, 40, 250.0).unwrap_err().to_string().contains("pi0(0)"));
        assert!(column(19, 40, 250.0).unwrap_err().to_string().contains("3000"));
    }

    /// CM1 r21.1's own base-state columns (gfortran 15.2, release tarball): every arm's th0, qv0,
    /// qc0 and qi0, word for word.
    #[test]
    fn every_arm_is_cm1s_column_word_for_word() {
        const TABLE: &str = include_str!("../evidence/cm1-base-columns.txt");
        let mut compared = 0;
        let mut cases = std::collections::BTreeSet::new();
        for line in TABLE.lines().filter(|line| !line.starts_with('#') && !line.is_empty()) {
            let words: Vec<&str> = line.split_whitespace().collect();
            let (case, isnd, imoist, testcase, eqtset) = (
                words[0],
                words[1].parse::<i32>().unwrap(),
                words[2].parse::<i32>().unwrap(),
                words[3].parse::<i32>().unwrap(),
                words[4].parse::<i32>().unwrap(),
            );
            let (dz, nk, field) = (words[5].parse::<f32>().unwrap(), words[6].parse::<usize>().unwrap(), words[7]);
            let cm1: Vec<u32> = words[8..].iter().map(|w| u32::from_str_radix(w, 16).unwrap()).collect();
            assert_eq!(cm1.len(), nk, "{case} {field}");
            let zh = heights(nk, dz);
            let above = 0.5_f32 * (dz * nk as f32 + dz * (nk + 1) as f32);
            let controls = AnalyticSoundingControls { eqtset, ..AnalyticSoundingControls::cm1(imoist, testcase) };
            let col = analytic_sounding_column(isnd, &zh, above, controls)
                .unwrap_or_else(|e| panic!("{case}: {e}"));
            let ours = match field {
                "th0" => &col.th0,
                "qv0" => &col.qv0,
                "qc0" => &col.qc0,
                "qi0" => &col.qi0,
                other => panic!("unknown field {other}"),
            };
            for k in 0..nk {
                assert_eq!(ours[k].to_bits(), cm1[k], "{case} {field} k={}: {} vs CM1 {}", k + 1, ours[k],
                           f32::from_bits(cm1[k]));
            }
            compared += nk;
            cases.insert(case);
        }
        // 17 arms, isnd 8's two testcases, isnd 19's testcase 11, isnd 10 at eqtset 2.
        assert_eq!(cases.len(), 21);
        assert!(compared > 2500, "{compared}");
    }

    #[test]
    fn dry_arms_carry_no_vapour_and_imoist_zero_clears_moist_arms() {
        for isnd in [1, 2, 3, 12, 13] {
            let col = column(isnd, 40, 250.0).unwrap();
            assert!(col.qv0.iter().all(|v| *v == 0.0), "isnd={isnd}");
        }
        let zh = heights(40, 250.0);
        let dry = analytic_sounding_column(4, &zh, 10_125.0, AnalyticSoundingControls::cm1(0, 0)).unwrap();
        assert!(dry.qv0.iter().chain(&dry.qc0).all(|v| *v == 0.0));
        assert!(analytic_sounding_column(11, &zh, 10_125.0, AnalyticSoundingControls::cm1(0, 0)).is_err());
    }
}
