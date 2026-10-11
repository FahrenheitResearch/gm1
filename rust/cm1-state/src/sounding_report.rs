//! Bind CM1's parcel and hodograph report to GM1's own initial base state.
//!
//! [`crate::getcape`] is the transcription; this module is the call site.  It
//! builds the single base-state column the run will integrate -- the `i=1,j=1`
//! column `base.F:3162` names -- runs `getcape.F` over it three times the way
//! `base.F:3165-3167` does, and returns the report as canonical JSON for the
//! run receipt.
//!
//! The column is built through the same two entry points the real initializer
//! uses, on a one-by-one horizontal domain.  Both lanes are horizontally
//! homogeneous with no terrain, so that column is word for word the column
//! every other column of the real domain holds; nothing here re-derives a base
//! state of its own.
//!
//! Three surface words are needed that the mass levels do not carry, and CM1
//! names each of them:
//!
//! * `psurf`/`tsurf`/`qsurf` for the thermodynamic surface level of the
//!   report, from `base.F:600-603` on the `isnd=7` lane and
//!   `base.F:440-449` on the `isnd=5` lane.
//! * the surface wind, from `parcel.F:67-71`'s `cgs1/cgs2/cgs3` bottom
//!   extrapolation of `ua`/`va` -- CM1's own way of putting a horizontal
//!   velocity on the ground -- with the weights `param.F:9038` forms from this
//!   grid's own levels rather than a uniform-grid constant.

use crate::getcape::{
    sounding_parameters, Adiabat, SoundingParameters, StormMotionChoice,
    DEFAULT_MIXED_LAYER_DEPTH_M, DEFAULT_PRESSURE_INCREMENT_PA,
};
use crate::idealized_initialization::{
    initialize_idealized_domain, IdealizedBaseStateLane, IdealizedInitializationFrame,
    IdealizedInitializationSelection, TranslatingFrame, WarmBubbleControls,
};
use crate::storm_initialization::initialize_analytic_idealized_domain;
use crate::terrain::{extrapolation_weights, DeclaredVerticalCoordinate};
use crate::{Result, StateError};

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

/// `base.F:378` for the `isnd=5` lane: the constant PBL mixing ratio the
/// analytic surface value is capped at (`base.F:446`).
const WK_QV_PBL: f32 = 0.014;
/// `base.F:373`.
const WK_PRS_SFC: f32 = 100_000.0;
/// `base.F:376-377` constants, repeated by `getcape.F:100-114`.
const RD: f32 = 287.04;
const CP: f32 = 1005.7;
const P00: f32 = 100_000.0;

/// The selectable half of the report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoundingReportControls {
    pub adiabat: Adiabat,
    pub mixed_layer_depth_m: f32,
    pub pressure_increment_pa: f32,
    pub storm_motion: StormMotionChoice,
}

impl Default for SoundingReportControls {
    fn default() -> Self {
        Self {
            adiabat: Adiabat::PseudoadiabaticLiquid,
            mixed_layer_depth_m: DEFAULT_MIXED_LAYER_DEPTH_M,
            pressure_increment_pa: DEFAULT_PRESSURE_INCREMENT_PA,
            storm_motion: StormMotionChoice::BunkersRight,
        }
    }
}

/// The vertical coordinate the report column is built on.
pub enum ReportCoordinate<'a> {
    /// `param.F:7789-7811`'s uniform midpoints, derived from `dz`.
    Uniform { nz: usize, dz_m: f32 },
    /// `param.F:7814-8139`'s declared coordinate, the levels the run carries.
    Declared(&'a DeclaredVerticalCoordinate),
}

impl ReportCoordinate<'_> {
    fn nz(&self) -> usize {
        match self {
            Self::Uniform { nz, .. } => *nz,
            Self::Declared(coordinate) => coordinate.sigma.len(),
        }
    }

    fn mass_heights(&self) -> Vec<f32> {
        match self {
            Self::Uniform { nz, dz_m } => (0..*nz)
                .map(|k| 0.5_f32 * (dz_m * k as f32 + dz_m * (k + 1) as f32))
                .collect(),
            Self::Declared(coordinate) => coordinate.sigma.clone(),
        }
    }

    fn face_heights(&self) -> Vec<f32> {
        match self {
            Self::Uniform { nz, dz_m } => (0..=*nz).map(|k| dz_m * k as f32).collect(),
            Self::Declared(coordinate) => coordinate.sigmaf.clone(),
        }
    }
}

/// Build the one-by-one base-state column this run will integrate.
fn report_frame(
    sounding_bytes: &[u8],
    selection: IdealizedInitializationSelection,
    coordinate: &ReportCoordinate<'_>,
    moving_frame: TranslatingFrame,
) -> Result<IdealizedInitializationFrame> {
    let nz = coordinate.nz();
    if nz < 4 {
        return Err(invalid(format!(
            "the t=0 sounding report is built on this run's own {nz} mass levels; \
             param.F needs at least four, and below that there is no column to \
             report the parameters of"
        )));
    }
    let heights = coordinate.mass_heights();
    // One column, at the domain origin.  The warm bubble does not enter the
    // base state, so its centre only has to be admissible.
    let xh = vec![0.5_f32];
    let yh = vec![0.5_f32];
    let controls = WarmBubbleControls::official(0.5_f32, 0.5_f32)?;
    let shape = [nz, 1usize, 1usize];
    match selection.lane() {
        IdealizedBaseStateLane::Isnd5WeismanKlempIwnd2 | IdealizedBaseStateLane::AnalyticBaseF => initialize_analytic_idealized_domain(
            selection,
            controls,
            &xh,
            &yh,
            &heights,
            &heights,
            &heights,
            shape,
            moving_frame,
        ),
        _ => initialize_idealized_domain(
            sounding_bytes,
            selection,
            controls,
            &xh,
            &yh,
            &heights,
            &heights,
            &heights,
            shape,
            moving_frame,
        ),
    }
}

/// CM1's `psurf`, `tsurf` and `qsurf` for this base state.
fn surface_words(
    frame: &IdealizedInitializationFrame,
    selection: IdealizedInitializationSelection,
    weights: [f32; 3],
) -> Result<(f32, f32, f32)> {
    let qv = frame.qv_base_kg_per_kg();
    if qv.len() < 3 {
        return Err(invalid(
            "base.F:444 forms qsurf from the lowest three base-state qv levels; \
             this column has fewer, so the surface moisture of the report parcel \
             would have to be invented",
        ));
    }
    if let Some(record) = frame.analytic_surface() {
        return Ok(analytic_surface_words(frame, record, weights));
    }
    match selection.lane() {
        IdealizedBaseStateLane::Isnd5WeismanKlempIwnd2 => {
            // base.F:440-449.
            let psurf = WK_PRS_SFC;
            let tsurf = frame.surface_theta_k() * (psurf / P00).powf(RD / CP);
            let qsurf = if qv[0] < WK_QV_PBL {
                weights[0] * qv[0] + weights[1] * qv[1] + weights[2] * qv[2]
            } else {
                WK_QV_PBL
            };
            Ok((psurf, tsurf, qsurf))
        }
        _ => {
            // base.F:596-603.  The header words of the supplied profile.
            let sounding = frame.input_sounding().ok_or_else(|| {
                invalid(
                    "base.F:600-603 takes psurf/tsurf/qsurf from the input_sounding \
                     header, and this lane preserved no header; the report's surface \
                     parcel would otherwise be extrapolated rather than the one CM1 \
                     lifts",
                )
            })?;
            let surface = sounding.surface();
            let psurf = surface.pressure_mb * 100.0_f32;
            let pi_sfc = (psurf / P00).powf(RD / CP);
            let tsurf = surface.potential_temperature_k * pi_sfc;
            let qsurf = surface.water_vapor_g_per_kg / 1000.0_f32;
            Ok((psurf, tsurf, qsurf))
        }
    }
}

/// `base.F:2212-2246` for the analytic `base.F` arms: the arm's own `psurf`
/// and `tsurf` when both are set, else level 1 extrapolated
/// hydrostatically; unset words then fall back to `prs0s` and to the `cgs`
/// extrapolations of `t0 = th0*pi0` and `qv0`.
fn analytic_surface_words(
    frame: &IdealizedInitializationFrame,
    record: crate::analytic_soundings::AnalyticSurface,
    cgs: [f32; 3],
) -> (f32, f32, f32) {
    use crate::analytic_soundings::{ArmSurface, TSMALL};
    let th0 = frame.theta_base_k();
    let pi0 = frame.pi_base();
    let qv0 = frame.qv_base_kg_per_kg();
    let qc0 = frame.qc_base_kg_per_kg();
    let qi0 = frame.qi_base_kg_per_kg();
    let extrapolate = |values: &[f32]| cgs[0] * values[0] + cgs[1] * values[1] + cgs[2] * values[2];
    let (mut psurf, mut tsurf) = match record.surface {
        ArmSurface::PsurfTsurf { psurf, tsurf } => (psurf, tsurf),
        ArmSurface::PsurfTsurfFromT0 { psurf, t0 } => (psurf, extrapolate(&t0)),
        ArmSurface::Unset => (0.0_f32, 0.0_f32),
    };
    let mut qsurf = record.qsurf_override.unwrap_or_else(|| extrapolate(qv0));
    if !(psurf >= TSMALL && tsurf >= TSMALL) {
        // base.F:2227-2239, then base.F:2243.
        let reps = 461.5_f32 / RD;
        let thsurf = extrapolate(th0);
        let q = extrapolate(qv0);
        let thvsurf = thsurf * (1.0_f32 + q * reps) / (1.0_f32 + q);
        let qvp = qv0[0].max(0.0_f32);
        let thv1 = (th0[0] + 0.0_f32) * (1.0_f32 + reps * qvp) / (1.0_f32 + qvp + (qc0[0] + qi0[0]).max(0.0_f32));
        let pi0s = pi0[0] + frame.mass_heights_m()[0] * 9.81_f32 / (CP * 0.5_f32 * (thvsurf + thv1));
        if psurf < TSMALL {
            psurf = P00 * pi0s.powf(CP / RD);
        }
    }
    if tsurf <= TSMALL {
        let t0: Vec<f32> = th0.iter().zip(pi0).take(3).map(|(t, p)| t * p).collect();
        tsurf = extrapolate(&t0);
    }
    if qsurf <= TSMALL {
        qsurf = extrapolate(qv0);
    }
    (psurf, tsurf, qsurf)
}

/// Build the complete t=0 environment report for one run's own base state.
pub fn initial_sounding_report(
    sounding_bytes: &[u8],
    selection: IdealizedInitializationSelection,
    coordinate: &ReportCoordinate<'_>,
    moving_frame: TranslatingFrame,
    controls: SoundingReportControls,
) -> Result<SoundingParameters> {
    let frame = report_frame(sounding_bytes, selection, coordinate, moving_frame)?;
    let mass = coordinate.mass_heights();
    let face = coordinate.face_heights();
    // param.F:9038.  The bottom Fornberg weights for this grid's own levels;
    // on param.F:7789-7811's uniform coordinate they are 1.875, -1.25, 0.375.
    let weights = extrapolation_weights(&mass, &face, false)?;
    let (psurf, tsurf, qsurf) = surface_words(&frame, selection, weights)?;

    // base.F:2667 already subtracted the translating frame from these two
    // carriers.  A storm motion and a helicity are ground-relative quantities,
    // so the frame is added back before the hodograph is read; reporting a
    // Bunkers motion in a frame that is itself moving with the storm would
    // give a storm-relative helicity near zero for any hodograph.
    let umove = moving_frame.u_move_m_per_s();
    let vmove = moving_frame.v_move_m_per_s();
    let u: Vec<f32> = frame
        .u_base_m_per_s()
        .iter()
        .map(|value| value + umove)
        .collect();
    let v: Vec<f32> = frame
        .v_base_m_per_s()
        .iter()
        .map(|value| value + vmove)
        .collect();
    if u.len() < 3 || v.len() < 3 {
        return Err(invalid(
            "parcel.F:67-71 extrapolates the surface wind from the lowest three \
             base-state levels; this column has fewer, so the 0-1 km shear and \
             both helicity layers would start from a wind that does not exist",
        ));
    }
    let surface_u = weights[0] * u[0] + weights[1] * u[1] + weights[2] * u[2];
    let surface_v = weights[0] * v[0] + weights[1] * v[1] + weights[2] * v[2];

    let temperature: Vec<f32> = frame
        .theta_base_k()
        .iter()
        .zip(frame.pi_base())
        .map(|(theta, pi)| theta * pi)
        .collect();

    sounding_parameters(
        &mass,
        frame.pressure_base_pa(),
        &temperature,
        frame.qv_base_kg_per_kg(),
        &u,
        &v,
        psurf,
        tsurf,
        qsurf,
        surface_u,
        surface_v,
        controls.mixed_layer_depth_m,
        controls.adiabat,
        controls.pressure_increment_pa,
        controls.storm_motion,
    )
}

// ---------------------------------------------------------------------------
// C ABI.  Same shape as `idealized_ffi`: a thread-local last error, an owned
// buffer the caller frees exactly once, and no bytes on disk.
// ---------------------------------------------------------------------------

use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn set_error(message: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into().into_bytes());
}

fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

/// Byte length of the last sounding-report error on this thread.
#[no_mangle]
pub extern "C" fn gm1_sounding_report_last_error_len() -> usize {
    LAST_ERROR.with(|slot| slot.borrow().len())
}

/// Copy the last sounding-report error into `buffer`.
///
/// # Safety
///
/// `buffer` must point to at least `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn gm1_sounding_report_last_error_copy(buffer: *mut u8, len: usize) -> usize {
    if buffer.is_null() || len == 0 {
        return 0;
    }
    LAST_ERROR.with(|slot| {
        let message = slot.borrow();
        let count = message.len().min(len);
        ptr::copy_nonoverlapping(message.as_ptr(), buffer, count);
        count
    })
}

/// Release one JSON buffer returned by [`gm1_sounding_report_json`].
///
/// # Safety
///
/// `pointer`/`len` must be exactly what one call returned, released once.
#[no_mangle]
pub unsafe extern "C" fn gm1_sounding_report_free(pointer: *mut u8, len: usize) {
    if pointer.is_null() || len == 0 {
        return;
    }
    drop(Vec::from_raw_parts(pointer, len, len));
}

/// Build one run's t=0 environment report and return it as UTF-8 JSON.
///
/// Returns null and sets the thread-local error on refusal.  `mass_heights`
/// and `face_heights` are null for `param.F:7789-7811`'s uniform coordinate
/// and otherwise carry `zh(1..nk)` and `zf(1..nk+1)`.  `storm_motion` is the
/// UTF-8 name of one Bunkers mover.
///
/// # Safety
///
/// Every pointer must be readable for its stated length for the duration of
/// the call.  `out_len` must be writable.  The returned buffer must be
/// released exactly once with [`gm1_sounding_report_free`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gm1_sounding_report_json(
    sounding: *const u8,
    sounding_len: usize,
    isnd: i32,
    iwnd: i32,
    nz: usize,
    dz_m: f32,
    mass_heights: *const f32,
    face_heights: *const f32,
    adiabat: i32,
    mixed_layer_depth_m: f32,
    pressure_increment_pa: f32,
    storm_motion: *const u8,
    storm_motion_len: usize,
    u_move_m_per_s: f32,
    v_move_m_per_s: f32,
    out_len: *mut usize,
) -> *mut u8 {
    sounding_report_json_impl(
        sounding, sounding_len, isnd, iwnd, nz, dz_m, mass_heights, face_heights, adiabat,
        mixed_layer_depth_m, pressure_increment_pa, storm_motion, storm_motion_len,
        u_move_m_per_s, v_move_m_per_s, out_len,
        ReportBaseWords { testcase: 0, imoist: 1, words: Default::default() },
    )
}

/// [`gm1_sounding_report_json`] with the base-state words beyond `isnd` and
/// `iwnd` that change the column: `testcase` (base.F's isnd=8/19 and iwnd=6
/// variants), `imoist`, the moist benchmark's `thec_mb`/`qt_mb` (isnd=4) and
/// `tsk0` (isnd=11, 21).
///
/// # Safety
///
/// As [`gm1_sounding_report_json`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gm1_sounding_report_json_v2(
    sounding: *const u8,
    sounding_len: usize,
    isnd: i32,
    iwnd: i32,
    nz: usize,
    dz_m: f32,
    mass_heights: *const f32,
    face_heights: *const f32,
    adiabat: i32,
    mixed_layer_depth_m: f32,
    pressure_increment_pa: f32,
    storm_motion: *const u8,
    storm_motion_len: usize,
    u_move_m_per_s: f32,
    v_move_m_per_s: f32,
    testcase: i32,
    imoist: i32,
    thec_mb_k: f32,
    qt_mb: f32,
    tsk0_k: f32,
    eqtset: i32,
    out_len: *mut usize,
) -> *mut u8 {
    sounding_report_json_impl(
        sounding, sounding_len, isnd, iwnd, nz, dz_m, mass_heights, face_heights, adiabat,
        mixed_layer_depth_m, pressure_increment_pa, storm_motion, storm_motion_len,
        u_move_m_per_s, v_move_m_per_s, out_len,
        ReportBaseWords {
            testcase,
            imoist,
            words: crate::idealized_initialization::AnalyticBaseWords {
                thec_mb_k,
                qt_mb,
                tsk0_k,
                eqtset,
            },
        },
    )
}

/// The base-state words beyond `isnd`/`iwnd` a report column is built with.
#[derive(Clone, Copy, Debug)]
struct ReportBaseWords {
    testcase: i32,
    imoist: i32,
    words: crate::idealized_initialization::AnalyticBaseWords,
}

#[allow(clippy::too_many_arguments)]
unsafe fn sounding_report_json_impl(
    sounding: *const u8,
    sounding_len: usize,
    isnd: i32,
    iwnd: i32,
    nz: usize,
    dz_m: f32,
    mass_heights: *const f32,
    face_heights: *const f32,
    adiabat: i32,
    mixed_layer_depth_m: f32,
    pressure_increment_pa: f32,
    storm_motion: *const u8,
    storm_motion_len: usize,
    u_move_m_per_s: f32,
    v_move_m_per_s: f32,
    out_len: *mut usize,
    base: ReportBaseWords,
) -> *mut u8 {
    clear_error();
    if out_len.is_null() {
        set_error("sounding report out_len pointer is null");
        return ptr::null_mut();
    }
    *out_len = 0;
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<String> {
        let bytes = if sounding.is_null() || sounding_len == 0 {
            &[][..]
        } else {
            slice::from_raw_parts(sounding, sounding_len)
        };
        let motion_name = if storm_motion.is_null() || storm_motion_len == 0 {
            return Err(invalid(
                "no storm motion was named; storm-relative helicity is an integral \
                 taken against one storm motion, and an unnamed motion publishes a \
                 helicity no reader can reproduce",
            ));
        } else {
            std::str::from_utf8(slice::from_raw_parts(storm_motion, storm_motion_len))
                .map_err(|error| invalid(format!("storm motion name is not UTF-8: {error}")))?
        };
        let controls = SoundingReportControls {
            adiabat: Adiabat::from_i32(adiabat)?,
            mixed_layer_depth_m,
            pressure_increment_pa,
            storm_motion: StormMotionChoice::parse(motion_name)?,
        };
        let selection = IdealizedInitializationSelection::new(isnd, iwnd, 0, 0, base.imoist)?
            .with_testcase(base.testcase)?
            .with_analytic_words(base.words)?;
        let frame = TranslatingFrame::new(u_move_m_per_s, v_move_m_per_s)?;
        let declared = if mass_heights.is_null() || face_heights.is_null() {
            None
        } else {
            Some(DeclaredVerticalCoordinate::new(
                slice::from_raw_parts(mass_heights, nz),
                slice::from_raw_parts(face_heights, nz + 1),
                dz_m,
            )?)
        };
        let coordinate = match &declared {
            None => ReportCoordinate::Uniform { nz, dz_m },
            Some(value) => ReportCoordinate::Declared(value),
        };
        let report = initial_sounding_report(bytes, selection, &coordinate, frame, controls)?;
        Ok(report.to_json())
    }));
    match result {
        Ok(Ok(json)) => {
            let mut bytes = json.into_bytes();
            bytes.shrink_to_fit();
            let len = bytes.len();
            let pointer = bytes.as_mut_ptr();
            std::mem::forget(bytes);
            *out_len = len;
            pointer
        }
        Ok(Err(error)) => {
            set_error(error.to_string());
            ptr::null_mut()
        }
        Err(_) => {
            set_error("the sounding report panicked; no report is returned");
            ptr::null_mut()
        }
    }
}

/// Run `getcape.F` over one caller-supplied column and return its eight words.
///
/// `p_mb`, `t_c` and `q_kg_per_kg` are `getcape.F:47-51`'s own units.  `out`
/// receives `cape, cin, zlcl, zlfc, zel, psource, tsource, qvsource, converged`
/// in that order.  This is the transcription itself, with no base state around
/// it, so an independent implementation can be compared against it column for
/// column.  Returns 0 on success and -1 on refusal.
///
/// # Safety
///
/// The three input pointers must each be readable for `levels` `f32` words and
/// `out` writable for nine.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gm1_getcape_column(
    levels: usize,
    p_mb: *const f32,
    t_c: *const f32,
    q_kg_per_kg: *const f32,
    parcel_source: i32,
    adiabat: i32,
    mixed_layer_depth_m: f32,
    pressure_increment_pa: f32,
    out: *mut f32,
) -> i32 {
    use crate::getcape::{getcape, GetCapeControls, ParcelSource};
    clear_error();
    if p_mb.is_null() || t_c.is_null() || q_kg_per_kg.is_null() || out.is_null() {
        set_error("gm1_getcape_column received a null pointer");
        return -1;
    }
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let controls = GetCapeControls {
            source: ParcelSource::from_i32(parcel_source)?,
            adiabat: Adiabat::from_i32(adiabat)?,
            mixed_layer_depth_m,
            pressure_increment_pa,
        };
        let value = getcape(
            controls,
            slice::from_raw_parts(p_mb, levels),
            slice::from_raw_parts(t_c, levels),
            slice::from_raw_parts(q_kg_per_kg, levels),
        )?;
        let words = [
            value.cape,
            value.cin,
            value.zlcl,
            value.zlfc,
            value.zel,
            value.psource,
            value.tsource,
            value.qvsource,
            if value.converged { 1.0 } else { 0.0 },
        ];
        ptr::copy_nonoverlapping(words.as_ptr(), out, words.len());
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_error(error.to_string());
            -1
        }
        Err(_) => {
            set_error("gm1_getcape_column panicked; no words are written");
            -1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::getcape::ParcelSource;
    use std::ffi::c_void;

    fn analytic_selection() -> IdealizedInitializationSelection {
        IdealizedInitializationSelection::new(5, 2, 0, 0, 1).unwrap()
    }

    #[test]
    fn the_analytic_lane_reports_a_convective_environment_from_its_own_base_state() {
        // base.F:363-450 with base.F:2321-2360's shear.  No external bytes, no
        // fixture: the whole column is CM1's own analytic construction, so this
        // test states what that construction's atmosphere actually is.
        let coordinate = ReportCoordinate::Uniform { nz: 40, dz_m: 500.0 };
        let report = initial_sounding_report(
            &[],
            analytic_selection(),
            &coordinate,
            TranslatingFrame::fixed(),
            SoundingReportControls::default(),
        )
        .unwrap();
        assert_eq!(report.levels, 40);
        assert!(report.surface_based.converged);
        assert!(
            report.surface_based.cape > 500.0,
            "the analytic sounding is meant to be convective: {}",
            report.surface_based.cape
        );
        assert!(report.surface_based.zlcl > 0.0 && report.surface_based.zel > 0.0);
        // base.F:2321-2360's hodograph turns and deepens, so the deep-layer
        // shear must exceed the low-level shear.
        assert!(
            report.shear_0_6km_m_per_s > report.shear_0_1km_m_per_s,
            "0-6 km {} is not above 0-1 km {}",
            report.shear_0_6km_m_per_s,
            report.shear_0_1km_m_per_s
        );
        assert!(report.shear_0_6km_m_per_s > 20.0);
        assert_eq!(report.storm_motion_choice, StormMotionChoice::BunkersRight);
    }

    #[test]
    fn the_report_is_taken_in_the_ground_relative_frame_whatever_frame_the_run_uses() {
        // base.F:2667 subtracts umove/vmove from every base-state wind word.  A
        // storm motion measured in that frame would be the storm's motion minus
        // its own translation, so the helicity would collapse toward zero for
        // any sounding at all.  The report adds the frame back, so a run in a
        // translating frame reports the same environment as the same run in a
        // fixed one.
        let coordinate = ReportCoordinate::Uniform { nz: 40, dz_m: 500.0 };
        let fixed = initial_sounding_report(
            &[],
            analytic_selection(),
            &coordinate,
            TranslatingFrame::fixed(),
            SoundingReportControls::default(),
        )
        .unwrap();
        let moving = initial_sounding_report(
            &[],
            analytic_selection(),
            &coordinate,
            TranslatingFrame::new(12.5, -3.0).unwrap(),
            SoundingReportControls::default(),
        )
        .unwrap();
        // Not bit-identical, and it cannot be: base.F:2667 subtracted umove
        // from every wind word in binary32 and this adds it back in binary32,
        // which is a rounding round trip.  What must hold is that the reported
        // environment is the same atmosphere -- a frame-relative report would
        // be smaller by hundreds of m2/s2, not by parts in a million.
        for (label, a, b) in [
            ("srh 0-1", fixed.srh_0_1km_m2_per_s2, moving.srh_0_1km_m2_per_s2),
            ("srh 0-3", fixed.srh_0_3km_m2_per_s2, moving.srh_0_3km_m2_per_s2),
            ("shear 0-6", fixed.shear_0_6km_m_per_s, moving.shear_0_6km_m_per_s),
            (
                "bunkers right u",
                fixed.bunkers.right.u_m_per_s,
                moving.bunkers.right.u_m_per_s,
            ),
            (
                "bunkers right v",
                fixed.bunkers.right.v_m_per_s,
                moving.bunkers.right.v_m_per_s,
            ),
        ] {
            let scale = a.abs().max(1.0_f32);
            assert!(
                (a - b).abs() <= 1.0e-4_f32 * scale,
                "{label}: fixed {a} vs moving {b} is more than a binary32 round trip"
            );
        }
        // And the frame really was nonzero, so the check above is not vacuous.
        assert!(moving.bunkers.right.u_m_per_s.abs() > 1.0);
    }

    #[test]
    fn the_three_parcels_are_three_different_parcels() {
        let coordinate = ReportCoordinate::Uniform { nz: 40, dz_m: 500.0 };
        let report = initial_sounding_report(
            &[],
            analytic_selection(),
            &coordinate,
            TranslatingFrame::fixed(),
            SoundingReportControls::default(),
        )
        .unwrap();
        // base.F:3165-3167 runs all three and this reports all three.
        // getcape.F:161-181 takes the maximum-theta-e parcel below 500 mb, so
        // the most-unstable CAPE can never be below the surface parcel's.
        assert!(
            report.most_unstable.cape >= report.surface_based.cape,
            "most-unstable {} is below surface-based {}",
            report.most_unstable.cape,
            report.surface_based.cape
        );
        // On this lane specifically the mixed-layer parcel coincides with the
        // surface parcel, and that is the base state saying something true:
        // base.F:428-429 caps qv at the constant qv_pbl through the boundary
        // layer, so getcape.F:216-217's depth-weighted average over the lowest
        // 500 m averages one number.  A lane whose moisture varies with height
        // separates them; this one cannot, and a report that claimed it did
        // would be describing a different sounding.
        assert_eq!(
            report.surface_based.qvsource.to_bits(),
            report.mixed_layer.qvsource.to_bits()
        );
        for parcel in [report.surface_based, report.most_unstable, report.mixed_layer] {
            assert!(parcel.converged);
            assert!(parcel.psource > 0.0 && parcel.tsource > 0.0);
        }
    }

    #[test]
    fn the_c_abi_returns_a_json_report_and_frees_it() {
        let motion = b"bunkers-right";
        let mut length: usize = 0;
        let pointer = unsafe {
            gm1_sounding_report_json(
                ptr::null(),
                0,
                5,
                2,
                40,
                500.0,
                ptr::null(),
                ptr::null(),
                1,
                500.0,
                200.0,
                motion.as_ptr(),
                motion.len(),
                0.0,
                0.0,
                &mut length,
            )
        };
        assert!(!pointer.is_null(), "the C ABI refused a valid analytic lane");
        let json = unsafe { slice::from_raw_parts(pointer, length) }.to_vec();
        unsafe { gm1_sounding_report_free(pointer, length) };
        let text = String::from_utf8(json).unwrap();
        crate::json::parse(text.as_bytes()).expect("the C ABI returned unparseable JSON");
        assert!(text.contains("\"schema\":\"gm1.sounding-parameters/v1\""));
        assert!(text.contains("srh_0_3km_m2_per_s2"));
    }

    #[test]
    fn the_c_abi_refuses_an_unnamed_storm_motion_by_value_and_returns_null() {
        let motion = b"mean-wind";
        let mut length: usize = 1;
        let pointer = unsafe {
            gm1_sounding_report_json(
                ptr::null(),
                0,
                5,
                2,
                40,
                500.0,
                ptr::null(),
                ptr::null(),
                1,
                500.0,
                200.0,
                motion.as_ptr(),
                motion.len(),
                0.0,
                0.0,
                &mut length,
            )
        };
        assert!(pointer.is_null());
        assert_eq!(length, 0);
        let len = gm1_sounding_report_last_error_len();
        let mut buffer = vec![0u8; len];
        unsafe { gm1_sounding_report_last_error_copy(buffer.as_mut_ptr(), len) };
        let message = String::from_utf8(buffer).unwrap();
        assert!(message.contains("Bunkers"), "{message}");
    }

    #[test]
    fn the_column_c_abi_matches_the_transcription_word_for_word() {
        let p: Vec<f32> = (0..41).map(|k| 1000.0_f32 * (-(k as f32 * 500.0) / 8500.0).exp()).collect();
        let t: Vec<f32> = (0..41)
            .map(|k| {
                let z = k as f32 * 500.0;
                if z < 12000.0 {
                    30.0 - 8.0e-3 * z
                } else {
                    30.0 - 8.0e-3 * 12000.0
                }
            })
            .collect();
        let q: Vec<f32> = (0..41)
            .map(|k| (0.017_f32 * (-(k as f32 * 500.0) / 2500.0).exp()).max(1.0e-8))
            .collect();
        let mut out = [0.0_f32; 9];
        let status = unsafe {
            gm1_getcape_column(
                p.len(),
                p.as_ptr(),
                t.as_ptr(),
                q.as_ptr(),
                3,
                1,
                500.0,
                200.0,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        let direct = crate::getcape::getcape(
            crate::getcape::GetCapeControls {
                source: ParcelSource::MixedLayer,
                adiabat: Adiabat::PseudoadiabaticLiquid,
                mixed_layer_depth_m: 500.0,
                pressure_increment_pa: 200.0,
            },
            &p,
            &t,
            &q,
        )
        .unwrap();
        assert_eq!(out[0].to_bits(), direct.cape.to_bits());
        assert_eq!(out[1].to_bits(), direct.cin.to_bits());
        assert_eq!(out[2].to_bits(), direct.zlcl.to_bits());
        assert_eq!(out[3].to_bits(), direct.zlfc.to_bits());
        assert_eq!(out[4].to_bits(), direct.zel.to_bits());
        assert_eq!(out[8], 1.0);
        let _unused: *const c_void = ptr::null();
    }
}
