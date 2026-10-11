//! Narrow in-memory C ABI for the source-audited idealized initializer.
//!
//! This is an ownership bridge, not a product, checkpoint, or persistence
//! format.  A caller supplies the complete sounding bytes, retains one opaque
//! handle while reading immutable carrier views, and destroys the handle once
//! its host-to-device copies are complete.  No carrier bytes touch disk.

use crate::base_state_wind::HurricaneBaseWind;
use crate::idealized_initialization::{
    initialize_flat_idealized_padded_state, initialize_stretched_idealized_padded_state,
    IdealizedBaseStateLane,
    IdealizedInitializationSelection, IdealizedPaddedCarrier, IdealizedPaddedState,
    TranslatingFrame, WarmBubbleControls,
};
use crate::storm_initialization::{
    initialize_flat_analytic_padded_state, initialize_stretched_analytic_padded_state,
};
use crate::terrain::DeclaredVerticalCoordinate;
use crate::terrain::TerrainStaticCarrier;
use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

pub const IDEALIZED_FFI_GROUP_TERRAIN_OPEN: u32 = 0;
pub const IDEALIZED_FFI_GROUP_BASE_WIND: u32 = 1;
pub const IDEALIZED_FFI_GROUP_MOIST: u32 = 2;
pub const IDEALIZED_FFI_GROUP_TERRAIN_STATIC: u32 = 3;
/// `base.F`'s `qc0` and `qi0` (padded mass layout).
pub const IDEALIZED_FFI_GROUP_BASE_CONDENSATE: u32 = 4;

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

#[repr(C)]
pub struct IdealizedFfiHandle {
    state: IdealizedPaddedState,
}

enum CarrierRef<'a> {
    Padded(&'a IdealizedPaddedCarrier),
    Static(&'a TerrainStaticCarrier),
}

impl CarrierRef<'_> {
    fn name(&self) -> &'static str {
        match self {
            Self::Padded(value) => value.name(),
            Self::Static(value) => value.name(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Padded(value) => value.shape(),
            Self::Static(value) => value.shape(),
        }
    }

    fn values(&self) -> &[f32] {
        match self {
            Self::Padded(value) => value.values(),
            Self::Static(value) => value.values(),
        }
    }
}

fn set_error(message: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into().into_bytes());
}

fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

fn carrier_count(state: &IdealizedPaddedState, group: u32) -> Option<usize> {
    match group {
        IDEALIZED_FFI_GROUP_TERRAIN_OPEN => Some(state.terrain_open_carriers().len()),
        IDEALIZED_FFI_GROUP_BASE_WIND => Some(state.base_wind_carriers().len()),
        IDEALIZED_FFI_GROUP_MOIST => Some(state.moist_carriers().len()),
        IDEALIZED_FFI_GROUP_TERRAIN_STATIC => Some(state.terrain_static().carriers().len()),
        IDEALIZED_FFI_GROUP_BASE_CONDENSATE => Some(state.base_condensate_carriers().len()),
        _ => None,
    }
}

fn carrier_at(state: &IdealizedPaddedState, group: u32, index: usize) -> Option<CarrierRef<'_>> {
    match group {
        IDEALIZED_FFI_GROUP_TERRAIN_OPEN => state
            .terrain_open_carriers()
            .get(index)
            .map(CarrierRef::Padded),
        IDEALIZED_FFI_GROUP_BASE_WIND => state
            .base_wind_carriers()
            .get(index)
            .map(CarrierRef::Padded),
        IDEALIZED_FFI_GROUP_MOIST => state.moist_carriers().get(index).map(CarrierRef::Padded),
        IDEALIZED_FFI_GROUP_TERRAIN_STATIC => state
            .terrain_static()
            .carriers()
            .get(index)
            .map(CarrierRef::Static),
        IDEALIZED_FFI_GROUP_BASE_CONDENSATE => state
            .base_condensate_carriers()
            .get(index)
            .map(CarrierRef::Padded),
        _ => None,
    }
}

/// Build one fresh, flat idealized carrier map and return its opaque owner.
///
/// The only admitted selector is `isnd=7, iwnd=0, irst=0, imoist=1`; `iinit`
/// may be zero or one.  Warm-bubble controls use the official CM1 values and
/// the caller-supplied domain center.
///
/// # Safety
///
/// `sounding` must point to `sounding_len` readable bytes for the duration of
/// this call.  The returned pointer must be released exactly once with
/// [`gm1_idealized_state_free`].
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_state_new(
    sounding: *const u8,
    sounding_len: usize,
    iinit: i32,
    nz: usize,
    ny: usize,
    nx: usize,
    dx_m: f64,
    dy_m: f64,
    dz_m: f64,
    origin_x_m: f32,
    origin_y_m: f32,
    bubble_center_x_m: f32,
    bubble_center_y_m: f32,
    umove_m_per_s: f32,
    vmove_m_per_s: f32,
) -> *mut IdealizedFfiHandle {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if sounding.is_null() || sounding_len == 0 {
            return Err("idealized FFI sounding pointer is null or empty".to_owned());
        }
        let source = unsafe { slice::from_raw_parts(sounding, sounding_len) };
        let selection = IdealizedInitializationSelection::new(7, 0, iinit, 0, 1)
            .map_err(|error| error.to_string())?;
        let controls = WarmBubbleControls::official(bubble_center_x_m, bubble_center_y_m)
            .map_err(|error| error.to_string())?;
        let moving_frame = TranslatingFrame::new(umove_m_per_s, vmove_m_per_s)
            .map_err(|error| error.to_string())?;
        let state = initialize_flat_idealized_padded_state(
            source,
            selection,
            controls,
            [nz, ny, nx],
            [dx_m, dy_m, dz_m],
            [origin_x_m, origin_y_m],
            moving_frame,
        )
        .map_err(|error| error.to_string())?;
        Ok::<_, String>(Box::into_raw(Box::new(IdealizedFfiHandle { state })))
    }));
    match result {
        Ok(Ok(handle)) => handle,
        Ok(Err(error)) => {
            set_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("idealized FFI initializer panicked");
            ptr::null_mut()
        }
    }
}

/// Build one fresh, flat analytic carrier map and return its opaque owner.
///
/// This is the second admitted selector: `isnd=5` (the analytic Weisman-Klemp
/// sounding of `base.F:363-454`) under `iwnd=2` (the analytic wind profile of
/// `base.F:2321-2360`), with `irst=0` and `imoist=1`.  `iinit` may be zero or
/// one.  No sounding bytes are read on this path; `isnd` and `iwnd` are passed
/// through so an unadmitted pair is refused here by value rather than
/// silently substituted.  The returned handle is the same opaque owner
/// [`gm1_idealized_state_new`] returns, so every carrier accessor below and
/// [`gm1_idealized_state_free`] apply unchanged.
#[no_mangle]
pub extern "C" fn gm1_analytic_state_new(
    isnd: i32,
    iwnd: i32,
    iinit: i32,
    nz: usize,
    ny: usize,
    nx: usize,
    dx_m: f64,
    dy_m: f64,
    dz_m: f64,
    origin_x_m: f32,
    origin_y_m: f32,
    bubble_center_x_m: f32,
    bubble_center_y_m: f32,
    umove_m_per_s: f32,
    vmove_m_per_s: f32,
) -> *mut IdealizedFfiHandle {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let selection = IdealizedInitializationSelection::new(isnd, iwnd, iinit, 0, 1)
            .map_err(|error| error.to_string())?;
        let controls = WarmBubbleControls::official(bubble_center_x_m, bubble_center_y_m)
            .map_err(|error| error.to_string())?;
        let moving_frame = TranslatingFrame::new(umove_m_per_s, vmove_m_per_s)
            .map_err(|error| error.to_string())?;
        let state = initialize_flat_analytic_padded_state(
            selection,
            controls,
            [nz, ny, nx],
            [dx_m, dy_m, dz_m],
            [origin_x_m, origin_y_m],
            moving_frame,
        )
        .map_err(|error| error.to_string())?;
        Ok::<_, String>(Box::into_raw(Box::new(IdealizedFfiHandle { state })))
    }));
    match result {
        Ok(Ok(handle)) => handle,
        Ok(Err(error)) => {
            set_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("analytic FFI initializer panicked");
            ptr::null_mut()
        }
    }
}

/// Read `mass_heights`/`face_heights` into one declared vertical coordinate.
///
/// # Safety
/// Both pointers must name readable f32 storage for their declared lengths.
unsafe fn declared_coordinate(
    mass_heights: *const f32,
    mass_heights_len: usize,
    face_heights: *const f32,
    face_heights_len: usize,
    dz_m: f64,
    nz: usize,
) -> std::result::Result<DeclaredVerticalCoordinate, String> {
    if mass_heights.is_null() || face_heights.is_null() {
        return Err("declared vertical coordinate pointer is null".to_owned());
    }
    if (mass_heights_len, face_heights_len) != (nz, nz + 1) {
        return Err(
            "declared vertical coordinate lengths differ from zh(1..nk) and zf(1..nk+1)".to_owned(),
        );
    }
    if mass_heights as usize % align_of::<f32>() != 0
        || face_heights as usize % align_of::<f32>() != 0
    {
        return Err("declared vertical coordinate pointer is misaligned".to_owned());
    }
    let dz = dz_m as f32;
    if !dz_m.is_finite() || f64::from(dz) != dz_m {
        return Err(
            "declared vertical dz reference must be exactly representable in default REAL"
                .to_owned(),
        );
    }
    // Safety: both pointers were just checked for null, alignment and length.
    let (zh, zf) = unsafe {
        (
            slice::from_raw_parts(mass_heights, mass_heights_len),
            slice::from_raw_parts(face_heights, face_heights_len),
        )
    };
    DeclaredVerticalCoordinate::new(zh, zf, dz).map_err(|error| error.to_string())
}

/// Build one fresh idealized carrier map over a declared vertical coordinate.
///
/// This is [`gm1_idealized_state_new`] with `param.F:7789-7811`'s uniform
/// levels replaced by `mass_heights` (`zh(1..nk)`) and `face_heights`
/// (`zf(1..nk+1)`), both in metres above ground.  `dz_m` stays CM1's namelist
/// `dz`, the metric reference of `param.F:8170`.
///
/// # Safety
///
/// `sounding` must point to `sounding_len` readable bytes, and the two
/// coordinate pointers to `nz` and `nz+1` readable f32 words, for the duration
/// of this call.  The returned pointer must be released exactly once with
/// [`gm1_idealized_state_free`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gm1_idealized_state_new_levels(
    sounding: *const u8,
    sounding_len: usize,
    iinit: i32,
    nz: usize,
    ny: usize,
    nx: usize,
    dx_m: f64,
    dy_m: f64,
    dz_m: f64,
    mass_heights: *const f32,
    mass_heights_len: usize,
    face_heights: *const f32,
    face_heights_len: usize,
    origin_x_m: f32,
    origin_y_m: f32,
    bubble_center_x_m: f32,
    bubble_center_y_m: f32,
    umove_m_per_s: f32,
    vmove_m_per_s: f32,
) -> *mut IdealizedFfiHandle {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        if sounding.is_null() || sounding_len == 0 {
            return Err("idealized FFI sounding pointer is null or empty".to_owned());
        }
        let vertical = unsafe {
            declared_coordinate(
                mass_heights,
                mass_heights_len,
                face_heights,
                face_heights_len,
                dz_m,
                nz,
            )
        }?;
        let source = unsafe { slice::from_raw_parts(sounding, sounding_len) };
        let selection = IdealizedInitializationSelection::new(7, 0, iinit, 0, 1)
            .map_err(|error| error.to_string())?;
        let controls = WarmBubbleControls::official(bubble_center_x_m, bubble_center_y_m)
            .map_err(|error| error.to_string())?;
        let moving_frame = TranslatingFrame::new(umove_m_per_s, vmove_m_per_s)
            .map_err(|error| error.to_string())?;
        let state = initialize_stretched_idealized_padded_state(
            source,
            selection,
            controls,
            [nz, ny, nx],
            [dx_m, dy_m, dz_m],
            [origin_x_m, origin_y_m],
            moving_frame,
            &vertical,
        )
        .map_err(|error| error.to_string())?;
        Ok::<_, String>(Box::into_raw(Box::new(IdealizedFfiHandle { state })))
    }));
    match result {
        Ok(Ok(handle)) => handle,
        Ok(Err(error)) => {
            set_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("idealized FFI initializer panicked");
            ptr::null_mut()
        }
    }
}

/// Build one fresh analytic carrier map over a declared vertical coordinate.
///
/// This is [`gm1_analytic_state_new`] with `param.F:7789-7811`'s uniform
/// levels replaced by the caller's; the Weisman-Klemp sounding and shear are
/// then evaluated at those levels.
///
/// # Safety
///
/// The two coordinate pointers must name `nz` and `nz+1` readable f32 words
/// for the duration of this call.  The returned pointer must be released
/// exactly once with [`gm1_idealized_state_free`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gm1_analytic_state_new_levels(
    isnd: i32,
    iwnd: i32,
    iinit: i32,
    nz: usize,
    ny: usize,
    nx: usize,
    dx_m: f64,
    dy_m: f64,
    dz_m: f64,
    mass_heights: *const f32,
    mass_heights_len: usize,
    face_heights: *const f32,
    face_heights_len: usize,
    origin_x_m: f32,
    origin_y_m: f32,
    bubble_center_x_m: f32,
    bubble_center_y_m: f32,
    umove_m_per_s: f32,
    vmove_m_per_s: f32,
) -> *mut IdealizedFfiHandle {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let vertical = unsafe {
            declared_coordinate(
                mass_heights,
                mass_heights_len,
                face_heights,
                face_heights_len,
                dz_m,
                nz,
            )
        }?;
        let selection = IdealizedInitializationSelection::new(isnd, iwnd, iinit, 0, 1)
            .map_err(|error| error.to_string())?;
        let controls = WarmBubbleControls::official(bubble_center_x_m, bubble_center_y_m)
            .map_err(|error| error.to_string())?;
        let moving_frame = TranslatingFrame::new(umove_m_per_s, vmove_m_per_s)
            .map_err(|error| error.to_string())?;
        let state = initialize_stretched_analytic_padded_state(
            selection,
            controls,
            [nz, ny, nx],
            [dx_m, dy_m, dz_m],
            [origin_x_m, origin_y_m],
            moving_frame,
            &vertical,
        )
        .map_err(|error| error.to_string())?;
        Ok::<_, String>(Box::into_raw(Box::new(IdealizedFfiHandle { state })))
    }));
    match result {
        Ok(Ok(handle)) => handle,
        Ok(Err(error)) => {
            set_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("analytic FFI initializer panicked");
            ptr::null_mut()
        }
    }
}

/// The complete `base.F` / `init3d.F` selection for one idealized state.
///
/// This is the single, extensible ABI: every selector CM1 reads from its
/// namelist for the initial condition has a field here, so a new one is added
/// by growing this struct rather than by adding another entry point with a
/// longer positional signature.  Integer fields carry CM1's own values;
/// `maintain_rh` is CM1's logical as 0/1.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Gm1IdealizedOptions {
    /// base.F:134-1960 sounding branch: 5, 7 or 17.
    pub isnd: i32,
    /// base.F:2263-2650 base-state wind branch.
    pub iwnd: i32,
    /// init3d.F:450-1416 3-D initialization branch.
    pub iinit: i32,
    /// param.F:891-903: a restart forces iinit=0.
    pub irst: i32,
    /// README.namelist:308.
    pub imoist: i32,
    /// run/namelist.input:29.
    pub testcase: i32,
    /// init3d.F:1743-1830.
    pub ibalance: i32,
    /// init3d.F:1446-1600.
    pub irandp: i32,
    /// The explicit seed for the irandp field; zero is refused when irandp=1.
    pub random_seed: u64,
    /// run/namelist.input:403,406 for the iwnd=8 branch.
    pub hurr_vg_m_per_s: f32,
    pub hurr_rotate_deg: f32,
    /// init3d.F:456-461 `ric`, `rjc`, `zc`, `bhrad`, `bvrad`, `bptpert`.
    pub ric_m: f32,
    pub rjc_m: f32,
    pub zc_m: f32,
    pub bhrad_m: f32,
    pub bvrad_m: f32,
    pub bptpert_k: f32,
    /// init3d.F:465 `maintain_rh`, as 0 or 1.
    pub maintain_rh: i32,
    /// init3d.F:525 `nbub` and the y step between its bubbles.
    pub nbub: i32,
    pub bubble_spacing_y_m: f32,
    /// init3d.F:494 `zdep`.
    pub zdep_m: f32,
    /// init3d.F:1275, the iinit=8 random amplitude.
    pub random_line_amplitude_k: f32,
    /// base.F:2661-2674.
    pub umove_m_per_s: f32,
    pub vmove_m_per_s: f32,
    /// 0: GM1's seeded stream; 1: CM1's own (init3d.F:427-440, libgfortran).
    pub random_generator: i32,
    /// init3d.F:697-1256 (iinit=7): CM1's fcor, the periodic images
    /// (wbc=ebc=1, sbc=nbc=1) and param.F's iice, each 0/1 where logical.
    pub fcor_s_inv: f32,
    pub periodic_x: i32,
    pub periodic_y: i32,
    pub iice: i32,
    /// base.F:267-268 (isnd=4, and init3d.F:625-645's iinit=4): the moist
    /// benchmark's wet-equivalent potential temperature and total water.
    pub thec_mb_k: f32,
    pub qt_mb: f32,
    /// The namelist `tsk0`, read by base.F's isnd=11 and isnd=21.
    pub tsk0_k: f32,
    /// param.F:5178-5205's imirror/jmirror (0/1), read by ibalance=2's poiss.
    pub imirror: i32,
    pub jmirror: i32,
    /// CM1's eqtset after param.F:936-950, read by base.F's isnd=10.
    pub eqtset: i32,
}

impl Gm1IdealizedOptions {
    fn selection(&self) -> Result<IdealizedInitializationSelection, String> {
        IdealizedInitializationSelection::new(
            self.isnd,
            self.iwnd,
            self.iinit,
            self.irst,
            self.imoist,
        )
        .and_then(|selection| selection.with_testcase(self.testcase))
        .and_then(|selection| selection.with_ibalance(self.ibalance))
        .and_then(|selection| selection.with_random_generator(self.random_generator))
        .and_then(|selection| {
            selection.with_vortex_environment(crate::idealized_initialization::VortexEnvironment {
                fcor: self.fcor_s_inv,
                periodic_x: self.periodic_x != 0,
                periodic_y: self.periodic_y != 0,
                iice: self.iice != 0,
            })
        })
        .and_then(|selection| {
            selection.with_analytic_words(crate::idealized_initialization::AnalyticBaseWords {
                thec_mb_k: self.thec_mb_k,
                qt_mb: self.qt_mb,
                tsk0_k: self.tsk0_k,
                eqtset: self.eqtset,
            })
        })
        .and_then(|selection| selection.with_random_perturbation(self.irandp, self.random_seed))
        .map(|selection| selection.with_elliptic_mirror(self.imirror != 0, self.jmirror != 0))
        .map(|selection| {
            selection.with_hurricane_wind(HurricaneBaseWind {
                gradient_wind_m_per_s: self.hurr_vg_m_per_s,
                rotate_deg: self.hurr_rotate_deg,
            })
        })
        .map_err(|error| error.to_string())
    }

    fn controls(&self) -> Result<WarmBubbleControls, String> {
        WarmBubbleControls::new(
            self.ric_m,
            self.rjc_m,
            self.zc_m,
            self.bhrad_m,
            self.bvrad_m,
            self.bptpert_k,
            self.maintain_rh != 0,
        )
        .and_then(|controls| controls.with_bubble_line(self.nbub, self.bubble_spacing_y_m))
        .and_then(|controls| controls.with_cold_pool_depth(self.zdep_m))
        .and_then(|controls| controls.with_random_line_amplitude(self.random_line_amplitude_k))
        .map_err(|error| error.to_string())
    }
}

/// Build one idealized carrier map from the complete option set.
///
/// `sounding`/`sounding_len` may be null/zero when `isnd=5`, which reads no
/// file.  `mass_heights`/`face_heights` may be null, which selects
/// `param.F:7789-7811`'s uniform coordinate derived from `dz_m`.
///
/// # Safety
///
/// `options` must point to one readable [`Gm1IdealizedOptions`].  When
/// non-null, `sounding` must name `sounding_len` readable bytes and the two
/// coordinate pointers `nz` and `nz+1` readable f32 words, all for the
/// duration of this call.  The returned pointer must be released exactly once
/// with [`gm1_idealized_state_free`].
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gm1_idealized_state_new_options(
    sounding: *const u8,
    sounding_len: usize,
    options: *const Gm1IdealizedOptions,
    nz: usize,
    ny: usize,
    nx: usize,
    dx_m: f64,
    dy_m: f64,
    dz_m: f64,
    origin_x_m: f32,
    origin_y_m: f32,
    mass_heights: *const f32,
    mass_heights_len: usize,
    face_heights: *const f32,
    face_heights_len: usize,
) -> *mut IdealizedFfiHandle {
    clear_error();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(options) = (unsafe { options.as_ref() }) else {
            return Err("idealized FFI options pointer is null".to_owned());
        };
        let selection = options.selection()?;
        let controls = options.controls()?;
        let moving_frame =
            TranslatingFrame::new(options.umove_m_per_s, options.vmove_m_per_s)
                .map_err(|error| error.to_string())?;
        let vertical = if mass_heights.is_null() && face_heights.is_null() {
            None
        } else {
            Some(unsafe {
                declared_coordinate(
                    mass_heights,
                    mass_heights_len,
                    face_heights,
                    face_heights_len,
                    dz_m,
                    nz,
                )
            }?)
        };
        let shape = [nz, ny, nx];
        let spacing = [dx_m, dy_m, dz_m];
        let origin = [origin_x_m, origin_y_m];
        let state = match selection.lane() {
            IdealizedBaseStateLane::Isnd5WeismanKlempIwnd2
            | IdealizedBaseStateLane::AnalyticBaseF => match vertical.as_ref() {
                None => initialize_flat_analytic_padded_state(
                    selection, controls, shape, spacing, origin, moving_frame,
                ),
                Some(coordinate) => initialize_stretched_analytic_padded_state(
                    selection, controls, shape, spacing, origin, moving_frame, coordinate,
                ),
            },
            _ => {
                if sounding.is_null() || sounding_len == 0 {
                    return Err(format!(
                        "base.F:495-607's isnd={} reads input_sounding; the FFI was handed no \
                         bytes, so the base state would be built from an empty file",
                        options.isnd
                    ));
                }
                let source = unsafe { slice::from_raw_parts(sounding, sounding_len) };
                match vertical.as_ref() {
                    None => initialize_flat_idealized_padded_state(
                        source, selection, controls, shape, spacing, origin, moving_frame,
                    ),
                    Some(coordinate) => initialize_stretched_idealized_padded_state(
                        source, selection, controls, shape, spacing, origin, moving_frame,
                        coordinate,
                    ),
                }
            }
        }
        .map_err(|error| error.to_string())?;
        Ok::<_, String>(Box::into_raw(Box::new(IdealizedFfiHandle { state })))
    }));
    match result {
        Ok(Ok(handle)) => handle,
        Ok(Err(error)) => {
            set_error(error);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("idealized FFI options initializer panicked");
            ptr::null_mut()
        }
    }
}

/// Report one integer selector the handle's state was actually built with, so
/// a caller can prove the library honoured what it asked for.  `-1` names a
/// null handle or an unknown selector.
///
/// Selector codes: 0 isnd, 1 effective iwnd, 2 requested iwnd, 3 requested
/// iinit, 4 effective iinit, 5 irst, 6 imoist, 7 testcase, 8 ibalance,
/// 9 irandp.
///
/// # Safety
///
/// `handle` must point to a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_state_selector(
    handle: *const IdealizedFfiHandle,
    code: u32,
) -> i32 {
    clear_error();
    let Some(owner) = (unsafe { handle.as_ref() }) else {
        set_error("idealized FFI handle is null");
        return -1;
    };
    let selection = owner.state.selection();
    match code {
        0 => match selection.lane() {
            IdealizedBaseStateLane::Isnd7ExternalSounding => 7,
            IdealizedBaseStateLane::Isnd5WeismanKlempIwnd2 => 5,
            IdealizedBaseStateLane::Isnd17ExternalThermodynamics => 17,
            IdealizedBaseStateLane::AnalyticBaseF => selection.requested_isnd(),
        },
        1 => selection.effective_iwnd(),
        2 => selection.requested_iwnd(),
        3 => selection.requested_iinit(),
        4 => selection.effective_iinit(),
        5 => selection.irst(),
        6 => selection.imoist(),
        7 => selection.testcase(),
        8 => selection.ibalance(),
        9 => selection.irandp(),
        _ => {
            set_error("unknown idealized selector code");
            -1
        }
    }
}

/// The `irandp` seed this handle's perturbation field was drawn from.
/// Zero means no seeded field was drawn.
///
/// # Safety
///
/// `handle` must point to a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_state_random_seed(
    handle: *const IdealizedFfiHandle,
) -> u64 {
    clear_error();
    let Some(owner) = (unsafe { handle.as_ref() }) else {
        set_error("idealized FFI handle is null");
        return 0;
    };
    owner.state.selection().random_seed()
}

/// Report which admitted lane built this handle: `7` for the external
/// `input_sounding` lane, `5` for the analytic Weisman-Klemp lane.
///
/// Returns `-1` on a null handle so a caller cannot mistake a failure for a
/// lane it did not ask for.
///
/// # Safety
///
/// `handle` must point to a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_state_isnd(handle: *const IdealizedFfiHandle) -> i32 {
    clear_error();
    let Some(owner) = (unsafe { handle.as_ref() }) else {
        set_error("idealized FFI handle is null");
        return -1;
    };
    match owner.state.selection().lane() {
        IdealizedBaseStateLane::Isnd7ExternalSounding => 7,
        IdealizedBaseStateLane::Isnd5WeismanKlempIwnd2 => 5,
        IdealizedBaseStateLane::Isnd17ExternalThermodynamics => 17,
        IdealizedBaseStateLane::AnalyticBaseF => owner.state.selection().requested_isnd(),
    }
}

/// Report the `iwnd` this handle's base winds were actually built with, after
/// the `param.F:836-852` adjustment.  Returns `-1` on a null handle.
///
/// # Safety
///
/// `handle` must point to a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_state_effective_iwnd(
    handle: *const IdealizedFfiHandle,
) -> i32 {
    clear_error();
    let Some(owner) = (unsafe { handle.as_ref() }) else {
        set_error("idealized FFI handle is null");
        return -1;
    };
    owner.state.selection().effective_iwnd()
}

/// Report the `umove` (m/s) this handle's base winds were actually built
/// with, following `base.F:2663-2669`.
///
/// This is the readback half of the frame: a caller that asked for a
/// translating frame can prove the library it loaded honoured it, instead of
/// discovering later that every published position is wrong by `umove*t`.
/// Returns a quiet NaN on a null handle, which no admitted frame can be.
///
/// # Safety
///
/// `handle` must point to a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_state_umove(handle: *const IdealizedFfiHandle) -> f32 {
    clear_error();
    let Some(owner) = (unsafe { handle.as_ref() }) else {
        set_error("idealized FFI handle is null");
        return f32::NAN;
    };
    owner.state.moving_frame().u_move_m_per_s()
}

/// Report the `vmove` (m/s) this handle's base winds were built with.
/// Returns a quiet NaN on a null handle.  See
/// [`gm1_idealized_state_umove`].
///
/// # Safety
///
/// `handle` must point to a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_state_vmove(handle: *const IdealizedFfiHandle) -> f32 {
    clear_error();
    let Some(owner) = (unsafe { handle.as_ref() }) else {
        set_error("idealized FFI handle is null");
        return f32::NAN;
    };
    owner.state.moving_frame().v_move_m_per_s()
}

/// Destroy one handle returned by [`gm1_idealized_state_new`].
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by this library, and a
/// non-null handle must not be freed more than once.
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_state_free(handle: *mut IdealizedFfiHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Return the number of carriers in one exact existing carrier group.
///
/// # Safety
///
/// `handle` must point to a live handle returned by this library.
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_carrier_count(
    handle: *const IdealizedFfiHandle,
    group: u32,
) -> usize {
    clear_error();
    let Some(owner) = (unsafe { handle.as_ref() }) else {
        set_error("idealized FFI handle is null");
        return usize::MAX;
    };
    match carrier_count(&owner.state, group) {
        Some(count) => count,
        None => {
            set_error(format!("unsupported idealized FFI carrier group {group}"));
            usize::MAX
        }
    }
}

/// Borrow one immutable carrier name, shape, and binary32 value buffer.
///
/// On success this returns a non-null name pointer and writes the other
/// borrowed pointers/counts.  Every view remains valid only while `handle`
/// remains live.
///
/// # Safety
///
/// `handle` and all four output pointers must be valid and writable.  The
/// caller must not mutate the returned name, shape, or value buffers.
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_carrier_view(
    handle: *const IdealizedFfiHandle,
    group: u32,
    index: usize,
    name_len: *mut usize,
    shape: *mut *const usize,
    rank: *mut usize,
    values: *mut *const f32,
    value_len: *mut usize,
) -> *const u8 {
    clear_error();
    let Some(owner) = (unsafe { handle.as_ref() }) else {
        set_error("idealized FFI handle is null");
        return ptr::null();
    };
    if name_len.is_null()
        || shape.is_null()
        || rank.is_null()
        || values.is_null()
        || value_len.is_null()
    {
        set_error("idealized FFI carrier output pointer is null");
        return ptr::null();
    }
    let Some(carrier) = carrier_at(&owner.state, group, index) else {
        set_error(format!(
            "idealized FFI carrier index {index} is outside group {group}"
        ));
        return ptr::null();
    };
    let name = carrier.name().as_bytes();
    let carrier_shape = carrier.shape();
    let carrier_values = carrier.values();
    unsafe {
        *name_len = name.len();
        *shape = carrier_shape.as_ptr();
        *rank = carrier_shape.len();
        *values = carrier_values.as_ptr();
        *value_len = carrier_values.len();
    }
    name.as_ptr()
}

/// Return the byte length of the last error on this calling thread.
#[no_mangle]
pub extern "C" fn gm1_idealized_last_error_len() -> usize {
    LAST_ERROR.with(|slot| slot.borrow().len())
}

/// Copy the last error to `destination`, returning the complete error length.
///
/// # Safety
///
/// When `capacity > 0`, `destination` must point to at least `capacity`
/// writable bytes.
#[no_mangle]
pub unsafe extern "C" fn gm1_idealized_last_error_copy(
    destination: *mut u8,
    capacity: usize,
) -> usize {
    LAST_ERROR.with(|slot| {
        let error = slot.borrow();
        let count = error.len().min(capacity);
        if count != 0 && !destination.is_null() {
            unsafe { ptr::copy_nonoverlapping(error.as_ptr(), destination, count) };
        }
        error.len()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOUNDING: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../upstream/cm1-21.1/pristine/run/config_files/sea_breeze/input_sounding"
    ));

    #[test]
    fn the_frame_velocity_crosses_the_abi_and_reads_back() {
        // Both lanes take the frame across the C ABI, and both report it
        // back, so a caller can prove the library it loaded honoured the
        // request instead of discovering later that every published
        // position is wrong by umove*t.
        let analytic = gm1_analytic_state_new(
            5, 2, 1, 40, 2, 3, 1000.0, 1000.0, 500.0, 0.0, 0.0, 1500.0, 1000.0, 12.5, 3.0,
        );
        assert!(!analytic.is_null());
        assert_eq!(unsafe { gm1_idealized_state_umove(analytic) }, 12.5_f32);
        assert_eq!(unsafe { gm1_idealized_state_vmove(analytic) }, 3.0_f32);
        unsafe { gm1_idealized_state_free(analytic) };

        let sounding = unsafe {
            gm1_idealized_state_new(
                SOUNDING.as_ptr(),
                SOUNDING.len(),
                1,
                4,
                2,
                3,
                1000.0,
                1000.0,
                500.0,
                0.0,
                0.0,
                1500.0,
                1000.0,
                -7.5,
                2.25,
            )
        };
        assert!(!sounding.is_null());
        assert_eq!(unsafe { gm1_idealized_state_umove(sounding) }, -7.5_f32);
        assert_eq!(unsafe { gm1_idealized_state_vmove(sounding) }, 2.25_f32);
        unsafe { gm1_idealized_state_free(sounding) };

        // A refused frame is a null handle plus the named error, never a
        // silently fixed frame.
        let refused = gm1_analytic_state_new(
            5,
            2,
            1,
            40,
            2,
            3,
            1000.0,
            1000.0,
            500.0,
            0.0,
            0.0,
            1500.0,
            1000.0,
            f32::NAN,
            0.0,
        );
        assert!(refused.is_null());
        let len = gm1_idealized_last_error_len();
        assert!(len > 0);
        let mut buffer = vec![0_u8; len];
        unsafe { gm1_idealized_last_error_copy(buffer.as_mut_ptr(), len) };
        let message = String::from_utf8(buffer).unwrap();
        assert!(message.contains("umove"), "{message}");
        assert!(message.contains("not finite"), "{message}");

        // A null handle reports NaN, which no admitted frame can be.
        assert!(unsafe { gm1_idealized_state_umove(ptr::null()) }.is_nan());
        assert!(unsafe { gm1_idealized_state_vmove(ptr::null()) }.is_nan());
    }

    #[test]
    fn ffi_views_existing_groups_without_persistence() {
        let handle = unsafe {
            gm1_idealized_state_new(
                SOUNDING.as_ptr(),
                SOUNDING.len(),
                1,
                4,
                2,
                3,
                1000.0,
                1000.0,
                500.0,
                0.0,
                0.0,
                1500.0,
                1000.0,
                0.0,
                0.0,
            )
        };
        assert!(!handle.is_null());
        assert_eq!(unsafe { gm1_idealized_carrier_count(handle, 0) }, 54);
        assert_eq!(unsafe { gm1_idealized_carrier_count(handle, 1) }, 2);
        assert_eq!(unsafe { gm1_idealized_carrier_count(handle, 2) }, 4);
        assert_eq!(unsafe { gm1_idealized_carrier_count(handle, 3) }, 25);

        let mut name_len = 0;
        let mut shape = ptr::null();
        let mut rank = 0;
        let mut values = ptr::null();
        let mut value_len = 0;
        let name = unsafe {
            gm1_idealized_carrier_view(
                handle,
                0,
                0,
                &mut name_len,
                &mut shape,
                &mut rank,
                &mut values,
                &mut value_len,
            )
        };
        assert!(!name.is_null());
        assert_eq!(unsafe { slice::from_raw_parts(name, name_len) }, b"pi0");
        assert_eq!(unsafe { slice::from_raw_parts(shape, rank) }, &[6, 8, 9]);
        assert_eq!(value_len, 6 * 8 * 9);
        assert!(!values.is_null());
        unsafe { gm1_idealized_state_free(handle) };
    }

    #[test]
    fn ffi_exposes_the_analytic_selector_beside_the_sounding_selector() {
        let analytic = gm1_analytic_state_new(
            5, 2, 1, 40, 2, 3, 1000.0, 1000.0, 500.0, 0.0, 0.0, 1500.0, 1000.0, 0.0, 0.0,
        );
        assert!(!analytic.is_null());
        assert_eq!(unsafe { gm1_idealized_state_isnd(analytic) }, 5);
        assert_eq!(unsafe { gm1_idealized_state_effective_iwnd(analytic) }, 2);
        // The analytic lane fills the same four carrier groups.
        assert_eq!(unsafe { gm1_idealized_carrier_count(analytic, 0) }, 54);
        assert_eq!(unsafe { gm1_idealized_carrier_count(analytic, 1) }, 2);
        assert_eq!(unsafe { gm1_idealized_carrier_count(analytic, 2) }, 4);
        assert_eq!(unsafe { gm1_idealized_carrier_count(analytic, 3) }, 25);

        let mut name_len = 0;
        let mut shape = ptr::null();
        let mut rank = 0;
        let mut values = ptr::null();
        let mut value_len = 0;
        let name = unsafe {
            gm1_idealized_carrier_view(
                analytic,
                IDEALIZED_FFI_GROUP_BASE_WIND,
                0,
                &mut name_len,
                &mut shape,
                &mut rank,
                &mut values,
                &mut value_len,
            )
        };
        assert_eq!(unsafe { slice::from_raw_parts(name, name_len) }, b"u0");
        assert_eq!(unsafe { slice::from_raw_parts(shape, rank) }, &[40, 2, 4]);
        let u0 = unsafe { slice::from_raw_parts(values, value_len) };
        // k=0 is 250 m AGL; k=12 is 6250 m, above udep2, so u0 is umax2.
        assert!(u0[0] > 0.0 && u0[0] < 7.0);
        assert_eq!(u0[12 * 2 * 4].to_bits(), 31.0_f32.to_bits());
        unsafe { gm1_idealized_state_free(analytic) };

        // The sounding lane still reports itself, with iwnd forced to zero.
        let sounding = unsafe {
            gm1_idealized_state_new(
                SOUNDING.as_ptr(),
                SOUNDING.len(),
                1,
                4,
                2,
                3,
                1000.0,
                1000.0,
                500.0,
                0.0,
                0.0,
                1500.0,
                1000.0,
                0.0,
                0.0,
            )
        };
        assert!(!sounding.is_null());
        assert_eq!(unsafe { gm1_idealized_state_isnd(sounding) }, 7);
        assert_eq!(unsafe { gm1_idealized_state_effective_iwnd(sounding) }, 0);
        unsafe { gm1_idealized_state_free(sounding) };

        assert_eq!(unsafe { gm1_idealized_state_isnd(ptr::null()) }, -1);
    }

    fn deck_options(isnd: i32, iwnd: i32, iinit: i32) -> Gm1IdealizedOptions {
        Gm1IdealizedOptions {
            isnd,
            iwnd,
            iinit,
            irst: 0,
            imoist: 1,
            testcase: 0,
            ibalance: 0,
            irandp: 0,
            random_seed: 0,
            hurr_vg_m_per_s: 40.0,
            hurr_rotate_deg: 0.0,
            ric_m: 1500.0,
            rjc_m: 1000.0,
            zc_m: 1400.0,
            bhrad_m: 10_000.0,
            bvrad_m: 1400.0,
            bptpert_k: 1.0,
            maintain_rh: 0,
            nbub: 3,
            bubble_spacing_y_m: 30_000.0,
            zdep_m: 2500.0,
            random_line_amplitude_k: 0.2,
            umove_m_per_s: 0.0,
            vmove_m_per_s: 0.0,
            random_generator: 0,
            fcor_s_inv: 5.0e-5,
            periodic_x: 0,
            periodic_y: 0,
            iice: 1,
            thec_mb_k: 320.0,
            qt_mb: 0.020,
            tsk0_k: 299.28,
            imirror: 1,
            jmirror: 1,
            eqtset: 2,
        }
    }

    fn build_options(options: &Gm1IdealizedOptions) -> *mut IdealizedFfiHandle {
        build_options_shaped(options, 8, 8)
    }

    fn build_options_shaped(
        options: &Gm1IdealizedOptions,
        ny: usize,
        nx: usize,
    ) -> *mut IdealizedFfiHandle {
        unsafe {
            gm1_idealized_state_new_options(
                SOUNDING.as_ptr(),
                SOUNDING.len(),
                options,
                40,
                ny,
                nx,
                1000.0,
                1000.0,
                500.0,
                0.0,
                0.0,
                ptr::null(),
                0,
                ptr::null(),
                0,
            )
        }
    }

    #[test]
    fn options_entry_reaches_every_new_selector_and_reports_it_back() {
        // base.F:2286-2640: each iwnd branch must change the emitted u0.
        let mut previous: Option<Vec<f32>> = None;
        for iwnd in [0_i32, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11] {
            let options = deck_options(5, iwnd, 1);
            let handle = build_options(&options);
            assert!(!handle.is_null(), "iwnd={iwnd} refused");
            let owner = unsafe { handle.as_ref() }.unwrap();
            let u0 = owner
                .state
                .base_wind_carriers()
                .iter()
                .find(|carrier| carrier.name() == "u0")
                .unwrap()
                .values()
                .to_vec();
            assert_eq!(
                unsafe { gm1_idealized_state_selector(handle, 1) },
                iwnd,
                "the library must report the iwnd it built"
            );
            if let Some(before) = previous.as_ref() {
                assert_ne!(before, &u0, "iwnd={iwnd} emitted the previous hodograph");
            }
            previous = Some(u0);
            unsafe { gm1_idealized_state_free(handle) };
        }

        // init3d.F:450-1416: each implemented iinit must change tha.
        let mut seen: Vec<Vec<f32>> = Vec::new();
        for iinit in [0_i32, 1, 2, 3, 5, 8, 11, 13] {
            let mut options = deck_options(5, 2, iinit);
            // iinit=8 draws random numbers; give the stream a seed.
            options.random_seed = 4242;
            // A bubble small enough that an 8x8 km domain resolves the line
            // of three as three, not as one overlapping blob.
            options.ric_m = 3500.0;
            options.rjc_m = 1500.0;
            options.bhrad_m = 1200.0;
            options.bvrad_m = 900.0;
            options.bubble_spacing_y_m = 3000.0;
            let handle = build_options(&options);
            assert!(!handle.is_null(), "iinit={iinit} refused");
            let owner = unsafe { handle.as_ref() }.unwrap();
            let tha = owner
                .state
                .terrain_open_carriers()
                .iter()
                .find(|carrier| carrier.name() == "tha")
                .unwrap()
                .values()
                .to_vec();
            assert_eq!(unsafe { gm1_idealized_state_selector(handle, 4) }, iinit);
            for before in &seen {
                assert_ne!(before, &tha, "iinit={iinit} emitted an earlier field");
            }
            seen.push(tha);
            unsafe { gm1_idealized_state_free(handle) };
        }
        // init3d.F:1303-1396: the forcing arms build no tha at all, exactly
        // as iinit=0 does; their perturbation is applied during the run.
        for iinit in [9_i32, 10, 12] {
            let handle = build_options(&deck_options(5, 2, iinit));
            assert!(!handle.is_null(), "iinit={iinit} refused");
            let owner = unsafe { handle.as_ref() }.unwrap();
            let tha = owner
                .state
                .terrain_open_carriers()
                .iter()
                .find(|carrier| carrier.name() == "tha")
                .unwrap()
                .values()
                .to_vec();
            assert_eq!(unsafe { gm1_idealized_state_selector(handle, 4) }, iinit);
            assert_eq!(tha, seen[0], "iinit={iinit} must leave tha as iinit=0 does");
            unsafe { gm1_idealized_state_free(handle) };
        }

        // init3d.F:1743-1830: ibalance must change ppi and nothing else.
        let read_ppi = |ibalance: i32| {
            let mut options = deck_options(5, 2, 1);
            options.ibalance = ibalance;
            let handle = build_options(&options);
            assert!(!handle.is_null());
            let owner = unsafe { handle.as_ref() }.unwrap();
            let values = owner
                .state
                .terrain_open_carriers()
                .iter()
                .find(|carrier| carrier.name() == "ppi")
                .unwrap()
                .values()
                .to_vec();
            unsafe { gm1_idealized_state_free(handle) };
            values
        };
        let unbalanced = read_ppi(0);
        let hydrostatic = read_ppi(1);
        assert!(
            unbalanced.iter().all(|value| value.to_bits() == 0),
            "ibalance=0 must leave the initial pressure perturbation at exactly zero"
        );
        assert!(
            hydrostatic.iter().any(|value| *value != 0.0),
            "ibalance=1 must integrate a nonzero hydrostatic pressure perturbation"
        );

        // init3d.F:1446-1600: an unseeded irandp=1 is refused by value.
        let mut options = deck_options(5, 2, 0);
        options.irandp = 1;
        assert!(build_options(&options).is_null());
        options.random_seed = 987_654_321;
        let handle = build_options(&options);
        assert!(!handle.is_null());
        assert_eq!(
            unsafe { gm1_idealized_state_random_seed(handle) },
            987_654_321
        );
        assert_eq!(unsafe { gm1_idealized_state_selector(handle, 9) }, 1);
        unsafe { gm1_idealized_state_free(handle) };

        // isnd=17 pairs the external thermodynamics with an analytic hodograph.
        let options = deck_options(17, 6, 1);
        let handle = build_options(&options);
        assert!(!handle.is_null());
        assert_eq!(unsafe { gm1_idealized_state_selector(handle, 0) }, 17);
        assert_eq!(unsafe { gm1_idealized_state_selector(handle, 1) }, 6);
        unsafe { gm1_idealized_state_free(handle) };

        // README.namelist:308: a dry run carries no vapour at all.
        let mut options = deck_options(7, 0, 1);
        options.imoist = 0;
        let handle = build_options(&options);
        assert!(!handle.is_null(), "{}", LAST_ERROR.with(|slot| String::from_utf8_lossy(&slot.borrow()).into_owned()));
        let owner = unsafe { handle.as_ref() }.unwrap();
        let qa = owner
            .state
            .moist_carriers()
            .iter()
            .find(|carrier| carrier.name() == "qa")
            .unwrap()
            .values()
            .to_vec();
        assert!(
            qa.iter().all(|value| *value == 0.0),
            "imoist=0 must emit an exactly dry moisture carrier"
        );
        unsafe { gm1_idealized_state_free(handle) };
    }

    #[test]
    fn ffi_refuses_unadmitted_analytic_pairs_by_value() {
        for (isnd, iwnd) in [(5_i32, 7_i32), (5, 12), (6, 2), (7, 0)] {
            let handle = gm1_analytic_state_new(
                isnd, iwnd, 1, 40, 2, 3, 1000.0, 1000.0, 500.0, 0.0, 0.0, 1500.0, 1000.0, 0.0, 0.0,
            );
            if isnd == 7 {
                // isnd=7 admits as a selection but is not this entry's lane.
                assert!(handle.is_null(), "isnd=7 must not build an analytic state");
            } else {
                assert!(handle.is_null(), "({isnd},{iwnd}) must be refused");
            }
            assert!(gm1_idealized_last_error_len() > 0);
        }
    }

    #[test]
    fn ffi_refuses_unsupported_selector_and_group() {
        let handle = unsafe {
            gm1_idealized_state_new(
                SOUNDING.as_ptr(),
                SOUNDING.len(),
                6,
                4,
                2,
                3,
                1000.0,
                1000.0,
                500.0,
                0.0,
                0.0,
                1500.0,
                1000.0,
                0.0,
                0.0,
            )
        };
        assert!(handle.is_null());
        assert!(gm1_idealized_last_error_len() > 0);

        let good = unsafe {
            gm1_idealized_state_new(
                SOUNDING.as_ptr(),
                SOUNDING.len(),
                0,
                4,
                2,
                3,
                1000.0,
                1000.0,
                500.0,
                0.0,
                0.0,
                1500.0,
                1000.0,
                0.0,
                0.0,
            )
        };
        assert!(!good.is_null());
        assert_eq!(unsafe { gm1_idealized_carrier_count(good, 99) }, usize::MAX);
        assert!(gm1_idealized_last_error_len() > 0);
        unsafe { gm1_idealized_state_free(good) };
    }
}
