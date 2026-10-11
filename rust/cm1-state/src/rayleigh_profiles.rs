//! Source-ordered flat/uniform CM1 r21.1 upper-Rayleigh profiles.
//!
//! Coordinates follow param.F:7789-7811, the active-depth guard is at
//! 8633-8650, and tauh/taus/tauf are assigned at 8652-8671.  The coefficients
//! are dimensionless: solve1.F applies rdalpha separately, once.  This module
//! uses Rust's target f32 cosine; cross-libm identity is measured separately.

use crate::{Result, StateError};
use std::cell::RefCell;
use std::mem::{align_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

pub const RAYLEIGH_PARAMETER_COUNT: usize = 5;
// constants.F:14, default REAL: 3.1415926535897932384626433.
const CM1_PI: f32 = f32::from_bits(0x4049_0fdb);

#[derive(Debug)]
pub struct UniformRayleighProfiles {
    pub tauh: Vec<f32>,
    pub taus: Vec<f32>,
    pub tauf: Vec<f32>,
    /// dz, model top, damping start, damping depth, rdalpha; all default REAL.
    pub parameters: [f32; RAYLEIGH_PARAMETER_COUNT],
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

fn nonnegative_real(name: &str, input: f64) -> Result<f32> {
    let value = input as f32;
    if !input.is_finite() || input < 0.0 || !value.is_finite() {
        return Err(invalid(format!(
            "Rayleigh {name} must be finite and nonnegative in CM1 default REAL"
        )));
    }
    Ok(value)
}

fn checked_levels(nz: usize) -> Result<usize> {
    if nz == 0 || nz > (i32::MAX as usize - 2) {
        return Err(invalid(
            "Rayleigh nz must be positive and leave room for CM1 integer ghost indices",
        ));
    }
    let faces = nz
        .checked_add(1)
        .ok_or_else(|| invalid("Rayleigh face count overflows"))?;
    let bytes = faces
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid("Rayleigh profile byte size overflows"))?;
    if bytes > isize::MAX as usize {
        return Err(invalid(
            "Rayleigh profile exceeds the host allocation bound",
        ));
    }
    Ok(faces)
}

fn zeros(words: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(words)
        .map_err(|error| invalid(format!("Rayleigh profile allocation failed: {error}")))?;
    values.resize(words, 0.0_f32);
    Ok(values)
}

fn coefficient(height: f32, top: f32, start: f32) -> f32 {
    if height <= start {
        return 0.0_f32;
    }
    // Preserve every default-REAL operation in param.F:8656,8667.  Dividing
    // (height-start) before multiplying pi changes raw words on these grids.
    let distance = height - start;
    let angle = CM1_PI * distance;
    let angle = angle / (top - start);
    let cosine = angle.cos();
    0.5_f32 * (1.0_f32 - cosine)
}

pub fn uniform_rayleigh_profiles(
    nz: usize,
    dz_m: f64,
    damping_depth_m: f64,
    damping_rate_s_inv: f64,
) -> Result<UniformRayleighProfiles> {
    let faces = checked_levels(nz)?;
    let dz = nonnegative_real("dz_m", dz_m)?;
    if dz <= 0.0 {
        return Err(invalid(
            "Rayleigh dz_m must remain positive in CM1 default REAL",
        ));
    }
    // param.F:7789-7811 and :2608.  The uniform coordinate is rebuilt here
    // exactly as param.F writes it -- both faces rounded before the midpoint,
    // and the model top from zf(nk+1) -- so that this entry point and
    // `stretched_rayleigh_profiles` share one coefficient loop.
    let mut mass_heights = zeros(nz)?;
    let mut face_heights = zeros(faces)?;
    for (k, value) in face_heights.iter_mut().enumerate() {
        *value = dz * k as f32;
    }
    for (k, value) in mass_heights.iter_mut().enumerate() {
        *value = 0.5_f32 * (dz * k as f32 + dz * (k + 1) as f32);
    }
    rayleigh_profiles_on_levels(
        &mass_heights,
        &face_heights,
        dz,
        damping_depth_m,
        damping_rate_s_inv,
    )
}

/// The same `param.F:8633-8671` assignment over a caller-declared coordinate.
///
/// `mass_heights_m` is CM1 `zh(1..nk)` and `face_heights_m` is `zf(1..nk+1)`,
/// both in metres above ground.  `dz_reference_m` is the namelist `dz` that
/// `param.F:8170` divides every `mh` word by; it is carried into the parameter
/// receipt and is not used to reconstruct a level, because on a stretched grid
/// it is not any layer's thickness.  `maxz` is `zf(nk+1)` (`param.F:8193`),
/// the same number the uniform branch reaches through `dz*nk`.
pub fn stretched_rayleigh_profiles(
    mass_heights_m: &[f32],
    face_heights_m: &[f32],
    dz_reference_m: f64,
    damping_depth_m: f64,
    damping_rate_s_inv: f64,
) -> Result<UniformRayleighProfiles> {
    let nz = mass_heights_m.len();
    let faces = checked_levels(nz)?;
    if face_heights_m.len() != faces {
        return Err(invalid(format!(
            "Rayleigh face heights contain {} words; zf(1..nk+1) requires {faces}",
            face_heights_m.len()
        )));
    }
    let dz = nonnegative_real("dz_reference_m", dz_reference_m)?;
    if dz <= 0.0 {
        return Err(invalid(
            "Rayleigh dz_reference_m must remain positive in CM1 default REAL",
        ));
    }
    if face_heights_m[0] != 0.0 {
        return Err(invalid(
            "Rayleigh face heights must begin at the ground, zf(1)=0",
        ));
    }
    for (k, window) in face_heights_m.windows(2).enumerate() {
        if !window[0].is_finite() || !window[1].is_finite() || !(window[1] > window[0]) {
            return Err(invalid(format!(
                "Rayleigh layer {} has non-positive depth; param.F:8212-8224 stops CM1 here",
                k + 1
            )));
        }
    }
    for (k, &height) in mass_heights_m.iter().enumerate() {
        if !height.is_finite() || height <= face_heights_m[k] || height >= face_heights_m[k + 1] {
            return Err(invalid(format!(
                "Rayleigh mass height {} does not lie inside its own layer",
                k + 1
            )));
        }
    }
    rayleigh_profiles_on_levels(
        mass_heights_m,
        face_heights_m,
        dz,
        damping_depth_m,
        damping_rate_s_inv,
    )
}

fn rayleigh_profiles_on_levels(
    mass_heights_m: &[f32],
    face_heights_m: &[f32],
    dz: f32,
    damping_depth_m: f64,
    damping_rate_s_inv: f64,
) -> Result<UniformRayleighProfiles> {
    let nz = mass_heights_m.len();
    let depth = nonnegative_real("damping_depth_m", damping_depth_m)?;
    let rate = nonnegative_real("damping_rate_s_inv", damping_rate_s_inv)?;
    // param.F:8193.  maxz is the top w level, not nz*dz: on a stretched grid
    // those are different numbers and the damper uses this one.
    let top = face_heights_m[nz];
    let start = top - depth;
    if !top.is_finite() || top <= 0.0 || !start.is_finite() {
        return Err(invalid(
            "Rayleigh model top or damping start overflows default REAL",
        ));
    }
    if start < 0.5_f32 * top {
        return Err(invalid(
            "Rayleigh damping depth extends below half the model top; CM1 param.F:8635-8650 stops this configuration",
        ));
    }
    let mut tauh = zeros(nz)?;
    let mut tauf = zeros(nz + 1)?;
    // A zero effective depth makes zd==maxz; param.F:8633 leaves the
    // allocation-zero coefficients unchanged in that branch.
    if start < top {
        for (value, &height) in tauh.iter_mut().zip(mass_heights_m) {
            *value = coefficient(height, top, start);
        }
        for (value, &height) in tauf.iter_mut().zip(face_heights_m) {
            *value = coefficient(height, top, start);
        }
    }
    let mut taus = zeros(nz)?;
    taus.copy_from_slice(&tauh);
    Ok(UniformRayleighProfiles {
        tauh,
        taus,
        tauf,
        parameters: [dz, top, start, depth, rate],
    })
}

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn set_error(message: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.into().into_bytes());
}

fn checked_output(pointer: *mut f32, words: usize) -> Result<(usize, usize)> {
    let start = pointer as usize;
    if pointer.is_null() || !start.is_multiple_of(align_of::<f32>()) {
        return Err(invalid("Rayleigh output pointer is null or misaligned"));
    }
    let bytes = words
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid("Rayleigh output byte size overflows"))?;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| invalid("Rayleigh output address range overflows"))?;
    Ok((start, end))
}

fn checked_input(pointer: *const f32, words: usize) -> Result<()> {
    let start = pointer as usize;
    if pointer.is_null() || !start.is_multiple_of(align_of::<f32>()) {
        return Err(invalid("Rayleigh coordinate pointer is null or misaligned"));
    }
    let bytes = words
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| invalid("Rayleigh coordinate byte size overflows"))?;
    start
        .checked_add(bytes)
        .ok_or_else(|| invalid("Rayleigh coordinate address range overflows"))?;
    Ok(())
}

/// Fill caller-owned profiles and five default-REAL parameter words.
///
/// Returns zero on success and -1 on refusal.  Refusals leave every output
/// untouched.  The parameter order is dz, top, start, depth, rdalpha.
///
/// # Safety
/// Each pointer must name writable f32 storage for its declared length.
/// Buffers must not overlap.  This function retains no pointer after return.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gm1_rayleigh_profiles_fill(
    nz: usize,
    dz_m: f64,
    damping_depth_m: f64,
    damping_rate_s_inv: f64,
    tauh: *mut f32,
    tauh_len: usize,
    taus: *mut f32,
    taus_len: usize,
    tauf: *mut f32,
    tauf_len: usize,
    parameters: *mut f32,
    parameters_len: usize,
) -> i32 {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
    let result = catch_unwind(AssertUnwindSafe(|| {
        let faces = checked_levels(nz)?;
        if (tauh_len, taus_len, tauf_len, parameters_len)
            != (nz, nz, faces, RAYLEIGH_PARAMETER_COUNT)
        {
            return Err(invalid("Rayleigh output lengths differ from nz/nz/nz+1/5"));
        }
        let ranges = [
            checked_output(tauh, tauh_len)?,
            checked_output(taus, taus_len)?,
            checked_output(tauf, tauf_len)?,
            checked_output(parameters, parameters_len)?,
        ];
        for (index, &(start, end)) in ranges.iter().enumerate() {
            if ranges[index + 1..]
                .iter()
                .any(|&(other_start, other_end)| start < other_end && other_start < end)
            {
                return Err(invalid("Rayleigh output buffers must not overlap"));
            }
        }
        uniform_rayleigh_profiles(nz, dz_m, damping_depth_m, damping_rate_s_inv)
    }));
    match result {
        Ok(Ok(profiles)) => {
            unsafe {
                ptr::copy_nonoverlapping(profiles.tauh.as_ptr(), tauh, tauh_len);
                ptr::copy_nonoverlapping(profiles.taus.as_ptr(), taus, taus_len);
                ptr::copy_nonoverlapping(profiles.tauf.as_ptr(), tauf, tauf_len);
                ptr::copy_nonoverlapping(profiles.parameters.as_ptr(), parameters, parameters_len);
            }
            0
        }
        Ok(Err(error)) => {
            set_error(error.to_string());
            -1
        }
        Err(_) => {
            set_error("Rust Rayleigh profile producer panicked");
            -1
        }
    }
}

/// Fill caller-owned profiles from a caller-declared vertical coordinate.
///
/// `mass_heights` is CM1 `zh(1..nk)` and `face_heights` is `zf(1..nk+1)`, both
/// in metres above ground; `dz_reference_m` is the namelist `dz` of
/// `param.F:8170`.  Returns zero on success and -1 on refusal.  Refusals leave
/// every output untouched.  The parameter order is dz, top, start, depth,
/// rdalpha, and `top` is `zf(nk+1)` rather than `nz*dz`.
///
/// # Safety
/// Each pointer must name storage for its declared length -- readable for the
/// two coordinate arrays, writable for the four outputs.  The output buffers
/// must not overlap each other.  This function retains no pointer after return.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gm1_rayleigh_profiles_fill_levels(
    nz: usize,
    mass_heights: *const f32,
    mass_heights_len: usize,
    face_heights: *const f32,
    face_heights_len: usize,
    dz_reference_m: f64,
    damping_depth_m: f64,
    damping_rate_s_inv: f64,
    tauh: *mut f32,
    tauh_len: usize,
    taus: *mut f32,
    taus_len: usize,
    tauf: *mut f32,
    tauf_len: usize,
    parameters: *mut f32,
    parameters_len: usize,
) -> i32 {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
    let result = catch_unwind(AssertUnwindSafe(|| {
        let faces = checked_levels(nz)?;
        if (tauh_len, taus_len, tauf_len, parameters_len)
            != (nz, nz, faces, RAYLEIGH_PARAMETER_COUNT)
        {
            return Err(invalid("Rayleigh output lengths differ from nz/nz/nz+1/5"));
        }
        if (mass_heights_len, face_heights_len) != (nz, faces) {
            return Err(invalid(
                "Rayleigh coordinate lengths differ from zh(1..nk) and zf(1..nk+1)",
            ));
        }
        let ranges = [
            checked_output(tauh, tauh_len)?,
            checked_output(taus, taus_len)?,
            checked_output(tauf, tauf_len)?,
            checked_output(parameters, parameters_len)?,
        ];
        for (index, &(start, end)) in ranges.iter().enumerate() {
            if ranges[index + 1..]
                .iter()
                .any(|&(other_start, other_end)| start < other_end && other_start < end)
            {
                return Err(invalid("Rayleigh output buffers must not overlap"));
            }
        }
        checked_input(mass_heights, mass_heights_len)?;
        checked_input(face_heights, face_heights_len)?;
        // Safety: both pointers were just checked for null, alignment and a
        // non-overflowing byte range at the declared lengths.
        let (zh, zf) = unsafe {
            (
                std::slice::from_raw_parts(mass_heights, mass_heights_len),
                std::slice::from_raw_parts(face_heights, face_heights_len),
            )
        };
        stretched_rayleigh_profiles(zh, zf, dz_reference_m, damping_depth_m, damping_rate_s_inv)
    }));
    match result {
        Ok(Ok(profiles)) => {
            unsafe {
                ptr::copy_nonoverlapping(profiles.tauh.as_ptr(), tauh, tauh_len);
                ptr::copy_nonoverlapping(profiles.taus.as_ptr(), taus, taus_len);
                ptr::copy_nonoverlapping(profiles.tauf.as_ptr(), tauf, tauf_len);
                ptr::copy_nonoverlapping(profiles.parameters.as_ptr(), parameters, parameters_len);
            }
            0
        }
        Ok(Err(error)) => {
            set_error(error.to_string());
            -1
        }
        Err(_) => {
            set_error("Rust Rayleigh profile producer panicked");
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn gm1_rayleigh_profiles_last_error_len() -> usize {
    LAST_ERROR.with(|slot| slot.borrow().len())
}

/// Copy the last thread-local error as UTF-8; return zero for a short buffer.
///
/// # Safety
/// ``output`` must name writable byte storage for ``capacity`` bytes.
#[no_mangle]
pub unsafe extern "C" fn gm1_rayleigh_profiles_last_error_copy(
    output: *mut u8,
    capacity: usize,
) -> usize {
    LAST_ERROR.with(|slot| {
        let error = slot.borrow();
        if output.is_null() || capacity < error.len() {
            return 0;
        }
        unsafe { ptr::copy_nonoverlapping(error.as_ptr(), output, error.len()) };
        error.len()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256;

    fn raw_sha(values: &[f32]) -> String {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        sha256::digest_hex(&bytes)
    }

    #[test]
    fn profiles_preserve_the_measured_target_cosine_words() {
        // All native hashes come from the extracted pristine param.F loops,
        // with GNU Fortran 13.3 O0/O2/repeat raw-bit agreement.  Windows UCRT
        // differs at exactly one 237/75 mass word; its hash is a target-libm
        // regression, explicitly not an assertion of native raw-bit equality.
        for (nz, dz, native_h, native_f) in [
            (
                237,
                75.0,
                "234fa7317871742a34cc2c7a20a5a39fb2ed28dc13ef67d34207b473a53b4e6f",
                "8733989b9b3445f9cb9abdf01ca8660158e456aa3c1d5018da47b25ebd016a49",
            ),
            (
                75,
                200.0,
                "8099fc9a112da0260c8798d2d46726f74c8f81393774c9aa6129befb4441cc9b",
                "8b672d0eb23bb0fbd270543f168fb9a67bae0190eb3638d503380398016fc740",
            ),
            (
                237,
                75.1_f32 as f64,
                "dee566c2b13dec7daad9a7dd7eca9d3f399c93d53eae5560023fc8ede75afbe6",
                "cd46431a655a84fd7fd178a2a5905c3d365d001fdacd91dfbdd5bc9c6f2168b6",
            ),
        ] {
            let profiles = uniform_rayleigh_profiles(nz, dz, 3000.0, 0.0033333333333).unwrap();
            assert_eq!(profiles.tauh.len(), nz);
            assert_eq!(profiles.taus.len(), nz);
            assert_eq!(profiles.tauf.len(), nz + 1);
            assert_eq!(raw_sha(&profiles.tauh), raw_sha(&profiles.taus));
            assert_eq!(profiles.tauf[0].to_bits(), 0);
            assert_eq!(profiles.tauf[nz].to_bits(), 1.0_f32.to_bits());
            assert!(profiles
                .tauh
                .iter()
                .all(|value| (0.0..=1.0).contains(value)));
            if cfg!(target_os = "linux") || cfg!(all(target_os = "windows", target_env = "msvc")) {
                let expected_h = if cfg!(target_os = "windows") && nz == 237 && dz == 75.0 {
                    "97333bbf015ca79082904ed070f4d345fde6ad6ff1f6e9f5a6ab9be2f0cf829b"
                } else {
                    native_h
                };
                assert_eq!(raw_sha(&profiles.tauh), expected_h, "nz={nz}, dz={dz}");
                assert_eq!(raw_sha(&profiles.tauf), native_f, "nz={nz}, dz={dz}");
            }
        }
    }

    /// zf(1..nk+1) of a two-segment stretched column and its zh midpoints.
    fn stretched_column(thin: usize, thin_dz: f32, thick: usize, thick_dz: f32) -> (Vec<f32>, Vec<f32>) {
        let mut zf = vec![0.0_f32];
        for _ in 0..thin {
            zf.push(zf[zf.len() - 1] + thin_dz);
        }
        for _ in 0..thick {
            zf.push(zf[zf.len() - 1] + thick_dz);
        }
        let zh = zf
            .windows(2)
            .map(|window| 0.5_f32 * (window[0] + window[1]))
            .collect::<Vec<_>>();
        (zh, zf)
    }

    #[test]
    fn the_uniform_coordinate_through_the_level_entry_is_raw_bit_identical() {
        // param.F:8652-8671 reads zh and zf and nothing else, so the uniform
        // branch must be the level branch evaluated on param.F:7789-7811's
        // own coordinate -- to the raw word, not to a tolerance.
        for (nz, dz) in [(237_usize, 75.0_f64), (75, 200.0), (40, 500.0), (8, 1000.0)] {
            let uniform = uniform_rayleigh_profiles(nz, dz, 3000.0, 0.0033333333333).unwrap();
            let dz32 = dz as f32;
            let zf = (0..=nz).map(|k| dz32 * k as f32).collect::<Vec<_>>();
            let zh = (0..nz)
                .map(|k| 0.5_f32 * (dz32 * k as f32 + dz32 * (k + 1) as f32))
                .collect::<Vec<_>>();
            let levels =
                stretched_rayleigh_profiles(&zh, &zf, dz, 3000.0, 0.0033333333333).unwrap();
            assert_eq!(raw_sha(&levels.tauh), raw_sha(&uniform.tauh), "nz={nz}");
            assert_eq!(raw_sha(&levels.taus), raw_sha(&uniform.taus), "nz={nz}");
            assert_eq!(raw_sha(&levels.tauf), raw_sha(&uniform.tauf), "nz={nz}");
            assert_eq!(
                levels.parameters.map(f32::to_bits),
                uniform.parameters.map(f32::to_bits),
                "nz={nz}"
            );
        }
    }

    #[test]
    fn a_stretched_column_damps_from_its_own_top_face() {
        // 10 layers of 20 m then 20 of 320 m: maxz = zf(nk+1) = 6600 m, which
        // is nothing like nz*dz for any single dz.  param.F:8193 uses this
        // number and param.F:8656 divides by zf(nk+1)-zd.
        let (zh, zf) = stretched_column(10, 20.0, 20, 320.0);
        let nz = zh.len();
        assert_eq!(zf[nz].to_bits(), 6600.0_f32.to_bits());
        let profiles = stretched_rayleigh_profiles(&zh, &zf, 100.0, 3000.0, 0.0033333333333)
            .expect("a 3000 m layer under a 6600 m top is above half the domain");
        assert_eq!(profiles.parameters[1].to_bits(), 6600.0_f32.to_bits());
        assert_eq!(profiles.parameters[2].to_bits(), 3600.0_f32.to_bits());
        // param.F:8170 keeps dz as the metric reference, not a layer depth.
        assert_eq!(profiles.parameters[0].to_bits(), 100.0_f32.to_bits());
        assert_eq!(profiles.tauf[0].to_bits(), 0);
        assert_eq!(profiles.tauf[nz].to_bits(), 1.0_f32.to_bits());
        for (k, &height) in zh.iter().enumerate() {
            if height <= 3600.0 {
                assert_eq!(profiles.tauh[k].to_bits(), 0, "k={k}");
            } else {
                assert!(profiles.tauh[k] > 0.0, "k={k}");
            }
        }
        for window in profiles.tauf.windows(2) {
            assert!(window[1] >= window[0]);
        }
        // The uniform entry over the same nz cannot reproduce it, which is
        // the whole point: nz*dz is 6600 m only for dz = 220 m, and that
        // coordinate puts its first scalar level at 110 m instead of 10 m.
        let uniform = uniform_rayleigh_profiles(nz, 220.0, 3000.0, 0.0033333333333).unwrap();
        assert_eq!(uniform.parameters[1].to_bits(), 6600.0_f32.to_bits());
        assert_ne!(raw_sha(&uniform.tauh), raw_sha(&profiles.tauh));
    }

    #[test]
    fn a_stretched_column_refuses_by_the_exact_coordinate_defect() {
        let (zh, zf) = stretched_column(10, 20.0, 20, 320.0);
        let mut sunken = zf.clone();
        sunken[0] = 1.0;
        assert!(stretched_rayleigh_profiles(&zh, &sunken, 100.0, 3000.0, 0.0)
            .unwrap_err()
            .to_string()
            .contains("ground"));
        let mut folded = zf.clone();
        folded[5] = folded[6];
        assert!(stretched_rayleigh_profiles(&zh, &folded, 100.0, 3000.0, 0.0)
            .unwrap_err()
            .to_string()
            .contains("non-positive depth"));
        let mut displaced = zh.clone();
        displaced[3] = zf[9];
        assert!(stretched_rayleigh_profiles(&displaced, &zf, 100.0, 3000.0, 0.0)
            .unwrap_err()
            .to_string()
            .contains("inside its own layer"));
        assert!(stretched_rayleigh_profiles(&zh, &zf[..zf.len() - 1], 100.0, 3000.0, 0.0).is_err());
        // param.F:8635-8650 stops CM1 when the damper covers half the domain.
        assert!(stretched_rayleigh_profiles(&zh, &zf, 100.0, 4000.0, 0.0).is_err());
    }

    #[test]
    fn zero_depth_is_source_zero_and_rate_is_applied_only_by_the_consumer() {
        let zero = uniform_rayleigh_profiles(8, 75.0, 0.0, 0.0033333333333).unwrap();
        assert!(zero
            .tauh
            .iter()
            .chain(&zero.taus)
            .chain(&zero.tauf)
            .all(|value| value.to_bits() == 0));
        let inactive_rate = uniform_rayleigh_profiles(8, 75.0, 120.0, 0.0).unwrap();
        let active_rate = uniform_rayleigh_profiles(8, 75.0, 120.0, 0.0033333333333).unwrap();
        assert_eq!(raw_sha(&inactive_rate.tauh), raw_sha(&active_rate.tauh));
        assert_eq!(raw_sha(&inactive_rate.tauf), raw_sha(&active_rate.tauf));
        assert_eq!(active_rate.parameters[..4], [75.0, 600.0, 480.0, 120.0]);
        assert_eq!(
            active_rate.parameters[4].to_bits(),
            0.0033333333333_f32.to_bits()
        );
        assert_eq!(active_rate.tauf[8].to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn source_guard_and_invalid_dimensions_refuse_before_allocation() {
        assert!(uniform_rayleigh_profiles(8, 75.0, 300.0, 0.0).is_ok());
        for (nz, dz, depth, rate, message) in [
            (0, 75.0, 120.0, 0.0, "positive"),
            (usize::MAX, 75.0, 120.0, 0.0, "ghost indices"),
            (8, 0.0, 120.0, 0.0, "remain positive"),
            (8, f64::NAN, 120.0, 0.0, "finite"),
            (8, f32::MAX as f64, 120.0, 0.0, "overflows"),
            (8, 75.0, -1.0, 0.0, "nonnegative"),
            (8, 75.0, 300.01, 0.0, "half the model top"),
            (8, 75.0, 120.0, -0.1, "nonnegative"),
            (8, 75.0, 120.0, f64::INFINITY, "finite"),
        ] {
            let error = uniform_rayleigh_profiles(nz, dz, depth, rate).unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    #[test]
    fn ffi_copies_profiles_and_reports_source_rounded_parameters() {
        let mut tauh = [f32::NAN; 8];
        let mut taus = [f32::NAN; 8];
        let mut tauf = [f32::NAN; 9];
        let mut parameters = [f32::NAN; 5];
        let status = unsafe {
            gm1_rayleigh_profiles_fill(
                8,
                75.0,
                120.0,
                0.0033333333333,
                tauh.as_mut_ptr(),
                tauh.len(),
                taus.as_mut_ptr(),
                taus.len(),
                tauf.as_mut_ptr(),
                tauf.len(),
                parameters.as_mut_ptr(),
                parameters.len(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(gm1_rayleigh_profiles_last_error_len(), 0);
        let expected = uniform_rayleigh_profiles(8, 75.0, 120.0, 0.0033333333333).unwrap();
        assert_eq!(raw_sha(&tauh), raw_sha(&expected.tauh));
        assert_eq!(raw_sha(&taus), raw_sha(&expected.taus));
        assert_eq!(raw_sha(&tauf), raw_sha(&expected.tauf));
        assert_eq!(parameters, expected.parameters);
    }

    #[test]
    fn ffi_refusals_leave_all_caller_buffers_untouched() {
        let mut tauh = [123.0_f32; 8];
        let mut taus = [123.0_f32; 8];
        let mut tauf = [123.0_f32; 9];
        let mut parameters = [123.0_f32; 5];
        for mode in 0..5 {
            let h_pointer = if mode == 0 {
                ptr::null_mut()
            } else if mode == 1 {
                unsafe { tauh.as_mut_ptr().cast::<u8>().add(1).cast::<f32>() }
            } else {
                tauh.as_mut_ptr()
            };
            let s_pointer = if mode == 2 {
                tauh.as_mut_ptr()
            } else {
                taus.as_mut_ptr()
            };
            let status = unsafe {
                gm1_rayleigh_profiles_fill(
                    8,
                    75.0,
                    if mode == 4 { 400.0 } else { 120.0 },
                    0.0,
                    h_pointer,
                    if mode == 3 { 7 } else { 8 },
                    s_pointer,
                    8,
                    tauf.as_mut_ptr(),
                    9,
                    parameters.as_mut_ptr(),
                    5,
                )
            };
            assert_eq!(status, -1);
            let length = gm1_rayleigh_profiles_last_error_len();
            assert!(length > 0);
            let mut error = vec![0_u8; length];
            assert_eq!(
                unsafe { gm1_rayleigh_profiles_last_error_copy(error.as_mut_ptr(), length) },
                length
            );
            assert!(!std::str::from_utf8(&error).unwrap().is_empty());
            assert!(tauh
                .iter()
                .chain(&taus)
                .chain(&tauf)
                .chain(&parameters)
                .all(|value| value.to_bits() == 123.0_f32.to_bits()));
        }
    }
}
