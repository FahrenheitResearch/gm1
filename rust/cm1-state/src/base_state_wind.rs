//! CM1 r21.1 `base.F` base-state wind profiles: every `iwnd` branch.
//!
//! `base.F:2263-2650` is one `IF(isnd.ne.7)` block that first zeroes `u0` and
//! `v0` (`base.F:2263-2280`) and then selects one closed-form hodograph by
//! `iwnd`.  `param.F:836-852` forces `iwnd=0` whenever `isnd=7`, because that
//! sounding already carries its own winds; that rule lives in
//! [`crate::idealized_initialization::IdealizedInitializationSelection`] and
//! is not repeated here.
//!
//! Every branch is a pure function of one staggered face height, so the whole
//! set is transcribed as two functions of `(iwnd, testcase, z)` plus the two
//! `iwnd=8` hurricane words.  The `u` profile is evaluated at the mean of the
//! two scalar heights either side of the `u` point in `x` (`base.F:2294`) and
//! the `v` profile at the mean of the two either side in `y` (`base.F:2313`).
//!
//! `iwnd=7` does not exist in `base.F`; the eleven implemented branches are
//! 0, 1, 2, 3, 4, 5, 6, 8, 9, 10 and 11 (`README.namelist:653-669`).

use crate::idealized_initialization::{invalid, require_finite};
use crate::Result;

/// `constants.F:14` rounded to a default REAL, exactly as `base.F` sees it.
const PI: f32 = std::f32::consts::PI;

/// The `iwnd` values `base.F:2286-2640` implements, in source order.
pub const IMPLEMENTED_IWND: [i32; 11] = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11];

/// `base.F:2513-2520`: `iwnd=8` reads `hurr_vg` and `hurr_angle` rather than
/// literals.  `run/namelist.input:403` is the shipped gradient wind and
/// `param.F:8973` derives the angle in radians from `hurr_rotate` degrees --
/// and only under `lspgrad=3`, so zero is what every other deck sees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HurricaneBaseWind {
    pub gradient_wind_m_per_s: f32,
    pub rotate_deg: f32,
}

impl Default for HurricaneBaseWind {
    fn default() -> Self {
        Self {
            gradient_wind_m_per_s: 40.0_f32,
            rotate_deg: 0.0_f32,
        }
    }
}

impl HurricaneBaseWind {
    pub fn new(gradient_wind_m_per_s: f32, rotate_deg: f32) -> Result<Self> {
        require_finite("hurr_vg", gradient_wind_m_per_s)?;
        require_finite("hurr_rotate", rotate_deg)?;
        Ok(Self {
            gradient_wind_m_per_s,
            rotate_deg,
        })
    }

    /// `param.F:8973`: `hurr_angle = dble(hurr_rotate)*(pi/180)`, evaluated in
    /// double and used as a double by `base.F:2545-2546`.
    pub fn angle_rad(self) -> f64 {
        f64::from(self.rotate_deg) * (std::f64::consts::PI / 180.0_f64)
    }
}

/// `base.F:2467-2510`: the `iwnd=6` constant wind, selected by `testcase`.
///
/// Returns `(uval, vval)`.  The `else` arm at `base.F:2499-2502` is CM1's
/// default for every `testcase` the branch does not name.
pub fn constant_wind_for_testcase(testcase: i32) -> (f32, f32) {
    match testcase {
        // Sullivan and Patton (2011, JAS)
        1 => (1.0_f32, 0.0_f32),
        // Moeng and Sullivan (1994, JAS)
        2 => (15.0_f32, 0.0_f32),
        // nonprecipitating stratocumulus (Stevens et al. MWR 2005)
        4 => (7.0_f32, -5.5_f32),
        // stable boundary layer (Beare et al. 2006, BLM)
        9 => (8.0_f32, 0.0_f32),
        // SAS
        11 => (2.0_f32, 0.0_f32),
        // Martinuzzi and Tropea (1993) wind tunnel
        12 => (20.0_f32, 0.0_f32),
        // shallow Cu over land (Brown et al. 2002)
        14 => (10.0_f32, 0.0_f32),
        _ => (10.0_f32, 0.0_f32),
    }
}

/// Refuse an `iwnd` `base.F` does not implement.
///
/// `base.F:2286-2640` is a closed `if/elseif` chain with no `else`: an
/// unimplemented selector falls through it and leaves the zeroed `u0`/`v0` of
/// `base.F:2263-2280` behind, silently.  Naming the value here stops a run
/// that would report the requested hodograph in its receipt and integrate a
/// motionless atmosphere instead.
pub fn require_implemented_iwnd(iwnd: i32) -> Result<()> {
    if IMPLEMENTED_IWND.contains(&iwnd) {
        return Ok(());
    }
    Err(invalid(format!(
        "base.F:2286-2640 implements iwnd in {IMPLEMENTED_IWND:?}; got iwnd={iwnd}. CM1's own \
         chain has no else arm, so this selector would leave the base.F:2263-2280 zero wind in \
         place and the run would integrate a motionless environment while its receipt named a \
         shear profile"
    )))
}

/// `base.F:2286-2640`, the `u0` branch, evaluated on one u-face height.
pub fn base_wind_u(iwnd: i32, testcase: i32, zu_m: f32, hurricane: HurricaneBaseWind) -> f32 {
    match iwnd {
        // base.F:2263-2280: zero wind.
        0 => 0.0_f32,
        // base.F:2286-2311: RKW-type profile (Rotunno, Klemp and Weisman 1988).
        1 => {
            let (udep1, udep2) = (0.0_f32, 2500.0_f32);
            let (uconst1, uconst2) = (0.0_f32, 10.0_f32);
            if zu_m < udep1 {
                uconst1
            } else if zu_m > udep1 && zu_m < udep2 {
                (uconst2 - uconst1) * (zu_m - udep1) / (udep2 - udep1) + uconst1
            } else {
                uconst2
            }
        }
        // base.F:2321-2342: Weisman-Klemp supercell.
        2 => {
            let (udep1, udep2) = (2000.0_f32, 6000.0_f32);
            let (umax1, umax2) = (7.0_f32, 31.0_f32);
            if zu_m <= udep1 {
                let angle = 90.0_f32 * (zu_m / udep1) * (PI / 180.0_f32);
                umax1 - umax1 * angle.cos()
            } else if zu_m > udep1 && zu_m <= udep2 {
                umax1 + (zu_m - udep1) * (umax2 - umax1) / (udep2 - udep1)
            } else {
                umax2
            }
        }
        // base.F:2366-2396: multicell.
        3 => {
            let (udep1, udep2) = (0.0_f32, 7500.0_f32);
            let umax1 = -40.0_f32 / PI;
            let umax2 = 40.0_f32 / PI + 40.0_f32;
            if zu_m <= udep2 {
                umax1 + (zu_m - udep1) * (umax2 - umax1) / (udep2 - udep1)
            } else {
                umax2
            }
        }
        // base.F:2400-2419: Weisman-Klemp multicell.
        4 => 35.0_f32 * (zu_m / 3000.0_f32).tanh(),
        // base.F:2425-2461: Dornbrack et al. (2005) analytic profile.
        5 => {
            let umax = 15.0_f32;
            let (udep1, udep2) = (4000.0_f32, 6000.0_f32);
            if zu_m < udep1 {
                umax
            } else if zu_m < udep2 {
                let alpha =
                    0.25_f32 * PI * (1.0_f32 + (PI * (zu_m - udep1) / (udep2 - udep1)).cos());
                umax * alpha.sin()
            } else {
                0.0_f32
            }
        }
        // base.F:2467-2510: constant wind.
        6 => constant_wind_for_testcase(testcase).0,
        // base.F:2525-2552: simple hurricane boundary layer (Bryan et al. 2017).
        // ug(k)=0 and vg(k)=hurr_vg, then rotated into Cartesian components.
        8 => (-f64::from(hurricane.gradient_wind_m_per_s) * (-hurricane.angle_rad()).sin()) as f32,
        // base.F:2557-2589: Siebesma et al. (2003) shallow Cu.
        9 => {
            let (udep1, udep2) = (700.0_f32, 3000.0_f32);
            let (uconst1, uconst2) = (-8.75_f32, -4.61_f32);
            if zu_m <= udep1 {
                uconst1
            } else if zu_m > udep1 && zu_m < udep2 {
                (uconst2 - uconst1) * (zu_m - udep1) / (udep2 - udep1) + uconst1
            } else {
                uconst2
            }
        }
        // base.F:2593-2613: Ackerman et al. (2009) drizzling Sc.
        10 => 3.0_f32 + 4.3_f32 * zu_m / 1000.0_f32,
        // base.F:2617-2640: RICO precipitating shallow Cu.
        11 => {
            let (udep1, udep2) = (0.0_f32, 4000.0_f32);
            let (uconst1, uconst2) = (-9.9_f32, -1.9_f32);
            if zu_m <= udep1 {
                uconst1
            } else if zu_m > udep1 && zu_m < udep2 {
                (uconst2 - uconst1) * (zu_m - udep1) / (udep2 - udep1) + uconst1
            } else {
                uconst2
            }
        }
        // Unreachable: require_implemented_iwnd refuses every other value
        // before a profile is built.  base.F leaves the zeroed field here.
        _ => 0.0_f32,
    }
}

/// `base.F:2286-2640`, the `v0` branch, evaluated on one v-face height.
#[allow(clippy::if_same_then_else)] // base.F:2353-2356 writes both arms.
pub fn base_wind_v(iwnd: i32, testcase: i32, zv_m: f32, hurricane: HurricaneBaseWind) -> f32 {
    match iwnd {
        0 => 0.0_f32,
        // base.F:2305-2311.
        1 => 0.0_f32,
        // base.F:2344-2357: vmax1 = umax1 = 7 m/s.
        2 => {
            let udep1 = 2000.0_f32;
            let udep2 = 6000.0_f32;
            let vmax1 = 7.0_f32;
            if zv_m <= udep1 {
                let angle = 90.0_f32 * (zv_m / udep1) * (PI / 180.0_f32);
                vmax1 * angle.sin()
            } else if zv_m > udep1 && zv_m <= udep2 {
                vmax1
            } else {
                vmax1
            }
        }
        // base.F:2388-2396.
        3 => 40.0_f32 / PI,
        // base.F:2412-2419.
        4 => 0.0_f32,
        // base.F:2446-2461.
        5 => {
            let umax = 15.0_f32;
            let (udep1, udep2) = (4000.0_f32, 6000.0_f32);
            if zv_m < udep1 {
                0.0_f32
            } else if zv_m < udep2 {
                let alpha =
                    0.25_f32 * PI * (1.0_f32 + (PI * (zv_m - udep1) / (udep2 - udep1)).cos());
                umax * alpha.cos()
            } else {
                umax
            }
        }
        6 => constant_wind_for_testcase(testcase).1,
        // base.F:2546: v0 = vg*cos(-hurr_angle).
        8 => (f64::from(hurricane.gradient_wind_m_per_s) * (-hurricane.angle_rad()).cos()) as f32,
        // base.F:2581-2586.
        9 => 0.0_f32,
        // base.F:2605-2610.
        10 => -9.0_f32 + 5.6_f32 * zv_m / 1000.0_f32,
        // base.F:2634-2638.
        11 => -3.8_f32,
        _ => 0.0_f32,
    }
}

/// Build the native staggered `u0` `[nk,nj,ni+1]` and `v0` `[nk,nj+1,ni]`.
///
/// `zu` and `zv` are the face-averaged heights of `base.F:2294` and
/// `base.F:2313`; [`crate::storm_initialization::analytic_shear_face_heights`]
/// builds them from the mass heights.
pub fn build_base_wind_faces(
    iwnd: i32,
    testcase: i32,
    hurricane: HurricaneBaseWind,
    zu_m: &[f32],
    zv_m: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    require_implemented_iwnd(iwnd)?;
    let u0 = zu_m
        .iter()
        .map(|z| base_wind_u(iwnd, testcase, *z, hurricane))
        .collect::<Vec<_>>();
    let v0 = zv_m
        .iter()
        .map(|z| base_wind_v(iwnd, testcase, *z, hurricane))
        .collect::<Vec<_>>();
    for (label, values) in [("u0", &u0), ("v0", &v0)] {
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            return Err(invalid(format!(
                "base.F iwnd={iwnd} produced a non-finite {label} word at index {index}"
            )));
        }
    }
    Ok((u0, v0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storm_initialization::{analytic_shear_u, analytic_shear_v};

    #[test]
    fn iwnd2_matches_the_established_analytic_shear_transcription_bit_for_bit() {
        // The iwnd=2 branch already had a first-class transcription in
        // storm_initialization; the general evaluator must be the same words,
        // or the sealed isnd=5 carrier bytes would move.
        let hurricane = HurricaneBaseWind::default();
        let mut z = -500.0_f32;
        while z <= 20_000.0_f32 {
            assert_eq!(
                base_wind_u(2, 0, z, hurricane).to_bits(),
                analytic_shear_u(z).to_bits(),
                "u0 differs at z={z}"
            );
            assert_eq!(
                base_wind_v(2, 0, z, hurricane).to_bits(),
                analytic_shear_v(z).to_bits(),
                "v0 differs at z={z}"
            );
            z += 12.5_f32;
        }
    }

    #[test]
    fn every_documented_branch_is_implemented_and_the_missing_one_is_refused() {
        // README.namelist:653-669 lists 0,1,2,3,4,5,6,8,9,10,11 and no 7.
        for iwnd in IMPLEMENTED_IWND {
            require_implemented_iwnd(iwnd).unwrap();
        }
        for iwnd in [-1_i32, 7, 12, 99] {
            let error = require_implemented_iwnd(iwnd).unwrap_err().to_string();
            assert!(error.contains("motionless environment"), "{error}");
        }
    }

    #[test]
    fn branch_values_match_hand_evaluated_source_at_named_heights() {
        let h = HurricaneBaseWind::default();
        // iwnd=1, base.F:2286-2311.  udep1 is exactly 0, so base.F's own
        // chain -- `lt` then `gt .and. lt` then else -- sends zu=0 to the
        // else arm and uconst2.  The discontinuity is the source's; the
        // first scalar level sits at 0.5*dz and takes the linear arm.
        assert_eq!(base_wind_u(1, 0, 0.0, h), 10.0);
        assert_eq!(base_wind_u(1, 0, -1.0, h), 0.0);
        assert_eq!(base_wind_u(1, 0, 1250.0, h), 5.0);
        assert_eq!(base_wind_u(1, 0, 2500.0, h), 10.0);
        assert_eq!(base_wind_u(1, 0, 9000.0, h), 10.0);
        assert_eq!(base_wind_v(1, 0, 1250.0, h), 0.0);
        // iwnd=3, base.F:2366-2396.
        assert_eq!(base_wind_u(3, 0, 0.0, h), -40.0_f32 / PI);
        assert_eq!(base_wind_u(3, 0, 7500.0, h), 40.0_f32 / PI + 40.0);
        assert_eq!(base_wind_u(3, 0, 12000.0, h), 40.0_f32 / PI + 40.0);
        assert_eq!(base_wind_v(3, 0, 0.0, h), 40.0_f32 / PI);
        // iwnd=4, base.F:2400-2419.
        assert_eq!(base_wind_u(4, 0, 3000.0, h), 35.0_f32 * 1.0_f32.tanh());
        assert_eq!(base_wind_v(4, 0, 3000.0, h), 0.0);
        // iwnd=5, base.F:2425-2461.
        assert_eq!(base_wind_u(5, 0, 1000.0, h), 15.0);
        assert_eq!(base_wind_u(5, 0, 9000.0, h), 0.0);
        assert_eq!(base_wind_v(5, 0, 1000.0, h), 0.0);
        assert_eq!(base_wind_v(5, 0, 9000.0, h), 15.0);
        {
            // the transition arm at z=5000: alpha = 0.25*pi*(1+cos(pi/2))
            let alpha = 0.25_f32 * PI * (1.0_f32 + (PI * 0.5_f32).cos());
            assert_eq!(base_wind_u(5, 0, 5000.0, h), 15.0_f32 * alpha.sin());
            assert_eq!(base_wind_v(5, 0, 5000.0, h), 15.0_f32 * alpha.cos());
        }
        // iwnd=6, base.F:2467-2510: height-independent, testcase-selected.
        for (testcase, expected) in [
            (0_i32, (10.0_f32, 0.0_f32)),
            (1, (1.0, 0.0)),
            (2, (15.0, 0.0)),
            (3, (10.0, 0.0)),
            (4, (7.0, -5.5)),
            (9, (8.0, 0.0)),
            (11, (2.0, 0.0)),
            (12, (20.0, 0.0)),
            (14, (10.0, 0.0)),
        ] {
            assert_eq!(base_wind_u(6, testcase, 0.0, h), expected.0, "tc={testcase}");
            assert_eq!(
                base_wind_u(6, testcase, 9999.0, h),
                expected.0,
                "tc={testcase}"
            );
            assert_eq!(base_wind_v(6, testcase, 0.0, h), expected.1, "tc={testcase}");
        }
        // iwnd=8, base.F:2525-2552 with hurr_rotate=0: pure v.
        assert_eq!(base_wind_u(8, 0, 500.0, h), 0.0);
        assert_eq!(base_wind_v(8, 0, 500.0, h), 40.0);
        let rotated = HurricaneBaseWind::new(40.0, 90.0).unwrap();
        assert!((base_wind_u(8, 0, 0.0, rotated) - 40.0).abs() < 1.0e-4);
        assert!(base_wind_v(8, 0, 0.0, rotated).abs() < 1.0e-4);
        // iwnd=9, base.F:2557-2589.
        assert_eq!(base_wind_u(9, 0, 0.0, h), -8.75);
        assert_eq!(base_wind_u(9, 0, 700.0, h), -8.75);
        assert_eq!(base_wind_u(9, 0, 5000.0, h), -4.61);
        assert_eq!(base_wind_v(9, 0, 1000.0, h), 0.0);
        // iwnd=10, base.F:2593-2613.
        assert_eq!(base_wind_u(10, 0, 1000.0, h), 3.0 + 4.3);
        assert_eq!(base_wind_v(10, 0, 1000.0, h), -9.0 + 5.6);
        // iwnd=11, base.F:2617-2640.
        assert_eq!(base_wind_u(11, 0, 0.0, h), -9.9);
        assert_eq!(base_wind_u(11, 0, 4000.0, h), -1.9);
        assert_eq!(base_wind_u(11, 0, 2000.0, h), -9.9 + 8.0 * 0.5);
        assert_eq!(base_wind_v(11, 0, 2000.0, h), -3.8);
        // iwnd=0 is zero everywhere in both components.
        assert_eq!(base_wind_u(0, 0, 1234.0, h), 0.0);
        assert_eq!(base_wind_v(0, 0, 1234.0, h), 0.0);
    }
}
