//! CM1 r21.1's running domain-location integral, `domainlocx`/`domainlocy`.
//!
//! A translating frame (`imove=1`) buys a small domain by moving the grid with
//! the storm, and the price is that the grid's own coordinates stop being
//! ground coordinates the moment the first large step completes.  CM1 pays it
//! with one running integral: `domainlocx` and `domainlocy` are
//! `double precision` scalars (`src/cm1.F:266`), start at zero
//! (`src/cm1.F:348-349`), and advance once per completed large step by
//! `dt*umove` and `dt*vmove` (`src/cm1.F:3512-3513`).  They are the
//! ground-relative position of the domain, in metres, at the model time the
//! frame was written at.
//!
//! CM1 never reads them back into the solution.  A grep of the whole r21.1
//! source finds `domainloc` only in the declaration, the initialization, the
//! integral itself, the restart pair and the output writers -- never on the
//! right-hand side of a physics expression.  They are a diagnostic the model
//! owes its consumer, and the consumer cannot recompute them from the
//! artifact alone once the frame velocity is allowed to vary, which is why
//! CM1 carries the integral rather than the product.
//!
//! # Where CM1 puts them
//!
//! * Restart, so a resumed run continues the integral instead of restarting
//!   it: written at `src/restart_write.F:399-400` (binary) and
//!   `src/restart_write.F:542-546` (netCDF, defined `nf90_double` on the time
//!   dimension at `src/writeout_nc.F:810-811`); read back at
//!   `src/restart_read.F:419-420` and `src/restart_read.F:728-732` into an
//!   `intent(inout)` pair (`src/restart_read.F:151`) and broadcast to every
//!   rank at `src/restart_read.F:838-839`.
//! * History metadata, per history time, beside `umove`/`vmove`:
//!   `src/writeout.F:9614-9648` names them `domainlocx`/`domainlocy` and
//!   describes them as `x location of (center of) domain (m)`, under the same
//!   `imove.ge.1` guard that admits `umove`/`vmove` themselves.
//!   `src/domaindiag.F:464-483` writes the same pair with `varunit` `m`.
//!   `src/writeout_nc.F:281-291` and `src/writeout_nc.F:559-567` write
//!   `umove`/`vmove` per time index with units `m/s`.  The azimuthal-average
//!   statistics files carry the same pair under the same names
//!   (`src/azimavg.F:3931-3933,3965-3966` and
//!   `src/azimavg.F:4479-4481,4503-4504`), as single precision, because those
//!   are GrADS records.
//!
//! # Why this is exact, and why that is not automatic
//!
//! CM1 accumulates in binary64 and lets the error fall where it may.  GM1
//! accumulates the same way and then proves there is no error to fall.
//!
//! `umove` is CM1's `real`, binary32, so `dble(umove)` carries at most 24
//! significant bits.  This lane's `dt` is exactly 0.5 s -- a power of two --
//! so `dbldt*dble(umove)` is that same 24-bit significand with its exponent
//! decremented, which is exact.  After `n` steps the exact sum is `n` times a
//! 24-bit significand and needs at most `24 + ceil(log2(n))` significant
//! bits.  Binary64 holds 53, so every partial sum through
//! `n = 2^29 = 536_870_912` steps is representable exactly, and IEEE-754
//! addition of two exactly-representable operands whose exact sum is also
//! representable returns that sum with no rounding.  A 14,400-step two-hour
//! run is 1/37_282 of the way to the first rounding.
//!
//! That argument depends on facts a future edit could take away -- a
//! non-power-of-two `dt`, a frame speed carrying more than 24 significant
//! bits, or a step count past 2^29.  So it is not left as an argument:
//! [`DomainLocationIntegrator::advance`] checks the accumulated sum against
//! the closed form after every step and refuses the moment they part, rather
//! than publishing a domain location that has silently drifted away from
//! `umove*t`.

use crate::forecast::{
    ForecastClock, SchedulerScalar, SchedulerScalarSpec, SchedulerScalarType, SchedulerScalarValue,
};
use crate::idealized_initialization::TranslatingFrame;
use crate::{Result, StateError};
use std::io::{Error, ErrorKind};

/// `src/cm1.F:348`, `src/writeout.F:9635`, `src/writeout_nc.F:810`.
pub const DOMAIN_LOCATION_X_NAME: &str = "domainlocx";
/// `src/cm1.F:349`, `src/writeout.F:9644`, `src/writeout_nc.F:811`.
pub const DOMAIN_LOCATION_Y_NAME: &str = "domainlocy";
/// `src/writeout_nc.F:283-285`, `src/domaindiag.F:440-447`.
pub const MOVE_SPEED_X_NAME: &str = "umove";
/// `src/writeout_nc.F:287-289`, `src/domaindiag.F:452-459`.
pub const MOVE_SPEED_Y_NAME: &str = "vmove";
/// `src/domaindiag.F:466,478`: `varunit(nvar) = 'm'`.
pub const DOMAIN_LOCATION_UNITS: &str = "m";
/// `src/writeout_nc.F:285,289`: `units` attribute `m/s`.
pub const MOVE_SPEED_UNITS: &str = "m/s";
/// `src/writeout.F:9636`, `src/domaindiag.F:465`.
pub const DOMAIN_LOCATION_X_DESCRIPTION: &str = "x location of (center of) domain";
/// `src/writeout.F:9645`, `src/domaindiag.F:477`.
pub const DOMAIN_LOCATION_Y_DESCRIPTION: &str = "y location of (center of) domain";
/// `src/cm1.F:348-349`: both components start the run at zero.
pub const DOMAIN_LOCATION_INITIAL_M: f64 = 0.0;
/// `src/cm1.F:3510-3513`: advanced once per large step, immediately after
/// `mtime` is stepped forward and before any output or restart is written.
pub const DOMAIN_LOCATION_CADENCE: &str = "post-large-step-after-model-time";
/// The step count past which the exactness proof in this module's header no
/// longer holds for a binary32 frame speed and a power-of-two `dt`.
pub const DOMAIN_LOCATION_EXACT_STEP_LIMIT: u64 = 1 << 29;
/// The complete CM1 moving-frame scalar set this lane publishes per frame.
pub const DOMAIN_LOCATION_SCALAR_NAMES: [&str; 4] = [
    DOMAIN_LOCATION_X_NAME,
    DOMAIN_LOCATION_Y_NAME,
    MOVE_SPEED_X_NAME,
    MOVE_SPEED_Y_NAME,
];
/// What this module does and does not claim.
pub const DOMAIN_LOCATION_CLAIM_LIMIT: &str = "CM1 r21.1 cm1.F:348-349,3512-3513 domainlocx/domainlocy running integral at the completed-large-step boundary under one constant frame velocity, proved drift-free and carried bit-exactly through restart; no adaptive timestep, no change_uvmove, no adaptive domain-center diagnosis, and no trajectory, storm or forecast-skill claim";

fn invalid(message: impl Into<String>) -> StateError {
    Error::new(ErrorKind::InvalidInput, message.into()).into()
}

/// CM1's `domainlocx`/`domainlocy` at one completed-large-step boundary.
///
/// The pair is meaningless without the clock and frame that produced it, so
/// all four travel together: a value restored beside a different frame
/// velocity or a different step count is not the same integral and is
/// refused rather than continued.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DomainLocationIntegrator {
    dt_s: f64,
    frame: TranslatingFrame,
    completed_steps: u64,
    x_m: f64,
    y_m: f64,
}

impl DomainLocationIntegrator {
    /// A fresh run: `src/cm1.F:348-349`, both components at zero, no steps.
    pub fn new(dt_s: f64, frame: TranslatingFrame) -> Result<Self> {
        Self::resume(
            dt_s,
            frame,
            0,
            DOMAIN_LOCATION_INITIAL_M,
            DOMAIN_LOCATION_INITIAL_M,
        )
    }

    /// Continue a persisted integral: `src/restart_read.F:419-420`.
    ///
    /// CM1 declares the pair `intent(inout)` at `src/restart_read.F:151` and
    /// overwrites it from the restart file, so a resumed run continues the
    /// integral.  An accumulator that reset to zero here would mislocate
    /// every frame after the first restart by the whole distance the domain
    /// had already travelled, so the restored value is validated against the
    /// clock and frame it is being resumed under instead of being trusted.
    pub fn resume(
        dt_s: f64,
        frame: TranslatingFrame,
        completed_steps: u64,
        x_m: f64,
        y_m: f64,
    ) -> Result<Self> {
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(invalid(format!(
                "domain-location dt is {dt_s:?}; cm1.F:3512 multiplies it by umove \
                 once per large step, so a nonpositive or non-finite dt would make \
                 every published domain location non-finite or run the integral \
                 backwards"
            )));
        }
        for (name, value) in [(DOMAIN_LOCATION_X_NAME, x_m), (DOMAIN_LOCATION_Y_NAME, y_m)] {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "resumed {name} is {value:?}, which is not finite; cm1.F:3512 \
                     adds to it once per large step, so every domain location this \
                     run published from here on would be non-finite and no consumer \
                     could place a single frame on the ground"
                )));
            }
        }
        let restored = Self {
            dt_s,
            frame,
            completed_steps,
            x_m,
            y_m,
        };
        restored.require_exact("resumed")?;
        Ok(restored)
    }

    /// `src/cm1.F:3512-3513`, once per completed large step.
    ///
    /// CM1 advances this immediately after `mtime = mtime + dbldt`
    /// (`src/cm1.F:3510`) and before any history, statistics or restart write
    /// of that step, so the value published at step `n` is the integral over
    /// `n` completed steps, not `n-1`.
    pub fn advance(&mut self) -> Result<()> {
        let next = self.completed_steps.checked_add(1).ok_or_else(|| {
            invalid("domain-location step count overflowed u64; the integral has no defined value")
        })?;
        // cm1.F:3512: the x location gains the step times umove, in double precision.
        self.x_m += self.dt_s * f64::from(self.frame.u_move_m_per_s());
        // cm1.F:3513: the y location gains the step times vmove, in double precision.
        self.y_m += self.dt_s * f64::from(self.frame.v_move_m_per_s());
        self.completed_steps = next;
        self.require_exact("accumulated")
    }

    /// Advance `steps` completed large steps, one CM1 addition each.
    pub fn advance_steps(&mut self, steps: u64) -> Result<()> {
        for _ in 0..steps {
            self.advance()?;
        }
        Ok(())
    }

    /// The per-step increment `dbldt*dble(umove)` of `src/cm1.F:3512`.
    pub fn x_increment_m(&self) -> f64 {
        self.dt_s * f64::from(self.frame.u_move_m_per_s())
    }

    /// The per-step increment `dbldt*dble(vmove)` of `src/cm1.F:3513`.
    pub fn y_increment_m(&self) -> f64 {
        self.dt_s * f64::from(self.frame.v_move_m_per_s())
    }

    /// `domainlocx` (m).
    pub fn x_m(&self) -> f64 {
        self.x_m
    }

    /// `domainlocy` (m).
    pub fn y_m(&self) -> f64 {
        self.y_m
    }

    /// Completed large steps behind this value.
    pub fn completed_steps(&self) -> u64 {
        self.completed_steps
    }

    /// CM1's `mtime` for this value: `completed_steps * dt`.
    pub fn model_time_s(&self) -> f64 {
        self.completed_steps as f64 * self.dt_s
    }

    /// The frame velocity this integral is of.
    pub fn frame(&self) -> TranslatingFrame {
        self.frame
    }

    /// The large-step interval this integral is of.
    pub fn dt_s(&self) -> f64 {
        self.dt_s
    }

    /// Refuse the moment the running sum stops being `n*dt*umove` exactly.
    ///
    /// This is the module header's exactness argument turned into a check, so
    /// a widened `dt`, a frame speed with more than 24 significant bits or a
    /// run past [`DOMAIN_LOCATION_EXACT_STEP_LIMIT`] steps stops the run
    /// instead of publishing a location that has drifted away from the ground
    /// truth by an amount that grows for the rest of the integration.
    fn require_exact(&self, stage: &str) -> Result<()> {
        let steps = self.completed_steps as f64;
        for (name, seen, increment) in [
            (DOMAIN_LOCATION_X_NAME, self.x_m, self.x_increment_m()),
            (DOMAIN_LOCATION_Y_NAME, self.y_m, self.y_increment_m()),
        ] {
            let closed = steps * increment;
            if seen != closed || !seen.is_finite() {
                return Err(invalid(format!(
                    "{stage} {name} is {seen:?} m after {} completed large steps of \
                     {} s at {} m/s, but the exact integral is {closed:?} m. \
                     cm1.F:3512's running sum is exact only while dt is a power of \
                     two, the frame speed fits binary32 and the run is shorter than \
                     {} steps; outside that the published domain location drifts \
                     away from umove*t and every frame after this one is placed on \
                     the wrong patch of ground",
                    self.completed_steps,
                    self.dt_s,
                    if name == DOMAIN_LOCATION_X_NAME {
                        self.frame.u_move_m_per_s()
                    } else {
                        self.frame.v_move_m_per_s()
                    },
                    DOMAIN_LOCATION_EXACT_STEP_LIMIT,
                )));
            }
        }
        Ok(())
    }

    /// The four CM1 moving-frame scalars a history or checkpoint frame
    /// carries, in the envelope's own bit-exact scalar encoding.
    ///
    /// `umove`/`vmove` are CM1 `real`, so their binary32 values are widened
    /// to binary64 exactly and come back unchanged; `domainlocx`/`domainlocy`
    /// are CM1 `double precision` already (`src/cm1.F:266`,
    /// `src/writeout_nc.F:810-811`).
    pub fn scheduler_scalars(&self) -> Result<Vec<SchedulerScalar>> {
        Ok(vec![
            SchedulerScalar::new(DOMAIN_LOCATION_X_NAME, SchedulerScalarValue::F64(self.x_m))?,
            SchedulerScalar::new(DOMAIN_LOCATION_Y_NAME, SchedulerScalarValue::F64(self.y_m))?,
            SchedulerScalar::new(
                MOVE_SPEED_X_NAME,
                SchedulerScalarValue::F64(f64::from(self.frame.u_move_m_per_s())),
            )?,
            SchedulerScalar::new(
                MOVE_SPEED_Y_NAME,
                SchedulerScalarValue::F64(f64::from(self.frame.v_move_m_per_s())),
            )?,
        ])
    }

    /// The contract specs matching [`Self::scheduler_scalars`].
    pub fn scheduler_scalar_specs() -> Result<Vec<SchedulerScalarSpec>> {
        DOMAIN_LOCATION_SCALAR_NAMES
            .iter()
            .map(|name| SchedulerScalarSpec::new(*name, SchedulerScalarType::F64))
            .collect()
    }

    /// Restore the integral from a published frame's clock and scalars.
    ///
    /// This is the read half of `src/restart_read.F:419-420`: the persisted
    /// pair is put back under the clock and frame it was written with, and
    /// the run continues the integral from there.
    pub fn from_frame(clock: ForecastClock, scalars: &[SchedulerScalar]) -> Result<Self> {
        let mut found = [None::<f64>; DOMAIN_LOCATION_SCALAR_NAMES.len()];
        for scalar in scalars {
            let Some(index) = DOMAIN_LOCATION_SCALAR_NAMES
                .iter()
                .position(|name| *name == scalar.name())
            else {
                continue;
            };
            let SchedulerScalarValue::F64(value) = scalar.value() else {
                return Err(invalid(format!(
                    "published {} is not an f64; cm1.F:266 declares the domain \
                     location double precision and restart_read.F:419-420 reads it \
                     back as one, so a narrower carrier would resume a rounded \
                     integral",
                    scalar.name()
                )));
            };
            if found[index].replace(value).is_some() {
                return Err(invalid(format!(
                    "published frame carries {} twice; there is one domain-location \
                     integral per model time and two values cannot both continue it",
                    scalar.name()
                )));
            }
        }
        let mut values = [0.0_f64; DOMAIN_LOCATION_SCALAR_NAMES.len()];
        for (index, name) in DOMAIN_LOCATION_SCALAR_NAMES.iter().enumerate() {
            values[index] = found[index].ok_or_else(|| {
                invalid(format!(
                    "published frame carries no {name}; restart_read.F:419-420 \
                     restores the domain-location integral from the persisted pair, \
                     and resuming without it would restart the integral at zero and \
                     mislocate every frame after the restart by the whole distance \
                     the domain had already travelled"
                ))
            })?;
        }
        let [x_m, y_m, umove, vmove] = values;
        for (name, value) in [(MOVE_SPEED_X_NAME, umove), (MOVE_SPEED_Y_NAME, vmove)] {
            if f64::from(value as f32) != value {
                return Err(invalid(format!(
                    "published {name} is {value:?} m/s, which is not a CM1 `real`; \
                     base.F:2667 subtracts a binary32 word from every base-state wind, \
                     so a value that does not survive binary32 is not the frame the \
                     state was built in and the resumed integral would be of a \
                     different frame than the carriers"
                )));
            }
        }
        let frame = TranslatingFrame::new(umove as f32, vmove as f32)?;
        Self::resume(clock.dt_s(), frame, clock.step(), x_m, y_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.5;
    // tools/run_cartesian_storm.py's two-hour profile: 14_400 large steps of
    // 0.5 s.  The exactness claim is asserted over the whole of it.
    const TWO_HOUR_STEPS: u64 = 14_400;

    fn frame(u: f32, v: f32) -> TranslatingFrame {
        TranslatingFrame::new(u, v).unwrap()
    }

    #[test]
    fn a_fixed_frame_never_leaves_cm1s_initial_value() {
        // cm1.F:348-349 initialize both to zero and cm1.F:3512-3513 add
        // dt*0.  A fixed-frame run's domain never moves, so every frame is
        // published at the origin CM1 started it at.
        let mut integral = DomainLocationIntegrator::new(DT, TranslatingFrame::fixed()).unwrap();
        assert_eq!(integral.x_m(), DOMAIN_LOCATION_INITIAL_M);
        assert_eq!(integral.y_m(), DOMAIN_LOCATION_INITIAL_M);
        for expected in 1..=TWO_HOUR_STEPS {
            integral.advance().unwrap();
            assert_eq!(integral.completed_steps(), expected);
            assert_eq!(integral.x_m().to_bits(), 0_u64);
            assert_eq!(integral.y_m().to_bits(), 0_u64);
        }
    }

    #[test]
    fn a_two_hour_run_accumulates_umove_times_t_with_no_drift() {
        // CM1's own supercell deck velocity (run/namelist.input umove=12.5,
        // vmove=3.0).  Exact equality, not closeness: the module header
        // proves every partial sum is representable, and this asserts it.
        let moving = frame(12.5_f32, 3.0_f32);
        let mut integral = DomainLocationIntegrator::new(DT, moving).unwrap();
        for step in 1..=TWO_HOUR_STEPS {
            integral.advance().unwrap();
            let seconds = step as f64 * DT;
            assert_eq!(integral.x_m(), seconds * f64::from(moving.u_move_m_per_s()));
            assert_eq!(integral.y_m(), seconds * f64::from(moving.v_move_m_per_s()));
            assert_eq!(integral.model_time_s(), seconds);
        }
        // Two hours at 12.5 m/s is 90 km, to the metre.
        assert_eq!(integral.x_m(), 90_000.0);
        assert_eq!(integral.y_m(), 21_600.0);
    }

    #[test]
    fn exactness_holds_for_every_awkward_binary32_frame_speed() {
        // The proof is about binary32 significands, not about round numbers,
        // so it is asserted on speeds whose decimal forms are not exact.
        for speed in [
            0.1_f32,
            -0.1_f32,
            1.0_f32 / 3.0_f32,
            12.345_678_f32,
            -7.25_f32,
            MAX_FRAME_SPEED_M_PER_S_FOR_TEST,
            f32::MIN_POSITIVE,
        ] {
            let moving = frame(speed, -speed);
            let mut integral = DomainLocationIntegrator::new(DT, moving).unwrap();
            integral.advance_steps(TWO_HOUR_STEPS).unwrap();
            let seconds = TWO_HOUR_STEPS as f64 * DT;
            assert_eq!(
                integral.x_m(),
                seconds * f64::from(speed),
                "drifted at {speed} m/s"
            );
            assert_eq!(integral.y_m(), seconds * f64::from(-speed));
        }
    }

    const MAX_FRAME_SPEED_M_PER_S_FOR_TEST: f32 =
        crate::idealized_initialization::MAX_FRAME_SPEED_M_PER_S;

    #[test]
    fn a_non_power_of_two_dt_is_refused_the_moment_it_drifts() {
        // CM1 would silently drift here.  This lane refuses, because the
        // drifted value is published as the ground position of every frame.
        let mut integral = DomainLocationIntegrator::new(0.3, frame(0.1_f32, 0.0_f32)).unwrap();
        let mut error = None;
        for _ in 0..TWO_HOUR_STEPS {
            if let Err(problem) = integral.advance() {
                error = Some(problem.to_string());
                break;
            }
        }
        let message = error.expect("a non-power-of-two dt drifts inside two hours");
        assert!(message.contains("domainlocx"), "{message}");
        assert!(message.contains("drifts away from umove*t"), "{message}");
    }

    #[test]
    fn a_restart_continues_the_integral_rather_than_restarting_it() {
        // restart_write.F:399-400 then restart_read.F:419-420.  A run that
        // stops at 7_200 steps and resumes must land exactly where an
        // uninterrupted run lands.
        let moving = frame(12.5_f32, 3.0_f32);
        let mut whole = DomainLocationIntegrator::new(DT, moving).unwrap();
        whole.advance_steps(TWO_HOUR_STEPS).unwrap();

        let mut first = DomainLocationIntegrator::new(DT, moving).unwrap();
        first.advance_steps(TWO_HOUR_STEPS / 2).unwrap();
        let clock = ForecastClock::new(first.completed_steps(), first.model_time_s(), DT).unwrap();
        let persisted = first.scheduler_scalars().unwrap();

        let mut resumed = DomainLocationIntegrator::from_frame(clock, &persisted).unwrap();
        assert_eq!(resumed.x_m(), first.x_m());
        assert_eq!(resumed.completed_steps(), first.completed_steps());
        resumed.advance_steps(TWO_HOUR_STEPS / 2).unwrap();

        assert_eq!(resumed.x_m().to_bits(), whole.x_m().to_bits());
        assert_eq!(resumed.y_m().to_bits(), whole.y_m().to_bits());
        assert_eq!(resumed.completed_steps(), whole.completed_steps());
        // The defect this guards: an accumulator that reset on restart would
        // be exactly half the correct distance out.
        assert_ne!(resumed.x_m(), whole.x_m() / 2.0);
    }

    #[test]
    fn a_frame_missing_the_pair_is_refused_rather_than_restarted_at_zero() {
        let clock = ForecastClock::new(7_200, 3_600.0, DT).unwrap();
        let partial = vec![
            SchedulerScalar::new(MOVE_SPEED_X_NAME, SchedulerScalarValue::F64(12.5)).unwrap(),
            SchedulerScalar::new(MOVE_SPEED_Y_NAME, SchedulerScalarValue::F64(3.0)).unwrap(),
        ];
        let message = DomainLocationIntegrator::from_frame(clock, &partial)
            .unwrap_err()
            .to_string();
        assert!(message.contains("carries no domainlocx"), "{message}");
        assert!(
            message.contains("restart the integral at zero"),
            "{message}"
        );
    }

    #[test]
    fn a_persisted_value_from_another_frame_is_refused() {
        // The 90 km a 12.5 m/s frame reaches in two hours, offered back under
        // a 3.0 m/s frame: not the same integral, and continuing it would
        // put every later frame 87 km from where the carriers actually are.
        let clock = ForecastClock::new(TWO_HOUR_STEPS, TWO_HOUR_STEPS as f64 * DT, DT).unwrap();
        let mismatched = vec![
            SchedulerScalar::new(DOMAIN_LOCATION_X_NAME, SchedulerScalarValue::F64(90_000.0))
                .unwrap(),
            SchedulerScalar::new(DOMAIN_LOCATION_Y_NAME, SchedulerScalarValue::F64(21_600.0))
                .unwrap(),
            SchedulerScalar::new(MOVE_SPEED_X_NAME, SchedulerScalarValue::F64(3.0)).unwrap(),
            SchedulerScalar::new(MOVE_SPEED_Y_NAME, SchedulerScalarValue::F64(3.0)).unwrap(),
        ];
        let message = DomainLocationIntegrator::from_frame(clock, &mismatched)
            .unwrap_err()
            .to_string();
        assert!(message.contains("resumed domainlocx"), "{message}");
    }

    #[test]
    fn a_frame_speed_that_is_not_a_cm1_real_is_refused() {
        let clock = ForecastClock::new(0, 0.0, DT).unwrap();
        let widened = vec![
            SchedulerScalar::new(DOMAIN_LOCATION_X_NAME, SchedulerScalarValue::F64(0.0)).unwrap(),
            SchedulerScalar::new(DOMAIN_LOCATION_Y_NAME, SchedulerScalarValue::F64(0.0)).unwrap(),
            // Representable in binary64, not in binary32.
            SchedulerScalar::new(
                MOVE_SPEED_X_NAME,
                SchedulerScalarValue::F64(12.500_000_000_000_002),
            )
            .unwrap(),
            SchedulerScalar::new(MOVE_SPEED_Y_NAME, SchedulerScalarValue::F64(3.0)).unwrap(),
        ];
        let message = DomainLocationIntegrator::from_frame(clock, &widened)
            .unwrap_err()
            .to_string();
        assert!(message.contains("not a CM1 `real`"), "{message}");
    }

    #[test]
    fn a_nonpositive_dt_and_a_nonfinite_resume_are_refused_by_name() {
        let fixed = TranslatingFrame::fixed();
        let message = DomainLocationIntegrator::new(0.0, fixed)
            .unwrap_err()
            .to_string();
        assert!(message.contains("domain-location dt is 0.0"), "{message}");
        let message = DomainLocationIntegrator::resume(DT, fixed, 0, f64::NAN, 0.0)
            .unwrap_err()
            .to_string();
        assert!(message.contains("resumed domainlocx"), "{message}");
        assert!(message.contains("not finite"), "{message}");
    }

    #[test]
    fn the_pair_survives_the_cm1fhs1_and_cm1fcp1_envelopes_it_is_written_into() {
        // restart_write.F:399-400 puts the pair in CM1's restart file and
        // restart_read.F:419-420 reads it back.  This lane's persisted
        // artifacts are the forecast envelopes, so the same round trip is
        // asserted through their actual bytes rather than through a struct.
        use crate::forecast::{
            read_forecast_checkpoint, read_forecast_history, write_forecast_checkpoint_new,
            write_forecast_history_new, ForecastArray, ForecastArraySpec, ForecastContract,
            ForecastFrame,
        };

        let moving = frame(12.5_f32, 3.0_f32);
        let mut integral = DomainLocationIntegrator::new(DT, moving).unwrap();
        integral.advance_steps(TWO_HOUR_STEPS).unwrap();

        let contract = ForecastContract::new(
            "gm1.domain-location-round-trip/v1",
            "11".repeat(32),
            "22".repeat(32),
            vec![ForecastArraySpec::new(
                "th3d",
                "current potential-temperature perturbation",
                "K",
                "scalar",
                vec![1, 2],
                vec![2],
            )
            .unwrap()],
            DomainLocationIntegrator::scheduler_scalar_specs().unwrap(),
            Vec::new(),
        )
        .unwrap();
        let clock =
            ForecastClock::new(integral.completed_steps(), integral.model_time_s(), DT).unwrap();
        let published = ForecastFrame::new(
            clock,
            vec![ForecastArray::new("th3d", vec![0.5, -0.25])],
            integral.scheduler_scalars().unwrap(),
            Vec::new(),
        );

        let directory = std::env::temp_dir().join(format!(
            "gm1-domain-location-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        for label in ["history.cm1fhs", "checkpoint.cm1fcp"] {
            let path = directory.join(label);
            let is_history = label.ends_with("cm1fhs");
            if is_history {
                write_forecast_history_new(&contract, &published, &path).unwrap();
            } else {
                write_forecast_checkpoint_new(&contract, &published, &path).unwrap();
            }
            let restored = if is_history {
                read_forecast_history(&contract, &path).unwrap()
            } else {
                read_forecast_checkpoint(&contract, &path).unwrap()
            };
            let mut resumed = DomainLocationIntegrator::from_frame(
                restored.frame().clock(),
                restored.frame().scheduler_scalars(),
            )
            .unwrap();
            // Bit-exact, not close: cm1.F:266 is double precision and the
            // envelope stores the bits.
            assert_eq!(resumed.x_m().to_bits(), integral.x_m().to_bits(), "{label}");
            assert_eq!(resumed.y_m().to_bits(), integral.y_m().to_bits(), "{label}");
            assert_eq!(resumed.completed_steps(), TWO_HOUR_STEPS, "{label}");
            assert_eq!(resumed.frame(), moving, "{label}");
            // And it continues rather than restarting.
            resumed.advance().unwrap();
            assert_eq!(resumed.x_m(), integral.x_m() + integral.x_increment_m());
        }
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn the_published_scalar_set_is_the_cm1_names_and_contract_types() {
        let integral = DomainLocationIntegrator::new(DT, frame(12.5, 3.0)).unwrap();
        let specs = DomainLocationIntegrator::scheduler_scalar_specs().unwrap();
        let scalars = integral.scheduler_scalars().unwrap();
        assert_eq!(
            specs
                .iter()
                .map(SchedulerScalarSpec::name)
                .collect::<Vec<_>>(),
            vec!["domainlocx", "domainlocy", "umove", "vmove"],
        );
        assert_eq!(
            scalars
                .iter()
                .map(SchedulerScalar::name)
                .collect::<Vec<_>>(),
            vec!["domainlocx", "domainlocy", "umove", "vmove"],
        );
        for scalar in &scalars {
            assert_eq!(
                scalar.value().scalar_type().as_str(),
                "f64-bits",
                "{}",
                scalar.name()
            );
        }
        assert_eq!(DOMAIN_LOCATION_UNITS, "m");
        assert_eq!(MOVE_SPEED_UNITS, "m/s");
    }
}
