//! First-class CM1 r21.1 idealized `isnd=7` / `iinit=1` initialization.
//!
//! This module is intentionally separate from the real-product admission
//! path.  It preserves the caller's complete `input_sounding` byte stream,
//! applies the CM1 single-sounding hydrostatic construction in source order,
//! and optionally builds the official warm-bubble theta perturbation.  Its
//! outputs are idealized initial conditions, never observations, forecasts,
//! tornado detections, or validated tornado simulations.

use crate::initialization::{MoistInitializationAdmission, MoistInitializationConfig};
use crate::terrain::{
    compute_flat_static_levels, compute_flat_uniform_static, DeclaredVerticalCoordinate,
    HorizontalCrop, TerrainStaticConfig, TerrainStaticFrame,
};
use crate::base_state_wind::{
    require_implemented_iwnd, HurricaneBaseWind,
};
pub use crate::initial_perturbation::{
    apply_base_dependent_perturbation, apply_maintain_relative_humidity,
    apply_random_perturbation, build_theta_perturbation, iinit_refusal, random_perturbation_law,
    Cm1RandomStream, InitialPerturbationControls, WarmBubbleControls, IMPLEMENTED_IINIT,
    RANDOM_STREAM_GENERATOR,
};
use crate::terrain_admission::TERRAIN_OPEN_CARRIER_NAMES;
use crate::terrain_admission_science::{
    compute_surface_halos, pack_cold_start_mutable, PerturbationPolicy, RecoveryMetric,
    SurfaceHaloFrame, TerrainBaseFrame,
};
use crate::{sha256, Result, StateError};

pub const IDEALIZED_INITIALIZATION_SCHEMA: &str = "gm1.cm1-r21.1-isnd7-iinit1-initialization/v1";
pub const IDEALIZED_INITIALIZATION_STATUS: &str = "ADMITTED_IDEALIZED_CM1_ISND7_IINIT1";
pub const IDEALIZED_CLASSIFICATION: &str =
    "idealized initial condition; never an observed or forecast meteorological product";
pub const IDEALIZED_CLAIM_LIMIT: &str = "CM1 r21.1 isnd=7 single-sounding hydrostatic base and selectable iinit=1 theta perturbation only; no dynamics, physics trajectory, forecast, tornado, or model-validation claim";

pub const CM1_R21_BASE_F_SHA256: &str =
    "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef";
pub const CM1_R21_INIT3D_F_SHA256: &str =
    "9c45c0982ba194ea6ea74afd6a2516445cdd011fc90902091d089f4cb92dfd28";
pub const CM1_R21_PARAM_F_SHA256: &str =
    "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349";

pub(crate) const G: f32 = 9.81_f32;
pub(crate) const RD: f32 = 287.04_f32;
pub(crate) const RV: f32 = 461.5_f32;
pub(crate) const CP: f32 = 1005.7_f32;
pub(crate) const P00: f32 = 100_000.0_f32;
const MAX_SOUNDING_BYTES: usize = 64 * 1024 * 1024;
const MAX_PROFILE_ROWS: usize = 1_000_000;
const NGXY: usize = 3;

pub const IDEALIZED_MOIST_CARRIER_NAMES: [&str; 4] = ["qa", "q3d", "qten", "qten1"];
pub const IDEALIZED_BASE_WIND_CARRIER_NAMES: [&str; 2] = ["u0", "v0"];

pub(crate) fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

pub(crate) fn checked_product(shape: [usize; 3], label: &str) -> Result<usize> {
    shape.into_iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(extent)
            .ok_or_else(|| invalid(format!("{label} shape product overflows")))
    })
}

pub(crate) fn require_finite(label: &str, value: f32) -> Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(format!("{label} is not finite")))
    }
}

fn parse_word(token: &str, line: usize, column: usize) -> Result<f32> {
    // CM1's list-directed reader accepts the Fortran D exponent.  Parsing is
    // normalized only for conversion; the complete original bytes remain in
    // `InputSounding::source_bytes` and define its provenance hash.
    let normalized = token.replace('D', "E").replace('d', "e");
    let value = normalized.parse::<f32>().map_err(|error| {
        invalid(format!(
            "input_sounding line {line} column {column} is not a default-real word: {error}"
        ))
    })?;
    require_finite(
        &format!("input_sounding line {line} column {column}"),
        value,
    )
}

/// Which source-audited CM1 base-state construction one selection names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdealizedBaseStateLane {
    /// `base.F:495-786`: thermodynamics and winds both read from the caller's
    /// `input_sounding` bytes.  `param.F:836-852` forces `iwnd=0` here.
    Isnd7ExternalSounding,
    /// `base.F:363-450` under any `base.F:2286-2640` wind branch: the
    /// Weisman-Klemp analytic sounding.  No external bytes.
    Isnd5WeismanKlempIwnd2,
    /// `base.F:495-607` with `isnd=17`: the same external file supplies the
    /// thermodynamic profile, its wind columns are ignored, and the base-state
    /// wind comes from the `iwnd` branch instead (`param.F:849-851` names this
    /// as the way to pair a file sounding with an analytic hodograph).
    Isnd17ExternalThermodynamics,
    /// Every other analytic `base.F` arm (`crate::analytic_soundings`):
    /// `isnd` in 1, 2, 3, 4, 8-14, 18-23, under any `base.F:2286-2640` wind
    /// branch.  No external bytes.
    AnalyticBaseF,
}

/// `base.F`'s own words the analytic arms read beyond the namelist
/// selectors: the moist benchmark's `thec_mb`/`qt_mb` (`base.F:267-268`,
/// also `init3d.F:625-645`'s `iinit=4`) and `tsk0` (`isnd=11`, `21`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalyticBaseWords {
    pub thec_mb_k: f32,
    pub qt_mb: f32,
    pub tsk0_k: f32,
    /// CM1's `eqtset` after `param.F:936-950` (1 for `ptype` 7, GM1's WSM6;
    /// 2 for `ptype` 5, Morrison), read by `isnd = 10` only.
    pub eqtset: i32,
}

impl Default for AnalyticBaseWords {
    fn default() -> Self {
        Self {
            thec_mb_k: crate::analytic_soundings::CM1_THEC_MB_K,
            qt_mb: crate::analytic_soundings::CM1_QT_MB,
            tsk0_k: crate::analytic_soundings::CM1_TSK0_K,
            eqtset: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IdealizedInitializationSelection {
    lane: IdealizedBaseStateLane,
    requested_isnd: i32,
    requested_iwnd: i32,
    effective_iwnd: i32,
    requested_iinit: i32,
    effective_iinit: i32,
    irst: i32,
    imoist: i32,
    testcase: i32,
    ibalance: i32,
    irandp: i32,
    random_seed: u64,
    /// `true`: CM1's own random numbers (libgfortran's stream as
    /// init3d.F:427-440 seeds it); `false`: GM1's seeded splitmix64 stream.
    cm1_random: bool,
    hurricane: HurricaneBaseWind,
    /// What init3d.F:697-1256's vortex (iinit=7) reads beyond the grid: CM1's
    /// fcor (param.F:1985 zeroes it when icor=0), the periodic images and
    /// param.F's iice.
    vortex_environment: VortexEnvironment,
    analytic_words: AnalyticBaseWords,
    /// `param.F:5178-5205`'s imirror/jmirror, read by `ibalance = 2` only.
    elliptic_mirror: [bool; 2],
}

/// See [`IdealizedInitializationSelection::with_vortex_environment`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VortexEnvironment {
    pub fcor: f32,
    pub periodic_x: bool,
    pub periodic_y: bool,
    pub iice: bool,
}

impl Default for VortexEnvironment {
    fn default() -> Self {
        // run/namelist.input's fcor, open sides, and an ice-bearing scheme.
        Self { fcor: 5.0e-5, periodic_x: false, periodic_y: false, iice: true }
    }
}

impl IdealizedInitializationSelection {
    /// Admit only the source-audited CM1 `(isnd,iwnd)` lanes ported here.
    ///
    /// `param.F:836-852` forces `iwnd=0` for `isnd=7`, and
    /// `param.F:891-903` forces `iinit=0` on restart.  Both adjustments are
    /// represented explicitly rather than silently changing caller inputs.
    /// Every other `(isnd,iwnd)` pair is refused by value.
    pub fn new(isnd: i32, iwnd: i32, iinit: i32, irst: i32, imoist: i32) -> Result<Self> {
        let (lane, effective_iwnd) = match (isnd, iwnd) {
            (7, wind) if wind >= 0 => (IdealizedBaseStateLane::Isnd7ExternalSounding, 0),
            (5, wind) => {
                require_implemented_iwnd(wind)?;
                (IdealizedBaseStateLane::Isnd5WeismanKlempIwnd2, wind)
            }
            (17, wind) => {
                require_implemented_iwnd(wind)?;
                (IdealizedBaseStateLane::Isnd17ExternalThermodynamics, wind)
            }
            (other, wind)
                if crate::analytic_soundings::TRANSCRIBED_ANALYTIC_ISND.contains(&other) =>
            {
                require_implemented_iwnd(wind)?;
                (IdealizedBaseStateLane::AnalyticBaseF, wind)
            }
            (7, wind) => return Err(invalid(format!("CM1 iwnd must be nonnegative; got {wind}"))),
            (6, _) => {
                return Err(invalid(
                    "base.F:455-460: isnd=6 stops CM1 r21.1 ('isnd = 6 code has been removed'); \
                     there is no base state to build",
                ))
            }
            (15, _) => {
                return Err(invalid(
                    "base.F:1491-1510 runs isnd=15 (DYCOMS-II) only with testcase=4 or 5, and \
                     constants.F:78-98 then changes rd, cp and xlv for the whole model; GM1's \
                     kernels and ArWen's physics are built with constants.F's defaults, so the \
                     base state would be integrated by other thermodynamics than built it",
                ))
            }
            (other, wind) => {
                return Err(invalid(format!(
                    "base.F:134-2036 has sounding branches isnd in {{1,2,3,4,5,7,8,9,10,11,12,13,\
                     14,15,17,18,19,20,21,22,23}} (6 stops CM1); got isnd={other}, iwnd={wind}. \
                     base.F's chain has no else arm, so an unlisted value would leave the base \
                     state at its allocation zero"
                )))
            }
        };
        if !IMPLEMENTED_IINIT.contains(&iinit) {
            return Err(invalid(
                iinit_refusal(iinit).unwrap_or_else(|| format!("unimplemented iinit={iinit}")),
            ));
        }
        if !matches!(irst, 0 | 1) {
            return Err(invalid(format!(
                "source-audited restart selector admits irst=0 or 1 only; got {irst}"
            )));
        }
        if !matches!(imoist, 0 | 1) {
            return Err(invalid(format!(
                "README.namelist:308 admits imoist=0 or 1; got {imoist}. Any other value leaves \
                 base.F:389-397's rh0 and base.F:592-609's qvsnd undefined, so the emitted base \
                 state would carry no defined moisture at all"
            )));
        }
        Ok(Self {
            lane,
            requested_isnd: isnd,
            requested_iwnd: iwnd,
            effective_iwnd,
            requested_iinit: iinit,
            effective_iinit: if irst == 1 { 0 } else { iinit },
            irst,
            imoist,
            testcase: 0,
            // run/namelist.input:80 and all 23 shipped decks: ibalance=0.
            ibalance: 0,
            irandp: 0,
            random_seed: 0,
            cm1_random: false,
            vortex_environment: VortexEnvironment::default(),
            analytic_words: AnalyticBaseWords::default(),
            // run/namelist.input's four open sides.
            elliptic_mirror: [true, true],
            hurricane: HurricaneBaseWind {
                gradient_wind_m_per_s: 40.0,
                rotate_deg: 0.0,
            },
        })
    }

    /// `run/namelist.input:29`: the pre-configured test case, which selects
    /// `base.F:2467-2510`'s constant wind and `init3d.F:1459-1620`'s random
    /// amplitude and depth limit.
    pub fn with_testcase(mut self, testcase: i32) -> Result<Self> {
        if !(0..=15).contains(&testcase) {
            return Err(invalid(format!(
                "param.F reads testcase in 0..=15 (README.namelist:216-232); got {testcase}. An \
                 unlisted value would fall through base.F:2499's else arm and init3d.F's random \
                 chain and silently select the default deck under another name"
            )));
        }
        self.testcase = testcase;
        Ok(self)
    }

    /// `init3d.F:1743-1917`: the initial 3-D pressure balance assumption.
    /// 0 leaves `ppi` at zero, 1 integrates hydrostatic balance top down, 2
    /// solves for the anelastic buoyancy pressure
    /// (`crate::anelastic_balance`).
    pub fn with_ibalance(mut self, ibalance: i32) -> Result<Self> {
        match ibalance {
            0..=2 => {
                self.ibalance = ibalance;
                Ok(self)
            }
            other => Err(invalid(format!(
                "README.namelist:697-705 admits ibalance in {{0,1,2}}; got {other}. init3d.F's \
                 chain has no else arm, so the initial pressure perturbation would stay at \
                 exactly zero while the receipt named a balance assumption"
            ))),
        }
    }

    /// `param.F:5178-5205`: whether `poiss` mirrors x and y (an open or
    /// free-slip side on that axis), for `ibalance = 2`.
    pub fn with_elliptic_mirror(mut self, imirror: bool, jmirror: bool) -> Self {
        self.elliptic_mirror = [imirror, jmirror];
        self
    }

    pub fn elliptic_mirror(self) -> [bool; 2] {
        self.elliptic_mirror
    }

    /// `init3d.F:1446-1600`: seeded random theta perturbations.
    ///
    /// The seed is required.  `init3d.F:416-421` offers CM1 a wall-clock
    /// reseed (`use_truly_random_pert`); a run seeded from the clock cannot be
    /// reproduced from its own receipt, so it is refused here rather than
    /// offered.
    /// Which stream iinit=8 and irandp draw from: 0 is GM1's own seeded
    /// stream, 1 is CM1's (`Cm1RandomStream::cm1_gfortran`), whose seed is
    /// CM1's fixed init3d.F:427-440 scramble, so no seed may be declared.
    pub fn with_random_generator(mut self, generator: i32) -> Result<Self> {
        match generator {
            0 => self.cm1_random = false,
            1 => self.cm1_random = true,
            other => {
                return Err(invalid(format!(
                    "random_generator is 0 (GM1's seeded stream) or 1 (CM1's own gfortran \
                     stream); got {other}"
                )))
            }
        }
        Ok(self)
    }

    pub fn cm1_random(self) -> bool {
        self.cm1_random
    }

    /// init3d.F:697-1256 reads `fcor` for the gradient wind, the lateral
    /// boundary flags for its periodic images, and `iice` for the saturation
    /// it holds the vortex's relative humidity to.
    pub fn with_vortex_environment(mut self, environment: VortexEnvironment) -> Result<Self> {
        if !environment.fcor.is_finite() {
            return Err(invalid("the iinit=7 vortex needs a finite fcor"));
        }
        self.vortex_environment = environment;
        Ok(self)
    }

    pub fn vortex_environment(self) -> VortexEnvironment {
        self.vortex_environment
    }

    /// `base.F:267-268`'s `thec_mb`/`qt_mb` and the namelist `tsk0`.
    pub fn with_analytic_words(mut self, words: AnalyticBaseWords) -> Result<Self> {
        for (label, value) in [
            ("thec_mb", words.thec_mb_k),
            ("qt_mb", words.qt_mb),
            ("tsk0", words.tsk0_k),
        ] {
            require_finite(label, value)?;
        }
        if words.thec_mb_k <= 0.0 || words.tsk0_k <= 100.0 {
            return Err(invalid(format!(
                "thec_mb ({}) must be a positive temperature and tsk0 ({}) above 100 K",
                words.thec_mb_k, words.tsk0_k
            )));
        }
        if !matches!(words.eqtset, 1 | 2) {
            return Err(invalid(format!(
                "README.namelist's eqtset is 1 or 2; got {}",
                words.eqtset
            )));
        }
        if !(words.qt_mb > 0.0 && words.qt_mb < 1.0) {
            return Err(invalid(format!(
                "qt_mb is a total water mixing ratio (kg/kg) in (0, 1); got {}",
                words.qt_mb
            )));
        }
        self.analytic_words = words;
        Ok(self)
    }

    pub fn analytic_words(self) -> AnalyticBaseWords {
        self.analytic_words
    }

    /// `init3d.F:625-645` reads `qt_mb` from the `input` module, which
    /// `param.F:6336` zeroes and only `base.F:268` (`isnd=4`) sets: with any
    /// other sounding the moist-benchmark bubble runs with `qt_mb = 0`.
    pub fn effective_qt_mb(self) -> f32 {
        if self.requested_isnd == 4 {
            self.analytic_words.qt_mb
        } else {
            0.0_f32
        }
    }

    pub fn with_random_perturbation(mut self, irandp: i32, random_seed: u64) -> Result<Self> {
        if self.cm1_random && random_seed != 0 {
            return Err(invalid(
                "CM1's own random stream is seeded by init3d.F:427-440's fixed scramble; a \
                 declared seed would be recorded and never read",
            ));
        }
        match irandp {
            0 => {
                self.irandp = 0;
                self.random_seed = 0;
                Ok(self)
            }
            1 if self.cm1_random => {
                self.irandp = 1;
                self.random_seed = 0;
                Ok(self)
            }
            1 => {
                if random_seed == 0 {
                    return Err(invalid(
                        "irandp=1 requires an explicit nonzero seed. init3d.F:416-443 seeds one \
                         generator for the whole run; without a recorded seed the perturbation \
                         field -- and therefore every storm that grows out of it -- cannot be \
                         reproduced from the receipt that describes it",
                    ));
                }
                self.irandp = 1;
                self.random_seed = random_seed;
                Ok(self)
            }
            other => Err(invalid(format!(
                "README.namelist:692-695 admits irandp=0 or 1; got {other}"
            ))),
        }
    }

    /// `run/namelist.input:403,406`: the `iwnd=8` gradient wind and rotation.
    pub fn with_hurricane_wind(mut self, hurricane: HurricaneBaseWind) -> Self {
        self.hurricane = hurricane;
        self
    }

    pub fn testcase(self) -> i32 {
        self.testcase
    }

    pub fn ibalance(self) -> i32 {
        self.ibalance
    }

    pub fn irandp(self) -> i32 {
        self.irandp
    }

    pub fn random_seed(self) -> u64 {
        self.random_seed
    }

    pub fn hurricane(self) -> HurricaneBaseWind {
        self.hurricane
    }

    /// `base.F:2255-2262`: only `isnd=7` carries its own winds.
    pub fn builds_analytic_wind(self) -> bool {
        self.lane != IdealizedBaseStateLane::Isnd7ExternalSounding
    }

    pub fn lane(self) -> IdealizedBaseStateLane {
        self.lane
    }

    pub fn requested_isnd(self) -> i32 {
        self.requested_isnd
    }

    pub fn requested_iwnd(self) -> i32 {
        self.requested_iwnd
    }

    pub fn effective_iwnd(self) -> i32 {
        self.effective_iwnd
    }

    pub fn requested_iinit(self) -> i32 {
        self.requested_iinit
    }

    pub fn effective_iinit(self) -> i32 {
        self.effective_iinit
    }

    pub fn irst(self) -> i32 {
        self.irst
    }

    pub fn imoist(self) -> i32 {
        self.imoist
    }

    pub fn restart_disabled_perturbation(self) -> bool {
        self.irst == 1 && self.requested_iinit != 0 && self.effective_iinit == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SuppliedSurface {
    pub pressure_mb: f32,
    pub potential_temperature_k: f32,
    pub water_vapor_g_per_kg: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SuppliedSoundingLevel {
    pub height_m: f32,
    pub potential_temperature_k: f32,
    pub water_vapor_g_per_kg: f32,
    pub u_m_per_s: f32,
    pub v_m_per_s: f32,
}

#[derive(Clone, Debug)]
pub struct InputSounding {
    source_bytes: Vec<u8>,
    source_sha256: String,
    surface: SuppliedSurface,
    levels: Vec<SuppliedSoundingLevel>,
}

impl InputSounding {
    /// Parse the exact CM1 `input_sounding` envelope (`base.F:463-607`).
    ///
    /// The supported envelope is deliberately stricter than Fortran
    /// list-directed I/O: one nonempty header with exactly three words,
    /// followed by nonempty rows with exactly five words.  Ambiguous comments,
    /// commas, blank records, and trailing tokens fail closed.  The supplied
    /// bytes are retained unchanged.
    pub fn parse(source_bytes: &[u8]) -> Result<Self> {
        if source_bytes.is_empty() || source_bytes.len() > MAX_SOUNDING_BYTES {
            return Err(invalid(format!(
                "input_sounding byte length must be in 1..={MAX_SOUNDING_BYTES}; got {}",
                source_bytes.len()
            )));
        }
        let text = std::str::from_utf8(source_bytes)
            .map_err(|error| invalid(format!("input_sounding is not UTF-8 text: {error}")))?;
        let mut records = Vec::new();
        for (zero_line, line) in text.lines().enumerate() {
            let line_number = zero_line + 1;
            if line.trim().is_empty() {
                return Err(invalid(format!(
                    "input_sounding line {line_number} is blank; the strict supported envelope has no blank records"
                )));
            }
            records.push((line_number, line));
        }
        if records.is_empty() {
            return Err(invalid("input_sounding has no records"));
        }

        let (header_line, header_text) = records[0];
        let header_words: Vec<&str> = header_text.split_whitespace().collect();
        if header_words.len() != 3 {
            return Err(invalid(format!(
                "input_sounding header line {header_line} requires exactly 3 words; got {}",
                header_words.len()
            )));
        }
        let surface = SuppliedSurface {
            pressure_mb: parse_word(header_words[0], header_line, 1)?,
            potential_temperature_k: parse_word(header_words[1], header_line, 2)?,
            water_vapor_g_per_kg: parse_word(header_words[2], header_line, 3)?,
        };
        if surface.pressure_mb <= 0.0
            || surface.potential_temperature_k <= 0.0
            || surface.water_vapor_g_per_kg < 0.0
        {
            return Err(invalid(
                "input_sounding surface pressure/theta must be positive and qv must be nonnegative",
            ));
        }

        let profile_count = records.len() - 1;
        if !(2..=MAX_PROFILE_ROWS).contains(&profile_count) {
            return Err(invalid(format!(
                "input_sounding requires 2..={MAX_PROFILE_ROWS} above-surface rows; got {profile_count}"
            )));
        }
        let mut levels: Vec<SuppliedSoundingLevel> = Vec::with_capacity(profile_count);
        for (row_index, (line_number, line)) in records.into_iter().skip(1).enumerate() {
            let words: Vec<&str> = line.split_whitespace().collect();
            if words.len() != 5 {
                return Err(invalid(format!(
                    "input_sounding line {line_number} requires exactly 5 words; got {}",
                    words.len()
                )));
            }
            let level = SuppliedSoundingLevel {
                height_m: parse_word(words[0], line_number, 1)?,
                potential_temperature_k: parse_word(words[1], line_number, 2)?,
                water_vapor_g_per_kg: parse_word(words[2], line_number, 3)?,
                u_m_per_s: parse_word(words[3], line_number, 4)?,
                v_m_per_s: parse_word(words[4], line_number, 5)?,
            };
            let previous_height = if row_index == 0 {
                0.0_f32
            } else {
                levels[row_index - 1].height_m
            };
            if level.height_m <= previous_height
                || level.potential_temperature_k <= 0.0
                || level.water_vapor_g_per_kg < 0.0
            {
                return Err(invalid(format!(
                    "input_sounding line {line_number} requires strictly increasing positive height, positive theta, and nonnegative qv"
                )));
            }
            levels.push(level);
        }

        Ok(Self {
            source_bytes: source_bytes.to_vec(),
            source_sha256: sha256::digest_hex(source_bytes),
            surface,
            levels,
        })
    }

    pub fn source_bytes(&self) -> &[u8] {
        &self.source_bytes
    }

    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    pub fn surface(&self) -> SuppliedSurface {
        self.surface
    }

    pub fn levels(&self) -> &[SuppliedSoundingLevel] {
        &self.levels
    }

    pub fn derive(&self, selection: IdealizedInitializationSelection) -> Result<Cm1Isnd7Sounding> {
        if !matches!(selection.requested_isnd, 7 | 17) {
            return Err(invalid(format!(
                "input_sounding derivation is base.F:495-607's isnd=7/isnd=17 branch; got \
                 isnd={}",
                selection.requested_isnd
            )));
        }
        Cm1Isnd7Sounding::from_input(self, selection.imoist)
    }
}

#[derive(Clone, Debug)]
pub struct Cm1Isnd7Sounding {
    z_m: Vec<f32>,
    theta_k: Vec<f32>,
    qv_kg_per_kg: Vec<f32>,
    u_m_per_s: Vec<f32>,
    v_m_per_s: Vec<f32>,
    pressure_pa: Vec<f32>,
    temperature_k: Vec<f32>,
    relative_humidity: Vec<f32>,
    pi_surface: f32,
    theta_v_surface: f32,
}

impl Cm1Isnd7Sounding {
    fn from_input(input: &InputSounding, imoist: i32) -> Result<Self> {
        let count = input.levels.len() + 1;
        let mut z_m = Vec::with_capacity(count);
        let mut theta_k = Vec::with_capacity(count);
        let mut qv_kg_per_kg = Vec::with_capacity(count);
        let mut u_m_per_s = Vec::with_capacity(count);
        let mut v_m_per_s = Vec::with_capacity(count);

        let pressure_surface_pa = input.surface.pressure_mb * 100.0_f32;
        // base.F:592: `if(imoist.eq.0) qv_sfc = 0.0`, before the g/kg scaling.
        let qv_surface = if imoist == 0 {
            0.0_f32
        } else {
            input.surface.water_vapor_g_per_kg / 1000.0_f32
        };
        let pi_surface = (pressure_surface_pa / P00).powf(RD / CP);
        let reps = RV / RD;
        let theta_v_surface = input.surface.potential_temperature_k * (1.0_f32 + qv_surface * reps)
            / (1.0_f32 + qv_surface);

        let first = input.levels[0];
        let second = input.levels[1];
        let surface_u = first.u_m_per_s
            - first.height_m * (second.u_m_per_s - first.u_m_per_s)
                / (second.height_m - first.height_m);
        let surface_v = first.v_m_per_s
            - first.height_m * (second.v_m_per_s - first.v_m_per_s)
                / (second.height_m - first.height_m);

        z_m.push(0.0_f32);
        theta_k.push(input.surface.potential_temperature_k);
        qv_kg_per_kg.push(qv_surface);
        u_m_per_s.push(surface_u);
        v_m_per_s.push(surface_v);
        for level in &input.levels {
            z_m.push(level.height_m);
            theta_k.push(level.potential_temperature_k);
            // base.F:609: `if(imoist.eq.0) qvsnd = 0.0`.
            qv_kg_per_kg.push(if imoist == 0 {
                0.0_f32
            } else {
                level.water_vapor_g_per_kg / 1000.0_f32
            });
            u_m_per_s.push(level.u_m_per_s);
            v_m_per_s.push(level.v_m_per_s);
        }

        let mut theta_v = vec![0.0_f32; count];
        for k in 0..count {
            theta_v[k] =
                theta_k[k] * (1.0_f32 + reps * qv_kg_per_kg[k]) / (1.0_f32 + qv_kg_per_kg[k]);
        }
        let mut pi = vec![0.0_f32; count];
        pi[0] = pi_surface;
        for k in 1..count {
            pi[k] = pi[k - 1]
                - G * (z_m[k] - z_m[k - 1]) / (CP * 0.5_f32 * (theta_v[k] + theta_v[k - 1]));
        }
        let mut pressure_pa = vec![0.0_f32; count];
        let mut temperature_k = vec![0.0_f32; count];
        let mut relative_humidity = vec![0.0_f32; count];
        for k in 0..count {
            pressure_pa[k] = P00 * pi[k].powf(CP / RD);
            temperature_k[k] = theta_k[k] * pi[k];
            // base.F:660-663's dry-layer floor exists to keep rh0 defined in a
            // moist run.  A dry run (imoist=0) has no vapour at all and must
            // not have any put back, or every "dry" case would carry a thin
            // 0.01% film of moisture the caller did not ask for.
            if imoist == 1 && qv_kg_per_kg[k] < 1.0e-12_f32 {
                qv_kg_per_kg[k] = 0.0001_f32 * rslf(pressure_pa[k], theta_k[k] * pi[k]);
            }
            relative_humidity[k] = if imoist == 0 {
                0.0_f32
            } else {
                qv_kg_per_kg[k] / rslf(pressure_pa[k], temperature_k[k])
            };
            if !pi[k].is_finite()
                || pi[k] <= 0.0
                || !pressure_pa[k].is_finite()
                || pressure_pa[k] <= 0.0
                || !temperature_k[k].is_finite()
                || temperature_k[k] <= 0.0
                || !relative_humidity[k].is_finite()
                || relative_humidity[k] < 0.0
            {
                return Err(invalid(format!(
                    "derived input_sounding state is non-physical at source level {k}"
                )));
            }
        }

        Ok(Self {
            z_m,
            theta_k,
            qv_kg_per_kg,
            u_m_per_s,
            v_m_per_s,
            pressure_pa,
            temperature_k,
            relative_humidity,
            pi_surface,
            theta_v_surface,
        })
    }

    pub fn z_m(&self) -> &[f32] {
        &self.z_m
    }

    pub fn theta_k(&self) -> &[f32] {
        &self.theta_k
    }

    pub fn qv_kg_per_kg(&self) -> &[f32] {
        &self.qv_kg_per_kg
    }

    pub fn u_m_per_s(&self) -> &[f32] {
        &self.u_m_per_s
    }

    pub fn v_m_per_s(&self) -> &[f32] {
        &self.v_m_per_s
    }

    pub fn pressure_pa(&self) -> &[f32] {
        &self.pressure_pa
    }

    pub fn temperature_k(&self) -> &[f32] {
        &self.temperature_k
    }

    pub fn relative_humidity(&self) -> &[f32] {
        &self.relative_humidity
    }

    pub fn pi_surface(&self) -> f32 {
        self.pi_surface
    }

    pub fn theta_v_surface(&self) -> f32 {
        self.theta_v_surface
    }

    /// Interpolate the supplied profile onto one terrain/source column in the
    /// exact `base.F:675-755` order.  `u_heights_m` and `v_heights_m` are the
    /// source face-height averages for the same mass levels; CM1 intentionally
    /// uses the mass-height bracket for all three interpolations.
    pub fn interpolate_column(
        &self,
        mass_heights_m: &[f32],
        u_heights_m: &[f32],
        v_heights_m: &[f32],
        highest_required_mass_height_m: f32,
    ) -> Result<IdealizedBaseColumn> {
        let nz = mass_heights_m.len();
        if nz == 0 || u_heights_m.len() != nz || v_heights_m.len() != nz {
            return Err(invalid(
                "isnd=7 column requires nonempty equal-length mass/u/v height arrays",
            ));
        }
        require_finite(
            "highest required mass height",
            highest_required_mass_height_m,
        )?;
        let sounding_top = *self.z_m.last().expect("isnd=7 sounding is nonempty");
        if sounding_top < highest_required_mass_height_m {
            return Err(invalid(format!(
                "input_sounding top {sounding_top} m does not cover required mass height {highest_required_mass_height_m} m"
            )));
        }
        let mut theta_k = vec![0.0_f32; nz];
        let mut qv_kg_per_kg = vec![0.0_f32; nz];
        let mut u_m_per_s = vec![0.0_f32; nz];
        let mut v_m_per_s = vec![0.0_f32; nz];
        let mut pressure_pa = vec![0.0_f32; nz];
        let mut temperature_k = vec![0.0_f32; nz];
        let mut relative_humidity = vec![0.0_f32; nz];
        let mut theta_v = vec![0.0_f32; nz];
        let reps = RV / RD;

        for k in 0..nz {
            let height = require_finite("mass height", mass_heights_m[k])?;
            let u_height = require_finite("u height", u_heights_m[k])?;
            let v_height = require_finite("v height", v_heights_m[k])?;
            if height <= self.z_m[0] || height > sounding_top {
                return Err(invalid(format!(
                    "mass height {height} at level {k} lies outside the supported (0,{sounding_top}] sounding interval"
                )));
            }
            if k > 0 && height <= mass_heights_m[k - 1] {
                return Err(invalid("isnd=7 mass heights must be strictly increasing"));
            }
            let mut upper = 1usize;
            while self.z_m[upper] < height {
                upper += 1;
            }
            let lower = upper - 1;
            u_m_per_s[k] = interpolate_bracket(
                self.z_m[lower],
                self.z_m[upper],
                self.u_m_per_s[lower],
                self.u_m_per_s[upper],
                u_height,
            );
            v_m_per_s[k] = interpolate_bracket(
                self.z_m[lower],
                self.z_m[upper],
                self.v_m_per_s[lower],
                self.v_m_per_s[upper],
                v_height,
            );
            theta_k[k] = interpolate_bracket(
                self.z_m[lower],
                self.z_m[upper],
                self.theta_k[lower],
                self.theta_k[upper],
                height,
            );
            // The first qv interpolation is an official intermediate which is
            // overwritten below after linearly interpolating RH, p, and T.
            qv_kg_per_kg[k] = interpolate_bracket(
                self.z_m[lower],
                self.z_m[upper],
                self.qv_kg_per_kg[lower],
                self.qv_kg_per_kg[upper],
                height,
            );
            pressure_pa[k] = interpolate_bracket(
                self.z_m[lower],
                self.z_m[upper],
                self.pressure_pa[lower],
                self.pressure_pa[upper],
                height,
            );
            temperature_k[k] = interpolate_bracket(
                self.z_m[lower],
                self.z_m[upper],
                self.temperature_k[lower],
                self.temperature_k[upper],
                height,
            );
            relative_humidity[k] = interpolate_bracket(
                self.z_m[lower],
                self.z_m[upper],
                self.relative_humidity[lower],
                self.relative_humidity[upper],
                height,
            );
        }

        for k in 0..nz {
            qv_kg_per_kg[k] = relative_humidity[k] * rslf(pressure_pa[k], temperature_k[k]);
            theta_v[k] =
                theta_k[k] * (1.0_f32 + reps * qv_kg_per_kg[k]) / (1.0_f32 + qv_kg_per_kg[k]);
        }
        let mut pi = vec![0.0_f32; nz];
        pi[0] = self.pi_surface
            - G * mass_heights_m[0] / (CP * 0.5_f32 * (self.theta_v_surface + theta_v[0]));
        for k in 1..nz {
            pi[k] = pi[k - 1]
                - G * (mass_heights_m[k] - mass_heights_m[k - 1])
                    / (CP * 0.5_f32 * (theta_v[k] + theta_v[k - 1]));
        }
        for k in 0..nz {
            pressure_pa[k] = P00 * pi[k].powf(CP / RD);
            relative_humidity[k] = qv_kg_per_kg[k] / rslf(pressure_pa[k], theta_k[k] * pi[k]);
            for (label, value) in [
                ("theta", theta_k[k]),
                ("qv", qv_kg_per_kg[k]),
                ("u", u_m_per_s[k]),
                ("v", v_m_per_s[k]),
                ("pi", pi[k]),
                ("pressure", pressure_pa[k]),
                ("temperature", temperature_k[k]),
                ("relative humidity", relative_humidity[k]),
            ] {
                if !value.is_finite() {
                    return Err(invalid(format!(
                        "isnd=7 column {label} is not finite at level {k}"
                    )));
                }
            }
            if theta_k[k] <= 0.0
                || qv_kg_per_kg[k] < 0.0
                || pi[k] <= 0.0
                || pressure_pa[k] <= 0.0
                || temperature_k[k] <= 0.0
                || relative_humidity[k] < 0.0
            {
                return Err(invalid(format!(
                    "isnd=7 column is non-physical at level {k}"
                )));
            }
        }

        Ok(IdealizedBaseColumn {
            mass_heights_m: mass_heights_m.to_vec(),
            theta_k,
            qv_kg_per_kg,
            u_m_per_s,
            v_m_per_s,
            pi,
            pressure_pa,
            temperature_k,
            relative_humidity,
        })
    }
}

pub(crate) fn rslf(pressure: f32, temperature: f32) -> f32 {
    let eps = RD / RV;
    let exponent = 17.67_f32 * (temperature - 273.15_f32) / (temperature - 29.65_f32);
    let vapor_pressure = (611.2_f32 * exponent.exp()).min(pressure * 0.5_f32);
    eps * vapor_pressure / (pressure - vapor_pressure)
}

fn interpolate_bracket(z_lower: f32, z_upper: f32, lower: f32, upper: f32, z: f32) -> f32 {
    let fraction = (z - z_lower) / (z_upper - z_lower);
    lower + (upper - lower) * fraction
}

#[derive(Clone, Debug)]
pub struct IdealizedBaseColumn {
    mass_heights_m: Vec<f32>,
    theta_k: Vec<f32>,
    qv_kg_per_kg: Vec<f32>,
    u_m_per_s: Vec<f32>,
    v_m_per_s: Vec<f32>,
    pi: Vec<f32>,
    pressure_pa: Vec<f32>,
    temperature_k: Vec<f32>,
    relative_humidity: Vec<f32>,
}

impl IdealizedBaseColumn {
    pub fn mass_heights_m(&self) -> &[f32] {
        &self.mass_heights_m
    }
    pub fn theta_k(&self) -> &[f32] {
        &self.theta_k
    }
    pub fn qv_kg_per_kg(&self) -> &[f32] {
        &self.qv_kg_per_kg
    }
    pub fn u_m_per_s(&self) -> &[f32] {
        &self.u_m_per_s
    }
    pub fn v_m_per_s(&self) -> &[f32] {
        &self.v_m_per_s
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
    pub fn relative_humidity(&self) -> &[f32] {
        &self.relative_humidity
    }
}


#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WarmBubbleDisposition {
    AppliedIinit1,
    NotRequested,
    DisabledByRestart,
}

#[derive(Clone, Debug)]
pub struct WarmBubbleFrame {
    pub(crate) shape_zyx: [usize; 3],
    pub(crate) theta_perturbation_k: Vec<f32>,
    pub(crate) disposition: WarmBubbleDisposition,
}

impl WarmBubbleFrame {
    pub fn shape_zyx(&self) -> [usize; 3] {
        self.shape_zyx
    }

    pub fn theta_perturbation_k(&self) -> &[f32] {
        &self.theta_perturbation_k
    }

    pub fn disposition(&self) -> WarmBubbleDisposition {
        self.disposition
    }
}

/// The largest frame speed this port admits, in m/s.
///
/// CM1 imposes no bound of its own: `param.F:701-714` only forces the speeds
/// to zero when `imove=0`.  The bound exists here because `base.F:2667`
/// subtracts the constant from every base-state wind word, so an absurd
/// speed is not an absurd number in one place, it is a uniform jet through
/// the whole depth of the declared base state.
pub const MAX_FRAME_SPEED_M_PER_S: f32 = 150.0_f32;

/// CM1 `base.F:2661-2674`: the constant translation of the model frame.
///
/// `imove=1` moves the whole domain at `(umove, vmove)`, and base.F
/// implements exactly that by subtracting the constant from the base-state
/// winds once, at the end of base-state construction, after every `isnd` and
/// `iwnd` branch has run (`base.F:2663-2669`).  `init3d.F:210-219` then
/// copies `u0`/`v0` straight into `ua`/`u3d` and `va`/`v3d`, so the initial
/// condition inherits the same offset from the same subtraction and there is
/// no second one anywhere in the model.  Ground-relative winds are therefore
/// `umove+ua` and `vmove+va` (`init3d.F:32-33`, `README.namelist:772-780`).
///
/// The change of variable is exact only without Coriolis.  CM1's own
/// Coriolis and large-scale pressure-gradient terms are the only places that
/// add the constant back (`solve1.F:605-609`, `solve1.F:623-624`), and CM1's
/// supercell deck runs `icor=0` (`run/namelist.input:59`) precisely so that
/// they do not exist.  GM1's Cartesian momentum tendencies carry no Coriolis
/// term at all, so the same exactness holds here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TranslatingFrame {
    u_move_m_per_s: f32,
    v_move_m_per_s: f32,
}

impl TranslatingFrame {
    /// `imove=0`: the fixed frame `param.F:701-702` forces zero speeds for.
    pub const fn fixed() -> Self {
        Self {
            u_move_m_per_s: 0.0_f32,
            v_move_m_per_s: 0.0_f32,
        }
    }

    /// Admit one frame velocity by value, or refuse it by name.
    pub fn new(u_move_m_per_s: f32, v_move_m_per_s: f32) -> Result<Self> {
        for (name, value) in [("umove", u_move_m_per_s), ("vmove", v_move_m_per_s)] {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "moving-frame {name} is {value}, which is not finite. base.F:2667 \
                     subtracts it from every base-state wind word, so u0/v0 and the \
                     ua/va that init3d.F:210-219 copies from them would be non-finite \
                     at every level and the first large step would integrate a domain \
                     of NaN instead of stopping"
                )));
            }
            if value.abs() > MAX_FRAME_SPEED_M_PER_S {
                return Err(invalid(format!(
                    "moving-frame {name} is {value} m/s, beyond the {MAX_FRAME_SPEED_M_PER_S} \
                     m/s this lane admits. The frame velocity is a storm motion; base.F:2667 \
                     subtracts it from every base-state wind word, so a speed this large \
                     would replace the declared hodograph with a uniform frame-relative jet \
                     through the whole depth and the run would integrate an atmosphere \
                     nobody declared"
                )));
            }
        }
        Ok(Self {
            u_move_m_per_s,
            v_move_m_per_s,
        })
    }

    /// `umove` (m/s).
    pub fn u_move_m_per_s(self) -> f32 {
        self.u_move_m_per_s
    }

    /// `vmove` (m/s).
    pub fn v_move_m_per_s(self) -> f32 {
        self.v_move_m_per_s
    }

    /// CM1's `imove` for this frame: `1` when it translates, `0` when fixed.
    pub fn imove(self) -> i32 {
        i32::from(self.is_active())
    }

    /// Whether `base.F:2663-2669` has any work to do for this frame.
    pub fn is_active(self) -> bool {
        self.u_move_m_per_s != 0.0_f32 || self.v_move_m_per_s != 0.0_f32
    }
}

/// `base.F:2661-2674`, applied once to one lane's constructed base winds.
///
/// base.F reaches this block after the `isnd=7` file branch and after every
/// `iwnd` branch, so both admitted lanes subtract the same constant at the
/// same point in the construction.  It is applied to the base state only,
/// because `init3d.F:210-219` builds the initial condition by copying it.
pub fn apply_translating_frame(
    u_base_m_per_s: &mut [f32],
    v_base_m_per_s: &mut [f32],
    frame: TranslatingFrame,
) {
    if !frame.is_active() {
        // base.F:2670-2674: with imove=0 the loop does not run at all and
        // CM1 forces umove=vmove=0.  Returning here rather than subtracting
        // a zero keeps a fixed frame raw-bit identical to a build that has
        // no moving frame, including the signed zeros that `x - 0.0` and
        // `x - (-0.0)` would otherwise flip.
        return;
    }
    for value in u_base_m_per_s.iter_mut() {
        // base.F:2667
        *value -= frame.u_move_m_per_s;
    }
    for value in v_base_m_per_s.iter_mut() {
        // base.F:2668
        *value -= frame.v_move_m_per_s;
    }
}

/// See [`IdealizedInitializationFrame`]'s `vortex`.
#[derive(Clone, Debug)]
pub struct VortexCarriers {
    /// (nz, ny, nx), (nz, ny, nx+1), (nz, ny+1, nx)
    pub ppi: Vec<f32>,
    pub ua: Vec<f32>,
    pub va: Vec<f32>,
}

/// One physical-domain idealized base state, whatever lane produced it.
///
/// `input_sounding` is present only for the `isnd=7` lane; the analytic
/// `isnd=5` lane has no external byte stream to preserve.  `psurf`/`tsurf`
/// are carried explicitly because both lanes define them (`base.F:441-442`
/// for `isnd=5`, `base.F:828-835` for `isnd=7`) and the padded surface
/// carriers below are built from them, not from any file record.
#[derive(Clone, Debug)]
pub struct IdealizedInitializationFrame {
    pub(crate) selection: IdealizedInitializationSelection,
    pub(crate) input_sounding: Option<InputSounding>,
    pub(crate) surface_pressure_pa: f32,
    pub(crate) surface_theta_k: f32,
    pub(crate) shape_zyx: [usize; 3],
    pub(crate) mass_heights_m: Vec<f32>,
    pub(crate) theta_base_k: Vec<f32>,
    pub(crate) qv_base_kg_per_kg: Vec<f32>,
    pub(crate) u_base_m_per_s: Vec<f32>,
    pub(crate) v_base_m_per_s: Vec<f32>,
    pub(crate) pi_base: Vec<f32>,
    pub(crate) pressure_base_pa: Vec<f32>,
    pub(crate) temperature_base_k: Vec<f32>,
    pub(crate) relative_humidity_base: Vec<f32>,
    pub(crate) warm_bubble: WarmBubbleFrame,
    /// `init3d.F:1424-1441`: the vapour field after `maintain_rh` rescaled it.
    /// `None` is CM1's default, where `qa(:,:,:,nqv)` is exactly `qv0`.
    pub(crate) initial_qv_kg_per_kg: Option<Vec<f32>>,
    /// init3d.F:697-1256 (iinit=7): the vortex's pressure perturbation and its
    /// part of `ua`/`va` (the base wind is added at the faces).  `None` for
    /// every other arm, whose `ppi` comes from `ibalance`.
    pub(crate) vortex: Option<VortexCarriers>,
    /// The frame `base.F:2663-2669` already subtracted from the two base
    /// winds above.  It is carried, not re-applied, so the state this frame
    /// becomes can report the offset its own coordinates are expressed in.
    pub(crate) moving_frame: TranslatingFrame,
    /// `base.F`'s `qc0`/`qi0`, the base-state cloud water and ice some
    /// analytic arms carry (`isnd = 4, 9, 10, 11`); exact zero on every other
    /// lane.  They enter `thv0` (`base.F:2127-2135`), the buoyancy
    /// (`solve2.F:404`) and the initial cloud (`init3d.F:295-317`).
    pub(crate) qc_base_kg_per_kg: Vec<f32>,
    pub(crate) qi_base_kg_per_kg: Vec<f32>,
    /// `base.F:2212-2246`'s inputs on the analytic `base.F` arms; `None` keeps
    /// the surface words the isnd=5/7/17 lanes have always published.
    pub(crate) analytic_surface: Option<crate::analytic_soundings::AnalyticSurface>,
    /// `init3d.F:625-645`'s `qa(nqc)` after the `iinit = 4` bubble.  `None`
    /// means `init3d.F:295-306`'s copy of `qc0` (or zero) stands.
    pub(crate) initial_qc_kg_per_kg: Option<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct FlatIdealizedMoistInputs {
    shape_zyx: [usize; 3],
    pi0_profile: Vec<f32>,
    th0_profile: Vec<f32>,
    theta_perturbation: Vec<f32>,
    packed_moisture: Vec<f32>,
}

impl FlatIdealizedMoistInputs {
    pub fn shape_zyx(&self) -> [usize; 3] {
        self.shape_zyx
    }

    pub fn pi0_profile(&self) -> &[f32] {
        &self.pi0_profile
    }

    pub fn th0_profile(&self) -> &[f32] {
        &self.th0_profile
    }

    pub fn theta_perturbation(&self) -> &[f32] {
        &self.theta_perturbation
    }

    /// Packed source order is exactly `[qv,qc,qr,qi,qs,qg,k,j,i]`.
    pub fn packed_moisture(&self) -> &[f32] {
        &self.packed_moisture
    }
}

/// One in-memory carrier in the already established CM1 terrain/open layout.
/// This is deliberately not a new persistence or product schema.
#[derive(Clone, Debug)]
pub struct IdealizedPaddedCarrier {
    name: &'static str,
    role: &'static str,
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl IdealizedPaddedCarrier {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn role(&self) -> &'static str {
        self.role
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn raw_sha256(&self) -> String {
        let mut bytes = Vec::with_capacity(self.values.len() * 4);
        for value in &self.values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        sha256::digest_hex(&bytes)
    }
}

/// Selectable fresh idealized state in the same padded/native carrier layout
/// consumed by the existing GM1 terrain and large-step paths.
#[derive(Clone, Debug)]
pub struct IdealizedPaddedState {
    selection: IdealizedInitializationSelection,
    input_sounding_sha256: Option<String>,
    shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    warm_bubble_disposition: WarmBubbleDisposition,
    terrain_static: TerrainStaticFrame,
    terrain_open_carriers: Vec<IdealizedPaddedCarrier>,
    base_wind_carriers: Vec<IdealizedPaddedCarrier>,
    moist_carriers: Vec<IdealizedPaddedCarrier>,
    /// `qc0`, `qi0` (padded mass layout); a group of their own so every
    /// carrier the lanes before them published keeps its place and bytes.
    base_condensate_carriers: Vec<IdealizedPaddedCarrier>,
    pi_recovery: RecoveryMetric,
    theta_recovery: RecoveryMetric,
    moving_frame: TranslatingFrame,
}

impl IdealizedPaddedState {
    pub fn selection(&self) -> IdealizedInitializationSelection {
        self.selection
    }

    /// The `isnd=7` sounding provenance hash, or `None` on an analytic lane
    /// where no external byte stream exists to hash.
    pub fn input_sounding_sha256(&self) -> Option<&str> {
        self.input_sounding_sha256.as_deref()
    }

    pub fn shape_zyx(&self) -> [usize; 3] {
        self.shape_zyx
    }

    pub fn spacing_m(&self) -> [f64; 3] {
        self.spacing_m
    }

    pub fn warm_bubble_disposition(&self) -> WarmBubbleDisposition {
        self.warm_bubble_disposition
    }

    /// The `base.F:2663-2669` frame these carriers are expressed in.
    ///
    /// Every wind word in this state is frame-relative; the ground-relative
    /// wind is this velocity plus the carried word (`init3d.F:32-33`).  A
    /// consumer that reports positions has to add `umove*t` back, so the
    /// velocity travels with the state rather than beside it.
    pub fn moving_frame(&self) -> TranslatingFrame {
        self.moving_frame
    }

    pub fn terrain_static(&self) -> &TerrainStaticFrame {
        &self.terrain_static
    }

    /// Exact existing terrain/open carrier order (50 entries).
    pub fn terrain_open_carriers(&self) -> &[IdealizedPaddedCarrier] {
        &self.terrain_open_carriers
    }

    /// Exact physical base winds on their native horizontal faces.
    pub fn base_wind_carriers(&self) -> &[IdealizedPaddedCarrier] {
        &self.base_wind_carriers
    }

    /// Exact native moisture order: `qa,q3d,qten,qten1`.
    pub fn moist_carriers(&self) -> &[IdealizedPaddedCarrier] {
        &self.moist_carriers
    }

    /// `base.F`'s `qc0` and `qi0`, in that order.
    pub fn base_condensate_carriers(&self) -> &[IdealizedPaddedCarrier] {
        &self.base_condensate_carriers
    }

    pub fn carrier(&self, name: &str) -> Option<&IdealizedPaddedCarrier> {
        self.terrain_open_carriers
            .iter()
            .chain(&self.base_wind_carriers)
            .chain(&self.moist_carriers)
            .chain(&self.base_condensate_carriers)
            .find(|carrier| carrier.name == name)
    }

    pub fn pi_recovery_mismatch_count(&self) -> u64 {
        self.pi_recovery.mismatch_count
    }

    pub fn theta_recovery_mismatch_count(&self) -> u64 {
        self.theta_recovery.mismatch_count
    }

    /// One order-sensitive digest over every emitted carrier in this state.
    ///
    /// Names, ranks, extents, and raw binary32 words all feed the hash, so a
    /// reordered carrier list or a single changed word changes the digest.
    /// It exists so a new initialization lane can prove it left an existing
    /// lane's carriers untouched.
    pub fn carrier_digest_sha256(&self) -> String {
        let mut bytes = Vec::new();
        for carrier in self
            .terrain_open_carriers
            .iter()
            .chain(&self.base_wind_carriers)
            .chain(&self.moist_carriers)
            .chain(std::iter::empty())
        {
            bytes.extend_from_slice(carrier.name.as_bytes());
            bytes.push(0);
            for extent in &carrier.shape {
                bytes.extend_from_slice(&(*extent as u64).to_le_bytes());
            }
            bytes.push(0);
            for value in &carrier.values {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        // The base-state condensate joins the digest only where a sounding
        // carries some, so every lane without it keeps its sealed digest.
        if self
            .base_condensate_carriers
            .iter()
            .any(|carrier| carrier.values.iter().any(|value| *value != 0.0))
        {
            for carrier in &self.base_condensate_carriers {
                bytes.extend_from_slice(carrier.name.as_bytes());
                bytes.push(0);
                for extent in &carrier.shape {
                    bytes.extend_from_slice(&(*extent as u64).to_le_bytes());
                }
                bytes.push(0);
                for value in &carrier.values {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }
        for carrier in self.terrain_static.carriers() {
            bytes.extend_from_slice(carrier.name().as_bytes());
            bytes.push(0);
            for extent in carrier.shape() {
                bytes.extend_from_slice(&(*extent as u64).to_le_bytes());
            }
            bytes.push(0);
            for value in carrier.values() {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        sha256::digest_hex(&bytes)
    }
}

fn padded_carrier(
    name: &'static str,
    role: &'static str,
    shape: Vec<usize>,
    values: Vec<f32>,
) -> Result<IdealizedPaddedCarrier> {
    let required = shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("idealized carrier {name} shape product overflows")))
    })?;
    if values.len() != required {
        return Err(invalid(format!(
            "idealized carrier {name} has {} words; shape {shape:?} requires {required}",
            values.len()
        )));
    }
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid(format!(
            "idealized carrier {name} word {index} is non-finite; bits={:08x}",
            value.to_bits()
        )));
    }
    Ok(IdealizedPaddedCarrier {
        name,
        role,
        shape,
        values,
    })
}

pub(crate) fn index3(k: usize, j: usize, i: usize, ny: usize, nx: usize) -> usize {
    (k * ny + j) * nx + i
}

fn extrapolated_profile(profile: &[f32], bottom: [f32; 3], top: [f32; 3]) -> Result<Vec<f32>> {
    if profile.len() < 4 {
        return Err(invalid(
            "idealized padded state requires at least four vertical levels",
        ));
    }
    let nz = profile.len();
    let mut values = Vec::with_capacity(nz + 2);
    values.push(bottom[0] * profile[0] + bottom[1] * profile[1] + bottom[2] * profile[2]);
    values.extend_from_slice(profile);
    values.push(top[0] * profile[nz - 1] + top[1] * profile[nz - 2] + top[2] * profile[nz - 3]);
    Ok(values)
}

/// `base.F`'s `qc0`/`qi0` on the padded mass layout (`extrapbcs` ghosts).
pub(crate) struct PaddedBaseCondensate {
    qc0: Vec<f32>,
    qi0: Vec<f32>,
}

fn build_flat_idealized_base(
    frame: &IdealizedInitializationFrame,
    inputs: &FlatIdealizedMoistInputs,
    terrain_static: &TerrainStaticFrame,
) -> Result<(TerrainBaseFrame, PaddedBaseCondensate)> {
    let [nz, ny, nx] = frame.shape_zyx;
    let halo_ny = ny + 2 * NGXY;
    let halo_nx = nx + 2 * NGXY;
    let mass_shape = [nz + 2, halo_ny, halo_nx];
    let mass_words = mass_shape.iter().product();
    let surface_words = halo_ny * halo_nx;
    let horizontal_words = ny * nx;
    let bottom = terrain_static.bottom_fornberg();
    let top = terrain_static.top_fornberg();
    let qv_profile = (0..nz)
        .map(|k| frame.qv_base_kg_per_kg[k * horizontal_words])
        .collect::<Vec<_>>();
    let pi_column = extrapolated_profile(inputs.pi0_profile(), bottom, top)?;
    let theta_column = extrapolated_profile(inputs.th0_profile(), bottom, top)?;
    let qv_column = extrapolated_profile(&qv_profile, bottom, top)?;
    // base.F:2048-2049,2077-2078: bcs/extrapbcs of qc0 and qi0 as for qv0.
    let qc_profile = (0..nz)
        .map(|k| frame.qc_base_kg_per_kg[k * horizontal_words])
        .collect::<Vec<_>>();
    let qi_profile = (0..nz)
        .map(|k| frame.qi_base_kg_per_kg[k * horizontal_words])
        .collect::<Vec<_>>();
    let qc_column = extrapolated_profile(&qc_profile, bottom, top)?;
    let qi_column = extrapolated_profile(&qi_profile, bottom, top)?;
    let mut qc0 = vec![0.0_f32; mass_words];
    let mut qi0 = vec![0.0_f32; mass_words];
    let dry = frame.selection.imoist() == 0;
    let mut pi0 = vec![0.0_f32; mass_words];
    let mut prs0 = vec![0.0_f32; mass_words];
    let mut th0 = vec![0.0_f32; mass_words];
    let mut thv0 = vec![0.0_f32; mass_words];
    let mut rth0 = vec![0.0_f32; mass_words];
    let mut qv0 = vec![0.0_f32; mass_words];
    let mut rho0 = vec![0.0_f32; mass_words];
    let mut rr0 = vec![0.0_f32; mass_words];
    let mut rf0 = vec![0.0_f32; mass_words];
    let mut rrf0 = vec![0.0_f32; mass_words];
    let reps = RV / RD;

    // isnd=7 fills ib:ie,jb:je, then extrapbcs.  A flat grid therefore has
    // the same source word in every horizontal allocation cell.
    for j in 0..halo_ny {
        for i in 0..halo_nx {
            for k in 0..=nz + 1 {
                let index = index3(k, j, i, halo_ny, halo_nx);
                pi0[index] = pi_column[k];
                th0[index] = theta_column[k];
                qv0[index] = qv_column[k];
                qc0[index] = qc_column[k];
                qi0[index] = qi_column[k];
                let qv = qv0[index];
                let qv_nonnegative = qv.max(0.0_f32);
                // base.F:2127-2135.  qc0+qi0 is exact zero on every lane
                // without base-state condensate, so those lanes keep the
                // words they always had.
                thv0[index] = if dry {
                    th0[index]
                } else {
                    (th0[index] + 0.0_f32) * (1.0_f32 + reps * qv_nonnegative)
                        / (1.0_f32 + qv_nonnegative + (qc0[index] + qi0[index]).max(0.0_f32))
                };
                prs0[index] = P00 * pi0[index].powf(CP / RD);
                rho0[index] = prs0[index] / (RD * th0[index] * pi0[index] * (1.0_f32 + qv * reps));
                rth0[index] = 1.0_f32 / th0[index];
            }
            // base.F calls extrapbcs(rho0) after the diagnostic calculation.
            rho0[index3(0, j, i, halo_ny, halo_nx)] = bottom[0]
                * rho0[index3(1, j, i, halo_ny, halo_nx)]
                + bottom[1] * rho0[index3(2, j, i, halo_ny, halo_nx)]
                + bottom[2] * rho0[index3(3, j, i, halo_ny, halo_nx)];
            rho0[index3(nz + 1, j, i, halo_ny, halo_nx)] = top[0]
                * rho0[index3(nz, j, i, halo_ny, halo_nx)]
                + top[1] * rho0[index3(nz - 1, j, i, halo_ny, halo_nx)]
                + top[2] * rho0[index3(nz - 2, j, i, halo_ny, halo_nx)];
            for k in 0..=nz + 1 {
                let index = index3(k, j, i, halo_ny, halo_nx);
                rr0[index] = 1.0_f32 / rho0[index];
            }
        }
    }
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let physical = index3(k, j, i, ny, nx);
                let padded = index3(k + 1, j + NGXY, i + NGXY, halo_ny, halo_nx);
                for (name, actual, expected) in [
                    ("pi0", pi0[padded], frame.pi_base[physical]),
                    ("th0", th0[padded], frame.theta_base_k[physical]),
                    ("qv0", qv0[padded], frame.qv_base_kg_per_kg[physical]),
                    ("prs0", prs0[padded], frame.pressure_base_pa[physical]),
                ] {
                    if actual.to_bits() != expected.to_bits() {
                        return Err(invalid(format!(
                            "idealized padded {name} differs from the source column at k={k},j={j},i={i}: expected {:08x}, got {:08x}",
                            expected.to_bits(), actual.to_bits()
                        )));
                    }
                }
            }
        }
    }

    let c1 = terrain_static
        .carrier("c1")
        .ok_or_else(|| invalid("idealized terrain static frame lacks c1"))?
        .values();
    let c2 = terrain_static
        .carrier("c2")
        .ok_or_else(|| invalid("idealized terrain static frame lacks c2"))?
        .values();
    for j in 0..halo_ny {
        let local_j = (j as isize - NGXY as isize).clamp(0, ny as isize - 1) as usize;
        for i in 0..halo_nx {
            let local_i = (i as isize - NGXY as isize).clamp(0, nx as isize - 1) as usize;
            let bottom_density = bottom[0] * rho0[index3(1, j, i, halo_ny, halo_nx)]
                + bottom[1] * rho0[index3(2, j, i, halo_ny, halo_nx)]
                + bottom[2] * rho0[index3(3, j, i, halo_ny, halo_nx)];
            rf0[index3(0, j, i, halo_ny, halo_nx)] = bottom_density;
            rf0[index3(1, j, i, halo_ny, halo_nx)] = bottom_density;
            for k in 2..=nz {
                let weight = index3(k - 1, local_j, local_i, ny, nx);
                rf0[index3(k, j, i, halo_ny, halo_nx)] = c1[weight]
                    * rho0[index3(k - 1, j, i, halo_ny, halo_nx)]
                    + c2[weight] * rho0[index3(k, j, i, halo_ny, halo_nx)];
            }
            rf0[index3(nz + 1, j, i, halo_ny, halo_nx)] = top[0]
                * rho0[index3(nz, j, i, halo_ny, halo_nx)]
                + top[1] * rho0[index3(nz - 1, j, i, halo_ny, halo_nx)]
                + top[2] * rho0[index3(nz - 2, j, i, halo_ny, halo_nx)];
            for k in 0..=nz + 1 {
                let index = index3(k, j, i, halo_ny, halo_nx);
                rrf0[index] = 1.0_f32 / rf0[index];
            }
        }
    }
    for (name, values) in [
        ("pi0", &pi0),
        ("prs0", &prs0),
        ("th0", &th0),
        ("thv0", &thv0),
        ("rho0", &rho0),
        ("rf0", &rf0),
    ] {
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || *value <= 0.0_f32)
        {
            return Err(invalid(format!(
                "idealized base {name} word {index} is not positive finite; bits={:08x}",
                value.to_bits()
            )));
        }
    }
    // base.F:592,609 zero qv0 outright under imoist=0, so a dry base state's
    // vapour is exactly zero and only a negative or non-finite word is wrong.
    if let Some((index, value)) = qv0
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value < 0.0_f32)
    {
        return Err(invalid(format!(
            "idealized base qv0 word {index} is negative or non-finite; bits={:08x}",
            value.to_bits()
        )));
    }

    let (surface_pi, surface_pressure, surface_rth) = match frame.analytic_surface {
        None => {
            let surface_pressure = frame.surface_pressure_pa;
            let surface_pi = (surface_pressure / P00).powf(RD / CP);
            // base.F first stores tsurf=th_sfc*pi_sfc, then recovers thsurf from
            // tsurf/(psurf/p00)**(rd/cp) before assigning rth0s.  Preserve both
            // binary32 operations instead of simplifying them algebraically.
            let surface_temperature = frame.surface_theta_k * surface_pi;
            let recovered_surface_theta =
                surface_temperature / (surface_pressure / P00).powf(RD / CP);
            (surface_pi, surface_pressure, recovered_surface_theta.powi(-1))
        }
        Some(record) => {
            let interior = index3(1, NGXY, NGXY, halo_ny, halo_nx);
            let level = |values: &[f32], k: usize| values[index3(k, NGXY, NGXY, halo_ny, halo_nx)];
            cm1_surface_words(
                record.surface,
                bottom,
                [level(&th0, 1), level(&th0, 2), level(&th0, 3)],
                [level(&qv0, 1), level(&qv0, 2), level(&qv0, 3)],
                pi0[interior],
                thv0[interior],
                frame.mass_heights_m[0],
            )
        }
    };
    let mut rho0s = vec![0.0_f32; surface_words];
    let pi0s = vec![surface_pi; surface_words];
    let prs0s = vec![surface_pressure; surface_words];
    let rth0s = vec![surface_rth; surface_words];
    for j in 0..halo_ny {
        for i in 0..halo_nx {
            rho0s[j * halo_nx + i] = rf0[index3(1, j, i, halo_ny, halo_nx)];
        }
    }
    Ok((
        TerrainBaseFrame {
            pi0,
            prs0,
            th0,
            thv0,
            rth0,
            qv0,
            rho0,
            rr0,
            rf0,
            rrf0,
            rho0s,
            pi0s,
            prs0s,
            rth0s,
        },
        PaddedBaseCondensate { qc0, qi0 },
    ))
}

/// `base.F:2212-2226` (psurf and tsurf both set) or `:2227-2239` (not set):
/// the surface Exner, pressure and reciprocal theta, on a flat grid.
/// Returns `(pi0s, prs0s, rth0s)`.
fn cm1_surface_words(
    surface: crate::analytic_soundings::ArmSurface,
    cgs: [f32; 3],
    th0: [f32; 3],
    qv0: [f32; 3],
    pi0_1: f32,
    thv0_1: f32,
    zh_1: f32,
) -> (f32, f32, f32) {
    use crate::analytic_soundings::{ArmSurface, TSMALL};
    let reps = RV / RD;
    let rp00 = 1.0_f32 / P00;
    let rovcp = RD / CP;
    let (psurf, tsurf) = match surface {
        ArmSurface::PsurfTsurf { psurf, tsurf } => (psurf, tsurf),
        ArmSurface::PsurfTsurfFromT0 { psurf, t0 } => {
            (psurf, cgs[0] * t0[0] + cgs[1] * t0[1] + cgs[2] * t0[2])
        }
        ArmSurface::Unset => (0.0_f32, 0.0_f32),
    };
    if psurf >= TSMALL && tsurf >= TSMALL {
        let thsurf = tsurf / (psurf * rp00).powf(rovcp);
        let pi0s = (psurf * rp00).powf(rovcp);
        (pi0s, psurf, 1.0_f32 / thsurf)
    } else {
        let thsurf = cgs[0] * th0[0] + cgs[1] * th0[1] + cgs[2] * th0[2];
        let qsurf = cgs[0] * qv0[0] + cgs[1] * qv0[1] + cgs[2] * qv0[2];
        let thvsurf = thsurf * (1.0_f32 + qsurf * reps) / (1.0_f32 + qsurf);
        let pi0s = pi0_1 + zh_1 * G / (CP * 0.5_f32 * (thvsurf + thv0_1));
        let prs0s = P00 * pi0s.powf(CP / RD);
        (pi0s, prs0s, 1.0_f32 / thsurf)
    }
}

fn flat_face_winds(frame: &IdealizedInitializationFrame) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let [nz, ny, nx] = frame.shape_zyx;
    let horizontal_words = ny * nx;
    let mut ua = vec![0.0_f32; nz * ny * (nx + 1)];
    let mut va = vec![0.0_f32; nz * (ny + 1) * nx];
    for k in 0..nz {
        let first = k * horizontal_words;
        let u = frame.u_base_m_per_s[first];
        let v = frame.v_base_m_per_s[first];
        for horizontal in 1..horizontal_words {
            if frame.u_base_m_per_s[first + horizontal].to_bits() != u.to_bits()
                || frame.v_base_m_per_s[first + horizontal].to_bits() != v.to_bits()
            {
                return Err(invalid(format!(
                    "idealized flat padded adapter refuses horizontally varying base wind at k={k}, word {horizontal}"
                )));
            }
        }
        for j in 0..ny {
            for i in 0..=nx {
                ua[index3(k, j, i, ny, nx + 1)] = u;
            }
        }
        for j in 0..=ny {
            for i in 0..nx {
                va[index3(k, j, i, ny + 1, nx)] = v;
            }
        }
    }
    Ok((ua, va, vec![0.0_f32; (nz + 1) * ny * nx]))
}

/// `init3d.F:295-317`'s condensate copies and `init3d.F:625-645`'s `iinit=4`
/// cloud, for [`pack_idealized_moisture`].
#[derive(Clone, Copy, Default)]
struct InitialCondensate<'a> {
    /// Padded `qc0`, when `init3d.F:295-306` copies it into `qa(nqc)`.
    qc0: Option<&'a [f32]>,
    /// Padded `qi0`, when `init3d.F:308-317` copies it into `qa(nqi)`.
    qi0: Option<&'a [f32]>,
    /// Physical `qa(nqc)` after the `iinit = 4` bubble.
    initial_qc: Option<&'a [f32]>,
}

fn pack_idealized_moisture(
    base: &TerrainBaseFrame,
    shape_zyx: [usize; 3],
    initial_qv: Option<&[f32]>,
    condensate: InitialCondensate<'_>,
) -> Result<Vec<IdealizedPaddedCarrier>> {
    let [nz, ny, nx] = shape_zyx;
    let scalar_shape = [nz + 2, ny + 2 * NGXY, nx + 2 * NGXY];
    let scalar_words = scalar_shape.iter().product::<usize>();
    if base.qv0.len() != scalar_words {
        return Err(invalid("idealized padded qv0 geometry drifted"));
    }
    let mut qa = vec![0.0_f32; 6 * scalar_words];
    // init3d.F:283-290 assigns qv0 over the complete allocated qa domain;
    // the five other WSM6 species retain allocation +0.
    qa[..scalar_words].copy_from_slice(&base.qv0);
    // init3d.F:1424-1441.  maintain_rh rescales qa(:,:,:,nqv) across the
    // perturbation; without this the published vapour is qv0 everywhere and
    // the flag has no effect on anything a consumer can see.
    if let Some(values) = initial_qv {
        let physical = checked_product(shape_zyx, "maintain_rh vapour")?;
        if values.len() != physical {
            return Err(invalid("maintain_rh vapour field is not the physical shape"));
        }
        let padded = pad_physical_field(values, [nz, ny, nx], scalar_shape, 1)?;
        qa[..scalar_words].copy_from_slice(&padded);
    }
    // WSM6 row order (param.F:3641-3678): qv, qc, qr, qi, qs, qg.
    if let Some(values) = condensate.qc0 {
        qa[scalar_words..2 * scalar_words].copy_from_slice(values);
    }
    if let Some(values) = condensate.initial_qc {
        let padded = pad_physical_field(values, [nz, ny, nx], scalar_shape, 1)?;
        qa[scalar_words..2 * scalar_words].copy_from_slice(&padded);
    }
    if let Some(values) = condensate.qi0 {
        qa[3 * scalar_words..4 * scalar_words].copy_from_slice(values);
    }
    let q3d = qa.clone();
    let packed_shape = vec![6, nz + 2, ny + 2 * NGXY, nx + 2 * NGXY];
    let zero = vec![0.0_f32; qa.len()];
    Ok(vec![
        padded_carrier(
            "qa",
            "source-initialized six-species moisture A carrier",
            packed_shape.clone(),
            qa,
        )?,
        padded_carrier(
            "q3d",
            "source-initialized six-species current moisture carrier",
            packed_shape.clone(),
            q3d,
        )?,
        padded_carrier(
            "qten",
            "exact cm1.F allocation-zero stage moisture tendency",
            packed_shape.clone(),
            zero.clone(),
        )?,
        padded_carrier(
            "qten1",
            "exact cold-start held moisture tendency",
            packed_shape,
            zero,
        )?,
    ])
}

/// Pad one physical `[nk,ny,nx]` field into the halo storage the runtime
/// reads, by the zero-gradient edge copy this flat lane's open boundary uses.
///
/// `k_physical_start` is the padded k index the first physical level occupies:
/// `1` for both the mass family (levels 1..nk) and the w family (faces
/// 1..nk+1), which is how `build_flat_idealized_base` lays the base arrays out.
/// Only the physical rectangle is ever read back through
/// `gm1_idealized_residency._physical`; the halos are filled so nothing in the
/// carrier is uninitialised.
fn pad_physical_field(
    values: &[f32],
    physical_shape: [usize; 3],
    padded_shape: [usize; 3],
    k_physical_start: usize,
) -> Result<Vec<f32>> {
    let [pk, py, px] = physical_shape;
    let [tk, ty, tx] = padded_shape;
    if values.len() != checked_product(physical_shape, "padded source")? {
        return Err(invalid("padded source length does not match its shape"));
    }
    if pk == 0 || py == 0 || px == 0 || tk < pk || ty < py || tx < px {
        return Err(invalid("padded target is smaller than its physical field"));
    }
    let halo_y = (ty - py) / 2;
    let halo_x = (tx - px) / 2;
    let mut padded = vec![0.0_f32; checked_product(padded_shape, "padded target")?];
    for k in 0..tk {
        let source_k = (k as isize - k_physical_start as isize).clamp(0, pk as isize - 1) as usize;
        for j in 0..ty {
            let source_j = (j as isize - halo_y as isize).clamp(0, py as isize - 1) as usize;
            for i in 0..tx {
                let source_i = (i as isize - halo_x as isize).clamp(0, px as isize - 1) as usize;
                padded[index3(k, j, i, ty, tx)] =
                    values[index3(source_k, source_j, source_i, py, px)];
            }
        }
    }
    Ok(padded)
}

fn append_terrain_open_carriers(
    output: &mut Vec<IdealizedPaddedCarrier>,
    base: TerrainBaseFrame,
    surface: SurfaceHaloFrame,
    mutable: crate::terrain_admission_science::MutableColdStartFrame,
    shape_zyx: [usize; 3],
) -> Result<()> {
    let [nz, ny, nx] = shape_zyx;
    let mass = vec![nz + 2, ny + 2 * NGXY, nx + 2 * NGXY];
    let surface_shape = vec![ny + 2 * NGXY, nx + 2 * NGXY];
    for (name, role, values) in [
        ("pi0", "idealized hydrostatic base Exner", base.pi0),
        ("prs0", "idealized hydrostatic base pressure", base.prs0),
        ("th0", "idealized base potential temperature", base.th0),
        (
            "thv0",
            "source-rebuilt base virtual potential temperature",
            base.thv0,
        ),
        (
            "rth0",
            "reciprocal idealized base potential temperature",
            base.rth0,
        ),
        ("qv0", "idealized base water-vapor mixing ratio", base.qv0),
        ("rho0", "idealized hydrostatic base density", base.rho0),
        ("rr0", "reciprocal idealized base density", base.rr0),
        (
            "rf0",
            "source-interpolated idealized base face density",
            base.rf0,
        ),
        ("rrf0", "reciprocal idealized base face density", base.rrf0),
    ] {
        output.push(padded_carrier(name, role, mass.clone(), values)?);
    }
    for (name, role, values) in [
        ("rho0s", "idealized surface base density", base.rho0s),
        ("pi0s", "supplied-sounding surface Exner", base.pi0s),
        ("prs0s", "supplied-sounding surface pressure", base.prs0s),
        (
            "rth0s",
            "reciprocal supplied-sounding surface theta",
            base.rth0s,
        ),
        (
            "terrain_msl_halo",
            "flat idealized terrain after source open edge law",
            surface.terrain_msl_halo,
        ),
        (
            "gz",
            "source-valid padded terrain Jacobian",
            surface.gz_halo,
        ),
        (
            "rgz",
            "source-valid padded reciprocal terrain Jacobian",
            surface.rgz_halo,
        ),
        (
            "gzu",
            "source-valid padded u-face terrain Jacobian",
            surface.gzu_halo,
        ),
        (
            "rgzu",
            "source-valid padded reciprocal u-face terrain Jacobian",
            surface.rgzu_halo,
        ),
        (
            "gzv",
            "source-valid padded v-face terrain Jacobian",
            surface.gzv_halo,
        ),
        (
            "rgzv",
            "source-valid padded reciprocal v-face terrain Jacobian",
            surface.rgzv_halo,
        ),
        (
            "dzdx",
            "source flx6 padded terrain x-gradient",
            surface.dzdx_halo,
        ),
        (
            "dzdy",
            "source flx6 padded terrain y-gradient",
            surface.dzdy_halo,
        ),
    ] {
        output.push(padded_carrier(name, role, surface_shape.clone(), values)?);
    }
    for carrier in mutable.carriers {
        output.push(padded_carrier(
            carrier.name,
            carrier.role,
            carrier.shape,
            carrier.values,
        )?);
    }
    if output.len() != TERRAIN_OPEN_CARRIER_NAMES.len()
        || output
            .iter()
            .map(IdealizedPaddedCarrier::name)
            .ne(TERRAIN_OPEN_CARRIER_NAMES)
    {
        return Err(invalid(
            "idealized terrain/open carrier order drifted from the established layout",
        ));
    }
    Ok(())
}

/// Adapt a source-ordered physical idealized frame to the established padded
/// terrain/open and moisture layouts.  This first runnable gate is deliberately
/// flat, unstretched, open-boundary, fresh-start only.
pub fn adapt_flat_idealized_frame_to_padded_state(
    frame: IdealizedInitializationFrame,
    spacing_m: [f64; 3],
) -> Result<IdealizedPaddedState> {
    adapt_idealized_frame_to_padded_state(frame, spacing_m, None)
}

/// The same adaptation over a declared CM1 vertical coordinate.
///
/// `spacing_m` still carries CM1's `dx`, `dy` and namelist `dz`; on a
/// stretched grid `param.F:8170` makes that third word the metric reference
/// every `mh` divides by rather than a layer thickness, and the levels
/// themselves arrive in `vertical`.  The frame's own mass heights are then
/// held to `sigma(1..nk)` raw word for raw word, exactly as the flat entry
/// holds them to `param.F:7792,7801`'s midpoints.
pub fn adapt_stretched_idealized_frame_to_padded_state(
    frame: IdealizedInitializationFrame,
    spacing_m: [f64; 3],
    vertical: &DeclaredVerticalCoordinate,
) -> Result<IdealizedPaddedState> {
    adapt_idealized_frame_to_padded_state(frame, spacing_m, Some(vertical))
}

fn adapt_idealized_frame_to_padded_state(
    frame: IdealizedInitializationFrame,
    spacing_m: [f64; 3],
    vertical: Option<&DeclaredVerticalCoordinate>,
) -> Result<IdealizedPaddedState> {
    if frame.selection.irst() != 0 {
        return Err(invalid(
            "fresh idealized padded entry refuses irst=1; restored state must enter through the restart path",
        ));
    }
    for (axis, value) in ["dx", "dy", "dz"].into_iter().zip(spacing_m) {
        if !value.is_finite() || value <= 0.0 || (value as f32) as f64 != value {
            return Err(invalid(format!(
                "idealized padded {axis} must be positive finite and exactly representable in default REAL"
            )));
        }
    }
    let [nz, ny, nx] = frame.shape_zyx;
    if nz < 4 || ny == 0 || nx == 0 {
        return Err(invalid(
            "idealized padded entry requires nz>=4 and positive horizontal extents",
        ));
    }
    let dz = spacing_m[2] as f32;
    if let Some(coordinate) = vertical {
        if coordinate.sigma.len() != nz {
            return Err(invalid(format!(
                "declared vertical coordinate has {} scalar levels; this frame has {nz}",
                coordinate.sigma.len()
            )));
        }
        if coordinate.dz_reference_m.to_bits() != dz.to_bits() {
            return Err(invalid(
                "declared vertical dz reference differs from spacing_m[2]; param.F:8170 divides every mh word by exactly one number",
            ));
        }
    }
    for k in 0..nz {
        // param.F:7792,7801: round both face coordinates before the midpoint.
        // param.F:8169 is the same statement on a declared coordinate, and
        // sigma already holds its result.
        let expected = match vertical {
            None => 0.5_f32 * (dz * k as f32 + dz * (k + 1) as f32),
            Some(coordinate) => coordinate.sigma[k],
        };
        for j in 0..ny {
            for i in 0..nx {
                let index = index3(k, j, i, ny, nx);
                if frame.mass_heights_m[index].to_bits() != expected.to_bits() {
                    return Err(invalid(format!(
                        "idealized padded flat-grid entry refuses mass height k={k},j={j},i={i}; expected bits {:08x}, got {:08x}",
                        expected.to_bits(), frame.mass_heights_m[index].to_bits()
                    )));
                }
            }
        }
    }
    let inputs = frame.flat_moist_inputs()?;
    let horizontal_words = ny * nx;
    let qv_profile = (0..nz)
        .map(|k| frame.qv_base_kg_per_kg[k * horizontal_words])
        .collect::<Vec<_>>();
    let rho_profile = (0..nz)
        .map(|k| {
            let pi = inputs.pi0_profile[k];
            let theta = inputs.th0_profile[k];
            let qv = qv_profile[k];
            let pressure = P00 * pi.powf(CP / RD);
            pressure / (RD * theta * pi * (1.0_f32 + qv * (RV / RD)))
        })
        .collect::<Vec<_>>();
    let terrain = vec![0.0_f32; horizontal_words];
    let crop = HorizontalCrop::full(ny, nx);
    let terrain_static = match vertical {
        None => compute_flat_uniform_static(
            &rho_profile,
            frame.shape_zyx,
            spacing_m,
            crop,
            TerrainStaticConfig::new(5, 5, [2, 2, 2, 2])?,
        )?,
        Some(coordinate) => compute_flat_static_levels(
            &rho_profile,
            frame.shape_zyx,
            [spacing_m[0], spacing_m[1]],
            coordinate,
            crop,
            TerrainStaticConfig::new(5, 5, [2, 2, 2, 2])?,
        )?,
    };
    let zh = terrain_static
        .carrier("zh")
        .ok_or_else(|| invalid("idealized terrain static frame lacks zh"))?
        .values();
    for (index, (actual, expected)) in zh.iter().zip(&frame.mass_heights_m).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            return Err(invalid(format!(
                "idealized terrain-static zh differs from initialized frame at word {index}"
            )));
        }
    }
    let c1 = terrain_static
        .carrier("c1")
        .ok_or_else(|| invalid("idealized terrain static frame lacks c1"))?
        .values();
    let c2 = terrain_static
        .carrier("c2")
        .ok_or_else(|| invalid("idealized terrain static frame lacks c2"))?
        .values();
    // init3d.F:1787-1917: ibalance=2's anelastic buoyancy pressure, handed to
    // the getset completion below exactly as iinit=7 hands its own.
    let anelastic_ppi = if frame.selection.ibalance() == 2 && frame.vortex.is_none() {
        let column = |name: &str, n: usize| -> Result<Vec<f32>> {
            let carrier = terrain_static
                .carrier(name)
                .ok_or_else(|| invalid(format!("idealized terrain static frame lacks {name}")))?;
            Ok((0..n).map(|k| carrier.values()[k * horizontal_words]).collect())
        };
        let mh = column("mh", nz)?;
        let mf = column("mf", nz + 1)?;
        let rf0 = column("rf0", nz + 1)?;
        let reps = RV / RD;
        let moist_run = frame.selection.imoist() == 1;
        let thv0: Vec<f32> = (0..nz)
            .map(|k| {
                if !moist_run {
                    return inputs.th0_profile[k];
                }
                let qv = qv_profile[k].max(0.0_f32);
                let condensate = (frame.qc_base_kg_per_kg[k * horizontal_words]
                    + frame.qi_base_kg_per_kg[k * horizontal_words])
                    .max(0.0_f32);
                (inputs.th0_profile[k] + 0.0_f32) * (1.0_f32 + reps * qv)
                    / (1.0_f32 + qv + condensate)
            })
            .collect();
        let words = nz * horizontal_words;
        let packed = inputs.packed_moisture();
        let mut qt = vec![0.0_f32; words];
        for (index, value) in qt.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for species in 1..6 {
                sum += packed[species * words + index];
            }
            *value = sum;
        }
        let [imirror, jmirror] = frame.selection.elliptic_mirror();
        Some(crate::anelastic_balance::anelastic_balance_ppi(
            &crate::anelastic_balance::AnelasticBalanceInputs {
                nx,
                ny,
                nz,
                dx: spacing_m[0] as f32,
                dz,
                imirror,
                jmirror,
                moist: moist_run,
                mh: &mh,
                mf: &mf,
                rho0: &rho_profile,
                rf0: &rf0,
                thv0: &thv0,
                th0: inputs.th0_profile(),
                tha: inputs.theta_perturbation(),
                qv: &packed[..words],
                qt: &qt,
            },
        )?)
    } else {
        None
    };
    let moist = crate::initialization::complete_moist_initialization_with_ppi(
        inputs.pi0_profile(),
        inputs.th0_profile(),
        inputs.theta_perturbation(),
        inputs.packed_moisture(),
        zh,
        c1,
        c2,
        terrain_static.bottom_fornberg(),
        terrain_static.top_fornberg(),
        frame.shape_zyx,
        // init3d.F:1743-1830.  run/namelist.input:80 and every one of CM1's
        // 23 shipped decks select ibalance=0; the selector carries it here.
        MoistInitializationConfig::new(
            G,
            CP,
            RD,
            RV,
            P00,
            3,
            // ibalance=2's pressure arrives as the prescribed ppi below.
            if frame.selection.ibalance() == 2 { 0 } else { frame.selection.ibalance() as u8 },
            true,
            1,
        )?
        .with_imoist(frame.selection.imoist() as u8)?,
        MoistInitializationAdmission::Generic,
        // init3d.F:1233: iinit=7 sets setppi = .false.; its pressure is the vortex's.
        // ibalance=2: init3d.F:1893's poiss solution (bcs(ppi) at :1915).
        frame
            .vortex
            .as_ref()
            .map(|vortex| vortex.ppi.as_slice())
            .or(anelastic_ppi.as_deref()),
    )?;
    let (base, condensate) = build_flat_idealized_base(&frame, &inputs, &terrain_static)?;
    let (copy_qc, copy_qi) = frame.initial_condensate_copies();
    let moist_carriers = pack_idealized_moisture(
        &base,
        frame.shape_zyx,
        frame.initial_qv_kg_per_kg(),
        InitialCondensate {
            qc0: copy_qc.then_some(condensate.qc0.as_slice()),
            qi0: copy_qi.then_some(condensate.qi0.as_slice()),
            initial_qc: frame.initial_qc_kg_per_kg(),
        },
    )?;
    let mass_shape_padded = vec![nz + 2, ny + 2 * NGXY, nx + 2 * NGXY];
    let base_condensate_carriers = vec![
        padded_carrier(
            "qc0",
            "base.F base-state cloud water (zero unless the sounding carries it)",
            mass_shape_padded.clone(),
            condensate.qc0,
        )?,
        padded_carrier(
            "qi0",
            "base.F base-state cloud ice (zero unless the sounding carries it)",
            mass_shape_padded,
            condensate.qi0,
        )?,
    ];
    let surface = match vertical {
        None => compute_surface_halos(&terrain, [ny, nx], crop, spacing_m, nz)?,
        // param.F:8193: the declared top w level, not dz*nk.
        Some(coordinate) => crate::terrain_admission_science::compute_surface_halos_at_top(
            &terrain,
            [ny, nx],
            crop,
            spacing_m,
            coordinate.sigmaf[nz],
        )?,
    };
    let (ua, va, wa) = flat_face_winds(&frame)?;
    // init3d.F:1206-1214: ua = vortex + u0, va = vortex + v0 (iinit=7 only);
    // u0/v0 themselves stay the base wind.
    let (ua_initial, va_initial) = match frame.vortex.as_ref() {
        None => (ua.clone(), va.clone()),
        Some(vortex) => (
            vortex.ua.iter().zip(&ua).map(|(a, b)| a + b).collect::<Vec<f32>>(),
            vortex.va.iter().zip(&va).map(|(a, b)| a + b).collect::<Vec<f32>>(),
        ),
    };
    let mutable = pack_cold_start_mutable(
        frame.shape_zyx,
        crop,
        inputs.pi0_profile(),
        inputs.th0_profile(),
        &ua_initial,
        &va_initial,
        &wa,
        moist.ppi(),
        Some(moist.ppi()),
        inputs.theta_perturbation(),
        inputs.packed_moisture(),
        &base,
        &surface,
        PerturbationPolicy::PreserveInitializedPerturbation,
    )?;
    let base_wind_carriers = vec![
        padded_carrier(
            "u0",
            "isnd=7 interpolated base zonal wind",
            vec![nz, ny, nx + 1],
            ua,
        )?,
        padded_carrier(
            "v0",
            "isnd=7 interpolated base meridional wind",
            vec![nz, ny + 1, nx],
            va,
        )?,
    ];
    let pi_recovery = mutable.pi_recovery;
    let theta_recovery = mutable.theta_recovery;
    if pi_recovery.mismatch_count != 0 || theta_recovery.mismatch_count != 0 {
        return Err(invalid(format!(
            "idealized padded current/A recovery is not raw-bit exact: pi={}, theta={}",
            pi_recovery.mismatch_count, theta_recovery.mismatch_count
        )));
    }
    let mut terrain_open_carriers = Vec::with_capacity(TERRAIN_OPEN_CARRIER_NAMES.len() + 4);
    append_terrain_open_carriers(
        &mut terrain_open_carriers,
        base,
        surface,
        mutable,
        frame.shape_zyx,
    )?;
    // init3d.F:2386-2418 and :2432-2440 are the INITIALIZED pressure, density
    // and reciprocal/face density: prs = p00*(pi0+pp3d)^(cp/rd) and
    // rho = prs/((th0+tha)*(pi0+pp3d)*(rd+max(0,qv)*rv)).  They are not the
    // base-state prs0/rho0 above, which carry neither the perturbation nor
    // the initial pressure balance, and the difference is exactly the
    // perturbation's own buoyancy: a 1 K bubble on a 300 K base makes rho0
    // wrong by 0.33% at the bubble, the same order as the signal it is there
    // to provide.  Both sets are published, under CM1's own two names.
    let mass_padded = [nz + 2, ny + 2 * NGXY, nx + 2 * NGXY];
    for (name, role, values, physical, start) in [
        (
            "prs",
            "init3d.F:2393 initialized air pressure",
            moist.prs(),
            [nz, ny, nx],
            1usize,
        ),
        (
            "rho",
            "init3d.F:2394 initialized air density",
            moist.rho(),
            [nz, ny, nx],
            1usize,
        ),
        (
            "rr",
            "init3d.F:2432 reciprocal initialized air density",
            moist.rr(),
            [nz, ny, nx],
            1usize,
        ),
        (
            "rf",
            "init3d.F:2433-2466 initialized w-face air density",
            moist.rf(),
            [nz + 1, ny, nx],
            1usize,
        ),
    ] {
        let padded = pad_physical_field(values, physical, mass_padded, start)?;
        terrain_open_carriers.push(padded_carrier(
            name,
            role,
            mass_padded.to_vec(),
            padded,
        )?);
    }
    Ok(IdealizedPaddedState {
        selection: frame.selection,
        input_sounding_sha256: frame
            .input_sounding
            .as_ref()
            .map(|sounding| sounding.source_sha256.clone()),
        shape_zyx: frame.shape_zyx,
        spacing_m,
        warm_bubble_disposition: frame.warm_bubble.disposition,
        terrain_static,
        terrain_open_carriers,
        base_wind_carriers,
        moist_carriers,
        base_condensate_carriers,
        pi_recovery,
        theta_recovery,
        moving_frame: frame.moving_frame,
    })
}

/// First-class selectable flat idealized entry.  Grid coordinates are the
/// official unstretched horizontal mass centers and source-ordered vertical
/// face midpoints from param.F:7789-7811.
#[allow(clippy::too_many_arguments)]
pub fn initialize_flat_idealized_padded_state(
    source_bytes: &[u8],
    selection: IdealizedInitializationSelection,
    controls: WarmBubbleControls,
    shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    origin_xy_m: [f32; 2],
    moving_frame: TranslatingFrame,
) -> Result<IdealizedPaddedState> {
    initialize_idealized_padded_state(
        source_bytes,
        selection,
        controls,
        shape_zyx,
        spacing_m,
        origin_xy_m,
        moving_frame,
        None,
    )
}

/// First-class selectable idealized entry over a declared vertical coordinate.
///
/// Horizontal mass centres are `param.F`'s unstretched ones; the vertical mass
/// levels are `sigma(1..nk)` from `vertical` rather than `param.F:7792,7801`'s
/// midpoints, which is what lets a 15 km column put its first scalar level at
/// tens of metres instead of `0.5*dz`.
#[allow(clippy::too_many_arguments)]
pub fn initialize_stretched_idealized_padded_state(
    source_bytes: &[u8],
    selection: IdealizedInitializationSelection,
    controls: WarmBubbleControls,
    shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    origin_xy_m: [f32; 2],
    moving_frame: TranslatingFrame,
    vertical: &DeclaredVerticalCoordinate,
) -> Result<IdealizedPaddedState> {
    initialize_idealized_padded_state(
        source_bytes,
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
fn initialize_idealized_padded_state(
    source_bytes: &[u8],
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
            "flat idealized padded entry requires nz>=4 and positive horizontal extents",
        ));
    }
    for value in origin_xy_m {
        require_finite("idealized domain origin", value)?;
    }
    for (axis, value) in ["dx", "dy", "dz"].into_iter().zip(spacing_m) {
        if !value.is_finite() || value <= 0.0 || (value as f32) as f64 != value {
            return Err(invalid(format!(
                "flat idealized padded {axis} must be positive finite and exactly representable in default REAL"
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
    let mut frame = initialize_idealized_domain(
        source_bytes,
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
    apply_tc_vortex_arm(&mut frame, &xh, &yh, [dx, dy, dz], origin_xy_m, vertical)?;
    apply_moist_benchmark_arm(&mut frame, controls, &xh)?;
    adapt_idealized_frame_to_padded_state(frame, spacing_m, vertical)
}

impl IdealizedInitializationFrame {
    pub fn selection(&self) -> IdealizedInitializationSelection {
        self.selection
    }

    /// The preserved `isnd=7` byte stream, or `None` for an analytic lane.
    pub fn input_sounding(&self) -> Option<&InputSounding> {
        self.input_sounding.as_ref()
    }

    pub fn surface_pressure_pa(&self) -> f32 {
        self.surface_pressure_pa
    }

    pub fn surface_theta_k(&self) -> f32 {
        self.surface_theta_k
    }

    pub fn shape_zyx(&self) -> [usize; 3] {
        self.shape_zyx
    }

    pub fn mass_heights_m(&self) -> &[f32] {
        &self.mass_heights_m
    }

    pub fn theta_base_k(&self) -> &[f32] {
        &self.theta_base_k
    }

    pub fn qv_base_kg_per_kg(&self) -> &[f32] {
        &self.qv_base_kg_per_kg
    }

    pub fn u_base_m_per_s(&self) -> &[f32] {
        &self.u_base_m_per_s
    }

    pub fn v_base_m_per_s(&self) -> &[f32] {
        &self.v_base_m_per_s
    }

    pub fn pi_base(&self) -> &[f32] {
        &self.pi_base
    }

    pub fn pressure_base_pa(&self) -> &[f32] {
        &self.pressure_base_pa
    }

    pub fn temperature_base_k(&self) -> &[f32] {
        &self.temperature_base_k
    }

    pub fn relative_humidity_base(&self) -> &[f32] {
        &self.relative_humidity_base
    }

    pub fn warm_bubble(&self) -> &WarmBubbleFrame {
        &self.warm_bubble
    }

    /// `init3d.F:1424-1441`'s rescaled `qa(:,:,:,nqv)`, when `maintain_rh` is
    /// set.  `None` means CM1's default: the vapour field is exactly `qv0`.
    pub fn initial_qv_kg_per_kg(&self) -> Option<&[f32]> {
        self.initial_qv_kg_per_kg.as_deref()
    }

    /// `init3d.F:625-645`'s `qa(nqc)` after the `iinit=4` bubble.
    pub fn initial_qc_kg_per_kg(&self) -> Option<&[f32]> {
        self.initial_qc_kg_per_kg.as_deref()
    }

    /// `base.F`'s `qc0` (physical words).
    pub fn qc_base_kg_per_kg(&self) -> &[f32] {
        &self.qc_base_kg_per_kg
    }

    /// `base.F`'s `qi0` (physical words).
    pub fn qi_base_kg_per_kg(&self) -> &[f32] {
        &self.qi_base_kg_per_kg
    }

    /// The analytic arm's surface record, if this frame came from one.
    pub fn analytic_surface(&self) -> Option<crate::analytic_soundings::AnalyticSurface> {
        self.analytic_surface
    }

    /// Whether `init3d.F:295-317` copies `qc0`/`qi0` into `qa` for this
    /// sounding (CM1 does so under `imoist = 1` and `ptype >= 1`, which every
    /// moist GM1 run is).
    pub fn initial_condensate_copies(&self) -> (bool, bool) {
        let isnd = self.selection.requested_isnd();
        let moist = self.selection.imoist() == 1;
        (
            moist && crate::analytic_soundings::ISND_WITH_INITIAL_CLOUD.contains(&isnd),
            moist
                && self.selection.vortex_environment().iice
                && crate::analytic_soundings::ISND_WITH_INITIAL_ICE.contains(&isnd),
        )
    }

    /// Project a horizontally uniform idealized base into the existing exact
    /// moist-initialization seam without inventing condensate species.
    ///
    /// CM1 initializes `qa(:,:,:,nqv)=qv0` before applying `iinit`; the other
    /// five WSM6 species remain exact positive zero for this `isnd=7` lane.
    /// Terrain-varying bases fail closed here because the existing downstream
    /// moist seam intentionally accepts one `pi0`/`th0` profile.  They require
    /// a separate full-column completion rather than an implicit flattening.
    pub fn flat_moist_inputs(&self) -> Result<FlatIdealizedMoistInputs> {
        let [nz, ny, nx] = self.shape_zyx;
        let horizontal_words = ny
            .checked_mul(nx)
            .ok_or_else(|| invalid("idealized horizontal shape product overflows"))?;
        let words = checked_product(self.shape_zyx, "idealized moist input")?;
        let mut pi0_profile = vec![0.0_f32; nz];
        let mut th0_profile = vec![0.0_f32; nz];
        for k in 0..nz {
            let first = k * horizontal_words;
            pi0_profile[k] = self.pi_base[first];
            th0_profile[k] = self.theta_base_k[first];
            for horizontal in 1..horizontal_words {
                let index = first + horizontal;
                if self.pi_base[index].to_bits() != pi0_profile[k].to_bits()
                    || self.theta_base_k[index].to_bits() != th0_profile[k].to_bits()
                    || self.qv_base_kg_per_kg[index].to_bits()
                        != self.qv_base_kg_per_kg[first].to_bits()
                {
                    return Err(invalid(format!(
                        "idealized flat moist projection refuses horizontally varying pi0/th0/qv0 at k={k}, horizontal word {horizontal}"
                    )));
                }
            }
        }
        let moisture_words = 6usize
            .checked_mul(words)
            .ok_or_else(|| invalid("idealized packed moisture length overflows"))?;
        let mut packed_moisture = vec![0.0_f32; moisture_words];
        packed_moisture[..words].copy_from_slice(
            self.initial_qv_kg_per_kg
                .as_deref()
                .unwrap_or(&self.qv_base_kg_per_kg),
        );
        // init3d.F:295-317: qa(nqc) = qc0 for isnd 4, 9, 10, 11, 15 and
        // qa(nqi) = qi0 for isnd 4, 9, 10 with iice (both GM1 schemes carry
        // ice), whenever imoist=1 and ptype >= 1.  Then init3d.F:625-645's
        // iinit=4 bubble replaces qa(nqc).
        let (copy_qc, copy_qi) = self.initial_condensate_copies();
        if copy_qc {
            packed_moisture[words..2 * words].copy_from_slice(&self.qc_base_kg_per_kg);
        }
        if let Some(values) = self.initial_qc_kg_per_kg.as_deref() {
            packed_moisture[words..2 * words].copy_from_slice(values);
        }
        if copy_qi {
            packed_moisture[3 * words..4 * words].copy_from_slice(&self.qi_base_kg_per_kg);
        }
        Ok(FlatIdealizedMoistInputs {
            shape_zyx: self.shape_zyx,
            pi0_profile,
            th0_profile,
            theta_perturbation: self.warm_bubble.theta_perturbation_k.clone(),
            packed_moisture,
        })
    }
}

/// `init3d.F:450-1600` in native loop order on C-order `[k,j,i]` words.
///
/// Every `iinit` arm this port implements is dispatched here, followed by the
/// `irandp=1` random field.  `iinit=5`'s conversion through `pi0` and the
/// `maintain_rh` moisture rescaling both need the base state and are applied
/// by the domain builders after it exists.
pub fn build_initial_perturbation(
    selection: IdealizedInitializationSelection,
    controls: WarmBubbleControls,
    xh_m: &[f32],
    yh_m: &[f32],
    zh_agl_m: &[f32],
    shape_zyx: [usize; 3],
    domain_span_m: [f32; 2],
) -> Result<WarmBubbleFrame> {
    controls.validate()?;
    let [nz, ny, nx] = shape_zyx;
    if nz == 0 || ny == 0 || nx == 0 || xh_m.len() != nx || yh_m.len() != ny {
        return Err(invalid(
            "perturbation shape must be positive and match xh/yh coordinate lengths",
        ));
    }
    let words = checked_product(shape_zyx, "perturbation")?;
    if zh_agl_m.len() != words {
        return Err(invalid(format!(
            "perturbation zh length {} does not match shape {shape_zyx:?} ({words} words)",
            zh_agl_m.len()
        )));
    }
    if xh_m
        .iter()
        .chain(yh_m)
        .chain(zh_agl_m)
        .any(|v| !v.is_finite())
    {
        return Err(invalid("perturbation coordinates must be finite"));
    }
    let disposition = match (selection.requested_iinit, selection.effective_iinit) {
        (0, 0) => WarmBubbleDisposition::NotRequested,
        (requested, 0) if requested != 0 && selection.restart_disabled_perturbation() => {
            WarmBubbleDisposition::DisabledByRestart
        }
        (requested, effective) if requested == effective => {
            WarmBubbleDisposition::AppliedIinit1
        }
        pair => {
            return Err(invalid(format!(
                "unadmitted requested/effective iinit pair {pair:?}"
            )))
        }
    };
    // init3d.F:416-443 seeds one generator before the iinit chain; iinit=8 and
    // irandp draw from it in that order.
    let mut stream = if selection.cm1_random() {
        Cm1RandomStream::cm1_gfortran()
    } else {
        Cm1RandomStream::new(selection.random_seed())
    };
    let mut theta_perturbation_k = build_theta_perturbation(
        selection.effective_iinit(),
        controls,
        xh_m,
        yh_m,
        zh_agl_m,
        shape_zyx,
        domain_span_m[0],
        domain_span_m[1],
        &mut stream,
    )?;
    // init3d.F:1239-1256 draws the vortex's own noise before irandp's, and the
    // vortex needs the base state; apply_tc_vortex_arm draws both, in order.
    // init3d.F:579-647 (iinit=4) overwrites tha after the base state exists,
    // and irandp's noise lands on top of it; apply_moist_benchmark_arm draws it.
    if selection.irandp() == 1
        && selection.irst() == 0
        && !matches!(selection.effective_iinit(), 4 | 7)
    {
        let law = random_perturbation_law(selection.testcase())?;
        apply_random_perturbation(law, &mut theta_perturbation_k, zh_agl_m, shape_zyx, &mut stream)?;
    }
    Ok(WarmBubbleFrame {
        shape_zyx,
        theta_perturbation_k,
        disposition,
    })
}

/// The `iinit=1`-only entry this module exposed before the rest of
/// `init3d.F`'s chain was ported.  It is exactly
/// [`build_initial_perturbation`] with no domain span, which only
/// `iinit=13` reads.
pub fn build_warm_bubble(
    selection: IdealizedInitializationSelection,
    controls: WarmBubbleControls,
    xh_m: &[f32],
    yh_m: &[f32],
    zh_agl_m: &[f32],
    shape_zyx: [usize; 3],
) -> Result<WarmBubbleFrame> {
    build_initial_perturbation(
        selection,
        controls,
        xh_m,
        yh_m,
        zh_agl_m,
        shape_zyx,
        [1.0, 1.0],
    )
}

/// Build one physical-domain CM1 `isnd=7` base plus its selected `iinit`
/// perturbation without replacing a single supplied sounding record.
///
/// All three height arrays use C-order `[k,j,i]` with the same physical mass
/// shape.  `u_heights_m` and `v_heights_m` carry the CM1 face-averaged heights
/// used to interpolate the base winds into the mass-shaped `u0`/`v0` arrays.
/// This slice intentionally emits no halos, terrain metrics, A/current copies,
/// tendencies, closure state, pressure solve, or physics state.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn initialize_idealized_domain(
    source_bytes: &[u8],
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
    let [nz, ny, nx] = shape_zyx;
    if nz == 0 || ny == 0 || nx == 0 || xh_m.len() != nx || yh_m.len() != ny {
        return Err(invalid(
            "idealized domain shape must be positive and match xh/yh coordinate lengths",
        ));
    }
    let words = checked_product(shape_zyx, "idealized domain")?;
    for (label, values) in [
        ("mass heights", mass_heights_m),
        ("u heights", u_heights_m),
        ("v heights", v_heights_m),
    ] {
        if values.len() != words {
            return Err(invalid(format!(
                "idealized domain {label} length {} does not match shape {shape_zyx:?} ({words} words)",
                values.len()
            )));
        }
    }

    let input_sounding = InputSounding::parse(source_bytes)?;
    let sounding = input_sounding.derive(selection)?;
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
    let mut mass_column = vec![0.0_f32; nz];
    let mut u_column = vec![0.0_f32; nz];
    let mut v_column = vec![0.0_f32; nz];

    for j in 0..ny {
        for i in 0..nx {
            for k in 0..nz {
                let index = (k * ny + j) * nx + i;
                mass_column[k] = mass_heights_m[index];
                u_column[k] = u_heights_m[index];
                v_column[k] = v_heights_m[index];
            }
            let required_top = mass_column[nz - 1];
            let column =
                sounding.interpolate_column(&mass_column, &u_column, &v_column, required_top)?;
            for k in 0..nz {
                let index = (k * ny + j) * nx + i;
                theta_base_k[index] = column.theta_k[k];
                qv_base_kg_per_kg[index] = column.qv_kg_per_kg[k];
                // base.F:2255-2262: only isnd=7 keeps the file's own winds.
                // isnd=17 reads the same file for thermodynamics and takes
                // the base-state wind from the iwnd branch instead.
                if selection.builds_analytic_wind() {
                    u_base_m_per_s[index] = crate::base_state_wind::base_wind_u(
                        selection.effective_iwnd(),
                        selection.testcase(),
                        u_heights_m[index],
                        selection.hurricane(),
                    );
                    v_base_m_per_s[index] = crate::base_state_wind::base_wind_v(
                        selection.effective_iwnd(),
                        selection.testcase(),
                        v_heights_m[index],
                        selection.hurricane(),
                    );
                } else {
                    u_base_m_per_s[index] = column.u_m_per_s[k];
                    v_base_m_per_s[index] = column.v_m_per_s[k];
                }
                pi_base[index] = column.pi[k];
                pressure_base_pa[index] = column.pressure_pa[k];
                temperature_base_k[index] = column.temperature_k[k];
                relative_humidity_base[index] = column.relative_humidity[k];
            }
        }
    }

    // base.F:2661-2674.  The source reaches this block after the isnd=7
    // file branch above and after every iwnd branch, and subtracts the frame
    // from the base winds exactly once.
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

    let surface = input_sounding.surface();
    Ok(IdealizedInitializationFrame {
        selection,
        surface_pressure_pa: surface.pressure_mb * 100.0_f32,
        surface_theta_k: surface.potential_temperature_k,
        input_sounding: Some(input_sounding),
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
        qc_base_kg_per_kg: vec![0.0_f32; words],
        qi_base_kg_per_kg: vec![0.0_f32; words],
        analytic_surface: None,
        initial_qc_kg_per_kg: None,
    })
}

/// `init3d.F:697-1256`: build the iinit=7 vortex into `frame`, then
/// `init3d.F:1446-1600`'s irandp noise after it, from one stream seeded as
/// init3d.F:416-443 seeds it.  Does nothing for any other arm.
///
/// `xh`/`yh` are the physical mass coordinates the frame was built on,
/// `spacing` CM1's dx/dy/dz and `origin_xy_m` param.F:7212-7213's
/// `xfdp(1)`/`yfdp(1)`.
pub(crate) fn apply_tc_vortex_arm(
    frame: &mut IdealizedInitializationFrame,
    xh: &[f32],
    yh: &[f32],
    spacing: [f32; 3],
    origin_xy_m: [f32; 2],
    vertical: Option<&DeclaredVerticalCoordinate>,
) -> Result<()> {
    let selection = frame.selection;
    if selection.effective_iinit() != 7 {
        return Ok(());
    }
    let [nz, ny, nx] = frame.shape_zyx;
    let [dx, dy, dz] = spacing;
    if frame.warm_bubble.theta_perturbation_k.iter().any(|value| *value != 0.0) {
        return Err(invalid("iinit=7 expects an untouched tha before the vortex is built"));
    }
    if selection.ibalance() != 0 {
        return Err(invalid(
            "init3d.F:800-806 stops an iinit=7 run with ibalance /= 0: the vortex brings its own balanced pressure",
        ));
    }
    // param.F:6649/6659 in double, stored in default REAL: xfdp(n) = origin + dx*(n-1).
    let face = |origin: f32, d: f32, n: usize| -> f32 { (f64::from(origin) + f64::from(d) * n as f64) as f32 };
    let centre = |origin: f32, d: f32, n: usize| -> f32 {
        (f64::from(origin) + f64::from(d) * (n as f64 + 0.5)) as f32
    };
    let xhref: Vec<f32> = (0..nx + 3).map(|n| centre(origin_xy_m[0], dx, n)).collect();
    let xf: Vec<f32> = (0..=nx).map(|n| face(origin_xy_m[0], dx, n)).collect();
    let yf: Vec<f32> = (0..=ny).map(|n| face(origin_xy_m[1], dy, n)).collect();
    // param.F:7406-7408: minx, maxx and centerx in default REAL.
    let (minx, maxx) = (xf[0], xf[nx]);
    let (miny, maxy) = (yf[0], yf[ny]);
    let column = |values: &[f32]| -> Vec<f32> { (0..nz).map(|k| values[k * ny * nx]).collect() };
    let zh = column(&frame.mass_heights_m);
    let th0 = column(&frame.theta_base_k);
    let pi0 = column(&frame.pi_base);
    let qv0 = column(&frame.qv_base_kg_per_kg);
    let prs0 = column(&frame.pressure_base_pa);
    let reps = RV / RD;
    // base.F:2127-2135, the thv0 init3d reads: th0*(1+reps*qv0)/(1+qv0+qc0+qi0).
    // Adding the exact-zero condensate of a sounding without any leaves the
    // words base.F:412/738's th0*(1+reps*qv0)/(1+qv0) gave.
    let qc0 = column(&frame.qc_base_kg_per_kg);
    let qi0 = column(&frame.qi_base_kg_per_kg);
    let thv0: Vec<f32> = (0..nz)
        .map(|k| {
            th0[k] * (1.0_f32 + reps * qv0[k]) / (1.0_f32 + qv0[k] + (qc0[k] + qi0[k]).max(0.0_f32))
        })
        .collect();
    // param.F:7793 (uniform: mf = 1) or :8476 (mf = dz/(zh(k)-zh(k-1)), k = 2..nk; the
    // vortex reads mf(2..nk+1) and multiplies mf(nk+1) by a zero difference).
    let mf: Vec<f32> = match vertical {
        None => vec![1.0_f32; nz + 1],
        Some(coordinate) => {
            let sigma = &coordinate.sigma;
            let mut out = vec![1.0_f32; nz + 1];
            for k in 1..nz {
                out[k] = dz / (sigma[k] - sigma[k - 1]);
            }
            out[nz] = out[nz - 1];
            out[0] = out[1];
            out
        }
    };
    let environment = selection.vortex_environment();
    let inputs = crate::tc_vortex::TcVortexInputs {
        nx,
        ny,
        nz,
        xhref: &xhref,
        xh,
        yh,
        xf: &xf,
        yf: &yf,
        zh: &zh,
        rdz: 1.0_f32 / dz,
        mf: &mf,
        th0: &th0,
        pi0: &pi0,
        qv0: &qv0,
        prs0: &prs0,
        thv0: &thv0,
        fcor: environment.fcor,
        imoist: selection.imoist() == 1,
        iice: environment.iice,
        periodic_x: environment.periodic_x,
        periodic_y: environment.periodic_y,
        xref: minx + 0.5_f32 * (maxx - minx),
        yref: miny + 0.5_f32 * (maxy - miny),
        span_x: maxx - minx,
        span_y: maxy - miny,
    };
    let mut stream = if selection.cm1_random() {
        Cm1RandomStream::cm1_gfortran()
    } else {
        Cm1RandomStream::new(selection.random_seed())
    };
    let fields = crate::tc_vortex::build_tc_vortex(
        &inputs,
        crate::tc_vortex::TcVortexControls::official(),
        &mut stream,
    )?;
    let mut tha = fields.tha;
    if selection.irandp() == 1 {
        let law = random_perturbation_law(selection.testcase())?;
        apply_random_perturbation(law, &mut tha, &frame.mass_heights_m, frame.shape_zyx, &mut stream)?;
    }
    frame.warm_bubble.theta_perturbation_k = tha;
    frame.initial_qv_kg_per_kg = if selection.imoist() == 1 { Some(fields.qv) } else { None };
    frame.vortex = Some(VortexCarriers { ppi: fields.ppi, ua: fields.ua, va: fields.va });
    Ok(())
}

/// `init3d.F:579-647`: the Bryan and Fritsch (2002) moist benchmark bubble.
///
/// A dry `cos^2` bubble of `bptpert` (CM1: 2 K at `ric = 0`, `zc = 2 km`,
/// radii 2 km, uniform in y) is turned into a saturated one with the same
/// density: 30 passes of `qv = rh*rslf(prs0,(th0+tha)*pi0)`,
/// `qc = max(qt_mb - qv, 0)` and
/// `tha = (dum1/300 + (1+qt_mb)/(1+qv))*thv0*(1+qv)/(1+reps*qv) - th0`,
/// with `rh = qv0/rslf(prs0,th0*pi0)` the base relative humidity and `thv0`
/// base.F:2127-2135's.  `qt_mb` is `effective_qt_mb` (zero unless `isnd=4`
/// set it, as in CM1).  `irandp`'s noise is drawn after it, from the stream
/// init3d.F:416-443 seeds.
pub(crate) fn apply_moist_benchmark_arm(
    frame: &mut IdealizedInitializationFrame,
    controls: WarmBubbleControls,
    xh: &[f32],
) -> Result<()> {
    let selection = frame.selection;
    if selection.effective_iinit() != 4 {
        return Ok(());
    }
    if selection.imoist() != 1 {
        return Err(invalid(
            "init3d.F:614-647 writes qa(:,:,:,nqv) and qa(:,:,:,nqc), which a dry CM1 run \
             (imoist=0) does not allocate; the moist benchmark bubble needs imoist=1",
        ));
    }
    if controls.maintain_relative_humidity() {
        return Err(invalid(
            "init3d.F:579-647 never sets maintain_rh: the moist benchmark arm holds relative \
             humidity itself, and init3d.F:1422's rescaling after it would replace the \
             saturated bubble's vapour with another field",
        ));
    }
    if selection.ibalance() == 2 {
        // Admitted: ibalance=2 runs after the arm on qa, as in CM1.
    }
    let [nz, ny, nx] = frame.shape_zyx;
    if xh.len() != nx {
        return Err(invalid("moist benchmark xh must carry nx words"));
    }
    if frame.warm_bubble.theta_perturbation_k.iter().any(|value| *value != 0.0) {
        return Err(invalid("iinit=4 expects an untouched tha before the bubble is built"));
    }
    let qt_mb = selection.effective_qt_mb();
    let reps = RV / RD;
    let ric = controls.center_x_m();
    let zc = controls.center_z_agl_m();
    let bhrad = controls.horizontal_radius_m();
    let bvrad = controls.vertical_radius_m();
    let bptpert = controls.amplitude_k();
    let words = nz * ny * nx;
    let mut tha = vec![0.0_f32; words];
    let mut qv = vec![0.0_f32; words];
    let mut qc = vec![0.0_f32; words];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let index = index3(k, j, i, ny, nx);
                let zh = frame.mass_heights_m[index];
                let th0 = frame.theta_base_k[index];
                let pi0 = frame.pi_base[index];
                let prs0 = frame.pressure_base_pa[index];
                let qv0 = frame.qv_base_kg_per_kg[index];
                let qc0 = frame.qc_base_kg_per_kg[index];
                let qi0 = frame.qi_base_kg_per_kg[index];
                // base.F:2127-2135, the thv0 init3d reads.
                let qvp = qv0.max(0.0_f32);
                let thv0 = (th0 + 0.0_f32) * (1.0_f32 + reps * qvp)
                    / (1.0_f32 + qvp + (qc0 + qi0).max(0.0_f32));
                // init3d.F:593-603.
                let rx = (xh[i] - ric) / bhrad;
                let rz = (zh - zc) / bvrad;
                let beta = (rx * rx + rz * rz).sqrt();
                let dum1 = if beta < 1.0_f32 {
                    let cosine = (0.5_f32 * std::f32::consts::PI * beta).cos();
                    bptpert * (cosine * cosine)
                } else {
                    0.0_f32
                };
                // init3d.F:605-612.
                let mut t = 0.0_f32;
                let dum2 = qv0 / rslf(prs0, th0 * pi0);
                let mut q = 0.0_f32;
                // init3d.F:614-640.
                for _ in 0..30 {
                    q = dum2 * rslf(prs0, (th0 + t) * pi0);
                    t = ((dum1 / 300.0_f32) + (1.0_f32 + qt_mb) / (1.0_f32 + q)) * thv0
                        * (1.0_f32 + q)
                        / (1.0_f32 + reps * q)
                        - th0;
                    if t.abs() < 1.0e-4_f32 {
                        t = 0.0_f32;
                    }
                }
                let _ = q;
                // init3d.F:642-647.
                let q = rslf(prs0, (th0 + t) * pi0);
                tha[index] = t;
                qv[index] = q;
                qc[index] = (qt_mb - q).max(0.0_f32);
            }
        }
    }
    if selection.irandp() == 1 {
        let mut stream = if selection.cm1_random() {
            Cm1RandomStream::cm1_gfortran()
        } else {
            Cm1RandomStream::new(selection.random_seed())
        };
        let law = random_perturbation_law(selection.testcase())?;
        apply_random_perturbation(law, &mut tha, &frame.mass_heights_m, frame.shape_zyx, &mut stream)?;
    }
    frame.warm_bubble.theta_perturbation_k = tha;
    frame.initial_qv_kg_per_kg = Some(qv);
    frame.initial_qc_kg_per_kg = Some(qc);
    Ok(())
}

/// The `maxx-minx` and `maxy-miny` of `init3d.F:1406-1407`, from the mass
/// centres CM1 builds them from (`param.F:7212-7213`).
pub fn domain_span(xh_m: &[f32], yh_m: &[f32]) -> [f32; 2] {
    let span = |axis: &[f32]| -> f32 {
        if axis.len() < 2 {
            return 1.0_f32;
        }
        let first = axis[0];
        let last = axis[axis.len() - 1];
        let cell = (last - first) / ((axis.len() - 1) as f32);
        last - first + cell
    };
    [span(xh_m), span(yh_m)]
}

/// Run `init3d.F`'s two base-dependent finishing steps and report the vapour
/// field they leave behind.
///
/// `iinit=5` (`init3d.F:654-695`) converts a temperature perturbation through
/// `pi0`, and `maintain_rh` (`init3d.F:1424-1441`) rescales `qv` so relative
/// humidity, not mixing ratio, is what the perturbation preserves.  Returning
/// `None` means the vapour field is exactly `qv0`, which is CM1's default.
#[allow(clippy::too_many_arguments)]
pub fn finish_base_dependent_perturbation(
    selection: IdealizedInitializationSelection,
    controls: WarmBubbleControls,
    xh_m: &[f32],
    mass_heights_m: &[f32],
    pi0: &[f32],
    prs0: &[f32],
    th0: &[f32],
    qv0: &[f32],
    theta_perturbation: &mut [f32],
    shape_zyx: [usize; 3],
) -> Result<Option<Vec<f32>>> {
    apply_base_dependent_perturbation(
        selection.effective_iinit(),
        controls,
        xh_m,
        mass_heights_m,
        pi0,
        th0,
        theta_perturbation,
        shape_zyx,
    )?;
    if !controls.maintain_relative_humidity() {
        return Ok(None);
    }
    if selection.imoist() == 0 {
        return Err(invalid(
            "init3d.F:1422 guards the maintain_rh block with imoist=1; a dry run has no vapour              to hold at constant relative humidity, so honouring the request would divide by a              zero saturation ratio and emit a moisture field in a run declared dry",
        ));
    }
    let mut qv = qv0.to_vec();
    apply_maintain_relative_humidity(&mut qv, qv0, prs0, th0, pi0, theta_perturbation)?;
    Ok(Some(qv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    const SOUNDING: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../upstream/cm1-21.1/pristine/run/config_files/sea_breeze/input_sounding"
    ));

    fn repository() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("cm1-state is nested under repository/rust")
            .to_path_buf()
    }

    fn selection() -> IdealizedInitializationSelection {
        IdealizedInitializationSelection::new(7, 4, 1, 0, 1).unwrap()
    }

    #[test]
    fn supplied_words_and_source_bytes_are_preserved() {
        let input = InputSounding::parse(SOUNDING).unwrap();
        assert_eq!(input.source_bytes(), SOUNDING);
        assert_eq!(
            input.source_sha256(),
            "56df89107f712b4911a7eb53a5c4df637826ee02ca14fe5322eb0cd2916bdea0"
        );
        assert_eq!(input.surface().pressure_mb.to_bits(), 1000.0_f32.to_bits());
        assert_eq!(input.levels().len(), 80);
        assert_eq!(
            input.levels()[1].water_vapor_g_per_kg.to_bits(),
            5.0_f32.to_bits()
        );
    }

    #[test]
    fn source_ordered_sounding_and_column_words_are_sealed() {
        let input = InputSounding::parse(SOUNDING).unwrap();
        let sounding = input.derive(selection()).unwrap();
        assert_eq!(sounding.u_m_per_s()[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(sounding.v_m_per_s()[0].to_bits(), 0.0_f32.to_bits());
        let column = sounding
            .interpolate_column(
                &[250.0_f32, 1000.0_f32, 2500.0_f32],
                &[260.0_f32, 1010.0_f32, 2510.0_f32],
                &[240.0_f32, 990.0_f32, 2490.0_f32],
                2500.0_f32,
            )
            .unwrap();
        let actual = [
            sounding.pi_surface().to_bits(),
            sounding.theta_v_surface().to_bits(),
            sounding.pressure_pa()[1].to_bits(),
            sounding.relative_humidity()[2].to_bits(),
            column.theta_k()[0].to_bits(),
            column.qv_kg_per_kg()[1].to_bits(),
            column.u_m_per_s()[2].to_bits(),
            column.v_m_per_s()[0].to_bits(),
            column.pi()[0].to_bits(),
            column.pi()[2].to_bits(),
            column.pressure_pa()[2].to_bits(),
            column.relative_humidity()[2].to_bits(),
        ];
        let expected = [
            0x3f80_0000,
            0x4396_b44f,
            0x47c0_8ee8,
            0x3e81_efaf,
            0x4396_443a,
            0x3ba4_0339,
            0x0000_0000,
            0x0000_0000,
            0x3f7d_edcd,
            0x3f6b_7228,
            0x4791_abf1,
            0x3f0a_0d07,
        ];
        assert_eq!(
            actual, expected,
            "seal these source-order words: {actual:08x?}"
        );
    }

    #[test]
    fn official_warm_bubble_words_are_sealed_and_restart_disables_it() {
        let controls = WarmBubbleControls::official(0.0, 0.0).unwrap();
        let active = build_warm_bubble(
            selection(),
            controls,
            &[0.0, 5000.0, 10_000.0],
            &[0.0],
            &[1400.0, 1400.0, 1400.0],
            [1, 1, 3],
        )
        .unwrap();
        assert_eq!(active.disposition(), WarmBubbleDisposition::AppliedIinit1);
        let actual: Vec<u32> = active
            .theta_perturbation_k()
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(
            actual,
            vec![0x3f80_0000, 0x3eff_ffff, 0x0000_0000],
            "sealed init3d.F warm-bubble binary32 words changed: {actual:08x?}"
        );

        let restart = IdealizedInitializationSelection::new(7, 9, 1, 1, 1).unwrap();
        assert_eq!(restart.effective_iwnd(), 0);
        assert_eq!(restart.effective_iinit(), 0);
        let disabled = build_warm_bubble(
            restart,
            controls,
            &[0.0, 5000.0, 10_000.0],
            &[0.0],
            &[1400.0, 1400.0, 1400.0],
            [1, 1, 3],
        )
        .unwrap();
        assert_eq!(
            disabled.disposition(),
            WarmBubbleDisposition::DisabledByRestart
        );
        assert!(disabled
            .theta_perturbation_k()
            .iter()
            .all(|value| value.to_bits() == 0));
    }

    #[test]
    fn unsupported_or_ambiguous_inputs_fail_closed() {
        for bad in [
            // base.F:455-460 stops CM1 for isnd=6; isnd=15 needs testcase
            // 4/5's constants; 16 has no arm at all.
            IdealizedInitializationSelection::new(6, 0, 1, 0, 1),
            IdealizedInitializationSelection::new(15, 0, 1, 0, 1),
            IdealizedInitializationSelection::new(16, 6, 0, 0, 1),
            // init3d.F has no arm 6, and a value with no arm.
            IdealizedInitializationSelection::new(7, 0, 6, 0, 1),
            IdealizedInitializationSelection::new(7, 0, 99, 0, 1),
            // README.namelist:308 admits imoist 0 or 1 and nothing else.
            IdealizedInitializationSelection::new(7, 0, 1, 0, 2),
        ] {
            assert!(bad.is_err());
        }
        // isnd=17, every analytic base.F arm, imoist=0 and every implemented
        // iinit are now admitted.
        IdealizedInitializationSelection::new(17, 6, 1, 0, 1).unwrap();
        for isnd in crate::analytic_soundings::TRANSCRIBED_ANALYTIC_ISND {
            let selection = IdealizedInitializationSelection::new(isnd, 2, 4, 0, 1).unwrap();
            assert_eq!(selection.lane(), IdealizedBaseStateLane::AnalyticBaseF);
        }
        IdealizedInitializationSelection::new(7, 0, 1, 0, 0).unwrap();
        for iinit in IMPLEMENTED_IINIT {
            IdealizedInitializationSelection::new(7, 0, iinit, 0, 1).unwrap();
        }
        assert!(InputSounding::parse(b"1000 300 10\n500 301 9 1 2\n").is_err());
        assert!(InputSounding::parse(b"1000 300 10\n500 301 9 1 2\n500 302 8 2 3\n").is_err());
        assert!(
            InputSounding::parse(b"1000 300 10 extra\n500 301 9 1 2\n1000 302 8 2 3\n").is_err()
        );
        // init3d.F:1424-1441's maintain_rh is now built, so the guard that
        // used to refuse it is retired: the controls admit it by value.
        WarmBubbleControls::new(0.0, 0.0, 1400.0, 10_000.0, 1400.0, 1.0, true).unwrap();
        // It still needs vapour to hold at constant relative humidity.
        let dry = IdealizedInitializationSelection::new(7, 0, 1, 0, 0).unwrap();
        let mut tha = vec![0.0_f32];
        let error = finish_base_dependent_perturbation(
            dry,
            WarmBubbleControls::new(0.0, 0.0, 1400.0, 10_000.0, 1400.0, 1.0, true).unwrap(),
            &[0.0],
            &[250.0],
            &[1.0],
            &[100_000.0],
            &[300.0],
            &[0.0],
            &mut tha,
            [1, 1, 1],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("declared dry"), "{error}");

        let sounding = InputSounding::parse(SOUNDING)
            .unwrap()
            .derive(selection())
            .unwrap();
        let error = sounding
            .interpolate_column(&[250.0], &[250.0], &[250.0], 30_000.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not cover"), "{error}");
    }

    #[test]
    fn fortran_d_exponents_parse_without_changing_provenance() {
        let bytes =
            b"1.0D3 3.0D2 1.0D1\n5.0D2 3.01D2 9.0D0 1.0D0 2.0D0\n1.0D3 3.02D2 8.0D0 2.0D0 3.0D0\n";
        let input = InputSounding::parse(bytes).unwrap();
        assert_eq!(input.source_bytes(), bytes);
        assert_eq!(input.surface().pressure_mb.to_bits(), 1000.0_f32.to_bits());
        assert_eq!(input.levels()[1].u_m_per_s.to_bits(), 2.0_f32.to_bits());
    }

    #[test]
    fn official_source_authority_and_sounding_fixture_are_pinned() {
        let root = repository();
        for (relative, expected) in [
            (
                "upstream/cm1-21.1/pristine/src/base.F",
                CM1_R21_BASE_F_SHA256,
            ),
            (
                "upstream/cm1-21.1/pristine/src/init3d.F",
                CM1_R21_INIT3D_F_SHA256,
            ),
            (
                "upstream/cm1-21.1/pristine/src/param.F",
                CM1_R21_PARAM_F_SHA256,
            ),
        ] {
            let bytes = fs::read(root.join(relative)).unwrap();
            assert_eq!(sha256::digest_hex(&bytes), expected, "{relative}");
        }
        assert_eq!(
            sha256::digest_hex(SOUNDING),
            "56df89107f712b4911a7eb53a5c4df637826ee02ca14fe5322eb0cd2916bdea0"
        );
        assert_eq!(
            IDEALIZED_INITIALIZATION_SCHEMA,
            "gm1.cm1-r21.1-isnd7-iinit1-initialization/v1"
        );
        assert!(IDEALIZED_CLASSIFICATION.contains("idealized"));
        assert!(IDEALIZED_CLAIM_LIMIT.contains("no dynamics"));
    }

    fn raw_sha256(values: &[f32]) -> String {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        sha256::digest_hex(&bytes)
    }

    #[test]
    fn combined_official_sounding_and_warm_bubble_domain_is_selectable_and_sealed() {
        let frame = initialize_idealized_domain(
            SOUNDING,
            selection(),
            WarmBubbleControls::official(5000.0, 0.0).unwrap(),
            &[0.0, 5000.0, 10_000.0],
            &[0.0],
            &[250.0, 300.0, 350.0, 1000.0, 1100.0, 1200.0],
            &[260.0, 310.0, 360.0, 1010.0, 1110.0, 1210.0],
            &[240.0, 290.0, 340.0, 990.0, 1090.0, 1190.0],
            [2, 1, 3],
            TranslatingFrame::fixed(),
        )
        .unwrap();
        assert_eq!(frame.shape_zyx(), [2, 1, 3]);
        assert_eq!(frame.input_sounding().unwrap().source_bytes(), SOUNDING);
        assert_eq!(frame.selection().effective_iwnd(), 0);
        assert_eq!(
            frame.warm_bubble().disposition(),
            WarmBubbleDisposition::AppliedIinit1
        );
        let actual = [
            raw_sha256(frame.theta_base_k()),
            raw_sha256(frame.qv_base_kg_per_kg()),
            raw_sha256(frame.u_base_m_per_s()),
            raw_sha256(frame.v_base_m_per_s()),
            raw_sha256(frame.pi_base()),
            raw_sha256(frame.pressure_base_pa()),
            raw_sha256(frame.temperature_base_k()),
            raw_sha256(frame.relative_humidity_base()),
            raw_sha256(frame.warm_bubble().theta_perturbation_k()),
        ];
        let expected = [
            "01ad14f60cb1851a6d316b21f1be7f5b6a7750fc6aa64db9fb8f32f3e335c533",
            "bed78b5a3d6bb384ff4c419c7e2b8a3f679ecbb4421a2c4059bf6faeda7d0032",
            "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0",
            "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0",
            "01606ac2a3bb8e4aaf649c1e0a3318c66c3845b8b5c337a8b40285b63fe3a812",
            "c288b739eaa05ae617b72b6be4f804fd73d3e4f4768e765cfc97c9bdbd72e66b",
            "3d7a6a3057dc19cd503dc37e5435d5cb49846a9ef881a0a441942509a6c4634a",
            "e5e526fac0b868a606a29f5c817da88529470de17e909fb63e965bb8e798cbd8",
            "cb86bd042dbea90d8cfbb3b236c073a44b9e023c970076ba8b3dababda7fc000",
        ]
        .map(str::to_owned);
        assert_eq!(actual, expected, "seal combined domain hashes: {actual:#?}");
    }

    #[test]
    fn flat_domain_projects_exact_qv_first_moisture_inputs_and_terrain_refuses() {
        let shape = [2, 1, 3];
        let flat_heights = &[250.0, 250.0, 250.0, 1000.0, 1000.0, 1000.0];
        let frame = initialize_idealized_domain(
            SOUNDING,
            selection(),
            WarmBubbleControls::official(5000.0, 0.0).unwrap(),
            &[0.0, 5000.0, 10_000.0],
            &[0.0],
            flat_heights,
            flat_heights,
            flat_heights,
            shape,
            TranslatingFrame::fixed(),
        )
        .unwrap();
        let inputs = frame.flat_moist_inputs().unwrap();
        assert_eq!(inputs.shape_zyx(), shape);
        assert_eq!(inputs.pi0_profile().len(), 2);
        assert_eq!(inputs.th0_profile().len(), 2);
        assert_eq!(
            inputs.theta_perturbation(),
            frame.warm_bubble().theta_perturbation_k()
        );
        assert_eq!(&inputs.packed_moisture()[..6], frame.qv_base_kg_per_kg());
        assert!(inputs.packed_moisture()[6..]
            .iter()
            .all(|value| value.to_bits() == 0));
        assert_eq!(
            raw_sha256(inputs.packed_moisture()),
            "c99c4b4189a75fd49174682c1b958126af48993670de7659ce6111323bf18aa6"
        );

        let varying = initialize_idealized_domain(
            SOUNDING,
            selection(),
            WarmBubbleControls::official(5000.0, 0.0).unwrap(),
            &[0.0, 5000.0, 10_000.0],
            &[0.0],
            &[250.0, 300.0, 350.0, 1000.0, 1100.0, 1200.0],
            &[250.0, 300.0, 350.0, 1000.0, 1100.0, 1200.0],
            &[250.0, 300.0, 350.0, 1000.0, 1100.0, 1200.0],
            shape,
            TranslatingFrame::fixed(),
        )
        .unwrap();
        let error = varying.flat_moist_inputs().unwrap_err().to_string();
        assert!(error.contains("refuses horizontally varying"), "{error}");
    }

    #[test]
    fn selectable_flat_entry_builds_existing_padded_current_a_and_moist_layouts() {
        let selection = IdealizedInitializationSelection::new(7, 9, 1, 0, 1).unwrap();
        let state = initialize_flat_idealized_padded_state(
            SOUNDING,
            selection,
            WarmBubbleControls::official(1500.0, 1000.0).unwrap(),
            [4, 2, 3],
            [1000.0, 1000.0, 500.0],
            [0.0, 0.0],
            TranslatingFrame::fixed(),
        )
        .unwrap();
        assert_eq!(state.selection(), selection);
        assert_eq!(
            state.input_sounding_sha256(),
            Some(sha256::digest_hex(SOUNDING).as_str())
        );
        assert_eq!(state.shape_zyx(), [4, 2, 3]);
        assert_eq!(state.spacing_m(), [1000.0, 1000.0, 500.0]);
        assert_eq!(
            state.warm_bubble_disposition(),
            WarmBubbleDisposition::AppliedIinit1
        );
        assert_eq!(state.pi_recovery_mismatch_count(), 0);
        assert_eq!(state.theta_recovery_mismatch_count(), 0);
        assert_eq!(state.terrain_open_carriers().len(), 54);
        // The fifty established base and mutable carriers, in their
        // established order, followed by init3d.F:2386-2466's four
        // initialized prs/rho/rr/rf.
        assert!(state
            .terrain_open_carriers()
            .iter()
            .map(IdealizedPaddedCarrier::name)
            .take(TERRAIN_OPEN_CARRIER_NAMES.len())
            .eq(TERRAIN_OPEN_CARRIER_NAMES));
        assert_eq!(
            state
                .terrain_open_carriers()
                .iter()
                .map(IdealizedPaddedCarrier::name)
                .skip(TERRAIN_OPEN_CARRIER_NAMES.len())
                .collect::<Vec<_>>(),
            vec!["prs", "rho", "rr", "rf"]
        );
        assert!(state
            .moist_carriers()
            .iter()
            .map(IdealizedPaddedCarrier::name)
            .eq(IDEALIZED_MOIST_CARRIER_NAMES));
        assert!(state
            .base_wind_carriers()
            .iter()
            .map(IdealizedPaddedCarrier::name)
            .eq(IDEALIZED_BASE_WIND_CARRIER_NAMES));
        assert_eq!(state.carrier("u0").unwrap().shape(), &[4, 2, 4]);
        assert_eq!(state.carrier("v0").unwrap().shape(), &[4, 3, 3]);

        for (left, right) in [
            ("ppi", "pp3d"),
            ("ppi", "ppx"),
            ("tha", "th3d"),
            ("ua", "u3d"),
            ("va", "v3d"),
            ("wa", "w3d"),
            ("qa", "q3d"),
        ] {
            let left = state.carrier(left).unwrap();
            let right = state.carrier(right).unwrap();
            assert_eq!(left.shape(), right.shape());
            assert!(left
                .values()
                .iter()
                .zip(right.values())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        let qa = state.carrier("qa").unwrap();
        assert_eq!(qa.shape(), &[6, 6, 8, 9]);
        let species_words = 6 * 8 * 9;
        assert!(qa.values()[species_words..]
            .iter()
            .all(|value| value.to_bits() == 0));
        assert!(state
            .carrier("qten")
            .unwrap()
            .values()
            .iter()
            .chain(state.carrier("qten1").unwrap().values())
            .all(|value| value.to_bits() == 0));
        assert!(state
            .carrier("terrain_msl_halo")
            .unwrap()
            .values()
            .iter()
            .all(|value| value.to_bits() == 0));
        assert!(state
            .terrain_static()
            .carrier("zh")
            .unwrap()
            .values()
            .iter()
            .enumerate()
            .all(|(word, value)| {
                let k = word / (2 * 3);
                value.to_bits() == ((k as f32 + 0.5_f32) * 500.0_f32).to_bits()
            }));

        let selected_hashes = ["pi0", "ppi", "tha", "ua", "qa"]
            .map(|name| (name, state.carrier(name).unwrap().raw_sha256()));
        assert_eq!(
            selected_hashes,
            [
                (
                    "pi0",
                    "7be5ea48e48b6e41ada959266dcbec2fb1c635de3fd975585645067cb29a5f93".to_owned(),
                ),
                (
                    "ppi",
                    "4b48f21a4b7a02bfbec19ef880a967a02334a3cdcef8ae83de2ef327ba8bc5dd".to_owned(),
                ),
                (
                    "tha",
                    "b1e69daba32360f140e6f4d0b459b90e8a711bc9cf794f920288492a2a610bc4".to_owned(),
                ),
                (
                    "ua",
                    "155e437b946ac82ae591ff382b8d19efda9397b2282672dbabd91ec31ce8a651".to_owned(),
                ),
                (
                    "qa",
                    "fd2ade272766091b268e77a7457c8062292f497266befc3dbea60cc82db16a5a".to_owned(),
                ),
            ]
        );

        let restart_selection = IdealizedInitializationSelection::new(7, 0, 1, 1, 1).unwrap();
        let error = initialize_flat_idealized_padded_state(
            SOUNDING,
            restart_selection,
            WarmBubbleControls::official(1500.0, 1000.0).unwrap(),
            [4, 2, 3],
            [1000.0, 1000.0, 500.0],
            [0.0, 0.0],
            TranslatingFrame::fixed(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("refuses irst=1"), "{error}");
    }

    /// zh(1..nk)/zf(1..nk+1) of a two-segment column, param.F:8169 midpoints.
    fn declared_column(thin: usize, thin_dz: f32, thick: usize, thick_dz: f32) -> DeclaredVerticalCoordinate {
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
        DeclaredVerticalCoordinate::new(&zh, &zf, 100.0).unwrap()
    }

    #[test]
    fn declaring_the_uniform_coordinate_reproduces_the_flat_entry_raw_bit_exactly() {
        // param.F has one vertical tail; a declared coordinate that happens to
        // be param.F:7789-7811's own must therefore reach the same words, not
        // merely the same physics.
        let shape = [8_usize, 3, 4];
        let spacing = [1000.0_f64, 1000.0, 500.0];
        let zf = (0..=shape[0]).map(|k| 500.0_f32 * k as f32).collect::<Vec<_>>();
        let zh = (0..shape[0])
            .map(|k| 0.5_f32 * (500.0_f32 * k as f32 + 500.0_f32 * (k + 1) as f32))
            .collect::<Vec<_>>();
        let declared = DeclaredVerticalCoordinate::new(&zh, &zf, 500.0).unwrap();
        let selection = IdealizedInitializationSelection::new(7, 0, 1, 0, 1).unwrap();
        let controls = WarmBubbleControls::official(1500.0, 1000.0).unwrap();
        let flat = initialize_flat_idealized_padded_state(
            SOUNDING, selection, controls, shape, spacing, [0.0, 0.0],
            TranslatingFrame::fixed(),
        )
        .unwrap();
        let levels = initialize_stretched_idealized_padded_state(
            SOUNDING, selection, controls, shape, spacing, [0.0, 0.0],
            TranslatingFrame::fixed(), &declared,
        )
        .unwrap();
        assert_eq!(flat.terrain_static().bottom_fornberg(), levels.terrain_static().bottom_fornberg());
        assert_eq!(flat.terrain_static().top_fornberg(), levels.terrain_static().top_fornberg());
        for carrier in flat.terrain_static().carriers() {
            let other = levels
                .terrain_static()
                .carrier(carrier.name())
                .unwrap_or_else(|| panic!("declared state lacks {}", carrier.name()));
            assert_eq!(
                carrier.values().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                other.values().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "terrain-static carrier {}", carrier.name()
            );
        }
        for carrier in flat.terrain_open_carriers() {
            let other = levels
                .terrain_open_carriers()
                .iter()
                .find(|candidate| candidate.name() == carrier.name())
                .unwrap_or_else(|| panic!("declared state lacks {}", carrier.name()));
            assert_eq!(
                carrier.values().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                other.values().iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "terrain-open carrier {}", carrier.name()
            );
        }
    }

    #[test]
    fn a_declared_stretched_column_moves_the_levels_metrics_and_fornberg_weights() {
        // 4 layers of 20 m then 8 of 320 m under a 2640 m top.  This is the
        // shape a tornado-grade deck has: the first scalar level is 10 m, not
        // 0.5*dz for any dz that also reaches the model top.
        let declared = declared_column(4, 20.0, 8, 320.0);
        let nz = declared.sigma.len();
        assert_eq!(declared.sigma[0].to_bits(), 10.0_f32.to_bits());
        let state = initialize_stretched_idealized_padded_state(
            SOUNDING,
            IdealizedInitializationSelection::new(7, 0, 1, 0, 1).unwrap(),
            WarmBubbleControls::official(1500.0, 1000.0).unwrap(),
            [nz, 3, 4],
            [1000.0, 1000.0, 100.0],
            [0.0, 0.0],
            TranslatingFrame::fixed(),
            &declared,
        )
        .unwrap();
        let stat = state.terrain_static();
        let zh = stat.carrier("zh").unwrap().values();
        let zf = stat.carrier("zf").unwrap().values();
        let mh = stat.carrier("mh").unwrap().values();
        for k in 0..nz {
            for word in 0..12 {
                assert_eq!(zh[k * 12 + word].to_bits(), declared.sigma[k].to_bits());
            }
        }
        for k in 0..=nz {
            assert_eq!(zf[k * 12].to_bits(), declared.sigmaf[k].to_bits());
        }
        // param.F:8170.  dz=100 over a 20 m layer is 5, over a 320 m layer
        // is 0.3125; a uniform grid would have had 1.0 everywhere.
        assert_eq!(mh[0].to_bits(), 5.0_f32.to_bits());
        assert_eq!(mh[(nz - 1) * 12].to_bits(), 0.3125_f32.to_bits());
        // param.F:8985-9060 over uniformly spaced lowest and highest three
        // scalar levels is exactly (1.875, -1.25, 0.375).  Four 20 m layers
        // keep the bottom three uniform; eight 320 m layers keep the top
        // three uniform.
        assert_eq!(stat.bottom_fornberg(), [1.875, -1.25, 0.375]);
        assert_eq!(stat.top_fornberg(), [1.875, -1.25, 0.375]);
    }

    #[test]
    fn fornberg_weights_leave_the_pinned_triple_when_the_lowest_levels_stretch() {
        // The refusal that matters: a column that starts stretching inside
        // the first three layers no longer has the quadratic the compiled
        // kernels carry inline, so rf0 at the ground would be extrapolated
        // from a fit to the wrong three levels.
        let zf = vec![0.0_f32, 20.0, 60.0, 140.0, 300.0, 620.0, 940.0, 1260.0, 1580.0];
        let zh = zf
            .windows(2)
            .map(|window| 0.5_f32 * (window[0] + window[1]))
            .collect::<Vec<_>>();
        let declared = DeclaredVerticalCoordinate::new(&zh, &zf, 100.0).unwrap();
        let state = initialize_stretched_idealized_padded_state(
            SOUNDING,
            IdealizedInitializationSelection::new(7, 0, 1, 0, 1).unwrap(),
            WarmBubbleControls::official(1500.0, 1000.0).unwrap(),
            [zh.len(), 3, 4],
            [1000.0, 1000.0, 100.0],
            [0.0, 0.0],
            TranslatingFrame::fixed(),
            &declared,
        )
        .unwrap();
        let bottom = state.terrain_static().bottom_fornberg();
        assert_ne!(bottom, [1.875, -1.25, 0.375]);
        // Fornberg's zeroth-derivative weights are an interpolation, so they
        // still sum to one and still reproduce the fitted linear function.
        assert!((bottom.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{bottom:?}");
        let predicted = bottom[0] * zh[0] + bottom[1] * zh[1] + bottom[2] * zh[2];
        assert!(predicted.abs() < 1e-3, "extrapolated ground height {predicted}");
    }

    #[test]
    fn a_declared_coordinate_refuses_by_the_exact_defect() {
        let declared = declared_column(4, 20.0, 8, 320.0);
        let nz = declared.sigma.len();
        let selection = IdealizedInitializationSelection::new(7, 0, 1, 0, 1).unwrap();
        let controls = WarmBubbleControls::official(1500.0, 1000.0).unwrap();
        // param.F:8170 has exactly one dz.
        let error = initialize_stretched_idealized_padded_state(
            SOUNDING, selection, controls, [nz, 3, 4], [1000.0, 1000.0, 500.0],
            [0.0, 0.0], TranslatingFrame::fixed(), &declared,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("dz reference"), "{error}");
        // The level count is the grid's own.
        let error = initialize_stretched_idealized_padded_state(
            SOUNDING, selection, controls, [nz + 1, 3, 4], [1000.0, 1000.0, 100.0],
            [0.0, 0.0], TranslatingFrame::fixed(), &declared,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("scalar levels"), "{error}");
        let mut folded = declared.sigmaf.clone();
        folded[3] = folded[2];
        assert!(
            DeclaredVerticalCoordinate::new(&declared.sigma, &folded, 100.0)
                .unwrap_err()
                .to_string()
                .contains("non-positive depth")
        );
        let mut sunken = declared.sigmaf.clone();
        sunken[0] = 5.0;
        assert!(
            DeclaredVerticalCoordinate::new(&declared.sigma, &sunken, 100.0)
                .unwrap_err()
                .to_string()
                .contains("ground")
        );
    }

    fn sounding_state(frame: TranslatingFrame) -> IdealizedPaddedState {
        initialize_flat_idealized_padded_state(
            SOUNDING,
            IdealizedInitializationSelection::new(7, 0, 1, 0, 1).unwrap(),
            WarmBubbleControls::official(1500.0, 1000.0).unwrap(),
            [8, 3, 4],
            [1000.0, 1000.0, 500.0],
            [0.0, 0.0],
            frame,
        )
        .unwrap()
    }

    /// A profile with level-varying, nonzero winds.
    ///
    /// The pinned `sea_breeze` fixture is calm at every level, so on it a
    /// frame test cannot tell "the interpolated winds were shifted" apart
    /// from "the winds were replaced by minus the frame".  This one can.
    fn sheared_sounding() -> Vec<u8> {
        let mut text = String::from("   1000.0   300.0   14.0\n");
        for level in 0..24 {
            let z = 125.0_f32 + 500.0_f32 * level as f32;
            let theta = 300.0_f32 + 0.004_f32 * z;
            let u = 3.0_f32 + 0.002_f32 * z;
            let v = -4.0_f32 + 0.0005_f32 * z;
            text.push_str(&format!("  {z:.4}  {theta:.4}  8.0  {u:.5}  {v:.5}\n"));
        }
        text.into_bytes()
    }

    #[test]
    fn the_sounding_lanes_interpolated_hodograph_is_shifted_not_replaced() {
        // base.F:2667-2668 subtracts a constant.  A constant subtraction
        // leaves every vertical wind difference unchanged, which is exactly
        // what makes the frame a change of variable rather than a different
        // atmosphere.  A lane that overwrote the winds instead would flatten
        // the shear and the storm would never organise.
        let bytes = sheared_sounding();
        let umove = 12.5_f32;
        let vmove = 3.0_f32;
        let build = |frame: TranslatingFrame| {
            initialize_flat_idealized_padded_state(
                &bytes,
                IdealizedInitializationSelection::new(7, 0, 1, 0, 1).unwrap(),
                WarmBubbleControls::official(1500.0, 1000.0).unwrap(),
                [8, 3, 4],
                [1000.0, 1000.0, 500.0],
                [0.0, 0.0],
                frame,
            )
            .unwrap()
        };
        let fixed = build(TranslatingFrame::fixed());
        let moving = build(TranslatingFrame::new(umove, vmove).unwrap());

        for (name, offset, faces) in [("u0", umove, 3 * 5_usize), ("v0", vmove, 4 * 4)] {
            let still = fixed.carrier(name).unwrap().values();
            let moved = moving.carrier(name).unwrap().values();
            // The profile is genuinely sheared and genuinely nonzero, so
            // neither "all equal" nor "all zero" can pass by accident.
            assert!(still.iter().any(|word| *word != 0.0), "{name} is calm");
            assert!(
                still[0].to_bits() != still[7 * faces].to_bits(),
                "{name} has no vertical variation to preserve"
            );
            for (index, (a, b)) in still.iter().zip(moved).enumerate() {
                assert_eq!(b.to_bits(), (a - offset).to_bits(), "{name} word {index}");
            }
            // Every level-to-level difference survives the change of frame.
            for level in 1..8 {
                let still_shear = still[level * faces] - still[(level - 1) * faces];
                let moved_shear = moved[level * faces] - moved[(level - 1) * faces];
                assert_eq!(
                    moved_shear.to_bits(),
                    still_shear.to_bits(),
                    "{name} shear across level {level}"
                );
            }
        }
    }

    #[test]
    fn a_fixed_frame_leaves_the_sounding_lane_raw_bit_identical() {
        let fixed = sounding_state(TranslatingFrame::fixed());
        let zero = sounding_state(TranslatingFrame::new(0.0, 0.0).unwrap());
        assert_eq!(fixed.carrier_digest_sha256(), zero.carrier_digest_sha256());
        assert_eq!(fixed.moving_frame(), TranslatingFrame::fixed());
    }

    #[test]
    fn the_sounding_lane_offsets_every_staggered_face_by_exactly_the_frame() {
        // The tornado run reads a real observed profile through isnd=7, so
        // the frame has to reach this lane's interpolated winds too, not
        // only the analytic hodograph.  base.F:2663-2669 runs after the
        // isnd=7 branch, so one subtraction covers both.
        let umove = 12.5_f32;
        let vmove = -3.25_f32;
        let fixed = sounding_state(TranslatingFrame::fixed());
        let moving = sounding_state(TranslatingFrame::new(umove, vmove).unwrap());
        assert_eq!(moving.moving_frame().u_move_m_per_s(), umove);
        assert_eq!(moving.moving_frame().v_move_m_per_s(), vmove);

        for (name, offset, shape) in [
            ("u0", umove, vec![8_usize, 3, 5]),
            ("v0", vmove, vec![8, 4, 4]),
        ] {
            let still = fixed.carrier(name).unwrap();
            let moved = moving.carrier(name).unwrap();
            assert_eq!(still.shape(), shape.as_slice(), "{name} shape");
            assert_eq!(moved.shape(), shape.as_slice(), "{name} shape");
            let mut moved_words = 0_usize;
            for (index, (a, b)) in still.values().iter().zip(moved.values()).enumerate() {
                assert_eq!(b.to_bits(), (a - offset).to_bits(), "{name} word {index}");
                if a.to_bits() != b.to_bits() {
                    moved_words += 1;
                }
            }
            // The sounding's own winds could in principle be zero at a
            // level, but not at every one of them; a carrier that did not
            // move at all would pass the equality above and prove nothing.
            assert_eq!(
                moved_words,
                still.values().len(),
                "{name} left words unmoved"
            );
        }
    }

    #[test]
    fn flat_entry_matches_native_uniform_vertical_grid_at_nondefault_spacing() {
        // Raw profile hashes from unmodified CM1 r21.1 param.F:7789-7811,
        // 8464-8494, extracted into a standalone GNU Fortran 13.3.0 oracle.
        // O0, O2 and an O2 repeat agree exactly with fast math and FMA off.
        // The fractional-dz case detects reassociation of face midpoints;
        // the 237-level case detects substituting zero-height terrain math.
        for (nz, dz, expected) in [
            (
                237,
                75.0_f32,
                [
                    "c9d186a3984737d79d77c69df5a305fefbbde771f8e7916c3e4d24648cc6c685",
                    "5e8bb0bbcda5b45650cd2759f1d1128fd457546f49633af3e7731ebaffccd7d1",
                    "4b7ebc1789dd55ce3d8ecc8efc1240ebe26b0174d6c95328eb90b1f366dbc097",
                    "4b7ebc1789dd55ce3d8ecc8efc1240ebe26b0174d6c95328eb90b1f366dbc097",
                    "ba15b573453b0c26f6d5b505e4b49481826a8c176092cfd6251f9fbcc10eaea7",
                    "ba15b573453b0c26f6d5b505e4b49481826a8c176092cfd6251f9fbcc10eaea7",
                ],
            ),
            (
                237,
                75.1_f32,
                [
                    "e4a5552e1d3c484fce65da30ae0b830894a1972a50d1f8864a1cf318f10c4d8b",
                    "ef0d42eb0ebcd0b6ad7df028e52d17286bb1a1916dedf3ea8dfb2e49d60ce57e",
                    "5c2c29dece8c58d94f42b64550fbad660101f155eca8e8dea8fe3b151c292600",
                    "814f836d618b678bfb556a7dfbf9cae089518b7ca9172644631d3636d82f295b",
                    "b7ecf40536fd5d8ecd19030ef769ce52343e11f0545e52a81e09476ee11c5b4c",
                    "6ec4f2c646c4de25530824b34848983d71b95c3ec26b8a13a21e9fa2a4518977",
                ],
            ),
            (
                75,
                200.0_f32,
                [
                    "458b7ecc933a04b929e2d59e2a1b0cde856e583189d9690d5c481fed4e4f9419",
                    "9a25165c25b86de3f86bf78876cb5ccfa3d7723b11d451513cbb68c95b3f0732",
                    "3c06904596e488f41166ab83f4b9909392ff9132464cd9544c7173944d814d17",
                    "3c06904596e488f41166ab83f4b9909392ff9132464cd9544c7173944d814d17",
                    "11043eed5a4df1b4865ebfbc3566d409a784c73c586352393ada7fb4f007adcb",
                    "11043eed5a4df1b4865ebfbc3566d409a784c73c586352393ada7fb4f007adcb",
                ],
            ),
        ] {
            let state = initialize_flat_idealized_padded_state(
                SOUNDING,
                selection(),
                WarmBubbleControls::official(112.5, 75.0).unwrap(),
                [nz, 2, 3],
                [75.0, 75.0, dz as f64],
                [0.0, 0.0],
                TranslatingFrame::fixed(),
            )
            .unwrap();
            assert_eq!(state.pi_recovery_mismatch_count(), 0);
            assert_eq!(state.theta_recovery_mismatch_count(), 0);
            assert_eq!(state.terrain_open_carriers().len(), 54);
            assert_eq!(state.carrier("qa").unwrap().shape(), &[6, nz + 2, 8, 9]);
            for (name, expected_hash) in ["zh", "zf", "mh", "rmh", "mf", "rmf"]
                .into_iter()
                .zip(expected)
            {
                let carrier = state.terrain_static().carrier(name).unwrap();
                let column = carrier
                    .values()
                    .iter()
                    .step_by(6)
                    .copied()
                    .collect::<Vec<_>>();
                assert_eq!(
                    raw_sha256(&column),
                    expected_hash,
                    "{name}: nz={nz}, dz={dz}"
                );
                for plane in carrier.values().chunks_exact(6) {
                    assert!(plane
                        .iter()
                        .all(|word| word.to_bits() == plane[0].to_bits()));
                }
            }
        }
    }

    #[test]
    fn isnd7_padded_carrier_digest_is_sealed() {
        // This digest is the regression fence for the established isnd=7
        // lane.  Any new initialization lane added beside it must leave this
        // value unchanged; a changed digest means the isnd=7 carriers moved.
        let state = initialize_flat_idealized_padded_state(
            SOUNDING,
            selection(),
            WarmBubbleControls::official(1500.0, 1000.0).unwrap(),
            [6, 3, 4],
            [1000.0, 1000.0, 500.0],
            [0.0, 0.0],
            TranslatingFrame::fixed(),
        )
        .unwrap();
        assert_eq!(
            state.carrier_digest_sha256(),
            "dcfda0d231693937e2a8874bfd51d78f598255e3cc62a5b9cbd70277c54d75a3",
            "the established isnd=7 padded carrier bytes changed"
        );
    }
}
