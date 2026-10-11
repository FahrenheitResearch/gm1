//! Strict Rust-native regridding of decoded HRRR hybrid-level windows.
//!
//! This module begins at a deliberately explicit ownership boundary: the
//! caller supplies finite little-endian float32 windows decoded from the
//! operational GRIB2 messages.  Rust then owns carrier inventory validation,
//! Lambert coordinates, wind rotation, horizontal and vertical interpolation,
//! thermodynamic conversion, CM1 staggering, f00/f01 frame construction, and
//! deterministic initialization receipts.  GRIB2 entropy decoding is outside
//! this crate; no Python transform or scientific fallback is accepted here.

use crate::forecast::{
    ComponentBlobSpec, ForecastArray, ForecastArraySpec, ForecastContract, ForecastFrame,
    ForecastReceipt, SchedulerScalar, SchedulerScalarSpec, SchedulerScalarType,
    SchedulerScalarValue, MAX_FORECAST_PAYLOAD_BYTES,
};
use crate::json::JsonValue;
use crate::sha256;
use crate::snapshot::atomic_write_new;
use crate::{Result, StateError};
use std::collections::BTreeSet;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
use std::fs;
use std::path::Path;

pub const HYBRID_LEVELS: usize = 50;
pub const INITIALIZATION_RECEIPT_SCHEMA: &str = "gpuwm-cm1.hrrr-cm1-initialization/v1";
pub const DECODED_WINDOW_SCHEMA: &str = "gpuwm-cm1.hrrr-decoded-f32-window/v1";
pub const HORIZONTAL_REMAP_AUDIT_SCHEMA: &str = "gm1.hrrr-horizontal-remap-audit/v1";
pub const HORIZONTAL_REMAP_ALGORITHM: &str = "target mass-cell latitude/longitude -> HRRR Lambert fractional full index -> decoded-window fractional index -> four-corner bilinear; reflectivity converts each dBZ corner to linear Z, bilinearly weights linear Z, then converts once back to dBZ";
pub const HORIZONTAL_REMAP_CLAIM_LIMIT: &str = "real-source coordinate and interpolation proof for one decoded HRRR field and one GM1 target grid; proves fractional Lambert mapping, four-corner linear-Z bilinear continuity, and non-equivalence to nearest-cell replication, but cannot create physical information below HRRR source resolution and makes no evolved forecast, storm-scale spin-up, tornado, or model-validation claim";
pub const THERMODYNAMIC_CONTRACT: &str = "SPFH specific humidity converts to vapor mixing ratio q/(1-q); HRRR hydrometeor mixing ratios remain mixing ratios; theta=T*(100000/P)^(Rd/Cp); Exner=(P/100000)^(Rd/Cp)";
pub const VERTICAL_MOTION_CONTRACT: &str = "required HRRR VVEL omega in Pa/s converts to geometric w=-omega/(rho*g) with rho=P/(Rd*Tv); no zero-w fallback";
pub const REFLECTIVITY_CONTRACT: &str = "required REFC and direct REFD 1000/4000 m AGL messages are horizontally interpolated in linear-Z space; no hydrometeor proxy";
pub const VERTICAL_INTERPOLATION_CONTRACT: &str = "hybrid HGT brackets CM1 terrain-following target MSL height; pressure is log-linear in height and remaining hybrid carriers are linear in height; bounded below-first-level extrapolation is permitted only down to terrain";
pub const TERRAIN_CONTRACT: &str = "bilinearly regridded source terrain is the CM1 zs field; mass and w-face heights use the CM1 r21.1 transform z=zs+sigma*(zt-zs)/zt with zt=nz*dz, preserving a level model top; interior u/v columns use the source-ordered average of adjacent transformed mass heights and outer faces use directly sampled face terrain";
pub const LIVE_WIRING_LIMIT: &str = "frame is CM1-carrier-shaped but live CUDA terrain metrics, halo population, pressure rebalance, and device injection remain deferred";

const RD: f64 = 287.04;
const CP: f64 = 1005.7;
const GRAVITY: f64 = 9.80665;
const REFERENCE_PRESSURE_PA: f64 = 100_000.0;
const MAX_BELOW_LOWEST_EXTRAPOLATION_M: f64 = 300.0;

const HYBRID_FILES: [&str; 11] = [
    "PRES.f32le",
    "QC.f32le",
    "QG.f32le",
    "QI.f32le",
    "QR.f32le",
    "QS.f32le",
    "HGT.f32le",
    "SPFH.f32le",
    "TT.f32le",
    "U_MASS.f32le",
    "V_MASS.f32le",
];

const SURFACE_FILES: [&str; 11] = [
    "LANDSEA.f32le",
    "PSFC.f32le",
    "Q2.f32le",
    "SKINTEMP.f32le",
    "SNOW.f32le",
    "SNOWH.f32le",
    "SOILHGT.f32le",
    "T2.f32le",
    "U10_MASS.f32le",
    "V10_MASS.f32le",
    "XICE.f32le",
];

const DIAGNOSTIC_FILES: [&str; 3] = ["REFC.f32le", "REFD-1000.f32le", "REFD-4000.f32le"];

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrrrLambertGrid {
    earth_radius_m: f64,
    nx: usize,
    ny: usize,
    lat1_deg: f64,
    lon1_deg_east: f64,
    dx_m: f64,
    dy_m: f64,
    latin1_deg: f64,
    latin2_deg: f64,
    lov_deg_east: f64,
    window_i_start: usize,
    window_j_start: usize,
    window_nx: usize,
    window_ny: usize,
}

impl HrrrLambertGrid {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        earth_radius_m: f64,
        nx: usize,
        ny: usize,
        lat1_deg: f64,
        lon1_deg_east: f64,
        dx_m: f64,
        dy_m: f64,
        latin1_deg: f64,
        latin2_deg: f64,
        lov_deg_east: f64,
        window_i_start: usize,
        window_j_start: usize,
        window_nx: usize,
        window_ny: usize,
    ) -> Result<Self> {
        let grid = Self {
            earth_radius_m,
            nx,
            ny,
            lat1_deg,
            lon1_deg_east,
            dx_m,
            dy_m,
            latin1_deg,
            latin2_deg,
            lov_deg_east,
            window_i_start,
            window_j_start,
            window_nx,
            window_ny,
        };
        grid.validate()?;
        Ok(grid)
    }

    /// Canonical NOAA HRRR CONUS Lambert grid with a caller-selected window.
    pub fn canonical_window(
        window_i_start: usize,
        window_j_start: usize,
        window_nx: usize,
        window_ny: usize,
    ) -> Result<Self> {
        Self::new(
            6_371_229.0,
            1799,
            1059,
            21.138_123,
            237.280_472,
            3000.0,
            3000.0,
            38.5,
            38.5,
            262.5,
            window_i_start,
            window_j_start,
            window_nx,
            window_ny,
        )
    }

    pub fn window_shape(&self) -> (usize, usize) {
        (self.window_nx, self.window_ny)
    }

    pub fn window_origin(&self) -> (usize, usize) {
        (self.window_i_start, self.window_j_start)
    }

    pub fn full_shape(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    pub fn grid_point_lat_lon(&self, i: f64, j: f64) -> Result<(f64, f64)> {
        if !i.is_finite() || !j.is_finite() {
            return Err(invalid("Lambert inverse indices must be finite"));
        }
        let (x_first, y_first) = self.project_raw(self.lat1_deg, self.lon1_deg_east)?;
        let x = x_first + i * self.dx_m;
        let y = y_first + j * self.dy_m;
        let (cone, scale) = self.cone_and_scale()?;
        let radius = x.hypot(-y);
        if radius == 0.0 {
            return Err(invalid("Lambert inverse is singular at the cone apex"));
        }
        let theta = x.atan2(-y);
        let tangent = (self.earth_radius_m * scale / radius).powf(1.0 / cone);
        let latitude = 2.0 * tangent.atan() - FRAC_PI_2;
        let longitude = self.lov_deg_east.to_radians() + theta / cone;
        Ok((
            latitude.to_degrees(),
            normalize_degrees(longitude.to_degrees()),
        ))
    }

    pub fn fractional_full_index(&self, lat_deg: f64, lon_deg: f64) -> Result<(f64, f64)> {
        let (x_first, y_first) = self.project_raw(self.lat1_deg, self.lon1_deg_east)?;
        let (x, y) = self.project_raw(lat_deg, lon_deg)?;
        Ok(((x - x_first) / self.dx_m, (y - y_first) / self.dy_m))
    }

    fn fractional_window_index(&self, lat_deg: f64, lon_deg: f64) -> Result<Sample> {
        let (i, j) = self.fractional_full_index(lat_deg, lon_deg)?;
        let x = i - self.window_i_start as f64;
        let y = j - self.window_j_start as f64;
        let tolerance = 1.0e-9;
        if x < -tolerance
            || y < -tolerance
            || x > (self.window_nx - 1) as f64 + tolerance
            || y > (self.window_ny - 1) as f64 + tolerance
        {
            return Err(invalid(format!(
                "target point ({lat_deg:.8},{lon_deg:.8}) maps to decoded-window ({x:.6},{y:.6}), outside 0..{} by 0..{}",
                self.window_nx - 1,
                self.window_ny - 1
            )));
        }
        let convergence = self.cone_and_scale()?.0
            * normalized_longitude_delta(lon_deg, self.lov_deg_east).to_radians();
        Ok(Sample {
            x: x.clamp(0.0, (self.window_nx - 1) as f64),
            y: y.clamp(0.0, (self.window_ny - 1) as f64),
            rotation_cos: convergence.cos(),
            rotation_sin: convergence.sin(),
        })
    }

    fn project_raw(&self, lat_deg: f64, lon_deg: f64) -> Result<(f64, f64)> {
        if !lat_deg.is_finite() || !lon_deg.is_finite() || !(-90.0..90.0).contains(&lat_deg) {
            return Err(invalid(format!(
                "Lambert coordinate must have finite latitude strictly between poles and finite longitude; got ({lat_deg:?},{lon_deg:?})"
            )));
        }
        let (cone, scale) = self.cone_and_scale()?;
        let latitude = lat_deg.to_radians();
        let theta = cone * normalized_longitude_delta(lon_deg, self.lov_deg_east).to_radians();
        let radius = self.earth_radius_m * scale / (FRAC_PI_4 + 0.5 * latitude).tan().powf(cone);
        Ok((radius * theta.sin(), -radius * theta.cos()))
    }

    fn cone_and_scale(&self) -> Result<(f64, f64)> {
        let first = self.latin1_deg.to_radians();
        let second = self.latin2_deg.to_radians();
        let cone = if (first - second).abs() < 1.0e-14 {
            first.sin()
        } else {
            (first.cos() / second.cos()).ln()
                / ((FRAC_PI_4 + 0.5 * second).tan() / (FRAC_PI_4 + 0.5 * first).tan()).ln()
        };
        if !cone.is_finite() || cone <= 0.0 {
            return Err(invalid(
                "this HRRR path supports only a finite northern-hemisphere Lambert cone",
            ));
        }
        let scale = first.cos() * (FRAC_PI_4 + 0.5 * first).tan().powf(cone) / cone;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid("invalid Lambert scale"));
        }
        Ok((cone, scale))
    }

    fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("earth radius", self.earth_radius_m),
            ("dx", self.dx_m),
            ("dy", self.dy_m),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!(
                    "Lambert {label} must be finite and positive"
                )));
            }
        }
        for (label, value) in [
            ("first latitude", self.lat1_deg),
            ("first standard parallel", self.latin1_deg),
            ("second standard parallel", self.latin2_deg),
        ] {
            if !value.is_finite() || !(-90.0..90.0).contains(&value) {
                return Err(invalid(format!(
                    "Lambert {label} must be finite and strictly between the poles"
                )));
            }
        }
        if !self.lon1_deg_east.is_finite() || !self.lov_deg_east.is_finite() {
            return Err(invalid("Lambert longitudes must be finite"));
        }
        if self.nx < 2 || self.ny < 2 || self.window_nx < 2 || self.window_ny < 2 {
            return Err(invalid(
                "Lambert full grid and decoded window must each be at least 2x2",
            ));
        }
        let i_end = self
            .window_i_start
            .checked_add(self.window_nx)
            .ok_or_else(|| invalid("Lambert window i extent overflows"))?;
        let j_end = self
            .window_j_start
            .checked_add(self.window_ny)
            .ok_or_else(|| invalid("Lambert window j extent overflows"))?;
        if i_end > self.nx || j_end > self.ny {
            return Err(invalid(
                "decoded Lambert window extends beyond the full source grid",
            ));
        }
        self.cone_and_scale()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cm1TargetGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx_m: f64,
    pub dy_m: f64,
    pub dz_m: f64,
    pub center_lat_deg: f64,
    pub center_lon_deg: f64,
}

impl Cm1TargetGrid {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx_m: f64,
        dy_m: f64,
        dz_m: f64,
        center_lat_deg: f64,
        center_lon_deg: f64,
    ) -> Result<Self> {
        let target = Self {
            nx,
            ny,
            nz,
            dx_m,
            dy_m,
            dz_m,
            center_lat_deg,
            center_lon_deg,
        };
        target.validate()?;
        Ok(target)
    }

    fn validate(&self) -> Result<()> {
        if self.nx < 2 || self.ny < 2 || self.nz < 2 {
            return Err(invalid("CM1 target dimensions must each be at least two"));
        }
        if self.nx > i32::MAX as usize || self.ny > i32::MAX as usize || self.nz > i32::MAX as usize
        {
            return Err(invalid(
                "CM1 target dimensions exceed forecast-contract signed bounds",
            ));
        }
        for (label, value) in [("dx", self.dx_m), ("dy", self.dy_m), ("dz", self.dz_m)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!(
                    "CM1 target {label} must be finite and positive"
                )));
            }
        }
        if !self.center_lat_deg.is_finite()
            || !(-90.0..90.0).contains(&self.center_lat_deg)
            || !self.center_lon_deg.is_finite()
        {
            return Err(invalid("CM1 target center latitude/longitude is invalid"));
        }
        self.nx
            .checked_mul(self.ny)
            .and_then(|value| value.checked_mul(self.nz))
            .ok_or_else(|| invalid("CM1 target cell count overflows"))?;
        let mass = self.nx * self.ny;
        let volume = mass * self.nz;
        let words = 7usize
            .checked_mul(self.nz)
            .and_then(|value| value.checked_add(8 * volume))
            .and_then(|value| value.checked_add(self.nz * self.ny * (self.nx + 1)))
            .and_then(|value| value.checked_add(self.nz * (self.ny + 1) * self.nx))
            .and_then(|value| value.checked_add((self.nz + 1) * mass))
            .and_then(|value| value.checked_add(4 * mass))
            .ok_or_else(|| invalid("CM1 target forecast payload estimate overflows"))?;
        let bytes = words
            .checked_mul(4)
            .ok_or_else(|| invalid("CM1 target forecast payload byte estimate overflows"))?;
        if bytes > MAX_FORECAST_PAYLOAD_BYTES {
            return Err(invalid(format!(
                "CM1 target forecast payload requires {bytes} bytes, exceeding the {MAX_FORECAST_PAYLOAD_BYTES}-byte envelope limit"
            )));
        }
        Ok(())
    }

    fn mass_lat_lon(&self, i: usize, j: usize) -> (f64, f64) {
        let x = (i as f64 + 0.5 - 0.5 * self.nx as f64) * self.dx_m;
        let y = (j as f64 + 0.5 - 0.5 * self.ny as f64) * self.dy_m;
        geodesic_offset(self.center_lat_deg, self.center_lon_deg, x, y)
    }

    /// Geographic center of one target mass cell.
    ///
    /// This exposes the same source-ordered mapping used by the interpolator so
    /// receipts can locate extrema without reimplementing the grid projection.
    pub fn mass_point_lat_lon(&self, i: usize, j: usize) -> Result<(f64, f64)> {
        if i >= self.nx || j >= self.ny {
            return Err(invalid(format!(
                "CM1 target mass index ({i},{j}) exceeds ({},{})",
                self.nx, self.ny
            )));
        }
        Ok(self.mass_lat_lon(i, j))
    }

    fn u_lat_lon(&self, i: usize, j: usize) -> (f64, f64) {
        let x = (i as f64 - 0.5 * self.nx as f64) * self.dx_m;
        let y = (j as f64 + 0.5 - 0.5 * self.ny as f64) * self.dy_m;
        geodesic_offset(self.center_lat_deg, self.center_lon_deg, x, y)
    }

    fn v_lat_lon(&self, i: usize, j: usize) -> (f64, f64) {
        let x = (i as f64 + 0.5 - 0.5 * self.nx as f64) * self.dx_m;
        let y = (j as f64 - 0.5 * self.ny as f64) * self.dy_m;
        geodesic_offset(self.center_lat_deg, self.center_lon_deg, x, y)
    }
}

#[derive(Clone, Debug)]
pub struct DecodedHrrrWindow {
    grid: HrrrLambertGrid,
    forecast_hour: u32,
    source_fingerprint_sha256: String,
    pressure: Vec<f32>,
    height: Vec<f32>,
    temperature: Vec<f32>,
    specific_humidity: Vec<f32>,
    qc: Vec<f32>,
    qi: Vec<f32>,
    qr: Vec<f32>,
    qs: Vec<f32>,
    qg: Vec<f32>,
    u_grid: Vec<f32>,
    v_grid: Vec<f32>,
    omega: Vec<f32>,
    terrain: Vec<f32>,
    refc: Vec<f32>,
    refd_1000: Vec<f32>,
    refd_4000: Vec<f32>,
}

impl DecodedHrrrWindow {
    pub fn read_directory(
        path: impl AsRef<Path>,
        grid: HrrrLambertGrid,
        forecast_hour: u32,
    ) -> Result<Self> {
        let path = path.as_ref();
        validate_exact_inventory(path)?;
        let cells = grid.window_nx * grid.window_ny;
        let hybrid_words = cells * HYBRID_LEVELS;
        let pressure = read_f32(path, "PRES.f32le", hybrid_words)?;
        let height = read_f32(path, "HGT.f32le", hybrid_words)?;
        let temperature = read_f32(path, "TT.f32le", hybrid_words)?;
        let specific_humidity = read_f32(path, "SPFH.f32le", hybrid_words)?;
        let qc = read_f32(path, "QC.f32le", hybrid_words)?;
        let qi = read_f32(path, "QI.f32le", hybrid_words)?;
        let qr = read_f32(path, "QR.f32le", hybrid_words)?;
        let qs = read_f32(path, "QS.f32le", hybrid_words)?;
        let qg = read_f32(path, "QG.f32le", hybrid_words)?;
        let u_grid = read_f32(path, "U_MASS.f32le", hybrid_words)?;
        let v_grid = read_f32(path, "V_MASS.f32le", hybrid_words)?;
        let mut omega = Vec::with_capacity(hybrid_words);
        for level in 1..=HYBRID_LEVELS {
            omega.extend(read_f32(path, &format!("VVEL-{level:02}.f32le"), cells)?);
        }
        let terrain = read_f32(path, "SOILHGT.f32le", cells)?;
        let psfc = read_f32(path, "PSFC.f32le", cells)?;
        let q2 = read_f32(path, "Q2.f32le", cells)?;
        let skin_temperature = read_f32(path, "SKINTEMP.f32le", cells)?;
        let snow = read_f32(path, "SNOW.f32le", cells)?;
        let snow_height = read_f32(path, "SNOWH.f32le", cells)?;
        let t2 = read_f32(path, "T2.f32le", cells)?;
        let _u10 = read_f32(path, "U10_MASS.f32le", cells)?;
        let _v10 = read_f32(path, "V10_MASS.f32le", cells)?;
        let landsea = read_f32(path, "LANDSEA.f32le", cells)?;
        let ice_fraction = read_f32(path, "XICE.f32le", cells)?;
        let refc = read_f32(path, "REFC.f32le", cells)?;
        let refd_1000 = read_f32(path, "REFD-1000.f32le", cells)?;
        let refd_4000 = read_f32(path, "REFD-4000.f32le", cells)?;

        validate_positive(&pressure, "PRES")?;
        validate_positive(&temperature, "TT")?;
        validate_fraction(&specific_humidity, "SPFH")?;
        validate_positive(&psfc, "PSFC")?;
        validate_positive(&skin_temperature, "SKINTEMP")?;
        validate_positive(&t2, "T2")?;
        validate_fraction(&q2, "Q2")?;
        validate_fraction_closed(&landsea, "LANDSEA")?;
        validate_fraction_closed(&ice_fraction, "XICE")?;
        validate_nonnegative(&snow, "SNOW")?;
        validate_nonnegative(&snow_height, "SNOWH")?;
        for (name, values) in [
            ("QC", &qc),
            ("QI", &qi),
            ("QR", &qr),
            ("QS", &qs),
            ("QG", &qg),
        ] {
            validate_nonnegative(values, name)?;
        }
        for (name, values) in [
            ("REFC", &refc),
            ("REFD-1000", &refd_1000),
            ("REFD-4000", &refd_4000),
        ] {
            if values.iter().any(|value| !(-200.0..=100.0).contains(value)) {
                return Err(invalid(format!(
                    "{name} contains a value outside the accepted direct dBZ range -200..100"
                )));
            }
        }
        validate_vertical_coordinates(&height, &pressure, cells)?;
        let source_fingerprint_sha256 = fingerprint_directory(path)?;
        Ok(Self {
            grid,
            forecast_hour,
            source_fingerprint_sha256,
            pressure,
            height,
            temperature,
            specific_humidity,
            qc,
            qi,
            qr,
            qs,
            qg,
            u_grid,
            v_grid,
            omega,
            terrain,
            refc,
            refd_1000,
            refd_4000,
        })
    }

    pub fn forecast_hour(&self) -> u32 {
        self.forecast_hour
    }

    pub fn source_fingerprint_sha256(&self) -> &str {
        &self.source_fingerprint_sha256
    }

    pub fn grid(&self) -> HrrrLambertGrid {
        self.grid
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MappingStats {
    pub total_theta_min_k: f32,
    pub total_theta_max_k: f32,
    pub pressure_min_pa: f32,
    pub pressure_max_pa: f32,
    pub theta_perturbation_min_k: f32,
    pub theta_perturbation_max_k: f32,
    pub exner_perturbation_min: f32,
    pub exner_perturbation_max: f32,
    pub w_min_mps: f32,
    pub w_max_mps: f32,
    pub nonzero_w_cells: usize,
    pub refc_min_dbz: f32,
    pub refc_max_dbz: f32,
    pub refd_1000_max_dbz: f32,
    pub refd_4000_max_dbz: f32,
}

#[derive(Clone, Debug)]
pub struct Cm1MappedFrame {
    target: Cm1TargetGrid,
    forecast_hour: u32,
    source_fingerprint_sha256: String,
    arrays: Vec<ForecastArray>,
    stats: MappingStats,
}

impl Cm1MappedFrame {
    pub fn target(&self) -> Cm1TargetGrid {
        self.target
    }

    pub fn forecast_hour(&self) -> u32 {
        self.forecast_hour
    }

    pub fn source_fingerprint_sha256(&self) -> &str {
        &self.source_fingerprint_sha256
    }

    pub fn stats(&self) -> &MappingStats {
        &self.stats
    }

    pub fn array(&self, name: &str) -> Option<&[f32]> {
        self.arrays
            .iter()
            .find(|array| array.name() == name)
            .map(ForecastArray::values)
    }

    pub fn into_forecast_frame(self, model_dt_s: f64) -> Result<ForecastFrame> {
        if !model_dt_s.is_finite() || model_dt_s <= 0.0 {
            return Err(invalid("model dt must be finite and positive"));
        }
        let valid_offset_s = f64::from(self.forecast_hour) * 3600.0;
        let exact_step = valid_offset_s / model_dt_s;
        let rounded_step = exact_step.round();
        if (exact_step - rounded_step).abs() > 1.0e-9 || rounded_step > u64::MAX as f64 {
            return Err(invalid(format!(
                "forecast-hour offset {valid_offset_s} s is not an integral model step for dt={model_dt_s} s"
            )));
        }
        let scalars = vec![
            SchedulerScalar::new(
                "boundary_ready",
                SchedulerScalarValue::Bool(self.forecast_hour > 0),
            )?,
            SchedulerScalar::new(
                "forcing_valid_offset_s",
                SchedulerScalarValue::F64(valid_offset_s),
            )?,
            SchedulerScalar::new(
                "source_forecast_hour",
                SchedulerScalarValue::U64(u64::from(self.forecast_hour)),
            )?,
            SchedulerScalar::new("target_dx_m", SchedulerScalarValue::F64(self.target.dx_m))?,
            SchedulerScalar::new("target_dy_m", SchedulerScalarValue::F64(self.target.dy_m))?,
            SchedulerScalar::new("target_dz_m", SchedulerScalarValue::F64(self.target.dz_m))?,
        ];
        Ok(ForecastFrame::new(
            crate::forecast::ForecastClock::new(rounded_step as u64, valid_offset_s, model_dt_s)?,
            self.arrays,
            scalars,
            Vec::new(),
        ))
    }
}

pub fn forecast_contract(
    target: Cm1TargetGrid,
    config_sha256: impl Into<String>,
    execution_identity_sha256: impl Into<String>,
) -> Result<ForecastContract> {
    let mass_shape = vec![target.nz, target.ny, target.nx];
    let u_shape = vec![target.nz, target.ny, target.nx + 1];
    let v_shape = vec![target.nz, target.ny + 1, target.nx];
    let w_shape = vec![target.nz + 1, target.ny, target.nx];
    let surface_shape = vec![target.ny, target.nx];
    let profile_shape = vec![target.nz];
    let mass_bounds = bounds_zyx(target.nz, target.ny, target.nx);
    let u_bounds = bounds_zyx(target.nz, target.ny, target.nx + 1);
    let v_bounds = bounds_zyx(target.nz, target.ny + 1, target.nx);
    let w_bounds = bounds_zyx(target.nz + 1, target.ny, target.nx);
    let surface_bounds = vec![1, target.ny as i32, 1, target.nx as i32];
    let profile_bounds = vec![1, target.nz as i32];
    let mut arrays = Vec::new();
    for (name, meaning, units) in [
        ("pi0", "domain-mean base Exner function", "1"),
        ("prs0", "domain-mean base pressure", "Pa"),
        (
            "qv0",
            "domain-mean base water-vapor mixing ratio",
            "kg kg-1",
        ),
        ("rho0", "diagnostic moist base density", "kg m-3"),
        ("th0", "domain-mean base potential temperature", "K"),
        ("u0", "domain-mean earth-east base wind", "m s-1"),
        ("v0", "domain-mean earth-north base wind", "m s-1"),
    ] {
        arrays.push(ForecastArraySpec::new(
            name,
            meaning,
            units,
            "vertical-profile",
            profile_bounds.clone(),
            profile_shape.clone(),
        )?);
    }
    for (name, meaning, units) in [
        ("pp3d", "Exner perturbation from pi0", "1"),
        ("th3d", "potential-temperature perturbation from th0", "K"),
        ("qv", "water-vapor mixing ratio", "kg kg-1"),
        ("qc", "cloud-water mixing ratio", "kg kg-1"),
        ("qi", "cloud-ice mixing ratio", "kg kg-1"),
        ("qr", "rain mixing ratio", "kg kg-1"),
        ("qs", "snow mixing ratio", "kg kg-1"),
        ("qg", "graupel mixing ratio", "kg kg-1"),
    ] {
        arrays.push(ForecastArraySpec::new(
            name,
            meaning,
            units,
            "scalar-mass-zyx",
            mass_bounds.clone(),
            mass_shape.clone(),
        )?);
    }
    arrays.push(ForecastArraySpec::new(
        "u3d",
        "earth-east wind on x faces",
        "m s-1",
        "x-face-zyx",
        u_bounds,
        u_shape,
    )?);
    arrays.push(ForecastArraySpec::new(
        "v3d",
        "earth-north wind on y faces",
        "m s-1",
        "y-face-zyx",
        v_bounds,
        v_shape,
    )?);
    arrays.push(ForecastArraySpec::new(
        "w3d",
        "geometric vertical velocity converted from required omega",
        "m s-1",
        "z-face-zyx",
        w_bounds,
        w_shape,
    )?);
    for (name, meaning, units) in [
        (
            "terrain_msl",
            "source terrain height above mean sea level",
            "m",
        ),
        ("refc_dbz", "direct composite reflectivity", "dBZ"),
        (
            "refd_1000m_agl_dbz",
            "direct derived radar reflectivity at 1000 m AGL",
            "dBZ",
        ),
        (
            "refd_4000m_agl_dbz",
            "direct derived radar reflectivity at 4000 m AGL",
            "dBZ",
        ),
    ] {
        arrays.push(ForecastArraySpec::new(
            name,
            meaning,
            units,
            "scalar-surface-yx",
            surface_bounds.clone(),
            surface_shape.clone(),
        )?);
    }
    let scalar_specs = vec![
        SchedulerScalarSpec::new("boundary_ready", SchedulerScalarType::Bool)?,
        SchedulerScalarSpec::new("forcing_valid_offset_s", SchedulerScalarType::F64)?,
        SchedulerScalarSpec::new("source_forecast_hour", SchedulerScalarType::U64)?,
        SchedulerScalarSpec::new("target_dx_m", SchedulerScalarType::F64)?,
        SchedulerScalarSpec::new("target_dy_m", SchedulerScalarType::F64)?,
        SchedulerScalarSpec::new("target_dz_m", SchedulerScalarType::F64)?,
    ];
    ForecastContract::new(
        "hrrr-cm1-real-state-v1",
        config_sha256,
        execution_identity_sha256,
        arrays,
        scalar_specs,
        Vec::<ComponentBlobSpec>::new(),
    )
}

pub fn configuration_sha256(
    source_grid: HrrrLambertGrid,
    target: Cm1TargetGrid,
    model_dt_s: f64,
) -> Result<String> {
    if !model_dt_s.is_finite() || model_dt_s <= 0.0 {
        return Err(invalid("model dt must be finite and positive"));
    }
    let value = object(vec![
        ("decoded_window_schema", string(DECODED_WINDOW_SCHEMA)),
        ("model_dt_s_f64_bits", f64_bits(model_dt_s)),
        (
            "source_grid",
            object(vec![
                ("dx_m_f64_bits", f64_bits(source_grid.dx_m)),
                ("dy_m_f64_bits", f64_bits(source_grid.dy_m)),
                (
                    "earth_radius_m_f64_bits",
                    f64_bits(source_grid.earth_radius_m),
                ),
                ("full_nx", number(source_grid.nx)),
                ("full_ny", number(source_grid.ny)),
                ("lat1_deg_f64_bits", f64_bits(source_grid.lat1_deg)),
                ("latin1_deg_f64_bits", f64_bits(source_grid.latin1_deg)),
                ("latin2_deg_f64_bits", f64_bits(source_grid.latin2_deg)),
                (
                    "lon1_deg_east_f64_bits",
                    f64_bits(source_grid.lon1_deg_east),
                ),
                ("lov_deg_east_f64_bits", f64_bits(source_grid.lov_deg_east)),
                ("window_i_start", number(source_grid.window_i_start)),
                ("window_j_start", number(source_grid.window_j_start)),
                ("window_nx", number(source_grid.window_nx)),
                ("window_ny", number(source_grid.window_ny)),
            ]),
        ),
        ("target", target_json(target)),
    ]);
    Ok(sha256::digest_hex(&value.canonical_bytes()))
}

pub fn execution_identity_sha256() -> String {
    let material = [
        DECODED_WINDOW_SCHEMA,
        THERMODYNAMIC_CONTRACT,
        VERTICAL_MOTION_CONTRACT,
        REFLECTIVITY_CONTRACT,
        VERTICAL_INTERPOLATION_CONTRACT,
        TERRAIN_CONTRACT,
        LIVE_WIRING_LIMIT,
    ]
    .join("\n");
    sha256::digest_hex(material.as_bytes())
}

pub fn map_to_cm1(source: &DecodedHrrrWindow, target: Cm1TargetGrid) -> Result<Cm1MappedFrame> {
    target.validate()?;
    let mass_cells = target.nx * target.ny;
    let volume_cells = mass_cells * target.nz;
    let mut terrain = vec![0.0_f32; mass_cells];
    let mut refc = vec![0.0_f32; mass_cells];
    let mut refd_1000 = vec![0.0_f32; mass_cells];
    let mut refd_4000 = vec![0.0_f32; mass_cells];
    let mut theta = vec![0.0_f32; volume_cells];
    let mut exner = vec![0.0_f32; volume_cells];
    let mut pressure_total = vec![0.0_f32; volume_cells];
    let mut qv = vec![0.0_f32; volume_cells];
    let mut qc = vec![0.0_f32; volume_cells];
    let mut qi = vec![0.0_f32; volume_cells];
    let mut qr = vec![0.0_f32; volume_cells];
    let mut qs = vec![0.0_f32; volume_cells];
    let mut qg = vec![0.0_f32; volume_cells];
    let mut w3d = vec![0.0_f32; mass_cells * (target.nz + 1)];

    let mut hgt_profile = [0.0_f64; HYBRID_LEVELS];
    let mut p_profile = [0.0_f64; HYBRID_LEVELS];
    let mut t_profile = [0.0_f64; HYBRID_LEVELS];
    let mut qv_profile = [0.0_f64; HYBRID_LEVELS];
    let mut qc_profile = [0.0_f64; HYBRID_LEVELS];
    let mut qi_profile = [0.0_f64; HYBRID_LEVELS];
    let mut qr_profile = [0.0_f64; HYBRID_LEVELS];
    let mut qs_profile = [0.0_f64; HYBRID_LEVELS];
    let mut qg_profile = [0.0_f64; HYBRID_LEVELS];
    let mut omega_profile = [0.0_f64; HYBRID_LEVELS];

    let model_top_m = target.nz as f64 * target.dz_m;
    let model_top_f32 = model_top_m as f32;
    if !model_top_f32.is_finite() || model_top_f32 <= 0.0 {
        return Err(invalid("CM1 target model top is invalid in float32"));
    }

    for j in 0..target.ny {
        for i in 0..target.nx {
            let (lat, lon) = target.mass_lat_lon(i, j);
            let sample = source.grid.fractional_window_index(lat, lon)?;
            let horizontal = j * target.nx + i;
            let surface = bilinear(&source.terrain, source.grid, sample);
            let surface_f32 = surface as f32;
            if !surface_f32.is_finite() || surface_f32 >= model_top_f32 {
                return Err(invalid(
                    "regridded source terrain reaches or exceeds the CM1 model top",
                ));
            }
            terrain[horizontal] = surface_f32;
            refc[horizontal] = bilinear_dbz(&source.refc, source.grid, sample) as f32;
            refd_1000[horizontal] = bilinear_dbz(&source.refd_1000, source.grid, sample) as f32;
            refd_4000[horizontal] = bilinear_dbz(&source.refd_4000, source.grid, sample) as f32;
            fill_profile(&source.height, source.grid, sample, &mut hgt_profile);
            fill_profile(&source.pressure, source.grid, sample, &mut p_profile);
            fill_profile(&source.temperature, source.grid, sample, &mut t_profile);
            fill_profile(
                &source.specific_humidity,
                source.grid,
                sample,
                &mut qv_profile,
            );
            fill_profile(&source.qc, source.grid, sample, &mut qc_profile);
            fill_profile(&source.qi, source.grid, sample, &mut qi_profile);
            fill_profile(&source.qr, source.grid, sample, &mut qr_profile);
            fill_profile(&source.qs, source.grid, sample, &mut qs_profile);
            fill_profile(&source.qg, source.grid, sample, &mut qg_profile);
            fill_profile(&source.omega, source.grid, sample, &mut omega_profile);
            validate_interpolated_column(&hgt_profile, &p_profile, f64::from(surface_f32))?;

            for k in 0..target.nz {
                let sigma_m = ((k as f64 + 0.5) * target.dz_m) as f32;
                let z_msl = f64::from(terrain_following_height_f32(
                    surface_f32,
                    sigma_m,
                    model_top_f32,
                ));
                let bracket = vertical_bracket(&hgt_profile, z_msl, f64::from(surface_f32))?;
                let pressure = interpolate_log(&p_profile, bracket);
                let temperature = interpolate_linear(&t_profile, bracket);
                let specific_q = interpolate_linear(&qv_profile, bracket);
                if !(pressure > 0.0 && temperature > 0.0 && (0.0..1.0).contains(&specific_q)) {
                    return Err(invalid(
                        "interpolated thermodynamic state left its physical range",
                    ));
                }
                let vapor_mixing_ratio = specific_q / (1.0 - specific_q);
                let pi = (pressure / REFERENCE_PRESSURE_PA).powf(RD / CP);
                let potential_temperature = temperature / pi;
                let index = (k * target.ny + j) * target.nx + i;
                pressure_total[index] = pressure as f32;
                exner[index] = pi as f32;
                theta[index] = potential_temperature as f32;
                qv[index] = vapor_mixing_ratio as f32;
                qc[index] = nonnegative_interpolate(&qc_profile, bracket, "QC")?;
                qi[index] = nonnegative_interpolate(&qi_profile, bracket, "QI")?;
                qr[index] = nonnegative_interpolate(&qr_profile, bracket, "QR")?;
                qs[index] = nonnegative_interpolate(&qs_profile, bracket, "QS")?;
                qg[index] = nonnegative_interpolate(&qg_profile, bracket, "QG")?;
            }
            for k in 0..=target.nz {
                let sigma_m = (k as f64 * target.dz_m) as f32;
                let z_msl = f64::from(terrain_following_height_f32(
                    surface_f32,
                    sigma_m,
                    model_top_f32,
                ));
                let bracket = vertical_bracket(&hgt_profile, z_msl, f64::from(surface_f32))?;
                let pressure = interpolate_log(&p_profile, bracket);
                let temperature = interpolate_linear(&t_profile, bracket);
                let specific_q = interpolate_linear(&qv_profile, bracket);
                let omega = interpolate_linear(&omega_profile, bracket);
                let virtual_temperature = temperature * (1.0 + 0.61 * specific_q);
                let density = pressure / (RD * virtual_temperature);
                let w = -omega / (density * GRAVITY);
                if !w.is_finite() {
                    return Err(invalid("omega-to-w conversion produced a non-finite value"));
                }
                w3d[(k * target.ny + j) * target.nx + i] = w as f32;
            }
        }
    }

    let mut u3d = vec![0.0_f32; target.nz * target.ny * (target.nx + 1)];
    map_horizontal_wind(source, target, true, &terrain, &mut u3d)?;
    let mut v3d = vec![0.0_f32; target.nz * (target.ny + 1) * target.nx];
    map_horizontal_wind(source, target, false, &terrain, &mut v3d)?;

    let mut th0 = horizontal_mean(&theta, target.nz, mass_cells);
    let mut pi0 = horizontal_mean(&exner, target.nz, mass_cells);
    let prs0 = horizontal_mean(&pressure_total, target.nz, mass_cells);
    let qv0 = horizontal_mean(&qv, target.nz, mass_cells);
    let u0 = staggered_mean_u(&u3d, target);
    let v0 = staggered_mean_v(&v3d, target);
    for k in 0..target.nz {
        for horizontal in 0..mass_cells {
            let index = k * mass_cells + horizontal;
            theta[index] -= th0[k];
            exner[index] -= pi0[k];
        }
    }
    let rho0 = (0..target.nz)
        .map(|k| {
            let temperature = f64::from(th0[k]) * f64::from(pi0[k]);
            (f64::from(prs0[k])
                / (RD * temperature * (1.0 + 0.61 * f64::from(qv0[k]) / (1.0 + f64::from(qv0[k])))))
                as f32
        })
        .collect::<Vec<_>>();

    let (theta_total_min, theta_total_max) = range_reconstructed(&theta, &th0, mass_cells);
    let (pressure_min, pressure_max) = range(&pressure_total);
    let (thpert_min, thpert_max) = range(&theta);
    let (pp_min, pp_max) = range(&exner);
    let (w_min, w_max) = range(&w3d);
    let nonzero_w_cells = w3d.iter().filter(|value| value.abs() > 1.0e-12).count();
    if nonzero_w_cells == 0 {
        return Err(invalid(
            "required VVEL mapped to an all-zero w field; refusing a zero-w initialization",
        ));
    }
    let (refc_min, refc_max) = range(&refc);
    let (_, refd_1000_max) = range(&refd_1000);
    let (_, refd_4000_max) = range(&refd_4000);
    let stats = MappingStats {
        total_theta_min_k: theta_total_min,
        total_theta_max_k: theta_total_max,
        pressure_min_pa: pressure_min,
        pressure_max_pa: pressure_max,
        theta_perturbation_min_k: thpert_min,
        theta_perturbation_max_k: thpert_max,
        exner_perturbation_min: pp_min,
        exner_perturbation_max: pp_max,
        w_min_mps: w_min,
        w_max_mps: w_max,
        nonzero_w_cells,
        refc_min_dbz: refc_min,
        refc_max_dbz: refc_max,
        refd_1000_max_dbz: refd_1000_max,
        refd_4000_max_dbz: refd_4000_max,
    };
    if stats.theta_perturbation_min_k == stats.theta_perturbation_max_k {
        return Err(invalid(
            "mapped theta perturbation is horizontally uniform; refusing a uniform-state replacement",
        ));
    }

    let arrays = vec![
        ForecastArray::new("pi0", std::mem::take(&mut pi0)),
        ForecastArray::new("prs0", prs0),
        ForecastArray::new("qv0", qv0),
        ForecastArray::new("rho0", rho0),
        ForecastArray::new("th0", std::mem::take(&mut th0)),
        ForecastArray::new("u0", u0),
        ForecastArray::new("v0", v0),
        ForecastArray::new("pp3d", exner),
        ForecastArray::new("th3d", theta),
        ForecastArray::new("qv", qv),
        ForecastArray::new("qc", qc),
        ForecastArray::new("qi", qi),
        ForecastArray::new("qr", qr),
        ForecastArray::new("qs", qs),
        ForecastArray::new("qg", qg),
        ForecastArray::new("u3d", u3d),
        ForecastArray::new("v3d", v3d),
        ForecastArray::new("w3d", w3d),
        ForecastArray::new("terrain_msl", terrain),
        ForecastArray::new("refc_dbz", refc),
        ForecastArray::new("refd_1000m_agl_dbz", refd_1000),
        ForecastArray::new("refd_4000m_agl_dbz", refd_4000),
    ];
    Ok(Cm1MappedFrame {
        target,
        forecast_hour: source.forecast_hour,
        source_fingerprint_sha256: source.source_fingerprint_sha256.clone(),
        arrays,
        stats,
    })
}

fn terrain_following_height_f32(surface_msl: f32, sigma_m: f32, model_top_m: f32) -> f32 {
    // CM1 r21.1 src/init_terrain.F:175-195. Keep the source operation order in
    // binary32 because zs, sigma[f], zt, zh, and zf are default REAL arrays.
    surface_msl + sigma_m * (model_top_m - surface_msl) / model_top_m
}

#[derive(Clone, Debug)]
pub struct HorizontalRemapAudit {
    source_fingerprint_sha256: String,
    source_refd_1000_sha256: String,
    mapped_refd_1000_sha256: String,
    nearest_refd_1000_sha256: String,
    target: Cm1TargetGrid,
    target_points: usize,
    fractional_samples: usize,
    four_corner_samples: usize,
    mapped_differs_from_nearest: usize,
    mapped_unique_f32_words: usize,
    nearest_unique_f32_words: usize,
    coordinate_roundtrip_max_abs_degrees: f64,
    source_boundary_probe_max_abs_dbz: f64,
}

impl HorizontalRemapAudit {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("algorithm", string(HORIZONTAL_REMAP_ALGORITHM)),
            ("claim_limit", string(HORIZONTAL_REMAP_CLAIM_LIMIT)),
            ("decoded_window_schema", string(DECODED_WINDOW_SCHEMA)),
            (
                "metrics",
                object(vec![
                    (
                        "coordinate_roundtrip_max_abs_degrees_f64_bits",
                        f64_bits(self.coordinate_roundtrip_max_abs_degrees),
                    ),
                    ("four_corner_samples", number(self.four_corner_samples)),
                    ("fractional_samples", number(self.fractional_samples)),
                    (
                        "mapped_differs_from_nearest",
                        number(self.mapped_differs_from_nearest),
                    ),
                    (
                        "mapped_unique_f32_words",
                        number(self.mapped_unique_f32_words),
                    ),
                    (
                        "nearest_unique_f32_words",
                        number(self.nearest_unique_f32_words),
                    ),
                    (
                        "source_boundary_probe_max_abs_dbz_f64_bits",
                        f64_bits(self.source_boundary_probe_max_abs_dbz),
                    ),
                    ("target_points", number(self.target_points)),
                ]),
            ),
            ("model", string("GM1")),
            (
                "result",
                string("PASS_COORDINATE_AWARE_BILINEAR_NOT_NEAREST_REPLICATION"),
            ),
            ("schema", string(HORIZONTAL_REMAP_AUDIT_SCHEMA)),
            (
                "source",
                object(vec![
                    ("field", string("refd_1000m_agl_dbz")),
                    ("field_f32le_sha256", string(&self.source_refd_1000_sha256)),
                    (
                        "fingerprint_sha256",
                        string(&self.source_fingerprint_sha256),
                    ),
                    ("forecast_hour", number(2)),
                    ("grid", string("HRRR canonical Lambert 3 km")),
                ]),
            ),
            (
                "target",
                object(vec![
                    (
                        "bilinear_field_f32le_sha256",
                        string(&self.mapped_refd_1000_sha256),
                    ),
                    (
                        "nearest_comparison_f32le_sha256",
                        string(&self.nearest_refd_1000_sha256),
                    ),
                    ("target_grid", target_json(self.target)),
                ]),
            ),
        ]);
        String::from_utf8(value.canonical_bytes()).expect("canonical remap audit is UTF-8")
    }

    pub fn sha256(&self) -> String {
        sha256::digest_hex(self.canonical_json().as_bytes())
    }

    pub fn write_new(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut bytes = self.canonical_json().into_bytes();
        bytes.push(b'\n');
        atomic_write_new(path.as_ref(), &bytes)
    }
}

pub fn audit_horizontal_remap(
    source: &DecodedHrrrWindow,
    target: Cm1TargetGrid,
) -> Result<HorizontalRemapAudit> {
    if source.forecast_hour != 2 {
        return Err(invalid(
            "horizontal remap audit requires the sealed f02 source",
        ));
    }
    let points = target
        .nx
        .checked_mul(target.ny)
        .ok_or_else(|| invalid("horizontal remap target point count overflow"))?;
    let mut mapped = Vec::with_capacity(points);
    let mut nearest = Vec::with_capacity(points);
    let mut mapped_words = BTreeSet::new();
    let mut nearest_words = BTreeSet::new();
    let mut fractional_samples = 0usize;
    let mut four_corner_samples = 0usize;
    let mut differs = 0usize;
    let mut coordinate_error = 0.0f64;
    for j in 0..target.ny {
        for i in 0..target.nx {
            let (latitude, longitude) = target.mass_lat_lon(i, j);
            let sample = source.grid.fractional_window_index(latitude, longitude)?;
            let fx = sample.x - sample.x.floor();
            let fy = sample.y - sample.y.floor();
            if fx != 0.0 || fy != 0.0 {
                fractional_samples += 1;
            }
            if fx > 0.0 && fx < 1.0 && fy > 0.0 && fy < 1.0 {
                four_corner_samples += 1;
            }
            let mapped_value = bilinear_dbz(&source.refd_1000, source.grid, sample) as f32;
            let nearest_i = (sample.x.round() as usize).min(source.grid.window_nx - 1);
            let nearest_j = (sample.y.round() as usize).min(source.grid.window_ny - 1);
            let nearest_value = source.refd_1000[nearest_j * source.grid.window_nx + nearest_i];
            if mapped_value.to_bits() != nearest_value.to_bits() {
                differs += 1;
            }
            mapped_words.insert(mapped_value.to_bits());
            nearest_words.insert(nearest_value.to_bits());
            mapped.push(mapped_value);
            nearest.push(nearest_value);

            let full_i = sample.x + source.grid.window_i_start as f64;
            let full_j = sample.y + source.grid.window_j_start as f64;
            let (roundtrip_lat, roundtrip_lon) = source.grid.grid_point_lat_lon(full_i, full_j)?;
            coordinate_error = coordinate_error
                .max((roundtrip_lat - latitude).abs())
                .max(normalized_longitude_delta(roundtrip_lon, longitude).abs());
        }
    }
    if fractional_samples == 0 || four_corner_samples == 0 || differs == 0 {
        return Err(invalid(
            "real horizontal remap does not demonstrate fractional four-corner interpolation distinct from nearest replication",
        ));
    }
    if mapped_words.len() <= nearest_words.len() {
        return Err(invalid(
            "real horizontal remap does not increase resolved f32 value diversity over nearest replication",
        ));
    }
    let epsilon = 1.0e-9;
    let mut boundary_probe = 0.0f64;
    for i in 1..source.grid.window_nx - 1 {
        for j in 0..source.grid.window_ny - 1 {
            let y = j as f64 + 0.371_093_75;
            let left = bilinear_dbz(
                &source.refd_1000,
                source.grid,
                Sample {
                    x: i as f64 - epsilon,
                    y,
                    rotation_cos: 1.0,
                    rotation_sin: 0.0,
                },
            );
            let right = bilinear_dbz(
                &source.refd_1000,
                source.grid,
                Sample {
                    x: i as f64 + epsilon,
                    y,
                    rotation_cos: 1.0,
                    rotation_sin: 0.0,
                },
            );
            boundary_probe = boundary_probe.max((right - left).abs());
        }
    }
    for j in 1..source.grid.window_ny - 1 {
        for i in 0..source.grid.window_nx - 1 {
            let x = i as f64 + 0.628_906_25;
            let south = bilinear_dbz(
                &source.refd_1000,
                source.grid,
                Sample {
                    x,
                    y: j as f64 - epsilon,
                    rotation_cos: 1.0,
                    rotation_sin: 0.0,
                },
            );
            let north = bilinear_dbz(
                &source.refd_1000,
                source.grid,
                Sample {
                    x,
                    y: j as f64 + epsilon,
                    rotation_cos: 1.0,
                    rotation_sin: 0.0,
                },
            );
            boundary_probe = boundary_probe.max((north - south).abs());
        }
    }
    if !coordinate_error.is_finite() || !boundary_probe.is_finite() {
        return Err(invalid(
            "horizontal remap audit produced a non-finite metric",
        ));
    }
    Ok(HorizontalRemapAudit {
        source_fingerprint_sha256: source.source_fingerprint_sha256.clone(),
        source_refd_1000_sha256: hash_f32_words(&source.refd_1000),
        mapped_refd_1000_sha256: hash_f32_words(&mapped),
        nearest_refd_1000_sha256: hash_f32_words(&nearest),
        target,
        target_points: points,
        fractional_samples,
        four_corner_samples,
        mapped_differs_from_nearest: differs,
        mapped_unique_f32_words: mapped_words.len(),
        nearest_unique_f32_words: nearest_words.len(),
        coordinate_roundtrip_max_abs_degrees: coordinate_error,
        source_boundary_probe_max_abs_dbz: boundary_probe,
    })
}

fn hash_f32_words(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    sha256::digest_hex(&bytes)
}

#[derive(Clone, Debug)]
pub struct InitializationReceipt {
    target: Cm1TargetGrid,
    contract_sha256: String,
    initial_source_sha256: String,
    boundary_source_sha256: String,
    initial_artifact: ForecastReceipt,
    boundary_artifact: ForecastReceipt,
    initial_stats: MappingStats,
    boundary_stats: MappingStats,
}

impl InitializationReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target: Cm1TargetGrid,
        contract: &ForecastContract,
        initial_source_sha256: impl Into<String>,
        boundary_source_sha256: impl Into<String>,
        initial_artifact: ForecastReceipt,
        boundary_artifact: ForecastReceipt,
        initial_stats: MappingStats,
        boundary_stats: MappingStats,
    ) -> Result<Self> {
        if initial_artifact.contract_sha256 != contract.sha256()
            || boundary_artifact.contract_sha256 != contract.sha256()
        {
            return Err(invalid(
                "initialization artifacts do not carry the supplied forecast contract",
            ));
        }
        if initial_artifact.model_time_f64_bits != "0000000000000000" {
            return Err(invalid(
                "initialization checkpoint must be at model time zero",
            ));
        }
        if boundary_artifact.step <= initial_artifact.step {
            return Err(invalid(
                "boundary frame must be later than the initialization frame",
            ));
        }
        let receipt = Self {
            target,
            contract_sha256: contract.sha256(),
            initial_source_sha256: initial_source_sha256.into(),
            boundary_source_sha256: boundary_source_sha256.into(),
            initial_artifact,
            boundary_artifact,
            initial_stats,
            boundary_stats,
        };
        for (label, value) in [
            ("initial source SHA-256", &receipt.initial_source_sha256),
            ("boundary source SHA-256", &receipt.boundary_source_sha256),
        ] {
            validate_sha256(value, label)?;
        }
        Ok(receipt)
    }

    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            (
                "boundary_artifact",
                forecast_receipt_json(&self.boundary_artifact),
            ),
            (
                "boundary_source_sha256",
                string(&self.boundary_source_sha256),
            ),
            ("boundary_stats", stats_json(&self.boundary_stats)),
            ("contract_sha256", string(&self.contract_sha256)),
            ("decoded_window_schema", string(DECODED_WINDOW_SCHEMA)),
            (
                "initial_artifact",
                forecast_receipt_json(&self.initial_artifact),
            ),
            ("initial_source_sha256", string(&self.initial_source_sha256)),
            ("initial_stats", stats_json(&self.initial_stats)),
            ("live_wiring_limit", string(LIVE_WIRING_LIMIT)),
            ("reflectivity_contract", string(REFLECTIVITY_CONTRACT)),
            ("schema", string(INITIALIZATION_RECEIPT_SCHEMA)),
            ("target", target_json(self.target)),
            ("terrain_contract", string(TERRAIN_CONTRACT)),
            ("thermodynamic_contract", string(THERMODYNAMIC_CONTRACT)),
            (
                "vertical_interpolation_contract",
                string(VERTICAL_INTERPOLATION_CONTRACT),
            ),
            ("vertical_motion_contract", string(VERTICAL_MOTION_CONTRACT)),
        ]);
        String::from_utf8(value.canonical_bytes()).expect("canonical JSON is UTF-8")
    }

    pub fn write_new(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut bytes = self.canonical_json().into_bytes();
        bytes.push(b'\n');
        atomic_write_new(path.as_ref(), &bytes)
    }
}

#[derive(Clone, Copy)]
struct Sample {
    x: f64,
    y: f64,
    rotation_cos: f64,
    rotation_sin: f64,
}

#[derive(Clone, Copy)]
struct Bracket {
    lower: usize,
    fraction: f64,
}

fn map_horizontal_wind(
    source: &DecodedHrrrWindow,
    target: Cm1TargetGrid,
    u_component: bool,
    mass_terrain: &[f32],
    output: &mut [f32],
) -> Result<()> {
    let (out_nx, out_ny) = if u_component {
        (target.nx + 1, target.ny)
    } else {
        (target.nx, target.ny + 1)
    };
    let mut hgt = [0.0_f64; HYBRID_LEVELS];
    let mut u = [0.0_f64; HYBRID_LEVELS];
    let mut v = [0.0_f64; HYBRID_LEVELS];
    let model_top_f32 = (target.nz as f64 * target.dz_m) as f32;
    if mass_terrain.len() != target.nx * target.ny {
        return Err(invalid(
            "mass terrain shape changed before staggered-wind mapping",
        ));
    }
    for j in 0..out_ny {
        for i in 0..out_nx {
            let (lat, lon) = if u_component {
                target.u_lat_lon(i, j)
            } else {
                target.v_lat_lon(i, j)
            };
            let sample = source.grid.fractional_window_index(lat, lon)?;
            let face_terrain = bilinear(&source.terrain, source.grid, sample) as f32;
            if !face_terrain.is_finite() || face_terrain >= model_top_f32 {
                return Err(invalid(
                    "staggered source terrain reaches or exceeds the CM1 model top",
                ));
            }
            fill_profile(&source.height, source.grid, sample, &mut hgt);
            fill_profile(&source.u_grid, source.grid, sample, &mut u);
            fill_profile(&source.v_grid, source.grid, sample, &mut v);
            for k in 0..target.nz {
                let sigma_m = ((k as f64 + 0.5) * target.dz_m) as f32;
                let (z_msl_f32, column_surface_f32) = staggered_wind_height_f32(
                    mass_terrain,
                    target,
                    u_component,
                    i,
                    j,
                    sigma_m,
                    model_top_f32,
                    face_terrain,
                );
                let z_msl = f64::from(z_msl_f32);
                let bracket = vertical_bracket(&hgt, z_msl, f64::from(column_surface_f32))?;
                let u_grid = interpolate_linear(&u, bracket);
                let v_grid = interpolate_linear(&v, bracket);
                let earth_u = sample.rotation_cos * u_grid + sample.rotation_sin * v_grid;
                let earth_v = -sample.rotation_sin * u_grid + sample.rotation_cos * v_grid;
                output[(k * out_ny + j) * out_nx + i] =
                    if u_component { earth_u } else { earth_v } as f32;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn staggered_wind_height_f32(
    mass_terrain: &[f32],
    target: Cm1TargetGrid,
    u_component: bool,
    i: usize,
    j: usize,
    sigma_m: f32,
    model_top_m: f32,
    outer_face_terrain_msl: f32,
) -> (f32, f32) {
    if u_component && i > 0 && i < target.nx {
        let left = mass_terrain[j * target.nx + i - 1];
        let right = mass_terrain[j * target.nx + i];
        let left_height = terrain_following_height_f32(left, sigma_m, model_top_m);
        let right_height = terrain_following_height_f32(right, sigma_m, model_top_m);
        return (
            0.5_f32 * (left_height + right_height),
            0.5_f32 * (left + right),
        );
    }
    if !u_component && j > 0 && j < target.ny {
        let south = mass_terrain[(j - 1) * target.nx + i];
        let north = mass_terrain[j * target.nx + i];
        let south_height = terrain_following_height_f32(south, sigma_m, model_top_m);
        let north_height = terrain_following_height_f32(north, sigma_m, model_top_m);
        return (
            0.5_f32 * (south_height + north_height),
            0.5_f32 * (south + north),
        );
    }
    (
        terrain_following_height_f32(outer_face_terrain_msl, sigma_m, model_top_m),
        outer_face_terrain_msl,
    )
}

fn validate_exact_inventory(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(invalid(format!(
            "decoded-window path {} must be a real directory",
            path.display()
        )));
    }
    let expected = expected_files();
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| invalid("decoded-window filename is not UTF-8"))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(invalid(format!(
                "decoded-window carrier {name:?} must be a regular non-symlink file"
            )));
        }
        actual.insert(name);
    }
    let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
    let extra = actual.difference(&expected).cloned().collect::<Vec<_>>();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(invalid(format!(
            "decoded-window carrier inventory mismatch: missing={missing:?}, extra={extra:?}; VVEL and direct REFC/REFD have no fallback"
        )));
    }
    Ok(())
}

fn expected_files() -> BTreeSet<String> {
    HYBRID_FILES
        .iter()
        .chain(SURFACE_FILES.iter())
        .chain(DIAGNOSTIC_FILES.iter())
        .map(|name| (*name).to_owned())
        .chain((1..=HYBRID_LEVELS).map(|level| format!("VVEL-{level:02}.f32le")))
        .collect()
}

fn read_f32(path: &Path, name: &str, words: usize) -> Result<Vec<f32>> {
    let file = path.join(name);
    let metadata = fs::symlink_metadata(&file)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(format!(
            "carrier {name} is not a regular non-symlink file"
        )));
    }
    let expected_bytes = words
        .checked_mul(4)
        .ok_or_else(|| invalid(format!("carrier {name} byte count overflows")))?;
    if metadata.len() != expected_bytes as u64 {
        return Err(invalid(format!(
            "carrier {name} has {} bytes; expected {expected_bytes}",
            metadata.len()
        )));
    }
    let bytes = fs::read(&file)?;
    let mut values = Vec::with_capacity(words);
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(chunk.try_into().expect("exact chunk"));
        if !value.is_finite() {
            return Err(invalid(format!(
                "carrier {name} value {index} is non-finite"
            )));
        }
        values.push(value);
    }
    Ok(values)
}

fn fingerprint_directory(path: &Path) -> Result<String> {
    let mut material = Vec::new();
    for name in expected_files() {
        let bytes = fs::read(path.join(&name))?;
        material.extend_from_slice(&(name.len() as u64).to_le_bytes());
        material.extend_from_slice(name.as_bytes());
        material.extend_from_slice(sha256::digest_hex(&bytes).as_bytes());
    }
    Ok(sha256::digest_hex(&material))
}

fn validate_positive(values: &[f32], name: &str) -> Result<()> {
    if values.iter().any(|value| *value <= 0.0) {
        return Err(invalid(format!("{name} contains a non-positive value")));
    }
    Ok(())
}

fn validate_nonnegative(values: &[f32], name: &str) -> Result<()> {
    if values.iter().any(|value| *value < 0.0) {
        return Err(invalid(format!("{name} contains a negative mixing ratio")));
    }
    Ok(())
}

fn validate_fraction(values: &[f32], name: &str) -> Result<()> {
    if values.iter().any(|value| !(0.0..1.0).contains(value)) {
        return Err(invalid(format!("{name} contains a value outside [0,1)")));
    }
    Ok(())
}

fn validate_fraction_closed(values: &[f32], name: &str) -> Result<()> {
    if values.iter().any(|value| !(0.0..=1.0).contains(value)) {
        return Err(invalid(format!("{name} contains a value outside [0,1]")));
    }
    Ok(())
}

fn validate_vertical_coordinates(height: &[f32], pressure: &[f32], cells: usize) -> Result<()> {
    for cell in 0..cells {
        for level in 1..HYBRID_LEVELS {
            let below = (level - 1) * cells + cell;
            let here = level * cells + cell;
            if height[here] <= height[below] {
                return Err(invalid(format!(
                    "HGT is not strictly increasing at source cell {cell}, hybrid levels {level} and {}",
                    level + 1
                )));
            }
            if pressure[here] >= pressure[below] {
                return Err(invalid(format!(
                    "PRES is not strictly decreasing at source cell {cell}, hybrid levels {level} and {}",
                    level + 1
                )));
            }
        }
    }
    Ok(())
}

fn validate_interpolated_column(
    height: &[f64; HYBRID_LEVELS],
    pressure: &[f64; HYBRID_LEVELS],
    terrain: f64,
) -> Result<()> {
    for level in 1..HYBRID_LEVELS {
        if height[level] <= height[level - 1] || pressure[level] >= pressure[level - 1] {
            return Err(invalid(
                "horizontal interpolation broke hybrid HGT/PRES monotonicity",
            ));
        }
    }
    if height[0] - terrain > MAX_BELOW_LOWEST_EXTRAPOLATION_M {
        return Err(invalid(format!(
            "lowest hybrid level is {:.3} m above terrain, beyond the {} m bounded extrapolation contract",
            height[0] - terrain,
            MAX_BELOW_LOWEST_EXTRAPOLATION_M
        )));
    }
    Ok(())
}

fn bilinear(values: &[f32], grid: HrrrLambertGrid, sample: Sample) -> f64 {
    let (i0, i1, fx) = interpolation_indices(sample.x, grid.window_nx);
    let (j0, j1, fy) = interpolation_indices(sample.y, grid.window_ny);
    let v00 = f64::from(values[j0 * grid.window_nx + i0]);
    let v10 = f64::from(values[j0 * grid.window_nx + i1]);
    let v01 = f64::from(values[j1 * grid.window_nx + i0]);
    let v11 = f64::from(values[j1 * grid.window_nx + i1]);
    (1.0 - fy) * ((1.0 - fx) * v00 + fx * v10) + fy * ((1.0 - fx) * v01 + fx * v11)
}

fn bilinear_dbz(values: &[f32], grid: HrrrLambertGrid, sample: Sample) -> f64 {
    let (i0, i1, fx) = interpolation_indices(sample.x, grid.window_nx);
    let (j0, j1, fy) = interpolation_indices(sample.y, grid.window_ny);
    let z = |i: usize, j: usize| 10.0_f64.powf(f64::from(values[j * grid.window_nx + i]) / 10.0);
    let linear = (1.0 - fy) * ((1.0 - fx) * z(i0, j0) + fx * z(i1, j0))
        + fy * ((1.0 - fx) * z(i0, j1) + fx * z(i1, j1));
    10.0 * linear.log10()
}

fn fill_profile(
    values: &[f32],
    grid: HrrrLambertGrid,
    sample: Sample,
    output: &mut [f64; HYBRID_LEVELS],
) {
    let cells = grid.window_nx * grid.window_ny;
    for (level, value) in output.iter_mut().enumerate() {
        *value = bilinear(&values[level * cells..(level + 1) * cells], grid, sample);
    }
}

fn interpolation_indices(coordinate: f64, extent: usize) -> (usize, usize, f64) {
    let lower = (coordinate.floor() as usize).min(extent - 2);
    (lower, lower + 1, coordinate - lower as f64)
}

fn vertical_bracket(
    height: &[f64; HYBRID_LEVELS],
    target_msl: f64,
    terrain_msl: f64,
) -> Result<Bracket> {
    if target_msl < terrain_msl - 1.0e-6 {
        return Err(invalid("target vertical coordinate lies below terrain"));
    }
    if target_msl > height[HYBRID_LEVELS - 1] {
        return Err(invalid(format!(
            "target height {target_msl:.3} m exceeds highest hybrid HGT {:.3} m",
            height[HYBRID_LEVELS - 1]
        )));
    }
    if target_msl < height[0] {
        if height[0] - target_msl > MAX_BELOW_LOWEST_EXTRAPOLATION_M {
            return Err(invalid("target lies too far below the lowest hybrid HGT"));
        }
        return Ok(Bracket {
            lower: 0,
            fraction: (target_msl - height[0]) / (height[1] - height[0]),
        });
    }
    let upper = height.partition_point(|value| *value <= target_msl);
    let lower = upper.saturating_sub(1).min(HYBRID_LEVELS - 2);
    Ok(Bracket {
        lower,
        fraction: (target_msl - height[lower]) / (height[lower + 1] - height[lower]),
    })
}

fn interpolate_linear(values: &[f64; HYBRID_LEVELS], bracket: Bracket) -> f64 {
    let lower = values[bracket.lower];
    lower + bracket.fraction * (values[bracket.lower + 1] - lower)
}

fn interpolate_log(values: &[f64; HYBRID_LEVELS], bracket: Bracket) -> f64 {
    let lower = values[bracket.lower].ln();
    (lower + bracket.fraction * (values[bracket.lower + 1].ln() - lower)).exp()
}

fn nonnegative_interpolate(
    values: &[f64; HYBRID_LEVELS],
    bracket: Bracket,
    name: &str,
) -> Result<f32> {
    let value = interpolate_linear(values, bracket);
    if value < -1.0e-12 || !value.is_finite() {
        return Err(invalid(format!(
            "vertical interpolation produced invalid {name} mixing ratio {value}"
        )));
    }
    Ok(value.max(0.0) as f32)
}

fn horizontal_mean(values: &[f32], levels: usize, cells: usize) -> Vec<f32> {
    (0..levels)
        .map(|level| {
            (values[level * cells..(level + 1) * cells]
                .iter()
                .map(|value| f64::from(*value))
                .sum::<f64>()
                / cells as f64) as f32
        })
        .collect()
}

fn staggered_mean_u(values: &[f32], target: Cm1TargetGrid) -> Vec<f32> {
    let cells = target.ny * (target.nx + 1);
    horizontal_mean(values, target.nz, cells)
}

fn staggered_mean_v(values: &[f32], target: Cm1TargetGrid) -> Vec<f32> {
    let cells = (target.ny + 1) * target.nx;
    horizontal_mean(values, target.nz, cells)
}

fn range(values: &[f32]) -> (f32, f32) {
    values.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(*value), maximum.max(*value)),
    )
}

fn range_reconstructed(perturbation: &[f32], base: &[f32], cells: usize) -> (f32, f32) {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for (level, base_value) in base.iter().enumerate() {
        for value in &perturbation[level * cells..(level + 1) * cells] {
            minimum = minimum.min(*value + *base_value);
            maximum = maximum.max(*value + *base_value);
        }
    }
    (minimum, maximum)
}

fn bounds_zyx(nz: usize, ny: usize, nx: usize) -> Vec<i32> {
    vec![1, nz as i32, 1, ny as i32, 1, nx as i32]
}

fn geodesic_offset(lat_deg: f64, lon_deg: f64, east_m: f64, north_m: f64) -> (f64, f64) {
    let distance = east_m.hypot(north_m);
    if distance == 0.0 {
        return (lat_deg, normalize_degrees(lon_deg));
    }
    let angular = distance / 6_371_229.0;
    let bearing = east_m.atan2(north_m);
    let lat1 = lat_deg.to_radians();
    let lon1 = lon_deg.to_radians();
    let lat2 = (lat1.sin() * angular.cos() + lat1.cos() * angular.sin() * bearing.cos()).asin();
    let lon2 = lon1
        + (bearing.sin() * angular.sin() * lat1.cos())
            .atan2(angular.cos() - lat1.sin() * lat2.sin());
    (lat2.to_degrees(), normalize_degrees(lon2.to_degrees()))
}

fn normalize_degrees(value: f64) -> f64 {
    let mut normalized = value % 360.0;
    if normalized >= 180.0 {
        normalized -= 360.0;
    }
    if normalized < -180.0 {
        normalized += 360.0;
    }
    normalized
}

fn normalized_longitude_delta(value: f64, origin: f64) -> f64 {
    normalize_degrees(value - normalize_degrees(origin))
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        )));
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

fn number(value: usize) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn f32_bits(value: f32) -> JsonValue {
    string(&format!("{:08x}", value.to_bits()))
}

fn f64_bits(value: f64) -> JsonValue {
    string(&format!("{:016x}", value.to_bits()))
}

fn target_json(target: Cm1TargetGrid) -> JsonValue {
    object(vec![
        ("center_lat_f64_bits", f64_bits(target.center_lat_deg)),
        ("center_lon_f64_bits", f64_bits(target.center_lon_deg)),
        ("dx_m_f64_bits", f64_bits(target.dx_m)),
        ("dy_m_f64_bits", f64_bits(target.dy_m)),
        ("dz_m_f64_bits", f64_bits(target.dz_m)),
        ("nx", number(target.nx)),
        ("ny", number(target.ny)),
        ("nz", number(target.nz)),
    ])
}

fn stats_json(stats: &MappingStats) -> JsonValue {
    object(vec![
        (
            "exner_perturbation_max_bits",
            f32_bits(stats.exner_perturbation_max),
        ),
        (
            "exner_perturbation_min_bits",
            f32_bits(stats.exner_perturbation_min),
        ),
        ("nonzero_w_cells", number(stats.nonzero_w_cells)),
        ("pressure_max_pa_bits", f32_bits(stats.pressure_max_pa)),
        ("pressure_min_pa_bits", f32_bits(stats.pressure_min_pa)),
        ("refc_max_dbz_bits", f32_bits(stats.refc_max_dbz)),
        ("refc_min_dbz_bits", f32_bits(stats.refc_min_dbz)),
        ("refd_1000_max_dbz_bits", f32_bits(stats.refd_1000_max_dbz)),
        ("refd_4000_max_dbz_bits", f32_bits(stats.refd_4000_max_dbz)),
        (
            "theta_perturbation_max_k_bits",
            f32_bits(stats.theta_perturbation_max_k),
        ),
        (
            "theta_perturbation_min_k_bits",
            f32_bits(stats.theta_perturbation_min_k),
        ),
        ("total_theta_max_k_bits", f32_bits(stats.total_theta_max_k)),
        ("total_theta_min_k_bits", f32_bits(stats.total_theta_min_k)),
        ("w_max_mps_bits", f32_bits(stats.w_max_mps)),
        ("w_min_mps_bits", f32_bits(stats.w_min_mps)),
    ])
}

fn forecast_receipt_json(receipt: &ForecastReceipt) -> JsonValue {
    object(vec![
        ("artifact_nbytes", number(receipt.artifact_nbytes)),
        ("artifact_sha256", string(&receipt.artifact_sha256)),
        ("contract_sha256", string(&receipt.contract_sha256)),
        ("header_sha256", string(&receipt.header_sha256)),
        ("kind", string(receipt.kind.as_str())),
        ("payload_nbytes", number(receipt.payload_nbytes)),
        ("payload_sha256", string(&receipt.payload_sha256)),
        ("step", JsonValue::Number(receipt.step.to_string())),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::{
        read_forecast_checkpoint, read_forecast_history, write_forecast_checkpoint_new,
        write_forecast_history_new,
    };
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("cm1-hrrr-{label}-{}-{nonce}", std::process::id()))
    }

    fn synthetic_grid() -> HrrrLambertGrid {
        HrrrLambertGrid::canonical_window(982, 633, 4, 4).unwrap()
    }

    fn write_values(path: &Path, values: impl Iterator<Item = f32>) {
        let mut file = fs::File::create(path).unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }

    fn make_window(path: &Path, time_delta: f32) {
        fs::create_dir_all(path).unwrap();
        let cells = 16;
        for name in expected_files() {
            let carrier_name = name.as_str();
            let levels = if HYBRID_FILES.contains(&name.as_str()) {
                HYBRID_LEVELS
            } else {
                1
            };
            let values = (0..levels).flat_map(|level| {
                (0..cells).map(move |cell| {
                    let x = (cell % 4) as f32;
                    let y = (cell / 4) as f32;
                    match carrier_name {
                        "PRES.f32le" => 98_000.0 * (-0.035 * level as f32).exp(),
                        "HGT.f32le" => 250.0 + 60.0 * x + 20.0 * y + 50.0 + 500.0 * level as f32,
                        "TT.f32le" => 298.0 - 2.8 * level as f32 + 0.2 * x + time_delta,
                        "SPFH.f32le" => (0.012 - 0.000_18 * level as f32).max(0.0002),
                        "QC.f32le" => 0.000_1 * ((level + cell) % 3) as f32,
                        "QI.f32le" => 0.000_02 * ((level + cell) % 2) as f32,
                        "QR.f32le" => 0.000_05 * ((level + cell) % 4) as f32,
                        "QS.f32le" => 0.000_01 * ((level + cell) % 2) as f32,
                        "QG.f32le" => 0.000_03 * ((level + cell) % 2) as f32,
                        "U_MASS.f32le" => 8.0 + 0.2 * x + 0.05 * level as f32,
                        "V_MASS.f32le" => 3.0 + 0.1 * y - 0.02 * level as f32,
                        "SOILHGT.f32le" => 250.0 + 60.0 * x + 20.0 * y,
                        "REFC.f32le" => 20.0 + x + y + 4.0 * time_delta,
                        "REFD-1000.f32le" => 15.0 + 2.0 * x + time_delta,
                        "REFD-4000.f32le" => 5.0 + 2.0 * y + time_delta,
                        value if value.starts_with("VVEL-") => {
                            -1.0 - 0.05 * x + 0.01 * level as f32 - time_delta
                        }
                        "PSFC.f32le" => 98_500.0,
                        "T2.f32le" | "SKINTEMP.f32le" => 299.0,
                        "Q2.f32le" => 0.012,
                        "LANDSEA.f32le" => 1.0,
                        "XICE.f32le" | "SNOW.f32le" | "SNOWH.f32le" => 0.0,
                        "U10_MASS.f32le" => 8.0,
                        "V10_MASS.f32le" => 3.0,
                        _ => unreachable!("unexpected synthetic carrier {carrier_name}"),
                    }
                })
            });
            write_values(&path.join(&name), values);
        }
    }

    fn synthetic_target(grid: HrrrLambertGrid) -> Cm1TargetGrid {
        let (lat, lon) = grid.grid_point_lat_lon(983.5, 634.5).unwrap();
        Cm1TargetGrid::new(2, 2, 3, 100.0, 100.0, 200.0, lat, lon).unwrap()
    }

    #[test]
    fn lambert_forward_inverse_round_trip_and_known_center_index() {
        let grid = HrrrLambertGrid::canonical_window(971, 622, 26, 26).unwrap();
        let (i, j) = grid.fractional_full_index(41.3053, -94.4619).unwrap();
        assert!((i - 983.854_700_616_960_4).abs() < 1.0e-9);
        assert!((j - 634.523_888_104_049_8).abs() < 1.0e-9);
        let (lat, lon) = grid.grid_point_lat_lon(i, j).unwrap();
        assert!((lat - 41.3053).abs() < 1.0e-10);
        assert!((normalize_degrees(lon + 94.4619)).abs() < 1.0e-10);
    }

    #[test]
    fn terrain_following_height_preserves_surface_and_level_model_top() {
        let surface = 312.75_f32;
        let top = 15_000.0_f32;
        assert_eq!(
            terrain_following_height_f32(surface, 0.0, top).to_bits(),
            surface.to_bits()
        );
        assert_eq!(
            terrain_following_height_f32(surface, top, top).to_bits(),
            top.to_bits()
        );
        let sigma = 7_500.0_f32;
        let expected = surface + sigma * (top - surface) / top;
        let actual = terrain_following_height_f32(surface, sigma, top);
        assert_eq!(actual.to_bits(), expected.to_bits());
        assert!(actual < surface + sigma);

        let target = Cm1TargetGrid::new(2, 2, 3, 100.0, 100.0, 200.0, 41.0, -94.0).unwrap();
        let terrain = [100.0_f32, 300.0_f32, 500.0_f32, 700.0_f32];
        let u = staggered_wind_height_f32(&terrain, target, true, 1, 0, 100.0, 600.0, 999.0);
        let u_expected = 0.5_f32
            * (terrain_following_height_f32(100.0, 100.0, 600.0)
                + terrain_following_height_f32(300.0, 100.0, 600.0));
        assert_eq!(u.0.to_bits(), u_expected.to_bits());
        assert_eq!(u.1.to_bits(), 200.0_f32.to_bits());
        let boundary = staggered_wind_height_f32(&terrain, target, true, 0, 0, 100.0, 600.0, 125.0);
        assert_eq!(
            boundary.0.to_bits(),
            terrain_following_height_f32(125.0, 100.0, 600.0).to_bits()
        );
        assert_eq!(boundary.1.to_bits(), 125.0_f32.to_bits());
    }

    #[test]
    fn strict_real_carriers_map_to_staggered_nonuniform_frames() {
        let root = temporary("map");
        let first = root.join("first");
        let second = root.join("second");
        make_window(&first, 0.0);
        make_window(&second, 1.0);
        let grid = synthetic_grid();
        let source0 = DecodedHrrrWindow::read_directory(&first, grid, 0).unwrap();
        let source1 = DecodedHrrrWindow::read_directory(&second, grid, 1).unwrap();
        let target = synthetic_target(grid);
        let mapped0 = map_to_cm1(&source0, target).unwrap();
        let mapped1 = map_to_cm1(&source1, target).unwrap();
        assert_eq!(mapped0.array("u3d").unwrap().len(), 3 * 2 * 3);
        assert_eq!(mapped0.array("v3d").unwrap().len(), 3 * 3 * 2);
        assert_eq!(mapped0.array("w3d").unwrap().len(), 4 * 2 * 2);
        assert!(mapped0.stats.nonzero_w_cells > 0);
        assert!(mapped0.stats.theta_perturbation_max_k > 0.0);
        assert!(mapped0.stats.theta_perturbation_min_k < 0.0);
        assert_ne!(mapped0.stats.refc_max_dbz, mapped1.stats.refc_max_dbz);
        assert_ne!(mapped0.array("w3d"), mapped1.array("w3d"));
        let initial_stats = mapped0.stats.clone();
        let boundary_stats = mapped1.stats.clone();
        let initial_source_sha = source0.source_fingerprint_sha256.clone();
        let boundary_source_sha = source1.source_fingerprint_sha256.clone();

        let config_sha = sha256::digest_hex(b"synthetic-target-config");
        let execution_sha = sha256::digest_hex(b"hrrr-module-test-execution");
        let contract = forecast_contract(target, config_sha, execution_sha).unwrap();
        let initial_path = root.join("initial.cm1fcp");
        let boundary_path = root.join("boundary.cm1fhs");
        let initial_receipt = write_forecast_checkpoint_new(
            &contract,
            &mapped0.into_forecast_frame(1.0).unwrap(),
            &initial_path,
        )
        .unwrap();
        let boundary_receipt = write_forecast_history_new(
            &contract,
            &mapped1.into_forecast_frame(1.0).unwrap(),
            &boundary_path,
        )
        .unwrap();
        assert_eq!(
            read_forecast_checkpoint(&contract, &initial_path)
                .unwrap()
                .frame()
                .clock()
                .step(),
            0
        );
        assert_eq!(
            read_forecast_history(&contract, &boundary_path)
                .unwrap()
                .frame()
                .clock()
                .step(),
            3600
        );
        assert_eq!(
            initial_receipt.contract_sha256,
            boundary_receipt.contract_sha256
        );
        let initialization_receipt = InitializationReceipt::new(
            target,
            &contract,
            initial_source_sha,
            boundary_source_sha,
            initial_receipt,
            boundary_receipt,
            initial_stats,
            boundary_stats,
        )
        .unwrap();
        let receipt_path = root.join("initialization-receipt.json");
        initialization_receipt.write_new(&receipt_path).unwrap();
        assert!(fs::read_to_string(receipt_path)
            .unwrap()
            .contains(INITIALIZATION_RECEIPT_SCHEMA));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn refuses_missing_vertical_motion_and_direct_radar_inputs() {
        for missing in [
            "VVEL-17.f32le",
            "REFC.f32le",
            "REFD-1000.f32le",
            "REFD-4000.f32le",
        ] {
            let root = temporary("missing");
            make_window(&root, 0.0);
            fs::remove_file(root.join(missing)).unwrap();
            let error = DecodedHrrrWindow::read_directory(&root, synthetic_grid(), 0)
                .unwrap_err()
                .to_string();
            assert!(error.contains("inventory mismatch"));
            assert!(error.contains("no fallback"));
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn refuses_extra_nonfinite_and_nonmonotone_carriers() {
        let extra = temporary("extra");
        make_window(&extra, 0.0);
        write_values(&extra.join("UNEXPECTED.f32le"), [1.0].into_iter());
        assert!(
            DecodedHrrrWindow::read_directory(&extra, synthetic_grid(), 0)
                .unwrap_err()
                .to_string()
                .contains("extra")
        );
        fs::remove_dir_all(extra).unwrap();

        let nan = temporary("nan");
        make_window(&nan, 0.0);
        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(nan.join("TT.f32le"))
            .unwrap();
        file.write_all(&f32::NAN.to_le_bytes()).unwrap();
        assert!(DecodedHrrrWindow::read_directory(&nan, synthetic_grid(), 0)
            .unwrap_err()
            .to_string()
            .contains("non-finite"));
        fs::remove_dir_all(nan).unwrap();

        let monotone = temporary("monotone");
        make_window(&monotone, 0.0);
        let mut file = fs::OpenOptions::new()
            .write(true)
            .open(monotone.join("HGT.f32le"))
            .unwrap();
        file.write_all(&20_000.0_f32.to_le_bytes()).unwrap();
        assert!(
            DecodedHrrrWindow::read_directory(&monotone, synthetic_grid(), 0)
                .unwrap_err()
                .to_string()
                .contains("not strictly increasing")
        );
        fs::remove_dir_all(monotone).unwrap();
    }
}
