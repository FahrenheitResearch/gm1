//! Source-anchored scientific initialization for the sealed product profile.
//!
//! This module is intentionally independent from the structural-sentinel
//! container.  It evaluates the CM1 r21.1 formulas in binary32, emits native
//! Fortran-order arrays, and can compare those arrays against a post-GETSET
//! capture without embedding any captured field values.

use crate::png::Canvas;
use crate::sha256;
use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;
use std::fs::{self, OpenOptions};
use std::io::{self, Write as IoWrite};
use std::path::{Path, PathBuf};

pub const SCIENTIFIC_SCHEMA: &str = "gpuwm-cm1.sealed-scientific-initialization/v1";
pub const COMPARISON_SCHEMA: &str = "gpuwm-cm1.native-scientific-comparison/v1";
pub const CLAIM_LIMIT: &str = "CM1 r21.1 flat-grid, isnd=5, iwnd=2 with moving frame, iinit=1, ibalance=0 initialization through the first nonrestart GETSET call for the sealed 120x120x40 product profile; no timestep, forecast, restart, CUDA, terrain, alternate namelist, or cross-platform libm bit-identity claim";

pub const NX: i32 = 120;
pub const NY: i32 = 120;
pub const NZ: i32 = 40;
pub const DX_M: f32 = 1000.0;
pub const DY_M: f32 = 1000.0;
pub const DZ_M: f32 = 500.0;

const SCALAR_BOUNDS: Bounds3 = Bounds3::new(-2, 123, -2, 123, 0, 41);
const U_BOUNDS: Bounds3 = Bounds3::new(-2, 124, -2, 123, 0, 41);
const V_BOUNDS: Bounds3 = Bounds3::new(-2, 123, -2, 124, 0, 41);
const W_BOUNDS: Bounds3 = Bounds3::new(-2, 123, -2, 123, 0, 42);

const G: f32 = 9.81;
const RD: f32 = 287.04;
const RV: f32 = 461.5;
const CP: f32 = 1005.7;
const P00: f32 = 100_000.0;
const REPS: f32 = RV / RD;
const EPS: f32 = RD / RV;
const CPDRD: f32 = CP / RD;
const EXTRAP_1: f32 = 1.875;
const EXTRAP_2: f32 = -1.25;
const EXTRAP_3: f32 = 0.375;

const SOURCE_ANCHORS: &[&str] = &[
    "run/config_files/supercell/namelist.input:27-30,44-47,53-57,64-81,99-100 (sealed inputs)",
    "src/constants.F:111-158 (default thermodynamic constants)",
    "src/param.F:6640-6695,7327-7408,7695-7745,7789-7812 (flat centered grid)",
    "src/base.F:359-450 (isnd=5 WK sounding)",
    "src/base.F:2041-2193,3204-3212 (base fills and reciprocals)",
    "src/base.F:2317-2360,2660-2718 (iwnd=2 and moving-frame fills)",
    "src/bc.F:528-934,1467-1711 (staggered open-boundary bands)",
    "src/comm.F:6511-6881 (MPI/global diagonal-corner semantics)",
    "src/init3d.F:204-223,280-324,445-482,1733-1743 (initial carriers and bubble)",
    "src/init3d.F:2140-2333 (GETSET fills and carrier copies)",
    "src/cm1.F:2390-2403,2660-2743,2884-2907 (call cadence and observation point)",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bounds3 {
    pub i_lo: i32,
    pub i_hi: i32,
    pub j_lo: i32,
    pub j_hi: i32,
    pub k_lo: i32,
    pub k_hi: i32,
}

impl Bounds3 {
    pub const fn new(i_lo: i32, i_hi: i32, j_lo: i32, j_hi: i32, k_lo: i32, k_hi: i32) -> Self {
        Self {
            i_lo,
            i_hi,
            j_lo,
            j_hi,
            k_lo,
            k_hi,
        }
    }

    pub fn shape_ijk(self) -> [usize; 3] {
        [
            (self.i_hi - self.i_lo + 1) as usize,
            (self.j_hi - self.j_lo + 1) as usize,
            (self.k_hi - self.k_lo + 1) as usize,
        ]
    }

    pub fn words(self) -> usize {
        let [ni, nj, nk] = self.shape_ijk();
        ni * nj * nk
    }

    fn contains(self, i: i32, j: i32, k: i32) -> bool {
        (self.i_lo..=self.i_hi).contains(&i)
            && (self.j_lo..=self.j_hi).contains(&j)
            && (self.k_lo..=self.k_hi).contains(&k)
    }

    fn offset(self, i: i32, j: i32, k: i32) -> usize {
        debug_assert!(self.contains(i, j, k));
        let [ni, nj, _] = self.shape_ijk();
        ((k - self.k_lo) as usize * nj + (j - self.j_lo) as usize) * ni + (i - self.i_lo) as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stagger {
    Scalar,
    UFace,
    VFace,
    WFace,
}

impl Stagger {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::UFace => "u_face",
            Self::VFace => "v_face",
            Self::WFace => "w_face",
        }
    }

    fn physical_bounds(self) -> Bounds3 {
        match self {
            Self::Scalar => Bounds3::new(1, NX, 1, NY, 1, NZ),
            Self::UFace => Bounds3::new(1, NX + 1, 1, NY, 1, NZ),
            Self::VFace => Bounds3::new(1, NX, 1, NY + 1, 1, NZ),
            Self::WFace => Bounds3::new(1, NX, 1, NY, 1, NZ + 1),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldFamily {
    Base,
    Core,
}

impl FieldFamily {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::Core => "core",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Coordinate1D {
    name: &'static str,
    axis: &'static str,
    location: &'static str,
    lower: i32,
    upper: i32,
    values: Vec<f32>,
}

impl Coordinate1D {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn lower(&self) -> i32 {
        self.lower
    }

    pub fn upper(&self) -> i32 {
        self.upper
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Clone, Debug)]
pub struct ScientificField {
    name: &'static str,
    family: FieldFamily,
    stagger: Stagger,
    bounds: Bounds3,
    values: Vec<f32>,
    native_file: &'static str,
}

impl ScientificField {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn family(&self) -> FieldFamily {
        self.family
    }

    pub fn stagger(&self) -> Stagger {
        self.stagger
    }

    pub fn bounds(&self) -> Bounds3 {
        self.bounds
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn value(&self, i: i32, j: i32, k: i32) -> Option<f32> {
        self.bounds
            .contains(i, j, k)
            .then(|| self.values[self.bounds.offset(i, j, k)])
    }

    pub fn native_file(&self) -> &'static str {
        self.native_file
    }
}

#[derive(Clone, Debug)]
pub struct ScientificState {
    coordinates: Vec<Coordinate1D>,
    fields: Vec<ScientificField>,
}

impl ScientificState {
    /// Evaluate the sealed scientific product profile from source formulas.
    #[allow(clippy::vec_init_then_push)] // Field order mirrors the source cadence and manifest.
    pub fn initialize_sealed_v1() -> Self {
        let coordinates = build_coordinates();
        let base = build_base_column();
        let (u0, v0) = build_wind_fields(&base.zh);

        let pi0 = expand_column(SCALAR_BOUNDS, &base.pi0);
        let prs0 = expand_column(SCALAR_BOUNDS, &base.prs0);
        let rho0 = expand_column(SCALAR_BOUNDS, &base.rho0);
        let thv0 = expand_column(SCALAR_BOUNDS, &base.thv0);
        let th0 = expand_column(SCALAR_BOUNDS, &base.th0);
        let rth0 = expand_column(SCALAR_BOUNDS, &base.rth0);
        let qv0 = expand_column(SCALAR_BOUNDS, &base.qv0);
        let bubble = build_warm_bubble(&coordinates, &base.zh);

        let mut fields = Vec::with_capacity(19);
        fields.push(field(
            "pi0",
            FieldFamily::Base,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            pi0,
        ));
        fields.push(field(
            "prs0",
            FieldFamily::Base,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            prs0,
        ));
        fields.push(field(
            "rho0",
            FieldFamily::Base,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            rho0,
        ));
        fields.push(field(
            "thv0",
            FieldFamily::Base,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            thv0,
        ));
        fields.push(field(
            "th0",
            FieldFamily::Base,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            th0,
        ));
        fields.push(field(
            "rth0",
            FieldFamily::Base,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            rth0,
        ));
        fields.push(field(
            "qv0",
            FieldFamily::Base,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            qv0,
        ));
        fields.push(field(
            "u0",
            FieldFamily::Base,
            Stagger::UFace,
            U_BOUNDS,
            u0.clone(),
        ));
        fields.push(field(
            "v0",
            FieldFamily::Base,
            Stagger::VFace,
            V_BOUNDS,
            v0.clone(),
        ));

        fields.push(field(
            "ua",
            FieldFamily::Core,
            Stagger::UFace,
            U_BOUNDS,
            u0.clone(),
        ));
        fields.push(field(
            "u3d",
            FieldFamily::Core,
            Stagger::UFace,
            U_BOUNDS,
            u0,
        ));
        fields.push(field(
            "va",
            FieldFamily::Core,
            Stagger::VFace,
            V_BOUNDS,
            v0.clone(),
        ));
        fields.push(field(
            "v3d",
            FieldFamily::Core,
            Stagger::VFace,
            V_BOUNDS,
            v0,
        ));
        fields.push(field(
            "wa",
            FieldFamily::Core,
            Stagger::WFace,
            W_BOUNDS,
            vec![0.0; W_BOUNDS.words()],
        ));
        fields.push(field(
            "w3d",
            FieldFamily::Core,
            Stagger::WFace,
            W_BOUNDS,
            vec![0.0; W_BOUNDS.words()],
        ));
        fields.push(field(
            "ppi",
            FieldFamily::Core,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            vec![0.0; SCALAR_BOUNDS.words()],
        ));
        fields.push(field(
            "pp3d",
            FieldFamily::Core,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            vec![0.0; SCALAR_BOUNDS.words()],
        ));
        fields.push(field(
            "tha",
            FieldFamily::Core,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            bubble.clone(),
        ));
        fields.push(field(
            "th3d",
            FieldFamily::Core,
            Stagger::Scalar,
            SCALAR_BOUNDS,
            bubble,
        ));

        Self {
            coordinates,
            fields,
        }
    }

    pub fn coordinates(&self) -> &[Coordinate1D] {
        &self.coordinates
    }

    pub fn fields(&self) -> &[ScientificField] {
        &self.fields
    }

    pub fn field(&self, name: &str) -> Option<&ScientificField> {
        self.fields.iter().find(|field| field.name == name)
    }

    /// Write all arrays and a deterministic LF-only manifest into a new directory.
    pub fn write_new(&self, output: impl AsRef<Path>) -> io::Result<WriteReceipt> {
        let output = output.as_ref();
        fs::create_dir(output).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "scientific output directory must be new ({}): {error}",
                    output.display()
                ),
            )
        })?;

        let mut entries = Vec::with_capacity(self.coordinates.len() + self.fields.len());
        for coordinate in &self.coordinates {
            let path = format!("{}.f32le", coordinate.name);
            let bytes = encode_f32le(&coordinate.values);
            write_new_file(&output.join(&path), &bytes)?;
            entries.push(ManifestEntry::coordinate(coordinate, path, &bytes));
        }
        for field in &self.fields {
            let path = format!("{}.f32le", field.name);
            let bytes = encode_f32le(&field.values);
            write_new_file(&output.join(&path), &bytes)?;
            entries.push(ManifestEntry::field(field, path, &bytes));
        }

        let manifest = manifest_json(&entries);
        let manifest_path = output.join("manifest.json");
        write_new_file(&manifest_path, manifest.as_bytes())?;

        Ok(WriteReceipt {
            output_dir: output.to_path_buf(),
            manifest_path,
            manifest_sha256: sha256::digest_hex(manifest.as_bytes()),
            arrays: entries.len(),
            total_array_bytes: entries.iter().map(|entry| entry.nbytes).sum(),
        })
    }

    /// Compare each emitted base/core field to a native post-GETSET capture.
    pub fn compare_native(&self, reference_dir: impl AsRef<Path>) -> io::Result<ComparisonReport> {
        let reference_dir = reference_dir.as_ref();
        let prefix = native_capture_prefix(reference_dir)?;
        validate_native_layout(reference_dir, &prefix, &self.fields)?;
        let mut comparisons = Vec::with_capacity(self.fields.len());
        for field in &self.fields {
            let native_path = reference_dir.join(format!("{prefix}{}", field.native_file));
            let native = fs::read(&native_path).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "cannot read native field {}: {error}",
                        native_path.display()
                    ),
                )
            })?;
            let expected_bytes = field.bounds.words() * 4;
            if native.len() != expected_bytes {
                return Err(invalid_data(format!(
                    "native field {} has {} bytes; expected {expected_bytes}",
                    native_path.display(),
                    native.len()
                )));
            }
            comparisons.push(compare_field(field, &native));
        }
        Ok(ComparisonReport {
            reference_dir: reference_dir.to_path_buf(),
            fields: comparisons,
        })
    }

    /// Render a labeled, dependency-free PNG summary of a native comparison.
    pub fn write_comparison_png(
        &self,
        report: &ComparisonReport,
        reference_dir: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> io::Result<String> {
        let png = comparison_png(self, report, reference_dir.as_ref())?;
        write_new_file(output.as_ref(), &png)?;
        Ok(sha256::digest_hex(&png))
    }
}

fn field(
    name: &'static str,
    family: FieldFamily,
    stagger: Stagger,
    bounds: Bounds3,
    values: Vec<f32>,
) -> ScientificField {
    debug_assert_eq!(bounds.words(), values.len());
    ScientificField {
        name,
        family,
        stagger,
        bounds,
        values,
        native_file: native_filename(name),
    }
}

fn native_filename(name: &'static str) -> &'static str {
    match name {
        "pi0" => "post-getset-pi0.f32le",
        "prs0" => "post-getset-prs0.f32le",
        "rho0" => "post-getset-rho0.f32le",
        "thv0" => "post-getset-thv0.f32le",
        "th0" => "post-getset-th0.f32le",
        "rth0" => "post-getset-rth0.f32le",
        "qv0" => "post-getset-qv0.f32le",
        "u0" => "post-getset-u0.f32le",
        "v0" => "post-getset-v0.f32le",
        "ua" => "post-getset-ua.f32le",
        "u3d" => "post-getset-u3d.f32le",
        "va" => "post-getset-va.f32le",
        "v3d" => "post-getset-v3d.f32le",
        "wa" => "post-getset-wa.f32le",
        "w3d" => "post-getset-w3d.f32le",
        "ppi" => "post-getset-ppi.f32le",
        "pp3d" => "post-getset-pp3d.f32le",
        "tha" => "post-getset-tha.f32le",
        "th3d" => "post-getset-th3d.f32le",
        _ => "",
    }
}

fn build_coordinates() -> Vec<Coordinate1D> {
    let x_face = coordinate_values(-2, 124, |i| {
        ((DX_M as f64) * ((i - 1) as f64) - 0.5_f64 * (DX_M as f64) * (NX as f64)) as f32
    });
    let x_mass = coordinate_values(-2, 123, |i| {
        let left = (DX_M as f64) * ((i - 1) as f64) - 0.5_f64 * (DX_M as f64) * (NX as f64);
        let right = (DX_M as f64) * (i as f64) - 0.5_f64 * (DX_M as f64) * (NX as f64);
        (0.5_f64 * (left + right)) as f32
    });
    let y_face = coordinate_values(-2, 124, |j| {
        ((DY_M as f64) * ((j - 1) as f64) - 0.5_f64 * (DY_M as f64) * (NY as f64)) as f32
    });
    let y_mass = coordinate_values(-2, 123, |j| {
        let left = (DY_M as f64) * ((j - 1) as f64) - 0.5_f64 * (DY_M as f64) * (NY as f64);
        let right = (DY_M as f64) * (j as f64) - 0.5_f64 * (DY_M as f64) * (NY as f64);
        (0.5_f64 * (left + right)) as f32
    });
    let z_face = coordinate_values(0, 42, |k| DZ_M * ((k - 1) as f32));
    let z_mass = coordinate_values(0, 41, |k| {
        let lower = DZ_M * ((k - 1) as f32);
        let upper = DZ_M * (k as f32);
        0.5 * (lower + upper)
    });

    vec![
        Coordinate1D {
            name: "xh",
            axis: "i",
            location: "mass",
            lower: -2,
            upper: 123,
            values: x_mass,
        },
        Coordinate1D {
            name: "xf",
            axis: "i",
            location: "face",
            lower: -2,
            upper: 124,
            values: x_face,
        },
        Coordinate1D {
            name: "yh",
            axis: "j",
            location: "mass",
            lower: -2,
            upper: 123,
            values: y_mass,
        },
        Coordinate1D {
            name: "yf",
            axis: "j",
            location: "face",
            lower: -2,
            upper: 124,
            values: y_face,
        },
        Coordinate1D {
            name: "zh",
            axis: "k",
            location: "mass-flat-reduction",
            lower: 0,
            upper: 41,
            values: z_mass,
        },
        Coordinate1D {
            name: "zf",
            axis: "k",
            location: "face-flat-reduction",
            lower: 0,
            upper: 42,
            values: z_face,
        },
    ]
}

fn coordinate_values(mut lower: i32, upper: i32, mut value: impl FnMut(i32) -> f32) -> Vec<f32> {
    let mut values = Vec::with_capacity((upper - lower + 1) as usize);
    while lower <= upper {
        values.push(value(lower));
        lower += 1;
    }
    values
}

struct BaseColumn {
    zh: Vec<f32>,
    pi0: Vec<f32>,
    prs0: Vec<f32>,
    rho0: Vec<f32>,
    thv0: Vec<f32>,
    th0: Vec<f32>,
    rth0: Vec<f32>,
    qv0: Vec<f32>,
}

fn build_base_column() -> BaseColumn {
    let zh = coordinate_values(0, 41, |k| {
        let lower = DZ_M * ((k - 1) as f32);
        let upper = DZ_M * (k as f32);
        0.5 * (lower + upper)
    });
    let mut th0 = vec![0.0_f32; 42];
    let mut rh0 = vec![0.0_f32; 42];
    let mut qv0 = vec![0.0_f32; 42];
    let mut thv0 = vec![0.0_f32; 42];
    let mut pi0 = vec![0.0_f32; 42];
    let mut prs0 = vec![0.0_f32; 42];

    let z_trop = 12_000.0_f32;
    let th_trop = 343.0_f32;
    let t_trop = 213.0_f32;
    let th_sfc = 300.0_f32;
    let prs_sfc = 100_000.0_f32;
    let qv_pbl = 0.014_f32;
    let pi_sfc = (prs_sfc / P00).powf(RD / CP);
    let qv_sfc = rslf(prs_sfc, th_sfc * pi_sfc);
    let thv_sfc = th_sfc * (1.0 + qv_sfc * REPS) / (1.0 + qv_sfc);

    for k in 1..=NZ as usize {
        if zh[k] < z_trop {
            let power = (zh[k] / z_trop).powf(1.25);
            th0[k] = th_sfc + (th_trop - th_sfc) * power;
            rh0[k] = 1.0 - 0.75 * power;
        } else {
            th0[k] = th_trop * ((G / (t_trop * CP)) * (zh[k] - z_trop)).exp();
            rh0[k] = 0.25;
        }
    }
    th0[0] = th0[1];
    th0[NZ as usize + 1] = th0[NZ as usize];

    for _ in 0..20 {
        for k in 0..=NZ as usize + 1 {
            thv0[k] = th0[k] * (1.0 + REPS * qv0[k]) / (1.0 + qv0[k]);
        }
        pi0[1] = pi_sfc - G * zh[1] / (CP * 0.5 * (thv_sfc + thv0[1]));
        for k in 2..=NZ as usize {
            pi0[k] = pi0[k - 1] - G * (zh[k] - zh[k - 1]) / (CP * 0.5 * (thv0[k] + thv0[k - 1]));
        }
        for k in 1..=NZ as usize {
            prs0[k] = P00 * pi0[k].powf(CPDRD);
        }
        for k in 1..=NZ as usize {
            qv0[k] = rh0[k] * rslf(prs0[k], th0[k] * pi0[k]);
            if qv0[k] > qv_pbl {
                qv0[k] = qv_pbl;
            }
        }
    }

    extrapolate(&mut pi0);
    extrapolate(&mut prs0);
    extrapolate(&mut th0);
    extrapolate(&mut qv0);
    extrapolate(&mut rh0);

    let mut rho0 = vec![0.0_f32; 42];
    let mut rth0 = vec![0.0_f32; 42];
    for k in 0..=NZ as usize + 1 {
        thv0[k] = (th0[k] + 0.0) * (1.0 + REPS * qv0[k].max(0.0))
            / (1.0 + qv0[k].max(0.0) + 0.0_f32.max(0.0));
        prs0[k] = P00 * pi0[k].powf(CPDRD);
        rho0[k] = prs0[k] / (RD * th0[k] * pi0[k] * (1.0 + qv0[k] * REPS));
        rth0[k] = 1.0 / th0[k];
    }
    extrapolate(&mut rho0);

    BaseColumn {
        zh,
        pi0,
        prs0,
        rho0,
        thv0,
        th0,
        rth0,
        qv0,
    }
}

fn rslf(pressure: f32, temperature: f32) -> f32 {
    let mut esl = 611.2 * (17.67 * (temperature - 273.15) / (temperature - 29.65)).exp();
    esl = esl.min(pressure * 0.5);
    EPS * esl / (pressure - esl)
}

fn extrapolate(values: &mut [f32]) {
    debug_assert_eq!(values.len(), NZ as usize + 2);
    values[0] = EXTRAP_1 * values[1] + EXTRAP_2 * values[2] + EXTRAP_3 * values[3];
    let top = NZ as usize + 1;
    values[top] =
        EXTRAP_1 * values[top - 1] + EXTRAP_2 * values[top - 2] + EXTRAP_3 * values[top - 3];
}

fn build_wind_fields(zh: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut u = vec![0.0_f32; U_BOUNDS.words()];
    let mut v = vec![0.0_f32; V_BOUNDS.words()];

    // base.F:2328-2360 initializes only the physical staggered bands.
    for k in 1..=NZ {
        let zu = 0.5 * (zh[k as usize] + zh[k as usize]);
        let u_value = if zu <= 2000.0 {
            let angle = 90.0 * (zu / 2000.0) * (std::f32::consts::PI / 180.0);
            7.0 - 7.0 * angle.cos()
        } else if zu <= 6000.0 {
            7.0 + (zu - 2000.0) * (31.0 - 7.0) / (6000.0 - 2000.0)
        } else {
            31.0
        };
        for j in 1..=NY {
            for i in 1..=NX + 1 {
                u[U_BOUNDS.offset(i, j, k)] = u_value;
            }
        }

        let zv = 0.5 * (zh[k as usize] + zh[k as usize]);
        let v_value = if zv <= 2000.0 {
            let angle = 90.0 * (zv / 2000.0) * (std::f32::consts::PI / 180.0);
            7.0 * angle.sin()
        } else {
            7.0
        };
        for j in 1..=NY + 1 {
            for i in 1..=NX {
                v[V_BOUNDS.offset(i, j, k)] = v_value;
            }
        }
    }

    // base.F:2663-2671 applies both moving-frame components over the shared
    // i=1:ni+1, j=1:nj+1 band, including the extra stagger cross-lines.
    for k in 1..=NZ {
        for j in 1..=NY + 1 {
            for i in 1..=NX + 1 {
                let u_offset = U_BOUNDS.offset(i, j, k);
                let v_offset = V_BOUNDS.offset(i, j, k);
                u[u_offset] -= 12.5;
                v[v_offset] -= 3.0;
            }
        }
    }

    // base.F:2677-2718: open physical bands, no diagonal MPI neighbor at a
    // global corner, then vertical extrapolation over the full allocations.
    bcu_open(&mut u);
    bcv_open(&mut v);
    extrapolate_field_vertical(&mut u, U_BOUNDS);
    extrapolate_field_vertical(&mut v, V_BOUNDS);

    // init3d.F:2149-2150 repeats the physical-k boundary fills during GETSET.
    // It repairs the intersection bands at k=1:nk but cannot change k ghosts
    // or the deliberately unfilled outer Cartesian corner rectangles.
    bcu_open(&mut u);
    bcv_open(&mut v);
    (u, v)
}

fn bcu_open(values: &mut [f32]) {
    for k in 1..=NZ {
        for j in 0..=NY + 1 {
            let west = values[U_BOUNDS.offset(1, j, k)];
            for i in -2..=0 {
                values[U_BOUNDS.offset(i, j, k)] = west;
            }
            let east = values[U_BOUNDS.offset(NX + 1, j, k)];
            for i in NX + 2..=NX + 4 {
                values[U_BOUNDS.offset(i, j, k)] = east;
            }
        }
        for j in -2..=0 {
            for i in 0..=NX + 2 {
                values[U_BOUNDS.offset(i, j, k)] = values[U_BOUNDS.offset(i, 1, k)];
            }
        }
        for j in NY + 1..=NY + 3 {
            for i in 0..=NX + 2 {
                values[U_BOUNDS.offset(i, j, k)] = values[U_BOUNDS.offset(i, NY, k)];
            }
        }
    }
}

fn bcv_open(values: &mut [f32]) {
    for k in 1..=NZ {
        for j in -2..=0 {
            for i in 0..=NX + 1 {
                values[V_BOUNDS.offset(i, j, k)] = values[V_BOUNDS.offset(i, 1, k)];
            }
        }
        for j in NY + 2..=NY + 4 {
            for i in 0..=NX + 1 {
                values[V_BOUNDS.offset(i, j, k)] = values[V_BOUNDS.offset(i, NY + 1, k)];
            }
        }
        for j in 0..=NY + 2 {
            let west = values[V_BOUNDS.offset(1, j, k)];
            for i in -2..=0 {
                values[V_BOUNDS.offset(i, j, k)] = west;
            }
            let east = values[V_BOUNDS.offset(NX, j, k)];
            for i in NX + 1..=NX + 3 {
                values[V_BOUNDS.offset(i, j, k)] = east;
            }
        }
    }
}

fn extrapolate_field_vertical(values: &mut [f32], bounds: Bounds3) {
    for j in bounds.j_lo..=bounds.j_hi {
        for i in bounds.i_lo..=bounds.i_hi {
            values[bounds.offset(i, j, 0)] = EXTRAP_1 * values[bounds.offset(i, j, 1)]
                + EXTRAP_2 * values[bounds.offset(i, j, 2)]
                + EXTRAP_3 * values[bounds.offset(i, j, 3)];
            values[bounds.offset(i, j, NZ + 1)] = EXTRAP_1 * values[bounds.offset(i, j, NZ)]
                + EXTRAP_2 * values[bounds.offset(i, j, NZ - 1)]
                + EXTRAP_3 * values[bounds.offset(i, j, NZ - 2)];
        }
    }
}

fn expand_column(bounds: Bounds3, column: &[f32]) -> Vec<f32> {
    let mut values = Vec::with_capacity(bounds.words());
    for k in bounds.k_lo..=bounds.k_hi {
        let value = column[k as usize];
        for _j in bounds.j_lo..=bounds.j_hi {
            for _i in bounds.i_lo..=bounds.i_hi {
                values.push(value);
            }
        }
    }
    values
}

fn build_warm_bubble(coordinates: &[Coordinate1D], zh: &[f32]) -> Vec<f32> {
    let xh = coordinates
        .iter()
        .find(|coordinate| coordinate.name == "xh")
        .unwrap();
    let yh = coordinates
        .iter()
        .find(|coordinate| coordinate.name == "yh")
        .unwrap();
    let mut values = vec![0.0_f32; SCALAR_BOUNDS.words()];
    for k in 1..=NZ {
        for j in 1..=NY {
            let y = yh.values[(j - yh.lower) as usize];
            for i in 1..=NX {
                let x = xh.values[(i - xh.lower) as usize];
                let bx = (x - 0.0) / 10_000.0;
                let by = (y - 0.0) / 10_000.0;
                let bz = (zh[k as usize] - 1400.0) / 1400.0;
                let beta = (bx * bx + by * by + bz * bz).sqrt();
                if beta < 1.0 {
                    let cosine = (0.5 * std::f32::consts::PI * beta).cos();
                    values[SCALAR_BOUNDS.offset(i, j, k)] = 1.0 * (cosine * cosine);
                }
            }
        }
    }
    values
}

fn encode_f32le(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn write_new_file(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.flush()
}

struct ManifestEntry {
    name: &'static str,
    family: &'static str,
    stagger: &'static str,
    axes: &'static str,
    bounds: Vec<[i32; 2]>,
    shape: Vec<usize>,
    path: String,
    nbytes: usize,
    sha256: String,
    native_file: Option<&'static str>,
    location: Option<&'static str>,
}

impl ManifestEntry {
    fn coordinate(coordinate: &Coordinate1D, path: String, bytes: &[u8]) -> Self {
        Self {
            name: coordinate.name,
            family: "coordinate",
            stagger: coordinate.location,
            axes: coordinate.axis,
            bounds: vec![[coordinate.lower, coordinate.upper]],
            shape: vec![coordinate.values.len()],
            path,
            nbytes: bytes.len(),
            sha256: sha256::digest_hex(bytes),
            native_file: None,
            location: Some(coordinate.location),
        }
    }

    fn field(field: &ScientificField, path: String, bytes: &[u8]) -> Self {
        let [ni, nj, nk] = field.bounds.shape_ijk();
        Self {
            name: field.name,
            family: field.family.as_str(),
            stagger: field.stagger.as_str(),
            axes: "ijk",
            bounds: vec![
                [field.bounds.i_lo, field.bounds.i_hi],
                [field.bounds.j_lo, field.bounds.j_hi],
                [field.bounds.k_lo, field.bounds.k_hi],
            ],
            shape: vec![ni, nj, nk],
            path,
            nbytes: bytes.len(),
            sha256: sha256::digest_hex(bytes),
            native_file: Some(field.native_file),
            location: None,
        }
    }
}

fn manifest_json(entries: &[ManifestEntry]) -> String {
    let mut json = String::new();
    writeln!(&mut json, "{{").unwrap();
    writeln!(&mut json, "  \"schema\": \"{SCIENTIFIC_SCHEMA}\",").unwrap();
    writeln!(&mut json, "  \"claim_limit\": \"{CLAIM_LIMIT}\",").unwrap();
    writeln!(
        &mut json,
        "  \"profile\": \"sealed-product-120x120x40/v1\","
    )
    .unwrap();
    writeln!(
        &mut json,
        "  \"dimensions\": {{\"nx\": {NX}, \"ny\": {NY}, \"nz\": {NZ}}},"
    )
    .unwrap();
    writeln!(
        &mut json,
        "  \"spacing_m\": [{DX_M:.1}, {DY_M:.1}, {DZ_M:.1}],"
    )
    .unwrap();
    writeln!(&mut json, "  \"binary_contract\": \"raw f32 little-endian; fields use Fortran storage order (i fastest, then j, then k)\",").unwrap();
    writeln!(&mut json, "  \"coordinate_contract\": \"flat-grid 1-D source reductions; field arrays retain complete native allocation bounds\",").unwrap();
    writeln!(&mut json, "  \"source_authority\": {{").unwrap();
    writeln!(&mut json, "    \"model\": \"CM1 r21.1\",").unwrap();
    writeln!(&mut json, "    \"anchors\": [").unwrap();
    for (index, anchor) in SOURCE_ANCHORS.iter().enumerate() {
        let comma = if index + 1 == SOURCE_ANCHORS.len() {
            ""
        } else {
            ","
        };
        writeln!(&mut json, "      \"{anchor}\"{comma}").unwrap();
    }
    writeln!(&mut json, "    ]").unwrap();
    writeln!(&mut json, "  }},").unwrap();
    writeln!(&mut json, "  \"arrays\": [").unwrap();
    for (index, entry) in entries.iter().enumerate() {
        let comma = if index + 1 == entries.len() { "" } else { "," };
        let bounds = entry
            .bounds
            .iter()
            .map(|pair| format!("[{}, {}]", pair[0], pair[1]))
            .collect::<Vec<_>>()
            .join(", ");
        let shape = entry
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let native = entry
            .native_file
            .map(|path| format!("\"{path}\""))
            .unwrap_or_else(|| "null".to_string());
        let location = entry
            .location
            .map(|value| format!(", \"location\": \"{value}\""))
            .unwrap_or_default();
        writeln!(
            &mut json,
            "    {{\"name\": \"{}\", \"family\": \"{}\", \"stagger\": \"{}\", \"axes\": \"{}\", \"bounds\": [{}], \"shape\": [{}], \"dtype\": \"<f4\", \"path\": \"{}\", \"nbytes\": {}, \"sha256\": \"{}\", \"native_reference_path\": {}{}}}{}",
            entry.name,
            entry.family,
            entry.stagger,
            entry.axes,
            bounds,
            shape,
            entry.path,
            entry.nbytes,
            entry.sha256,
            native,
            location,
            comma
        )
        .unwrap();
    }
    writeln!(&mut json, "  ],").unwrap();
    writeln!(&mut json, "  \"deferred\": [\"moisture carrier families\", \"TKE\", \"derived rho/rr/rf/prs/ppx diagnostics\"]").unwrap();
    writeln!(&mut json, "}}").unwrap();
    json
}

#[derive(Clone, Debug)]
pub struct WriteReceipt {
    pub output_dir: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub arrays: usize,
    pub total_array_bytes: usize,
}

#[derive(Clone, Debug, Default)]
pub struct RegionStats {
    pub words: u64,
    pub exact_bits: u64,
    pub max_ulp: u32,
    pub max_abs: f64,
    pub non_finite_mismatches: u64,
}

impl RegionStats {
    fn observe(&mut self, actual: f32, expected: f32) {
        self.words += 1;
        if actual.to_bits() == expected.to_bits() {
            self.exact_bits += 1;
            return;
        }
        if actual.is_finite() && expected.is_finite() {
            self.max_ulp = self.max_ulp.max(ulp_distance(actual, expected));
            self.max_abs = self.max_abs.max((actual as f64 - expected as f64).abs());
        } else {
            self.non_finite_mismatches += 1;
            self.max_ulp = u32::MAX;
            self.max_abs = f64::INFINITY;
        }
    }
}

#[derive(Clone, Debug)]
pub struct FieldComparison {
    pub name: &'static str,
    pub stagger: Stagger,
    pub bounds: Bounds3,
    pub generated_sha256: String,
    pub native_sha256: String,
    pub all: RegionStats,
    pub interior: RegionStats,
    pub boundary: RegionStats,
}

#[derive(Clone, Debug)]
pub struct ComparisonReport {
    pub reference_dir: PathBuf,
    pub fields: Vec<FieldComparison>,
}

impl ComparisonReport {
    pub fn exact_fields(&self) -> usize {
        self.fields
            .iter()
            .filter(|comparison| comparison.all.exact_bits == comparison.all.words)
            .count()
    }

    pub fn max_ulp(&self) -> u32 {
        self.fields
            .iter()
            .map(|field| field.all.max_ulp)
            .max()
            .unwrap_or(0)
    }

    pub fn max_abs(&self) -> f64 {
        self.fields
            .iter()
            .map(|field| field.all.max_abs)
            .fold(0.0, f64::max)
    }

    pub fn to_json(&self) -> String {
        comparison_json(self)
    }

    pub fn write_new(&self, path: impl AsRef<Path>) -> io::Result<String> {
        let json = self.to_json();
        write_new_file(path.as_ref(), json.as_bytes())?;
        Ok(sha256::digest_hex(json.as_bytes()))
    }
}

fn compare_field(field: &ScientificField, native: &[u8]) -> FieldComparison {
    let mut all = RegionStats::default();
    let mut interior = RegionStats::default();
    let mut boundary = RegionStats::default();
    let physical = field.stagger.physical_bounds();
    let mut word = 0_usize;
    for k in field.bounds.k_lo..=field.bounds.k_hi {
        for j in field.bounds.j_lo..=field.bounds.j_hi {
            for i in field.bounds.i_lo..=field.bounds.i_hi {
                let start = word * 4;
                let expected = f32::from_le_bytes(native[start..start + 4].try_into().unwrap());
                let actual = field.values[word];
                all.observe(actual, expected);
                if physical.contains(i, j, k) {
                    interior.observe(actual, expected);
                } else {
                    boundary.observe(actual, expected);
                }
                word += 1;
            }
        }
    }
    FieldComparison {
        name: field.name,
        stagger: field.stagger,
        bounds: field.bounds,
        generated_sha256: sha256::digest_hex(&encode_f32le(&field.values)),
        native_sha256: sha256::digest_hex(native),
        all,
        interior,
        boundary,
    }
}

fn ulp_distance(left: f32, right: f32) -> u32 {
    fn ordered(value: f32) -> u32 {
        let bits = value.to_bits();
        if bits & 0x8000_0000 == 0 {
            bits | 0x8000_0000
        } else {
            !bits
        }
    }
    ordered(left).abs_diff(ordered(right))
}

fn comparison_json(report: &ComparisonReport) -> String {
    let mut json = String::new();
    writeln!(&mut json, "{{").unwrap();
    writeln!(&mut json, "  \"schema\": \"{COMPARISON_SCHEMA}\",").unwrap();
    writeln!(&mut json, "  \"claim_limit\": \"{CLAIM_LIMIT}\",").unwrap();
    writeln!(&mut json, "  \"reference_contract\": \"post_getset call 1, raw f32 little-endian, Fortran i-fastest, identical native indices, no regridding\",").unwrap();
    writeln!(&mut json, "  \"fields_exact\": {},", report.exact_fields()).unwrap();
    writeln!(&mut json, "  \"fields_compared\": {},", report.fields.len()).unwrap();
    writeln!(&mut json, "  \"max_ulp\": {},", report.max_ulp()).unwrap();
    writeln!(
        &mut json,
        "  \"max_abs\": {},",
        json_float(report.max_abs())
    )
    .unwrap();
    writeln!(&mut json, "  \"fields\": [").unwrap();
    for (index, field) in report.fields.iter().enumerate() {
        let comma = if index + 1 == report.fields.len() {
            ""
        } else {
            ","
        };
        writeln!(&mut json, "    {{").unwrap();
        writeln!(&mut json, "      \"name\": \"{}\",", field.name).unwrap();
        writeln!(
            &mut json,
            "      \"stagger\": \"{}\",",
            field.stagger.as_str()
        )
        .unwrap();
        writeln!(
            &mut json,
            "      \"shape_ijk\": {:?},",
            field.bounds.shape_ijk()
        )
        .unwrap();
        writeln!(
            &mut json,
            "      \"generated_sha256\": \"{}\",",
            field.generated_sha256
        )
        .unwrap();
        writeln!(
            &mut json,
            "      \"native_sha256\": \"{}\",",
            field.native_sha256
        )
        .unwrap();
        writeln!(&mut json, "      \"all\": {},", stats_json(&field.all)).unwrap();
        writeln!(
            &mut json,
            "      \"interior\": {},",
            stats_json(&field.interior)
        )
        .unwrap();
        writeln!(
            &mut json,
            "      \"boundary\": {}",
            stats_json(&field.boundary)
        )
        .unwrap();
        writeln!(&mut json, "    }}{comma}").unwrap();
    }
    writeln!(&mut json, "  ]").unwrap();
    writeln!(&mut json, "}}").unwrap();
    json
}

fn comparison_png(
    state: &ScientificState,
    report: &ComparisonReport,
    reference_dir: &Path,
) -> io::Result<Vec<u8>> {
    const INK: [u8; 3] = [21, 28, 43];
    const MUTED: [u8; 3] = [73, 85, 105];
    const GREEN: [u8; 3] = [23, 145, 105];
    const PURPLE: [u8; 3] = [126, 34, 206];
    const GOLD: [u8; 3] = [245, 158, 11];

    let bubble = state
        .field("tha")
        .ok_or_else(|| invalid_data("scientific state is missing tha"))?;
    let prefix = native_capture_prefix(reference_dir)?;
    let native_path = reference_dir.join(format!("{prefix}{}", bubble.native_file));
    let native = fs::read(&native_path)?;
    if native.len() != bubble.bounds.words() * 4 {
        return Err(invalid_data(format!(
            "native tha has {} bytes; expected {}",
            native.len(),
            bubble.bounds.words() * 4
        )));
    }

    let mut best_k = 1_i32;
    let mut best_mismatches = 0_u64;
    for k in 1..=NZ {
        let mut mismatches = 0_u64;
        for j in 1..=NY {
            for i in 1..=NX {
                let offset = bubble.bounds.offset(i, j, k);
                let expected = native_word(&native, offset);
                if bubble.values[offset].to_bits() != expected.to_bits() {
                    mismatches += 1;
                }
            }
        }
        if mismatches > best_mismatches {
            best_mismatches = mismatches;
            best_k = k;
        }
    }

    let mut canvas = Canvas::new(1200, 860, [248, 250, 252]);
    canvas.fill_rect(0, 0, canvas.width() as i32, 82, [15, 23, 42]);
    canvas.text(
        34,
        20,
        "CM1 R21.1 SCIENTIFIC INITIALIZER",
        3,
        [241, 245, 249],
    );
    canvas.text(
        36,
        58,
        "RUST FORMULAS VS NATIVE POST-GETSET, IDENTICAL INDICES",
        2,
        [148, 163, 184],
    );

    let z_metres = DZ_M * (best_k as f32 - 0.5);
    canvas.text(
        50,
        96,
        &format!("WARM BUBBLE THA, RUST, Z={z_metres:.0} M"),
        2,
        INK,
    );
    canvas.text(
        650,
        96,
        &format!("NATIVE DIFFERENCE, ULP, Z={z_metres:.0} M"),
        2,
        INK,
    );

    const MAP_X_LEFT: i32 = 50;
    const MAP_X_RIGHT: i32 = 650;
    const MAP_Y: i32 = 126;
    const CELL: i32 = 4;
    for j in 1..=NY {
        for i in 1..=NX {
            let offset = bubble.bounds.offset(i, j, best_k);
            let generated = bubble.values[offset];
            let expected = native_word(&native, offset);
            let y = MAP_Y + (NY - j) * CELL;
            let x_left = MAP_X_LEFT + (i - 1) * CELL;
            let x_right = MAP_X_RIGHT + (i - 1) * CELL;
            canvas.fill_rect(x_left, y, CELL, CELL, bubble_color(generated));
            let difference = if generated.to_bits() == expected.to_bits() {
                [226, 240, 233]
            } else {
                match ulp_distance(generated, expected) {
                    1 => GOLD,
                    2 => PURPLE,
                    _ => [220, 38, 38],
                }
            };
            canvas.fill_rect(x_right, y, CELL, CELL, difference);
        }
    }
    canvas.rect(MAP_X_LEFT - 1, MAP_Y - 1, 482, 482, INK);
    canvas.rect(MAP_X_RIGHT - 1, MAP_Y - 1, 482, 482, INK);
    canvas.text(50, 614, "-60 KM", 1, MUTED);
    canvas.text(484, 614, "+60 KM", 1, MUTED);
    canvas.text(650, 614, "EXACT", 1, GREEN);
    canvas.fill_rect(692, 612, 14, 8, [226, 240, 233]);
    canvas.text(724, 614, "1 ULP", 1, GOLD);
    canvas.fill_rect(760, 612, 14, 8, GOLD);
    canvas.text(794, 614, "2 ULP", 1, PURPLE);
    canvas.fill_rect(830, 612, 14, 8, PURPLE);

    let exact_fields = report.exact_fields();
    canvas.text(
        50,
        642,
        &format!(
            "SCOPE: {} FIELDS, {} FULL-ALLOCATION BIT-EXACT; THA SLICE MISMATCHES={best_mismatches}",
            report.fields.len(),
            exact_fields
        ),
        2,
        INK,
    );
    let thermo_names = ["pi0", "prs0", "rho0", "thv0", "th0", "rth0", "qv0"];
    let thermo_exact = thermo_names
        .iter()
        .filter(|name| {
            report
                .fields
                .iter()
                .find(|field| field.name == **name)
                .is_some_and(|field| field.all.words == field.all.exact_bits)
        })
        .count();
    canvas.text(
        50,
        670,
        &format!("BASE THERMODYNAMICS: {thermo_exact}/7 FULL-ALLOCATION BIT-EXACT"),
        2,
        GREEN,
    );
    let winds_interior_exact = ["u0", "v0", "ua", "u3d", "va", "v3d"].iter().all(|name| {
        report
            .fields
            .iter()
            .find(|field| field.name == *name)
            .is_some_and(|field| field.interior.words == field.interior.exact_bits)
    });
    canvas.text(
        50,
        698,
        if winds_interior_exact {
            "U/V LIVE INTERIORS: EXACT; WIND DELTAS ARE NON-LIVE CORNER/GHOST CADENCE"
        } else {
            "U/V LIVE INTERIORS: SEE COMPARISON.JSON"
        },
        2,
        if winds_interior_exact { GREEN } else { PURPLE },
    );

    canvas.text(50, 730, "MISMATCH WORDS BY FIELD, LOG SCALE", 1, MUTED);
    let max_mismatch = report
        .fields
        .iter()
        .map(|field| field.all.words - field.all.exact_bits)
        .max()
        .unwrap_or(0);
    let baseline = 824_i32;
    for (index, field) in report.fields.iter().enumerate() {
        let x = 42 + index as i32 * 60;
        let mismatches = field.all.words - field.all.exact_bits;
        let height = if mismatches == 0 || max_mismatch == 0 {
            3
        } else {
            10 + ((mismatches as f64).ln_1p() / (max_mismatch as f64).ln_1p() * 66.0) as i32
        };
        canvas.fill_rect(
            x,
            baseline - height,
            42,
            height,
            if mismatches == 0 { GREEN } else { PURPLE },
        );
        canvas.text(x, baseline + 5, field.name, 1, INK);
    }
    canvas.line(40, baseline, 1170, baseline, INK);
    Ok(canvas.encode())
}

fn native_word(bytes: &[u8], word: usize) -> f32 {
    let start = word * 4;
    f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
}

fn bubble_color(value: f32) -> [u8; 3] {
    let fraction = value.clamp(0.0, 1.0);
    let red = (38.0 + 217.0 * fraction) as u8;
    let green = (74.0 + 90.0 * (1.0 - (2.0 * fraction - 1.0).abs())) as u8;
    let blue = (162.0 * (1.0 - fraction) + 35.0) as u8;
    [red, green, blue]
}

fn stats_json(stats: &RegionStats) -> String {
    format!(
        "{{\"words\":{},\"exact_bits\":{},\"max_ulp\":{},\"max_abs\":{},\"non_finite_mismatches\":{}}}",
        stats.words,
        stats.exact_bits,
        stats.max_ulp,
        json_float(stats.max_abs),
        stats.non_finite_mismatches
    )
}

fn json_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.17e}")
    } else {
        "null".to_string()
    }
}

/// The stream names' common prefix in a native post-GETSET capture.
///
/// A capture names its streams `<prefix>post-getset-<field>.f32le` beside one
/// `<prefix>post-getset-layout.tsv`; the prefix is whatever the instrument
/// that wrote the capture put there (it may be empty), and the layout's `path`
/// column must name each field's stream with that same prefix.
fn native_capture_prefix(reference_dir: &Path) -> io::Result<String> {
    const LAYOUT: &str = "post-getset-layout.tsv";
    let mut prefixes = Vec::new();
    for entry in fs::read_dir(reference_dir).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "cannot list native capture {}: {error}",
                reference_dir.display()
            ),
        )
    })? {
        let name = entry?.file_name();
        if let Some(prefix) = name.to_str().and_then(|name| name.strip_suffix(LAYOUT)) {
            prefixes.push(prefix.to_string());
        }
    }
    match prefixes.len() {
        1 => Ok(prefixes.remove(0)),
        0 => Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "native capture {} holds no *{LAYOUT}",
                reference_dir.display()
            ),
        )),
        _ => Err(invalid_data(format!(
            "native capture {} holds {} *{LAYOUT} files; expected one",
            reference_dir.display(),
            prefixes.len()
        ))),
    }
}

fn validate_native_layout(
    reference_dir: &Path,
    prefix: &str,
    fields: &[ScientificField],
) -> io::Result<()> {
    let layout_path = reference_dir.join(format!("{prefix}post-getset-layout.tsv"));
    let text = fs::read_to_string(&layout_path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "cannot read native layout {}: {error}",
                layout_path.display()
            ),
        )
    })?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| invalid_data("native layout is empty"))?;
    let expected_header = "field\tstagger\tilo\tihi\tjlo\tjhi\tklo\tkhi\twords\tphase\tcall_ordinal\tdtype\tbyte_order\tpath";
    if header != expected_header {
        return Err(invalid_data(format!(
            "native layout header mismatch: {header:?}"
        )));
    }
    let mut rows = BTreeMap::new();
    for (line_index, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let columns = line.split('\t').collect::<Vec<_>>();
        if columns.len() != 14 {
            return Err(invalid_data(format!(
                "native layout line {} has {} columns; expected 14",
                line_index + 2,
                columns.len()
            )));
        }
        if rows.insert(columns[0].to_string(), columns).is_some() {
            return Err(invalid_data(format!(
                "native layout contains duplicate field {}",
                line.split('\t').next().unwrap_or("")
            )));
        }
    }
    for field in fields {
        let row = rows.get(field.name).ok_or_else(|| {
            invalid_data(format!("native layout is missing field {}", field.name))
        })?;
        let expected = [
            field.name.to_string(),
            field.stagger.as_str().to_string(),
            field.bounds.i_lo.to_string(),
            field.bounds.i_hi.to_string(),
            field.bounds.j_lo.to_string(),
            field.bounds.j_hi.to_string(),
            field.bounds.k_lo.to_string(),
            field.bounds.k_hi.to_string(),
            field.bounds.words().to_string(),
            "post_getset".to_string(),
            "1".to_string(),
            "f32".to_string(),
            "little".to_string(),
            format!("{prefix}{}", field.native_file),
        ];
        for (column, (actual, expected)) in row.iter().zip(expected.iter()).enumerate() {
            if actual != expected {
                return Err(invalid_data(format!(
                    "native layout field {} column {} is {:?}; expected {:?}",
                    field.name,
                    column + 1,
                    actual,
                    expected
                )));
            }
        }
    }
    Ok(())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn coordinate<'a>(state: &'a ScientificState, name: &str) -> &'a Coordinate1D {
        state
            .coordinates()
            .iter()
            .find(|item| item.name() == name)
            .unwrap()
    }

    #[test]
    fn sealed_geometry_and_source_reductions_are_exact() {
        let state = ScientificState::initialize_sealed_v1();
        assert_eq!(state.coordinates().len(), 6);
        assert_eq!(state.fields().len(), 19);
        assert_eq!(coordinate(&state, "xf").values()[3], -60_000.0);
        assert_eq!(coordinate(&state, "xh").values()[3], -59_500.0);
        assert_eq!(coordinate(&state, "xh").values()[63], 500.0);
        assert_eq!(coordinate(&state, "zf").values()[1], 0.0);
        assert_eq!(coordinate(&state, "zh").values()[1], 250.0);
        assert_eq!(coordinate(&state, "zf").values()[41], 20_000.0);
        assert_eq!(state.field("pi0").unwrap().bounds(), SCALAR_BOUNDS);
        assert_eq!(state.field("ua").unwrap().bounds(), U_BOUNDS);
        assert_eq!(state.field("va").unwrap().bounds(), V_BOUNDS);
        assert_eq!(state.field("wa").unwrap().bounds(), W_BOUNDS);
    }

    #[test]
    fn source_formulas_preserve_carriers_and_claim_boundaries() {
        let state = ScientificState::initialize_sealed_v1();
        assert_eq!(
            state.field("u0").unwrap().values(),
            state.field("ua").unwrap().values()
        );
        assert_eq!(
            state.field("ua").unwrap().values(),
            state.field("u3d").unwrap().values()
        );
        assert_eq!(
            state.field("v0").unwrap().values(),
            state.field("va").unwrap().values()
        );
        assert_eq!(
            state.field("va").unwrap().values(),
            state.field("v3d").unwrap().values()
        );
        assert_eq!(
            state.field("tha").unwrap().values(),
            state.field("th3d").unwrap().values()
        );

        // The physical-k fill bands intentionally leave global outer corner
        // rectangles at allocation zero. GETSET refills the intersection
        // bands, but not their already-extrapolated vertical ghost words.
        let u0 = state.field("u0").unwrap();
        assert_eq!(u0.value(-2, -2, 1).unwrap().to_bits(), 0);
        assert_eq!(u0.value(-2, 0, 1), u0.value(1, 1, 1));
        assert_eq!(u0.value(-2, 0, 0).unwrap().to_bits(), 0);
        assert_eq!(u0.value(-2, 1, 0), u0.value(1, 1, 0));
        let v0 = state.field("v0").unwrap();
        assert_eq!(v0.value(-2, -2, 1).unwrap().to_bits(), 0);
        assert_eq!(v0.value(0, -2, 1), v0.value(1, 1, 1));
        assert_eq!(v0.value(0, -2, 0).unwrap().to_bits(), 0);
        assert_eq!(v0.value(1, -2, 0), v0.value(1, 1, 0));

        assert!(state
            .field("wa")
            .unwrap()
            .values()
            .iter()
            .all(|value| value.to_bits() == 0));
        assert!(state
            .field("ppi")
            .unwrap()
            .values()
            .iter()
            .all(|value| value.to_bits() == 0));

        let bubble = state.field("tha").unwrap();
        assert!(bubble.value(60, 60, 3).unwrap() > 0.0);
        assert_eq!(bubble.value(1, 1, 1).unwrap().to_bits(), 0);
        assert_eq!(bubble.value(-2, -2, 0).unwrap().to_bits(), 0);
        let qv0 = state.field("qv0").unwrap();
        assert!(qv0
            .values()
            .iter()
            .all(|value| value.is_finite() && *value <= 0.014));

        assert!(CLAIM_LIMIT.contains("no timestep"));
        assert!(SOURCE_ANCHORS
            .iter()
            .any(|anchor| anchor.contains("base.F:359-450")));
        assert!(SOURCE_ANCHORS
            .iter()
            .any(|anchor| anchor.contains("cm1.F:2390-2403")));
    }

    #[test]
    fn comparison_splits_identical_native_indices() {
        let bounds = Bounds3::new(0, 2, 0, 2, 0, 2);
        let mut values = vec![0.0_f32; bounds.words()];
        values[bounds.offset(1, 1, 1)] = 2.0;
        let field = ScientificField {
            name: "tiny",
            family: FieldFamily::Core,
            stagger: Stagger::Scalar,
            bounds,
            values,
            native_file: "tiny.f32le",
        };
        let mut native = encode_f32le(field.values());
        let control = compare_field(&field, &native);
        assert_eq!(control.all.exact_bits, control.all.words);
        assert_eq!(control.interior.exact_bits, control.interior.words);
        assert_eq!(control.boundary.exact_bits, control.boundary.words);

        // Negative control: one physical-interior word and one boundary word
        // are independently perturbed without changing the array envelope.
        native[bounds.offset(1, 1, 1) * 4..bounds.offset(1, 1, 1) * 4 + 4]
            .copy_from_slice(&2.5_f32.to_le_bytes());
        native[bounds.offset(0, 0, 0) * 4..bounds.offset(0, 0, 0) * 4 + 4]
            .copy_from_slice(&1.0_f32.to_le_bytes());
        let compared = compare_field(&field, &native);
        assert_eq!(compared.all.words, 27);
        assert_eq!(compared.all.exact_bits, 25);
        assert_eq!(compared.interior.words, 8);
        assert_eq!(compared.interior.exact_bits, 7);
        assert_eq!(compared.boundary.words, 19);
        assert_eq!(compared.boundary.exact_bits, 18);
        assert_eq!(compared.all.max_abs, 1.0);
    }

    #[test]
    fn tracked_native_receipt_is_bounded_lf_json() {
        let bytes = include_bytes!("../evidence/sealed-product-o0-comparison.json");
        assert!(!bytes.contains(&b'\r'));
        let receipt = crate::json::parse(bytes).unwrap();
        assert_eq!(
            receipt.get("schema").unwrap().as_str().unwrap(),
            "gpuwm-cm1.rust-scientific-comparison-receipt/v1"
        );
        let result = receipt.get("result").unwrap();
        assert_eq!(result.get("fields_compared").unwrap().as_u64().unwrap(), 19);
        assert_eq!(
            result.get("fields_bit_exact").unwrap().as_u64().unwrap(),
            17
        );
        assert_eq!(result.get("max_ulp").unwrap().as_u64().unwrap(), 2);
        assert!(receipt
            .get("command")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("compare-scientific"));
        assert!(receipt
            .get("claim_limit")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("no timestep"));
        assert_eq!(
            receipt
                .get("ignored_outputs")
                .unwrap()
                .get("png_path")
                .unwrap()
                .as_str()
                .unwrap(),
            "comparison.png beside the comparison output (not tracked)"
        );
        let readme = include_str!("../README.md");
        assert!(readme.contains(
            "cargo run -- render-scientific-comparison NATIVE_POST_GETSET_DIR NEW_OUTPUT.png"
        ));
    }

    #[test]
    fn writer_is_lf_only_and_refuses_existing_directory() {
        let state = ScientificState::initialize_sealed_v1();
        let path = temp_path("scientific-write");
        let receipt = state.write_new(&path).unwrap();
        let manifest = fs::read(&receipt.manifest_path).unwrap();
        assert!(!manifest.contains(&b'\r'));
        assert!(String::from_utf8(manifest)
            .unwrap()
            .contains(SCIENTIFIC_SCHEMA));
        assert_eq!(receipt.arrays, 25);
        assert!(receipt.total_array_bytes > 40_000_000);
        assert!(state.write_new(&path).is_err());
        fs::remove_dir_all(&path).unwrap();
    }

    #[test]
    fn a_capture_names_its_streams_with_its_layouts_own_prefix() {
        let dir = temp_path("capture-prefix");
        fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            native_capture_prefix(&dir).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        fs::write(dir.join("post-getset-layout.tsv"), b"").unwrap();
        assert_eq!(native_capture_prefix(&dir).unwrap(), "");
        fs::remove_file(dir.join("post-getset-layout.tsv")).unwrap();
        let state = ScientificState::initialize_sealed_v1();
        let mut layout = String::from(
            "field\tstagger\tilo\tihi\tjlo\tjhi\tklo\tkhi\twords\tphase\tcall_ordinal\tdtype\tbyte_order\tpath\n",
        );
        for field in state.fields() {
            let b = field.bounds();
            layout.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tpost_getset\t1\tf32\tlittle\to0-{}\n",
                field.name(),
                field.stagger().as_str(),
                b.i_lo,
                b.i_hi,
                b.j_lo,
                b.j_hi,
                b.k_lo,
                b.k_hi,
                b.words(),
                field.native_file()
            ));
        }
        fs::write(dir.join("o0-post-getset-layout.tsv"), &layout).unwrap();
        assert_eq!(native_capture_prefix(&dir).unwrap(), "o0-");
        validate_native_layout(&dir, "o0-", state.fields()).unwrap();
        // A layout whose path column leaves out the capture's own prefix is refused.
        let unprefixed = layout.replace("\to0-post-getset-", "\tpost-getset-");
        fs::write(dir.join("o0-post-getset-layout.tsv"), unprefixed).unwrap();
        let error = validate_native_layout(&dir, "o0-", state.fields()).unwrap_err();
        assert!(error.to_string().contains("column 14"), "{error}");
        fs::write(dir.join("o2-post-getset-layout.tsv"), b"").unwrap();
        assert_eq!(
            native_capture_prefix(&dir).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert!(state
            .fields()
            .iter()
            .all(|field| field.native_file().starts_with("post-getset-")));
        fs::remove_dir_all(&dir).unwrap();
    }

    fn temp_path(label: &str) -> PathBuf {
        let nonce = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("cm1-state-{label}-{}-{nonce}", std::process::id()))
    }
}
