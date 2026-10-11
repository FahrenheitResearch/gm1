//! Source-ordered CM1 r21.1 moist pressure and density completion.
//!
//! The transform is deliberately physical-domain only.  It consumes terrain
//! heights and face coefficients already owned by [`crate::terrain`], plus the
//! checkpoint base profiles, potential-temperature perturbation, and packed
//! moisture.  It does not synthesize halos or boundary history.

use crate::{Result, StateError};

pub const MOIST_INITIALIZATION_CLAIM_LIMIT: &str = "source-ordered CM1 r21.1 ibalance=1 moist physical-domain pressure and density completion only; no halos, open-boundary history or tendencies, pressure operator, dynamics, physics, closure, trajectory, forecast, tornado, or model-validation claim";
pub const MOIST_SPECIES_ORDER: [&str; 6] = ["qv", "qc", "qr", "qi", "qs", "qg"];

const PRODUCT_G_BITS: u32 = 0x411c_f5c3;
const PRODUCT_CP_BITS: u32 = 0x447b_6ccd;
const PRODUCT_RD_BITS: u32 = 0x438f_851f;
const PRODUCT_RV_BITS: u32 = 0x43e6_c000;
const PRODUCT_P00_BITS: u32 = 0x47c3_5000;
const PPI_ZERO_THRESHOLD: f32 = 1.0e-6_f32;

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MoistInitializationAdmission {
    Generic,
    RealProduct,
}

/// Explicit source configuration.  There is intentionally no `Default`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoistInitializationConfig {
    pub g: f32,
    pub cp: f32,
    pub rd: f32,
    pub rv: f32,
    pub p00: f32,
    pub psolver: u8,
    pub ibalance: u8,
    pub setppi: bool,
    pub iice: u8,
    /// README.namelist:308's imoist.  `new` binds 1; a dry base state
    /// (imoist = 0) takes init3d.F's dry arms: `thvnew = th0+tha`
    /// (init3d.F:1766-1768) and `rho = prs/(rd*th*pi)` (init3d.F:2408-2410).
    pub imoist: u8,
}

impl MoistInitializationConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        g: f32,
        cp: f32,
        rd: f32,
        rv: f32,
        p00: f32,
        psolver: u8,
        ibalance: u8,
        setppi: bool,
        iice: u8,
    ) -> Result<Self> {
        let value = Self {
            g,
            cp,
            rd,
            rv,
            p00,
            psolver,
            ibalance,
            setppi,
            iice,
            imoist: 1,
        };
        value.validate()?;
        Ok(value)
    }

    /// Select README.namelist:308's imoist (0 or 1) for init3d.F's
    /// thermodynamic completion.
    pub fn with_imoist(mut self, imoist: u8) -> Result<Self> {
        if imoist > 1 {
            return Err(invalid(format!(
                "README.namelist:308 admits imoist 0 or 1; got {imoist}"
            )));
        }
        self.imoist = imoist;
        Ok(self)
    }

    fn validate(self) -> Result<()> {
        let actual = [
            self.g.to_bits(),
            self.cp.to_bits(),
            self.rd.to_bits(),
            self.rv.to_bits(),
            self.p00.to_bits(),
        ];
        let expected = [
            PRODUCT_G_BITS,
            PRODUCT_CP_BITS,
            PRODUCT_RD_BITS,
            PRODUCT_RV_BITS,
            PRODUCT_P00_BITS,
        ];
        if actual != expected {
            return Err(invalid(format!(
                "moist initialization requires exact CM1 default-real constants g/cp/rd/rv/p00={expected:08x?}; got {actual:08x?}, which would change pressure and density bits"
            )));
        }
        if self.psolver != 3 || !matches!(self.ibalance, 0 | 1) || !self.setppi || self.iice != 1 {
            return Err(invalid(format!(
                "moist initialization admits psolver=3, ibalance=0 or 1, setppi=true, iice=1 only; got psolver={}, ibalance={}, setppi={}, iice={}, which selects a different init3d branch",
                self.psolver, self.ibalance, self.setppi, self.iice
            )));
        }
        Ok(())
    }

    pub fn constants_bits(self) -> [u32; 5] {
        [
            self.g.to_bits(),
            self.cp.to_bits(),
            self.rd.to_bits(),
            self.rv.to_bits(),
            self.p00.to_bits(),
        ]
    }
}

#[derive(Clone, Debug)]
pub struct MoistInitializationFrame {
    shape_zyx: [usize; 3],
    qt: Vec<f32>,
    thv: Vec<f32>,
    pinew: Vec<f32>,
    ppi: Vec<f32>,
    prs: Vec<f32>,
    rho: Vec<f32>,
    rr: Vec<f32>,
    rf: Vec<f32>,
}

impl MoistInitializationFrame {
    pub fn shape_zyx(&self) -> [usize; 3] {
        self.shape_zyx
    }

    pub fn qt(&self) -> &[f32] {
        &self.qt
    }

    pub fn thv(&self) -> &[f32] {
        &self.thv
    }

    pub fn pinew(&self) -> &[f32] {
        &self.pinew
    }

    pub fn ppi(&self) -> &[f32] {
        &self.ppi
    }

    pub fn prs(&self) -> &[f32] {
        &self.prs
    }

    pub fn rho(&self) -> &[f32] {
        &self.rho
    }

    pub fn rr(&self) -> &[f32] {
        &self.rr
    }

    pub fn rf(&self) -> &[f32] {
        &self.rf
    }
}

/// Apply `init3d.F:1733-1785,2365-2467` in source loop and assignment order.
///
/// Arrays are little-endian binary32 values in C-order.  Mass carriers use
/// `[k,j,i]`; packed moisture uses `[species,k,j,i]`; face carriers use
/// `[face_k,j,i]`.  `bottom_cgs` and `top_cgt` must be the exact coefficients
/// produced by `param.F:8988-9052` for the matching vertical grid.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn complete_moist_initialization(
    pi0_profile: &[f32],
    th0_profile: &[f32],
    theta_perturbation: &[f32],
    packed_moisture: &[f32],
    zh: &[f32],
    c1: &[f32],
    c2: &[f32],
    bottom_cgs: [f32; 3],
    top_cgt: [f32; 3],
    shape_zyx: [usize; 3],
    config: MoistInitializationConfig,
    admission: MoistInitializationAdmission,
) -> Result<MoistInitializationFrame> {
    complete_moist_initialization_with_ppi(
        pi0_profile,
        th0_profile,
        theta_perturbation,
        packed_moisture,
        zh,
        c1,
        c2,
        bottom_cgs,
        top_cgt,
        shape_zyx,
        config,
        admission,
        None,
    )
}

/// [`complete_moist_initialization`] with an arm's own initial pressure
/// perturbation: init3d.F's `setppi = .false.` (iinit=7), where the
/// ibalance block is skipped and `ppi` is the arm's.
#[allow(clippy::too_many_arguments)]
pub fn complete_moist_initialization_with_ppi(
    pi0_profile: &[f32],
    th0_profile: &[f32],
    theta_perturbation: &[f32],
    packed_moisture: &[f32],
    zh: &[f32],
    c1: &[f32],
    c2: &[f32],
    bottom_cgs: [f32; 3],
    top_cgt: [f32; 3],
    shape_zyx: [usize; 3],
    config: MoistInitializationConfig,
    admission: MoistInitializationAdmission,
    prescribed_ppi: Option<&[f32]>,
) -> Result<MoistInitializationFrame> {
    config.validate()?;
    let [nz, ny, nx] = shape_zyx;
    if nz < 4 || ny == 0 || nx == 0 {
        return Err(invalid(
            "moist initialization requires at least four vertical levels and positive horizontal extents",
        ));
    }
    if admission == MoistInitializationAdmission::RealProduct && nz != 75 {
        return Err(invalid(format!(
            "real-product moist initialization requires all 75 source vertical levels; got {shape_zyx:?}"
        )));
    }
    let mass_words = product(&[nz, ny, nx], "mass")?;
    let face_words = product(&[nz + 1, ny, nx], "z-face")?;
    let moisture_words = product(&[MOIST_SPECIES_ORDER.len(), nz, ny, nx], "moisture")?;
    require_len("pi0", pi0_profile, nz)?;
    require_len("th0", th0_profile, nz)?;
    require_len("theta perturbation", theta_perturbation, mass_words)?;
    if let Some(given) = prescribed_ppi {
        require_len("prescribed ppi", given, mass_words)?;
        validate_finite("prescribed ppi", given)?;
    }
    require_len("packed moisture", packed_moisture, moisture_words)?;
    require_len("zh", zh, mass_words)?;
    require_len("c1", c1, face_words)?;
    require_len("c2", c2, face_words)?;
    validate_finite("pi0", pi0_profile)?;
    validate_finite("th0", th0_profile)?;
    validate_finite("theta perturbation", theta_perturbation)?;
    validate_finite("packed moisture", packed_moisture)?;
    validate_finite("zh", zh)?;
    validate_finite("c1", c1)?;
    validate_finite("c2", c2)?;
    validate_finite("bottom cgs", &bottom_cgs)?;
    validate_finite("top cgt", &top_cgt)?;
    if pi0_profile.iter().any(|value| *value <= 0.0)
        || th0_profile.iter().any(|value| *value <= 0.0)
    {
        return Err(invalid(
            "moist initialization pi0 and th0 profiles must be positive",
        ));
    }
    for index in 0..face_words {
        if !(0.0..=1.0).contains(&c1[index])
            || !(0.0..=1.0).contains(&c2[index])
            || (c1[index] + c2[index]).to_bits() != 1.0_f32.to_bits()
        {
            return Err(invalid(format!(
                "moist initialization face weights at word {index} are not exact complementary binary32 fractions"
            )));
        }
    }
    if admission == MoistInitializationAdmission::RealProduct {
        for (index, value) in packed_moisture.iter().copied().enumerate() {
            if !(0.0..1.0).contains(&value) {
                return Err(invalid(format!(
                    "real-product packed moisture word {index} leaves [0,1); bits={:08x}",
                    value.to_bits()
                )));
            }
        }
    }

    let reps = config.rv / config.rd;
    let cpdrd = config.cp / config.rd;
    let mut qt = vec![0.0_f32; mass_words];
    let mut thv = vec![0.0_f32; mass_words];
    let mut pinew = vec![0.0_f32; mass_words];
    let mut ppi = vec![0.0_f32; mass_words];
    let mut prs = vec![0.0_f32; mass_words];
    let mut rho = vec![0.0_f32; mass_words];
    let mut rr = vec![0.0_f32; mass_words];
    let mut rf = vec![0.0_f32; face_words];

    // init3d.F loops horizontal columns around its vertical calculations.
    for j in 0..ny {
        for i in 0..nx {
            for k in 0..nz {
                let index = index3(k, j, i, ny, nx);
                let mut total_condensate = 0.0_f32;
                for species in 1..MOIST_SPECIES_ORDER.len() {
                    total_condensate += packed_moisture[index4(species, k, j, i, nz, ny, nx)];
                }
                qt[index] = total_condensate;
                let qv = packed_moisture[index4(0, k, j, i, nz, ny, nx)];
                // init3d.F:1751-1769: the moist thvnew, or th0+tha when dry.
                thv[index] = if config.imoist == 0 {
                    th0_profile[k] + theta_perturbation[index]
                } else {
                    (th0_profile[k] + theta_perturbation[index]) * (1.0_f32 + reps * qv)
                        / (1.0_f32 + qv + total_condensate)
                };
                if !thv[index].is_finite() || thv[index] <= 0.0 {
                    return Err(invalid(format!(
                        "moist initialization virtual potential temperature is invalid at k={k}, j={j}, i={i}"
                    )));
                }
            }
            // init3d.F:1735-1741 then :1743-1786.  ibalance=0 leaves the
            // initial pressure perturbation at exactly zero -- which is what
            // every one of CM1's 23 shipped decks selects (run/namelist.input:80)
            // -- and ibalance=1 integrates hydrostatic balance top-down.
            if let Some(given) = prescribed_ppi {
                for k in 0..nz {
                    let index = index3(k, j, i, ny, nx);
                    ppi[index] = given[index];
                    pinew[index] = pi0_profile[k] + given[index];
                }
            } else if config.ibalance == 1 {
                let top_index = index3(nz - 1, j, i, ny, nx);
                pinew[top_index] = pi0_profile[nz - 1];
                for k in (0..nz - 1).rev() {
                    let index = index3(k, j, i, ny, nx);
                    let above = index3(k + 1, j, i, ny, nx);
                    pinew[index] = pinew[above]
                        + config.g * (zh[above] - zh[index])
                            / (config.cp * 0.5_f32 * (thv[above] + thv[index]));
                }
                for k in 0..nz {
                    let index = index3(k, j, i, ny, nx);
                    let perturbation = pinew[index] - pi0_profile[k];
                    ppi[index] = if perturbation.abs() < PPI_ZERO_THRESHOLD {
                        0.0_f32
                    } else {
                        perturbation
                    };
                }
            } else {
                for k in 0..nz {
                    let index = index3(k, j, i, ny, nx);
                    pinew[index] = pi0_profile[k];
                    ppi[index] = 0.0_f32;
                }
            }
        }
    }

    // init3d.F diagnostic completion uses k/j/i loop order.  Each cell is
    // independent, but retaining that order also makes raw record emission
    // and validation straightforward.
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let index = index3(k, j, i, ny, nx);
                let pi = pi0_profile[k] + ppi[index];
                if !pi.is_finite() || pi <= 0.0 {
                    return Err(invalid(format!(
                        "moist initialization total Exner pressure is non-positive at k={k}, j={j}, i={i}"
                    )));
                }
                prs[index] = config.p00 * pi.powf(cpdrd);
                let qv = packed_moisture[index4(0, k, j, i, nz, ny, nx)];
                // init3d.F:2393-2396 (imoist = 1) or :2408-2410 (imoist = 0).
                rho[index] = if config.imoist == 0 {
                    prs[index]
                        / (config.rd * (th0_profile[k] + theta_perturbation[index]) * pi)
                } else {
                    prs[index]
                        / ((th0_profile[k] + theta_perturbation[index])
                            * pi
                            * (config.rd + qv.max(0.0_f32) * config.rv))
                };
                if !prs[index].is_finite()
                    || prs[index] <= 0.0
                    || !rho[index].is_finite()
                    || rho[index] <= 0.0
                {
                    return Err(invalid(format!(
                        "moist initialization pressure/density is invalid at k={k}, j={j}, i={i}"
                    )));
                }
                rr[index] = 1.0_f32 / rho[index];
            }
        }
    }

    for j in 0..ny {
        for i in 0..nx {
            for face in 1..nz {
                let face_index = index3(face, j, i, ny, nx);
                rf[face_index] = c1[face_index] * rho[index3(face - 1, j, i, ny, nx)]
                    + c2[face_index] * rho[index3(face, j, i, ny, nx)];
            }
            let bottom = index3(0, j, i, ny, nx);
            rf[bottom] = bottom_cgs[0] * rho[index3(0, j, i, ny, nx)]
                + bottom_cgs[1] * rho[index3(1, j, i, ny, nx)]
                + bottom_cgs[2] * rho[index3(2, j, i, ny, nx)];
            let top = index3(nz, j, i, ny, nx);
            rf[top] = top_cgt[0] * rho[index3(nz - 1, j, i, ny, nx)]
                + top_cgt[1] * rho[index3(nz - 2, j, i, ny, nx)]
                + top_cgt[2] * rho[index3(nz - 3, j, i, ny, nx)];
        }
    }
    if rf.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(invalid(
            "moist initialization extrapolated face density is not positive finite",
        ));
    }

    Ok(MoistInitializationFrame {
        shape_zyx,
        qt,
        thv,
        pinew,
        ppi,
        prs,
        rho,
        rr,
        rf,
    })
}

fn require_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        return Err(invalid(format!(
            "moist initialization {name} contains {} words; expected {expected}",
            values.len()
        )));
    }
    Ok(())
}

fn validate_finite(name: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid(format!(
            "moist initialization {name} has non-finite word {index}; bits={:08x}",
            value.to_bits()
        )));
    }
    Ok(())
}

fn product(shape: &[usize], name: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("moist initialization {name} shape overflows")))
    })
}

fn index3(k: usize, j: usize, i: usize, ny: usize, nx: usize) -> usize {
    (k * ny + j) * nx + i
}

fn index4(species: usize, k: usize, j: usize, i: usize, nz: usize, ny: usize, nx: usize) -> usize {
    ((species * nz + k) * ny + j) * nx + i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;

    type Key = (String, usize, usize, usize, usize, usize);

    fn records(path: &Path) -> BTreeMap<Key, u32> {
        let text = fs::read_to_string(path).unwrap();
        let mut values = BTreeMap::new();
        for line in text.lines().skip(1) {
            let columns = line.split('\t').collect::<Vec<_>>();
            assert_eq!(columns.len(), 12);
            values.insert(
                (
                    columns[3].to_string(),
                    columns[4].parse().unwrap(),
                    columns[5].parse().unwrap(),
                    columns[6].parse().unwrap(),
                    columns[7].parse().unwrap(),
                    columns[8].parse().unwrap(),
                ),
                u32::from_str_radix(columns[11], 16).unwrap(),
            );
        }
        values
    }

    fn bits(
        values: &BTreeMap<Key, u32>,
        field: &str,
        species: usize,
        i: usize,
        j: usize,
        k: usize,
        face: usize,
    ) -> u32 {
        values[&(field.to_string(), species, i, j, k, face)]
    }

    fn config() -> MoistInitializationConfig {
        MoistInitializationConfig::new(
            f32::from_bits(PRODUCT_G_BITS),
            f32::from_bits(PRODUCT_CP_BITS),
            f32::from_bits(PRODUCT_RD_BITS),
            f32::from_bits(PRODUCT_RV_BITS),
            f32::from_bits(PRODUCT_P00_BITS),
            3,
            1,
            true,
            1,
        )
        .unwrap()
    }

    #[test]
    fn exact_native_moist_psolver3_fixture_matches_raw_bits() {
        // The development repository's oracle folder for this fixture (its name may carry a prefix).
        let oracles = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../evidence/oracles");
        let root = fs::read_dir(&oracles)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with("moist-psolver3-init-v1"))
            })
            .unwrap();
        let fixture_path = root.join("fixture-inputs.tsv");
        let oracle_path = root.join("moist-initialization-records.tsv");
        assert_eq!(
            sha256::digest_hex(&fs::read(&fixture_path).unwrap()),
            "c6314b6a526d8944b012d0590f45600e4b32b03a7cd66e4203b4bc62542143eb"
        );
        assert_eq!(
            sha256::digest_hex(&fs::read(&oracle_path).unwrap()),
            "1371239173e7f50a57326e8066fb8b9d09b2756573a57b308dc5e3f8782510e8"
        );
        let input = records(&fixture_path);
        let expected = records(&oracle_path);
        let (nz, ny, nx) = (5usize, 2usize, 3usize);
        let pi0 = (1..=nz)
            .map(|k| f32::from_bits(bits(&input, "pi0", 0, 0, 0, k, 0)))
            .collect::<Vec<_>>();
        let th0 = (1..=nz)
            .map(|k| f32::from_bits(bits(&input, "th0", 0, 0, 0, k, 0)))
            .collect::<Vec<_>>();
        let mut theta = Vec::with_capacity(nz * ny * nx);
        for k in 1..=nz {
            for j in 1..=ny {
                for i in 1..=nx {
                    theta.push(f32::from_bits(bits(&input, "tha", 0, i, j, k, 0)));
                }
            }
        }
        let mut moisture = Vec::with_capacity(6 * nz * ny * nx);
        for species in 1..=6 {
            for k in 1..=nz {
                for j in 1..=ny {
                    for i in 1..=nx {
                        moisture.push(f32::from_bits(bits(&input, "qa", species, i, j, k, 0)));
                    }
                }
            }
        }
        let mut zh = Vec::with_capacity(nz * ny * nx);
        for k in 1..=nz {
            for j in 1..=ny {
                for i in 1..=nx {
                    zh.push(f32::from_bits(bits(&expected, "zh", 0, i, j, k, 0)));
                }
            }
        }
        let mut c1 = Vec::with_capacity((nz + 1) * ny * nx);
        let mut c2 = Vec::with_capacity((nz + 1) * ny * nx);
        for face in 1..=nz + 1 {
            for j in 1..=ny {
                for i in 1..=nx {
                    c1.push(f32::from_bits(bits(&expected, "c1", 0, i, j, 0, face)));
                    c2.push(f32::from_bits(bits(&expected, "c2", 0, i, j, 0, face)));
                }
            }
        }
        let bottom = [
            f32::from_bits(bits(&expected, "cgs1", 0, 0, 0, 0, 0)),
            f32::from_bits(bits(&expected, "cgs2", 0, 0, 0, 0, 0)),
            f32::from_bits(bits(&expected, "cgs3", 0, 0, 0, 0, 0)),
        ];
        let top = [
            f32::from_bits(bits(&expected, "cgt1", 0, 0, 0, 0, 0)),
            f32::from_bits(bits(&expected, "cgt2", 0, 0, 0, 0, 0)),
            f32::from_bits(bits(&expected, "cgt3", 0, 0, 0, 0, 0)),
        ];
        let frame = complete_moist_initialization(
            &pi0,
            &th0,
            &theta,
            &moisture,
            &zh,
            &c1,
            &c2,
            bottom,
            top,
            [nz, ny, nx],
            config(),
            MoistInitializationAdmission::Generic,
        )
        .unwrap();
        for (name, values) in [
            ("qt", frame.qt()),
            ("thvnew", frame.thv()),
            ("pinew", frame.pinew()),
            ("ppi", frame.ppi()),
            ("prs", frame.prs()),
            ("rho", frame.rho()),
            ("rr", frame.rr()),
        ] {
            for k in 1..=nz {
                for j in 1..=ny {
                    for i in 1..=nx {
                        let index = index3(k - 1, j - 1, i - 1, ny, nx);
                        assert_eq!(
                            values[index].to_bits(),
                            bits(&expected, name, 0, i, j, k, 0),
                            "{name} i={i} j={j} k={k}"
                        );
                    }
                }
            }
        }
        for face in 1..=nz + 1 {
            for j in 1..=ny {
                for i in 1..=nx {
                    let index = index3(face - 1, j - 1, i - 1, ny, nx);
                    assert_eq!(
                        frame.rf()[index].to_bits(),
                        bits(&expected, "rf", 0, i, j, 0, face),
                        "rf i={i} j={j} face={face}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_dry_base_state_takes_init3d_dry_arms() {
        // init3d.F:1766-1768 (thvnew = th0+tha) and :2408-2410
        // (rho = prs/(rd*th*pi)), against the moist arms on the same words.
        let shape = [4, 1, 1];
        let pi0 = [0.99_f32, 0.97, 0.95, 0.93];
        let th0 = [300.0_f32, 301.0, 302.5, 304.0];
        let tha = [0.5_f32, 0.25, 0.0, -0.125];
        let moisture = vec![0.0_f32; 24];
        let zh = [250.0_f32, 750.0, 1250.0, 1750.0];
        let c = vec![0.5_f32; 5];
        let cgs = [1.875, -1.25, 0.375];
        let run = |imoist: u8| {
            complete_moist_initialization(
                &pi0, &th0, &tha, &moisture, &zh, &c, &c, cgs, cgs, shape,
                config().with_imoist(imoist).unwrap(), MoistInitializationAdmission::Generic,
            )
            .unwrap()
        };
        let (dry, moist) = (run(0), run(1));
        let cfg = config();
        for k in 0..4 {
            assert_eq!(dry.thv()[k].to_bits(), (th0[k] + tha[k]).to_bits());
            let pi = pi0[k] + dry.ppi()[k];
            let expected = dry.prs()[k] / (cfg.rd * (th0[k] + tha[k]) * pi);
            assert_eq!(dry.rho()[k].to_bits(), expected.to_bits(), "k={k}");
            let moist_pi = pi0[k] + moist.ppi()[k];
            let moist_expected =
                moist.prs()[k] / ((th0[k] + tha[k]) * moist_pi * (cfg.rd + 0.0_f32 * cfg.rv));
            assert_eq!(moist.rho()[k].to_bits(), moist_expected.to_bits(), "k={k}");
        }
        assert!(config().with_imoist(2).is_err());
    }

    #[test]
    fn strict_config_shape_weights_and_product_moisture_refuse() {
        assert!(MoistInitializationConfig::new(
            f32::from_bits(PRODUCT_G_BITS + 1),
            f32::from_bits(PRODUCT_CP_BITS),
            f32::from_bits(PRODUCT_RD_BITS),
            f32::from_bits(PRODUCT_RV_BITS),
            f32::from_bits(PRODUCT_P00_BITS),
            3,
            1,
            true,
            1,
        )
        .unwrap_err()
        .to_string()
        .contains("exact CM1"));
        assert!(MoistInitializationConfig::new(
            f32::from_bits(PRODUCT_G_BITS),
            f32::from_bits(PRODUCT_CP_BITS),
            f32::from_bits(PRODUCT_RD_BITS),
            f32::from_bits(PRODUCT_RV_BITS),
            f32::from_bits(PRODUCT_P00_BITS),
            2,
            1,
            true,
            1,
        )
        .unwrap_err()
        .to_string()
        .contains("different init3d branch"));

        let shape = [4, 1, 1];
        let mut c1 = vec![0.5_f32; 5];
        let c2 = vec![0.5_f32; 5];
        c1[2] = 0.25;
        let error = complete_moist_initialization(
            &[1.0; 4],
            &[300.0; 4],
            &[0.0; 4],
            &[0.0; 24],
            &[100.0, 300.0, 500.0, 700.0],
            &c1,
            &c2,
            [1.875, -1.25, 0.375],
            [1.875, -1.25, 0.375],
            shape,
            config(),
            MoistInitializationAdmission::Generic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("complementary"), "{error}");
    }
}
