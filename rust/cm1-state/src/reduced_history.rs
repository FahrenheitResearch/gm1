//! Reader-side reconstruction of the carriers a reduced GM1 history omits.
//!
//! A GM1 Cartesian history frame published under the
//! [`REDUCED_HISTORY_PROFILE_ID`] profile drops `prs` and `zf` and carries
//! instead the two compact columns their CM1 reconstruction reads: the
//! base-state Exner column `base.pi0` and the w-face height column
//! `metrics.zf_profile`.  This module is the reader half of that trade.  It
//! is the only place in the distribution that applies the recorded formulas,
//! so every consumer of a reduced frame -- the renderer today, anything else
//! tomorrow -- rebuilds the same words from the same expression.
//!
//! * `prs` is CM1 r21.1 `solve2.F:1588`:
//!   `p00` times the total Exner function `pi0 + pp3d` raised to `cpdrd`, with `p00` from
//!   `constants.F:17`, `rd`/`cp` from `constants.F:113-114` and
//!   `cpdrd = cp/rd` from `constants.F:137`.  The frame carries `pp3d` at
//!   full extent and `pi0` as the single column the producing runtime
//!   broadcasts it from, so every input of the expression is published.
//! * `zf` is a broadcast of one vertical profile, proved plane by plane by
//!   the producing runtime before upload, so `zf[k][j][i] = zf_profile[k]`
//!   is an exact copy and not an interpolation.
//!
//! The `prs` expression is evaluated in binary32 with this platform's `powf`.
//! On a glibc 2.39 host the rebuilt words are the words a complete frame would
//! have carried, bit for bit, and that rests on two premises rather than one.
//! Both are measured, not asserted:
//!
//! 1. `f32::powf` here *is* the host libm's `powf` -- Rust lowers it to the
//!    platform call rather than computing in `f64` and rounding once.
//!    [`the_rebuilt_pressure_is_the_host_libm_powf_word_for_word`] holds the
//!    two to raw-word equality over the Exner range a GM1 column spans.
//! 2. The producing CUDA kernel's `gm1_glibc239_pow_cpdrd_positive_normal`
//!    (`cuda_gm1_tilestream_rk3_eqtset2_remove_eos.py:128-132`, over the
//!    `gm1.cuda-glibc239-pow13/v1` device source) is glibc 2.39
//!    `sysdeps/ieee754/flt-32/e_powf.c` statement for statement.  Compiled
//!    with the port's own NVRTC options and run on sm_86, that kernel's
//!    `__fmul_rn(p00, pow_cpdrd(pi))` returned all 510,000 of 510,000 sampled
//!    words bit-identical to the same expression through glibc 2.39 `powf`.
//!
//! Neither premise is a claim about other hosts.  On a host whose `powf` is a
//! different implementation the rebuilt pressures are that implementation's,
//! within its own accuracy, and premise 1's test is what says so out loud; the
//! copy of `zf` is exact everywhere.
//!
//! `rf` is deliberately not reconstructible and stays published: CM1 builds
//! it from `rho` with the `c1`/`c2` interpolation weights (`solve3.F:666`)
//! and the second-order edge extrapolations at `k=1` and `k=nk+1`
//! (`solve3.F:678-679`), none of which a history frame carries.

use crate::forecast::{ForecastArray, ForecastArraySpec, ForecastContract, ForecastFrame};
use crate::{Result, StateError};
use std::io::{Error, ErrorKind};

/// `profile_id` of a complete GM1 Cartesian history frame.
pub const COMPLETE_HISTORY_PROFILE_ID: &str = "gm1.cartesian-history/v1";
/// `profile_id` of the opt-in frame this module reads back.
pub const REDUCED_HISTORY_PROFILE_ID: &str = "gm1.cartesian-history-reconstructible-omitted/v1";

/// The carrier a reduced frame publishes `pi0` as.
pub const PI0_COLUMN_CARRIER: &str = "base.pi0";
/// The carrier a reduced frame publishes the `zf` profile as.
pub const ZF_PROFILE_COLUMN_CARRIER: &str = "metrics.zf_profile";
/// The full-extent carrier the `prs` expression reads.
pub const PP3D_CARRIER: &str = "pp3d";
/// The two carriers this module rebuilds, in published (sorted) order.
pub const RECONSTRUCTED_CARRIERS: [&str; 2] = ["prs", "zf"];

/// CM1 r21.1 `constants.F:17`: `p00 = 1.0e5`.
pub const P00_PA: f32 = 1.0e5;
/// CM1 r21.1 `constants.F:113`: `rd = 287.04`.
pub const RD_J_PER_KG_K: f32 = 287.04;
/// CM1 r21.1 `constants.F:114`: `cp = 1005.7`.
pub const CP_J_PER_KG_K: f32 = 1005.7;
/// CM1 r21.1 `constants.F:137`: `cpdrd = cp/rd`, in binary32 as the producing
/// kernel holds it (`0x40603c81`).
pub const CPDRD: f32 = CP_J_PER_KG_K / RD_J_PER_KG_K;

const PRS_MEANING: &str = "current air pressure";
const PRS_UNITS: &str = "Pa";
const PRS_STAGGER: &str = "mass";
const PRS_FORMULA: &str =
    "CM1 r21.1 solve2.F:1588 prs = p00*((pi0+pp3d)**cpdrd) in binary32, with constants.F:17 p00, constants.F:113-114 rd/cp and constants.F:137 cpdrd = cp/rd";
const ZF_MEANING: &str = "model w-face height above mean sea level";
const ZF_UNITS: &str = "m";
const ZF_STAGGER: &str = "w";
const ZF_FORMULA: &str =
    "zf[k][j][i] = metrics.zf_profile[k], the single vertical profile the producing runtime proved equal on every horizontal plane before upload";

fn invalid(message: impl Into<String>) -> StateError {
    Error::new(ErrorKind::InvalidInput, message.into()).into()
}

/// One carrier rebuilt from a reduced frame's own remaining words.
#[derive(Clone, Debug)]
pub struct ReconstructedCarrier {
    spec: ForecastArraySpec,
    array: ForecastArray,
    formula: &'static str,
}

impl ReconstructedCarrier {
    pub fn name(&self) -> &str {
        self.spec.name()
    }

    pub fn shape(&self) -> &[usize] {
        self.spec.shape()
    }

    pub fn spec(&self) -> &ForecastArraySpec {
        &self.spec
    }

    pub fn values(&self) -> &[f32] {
        self.array.values()
    }

    pub fn array(&self) -> &ForecastArray {
        &self.array
    }

    /// The recorded CM1 expression these words were rebuilt by.
    pub fn formula(&self) -> &'static str {
        self.formula
    }
}

/// True when `contract` is a GM1 history frame that omits reconstructible
/// carriers, so a reader that wants `prs` or `zf` must rebuild them.
pub fn is_reduced_history(contract: &ForecastContract) -> bool {
    contract.profile_id() == REDUCED_HISTORY_PROFILE_ID
}

fn column<'a>(
    contract: &ForecastContract,
    frame: &'a ForecastFrame,
    name: &str,
    expected_len: Option<usize>,
) -> Result<&'a [f32]> {
    let spec = contract
        .arrays()
        .iter()
        .find(|spec| spec.name() == name)
        .ok_or_else(|| {
            invalid(format!(
                "reduced history frame declares no {name} carrier; without it {:?} cannot be rebuilt and every word of the reconstruction would be invented",
                RECONSTRUCTED_CARRIERS
            ))
        })?;
    if spec.shape().len() != 1 {
        return Err(invalid(format!(
            "reduced history carrier {name} has shape {:?}; the reconstruction reads one vertical column and a higher-rank carrier would be sampled at an arbitrary horizontal point",
            spec.shape()
        )));
    }
    if let Some(expected) = expected_len {
        if spec.shape()[0] != expected {
            return Err(invalid(format!(
                "reduced history carrier {name} has {} levels; the frame's pp3d extent requires {expected}, and a mismatched column would misplace every level of the rebuilt carrier",
                spec.shape()[0]
            )));
        }
    }
    let array = frame
        .arrays()
        .iter()
        .find(|array| array.name() == name)
        .ok_or_else(|| invalid(format!("reduced history payload lacks {name}")))?;
    if array.values().len() != spec.shape()[0] {
        return Err(invalid(format!(
            "reduced history carrier {name} holds {} words against a declared {} levels",
            array.values().len(),
            spec.shape()[0]
        )));
    }
    Ok(array.values())
}

fn bounds(shape: &[usize]) -> Result<Vec<i32>> {
    let mut values = Vec::with_capacity(2 * shape.len());
    for extent in shape {
        let upper = i32::try_from(*extent).map_err(|_| {
            invalid("reconstructed carrier extent does not fit a CM1 one-based bound")
        })?;
        values.push(1);
        values.push(upper);
    }
    Ok(values)
}

/// Rebuild every carrier a reduced GM1 history frame omits.
///
/// Refuses a complete frame by name: a complete frame already carries `prs`
/// and `zf`, and returning recomputed words for it would publish a second,
/// silently different copy of a field the artifact already holds.
pub fn reconstruct_reduced_history(
    contract: &ForecastContract,
    frame: &ForecastFrame,
) -> Result<Vec<ReconstructedCarrier>> {
    if !is_reduced_history(contract) {
        return Err(invalid(format!(
            "forecast profile {:?} is not {REDUCED_HISTORY_PROFILE_ID:?}; reconstructing {:?} for it would return recomputed words beside the carriers the artifact already holds",
            contract.profile_id(),
            RECONSTRUCTED_CARRIERS
        )));
    }
    let pp3d_spec = contract
        .arrays()
        .iter()
        .find(|spec| spec.name() == PP3D_CARRIER)
        .ok_or_else(|| {
            invalid(format!(
                "reduced history frame declares no {PP3D_CARRIER} carrier; the prs expression reads it at full extent, so nothing in the frame can rebuild prs"
            ))
        })?;
    if pp3d_spec.shape().len() != 3 {
        return Err(invalid(format!(
            "reduced history carrier {PP3D_CARRIER} has shape {:?}; the reconstruction requires a [z,y,x] mass carrier",
            pp3d_spec.shape()
        )));
    }
    let (nz, ny, nx) = (
        pp3d_spec.shape()[0],
        pp3d_spec.shape()[1],
        pp3d_spec.shape()[2],
    );
    let mass_words = nz
        .checked_mul(ny)
        .and_then(|value| value.checked_mul(nx))
        .ok_or_else(|| invalid("reduced history mass word count overflows"))?;
    let face_words = nz
        .checked_add(1)
        .and_then(|levels| levels.checked_mul(ny))
        .and_then(|value| value.checked_mul(nx))
        .ok_or_else(|| invalid("reduced history face word count overflows"))?;
    let pp3d = frame
        .arrays()
        .iter()
        .find(|array| array.name() == PP3D_CARRIER)
        .ok_or_else(|| invalid(format!("reduced history payload lacks {PP3D_CARRIER}")))?
        .values();
    if pp3d.len() != mass_words {
        return Err(invalid(format!(
            "reduced history {PP3D_CARRIER} holds {} words against a declared [{nz},{ny},{nx}]",
            pp3d.len()
        )));
    }
    let pi0 = column(contract, frame, PI0_COLUMN_CARRIER, Some(nz))?;
    let zf_profile = column(contract, frame, ZF_PROFILE_COLUMN_CARRIER, Some(nz + 1))?;

    let mut prs = Vec::new();
    prs.try_reserve_exact(mass_words).map_err(|_| {
        invalid(format!(
            "rebuilding prs needs {} host bytes, which this host did not give back",
            mass_words.saturating_mul(4)
        ))
    })?;
    let plane = ny
        .checked_mul(nx)
        .ok_or_else(|| invalid("reduced history horizontal plane overflows"))?;
    for level in 0..nz {
        let base = pi0[level];
        for word in &pp3d[level * plane..(level + 1) * plane] {
            // solve2.F:1588 evaluates the total Exner pressure first; the
            // producing kernel refuses a non-positive-normal one rather than
            // raising it to a power, and so does this.
            let pi = base + *word;
            if !pi.is_finite() || pi <= 0.0 {
                return Err(invalid(format!(
                    "reduced history total Exner pressure pi0+pp3d is {pi} at level {level}; raising it to cpdrd would publish a NaN or a negative pressure as rebuilt model output"
                )));
            }
            prs.push(P00_PA * pi.powf(CPDRD));
        }
    }

    let mut zf = Vec::new();
    zf.try_reserve_exact(face_words).map_err(|_| {
        invalid(format!(
            "rebuilding zf needs {} host bytes, which this host did not give back",
            face_words.saturating_mul(4)
        ))
    })?;
    for height in zf_profile.iter().copied() {
        if !height.is_finite() {
            return Err(invalid(
                "reduced history zf profile carries a non-finite height; broadcasting it would publish a non-finite grid height at every column",
            ));
        }
        zf.extend(std::iter::repeat_n(height, plane));
    }

    let mass_shape = vec![nz, ny, nx];
    let face_shape = vec![nz + 1, ny, nx];
    Ok(vec![
        ReconstructedCarrier {
            spec: ForecastArraySpec::new(
                "prs",
                PRS_MEANING,
                PRS_UNITS,
                PRS_STAGGER,
                bounds(&mass_shape)?,
                mass_shape,
            )?,
            array: ForecastArray::new("prs", prs),
            formula: PRS_FORMULA,
        },
        ReconstructedCarrier {
            spec: ForecastArraySpec::new(
                "zf",
                ZF_MEANING,
                ZF_UNITS,
                ZF_STAGGER,
                bounds(&face_shape)?,
                face_shape,
            )?,
            array: ForecastArray::new("zf", zf),
            formula: ZF_FORMULA,
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::{
        read_forecast_history_embedded, write_forecast_history_new, ForecastClock,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    const NZ: usize = 3;
    const NY: usize = 2;
    const NX: usize = 4;

    fn temp_path(label: &str) -> PathBuf {
        let sequence = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "cm1-reduced-history-{}-{label}-{sequence}",
            std::process::id()
        ))
    }

    fn hash(seed: u8) -> String {
        format!("{seed:02x}").repeat(32)
    }

    fn pi0_column() -> Vec<f32> {
        vec![0.9, 0.8, 0.7]
    }

    fn zf_column() -> Vec<f32> {
        vec![0.0, 250.0, 500.0, 750.0]
    }

    fn pp3d_words() -> Vec<f32> {
        (0..NZ * NY * NX)
            .map(|word| 0.01_f32 * word as f32)
            .collect()
    }

    fn spec(name: &str, units: &str, stagger: &str, shape: Vec<usize>) -> ForecastArraySpec {
        let bounds = shape
            .iter()
            .flat_map(|extent| [1i32, i32::try_from(*extent).unwrap()])
            .collect::<Vec<_>>();
        ForecastArraySpec::new(name, name, units, stagger, bounds, shape).unwrap()
    }

    fn reduced_contract(profile_id: &str) -> ForecastContract {
        ForecastContract::new(
            profile_id,
            hash(0x11),
            hash(0x22),
            vec![
                spec(PI0_COLUMN_CARRIER, "nondimensional", "mass-column", vec![NZ]),
                spec(ZF_PROFILE_COLUMN_CARRIER, "m", "w-column", vec![NZ + 1]),
                spec(PP3D_CARRIER, "nondimensional", "mass", vec![NZ, NY, NX]),
            ],
            Vec::new(),
            Vec::new(),
        )
        .unwrap()
    }

    fn reduced_frame() -> ForecastFrame {
        ForecastFrame::new(
            ForecastClock::new(600, 3600.0, 0.5).unwrap(),
            vec![
                ForecastArray::new(PI0_COLUMN_CARRIER, pi0_column()),
                ForecastArray::new(ZF_PROFILE_COLUMN_CARRIER, zf_column()),
                ForecastArray::new(PP3D_CARRIER, pp3d_words()),
            ],
            Vec::new(),
            Vec::new(),
        )
    }

    #[test]
    fn the_pinned_constants_are_the_words_the_producing_kernel_holds() {
        // The GM1 pressure kernels evaluate p00 as the binary32 constant
        // 0x47c35000 and raise pi0+pp3d to the binary32 cpdrd 0x40603c81.  A
        // reconstruction that used any other word would rebuild a different
        // field under a CM1 name.
        assert_eq!(P00_PA.to_bits(), 0x47c3_5000);
        assert_eq!(CPDRD.to_bits(), 0x4060_3c81);
        assert_eq!(RD_J_PER_KG_K.to_bits(), 0x438f_851f);
        assert_eq!(CP_J_PER_KG_K.to_bits(), 0x447b_6ccd);
    }

    /// Premise 1 of the module's bit-identity claim, measured on this host.
    ///
    /// `reduced_history` rebuilds `prs` with `f32::powf`.  That is only the
    /// word a complete frame carries if `f32::powf` is the platform `powf` the
    /// producing kernel transcribes -- if Rust instead computed in `f64` and
    /// rounded once, or linked a different libm, the rebuilt pressures would be
    /// a different field published under a CM1 name.  So compare the two
    /// directly, over the Exner range a GM1 column actually spans: `pi0+pp3d`
    /// runs from roughly 0.15 at a 20 km model top to a little over 1.0 at the
    /// ground, and the neighbourhood of 1.0 is where the log2/exp2 pair is most
    /// likely to split a word.
    #[test]
    fn the_rebuilt_pressure_is_the_host_libm_powf_word_for_word() {
        extern "C" {
            fn powf(x: f32, y: f32) -> f32;
        }

        let mut sampled = 0usize;
        let mut check = |pi: f32| {
            assert!(pi.is_finite() && pi > 0.0 && pi.is_normal());
            let rebuilt = P00_PA * pi.powf(CPDRD);
            let platform = P00_PA * unsafe { powf(pi, CPDRD) };
            assert_eq!(
                rebuilt.to_bits(),
                platform.to_bits(),
                "prs at pi={pi:e} is not the platform powf word"
            );
            sampled += 1;
        };
        // The physical range, walked finely enough to cross every table entry
        // of the glibc log2 stage many times over.
        for step in 0..90_001u32 {
            check(0.15 + (step as f32) * 1.0e-5);
        }
        // Ten thousand consecutive binary32 words either side of one.
        let one = 1.0_f32.to_bits();
        for offset in -5_000i32..=5_000 {
            check(f32::from_bits((one as i64 + offset as i64) as u32));
        }
        // Both ends of what a positive-normal Exner word can be at all.
        check(f32::MIN_POSITIVE);
        check(f32::MAX);
        assert!(sampled >= 100_000, "only {sampled} words sampled");
    }

    #[test]
    fn a_published_reduced_frame_reads_back_and_rebuilds_prs_and_zf() {
        let contract = reduced_contract(REDUCED_HISTORY_PROFILE_ID);
        let frame = reduced_frame();
        let path = temp_path("round-trip.cm1fhs");
        let receipt = write_forecast_history_new(&contract, &frame, &path).unwrap();
        assert_eq!(
            receipt.payload_nbytes,
            4 * (NZ + (NZ + 1) + NZ * NY * NX)
        );

        let (read_contract, restored) =
            read_forecast_history_embedded(&path, &contract.sha256()).unwrap();
        assert!(is_reduced_history(&read_contract));
        assert!(read_contract
            .arrays()
            .iter()
            .all(|spec| spec.name() != "prs" && spec.name() != "zf"));

        let rebuilt = reconstruct_reduced_history(&read_contract, restored.frame()).unwrap();
        let names = rebuilt
            .iter()
            .map(ReconstructedCarrier::name)
            .collect::<Vec<_>>();
        assert_eq!(names, RECONSTRUCTED_CARRIERS.to_vec());

        let prs = &rebuilt[0];
        assert_eq!(prs.shape(), [NZ, NY, NX]);
        assert_eq!(prs.values().len(), NZ * NY * NX);
        assert!(prs.formula().contains("solve2.F:1588"));
        let pi0 = pi0_column();
        let pp3d = pp3d_words();
        for level in 0..NZ {
            for row in 0..NY {
                for column in 0..NX {
                    let word = (level * NY + row) * NX + column;
                    let pi = pi0[level] + pp3d[word];
                    assert_eq!(
                        prs.values()[word].to_bits(),
                        (P00_PA * pi.powf(CPDRD)).to_bits(),
                        "prs word {word} is not p00*((pi0+pp3d)**cpdrd)"
                    );
                }
            }
        }
        // Index order, not just the expression: level 1, row 1, column 2 reads
        // pi0[1] and the word at that exact position, never a transpose.
        let word = (1 * NY + 1) * NX + 2;
        assert_eq!(
            prs.values()[word].to_bits(),
            (P00_PA * (0.8_f32 + 0.01_f32 * word as f32).powf(CPDRD)).to_bits()
        );

        let zf = &rebuilt[1];
        assert_eq!(zf.shape(), [NZ + 1, NY, NX]);
        assert!(zf.formula().contains("zf_profile"));
        for (level, height) in zf_column().into_iter().enumerate() {
            for word in 0..NY * NX {
                assert_eq!(zf.values()[level * NY * NX + word].to_bits(), height.to_bits());
            }
        }
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn the_rebuilt_carriers_are_the_complete_frame_s_own_words() {
        // The trade is only honest if a reader of the reduced frame sees the
        // same field a reader of the complete frame does.  Publish both from
        // one state, read both back, and compare bit for bit.
        let mass = vec![NZ, NY, NX];
        let faces = vec![NZ + 1, NY, NX];
        let pi0 = pi0_column();
        let zf_column = zf_column();
        let pp3d = pp3d_words();
        let prs = (0..NZ * NY * NX)
            .map(|word| {
                let pi = pi0[word / (NY * NX)] + pp3d[word];
                P00_PA * pi.powf(CPDRD)
            })
            .collect::<Vec<f32>>();
        let zf = zf_column
            .iter()
            .flat_map(|height| std::iter::repeat_n(*height, NY * NX))
            .collect::<Vec<f32>>();

        let complete_contract = ForecastContract::new(
            COMPLETE_HISTORY_PROFILE_ID,
            hash(0x11),
            hash(0x22),
            vec![
                spec(PP3D_CARRIER, "nondimensional", "mass", mass.clone()),
                spec("prs", "Pa", "mass", mass.clone()),
                spec("zf", "m", "w", faces.clone()),
            ],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let complete_frame = ForecastFrame::new(
            ForecastClock::new(600, 3600.0, 0.5).unwrap(),
            vec![
                ForecastArray::new(PP3D_CARRIER, pp3d.clone()),
                ForecastArray::new("prs", prs),
                ForecastArray::new("zf", zf),
            ],
            Vec::new(),
            Vec::new(),
        );
        let complete_path = temp_path("complete.cm1fhs");
        write_forecast_history_new(&complete_contract, &complete_frame, &complete_path).unwrap();

        let reduced_path = temp_path("reduced.cm1fhs");
        let reduced_contract = reduced_contract(REDUCED_HISTORY_PROFILE_ID);
        write_forecast_history_new(&reduced_contract, &reduced_frame(), &reduced_path).unwrap();
        // The reduced frame really is smaller, and by the two carriers it drops.
        assert!(
            fs::metadata(&reduced_path).unwrap().len() < fs::metadata(&complete_path).unwrap().len()
        );

        let (complete_read, complete_restored) =
            read_forecast_history_embedded(&complete_path, &complete_contract.sha256()).unwrap();
        assert!(!is_reduced_history(&complete_read));
        let (reduced_read, reduced_restored) =
            read_forecast_history_embedded(&reduced_path, &reduced_contract.sha256()).unwrap();
        let rebuilt = reconstruct_reduced_history(&reduced_read, reduced_restored.frame()).unwrap();

        for carrier in &rebuilt {
            let published = complete_restored
                .frame()
                .arrays()
                .iter()
                .find(|array| array.name() == carrier.name())
                .expect("the complete frame carries it");
            assert_eq!(
                published.values().len(),
                carrier.values().len(),
                "{} word count",
                carrier.name()
            );
            for (word, (want, got)) in published
                .values()
                .iter()
                .zip(carrier.values())
                .enumerate()
            {
                assert_eq!(
                    want.to_bits(),
                    got.to_bits(),
                    "{} word {word} differs",
                    carrier.name()
                );
            }
        }
        fs::remove_file(&complete_path).unwrap();
        fs::remove_file(&reduced_path).unwrap();
    }

    #[test]
    fn a_complete_frame_is_refused_rather_than_given_recomputed_carriers() {
        let contract = reduced_contract(COMPLETE_HISTORY_PROFILE_ID);
        let frame = reduced_frame();
        assert!(!is_reduced_history(&contract));
        let refusal = reconstruct_reduced_history(&contract, &frame)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains(COMPLETE_HISTORY_PROFILE_ID), "{refusal}");
        assert!(
            refusal.contains("carriers the artifact already holds"),
            "{refusal}"
        );
    }

    #[test]
    fn a_reduced_frame_missing_an_input_of_its_own_formula_refuses_by_name() {
        for absent in [PI0_COLUMN_CARRIER, ZF_PROFILE_COLUMN_CARRIER, PP3D_CARRIER] {
            let specs = reduced_contract(REDUCED_HISTORY_PROFILE_ID)
                .arrays()
                .iter()
                .filter(|spec| spec.name() != absent)
                .cloned()
                .collect::<Vec<_>>();
            let contract = ForecastContract::new(
                REDUCED_HISTORY_PROFILE_ID,
                hash(0x11),
                hash(0x22),
                specs,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
            let arrays = reduced_frame()
                .arrays()
                .iter()
                .filter(|array| array.name() != absent)
                .cloned()
                .collect::<Vec<_>>();
            let frame = ForecastFrame::new(
                ForecastClock::new(600, 3600.0, 0.5).unwrap(),
                arrays,
                Vec::new(),
                Vec::new(),
            );
            let refusal = reconstruct_reduced_history(&contract, &frame)
                .unwrap_err()
                .to_string();
            assert!(refusal.contains(absent), "{absent}: {refusal}");
        }
    }

    #[test]
    fn a_column_of_the_wrong_depth_or_rank_refuses_before_it_misplaces_a_level() {
        let contract = ForecastContract::new(
            REDUCED_HISTORY_PROFILE_ID,
            hash(0x11),
            hash(0x22),
            vec![
                spec(PI0_COLUMN_CARRIER, "nondimensional", "mass-column", vec![NZ]),
                // One level short of the nz+1 faces pp3d implies.
                spec(ZF_PROFILE_COLUMN_CARRIER, "m", "w-column", vec![NZ]),
                spec(PP3D_CARRIER, "nondimensional", "mass", vec![NZ, NY, NX]),
            ],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let frame = ForecastFrame::new(
            ForecastClock::new(600, 3600.0, 0.5).unwrap(),
            vec![
                ForecastArray::new(PI0_COLUMN_CARRIER, pi0_column()),
                ForecastArray::new(ZF_PROFILE_COLUMN_CARRIER, vec![0.0, 250.0, 500.0]),
                ForecastArray::new(PP3D_CARRIER, pp3d_words()),
            ],
            Vec::new(),
            Vec::new(),
        );
        let refusal = reconstruct_reduced_history(&contract, &frame)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("misplace every level"), "{refusal}");

        let contract = ForecastContract::new(
            REDUCED_HISTORY_PROFILE_ID,
            hash(0x11),
            hash(0x22),
            vec![
                // A full field where the reconstruction expects one column.
                spec(PI0_COLUMN_CARRIER, "nondimensional", "mass", vec![NZ, NY, NX]),
                spec(ZF_PROFILE_COLUMN_CARRIER, "m", "w-column", vec![NZ + 1]),
                spec(PP3D_CARRIER, "nondimensional", "mass", vec![NZ, NY, NX]),
            ],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let frame = ForecastFrame::new(
            ForecastClock::new(600, 3600.0, 0.5).unwrap(),
            vec![
                ForecastArray::new(PI0_COLUMN_CARRIER, pp3d_words()),
                ForecastArray::new(ZF_PROFILE_COLUMN_CARRIER, zf_column()),
                ForecastArray::new(PP3D_CARRIER, pp3d_words()),
            ],
            Vec::new(),
            Vec::new(),
        );
        let refusal = reconstruct_reduced_history(&contract, &frame)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("arbitrary horizontal point"), "{refusal}");
    }

    #[test]
    fn a_non_positive_total_exner_pressure_refuses_instead_of_rebuilding_a_nan() {
        let contract = reduced_contract(REDUCED_HISTORY_PROFILE_ID);
        let mut arrays = reduced_frame().arrays().to_vec();
        arrays[0] = ForecastArray::new(PI0_COLUMN_CARRIER, vec![0.9, -1.0, 0.7]);
        let frame = ForecastFrame::new(
            ForecastClock::new(600, 3600.0, 0.5).unwrap(),
            arrays,
            Vec::new(),
            Vec::new(),
        );
        let refusal = reconstruct_reduced_history(&contract, &frame)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("total Exner pressure"), "{refusal}");
        assert!(refusal.contains("level 1"), "{refusal}");
    }
}
