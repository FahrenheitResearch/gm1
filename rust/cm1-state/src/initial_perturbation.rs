//! CM1 r21.1 `init3d.F` initial 3-D perturbations: the `iinit` set and `irandp`.
//!
//! `init3d.F:450-1416` is one `IF(iinit.eq.1) ... ELSEIF ...` chain, each arm
//! opening with its own named parameter block that a CM1 user edits in the
//! source.  Those blocks are this module's configuration; every name here is
//! CM1's own (`ric`, `rjc`, `zc`, `bhrad`, `bvrad`, `bptpert`, `nbub`,
//! `zdep`, `maintain_rh`).
//!
//! Two arms need the base state and therefore run after it:
//! `iinit=5` converts a temperature perturbation through `pi0`
//! (`init3d.F:686-693`) and `maintain_rh` rescales `qv` through `rslf`
//! (`init3d.F:1424-1441`).  [`ThetaPerturbationField`] carries what the
//! coordinate-only arms produce; [`apply_base_dependent_perturbation`] and
//! [`apply_maintain_relative_humidity`] finish the job.
//!
//! `init3d.F:1446-1600`'s `irandp=1` adds a uniform random field to `tha`.
//! CM1 draws from gfortran's `random_number` seeded by `init3d.F:427-440`;
//! that sequence is a property of one Fortran runtime and is not reproducible
//! here, so this port declares its own generator and its own seed and records
//! both.  The *structure* is CM1's: the same traversal, the same number of
//! draws per point, the same amplitude and the same depth limit.

use crate::idealized_initialization::{checked_product, index3, invalid, require_finite, rslf};
use crate::Result;

const PI: f32 = std::f32::consts::PI;

/// Which `init3d.F` arm one selection names, and whether this port has it.
/// 9, 10 and 12 build no `tha` at all: their `init3d.F` arms
/// (`:1303-1333`, `:1334-1347`, `:1370-1396`) only set control words, and the
/// perturbation is a forcing the runtime applies during the first minutes
/// (`cm1core.initiation_forcing`).
/// 7 builds no tha here either: its vortex needs the base state and is
/// built after it (`idealized_initialization::apply_tc_vortex_arm`), and so
/// does 4, the moist benchmark bubble
/// (`idealized_initialization::apply_moist_benchmark_arm`).
pub const IMPLEMENTED_IINIT: [i32; 13] = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13];

/// `init3d.F` arms this port does not build, each with the concrete wrong
/// result admitting it would produce.
pub fn iinit_refusal(iinit: i32) -> Option<String> {
    if IMPLEMENTED_IINIT.contains(&iinit) {
        return None;
    }
    Some(format!(
        "init3d.F:450-1416 implements iinit in {{0,1,2,3,4,5,7,8,9,10,11,12,13}}; got \
         iinit={iinit}. CM1's chain has no else arm, so an unlisted value would leave \
         tha at exactly zero and the run would start from an undisturbed environment \
         while its receipt named a trigger"
    ))
}

/// The `init3d.F` named parameter block for whichever arm is selected.
///
/// Every field is CM1's own variable.  `official_warm_bubble` is
/// `init3d.F:456-461`; the other arms' shipped values are named on their
/// constructors so a caller can reproduce a CM1 deck exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InitialPerturbationControls {
    center_x_m: f32,
    center_y_m: f32,
    center_z_agl_m: f32,
    horizontal_radius_m: f32,
    vertical_radius_m: f32,
    amplitude_k: f32,
    maintain_relative_humidity: bool,
    bubble_count: i32,
    bubble_spacing_y_m: f32,
    cold_pool_depth_m: f32,
    random_line_amplitude_k: f32,
}

/// The name this module had when `iinit=1` was its only arm.
pub type WarmBubbleControls = InitialPerturbationControls;

impl InitialPerturbationControls {
    /// Official CM1 `iinit=1` controls (`init3d.F:456-461`), with the
    /// caller-provided domain center for `ric`/`rjc`.
    pub fn official(center_x_m: f32, center_y_m: f32) -> Result<Self> {
        Self::new(
            center_x_m,
            center_y_m,
            1400.0_f32,
            10_000.0_f32,
            1400.0_f32,
            1.0_f32,
            false,
        )
    }

    /// The seven `iinit=1` words; every other arm's extra words keep their
    /// `init3d.F` defaults and are set with the builders below.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        center_x_m: f32,
        center_y_m: f32,
        center_z_agl_m: f32,
        horizontal_radius_m: f32,
        vertical_radius_m: f32,
        amplitude_k: f32,
        maintain_relative_humidity: bool,
    ) -> Result<Self> {
        let controls = Self {
            center_x_m,
            center_y_m,
            center_z_agl_m,
            horizontal_radius_m,
            vertical_radius_m,
            amplitude_k,
            maintain_relative_humidity,
            // init3d.F:521-566.
            bubble_count: 3,
            bubble_spacing_y_m: 30_000.0_f32,
            // init3d.F:489-518.
            cold_pool_depth_m: 2500.0_f32,
            // init3d.F:1275.
            random_line_amplitude_k: 0.20_f32,
        };
        controls.validate()?;
        Ok(controls)
    }

    /// `init3d.F:525-530`: `nbub` bubbles, `rjc` stepping 30 km from 3 km.
    pub fn with_bubble_line(mut self, bubble_count: i32, bubble_spacing_y_m: f32) -> Result<Self> {
        self.bubble_count = bubble_count;
        self.bubble_spacing_y_m = bubble_spacing_y_m;
        self.validate()?;
        Ok(self)
    }

    /// `init3d.F:494`: `zdep`, the depth of the dam-break cold pool.
    pub fn with_cold_pool_depth(mut self, cold_pool_depth_m: f32) -> Result<Self> {
        self.cold_pool_depth_m = cold_pool_depth_m;
        self.validate()?;
        Ok(self)
    }

    /// `init3d.F:1275`: the `iinit=8` random amplitude added inside the line.
    pub fn with_random_line_amplitude(mut self, amplitude_k: f32) -> Result<Self> {
        self.random_line_amplitude_k = amplitude_k;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn validate(self) -> Result<()> {
        for (label, value) in [
            ("center x", self.center_x_m),
            ("center y", self.center_y_m),
            ("center z", self.center_z_agl_m),
            ("horizontal radius", self.horizontal_radius_m),
            ("vertical radius", self.vertical_radius_m),
            ("amplitude", self.amplitude_k),
            ("bubble spacing y", self.bubble_spacing_y_m),
            ("cold pool depth", self.cold_pool_depth_m),
            ("random line amplitude", self.random_line_amplitude_k),
        ] {
            require_finite(label, value)?;
        }
        if self.horizontal_radius_m <= 0.0 || self.vertical_radius_m <= 0.0 {
            return Err(invalid("warm-bubble radii must be positive"));
        }
        if self.bubble_count < 1 {
            return Err(invalid(format!(
                "init3d.F:525 loops n=1,nbub; nbub={} would place no bubble at all and the \
                 run would start undisturbed",
                self.bubble_count
            )));
        }
        if self.cold_pool_depth_m <= 0.0 {
            return Err(invalid(
                "init3d.F:497 divides by zdep; a non-positive cold-pool depth has no perturbation",
            ));
        }
        Ok(())
    }

    pub fn center_x_m(self) -> f32 {
        self.center_x_m
    }
    pub fn center_y_m(self) -> f32 {
        self.center_y_m
    }
    pub fn center_z_agl_m(self) -> f32 {
        self.center_z_agl_m
    }
    pub fn horizontal_radius_m(self) -> f32 {
        self.horizontal_radius_m
    }
    pub fn vertical_radius_m(self) -> f32 {
        self.vertical_radius_m
    }
    pub fn amplitude_k(self) -> f32 {
        self.amplitude_k
    }
    pub fn maintain_relative_humidity(self) -> bool {
        self.maintain_relative_humidity
    }
    pub fn bubble_count(self) -> i32 {
        self.bubble_count
    }
    pub fn bubble_spacing_y_m(self) -> f32 {
        self.bubble_spacing_y_m
    }
    pub fn cold_pool_depth_m(self) -> f32 {
        self.cold_pool_depth_m
    }
    pub fn random_line_amplitude_k(self) -> f32 {
        self.random_line_amplitude_k
    }
}

/// A named, seeded, reproducible uniform stream standing in for gfortran's.
///
/// `init3d.F:416-443` seeds one process-wide Fortran generator and every
/// random arm below draws from it in source order.  This is that stream.  It
/// is SplitMix64 (Steele, Lea and Flood 2014) taking the top 24 bits of each
/// output, so one draw is an exact multiple of `2^-24` in `[0,1)` and every
/// platform with IEEE binary32 produces the same words from the same seed.
/// It is deliberately NOT gfortran's xorshift1024\*: matching CM1's numbers
/// would require pinning one libgfortran build, and a seed recorded in a
/// receipt is worth more than a sequence that is only reproducible on one
/// compiler.
#[derive(Clone, Debug)]
pub struct Cm1RandomStream {
    state: u64,
    seed: u64,
    draws: u64,
    /// `Some` for CM1's own generator (see [`Cm1RandomStream::cm1_gfortran`]).
    xoshiro: Option<[u64; 4]>,
}

pub const RANDOM_STREAM_GENERATOR: &str = "splitmix64/top24-binary32";

/// CM1's own stream: libgfortran's `random_number` (xoshiro256**, GCC 9 and
/// later) as `init3d.F:427-440` seeds it.
pub const CM1_RANDOM_STREAM_GENERATOR: &str =
    "gfortran-xoshiro256starstar/init3d.F:427-440-seed/top24-binary32";

/// libgfortran `random.c`'s `xor_keys`, which `random_seed(put=)` XORs into
/// the state after copying the put array in reverse order.  Read from
/// libgfortran.so.5 (GCC 15.2, Linux) and checked against its output.
const GFORTRAN_XOR_KEYS: [u64; 4] = [
    0xbd0c_5b6e_50c2_df49,
    0xd460_61cd_46e1_df38,
    0xbb4f_4d4e_d610_3544,
    0x114a_583d_0756_ad39,
];

impl Cm1RandomStream {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            seed,
            draws: 0,
            xoshiro: None,
        }
    }

    /// CM1's own random numbers, exactly: `init3d.F:427-440` with
    /// `use_truly_random_pert` false (CM1's default).
    ///
    /// `random_seed(size=k)` is 8 for xoshiro256**; seed word `n` is the
    /// nearest integer to 2.0e9 times `2(n-1)/(k-1) - 1`, in default REAL;
    /// `random_seed(put=sand)` copies the eight words in reverse order and
    /// XORs libgfortran's keys in; then `init3d.F:437` draws one number (the
    /// logged `rand-1`) before any arm draws.  The stream returned has taken
    /// that draw.
    pub fn cm1_gfortran() -> Self {
        let k = 8_i32;
        let mut sand = [0_i32; 8];
        for (n, word) in sand.iter_mut().enumerate() {
            let x = 2.0e9_f32 * (2.0_f32 * (n as f32) / ((k - 1) as f32) - 1.0_f32);
            // Fortran nint: round half away from zero.
            *word = x.round() as i32;
        }
        let mut bytes = [0_u8; 32];
        for (i, word) in sand.iter().rev().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
        }
        let mut s = [0_u64; 4];
        for (i, lane) in s.iter_mut().enumerate() {
            let mut word = [0_u8; 8];
            word.copy_from_slice(&bytes[8 * i..8 * i + 8]);
            *lane = u64::from_le_bytes(word) ^ GFORTRAN_XOR_KEYS[i];
        }
        let mut stream = Self {
            state: 0,
            seed: 0,
            draws: 0,
            xoshiro: Some(s),
        };
        // init3d.F:437: call random_number(rand); print rand-1.
        let _ = stream.next_unit();
        stream
    }

    pub fn is_cm1_gfortran(&self) -> bool {
        self.xoshiro.is_some()
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn draws(&self) -> u64 {
        self.draws
    }

    /// One `call random_number(rand)`: a binary32 in `[0,1)`.
    pub fn next_unit(&mut self) -> f32 {
        self.draws += 1;
        if let Some(s) = self.xoshiro.as_mut() {
            // libgfortran prng_next (xoshiro256**), then random_r4: the high
            // 32 bits, masked to binary32's 24 digits, times 2**-32.
            let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
            let t = s[1] << 17;
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = s[3].rotate_left(45);
            let high = ((result >> 32) as u32) & (u32::MAX << 8);
            return (high as f32) * (1.0_f32 / 4_294_967_296.0_f32);
        }
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32) * (1.0_f32 / 16_777_216.0_f32)
    }
}

/// `init3d.F:1446-1600`: the `irandp=1` amplitude and depth limit, by
/// `testcase`.
///
/// `testcase=0` (`init3d.F:1459-1482`) perturbs every level at 0.25 K and
/// draws twice per point.  The pre-configured cases perturb only below a
/// named height.  `depth_limit_m = None` means "every level".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomPerturbationLaw {
    pub amplitude_k: f32,
    pub depth_limit_m: Option<f32>,
    pub draws_per_point: u32,
    pub margin_cells: usize,
}

/// The `testcase` arms of `init3d.F:1459-1590` this port implements.
pub fn random_perturbation_law(testcase: i32) -> Result<RandomPerturbationLaw> {
    Ok(match testcase {
        // init3d.F:1459-1482.
        0 => RandomPerturbationLaw {
            amplitude_k: 0.25_f32,
            depth_limit_m: None,
            draws_per_point: 2,
            margin_cells: 2,
        },
        // init3d.F:1484-1510: convective/shear/hurricane boundary layers.
        1 | 2 | 6 | 10 | 11 | 15 => RandomPerturbationLaw {
            amplitude_k: 0.10_f32,
            depth_limit_m: Some(100.0_f32),
            draws_per_point: 2,
            margin_cells: 1,
        },
        // init3d.F:1533-1552.
        4 | 5 | 7 => RandomPerturbationLaw {
            amplitude_k: 0.10_f32,
            depth_limit_m: Some(1000.0_f32),
            draws_per_point: 1,
            margin_cells: 1,
        },
        // init3d.F:1575-1590: Beare et al. (2006) stable boundary layer.
        9 => RandomPerturbationLaw {
            amplitude_k: 0.10_f32,
            depth_limit_m: Some(50.0_f32),
            draws_per_point: 1,
            margin_cells: 1,
        },
        // init3d.F:1512-1531 (testcase=3) also perturbs qv, and
        // init3d.F:1554-1573 (testcase=8) uses a per-level amplitude over the
        // lowest five levels; init3d.F:1592-1620 (testcase 12/13) perturbs
        // velocity, not theta.  None of the three is a theta-only uniform
        // field, so none is claimed here.
        3 | 8 | 12 | 13 => {
            return Err(invalid(format!(
                "init3d.F's irandp arm for testcase={testcase} is not a uniform theta field: \
                 testcase=3 also perturbs qv, testcase=8 varies the amplitude per level over the \
                 lowest five, and testcase=12/13 perturb u and v instead of theta. Admitting it \
                 here would add a plain theta field and record the named test case"
            )))
        }
        other => {
            return Err(invalid(format!(
                "init3d.F:1459-1620 branches on testcase in 0..15; got testcase={other}. CM1's \
                 chain has no else arm, so no perturbation at all would be added and the run \
                 would stay laminar under a receipt naming irandp=1"
            )))
        }
    })
}

/// What `init3d.F` did with the requested `iinit`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PerturbationDisposition {
    NotRequested,
    Applied(i32),
    DisabledByRestart,
}

/// One built `tha` field plus what produced it.
#[derive(Clone, Debug)]
pub struct ThetaPerturbationField {
    pub shape_zyx: [usize; 3],
    pub theta_perturbation_k: Vec<f32>,
    pub disposition: PerturbationDisposition,
    pub random_draws: u64,
}

/// `init3d.F:450-1416`'s coordinate-only arms, in native loop order.
///
/// `stream` is consumed only by `iinit=8` and by `irandp`; a caller that
/// selects neither may pass a stream and it stays untouched.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn build_theta_perturbation(
    effective_iinit: i32,
    controls: InitialPerturbationControls,
    xh_m: &[f32],
    yh_m: &[f32],
    zh_agl_m: &[f32],
    shape_zyx: [usize; 3],
    domain_x_span_m: f32,
    domain_y_span_m: f32,
    stream: &mut Cm1RandomStream,
) -> Result<Vec<f32>> {
    controls.validate()?;
    let [nz, ny, nx] = shape_zyx;
    let words = checked_product(shape_zyx, "theta perturbation")?;
    let mut tha = vec![0.0_f32; words];
    match effective_iinit {
        // init3d.F:1303-1396: forced convergence (9), momentum forcing (10)
        // and updraft nudging (12) assign control words only; tha stays at
        // init3d.F's zero and the forcing is applied during the integration.
        // init3d.F:697-1256's vortex and init3d.F:579-647's moist benchmark
        // bubble are built after the base state.
        0 | 4 | 7 | 9 | 10 | 12 => {}
        // init3d.F:450-482.
        1 => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let index = index3(k, j, i, ny, nx);
                        let rx = (xh_m[i] - controls.center_x_m) / controls.horizontal_radius_m;
                        let ry = (yh_m[j] - controls.center_y_m) / controls.horizontal_radius_m;
                        let rz = (zh_agl_m[index] - controls.center_z_agl_m)
                            / controls.vertical_radius_m;
                        let beta = (rx * rx + ry * ry + rz * rz).sqrt();
                        tha[index] = if beta < 1.0_f32 {
                            let cosine = (0.5_f32 * PI * beta).cos();
                            controls.amplitude_k * (cosine * cosine)
                        } else {
                            0.0_f32
                        };
                    }
                }
            }
        }
        // init3d.F:489-518: dam-break cold pool, everything west of ric and
        // below zdep, linear in height.
        2 => {
            let zdep = controls.cold_pool_depth_m;
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let index = index3(k, j, i, ny, nx);
                        tha[index] = if xh_m[i] <= controls.center_x_m && zh_agl_m[index] < zdep {
                            controls.amplitude_k * (zdep - zh_agl_m[index]) / zdep
                        } else {
                            0.0_f32
                        };
                    }
                }
            }
        }
        // init3d.F:521-576: a line of nbub bubbles stepping in y.
        3 => {
            for n in 0..controls.bubble_count {
                let rjc = controls.center_y_m + (n as f32) * controls.bubble_spacing_y_m;
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            let index = index3(k, j, i, ny, nx);
                            let rx = (xh_m[i] - controls.center_x_m) / controls.horizontal_radius_m;
                            let ry = (yh_m[j] - rjc) / controls.horizontal_radius_m;
                            let rz = (zh_agl_m[index] - controls.center_z_agl_m)
                                / controls.vertical_radius_m;
                            let beta = (rx * rx + ry * ry + rz * rz).sqrt();
                            tha[index] = if beta < 1.0_f32 {
                                let cosine = (0.5_f32 * PI * beta).cos();
                                controls.amplitude_k * (cosine * cosine)
                            } else {
                                // init3d.F:568: max(0,tha) keeps an earlier
                                // bubble's words instead of overwriting them.
                                tha[index].max(0.0_f32)
                            };
                        }
                    }
                }
            }
        }
        // init3d.F:654-695 needs pi0; it is finished by
        // apply_base_dependent_perturbation and leaves tha at zero here.
        5 => {}
        // init3d.F:1257-1296: a 2-D (x,z) line thermal with random noise.
        8 => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let index = index3(k, j, i, ny, nx);
                        // init3d.F:1277-1281 draws once per (k,jj,ii) over the
                        // whole global domain, before testing ownership.
                        let rand = stream.next_unit();
                        let rx = (xh_m[i] - controls.center_x_m) / controls.horizontal_radius_m;
                        let rz = (zh_agl_m[index] - controls.center_z_agl_m)
                            / controls.vertical_radius_m;
                        let beta = (rx * rx + rz * rz).sqrt();
                        tha[index] = if beta < 1.0_f32 {
                            let cosine = (0.5_f32 * PI * beta).cos();
                            controls.amplitude_k * (cosine * cosine)
                                + controls.random_line_amplitude_k * (2.0_f32 * rand - 1.0_f32)
                        } else {
                            0.0_f32
                        };
                    }
                }
            }
        }
        // init3d.F:1348-1368: Skamarock and Klemp (1994) inertia-gravity wave.
        // The source spells the three lengths as literals; they are read here
        // from the bubble block so a different domain can place the same wave.
        11 => {
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let index = index3(k, j, i, ny, nx);
                        let vertical = (PI * zh_agl_m[index] / controls.vertical_radius_m).sin();
                        let shape = (xh_m[i] - controls.center_x_m) / controls.horizontal_radius_m;
                        tha[index] = controls.amplitude_k * vertical / (1.0_f32 + shape * shape);
                    }
                }
            }
        }
        // init3d.F:1398-1414: a sinusoidal near-surface theta wave whose
        // wavelengths are a quarter of the domain in x and y.
        13 => {
            let bvrad = controls.vertical_radius_m;
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let index = index3(k, j, i, ny, nx);
                        if zh_agl_m[index] < bvrad {
                            tha[index] = controls.amplitude_k
                                * (2.0_f32 * PI * xh_m[i] / (0.25_f32 * domain_x_span_m)).sin()
                                * (2.0_f32 * PI * yh_m[j] / (0.25_f32 * domain_y_span_m)).sin()
                                * (2.0_f32 * PI * zh_agl_m[index] / (2.0_f32 * bvrad)).sin();
                        }
                    }
                }
            }
        }
        other => {
            return Err(invalid(
                iinit_refusal(other).unwrap_or_else(|| format!("unimplemented iinit={other}")),
            ))
        }
    }
    Ok(tha)
}

/// `init3d.F:654-695`: the Straka cold blob, a TEMPERATURE perturbation
/// converted through `pi0`.
pub fn apply_base_dependent_perturbation(
    effective_iinit: i32,
    controls: InitialPerturbationControls,
    xh_m: &[f32],
    zh_agl_m: &[f32],
    pi0: &[f32],
    th0: &[f32],
    theta_perturbation: &mut [f32],
    shape_zyx: [usize; 3],
) -> Result<()> {
    if effective_iinit != 5 {
        return Ok(());
    }
    let [nz, ny, nx] = shape_zyx;
    let words = checked_product(shape_zyx, "cold blob")?;
    if pi0.len() != words || th0.len() != words || theta_perturbation.len() != words || zh_agl_m.len() != words {
        return Err(invalid("cold-blob fields must all carry the mass shape"));
    }
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let index = index3(k, j, i, ny, nx);
                let rx = (xh_m[i] - controls.center_x_m) / controls.horizontal_radius_m;
                let rz = (zh_agl_m[index] - controls.center_z_agl_m) / controls.vertical_radius_m;
                let beta = (rx * rx + rz * rz).sqrt();
                // init3d.F:678-684: cos(pi*beta) here, not cos(0.5*pi*beta)^2.
                let temperature_perturbation = if beta < 1.0_f32 {
                    controls.amplitude_k * ((PI * beta).cos() + 1.0_f32) * 0.5_f32
                } else {
                    0.0_f32
                };
                // init3d.F:688-692: tmp = th0*pi0 + dum1; tha = tmp/pi0 - th0,
                // in default REAL and in that order (dum1/pi0 is not the same
                // word: it differed from CM1's by up to 3e-5 K).
                let tmp = th0[index] * pi0[index] + temperature_perturbation;
                let value = tmp / pi0[index] - th0[index];
                theta_perturbation[index] = if value.abs() < 1.0e-4_f32 { 0.0_f32 } else { value };
            }
        }
    }
    Ok(())
}

/// `init3d.F:1424-1441`: hold relative humidity fixed across the perturbation.
///
/// `dum2 = qv0/rslf(prs0, th0*pi0)` is the base relative humidity with respect
/// to liquid, and `qv = dum2*rslf(prs0, (th0+tha)*pi0)` rescales the vapour to
/// keep it.  Without this CM1 holds `qv` itself fixed, which changes RH inside
/// the perturbation.
pub fn apply_maintain_relative_humidity(
    qv: &mut [f32],
    qv0: &[f32],
    prs0: &[f32],
    th0: &[f32],
    pi0: &[f32],
    theta_perturbation: &[f32],
) -> Result<()> {
    let words = qv.len();
    for (label, values) in [
        ("qv0", qv0),
        ("prs0", prs0),
        ("th0", th0),
        ("pi0", pi0),
        ("tha", theta_perturbation),
    ] {
        if values.len() != words {
            return Err(invalid(format!(
                "maintain_rh {label} length {} does not match qv length {words}",
                values.len()
            )));
        }
    }
    for index in 0..words {
        let saturation = rslf(prs0[index], th0[index] * pi0[index]);
        if !(saturation > 0.0) {
            return Err(invalid(format!(
                "init3d.F:1434 divides by rslf(prs0,t0); it is {saturation} at word {index}, so \
                 the base relative humidity is undefined and maintain_rh cannot be honoured"
            )));
        }
        let relative_humidity = qv0[index] / saturation;
        qv[index] = relative_humidity
            * rslf(
                prs0[index],
                (th0[index] + theta_perturbation[index]) * pi0[index],
            );
    }
    Ok(())
}

/// `init3d.F:1446-1600`: add the seeded uniform theta field.
///
/// The traversal is CM1's: `do k` outermost, then the global `jj`/`ii` sweep
/// with its margin cells, drawing `draws_per_point` numbers at every global
/// point whether or not this rank owns it, so the field does not depend on the
/// decomposition.  This port is serial and owns every point.
pub fn apply_random_perturbation(
    law: RandomPerturbationLaw,
    theta_perturbation: &mut [f32],
    zh_agl_m: &[f32],
    shape_zyx: [usize; 3],
    stream: &mut Cm1RandomStream,
) -> Result<()> {
    let [nz, ny, nx] = shape_zyx;
    let words = checked_product(shape_zyx, "random perturbation")?;
    if theta_perturbation.len() != words || zh_agl_m.len() != words {
        return Err(invalid(
            "random perturbation fields must carry the mass shape",
        ));
    }
    require_finite("irandp amplitude", law.amplitude_k)?;
    let margin = law.margin_cells;
    // CM1's own loops (init3d.F:1471-1472 and the testcase arms): jj runs
    // 0..ny+margin and j = jj, so interior row j (1-based) is draw row jj=j.
    // GM1's own stream keeps the traversal it has always had, so every seeded
    // field recorded before CM1's generator existed is unchanged.
    let (rows, cols, offset) = if stream.is_cm1_gfortran() {
        (ny + margin + 1, nx + margin + 1, 1_isize)
    } else {
        (ny + 2 * margin, nx + 2 * margin, margin as isize)
    };
    for k in 0..nz {
        // init3d.F:1493 tests zh(1,1,k), one column, for the whole level.
        if let Some(limit) = law.depth_limit_m {
            if !(zh_agl_m[index3(k, 0, 0, ny, nx)] <= limit) {
                continue;
            }
        }
        for jj in 0..rows {
            for ii in 0..cols {
                let mut first = 0.0_f32;
                for draw in 0..law.draws_per_point {
                    let value = stream.next_unit();
                    if draw == 0 {
                        first = value;
                    }
                }
                let j = jj as isize - offset;
                let i = ii as isize - offset;
                if j < 0 || i < 0 || j >= ny as isize || i >= nx as isize {
                    continue;
                }
                let index = index3(k, j as usize, i as usize, ny, nx);
                theta_perturbation[index] += law.amplitude_k * (2.0_f32 * first - 1.0_f32);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coords(nx: usize, ny: usize, nz: usize, dx: f32, dy: f32, dz: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let xh = (0..nx).map(|i| (i as f32 + 0.5) * dx).collect::<Vec<_>>();
        let yh = (0..ny).map(|j| (j as f32 + 0.5) * dy).collect::<Vec<_>>();
        let mut zh = vec![0.0_f32; nz * ny * nx];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    zh[index3(k, j, i, ny, nx)] = (k as f32 + 0.5) * dz;
                }
            }
        }
        (xh, yh, zh)
    }

    #[test]
    fn warm_bubble_arm_is_unchanged_and_placement_moves_the_maximum() {
        let (nx, ny, nz) = (20_usize, 20_usize, 10_usize);
        let (xh, yh, zh) = coords(nx, ny, nz, 1000.0, 1000.0, 500.0);
        let mut stream = Cm1RandomStream::new(1);
        let centered = InitialPerturbationControls::official(10_000.0, 10_000.0).unwrap();
        let tha = build_theta_perturbation(
            1, centered, &xh, &yh, &zh, [nz, ny, nx], 20_000.0, 20_000.0, &mut stream,
        )
        .unwrap();
        assert_eq!(stream.draws(), 0, "iinit=1 draws no random numbers");
        let peak = tha.iter().cloned().fold(f32::MIN, f32::max);
        // ric=10 km is a cell FACE on a 1 km grid, so the nearest mass point
        // sits half a cell off centre and the sampled peak is just under
        // bptpert.  That is CM1's own sampling, not a tolerance.
        assert!(peak > 0.9 && peak <= 1.0, "peak={peak}");
        let argmax = tha.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let (k, rest) = (argmax / (ny * nx), argmax % (ny * nx));
        let (j, i) = (rest / nx, rest % nx);
        assert!((j, i) == (9, 9) || (j, i) == (10, 10), "(j,i)={:?}", (j, i));
        assert_eq!(k, 2); // 1400 m sits between 1250 m and 1750 m

        // The same bubble placed a quarter domain west and south.
        let offset = InitialPerturbationControls::official(5_000.0, 5_000.0).unwrap();
        let moved = build_theta_perturbation(
            1, offset, &xh, &yh, &zh, [nz, ny, nx], 20_000.0, 20_000.0, &mut stream,
        )
        .unwrap();
        let argmax = moved.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let rest = argmax % (ny * nx);
        assert!(
            (rest / nx, rest % nx) == (4, 4) || (rest / nx, rest % nx) == (5, 5),
            "{:?}",
            (rest / nx, rest % nx)
        );
    }

    #[test]
    fn cold_pool_is_west_of_ric_and_linear_in_depth() {
        let (nx, ny, nz) = (8_usize, 4_usize, 8_usize);
        let (xh, yh, zh) = coords(nx, ny, nz, 1000.0, 1000.0, 500.0);
        let mut stream = Cm1RandomStream::new(7);
        let controls = InitialPerturbationControls::new(4000.0, 0.0, 0.0, 1.0, 1.0, -6.0, false)
            .unwrap()
            .with_cold_pool_depth(2500.0)
            .unwrap();
        let tha = build_theta_perturbation(
            2, controls, &xh, &yh, &zh, [nz, ny, nx], 8000.0, 4000.0, &mut stream,
        )
        .unwrap();
        // x = 3500 m (i=3) is west of ric=4000; x = 4500 m (i=4) is east.
        assert_eq!(tha[index3(0, 0, 4, ny, nx)], 0.0);
        assert_eq!(tha[index3(0, 0, 3, ny, nx)], -6.0 * (2500.0 - 250.0) / 2500.0);
        // z = 2750 m (k=5) is deeper than zdep.
        assert_eq!(tha[index3(5, 0, 3, ny, nx)], 0.0);
    }

    #[test]
    fn bubble_line_places_every_bubble_and_keeps_the_earlier_ones() {
        let (nx, ny, nz) = (8_usize, 90_usize, 6_usize);
        let (xh, yh, zh) = coords(nx, ny, nz, 1000.0, 1000.0, 500.0);
        let mut stream = Cm1RandomStream::new(3);
        // init3d.F:525-530's own deck: ric=30 km, rjc = 3, 33, 63 km.
        let controls = InitialPerturbationControls::new(
            4000.0, 3000.0, 1400.0, 10_000.0, 1400.0, 2.0, false,
        )
        .unwrap()
        .with_bubble_line(3, 30_000.0)
        .unwrap();
        let tha = build_theta_perturbation(
            3, controls, &xh, &yh, &zh, [nz, ny, nx], 8000.0, 90_000.0, &mut stream,
        )
        .unwrap();
        for centre in [3000.0_f32, 33_000.0, 63_000.0] {
            let j = (centre / 1000.0 - 0.5).round() as usize;
            let column = (0..nz)
                .map(|k| tha[index3(k, j, 3, ny, nx)])
                .fold(f32::MIN, f32::max);
            assert!(column > 1.9, "bubble at y={centre} m is missing (max={column})");
        }
        // Between two bubbles the field is back to zero.
        let between = (0..nz)
            .map(|k| tha[index3(k, 18, 3, ny, nx)])
            .fold(f32::MIN, f32::max);
        assert_eq!(between, 0.0);
    }

    #[test]
    fn random_field_is_seeded_reproducible_bounded_and_seed_sensitive() {
        let (nx, ny, nz) = (16_usize, 16_usize, 4_usize);
        let (_xh, _yh, zh) = coords(nx, ny, nz, 1000.0, 1000.0, 500.0);
        let law = random_perturbation_law(0).unwrap();
        assert_eq!(law.amplitude_k, 0.25);
        assert_eq!(law.draws_per_point, 2);
        let build = |seed: u64| {
            let mut field = vec![0.0_f32; nz * ny * nx];
            let mut stream = Cm1RandomStream::new(seed);
            apply_random_perturbation(law, &mut field, &zh, [nz, ny, nx], &mut stream).unwrap();
            (field, stream.draws())
        };
        let (a, draws_a) = build(20240611);
        let (b, _) = build(20240611);
        let (c, _) = build(20240612);
        assert_eq!(a, b, "the same seed must give the same field");
        assert_ne!(a, c, "a different seed must give a different field");
        // init3d.F:1471-1473 draws twice per global point including the two
        // margin cells on each side.
        assert_eq!(draws_a as usize, nz * (ny + 4) * (nx + 4) * 2);
        assert!(a.iter().all(|v| v.abs() <= 0.25), "amplitude bound broken");
        let mean = a.iter().sum::<f32>() / (a.len() as f32);
        assert!(mean.abs() < 0.01, "mean={mean} is not centred");
        // A uniform on [-A,A] has variance A^2/3.
        let variance = a.iter().map(|v| v * v).sum::<f32>() / (a.len() as f32);
        let expected = 0.25_f32 * 0.25_f32 / 3.0_f32;
        assert!(
            (variance / expected - 1.0).abs() < 0.08,
            "variance {variance} vs {expected}"
        );
    }

    #[test]
    fn depth_limited_testcases_perturb_only_below_their_named_height() {
        let (nx, ny, nz) = (8_usize, 8_usize, 8_usize);
        let (_xh, _yh, zh) = coords(nx, ny, nz, 1000.0, 1000.0, 50.0);
        let law = random_perturbation_law(9).unwrap();
        assert_eq!(law.depth_limit_m, Some(50.0));
        let mut field = vec![0.0_f32; nz * ny * nx];
        let mut stream = Cm1RandomStream::new(5);
        apply_random_perturbation(law, &mut field, &zh, [nz, ny, nx], &mut stream).unwrap();
        // zh(k=0) = 25 m <= 50 m; zh(k=1) = 75 m > 50 m.
        assert!(field[index3(0, 0, 0, ny, nx)] != 0.0);
        for k in 1..nz {
            for j in 0..ny {
                for i in 0..nx {
                    assert_eq!(field[index3(k, j, i, ny, nx)], 0.0, "level {k} perturbed");
                }
            }
        }
    }

    #[test]
    fn cm1_gfortran_stream_is_libgfortrans_sequence() {
        // gfortran 15.2 on Linux, a program that does what CM1 does:
        // CM1's init3d.F:427-440 seeding, then 20 random_number draws.  The
        // first is CM1's logged "rand-1 = 0.620773196".
        let want: [u32; 20] = [
            0x3F1EEAFE, 0x3F3585ED, 0x3F28FBEF, 0x3CE494E0, 0x3E44B9DC, 0x3EC8E15C, 0x3EF1B6DA,
            0x3F156CB9, 0x3F673895, 0x3E32F160, 0x3EAB3D5E, 0x3E203BA8, 0x3F582BD8, 0x3F240AB1,
            0x3F25CFDC, 0x3E275C88, 0x3E4E8964, 0x3ED4B5C4, 0x3F191EE6, 0x3D833530,
        ];
        let mut stream = Cm1RandomStream::cm1_gfortran();
        assert_eq!(stream.draws(), 1);
        for expected in want.iter().skip(1) {
            assert_eq!(stream.next_unit().to_bits(), *expected);
        }
    }

    #[test]
    fn unported_iinit_arms_name_the_wrong_result_they_prevent() {
        for (iinit, needle) in [(6_i32, "no else arm"), (99, "no else arm")] {
            let message = iinit_refusal(iinit).unwrap();
            assert!(message.contains(needle), "iinit={iinit}: {message}");
        }
        // init3d.F:579-647's moist benchmark is transcribed now.
        assert!(iinit_refusal(4).is_none());
    }
}
