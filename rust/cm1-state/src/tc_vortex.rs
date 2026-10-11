//! CM1 r21.1 `init3d.F:697-1256`: the `iinit=7` analytic tropical-cyclone vortex.
//!
//! The arm builds an axisymmetric vortex on a radial reference grid, balances
//! it (gradient wind for the pressure, hydrostatic for the temperature,
//! `init3d.F:895-966`, twenty passes), interpolates it to the Cartesian grid
//! from every periodic image (`init3d.F:995-1197`), adds the base-state wind
//! (`init3d.F:1206-1214`) and puts random theta noise of 0.1 K in the warm core
//! (`init3d.F:1239-1256`).  It sets `setppi = .false.`, so the initial pressure
//! perturbation is the vortex's own, not `ibalance`'s.
//!
//! Every name here is CM1's.  The arithmetic follows CM1's declarations:
//! `rref`, `vref`, `piref`, `thref`, `thvref`, `qvref` and `rr` are default
//! REAL; `tem1`, `tem2`, `temd` and `frac` are double precision
//! (`init3d.F:130-160`).

use crate::idealized_initialization::{invalid, rslf};
use crate::initial_perturbation::Cm1RandomStream;
use crate::Result;

const G: f32 = 9.81_f32;
const RD: f32 = 287.04_f32;
const RV: f32 = 461.5_f32;
const CP: f32 = 1005.7_f32;
const P00: f32 = 100_000.0_f32;
/// cm1.F:312, the single-precision build's `tsmall`.
const TSMALL: f32 = 0.0001_f32;

/// `init3d.F:704-731`: the vortex's named parameter block (CM1 edits it in
/// the source).  `official()` is CM1 r21.1's own block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TcVortexControls {
    /// 1 = Rotunno and Emanuel (1987), 2 = modified Rankine (CM1's default since r20.1).
    pub tctype: i32,
    pub vmax: f32,
    pub rm1: f32,
    pub rm2: f32,
    pub rm3: f32,
    pub rdc: f32,
    pub r0_re87: f32,
    pub rmax_re87: f32,
    pub vmax_re87: f32,
    /// Height (ASL) where V = 0.
    pub zdd: f32,
}

impl TcVortexControls {
    pub fn official() -> Self {
        Self {
            tctype: 2,
            vmax: 15.0,
            rm1: 75_000.0,
            rm2: 200_000.0,
            rm3: 500_000.0,
            rdc: -0.35,
            r0_re87: 412_500.0,
            rmax_re87: 82_500.0,
            vmax_re87: 15.0,
            zdd: 15_000.0,
        }
    }
}

/// The grid and base state the arm reads.
pub struct TcVortexInputs<'a> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// param.F's global `xhref(i)`, i = 1..nx+3 (the uniform extension past
    /// the east edge), as default REAL; index 0 is CM1's i = 1.
    pub xhref: &'a [f32],
    /// Mass-point and face coordinates of the physical domain, CM1's i = 1..
    pub xh: &'a [f32],
    pub yh: &'a [f32],
    pub xf: &'a [f32],
    pub yf: &'a [f32],
    /// `zh(1,1,k)`, k = 1..nk.
    pub zh: &'a [f32],
    /// `rdz` (param.F:6327) and `mf(1,1,k)` for k = 1..nk+1 (index 0 is k = 1).
    pub rdz: f32,
    pub mf: &'a [f32],
    /// Base state at column (1,1): th0, pi0, qv0, prs0, thv0 (k = 1..nk).
    pub th0: &'a [f32],
    pub pi0: &'a [f32],
    pub qv0: &'a [f32],
    pub prs0: &'a [f32],
    pub thv0: &'a [f32],
    pub fcor: f32,
    pub imoist: bool,
    pub iice: bool,
    /// `ebc = wbc = 1` and `sbc = nbc = 1`: the periodic images of init3d.F:998-1054.
    pub periodic_x: bool,
    pub periodic_y: bool,
    /// `centerx`, `centery`, `maxx-minx`, `maxy-miny`.
    pub xref: f32,
    pub yref: f32,
    pub span_x: f32,
    pub span_y: f32,
}

/// What the arm leaves in `tha`, `ppi`, `qa(nqv)`, `ua` and `va` before the
/// base wind is added: the vortex part of `ua`/`va` only.
#[derive(Clone, Debug, PartialEq)]
pub struct TcVortexFields {
    /// (nz, ny, nx)
    pub tha: Vec<f32>,
    pub ppi: Vec<f32>,
    pub qv: Vec<f32>,
    /// (nz, ny, nx+1) and (nz, ny+1, nx)
    pub ua: Vec<f32>,
    pub va: Vec<f32>,
}

fn rsif(p: f32, t: f32) -> f32 {
    // cm1libs.F:69-118, the Tao et al. (1989) branch every GM1 scheme takes.
    let eps = RD / RV;
    let esi = (611.2_f32 * (21.874_558_4_f32 * (t - 273.15_f32) / (t - 7.66_f32)).exp()).min(p * 0.5_f32);
    eps * esi / (p - esi)
}

/// init3d.F:1063-1082: the first reference radius not below `rr`.
fn bracket(rref: &[f32], rr: f32) -> Result<(usize, usize)> {
    let mut ii = 0usize; // 1-based in CM1; here the 0-based index + 1
    loop {
        ii += 1;
        if ii > rref.len() {
            return Err(invalid(format!(
                "init3d.F:1068-1076: radius {rr} is beyond the reference grid's {}; CM1 stops here",
                rref[rref.len() - 1]
            )));
        }
        let diff = rref[ii - 1] - rr;
        if diff >= 0.0 {
            break;
        }
    }
    if (rr - rref[ii - 1]).abs() < TSMALL && ii == 1 {
        ii = 2;
    }
    if ii < 2 {
        return Err(invalid(format!(
            "init3d.F:1083: radius {rr} is inside the first reference radius {}; CM1 would read rref(0)",
            rref[0]
        )));
    }
    Ok((ii - 2, ii - 1))
}

/// init3d.F:697-1256 on a 3-D Cartesian grid.  `stream` has taken
/// init3d.F:437's first draw; the arm's own draws follow (init3d.F:1245-1254).
#[allow(clippy::needless_range_loop, clippy::too_many_lines)]
pub fn build_tc_vortex(
    inputs: &TcVortexInputs<'_>,
    controls: TcVortexControls,
    stream: &mut Cm1RandomStream,
) -> Result<TcVortexFields> {
    let (nx, ny, nz) = (inputs.nx, inputs.ny, inputs.nz);
    let reps = RV / RD;
    let cpdrd = CP / RD;
    if inputs.xh.len() != nx
        || inputs.yh.len() != ny
        || inputs.xf.len() != nx + 1
        || inputs.yf.len() != ny + 1
        || inputs.zh.len() != nz
        || inputs.mf.len() != nz + 1
        || [inputs.th0, inputs.pi0, inputs.qv0, inputs.prs0, inputs.thv0]
            .iter()
            .any(|column| column.len() != nz)
    {
        return Err(invalid("iinit=7 vortex inputs do not match the grid"));
    }
    // init3d.F:763-771: nref = nx/2 + 3, rref(i) = xhref(nx/2 + i).
    let nref = nx / 2 + 3;
    if inputs.xhref.len() < nx / 2 + nref {
        return Err(invalid("iinit=7 needs xhref through nx/2 + nref"));
    }
    let rref: Vec<f32> = (1..=nref).map(|i| inputs.xhref[nx / 2 + i - 1]).collect();
    let rrmax = rref[nref - 1];

    // init3d.F:815-853: the tangential wind profile.
    let mut vt = vec![0.0_f32; nref];
    match controls.tctype {
        1 => {
            let c = controls;
            let dd2 = 2.0_f32 * c.rmax_re87 / (c.r0_re87 + c.rmax_re87);
            for i in 0..nref {
                if rref[i] < c.r0_re87 {
                    let dd1 = 2.0_f32 * c.rmax_re87 / (rref[i] + c.rmax_re87);
                    let q = rref[i] / c.rmax_re87;
                    vt[i] = ((c.vmax_re87 * c.vmax_re87) * (q * q)
                        * (dd1 * dd1 * dd1 - dd2 * dd2 * dd2)
                        + 0.25_f32 * inputs.fcor * inputs.fcor * rref[i] * rref[i])
                        .sqrt()
                        - 0.5_f32 * inputs.fcor * rref[i];
                }
            }
        }
        2 => {
            let c = controls;
            for i in 0..nref {
                let r = rref[i];
                vt[i] = if r < c.rm1 {
                    c.vmax * r / c.rm1
                } else if r < c.rm2 {
                    c.vmax * (r / c.rm1).powf(c.rdc)
                } else if r < c.rm3 {
                    let v2 = c.vmax * (r / c.rm1).powf(c.rdc);
                    let v3 = c.vmax * 0.5_f32 * (1.0_f32 - (r - c.rm2) / (c.rm3 - c.rm2));
                    let w3 = (r - c.rm2) / (c.rm3 - c.rm2);
                    let w2 = 1.0_f32 - w3;
                    w2 * v2 + w3 * v3
                } else {
                    0.0_f32
                };
            }
        }
        other => return Err(invalid(format!("init3d.F:813: tctype is 1 or 2, got {other}"))),
    }

    // Reference arrays, (nref, 0:nk+1) as [i][k+1].
    let row = nz + 2;
    let at = |i: usize, k: usize| i * row + k; // k is CM1's 0..nk+1
    let mut vref = vec![0.0_f32; nref * row];
    let mut piref = vec![0.0_f32; nref * row];
    let mut pref = vec![0.0_f32; nref * row];
    let mut thref = vec![0.0_f32; nref * row];
    let mut thvref = vec![0.0_f32; nref * row];
    let mut qvref = vec![0.0_f32; nref * row];
    let mut rhref = vec![0.0_f32; nref * row];
    // init3d.F:867-874: linear to zero at zdd.
    for k in 1..=nz {
        for i in 0..nref {
            let z = inputs.zh[k - 1];
            vref[at(i, k)] = if z < controls.zdd {
                vt[i] * (controls.zdd - z) / (controls.zdd - 0.0_f32)
            } else {
                0.0_f32
            };
        }
    }
    // init3d.F:876-885 (rhmods = .false.).
    for k in 1..=nz {
        let rh = inputs.qv0[k - 1] / rslf(inputs.prs0[k - 1], inputs.th0[k - 1] * inputs.pi0[k - 1]);
        for i in 0..nref {
            rhref[at(i, k)] = rh;
            thref[at(i, k)] = 0.0;
            piref[at(i, k)] = 0.0;
            pref[at(i, k)] = inputs.prs0[k - 1];
        }
    }
    // init3d.F:895-966: twenty passes of gradient-wind and hydrostatic balance.
    for _ in 0..20 {
        for k in 1..=nz {
            for i in 0..nref {
                let p = at(i, k);
                pref[p] = P00 * (inputs.pi0[k - 1] + piref[p]).powf(cpdrd);
                if inputs.imoist {
                    let tnew = (inputs.th0[k - 1] + thref[p]) * (inputs.pi0[k - 1] + piref[p]);
                    let qvs = if inputs.iice {
                        let fliq = ((tnew - 233.15_f32) / (273.15_f32 - 233.15_f32)).min(1.0).max(0.0);
                        let fice = 1.0_f32 - fliq;
                        let qsw = if tnew > 233.15_f32 { fliq * rslf(pref[p], tnew) } else { 0.0 };
                        let qsi = if tnew < 273.15_f32 { fice * rsif(pref[p], tnew) } else { 0.0 };
                        qsw + qsi
                    } else {
                        rslf(pref[p], tnew)
                    };
                    qvref[p] = rhref[p] * qvs;
                }
                thvref[p] = (inputs.th0[k - 1] + thref[p]) * (1.0_f32 + reps * qvref[p]) / (1.0_f32 + qvref[p]);
            }
        }
        for k in 1..=nz {
            piref[at(nref - 1, k)] = 0.0;
            for i in (1..nref).rev() {
                let (a, b) = (at(i, k), at(i - 1, k));
                piref[b] = piref[a]
                    + (rref[i - 1] - rref[i]) / (CP * 0.5_f32 * (thvref[b] + thvref[a]))
                        * 0.5_f32
                        * (vref[a] * vref[a] / rref[i]
                            + vref[b] * vref[b] / rref[i - 1]
                            + inputs.fcor * (vref[a] + vref[b]));
            }
        }
        for i in 0..nref {
            piref[at(i, 0)] = piref[at(i, 1)];
            piref[at(i, nz + 1)] = piref[at(i, nz)];
        }
        for k in 2..=nz {
            for i in 0..nref {
                let (p, up, dn) = (at(i, k), at(i, k + 1), at(i, k - 1));
                // mf(1,1,k+1) is index k, mf(1,1,k) index k-1.
                let upper = CP * 0.5_f32 * (thvref[p] + thvref[up]) * (piref[up] - piref[p]) * inputs.rdz * inputs.mf[k];
                let lower = CP * 0.5_f32 * (thvref[p] + thvref[dn]) * (piref[p] - piref[dn]) * inputs.rdz * inputs.mf[k - 1];
                let th = 0.5_f32 * (upper + lower) * inputs.thv0[k - 1] / G;
                thref[p] = (inputs.thv0[k - 1] + th) * (1.0_f32 + qvref[p]) / (1.0_f32 + reps * qvref[p])
                    - inputs.th0[k - 1];
            }
        }
        for i in 0..nref {
            let (p, up) = (at(i, 1), at(i, 2));
            let th = (CP * 0.5_f32 * (thvref[p] + thvref[up]) * (piref[up] - piref[p]) * inputs.rdz * inputs.mf[1])
                * inputs.thv0[0]
                / G;
            thref[p] = (inputs.thv0[0] + th) * (1.0_f32 + qvref[p]) / (1.0_f32 + reps * qvref[p]) - inputs.th0[0];
        }
    }

    // init3d.F:990-1197: interpolate from every image the boundaries admit.
    let mut out = TcVortexFields {
        tha: vec![0.0; nz * ny * nx],
        ppi: vec![0.0; nz * ny * nx],
        qv: vec![0.0; nz * ny * nx],
        ua: vec![0.0; nz * ny * (nx + 1)],
        va: vec![0.0; nz * (ny + 1) * nx],
    };
    let (sx, sy) = (inputs.span_x, inputs.span_y);
    let both = inputs.periodic_x && inputs.periodic_y;
    let images: [(bool, f32, f32); 9] = [
        (true, 0.0, 0.0),
        (both, sx, sy),
        (both, -sx, -sy),
        (both, sx, -sy),
        (both, -sx, sy),
        (inputs.periodic_x, sx, 0.0),
        (inputs.periodic_x, -sx, 0.0),
        (inputs.periodic_y, 0.0, sy),
        (inputs.periodic_y, 0.0, -sy),
    ];
    let radius = |x: f32, y: f32, xref: f32, yref: f32| -> (f64, f64, f32) {
        let tem1 = f64::from(x - xref);
        let tem2 = f64::from(y - yref);
        (tem1, tem2, (tem1 * tem1 + tem2 * tem2).sqrt() as f32)
    };
    for (nn, (doit, ox, oy)) in images.iter().enumerate() {
        if !*doit {
            continue;
        }
        let (xref, yref) = (inputs.xref + ox, inputs.yref + oy);
        for k in 1..=nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (_, _, rr) = radius(inputs.xh[i], inputs.yh[j], xref, yref);
                    let o = ((k - 1) * ny + j) * nx + i;
                    if rr < rrmax {
                        let (i1, i2) = bracket(&rref, rr)?;
                        let frac = f64::from((rr - rref[i1]) / (rref[i2] - rref[i1]));
                        let lerp = |a: &[f32]| -> f32 {
                            (f64::from(a[at(i1, k)]) + f64::from(a[at(i2, k)] - a[at(i1, k)]) * frac) as f32
                        };
                        out.ppi[o] = lerp(&piref);
                        out.tha[o] = lerp(&thref);
                        if inputs.imoist {
                            out.qv[o] = lerp(&qvref);
                        }
                    } else if nn == 0 {
                        out.tha[o] = thref[at(nref - 1, k)];
                        out.ppi[o] = piref[at(nref - 1, k)];
                        out.qv[o] = qvref[at(nref - 1, k)];
                    }
                }
            }
            for j in 0..ny {
                for i in 0..=nx {
                    let (tem1, tem2, rr) = radius(inputs.xf[i], inputs.yh[j], xref, yref);
                    let o = ((k - 1) * ny + j) * (nx + 1) + i;
                    if rr < TSMALL {
                        out.ua[o] = 0.0;
                    } else if rr < rrmax {
                        let (i1, i2) = bracket(&rref, rr)?;
                        let frac = f64::from((rr - rref[i1]) / (rref[i2] - rref[i1]));
                        let temd = -tem2.atan2(tem1).sin();
                        let v = f64::from(vref[at(i1, k)]) + f64::from(vref[at(i2, k)] - vref[at(i1, k)]) * frac;
                        out.ua[o] = (f64::from(out.ua[o]) + v * temd) as f32;
                    }
                }
            }
            for j in 0..=ny {
                for i in 0..nx {
                    let (tem1, tem2, rr) = radius(inputs.xh[i], inputs.yf[j], xref, yref);
                    let o = ((k - 1) * (ny + 1) + j) * nx + i;
                    if rr < TSMALL {
                        out.va[o] = 0.0;
                    } else if rr < rrmax {
                        let (i1, i2) = bracket(&rref, rr)?;
                        let frac = f64::from((rr - rref[i1]) / (rref[i2] - rref[i1]));
                        let temd = tem2.atan2(tem1).cos();
                        let v = f64::from(vref[at(i1, k)]) + f64::from(vref[at(i2, k)] - vref[at(i1, k)]) * frac;
                        out.va[o] = (f64::from(out.va[o]) + v * temd) as f32;
                    }
                }
            }
        }
    }

    // init3d.F:1236-1256: 0.1 K of random noise in the warm core, drawn over
    // the whole domain in (k, jj, ii) order.
    if nx > 3 && ny > 3 {
        let amplitude = 0.1_f32;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let rand = stream.next_unit();
                    let o = (k * ny + j) * nx + i;
                    if out.tha[o] >= 0.1_f32 {
                        out.tha[o] += amplitude * (2.0_f32 * rand - 1.0_f32);
                    }
                }
            }
        }
    }
    Ok(out)
}
