//! CM1 r21.1 `ibalance = 2`: the initial pressure perturbation as the
//! anelastic buoyancy pressure (`init3d.F:1787-1917`).
//!
//! `init3d.F:1802-1878` forms the buoyancy `g*(thv/thv0 - 1)` of the
//! initialized state, averages it to the w faces, and hands it to
//! `poiss.F:10-220` with zero horizontal forcing and zero `divx`.  `poiss`
//! Fourier transforms each level on the (mirrored, `param.F:5178-5205`)
//! horizontal rectangle, solves the vertical tridiagonal system of
//! `init3d.F:1668-1692`'s `psolver <= 3` coefficients mode by mode, replaces
//! the singular (1,1) mode by its bottom-gauged upward recurrence
//! (`poiss.F:176-185`), and transforms back.
//!
//! Every step is CM1's single-precision statement order, the transform
//! included: `crate::singleton_fft` is `singleton.F` transcribed.
//! `init3d.F:1823-1827` has no MPI arm and stops CM1 for horizontal
//! stretching; this lane has neither.

use crate::idealized_initialization::{invalid, CP, G, RD, RV};
use crate::Result;

/// Everything `init3d.F:1787-1917` and `poiss.F` read, on one flat grid.
///
/// Columns are CM1's `k = 1..nk` (mass) and `k = 1..nk+1` (faces); 3-D fields
/// are physical `[k,j,i]`.  `qt` is `init3d.F:1839-1847`'s condensate sum.
pub struct AnelasticBalanceInputs<'a> {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f32,
    pub dz: f32,
    pub imirror: bool,
    pub jmirror: bool,
    pub moist: bool,
    pub mh: &'a [f32],
    pub mf: &'a [f32],
    pub rho0: &'a [f32],
    pub rf0: &'a [f32],
    pub thv0: &'a [f32],
    pub th0: &'a [f32],
    pub tha: &'a [f32],
    pub qv: &'a [f32],
    pub qt: &'a [f32],
}

use crate::singleton_fft::{fft2d, C32};

/// `init3d.F:1787-1917` then `poiss.F:10-220`: the physical `[k,j,i]` ppi.
#[allow(clippy::needless_range_loop)]
pub fn anelastic_balance_ppi(inputs: &AnelasticBalanceInputs<'_>) -> Result<Vec<f32>> {
    let (nx, ny, nz) = (inputs.nx, inputs.ny, inputs.nz);
    let words = nx * ny * nz;
    if nz < 3 || nx == 0 || ny == 0 {
        return Err(invalid("ibalance=2 needs a grid of at least three levels"));
    }
    for (label, values, n) in [
        ("mh", inputs.mh, nz),
        ("mf", inputs.mf, nz + 1),
        ("rho0", inputs.rho0, nz),
        ("rf0", inputs.rf0, nz + 1),
        ("thv0", inputs.thv0, nz),
        ("th0", inputs.th0, nz),
        ("tha", inputs.tha, words),
        ("qv", inputs.qv, words),
        ("qt", inputs.qt, words),
    ] {
        if values.len() != n {
            return Err(invalid(format!("ibalance=2 {label} has {} words, expected {n}", values.len())));
        }
    }
    let reps = RV / RD;
    let (dx, dz) = (inputs.dx, inputs.dz);
    let rdz = 1.0_f32 / dz;
    let idx = |k: usize, j: usize, i: usize| (k * ny + j) * nx + i;

    // init3d.F:1829-1872: thv, then dum4 = g*(thv/thv0-1) with odd-reflected
    // ghosts, then wten on the faces.
    let mut dum4 = vec![0.0_f32; (nz + 2) * ny * nx];
    let at = |k: usize, j: usize, i: usize| (k * ny + j) * nx + i;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = idx(k, j, i);
                let thv = if inputs.moist {
                    (inputs.th0[k] + inputs.tha[c]) * (1.0_f32 + reps * inputs.qv[c])
                        / (1.0_f32 + inputs.qv[c] + inputs.qt[c])
                } else {
                    inputs.th0[k] + inputs.tha[c]
                };
                dum4[at(k + 1, j, i)] = G * (thv / inputs.thv0[k] - 1.0_f32);
            }
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            dum4[at(0, j, i)] = -dum4[at(1, j, i)];
            dum4[at(nz + 1, j, i)] = -dum4[at(nz, j, i)];
        }
    }
    // wten(k) for CM1 k = 1..nk+1, stored at index k-1.
    let mut wten = vec![0.0_f32; (nz + 1) * ny * nx];
    for k in 0..=nz {
        for j in 0..ny {
            for i in 0..nx {
                wten[at(k, j, i)] = 0.5_f32 * (dum4[at(k, j, i)] + dum4[at(k + 1, j, i)]);
            }
        }
    }

    // init3d.F:1668-1692 (psolver <= 3).
    let dz2 = dz * dz;
    let mut cfa = vec![0.0_f32; nz];
    let mut cfc = vec![0.0_f32; nz];
    let mut d1 = vec![0.0_f32; nz];
    for k in 0..nz {
        if k > 0 {
            cfa[k] = inputs.mh[k] * inputs.mf[k] * inputs.rf0[k] * 0.5_f32
                * (inputs.thv0[k - 1] + inputs.thv0[k])
                / (dz2 * inputs.rho0[k] * inputs.thv0[k]);
        }
        if k + 1 < nz {
            cfc[k] = inputs.mh[k] * inputs.mf[k + 1] * inputs.rf0[k + 1] * 0.5_f32
                * (inputs.thv0[k] + inputs.thv0[k + 1])
                / (dz2 * inputs.rho0[k] * inputs.thv0[k]);
        }
        d1[k] = 1.0_f32 / (CP * inputs.rho0[k] * inputs.thv0[k]);
    }
    let ipe = if inputs.imirror { 2 * nx } else { nx };
    let jpe = if inputs.jmirror { 2 * ny } else { ny };
    let dpi = 4.0_f64 * 1.0_f64.atan();
    let dxdx = f64::from(dx * dx);
    let mut cfb = vec![0.0_f32; nz * jpe * ipe];
    for k in 0..nz {
        for j in 0..jpe {
            for i in 0..ipe {
                let mode = 2.0_f64
                    * ((2.0 * dpi * i as f64 / ipe as f64).cos()
                        + (2.0 * dpi * j as f64 / jpe as f64).cos()
                        - 2.0)
                    / dxdx;
                cfb[(k * jpe + j) * ipe + i] = (mode - f64::from(cfa[k]) - f64::from(cfc[k])) as f32;
            }
        }
    }

    // poiss.F:62-116: the forcing of each level, mirrored, transformed.
    let plane = ipe * jpe;
    let mut trans = vec![C32::default(); nz * plane];
    for k in 0..nz {
        let level = &mut trans[k * plane..(k + 1) * plane];
        for j in 0..ny {
            for i in 0..nx {
                // rho0*((uten..)*rdx*uh+(vten..)*rdy*vh) is exact +0 with zero
                // horizontal forcing, and divx*(1/dtl) is exact +0 too.
                let vertical = (inputs.rf0[k + 1] * wten[at(k + 1, j, i)]
                    - inputs.rf0[k] * wten[at(k, j, i)])
                    * rdz
                    * inputs.mh[k];
                let def = 0.0_f32 + vertical + 0.0_f32;
                // rhs(i,j) = cmplx(def*d1(k), 0.0), column-major (i fastest).
                let value = C32::new(def * d1[k], 0.0);
                level[j * ipe + i] = value;
                if inputs.imirror {
                    level[j * ipe + (ipe - 1 - i)] = value;
                }
                if inputs.jmirror {
                    level[(jpe - 1 - j) * ipe + i] = value;
                }
                if inputs.imirror && inputs.jmirror {
                    level[(jpe - 1 - j) * ipe + (ipe - 1 - i)] = value;
                }
            }
        }
        // poiss.F:116: trans = fft(rhs).
        fft2d(level, ipe, jpe, false);
    }

    // poiss.F:118-185 in default REAL.  lgbth is COMPLEX in CM1 but only
    // ever holds a real word (-cfc*tem), so cfa*lgbth+cfb and 1.0/(...) are
    // the real operations gfortran's complex arithmetic reduces to.
    let mut pdt = vec![(0.0_f32, 0.0_f32); nz * plane];
    let mut lgbth = vec![0.0_f32; nz * plane];
    let mut lgbph = vec![(0.0_f32, 0.0_f32); nz * plane];
    for m in 0..plane {
        let t0 = trans[m];
        let tem = 1.0_f32 / cfb[m];
        lgbth[m] = -cfc[0] * tem;
        lgbph[m] = (t0.re * tem, t0.im * tem);
        for k in 1..nz {
            let at_k = k * plane + m;
            let t = trans[at_k];
            let temc = 1.0_f32 / (cfa[k] * lgbth[(k - 1) * plane + m] + cfb[at_k]);
            lgbth[at_k] = -cfc[k] * temc;
            let previous = lgbph[(k - 1) * plane + m];
            lgbph[at_k] = ((t.re - cfa[k] * previous.0) * temc, (t.im - cfa[k] * previous.1) * temc);
        }
        pdt[(nz - 1) * plane + m] = lgbph[(nz - 1) * plane + m];
        for k in (0..nz - 1).rev() {
            let above = pdt[(k + 1) * plane + m];
            let (th, ph) = (lgbth[k * plane + m], lgbph[k * plane + m]);
            pdt[k * plane + m] = (th * above.0 + ph.0, th * above.1 + ph.1);
        }
    }
    // poiss.F:176-185: the (1,1) mode from its bottom gauge upward.
    let deft2: Vec<f32> = (0..nz).map(|k| trans[k * plane].re).collect();
    let mut r = vec![0.0_f32; nz + 1];
    pdt[0] = (0.0, 0.0);
    r[1] = (deft2[0] - cfb[0] * r[0]) / cfc[0];
    pdt[plane] = (r[1], 0.0);
    for k in 1..nz - 1 {
        r[k + 1] = (deft2[k] - cfa[k] * r[k - 1] - cfb[k * plane] * r[k]) / cfc[k];
        pdt[(k + 1) * plane] = (r[k + 1], 0.0);
    }

    // poiss.F:187-213: the inverse transform, the real part times d2 (= 1).
    let mut ppi = vec![0.0_f32; words];
    let mut level = vec![C32::default(); plane];
    for k in 0..nz {
        for m in 0..plane {
            let (re, im) = pdt[k * plane + m];
            level[m] = C32::new(re, im);
        }
        fft2d(&mut level, ipe, jpe, true);
        for j in 0..ny {
            for i in 0..nx {
                ppi[idx(k, j, i)] = level[j * ipe + i].re * 1.0_f32;
            }
        }
    }
    if ppi.iter().any(|value| !value.is_finite()) {
        return Err(invalid("ibalance=2 produced a non-finite pressure perturbation"));
    }
    Ok(ppi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_uniform_state_has_no_buoyancy_pressure() {
        let (nx, ny, nz) = (6, 5, 8);
        let ones = |n: usize| vec![1.0_f32; n];
        let th0: Vec<f32> = (0..nz).map(|k| 300.0 + k as f32).collect();
        let thv0 = th0.clone();
        let zero = vec![0.0_f32; nx * ny * nz];
        let rho0: Vec<f32> = (0..nz).map(|k| 1.1 - 0.05 * k as f32).collect();
        let rf0: Vec<f32> = (0..=nz).map(|k| 1.1 - 0.05 * k as f32 + 0.025).collect();
        let inputs = AnelasticBalanceInputs {
            nx, ny, nz, dx: 1000.0, dz: 500.0, imirror: true, jmirror: true, moist: false,
            mh: &ones(nz), mf: &ones(nz + 1), rho0: &rho0, rf0: &rf0, thv0: &thv0, th0: &th0,
            tha: &zero, qv: &zero, qt: &zero,
        };
        let ppi = anelastic_balance_ppi(&inputs).unwrap();
        assert!(ppi.iter().all(|value| value.abs() < 1e-9), "{:?}", &ppi[..4]);
    }
}
