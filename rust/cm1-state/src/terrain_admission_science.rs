//! Source-ordered CM1 r21.1 terrain/base and cold-start halo transforms.
//!
//! This module deliberately contains no filesystem or publication code.  The
//! companion `terrain_admission` module owns all strict envelopes and hashes.

use crate::terrain::{HorizontalCrop, TerrainStaticFrame};
use crate::{Result, StateError};

pub(crate) const G: f32 = 9.81;
pub(crate) const RD: f32 = 287.04;
pub(crate) const RV: f32 = 461.5;
pub(crate) const CP: f32 = 1005.7;
pub(crate) const CPV: f32 = 1870.0;
pub(crate) const CPL: f32 = 4190.0;
pub(crate) const CPI: f32 = 2106.0;
pub(crate) const P00: f32 = 100_000.0;
pub(crate) const NGXY: usize = 3;

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

fn checked_product(shape: &[usize], label: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("{label} shape product overflows")))
    })
}

#[inline]
fn index3(k: usize, j: usize, i: usize, ny: usize, nx: usize) -> usize {
    (k * ny + j) * nx + i
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RecoveryMetric {
    pub mismatch_count: u64,
    pub max_ulp: u32,
    pub max_abs_f32_bits: u32,
}

impl RecoveryMetric {
    fn observe(&mut self, expected: f32, actual: f32) {
        if expected.to_bits() != actual.to_bits() {
            self.mismatch_count += 1;
        }
        self.max_ulp = self.max_ulp.max(ulp_distance(expected, actual));
        let absolute = (actual - expected).abs();
        let current = f32::from_bits(self.max_abs_f32_bits);
        if absolute > current {
            self.max_abs_f32_bits = absolute.to_bits();
        }
    }
}

fn ordered_f32(bits: u32) -> u32 {
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

fn ulp_distance(left: f32, right: f32) -> u32 {
    ordered_f32(left.to_bits()).abs_diff(ordered_f32(right.to_bits()))
}

#[derive(Clone, Debug)]
pub(crate) struct TerrainBaseFrame {
    /// Source allocation order `[nk+2,nj+6,ni+6]`, with native indices
    /// `k=0..nk+1,j=-2..nj+3,i=-2..ni+3`.  These are not generic padded
    /// arrays: every horizontal word follows the open `bcs` edge law and
    /// every vertical ghost follows `extrapbcs` before the source-ordered
    /// diagnostic rebuild.
    pub pi0: Vec<f32>,
    pub prs0: Vec<f32>,
    pub th0: Vec<f32>,
    pub thv0: Vec<f32>,
    pub rth0: Vec<f32>,
    pub qv0: Vec<f32>,
    pub rho0: Vec<f32>,
    pub rr0: Vec<f32>,
    /// Source allocation order `[nk+2,nj+6,ni+6]`; native face `k=0` is the
    /// exact duplicate of `k=1` required by `base.F:2179`.
    pub rf0: Vec<f32>,
    pub rrf0: Vec<f32>,
    /// Source surface-base allocations `[nj+6,ni+6]`.
    pub rho0s: Vec<f32>,
    pub pi0s: Vec<f32>,
    pub prs0s: Vec<f32>,
    pub rth0s: Vec<f32>,
}

struct Sounding {
    z: Vec<f32>,
    theta: Vec<f32>,
    pressure: Vec<f32>,
    temperature: Vec<f32>,
    relative_humidity: Vec<f32>,
    pi_surface: f32,
    theta_v_surface: f32,
}

/// CM1 liquid saturation mixing ratio, `cm1libs.F:16-62`, for the admitted
/// ptype=5 family.  Rust's binary32 exponential is not claimed to be raw-bit
/// identical to every Fortran libm; the publication records the resulting
/// recovery metrics rather than asserting an impossible compiler-wide bit pin.
fn rslf(pressure: f32, temperature: f32) -> f32 {
    let eps = RD / RV;
    let exponent = 17.67_f32 * (temperature - 273.15_f32) / (temperature - 29.65_f32);
    let vapor_pressure = (611.2_f32 * exponent.exp()).min(pressure * 0.5_f32);
    eps * vapor_pressure / (pressure - vapor_pressure)
}

fn extrapolate_bottom(values: &[f32], weights: [f32; 3]) -> f32 {
    weights[0] * values[0] + weights[1] * values[1] + weights[2] * values[2]
}

fn extrapolate_top(values: &[f32], weights: [f32; 3]) -> f32 {
    let n = values.len();
    weights[0] * values[n - 1] + weights[1] * values[n - 2] + weights[2] * values[n - 3]
}

fn build_sounding(
    pi_profile: &[f32],
    theta_profile: &[f32],
    qv_profile: &[f32],
    dz: f32,
    bottom: [f32; 3],
    top: [f32; 3],
) -> Result<Sounding> {
    let nz = pi_profile.len();
    if nz < 4 || theta_profile.len() != nz || qv_profile.len() != nz {
        return Err(invalid(
            "terrain hydrostatic base requires matching profiles with at least four levels",
        ));
    }
    let pi_surface = extrapolate_bottom(pi_profile, bottom);
    let theta_surface = extrapolate_bottom(theta_profile, bottom);
    let qv_surface = extrapolate_bottom(qv_profile, bottom);
    if !pi_surface.is_finite()
        || pi_surface <= 0.0
        || !theta_surface.is_finite()
        || theta_surface <= 0.0
        || !qv_surface.is_finite()
        || qv_surface < 0.0
    {
        return Err(invalid(
            "Fornberg-derived surface sounding values are not positive finite physical values",
        ));
    }
    let theta_top = extrapolate_top(theta_profile, top);
    let qv_top = extrapolate_top(qv_profile, top);
    if !theta_top.is_finite() || theta_top <= 0.0 || !qv_top.is_finite() || qv_top < 0.0 {
        return Err(invalid(
            "Fornberg-derived top sounding values are not positive finite physical values",
        ));
    }

    let mut z = Vec::with_capacity(nz + 2);
    let mut theta = Vec::with_capacity(nz + 2);
    let mut qv = Vec::with_capacity(nz + 2);
    z.push(0.0);
    theta.push(theta_surface);
    qv.push(qv_surface);
    for k in 0..nz {
        z.push((k as f32 + 0.5_f32) * dz);
        theta.push(theta_profile[k]);
        qv.push(qv_profile[k]);
    }
    z.push(nz as f32 * dz);
    theta.push(theta_top);
    qv.push(qv_top);

    let reps = RV / RD;
    let mut pi = vec![0.0_f32; z.len()];
    let mut pressure = vec![0.0_f32; z.len()];
    let mut temperature = vec![0.0_f32; z.len()];
    let mut relative_humidity = vec![0.0_f32; z.len()];
    let theta_v_surface = theta_surface * (1.0_f32 + reps * qv_surface) / (1.0_f32 + qv_surface);
    pi[0] = pi_surface;
    for k in 1..z.len() {
        let previous_theta_v = theta[k - 1] * (1.0_f32 + reps * qv[k - 1]) / (1.0_f32 + qv[k - 1]);
        let theta_v = theta[k] * (1.0_f32 + reps * qv[k]) / (1.0_f32 + qv[k]);
        pi[k] = pi[k - 1] - G * (z[k] - z[k - 1]) / (CP * 0.5_f32 * (theta_v + previous_theta_v));
    }
    for k in 0..z.len() {
        pressure[k] = P00 * pi[k].powf(CP / RD);
        temperature[k] = theta[k] * pi[k];
        let saturation = rslf(pressure[k], temperature[k]);
        relative_humidity[k] = qv[k] / saturation;
        if !pressure[k].is_finite()
            || pressure[k] <= 0.0
            || !temperature[k].is_finite()
            || temperature[k] <= 0.0
            || !relative_humidity[k].is_finite()
            || relative_humidity[k] < 0.0
        {
            return Err(invalid(format!(
                "derived sounding is non-physical at index {k}"
            )));
        }
    }
    Ok(Sounding {
        z,
        theta,
        pressure,
        temperature,
        relative_humidity,
        pi_surface,
        theta_v_surface,
    })
}

fn interpolate(nodes: &[f32], values: &[f32], target: f32) -> Result<f32> {
    if nodes.len() != values.len() || nodes.len() < 2 || !target.is_finite() {
        return Err(invalid("invalid terrain-base interpolation operands"));
    }
    if target < nodes[0] || target > *nodes.last().expect("nonempty nodes") {
        return Err(invalid(format!(
            "terrain-base target height {target} lies outside sounding [{},{}]",
            nodes[0],
            nodes[nodes.len() - 1]
        )));
    }
    let mut upper = 1usize;
    while nodes[upper] < target {
        upper += 1;
    }
    let lower = upper - 1;
    let fraction = (target - nodes[lower]) / (nodes[upper] - nodes[lower]);
    Ok(values[lower] + (values[upper] - values[lower]) * fraction)
}

struct ColumnBase {
    pi: Vec<f32>,
    theta: Vec<f32>,
    qv: Vec<f32>,
}

fn column_base(sounding: &Sounding, heights: &[f32]) -> Result<ColumnBase> {
    let nz = heights.len();
    let reps = RV / RD;
    let mut theta = vec![0.0_f32; nz];
    let mut qv = vec![0.0_f32; nz];
    let mut theta_v = vec![0.0_f32; nz];
    for k in 0..nz {
        theta[k] = interpolate(&sounding.z, &sounding.theta, heights[k])?;
        let pressure_interp = interpolate(&sounding.z, &sounding.pressure, heights[k])?;
        let temperature_interp = interpolate(&sounding.z, &sounding.temperature, heights[k])?;
        let rh_interp = interpolate(&sounding.z, &sounding.relative_humidity, heights[k])?;
        qv[k] = rh_interp * rslf(pressure_interp, temperature_interp);
        theta_v[k] = theta[k] * (1.0_f32 + reps * qv[k]) / (1.0_f32 + qv[k]);
    }
    let mut pi = vec![0.0_f32; nz];
    pi[0] = sounding.pi_surface
        - G * heights[0] / (CP * 0.5_f32 * (sounding.theta_v_surface + theta_v[0]));
    for k in 1..nz {
        pi[k] = pi[k - 1]
            - G * (heights[k] - heights[k - 1]) / (CP * 0.5_f32 * (theta_v[k] + theta_v[k - 1]));
    }
    for k in 0..nz {
        if !pi[k].is_finite()
            || pi[k] <= 0.0
            || !theta[k].is_finite()
            || theta[k] <= 0.0
            || !qv[k].is_finite()
            || qv[k] < 0.0
        {
            return Err(invalid(format!(
                "terrain hydrostatic column is non-physical at level {k}"
            )));
        }
    }
    Ok(ColumnBase { pi, theta, qv })
}

/// Reconstruct the admitted terrain-varying hydrostatic base.  The profile is
/// interpreted exactly like an external CM1 sounding at the uniform sigma mass
/// levels, with source Fornberg surface/top endpoints and `base.F:625-749`
/// statement ordering.
#[allow(clippy::too_many_arguments)] // Mirrors the independent CM1 source carriers explicitly.
pub(crate) fn compute_terrain_base(
    terrain: &[f32],
    source_shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    crop: HorizontalCrop,
    pi_profile: &[f32],
    theta_profile: &[f32],
    qv_profile: &[f32],
    terrain_frame: &TerrainStaticFrame,
) -> Result<TerrainBaseFrame> {
    let [nz, source_ny, source_nx] = source_shape_zyx;
    if terrain.len() != source_ny * source_nx
        || crop.ny == 0
        || crop.nx == 0
        || crop.j_start + crop.ny > source_ny
        || crop.i_start + crop.nx > source_nx
    {
        return Err(invalid(
            "terrain hydrostatic base source/crop geometry is invalid",
        ));
    }
    if terrain_frame.source_shape_zyx() != source_shape_zyx
        || terrain_frame.crop() != crop
        || terrain_frame.spacing_m().map(f64::to_bits) != spacing_m.map(f64::to_bits)
    {
        return Err(invalid(
            "terrain hydrostatic base and terrain-static frame identities differ",
        ));
    }
    let dz = spacing_m[2] as f32;
    let top_height = nz as f32 * dz;
    let sounding = build_sounding(
        pi_profile,
        theta_profile,
        qv_profile,
        dz,
        terrain_frame.bottom_fornberg(),
        terrain_frame.top_fornberg(),
    )?;
    let halo_ny = crop.ny + 2 * NGXY;
    let halo_nx = crop.nx + 2 * NGXY;
    let mass_shape = [nz + 2, halo_ny, halo_nx];
    let mass_words = checked_product(&mass_shape, "terrain base source allocation")?;
    let surface_words = checked_product(&[halo_ny, halo_nx], "terrain base surface")?;
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
    let mut rho0s = vec![0.0_f32; surface_words];
    let mut pi0s = vec![0.0_f32; surface_words];
    let mut prs0s = vec![0.0_f32; surface_words];
    let mut rth0s = vec![0.0_f32; surface_words];
    let c1 = terrain_frame
        .carrier("c1")
        .ok_or_else(|| invalid("terrain frame lacks c1"))?
        .values();
    let c2 = terrain_frame
        .carrier("c2")
        .ok_or_else(|| invalid("terrain frame lacks c2"))?
        .values();
    if c1.len() != (nz + 1) * crop.ny * crop.nx || c2.len() != (nz + 1) * crop.ny * crop.nx {
        return Err(invalid("terrain frame c1/c2 geometry is invalid"));
    }
    let bottom = terrain_frame.bottom_fornberg();
    let top = terrain_frame.top_fornberg();
    let reps = RV / RD;

    // base.F computes over ib:ie,jb:je, not only over the physical domain.
    // For the single-domain open tuple, bc2d makes those terrain coordinates
    // exact edge copies before base is called.  Clamping here is therefore the
    // source operation, not a generic padding policy.
    for out_j_halo in 0..halo_ny {
        let native_j = out_j_halo as isize + 1 - NGXY as isize;
        let global_j =
            (crop.j_start as isize + native_j - 1).clamp(0, source_ny as isize - 1) as usize;
        for out_i_halo in 0..halo_nx {
            let native_i = out_i_halo as isize + 1 - NGXY as isize;
            let global_i =
                (crop.i_start as isize + native_i - 1).clamp(0, source_nx as isize - 1) as usize;
            let surface = terrain[global_j * source_nx + global_i];
            let heights = (0..nz)
                .map(|k| {
                    let sigma = (k as f32 + 0.5_f32) * dz;
                    surface + sigma * (top_height - surface) / top_height
                })
                .collect::<Vec<_>>();
            let column = column_base(&sounding, &heights)?;
            for k in 0..nz {
                let p = index3(k + 1, out_j_halo, out_i_halo, halo_ny, halo_nx);
                pi0[p] = column.pi[k];
                th0[p] = column.theta[k];
                qv0[p] = column.qv[k];
            }

            // extrapbcs is applied independently to pi0/th0/qv0 before CM1
            // rebuilds thv0, prs0, rho0, rth0 and rr0 over kb:ke.
            for array in [&mut pi0, &mut th0, &mut qv0] {
                let k1 = array[index3(1, out_j_halo, out_i_halo, halo_ny, halo_nx)];
                let k2 = array[index3(2, out_j_halo, out_i_halo, halo_ny, halo_nx)];
                let k3 = array[index3(3, out_j_halo, out_i_halo, halo_ny, halo_nx)];
                let kn = array[index3(nz, out_j_halo, out_i_halo, halo_ny, halo_nx)];
                let kn1 = array[index3(nz - 1, out_j_halo, out_i_halo, halo_ny, halo_nx)];
                let kn2 = array[index3(nz - 2, out_j_halo, out_i_halo, halo_ny, halo_nx)];
                array[index3(0, out_j_halo, out_i_halo, halo_ny, halo_nx)] =
                    bottom[0] * k1 + bottom[1] * k2 + bottom[2] * k3;
                array[index3(nz + 1, out_j_halo, out_i_halo, halo_ny, halo_nx)] =
                    top[0] * kn + top[1] * kn1 + top[2] * kn2;
            }

            for k in 0..=nz + 1 {
                let p = index3(k, out_j_halo, out_i_halo, halo_ny, halo_nx);
                let qv = qv0[p];
                let qv_nonnegative = qv.max(0.0);
                thv0[p] = (th0[p] + 0.0_f32) * (1.0_f32 + reps * qv_nonnegative)
                    / (1.0_f32 + qv_nonnegative);
                prs0[p] = P00 * pi0[p].powf(CP / RD);
                rho0[p] = prs0[p] / (RD * th0[p] * pi0[p] * (1.0_f32 + qv * reps));
                rth0[p] = 1.0_f32 / th0[p];
                rr0[p] = 1.0_f32 / rho0[p];
                if !thv0[p].is_finite()
                    || !prs0[p].is_finite()
                    || prs0[p] <= 0.0
                    || !rho0[p].is_finite()
                    || rho0[p] <= 0.0
                {
                    return Err(invalid(format!(
                        "terrain base diagnostic rebuild is non-physical at k={k},j={native_j},i={native_i}"
                    )));
                }
            }

            let thsurf = bottom[0] * th0[index3(1, out_j_halo, out_i_halo, halo_ny, halo_nx)]
                + bottom[1] * th0[index3(2, out_j_halo, out_i_halo, halo_ny, halo_nx)]
                + bottom[2] * th0[index3(3, out_j_halo, out_i_halo, halo_ny, halo_nx)];
            let qsurf = bottom[0] * qv0[index3(1, out_j_halo, out_i_halo, halo_ny, halo_nx)]
                + bottom[1] * qv0[index3(2, out_j_halo, out_i_halo, halo_ny, halo_nx)]
                + bottom[2] * qv0[index3(3, out_j_halo, out_i_halo, halo_ny, halo_nx)];
            let thvsurf = thsurf * (1.0_f32 + qsurf * reps) / (1.0_f32 + qsurf);
            let first_height = surface + 0.5_f32 * dz * (top_height - surface) / top_height;
            let surface_index = out_j_halo * halo_nx + out_i_halo;
            pi0s[surface_index] = pi0[index3(1, out_j_halo, out_i_halo, halo_ny, halo_nx)]
                + first_height * G
                    / (CP
                        * 0.5_f32
                        * (thvsurf + thv0[index3(1, out_j_halo, out_i_halo, halo_ny, halo_nx)]));
            prs0s[surface_index] = P00 * pi0s[surface_index].powf(CP / RD);
            rth0s[surface_index] = thsurf.powi(-1);
        }
    }

    // base.F:2160-2190: source face interpolation, including k=0=k=1.
    for out_j_halo in 0..halo_ny {
        let local_j = (out_j_halo as isize - NGXY as isize).clamp(0, crop.ny as isize - 1) as usize;
        for out_i_halo in 0..halo_nx {
            let local_i =
                (out_i_halo as isize - NGXY as isize).clamp(0, crop.nx as isize - 1) as usize;
            let bottom_density = bottom[0]
                * rho0[index3(1, out_j_halo, out_i_halo, halo_ny, halo_nx)]
                + bottom[1] * rho0[index3(2, out_j_halo, out_i_halo, halo_ny, halo_nx)]
                + bottom[2] * rho0[index3(3, out_j_halo, out_i_halo, halo_ny, halo_nx)];
            for k in [0usize, 1usize] {
                rf0[index3(k, out_j_halo, out_i_halo, halo_ny, halo_nx)] = bottom_density;
            }
            for k in 2..=nz {
                let weight = index3(k - 1, local_j, local_i, crop.ny, crop.nx);
                rf0[index3(k, out_j_halo, out_i_halo, halo_ny, halo_nx)] = c1[weight]
                    * rho0[index3(k - 1, out_j_halo, out_i_halo, halo_ny, halo_nx)]
                    + c2[weight] * rho0[index3(k, out_j_halo, out_i_halo, halo_ny, halo_nx)];
            }
            rf0[index3(nz + 1, out_j_halo, out_i_halo, halo_ny, halo_nx)] = top[0]
                * rho0[index3(nz, out_j_halo, out_i_halo, halo_ny, halo_nx)]
                + top[1] * rho0[index3(nz - 1, out_j_halo, out_i_halo, halo_ny, halo_nx)]
                + top[2] * rho0[index3(nz - 2, out_j_halo, out_i_halo, halo_ny, halo_nx)];
            for k in 0..=nz + 1 {
                let p = index3(k, out_j_halo, out_i_halo, halo_ny, halo_nx);
                if !rf0[p].is_finite() || rf0[p] <= 0.0 {
                    return Err(invalid(format!(
                        "terrain base face density is non-positive at k={k},j={out_j_halo},i={out_i_halo}"
                    )));
                }
                rrf0[p] = 1.0_f32 / rf0[p];
            }
            rho0s[out_j_halo * halo_nx + out_i_halo] =
                rf0[index3(1, out_j_halo, out_i_halo, halo_ny, halo_nx)];
        }
    }

    Ok(TerrainBaseFrame {
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
    })
}

/// Build the source-backed base state on the sealed stretched, terrain-following
/// child grid.  The sounding reconstruction follows `base.F:625-749`; only the
/// child `zh` and vertical metrics differ from the older uniform-grid path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_stretched_terrain_base(
    terrain: &[f32],
    shape_zyx: [usize; 3],
    zh_physical: &[f32],
    sigma_mass: &[f32],
    top_height: f32,
    source_profile_dz: f32,
    pi_profile: &[f32],
    theta_profile: &[f32],
    qv_profile: &[f32],
    c1_physical: &[f32],
    c2_physical: &[f32],
    child_bottom: [f32; 3],
    child_top: [f32; 3],
) -> Result<TerrainBaseFrame> {
    let [nz, ny, nx] = shape_zyx;
    let mass_physical_words = checked_product(&shape_zyx, "stretched child mass")?;
    let face_physical_words = checked_product(&[nz + 1, ny, nx], "stretched child face")?;
    if nz < 4
        || terrain.len() != ny * nx
        || zh_physical.len() != mass_physical_words
        || sigma_mass.len() != nz
        || c1_physical.len() != face_physical_words
        || c2_physical.len() != face_physical_words
        || !top_height.is_finite()
        || top_height <= 0.0
        || !source_profile_dz.is_finite()
        || source_profile_dz <= 0.0
    {
        return Err(invalid("stretched terrain-base geometry is invalid"));
    }
    if terrain
        .iter()
        .any(|value| !value.is_finite() || *value >= top_height)
        || sigma_mass
            .windows(2)
            .any(|pair| !pair[0].is_finite() || pair[1] <= pair[0])
    {
        return Err(invalid(
            "stretched terrain or mass-coordinate profile is invalid",
        ));
    }

    let generator = StretchedTerrainBaseColumnGenerator::new(
        sigma_mass,
        top_height,
        source_profile_dz,
        pi_profile,
        theta_profile,
        qv_profile,
        child_bottom,
        child_top,
    )?;
    let halo_ny = ny + 2 * NGXY;
    let halo_nx = nx + 2 * NGXY;
    let mass_shape = [nz + 2, halo_ny, halo_nx];
    let mass_words = checked_product(&mass_shape, "stretched terrain base allocation")?;
    let surface_words = checked_product(&[halo_ny, halo_nx], "stretched terrain base surface")?;
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
    let mut rho0s = vec![0.0_f32; surface_words];
    let mut pi0s = vec![0.0_f32; surface_words];
    let mut prs0s = vec![0.0_f32; surface_words];
    let mut rth0s = vec![0.0_f32; surface_words];

    for jh in 0..halo_ny {
        let j = (jh as isize - NGXY as isize).clamp(0, ny as isize - 1) as usize;
        for ih in 0..halo_nx {
            let i = (ih as isize - NGXY as isize).clamp(0, nx as isize - 1) as usize;
            let surface = terrain[j * nx + i];
            let published_zh =
                (jh >= NGXY && jh < NGXY + ny && ih >= NGXY && ih < NGXY + nx).then(|| {
                    (0..nz)
                        .map(|k| zh_physical[index3(k, j, i, ny, nx)])
                        .collect::<Vec<_>>()
                });
            let c1_column = (0..=nz)
                .map(|k| c1_physical[index3(k, j, i, ny, nx)])
                .collect::<Vec<_>>();
            let c2_column = (0..=nz)
                .map(|k| c2_physical[index3(k, j, i, ny, nx)])
                .collect::<Vec<_>>();
            let column =
                generator.generate(surface, published_zh.as_deref(), &c1_column, &c2_column)?;
            for k in 0..=nz + 1 {
                let p = index3(k, jh, ih, halo_ny, halo_nx);
                pi0[p] = column.pi0[k];
                prs0[p] = column.prs0[k];
                th0[p] = column.th0[k];
                thv0[p] = column.thv0[k];
                rth0[p] = column.rth0[k];
                qv0[p] = column.qv0[k];
                rho0[p] = column.rho0[k];
                rr0[p] = column.rr0[k];
                rf0[p] = column.rf0[k];
                rrf0[p] = column.rrf0[k];
            }
            let surface_index = jh * halo_nx + ih;
            rho0s[surface_index] = column.rho0s;
            pi0s[surface_index] = column.pi0s;
            prs0s[surface_index] = column.prs0s;
            rth0s[surface_index] = column.rth0s;
        }
    }

    Ok(TerrainBaseFrame {
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
    })
}

/// Reusable source-ordered hydrostatic column generator for bounded-memory
/// child publishers. The expensive sounding reconstruction is performed once;
/// each call then emits exactly one terrain-following column and its surface
/// diagnostics without allocating a horizontal domain.
pub(crate) struct StretchedTerrainBaseColumnGenerator {
    sounding: Sounding,
    sigma_mass: Vec<f32>,
    top_height: f32,
    child_bottom: [f32; 3],
    child_top: [f32; 3],
}

#[derive(Clone, Debug)]
pub(crate) struct StretchedTerrainBaseColumn {
    pub pi0: Vec<f32>,
    pub prs0: Vec<f32>,
    pub th0: Vec<f32>,
    pub thv0: Vec<f32>,
    pub rth0: Vec<f32>,
    pub qv0: Vec<f32>,
    pub rho0: Vec<f32>,
    pub rr0: Vec<f32>,
    pub rf0: Vec<f32>,
    pub rrf0: Vec<f32>,
    pub rho0s: f32,
    pub pi0s: f32,
    pub prs0s: f32,
    pub rth0s: f32,
}

impl StretchedTerrainBaseColumnGenerator {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        sigma_mass: &[f32],
        top_height: f32,
        source_profile_dz: f32,
        pi_profile: &[f32],
        theta_profile: &[f32],
        qv_profile: &[f32],
        child_bottom: [f32; 3],
        child_top: [f32; 3],
    ) -> Result<Self> {
        if sigma_mass.len() < 4
            || !top_height.is_finite()
            || top_height <= 0.0
            || !source_profile_dz.is_finite()
            || source_profile_dz <= 0.0
            || sigma_mass
                .windows(2)
                .any(|pair| !pair[0].is_finite() || pair[1] <= pair[0])
            || sigma_mass.last().is_some_and(|value| !value.is_finite())
        {
            return Err(invalid("stretched terrain-base column geometry is invalid"));
        }
        let source_endpoint = [1.875_f32, -1.25_f32, 0.375_f32];
        let sounding = build_sounding(
            pi_profile,
            theta_profile,
            qv_profile,
            source_profile_dz,
            source_endpoint,
            source_endpoint,
        )?;
        Ok(Self {
            sounding,
            sigma_mass: sigma_mass.to_vec(),
            top_height,
            child_bottom,
            child_top,
        })
    }

    pub(crate) fn nz(&self) -> usize {
        self.sigma_mass.len()
    }

    pub(crate) fn generate(
        &self,
        surface: f32,
        published_zh: Option<&[f32]>,
        c1_physical: &[f32],
        c2_physical: &[f32],
    ) -> Result<StretchedTerrainBaseColumn> {
        let nz = self.nz();
        if !surface.is_finite()
            || surface >= self.top_height
            || published_zh.is_some_and(|values| values.len() != nz)
            || c1_physical.len() != nz + 1
            || c2_physical.len() != nz + 1
            || c1_physical.iter().any(|value| !value.is_finite())
            || c2_physical.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("stretched terrain-base column inputs are invalid"));
        }
        let mut heights = Vec::with_capacity(nz);
        for (k, sigma) in self.sigma_mass.iter().copied().enumerate() {
            let height = surface + sigma * (self.top_height - surface) / self.top_height;
            if published_zh.is_some_and(|values| values[k].to_bits() != height.to_bits()) {
                return Err(invalid(format!(
                    "stretched child zh differs from the source-order terrain transform at level {k}"
                )));
            }
            heights.push(height);
        }
        let physical = column_base(&self.sounding, &heights)?;
        let mut pi0 = vec![0.0_f32; nz + 2];
        let mut th0 = vec![0.0_f32; nz + 2];
        let mut qv0 = vec![0.0_f32; nz + 2];
        pi0[1..=nz].copy_from_slice(&physical.pi);
        th0[1..=nz].copy_from_slice(&physical.theta);
        qv0[1..=nz].copy_from_slice(&physical.qv);
        for values in [&mut pi0, &mut th0, &mut qv0] {
            values[0] = self.child_bottom[0] * values[1]
                + self.child_bottom[1] * values[2]
                + self.child_bottom[2] * values[3];
            values[nz + 1] = self.child_top[0] * values[nz]
                + self.child_top[1] * values[nz - 1]
                + self.child_top[2] * values[nz - 2];
        }
        let mut prs0 = vec![0.0_f32; nz + 2];
        let mut thv0 = vec![0.0_f32; nz + 2];
        let mut rth0 = vec![0.0_f32; nz + 2];
        let mut rho0 = vec![0.0_f32; nz + 2];
        let mut rr0 = vec![0.0_f32; nz + 2];
        let reps = RV / RD;
        for k in 0..=nz + 1 {
            let qv = qv0[k];
            let qv_nonnegative = qv.max(0.0);
            thv0[k] = th0[k] * (1.0 + reps * qv_nonnegative) / (1.0 + qv_nonnegative);
            prs0[k] = P00 * pi0[k].powf(CP / RD);
            rho0[k] = prs0[k] / (RD * th0[k] * pi0[k] * (1.0 + qv * reps));
            rth0[k] = 1.0 / th0[k];
            rr0[k] = 1.0 / rho0[k];
            if !thv0[k].is_finite()
                || !prs0[k].is_finite()
                || prs0[k] <= 0.0
                || !rho0[k].is_finite()
                || rho0[k] <= 0.0
            {
                return Err(invalid(format!(
                    "stretched terrain-base column is non-physical at level {k}"
                )));
            }
        }
        let thsurf = self.child_bottom[0] * th0[1]
            + self.child_bottom[1] * th0[2]
            + self.child_bottom[2] * th0[3];
        let qsurf = self.child_bottom[0] * qv0[1]
            + self.child_bottom[1] * qv0[2]
            + self.child_bottom[2] * qv0[3];
        let thvsurf = thsurf * (1.0 + qsurf * reps) / (1.0 + qsurf);
        let pi0s = pi0[1] + heights[0] * G / (CP * 0.5 * (thvsurf + thv0[1]));
        let prs0s = P00 * pi0s.powf(CP / RD);
        let rth0s = thsurf.powi(-1);

        let bottom_density = self.child_bottom[0] * rho0[1]
            + self.child_bottom[1] * rho0[2]
            + self.child_bottom[2] * rho0[3];
        let mut rf0 = vec![0.0_f32; nz + 2];
        rf0[0] = bottom_density;
        rf0[1] = bottom_density;
        for k in 2..=nz {
            rf0[k] = c1_physical[k - 1] * rho0[k - 1] + c2_physical[k - 1] * rho0[k];
        }
        rf0[nz + 1] = self.child_top[0] * rho0[nz]
            + self.child_top[1] * rho0[nz - 1]
            + self.child_top[2] * rho0[nz - 2];
        let mut rrf0 = vec![0.0_f32; nz + 2];
        for k in 0..=nz + 1 {
            if !rf0[k].is_finite() || rf0[k] <= 0.0 {
                return Err(invalid(format!(
                    "stretched terrain-base face density is non-positive at level {k}"
                )));
            }
            rrf0[k] = 1.0 / rf0[k];
        }
        Ok(StretchedTerrainBaseColumn {
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
            rho0s: bottom_density,
            pi0s,
            prs0s,
            rth0s,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SurfaceHaloFrame {
    pub terrain_msl_halo: Vec<f32>,
    pub gz_halo: Vec<f32>,
    pub rgz_halo: Vec<f32>,
    pub gzu_halo: Vec<f32>,
    pub rgzu_halo: Vec<f32>,
    pub gzv_halo: Vec<f32>,
    pub rgzv_halo: Vec<f32>,
    pub dzdx_halo: Vec<f32>,
    pub dzdy_halo: Vec<f32>,
}

fn flx6(s1: f32, s2: f32, s3: f32, s4: f32, s5: f32, s6: f32) -> f32 {
    (37.0_f32 / 60.0_f32) * (s4 + s3)
        + (-8.0_f32 / 60.0_f32) * (s5 + s2)
        + (1.0_f32 / 60.0_f32) * (s6 + s1)
}

pub(crate) fn compute_surface_halos(
    terrain: &[f32],
    source_shape_yx: [usize; 2],
    crop: HorizontalCrop,
    spacing_m: [f64; 3],
    nz: usize,
) -> Result<SurfaceHaloFrame> {
    // param.F:2608's model top for the unstretched coordinate.
    let top = spacing_m[2] as f32 * nz as f32;
    compute_surface_halos_at_top(terrain, source_shape_yx, crop, spacing_m, top)
}

/// The same halos over a declared model top.
///
/// `param.F:8193` makes the top `sigmaf(nk+1)`, which equals `dz*nk` only on
/// an unstretched grid; `init_terrain.F:175-196` divides by it to form `gz`.
pub(crate) fn compute_surface_halos_at_top(
    terrain: &[f32],
    source_shape_yx: [usize; 2],
    crop: HorizontalCrop,
    spacing_m: [f64; 3],
    top: f32,
) -> Result<SurfaceHaloFrame> {
    if !top.is_finite() || top <= 0.0 {
        return Err(invalid("surface halo model top is invalid in binary32"));
    }
    let [source_ny, source_nx] = source_shape_yx;
    if terrain.len() != source_ny * source_nx
        || crop.j_start + crop.ny > source_ny
        || crop.i_start + crop.nx > source_nx
    {
        return Err(invalid("surface halo source/crop geometry is invalid"));
    }
    let halo_ny = crop.ny + 2 * NGXY;
    let halo_nx = crop.nx + 2 * NGXY;
    let rdx = 1.0_f32 / spacing_m[0] as f32;
    let rdy = 1.0_f32 / spacing_m[1] as f32;
    let terrain_at = |global_i: isize, global_j: isize| -> f32 {
        let i = global_i.clamp(0, source_nx as isize - 1) as usize;
        let j = global_j.clamp(0, source_ny as isize - 1) as usize;
        terrain[j * source_nx + i]
    };
    let x_flux = |face_i: isize, global_j: isize| -> f32 {
        flx6(
            terrain_at(face_i - 3, global_j),
            terrain_at(face_i - 2, global_j),
            terrain_at(face_i - 1, global_j),
            terrain_at(face_i, global_j),
            terrain_at(face_i + 1, global_j),
            terrain_at(face_i + 2, global_j),
        )
    };
    let y_flux = |global_i: isize, face_j: isize| -> f32 {
        flx6(
            terrain_at(global_i, face_j - 3),
            terrain_at(global_i, face_j - 2),
            terrain_at(global_i, face_j - 1),
            terrain_at(global_i, face_j),
            terrain_at(global_i, face_j + 1),
            terrain_at(global_i, face_j + 2),
        )
    };
    let mut terrain_msl_halo = vec![0.0_f32; halo_ny * halo_nx];
    let mut gz_halo = vec![0.0_f32; halo_ny * halo_nx];
    let mut rgz_halo = vec![0.0_f32; halo_ny * halo_nx];
    let mut gzu_halo = vec![0.0_f32; halo_ny * halo_nx];
    let mut rgzu_halo = vec![0.0_f32; halo_ny * halo_nx];
    let mut gzv_halo = vec![0.0_f32; halo_ny * halo_nx];
    let mut rgzv_halo = vec![0.0_f32; halo_ny * halo_nx];
    let mut dzdx_halo = vec![0.0_f32; halo_ny * halo_nx];
    let mut dzdy_halo = vec![0.0_f32; halo_ny * halo_nx];
    for jh in 0..halo_ny {
        let global_j = crop.j_start as isize + jh as isize - NGXY as isize;
        for ih in 0..halo_nx {
            let global_i = crop.i_start as isize + ih as isize - NGXY as isize;
            let index = jh * halo_nx + ih;
            let surface = terrain_at(global_i, global_j);
            terrain_msl_halo[index] = surface;
            gz_halo[index] = top / (top - surface);
            rgz_halo[index] = (top - surface) / top;
            let bounded_i = global_i.clamp(0, source_nx as isize - 1);
            let bounded_j = global_j.clamp(0, source_ny as isize - 1);
            dzdx_halo[index] =
                (x_flux(bounded_i + 1, bounded_j) - x_flux(bounded_i, bounded_j)) * rdx;
            dzdy_halo[index] =
                (y_flux(bounded_i, bounded_j + 1) - y_flux(bounded_i, bounded_j)) * rdy;
        }
    }
    // init_terrain.F:247-253 intentionally leaves the first allocated row
    // and column at their cm1.F allocation-zero value.
    for jh in 1..halo_ny {
        let global_j = crop.j_start as isize + jh as isize - NGXY as isize;
        for ih in 1..halo_nx {
            let global_i = crop.i_start as isize + ih as isize - NGXY as isize;
            let index = jh * halo_nx + ih;
            let u_surface =
                0.5_f32 * (terrain_at(global_i - 1, global_j) + terrain_at(global_i, global_j));
            let v_surface =
                0.5_f32 * (terrain_at(global_i, global_j - 1) + terrain_at(global_i, global_j));
            gzu_halo[index] = top / (top - u_surface);
            rgzu_halo[index] = (top - u_surface) / top;
            gzv_halo[index] = top / (top - v_surface);
            rgzv_halo[index] = (top - v_surface) / top;
        }
    }
    Ok(SurfaceHaloFrame {
        terrain_msl_halo,
        gz_halo,
        rgz_halo,
        gzu_halo,
        rgzu_halo,
        gzv_halo,
        rgzv_halo,
        dzdx_halo,
        dzdy_halo,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct MutableColdStartFrame {
    pub carriers: Vec<MutableCarrier>,
    pub pi_recovery: RecoveryMetric,
    pub theta_recovery: RecoveryMetric,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PerturbationPolicy {
    /// Repartition a source total state against a newly diagnosed base.  This
    /// is the policy used by the older initialized-product terrain admission.
    RepartitionTotal,
    /// Preserve the perturbation carriers written by CM1 restart-grid
    /// interpolation.  `restart_read.F` overwrites the provisional `init3d`
    /// fields, so the fixed 50 m child must not repartition them a second time.
    PreserveRestartGrid,
    /// Preserve perturbations produced directly by the source-ordered
    /// `init3d` idealized initializer.  Numerically this is the same carrier
    /// operation as restart preservation, but keeping the provenance label
    /// distinct prevents a fresh idealized state from being described as a
    /// restart-grid product.
    PreserveInitializedPerturbation,
}

#[derive(Clone, Debug)]
pub(crate) struct MutableCarrier {
    pub name: &'static str,
    pub role: &'static str,
    pub shape: Vec<usize>,
    pub live_region: &'static str,
    pub values: Vec<f32>,
}

fn crop_mass_value(
    source: &[f32],
    source_ny: usize,
    source_nx: usize,
    crop: HorizontalCrop,
    k: usize,
    j: usize,
    i: usize,
) -> f32 {
    source[index3(k, crop.j_start + j, crop.i_start + i, source_ny, source_nx)]
}

fn crop_u_value(
    source: &[f32],
    source_ny: usize,
    source_nx: usize,
    crop: HorizontalCrop,
    k: usize,
    j: usize,
    i: usize,
) -> f32 {
    source[index3(
        k,
        crop.j_start + j,
        crop.i_start + i,
        source_ny,
        source_nx + 1,
    )]
}

fn crop_v_value(
    source: &[f32],
    source_ny: usize,
    source_nx: usize,
    crop: HorizontalCrop,
    k: usize,
    j: usize,
    i: usize,
) -> f32 {
    source[index3(
        k,
        crop.j_start + j,
        crop.i_start + i,
        source_ny + 1,
        source_nx,
    )]
}

#[inline]
fn native_xy(value: isize) -> usize {
    usize::try_from(value + NGXY as isize - 1).expect("native index is inside allocation")
}

fn bcp_open(array: &mut [f32], nk: usize, nj: usize, ni: usize) {
    let ny = nj + 2;
    let nx = ni + 2;
    for k in 1..=nk {
        for j in 0..=nj + 1 {
            array[index3(k, j, 0, ny, nx)] = array[index3(k, j, 1, ny, nx)];
            array[index3(k, j, ni + 1, ny, nx)] = array[index3(k, j, ni, ny, nx)];
        }
        for i in 0..=ni + 1 {
            array[index3(k, 0, i, ny, nx)] = array[index3(k, 1, i, ny, nx)];
            array[index3(k, nj + 1, i, ny, nx)] = array[index3(k, nj, i, ny, nx)];
        }
    }
}

fn bcs_open(array: &mut [f32], nk: usize, nj: usize, ni: usize) {
    let ny = nj + 2 * NGXY;
    let nx = ni + 2 * NGXY;
    for k in 1..=nk {
        for j_native in 0..=nj as isize + 1 {
            let j = native_xy(j_native);
            for i_native in (1 - NGXY as isize)..=0 {
                array[index3(k, j, native_xy(i_native), ny, nx)] =
                    array[index3(k, j, native_xy(1), ny, nx)];
            }
            for i_native in ni as isize + 1..=ni as isize + NGXY as isize {
                array[index3(k, j, native_xy(i_native), ny, nx)] =
                    array[index3(k, j, native_xy(ni as isize), ny, nx)];
            }
        }
        for j_native in (1 - NGXY as isize)..=0 {
            for i_native in 0..=ni as isize + 1 {
                array[index3(k, native_xy(j_native), native_xy(i_native), ny, nx)] =
                    array[index3(k, native_xy(1), native_xy(i_native), ny, nx)];
            }
        }
        for j_native in nj as isize + 1..=nj as isize + NGXY as isize {
            for i_native in 0..=ni as isize + 1 {
                array[index3(k, native_xy(j_native), native_xy(i_native), ny, nx)] =
                    array[index3(k, native_xy(nj as isize), native_xy(i_native), ny, nx)];
            }
        }
    }
}

fn bcu_open(array: &mut [f32], nk: usize, nj: usize, ni: usize) {
    let ny = nj + 2 * NGXY;
    let nx = ni + 1 + 2 * NGXY;
    for k in 1..=nk {
        for j_native in 0..=nj as isize + 1 {
            let j = native_xy(j_native);
            for i_native in (1 - NGXY as isize)..=0 {
                array[index3(k, j, native_xy(i_native), ny, nx)] =
                    array[index3(k, j, native_xy(1), ny, nx)];
            }
            for i_native in ni as isize + 2..=ni as isize + 1 + NGXY as isize {
                array[index3(k, j, native_xy(i_native), ny, nx)] =
                    array[index3(k, j, native_xy(ni as isize + 1), ny, nx)];
            }
        }
        for j_native in (1 - NGXY as isize)..=0 {
            for i_native in 0..=ni as isize + 2 {
                array[index3(k, native_xy(j_native), native_xy(i_native), ny, nx)] =
                    array[index3(k, native_xy(1), native_xy(i_native), ny, nx)];
            }
        }
        for j_native in nj as isize + 1..=nj as isize + NGXY as isize {
            for i_native in 0..=ni as isize + 2 {
                array[index3(k, native_xy(j_native), native_xy(i_native), ny, nx)] =
                    array[index3(k, native_xy(nj as isize), native_xy(i_native), ny, nx)];
            }
        }
    }
}

fn bcv_open(array: &mut [f32], nk: usize, nj: usize, ni: usize) {
    let ny = nj + 1 + 2 * NGXY;
    let nx = ni + 2 * NGXY;
    for k in 1..=nk {
        for j_native in (1 - NGXY as isize)..=0 {
            for i_native in 0..=ni as isize + 1 {
                array[index3(k, native_xy(j_native), native_xy(i_native), ny, nx)] =
                    array[index3(k, native_xy(1), native_xy(i_native), ny, nx)];
            }
        }
        for j_native in nj as isize + 2..=nj as isize + 1 + NGXY as isize {
            for i_native in 0..=ni as isize + 1 {
                array[index3(k, native_xy(j_native), native_xy(i_native), ny, nx)] =
                    array[index3(k, native_xy(nj as isize + 1), native_xy(i_native), ny, nx)];
            }
        }
        for j_native in 0..=nj as isize + 2 {
            let j = native_xy(j_native);
            for i_native in (1 - NGXY as isize)..=0 {
                array[index3(k, j, native_xy(i_native), ny, nx)] =
                    array[index3(k, j, native_xy(1), ny, nx)];
            }
            for i_native in ni as isize + 1..=ni as isize + NGXY as isize {
                array[index3(k, j, native_xy(i_native), ny, nx)] =
                    array[index3(k, j, native_xy(ni as isize), ny, nx)];
            }
        }
    }
}

fn bcw_open(array: &mut [f32], nk: usize, nj: usize, ni: usize) {
    let ny = nj + 2 * NGXY;
    let nx = ni + 2 * NGXY;
    for k in 2..=nk {
        for j_native in 0..=nj as isize + 1 {
            let j = native_xy(j_native);
            for i_native in (1 - NGXY as isize)..=0 {
                array[index3(k, j, native_xy(i_native), ny, nx)] =
                    array[index3(k, j, native_xy(1), ny, nx)];
            }
            for i_native in ni as isize + 1..=ni as isize + NGXY as isize {
                array[index3(k, j, native_xy(i_native), ny, nx)] =
                    array[index3(k, j, native_xy(ni as isize), ny, nx)];
            }
        }
        for j_native in (1 - NGXY as isize)..=0 {
            for i_native in 0..=ni as isize + 1 {
                array[index3(k, native_xy(j_native), native_xy(i_native), ny, nx)] =
                    array[index3(k, native_xy(1), native_xy(i_native), ny, nx)];
            }
        }
        for j_native in nj as isize + 1..=nj as isize + NGXY as isize {
            for i_native in 0..=ni as isize + 1 {
                array[index3(k, native_xy(j_native), native_xy(i_native), ny, nx)] =
                    array[index3(k, native_xy(nj as isize), native_xy(i_native), ny, nx)];
            }
        }
    }
    for j_native in 0..=nj as isize + 1 {
        for i_native in 0..=ni as isize + 1 {
            array[index3(1, native_xy(j_native), native_xy(i_native), ny, nx)] = 0.0;
            array[index3(nk + 1, native_xy(j_native), native_xy(i_native), ny, nx)] = 0.0;
        }
    }
}

#[allow(clippy::too_many_arguments)] // Keeps the CM1 bcwsfc inputs visibly source-ordered.
fn bcwsfc(
    u: &[f32],
    v: &[f32],
    w: &mut [f32],
    gz_halo: &[f32],
    dzdx_halo: &[f32],
    dzdy_halo: &[f32],
    nj: usize,
    ni: usize,
) {
    let nyu = nj + 2 * NGXY;
    let nxu = ni + 1 + 2 * NGXY;
    let nyv = nj + 1 + 2 * NGXY;
    let nxv = ni + 2 * NGXY;
    let nyw = nj + 2 * NGXY;
    let nxw = ni + 2 * NGXY;
    for j_native in 0..=nj as isize + 1 {
        let j = native_xy(j_native);
        for i_native in 0..=ni as isize + 1 {
            let i = native_xy(i_native);
            let u_sum = u[index3(1, j, i, nyu, nxu)] + u[index3(1, j, i + 1, nyu, nxu)];
            let v_sum = v[index3(1, j, i, nyv, nxv)] + v[index3(1, j + 1, i, nyv, nxv)];
            let metric = j * (ni + 2 * NGXY) + i;
            w[index3(1, j, i, nyw, nxw)] =
                0.5_f32 * (u_sum * dzdx_halo[metric] + v_sum * dzdy_halo[metric]) * gz_halo[metric];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_cold_start_mutable(
    source_shape_zyx: [usize; 3],
    crop: HorizontalCrop,
    pi_profile: &[f32],
    theta_profile: &[f32],
    ua_source: &[f32],
    va_source: &[f32],
    wa_source: &[f32],
    ppi_source: &[f32],
    ppx_source: Option<&[f32]>,
    theta_perturbation_source: &[f32],
    qa_source: &[f32],
    base: &TerrainBaseFrame,
    surface_halos: &SurfaceHaloFrame,
    perturbation_policy: PerturbationPolicy,
) -> Result<MutableColdStartFrame> {
    let [nk, source_ny, source_nx] = source_shape_zyx;
    let nj = crop.ny;
    let ni = crop.nx;
    let source_mass_words = checked_product(&source_shape_zyx, "source mass")?;
    if pi_profile.len() != nk
        || theta_profile.len() != nk
        || ua_source.len() != nk * source_ny * (source_nx + 1)
        || va_source.len() != nk * (source_ny + 1) * source_nx
        || wa_source.len() != (nk + 1) * source_ny * source_nx
        || ppi_source.len() != source_mass_words
        || ppx_source.is_some_and(|values| values.len() != source_mass_words)
        || theta_perturbation_source.len() != source_mass_words
        || qa_source.len() != 6 * source_mass_words
    {
        return Err(invalid(
            "mutable cold-start source carrier geometry is invalid",
        ));
    }
    let pressure_shape = vec![nk + 2, nj + 2, ni + 2];
    let scalar_shape = vec![nk + 2, nj + 2 * NGXY, ni + 2 * NGXY];
    let u_shape = vec![nk + 2, nj + 2 * NGXY, ni + 1 + 2 * NGXY];
    let v_shape = vec![nk + 2, nj + 1 + 2 * NGXY, ni + 2 * NGXY];
    let w_shape = vec![nk + 3, nj + 2 * NGXY, ni + 2 * NGXY];
    let pressure_words = checked_product(&pressure_shape, "pressure family")?;
    let scalar_words = checked_product(&scalar_shape, "scalar family")?;
    let u_words = checked_product(&u_shape, "u family")?;
    let v_words = checked_product(&v_shape, "v family")?;
    let w_words = checked_product(&w_shape, "w family")?;
    let mut ppi = vec![0.0_f32; pressure_words];
    let mut pp3d = vec![0.0_f32; pressure_words];
    let mut ppx = vec![0.0_f32; pressure_words];
    let mut piadv = vec![0.0_f32; pressure_words];
    let mut ppterm = vec![0.0_f32; pressure_words];
    let ppten = vec![0.0_f32; pressure_words];
    let mut thv = vec![0.0_f32; pressure_words];
    let mut tha = vec![0.0_f32; scalar_words];
    let mut ua = vec![0.0_f32; u_words];
    let mut u3d = vec![0.0_f32; u_words];
    let uten = vec![0.0_f32; u_words];
    let mut va = vec![0.0_f32; v_words];
    let mut v3d = vec![0.0_f32; v_words];
    let vten = vec![0.0_f32; v_words];
    let mut wa = vec![0.0_f32; w_words];
    let mut w3d = vec![0.0_f32; w_words];
    let wten = vec![0.0_f32; w_words];
    let mut pi_recovery = RecoveryMetric::default();
    let mut theta_recovery = RecoveryMetric::default();
    let reps = RV / RD;
    let cv = CP - RD;
    let cvv = CPV - RV;
    let rovcp = RD / CP;
    let base_ny = nj + 2 * NGXY;
    let base_nx = ni + 2 * NGXY;
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let source_index =
                    index3(k, crop.j_start + j, crop.i_start + i, source_ny, source_nx);
                let base_index = index3(k + 1, j + NGXY, i + NGXY, base_ny, base_nx);
                let (total_pi, adjusted_ppi) = match perturbation_policy {
                    PerturbationPolicy::RepartitionTotal => {
                        let total = pi_profile[k] + ppi_source[source_index];
                        (total, total - base.pi0[base_index])
                    }
                    PerturbationPolicy::PreserveRestartGrid
                    | PerturbationPolicy::PreserveInitializedPerturbation => {
                        let perturbation = ppi_source[source_index];
                        (base.pi0[base_index] + perturbation, perturbation)
                    }
                };
                let recovered_pi = base.pi0[base_index] + adjusted_ppi;
                pi_recovery.observe(total_pi, recovered_pi);
                let (total_theta, adjusted_theta) = match perturbation_policy {
                    PerturbationPolicy::RepartitionTotal => {
                        let total = theta_profile[k] + theta_perturbation_source[source_index];
                        (total, total - base.th0[base_index])
                    }
                    PerturbationPolicy::PreserveRestartGrid
                    | PerturbationPolicy::PreserveInitializedPerturbation => {
                        let perturbation = theta_perturbation_source[source_index];
                        (base.th0[base_index] + perturbation, perturbation)
                    }
                };
                let recovered_theta = base.th0[base_index] + adjusted_theta;
                theta_recovery.observe(total_theta, recovered_theta);
                let pressure_index = index3(k + 1, j + 1, i + 1, nj + 2, ni + 2);
                ppi[pressure_index] = adjusted_ppi;
                pp3d[pressure_index] = adjusted_ppi;
                ppx[pressure_index] = ppx_source
                    .map(|values| values[source_index])
                    .unwrap_or(adjusted_ppi);
                piadv[pressure_index] = recovered_pi;
                let scalar_index = index3(k + 1, j + NGXY, i + NGXY, nj + 2 * NGXY, ni + 2 * NGXY);
                tha[scalar_index] = adjusted_theta;
                let qv = qa_source[source_index].max(0.0);
                let liquid = qa_source[source_mass_words + source_index]
                    + qa_source[2 * source_mass_words + source_index];
                let solid = qa_source[3 * source_mass_words + source_index]
                    + qa_source[4 * source_mass_words + source_index]
                    + qa_source[5 * source_mass_words + source_index];
                let qli = (liquid + solid).max(0.0);
                let cpli = CPL * liquid.max(0.0) + CPI * solid.max(0.0);
                thv[pressure_index] =
                    recovered_theta * (1.0_f32 + reps * qv) / (1.0_f32 + qv + qli);
                let cpm = CP + CPV * qv + cpli;
                let cvm = 1.0_f32 / (cv + cvv * qv + cpli);
                ppterm[pressure_index] = recovered_pi * rovcp * cpm * cvm;
            }
        }
    }
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..=ni {
                let value = crop_u_value(ua_source, source_ny, source_nx, crop, k, j, i);
                let destination = index3(k + 1, j + NGXY, i + NGXY, u_shape[1], u_shape[2]);
                ua[destination] = value;
                u3d[destination] = value;
            }
        }
        for j in 0..=nj {
            for i in 0..ni {
                let value = crop_v_value(va_source, source_ny, source_nx, crop, k, j, i);
                let destination = index3(k + 1, j + NGXY, i + NGXY, v_shape[1], v_shape[2]);
                va[destination] = value;
                v3d[destination] = value;
            }
        }
    }
    for k in 0..=nk {
        for j in 0..nj {
            for i in 0..ni {
                let value = crop_mass_value(wa_source, source_ny, source_nx, crop, k, j, i);
                let destination = index3(k + 1, j + NGXY, i + NGXY, w_shape[1], w_shape[2]);
                wa[destination] = value;
                w3d[destination] = value;
            }
        }
    }
    bcu_open(&mut ua, nk, nj, ni);
    bcu_open(&mut u3d, nk, nj, ni);
    bcv_open(&mut va, nk, nj, ni);
    bcv_open(&mut v3d, nk, nj, ni);
    bcw_open(&mut wa, nk, nj, ni);
    bcw_open(&mut w3d, nk, nj, ni);
    bcwsfc(
        &ua,
        &va,
        &mut wa,
        &surface_halos.gz_halo,
        &surface_halos.dzdx_halo,
        &surface_halos.dzdy_halo,
        nj,
        ni,
    );
    bcwsfc(
        &u3d,
        &v3d,
        &mut w3d,
        &surface_halos.gz_halo,
        &surface_halos.dzdx_halo,
        &surface_halos.dzdy_halo,
        nj,
        ni,
    );
    bcp_open(&mut ppi, nk, nj, ni);
    bcp_open(&mut pp3d, nk, nj, ni);
    bcp_open(&mut ppx, nk, nj, ni);
    bcp_open(&mut thv, nk, nj, ni);
    bcs_open(&mut tha, nk, nj, ni);
    let th3d = tha.clone();
    let thten = vec![0.0_f32; scalar_words];
    let thten1 = vec![0.0_f32; scalar_words];
    // solve2.F:1041-1049 evaluates terrain piadv over the complete pressure
    // live region after pp3d's open bcp fill.
    for k in 1..=nk {
        for j in 0..=nj + 1 {
            for i in 0..=ni + 1 {
                let p = index3(k, j, i, nj + 2, ni + 2);
                let base_index = index3(
                    k,
                    native_xy(j as isize),
                    native_xy(i as isize),
                    base_ny,
                    base_nx,
                );
                piadv[p] = base.pi0[base_index] + pp3d[p];
            }
        }
    }
    let history_we_shape = vec![nk + 2, nj + 2 * NGXY];
    let history_sn_shape = vec![nk + 2, ni + 2 * NGXY];
    let history_we_words = checked_product(&history_we_shape, "west/east history")?;
    let history_sn_words = checked_product(&history_sn_shape, "south/north history")?;
    let (exner_role, theta_role) = match perturbation_policy {
        PerturbationPolicy::RepartitionTotal => {
            ("base-repartitioned Exner", "base-repartitioned theta")
        }
        PerturbationPolicy::PreserveRestartGrid => (
            "restart-grid-preserved Exner",
            "restart-grid-preserved theta",
        ),
        PerturbationPolicy::PreserveInitializedPerturbation => {
            ("source-initialized Exner", "source-initialized theta")
        }
    };
    let carriers = vec![
        mutable(
            "ppi",
            exner_role,
            pressure_shape.clone(),
            "bcp-live-horizontal-one-halo; vertical ghosts allocation-zero",
            ppi,
        ),
        mutable(
            "pp3d",
            exner_role,
            pressure_shape.clone(),
            "bcp-live-horizontal-one-halo; vertical ghosts allocation-zero",
            pp3d,
        ),
        mutable(
            "ppx",
            "restart-grid-preserved held Exner pressure helper",
            pressure_shape.clone(),
            "bcp-live-horizontal-one-halo; vertical ghosts allocation-zero",
            ppx,
        ),
        mutable(
            "piadv",
            "terrain-total-Exner-advection",
            pressure_shape.clone(),
            "solve2.F:1041-1049 full pressure horizontal live region",
            piadv,
        ),
        mutable(
            "ppterm",
            "eqtset2-pressure-equation-term",
            pressure_shape.clone(),
            "physical mass cells only; allocation-zero halo",
            ppterm,
        ),
        mutable(
            "ppten",
            "held-pressure-tendency",
            pressure_shape.clone(),
            "exact cm1.F allocation-zero cold-start",
            ppten,
        ),
        mutable(
            "thv",
            "theta-rho",
            pressure_shape,
            "bcp-live-horizontal-one-halo; vertical ghosts allocation-zero",
            thv,
        ),
        mutable(
            "tha",
            theta_role,
            scalar_shape.clone(),
            "bcs open live subset; untouched allocation corners/vertical ghosts remain zero",
            tha,
        ),
        mutable(
            "th3d",
            theta_role,
            scalar_shape.clone(),
            "bcs open live subset; untouched allocation corners/vertical ghosts remain zero",
            th3d,
        ),
        mutable(
            "thten",
            "stage-theta-tendency",
            scalar_shape.clone(),
            "exact cm1.F allocation-zero cold-start",
            thten,
        ),
        mutable(
            "thten1",
            "held-theta-tendency",
            scalar_shape,
            "exact cm1.F allocation-zero cold-start",
            thten1,
        ),
        mutable(
            "ua",
            "zonal-velocity-A",
            u_shape.clone(),
            "bcu open live subset from corrected source physical faces",
            ua,
        ),
        mutable(
            "u3d",
            "current-zonal-velocity",
            u_shape.clone(),
            "bcu open live subset from corrected source physical faces",
            u3d,
        ),
        mutable(
            "uten",
            "stage-zonal-tendency",
            u_shape.clone(),
            "exact cm1.F allocation-zero cold-start",
            uten,
        ),
        mutable(
            "uten1",
            "held-zonal-tendency",
            u_shape.clone(),
            "exact cm1.F allocation-zero cold-start",
            vec![0.0; u_words],
        ),
        mutable(
            "va",
            "meridional-velocity-A",
            v_shape.clone(),
            "bcv open live subset from corrected source physical faces",
            va,
        ),
        mutable(
            "v3d",
            "current-meridional-velocity",
            v_shape.clone(),
            "bcv open live subset from corrected source physical faces",
            v3d,
        ),
        mutable(
            "vten",
            "stage-meridional-tendency",
            v_shape.clone(),
            "exact cm1.F allocation-zero cold-start",
            vten,
        ),
        mutable(
            "vten1",
            "held-meridional-tendency",
            v_shape.clone(),
            "exact cm1.F allocation-zero cold-start",
            vec![0.0; v_words],
        ),
        mutable(
            "wa",
            "vertical-velocity-A",
            w_shape.clone(),
            "bcw open plus terrain bcwsfc live subset",
            wa,
        ),
        mutable(
            "w3d",
            "current-vertical-velocity",
            w_shape.clone(),
            "bcw open plus terrain bcwsfc live subset",
            w3d,
        ),
        mutable(
            "wten",
            "stage-vertical-tendency",
            w_shape.clone(),
            "exact cm1.F allocation-zero cold-start",
            wten,
        ),
        mutable(
            "wten1",
            "held-vertical-tendency",
            w_shape.clone(),
            "exact cm1.F allocation-zero cold-start",
            vec![0.0; w_words],
        ),
        mutable(
            "radbcw",
            "west-radiation-history",
            history_we_shape.clone(),
            "cm1.F:1559-1560 exact zero cold-start; resume import required",
            vec![0.0; history_we_words],
        ),
        mutable(
            "radbce",
            "east-radiation-history",
            history_we_shape,
            "cm1.F:1561-1562 exact zero cold-start; resume import required",
            vec![0.0; history_we_words],
        ),
        mutable(
            "radbcs",
            "south-radiation-history",
            history_sn_shape.clone(),
            "cm1.F:1563-1564 exact zero cold-start; resume import required",
            vec![0.0; history_sn_words],
        ),
        mutable(
            "radbcn",
            "north-radiation-history",
            history_sn_shape,
            "cm1.F:1565-1566 exact zero cold-start; resume import required",
            vec![0.0; history_sn_words],
        ),
    ];
    Ok(MutableColdStartFrame {
        carriers,
        pi_recovery,
        theta_recovery,
    })
}

fn mutable(
    name: &'static str,
    role: &'static str,
    shape: Vec<usize>,
    live_region: &'static str,
    values: Vec<f32>,
) -> MutableCarrier {
    MutableCarrier {
        name,
        role,
        shape,
        live_region,
        values,
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LiveDensityFrame {
    pub carriers: Vec<MutableCarrier>,
}

/// Rebuild the selected `psolver=3,imoist=1` live pressure/density family
/// after restart-grid carriers overwrite the provisional initialization.
/// This is `init3d.F:2386-2467`; output is the physical domain and the CUDA
/// consumer owns the corresponding source-family halo pack.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_child_live_density(
    shape_zyx: [usize; 3],
    pp3d: &[f32],
    th3d: &[f32],
    qa: &[f32],
    base: &TerrainBaseFrame,
    c1: &[f32],
    c2: &[f32],
    bottom: [f32; 3],
    top: [f32; 3],
) -> Result<LiveDensityFrame> {
    let [nz, ny, nx] = shape_zyx;
    let mass_words = checked_product(&shape_zyx, "child live density mass")?;
    let face_words = checked_product(&[nz + 1, ny, nx], "child live density face")?;
    let base_words = checked_product(
        &[nz + 2, ny + 2 * NGXY, nx + 2 * NGXY],
        "child live density base",
    )?;
    if pp3d.len() != mass_words
        || th3d.len() != mass_words
        || qa.len() != 6 * mass_words
        || base.pi0.len() != base_words
        || base.th0.len() != base_words
        || c1.len() != face_words
        || c2.len() != face_words
    {
        return Err(invalid("child live-density carrier geometry is invalid"));
    }
    let mut prs = vec![0.0_f32; mass_words];
    let mut rho = vec![0.0_f32; mass_words];
    let mut rr = vec![0.0_f32; mass_words];
    let mut rf = vec![0.0_f32; face_words];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let physical = index3(k, j, i, ny, nx);
                let padded = index3(k + 1, j + NGXY, i + NGXY, ny + 2 * NGXY, nx + 2 * NGXY);
                let total_pi = base.pi0[padded] + pp3d[physical];
                let total_theta = base.th0[padded] + th3d[physical];
                let qv = qa[physical].max(0.0);
                prs[physical] = P00 * total_pi.powf(CP / RD);
                rho[physical] = prs[physical] / (total_theta * total_pi * (RD + qv * RV));
                rr[physical] = 1.0 / rho[physical];
                if !total_pi.is_finite()
                    || total_pi <= 0.0
                    || !total_theta.is_finite()
                    || total_theta <= 0.0
                    || !prs[physical].is_finite()
                    || prs[physical] <= 0.0
                    || !rho[physical].is_finite()
                    || rho[physical] <= 0.0
                {
                    return Err(invalid(format!(
                        "child live pressure/density is non-physical at k={k},j={j},i={i}"
                    )));
                }
            }
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            let column = |k: usize| rho[index3(k, j, i, ny, nx)];
            rf[index3(0, j, i, ny, nx)] =
                bottom[0] * column(0) + bottom[1] * column(1) + bottom[2] * column(2);
            for face in 1..nz {
                let p = index3(face, j, i, ny, nx);
                rf[p] = c1[p] * column(face - 1) + c2[p] * column(face);
            }
            rf[index3(nz, j, i, ny, nx)] =
                top[0] * column(nz - 1) + top[1] * column(nz - 2) + top[2] * column(nz - 3);
        }
    }
    if rf.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(invalid(
            "child live face-density reconstruction is non-positive",
        ));
    }
    Ok(LiveDensityFrame {
        carriers: vec![
            mutable(
                "prs",
                "restart-grid child live pressure",
                vec![nz, ny, nx],
                "physical mass cells; source halo pack remains consumer-owned",
                prs,
            ),
            mutable(
                "rf",
                "restart-grid child live face density",
                vec![nz + 1, ny, nx],
                "physical z faces; source halo pack remains consumer-owned",
                rf,
            ),
            mutable(
                "rho",
                "restart-grid child live density",
                vec![nz, ny, nx],
                "physical mass cells; source halo pack remains consumer-owned",
                rho,
            ),
            mutable(
                "rr",
                "reciprocal restart-grid child live density",
                vec![nz, ny, nx],
                "physical mass cells; source halo pack remains consumer-owned",
                rr,
            ),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_stretched_column_is_raw_bit_exact_to_the_full_frame() {
        let shape = [4usize, 2usize, 2usize];
        let [nz, ny, nx] = shape;
        let terrain = vec![0.0_f32, 50.0, 100.0, 150.0];
        let sigma = vec![50.0_f32, 150.0, 300.0, 600.0];
        let top = 800.0_f32;
        let mut zh = vec![0.0_f32; nz * ny * nx];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let surface = terrain[j * nx + i];
                    zh[index3(k, j, i, ny, nx)] = surface + sigma[k] * (top - surface) / top;
                }
            }
        }
        let c1 = vec![0.5_f32; (nz + 1) * ny * nx];
        let c2 = vec![0.5_f32; (nz + 1) * ny * nx];
        let pi = vec![0.99_f32, 0.97, 0.95, 0.93];
        let theta = vec![300.0_f32, 302.0, 304.0, 306.0];
        let qv = vec![0.01_f32, 0.009, 0.008, 0.007];
        let endpoint = [1.875_f32, -1.25, 0.375];
        let full = compute_stretched_terrain_base(
            &terrain, shape, &zh, &sigma, top, 200.0, &pi, &theta, &qv, &c1, &c2, endpoint,
            endpoint,
        )
        .unwrap();
        let generator = StretchedTerrainBaseColumnGenerator::new(
            &sigma, top, 200.0, &pi, &theta, &qv, endpoint, endpoint,
        )
        .unwrap();
        let published_zh = (0..nz)
            .map(|k| zh[index3(k, 0, 0, ny, nx)])
            .collect::<Vec<_>>();
        let c1_column = (0..=nz)
            .map(|k| c1[index3(k, 0, 0, ny, nx)])
            .collect::<Vec<_>>();
        let c2_column = (0..=nz)
            .map(|k| c2[index3(k, 0, 0, ny, nx)])
            .collect::<Vec<_>>();
        let column = generator
            .generate(terrain[0], Some(&published_zh), &c1_column, &c2_column)
            .unwrap();
        let halo_ny = ny + 2 * NGXY;
        let halo_nx = nx + 2 * NGXY;
        let compare = |full_values: &[f32], column_values: &[f32]| {
            for (k, actual) in column_values.iter().copied().enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    full_values[index3(k, NGXY, NGXY, halo_ny, halo_nx)].to_bits()
                );
            }
        };
        compare(&full.pi0, &column.pi0);
        compare(&full.prs0, &column.prs0);
        compare(&full.th0, &column.th0);
        compare(&full.thv0, &column.thv0);
        compare(&full.rth0, &column.rth0);
        compare(&full.qv0, &column.qv0);
        compare(&full.rho0, &column.rho0);
        compare(&full.rr0, &column.rr0);
        compare(&full.rf0, &column.rf0);
        compare(&full.rrf0, &column.rrf0);
        let surface = NGXY * halo_nx + NGXY;
        assert_eq!(column.rho0s.to_bits(), full.rho0s[surface].to_bits());
        assert_eq!(column.pi0s.to_bits(), full.pi0s[surface].to_bits());
        assert_eq!(column.prs0s.to_bits(), full.prs0s[surface].to_bits());
        assert_eq!(column.rth0s.to_bits(), full.rth0s[surface].to_bits());
    }

    #[test]
    fn open_pressure_fill_is_source_ordered() {
        let (nk, nj, ni) = (2, 2, 3);
        let mut value = vec![0.0_f32; (nk + 2) * (nj + 2) * (ni + 2)];
        for k in 1..=nk {
            for j in 1..=nj {
                for i in 1..=ni {
                    value[index3(k, j, i, nj + 2, ni + 2)] = (100 * k + 10 * j + i) as f32;
                }
            }
        }
        bcp_open(&mut value, nk, nj, ni);
        assert_eq!(value[index3(1, 0, 0, nj + 2, ni + 2)], 111.0);
        assert_eq!(value[index3(1, nj + 1, ni + 1, nj + 2, ni + 2)], 123.0);
        assert_eq!(value[index3(0, 0, 0, nj + 2, ni + 2)].to_bits(), 0);
    }

    #[test]
    fn source_open_scalar_leaves_unwritten_corners_zero() {
        let (nk, nj, ni) = (1, 2, 2);
        let ny = nj + 2 * NGXY;
        let nx = ni + 2 * NGXY;
        let mut value = vec![0.0_f32; (nk + 2) * ny * nx];
        for j in 0..nj {
            for i in 0..ni {
                value[index3(1, j + NGXY, i + NGXY, ny, nx)] = 7.0;
            }
        }
        bcs_open(&mut value, nk, nj, ni);
        assert_eq!(value[index3(1, native_xy(0), native_xy(0), ny, nx)], 7.0);
        assert_eq!(value[index3(1, 0, 0, ny, nx)].to_bits(), 0);
    }

    #[test]
    fn recovery_metric_reports_one_ulp() {
        let mut metric = RecoveryMetric::default();
        metric.observe(1.0, f32::from_bits(1.0_f32.to_bits() + 1));
        assert_eq!(metric.mismatch_count, 1);
        assert_eq!(metric.max_ulp, 1);
    }
}
