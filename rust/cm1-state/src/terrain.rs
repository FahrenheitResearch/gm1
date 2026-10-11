//! CM1 r21.1 terrain-coordinate and static-base auxiliary transport.
//!
//! This module evaluates the pinned default-REAL formulas in Rust from the
//! independently hashed real checkpoint.  It deliberately keeps the current
//! one-dimensional `rho0` contract as a horizontal broadcast.  That is a
//! faithful transform of the supplied carrier, not a claim that CM1's final
//! terrain-varying hydrostatic base state has been reconstructed.

use crate::forecast::{
    read_forecast_checkpoint_embedded, ForecastArray, ForecastContract, ForecastFrame,
    ForecastReceipt, SchedulerScalarValue,
};
use crate::json::JsonValue;
use crate::sha256::{self, Sha256};
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const TERRAIN_STATIC_SCHEMA: &str = "gpuwm-cm1.terrain-static-export/v1";
pub const TERRAIN_STATIC_RECEIPT_SCHEMA: &str = "gpuwm-cm1.terrain-static-export-receipt/v1";
pub const TERRAIN_STATIC_CARRIER_ABI: &str = "gpuwm-cm1.terrain-static-carriers/v1";
pub const TERRAIN_STATIC_PAYLOAD_FILE: &str = "terrain-static.f32le";
pub const TERRAIN_STATIC_MANIFEST_FILE: &str = "manifest.json";
pub const TERRAIN_STATIC_RECEIPT_FILE: &str = "receipt.json";
pub const TERRAIN_STATIC_PUBLICATION_CONTRACT: &str = "write the binary32 payload, canonical manifest, and canonical receipt into a new sibling staging directory; synchronize every file; rename the directory only after all hashes validate; refuse an existing destination";
pub const TERRAIN_STATIC_BASE_POLICY: &str = "exact horizontal broadcast of the checkpoint rho0 profile followed by source-ordered rr0, c1/c2 face interpolation, and bottom/top Fornberg extrapolation; replaceable by a future source-backed terrain-varying hydrostatic base producer";
pub const TERRAIN_STATIC_CLAIM_LIMIT: &str = "CM1 r21.1 physical-domain terrain coordinates and static auxiliaries from real terrain plus the supplied one-dimensional rho0 profile; no CM1 halo storage, hydrostatic terrain-base reconstruction, pressure rebalance, boundary-history state, dynamics, physics, trajectory, forecast, or model-validation claim";

pub const TERRAIN_STATIC_CARRIER_NAMES: [&str; 25] = [
    "terrain_msl",
    "zh",
    "zf",
    "mh",
    "rmh",
    "mf",
    "rmf",
    "c1",
    "c2",
    "rho0_3d",
    "rr0",
    "rf0",
    "rrf0",
    "gz",
    "rgz",
    "gzu",
    "rgzu",
    "gzv",
    "rgzv",
    "dzdx",
    "dzdy",
    "gx",
    "gxu",
    "gy",
    "gyv",
];

const SOURCE_PROFILE_ID: &str = "hrrr-cm1-real-state-v1";
const REAL_NX: usize = 300;
const REAL_NY: usize = 300;
const REAL_NZ: usize = 75;
const REAL_DX_M: f64 = 100.0;
const REAL_DY_M: f64 = 100.0;
const REAL_DZ_M: f64 = 200.0;
const ITEMSIZE: usize = 4;
const IO_CHUNK_WORDS: usize = 64 * 1024;

const AUTHORITY_TARBALL_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
const AUTHORITY_FILES: [(&str, &str, &str); 5] = [
    (
        "src/adv_routines.F",
        "265723c8b19e300592ddbad3839ee689aa4933ac2f0ae236c61f4d5d40ef7f00",
        "5990-6037,9255-9322",
    ),
    (
        "src/base.F",
        "9c88a1021ddde22d02680786246c52bcffb040cbd72c3c4708f24fe24eec32ef",
        "2160-2190",
    ),
    (
        "src/bc.F",
        "5b7353a08b13eb9f69e4b89e250aec0e7918df3c93bd48c8539ab850787716e3",
        "12-172",
    ),
    (
        "src/init_terrain.F",
        "813e579983c0f55347d5eb54828709eedb5911a08587b216ea668c9d03cbca7c",
        "175-297",
    ),
    (
        "src/param.F",
        "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349",
        "6325-6327,7789-7812,8162-8207,8464-8492,8603-8627,8988-9062",
    ),
];

const SOURCE_ARRAYS: [&str; 22] = [
    "pi0",
    "pp3d",
    "prs0",
    "qc",
    "qg",
    "qi",
    "qr",
    "qs",
    "qv",
    "qv0",
    "refc_dbz",
    "refd_1000m_agl_dbz",
    "refd_4000m_agl_dbz",
    "rho0",
    "terrain_msl",
    "th0",
    "th3d",
    "u0",
    "u3d",
    "v0",
    "v3d",
    "w3d",
];

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerrainStaticAdmission {
    Generic,
    RealProduct,
}

impl TerrainStaticAdmission {
    fn as_str(self) -> &'static str {
        match self {
            Self::Generic => "generic-schema",
            Self::RealProduct => "real-product-300x300x75",
        }
    }
}

/// Explicit terrain metric configuration.  No default is provided because
/// both the flux order and lateral boundary tuple change the metric bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerrainStaticConfig {
    pub hadvordrs: u8,
    pub hadvordrv: u8,
    /// CM1 order: west, east, south, north.
    pub lateral_bc: [u8; 4],
}

impl TerrainStaticConfig {
    pub fn new(hadvordrs: u8, hadvordrv: u8, lateral_bc: [u8; 4]) -> Result<Self> {
        let value = Self {
            hadvordrs,
            hadvordrv,
            lateral_bc,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(self) -> Result<()> {
        if (self.hadvordrs, self.hadvordrv) != (5, 5) {
            return Err(invalid(format!(
                "terrain-static metric admission supports hadvordrs=5 and hadvordrv=5 only; got ({},{}), which would select a different zsgrad flux stencil",
                self.hadvordrs, self.hadvordrv
            )));
        }
        if self.lateral_bc != [2, 2, 2, 2] {
            return Err(invalid(format!(
                "terrain-static metric admission supports CM1 open bc2d tuple [2,2,2,2] only; got {:?}, which would change terrain halo values",
                self.lateral_bc
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HorizontalCrop {
    pub j_start: usize,
    pub i_start: usize,
    pub ny: usize,
    pub nx: usize,
}

impl HorizontalCrop {
    pub fn full(ny: usize, nx: usize) -> Self {
        Self {
            j_start: 0,
            i_start: 0,
            ny,
            nx,
        }
    }

    fn validate(self, ny: usize, nx: usize) -> Result<()> {
        if self.ny == 0 || self.nx == 0 {
            return Err(invalid("terrain-static crop extents must be positive"));
        }
        let j_end = self
            .j_start
            .checked_add(self.ny)
            .ok_or_else(|| invalid("terrain-static crop j range overflows"))?;
        let i_end = self
            .i_start
            .checked_add(self.nx)
            .ok_or_else(|| invalid("terrain-static crop i range overflows"))?;
        if j_end > ny || i_end > nx {
            return Err(invalid(format!(
                "terrain-static crop j={}..{}, i={}..{} exceeds source grid [{ny},{nx}]",
                self.j_start, j_end, self.i_start, i_end
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TerrainStaticCarrier {
    name: &'static str,
    role: &'static str,
    source_arrays: &'static [&'static str],
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl TerrainStaticCarrier {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn role(&self) -> &'static str {
        self.role
    }

    pub fn source_arrays(&self) -> &'static [&'static str] {
        self.source_arrays
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Clone, Debug)]
pub struct TerrainStaticFrame {
    source_shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    crop: HorizontalCrop,
    config: TerrainStaticConfig,
    bottom_fornberg: [f32; 3],
    top_fornberg: [f32; 3],
    carriers: Vec<TerrainStaticCarrier>,
}

impl TerrainStaticFrame {
    pub fn source_shape_zyx(&self) -> [usize; 3] {
        self.source_shape_zyx
    }

    pub fn spacing_m(&self) -> [f64; 3] {
        self.spacing_m
    }

    pub fn crop(&self) -> HorizontalCrop {
        self.crop
    }

    pub fn config(&self) -> TerrainStaticConfig {
        self.config
    }

    pub fn bottom_fornberg(&self) -> [f32; 3] {
        self.bottom_fornberg
    }

    pub fn top_fornberg(&self) -> [f32; 3] {
        self.top_fornberg
    }

    pub fn carriers(&self) -> &[TerrainStaticCarrier] {
        &self.carriers
    }

    pub fn carrier(&self, name: &str) -> Option<&TerrainStaticCarrier> {
        self.carriers.iter().find(|carrier| carrier.name == name)
    }
}

/// Evaluate the source-ordered physical-domain auxiliaries without I/O.
///
/// `source_shape_zyx` names the complete checkpoint grid; `crop` may select a
/// horizontal rectangle while retaining all vertical levels and the terminal
/// u/v faces required by the metric staggering.
#[allow(clippy::too_many_arguments)]
pub fn compute_terrain_static(
    terrain_msl: &[f32],
    rho0_profile: &[f32],
    source_shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    crop: HorizontalCrop,
    config: TerrainStaticConfig,
    admission: TerrainStaticAdmission,
) -> Result<TerrainStaticFrame> {
    compute_static_geometry(
        terrain_msl,
        rho0_profile,
        source_shape_zyx,
        spacing_m,
        crop,
        config,
        admission,
        true,
        None,
    )
}

/// Build the `terrain_flag=false, stretch_z=0` static carriers.
///
/// CM1 param.F:7789-7811 retains its uniform coordinates when the terrain
/// dispatch at init_terrain.F:175-196 is disabled.  A zero-height terrain-following
/// surface is a different arithmetic branch: `sigma * top / top` can round
/// away from `sigma`, so it must not be substituted for the flat branch.
pub(crate) fn compute_flat_uniform_static(
    rho0_profile: &[f32],
    source_shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    crop: HorizontalCrop,
    config: TerrainStaticConfig,
) -> Result<TerrainStaticFrame> {
    let terrain_words = checked_product(&source_shape_zyx[1..], "flat surface")?;
    let terrain_msl = vec![0.0_f32; terrain_words];
    compute_static_geometry(
        &terrain_msl,
        rho0_profile,
        source_shape_zyx,
        spacing_m,
        crop,
        config,
        TerrainStaticAdmission::Generic,
        false,
        None,
    )
}

/// One declared CM1 vertical coordinate, in the two arrays `param.F` keeps.
///
/// `sigma` is `sigma(1..nk)` and `sigmaf` is `sigmaf(1..nk+1)`, both in metres
/// (`param.F:8189-8191` and `:7947`).  `top_ghost_m` is `zf(nk+2)`, which
/// `param.F:8166` forms as `zf(nk+1)+(zf(nk+1)-zf(nk))` and which the face
/// metric `mf(nk+1)` at `param.F:8176` then reaches through `zh(nk+1)`.
/// `dz_reference_m` is the namelist `dz` that `param.F:8170` divides every
/// `mh` word by; on a stretched grid it is not any layer's thickness.
#[derive(Debug, Clone)]
pub struct DeclaredVerticalCoordinate {
    pub sigma: Vec<f32>,
    pub sigmaf: Vec<f32>,
    pub dz_reference_m: f32,
    pub top_ghost_m: f32,
}

impl DeclaredVerticalCoordinate {
    /// Build from `zh(1..nk)` and `zf(1..nk+1)`, forming `param.F:8166`'s ghost.
    pub fn new(mass_heights_m: &[f32], face_heights_m: &[f32], dz_reference_m: f32) -> Result<Self> {
        let nz = mass_heights_m.len();
        if nz < 4 || face_heights_m.len() != nz + 1 {
            return Err(invalid(
                "declared vertical coordinate needs zh(1..nk) with nk>=4 and zf(1..nk+1)",
            ));
        }
        if !dz_reference_m.is_finite() || dz_reference_m <= 0.0 {
            return Err(invalid(
                "declared vertical dz reference must be positive and finite",
            ));
        }
        if face_heights_m[0] != 0.0 {
            return Err(invalid(
                "declared vertical coordinate must begin at the ground, zf(1)=0",
            ));
        }
        for (k, window) in face_heights_m.windows(2).enumerate() {
            if !window[0].is_finite() || !window[1].is_finite() || !(window[1] > window[0]) {
                return Err(invalid(format!(
                    "declared vertical layer {} has non-positive depth; param.F:8212-8224 stops CM1 here",
                    k + 1
                )));
            }
        }
        for (k, &height) in mass_heights_m.iter().enumerate() {
            if !height.is_finite() || height <= face_heights_m[k] || height >= face_heights_m[k + 1]
            {
                return Err(invalid(format!(
                    "declared vertical mass height {} does not lie inside its own layer",
                    k + 1
                )));
            }
        }
        let top_ghost_m = face_heights_m[nz] + (face_heights_m[nz] - face_heights_m[nz - 1]);
        Ok(Self {
            sigma: mass_heights_m.to_vec(),
            sigmaf: face_heights_m.to_vec(),
            dz_reference_m,
            top_ghost_m,
        })
    }
}

/// The `terrain_flag=false` static carriers over a declared vertical grid.
///
/// This is `compute_flat_uniform_static` with `param.F:7789-7811`'s coordinate
/// replaced by the caller's.  Everything below the coordinate -- `mh`, `mf`,
/// `c1`, `c2`, the `param.F:8985-9060` Fornberg weights and the `rf0` they
/// extrapolate -- is the same arithmetic, which is exactly why the weights
/// must be recomputed rather than pinned: on a non-uniform column the three
/// nodes they fit are no longer equally spaced.
pub(crate) fn compute_flat_static_levels(
    rho0_profile: &[f32],
    source_shape_zyx: [usize; 3],
    horizontal_spacing_m: [f64; 2],
    vertical: &DeclaredVerticalCoordinate,
    crop: HorizontalCrop,
    config: TerrainStaticConfig,
) -> Result<TerrainStaticFrame> {
    let terrain_words = checked_product(&source_shape_zyx[1..], "flat surface")?;
    let terrain_msl = vec![0.0_f32; terrain_words];
    if vertical.sigma.len() != source_shape_zyx[0] {
        return Err(invalid(format!(
            "declared vertical coordinate has {} scalar levels; this grid has {}",
            vertical.sigma.len(),
            source_shape_zyx[0]
        )));
    }
    compute_static_geometry(
        &terrain_msl,
        rho0_profile,
        source_shape_zyx,
        [
            horizontal_spacing_m[0],
            horizontal_spacing_m[1],
            f64::from(vertical.dz_reference_m),
        ],
        crop,
        config,
        TerrainStaticAdmission::Generic,
        false,
        Some(vertical),
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_static_geometry(
    terrain_msl: &[f32],
    rho0_profile: &[f32],
    source_shape_zyx: [usize; 3],
    spacing_m: [f64; 3],
    crop: HorizontalCrop,
    config: TerrainStaticConfig,
    admission: TerrainStaticAdmission,
    terrain_following: bool,
    declared: Option<&DeclaredVerticalCoordinate>,
) -> Result<TerrainStaticFrame> {
    config.validate()?;
    let [nz, ny, nx] = source_shape_zyx;
    if nz < 4 || ny == 0 || nx == 0 {
        return Err(invalid(
            "terrain-static source requires at least four vertical levels and positive horizontal extents",
        ));
    }
    crop.validate(ny, nx)?;
    let terrain_words = ny
        .checked_mul(nx)
        .ok_or_else(|| invalid("terrain-static source surface size overflows"))?;
    if terrain_msl.len() != terrain_words {
        return Err(invalid(format!(
            "terrain-static terrain contains {} words; source grid requires {terrain_words}",
            terrain_msl.len()
        )));
    }
    if rho0_profile.len() != nz {
        return Err(invalid(format!(
            "terrain-static rho0 profile contains {} words; source grid requires {nz}",
            rho0_profile.len()
        )));
    }
    for (axis, value) in ["dx", "dy", "dz"].into_iter().zip(spacing_m) {
        if !value.is_finite() || value <= 0.0 || (value as f32) as f64 != value {
            return Err(invalid(format!(
                "terrain-static {axis} must be a positive finite value exactly representable in CM1 default REAL"
            )));
        }
    }
    if admission == TerrainStaticAdmission::RealProduct
        && (source_shape_zyx != [REAL_NZ, REAL_NY, REAL_NX]
            || spacing_m.map(f64::to_bits) != [REAL_DX_M, REAL_DY_M, REAL_DZ_M].map(f64::to_bits))
    {
        return Err(invalid(
            "real-product terrain-static admission requires [75,300,300] at 100x100x200 m",
        ));
    }

    let dx = spacing_m[0] as f32;
    let dy = spacing_m[1] as f32;
    let dz = spacing_m[2] as f32;
    let rdx = 1.0_f32 / dx;
    let rdy = 1.0_f32 / dy;
    // param.F:8193.  maxz is the top w level, which is dz*nz only when the
    // grid is unstretched.
    let top = match declared {
        None => dz * nz as f32,
        Some(coordinate) => coordinate.sigmaf[nz],
    };
    if !top.is_finite() || top <= 0.0 {
        return Err(invalid("terrain-static model top is invalid in binary32"));
    }
    if declared.is_some() && terrain_following {
        return Err(invalid(
            "terrain-following metrics over a declared vertical coordinate are not built here; init_terrain.F:175-196 composes sigma with the surface and that composition has no oracle in this port",
        ));
    }
    for (index, value) in terrain_msl.iter().copied().enumerate() {
        if !value.is_finite() || value >= top {
            return Err(invalid(format!(
                "terrain-static terrain word {index} is non-finite or reaches the model top; bits={:08x}",
                value.to_bits()
            )));
        }
    }
    for (index, value) in rho0_profile.iter().copied().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid(format!(
                "terrain-static rho0 profile word {index} must be positive finite; bits={:08x}",
                value.to_bits()
            )));
        }
    }

    // param.F:7789-7811 when nothing is declared; the caller's own coordinate
    // otherwise.  The two are one code path from here down: every metric,
    // weight and interpolation below reads sigma/sigmaf and never rebuilds a
    // level from dz.
    let (sigma, sigmaf) = match declared {
        None => {
            let mut sigmaf = Vec::with_capacity(nz + 1);
            for k in 0..=nz {
                sigmaf.push(dz * k as f32);
            }
            let mut sigma = Vec::with_capacity(nz);
            for k in 0..nz {
                sigma.push(0.5_f32 * (sigmaf[k] + sigmaf[k + 1]));
            }
            (sigma, sigmaf)
        }
        Some(coordinate) => (coordinate.sigma.clone(), coordinate.sigmaf.clone()),
    };
    let bottom_fornberg = extrapolation_weights(&sigma, &sigmaf, false)?;
    let top_fornberg = extrapolation_weights(&sigma, &sigmaf, true)?;

    let mass_words = checked_product(&[nz, crop.ny, crop.nx], "mass auxiliary")?;
    let zface_words = checked_product(&[nz + 1, crop.ny, crop.nx], "z-face auxiliary")?;
    let mut terrain = Vec::with_capacity(crop.ny * crop.nx);
    let mut zh = vec![0.0_f32; mass_words];
    let mut zf = vec![0.0_f32; zface_words];
    let mut mh = vec![0.0_f32; mass_words];
    let mut rmh = vec![0.0_f32; mass_words];
    let mut mf = vec![0.0_f32; zface_words];
    let mut rmf = vec![0.0_f32; zface_words];
    let mut c1 = vec![0.0_f32; zface_words];
    let mut c2 = vec![0.0_f32; zface_words];
    let mut rho0_3d = vec![0.0_f32; mass_words];
    let mut rr0 = vec![0.0_f32; mass_words];
    let mut rf0 = vec![0.0_f32; zface_words];
    let mut rrf0 = vec![0.0_f32; zface_words];

    for out_j in 0..crop.ny {
        let global_j = crop.j_start + out_j;
        for out_i in 0..crop.nx {
            let global_i = crop.i_start + out_i;
            let surface = terrain_msl[global_j * nx + global_i];
            terrain.push(surface);
            let mut column_zh = vec![0.0_f32; nz];
            let mut column_zf = vec![0.0_f32; nz + 1];
            for k in 0..=nz {
                column_zf[k] = if terrain_following {
                    terrain_height(surface, sigmaf[k], top)
                } else {
                    sigmaf[k]
                };
                zf[index3(k, out_j, out_i, crop.ny, crop.nx)] = column_zf[k];
            }
            for k in 0..nz {
                column_zh[k] = if terrain_following {
                    terrain_height(surface, sigma[k], top)
                } else {
                    sigma[k]
                };
                let index = index3(k, out_j, out_i, crop.ny, crop.nx);
                zh[index] = column_zh[k];
                let layer_metric = dz / (column_zf[k + 1] - column_zf[k]);
                mh[index] = layer_metric;
                rmh[index] = 1.0_f32 / layer_metric;
                rho0_3d[index] = rho0_profile[k];
                rr0[index] = 1.0_f32 / rho0_profile[k];
            }
            let zf_bottom_ghost = column_zf[0] - (column_zf[1] - column_zf[0]);
            let zf_top_ghost = match declared {
                // param.F:8166 in both branches; for the uniform coordinate
                // the established dz*(nz+1) spelling is kept word for word.
                Some(coordinate) => coordinate.top_ghost_m,
                None if terrain_following => {
                    column_zf[nz] + (column_zf[nz] - column_zf[nz - 1])
                }
                None => dz * (nz + 1) as f32,
            };
            let zh_bottom_ghost = 0.5_f32 * (zf_bottom_ghost + column_zf[0]);
            let zh_top_ghost = 0.5_f32 * (column_zf[nz] + zf_top_ghost);
            for face in 0..=nz {
                let lower = if face == 0 {
                    zh_bottom_ghost
                } else {
                    column_zh[face - 1]
                };
                let upper = if face == nz {
                    zh_top_ghost
                } else {
                    column_zh[face]
                };
                let index = index3(face, out_j, out_i, crop.ny, crop.nx);
                let face_metric = dz / (upper - lower);
                mf[index] = face_metric;
                rmf[index] = 1.0_f32 / face_metric;
                let upper_weight = (column_zf[face] - lower) / (upper - lower);
                c2[index] = upper_weight;
                c1[index] = 1.0_f32 - upper_weight;
                let density = if face == 0 {
                    bottom_fornberg[0] * rho0_profile[0]
                        + bottom_fornberg[1] * rho0_profile[1]
                        + bottom_fornberg[2] * rho0_profile[2]
                } else if face == nz {
                    top_fornberg[0] * rho0_profile[nz - 1]
                        + top_fornberg[1] * rho0_profile[nz - 2]
                        + top_fornberg[2] * rho0_profile[nz - 3]
                } else {
                    c1[index] * rho0_profile[face - 1] + c2[index] * rho0_profile[face]
                };
                if !density.is_finite() || density <= 0.0 {
                    return Err(invalid(format!(
                        "terrain-static rf0 is non-positive at face {face}, j={global_j}, i={global_i}; bits={:08x}",
                        density.to_bits()
                    )));
                }
                rf0[index] = density;
                rrf0[index] = 1.0_f32 / density;
            }
        }
    }

    let surface_words = crop.ny * crop.nx;
    let uface2_words = crop.ny * (crop.nx + 1);
    let vface2_words = (crop.ny + 1) * crop.nx;
    let mut gz = vec![0.0_f32; surface_words];
    let mut rgz = vec![0.0_f32; surface_words];
    let mut gzu = vec![0.0_f32; uface2_words];
    let mut rgzu = vec![0.0_f32; uface2_words];
    let mut gzv = vec![0.0_f32; vface2_words];
    let mut rgzv = vec![0.0_f32; vface2_words];
    let mut dzdx = vec![0.0_f32; surface_words];
    let mut dzdy = vec![0.0_f32; surface_words];

    let terrain_at = |i: isize, j: isize| -> f32 {
        let bounded_i = i.clamp(0, nx as isize - 1) as usize;
        let bounded_j = j.clamp(0, ny as isize - 1) as usize;
        terrain_msl[bounded_j * nx + bounded_i]
    };
    let rgz_at = |i: isize, j: isize| -> f32 {
        let surface = terrain_at(i, j);
        (top - surface) / top
    };
    let rgzu_at = |face_i: isize, j: isize| -> f32 {
        let average = 0.5_f32 * (terrain_at(face_i - 1, j) + terrain_at(face_i, j));
        (top - average) / top
    };
    let rgzv_at = |i: isize, face_j: isize| -> f32 {
        let average = 0.5_f32 * (terrain_at(i, face_j - 1) + terrain_at(i, face_j));
        (top - average) / top
    };
    let x_flux = |face_i: isize, j: isize| -> f32 {
        flx6(
            terrain_at(face_i - 3, j),
            terrain_at(face_i - 2, j),
            terrain_at(face_i - 1, j),
            terrain_at(face_i, j),
            terrain_at(face_i + 1, j),
            terrain_at(face_i + 2, j),
        )
    };
    let y_flux = |i: isize, face_j: isize| -> f32 {
        flx6(
            terrain_at(i, face_j - 3),
            terrain_at(i, face_j - 2),
            terrain_at(i, face_j - 1),
            terrain_at(i, face_j),
            terrain_at(i, face_j + 1),
            terrain_at(i, face_j + 2),
        )
    };

    for out_j in 0..crop.ny {
        let j = (crop.j_start + out_j) as isize;
        for out_i in 0..crop.nx {
            let i = (crop.i_start + out_i) as isize;
            let index = out_j * crop.nx + out_i;
            let surface = terrain_at(i, j);
            gz[index] = top / (top - surface);
            rgz[index] = (top - surface) / top;
            dzdx[index] = (x_flux(i + 1, j) - x_flux(i, j)) * rdx;
            dzdy[index] = (y_flux(i, j + 1) - y_flux(i, j)) * rdy;
        }
        for out_face_i in 0..=crop.nx {
            let face_i = (crop.i_start + out_face_i) as isize;
            let average = 0.5_f32 * (terrain_at(face_i - 1, j) + terrain_at(face_i, j));
            let index = out_j * (crop.nx + 1) + out_face_i;
            gzu[index] = top / (top - average);
            rgzu[index] = (top - average) / top;
        }
    }
    for out_face_j in 0..=crop.ny {
        let face_j = (crop.j_start + out_face_j) as isize;
        for out_i in 0..crop.nx {
            let i = (crop.i_start + out_i) as isize;
            let average = 0.5_f32 * (terrain_at(i, face_j - 1) + terrain_at(i, face_j));
            let index = out_face_j * crop.nx + out_i;
            gzv[index] = top / (top - average);
            rgzv[index] = (top - average) / top;
        }
    }

    let mut gx = vec![0.0_f32; zface_words];
    let mut gy = vec![0.0_f32; zface_words];
    let mut gxu = vec![0.0_f32; (nz + 1) * crop.ny * (crop.nx + 1)];
    let mut gyv = vec![0.0_f32; (nz + 1) * (crop.ny + 1) * crop.nx];
    for (k, sigma_face) in sigmaf.iter().copied().enumerate() {
        let height_factor = top - sigma_face;
        for out_j in 0..crop.ny {
            let j = (crop.j_start + out_j) as isize;
            for out_i in 0..crop.nx {
                let i = (crop.i_start + out_i) as isize;
                let surface_index = out_j * crop.nx + out_i;
                let index = index3(k, out_j, out_i, crop.ny, crop.nx);
                gx[index] =
                    height_factor * gz[surface_index] * (rgzu_at(i + 1, j) - rgzu_at(i, j)) * rdx;
                gy[index] =
                    height_factor * gz[surface_index] * (rgzv_at(i, j + 1) - rgzv_at(i, j)) * rdy;
            }
            for out_face_i in 0..=crop.nx {
                let face_i = (crop.i_start + out_face_i) as isize;
                let face_index = out_j * (crop.nx + 1) + out_face_i;
                let index = index3(k, out_j, out_face_i, crop.ny, crop.nx + 1);
                gxu[index] = height_factor
                    * gzu[face_index]
                    * (rgz_at(face_i, j) - rgz_at(face_i - 1, j))
                    * rdx;
            }
        }
        for out_face_j in 0..=crop.ny {
            let face_j = (crop.j_start + out_face_j) as isize;
            for out_i in 0..crop.nx {
                let i = (crop.i_start + out_i) as isize;
                let face_index = out_face_j * crop.nx + out_i;
                let index = index3(k, out_face_j, out_i, crop.ny + 1, crop.nx);
                gyv[index] = height_factor
                    * gzv[face_index]
                    * (rgz_at(i, face_j) - rgz_at(i, face_j - 1))
                    * rdy;
            }
        }
    }

    let carriers = vec![
        carrier(
            "terrain_msl",
            "source-real-terrain",
            &["terrain_msl"],
            vec![crop.ny, crop.nx],
            terrain,
        ),
        carrier(
            "zh",
            "terrain-mass-height",
            &["terrain_msl", "target_dz_m"],
            vec![nz, crop.ny, crop.nx],
            zh,
        ),
        carrier(
            "zf",
            "terrain-z-face-height",
            &["terrain_msl", "target_dz_m"],
            vec![nz + 1, crop.ny, crop.nx],
            zf,
        ),
        carrier(
            "mh",
            "mass-level-vertical-metric",
            &["terrain_msl", "target_dz_m"],
            vec![nz, crop.ny, crop.nx],
            mh,
        ),
        carrier(
            "rmh",
            "reciprocal-mass-level-vertical-metric",
            &["mh"],
            vec![nz, crop.ny, crop.nx],
            rmh,
        ),
        carrier(
            "mf",
            "z-face-vertical-metric",
            &["terrain_msl", "target_dz_m"],
            vec![nz + 1, crop.ny, crop.nx],
            mf,
        ),
        carrier(
            "rmf",
            "reciprocal-z-face-vertical-metric",
            &["mf"],
            vec![nz + 1, crop.ny, crop.nx],
            rmf,
        ),
        carrier(
            "c1",
            "lower-mass-to-z-face-weight",
            &["zh", "zf"],
            vec![nz + 1, crop.ny, crop.nx],
            c1,
        ),
        carrier(
            "c2",
            "upper-mass-to-z-face-weight",
            &["zh", "zf"],
            vec![nz + 1, crop.ny, crop.nx],
            c2,
        ),
        carrier(
            "rho0_3d",
            "broadcast-source-base-density",
            &["rho0"],
            vec![nz, crop.ny, crop.nx],
            rho0_3d,
        ),
        carrier(
            "rr0",
            "reciprocal-broadcast-base-density",
            &["rho0"],
            vec![nz, crop.ny, crop.nx],
            rr0,
        ),
        carrier(
            "rf0",
            "z-face-broadcast-base-density",
            &["rho0", "c1", "c2"],
            vec![nz + 1, crop.ny, crop.nx],
            rf0,
        ),
        carrier(
            "rrf0",
            "reciprocal-z-face-broadcast-base-density",
            &["rf0"],
            vec![nz + 1, crop.ny, crop.nx],
            rrf0,
        ),
        carrier(
            "gz",
            "mass-horizontal-terrain-Jacobian",
            &["terrain_msl"],
            vec![crop.ny, crop.nx],
            gz,
        ),
        carrier(
            "rgz",
            "reciprocal-mass-horizontal-terrain-Jacobian",
            &["terrain_msl"],
            vec![crop.ny, crop.nx],
            rgz,
        ),
        carrier(
            "gzu",
            "u-face-horizontal-terrain-Jacobian",
            &["terrain_msl"],
            vec![crop.ny, crop.nx + 1],
            gzu,
        ),
        carrier(
            "rgzu",
            "reciprocal-u-face-horizontal-terrain-Jacobian",
            &["terrain_msl"],
            vec![crop.ny, crop.nx + 1],
            rgzu,
        ),
        carrier(
            "gzv",
            "v-face-horizontal-terrain-Jacobian",
            &["terrain_msl"],
            vec![crop.ny + 1, crop.nx],
            gzv,
        ),
        carrier(
            "rgzv",
            "reciprocal-v-face-horizontal-terrain-Jacobian",
            &["terrain_msl"],
            vec![crop.ny + 1, crop.nx],
            rgzv,
        ),
        carrier(
            "dzdx",
            "source-flx6-terrain-x-gradient",
            &["terrain_msl"],
            vec![crop.ny, crop.nx],
            dzdx,
        ),
        carrier(
            "dzdy",
            "source-flx6-terrain-y-gradient",
            &["terrain_msl"],
            vec![crop.ny, crop.nx],
            dzdy,
        ),
        carrier(
            "gx",
            "mass-horizontal-x-transform-metric",
            &["terrain_msl", "gz", "rgzu"],
            vec![nz + 1, crop.ny, crop.nx],
            gx,
        ),
        carrier(
            "gxu",
            "u-face-x-transform-metric",
            &["terrain_msl", "gzu", "rgz"],
            vec![nz + 1, crop.ny, crop.nx + 1],
            gxu,
        ),
        carrier(
            "gy",
            "mass-horizontal-y-transform-metric",
            &["terrain_msl", "gz", "rgzv"],
            vec![nz + 1, crop.ny, crop.nx],
            gy,
        ),
        carrier(
            "gyv",
            "v-face-y-transform-metric",
            &["terrain_msl", "gzv", "rgz"],
            vec![nz + 1, crop.ny + 1, crop.nx],
            gyv,
        ),
    ];
    if carriers.iter().map(|item| item.name).collect::<Vec<_>>() != TERRAIN_STATIC_CARRIER_NAMES {
        return Err(invalid("terrain-static internal carrier order drifted"));
    }
    for item in &carriers {
        validate_carrier(item)?;
    }

    Ok(TerrainStaticFrame {
        source_shape_zyx,
        spacing_m,
        crop,
        config,
        bottom_fornberg,
        top_fornberg,
        carriers,
    })
}

fn carrier(
    name: &'static str,
    role: &'static str,
    source_arrays: &'static [&'static str],
    shape: Vec<usize>,
    values: Vec<f32>,
) -> TerrainStaticCarrier {
    TerrainStaticCarrier {
        name,
        role,
        source_arrays,
        shape,
        values,
    }
}

fn validate_carrier(carrier: &TerrainStaticCarrier) -> Result<()> {
    let expected = checked_product(&carrier.shape, carrier.name)?;
    if carrier.values.len() != expected {
        return Err(invalid(format!(
            "terrain-static carrier {} has {} words; shape {:?} requires {expected}",
            carrier.name,
            carrier.values.len(),
            carrier.shape
        )));
    }
    for (index, value) in carrier.values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(invalid(format!(
                "terrain-static carrier {} has non-finite word {index}",
                carrier.name
            )));
        }
    }
    match carrier.name {
        "mh" | "rmh" | "mf" | "rmf" | "rho0_3d" | "rr0" | "rf0" | "rrf0" | "gz" | "rgz" | "gzu"
        | "rgzu" | "gzv" | "rgzv" => {
            if carrier.values.iter().any(|value| *value <= 0.0) {
                return Err(invalid(format!(
                    "terrain-static positive carrier {} contains a non-positive word",
                    carrier.name
                )));
            }
        }
        "c1" | "c2" => {
            if carrier
                .values
                .iter()
                .any(|value| !(0.0..=1.0).contains(value))
            {
                return Err(invalid(format!(
                    "terrain-static interpolation carrier {} leaves [0,1]",
                    carrier.name
                )));
            }
        }
        _ => {}
    }
    Ok(())
}

fn checked_product(shape: &[usize], label: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| invalid(format!("terrain-static {label} shape product overflows")))
    })
}

fn index3(k: usize, j: usize, i: usize, ny: usize, nx: usize) -> usize {
    (k * ny + j) * nx + i
}

fn terrain_height(surface: f32, sigma: f32, top: f32) -> f32 {
    surface + sigma * (top - surface) / top
}

fn flx6(s1: f32, s2: f32, s3: f32, s4: f32, s5: f32, s6: f32) -> f32 {
    (37.0_f32 / 60.0_f32) * (s4 + s3)
        + (-8.0_f32 / 60.0_f32) * (s5 + s2)
        + (1.0_f32 / 60.0_f32) * (s6 + s1)
}

pub(crate) fn extrapolation_weights(sigma: &[f32], sigmaf: &[f32], top: bool) -> Result<[f32; 3]> {
    let nz = sigma.len();
    if nz < 4 || sigmaf.len() != nz + 1 {
        return Err(invalid(
            "terrain-static Fornberg extrapolation requires four mass levels and matching faces",
        ));
    }
    let denominator = if top {
        sigmaf[nz] - sigmaf[nz - 1]
    } else {
        sigmaf[1] - sigmaf[0]
    };
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(invalid("terrain-static Fornberg spacing is invalid"));
    }
    // The source evaluates these default-REAL divisions before assigning the
    // results to DOUBLE PRECISION x0/alpha.
    let x0 = f64::from(if top {
        sigmaf[nz] / denominator
    } else {
        sigmaf[0] / denominator
    });
    let mut alpha = [0.0_f64; 4];
    for n in 0..4 {
        let source = if top { sigma[nz - 1 - n] } else { sigma[n] };
        alpha[n] = f64::from(source / denominator);
    }
    let delta = fornberg_delta(alpha, x0)?;
    Ok([
        delta[2][1][0] as f32,
        delta[2][1][1] as f32,
        delta[2][1][2] as f32,
    ])
}

/// `delta[n][m+1][nu]` mirrors `delta(n,m,nu)` with source m=-1..4.
fn fornberg_delta(alpha: [f64; 4], x0: f64) -> Result<[[[f64; 4]; 6]; 4]> {
    let mut delta = [[[0.0_f64; 4]; 6]; 4];
    delta[0][1][0] = 1.0;
    let mut b1 = 1.0_f64;
    for n in 1..=3 {
        let mut b2 = 1.0_f64;
        for nu in 0..n {
            let b3 = alpha[n] - alpha[nu];
            if !b3.is_finite() || b3 == 0.0 {
                return Err(invalid("terrain-static Fornberg nodes are duplicated"));
            }
            b2 *= b3;
            delta[n - 1][n + 1][nu] = 0.0;
            for m in 0..=n.min(4) {
                delta[n][m + 1][nu] = ((alpha[n] - x0) * delta[n - 1][m + 1][nu]
                    - m as f64 * delta[n - 1][m][nu])
                    / b3;
            }
        }
        for m in 0..=n.min(4) {
            delta[n][m + 1][n] = (b1 / b2)
                * (m as f64 * delta[n - 1][m][n - 1]
                    - (alpha[n - 1] - x0) * delta[n - 1][m + 1][n - 1]);
        }
        b1 = b2;
    }
    Ok(delta)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TerrainStaticReceipt {
    pub source_artifact_sha256: String,
    pub source_contract_sha256: String,
    pub payload_sha256: String,
    pub payload_nbytes: u64,
    pub manifest_sha256: String,
    pub manifest_nbytes: usize,
    pub carrier_count: usize,
    pub admission: TerrainStaticAdmission,
}

impl TerrainStaticReceipt {
    pub fn canonical_json(&self) -> String {
        let value = object(vec![
            ("admission", string(self.admission.as_str())),
            ("base_policy", string(TERRAIN_STATIC_BASE_POLICY)),
            ("carrier_abi", string(TERRAIN_STATIC_CARRIER_ABI)),
            ("carrier_count", number(self.carrier_count)),
            ("claim_limit", string(TERRAIN_STATIC_CLAIM_LIMIT)),
            ("manifest_nbytes", number(self.manifest_nbytes)),
            ("manifest_sha256", string(&self.manifest_sha256)),
            ("payload_nbytes", u64_number(self.payload_nbytes)),
            ("payload_sha256", string(&self.payload_sha256)),
            (
                "publication_contract",
                string(TERRAIN_STATIC_PUBLICATION_CONTRACT),
            ),
            ("schema", string(TERRAIN_STATIC_RECEIPT_SCHEMA)),
            (
                "source_artifact_sha256",
                string(&self.source_artifact_sha256),
            ),
            (
                "source_contract_sha256",
                string(&self.source_contract_sha256),
            ),
        ]);
        String::from_utf8(value.canonical_bytes())
            .expect("canonical terrain-static receipt is UTF-8")
    }
}

/// Independently verify one CM1FCP1 checkpoint, derive the terrain/static
/// carrier frame in Rust, and publish it atomically into a new directory.
#[allow(clippy::too_many_arguments)]
pub fn export_checkpoint_terrain_static_new(
    input: impl AsRef<Path>,
    output_directory: impl AsRef<Path>,
    expected_contract_sha256: &str,
    expected_artifact_sha256: &str,
    crop: Option<HorizontalCrop>,
    config: TerrainStaticConfig,
    admission: TerrainStaticAdmission,
) -> Result<TerrainStaticReceipt> {
    validate_sha256(
        expected_artifact_sha256,
        "expected checkpoint artifact SHA-256",
    )?;
    config.validate()?;
    let output = output_directory.as_ref();
    if fs::symlink_metadata(output).is_ok() {
        return Err(invalid(format!(
            "refusing to replace terrain-static destination {}",
            output.display()
        )));
    }
    let parent = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "terrain-static parent directory does not exist: {}",
            parent.display()
        )));
    }

    let (contract, restored) =
        read_forecast_checkpoint_embedded(input.as_ref(), expected_contract_sha256)?;
    if restored.receipt().artifact_sha256 != expected_artifact_sha256 {
        return Err(invalid(format!(
            "checkpoint artifact SHA-256 is {}; independently expected {expected_artifact_sha256}",
            restored.receipt().artifact_sha256
        )));
    }
    let ([nz, ny, nx], spacing_m) =
        validate_checkpoint_contract(&contract, restored.frame(), admission)?;
    let crop = crop.unwrap_or_else(|| HorizontalCrop::full(ny, nx));
    crop.validate(ny, nx)?;
    let arrays = array_map(restored.frame())?;
    let frame = compute_terrain_static(
        array(&arrays, "terrain_msl")?,
        array(&arrays, "rho0")?,
        [nz, ny, nx],
        spacing_m,
        crop,
        config,
        admission,
    )?;

    let staging = staging_path(output)?;
    fs::create_dir(&staging)?;
    let result = write_frame_into_directory(
        &staging,
        &frame,
        &contract,
        restored.frame(),
        restored.receipt(),
        admission,
    );
    match result {
        Ok(receipt) => {
            if let Err(error) = fs::rename(&staging, output) {
                let _ = fs::remove_dir_all(&staging);
                return Err(error.into());
            }
            Ok(receipt)
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            Err(error)
        }
    }
}

fn validate_checkpoint_contract(
    contract: &ForecastContract,
    frame: &ForecastFrame,
    admission: TerrainStaticAdmission,
) -> Result<([usize; 3], [f64; 3])> {
    if contract.profile_id() != SOURCE_PROFILE_ID {
        return Err(invalid(format!(
            "terrain-static export requires profile {SOURCE_PROFILE_ID:?}; got {:?}",
            contract.profile_id()
        )));
    }
    let actual = contract
        .arrays()
        .iter()
        .map(|spec| spec.name())
        .collect::<BTreeSet<_>>();
    let expected = SOURCE_ARRAYS.into_iter().collect::<BTreeSet<_>>();
    if actual != expected {
        let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
        let extra = actual.difference(&expected).copied().collect::<Vec<_>>();
        return Err(invalid(format!(
            "terrain-static source inventory mismatch; missing={missing:?}, extra={extra:?}"
        )));
    }
    let mass = shape(contract, "pp3d")?;
    if mass.len() != 3 {
        return Err(invalid("terrain-static pp3d must have z/y/x rank three"));
    }
    let [nz, ny, nx] = <[usize; 3]>::try_from(mass)
        .map_err(|_| invalid("terrain-static source mass shape conversion failed"))?;
    for name in ["th3d", "qv", "qc", "qr", "qi", "qs", "qg"] {
        require_shape(contract, name, &[nz, ny, nx])?;
    }
    for name in ["pi0", "prs0", "qv0", "rho0", "th0", "u0", "v0"] {
        require_shape(contract, name, &[nz])?;
    }
    require_shape(contract, "u3d", &[nz, ny, nx + 1])?;
    require_shape(contract, "v3d", &[nz, ny + 1, nx])?;
    require_shape(contract, "w3d", &[nz + 1, ny, nx])?;
    require_shape(contract, "terrain_msl", &[ny, nx])?;
    for name in ["refc_dbz", "refd_1000m_agl_dbz", "refd_4000m_agl_dbz"] {
        require_shape(contract, name, &[ny, nx])?;
    }
    if frame.clock().step() != 0 || frame.clock().model_time_s().to_bits() != 0 {
        return Err(invalid(
            "terrain-static initialization requires a step-zero, time-zero checkpoint",
        ));
    }
    require_scalar(frame, "boundary_ready", SchedulerScalarValue::Bool(false))?;
    require_scalar(
        frame,
        "forcing_valid_offset_s",
        SchedulerScalarValue::F64(0.0),
    )?;
    require_scalar(frame, "source_forecast_hour", SchedulerScalarValue::U64(0))?;
    let spacing_m = [
        scalar_f64(frame, "target_dx_m")?,
        scalar_f64(frame, "target_dy_m")?,
        scalar_f64(frame, "target_dz_m")?,
    ];
    if admission == TerrainStaticAdmission::RealProduct
        && ([nz, ny, nx] != [REAL_NZ, REAL_NY, REAL_NX]
            || spacing_m.map(f64::to_bits) != [REAL_DX_M, REAL_DY_M, REAL_DZ_M].map(f64::to_bits)
            || frame.clock().dt_s().to_bits() != 0.5_f64.to_bits())
    {
        return Err(invalid(
            "real-product terrain-static checkpoint must be [75,300,300], 100x100x200 m, dt=0.5 s",
        ));
    }
    Ok(([nz, ny, nx], spacing_m))
}

fn require_scalar(frame: &ForecastFrame, name: &str, expected: SchedulerScalarValue) -> Result<()> {
    let actual = frame
        .scheduler_scalars()
        .iter()
        .find(|scalar| scalar.name() == name)
        .ok_or_else(|| {
            invalid(format!(
                "terrain-static source lacks scheduler scalar {name:?}"
            ))
        })?
        .value();
    let equal = match (actual, expected) {
        (SchedulerScalarValue::F64(left), SchedulerScalarValue::F64(right)) => {
            left.to_bits() == right.to_bits()
        }
        _ => actual == expected,
    };
    if !equal {
        return Err(invalid(format!(
            "terrain-static scheduler scalar {name:?} is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn scalar_f64(frame: &ForecastFrame, name: &str) -> Result<f64> {
    match frame
        .scheduler_scalars()
        .iter()
        .find(|scalar| scalar.name() == name)
        .ok_or_else(|| {
            invalid(format!(
                "terrain-static source lacks scheduler scalar {name:?}"
            ))
        })?
        .value()
    {
        SchedulerScalarValue::F64(value) => Ok(value),
        other => Err(invalid(format!(
            "terrain-static scheduler scalar {name:?} has wrong type {other:?}"
        ))),
    }
}

fn shape<'a>(contract: &'a ForecastContract, name: &str) -> Result<&'a [usize]> {
    contract
        .arrays()
        .iter()
        .find(|spec| spec.name() == name)
        .map(|spec| spec.shape())
        .ok_or_else(|| invalid(format!("terrain-static contract lacks array {name:?}")))
}

fn require_shape(contract: &ForecastContract, name: &str, expected: &[usize]) -> Result<()> {
    let actual = shape(contract, name)?;
    if actual != expected {
        return Err(invalid(format!(
            "terrain-static source array {name:?} shape is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn array_map(frame: &ForecastFrame) -> Result<BTreeMap<&str, &ForecastArray>> {
    let mut arrays = BTreeMap::new();
    for item in frame.arrays() {
        if arrays.insert(item.name(), item).is_some() {
            return Err(invalid(format!(
                "terrain-static restored frame repeats array {:?}",
                item.name()
            )));
        }
    }
    Ok(arrays)
}

fn array<'a>(arrays: &BTreeMap<&str, &'a ForecastArray>, name: &str) -> Result<&'a [f32]> {
    arrays.get(name).map(|item| item.values()).ok_or_else(|| {
        invalid(format!(
            "terrain-static restored frame lacks array {name:?}"
        ))
    })
}

#[derive(Clone, Debug)]
struct CarrierStats {
    finite_count: usize,
    nonzero_count: usize,
    minimum: f32,
    maximum: f32,
}

impl CarrierStats {
    fn new(name: &str, values: &[f32]) -> Result<Self> {
        if values.is_empty() {
            return Err(invalid(format!("terrain-static carrier {name} is empty")));
        }
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        let mut nonzero_count = 0usize;
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(invalid(format!(
                    "terrain-static carrier {name} has non-finite word {index}"
                )));
            }
            minimum = minimum.min(value);
            maximum = maximum.max(value);
            if value.to_bits() & 0x7fff_ffff != 0 {
                nonzero_count += 1;
            }
        }
        Ok(Self {
            finite_count: values.len(),
            nonzero_count,
            minimum,
            maximum,
        })
    }

    fn json(&self) -> JsonValue {
        object(vec![
            ("finite_count", number(self.finite_count)),
            ("maximum_f32_bits", f32_bits(self.maximum)),
            ("minimum_f32_bits", f32_bits(self.minimum)),
            ("nonzero_count", number(self.nonzero_count)),
        ])
    }
}

#[derive(Clone, Debug)]
struct CarrierRecord {
    name: String,
    role: String,
    source_arrays: Vec<String>,
    shape: Vec<usize>,
    offset_nbytes: u64,
    nbytes: u64,
    sha256: String,
    stats: CarrierStats,
}

impl CarrierRecord {
    fn json(&self) -> JsonValue {
        object(vec![
            ("dtype", string("<f4")),
            ("itemsize", number(ITEMSIZE)),
            ("name", string(&self.name)),
            ("nbytes", u64_number(self.nbytes)),
            ("offset_nbytes", u64_number(self.offset_nbytes)),
            ("order", string("C")),
            ("role", string(&self.role)),
            (
                "shape",
                JsonValue::Array(self.shape.iter().copied().map(number).collect()),
            ),
            ("sha256", string(&self.sha256)),
            (
                "source_arrays",
                JsonValue::Array(self.source_arrays.iter().map(|item| string(item)).collect()),
            ),
            ("statistics", self.stats.json()),
        ])
    }
}

fn write_frame_into_directory(
    directory: &Path,
    frame: &TerrainStaticFrame,
    contract: &ForecastContract,
    source_frame: &ForecastFrame,
    source_receipt: &ForecastReceipt,
    admission: TerrainStaticAdmission,
) -> Result<TerrainStaticReceipt> {
    let payload_path = directory.join(TERRAIN_STATIC_PAYLOAD_FILE);
    let payload_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&payload_path)?;
    let mut writer = BufWriter::new(payload_file);
    let mut aggregate = Sha256::default();
    let mut offset = 0u64;
    let mut records = Vec::with_capacity(frame.carriers.len());
    for item in &frame.carriers {
        records.push(write_carrier(
            &mut writer,
            &mut aggregate,
            &mut offset,
            item,
        )?);
    }
    writer.flush()?;
    let payload_file = writer.into_inner().map_err(|error| error.into_error())?;
    payload_file.sync_all()?;
    if payload_file.metadata()?.len() != offset {
        return Err(invalid(
            "terrain-static payload size changed after synchronized write",
        ));
    }
    let payload_sha256 = sha256::hex(&aggregate.finalize());
    let manifest = manifest_json(
        frame,
        contract,
        source_frame,
        source_receipt,
        admission,
        offset,
        &payload_sha256,
        &records,
    );
    let manifest_bytes = manifest.canonical_bytes();
    let manifest_sha256 = sha256::digest_hex(&manifest_bytes);
    write_new_synced(
        &directory.join(TERRAIN_STATIC_MANIFEST_FILE),
        &manifest_bytes,
    )?;
    let receipt = TerrainStaticReceipt {
        source_artifact_sha256: source_receipt.artifact_sha256.clone(),
        source_contract_sha256: contract.sha256(),
        payload_sha256,
        payload_nbytes: offset,
        manifest_sha256,
        manifest_nbytes: manifest_bytes.len(),
        carrier_count: records.len(),
        admission,
    };
    write_new_synced(
        &directory.join(TERRAIN_STATIC_RECEIPT_FILE),
        receipt.canonical_json().as_bytes(),
    )?;
    Ok(receipt)
}

fn write_carrier(
    writer: &mut BufWriter<File>,
    aggregate: &mut Sha256,
    offset: &mut u64,
    carrier: &TerrainStaticCarrier,
) -> Result<CarrierRecord> {
    validate_carrier(carrier)?;
    let nbytes = u64::try_from(
        carrier
            .values
            .len()
            .checked_mul(ITEMSIZE)
            .ok_or_else(|| invalid("terrain-static carrier byte count overflows"))?,
    )
    .map_err(|_| invalid("terrain-static carrier byte count exceeds u64"))?;
    let carrier_offset = *offset;
    let mut field_hasher = Sha256::default();
    let mut bytes = Vec::with_capacity(IO_CHUNK_WORDS * ITEMSIZE);
    for chunk in carrier.values.chunks(IO_CHUNK_WORDS) {
        bytes.clear();
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes)?;
        field_hasher.update(&bytes);
        aggregate.update(&bytes);
    }
    *offset = offset
        .checked_add(nbytes)
        .ok_or_else(|| invalid("terrain-static payload offset overflows"))?;
    Ok(CarrierRecord {
        name: carrier.name.to_string(),
        role: carrier.role.to_string(),
        source_arrays: carrier
            .source_arrays
            .iter()
            .map(|item| (*item).to_string())
            .collect(),
        shape: carrier.shape.clone(),
        offset_nbytes: carrier_offset,
        nbytes,
        sha256: sha256::hex(&field_hasher.finalize()),
        stats: CarrierStats::new(carrier.name, &carrier.values)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn manifest_json(
    frame: &TerrainStaticFrame,
    contract: &ForecastContract,
    source_frame: &ForecastFrame,
    source_receipt: &ForecastReceipt,
    admission: TerrainStaticAdmission,
    payload_nbytes: u64,
    payload_sha256: &str,
    records: &[CarrierRecord],
) -> JsonValue {
    object(vec![
        ("admission", string(admission.as_str())),
        (
            "authority",
            object(vec![
                ("release", string("CM1 21.1")),
                ("tarball_sha256", string(AUTHORITY_TARBALL_SHA256)),
                (
                    "source_files",
                    JsonValue::Array(
                        AUTHORITY_FILES
                            .iter()
                            .map(|(path, digest, anchors)| {
                                object(vec![
                                    ("anchors", string(anchors)),
                                    ("path", string(path)),
                                    ("sha256", string(digest)),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
        ),
        ("base_policy", string(TERRAIN_STATIC_BASE_POLICY)),
        ("carrier_abi", string(TERRAIN_STATIC_CARRIER_ABI)),
        (
            "carriers",
            JsonValue::Array(records.iter().map(CarrierRecord::json).collect()),
        ),
        ("claim_limit", string(TERRAIN_STATIC_CLAIM_LIMIT)),
        (
            "configuration",
            object(vec![
                (
                    "hadvordrs",
                    number(usize::from(frame.config.hadvordrs)),
                ),
                (
                    "hadvordrv",
                    number(usize::from(frame.config.hadvordrv)),
                ),
                (
                    "lateral_bc_west_east_south_north",
                    JsonValue::Array(
                        frame
                            .config
                            .lateral_bc
                            .map(|item| number(usize::from(item)))
                            .to_vec(),
                    ),
                ),
                (
                    "zsgrad_effective_order",
                    number(usize::from(
                        frame.config.hadvordrs.max(frame.config.hadvordrv),
                    )),
                ),
            ]),
        ),
        (
            "crop",
            object(vec![
                (
                    "count_yx",
                    JsonValue::Array([frame.crop.ny, frame.crop.nx].map(number).to_vec()),
                ),
                (
                    "origin_yx",
                    JsonValue::Array(
                        [frame.crop.j_start, frame.crop.i_start]
                            .map(number)
                            .to_vec(),
                    ),
                ),
                (
                    "stagger_rule",
                    string("all z levels retained; horizontal crop retains terminal u/v faces"),
                ),
            ]),
        ),
        ("dtype", string("<f4")),
        (
            "fornberg",
            object(vec![
                (
                    "bottom_cgs_f32_bits",
                    JsonValue::Array(frame.bottom_fornberg.map(f32_bits).to_vec()),
                ),
                (
                    "source_contract",
                    string("param.F:8988-9052 DOUBLE PRECISION recurrence; default-REAL normalized coordinates and default-REAL stored weights"),
                ),
                (
                    "top_cgt_f32_bits",
                    JsonValue::Array(frame.top_fornberg.map(f32_bits).to_vec()),
                ),
            ]),
        ),
        (
            "formula_contracts",
            object(vec![
                ("base_reciprocals", string("base.F:2160-2190")),
                ("horizontal_gradients", string("adv_routines.F:9255-9322 with flx6 at 6026-6037 and bc2d open-edge halos")),
                ("terrain_coordinates", string("init_terrain.F:175-297")),
                ("vertical_metrics", string("param.F:8464-8492,8603-8627")),
            ]),
        ),
        ("itemsize", number(ITEMSIZE)),
        ("order", string("C")),
        (
            "payload",
            object(vec![
                ("file", string(TERRAIN_STATIC_PAYLOAD_FILE)),
                ("nbytes", u64_number(payload_nbytes)),
                ("sha256", string(payload_sha256)),
            ]),
        ),
        ("schema", string(TERRAIN_STATIC_SCHEMA)),
        (
            "source",
            object(vec![
                ("artifact_nbytes", number(source_receipt.artifact_nbytes)),
                ("artifact_sha256", string(&source_receipt.artifact_sha256)),
                ("contract_sha256", string(&contract.sha256())),
                ("dt_f64_bits", f64_bits(source_frame.clock().dt_s())),
                (
                    "model_time_f64_bits",
                    f64_bits(source_frame.clock().model_time_s()),
                ),
                ("profile_id", string(contract.profile_id())),
                ("step", u64_number(source_frame.clock().step())),
            ]),
        ),
        (
            "source_grid",
            object(vec![
                (
                    "shape_zyx",
                    JsonValue::Array(frame.source_shape_zyx.map(number).to_vec()),
                ),
                ("target_dx_m_f64_bits", f64_bits(frame.spacing_m[0])),
                ("target_dy_m_f64_bits", f64_bits(frame.spacing_m[1])),
                ("target_dz_m_f64_bits", f64_bits(frame.spacing_m[2])),
            ]),
        ),
    ])
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|item| item.to_str())
        .ok_or_else(|| invalid("terrain-static destination needs a UTF-8 file name"))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| invalid(format!("system clock precedes epoch: {error}")))?
        .as_nanos();
    Ok(output.with_file_name(format!(".{name}.partial-{}-{nonce}", std::process::id())))
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!(
            "{label} must be 64 lower-case hexadecimal characters"
        )));
    }
    Ok(())
}

fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    )
}

fn string(value: &str) -> JsonValue {
    JsonValue::String(value.to_string())
}

fn number(value: usize) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn u64_number(value: u64) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn f32_bits(value: f32) -> JsonValue {
    string(&format!("{:08x}", value.to_bits()))
}

fn f64_bits(value: f64) -> JsonValue {
    string(&format!("{:016x}", value.to_bits()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecast::{
        write_forecast_checkpoint_new, ComponentBlobSpec, ForecastArraySpec, ForecastClock,
        SchedulerScalar, SchedulerScalarSpec, SchedulerScalarType,
    };

    fn temporary(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "gpuwm-cm1-terrain-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        path
    }

    fn spec(name: &str, shape: Vec<usize>) -> ForecastArraySpec {
        let mut bounds = Vec::with_capacity(shape.len() * 2);
        for extent in &shape {
            bounds.push(1);
            bounds.push(i32::try_from(*extent).unwrap());
        }
        ForecastArraySpec::new(name, name, "1", "test", bounds, shape).unwrap()
    }

    fn source_shape(name: &str, nz: usize, ny: usize, nx: usize) -> Vec<usize> {
        match name {
            "pi0" | "prs0" | "qv0" | "rho0" | "th0" | "u0" | "v0" => vec![nz],
            "pp3d" | "th3d" | "qv" | "qc" | "qr" | "qi" | "qs" | "qg" => {
                vec![nz, ny, nx]
            }
            "u3d" => vec![nz, ny, nx + 1],
            "v3d" => vec![nz, ny + 1, nx],
            "w3d" => vec![nz + 1, ny, nx],
            "terrain_msl" | "refc_dbz" | "refd_1000m_agl_dbz" | "refd_4000m_agl_dbz" => {
                vec![ny, nx]
            }
            other => panic!("unhandled source fixture array {other}"),
        }
    }

    fn fixture(root: &Path, mutate: impl Fn(&str, &mut Vec<f32>)) -> (PathBuf, String, String) {
        let (nz, ny, nx) = (4usize, 4usize, 5usize);
        let specs = SOURCE_ARRAYS
            .iter()
            .map(|name| spec(name, source_shape(name, nz, ny, nx)))
            .collect::<Vec<_>>();
        let scalar_specs = vec![
            SchedulerScalarSpec::new("boundary_ready", SchedulerScalarType::Bool).unwrap(),
            SchedulerScalarSpec::new("forcing_valid_offset_s", SchedulerScalarType::F64).unwrap(),
            SchedulerScalarSpec::new("source_forecast_hour", SchedulerScalarType::U64).unwrap(),
            SchedulerScalarSpec::new("target_dx_m", SchedulerScalarType::F64).unwrap(),
            SchedulerScalarSpec::new("target_dy_m", SchedulerScalarType::F64).unwrap(),
            SchedulerScalarSpec::new("target_dz_m", SchedulerScalarType::F64).unwrap(),
        ];
        let contract = ForecastContract::new(
            SOURCE_PROFILE_ID,
            "11".repeat(32),
            "22".repeat(32),
            specs,
            scalar_specs,
            Vec::<ComponentBlobSpec>::new(),
        )
        .unwrap();
        let mut arrays = Vec::new();
        for source in contract.arrays() {
            let words = source.shape().iter().product::<usize>();
            let mut values = (0..words)
                .map(|index| 0.01_f32 + index as f32 * 0.0001_f32)
                .collect::<Vec<_>>();
            match source.name() {
                "pi0" | "prs0" | "rho0" | "th0" => {
                    values.iter_mut().for_each(|value| *value += 1.0)
                }
                "qv0" | "qv" | "qc" | "qr" | "qi" | "qs" | "qg" => {
                    values.iter_mut().for_each(|value| *value *= 0.001)
                }
                "terrain_msl" => {
                    for (index, value) in values.iter_mut().enumerate() {
                        *value = 10.0_f32 + index as f32 * 0.25_f32;
                    }
                }
                "refc_dbz" | "refd_1000m_agl_dbz" | "refd_4000m_agl_dbz" => {
                    values.iter_mut().for_each(|value| *value -= 10.0)
                }
                _ => {}
            }
            mutate(source.name(), &mut values);
            arrays.push(ForecastArray::new(source.name(), values));
        }
        let scalars = vec![
            SchedulerScalar::new("boundary_ready", SchedulerScalarValue::Bool(false)).unwrap(),
            SchedulerScalar::new("forcing_valid_offset_s", SchedulerScalarValue::F64(0.0)).unwrap(),
            SchedulerScalar::new("source_forecast_hour", SchedulerScalarValue::U64(0)).unwrap(),
            SchedulerScalar::new("target_dx_m", SchedulerScalarValue::F64(100.0)).unwrap(),
            SchedulerScalar::new("target_dy_m", SchedulerScalarValue::F64(100.0)).unwrap(),
            SchedulerScalar::new("target_dz_m", SchedulerScalarValue::F64(200.0)).unwrap(),
        ];
        let frame = ForecastFrame::new(
            ForecastClock::new(0, 0.0, 0.5).unwrap(),
            arrays,
            scalars,
            Vec::new(),
        );
        let artifact = root.join("input.cm1fcp");
        let receipt = write_forecast_checkpoint_new(&contract, &frame, &artifact).unwrap();
        (artifact, contract.sha256(), receipt.artifact_sha256)
    }

    fn values<'a>(frame: &'a TerrainStaticFrame, name: &str) -> &'a [f32] {
        frame.carrier(name).unwrap().values()
    }

    #[test]
    fn flat_column_has_exact_source_metrics_weights_and_base_faces() {
        let rho0 = [1.2_f32, 1.1_f32, 1.0_f32, 0.9_f32];
        let frame = compute_terrain_static(
            &[0.0; 30],
            &rho0,
            [4, 5, 6],
            [100.0, 100.0, 200.0],
            HorizontalCrop::full(5, 6),
            TerrainStaticConfig::new(5, 5, [2; 4]).unwrap(),
            TerrainStaticAdmission::Generic,
        )
        .unwrap();
        assert_eq!(
            frame.bottom_fornberg().map(f32::to_bits),
            [
                1.875_f32.to_bits(),
                (-1.25_f32).to_bits(),
                0.375_f32.to_bits(),
            ]
        );
        assert_eq!(
            frame.top_fornberg().map(f32::to_bits),
            [
                1.875_f32.to_bits(),
                (-1.25_f32).to_bits(),
                0.375_f32.to_bits(),
            ]
        );
        for name in [
            "mh", "rmh", "mf", "rmf", "gz", "rgz", "gzu", "rgzu", "gzv", "rgzv",
        ] {
            assert!(
                values(&frame, name)
                    .iter()
                    .all(|value| value.to_bits() == 1.0_f32.to_bits()),
                "{name}"
            );
        }
        for name in ["dzdx", "dzdy", "gx", "gxu", "gy", "gyv"] {
            assert!(
                values(&frame, name).iter().all(|value| *value == 0.0),
                "{name}"
            );
        }
        assert!(values(&frame, "c1")
            .iter()
            .all(|value| value.to_bits() == 0.5_f32.to_bits()));
        assert!(values(&frame, "c2")
            .iter()
            .all(|value| value.to_bits() == 0.5_f32.to_bits()));
        assert_eq!(values(&frame, "zh")[0].to_bits(), 100.0_f32.to_bits());
        assert_eq!(values(&frame, "zf")[0].to_bits(), 0.0_f32.to_bits());
        let expected_bottom = 1.875_f32 * rho0[0] + (-1.25_f32) * rho0[1] + 0.375_f32 * rho0[2];
        let expected_top = 1.875_f32 * rho0[3] + (-1.25_f32) * rho0[2] + 0.375_f32 * rho0[1];
        assert_eq!(
            values(&frame, "rf0")[0].to_bits(),
            expected_bottom.to_bits()
        );
        assert_eq!(
            values(&frame, "rf0")[4 * 30].to_bits(),
            expected_top.to_bits()
        );
    }

    #[test]
    fn flat_uniform_and_zero_height_terrain_keep_their_distinct_source_arithmetic() {
        let rho = vec![1.0_f32; 237];
        let config = TerrainStaticConfig::new(5, 5, [2; 4]).unwrap();
        let flat = compute_flat_uniform_static(
            &rho,
            [237, 2, 3],
            [75.0; 3],
            HorizontalCrop::full(2, 3),
            config,
        )
        .unwrap();
        let terrain = compute_terrain_static(
            &[0.0; 6],
            &rho,
            [237, 2, 3],
            [75.0; 3],
            HorizontalCrop::full(2, 3),
            config,
            TerrainStaticAdmission::Generic,
        )
        .unwrap();
        // Native param.F leaves sigma unchanged without terrain; native
        // init_terrain.F:187 evaluates this expression even at zero surface.
        let k = 6;
        let sigma = 487.5_f32;
        let top = 17_775.0_f32;
        assert_eq!(values(&flat, "zh")[k * 6].to_bits(), sigma.to_bits());
        assert_eq!(
            values(&terrain, "zh")[k * 6].to_bits(),
            (sigma * top / top).to_bits()
        );
        assert_ne!(
            values(&flat, "zh")[k * 6].to_bits(),
            values(&terrain, "zh")[k * 6].to_bits()
        );
    }

    #[test]
    fn flat_grid_and_base_match_native_instrument_oracle_raw_bits() {
        // The development repository's grid/base oracle (its folder and file names may carry a prefix).
        let named = |dir: &Path, suffix: &str| {
            fs::read_dir(dir)
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with(suffix))
                })
                .unwrap()
        };
        let oracles = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../evidence/oracles");
        let oracle = named(&named(&oracles, "grid-base-v1"), "grid-base-outputs.tsv");
        let text = fs::read_to_string(oracle).unwrap();
        let mut captured = BTreeMap::<(String, usize), u32>::new();
        for line in text.lines().skip(1) {
            let columns = line.split('\t').collect::<Vec<_>>();
            if columns.len() != 11 || columns[4] != "1" || columns[5] != "1" {
                continue;
            }
            let field = columns[2];
            if ![
                "zh", "zf", "mh", "rmh", "mf", "rmf", "c1", "c2", "rho0", "rr0", "rf0", "rrf0",
            ]
            .contains(&field)
            {
                continue;
            }
            captured.insert(
                (field.to_string(), columns[6].parse::<usize>().unwrap()),
                u32::from_str_radix(columns[10], 16).unwrap(),
            );
        }
        let rho0 = (1..=40)
            .map(|k| f32::from_bits(captured[&("rho0".to_string(), k)]))
            .collect::<Vec<_>>();
        let frame = compute_terrain_static(
            &[0.0],
            &rho0,
            [40, 1, 1],
            [1000.0, 1000.0, 500.0],
            HorizontalCrop::full(1, 1),
            TerrainStaticConfig::new(5, 5, [2; 4]).unwrap(),
            TerrainStaticAdmission::Generic,
        )
        .unwrap();
        for (carrier, oracle_field) in [
            ("zh", "zh"),
            ("mh", "mh"),
            ("rmh", "rmh"),
            ("rho0_3d", "rho0"),
            ("rr0", "rr0"),
        ] {
            for (k, value) in values(&frame, carrier).iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    captured[&(oracle_field.to_string(), k + 1)],
                    "{carrier} k={}",
                    k + 1
                );
            }
        }
        for carrier in ["zf", "mf", "rmf", "c1", "c2", "rf0", "rrf0"] {
            for (k, value) in values(&frame, carrier).iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    captured[&(carrier.to_string(), k + 1)],
                    "{carrier} k={}",
                    k + 1
                );
            }
        }
    }

    #[test]
    fn strict_export_publishes_canonical_hash_bound_companion() {
        let root = temporary("export");
        let (artifact, contract_sha, artifact_sha) = fixture(&root, |_, _| {});
        let output = root.join("terrain-static");
        let crop = HorizontalCrop {
            j_start: 1,
            i_start: 1,
            ny: 2,
            nx: 3,
        };
        let receipt = export_checkpoint_terrain_static_new(
            &artifact,
            &output,
            &contract_sha,
            &artifact_sha,
            Some(crop),
            TerrainStaticConfig::new(5, 5, [2; 4]).unwrap(),
            TerrainStaticAdmission::Generic,
        )
        .unwrap();
        assert_eq!(receipt.carrier_count, TERRAIN_STATIC_CARRIER_NAMES.len());
        let payload = fs::read(output.join(TERRAIN_STATIC_PAYLOAD_FILE)).unwrap();
        assert_eq!(payload.len() as u64, receipt.payload_nbytes);
        assert_eq!(sha256::digest_hex(&payload), receipt.payload_sha256);
        let manifest_bytes = fs::read(output.join(TERRAIN_STATIC_MANIFEST_FILE)).unwrap();
        assert_eq!(sha256::digest_hex(&manifest_bytes), receipt.manifest_sha256);
        assert_eq!(
            manifest_bytes,
            crate::json::parse(&manifest_bytes)
                .unwrap()
                .canonical_bytes()
        );
        let manifest = crate::json::parse(&manifest_bytes).unwrap();
        assert_eq!(
            manifest.get("schema").unwrap().as_str().unwrap(),
            TERRAIN_STATIC_SCHEMA
        );
        let carriers = manifest.get("carriers").unwrap().as_array().unwrap();
        assert_eq!(carriers.len(), 25);
        assert_eq!(
            carriers
                .iter()
                .map(|item| item.get("name").unwrap().as_str().unwrap())
                .collect::<Vec<_>>(),
            TERRAIN_STATIC_CARRIER_NAMES
        );
        assert_eq!(
            carriers[1]
                .get("shape")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap())
                .collect::<Vec<_>>(),
            [4, 2, 3]
        );
        assert_eq!(
            carriers[15]
                .get("shape")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap())
                .collect::<Vec<_>>(),
            [2, 4]
        );
        assert!(export_checkpoint_terrain_static_new(
            &artifact,
            &output,
            &contract_sha,
            &artifact_sha,
            Some(crop),
            TerrainStaticConfig::new(5, 5, [2; 4]).unwrap(),
            TerrainStaticAdmission::Generic,
        )
        .unwrap_err()
        .to_string()
        .contains("refusing to replace"));
    }

    #[test]
    fn strict_negatives_refuse_config_hash_crop_shape_and_values() {
        assert!(TerrainStaticConfig::new(6, 5, [2; 4])
            .unwrap_err()
            .to_string()
            .contains("different zsgrad"));
        assert!(TerrainStaticConfig::new(5, 5, [1, 2, 2, 2])
            .unwrap_err()
            .to_string()
            .contains("change terrain halo"));
        let error = compute_terrain_static(
            &[800.0],
            &[1.0; 4],
            [4, 1, 1],
            [100.0, 100.0, 200.0],
            HorizontalCrop::full(1, 1),
            TerrainStaticConfig::new(5, 5, [2; 4]).unwrap(),
            TerrainStaticAdmission::Generic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("reaches the model top"), "{error}");
        let error = compute_terrain_static(
            &[0.0],
            &[1.0, 1.0, 0.0, 1.0],
            [4, 1, 1],
            [100.0, 100.0, 200.0],
            HorizontalCrop::full(1, 1),
            TerrainStaticConfig::new(5, 5, [2; 4]).unwrap(),
            TerrainStaticAdmission::Generic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("positive finite"), "{error}");

        let root = temporary("negative");
        let (artifact, contract_sha, artifact_sha) = fixture(&root, |_, _| {});
        let error = export_checkpoint_terrain_static_new(
            &artifact,
            root.join("bad-hash"),
            &contract_sha,
            &"ff".repeat(32),
            None,
            TerrainStaticConfig::new(5, 5, [2; 4]).unwrap(),
            TerrainStaticAdmission::Generic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("independently expected"), "{error}");
        let error = export_checkpoint_terrain_static_new(
            &artifact,
            root.join("bad-crop"),
            &contract_sha,
            &artifact_sha,
            Some(HorizontalCrop {
                j_start: 3,
                i_start: 0,
                ny: 2,
                nx: 1,
            }),
            TerrainStaticConfig::new(5, 5, [2; 4]).unwrap(),
            TerrainStaticAdmission::Generic,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exceeds source grid"), "{error}");
        let error = export_checkpoint_terrain_static_new(
            &artifact,
            root.join("wrong-product"),
            &contract_sha,
            &artifact_sha,
            None,
            TerrainStaticConfig::new(5, 5, [2; 4]).unwrap(),
            TerrainStaticAdmission::RealProduct,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("must be [75,300,300]"), "{error}");
    }
}
