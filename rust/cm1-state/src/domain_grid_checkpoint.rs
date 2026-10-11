//! Strict, versioned bindings for CM1 vertical-grid and moving-domain state.
//!
//! Static grid metrics are rebuilt from their source parameters and verified
//! by raw-byte hashes.  Moving-domain scalars are persisted bit-for-bit at the
//! completed-large-step boundary.  The two components remain independent:
//! CM1 r21.1 forbids combining terrain with a moving domain, and this module
//! fails closed on that selection instead of inventing a coupling.

use crate::json::{self, JsonValue};
use crate::sha256;
use crate::snapshot::atomic_write_new;
use crate::{Result, StateError};
use std::path::Path;

pub const DOMAIN_GRID_AUTHORITY_TARBALL_SHA256: &str =
    "dc49fe84531056d1ae6249b37a5e3ee453fd96861c3b6bafd63828d92e64edf7";
pub const STRETCH_Z1_COMPONENT_SCHEMA: &str = "gm1.stretch-z1-checkpoint-component/v1";
pub const MOVING_DOMAIN_COMPONENT_SCHEMA: &str = "gm1.moving-domain-checkpoint-component/v1";
pub const F02_INTERIOR_ROI_SCHEMA: &str = "gm1.hrrr-f02-interior-roi/v1";
pub const COMPLETED_LARGE_STEP_CADENCE: &str = "post-large-step-after-adapt-move";
pub const CUDA_STRETCH_Z1_SCHEMA: &str = "gpuwm-cm1.cuda-stretch-z1/v1";
pub const CUDA_STRETCH_Z1_ABI: &str = "gpuwm-cm1.cuda-stretch-z1-kernel/v1";
pub const CUDA_STRETCH_Z1_SOURCE_SHA256: &str =
    "fb015416dc07405654ea264b50b13ce7eb986b5406c4894ab4e5d082bb6a7441";
pub const CUDA_STRETCH_Z1_PROOF_SHA256: &str =
    "d36796c3de7adab3065c69d85d9a048de543ea77ca329a6778381705c26f97a5";
pub const STRETCH_Z1_GRID_SHA256: &str =
    "951701a06998443302f68c7006a80bf52c4a49f199dd492560442cbbc526268c";
pub const STRETCH_Z1_COMMIT: &str = "c6779e2dd10293f08ae09bc14101e5e463a00117";
pub const STRETCH_Z1_CLAIM_LIMIT: &str = "CM1 r21.1 param.F Wilhelmson-Chen stretch_z=1 reference levels and mh/mf/rmh/rmf for a flat Cartesian grid, rebuilt from raw binary32 parameters and hash-verified; no stretch_z=2/3/4, terrain composition, horizontal stretch, moving-domain coupling, trajectory, storm, tornado, or performance claim";
pub const MOVING_DOMAIN_CLAIM_LIMIT: &str = "bit-exact persistence of the CM1 r21.1 moving-domain restart scalars and output counters at a completed-large-step boundary; validates documented selector conflicts but does not implement change_uvmove, adaptive center diagnosis, halo translation, MPI redistribution, dynamics, trajectory, storm, or tornado claims";
pub const F02_INTERIOR_ROI_CLAIM_LIMIT: &str = "real HRRR f02 coordinate-selected 300x300x75 remap centered on the observed low-level reflectivity maximum, with Rust-owned history/product/restart/geopotential hashes; no array translation, synthetic displacement, evolved trajectory, tornado, or model-validation claim";

const PARAM_F_SHA256: &str = "cac64a6cb4363c6b88367b5cb9391f1bcf2130c63ffedef6e5973c03b190c349";
const CM1_F_SHA256: &str = "7650be7688b45288997b4a869f5c996869aa22eae0c3c1cffa04c372e7deedb4";
const AZIMAVG_F_SHA256: &str = "169a9f82d0929ab88c4add87441b6f9ace70f5c83fd82ecbb9f43d700af75c2a";
const RESTART_WRITE_F_SHA256: &str =
    "d9c21d7071a4548cb9d3ffee1658550ce0e2a4ee7db9c67758ac8c61c07cddba";
const RESTART_READ_F_SHA256: &str =
    "72a041737a4025ce445cd303cd169eb6ff4c001a1140f9fca9e9cdb918884e2a";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StretchZ1Parameters {
    pub nk: usize,
    pub dz: f32,
    pub ztop: f32,
    pub str_bot: f32,
    pub str_top: f32,
    pub dz_bot: f32,
    pub dz_top: f32,
}

impl StretchZ1Parameters {
    pub const fn gm1_selected() -> Self {
        Self {
            nk: 75,
            dz: 100.0,
            ztop: 17_650.0,
            str_bot: 200.0,
            str_top: 9_650.0,
            dz_bot: 20.0,
            dz_top: 400.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StretchZ1Derived {
    pub nk1: usize,
    pub nk2: usize,
    pub nk3: usize,
    pub nominal_dz: f32,
    pub c1: f32,
    pub c2: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StretchZ1Grid {
    pub zf: Vec<f32>,
    pub zh: Vec<f32>,
    pub sigmaf: Vec<f32>,
    pub sigma: Vec<f32>,
    pub mh: Vec<f32>,
    pub mf: Vec<f32>,
    pub rmh: Vec<f32>,
    pub rmf: Vec<f32>,
    pub derived: StretchZ1Derived,
}

impl StretchZ1Grid {
    pub fn profile_sha256(&self, name: &str) -> Result<String> {
        let values = match name {
            "zf" => &self.zf,
            "zh" => &self.zh,
            "sigmaf" => &self.sigmaf,
            "sigma" => &self.sigma,
            "mh" => &self.mh,
            "mf" => &self.mf,
            "rmh" => &self.rmh,
            "rmf" => &self.rmf,
            _ => return Err(invalid(format!("unknown stretch_z=1 profile {name:?}"))),
        };
        Ok(hash_f32(values))
    }

    pub fn combined_sha256(&self) -> String {
        let mut bytes = Vec::new();
        for values in [
            &self.zf,
            &self.zh,
            &self.sigmaf,
            &self.sigma,
            &self.mh,
            &self.mf,
            &self.rmh,
            &self.rmf,
        ] {
            append_f32_bytes(&mut bytes, values);
        }
        sha256::digest_hex(&bytes)
    }
}

pub fn build_stretch_z1(parameters: StretchZ1Parameters) -> Result<StretchZ1Grid> {
    let derived = derive_stretch_z1(parameters)?;
    let nk = parameters.nk;
    let mut zf = vec![0.0f32; nk + 3];
    let mut sigmaf = vec![0.0f32; nk + 3];
    for k in 1..=(derived.nk1 + 1) {
        zf[k] = mul(k.saturating_sub(1) as f32, parameters.dz_bot);
        sigmaf[k] = zf[k];
    }
    for k in (derived.nk1 + 1)..=(derived.nk1 + derived.nk2 + 1) {
        let delta = (k - 1 - derived.nk1) as f32;
        let factor = add(derived.c1, mul(mul(derived.c2, delta), derived.nominal_dz));
        let distance = mul(mul(factor, delta), derived.nominal_dz);
        zf[k] = add(zf[derived.nk1 + 1], distance);
        sigmaf[k] = zf[k];
    }
    for k in (derived.nk1 + derived.nk2 + 2)..=(nk + 1) {
        zf[k] = add(zf[k - 1], parameters.dz_top);
        sigmaf[k] = zf[k];
    }
    zf[0] = -zf[2];
    zf[nk + 2] = add(zf[nk + 1], sub(zf[nk + 1], zf[nk]));
    sigmaf[0] = -sigmaf[2];
    sigmaf[nk + 2] = add(sigmaf[nk + 1], sub(sigmaf[nk + 1], sigmaf[nk]));

    let mut zh = vec![0.0f32; nk + 2];
    let mut mh = vec![0.0f32; nk + 2];
    for k in 0..=(nk + 1) {
        zh[k] = mul(0.5, add(zf[k + 1], zf[k]));
        mh[k] = div(parameters.dz, sub(zf[k + 1], zf[k]));
    }
    zh[0] = -zh[1];
    zh[nk + 1] = add(zh[nk], mul(2.0, sub(zf[nk + 1], zh[nk])));

    let mut mf = vec![0.0f32; nk + 3];
    for k in 1..=(nk + 1) {
        mf[k] = div(parameters.dz, sub(zh[k], zh[k - 1]));
    }
    mf[0] = mf[1];
    mf[nk + 2] = mf[nk + 1];

    let mut sigma = vec![0.0f32; nk + 2];
    let mut rmh = vec![0.0f32; nk + 2];
    for k in 0..=(nk + 1) {
        sigma[k] = mul(0.5, add(sigmaf[k], sigmaf[k + 1]));
        rmh[k] = div(1.0, mh[k]);
    }
    let mut rmf = vec![0.0f32; nk + 3];
    for k in 0..=(nk + 2) {
        rmf[k] = div(1.0, mf[k]);
    }
    if zf[1..=nk + 1].windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(invalid(
            "stretch_z=1 generated nonpositive physical spacing",
        ));
    }
    Ok(StretchZ1Grid {
        zf,
        zh,
        sigmaf,
        sigma,
        mh,
        mf,
        rmh,
        rmf,
        derived,
    })
}

fn derive_stretch_z1(parameters: StretchZ1Parameters) -> Result<StretchZ1Derived> {
    if parameters.nk < 4 {
        return Err(invalid("stretch_z=1 nk must be at least four"));
    }
    for (name, value) in [
        ("dz", parameters.dz),
        ("ztop", parameters.ztop),
        ("str_bot", parameters.str_bot),
        ("str_top", parameters.str_top),
        ("dz_bot", parameters.dz_bot),
        ("dz_top", parameters.dz_top),
    ] {
        if !value.is_finite() {
            return Err(invalid(format!("stretch_z=1 {name} is not finite")));
        }
    }
    if parameters.dz <= 0.0
        || parameters.ztop <= 0.0
        || parameters.dz_bot <= 0.0
        || parameters.dz_top <= 0.0
        || parameters.str_bot < 0.0
        || parameters.str_top <= parameters.str_bot
        || parameters.str_top > parameters.ztop
    {
        return Err(invalid("stretch_z=1 parameter ordering is invalid"));
    }
    let nominal = mul(0.5, add(parameters.dz_bot, parameters.dz_top));
    let nk1 = nint_positive(div(parameters.str_bot, parameters.dz_bot))?;
    let nk3 = nint_positive(div(
        sub(parameters.ztop, parameters.str_top),
        parameters.dz_top,
    ))?;
    let nk2 = parameters
        .nk
        .checked_sub(nk1 + nk3)
        .ok_or_else(|| invalid("stretch_z=1 segment counts exceed nk"))?;
    if nk2 < 2 {
        return Err(invalid("stretch_z=1 requires at least two middle levels"));
    }
    let bottom_error = sub(mul(nk1 as f32, parameters.dz_bot), parameters.str_bot).abs();
    let middle_error = sub(
        mul(nk2 as f32, nominal),
        sub(parameters.str_top, parameters.str_bot),
    )
    .abs();
    let top_error = sub(
        mul(nk3 as f32, parameters.dz_top),
        sub(parameters.ztop, parameters.str_top),
    )
    .abs();
    if bottom_error > 0.01 || middle_error >= 0.01 || top_error > 0.01 {
        return Err(invalid("stretch_z=1 segment feasibility check failed"));
    }
    let c2 = div(
        sub(nominal, parameters.dz_bot),
        mul(mul(nominal, nominal), (nk2 - 1) as f32),
    );
    let c1 = sub(div(parameters.dz_bot, nominal), mul(c2, nominal));
    let test_len = add(
        add(mul(parameters.dz_bot, nk1 as f32), mul(nominal, nk2 as f32)),
        mul(parameters.dz_top, nk3 as f32),
    );
    if sub(parameters.ztop, test_len).abs() > mul(1.0e-6, parameters.ztop) {
        return Err(invalid("stretch_z=1 total length check failed"));
    }
    Ok(StretchZ1Derived {
        nk1,
        nk2,
        nk3,
        nominal_dz: nominal,
        c1,
        c2,
    })
}

pub fn encode_stretch_z1_component(parameters: StretchZ1Parameters) -> Result<Vec<u8>> {
    let grid = build_stretch_z1(parameters)?;
    let hashes = stretch_hashes_json(&grid)?;
    let value = object(vec![
        (
            "authority",
            authority_json(vec![("src/param.F", PARAM_F_SHA256, "7814-8225")]),
        ),
        ("cadence", string("immutable-rebuild-on-restore")),
        ("claim_limit", string(STRETCH_Z1_CLAIM_LIMIT)),
        (
            "cuda_proof",
            object(vec![
                ("abi", string(CUDA_STRETCH_Z1_ABI)),
                ("commit", string(STRETCH_Z1_COMMIT)),
                ("proof_sha256", string(CUDA_STRETCH_Z1_PROOF_SHA256)),
                ("schema", string(CUDA_STRETCH_Z1_SCHEMA)),
                ("source_sha256", string(CUDA_STRETCH_Z1_SOURCE_SHA256)),
                ("status", string("PASS_CPU_CUDA_RAW_BIT_EXACT")),
            ]),
        ),
        (
            "derived",
            object(vec![
                ("c1_f32_bits", f32_bits(grid.derived.c1)),
                ("c2_f32_bits", f32_bits(grid.derived.c2)),
                ("nk1", number(grid.derived.nk1)),
                ("nk2", number(grid.derived.nk2)),
                ("nk3", number(grid.derived.nk3)),
                ("nominal_dz_f32_bits", f32_bits(grid.derived.nominal_dz)),
            ]),
        ),
        ("hashes", hashes),
        (
            "parameters",
            object(vec![
                ("dz_bot_f32_bits", f32_bits(parameters.dz_bot)),
                ("dz_f32_bits", f32_bits(parameters.dz)),
                ("dz_top_f32_bits", f32_bits(parameters.dz_top)),
                ("nk", number(parameters.nk)),
                ("str_bot_f32_bits", f32_bits(parameters.str_bot)),
                ("str_top_f32_bits", f32_bits(parameters.str_top)),
                ("ztop_f32_bits", f32_bits(parameters.ztop)),
            ]),
        ),
        (
            "selection",
            object(vec![
                ("axisymm", JsonValue::Bool(false)),
                ("stretch_x", number(0)),
                ("stretch_y", number(0)),
                ("stretch_z", number(1)),
                ("terrain", JsonValue::Bool(false)),
            ]),
        ),
        ("schema", string(STRETCH_Z1_COMPONENT_SCHEMA)),
    ]);
    Ok(value.canonical_bytes())
}

pub fn write_stretch_z1_component_new(
    parameters: StretchZ1Parameters,
    path: impl AsRef<Path>,
) -> Result<()> {
    let mut bytes = encode_stretch_z1_component(parameters)?;
    bytes.push(b'\n');
    atomic_write_new(path.as_ref(), &bytes)
}

pub fn decode_stretch_z1_component(bytes: &[u8]) -> Result<(StretchZ1Parameters, StretchZ1Grid)> {
    let value =
        json::parse(bytes).map_err(|error| invalid(format!("stretch component: {error}")))?;
    exact_keys(
        &value,
        &[
            "authority",
            "cadence",
            "claim_limit",
            "cuda_proof",
            "derived",
            "hashes",
            "parameters",
            "schema",
            "selection",
        ],
        "stretch component",
    )?;
    fixed_string(
        &value,
        "schema",
        STRETCH_Z1_COMPONENT_SCHEMA,
        "stretch component",
    )?;
    fixed_string(
        &value,
        "cadence",
        "immutable-rebuild-on-restore",
        "stretch component",
    )?;
    fixed_string(
        &value,
        "claim_limit",
        STRETCH_Z1_CLAIM_LIMIT,
        "stretch component",
    )?;
    validate_stretch_authority(value.get("authority").map_err(json_error)?)?;
    validate_stretch_selection(value.get("selection").map_err(json_error)?)?;
    validate_cuda_proof(value.get("cuda_proof").map_err(json_error)?)?;
    let parameters = parse_stretch_parameters(value.get("parameters").map_err(json_error)?)?;
    let grid = build_stretch_z1(parameters)?;
    validate_derived(value.get("derived").map_err(json_error)?, grid.derived)?;
    let expected_hashes = stretch_hashes_json(&grid)?;
    if value.get("hashes").map_err(json_error)? != &expected_hashes {
        return Err(invalid(
            "stretch_z=1 rebuilt profile hashes differ from component",
        ));
    }
    if value.canonical_bytes() != bytes {
        return Err(invalid("stretch_z=1 component is not canonical JSON"));
    }
    Ok((parameters, grid))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MovingDomainMode {
    Fixed,
    Constant,
    Adaptive,
}

impl MovingDomainMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Constant => "constant",
            Self::Adaptive => "adaptive",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "fixed" => Ok(Self::Fixed),
            "constant" => Ok(Self::Constant),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err(invalid("moving-domain mode is unsupported")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovingDomainSelection {
    pub mode: MovingDomainMode,
    pub terrain: bool,
    pub axisymm: bool,
    pub initsfc: u8,
    pub lateral_bc_west_east_south_north: [u8; 4],
    pub adapt_move_frq: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MovingDomainState {
    pub icenter: i32,
    pub jcenter: i32,
    pub xcenter: f32,
    pub ycenter: f32,
    pub umove: f32,
    pub umovem1: f32,
    pub umovem2: f32,
    pub vmove: f32,
    pub vmovem1: f32,
    pub vmovem2: f32,
    pub domainlocx: f64,
    pub domainlocy: f64,
    pub adaptmovetim: f64,
    pub mvrec: i32,
    pub nwritemv: i32,
}

pub fn encode_moving_domain_component(
    selection: MovingDomainSelection,
    state: MovingDomainState,
    nx: usize,
    ny: usize,
) -> Result<Vec<u8>> {
    validate_moving(selection, state, nx, ny)?;
    let value = object(vec![
        (
            "authority",
            authority_json(vec![
                ("src/cm1.F", CM1_F_SHA256, "264-269,3512-3513,3642-3679"),
                ("src/azimavg.F", AZIMAVG_F_SHA256, "4400-4465"),
                ("src/param.F", PARAM_F_SHA256, "1890-1903"),
                (
                    "src/restart_read.F",
                    RESTART_READ_F_SHA256,
                    "406-421,689-735",
                ),
                (
                    "src/restart_write.F",
                    RESTART_WRITE_F_SHA256,
                    "386-401,503-549",
                ),
            ]),
        ),
        ("cadence", string(COMPLETED_LARGE_STEP_CADENCE)),
        ("claim_limit", string(MOVING_DOMAIN_CLAIM_LIMIT)),
        (
            "domain",
            object(vec![("nx", number(nx)), ("ny", number(ny))]),
        ),
        (
            "selection",
            object(vec![
                (
                    "adapt_move_frq_f32_bits",
                    f32_bits(selection.adapt_move_frq),
                ),
                ("axisymm", JsonValue::Bool(selection.axisymm)),
                (
                    "do_adapt_move",
                    JsonValue::Bool(selection.mode == MovingDomainMode::Adaptive),
                ),
                (
                    "imove",
                    number(usize::from(selection.mode != MovingDomainMode::Fixed)),
                ),
                ("initsfc", number(selection.initsfc.into())),
                (
                    "lateral_bc_west_east_south_north",
                    JsonValue::Array(
                        selection
                            .lateral_bc_west_east_south_north
                            .iter()
                            .map(|value| number((*value).into()))
                            .collect(),
                    ),
                ),
                ("mode", string(selection.mode.as_str())),
                ("terrain", JsonValue::Bool(selection.terrain)),
            ]),
        ),
        ("schema", string(MOVING_DOMAIN_COMPONENT_SCHEMA)),
        ("state", moving_state_json(state)),
    ]);
    Ok(value.canonical_bytes())
}

pub fn write_moving_domain_component_new(
    selection: MovingDomainSelection,
    state: MovingDomainState,
    nx: usize,
    ny: usize,
    path: impl AsRef<Path>,
) -> Result<()> {
    let mut bytes = encode_moving_domain_component(selection, state, nx, ny)?;
    bytes.push(b'\n');
    atomic_write_new(path.as_ref(), &bytes)
}

pub fn decode_moving_domain_component(
    bytes: &[u8],
) -> Result<(MovingDomainSelection, MovingDomainState)> {
    let value =
        json::parse(bytes).map_err(|error| invalid(format!("moving component: {error}")))?;
    exact_keys(
        &value,
        &[
            "authority",
            "cadence",
            "claim_limit",
            "domain",
            "schema",
            "selection",
            "state",
        ],
        "moving component",
    )?;
    fixed_string(
        &value,
        "schema",
        MOVING_DOMAIN_COMPONENT_SCHEMA,
        "moving component",
    )?;
    fixed_string(
        &value,
        "cadence",
        COMPLETED_LARGE_STEP_CADENCE,
        "moving component",
    )?;
    fixed_string(
        &value,
        "claim_limit",
        MOVING_DOMAIN_CLAIM_LIMIT,
        "moving component",
    )?;
    validate_moving_authority(value.get("authority").map_err(json_error)?)?;
    let domain = value.get("domain").map_err(json_error)?;
    exact_keys(domain, &["nx", "ny"], "moving domain")?;
    let nx = usize_member(domain, "nx", "moving domain")?;
    let ny = usize_member(domain, "ny", "moving domain")?;
    let selection = parse_moving_selection(value.get("selection").map_err(json_error)?)?;
    let state = parse_moving_state(value.get("state").map_err(json_error)?)?;
    validate_moving(selection, state, nx, ny)?;
    if value.canonical_bytes() != bytes {
        return Err(invalid("moving-domain component is not canonical JSON"));
    }
    Ok((selection, state))
}

fn validate_moving(
    selection: MovingDomainSelection,
    state: MovingDomainState,
    nx: usize,
    ny: usize,
) -> Result<()> {
    if nx == 0 || ny == 0 || nx > i32::MAX as usize || ny > i32::MAX as usize {
        return Err(invalid("moving-domain dimensions are invalid"));
    }
    if !selection.adapt_move_frq.is_finite() || selection.adapt_move_frq < 0.0 {
        return Err(invalid("adaptive moving-domain frequency is invalid"));
    }
    if selection.mode != MovingDomainMode::Fixed {
        if selection.terrain {
            return Err(invalid("CM1 r21.1 forbids terrain with imove nonzero"));
        }
        if selection.axisymm {
            return Err(invalid("CM1 r21.1 axisymmetric mode requires imove=0"));
        }
        if selection.initsfc != 1 {
            return Err(invalid("CM1 r21.1 moving domain requires initsfc=1"));
        }
    }
    if selection.mode == MovingDomainMode::Adaptive && selection.adapt_move_frq <= 0.0 {
        return Err(invalid("adaptive moving-domain frequency must be positive"));
    }
    for (name, value) in [
        ("xcenter", state.xcenter),
        ("ycenter", state.ycenter),
        ("umove", state.umove),
        ("umovem1", state.umovem1),
        ("umovem2", state.umovem2),
        ("vmove", state.vmove),
        ("vmovem1", state.vmovem1),
        ("vmovem2", state.vmovem2),
    ] {
        if !value.is_finite() {
            return Err(invalid(format!("moving-domain {name} is not finite")));
        }
    }
    for (name, value) in [
        ("domainlocx", state.domainlocx),
        ("domainlocy", state.domainlocy),
        ("adaptmovetim", state.adaptmovetim),
    ] {
        if !value.is_finite() {
            return Err(invalid(format!("moving-domain {name} is not finite")));
        }
    }
    if !(1..=nx as i32).contains(&state.icenter) || !(1..=ny as i32).contains(&state.jcenter) {
        return Err(invalid(
            "moving-domain center indices are outside the physical grid",
        ));
    }
    if state.mvrec < 1 || state.nwritemv < 1 {
        return Err(invalid("moving-domain output counters must be positive"));
    }
    if selection.mode == MovingDomainMode::Fixed {
        let zero_f32 = [
            state.umove,
            state.umovem1,
            state.umovem2,
            state.vmove,
            state.vmovem1,
            state.vmovem2,
        ]
        .iter()
        .all(|value| value.to_bits() & 0x7fff_ffff == 0);
        let zero_f64 = [state.domainlocx, state.domainlocy, state.adaptmovetim]
            .iter()
            .all(|value| value.to_bits() & 0x7fff_ffff_ffff_ffff == 0);
        if !zero_f32 || !zero_f64 {
            return Err(invalid("fixed-domain checkpoint contains moving state"));
        }
    } else {
        const RIGID: u8 = 1;
        let bc = selection.lateral_bc_west_east_south_north;
        if state.umove.to_bits() & 0x7fff_ffff != 0 && (bc[0] == RIGID || bc[1] == RIGID) {
            return Err(invalid(
                "nonzero umove conflicts with a rigid west/east boundary",
            ));
        }
        if state.vmove.to_bits() & 0x7fff_ffff != 0 && (bc[2] == RIGID || bc[3] == RIGID) {
            return Err(invalid(
                "nonzero vmove conflicts with a rigid south/north boundary",
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct F02InteriorRoiReceipt {
    value: JsonValue,
}

impl F02InteriorRoiReceipt {
    pub fn sealed() -> Self {
        let value = object(vec![
            ("claim_limit", string(F02_INTERIOR_ROI_CLAIM_LIMIT)),
            (
                "geopotential",
                object(vec![
                    ("manifest_sha256", string("da37e1a22f7ef93a4abdd98fa8373fc3d50f26091aeac4812a9c216970433d1a")),
                    ("path", string("work/initialized-product-hrrr-f02-roi-v1/wsm6-geopotential")),
                    ("payload_sha256", string("74625673f6a3aba254b90eff65dedda0688dd64d62f94eeec8709b365748ae3e")),
                    ("receipt_sha256", string("09eb31934987b30ff953fb9846a2ab10bf73c8f13452e2612b26a0b2306b64b4")),
                ]),
            ),
            (
                "history",
                object(vec![
                    ("name", string("boundary-f02.cm1fhs")),
                    ("path", string("work/real-hrrr-20240521-19z-f02-f03/cm1-mapped-100m-f02-f03-v3-storm-centered/boundary-f02.cm1fhs")),
                    ("series_receipt_sha256", string("7c2c5d04f6d3a25a7cb6e9508ba95cba18718fb998cece0c49a1d068de16956e")),
                    ("sha256", string("356fe3022ce83909f5c14e7ce9a3639811d9f6dfc5ed9b33a0782cfffb7cd79d")),
                ]),
            ),
            (
                "initialized_product",
                object(vec![
                    ("manifest_sha256", string("b25de4008d17ddad46b8057f022b50a76e6bff61b537fa9952f68e638adf2567")),
                    ("path", string("work/initialized-product-hrrr-f02-roi-v1/full")),
                    ("payload_sha256", string("ddafeaa8ca523e81925cb4b53e0577963a3e57e06114b6b7df8c6d436e0d68b1")),
                    ("receipt_sha256", string("379b989884e773a942ade62069e99ce5e032374c2ae2ef13679f32a524fbbdcd")),
                    ("restart_sha256", string("9efe09771763a3f727af715f7dd41a2c6256cd5613aa2f31637cc08c9721ed6e")),
                ]),
            ),
            ("model", string("GM1")),
            ("schema", string(F02_INTERIOR_ROI_SCHEMA)),
            (
                "source",
                object(vec![
                    ("absolute_source_time_s_f64_bits", string("40bc200000000000")),
                    ("artifact_sha256", string("356fe3022ce83909f5c14e7ce9a3639811d9f6dfc5ed9b33a0782cfffb7cd79d")),
                    ("contract_sha256", string("a9ccd6f0c6650fd89cd42f16977209f57695464181bcb62beff14e9a9f61d896")),
                    ("forecast_hour", number(2)),
                    ("source_kind", string("boundary-analysis-history-f02-roi")),
                    ("step", number(14_400)),
                    ("synthetic_translation", JsonValue::Bool(false)),
                ]),
            ),
            (
                "storm_anchor",
                object(vec![
                    ("kind", string("refd_1000m_agl_max")),
                    ("latitude_f64_bits", string("4044b274de47f4a3")),
                    ("longitude_f64_bits", string("c057917a1fcc064c")),
                    ("source_full_i", number(989)),
                    ("source_full_j", number(638)),
                ]),
            ),
            ("status", string("PASS_REAL_COORDINATE_SELECTED_INTERIOR_ROI")),
            (
                "target",
                object(vec![
                    ("anchor_i", number(157)),
                    ("anchor_j", number(149)),
                    ("anchor_latitude_f64_bits", string("4044b274d4e2432e")),
                    ("anchor_longitude_f64_bits", string("c05791842c33cd17")),
                    ("center_lat_f64_bits", string("4044b28393b636d0")),
                    ("center_lon_f64_bits", string("c05792177a2c3f9c")),
                    ("distance_to_nearest_edge_cells", number(142)),
                    ("dx_m_f64_bits", string("4059000000000000")),
                    ("dy_m_f64_bits", string("4059000000000000")),
                    ("dz_m_f64_bits", string("4069000000000000")),
                    ("nx", number(300)),
                    ("ny", number(300)),
                    ("nz", number(75)),
                ]),
            ),
        ]);
        Self { value }
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.value.canonical_bytes()
    }

    pub fn sha256(&self) -> String {
        sha256::digest_hex(&self.canonical_bytes())
    }

    pub fn write_new(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut bytes = self.canonical_bytes();
        bytes.push(b'\n');
        atomic_write_new(path.as_ref(), &bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let expected = Self::sealed();
        let value =
            json::parse(bytes).map_err(|error| invalid(format!("f02 ROI receipt: {error}")))?;
        if value.canonical_bytes() != bytes || expected.canonical_bytes() != bytes {
            return Err(invalid(
                "f02 ROI receipt differs from the sealed coordinate-selected identity",
            ));
        }
        Ok(expected)
    }
}

fn stretch_hashes_json(grid: &StretchZ1Grid) -> Result<JsonValue> {
    Ok(object(vec![
        ("combined_sha256", string(&grid.combined_sha256())),
        ("mf_sha256", string(&grid.profile_sha256("mf")?)),
        ("mh_sha256", string(&grid.profile_sha256("mh")?)),
        ("rmf_sha256", string(&grid.profile_sha256("rmf")?)),
        ("rmh_sha256", string(&grid.profile_sha256("rmh")?)),
        ("sigma_sha256", string(&grid.profile_sha256("sigma")?)),
        ("sigmaf_sha256", string(&grid.profile_sha256("sigmaf")?)),
        ("zf_sha256", string(&grid.profile_sha256("zf")?)),
        ("zh_sha256", string(&grid.profile_sha256("zh")?)),
    ]))
}

fn moving_state_json(state: MovingDomainState) -> JsonValue {
    object(vec![
        ("adaptmovetim_f64_bits", f64_bits(state.adaptmovetim)),
        ("domainlocx_f64_bits", f64_bits(state.domainlocx)),
        ("domainlocy_f64_bits", f64_bits(state.domainlocy)),
        ("icenter", i32_number(state.icenter)),
        ("jcenter", i32_number(state.jcenter)),
        ("mvrec", i32_number(state.mvrec)),
        ("nwritemv", i32_number(state.nwritemv)),
        ("umove_f32_bits", f32_bits(state.umove)),
        ("umovem1_f32_bits", f32_bits(state.umovem1)),
        ("umovem2_f32_bits", f32_bits(state.umovem2)),
        ("vmove_f32_bits", f32_bits(state.vmove)),
        ("vmovem1_f32_bits", f32_bits(state.vmovem1)),
        ("vmovem2_f32_bits", f32_bits(state.vmovem2)),
        ("xcenter_f32_bits", f32_bits(state.xcenter)),
        ("ycenter_f32_bits", f32_bits(state.ycenter)),
    ])
}

fn parse_moving_selection(value: &JsonValue) -> Result<MovingDomainSelection> {
    exact_keys(
        value,
        &[
            "adapt_move_frq_f32_bits",
            "axisymm",
            "do_adapt_move",
            "imove",
            "initsfc",
            "lateral_bc_west_east_south_north",
            "mode",
            "terrain",
        ],
        "moving selection",
    )?;
    let mode = MovingDomainMode::parse(string_member(value, "mode", "moving selection")?)?;
    let imove = usize_member(value, "imove", "moving selection")?;
    let adaptive = bool_member(value, "do_adapt_move", "moving selection")?;
    if imove != usize::from(mode != MovingDomainMode::Fixed)
        || adaptive != (mode == MovingDomainMode::Adaptive)
    {
        return Err(invalid(
            "moving selection mode/imove/do_adapt_move disagree",
        ));
    }
    let initsfc = u8::try_from(usize_member(value, "initsfc", "moving selection")?)
        .map_err(|_| invalid("moving selection initsfc does not fit u8"))?;
    let bc_value = value
        .get("lateral_bc_west_east_south_north")
        .map_err(json_error)?;
    let bc_array = bc_value
        .as_array()
        .map_err(|error| invalid(format!("moving selection boundaries: {error}")))?;
    if bc_array.len() != 4 {
        return Err(invalid("moving selection requires four lateral boundaries"));
    }
    let mut bc = [0u8; 4];
    for (index, entry) in bc_array.iter().enumerate() {
        bc[index] = u8::try_from(
            entry
                .as_u64()
                .map_err(|error| invalid(format!("moving boundary {index}: {error}")))?,
        )
        .map_err(|_| invalid("moving boundary code does not fit u8"))?;
    }
    Ok(MovingDomainSelection {
        mode,
        terrain: bool_member(value, "terrain", "moving selection")?,
        axisymm: bool_member(value, "axisymm", "moving selection")?,
        initsfc,
        lateral_bc_west_east_south_north: bc,
        adapt_move_frq: f32_member(value, "adapt_move_frq_f32_bits", "moving selection")?,
    })
}

fn parse_moving_state(value: &JsonValue) -> Result<MovingDomainState> {
    exact_keys(
        value,
        &[
            "adaptmovetim_f64_bits",
            "domainlocx_f64_bits",
            "domainlocy_f64_bits",
            "icenter",
            "jcenter",
            "mvrec",
            "nwritemv",
            "umove_f32_bits",
            "umovem1_f32_bits",
            "umovem2_f32_bits",
            "vmove_f32_bits",
            "vmovem1_f32_bits",
            "vmovem2_f32_bits",
            "xcenter_f32_bits",
            "ycenter_f32_bits",
        ],
        "moving state",
    )?;
    Ok(MovingDomainState {
        icenter: i32_member(value, "icenter", "moving state")?,
        jcenter: i32_member(value, "jcenter", "moving state")?,
        xcenter: f32_member(value, "xcenter_f32_bits", "moving state")?,
        ycenter: f32_member(value, "ycenter_f32_bits", "moving state")?,
        umove: f32_member(value, "umove_f32_bits", "moving state")?,
        umovem1: f32_member(value, "umovem1_f32_bits", "moving state")?,
        umovem2: f32_member(value, "umovem2_f32_bits", "moving state")?,
        vmove: f32_member(value, "vmove_f32_bits", "moving state")?,
        vmovem1: f32_member(value, "vmovem1_f32_bits", "moving state")?,
        vmovem2: f32_member(value, "vmovem2_f32_bits", "moving state")?,
        domainlocx: f64_member(value, "domainlocx_f64_bits", "moving state")?,
        domainlocy: f64_member(value, "domainlocy_f64_bits", "moving state")?,
        adaptmovetim: f64_member(value, "adaptmovetim_f64_bits", "moving state")?,
        mvrec: i32_member(value, "mvrec", "moving state")?,
        nwritemv: i32_member(value, "nwritemv", "moving state")?,
    })
}

fn parse_stretch_parameters(value: &JsonValue) -> Result<StretchZ1Parameters> {
    exact_keys(
        value,
        &[
            "dz_bot_f32_bits",
            "dz_f32_bits",
            "dz_top_f32_bits",
            "nk",
            "str_bot_f32_bits",
            "str_top_f32_bits",
            "ztop_f32_bits",
        ],
        "stretch parameters",
    )?;
    Ok(StretchZ1Parameters {
        nk: usize_member(value, "nk", "stretch parameters")?,
        dz: f32_member(value, "dz_f32_bits", "stretch parameters")?,
        ztop: f32_member(value, "ztop_f32_bits", "stretch parameters")?,
        str_bot: f32_member(value, "str_bot_f32_bits", "stretch parameters")?,
        str_top: f32_member(value, "str_top_f32_bits", "stretch parameters")?,
        dz_bot: f32_member(value, "dz_bot_f32_bits", "stretch parameters")?,
        dz_top: f32_member(value, "dz_top_f32_bits", "stretch parameters")?,
    })
}

fn validate_derived(value: &JsonValue, expected: StretchZ1Derived) -> Result<()> {
    exact_keys(
        value,
        &[
            "c1_f32_bits",
            "c2_f32_bits",
            "nk1",
            "nk2",
            "nk3",
            "nominal_dz_f32_bits",
        ],
        "stretch derived",
    )?;
    if usize_member(value, "nk1", "stretch derived")? != expected.nk1
        || usize_member(value, "nk2", "stretch derived")? != expected.nk2
        || usize_member(value, "nk3", "stretch derived")? != expected.nk3
        || f32_member(value, "nominal_dz_f32_bits", "stretch derived")?.to_bits()
            != expected.nominal_dz.to_bits()
        || f32_member(value, "c1_f32_bits", "stretch derived")?.to_bits() != expected.c1.to_bits()
        || f32_member(value, "c2_f32_bits", "stretch derived")?.to_bits() != expected.c2.to_bits()
    {
        return Err(invalid(
            "stretch_z=1 derived values differ from rebuilt source arithmetic",
        ));
    }
    Ok(())
}

fn validate_stretch_selection(value: &JsonValue) -> Result<()> {
    exact_keys(
        value,
        &["axisymm", "stretch_x", "stretch_y", "stretch_z", "terrain"],
        "stretch selection",
    )?;
    if bool_member(value, "axisymm", "stretch selection")?
        || bool_member(value, "terrain", "stretch selection")?
        || usize_member(value, "stretch_x", "stretch selection")? != 0
        || usize_member(value, "stretch_y", "stretch selection")? != 0
        || usize_member(value, "stretch_z", "stretch selection")? != 1
    {
        return Err(invalid(
            "bounded stretch_z=1 selection requires flat Cartesian stretch_z=1 only",
        ));
    }
    Ok(())
}

fn validate_cuda_proof(value: &JsonValue) -> Result<()> {
    exact_keys(
        value,
        &[
            "abi",
            "commit",
            "proof_sha256",
            "schema",
            "source_sha256",
            "status",
        ],
        "stretch CUDA proof",
    )?;
    fixed_string(value, "abi", CUDA_STRETCH_Z1_ABI, "stretch CUDA proof")?;
    fixed_string(value, "commit", STRETCH_Z1_COMMIT, "stretch CUDA proof")?;
    fixed_string(
        value,
        "proof_sha256",
        CUDA_STRETCH_Z1_PROOF_SHA256,
        "stretch CUDA proof",
    )?;
    fixed_string(
        value,
        "schema",
        CUDA_STRETCH_Z1_SCHEMA,
        "stretch CUDA proof",
    )?;
    fixed_string(
        value,
        "source_sha256",
        CUDA_STRETCH_Z1_SOURCE_SHA256,
        "stretch CUDA proof",
    )?;
    fixed_string(
        value,
        "status",
        "PASS_CPU_CUDA_RAW_BIT_EXACT",
        "stretch CUDA proof",
    )
}

fn authority_json(files: Vec<(&str, &str, &str)>) -> JsonValue {
    object(vec![
        (
            "files",
            JsonValue::Array(
                files
                    .into_iter()
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
        ("model_source", string("CM1 r21.1")),
        (
            "tarball_sha256",
            string(DOMAIN_GRID_AUTHORITY_TARBALL_SHA256),
        ),
    ])
}

fn validate_stretch_authority(value: &JsonValue) -> Result<()> {
    let expected = authority_json(vec![("src/param.F", PARAM_F_SHA256, "7814-8225")]);
    if value != &expected {
        return Err(invalid(
            "stretch_z=1 source authority differs from the sealed identity",
        ));
    }
    Ok(())
}

fn validate_moving_authority(value: &JsonValue) -> Result<()> {
    let expected = authority_json(vec![
        ("src/cm1.F", CM1_F_SHA256, "264-269,3512-3513,3642-3679"),
        ("src/azimavg.F", AZIMAVG_F_SHA256, "4400-4465"),
        ("src/param.F", PARAM_F_SHA256, "1890-1903"),
        (
            "src/restart_read.F",
            RESTART_READ_F_SHA256,
            "406-421,689-735",
        ),
        (
            "src/restart_write.F",
            RESTART_WRITE_F_SHA256,
            "386-401,503-549",
        ),
    ]);
    if value != &expected {
        return Err(invalid(
            "moving-domain source authority differs from the sealed identity",
        ));
    }
    Ok(())
}

fn add(left: f32, right: f32) -> f32 {
    left + right
}

fn sub(left: f32, right: f32) -> f32 {
    left - right
}

fn mul(left: f32, right: f32) -> f32 {
    left * right
}

fn div(left: f32, right: f32) -> f32 {
    left / right
}

fn nint_positive(value: f32) -> Result<usize> {
    if !value.is_finite() || value < 0.0 {
        return Err(invalid("stretch_z=1 NINT input is invalid"));
    }
    let rounded = (f64::from(value) + 0.5).floor();
    if rounded > usize::MAX as f64 {
        return Err(invalid("stretch_z=1 NINT result does not fit usize"));
    }
    Ok(rounded as usize)
}

fn append_f32_bytes(output: &mut Vec<u8>, values: &[f32]) {
    output.reserve(values.len() * 4);
    for value in values {
        output.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn hash_f32(values: &[f32]) -> String {
    let mut bytes = Vec::new();
    append_f32_bytes(&mut bytes, values);
    sha256::digest_hex(&bytes)
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

fn i32_number(value: i32) -> JsonValue {
    JsonValue::Number(value.to_string())
}

fn f32_bits(value: f32) -> JsonValue {
    string(&format!("{:08x}", value.to_bits()))
}

fn f64_bits(value: f64) -> JsonValue {
    string(&format!("{:016x}", value.to_bits()))
}

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) -> Result<()> {
    let mut actual = value
        .as_object()
        .map_err(|error| invalid(format!("{label}: {error}")))?
        .iter()
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err(invalid(format!(
            "{label} keys are {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn fixed_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = string_member(value, key, label)?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} differs from the sealed value"
        )));
    }
    Ok(())
}

fn string_member<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn bool_member(value: &JsonValue, key: &str, label: &str) -> Result<bool> {
    match value.get(key).map_err(json_error)? {
        JsonValue::Bool(result) => Ok(*result),
        _ => Err(invalid(format!("{label}.{key} must be a JSON boolean"))),
    }
}

fn usize_member(value: &JsonValue, key: &str, label: &str) -> Result<usize> {
    let raw = value
        .get(key)
        .and_then(JsonValue::as_u64)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
    usize::try_from(raw).map_err(|_| invalid(format!("{label}.{key} does not fit usize")))
}

fn i32_member(value: &JsonValue, key: &str, label: &str) -> Result<i32> {
    let raw = value
        .get(key)
        .and_then(JsonValue::as_i64)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))?;
    i32::try_from(raw).map_err(|_| invalid(format!("{label}.{key} does not fit i32")))
}

fn f32_member(value: &JsonValue, key: &str, label: &str) -> Result<f32> {
    let word = string_member(value, key, label)?;
    if word.len() != 8 || !word.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(format!("{label}.{key} is not eight hex digits")));
    }
    let result = f32::from_bits(
        u32::from_str_radix(word, 16)
            .map_err(|_| invalid(format!("{label}.{key} does not parse")))?,
    );
    if !result.is_finite() {
        return Err(invalid(format!("{label}.{key} is not finite")));
    }
    Ok(result)
}

fn f64_member(value: &JsonValue, key: &str, label: &str) -> Result<f64> {
    let word = string_member(value, key, label)?;
    if word.len() != 16 || !word.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(format!("{label}.{key} is not sixteen hex digits")));
    }
    let result = f64::from_bits(
        u64::from_str_radix(word, 16)
            .map_err(|_| invalid(format!("{label}.{key} does not parse")))?,
    );
    if !result.is_finite() {
        return Err(invalid(format!("{label}.{key} is not finite")));
    }
    Ok(result)
}

fn json_error(error: String) -> StateError {
    invalid(error)
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_state() -> MovingDomainState {
        MovingDomainState {
            icenter: 151,
            jcenter: 151,
            xcenter: 15_000.0,
            ycenter: 15_000.0,
            umove: 0.0,
            umovem1: -0.0,
            umovem2: 0.0,
            vmove: -0.0,
            vmovem1: 0.0,
            vmovem2: -0.0,
            domainlocx: 0.0,
            domainlocy: -0.0,
            adaptmovetim: 0.0,
            mvrec: 1,
            nwritemv: 1,
        }
    }

    #[test]
    fn selected_stretch_grid_matches_cpu_cuda_proof_hashes() {
        let grid = build_stretch_z1(StretchZ1Parameters::gm1_selected()).unwrap();
        assert_eq!(
            (grid.derived.nk1, grid.derived.nk2, grid.derived.nk3),
            (10, 45, 20)
        );
        assert_eq!(
            grid.profile_sha256("sigmaf").unwrap(),
            "e77369ceb8181996e9695b73c28175a84a819bed9e8e9cadd575ca91d6de0837"
        );
        assert_eq!(
            grid.profile_sha256("sigma").unwrap(),
            "41e849887b91a7f832ec5afb4e542462bbb96b7c857db6479b688b6c62a70d85"
        );
        assert_eq!(
            grid.profile_sha256("mh").unwrap(),
            "a04a45f76476a6f639242a037ce3285115870ac2ff66bc7bfa591fda06ad18d6"
        );
        assert_eq!(
            grid.profile_sha256("mf").unwrap(),
            "919c1ec2f4daedfb2c4c7581496b8129278675085e9b8f0ecec29257a8cb869a"
        );
        assert_eq!(
            grid.profile_sha256("rmh").unwrap(),
            "7d3e116310a0fb0e4e016ba5b4a517fdba1488462c21b51eb9ea74b3673c730c"
        );
        assert_eq!(
            grid.profile_sha256("rmf").unwrap(),
            "8d00ac691878f2a15127ccdcb6674c13e6ff4e128cba21161f295fbbd87f2878"
        );
        assert_eq!(grid.combined_sha256(), STRETCH_Z1_GRID_SHA256);
    }

    #[test]
    fn stretch_component_roundtrip_rebuilds_and_rejects_drift() {
        let parameters = StretchZ1Parameters::gm1_selected();
        let bytes = encode_stretch_z1_component(parameters).unwrap();
        let (decoded, grid) = decode_stretch_z1_component(&bytes).unwrap();
        assert_eq!(decoded, parameters);
        assert_eq!(grid.combined_sha256(), STRETCH_Z1_GRID_SHA256);
        let mut drift = bytes;
        let offset = drift
            .windows(8)
            .position(|word| word == b"42c80000")
            .unwrap();
        drift[offset + 7] = b'1';
        assert!(decode_stretch_z1_component(&drift).is_err());
    }

    #[test]
    fn moving_component_preserves_every_raw_word_and_counter() {
        let selection = MovingDomainSelection {
            mode: MovingDomainMode::Adaptive,
            terrain: false,
            axisymm: false,
            initsfc: 1,
            lateral_bc_west_east_south_north: [2, 2, 2, 2],
            adapt_move_frq: f32::from_bits(0x447a_0000),
        };
        let state = MovingDomainState {
            icenter: 141,
            jcenter: 163,
            xcenter: f32::from_bits(0x465a_6000),
            ycenter: f32::from_bits(0x4669_9000),
            umove: f32::from_bits(0x3fc0_0001),
            umovem1: f32::from_bits(0x3fa0_0001),
            umovem2: f32::from_bits(0x3f80_0001),
            vmove: f32::from_bits(0xbfa0_0001),
            vmovem1: f32::from_bits(0xbf80_0001),
            vmovem2: f32::from_bits(0xbf40_0001),
            domainlocx: f64::from_bits(0x40c8_1cd6_e631_f8a1),
            domainlocy: f64::from_bits(0xc0b8_1cd6_e631_f8a1),
            adaptmovetim: f64::from_bits(0x40c3_8800_0000_0001),
            mvrec: 27,
            nwritemv: 18,
        };
        let bytes = encode_moving_domain_component(selection, state, 300, 300).unwrap();
        let (restored_selection, restored_state) = decode_moving_domain_component(&bytes).unwrap();
        assert_eq!(restored_selection, selection);
        assert_eq!(restored_state, state);
    }

    #[test]
    fn source_conflicts_fail_closed() {
        let moving = MovingDomainSelection {
            mode: MovingDomainMode::Constant,
            terrain: true,
            axisymm: false,
            initsfc: 1,
            lateral_bc_west_east_south_north: [2; 4],
            adapt_move_frq: 0.0,
        };
        assert!(encode_moving_domain_component(moving, fixed_state(), 300, 300).is_err());
        let rigid = MovingDomainSelection {
            mode: MovingDomainMode::Constant,
            terrain: false,
            axisymm: false,
            initsfc: 1,
            lateral_bc_west_east_south_north: [1, 2, 2, 2],
            adapt_move_frq: 0.0,
        };
        let mut state = fixed_state();
        state.umove = 1.0;
        assert!(encode_moving_domain_component(rigid, state, 300, 300).is_err());
    }

    #[test]
    fn fixed_domain_accepts_signed_zero_but_not_hidden_motion() {
        let selection = MovingDomainSelection {
            mode: MovingDomainMode::Fixed,
            terrain: true,
            axisymm: false,
            initsfc: 1,
            lateral_bc_west_east_south_north: [2; 4],
            adapt_move_frq: 0.0,
        };
        let state = fixed_state();
        let bytes = encode_moving_domain_component(selection, state, 300, 300).unwrap();
        let (_, restored) = decode_moving_domain_component(&bytes).unwrap();
        assert_eq!(restored.vmovem2.to_bits(), 0x8000_0000);
        assert_eq!(restored.domainlocy.to_bits(), 0x8000_0000_0000_0000);
        let mut moved = state;
        moved.domainlocx = f64::from_bits(1);
        assert!(encode_moving_domain_component(selection, moved, 300, 300).is_err());
    }

    #[test]
    fn f02_roi_receipt_is_canonical_and_fail_closed() {
        let receipt = F02InteriorRoiReceipt::sealed();
        let bytes = receipt.canonical_bytes();
        assert_eq!(F02InteriorRoiReceipt::parse(&bytes).unwrap(), receipt);
        let mut drift = bytes;
        let offset = drift.windows(3).position(|word| word == b"142").unwrap();
        drift[offset + 2] = b'1';
        assert!(F02InteriorRoiReceipt::parse(&drift).is_err());
    }
}
