"""Profile-aware solve1 scalar turbulence for the real stretched GM1 child.

The legacy scalar-turbulence operator is exact only when every vertical
interpolation weight is one half and every terrain cross derivative uses one
uniform ``1/dz``.  CM1's ``stretch_z=1`` terrain branch instead consumes
``sigma``, ``sigmaf``, ``rds``, and ``rdsf`` at those seams.  This module
patches only those source-audited expressions, binds the already sealed
``Gm1ChildStretchedMetrics`` object, and keeps the reference ``dz`` solely in
the source-defined implicit vertical Thomas solve.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .cuda_gm1_child_stretched_metrics import (
    GM1_CHILD_STRETCHED_GRID_SHA256,
    GM1_CHILD_STRETCHED_PROFILE_SHA256,
    GM1_CHILD_STRETCHED_PROOF_SHA256,
    Gm1ChildStretchedMetrics,
)
from .cuda_gm1_scalar_turbulence import (
    ALL_ARRAY_NAMES,
    AUTHORITY_SOURCE_SHA256,
    AUTHORITY_TARBALL_SHA256,
    CUDA_GM1_SCALAR_TURBULENCE_KERNEL_NAMES,
    CUDA_GM1_SCALAR_TURBULENCE_SOURCE,
    CUDA_GM1_SCALAR_TURBULENCE_SOURCE_SHA256,
    CUDA_GM1_SCALAR_TURBULENCE_THREADS,
    DEFAULT_GM1_SCALAR_TURBULENCE_SELECTION,
    GM1_SCALAR_TURBULENCE_FIELD_ORDER,
    CompiledGm1ScalarTurbulenceKernels,
    CudaGm1ScalarTurbulenceLaunch,
    Gm1ScalarTurbulenceSelection,
    Gm1ScalarTurbulenceWorkspace,
    _canonical_arch,
    _spacing,
    _validate_grid,
    _validate_selection,
    _validate_workspace,
)


CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SCHEMA = (
    "gm1.cuda-child-stretched-scalar-turbulence/v1"
)
CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_ABI = (
    "gm1.cuda-child-stretched-scalar-turbulence-kernels/v1"
)
CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_KERNEL_NAMES = tuple(
    name.replace("_v1", "_stretched_v1")
    for name in CUDA_GM1_SCALAR_TURBULENCE_KERNEL_NAMES
)
CM1_TURBTEND_F_SHA256 = AUTHORITY_SOURCE_SHA256["src/turbtend.F"]
CM1_SOLVE1_F_SHA256 = AUTHORITY_SOURCE_SHA256["src/solve1.F"]
_CLAIM_LIMIT = (
    "GM1 child solve1 theta-plus-six-moisture scalar turbulent-diffusion held "
    "contribution on the sealed 15 km stretch_z=1 terrain grid: source-order "
    "sigma/rds interpolation, rdsf terrain cross derivatives, 3-D child th0, "
    "and the unchanged reference-dz implicit vertical Thomas solve; no scalar "
    "advection/pdef, momentum stress, TKE, idiss, RK, full trajectory, forecast "
    "skill, or tornado claim"
)


def _replace_once(source: str, old: str, new: str, *, seam: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"legacy scalar-turbulence source drifted at {seam}")
    return source.replace(old, new, 1)


def _stretched_source() -> str:
    source = _replace_once(
        CUDA_GM1_SCALAR_TURBULENCE_SOURCE,
        "// gm1.cuda-scalar-turbulence-kernels/v1",
        f"// {CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_ABI}",
        seam="ABI",
    )
    for old, new in zip(
        CUDA_GM1_SCALAR_TURBULENCE_KERNEL_NAMES,
        CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_KERNEL_NAMES,
        strict=True,
    ):
        source = _replace_once(source, old, new, seam=f"kernel name {old}")

    # Child th0 is terrain-varying [k,j,i], not the legacy 1-D profile.
    source = _replace_once(
        source,
        "A(S(th0[k],300.0f),tha[p])",
        "A(S(th0[p],300.0f),tha[p])",
        seam="three-dimensional base theta",
    )

    source = _replace_once(
        source,
        "const float*sh,float*sw,unsigned ni,unsigned nj,unsigned nk){",
        "const float*sh,const float*sigma,const float*sigmaf,const float*rds,float*sw,unsigned ni,unsigned nj,unsigned nk){",
        seam="scalar w reconstruction signature",
    )
    source = _replace_once(
        source,
        "else sw[z]=A(M(0.5f,sh[(unsigned long long)(f-1u)*hxy+p]),M(0.5f,sh[(unsigned long long)f*hxy+p]));",
        "else {float r2=M(S(sigmaf[f+1u],sigma[f]),rds[f+1u]),r1=S(1.0f,r2);sw[z]=A(M(r1,sh[(unsigned long long)(f-1u)*hxy+p]),M(r2,sh[(unsigned long long)f*hxy+p]));}",
        seam="scalar w reconstruction weights",
    )

    source = _replace_once(
        source,
        "const float*rgzu,const float*rgzv,const float*gxu,const float*gyv,float*fx,float*fy,\n unsigned ni,unsigned nj,unsigned nk,float dx,float dy,float dz){",
        "const float*rgzu,const float*rgzv,const float*gxu,const float*gyv,const float*rdsf,float*fx,float*fy,\n unsigned ni,unsigned nj,unsigned nk,float dx,float dy){",
        seam="horizontal flux profile signature",
    )
    source = _replace_once(
        source,
        "ny=(unsigned long long)nk*(nj+1u)*ni;float rx=D(1.0f,dx),ry=D(1.0f,dy),rz=D(1.0f,dz);",
        "ny=(unsigned long long)nk*(nj+1u)*ni;float rx=D(1.0f,dx),ry=D(1.0f,dy);",
        seam="horizontal flux uniform rz",
    )
    source = _replace_once(
        source,
        "ter=M(M(M(0.5f,S(up,lo)),rz),rgzu[(unsigned long long)j*(ni+1u)+f])",
        "ter=M(M(M(0.5f,S(up,lo)),rdsf[k+1u]),rgzu[(unsigned long long)j*(ni+1u)+f])",
        seam="x-flux rdsf",
    )
    source = _replace_once(
        source,
        "ter=M(M(M(0.5f,S(up,lo)),rz),rgzv[(unsigned long long)f*ni+i])",
        "ter=M(M(M(0.5f,S(up,lo)),rdsf[k+1u]),rgzv[(unsigned long long)f*ni+i])",
        seam="y-flux rdsf",
    )

    source = _replace_once(
        source,
        "const float*fx,const float*fy,float*fxw,float*fyw,unsigned ni,unsigned nj,unsigned nk){",
        "const float*fx,const float*fy,const float*sigma,const float*sigmaf,const float*rds,float*fxw,float*fyw,unsigned ni,unsigned nj,unsigned nk){",
        seam="flux interpolation signature",
    )
    source = _replace_once(
        source,
        "else fxw[z]=A(M(0.5f,fx[(unsigned long long)(f-1u)*nx+p]),M(0.5f,fx[(unsigned long long)f*nx+p]));",
        "else {float r2=M(S(sigmaf[f+1u],sigma[f]),rds[f+1u]),r1=S(1.0f,r2);fxw[z]=A(M(r1,fx[(unsigned long long)(f-1u)*nx+p]),M(r2,fx[(unsigned long long)f*nx+p]));}",
        seam="x-flux interpolation weights",
    )
    source = _replace_once(
        source,
        "else fyw[z]=A(M(0.5f,fy[(unsigned long long)(f-1u)*ny+p]),M(0.5f,fy[(unsigned long long)f*ny+p]));",
        "else {float r2=M(S(sigmaf[f+1u],sigma[f]),rds[f+1u]),r1=S(1.0f,r2);fyw[z]=A(M(r1,fy[(unsigned long long)(f-1u)*ny+p]),M(r2,fy[(unsigned long long)f*ny+p]));}",
        seam="y-flux interpolation weights",
    )

    source = _replace_once(
        source,
        "const float*gz,const float*rgzu,const float*rgzv,const float*gx,const float*gy,\n const float*fx,const float*fy,const float*fxw,const float*fyw,const float*sflux,float*cp,float*dp,float*target,",
        "const float*gz,const float*rgzu,const float*rgzv,const float*gx,const float*gy,const float*rdsf,\n const float*fx,const float*fy,const float*fxw,const float*fyw,const float*sflux,float*cp,float*dp,float*target,",
        seam="implicit/tendency rdsf signature",
    )
    source = _replace_once(
        source,
        "),0.5f),rz));if(i==0u||i+1u==ni)tx=0.0f;",
        "),0.5f),rdsf[ku+1u]));if(i==0u||i+1u==ni)tx=0.0f;",
        seam="x tendency rdsf",
    )
    source = _replace_once(
        source,
        "),0.5f),rz));if(j==0u||j+1u==nj)ty=0.0f;",
        "),0.5f),rdsf[ku+1u]));if(j==0u||j+1u==nj)ty=0.0f;",
        seam="y tendency rdsf",
    )
    return source


CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE = _stretched_source()
CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_BYTES = (
    CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE.encode("utf-8")
)
# Updated only after the reviewed source transformation above changes.
CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256 = (
    "7f57b86da8aaf2f3d80a61e1f7dfada465cc2878f6d5aeaf6fccb5a6e3950506"
)


def child_stretched_scalar_turbulence_shapes(
    ni: int, nj: int, nk: int, numq: int = 6
) -> dict[str, tuple[int, ...]]:
    """``src/solve1.F:1213-1231``: ``turbs`` runs once per ``n=1,numq``."""

    _validate_grid(ni, nj, nk)
    if isinstance(numq, bool) or not isinstance(numq, int) or numq < 1:
        raise ValueError("numq must be a positive integer")
    mass = (nk, nj, ni)
    zface = (nk + 1, nj, ni)
    return {
        "th0": mass,
        "tha": mass,
        "qa": (numq, *mass),
        "rho": mass,
        "rr": mass,
        "rf": zface,
        "mh": mass,
        "mf": zface,
        "khh": zface,
        "khv": zface,
        "gz": (nj, ni),
        "rgz": (nj, ni),
        "rgzu": (nj, ni + 1),
        "rgzv": (nj + 1, ni),
        "gx": zface,
        "gxu": (nk + 1, nj, ni + 1),
        "gy": zface,
        "gyv": (nk + 1, nj + 1, ni),
        "thten1": mass,
        "qten": (numq, *mass),
    }


def _require_device_arrays(
    cupy: Any, arrays: Mapping[str, Any], *, ni: int, nj: int, nk: int
) -> tuple[dict[str, Any], int]:
    missing = [name for name in ALL_ARRAY_NAMES if name not in arrays]
    if missing:
        raise ValueError(
            f"missing child scalar-turbulence arrays: {', '.join(missing)}"
        )
    # src/solve1.F:1213-1231 calls turbs once per n=1,numq, so the scalar axis
    # the caller bound is the axis diffused, and qten must carry the same rows.
    qa_shape = tuple(getattr(arrays["qa"], "shape", ()))
    if len(qa_shape) != 4 or int(qa_shape[0]) < 1:
        raise ValueError("qa must be C-order CUDA float32 shape (numq, nk, nj, ni)")
    numq = int(qa_shape[0])
    shapes = child_stretched_scalar_turbulence_shapes(ni, nj, nk, numq)
    result = {}
    for name in ALL_ARRAY_NAMES:
        value = arrays[name]
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shapes[name]
            or not value.flags.c_contiguous
        ):
            raise ValueError(
                f"{name} must be C-order CUDA float32 shape {shapes[name]}"
            )
        result[name] = value
    if len({int(value.data.ptr) for value in result.values()}) != len(result):
        raise ValueError("child scalar-turbulence arrays must not alias")
    return result, numq


def _require_vertical(
    cupy: Any,
    vertical: Gm1ChildStretchedMetrics,
    *,
    compiled: CompiledGm1ScalarTurbulenceKernels,
    allow_compact_validation: bool,
    allow_uniform_idealized_validation: bool,
) -> None:
    if not isinstance(vertical, Gm1ChildStretchedMetrics):
        raise TypeError("vertical must be the exact Gm1ChildStretchedMetrics binding")
    for name, shape in (
        ("sigma", (77,)),
        ("sigmaf", (78,)),
        ("rds", (77,)),
        ("rdsf", (78,)),
    ):
        value = getattr(vertical, name)
        if (
            not isinstance(value, cupy.ndarray)
            or value.dtype != cupy.float32
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ValueError(f"vertical.{name} must be C-order CUDA float32 {shape}")
    if len({int(value.data.ptr) for value in vertical.arrays.values()}) != 4:
        raise ValueError("child stretched vertical profiles alias")
    if vertical.target_arch != compiled.target_arch:
        raise ValueError(
            "scalar-turbulence and stretched profiles target different architectures"
        )
    if allow_uniform_idealized_validation:
        if (
            float(vertical.ztop_f32) != 15000.0
            or float(vertical.dz_reference_f32) != 200.0
            or vertical.identity.get("fixture_class")
            != "source-audited-flat-idealized-uniform-200m"
            or vertical.identity.get("ztop_f32") != 15000.0
            or vertical.identity.get("dz_reference_f32") != 200.0
        ):
            raise ValueError(
                "uniform idealized scalar-turbulence vertical identity changed"
            )
    elif not allow_compact_validation:
        expected = {
            "grid_sha256": GM1_CHILD_STRETCHED_GRID_SHA256,
            "proof_sha256": GM1_CHILD_STRETCHED_PROOF_SHA256,
            "profile_sha256": dict(GM1_CHILD_STRETCHED_PROFILE_SHA256),
        }
        for name, wanted in expected.items():
            if vertical.identity.get(name) != wanted:
                raise ValueError(
                    f"vertical identity {name} differs from the sealed child"
                )


def compile_cuda_gm1_child_stretched_scalar_turbulence() -> (
    CompiledGm1ScalarTurbulenceKernels
):
    actual = hashlib.sha256(
        CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_BYTES
    ).hexdigest()
    if actual != CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256:
        raise RuntimeError(
            f"child stretched scalar-turbulence source pin differs: {actual}"
        )
    cupy = importlib.import_module("cupy")
    compiler = importlib.import_module("cupy.cuda.compiler")
    function = importlib.import_module("cupy.cuda.function")
    device = cupy.cuda.Device()
    arch = _canonical_arch(device.compute_capability)
    if arch != _canonical_arch(compiler._get_arch()):
        raise ValueError("NVRTC target differs from active CUDA device")
    nvrtc = tuple(compiler._get_nvrtc_version())
    common = (
        "--std=c++14",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    if int(nvrtc[0]) >= 12:
        common += ("--device-as-default-execution-space",)
    ptx_options = common + (f"--gpu-architecture=compute_{arch}",)
    cubin_options = common + (f"--gpu-architecture=sm_{arch}",)
    ptx, _ = compiler._NVRTCProgram(
        CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE,
        "gm1_child_stretched_scalar_turbs_ptx.cu",
        method="ptx",
    ).compile(ptx_options)
    cubin, _ = compiler._NVRTCProgram(
        CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE,
        "gm1_child_stretched_scalar_turbs_cubin.cu",
        method="cubin",
    ).compile(cubin_options)
    ptx = ptx.encode("utf-8") if isinstance(ptx, str) else bytes(ptx)
    cubin = cubin.encode("latin-1") if isinstance(cubin, str) else bytes(cubin)
    decoded = ptx.decode("utf-8")
    if not all(
        re.search(rf"\.entry\s+{name}(?:\s|\()", decoded)
        for name in CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_KERNEL_NAMES
    ):
        raise ValueError("compiled PTX lacks child stretched scalar-turbulence entries")
    if (
        len(cubin) < 20
        or cubin[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(cubin[18:20], "little") != 190
    ):
        raise ValueError("compiled child scalar-turbulence image is not ELF64 CUDA")
    module = function.Module()
    module.load(cubin)
    kernels = tuple(
        module.get_function(name)
        for name in CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_KERNEL_NAMES
    )
    properties = cupy.cuda.runtime.getDeviceProperties(int(device.id))
    device_name = properties.get("name", "unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    identity = (
        ("cupy_version", str(cupy.__version__)),
        ("cuda_driver_version", str(cupy.cuda.runtime.driverGetVersion())),
        ("cuda_runtime_version", str(cupy.cuda.runtime.runtimeGetVersion())),
        ("nvrtc_version", f"{int(nvrtc[0])}.{int(nvrtc[1])}"),
        ("device_id", str(int(device.id))),
        ("device_name", str(device_name)),
    )
    return CompiledGm1ScalarTurbulenceKernels(
        module, kernels, ptx, cubin, arch, ptx_options, cubin_options, identity
    )


def launch_cuda_gm1_child_stretched_scalar_turbulence(
    arrays: Mapping[str, Any],
    *,
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    dy: float,
    dt: float,
    vertical: Gm1ChildStretchedMetrics,
    selection: Gm1ScalarTurbulenceSelection = DEFAULT_GM1_SCALAR_TURBULENCE_SELECTION,
    stream: Any | None = None,
    compiled: CompiledGm1ScalarTurbulenceKernels | None = None,
    workspace: Gm1ScalarTurbulenceWorkspace | None = None,
    allow_compact_validation: bool = False,
    allow_uniform_idealized_validation: bool = False,
    scalar_names: Sequence[str] | None = None,
) -> CudaGm1ScalarTurbulenceLaunch:
    """Enqueue profile-aware theta and every q3d scalar's held contribution.

    ``scalar_names`` is CM1's ``qname`` for each ``qa`` row (the component
    profile's ``names``), which the receipt's ``source_call_order`` publishes.
    A six-row axis without it is WSM6's ``qv..qg``; any other axis must name
    its rows, or the receipt could not say which scalar each call diffused.
    """

    _validate_selection(selection)
    _validate_grid(ni, nj, nk)
    if type(allow_compact_validation) is not bool:
        raise TypeError("allow_compact_validation must be bool")
    if type(allow_uniform_idealized_validation) is not bool:
        raise TypeError("allow_uniform_idealized_validation must be bool")
    if allow_uniform_idealized_validation and (
        not allow_compact_validation or nk != 75
    ):
        raise ValueError(
            "uniform idealized scalar turbulence requires a tiled 75-level validation launch"
        )
    if (
        not allow_compact_validation
        and not allow_uniform_idealized_validation
        and (ni, nj, nk) != (300, 300, 75)
    ):
        raise ValueError(
            "production scalar turbulence requires the sealed 300x300x75 child"
        )
    dx, dy, dt = (
        _spacing(name, value) for name, value in (("dx", dx), ("dy", dy), ("dt", dt))
    )
    if not allow_compact_validation and (dx, dy) != (50.0, 50.0):
        raise ValueError("production scalar turbulence requires dx=dy=50 m")
    cupy = importlib.import_module("cupy")
    a, numq = _require_device_arrays(cupy, arrays, ni=ni, nj=nj, nk=nk)
    wsm6_names = tuple(GM1_SCALAR_TURBULENCE_FIELD_ORDER[1:])
    if scalar_names is None:
        if numq != len(wsm6_names):
            raise ValueError(
                f"qa carries numq={numq} rows; pass scalar_names (CM1's qname for "
                "each row) so the receipt's source_call_order names every scalar "
                "turbs diffused"
            )
        scalar_names = wsm6_names
    scalar_names = tuple(scalar_names)
    if (
        len(scalar_names) != numq
        or any(not isinstance(name, str) or not name for name in scalar_names)
        or len(set(scalar_names)) != numq
    ):
        raise ValueError(
            f"scalar_names {scalar_names!r} must be {numq} distinct CM1 qnames, "
            "one per qa row"
        )
    compiled = compiled or compile_cuda_gm1_child_stretched_scalar_turbulence()
    _require_vertical(
        cupy,
        vertical,
        compiled=compiled,
        allow_compact_validation=allow_compact_validation,
        allow_uniform_idealized_validation=allow_uniform_idealized_validation,
    )
    stream = stream or cupy.cuda.get_current_stream()
    if int(stream.ptr) == 0:
        raise ValueError("scalar-turbulence stream pointer must be nonzero")
    if workspace is None:
        from .cuda_gm1_scalar_turbulence import (
            create_cuda_gm1_scalar_turbulence_workspace,
        )

        workspace = create_cuda_gm1_scalar_turbulence_workspace(ni=ni, nj=nj, nk=nk)
    _validate_workspace(
        cupy,
        workspace,
        ni=ni,
        nj=nj,
        nk=nk,
        input_pointers={int(value.data.ptr) for value in a.values()}
        | {int(value.data.ptr) for value in vertical.arrays.values()},
    )
    if {int(value.data.ptr) for value in a.values()}.intersection(
        int(value.data.ptr) for value in vertical.arrays.values()
    ):
        raise ValueError("child scalar carriers alias stretched profiles")

    kpad, kpack, krecon, kflux, kinterp, kapply = compiled.kernels
    threads = (CUDA_GM1_SCALAR_TURBULENCE_THREADS, 1, 1)
    u32, f32 = cupy.uint32, cupy.float32
    hxy = (nj + 2) * (ni + 2)
    largest = max((nk + 1) * hxy, nk * nj * (ni + 1), nk * (nj + 1) * ni)
    grid = (
        (largest + CUDA_GM1_SCALAR_TURBULENCE_THREADS - 1)
        // CUDA_GM1_SCALAR_TURBULENCE_THREADS,
        1,
        1,
    )
    column_grid = (
        (ni * nj + CUDA_GM1_SCALAR_TURBULENCE_THREADS - 1)
        // CUDA_GM1_SCALAR_TURBULENCE_THREADS,
        1,
        1,
    )
    dz_reference = float(vertical.dz_reference_f32)
    with stream:
        kpad(
            grid,
            threads,
            (
                a["rho"],
                a["khh"],
                a["gz"],
                a["rgz"],
                workspace.rho_halo,
                workspace.khh_halo,
                workspace.gz_halo,
                workspace.rgz_halo,
                u32(ni),
                u32(nj),
                u32(nk),
            ),
            stream=stream,
        )
        order = ((0xFFFF_FFFF, a["thten1"]),) + tuple(
            (index, a["qten"][index]) for index in range(numq)
        )
        for species, target in order:
            kpack(
                grid,
                threads,
                (
                    a["th0"],
                    a["tha"],
                    a["qa"],
                    workspace.scalar_halo,
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    u32(species),
                ),
                stream=stream,
            )
            krecon(
                grid,
                threads,
                (
                    workspace.scalar_halo,
                    vertical.sigma,
                    vertical.sigmaf,
                    vertical.rds,
                    workspace.scalar_w,
                    u32(ni),
                    u32(nj),
                    u32(nk),
                ),
                stream=stream,
            )
            kflux(
                grid,
                threads,
                (
                    workspace.scalar_halo,
                    workspace.scalar_w,
                    workspace.rho_halo,
                    workspace.khh_halo,
                    workspace.gz_halo,
                    workspace.rgz_halo,
                    a["rgzu"],
                    a["rgzv"],
                    a["gxu"],
                    a["gyv"],
                    vertical.rdsf,
                    workspace.flux_x,
                    workspace.flux_y,
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    f32(dx),
                    f32(dy),
                ),
                stream=stream,
            )
            kinterp(
                grid,
                threads,
                (
                    workspace.flux_x,
                    workspace.flux_y,
                    vertical.sigma,
                    vertical.sigmaf,
                    vertical.rds,
                    workspace.flux_x_w,
                    workspace.flux_y_w,
                    u32(ni),
                    u32(nj),
                    u32(nk),
                ),
                stream=stream,
            )
            kapply(
                column_grid,
                threads,
                (
                    workspace.scalar_halo,
                    a["rho"],
                    a["rr"],
                    a["rf"],
                    a["mh"],
                    a["mf"],
                    a["khv"],
                    a["gz"],
                    a["rgzu"],
                    a["rgzv"],
                    a["gx"],
                    a["gy"],
                    vertical.rdsf,
                    workspace.flux_x,
                    workspace.flux_y,
                    workspace.flux_x_w,
                    workspace.flux_y_w,
                    # src/turbtend.F:588-596.  This sealed launcher's
                    # selection has surface_flux false, so the lower flux is
                    # the source's zero and the pointer is never read; gz is
                    # the (nj,ni) array every caller already supplies.
                    a["gz"],
                    workspace.thomas_c,
                    workspace.thomas_d,
                    target,
                    u32(ni),
                    u32(nj),
                    u32(nk),
                    f32(dx),
                    f32(dy),
                    f32(dz_reference),
                    f32(dt),
                    u32(0),
                ),
                stream=stream,
            )
        event = cupy.cuda.Event(disable_timing=True)
        event.record(stream)
    receipt = {
        "schema": CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SCHEMA,
        "claim_limit": _CLAIM_LIMIT,
        "authority": {
            "model": "GM1",
            "math_release": "CM1 r21.1",
            "tarball_sha256": AUTHORITY_TARBALL_SHA256,
            "source_sha256": {
                "src/solve1.F": CM1_SOLVE1_F_SHA256,
                "src/turbtend.F": CM1_TURBTEND_F_SHA256,
            },
            "anchors": {
                "coefficients_and_call_order": "solve1.F:1169-1227",
                "terrain_interpolation_and_flux": "turbtend.F:157-308",
                "implicit_vertical": "turbtend.F:442-692",
            },
        },
        "domain": {
            "ni": ni,
            "nj": nj,
            "nk": nk,
            "dx_m": dx,
            "dy_m": dy,
            "dt_s": dt,
            "reference_dz_m": dz_reference,
            "compact_validation": allow_compact_validation,
            "vertical_disposition": (
                "source-audited-flat-idealized-uniform-200m"
                if allow_uniform_idealized_validation
                else "sealed-stretched-child-or-compact-operator"
            ),
        },
        "vertical": dict(vertical.identity),
        "vertical_semantics": {
            "scalar_and_flux_w_interpolation": "(sigmaf(k)-sigma(k-1))*rds(k)",
            "terrain_cross_derivatives": "rdsf(k)",
            "implicit_thomas": "reference rdz squared with mf/mh metric factors",
            "uniform_scalar_dz_fallback": False,
        },
        "selection": {
            name: getattr(selection, name) for name in selection.__dataclass_fields__
        },
        "source_call_order": ["theta", *scalar_names],
        "cadence": "once per large step in solve1; held across all three RK stages",
        "kernel": {
            "abi": CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_ABI,
            "source_sha256": CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256,
            "legacy_source_sha256": CUDA_GM1_SCALAR_TURBULENCE_SOURCE_SHA256,
            "ptx_sha256": hashlib.sha256(compiled.ptx).hexdigest(),
            "cubin_sha256": hashlib.sha256(compiled.cubin).hexdigest(),
            "target_arch": compiled.target_arch,
        },
        "launch": {
            # One static pad, then pack/recon/flux/interp/apply per scalar.
            "kernel_launch_count": 1 + 5 * (1 + numq),
            "stream_pointer": int(stream.ptr),
            "event_pointer": int(event.ptr),
        },
        "runtime": dict(compiled.runtime_identity),
    }
    json.dumps(receipt, allow_nan=False, sort_keys=True)
    return CudaGm1ScalarTurbulenceLaunch(event, compiled, workspace, receipt)


__all__ = (
    "CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_ABI",
    "CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_KERNEL_NAMES",
    "CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SCHEMA",
    "CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE",
    "CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_BYTES",
    "CUDA_GM1_CHILD_STRETCHED_SCALAR_TURBULENCE_SOURCE_SHA256",
    "child_stretched_scalar_turbulence_shapes",
    "compile_cuda_gm1_child_stretched_scalar_turbulence",
    "launch_cuda_gm1_child_stretched_scalar_turbulence",
)
