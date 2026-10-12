"""Read-only admission of the bounded ArWen estate.

Nothing in this module imports :mod:`gpuwm` or CuPy.  A broken or drifting
engine must remain diagnosable without opening a CUDA context.
"""

from __future__ import annotations

import hashlib
from typing import Any
import importlib.util
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


GPUWM_REQUIREMENT = "gpuwm>=2.8.9,<2.8.10"
GPUWM_MEASURED_VERSION = "2.8.9"
#: ArWen's lookup-table and static-data distribution.  gpuwm 2.8.9 itself
#: requires ``gpuwm-data==2.8.9``; GM1 states the same window so the pair is
#: visible in its own metadata.  No file of it is byte-pinned here.
GPUWM_DATA_REQUIREMENT = "gpuwm-data>=2.8.9,<2.8.10"

# 2.5.8 was admitted from a tracked gpuwm checkout, so its source identity was a
# git commit.  2.7.4, 2.8.0 and 2.8.9 are admitted from their published
# distribution artifacts: a wheel carries no upstream commit in its metadata, and recording
# one anyway would be a fabricated provenance claim.  The identity below is the
# thing that was actually hashed -- the wheel this estate was extracted from.
GPUWM_AUDITED_DISTRIBUTION = "gpuwm-2.8.9-py3-none-manylinux_2_28_x86_64.whl"
GPUWM_AUDITED_DISTRIBUTION_SHA256 = (
    "0ece5eca8d08d693012027f812d541b42e097f396264c63c587ed73bf5250d50"
)
GPUWM_AUDITED_SOURCE_IDENTITY = (
    f"pypi:gpuwm=={GPUWM_MEASURED_VERSION} "
    f"sha256:{GPUWM_AUDITED_DISTRIBUTION_SHA256}"
)

#: The 2.8.0 estate this seam was pinned at before the 2.8.9 move (GM1
#: 0.1.0), as the published wheel it was extracted from.  It is the base of
#: the current estate-delta audit; nothing admits against it.
GPUWM_SUPERSEDED_2_8_0_DISTRIBUTION_SHA256 = (
    "6a8749c8acc8050de25857fb698ed5070fafeac49a002e127dfc664b14ff3577"
)
GPUWM_SUPERSEDED_2_8_0_SOURCE_IDENTITY = (
    f"pypi:gpuwm==2.8.0 sha256:{GPUWM_SUPERSEDED_2_8_0_DISTRIBUTION_SHA256}"
)

#: The 2.7.4 estate this seam was pinned at before the 2.8.0 move, as the
#: published wheel it was extracted from.  It is the base of the 2.8.0
#: estate-delta audit, kept as history; nothing admits against it.
GPUWM_SUPERSEDED_2_7_4_DISTRIBUTION_SHA256 = (
    "defe2d64ad97a95f3578c1d5145c4b36ed68f08b00c77218d644cc1a824f3afd"
)
GPUWM_SUPERSEDED_2_7_4_SOURCE_IDENTITY = (
    f"pypi:gpuwm==2.7.4 sha256:{GPUWM_SUPERSEDED_2_7_4_DISTRIBUTION_SHA256}"
)

#: The 2.5.8 source identity this seam was pinned at before the 2.7.4 move.
#: Kept so a historical receipt that names a commit stays readable next to the
#: record that superseded it; nothing admits against it.
GPUWM_SUPERSEDED_2_5_8_COMMIT = "f47f177d4b2e0da95568d70230f5ce9785e632c7"

# This is the complete byte-admitted surface reached by the CM1-owned adapter.
#
# Measured on the published gpuwm 2.8.9 wheel.  Against 2.8.0, six of the
# seventeen files 2.8.0 pinned are byte-identical (constants.py, common.cuh,
# morrison_constants.py, wsm6.py, wsm6_constants.py and
# kernel_compile_notice.py) and eleven changed.  Two of the eleven are kernel
# sources, so this is the first move since 2.5.8 in which the WSM6 kernel
# itself changed:
#
# - wsm6.cu: seven divisions by a constant are now written __fdiv_rn(a, b)
#   instead of a / b, plus a host shim for the same name.  __fdiv_rn is the
#   IEEE round-to-nearest quotient.  ArWen's notes say the device compiler
#   turned a constant division into a multiplication by the reciprocal on
#   RTX 50-series cards only, so those cards get different WSM6 words (up to
#   one unit in the last place at each site) and older cards should get the
#   same ones.  The estate-delta audit proves the text claim: writing every
#   __fdiv_rn(a, b) back as a / b reproduces the 2.8.0 kernel token for token.
#   Whether an RTX 30- or 40-series card gives the same words is a GPU
#   measurement, not something this file can state.
# - refl.cu: the same spelling change at one site, in the Thompson snow
#   reflectivity line.  The WSM6 and Morrison reflectivity kernels are
#   unchanged text.
#
# The other nine (config.py, core/kernels/__init__.py, core/microphysics.py,
# core/physics.py, core/refl.py, core/state.py, physics_registry.py,
# physics_registry_v2.json and physics_vertical_contract.py) are Python and
# registry changes; what each does to the WSM6 call is recorded per check in
# the estate-delta audit for 2.8.9 (tools/audit_arwen_wsm6_estate_delta.py)
# and in records/CM1-ARWEN-SEAM.md.  One of them reaches a WSM6 answer:
# core/microphysics.py now clamps the heating increment at the float32
# product float32(mp_tend_lim) * float32(dt) instead of the rounded double
# product, which can move a clamped increment by one float32 unit.  GM1
# passes the default limit of 10 K/s.
#
# Four files are new to the surface, because 2.8.9 runs every WSM6 or
# Morrison call through them: core/device_cache.py (the per-device cache that
# now holds every compiled module), physics_params.py (every kernel source
# passes through its edit_kernel_source before NVRTC; it returns the text
# unchanged unless a parameter set was declared, and GM1 declares none),
# wrf_exact.py (read while the source is assembled) and
# microphysics_schemes.py (the dispatch consults it).
AUDITED_WSM6_SURFACE: dict[str, str] = {
    "gpuwm/config.py": "efc959280c505ae7532827c5f7af6f6e39bbd5a7682d7b115a5583cb5f306137",
    "gpuwm/core/constants.py": "76c8c11d1a75d17fd54d0a80646cec9dde04f649247ec2347d3d32e6c0500e9e",
    "gpuwm/core/device_cache.py": "60edf18e58ea7d37d82312894c0950667431e9554cdf6fba4ccf5024c95fd729",
    "gpuwm/core/kernels/__init__.py": "e44a76e72880e6cbacee4a3cf299ab7e1a0b418e737355b3b64bfd60358b67cc",
    "gpuwm/core/kernels/common.cuh": "c78b17cb02ef67a2ad24d19e06e1129d7d5bcda74b972b38470fd33a6e58ff43",
    "gpuwm/core/kernels/refl.cu": "1e2c45699a721b4b042769f9f868d85d8a6eb4960fc547ce9affb59b69f43f66",
    "gpuwm/core/kernels/wsm6.cu": "0028b0c2d88d6087095cd2d4afab27bb2cb76b2a5bf51e3e5e908e0fdf2f0886",
    "gpuwm/core/microphysics.py": "35af3fcfc39bbd710d1c9ced44c4e2085f896402440087ed653245d493fcd90b",
    "gpuwm/core/morrison_constants.py": "100976309f219fe2466bb35a45ed6f68df70e0dbfb903375c22158cfa7a8044e",
    "gpuwm/core/physics.py": "78a4ab38d37814f1a37247ea3a495351ed759ce19cc335febe5b118cefa6ceb7",
    "gpuwm/core/refl.py": "d153a490c7f224a8b743b823857c5696cbb0c88700eb28abe73c7a3b7aac8de9",
    "gpuwm/core/state.py": "b4ec3e33ecc89f920dd6638e8c7dfecb2b27b42c68796da60c7522f0a2733296",
    "gpuwm/core/wsm6.py": "3ba72f8e6f3e4453d7c864e0556804a89da2f161f238043ab7670191e5d6af3d",
    "gpuwm/core/wsm6_constants.py": "4d110ed7f5a792f48d925fd0ff8e974305d870a11ebb31437ff4e6fce6d3c62d",
    "gpuwm/kernel_compile_notice.py": "3612648bdf4a1263e950ba2a2b671685e9d76f9b15a6675884c462682d5c99ae",
    "gpuwm/microphysics_schemes.py": "a9c667667eed75256e8394ed0abb82cc9d4cfc80ebb98d7b6e4f162f241bd433",
    "gpuwm/physics_params.py": "205833c7e3eccb75d2382cae1caf6c16c91d7ef1d9a26def6a7b86b522876f41",
    "gpuwm/physics_registry.py": "db15aa77fceeb830b32d9d1fe8e265ab41fc947d234beb6c41ef548e3e392c4f",
    "gpuwm/physics_registry_v2.json": "b7b770b44d3ec7f1f52a9812da5cd8022fb7cfcdf1f8e8c102523e486b70ffbf",
    "gpuwm/physics_vertical_contract.py": "c5cfb5210ddd6efc2ca55b6214d56e70b5e3e28427f5bd07e9ce777289f18ff1",
    "gpuwm/wrf_exact.py": "46a2451b5ac0408d8c1fb63ea801debd99c05a66bf110cb509f372020b1c3c57",
}

#: SHA-256 of the exact string ``gpuwm.core.kernels.module_source("wsm6")``
#: returns, which is what ``gpuwm.certify.kernel_manifest.record_module``
#: digests.  This is the string a shallow column really compiles:
#: ``gpuwm.core.wsm6.launch_wsm6`` takes the plain ``get_kernel`` route whenever
#: the selected capacity is the shallow tier, so no ``WSM6_KMAX`` define is
#: inserted at all and ``wsm6.cu``'s own ``#ifndef`` fallback of 64 applies.
#: Measured on 2.8.9.  It differs from the graded value below, for the first
#: time since the seam was built.
AUDITED_WSM6_NVRTC_SOURCE_SHA256 = (
    "d6798c079bbf2c6343d6d6575be2cd5fd67a128612c401b36e8ebfc3f1c10451"
)

#: The assembled WSM6 source the WRF v4.6.1 adapter oracle graded
#: (evidence/validation/arwen-wsm6-adapter-oracle-v1.json), which 2.5.8,
#: 2.7.4 and 2.8.0 all compiled.  2.8.9 compiles a string that differs from
#: it only in how seven constant divisions are spelled (see above), so the
#: grading describes the 2.8.9 kernel exactly wherever the compiler already
#: divided, and to within one unit in the last place at those seven sites
#: wherever it multiplied by a reciprocal.  The oracle has not been re-run
#: on 2.8.9.
WSM6_ORACLE_GRADED_NVRTC_SOURCE_SHA256 = (
    "1a6d20da0d450f235227fe609bdb12b368d96aec5ac231752074ff4dd9cc50e6"
)

# The two further files the public Morrison (CM1 ptype=5, ArWen mp_physics=10)
# dispatch reaches, at 2.8.9.  `gpuwm/core/morrison_constants.py` is already in
# the WSM6 surface above because `gpuwm/core/microphysics.py` imports it at
# module scope; these two are the lazily imported launcher and its CUDA
# translation unit.  They are a SEPARATE mapping, not an addition to the WSM6
# one, so a WSM6 run's audited surface, its SHA-256 and every receipt that
# carries it stay exactly the bytes they were before Morrison existed.
#
# Unlike the seven WSM6 numerical files, `morrison.cu` MOVED across the
# 2.5.8 -> 2.7.4 bump, and it moved numerically: the finalize kernel gained the
# `qv` write-back it was missing, PGAM's reference density is now rebuilt from
# the current temperature instead of being scaled from the frozen wrapper
# density, and cold cloud/rain donor limiting gained a wide-range arm.
# `gpuwm/core/morrison.py` is byte-identical across the two versions.  Morrison
# had never been bound in this port before 2.7.4, so there was no GM1 answer
# for that to move; the delta is named in records/CM1-ARWEN-SEAM.md rather
# than rounded to "unchanged".  Both files, and the assembled Morrison NVRTC
# source, are byte-identical between 2.7.4 and 2.8.0.
#
# Both files MOVED again between 2.8.0 and 2.8.9, and the kernel moved in ways
# a text comparison cannot price:
# - morrison.cu: twenty more divisions, most of them by a constant, go through __fdiv_rn (the
#   RTX 50-series change described for wsm6.cu above); sedimentation runs one
#   warp per hydrometeor category, with the five categories agreeing their
#   substep count through shared memory, and one fall-speed pass instead of
#   two; the finalize kernel leaves early for a cell that holds vapour and no
#   hydrometeor.
# - morrison.py: theta, the Exner function and the layer depths come from one
#   fused kernel in place of four array expressions, and the interface-sized
#   scratch volume is gone; the sedimentation block is 160 threads.
# ArWen states the restructured kernel gives identical output.  GM1 has not
# measured that: every Morrison comparison GM1 made against CM1 was made on
# 2.7.4 or 2.8.0 bytes and has to be repeated on a GPU before it is claimed
# for 2.8.9.
AUDITED_MORRISON_EXTRA_SURFACE: dict[str, str] = {
    "gpuwm/core/kernels/morrison.cu": "9d4aad5a012cf2751f3515d96dae48adc1765f65aaa934486e0a025bf8dca229",
    "gpuwm/core/morrison.py": "0902cff1b045ee0101effbf586498be21b31a0c7b965b56cdcbe593bfbf4a8ca",
}

# Classic Thompson (CM1 ptype=3, ArWen mp_physics=8).  The adapter body lives
# in the already-audited gpuwm/core/microphysics.py; these are the lazily
# imported launcher, its contract and runtime table loader, and the CUDA
# translation unit.  The reflectivity path is gpuwm/core/refl.py's own
# refl.cu, which the WSM6 surface already carries.  Re-measured on 2.8.9.
# 2.8.0 changed thompson.cu and thompson.py numerically (WRF's entry zeroing
# at R1, the column-exit mask, L_qc/L_qr presence handling), and 2.8.9 changes
# them again on every card: rain collecting graupel reads the table WRF builds
# (16 of ArWen's 42 fixture columns change), the adapter's entry, mask and
# finish steps are fused kernels, and the heating clamp is a float32 product.
# This surface admits bytes no GM1 run has measured on a GPU.
AUDITED_THOMPSON_EXTRA_SURFACE: dict[str, str] = {
    "gpuwm/core/kernels/thompson.cu": "43f76b17febd32acdfe96e9cca4d9c932ce2d72c5a5378ed73b5f3eac33998d8",
    "gpuwm/core/thompson.py": "00f661f03a86e5d10b4c27db93264a3dcef287b20cb25f0449304a95d22c2f16",
    "gpuwm/core/thompson_contract.py": "1f4655e688fd8f7a8d6291ca7e1dd041cc11ae4235ae2f8bee25733d53a211e2",
    "gpuwm/core/thompson_runtime.py": "d5c4513b43142985aac040d22cd4e2da39253337634d6a1594e2e6f41d118d10",
}

# WDM6 (ArWen mp_physics=16; CM1 r21.1 has no WDM6 ptype).  No lane selects
# this component today -- ArWen's WDM6 refuses without WRF's XLAND mask, which
# cm1core.column_component_blockers records -- and the surface is kept because
# it is measured against the deployed estate and is what a future binding needs.  wdm6_constants.py
# is listed here rather than in the WSM6 surface even though WSM6's own
# gpuwm/physics_vertical_contract.py imports it: adding it there would change
# the WSM6 audited-surface SHA-256 that already travels inside recorded
# receipts.  kernels/wdm6_refl.cu is WDM6's own reflectivity translation unit
# (gpuwm/core/refl.py launches refl10cm_wdm6_column from module
# "wdm6_refl", not from refl.cu).  Re-measured on 2.8.9: wdm6.cu changed
# (rain evaporation can no longer drive vapour negative, and constant
# divisions go through __fdiv_rn); the other three files are byte-identical
# to 2.8.0.
AUDITED_WDM6_EXTRA_SURFACE: dict[str, str] = {
    "gpuwm/core/kernels/wdm6.cu": "e1c1a666004e43f2471b69469b6938ec188d43e542330c741ed546c3a8d771aa",
    "gpuwm/core/kernels/wdm6_refl.cu": "5dff160d671d68c2236c964bfac94e0b8f275a6897b8840f860c0e1ddbf9fdcf",
    "gpuwm/core/wdm6.py": "128a1fa3906894e4c87681fc0e64e67565592e3a3d842f7bc0323b382a7a2000",
    "gpuwm/core/wdm6_constants.py": "e8c55cdd7eba4e8123f685ae92c0dbe1156ca4448b4630aab14d147a2776a2ea",
}

# Aerosol-aware Thompson (ArWen mp_physics=28; CM1 r21.1 has no aerosol-aware
# Thompson ptype).  It is classic Thompson's surface plus its own sibling
# adapter, its launcher and the six aerosol translation units that share one
# device header.  Re-measured on 2.8.9: all nine files changed, and the six
# translation units now also compile with three further device headers
# (glibc_flt32.cuh, glibc_flt64.cuh and thompson_aerosol_libm.cuh), which join
# the surface here.
AUDITED_THOMPSON_AEROSOL_EXTRA_SURFACE: dict[str, str] = {
    **AUDITED_THOMPSON_EXTRA_SURFACE,
    "gpuwm/core/microphysics_aerosol.py": "5e123908bba267c14bba5ef94212608d8751759faf13635b227a720a70c29c92",
    "gpuwm/core/thompson_aerosol.py": "dc4d8c44a1dfd2a84b8d18f6369caeb173ab0266474579723be65ef317532c23",
    "gpuwm/core/kernels/thompson_aerosol_common.cuh": "a24a96dab64e0688106dbdda5fe518ea9390d826c1f32169126a4d3893b38e9d",
    "gpuwm/core/kernels/thompson_aerosol_libm.cuh": "c22ea2f8b096e402685df8acf31c775a6f134af26be0646d9380e93e2df10ac0",
    "gpuwm/core/kernels/glibc_flt32.cuh": "b647335d355aefb5d9ece1726ff23667a01f09c8b263c136bd8a94764e904d4f",
    "gpuwm/core/kernels/glibc_flt64.cuh": "a14aff39d4acc74d9c782726d17d0bf7ef726c40235ddeb86d6a1a0bd7c4fcb8",
    "gpuwm/core/kernels/thompson_aerosol_probe.cu": "a4db6f915e5393c2a809f40c0b793ccba5e9bba51a672f7d5ea99a04c1a18274",
    "gpuwm/core/kernels/thompson_aerosol_state.cu": "c16b1552e1e0a0980aa83e9a3c197e25b71350521ce71de6badb136dd031e3e9",
    "gpuwm/core/kernels/thompson_aerosol_sat.cu": "88ee26846dde74b282b286802d0cd838f3332048c74b295cc3723109b2847a30",
    "gpuwm/core/kernels/thompson_aerosol_cold.cu": "b8f5496b68a29ee0de0a93203a8619175f428040c634bcd099e2e21bab1e49e7",
    "gpuwm/core/kernels/thompson_aerosol_warm.cu": "3e6972a7e5c41238c6ba664a8032851c187d007a4bf6366b17020b6dbed9635a",
    "gpuwm/core/kernels/thompson_aerosol_sed.cu": "bcd106943cec93f8faa8b6281b1bb8d3d9e90888b9b3a4b4e9dc44466bf58019",
}

# Component id -> the exact extra files that component's dispatch reaches.
# An id absent here has no admitted public surface at all.
COMPONENT_EXTRA_SURFACE: dict[str, dict[str, str]] = {
    "wsm6-mp6": {},
    "morrison-mp10": AUDITED_MORRISON_EXTRA_SURFACE,
    "thompson-mp8": AUDITED_THOMPSON_EXTRA_SURFACE,
    "thompson-aerosol-mp28": AUDITED_THOMPSON_AEROSOL_EXTRA_SURFACE,
    "wdm6-mp16": AUDITED_WDM6_EXTRA_SURFACE,
}


@dataclass(frozen=True, slots=True)
class EngineInspection:
    version: str | None
    root: Path | None
    matched: tuple[str, ...]
    moved: tuple[str, ...]
    absent: tuple[str, ...]
    capacity_profile: str | None = None
    component_id: str | None = None

    @property
    def admitted(self) -> bool:
        return (
            self.version == GPUWM_MEASURED_VERSION
            and self.root is not None
            and not self.moved
            and not self.absent
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "requirement": GPUWM_REQUIREMENT,
            "measured_version": GPUWM_MEASURED_VERSION,
            "installed_version": self.version,
            "root": str(self.root) if self.root is not None else None,
            "matched": list(self.matched),
            "moved": list(self.moved),
            "absent": list(self.absent),
            "admitted_audited_surface": self.admitted,
            "cm1_public_seam_available": True,
        }
        # Absent until a caller names a component, so a WSM6 inspection
        # serializes to exactly the object it did before Morrison existed.
        if self.component_id is not None:
            result["component_id"] = self.component_id
            result["component_extra_surface"] = dict(
                sorted(component_extra_surface(self.component_id).items())
            )
        if self.capacity_profile is not None:
            from .engine_capacity import capacity_profile_metadata, capacity_profile_sha256

            result["capacity_profile"] = capacity_profile_metadata(self.capacity_profile)
            result["capacity_profile_sha256"] = capacity_profile_sha256(self.capacity_profile)
        return result


def installed_version() -> str | None:
    try:
        return version("gpuwm")
    except PackageNotFoundError:
        return None


def installed_root() -> Path | None:
    try:
        spec = importlib.util.find_spec("gpuwm")
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent.parent


_INSPECTION_CACHE: dict[tuple[Any, ...], EngineInspection] = {}


def component_extra_surface(component_id: str | None) -> dict[str, str]:
    """The extra audited files one semantic component's dispatch reaches.

    ``None`` keeps the historical WSM6-only answer, so every caller that does
    not name a component gets exactly the surface it got before this argument
    existed.
    """

    if component_id is None:
        return {}
    try:
        return dict(COMPONENT_EXTRA_SURFACE[component_id])
    except KeyError:
        raise ValueError(
            f"component {component_id!r} has no admitted ArWen public surface; "
            "running it would execute bytes this port has never audited"
        ) from None


def audited_surface(
    capacity_profile: str | None = None, component_id: str | None = None
) -> dict[str, str]:
    """Select an explicit capacity build while leaving the default pins intact."""
    if capacity_profile is None:
        surface = dict(AUDITED_WSM6_SURFACE)
    else:
        from .engine_capacity import capacity_surface

        surface = capacity_surface(capacity_profile)
    surface.update(component_extra_surface(component_id))
    return surface


def inspect(
    *, capacity_profile: str | None = None, component_id: str | None = None
) -> EngineInspection:
    """Verify the installed ArWen bytes against the audited surface.

    The result is memoized per profile and per (path, size, mtime_ns) stamp of
    every audited file, because the tiled WSM6 launcher calls this once per
    horizontal tile.  Uncached that is 0.71 ms x 16 tiles per step, which is
    2.7 minutes of rehashing across the 14,400 steps of a two-hour run.

    The stamp is part of the key, so a file that changes underneath a running
    process is re-read and re-hashed rather than served from the cache: this
    memoizes the answer, it does not stop asking the question.
    """

    stamp = _surface_stamp(capacity_profile, component_id)
    cached = _INSPECTION_CACHE.get((capacity_profile, component_id, stamp))
    if cached is not None:
        return cached
    result = _inspect_uncached(
        capacity_profile=capacity_profile, component_id=component_id
    )
    _INSPECTION_CACHE[(capacity_profile, component_id, stamp)] = result
    return result


def _surface_stamp(
    capacity_profile: str | None, component_id: str | None = None
) -> tuple[Any, ...]:
    """Cheap identity of every audited file: path, size and mtime."""

    root = installed_root()
    if root is None:
        return (None,)
    stamp: list[Any] = [str(root)]
    for relative in sorted(audited_surface(capacity_profile, component_id)):
        try:
            info = (root / relative).stat()
        except OSError:
            stamp.append((relative, None))
        else:
            stamp.append((relative, info.st_size, info.st_mtime_ns))
    return tuple(stamp)


def _inspect_uncached(
    *, capacity_profile: str | None = None, component_id: str | None = None
) -> EngineInspection:
    surface = audited_surface(capacity_profile, component_id)
    root = installed_root()
    if root is None:
        return EngineInspection(
            installed_version(), None, (), (), tuple(surface), capacity_profile,
            component_id,
        )
    matched: list[str] = []
    moved: list[str] = []
    absent: list[str] = []
    for relative, expected in surface.items():
        path = root / relative
        try:
            payload = path.read_bytes()
        except OSError:
            absent.append(relative)
            continue
        if hashlib.sha256(payload).hexdigest() == expected:
            matched.append(relative)
        else:
            moved.append(relative)
    return EngineInspection(
        installed_version(),
        root,
        tuple(sorted(matched)),
        tuple(sorted(moved)),
        tuple(sorted(absent)),
        capacity_profile,
        component_id,
    )


__all__ = [
    "AUDITED_MORRISON_EXTRA_SURFACE",
    "AUDITED_THOMPSON_AEROSOL_EXTRA_SURFACE",
    "AUDITED_THOMPSON_EXTRA_SURFACE",
    "AUDITED_WDM6_EXTRA_SURFACE",
    "AUDITED_WSM6_SURFACE",
    "COMPONENT_EXTRA_SURFACE",
    "EngineInspection",
    "GPUWM_AUDITED_DISTRIBUTION",
    "GPUWM_AUDITED_DISTRIBUTION_SHA256",
    "GPUWM_AUDITED_SOURCE_IDENTITY",
    "GPUWM_DATA_REQUIREMENT",
    "component_extra_surface",
    "GPUWM_MEASURED_VERSION",
    "GPUWM_REQUIREMENT",
    "GPUWM_SUPERSEDED_2_5_8_COMMIT",
    "GPUWM_SUPERSEDED_2_7_4_DISTRIBUTION_SHA256",
    "GPUWM_SUPERSEDED_2_8_0_DISTRIBUTION_SHA256",
    "GPUWM_SUPERSEDED_2_8_0_SOURCE_IDENTITY",
    "WSM6_ORACLE_GRADED_NVRTC_SOURCE_SHA256",
    "GPUWM_SUPERSEDED_2_7_4_SOURCE_IDENTITY",
    "audited_surface",
    "inspect",
    "installed_root",
    "installed_version",
]
