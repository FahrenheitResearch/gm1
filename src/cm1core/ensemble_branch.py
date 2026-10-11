"""Reproducible ensemble members, branched from an initial state or a restart.

CM1's own ensemble perturbation is ``irandp`` (``README.namelist:692-695``,
``src/init3d.F:1446-1483``).  At ``testcase = 0`` it is exactly one operation:
with the amplitude fixed at 0.25 K in the source, every point of the domain
draws one ``random_number`` ``r`` in a fixed loop order and adds
``amplitude*(2r - 1)`` to the perturbation potential temperature -- a uniform
perturbation of plus or minus one amplitude, added to the perturbation carrier
over the whole domain, with the amplitude set in the source rather than in the
namelist.

This module is that operation with two deliberate differences, both named.

**The amplitude is configuration, not source.**  ``src/init3d.F:1466`` says
"set magnitude of perturbations in init3d.F", so CM1 users edit Fortran to
change it.  ``ensemble.theta_amplitude_k`` carries it, and CM1's own 0.25 K is
what the shipped example declares.

**The stream is reproducible.**  CM1 calls ``random_number``, whose sequence
depends on the compiler and is not seeded, so a CM1 ensemble member cannot be
re-run.  A member that cannot be re-run cannot be bisected, and the question an
ensemble exists to answer -- was this outcome inevitable, or did it turn on
this perturbation -- needs exactly that.  ``ensemble.seed`` drives splitmix64,
whose output is fully specified by five integer operations, so the same seed
gives the same field on any machine and any NumPy version, and the receipt
carries the SHA-256 of the field that was actually added.

The loop order is CM1's: ``k`` outermost, then ``j``, then ``i``, which is the
C order of this lane's ``(nz, ny, nx)`` mass carriers, so word *n* of the
stream lands on the cell CM1's *n*-th ``random_number`` call would have landed
on.

WHAT THIS DOES NOT DO
---------------------
It perturbs the potential-temperature carriers and nothing else.  CM1 perturbs
``tha`` at initialization and then rebuilds the pressure and density fields
from it (``src/init3d.F:2386-2467`` getset); a branch taken at a completed
large step cannot rebuild them, because they carry the trajectory rather than
the base state.  The perturbed state is therefore momentarily out of exact
thermodynamic balance and the first acoustic step adjusts it -- which is what
an unbalanced perturbation physically is, and is why the amplitude is small.
The receipt says so rather than implying a balanced perturbation.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Any, Mapping


ENSEMBLE_BRANCH_SCHEMA = "gm1.cartesian-ensemble-branch/v1"
ENSEMBLE_BRANCH_AUTHORITY = (
    "src/init3d.F:1446-1483 irandp=1, testcase=0; README.namelist:692-695"
)
#: ``src/init3d.F:1466``.  CM1's own amplitude, in kelvin.
CM1_IRANDP_AMPLITUDE_K = 0.25
#: The carriers CM1's ``irandp`` writes.  ``tha`` is the potential-temperature
#: perturbation from ``th0``; at the completed-large-step boundary this lane's
#: ``th3d`` holds the same words, and both are written so the A and current
#: carriers still agree when the next step reads them.
ENSEMBLE_PERTURBED_CARRIERS = ("field.tha", "field.th3d")
ENSEMBLE_BRANCH_CLAIM_LIMIT = (
    "one reproducible uniform potential-temperature perturbation of CM1's own "
    "irandp shape, added to the theta perturbation carriers before the first "
    "large step of this leg; the pressure and density carriers are not "
    "re-derived, so the state is momentarily out of exact thermodynamic "
    "balance and the first acoustic step adjusts it"
)

_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB


class EnsembleBranchError(RuntimeError):
    """A declared ensemble member could not be produced exactly."""


@dataclass(frozen=True, slots=True)
class EnsembleBranchReceipt:
    schema: str
    authority: str
    member: int
    seed: int
    theta_amplitude_k: float
    perturbed_carriers: tuple[str, ...]
    cells: int
    field_sha256: str
    minimum_k: float
    maximum_k: float
    mean_k: float
    branched_from_completed_steps: int
    claim_limit: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority,
            "branched_from_completed_steps": self.branched_from_completed_steps,
            "cells": self.cells,
            "claim_limit": self.claim_limit,
            "field_sha256": self.field_sha256,
            "maximum_k": self.maximum_k,
            "mean_k": self.mean_k,
            "member": self.member,
            "minimum_k": self.minimum_k,
            "perturbed_carriers": list(self.perturbed_carriers),
            "schema": self.schema,
            "seed": self.seed,
            "theta_amplitude_k": self.theta_amplitude_k,
        }


def splitmix64_words(seed: int, count: int) -> Any:
    """``count`` consecutive splitmix64 outputs for ``seed``, as uint64.

    Five integer operations per word, every one of them exact in 64-bit
    two's complement, so the sequence is a property of the seed alone: no
    library version, platform or device can change it.  The generator is
    stated here rather than imported so the stream a receipt names can be
    re-derived from this file.
    """

    numpy = importlib.import_module("numpy")
    if type(seed) is not int or seed < 0 or seed > _MASK:
        raise EnsembleBranchError("an ensemble seed is an unsigned 64-bit integer")
    if type(count) is not int or count < 1:
        raise EnsembleBranchError("an ensemble stream needs at least one word")
    index = numpy.arange(1, count + 1, dtype=numpy.uint64)
    state = numpy.uint64(seed) + index * numpy.uint64(_GOLDEN)
    z = state
    z = z ^ (z >> numpy.uint64(30))
    z = z * numpy.uint64(_MIX1)
    z = z ^ (z >> numpy.uint64(27))
    z = z * numpy.uint64(_MIX2)
    return z ^ (z >> numpy.uint64(31))


def theta_perturbation_field(
    *, seed: int, amplitude_k: float, shape: tuple[int, ...]
) -> Any:
    """CM1's ``amplitude*(2.0*rand-1.0)``, in CM1's own k, j, i order."""

    numpy = importlib.import_module("numpy")
    if not isinstance(amplitude_k, float) or not (amplitude_k > 0.0):
        raise EnsembleBranchError(
            "an ensemble theta amplitude is a positive number of kelvin; a zero "
            "amplitude would make every member the trajectory it branched from "
            "while the receipt named an ensemble"
        )
    cells = 1
    for extent in shape:
        cells *= int(extent)
    words = splitmix64_words(seed, cells)
    # The top 53 bits of a uint64 give a binary64 in [0, 1) exactly, which is
    # the closed form of what Fortran's random_number returns.
    unit = (words >> numpy.uint64(11)).astype(numpy.float64) * (2.0**-53)
    field = (float(amplitude_k) * (2.0 * unit - 1.0)).astype(numpy.float32)
    return numpy.ascontiguousarray(field.reshape(shape))


def apply_ensemble_branch(
    runtime: Any, *, member: int, seed: int, theta_amplitude_k: float
) -> EnsembleBranchReceipt:
    """Add one member's perturbation to this runtime, before it steps.

    The runtime must be standing at a completed-large-step boundary -- a
    freshly built one at step zero, or one just restored from a restart -- so
    the A and current theta carriers agree and both can be given the same
    words.  Perturbing anywhere else would leave the two RK carriers holding
    different model states.
    """

    numpy = importlib.import_module("numpy")
    dynamics = runtime.dynamics
    if dynamics._expected_phase != ("held", 1):
        raise EnsembleBranchError(
            f"the dynamics cadence is at {dynamics._expected_phase!r}; an "
            "ensemble perturbation is added at the completed-large-step "
            "boundary only, where the A and current theta carriers agree"
        )
    grid = runtime.config.grid
    shape = (int(grid.nz), int(grid.ny), int(grid.nx))
    registry = runtime.registry
    missing = [name for name in ENSEMBLE_PERTURBED_CARRIERS if name not in registry.buffers]
    if missing:
        raise EnsembleBranchError(
            f"this runtime has no {', '.join(missing)} carrier to perturb"
        )
    field = theta_perturbation_field(
        seed=seed, amplitude_k=theta_amplitude_k, shape=shape
    )
    digest = hashlib.sha256(field.tobytes(order="C")).hexdigest()
    targets = []
    for name in ENSEMBLE_PERTURBED_CARRIERS:
        target = registry.buffers[name]
        if tuple(int(extent) for extent in target.shape) != shape:
            raise EnsembleBranchError(
                f"{name} is {tuple(target.shape)} and the mass grid is {shape}"
            )
        targets.append(target)
    first = targets[0]
    if hasattr(first, "__cuda_array_interface__"):
        cupy = importlib.import_module("cupy")
        device_field = cupy.asarray(field)
        for target in targets:
            cupy.add(target, device_field, out=target)
        del device_field
    else:
        for target in targets:
            numpy.add(target, field, out=target)
    runtime.synchronize()
    return EnsembleBranchReceipt(
        schema=ENSEMBLE_BRANCH_SCHEMA,
        authority=ENSEMBLE_BRANCH_AUTHORITY,
        member=int(member),
        seed=int(seed),
        theta_amplitude_k=float(theta_amplitude_k),
        perturbed_carriers=ENSEMBLE_PERTURBED_CARRIERS,
        cells=int(field.size),
        field_sha256=digest,
        minimum_k=float(field.min()),
        maximum_k=float(field.max()),
        mean_k=float(field.astype(numpy.float64).mean()),
        branched_from_completed_steps=int(runtime.clock.completed_steps),
        claim_limit=ENSEMBLE_BRANCH_CLAIM_LIMIT,
    )


__all__ = [
    "CM1_IRANDP_AMPLITUDE_K",
    "ENSEMBLE_BRANCH_AUTHORITY",
    "ENSEMBLE_BRANCH_CLAIM_LIMIT",
    "ENSEMBLE_BRANCH_SCHEMA",
    "ENSEMBLE_PERTURBED_CARRIERS",
    "EnsembleBranchError",
    "EnsembleBranchReceipt",
    "apply_ensemble_branch",
    "splitmix64_words",
    "theta_perturbation_field",
]
