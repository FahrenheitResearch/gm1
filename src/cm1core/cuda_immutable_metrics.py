"""Source-proven sharing of immutable grid metrics, never evolving state."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import hashlib
from typing import Any

from .gm1_idealized_residency import DeviceBufferRegistry, IdealizedSource, SourceRequest


_PROVEN = object()


class ImmutableMetricMap(Mapping[str, Any]):
    """A CUDA mapping accompanied by host-source word/layout proofs."""

    def __init__(self, values: Mapping[str, Any], proofs: Mapping[str, Any], token: object) -> None:
        if token is not _PROVEN:
            raise TypeError("immutable metric maps require source validation")
        self._values = dict(values)
        self.proofs = dict(proofs)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def with_derived(self, name: str, value: Any, proof: Mapping[str, Any]) -> "ImmutableMetricMap":
        """One device metric this runtime derived from an already proven one.

        A derived word is not a source carrier and never claims to be: its
        proof records what it was built from and by which rule, so a receipt
        that reads the map can tell the two apart.  A name the source already
        carries is refused, because a derived value silently shadowing a
        source metric is exactly the substitution these proofs exist to stop.
        """

        if name in self._values:
            raise ValueError(f"{name} is already a source metric and cannot be derived over")
        proof = dict(proof)
        if proof.get("kind", "").startswith(("constant-source-words", "separate-source-array", "horizontal-profile")):
            raise ValueError("a derived metric may not claim a source-carrier proof kind")
        values, proofs = dict(self._values), dict(self.proofs)
        values[name], proofs[name] = value, proof
        return ImmutableMetricMap(values, proofs, _PROVEN)

    def allows_alias(self, left: str, left_value: Any, right: str, right_value: Any) -> bool:
        if self._values.get(left) is not left_value or self._values.get(right) is not right_value:
            return False
        a, b = self.proofs[left], self.proofs[right]
        return a["kind"] == b["kind"] == "constant-source-words" and a["word_u32"] == b["word_u32"]


def immutable_metric_subset(original: Mapping[str, Any], checked: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(original, ImmutableMetricMap):
        if any(original[name] is not value for name, value in checked.items()):
            raise ValueError("validated immutable metric subset changed source objects")
        return ImmutableMetricMap(checked, {name: original.proofs[name] for name in checked}, _PROVEN)
    return dict(checked)


def allows_compact_coordinate(metrics: Mapping[str, Any], name: str, value: Any) -> bool:
    return (
        name in ("zh", "zf") and isinstance(metrics, ImmutableMetricMap)
        and metrics[name] is value
        and metrics.proofs[name]["kind"] == "horizontal-profile"
    )


def require_cuda_argument_aliases(
    arguments: Mapping[str, Any], *, immutable_metrics: Mapping[str, Any] | None = None,
    message: str = "CUDA mutable/scratch arguments must not alias",
) -> None:
    """Allow equal-pointer metric reads only with source-validated proofs.

    Metric names are prefixed ``metric.`` by the numerical call site.  Every
    other argument retains the original no-alias rule, including state read
    arguments: this does not weaken mutable/output or scratch ownership.
    """
    seen: dict[int, tuple[str, Any]] = {}
    for name, value in arguments.items():
        pointer = int(value.data.ptr)
        if pointer in seen:
            other_name, other = seen[pointer]
            allowed = (
                isinstance(immutable_metrics, ImmutableMetricMap)
                and name.startswith("metric.") and other_name.startswith("metric.")
                and immutable_metrics.allows_alias(name[7:], value, other_name[7:], other)
            )
            if not allowed:
                raise ValueError(message)
        else:
            seen[pointer] = (name, value)


def extend_immutable_source_metrics(
    original: ImmutableMetricMap,
    source: IdealizedSource,
    registry: DeviceBufferRegistry,
    additions: Mapping[str, str],
) -> ImmutableMetricMap:
    """Add named carriers from another Rust group to an already proven map.

    ``additions`` maps the metric name a consumer asks for to the qualified
    ``group.carrier`` it comes from.  Each one is uploaded whole and keeps its
    own storage: it is recorded as a separate source array, never as a member
    of a constant-word sharing group, so it can never satisfy an alias proof
    and a consumer that finds it aliasing another metric still refuses.
    """

    import numpy as np

    if not isinstance(original, ImmutableMetricMap):
        raise TypeError("only a source-proven metric map can be extended")
    values = dict(original)
    proofs = dict(original.proofs)
    for name, qualified in additions.items():
        if name in values:
            raise ValueError(f"metric {name} is already proven in this map")
        group, _, carrier = qualified.partition(".")
        host = source.groups[group][carrier]
        if host.dtype != np.dtype("float32") or not host.flags.c_contiguous:
            raise ValueError(f"source metric {qualified} is not a contiguous binary32 carrier")
        if not host.size:
            raise ValueError(f"empty source metric {qualified} cannot be shared")
        registered = f"metric.{name}"
        values[name] = registry.upload(
            source, [SourceRequest(registered, qualified, "native", True)]
        )[registered]
        proofs[name] = {
            "source_sha256": hashlib.sha256(memoryview(host).cast("B")).hexdigest(),
            "shape": list(host.shape),
            "kind": "separate-source-array",
            "source_carrier": qualified,
        }
    return ImmutableMetricMap(values, proofs, _PROVEN)


def upload_immutable_source_metrics(source: IdealizedSource, registry: DeviceBufferRegistry) -> ImmutableMetricMap:
    """Share constant source words; compress only source-proven z coordinates.

    Each full source carrier is inspected in bounded chunks, including signed
    zero.  Sharing is restricted to this Rust terrain-static group.  There is
    no assumption that a metric equals one, zero, or another named carrier.
    """
    import numpy as np

    host = source.groups["terrain_static"]
    constant_groups: dict[int, list[str]] = {}
    variable: list[str] = []
    proofs: dict[str, Any] = {}
    for name, value in host.items():
        if value.dtype != np.dtype("float32") or not value.flags.c_contiguous:
            raise ValueError(f"source metric {name} is not a contiguous binary32 carrier")
        words = value.view(np.uint32).reshape(-1)
        if not words.size:
            raise ValueError("empty terrain metric cannot be shared")
        word = int(words[0])
        same = all(bool(np.all(words[start:start + 262_144] == word)) for start in range(0, words.size, 262_144))
        source_sha = hashlib.sha256(memoryview(value).cast("B")).hexdigest()
        proofs[name] = {"source_sha256": source_sha, "shape": list(value.shape), "kind": "separate-source-array"}
        if same:
            constant_groups.setdefault(word, []).append(name)
            proofs[name].update({"kind": "constant-source-words", "word_u32": word})
        else:
            variable.append(name)
    values: dict[str, Any] = {}
    for word, names in constant_groups.items():
        owner = max(names, key=lambda name: int(host[name].size))
        registered = f"metric.{owner}"
        values[owner] = registry.upload(source, [SourceRequest(registered, f"terrain_static.{owner}", "native", True)])[registered]
        for name in names:
            proofs[name]["storage_owner"] = owner
            if name != owner:
                values[name] = registry.alias_read_only_view(f"metric.{name}", registered, tuple(host[name].shape))
    for name in variable:
        if name in ("zh", "zf"):
            # field(profile) checks every horizontal plane before retaining a
            # compact host profile.  CUDA broadcasting changes storage only.
            source.field(f"terrain_static.{name}", layout="profile")
            registered = f"metric-profile.{name}"
            registry.upload(source, [SourceRequest(registered, f"terrain_static.{name}", "profile", True)])
            values[name] = registry.alias_read_only_broadcast(f"metric.{name}", registered, tuple(host[name].shape))
            proofs[name]["kind"] = "horizontal-profile"
        else:
            registered = f"metric.{name}"
            values[name] = registry.upload(source, [SourceRequest(registered, f"terrain_static.{name}", "native", True)])[registered]
    return ImmutableMetricMap({name: values[name] for name in host}, proofs, _PROVEN)
