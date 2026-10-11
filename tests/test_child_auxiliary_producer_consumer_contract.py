"""The Rust auxiliary producer and its only Python consumer agree by key.

`rust/cm1-state/src/child_auxiliary.rs` publishes the child auxiliary estate;
`cm1core.cuda_gm1_child_nm_def.load_gm1_child_auxiliary` is the only thing that
reads it, and the child composer and both real-child runners reach the estate
only through that loader.  When the producer stopped emitting one key of
`static_inputs` while the loader still required it, every estate the new
producer published raised ValueError before CUDA opened - the artifact the
change existed to enable could not be consumed at all.

This reads the key literals out of the producer's own source and compares them
to the exact sets the loader requires, so the two halves cannot drift apart
again without a test saying so.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "rust/cm1-state/src/child_auxiliary.rs"
CONSUMER = ROOT / "src/cm1core/cuda_gm1_child_nm_def.py"


def _rust_object_keys(source: str, block: str) -> set[str]:
    """The keys of one `("<block>", object(vec![...]))` literal."""

    start = source.index(f'"{block}",\n            object(vec![')
    cursor = source.index("object(vec![", start) + len("object(vec![")
    depth = 1
    end = cursor
    while depth:
        if source[end] == "[":
            depth += 1
        elif source[end] == "]":
            depth -= 1
            if depth == 0:
                break
        end += 1
    body = source[cursor:end]
    return set(re.findall(r'\(\s*\n?\s*"([a-z0-9_]+)",', body))


def _python_key_set(source: str, anchor: str) -> set[str]:
    """The literal set compared against in one `set(...) != { ... }` guard."""

    start = source.index(anchor)
    open_brace = source.index("{", start)
    close_brace = source.index("}", open_brace)
    return set(re.findall(r'"([a-z0-9_]+)"', source[open_brace:close_brace]))


def test_the_producer_and_the_loader_agree_on_static_inputs() -> None:
    consumer = CONSUMER.read_text(encoding="utf-8")
    produced = _rust_object_keys(PRODUCER.read_text(encoding="utf-8"), "static_inputs")
    required = _python_key_set(consumer, "    required = {")
    assert produced == required, f"produced={sorted(produced)} required={sorted(required)}"
    # The key the producer cannot honestly compute for a new child: the digest
    # of a CUDA terrain-metrics output.  A new estate must not carry it and the
    # loader must not require it.
    assert "combined_23_sha256" not in produced
    assert "combined_23_sha256" not in required
    # It is still *tolerated* on read, and only as an extra beside the required
    # set, because the estate this distribution ships was published before the
    # retirement and would otherwise be unloadable. That tolerance is one named
    # historical shape, not an open door: the loader admits the required set or
    # the required set plus exactly this key, and nothing else.
    assert 'required | {"combined_23_sha256"},' in consumer


def test_the_producer_and_the_loader_agree_on_the_source_atmosphere() -> None:
    produced = _rust_object_keys(
        PRODUCER.read_text(encoding="utf-8"), "source_atmosphere"
    )
    required = _python_key_set(
        CONSUMER.read_text(encoding="utf-8"),
        'if not isinstance(source_atmosphere, Mapping) or set(source_atmosphere) != {',
    )
    assert produced == required, f"produced={sorted(produced)} required={sorted(required)}"


def test_the_retired_combined_23_constant_is_gone_from_the_producer() -> None:
    """A dead constant beside its replacement is what a reader mistakes for the gate."""

    assert "STATIC_COMBINED_23_SHA256" not in PRODUCER.read_text(encoding="utf-8")
