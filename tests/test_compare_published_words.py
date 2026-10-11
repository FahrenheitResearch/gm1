"""tools/perf_probe/compare_published_words.py: the word comparison --cuda-graphs was held to."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "compare_published_words", ROOT / "tools/perf_probe/compare_published_words.py"
)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def _run(root: Path, *, words: bytes, identity: str, payload: str, seconds: float) -> Path:
    frame = root / "frames/frame-000001"
    frame.mkdir(parents=True)
    (frame / "history-carriers.f32").write_bytes(words)
    (frame / "history-scheduler-scalars.tsv").write_text("step\t40\n")
    receipt = {"execution_identity_sha256": identity, "rust_output": f'{{"payload_sha256":"{payload}"}}'}
    (frame / "history-publication.json").write_text(json.dumps(receipt, indent=1))
    (frame / "history-field-table.tsv").write_text(f"u\t{frame}/history-carriers.f32\t0\n")
    window = root / "frames/nest-child-window"
    window.mkdir(parents=True)
    (window / "window-stream.json").write_text(json.dumps({"device_copy_seconds": seconds}, indent=1))
    (root / "restarts").mkdir()
    return root


def test_words_agree_across_trees_and_every_file_within_one(tmp_path: Path) -> None:
    payload = "a" * 64
    a = _run(tmp_path / "a", words=b"\x00\x01", identity="1" * 64, payload=payload, seconds=0.1)
    b = _run(tmp_path / "b", words=b"\x00\x01", identity="2" * 64, payload=payload, seconds=0.2)
    differences, counts = tool.compare(a, b, same_tree=False)
    assert differences == [] and counts["word_files"] == 2 and counts["payload_hashes"] == 1
    # One tree: the identity field is a real difference; the path and the
    # wall-clock seconds are not.
    differences, _ = tool.compare(a, b, same_tree=True)
    assert differences == ["text differs: frames/frame-000001/history-publication.json"]
    c = _run(tmp_path / "c", words=b"\x00\x01", identity="1" * 64, payload=payload, seconds=0.3)
    assert tool.compare(a, c, same_tree=True)[0] == []


def test_a_changed_word_or_payload_is_found(tmp_path: Path) -> None:
    a = _run(tmp_path / "a", words=b"\x00\x01", identity="1" * 64, payload="a" * 64, seconds=0.1)
    b = _run(tmp_path / "b", words=b"\x00\x02", identity="1" * 64, payload="b" * 64, seconds=0.1)
    differences, _ = tool.compare(a, b, same_tree=False)
    assert differences == [
        "words differ: frames/frame-000001/history-carriers.f32",
        "payload hashes differ: frames/frame-000001/history-publication.json",
    ]
