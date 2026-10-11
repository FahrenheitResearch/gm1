from __future__ import annotations

import csv
import json
from importlib import resources


REQUIRED = {
    "CM1-AUTHORITY.json",
    "CM1-SOURCE-MAP.md",
    "CM1-STATE-MAP.csv",
    "CM1-CADENCE-MAP.md",
    "CM1-SUPPORT-MATRIX.csv",
    "CM1-ARITHMETIC-CONTRACT.md",
    "CM1-ARWEN-SEAM.md",
    "CM1-NEST-SEAM.md",
    "CM1-ORACLE-MANIFEST.json",
    "CM1-PROFILE-MANIFEST.json",
    "CM1-RISK-REGISTER.md",
}


def test_required_records_are_installed_resources() -> None:
    root = resources.files("cm1core.records")
    assert REQUIRED <= {item.name for item in root.iterdir() if item.is_file()}


def test_json_records_bind_ratified_oracles_without_overclaim() -> None:
    root = resources.files("cm1core.records")
    authority = json.loads(
        root.joinpath("CM1-AUTHORITY.json").read_text(encoding="utf-8")
    )
    oracle = json.loads(
        root.joinpath("CM1-ORACLE-MANIFEST.json").read_text(encoding="utf-8")
    )
    assert authority["mathematical_authority"]["sha256"].startswith("dc49fe8453")
    assert oracle["status"] == "partial-native-oracle-campaign"
    assert oracle["instrument_status"] == (
        "partial-grid-base-and-post-getset-passed"
    )
    assert len(oracle["fixtures"]) == 5
    fixtures = {item["id"]: item for item in oracle["fixtures"]}
    fixture = fixtures["O-GRID-BASE-V1"]
    assert fixture["id"] == "O-GRID-BASE-V1"
    assert fixture["status"] == "passed-exact-native"
    assert fixture["authority_sha256"] == authority["mathematical_authority"]["sha256"]
    assert fixture["receipt_sha256"] == (
        "d6558beb15856aa06d6c9ddb8bd2f27d88688a9d6fa8182467bbffa1ce290409"
    )
    assert fixture["deck_manifest_sha256"] == (
        "0eec3831de568e57fe597f812daa6697c0846d19938294fdc2ad1a4628356f87"
    )
    assert fixture["input_sha256"] == (
        "29f41f664adf36cc751db8997dce036b00eca4fcec03f7d5ea381730ee5646dc"
    )
    assert fixture["output_sha256"] == (
        "b5842936eec58e9717883bf376c51e129b1b6ab6ad334d33f9db2264f237869d"
    )
    assert fixture["comparison_contract"] == {"mode": "raw-bit", "max_ulp": 0}
    assert "tolerance" not in fixture
    assert fixture["evidence_directory"].startswith("development-repository:")
    assert fixture["receipt"].startswith("development-repository:")
    assert fixture["deck_manifest"].startswith("development-repository:")
    assert fixture["negative_controls"] == [
        "input-bit",
        "location",
        "cadence",
        "output",
    ]
    assert "not complete" in fixture["claim_limit"]
    assert "five sampled three-dimensional columns" in fixture["claim_limit"]
    assert "no complete 3-D halo/base-state coverage" in fixture["claim_limit"]
    assert "no u0 ie+1 or v0 je+1 sample" in fixture["claim_limit"]

    post = fixtures["O-POST-GETSET-V1"]
    assert post["status"] == "passed-exact-native-bounded"
    assert post["authority_sha256"] == authority["mathematical_authority"]["sha256"]
    assert post["receipt_sha256"] == (
        "c2c0666e871cf95b970d27fbcd34a1b149b11e7f623706591a9f9a16e6d09089"
    )
    assert post["deck_manifest_sha256"] == (
        "a583d6ceb0d532ad235386192edad468a6005d95c91fea5212e1d02e2d656170"
    )
    assert post["input_sha256"] == (
        "ffb8cf11975e047c33ac17c5a55a800775f8670f1fd8fb06eb676221e953eea7"
    )
    assert post["output_sha256"] == (
        "1e09fc90af50f22ed345afc29c56f016a334404227043a801dc90e7eb364e8c1"
    )
    assert post["layout_sha256"] == (
        "c75b49a11dbe0e4ad436e0654ff18bf9510e9b95e63c4fa8c9986091a4de7763"
    )
    assert post["region_manifest_sha256"] == (
        "b6f7f715ff7f0fba51e91d4a1679641b9038757c00583b23b89908ddb66df8d6"
    )
    assert post["artifact_inventory_sha256"] == (
        "b8ab9fba4521196d47c4dce36cb2e39abfbce7fb6d3ea0d0e76071ad1f6f71ac"
    )
    assert post["artifact_inventory"].startswith("development-repository:")
    assert post["comparison_contract"] == {"mode": "raw-bit", "max_ulp": 0}
    assert "tolerance" not in post
    assert post["complete_captured_field_files"] == 46
    assert post["complete_captured_field_bytes_per_run"] == 123_070_752
    assert post["source_adjacent_artifacts_per_run"] == 48
    assert post["source_adjacent_non_field_streams"] == [
        "post-getset-inputs.tsv",
        "post-getset-layout.tsv",
    ]
    assert {"rf0", "rr0", "rrf0", "qc0", "qi0"} <= set(post["known_omissions"])
    assert "prs halo is storage-only" in post["claim_limit"]
    assert "not complete first-solve state" in post["claim_limit"]

    idiss = fixtures["O-EQTSET2-IDISS-V1"]
    assert idiss["status"] == "passed-exact-native-and-sm86-bounded"
    assert idiss["authority_sha256"] == authority["mathematical_authority"][
        "sha256"
    ]
    assert idiss["receipt_sha256"] == (
        "c39001d12d208f8cfe28c8ed4605a6aa6f212ccd08a41adfb7699831b89fe1cc"
    )
    assert idiss["deck_manifest_sha256"] == (
        "3014a5384e82baccd1497b9fd944bf04cc1e1a47b3cb00e67a06628bdea14f4f"
    )
    assert idiss["cuda_receipt_sha256"] == (
        "1873306118eac27d80ff5c5c09208dc726838181ffd62aa85e37daf9ca0001a7"
    )
    assert idiss["artifact_inventory_sha256"] == (
        "cf2723e33538375fded349981181a928afbb43eaef0b0fc25e62e0eff155741d"
    )
    assert idiss["comparison_contract"] == {"mode": "raw-bit", "max_ulp": 0}
    assert idiss["cuda_output_words"] == 24
    assert "supplied dissten only" in idiss["claim_limit"]
    assert "no live SGS dissten producer" in idiss["claim_limit"]

    closure = fixtures["O-TERRAIN-DEFORMATION-SGS-V1"]
    assert closure["status"] == "passed-exact-native-and-sm86-bounded"
    assert closure["authority_sha256"] == authority["mathematical_authority"]["sha256"]
    assert closure["receipt_sha256"] == (
        "18e5cbd0576a1860d94b300fffeae7f4f18cbf672c3af4911877b99cbb3237a0"
    )
    assert closure["deck_manifest_sha256"] == (
        "251e6458e2925f220683d77ec84c679c7ee8c131a9d773f640227a9d2b099c95"
    )
    assert closure["cuda_receipt_sha256"] == (
        "a38cc53cf6afea02fbb1fb6a1d324384504fdedd968643a2edb9fdbf39d55bd8"
    )
    assert closure["real_receipt_sha256"] == (
        "b7e12ad93e70e609fe5db1164aa8439e043e3ab50d12185762ea2d04e617b1d0"
    )
    assert closure["comparison_contract"] == {"mode": "raw-bit", "max_ulp": 0}
    assert closure["record_count"] == 1241
    assert closure["real_live_closure_words"] == 6_660_000
    assert "selected terrain/all-open" in closure["claim_limit"]
    assert "no evolved/nonuniform-tkea turbt" in closure["claim_limit"]
    assert "tornado" in closure["claim_limit"]

    evolved = fixtures["O-EVOLVED-SGS-TKE-PRIMITIVES-V1"]
    assert evolved["status"] == "passed-exact-native-primitives-and-sm86-bounded"
    assert evolved["authority_sha256"] == authority["mathematical_authority"][
        "sha256"
    ]
    assert evolved["receipt_sha256"] == (
        "8291ceddd1e8eacb63a325dfb79774af7f7b196917dc573d98a627f91422bafa"
    )
    assert evolved["deck_manifest_sha256"] == (
        "c24778daae6a5fb21e91606ddb29fe97c1c7f8220e4ff453e3976158ea05ae44"
    )
    assert evolved["cuda_receipt_sha256"] == (
        "265d19974100505d1cb3e919ce342746ff7310565f0b5fa1177a33925092fc91"
    )
    assert evolved["real_receipt_sha256"] == (
        "1ce88abcc2133d43e29a00df068088a17f2e62417b0404855ae80939a578608c"
    )
    assert evolved["comparison_contract"] == {"mode": "raw-bit", "max_ulp": 0}
    assert evolved["record_count"] == 33
    assert "native proof covers only" in evolved["claim_limit"]
    assert "holding momentum and thermodynamic state fixed" in evolved[
        "claim_limit"
    ]
    assert "no coupled trajectory" in evolved["claim_limit"]


def test_instrument_records_keep_partial_ledger_and_bounded_sample_claims() -> None:
    root = resources.files("cm1core.records")
    source = root.joinpath("CM1-SOURCE-MAP.md").read_text(encoding="utf-8")
    cadence = root.joinpath("CM1-CADENCE-MAP.md").read_text(encoding="utf-8")
    risks = root.joinpath("CM1-RISK-REGISTER.md").read_text(encoding="utf-8")
    source_words = " ".join(source.split())

    assert "## Partial operator, call-site, and cadence ledger" in source
    assert "not an every-operator inventory" in source_words
    assert "`CORIOLIS-MOVING-FRAME` | `src/solve2.F:631-720`" in source
    assert "`BUOYANCY-APPLICATION` | `src/solve2.F:895-921`" in source
    assert "`src/param.F:6640-7451`, `7458-7726`" in source
    assert "development-repository:evidence/oracles/grid-base-v1/" in source
    assert "development-repository:evidence/oracles/post-getset-v1/" in source
    assert "exact write site `cm1.F:2905`" in source
    assert "ledger is exact" not in cadence
    assert "ledger is partial" in cadence
    r016 = next(line for line in risks.splitlines() if "R-016" in line)
    assert "partial and open" in r016
    assert "exact operator" not in r016

    rows = {
        row["state_name"]: row
        for row in csv.DictReader(
            root.joinpath("CM1-STATE-MAP.csv").open("r", encoding="utf-8", newline="")
        )
    }
    assert "full captured 1-D" in rows["xh"]["halo_policy"]
    assert "full captured 1-D" in rows["yf"]["halo_policy"]
    assert "only five sampled" in rows["zh"]["halo_policy"]
    assert "complete post-getset allocation" in rows["pi0"]["halo_policy"]
    assert "ie+1 extra x face" in rows["u0"]["halo_policy"]
    assert "je+1 extra y face" in rows["v0"]["halo_policy"]
    assert "physical interior" in rows["prs"]["halo_policy"]
    assert "stable zero storage" in rows["prs"]["halo_policy"]
    sampled = {
        "zh",
        "zf",
        "mh",
        "rmh",
        "mf",
        "rmf",
        "rr0",
        "rf0",
    }
    assert all(
        rows[name]["port_status"]
        == "native-five-column-post-base-sample-passed-cuda-unimplemented"
        for name in sampled
    )
    full_post_getset = {
        "pi0",
        "rho0",
        "prs0",
        "thv0",
        "th0",
        "rth0",
        "qv0",
        "u0",
        "v0",
        "ua",
        "u3d",
        "va",
        "v3d",
        "wa",
        "w3d",
        "ppi",
        "pp3d",
        "tha",
        "th3d",
    }
    assert all(
        "native-full-allocation-post-getset-oracle-passed" in rows[name]["port_status"]
        for name in full_post_getset
    )
    assert all(
        rows[name]["port_status"]
        == "native-masked-post-getset-oracle-passed-cuda-unimplemented"
        for name in ("rho", "rr", "rf")
    )


def test_canonical_profile_binds_every_native_option_by_fixture_bytes() -> None:
    from cm1core.config import load_config
    from cm1core.config_types import LaneName
    from cm1core.profiles import profile_identity

    root = resources.files("cm1core.records")
    manifest = json.loads(
        root.joinpath("CM1-PROFILE-MANIFEST.json").read_text(encoding="utf-8")
    )
    from pathlib import Path

    checkout = Path(__file__).resolve().parents[1]
    identity = load_config(
        checkout / "examples" / "official-supercell-product.toml"
    ).canonical_dict()["profile_identity"]
    expected_product = manifest["official_supercell"] | {"lane": LaneName.PRODUCT}
    assert identity == expected_product
    assert profile_identity(LaneName.PRODUCT) == expected_product

    expected_real = manifest["real_atmosphere_100m"] | {
        "lane": LaneName.REAL_PRODUCT
    }
    assert profile_identity(LaneName.REAL_PRODUCT) == expected_real

    expected_native = dict(manifest["official_supercell"])
    expected_native.pop("product_differences")
    expected_native |= {
        "lane": LaneName.NATIVE_CONTEXT,
        "execution_policy": "reference/context outside package; never a production door",
    }
    assert profile_identity(LaneName.NATIVE_CONTEXT) == expected_native

    expected_dry = manifest["dry_bootstrap"] | {"lane": LaneName.DRY_DYCORE}
    assert profile_identity(LaneName.DRY_DYCORE) == expected_dry


def test_support_matrix_has_one_row_per_parser_key() -> None:
    from cm1core.config import _SCHEMA, SUBTABLE_KEYS

    root = resources.files("cm1core.records")
    rows = list(
        csv.DictReader(
            root.joinpath("CM1-SUPPORT-MATRIX.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    expected = {f"{section}.{key}" for section, keys in _SCHEMA.items() for key in keys}
    # A sub-table documented key by key (config.SUBTABLE_KEYS) has a row for
    # itself, in _SCHEMA, and one per key beneath it.
    for table, keys in SUBTABLE_KEYS.items():
        assert table in expected
        expected |= {f"{table}.{key}" for key in keys}
    assert {row["option"] for row in rows} == expected


def test_state_map_has_distinct_core_families_and_complete_rows() -> None:
    root = resources.files("cm1core.records")
    rows = list(
        csv.DictReader(
            root.joinpath("CM1-STATE-MAP.csv").open("r", encoding="utf-8", newline="")
        )
    )
    assert rows
    assert all(None not in row for row in rows)
    assert all(all(value not in (None, "") for value in row.values()) for row in rows)
    names = {row["state_name"] for row in rows}
    assert all("|" not in name for name in names)
    assert {
        "ua",
        "u3d",
        "va",
        "v3d",
        "wa",
        "w3d",
        "ppi",
        "pp3d",
        "tha",
        "th3d",
        "qa",
        "q3d",
        "tkea",
        "tke3d",
        "pi0",
        "rho0",
        "th0",
        "xh",
        "xf",
        "nrk",
        "acoustic_substep",
        "component_persistence",
    } <= names
