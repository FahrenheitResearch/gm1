use std::{fs, path::Path};

use serde_json::Value;
use sha2::{Digest, Sha256};

fn sha256(path: &Path) -> String {
    let bytes = fs::read(path).expect("source should be readable");
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn real_arwen_wrfout_admission_receipt_is_descriptor_only_and_fail_closed() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let receipt_path =
        root.join("evidence/validation/gm1-view-arwen-wrfout-d04-geofield-v1/receipt.json");
    // The receipt is a measurement record of the development repository; a source tree without it (the public
    // snapshot) has nothing for this test to read.
    if !receipt_path.is_file() {
        eprintln!("skipped: {} is not in this tree", receipt_path.display());
        return;
    }
    let receipt: Value = serde_json::from_slice(
        &fs::read(&receipt_path).expect("real ArWen evidence receipt should exist"),
    )
    .expect("real ArWen evidence receipt should parse");

    assert_eq!(
        receipt["status"],
        "PASS_STRICT_REAL_CLASSIC_CDF_WRFOUT_GEOFRAME"
    );
    assert_eq!(receipt["source"]["bytes"], 3_394_407_204_u64);
    assert_eq!(
        receipt["source"]["sha256"],
        "cb5c9e214e420552308da2eea10ca3f0b1aedbc08c50fdcac4269056aaee84c9"
    );
    assert_eq!(receipt["field_counts"]["available"], 18);
    assert_eq!(receipt["field_counts"]["missing"], 3);
    assert_eq!(receipt["field_counts"]["refused"], 0);
    assert_eq!(receipt["timings_ms"]["total"], 114_369);
    assert_eq!(receipt["memory"]["peak_rss_bytes"], Value::Null);
    assert_eq!(
        receipt["memory"]["peak_rss_status"],
        "NOT_INSTRUMENTED_NO_RSS_CLAIM"
    );
    assert_eq!(receipt["claim_scope"]["renderer_payload_residency"], false);
    assert_eq!(receipt["claim_scope"]["destaggered_fields"], false);
    assert_eq!(
        receipt["claim_scope"]["synthetic_or_zero_filled_carriers"],
        false
    );
    assert_eq!(
        receipt["actual_fail_closed_probe"]["status"],
        "REFUSED_UNSUPPORTED_HDF5_NETCDF4_BEFORE_FIELD_ADMISSION"
    );

    let code = &receipt["source_code"];
    for (relative, key) in [
        (
            "rust/gm1-view/src/arwen_wrfout.rs",
            "arwen_wrfout_rs_sha256",
        ),
        (
            "rust/gm1-view/src/bin/gm1_wrfout_probe.rs",
            "gm1_wrfout_probe_rs_sha256",
        ),
        ("rust/gm1-view/src/geofield.rs", "geofield_rs_sha256"),
        ("rust/gm1-view/src/lib.rs", "lib_rs_sha256"),
    ] {
        assert_eq!(sha256(&root.join(relative)), code[key]);
    }

    let serialized = serde_json::to_string(&receipt).unwrap();
    for forbidden in [
        "finite_or_zero",
        "hydrostatic pressure fallback",
        "heuristic vortex",
        "synthetic opacity",
    ] {
        assert!(!serialized.contains(forbidden));
    }
}
