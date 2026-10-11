use std::{
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn missing_pack_emits_machine_readable_gpu_free_refusal() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let missing = std::env::temp_dir().join(format!(
        "gm1-view-verify-missing-{}-{nonce}",
        std::process::id()
    ));
    assert!(!missing.exists());
    let output = Command::new(env!("CARGO_BIN_EXE_gm1-view"))
        .arg("--verify-production-pack")
        .arg(&missing)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let receipt: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        receipt["schema"],
        "gm1.view.production-pack-verification/v1"
    );
    assert_eq!(
        receipt["status"],
        "REFUSED_STRICT_PRODUCTION_PACK_VERIFICATION"
    );
    assert_eq!(receipt["renderer_invoked"], false);
    assert_eq!(receipt["gpu_initialized"], false);
    assert!(
        receipt["error"]
            .as_str()
            .unwrap()
            .contains("production-pack path")
    );
}
