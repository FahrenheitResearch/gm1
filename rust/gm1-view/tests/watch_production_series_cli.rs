use std::{
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn missing_production_series_refuses_before_renderer_or_gpu_startup() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let missing = std::env::temp_dir().join(format!(
        "gm1-view-missing-production-series-{}-{nonce}",
        std::process::id()
    ));
    assert!(!missing.exists());
    let output = Command::new(env!("CARGO_BIN_EXE_gm1-view"))
        .arg("--watch-production-series")
        .arg(&missing)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("production series"));
    assert!(stderr.contains("root is unreadable"));
    assert!(!stderr.contains("wgpu"));
    assert!(!stderr.contains("adapter"));
}
