//! Opt-in stage timings for the forecast publisher.
//!
//! When the process environment sets `GPUWM_CM1_PUBLISH_TIMING=1`, the
//! encoder, the atomic writer and `cm1-forecast-publish` record how long each
//! stage of a publication took, and the binary prints one
//! `gm1-publish-timing<TAB>stage<TAB>seconds<TAB>bytes` line per stage on
//! stderr.  Nothing is recorded otherwise, and nothing here ever reaches
//! stdout or a published byte.

use std::sync::{Mutex, OnceLock};

pub const ENVIRONMENT: &str = "GPUWM_CM1_PUBLISH_TIMING";
pub const PREFIX: &str = "gm1-publish-timing";

static ENABLED: OnceLock<bool> = OnceLock::new();
static ROWS: Mutex<Vec<(&'static str, f64, u64)>> = Mutex::new(Vec::new());

pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var(ENVIRONMENT).is_ok_and(|value| value == "1"))
}

pub fn record(stage: &'static str, seconds: f64, bytes: u64) {
    if !enabled() {
        return;
    }
    if let Ok(mut rows) = ROWS.lock() {
        rows.push((stage, seconds, bytes));
    }
}

/// Every recorded stage as stderr lines, in recording order, draining them.
pub fn take_lines() -> String {
    let rows = match ROWS.lock() {
        Ok(mut rows) => std::mem::take(&mut *rows),
        Err(_) => return String::new(),
    };
    rows.into_iter()
        .map(|(stage, seconds, bytes)| format!("{PREFIX}\t{stage}\t{seconds:.6}\t{bytes}\n"))
        .collect()
}
