//! Publish the exact source-backed HRRR terrain context used by GM1 View.

use gm1_view::hrrr_surround_terrain::{
    EXACT_CHILD_TERRAIN_SHA256, HRRR_BRIDGE_BUILD_MANIFEST_SHA256, HRRR_BRIDGE_SHA256,
    HRRR_BRIDGE_SOURCE_COMMIT, HRRR_DECODED_WINDOW_SCHEMA, HRRR_FETCH_MANIFEST_SHA256,
    HRRR_SOIL_F02_SHA256, HRRR_SOIL_F03_SHA256, HRRR_SOURCE_F02_FORECAST_HOUR,
    HRRR_SOURCE_VALID_TIME_UTC, HRRR_SURROUND_ASSET_DIRECTORY, HRRR_SURROUND_CLAIM_LIMIT,
    HRRR_SURROUND_HEIGHT_TREATMENT, HRRR_SURROUND_MANIFEST_NAME, HRRR_SURROUND_MANIFEST_SCHEMA,
    HRRR_SURROUND_NX, HRRR_SURROUND_NY, HRRR_SURROUND_OWNERSHIP, HRRR_SURROUND_PAYLOAD_NAME,
    HRRR_SURROUND_RECEIPT_NAME, HRRR_SURROUND_RECEIPT_SCHEMA, HRRR_SURROUND_STATUS,
    HRRR_SURROUND_WINDOW_I_START, HRRR_SURROUND_WINDOW_J_START, HRRR_WRFNAT_F02_SHA256,
    HRRR_WRFNAT_F03_SHA256, source_backed_surround_mesh_from_envelope,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process,
};

const RAW_FILES: [(&str, u64, &str); 4] = [
    ("hrrr.t19z.soilf02.grib2", 431_574_022, HRRR_SOIL_F02_SHA256),
    ("hrrr.t19z.soilf03.grib2", 434_331_894, HRRR_SOIL_F03_SHA256),
    (
        "hrrr.t19z.wrfnatf02.grib2",
        763_838_413,
        HRRR_WRFNAT_F02_SHA256,
    ),
    (
        "hrrr.t19z.wrfnatf03.grib2",
        766_890_103,
        HRRR_WRFNAT_F03_SHA256,
    ),
];

fn main() {
    if let Err(error) = run() {
        eprintln!("gm1-view HRRR surrounding-terrain publisher refused: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let arguments = env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    if arguments.len() != 4 {
        return Err(format!(
            "usage: gm1_view_hrrr_surround <decoded-window-root> <raw-grib-root> <hrrr_grib2_bridge> <asset-parent>; publishes <asset-parent>/{HRRR_SURROUND_ASSET_DIRECTORY}"
        ));
    }
    let decoded_root = &arguments[0];
    let raw_root = &arguments[1];
    let bridge_path = &arguments[2];
    let asset_parent = &arguments[3];

    verify_file(bridge_path, None, HRRR_BRIDGE_SHA256)?;
    let mut source_files = Vec::with_capacity(RAW_FILES.len());
    for (name, nbytes, expected_sha256) in RAW_FILES {
        verify_file(&raw_root.join(name), Some(nbytes), expected_sha256)?;
        source_files.push(json!({"name":name,"nbytes":nbytes,"sha256":expected_sha256}));
    }

    let gate_path = decoded_root.join("gate.txt");
    let inventory_path = decoded_root.join("inventory.tsv");
    let gate_bytes = read_file(&gate_path)?;
    let inventory_bytes = read_file(&inventory_path)?;
    validate_gate(&gate_bytes)?;
    if !String::from_utf8_lossy(&inventory_bytes).contains("atmosphere-f02\t1032\tSOILHGT") {
        return Err("decoded inventory does not bind atmosphere-f02 SOILHGT".into());
    }

    // This viewer adapter consumes SOILHGT only. The selected sealed bridge
    // predates later VVEL/direct-reflectivity output additions, so the full
    // CM1 forecast-window reader is intentionally not invoked here. The raw
    // GRIBs, decoder gate/inventory, every decoded atmosphere byte, and the
    // exact terrain payload are nevertheless independently hashed.
    let source_f02_fingerprint = fingerprint_tree(&decoded_root.join("atmosphere-f02"))?;
    let source_f03_fingerprint = fingerprint_tree(&decoded_root.join("atmosphere-f03"))?;
    let payload = read_file(&decoded_root.join("atmosphere-f02/SOILHGT.f32le"))?;
    let payload_sha256 = sha256_bytes(&payload);
    let gate_sha256 = sha256_bytes(&gate_bytes);
    let inventory_sha256 = sha256_bytes(&inventory_bytes);
    let decoded_tree_sha256 = fingerprint_tree(decoded_root)?;

    let manifest = json!({
        "schema": HRRR_SURROUND_MANIFEST_SCHEMA,
        "status": HRRR_SURROUND_STATUS,
        "source": {
            "cycle_utc": "2024-05-21T19:00:00Z",
            "valid_time_utc": HRRR_SOURCE_VALID_TIME_UTC,
            "forecast_hour": HRRR_SOURCE_F02_FORECAST_HOUR,
            "fetch_manifest_sha256": HRRR_FETCH_MANIFEST_SHA256,
            "files": source_files
        },
        "decoder": {
            "name": format!("ArWen RW-WPS hrrr_grib2_bridge @ {HRRR_BRIDGE_SOURCE_COMMIT}"),
            "source_commit": HRRR_BRIDGE_SOURCE_COMMIT,
            "executable_sha256": HRRR_BRIDGE_SHA256,
            "build_manifest_sha256": HRRR_BRIDGE_BUILD_MANIFEST_SHA256,
            "output_schema": HRRR_DECODED_WINDOW_SCHEMA,
            "gate_sha256": gate_sha256,
            "inventory_sha256": inventory_sha256,
            "atmosphere_f02_fingerprint_sha256": source_f02_fingerprint,
            "atmosphere_f03_fingerprint_sha256": source_f03_fingerprint,
            "decoded_tree_sha256": decoded_tree_sha256
        },
        "payload": {
            "name": "SOILHGT.f32le",
            "nbytes": payload.len() as u64,
            "sha256": payload_sha256,
            "dtype": "little-endian binary32",
            "order": "south-to-north row-major",
            "shape": [HRRR_SURROUND_NY, HRRR_SURROUND_NX]
        },
        "grid": {
            "name": "HRRR Lambert GDT3.30",
            "full_shape_xy": [1799, 1059],
            "window_origin_ij": [HRRR_SURROUND_WINDOW_I_START, HRRR_SURROUND_WINDOW_J_START],
            "window_shape_xy": [HRRR_SURROUND_NX, HRRR_SURROUND_NY],
            "spacing_m_xy": [3000, 3000],
            "scan_mode": "0x40"
        },
        "transform": {
            "method": "HRRR Lambert inverse -> spherical ECEF -> child ENU",
            "earth_radius_m_f64_bits": "41584de740000000",
            "child_center_lat_f64_bits": "4044bb15cda77904",
            "child_center_lon_f64_bits": "c0578c5ebb59704f",
            "output_axes_units": "east-km, up-km, north-km"
        },
        "overlay": {
            "exact_child_terrain_sha256": EXACT_CHILD_TERRAIN_SHA256,
            "exact_child_half_extent_km_f32_bits": format!("{:08x}", 7.475_f32.to_bits()),
            "ownership": HRRR_SURROUND_OWNERSHIP,
            "height_treatment": HRRR_SURROUND_HEIGHT_TREATMENT
        },
        "claim_limit": HRRR_SURROUND_CLAIM_LIMIT
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed to serialize surround manifest: {error}"))?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let receipt = json!({
        "schema": HRRR_SURROUND_RECEIPT_SCHEMA,
        "status": HRRR_SURROUND_STATUS,
        "manifest_sha256": manifest_sha256,
        "payload_sha256": payload_sha256,
        "exact_child_terrain_sha256": EXACT_CHILD_TERRAIN_SHA256,
        "no_extrapolation": true,
        "earth_curvature_retained": true,
        "claim_limit": HRRR_SURROUND_CLAIM_LIMIT
    });
    let receipt_bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("failed to serialize surround receipt: {error}"))?;
    let (_, stats) =
        source_backed_surround_mesh_from_envelope(&manifest_bytes, &receipt_bytes, &payload)?;

    fs::create_dir_all(asset_parent).map_err(|error| {
        format!(
            "failed to create asset parent {}: {error}",
            asset_parent.display()
        )
    })?;
    let final_directory = asset_parent.join(HRRR_SURROUND_ASSET_DIRECTORY);
    if final_directory.exists() {
        return Err(format!(
            "refusing to overwrite immutable asset directory {}",
            final_directory.display()
        ));
    }
    let staging = asset_parent.join(format!(
        ".{HRRR_SURROUND_ASSET_DIRECTORY}.partial-{}",
        process::id()
    ));
    if staging.exists() {
        return Err(format!(
            "staging directory already exists: {}",
            staging.display()
        ));
    }
    fs::create_dir(&staging)
        .map_err(|error| format!("failed to create staging directory: {error}"))?;
    write_new_sync(&staging.join(HRRR_SURROUND_PAYLOAD_NAME), &payload)?;
    write_new_sync(&staging.join(HRRR_SURROUND_MANIFEST_NAME), &manifest_bytes)?;
    write_new_sync(&staging.join(HRRR_SURROUND_RECEIPT_NAME), &receipt_bytes)?;
    fs::rename(&staging, &final_directory)
        .map_err(|error| format!("atomic asset-directory publication failed: {error}"))?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": HRRR_SURROUND_STATUS,
            "directory": final_directory,
            "payload_sha256": payload_sha256,
            "manifest_sha256": manifest_sha256,
            "receipt_sha256": sha256_bytes(&receipt_bytes),
            "decoded_tree_sha256": decoded_tree_sha256,
            "bounds_east_km": [stats.minimum_east_km, stats.maximum_east_km],
            "bounds_north_km": [stats.minimum_north_km, stats.maximum_north_km],
            "bounds_up_km": [stats.minimum_up_km, stats.maximum_up_km],
            "horizontal_margin_km": stats.horizontal_margin_km(),
            "indexed_triangle_count": stats.indexed_triangle_count
        }))
        .map_err(|error| format!("failed to serialize publication summary: {error}"))?
    );
    Ok(())
}

fn validate_gate(bytes: &[u8]) -> Result<(), String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("decoded gate is not UTF-8: {error}"))?;
    for required in [
        "status\tPASS",
        "cycle\t2024-05-21 19:00:00",
        "valid_times\t2024-05-21 19:00:00+02h,2024-05-21 19:00:00+03h",
        "forecast_hours\t2,3",
        "source_forecast_hours\t2,3",
        "grid\tHRRR Lambert GDT3.30 1799x1059 3000m shape6 scan0x40 uv-grid-relative",
        "window_zero_based_inclusive\ti=959..1022 j=609..672",
        "window_shape\t64x64",
        "output\tFP32 little-endian south-to-north row-major; 3-D level-major",
    ] {
        if !text.lines().any(|line| line == required) {
            return Err(format!("decoded gate is missing exact line {required:?}"));
        }
    }
    Ok(())
}

fn verify_file(
    path: &Path,
    expected_nbytes: Option<u64>,
    expected_sha256: &str,
) -> Result<(), String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("failed to stat {}: {error}", path.display()))?;
    if !metadata.is_file() || expected_nbytes.is_some_and(|size| size != metadata.len()) {
        return Err(format!("file size/type changed for {}", path.display()));
    }
    let actual = sha256_file(path)?;
    if actual != expected_sha256 {
        return Err(format!(
            "file identity changed for {}: {actual}",
            path.display()
        ));
    }
    Ok(())
}

fn fingerprint_tree(root: &Path) -> Result<String, String> {
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    hasher.update(b"gm1-view-decoded-hrrr-tree/v1\0");
    for (relative, path) in files {
        let metadata = fs::metadata(&path)
            .map_err(|error| format!("failed to stat decoded file {}: {error}", path.display()))?;
        let digest = sha256_file(&path)?;
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update(metadata.len().to_le_bytes());
        hasher.update(digest.as_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_files(
    root: &Path,
    directory: &Path,
    output: &mut Vec<(String, PathBuf)>,
) -> Result<(), String> {
    for entry in fs::read_dir(directory)
        .map_err(|error| format!("failed to enumerate {}: {error}", directory.display()))?
    {
        let entry = entry.map_err(|error| format!("decoded tree entry failed: {error}"))?;
        let file_type = entry
            .file_type()
            .map_err(|error| format!("decoded entry type failed: {error}"))?;
        let path = entry.path();
        if file_type.is_symlink() {
            return Err(format!("decoded tree contains symlink {}", path.display()));
        }
        if file_type.is_dir() {
            collect_files(root, &path, output)?;
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|error| format!("decoded path escaped root: {error}"))?
                .to_string_lossy()
                .replace('\\', "/");
            output.push((relative, path));
        } else {
            return Err(format!(
                "decoded tree contains special file {}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn read_file(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_new_sync(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("failed to sync {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn gate_fixture() -> Vec<u8> {
        [
            "status\tPASS",
            "cycle\t2024-05-21 19:00:00",
            "valid_times\t2024-05-21 19:00:00+02h,2024-05-21 19:00:00+03h",
            "forecast_hours\t2,3",
            "source_forecast_hours\t2,3",
            "grid\tHRRR Lambert GDT3.30 1799x1059 3000m shape6 scan0x40 uv-grid-relative",
            "window_zero_based_inclusive\ti=959..1022 j=609..672",
            "window_shape\t64x64",
            "output\tFP32 little-endian south-to-north row-major; 3-D level-major",
        ]
        .join("\n")
        .into_bytes()
    }

    #[test]
    fn decoder_gate_is_exact_and_refuses_old_window() {
        let gate = gate_fixture();
        validate_gate(&gate).expect("exact 64x64 gate");
        let old = String::from_utf8(gate)
            .unwrap()
            .replace("i=959..1022 j=609..672", "i=971..996 j=622..647");
        assert!(validate_gate(old.as_bytes()).is_err());
    }

    #[test]
    fn decoded_tree_fingerprint_is_order_stable_and_byte_sensitive() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            env::temp_dir().join(format!("gm1-view-hrrr-tree-test-{}-{nonce}", process::id()));
        fs::create_dir_all(root.join("b")).unwrap();
        fs::write(root.join("b/z.bin"), b"z").unwrap();
        fs::write(root.join("a.bin"), b"a").unwrap();
        let first = fingerprint_tree(&root).unwrap();
        let second = fingerprint_tree(&root).unwrap();
        assert_eq!(first, second);
        fs::write(root.join("b/z.bin"), b"changed").unwrap();
        assert_ne!(first, fingerprint_tree(&root).unwrap());
        fs::remove_dir_all(root).unwrap();
    }
}
