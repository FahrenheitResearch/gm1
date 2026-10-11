//! Renderer-free strict production-pack verification.
//!
//! This path deliberately stops after the same v2 envelope, payload,
//! in-memory rehash, renderer-residency, and cloud-dataset gates used by the
//! normal production renderer. It does not construct a Bevy app, adapter,
//! window, render device, or GPU resource.

use crate::{
    geofield::GeoFieldGeometry,
    production_playback::{ProductionResidencyKind, load_production_pack},
};
use serde::Serialize;
use std::path::{Path, PathBuf};

pub const PRODUCTION_PACK_VERIFICATION_SCHEMA: &str = "gm1.view.production-pack-verification/v1";
pub const PRODUCTION_PACK_VERIFICATION_STATUS: &str =
    "PASS_GPU_FREE_STRICT_PRODUCTION_PACK_VERIFICATION";
pub const PRODUCTION_PACK_VERIFICATION_REFUSAL: &str =
    "REFUSED_STRICT_PRODUCTION_PACK_VERIFICATION";

#[derive(Debug, Serialize)]
pub struct ProductionPackVerificationField {
    pub id: String,
    pub units: String,
    pub native_stagger: String,
    pub shape: Vec<usize>,
    pub values_sha256: String,
    pub payload_nbytes: usize,
    pub residency: &'static str,
    pub geometry: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ProductionPackVerificationReceipt {
    pub schema: &'static str,
    pub status: &'static str,
    pub pack_path: String,
    pub source_profile: &'static str,
    pub source_scientific_label: &'static str,
    pub domain: String,
    pub vertical_grid: &'static str,
    pub terrain: &'static str,
    pub step: u64,
    pub model_time_f64_bits: String,
    pub manifest_sha256: String,
    pub payload_set_sha256: String,
    pub history_contract_sha256: String,
    pub operator_manifest_sha256: String,
    pub geofield_abi: &'static str,
    pub render_adapter: &'static str,
    pub field_count: usize,
    pub verified_payload_nbytes: usize,
    pub fields: Vec<ProductionPackVerificationField>,
    pub renderer_invoked: bool,
    pub gpu_initialized: bool,
    pub claim_limit: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ProductionPackVerificationRefusal {
    pub schema: &'static str,
    pub status: &'static str,
    pub pack_path: String,
    pub error: String,
    pub renderer_invoked: bool,
    pub gpu_initialized: bool,
}

pub fn verify_production_pack(
    directory: &Path,
) -> Result<ProductionPackVerificationReceipt, String> {
    let canonical = directory
        .canonicalize()
        .map_err(|error| format!("production-pack path {}: {error}", directory.display()))?;
    let frame = load_production_pack(&canonical)?;
    let residency = frame.renderer_residency()?;
    let dataset = frame.cloud_dataset()?;
    if frame.source_path != canonical
        || residency.manifest_sha256 != frame.manifest_sha256
        || residency.payload_set_sha256 != frame.payload_set_sha256
        || residency.frame.source_identity != frame.payload_set_sha256
        || residency.frame.time.step != frame.step
        || residency.frame.time.model_time_s.to_bits() != frame.model_time_s.to_bits()
        || dataset.source_path != frame.source_path
        || dataset.artifact_sha256 != frame.manifest_sha256
        || dataset.payload_sha256 != frame.payload_set_sha256
        || dataset.contract_sha256 != frame.history_contract_sha256
        || dataset.step != frame.step
        || dataset.model_time_s.to_bits() != frame.model_time_s.to_bits()
    {
        return Err(
            "production-pack renderer/dataset residency lineage changed after strict admission"
                .to_owned(),
        );
    }
    let cloud = dataset
        .fields
        .first()
        .ok_or_else(|| "production-pack cloud dataset is empty".to_owned())?;
    let resident_cloud = residency
        .field("cloud_condensate_kgkg")
        .ok_or_else(|| "production-pack residency lacks cloud condensate".to_owned())?;
    if dataset.fields.len() != 1
        || cloud.values_sha256 != resident_cloud.descriptor.values_sha256
        || cloud.values.as_slice() != resident_cloud.values.as_slice()
    {
        return Err("production-pack cloud dataset changed after residency validation".to_owned());
    }

    let mut verified_payload_nbytes = 0_usize;
    let mut fields = Vec::with_capacity(residency.fields.len());
    for field in &residency.fields {
        let payload_nbytes = field
            .values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "production-pack resident byte count overflowed".to_owned())?;
        verified_payload_nbytes = verified_payload_nbytes
            .checked_add(payload_nbytes)
            .ok_or_else(|| "production-pack aggregate byte count overflowed".to_owned())?;
        fields.push(ProductionPackVerificationField {
            id: field.descriptor.id.clone(),
            units: field.descriptor.units.clone(),
            native_stagger: field.stagger.clone(),
            shape: field.descriptor.shape.clone(),
            values_sha256: field.descriptor.values_sha256.clone(),
            payload_nbytes,
            residency: residency_label(field.residency),
            geometry: geometry_label(field.descriptor.geometry),
        });
    }
    if fields.len() != frame.fields.len()
        || fields.iter().zip(&frame.fields).any(|(resident, source)| {
            resident.id != source.name
                || resident.shape != source.shape
                || resident.values_sha256 != source.values_sha256
                || resident.native_stagger != source.stagger
        })
    {
        return Err("production-pack field inventory changed after strict residency".to_owned());
    }

    Ok(ProductionPackVerificationReceipt {
        schema: PRODUCTION_PACK_VERIFICATION_SCHEMA,
        status: PRODUCTION_PACK_VERIFICATION_STATUS,
        pack_path: canonical.display().to_string(),
        source_profile: frame.source_profile.id,
        source_scientific_label: frame.source_profile.scientific_label,
        domain: frame.source_profile.domain_label(),
        vertical_grid: frame.source_profile.vertical_grid_label(),
        terrain: frame.source_profile.terrain_label(),
        step: frame.step,
        model_time_f64_bits: format!("{:016x}", frame.model_time_s.to_bits()),
        manifest_sha256: frame.manifest_sha256,
        payload_set_sha256: frame.payload_set_sha256,
        history_contract_sha256: frame.history_contract_sha256,
        operator_manifest_sha256: frame.operator_manifest_sha256,
        geofield_abi: residency.frame.abi,
        render_adapter: residency.frame.adapter_id,
        field_count: fields.len(),
        verified_payload_nbytes,
        fields,
        renderer_invoked: false,
        gpu_initialized: false,
        claim_limit: "strict v2 authority, bytes, decoded words, and renderer residency only; no render, forecast-skill, derived-meteorology, or GPU claim",
    })
}

pub fn production_pack_verification_refusal(
    directory: &Path,
    error: &str,
) -> ProductionPackVerificationRefusal {
    ProductionPackVerificationRefusal {
        schema: PRODUCTION_PACK_VERIFICATION_SCHEMA,
        status: PRODUCTION_PACK_VERIFICATION_REFUSAL,
        pack_path: stable_path(directory).display().to_string(),
        error: error.to_owned(),
        renderer_invoked: false,
        gpu_initialized: false,
    }
}

fn stable_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

const fn residency_label(kind: ProductionResidencyKind) -> &'static str {
    match kind {
        ProductionResidencyKind::CloudScalarVolume => "cloud-scalar-volume",
        ProductionResidencyKind::XFaceWindVolume => "x-face-wind-volume",
        ProductionResidencyKind::YFaceWindVolume => "y-face-wind-volume",
        ProductionResidencyKind::ZFaceWindVolume => "z-face-wind-volume",
        ProductionResidencyKind::PressureScalarVolume => "pressure-scalar-volume",
        ProductionResidencyKind::TerrainSurface => "terrain-surface",
        ProductionResidencyKind::StaticVerticalFaces => "static-vertical-faces",
        ProductionResidencyKind::TerrainFollowingVerticalFaces => {
            "terrain-following-vertical-faces"
        }
    }
}

const fn geometry_label(geometry: GeoFieldGeometry) -> &'static str {
    match geometry {
        GeoFieldGeometry::VolumeZyx => "volume-zyx",
        GeoFieldGeometry::RasterYx => "raster-yx",
        GeoFieldGeometry::RgbaYx => "rgba-yx",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusal_is_compact_machine_readable_and_gpu_free() {
        let refusal = production_pack_verification_refusal(
            Path::new("missing-step"),
            "strict source bytes are absent",
        );
        let encoded = serde_json::to_string(&refusal).unwrap();
        assert!(!encoded.contains('\n'));
        let decoded: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded["schema"], PRODUCTION_PACK_VERIFICATION_SCHEMA);
        assert_eq!(decoded["status"], PRODUCTION_PACK_VERIFICATION_REFUSAL);
        assert_eq!(decoded["renderer_invoked"], false);
        assert_eq!(decoded["gpu_initialized"], false);
        assert_eq!(decoded["error"], "strict source bytes are absent");
    }

    #[test]
    fn every_renderer_residency_has_a_stable_receipt_label() {
        let labels = [
            ProductionResidencyKind::CloudScalarVolume,
            ProductionResidencyKind::XFaceWindVolume,
            ProductionResidencyKind::YFaceWindVolume,
            ProductionResidencyKind::ZFaceWindVolume,
            ProductionResidencyKind::PressureScalarVolume,
            ProductionResidencyKind::TerrainSurface,
            ProductionResidencyKind::StaticVerticalFaces,
            ProductionResidencyKind::TerrainFollowingVerticalFaces,
        ]
        .map(residency_label);
        assert_eq!(labels.len(), 8);
        assert!(labels.iter().all(|label| !label.is_empty()));
        assert_eq!(geometry_label(GeoFieldGeometry::VolumeZyx), "volume-zyx");
        assert_eq!(geometry_label(GeoFieldGeometry::RasterYx), "raster-yx");
        assert_eq!(geometry_label(GeoFieldGeometry::RgbaYx), "rgba-yx");
    }
}
