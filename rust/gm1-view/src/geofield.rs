//! Shared georeferenced weather-frame boundary for renderer layers.
//!
//! GM1 and strict classic-NetCDF ArWen/WRFOUT sources implement this boundary.
//! The types intentionally avoid either model's storage details.

use crate::{
    input::Gm1Dataset,
    lighting::{T300_CENTER_LATITUDE_DEG, T300_CENTER_LONGITUDE_DEG},
};
use serde::Serialize;

pub const GEO_FIELD_ABI: &str = "gm1.view.georeferenced-weather-frame/v1";
pub const GM1_ADAPTER_ID: &str = "gm1.strict-cm1fhs-cm1fcp-and-cloud-pack/v1";
pub const ARWEN_WRF_ADAPTER_ID: &str = "arwen.strict-wrfout-adapter/v1";
pub const ARWEN_WRF_ADAPTER_STATUS: &str = "IMPLEMENTED_STRICT_CLASSIC_CDF1_CDF2_FAIL_CLOSED";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum GeoFieldGeometry {
    VolumeZyx,
    RasterYx,
    RgbaYx,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub enum GeoFieldAdmission {
    Available,
    Refused,
    Missing,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GeoGrid {
    pub center_latitude_deg: f64,
    pub center_longitude_deg: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx_m: f64,
    pub dy_m: f64,
    pub model_top_m: f64,
    pub horizontal_crs: &'static str,
    pub vertical_coordinate: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GeoTime {
    pub step: u64,
    pub model_time_s: f64,
    pub dt_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GeoFieldDescriptor {
    pub id: String,
    pub label: String,
    pub units: String,
    pub meaning: String,
    pub geometry: GeoFieldGeometry,
    pub shape: Vec<usize>,
    pub values_sha256: String,
    pub admission: GeoFieldAdmission,
    pub admission_reason: String,
    pub provenance: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GeoreferencedFrame {
    pub abi: &'static str,
    pub adapter_id: &'static str,
    pub source_identity: String,
    pub grid: GeoGrid,
    pub time: GeoTime,
    pub fields: Vec<GeoFieldDescriptor>,
}

/// Renderer-facing adapter. Implementations must perform source-specific
/// admission before returning a frame; the renderer consumes only this common
/// georeferenced description plus separately owned exact field words.
pub trait GeoreferencedWeatherAdapter {
    type Source;

    fn adapter_id(&self) -> &'static str;
    fn adapt(&self, source: &Self::Source) -> Result<GeoreferencedFrame, String>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Gm1GeoreferencedAdapter;

impl Gm1GeoreferencedAdapter {
    pub fn adapt_with_grid(
        &self,
        source: &Gm1Dataset,
        grid: GeoGrid,
    ) -> Result<GeoreferencedFrame, String> {
        if source.fields.is_empty() {
            return Err("GM1 frame has no admitted fields".to_owned());
        }
        if grid.nz == 0 || grid.ny == 0 || grid.nx == 0 {
            return Err(format!(
                "GM1 georeferenced adapter refuses a grid with a zero extent \
                 ({}x{}x{} as nz,ny,nx); the raymarcher would divide by a zero \
                 cell count and sample outside the payload",
                grid.nz, grid.ny, grid.nx
            ));
        }
        if !(grid.dx_m.is_finite() && grid.dx_m > 0.0)
            || !(grid.dy_m.is_finite() && grid.dy_m > 0.0)
            || !(grid.model_top_m.is_finite() && grid.model_top_m > 0.0)
        {
            return Err(format!(
                "GM1 georeferenced adapter refuses non-positive or non-finite \
                 grid spacing (dx={} m, dy={} m, model top={} m); the volume \
                 would be placed at an undefined physical size and every \
                 distance the renderer reports would be wrong",
                grid.dx_m, grid.dy_m, grid.model_top_m
            ));
        }
        let expected = [grid.nz, grid.ny, grid.nx];
        if let Some(mismatched) = source
            .fields
            .iter()
            .find(|field| field.shape != expected)
        {
            return Err(format!(
                "GM1 georeferenced adapter refuses field {} with shape \
                 {:?} against grid {:?} (nz,ny,nx); a field indexed with the \
                 wrong strides renders one column's words at another column's \
                 position, so the displayed volume would not be the simulated \
                 state",
                mismatched.id, mismatched.shape, expected
            ));
        }
        let fields = source
            .fields
            .iter()
            .map(|field| GeoFieldDescriptor {
                id: field.id.clone(),
                label: field.label.clone(),
                units: field.units.clone(),
                meaning: field.meaning.clone(),
                geometry: GeoFieldGeometry::VolumeZyx,
                shape: field.shape.to_vec(),
                values_sha256: field.values_sha256.clone(),
                admission: GeoFieldAdmission::Available,
                admission_reason: "strict GM1 artifact/pack admission passed".to_owned(),
                provenance: field.source.clone(),
            })
            .collect();
        Ok(GeoreferencedFrame {
            abi: GEO_FIELD_ABI,
            adapter_id: self.adapter_id(),
            source_identity: source.payload_sha256.clone(),
            grid,
            time: GeoTime {
                step: source.step,
                model_time_s: source.model_time_s,
                dt_s: source.dt_s,
            },
            fields,
        })
    }
}

impl GeoreferencedWeatherAdapter for Gm1GeoreferencedAdapter {
    type Source = Gm1Dataset;

    fn adapter_id(&self) -> &'static str {
        GM1_ADAPTER_ID
    }

    fn adapt(&self, source: &Gm1Dataset) -> Result<GeoreferencedFrame, String> {
        self.adapt_with_grid(
            source,
            GeoGrid {
                center_latitude_deg: T300_CENTER_LATITUDE_DEG,
                center_longitude_deg: T300_CENTER_LONGITUDE_DEG,
                nx: 300,
                ny: 300,
                nz: 75,
                dx_m: 50.0,
                dy_m: 50.0,
                model_top_m: 15_000.0,
                horizontal_crs: "local ENU centered on sealed GM1 child center",
                vertical_coordinate: "exact terrain-composed stretched scalar mass levels",
            },
        )
    }
}

/// Compatibility marker for callers that have not supplied a WRFOUT source.
/// The strict implementation is [`crate::arwen_wrfout::ArwenWrfoutAdapter`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ArwenWrfAdapterBoundary;

impl ArwenWrfAdapterBoundary {
    pub const fn adapter_id(self) -> &'static str {
        ARWEN_WRF_ADAPTER_ID
    }

    pub const fn status(self) -> &'static str {
        ARWEN_WRF_ADAPTER_STATUS
    }

    pub fn refuse(self) -> Result<GeoreferencedFrame, String> {
        Err(format!(
            "{}: a source-free boundary cannot admit data; use ArwenWrfoutSource",
            self.status()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ARWEN_WRF_ADAPTER_STATUS, ArwenWrfAdapterBoundary, GEO_FIELD_ABI, GeoGrid,
        GeoreferencedWeatherAdapter, Gm1GeoreferencedAdapter,
    };
    use crate::input::{Gm1Dataset, VolumeField, f32_sha256};
    use std::path::PathBuf;

    #[test]
    fn gm1_adapter_exposes_common_georeferenced_field_time_contract() {
        let values = vec![0.0_f32; 75 * 300 * 300];
        let dataset = Gm1Dataset {
            source_path: PathBuf::from("exact-pack"),
            artifact_kind: "exact qc+qi render pack".to_owned(),
            artifact_sha256: "1".repeat(64),
            payload_sha256: "2".repeat(64),
            contract_sha256: "3".repeat(64),
            config_sha256: "4".repeat(64),
            profile_id: "gm1-real-50m-child-v1".to_owned(),
            step: 600,
            model_time_s: 300.0,
            dt_s: 0.5,
            fields: vec![VolumeField {
                id: "cloud_condensate".to_owned(),
                label: "Cloud condensate".to_owned(),
                units: "kg kg-1".to_owned(),
                meaning: "qc + qi".to_owned(),
                source: "strict exact pack".to_owned(),
                shape: [75, 300, 300],
                values_sha256: f32_sha256(&values),
                finite_min: 0.0,
                finite_max: 0.0,
                values,
            }],
            terrain_msl_m: None,
            face_height_m: None,
            tornado_surface_fields: Vec::new(),
        };
        let frame = Gm1GeoreferencedAdapter.adapt(&dataset).unwrap();
        assert_eq!(frame.abi, GEO_FIELD_ABI);
        assert_eq!(frame.time.step, 600);
        assert_eq!(frame.grid.nx, 300);
        assert_eq!(frame.grid.dx_m, 50.0);
        assert_eq!(frame.fields[0].shape, [75, 300, 300]);

        let idealized = Gm1GeoreferencedAdapter
            .adapt_with_grid(
                &dataset,
                GeoGrid {
                    center_latitude_deg: 0.0,
                    center_longitude_deg: 0.0,
                    nx: 300,
                    ny: 300,
                    nz: 75,
                    dx_m: 50.0,
                    dy_m: 50.0,
                    model_top_m: 15_000.0,
                    horizontal_crs: "idealized Cartesian x/y; no Earth geolocation",
                    vertical_coordinate: "uniform dz=200 m",
                },
            )
            .unwrap();
        assert_eq!(
            idealized.grid.center_latitude_deg.to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(idealized.grid.vertical_coordinate, "uniform dz=200 m");
    }

    fn volume_field(id: &str, shape: [usize; 3]) -> VolumeField {
        let values = vec![0.0_f32; shape.iter().product()];
        VolumeField {
            id: id.to_owned(),
            label: id.to_owned(),
            units: "kg kg-1".to_owned(),
            meaning: "qc + qi".to_owned(),
            source: "strict exact artifact".to_owned(),
            shape,
            values_sha256: f32_sha256(&values),
            finite_min: 0.0,
            finite_max: 0.0,
            values,
        }
    }

    fn dataset_with(fields: Vec<VolumeField>) -> Gm1Dataset {
        Gm1Dataset {
            source_path: PathBuf::from("frame"),
            artifact_kind: "CM1FHS1 history".to_owned(),
            artifact_sha256: "1".repeat(64),
            payload_sha256: "2".repeat(64),
            contract_sha256: "3".repeat(64),
            config_sha256: "4".repeat(64),
            profile_id: "gm1.cartesian-history/v1".to_owned(),
            step: 10_800,
            model_time_s: 5_400.0,
            dt_s: 0.5,
            fields,
            terrain_msl_m: None,
            face_height_m: None,
            tornado_surface_fields: Vec::new(),
        }
    }

    fn cartesian_grid(nz: usize, ny: usize, nx: usize) -> GeoGrid {
        GeoGrid {
            center_latitude_deg: 0.0,
            center_longitude_deg: 0.0,
            nx,
            ny,
            nz,
            dx_m: 1_000.0,
            dy_m: 1_000.0,
            model_top_m: 20_000.0,
            horizontal_crs: "idealized Cartesian x/y; no Earth geolocation",
            vertical_coordinate: "uniform dz from admitted profile",
        }
    }

    #[test]
    fn an_arbitrary_cartesian_grid_is_admitted_when_every_field_matches_it() {
        let dataset = dataset_with(vec![
            volume_field("cloud_condensate", [40, 220, 220]),
            volume_field("qv", [40, 220, 220]),
        ]);
        let frame = Gm1GeoreferencedAdapter
            .adapt_with_grid(&dataset, cartesian_grid(40, 220, 220))
            .expect("a self-consistent Cartesian grid is admitted");
        assert_eq!(frame.grid.nx, 220);
        assert_eq!(frame.grid.nz, 40);
        assert_eq!(frame.grid.dx_m.to_bits(), 1_000.0_f64.to_bits());
        assert_eq!(frame.fields.len(), 2);
        assert_eq!(frame.fields[0].shape, vec![40, 220, 220]);
    }

    #[test]
    fn a_mixed_shape_frame_is_refused_by_the_misplaced_words_it_would_render() {
        let dataset = dataset_with(vec![
            volume_field("cloud_condensate", [40, 220, 220]),
            volume_field("qv", [40, 200, 200]),
        ]);
        let error = Gm1GeoreferencedAdapter
            .adapt_with_grid(&dataset, cartesian_grid(40, 220, 220))
            .expect_err("one frame cannot carry two grids");
        assert!(error.contains("qv"), "{error}");
        assert!(error.contains("[40, 200, 200]"), "{error}");
        assert!(
            error.contains("another column's position"),
            "{error}"
        );
    }

    #[test]
    fn a_grid_that_disagrees_with_every_field_is_refused_the_same_way() {
        let dataset = dataset_with(vec![volume_field("cloud_condensate", [40, 220, 220])]);
        let error = Gm1GeoreferencedAdapter
            .adapt_with_grid(&dataset, cartesian_grid(75, 300, 300))
            .expect_err("the declared grid must match the decoded payload");
        assert!(error.contains("[75, 300, 300]"), "{error}");
        assert!(
            error.contains("would not be the simulated state"),
            "{error}"
        );
    }

    #[test]
    fn a_zero_extent_grid_is_refused_by_the_sampling_it_would_break() {
        let dataset = dataset_with(vec![volume_field("cloud_condensate", [40, 220, 220])]);
        let error = Gm1GeoreferencedAdapter
            .adapt_with_grid(&dataset, cartesian_grid(0, 220, 220))
            .expect_err("a grid with no levels cannot be raymarched");
        assert!(error.contains("zero extent"), "{error}");
        assert!(error.contains("sample outside the payload"), "{error}");
    }

    #[test]
    fn non_positive_spacing_is_refused_by_the_wrong_distances_it_would_report() {
        let dataset = dataset_with(vec![volume_field("cloud_condensate", [40, 220, 220])]);
        for mutate in [
            (|grid: &mut GeoGrid| grid.dx_m = 0.0) as fn(&mut GeoGrid),
            |grid: &mut GeoGrid| grid.dy_m = f64::NAN,
            |grid: &mut GeoGrid| grid.model_top_m = -1.0,
        ] {
            let mut grid = cartesian_grid(40, 220, 220);
            mutate(&mut grid);
            let error = Gm1GeoreferencedAdapter
                .adapt_with_grid(&dataset, grid)
                .expect_err("undefined spacing must not size a volume");
            assert!(
                error.contains("every \n                 distance the renderer reports would be wrong")
                    || error.contains("distance the renderer reports would be wrong"),
                "{error}"
            );
        }
    }

    #[test]
    fn arwen_wrf_boundary_requires_a_source_even_after_implementation() {
        let boundary = ArwenWrfAdapterBoundary;
        assert_eq!(boundary.status(), ARWEN_WRF_ADAPTER_STATUS);
        assert!(
            boundary
                .refuse()
                .unwrap_err()
                .contains("source-free boundary")
        );
    }
}
