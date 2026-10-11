//! Physical grid geometry for a published Cartesian history frame.
//!
//! The `.cm1fhs` envelope's embedded contract carries array shapes and
//! identities, but no `dx`/`dy`/`dz`/`ztop`. Shapes alone cannot distinguish a
//! 220 km domain from a 15 km one, so a viewer that guesses the spacing draws
//! the volume at the wrong physical size and every distance, speed, height and
//! camera framing it reports is wrong by that factor.
//!
//! The publisher writes a sibling `<stem>-publication.json` next to every
//! envelope. That document carries the run configuration, the grid geometry and
//! the same `config_sha256` the envelope's contract carries, so the geometry can
//! be bound to this exact frame instead of assumed. This module reads it and
//! refuses anything it cannot bind.

use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const FRAME_GEOMETRY_SIDECAR_SUFFIX: &str = "-publication.json";
pub const CARTESIAN_HISTORY_PUBLICATION_SCHEMA: &str = "gm1.cartesian-history-publication/v1";

/// Largest sidecar this reader will admit. The document is run metadata, not
/// payload; a file far past this size is not the sidecar the publisher writes.
const MAX_SIDECAR_BYTES: u64 = 4 * 1024 * 1024;

/// Exact physical geometry of one published Cartesian frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameGeometry {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx_m: f64,
    pub dy_m: f64,
    pub dz_m: f64,
    pub model_top_m: f64,
}

impl FrameGeometry {
    pub fn shape_zyx(self) -> [usize; 3] {
        [self.nz, self.ny, self.nx]
    }

    pub fn domain_width_m(self) -> f64 {
        self.nx as f64 * self.dx_m
    }

    pub fn domain_depth_m(self) -> f64 {
        self.ny as f64 * self.dy_m
    }

    /// Physical height of one scalar mass level above the column's floor.
    pub fn mass_level_height_m(self, k: usize) -> f64 {
        (k as f64 + 0.5) * self.dz_m
    }

    fn validate(self, source: &Path) -> Result<Self, String> {
        let source = source.display();
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(format!(
                "{source} declares a zero grid extent ({}x{}x{} as nx,ny,nz); \
                 a volume with no cells in one axis cannot be raymarched and \
                 would be drawn as an empty or infinitely thin box",
                self.nx, self.ny, self.nz
            ));
        }
        for (label, value) in [
            ("dx_m", self.dx_m),
            ("dy_m", self.dy_m),
            ("dz_m", self.dz_m),
            ("ztop_m", self.model_top_m),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "{source} declares {label}={value}; a non-finite or \
                     non-positive spacing places the volume at an undefined \
                     physical size, so every distance, speed and height the \
                     viewer reports would be wrong"
                ));
            }
        }
        let stacked_top_m = self.nz as f64 * self.dz_m;
        let disagreement = (stacked_top_m - self.model_top_m).abs();
        if disagreement > 1.0e-6 * self.model_top_m.abs().max(1.0) {
            return Err(format!(
                "{source} declares nz={} dz_m={} (stacking to {stacked_top_m} m) \
                 but ztop_m={}; the two disagree, so scalar levels would be \
                 drawn at altitudes the run never simulated",
                self.nz, self.dz_m, self.model_top_m
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Deserialize)]
struct SidecarDocument {
    schema: String,
    config_sha256: String,
    profile: SidecarProfile,
}

#[derive(Debug, Deserialize)]
struct SidecarProfile {
    profile_id: String,
    geometry: SidecarGeometry,
}

#[derive(Debug, Deserialize)]
struct SidecarGeometry {
    nx: usize,
    ny: usize,
    nz: usize,
    dx_m: f64,
    dy_m: f64,
    dz_m: f64,
    ztop_m: f64,
}

/// Sibling geometry document the publisher writes next to `input`.
pub fn sidecar_path_for(input: &Path) -> Result<PathBuf, String> {
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            format!(
                "{} has no file stem, so its sibling geometry document cannot \
                 be named; without it the domain size is unknown and the volume \
                 would be drawn at a guessed physical size",
                input.display()
            )
        })?;
    Ok(input.with_file_name(format!("{stem}{FRAME_GEOMETRY_SIDECAR_SUFFIX}")))
}

/// Geometry bound to one envelope by its embedded contract `config_sha256`.
///
/// `envelope_config_sha256` is the value the admitted envelope's own contract
/// carries. Binding on it is what makes the sidecar *this* frame's geometry
/// rather than a stale neighbour's.
pub fn load_frame_geometry(
    input: &Path,
    envelope_config_sha256: &str,
    envelope_profile_id: &str,
    admitted_shape_zyx: [usize; 3],
) -> Result<FrameGeometry, String> {
    let path = sidecar_path_for(input)?;
    let metadata = std::fs::metadata(&path).map_err(|error| {
        format!(
            "the sibling geometry document {} could not be read ({error}); the \
             envelope carries no dx/dy/dz/ztop, so without it the viewer would \
             have to guess the domain size and would draw the volume at the \
             wrong physical scale",
            path.display()
        )
    })?;
    if metadata.len() > MAX_SIDECAR_BYTES {
        return Err(format!(
            "{} is {} bytes, past the {MAX_SIDECAR_BYTES}-byte sidecar limit; \
             a document that large is not the publisher's geometry record and \
             reading it as one would bind an unrelated domain size",
            path.display(),
            metadata.len()
        ));
    }
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("{} could not be read: {error}", path.display()))?;
    let document: SidecarDocument = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "{} is not a readable publication document ({error}); its geometry \
             cannot be bound and the volume would be drawn at a guessed size",
            path.display()
        )
    })?;
    if document.schema != CARTESIAN_HISTORY_PUBLICATION_SCHEMA {
        return Err(format!(
            "{} declares schema {} rather than {CARTESIAN_HISTORY_PUBLICATION_SCHEMA}; \
             an unrecognized schema's geometry fields may not mean what this \
             reader assumes, so the domain size it produced could be wrong",
            path.display(),
            document.schema
        ));
    }
    if !document
        .config_sha256
        .eq_ignore_ascii_case(envelope_config_sha256)
    {
        return Err(format!(
            "{} records config_sha256={} but the admitted envelope's contract \
             carries {envelope_config_sha256}; the sidecar describes a different \
             run configuration, so binding its geometry would size this frame's \
             volume by another run's domain",
            path.display(),
            document.config_sha256
        ));
    }
    if !envelope_profile_id.is_empty() && document.profile.profile_id != envelope_profile_id {
        return Err(format!(
            "{} records profile_id={} but the admitted envelope's contract \
             carries {envelope_profile_id}; the sidecar describes a different \
             publication profile, so its geometry is not this frame's",
            path.display(),
            document.profile.profile_id
        ));
    }
    let geometry = FrameGeometry {
        nx: document.profile.geometry.nx,
        ny: document.profile.geometry.ny,
        nz: document.profile.geometry.nz,
        dx_m: document.profile.geometry.dx_m,
        dy_m: document.profile.geometry.dy_m,
        dz_m: document.profile.geometry.dz_m,
        model_top_m: document.profile.geometry.ztop_m,
    }
    .validate(&path)?;
    if geometry.shape_zyx() != admitted_shape_zyx {
        return Err(format!(
            "{} declares grid {:?} but the admitted envelope decoded fields of \
             shape {admitted_shape_zyx:?} (nz,ny,nx); indexing the payload with \
             the sidecar's strides would place one column's words at another \
             column's position, so the displayed volume would not be the \
             simulated state",
            path.display(),
            geometry.shape_zyx()
        ));
    }
    Ok(geometry)
}

#[cfg(test)]
mod tests {
    use super::{CARTESIAN_HISTORY_PUBLICATION_SCHEMA, load_frame_geometry, sidecar_path_for};
    use std::path::{Path, PathBuf};

    fn scratch(name: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!("gm1-view-frame-geometry-{name}"));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).expect("scratch directory");
        directory
    }

    fn write_sidecar(directory: &Path, body: &str) -> PathBuf {
        let envelope = directory.join("history.cm1fhs");
        std::fs::write(directory.join("history-publication.json"), body).expect("sidecar write");
        envelope
    }

    fn sidecar_json(schema: &str, config_sha256: &str, nz: usize, dz_m: f64, ztop_m: f64) -> String {
        format!(
            r#"{{"schema":"{schema}","config_sha256":"{config_sha256}",
               "profile":{{"profile_id":"gm1.cartesian-history/v1",
               "geometry":{{"nx":220,"ny":220,"nz":{nz},"dx_m":1000.0,
               "dy_m":1000.0,"dz_m":{dz_m},"ztop_m":{ztop_m}}}}}}}"#
        )
    }

    #[test]
    fn sidecar_path_is_the_envelope_stem_plus_the_publication_suffix() {
        assert_eq!(
            sidecar_path_for(Path::new("/frames/frame-000045/history.cm1fhs")).unwrap(),
            PathBuf::from("/frames/frame-000045/history-publication.json")
        );
    }

    #[test]
    fn geometry_binds_when_the_sidecar_matches_the_envelope_configuration() {
        let directory = scratch("bind");
        let envelope = write_sidecar(
            &directory,
            &sidecar_json(
                CARTESIAN_HISTORY_PUBLICATION_SCHEMA,
                &"b2".repeat(32),
                40,
                500.0,
                20_000.0,
            ),
        );
        let geometry = load_frame_geometry(
            &envelope,
            &"B2".repeat(32),
            "gm1.cartesian-history/v1",
            [40, 220, 220],
        )
        .expect("matching sidecar binds");
        assert_eq!(geometry.shape_zyx(), [40, 220, 220]);
        assert_eq!(geometry.dx_m.to_bits(), 1000.0_f64.to_bits());
        assert_eq!(geometry.domain_width_m().to_bits(), 220_000.0_f64.to_bits());
        assert_eq!(geometry.model_top_m.to_bits(), 20_000.0_f64.to_bits());
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_different_run_configuration_is_refused_by_the_wrong_domain_size_it_would_bind() {
        let directory = scratch("config-mismatch");
        let envelope = write_sidecar(
            &directory,
            &sidecar_json(
                CARTESIAN_HISTORY_PUBLICATION_SCHEMA,
                &"aa".repeat(32),
                40,
                500.0,
                20_000.0,
            ),
        );
        let error = load_frame_geometry(
            &envelope,
            &"bb".repeat(32),
            "gm1.cartesian-history/v1",
            [40, 220, 220],
        )
        .expect_err("a foreign configuration must not size this frame");
        assert!(error.contains("another run's domain"), "{error}");
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_missing_sidecar_is_refused_by_the_guessed_scale_it_would_force() {
        let directory = scratch("missing");
        let envelope = directory.join("history.cm1fhs");
        let error = load_frame_geometry(
            &envelope,
            &"cc".repeat(32),
            "gm1.cartesian-history/v1",
            [40, 220, 220],
        )
        .expect_err("a frame with no geometry document must not be sized by guess");
        assert!(error.contains("wrong physical scale"), "{error}");
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_vertical_stack_that_disagrees_with_the_model_top_is_refused() {
        let directory = scratch("ztop-mismatch");
        let envelope = write_sidecar(
            &directory,
            &sidecar_json(
                CARTESIAN_HISTORY_PUBLICATION_SCHEMA,
                &"dd".repeat(32),
                40,
                500.0,
                15_000.0,
            ),
        );
        let error = load_frame_geometry(
            &envelope,
            &"dd".repeat(32),
            "gm1.cartesian-history/v1",
            [40, 220, 220],
        )
        .expect_err("nz*dz must reproduce the declared model top");
        assert!(error.contains("altitudes the run never simulated"), "{error}");
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_sidecar_grid_that_differs_from_the_decoded_payload_is_refused() {
        let directory = scratch("shape-mismatch");
        let envelope = write_sidecar(
            &directory,
            &sidecar_json(
                CARTESIAN_HISTORY_PUBLICATION_SCHEMA,
                &"ee".repeat(32),
                40,
                500.0,
                20_000.0,
            ),
        );
        let error = load_frame_geometry(
            &envelope,
            &"ee".repeat(32),
            "gm1.cartesian-history/v1",
            [40, 200, 200],
        )
        .expect_err("sidecar strides must match the decoded payload");
        assert!(
            error.contains("another column's position"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn an_unrecognized_schema_is_refused() {
        let directory = scratch("schema");
        let envelope = write_sidecar(
            &directory,
            &sidecar_json("some.other-publication/v9", &"ff".repeat(32), 40, 500.0, 20_000.0),
        );
        let error = load_frame_geometry(
            &envelope,
            &"ff".repeat(32),
            "gm1.cartesian-history/v1",
            [40, 220, 220],
        )
        .expect_err("an unknown schema must not be read as geometry");
        assert!(error.contains("could be wrong"), "{error}");
        let _ = std::fs::remove_dir_all(&directory);
    }
}
