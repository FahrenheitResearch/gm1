//! FP32 Spherical Harmonic Exponentials core for the Cinema material path.
//!
//! This is a source-order port of the MIT-licensed upstream HLSL pinned by
//! [`UPSTREAM_COMMIT`].  It is distinct from SimSat's positive log-SH3 sky and
//! from the cloud-volume scattering path.

use bevy::math::Vec3;

pub const UPSTREAM_PROJECT: &str = "Avelina9X/SphericalHarmonicExponentials";
pub const UPSTREAM_COMMIT: &str = "3c14ddd5709705eebe00ec1305c891bc5f7d038b";
pub const UPSTREAM_LICENSE: &str = "MIT";
pub const UPSTREAM_COMMON_SH_SHA256: &str =
    "2ffeb4fcc001e57ba61b3f825177d70ed6283ebd5b710c39204acc16460aa56c";
pub const UPSTREAM_SHADE32_SHA256: &str =
    "3a4cf0fce13db21fc2a53a4a8affc09d5f4a750a4e3b89afa89d502f500356d1";
pub const WGSL_SOURCE: &str = include_str!("she_fp32.wgsl");

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SheDiffuseCoefficients(pub [Vec3; 9]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SheSpecularCoefficients(pub [Vec3; 33]);

/// HLSL `ComputeSHBasisL2`, flattened as c_1_4, c_5_8, c_0.
pub fn compute_sh_basis_l2(normal: Vec3) -> [f32; 9] {
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    [
        0.488603 * x,
        0.488603 * z,
        0.488603 * y,
        0.546274 * (x2 - y2),
        1.092548 * x * z,
        0.946176 * z2 - 0.315392,
        1.092548 * y * z,
        1.092548 * x * y,
        0.282095,
    ]
}

/// HLSL `ComputeSHBasisL4`, flattened as c_1_4 through c_21_24, then c_0.
pub fn compute_sh_basis_l4(normal: Vec3) -> [f32; 25] {
    let x = normal.x;
    let y = normal.y;
    let z = normal.z;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    [
        0.488603 * x,
        0.488603 * z,
        0.488603 * y,
        0.546274 * (x2 - y2),
        1.092548 * x * z,
        0.946176 * z2 - 0.315392,
        1.092548 * y * z,
        1.092548 * x * y,
        0.590044 * x * (x2 - 3.0 * y2),
        1.445306 * z * (x2 - y2),
        0.457046 * x * (5.0 * z2 - 1.0),
        0.373176 * z * (5.0 * z2 - 3.0),
        0.457046 * y * (5.0 * z2 - 1.0),
        2.890611 * x * y * z,
        0.590044 * y * (3.0 * x2 - y2),
        0.625836 * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2)),
        1.770131 * x * z * (x2 - 3.0 * y2),
        0.473087 * (x2 - y2) * (7.0 * z2 - 1.0),
        0.669047 * x * z * (7.0 * z2 - 3.0),
        0.105786 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0),
        0.669047 * y * z * (7.0 * z2 - 3.0),
        0.946174 * x * y * (7.0 * z2 - 1.0),
        1.770131 * y * z * (3.0 * x2 - y2),
        2.503343 * x * y * (x2 - y2),
        0.282095,
    ]
}

/// HLSL `ApplyVonMisesFisher` with its exact band factors.
pub fn apply_von_mises_fisher(p: &mut [f32; 25], q: &mut [f32; 9], alpha: f32) {
    let l1 = (-alpha).exp();
    let l2 = (-3.0 * alpha).exp();
    let l3 = (-6.0 * alpha).exp();
    let l4 = (-10.0 * alpha).exp();
    for value in &mut p[0..3] {
        *value *= l1;
    }
    p[3] *= l2;
    for value in &mut p[4..8] {
        *value *= l2;
    }
    for value in &mut p[8..15] {
        *value *= l3;
    }
    for value in &mut p[15..24] {
        *value *= l4;
    }
    for value in &mut q[0..3] {
        *value *= l1;
    }
    q[3] *= l2;
    for value in &mut q[4..8] {
        *value *= l2;
    }
}

/// HLSL `ConstructSHVectorA`; callers must supply normalized N/V with N dot V > 0.
pub fn construct_sh_vector_a(normal: Vec3, view: Vec3, alpha: f32) -> Result<[f32; 33], String> {
    if !normal.is_finite()
        || !view.is_finite()
        || !alpha.is_finite()
        || alpha < 0.0
        || (normal.length_squared() - 1.0).abs() > 2.0e-5
        || (view.length_squared() - 1.0).abs() > 2.0e-5
        || normal.dot(view) <= 0.0
    {
        return Err("SHE vector requires normalized finite N/V, NdotV>0, and alpha>=0".to_owned());
    }
    let reflected = (-view) - 2.0 * (-view).dot(normal) * normal;
    let half = (normal + reflected).normalize();
    if !half.is_finite() {
        return Err("SHE half-vector is degenerate".to_owned());
    }
    let mut p = compute_sh_basis_l4(reflected);
    let mut q = compute_sh_basis_l2(half);
    apply_von_mises_fisher(&mut p, &mut q, alpha);
    let mut output = [0.0_f32; 33];
    output[..24].copy_from_slice(&p[..24]);
    output[24..32].copy_from_slice(&q[..8]);
    output[32] = p[24];
    Ok(output)
}

pub fn reconstruct_diffuse(normal: Vec3, coefficients: &SheDiffuseCoefficients) -> Vec3 {
    let basis = compute_sh_basis_l2(normal);
    basis
        .iter()
        .zip(coefficients.0.iter())
        .fold(Vec3::ZERO, |sum, (weight, coefficient)| {
            sum + *weight * *coefficient
        })
}

/// FP32 Spherical Harmonic Exponentials reconstruction used by upstream Shade32.
pub fn reconstruct_specular(
    normal: Vec3,
    view: Vec3,
    roughness: f32,
    coefficients: &SheSpecularCoefficients,
) -> Result<Vec3, String> {
    let a = construct_sh_vector_a(normal, view, roughness * roughness)?;
    let log_radiance = a
        .iter()
        .zip(coefficients.0.iter())
        .fold(Vec3::ZERO, |sum, (weight, coefficient)| {
            sum + *weight * *coefficient
        });
    Ok(log_radiance.exp())
}

#[cfg(test)]
mod tests {
    use super::{
        SheDiffuseCoefficients, SheSpecularCoefficients, WGSL_SOURCE, compute_sh_basis_l2,
        construct_sh_vector_a, reconstruct_diffuse, reconstruct_specular,
    };
    use bevy::math::Vec3;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    #[test]
    fn l2_basis_preserves_upstream_axis_order() {
        let basis = compute_sh_basis_l2(Vec3::Z);
        assert_eq!(basis[0].to_bits(), 0);
        assert_eq!(basis[1].to_bits(), 0.488603_f32.to_bits());
        assert_eq!(basis[2].to_bits(), 0);
        assert_eq!(basis[5].to_bits(), (0.946176_f32 - 0.315392_f32).to_bits());
        assert_eq!(basis[8].to_bits(), 0.282095_f32.to_bits());
    }

    #[test]
    fn fp32_diffuse_and_exponential_specular_are_finite() {
        let diffuse = SheDiffuseCoefficients(std::array::from_fn(|index| {
            Vec3::splat((index as f32 + 1.0) * 0.01)
        }));
        let specular = SheSpecularCoefficients(std::array::from_fn(|index| {
            Vec3::new(
                index as f32 * 0.001,
                index as f32 * -0.0005,
                index as f32 * 0.00025,
            )
        }));
        let n = Vec3::new(0.3, 0.4, 0.8660254).normalize();
        let v = Vec3::new(-0.2, 0.35, 0.915).normalize();
        let d = reconstruct_diffuse(n, &diffuse);
        let s = reconstruct_specular(n, v, 0.42, &specular).expect("specular");
        assert!(d.is_finite());
        assert!(s.is_finite());
        assert!(s.cmpgt(Vec3::ZERO).all());
    }

    #[test]
    fn held_directions_cover_all_33_fp32_terms() {
        for index in 0..257_u32 {
            let azimuth = index as f32 * 2.399_963_1;
            let z = 0.15 + 0.84 * ((index * 37 % 257) as f32 / 256.0);
            let radius = (1.0 - z * z).sqrt();
            let normal = Vec3::new(radius * azimuth.cos(), z, radius * azimuth.sin()).normalize();
            let view = (normal + Vec3::new(0.13, 0.31, -0.19)).normalize();
            let a =
                construct_sh_vector_a(normal, view, index as f32 / 256.0).expect("held direction");
            assert!(a.iter().all(|value| value.is_finite()));
            assert_eq!(a[32].to_bits(), 0.282095_f32.to_bits());
        }
    }

    #[test]
    fn wgsl_fp32_compute_oracle_is_well_formed() {
        let module = naga::front::wgsl::parse_str(WGSL_SOURCE).expect("parse SHE WGSL");
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .expect("validate SHE WGSL");
    }
}
