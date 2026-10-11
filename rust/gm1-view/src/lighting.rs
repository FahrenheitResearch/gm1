//! Positive log-SH3 diffuse sky irradiance for cloud lighting.
//!
//! The basis/evaluation ordering is adapted from the SimSat project's positive
//! log-SH3 implementation (third_party/SimSat-PROVENANCE.md), distributed here
//! under GM1's Apache-2.0 licence. Direct solar irradiance remains a separate term.

pub const SH3_COUNT: usize = 16;
pub const LOG_EXP_LIMIT: f64 = 80.0;
pub const SIMSAT_PROVENANCE: &str = "SimSat positive log-SH3 sky irradiance (the SimSat project)";
pub const SIMSAT_SOURCE_REVISION: &str = "as adapted for GM1 View";
pub const SIMSAT_ATMOSPHERE_SHA256: &str =
    "7330c3d3a82124e3a46ab53fd8341e89463132a64ae575f6c2591f2fb1c4433b";
pub const SIMSAT_SOLAR_SHA256: &str =
    "35af97fc6e094608942c68cb4f5261dbb7716a0d31ac09019b3dce03d62b4c31";
pub const T300_SKY_ROW_SHA256: &str =
    "0e29c568b6f2a22b6f79f072d991a9db60e86fb0a8584e3dde9e36be3c24d703";
pub const T300_VALID_TIME_UTC: &str = "2024-05-21T21:05:00Z";
pub const T300_CENTER_LATITUDE_DEG: f64 = 41.394_640_411_35;
pub const T300_CENTER_LONGITUDE_DEG: f64 = -94.282_682_936_87;
// Preserve the independently derived solar geometry decimal pins verbatim.
#[allow(clippy::excessive_precision)]
pub const T300_SOLAR_ELEVATION_DEG: f64 = 48.305_803_125_363_973;
#[allow(clippy::excessive_precision)]
pub const T300_SOLAR_AZIMUTH_DEG: f64 = 253.137_338_192_791_987;

/// SimSat `AtmosphereLuts::build(AtmosphereParams::default())` followed by
/// `SkyShTable::build(..., 48).at(T300_SOLAR_ELEVATION_DEG)`, packed as RGB f32.
/// The first 16 triplets are direct log-SH3 irradiance coefficients; the last is
/// the explicit positive amplitude. These are sealed generated words, not a fit
/// authored inside GM1 View.
pub const T300_SKY_ROW_F32_BITS: [[u32; 3]; SH3_COUNT + 1] = [
    [0xbecf0da6, 0xbef0ad25, 0xbf1bc652],
    [0xba242f59, 0xba793e8f, 0xba9d0144],
    [0x3ece1289, 0x3ef867e4, 0x3f25ae2a],
    [0x3e569396, 0x3e3283e3, 0x3e1b1547],
    [0x3adca185, 0x3b006855, 0x3b1abd31],
    [0x3a06ff4b, 0x3a566f11, 0x3a8a7317],
    [0xbdabada7, 0xbd779cba, 0xbc90d95b],
    [0x3d03b64e, 0x3cce3642, 0x3ca12a9a],
    [0x3d629edb, 0x3d37f48f, 0x3d0aa2b5],
    [0xbad6858e, 0xbae6f1e0, 0xbb03c274],
    [0xba8fd144, 0xbaab5495, 0xbad32e34],
    [0xba3e6726, 0xba261551, 0xb9f6b409],
    [0xbe0004d7, 0xbde8fd8f, 0xbdc3f143],
    [0x3c3babae, 0x3c053929, 0x3b94888b],
    [0x3b33dae6, 0x3b3d4730, 0x3ada6ddf],
    [0xbb951232, 0xbb447ba0, 0xbb0ca695],
    [0x40a333d3, 0x4128866c, 0x41bcb66e],
];

/// Exact NOAA/Meeus SimSat solar vector for T+300 in local ENU order.
pub const T300_SUN_ENU_F32_BITS: [u32; 3] = [0xbf22f544, 0xbe4593f0, 0x3f3f2818];

/// SimSat default-atmosphere direct-normal RGB irradiance at T+300 in W m-2.
pub const T300_DIRECT_IRRADIANCE_F32_BITS: [u32; 3] = [0x431a9abe, 0x43110b32, 0x4301c0ab];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositiveSky {
    pub log_coef: [[f64; 3]; SH3_COUNT],
    pub amplitude: [f64; 3],
}

impl PositiveSky {
    /// The sealed, physically generated SimSat row for the real T+300 frame.
    pub fn t300_physical() -> Self {
        let log_coef = std::array::from_fn(|index| {
            T300_SKY_ROW_F32_BITS[index].map(|bits| f64::from(f32::from_bits(bits)))
        });
        Self {
            log_coef,
            amplitude: T300_SKY_ROW_F32_BITS[SH3_COUNT].map(|bits| f64::from(f32::from_bits(bits))),
        }
    }

    pub fn irradiance(self, direction: [f64; 3]) -> [f64; 3] {
        let basis = sh3_basis(normalize(direction));
        let mut log_value = [0.0; 3];
        for (coefficient, basis_value) in self.log_coef.iter().zip(basis) {
            for (channel, value) in log_value.iter_mut().enumerate() {
                *value += coefficient[channel] * basis_value;
            }
        }
        let mut output = [0.0; 3];
        for channel in 0..3 {
            let amplitude = self.amplitude[channel];
            if amplitude > 0.0 && amplitude.is_finite() {
                output[channel] = amplitude
                    * log_value[channel]
                        .clamp(-LOG_EXP_LIMIT, LOG_EXP_LIMIT)
                        .exp();
            }
        }
        output
    }
}

pub fn sh3_basis(direction: [f64; 3]) -> [f64; SH3_COUNT] {
    let [x, y, z] = direction;
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    [
        0.282094791773878,
        0.488602511902920 * y,
        0.488602511902920 * z,
        0.488602511902920 * x,
        1.092548430592079 * x * y,
        1.092548430592079 * y * z,
        0.315391565252520 * (3.0 * z2 - 1.0),
        1.092548430592079 * x * z,
        0.546274215296040 * (x2 - y2),
        0.590043589926644 * y * (3.0 * x2 - y2),
        2.890611442640554 * x * y * z,
        0.457045799464466 * y * (5.0 * z2 - 1.0),
        0.373176332590115 * z * (5.0 * z2 - 3.0),
        0.457045799464466 * x * (5.0 * z2 - 1.0),
        1.445305721320277 * z * (x2 - y2),
        0.590043589926644 * x * (x2 - 3.0 * y2),
    ]
}

fn normalize(direction: [f64; 3]) -> [f64; 3] {
    let length = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    assert!(length.is_finite() && length > 0.0);
    [
        direction[0] / length,
        direction[1] / length,
        direction[2] / length,
    ]
}

#[cfg(test)]
mod tests {
    use super::{PositiveSky, SH3_COUNT, T300_SKY_ROW_F32_BITS, T300_SKY_ROW_SHA256, sh3_basis};
    use sha2::{Digest, Sha256};

    fn wgsl_mirror(sky: PositiveSky, direction: [f32; 3]) -> [f32; 3] {
        let length = direction
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let direction = [
            direction[0] / length,
            direction[1] / length,
            direction[2] / length,
        ];
        let basis = sh3_basis([
            f64::from(direction[0]),
            f64::from(direction[1]),
            f64::from(direction[2]),
        ]);
        let mut log_value = [0.0_f32; 3];
        for (index, basis_value) in basis.into_iter().enumerate().take(SH3_COUNT) {
            for (channel, value) in log_value.iter_mut().enumerate() {
                *value += sky.log_coef[index][channel] as f32 * basis_value as f32;
            }
        }
        std::array::from_fn(|channel| {
            sky.amplitude[channel] as f32 * log_value[channel].clamp(-80.0, 80.0).exp()
        })
    }

    #[test]
    fn positive_sky_matches_wgsl_order_on_held_directions() {
        let sky = PositiveSky::t300_physical();
        for direction in [
            [0.0_f32, 0.0, 1.0],
            [0.6, 0.2, 0.77],
            [-0.8, 0.1, 0.59],
            [0.1, -0.95, 0.3],
        ] {
            let reference = sky.irradiance(direction.map(f64::from));
            let gpu = wgsl_mirror(sky, direction);
            for channel in 0..3 {
                let relative =
                    ((f64::from(gpu[channel]) - reference[channel]) / reference[channel]).abs();
                assert!(
                    relative < 8.0e-7,
                    "held {direction:?} channel {channel}: {relative}"
                );
                assert!(gpu[channel] > 0.0);
            }
        }
    }

    #[test]
    fn physical_simsat_row_matches_sealed_hash() {
        let bytes = T300_SKY_ROW_F32_BITS
            .iter()
            .flatten()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), T300_SKY_ROW_SHA256);
        assert!(
            PositiveSky::t300_physical()
                .amplitude
                .iter()
                .all(|value| *value > 0.0)
        );
    }
}
