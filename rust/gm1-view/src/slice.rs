//! Deterministic CPU preparation for the dockable scientific slice.

use cm1_state::nws_reflectivity::{NWS_DBZ_CLASSES, NWS_DBZ_FLOOR, nws_dbz_rgb};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub from: [f32; 2],
    pub to: [f32; 2],
}

pub fn quantile_range(values: &[f32]) -> Result<(f32, f32), String> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Err("slice contains no finite values".to_owned());
    }
    finite.sort_by(f32::total_cmp);
    let lo = finite[finite.len() / 100];
    let hi = finite[(finite.len() * 99 / 100).min(finite.len() - 1)];
    if hi > lo {
        Ok((lo, hi))
    } else {
        Ok((lo, lo + f32::EPSILON.max(lo.abs() * f32::EPSILON)))
    }
}

/// The published NWS dBZ class scale, for a slice whose units are dBZ.
///
/// Reflectivity is the one field in this viewer a meteorologist already has a
/// scale for. Drawing it with the generic sequential ramp would make a 50 dBZ
/// core and a 30 dBZ shield differ only in brightness, and would move the
/// class boundaries with every frame's own quantiles. The classes are fixed,
/// published, and read the same way on every frame; below the scale's floor
/// the product shows no echo, which is left transparent so the viewer's own
/// ground shows through.
pub fn nws_dbz_rgba(values: &[f32]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(values.len() * 4);
    for value in values {
        if !value.is_finite() || *value < NWS_DBZ_FLOOR {
            rgba.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            let rgb = nws_dbz_rgb(*value, [0, 0, 0]);
            rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
        }
    }
    rgba
}

/// The fixed display range of [`nws_dbz_rgba`], as the scale publishes it.
pub fn nws_dbz_range() -> (f32, f32) {
    (NWS_DBZ_FLOOR, NWS_DBZ_CLASSES[NWS_DBZ_CLASSES.len() - 1].0)
}

/// Whether a field carries reflectivity, by the units it declares.
pub fn is_reflectivity_units(units: &str) -> bool {
    units.eq_ignore_ascii_case("dbz")
}

pub fn scientific_rgba(values: &[f32], lo: f32, hi: f32) -> Vec<u8> {
    let span = (hi - lo).max(f32::MIN_POSITIVE);
    let mut rgba = Vec::with_capacity(values.len() * 4);
    for value in values {
        let t = if value.is_finite() {
            ((*value - lo) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let color = palette(t);
        rgba.extend_from_slice(&color);
    }
    rgba
}

fn palette(t: f32) -> [u8; 4] {
    let stops = [
        [20.0, 31.0, 48.0],
        [33.0, 97.0, 140.0],
        [42.0, 161.0, 152.0],
        [214.0, 202.0, 122.0],
        [224.0, 112.0, 66.0],
        [132.0, 39.0, 62.0],
    ];
    let x = t * (stops.len() - 1) as f32;
    let left = x.floor() as usize;
    let right = (left + 1).min(stops.len() - 1);
    let f = x - left as f32;
    let mut result = [0_u8; 4];
    for channel in 0..3 {
        result[channel] =
            (stops[left][channel] * (1.0 - f) + stops[right][channel] * f).round() as u8;
    }
    result[3] = 255;
    result
}

pub fn contours(
    values: &[f32],
    width: usize,
    height: usize,
    level: f32,
) -> Result<Vec<Segment>, String> {
    if width < 2 || height < 2 || values.len() != width * height {
        return Err("contour grid requires width,height >= 2 and width*height values".to_owned());
    }
    let mut segments = Vec::new();
    for y in 0..height - 1 {
        for x in 0..width - 1 {
            let v = [
                values[y * width + x],
                values[y * width + x + 1],
                values[(y + 1) * width + x + 1],
                values[(y + 1) * width + x],
            ];
            if !v.iter().all(|value| value.is_finite()) {
                continue;
            }
            let corners = [
                [x as f32, y as f32],
                [(x + 1) as f32, y as f32],
                [(x + 1) as f32, (y + 1) as f32],
                [x as f32, (y + 1) as f32],
            ];
            let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
            let mut hits = Vec::with_capacity(4);
            for (a, b) in edges {
                if (v[a] < level) == (v[b] < level) || v[a] == v[b] {
                    continue;
                }
                let fraction = ((level - v[a]) / (v[b] - v[a])).clamp(0.0, 1.0);
                hits.push([
                    corners[a][0] + fraction * (corners[b][0] - corners[a][0]),
                    corners[a][1] + fraction * (corners[b][1] - corners[a][1]),
                ]);
            }
            if hits.len() == 2 {
                segments.push(Segment {
                    from: hits[0],
                    to: hits[1],
                });
            } else if hits.len() == 4 {
                segments.push(Segment {
                    from: hits[0],
                    to: hits[1],
                });
                segments.push(Segment {
                    from: hits[2],
                    to: hits[3],
                });
            }
        }
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::{
        contours, is_reflectivity_units, nws_dbz_range, nws_dbz_rgba, scientific_rgba,
    };

    #[test]
    fn a_reflectivity_slice_is_painted_with_the_published_nws_classes() {
        // Fixed classes, and clear air left clear: a 50 dBZ core reads as a
        // 50 dBZ core on every frame, not as this frame's brightest quantile.
        assert!(is_reflectivity_units("dBZ"));
        assert!(is_reflectivity_units("dbz"));
        assert!(!is_reflectivity_units("kg kg-1"));
        assert_eq!(nws_dbz_range(), (5.0, 75.0));
        let rgba = nws_dbz_rgba(&[-35.0, 22.0, 52.0, f32::NAN, 120.0]);
        assert_eq!(rgba.len(), 20);
        assert_eq!(&rgba[..4], &[0, 0, 0, 0]);
        assert_eq!(&rgba[4..8], &[0x02, 0xFD, 0x02, 255]);
        assert_eq!(&rgba[8..12], &[0xFD, 0x00, 0x00, 255]);
        // A non-finite word is a defect, not clear air, but it is also not a
        // class; the slice leaves it out of the product.
        assert_eq!(&rgba[12..16], &[0, 0, 0, 0]);
        // The top class holds above its edge.
        assert_eq!(&rgba[16..], &[0xFD, 0xFD, 0xFD, 255]);
    }

    #[test]
    fn pseudocolor_is_light_ui_safe_and_deterministic() {
        let rgba = scientific_rgba(&[0.0, 0.5, 1.0], 0.0, 1.0);
        assert_eq!(rgba.len(), 12);
        assert_eq!(&rgba[..4], &[20, 31, 48, 255]);
        assert_eq!(&rgba[8..], &[132, 39, 62, 255]);
    }

    #[test]
    fn contour_crosses_a_discriminating_cell() {
        let segments = contours(&[0.0, 1.0, 1.0, 0.0], 2, 2, 0.5).unwrap();
        assert_eq!(segments.len(), 2);
    }
}
