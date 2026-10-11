//! The standard NWS dBZ colour scale, as one table both renderers read.
//!
//! Simulated reflectivity is only useful if it reads like radar. A continuous
//! heat ramp says "this number is bigger than that one"; the NWS scale says
//! "that is a 50 dBZ core with a 60 dBZ hail signature inside it", because
//! every meteorologist already knows what its greens, yellows and reds mean
//! and reads the boundaries between them as physical thresholds.
//!
//! This is the sixteen-class NEXRAD Level III reflectivity legend the US
//! National Weather Service publishes its base- and composite-reflectivity
//! products under: 5 dBZ per class from 5 dBZ to 75 dBZ, with everything
//! below 5 dBZ shown as no echo rather than as a colour. It is a *discrete*
//! scale by design -- the classes are the product, and interpolating between
//! them would report a precision the radar product does not have and would
//! blur the class boundary a forecaster reads the storm by.
//!
//! The scale is a display convention, not model output. It is applied to
//! whatever dBZ words it is handed; it never rescales, floors or derives
//! them. The reflectivity itself comes from the microphysics scheme that owns
//! the size distributions it depends on (CM1 r21.1 README.namelist:1366-1369
//! `output_dbz`; the composite is `misclibs.F:2259-2292`).

/// Class width of the published scale, in dBZ.
pub const NWS_DBZ_CLASS_WIDTH: f32 = 5.0;

/// Lowest reflectivity the scale gives a colour to, in dBZ. Below this the
/// product shows no echo, so a renderer keeps its own background there rather
/// than painting a colour that would read as light rain.
pub const NWS_DBZ_FLOOR: f32 = 5.0;

/// The sixteen NWS reflectivity classes: the inclusive lower edge of each
/// class in dBZ, and the class colour.
pub const NWS_DBZ_CLASSES: [(f32, [u8; 3]); 15] = [
    (5.0, [0x04, 0xE9, 0xE7]),
    (10.0, [0x01, 0x9F, 0xF4]),
    (15.0, [0x03, 0x00, 0xF4]),
    (20.0, [0x02, 0xFD, 0x02]),
    (25.0, [0x01, 0xC5, 0x01]),
    (30.0, [0x00, 0x8E, 0x00]),
    (35.0, [0xFD, 0xF8, 0x02]),
    (40.0, [0xE5, 0xBC, 0x00]),
    (45.0, [0xFD, 0x95, 0x00]),
    (50.0, [0xFD, 0x00, 0x00]),
    (55.0, [0xD4, 0x00, 0x00]),
    (60.0, [0xBC, 0x00, 0x00]),
    (65.0, [0xF8, 0x00, 0xFD]),
    (70.0, [0x98, 0x54, 0xC6]),
    (75.0, [0xFD, 0xFD, 0xFD]),
];

/// The class colour for one reflectivity value, or `None` for no echo.
///
/// `None` is returned for anything below [`NWS_DBZ_FLOOR`] and for a
/// non-finite word. A non-finite word is not "no echo" -- it is a defect --
/// so the caller decides what to show for it; painting it as the weakest
/// class here would hide it inside the product.
pub fn nws_dbz_class(dbz: f32) -> Option<[u8; 3]> {
    if !dbz.is_finite() || dbz < NWS_DBZ_FLOOR {
        return None;
    }
    let mut chosen = NWS_DBZ_CLASSES[0].1;
    for (lower, colour) in NWS_DBZ_CLASSES {
        if dbz >= lower {
            chosen = colour;
        }
    }
    Some(chosen)
}

/// The class colour, or `background` where the product shows no echo.
pub fn nws_dbz_rgb(dbz: f32, background: [u8; 3]) -> [u8; 3] {
    nws_dbz_class(dbz).unwrap_or(background)
}

/// The class index a value falls in, or `None` for no echo. Index 0 is the
/// 5 dBZ class; the last index is the 75 dBZ class and everything above it.
pub fn nws_dbz_class_index(dbz: f32) -> Option<usize> {
    if !dbz.is_finite() || dbz < NWS_DBZ_FLOOR {
        return None;
    }
    let mut chosen = 0usize;
    for (index, (lower, _)) in NWS_DBZ_CLASSES.iter().enumerate() {
        if dbz >= *lower {
            chosen = index;
        }
    }
    Some(chosen)
}

/// The legend a renderer draws beside the plane: one `(dBZ, colour)` pair per
/// class, in ascending order.
pub fn nws_dbz_legend() -> &'static [(f32, [u8; 3])] {
    &NWS_DBZ_CLASSES
}
