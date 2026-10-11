//! Fail-closed image freshness gate for GM1 View development captures.

use image::ImageReader;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, env, fs, path::Path};

fn main() {
    if let Err(error) = run() {
        eprintln!("gm1-view-png-gate: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let baseline = args.next().ok_or_else(|| {
        "usage: gm1-view-png-gate BASELINE.png CANDIDATE.png MIN_CHANGED_PIXELS".to_owned()
    })?;
    let candidate = args
        .next()
        .ok_or_else(|| "missing CANDIDATE.png".to_owned())?;
    let minimum = args
        .next()
        .ok_or_else(|| "missing MIN_CHANGED_PIXELS".to_owned())?
        .parse::<usize>()
        .map_err(|error| format!("invalid MIN_CHANGED_PIXELS: {error}"))?;
    if args.next().is_some() {
        return Err("unexpected extra argument".to_owned());
    }

    let baseline_image = decode(&baseline)?;
    let candidate_image = decode(&candidate)?;
    if baseline_image.dimensions() != candidate_image.dimensions() {
        return Err("baseline and candidate dimensions differ".to_owned());
    }
    let width = baseline_image.width();
    let scene_width = width * 3 / 4;
    let mut colors = HashMap::<[u8; 3], usize>::new();
    for y in 0..candidate_image.height() {
        for x in 0..scene_width {
            let pixel = candidate_image.get_pixel(x, y).0;
            *colors.entry([pixel[0], pixel[1], pixel[2]]).or_default() += 1;
        }
    }
    let clear = colors
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(color, _)| color)
        .ok_or_else(|| "candidate scene is empty".to_owned())?;
    let mut changed = 0usize;
    let mut non_background = 0usize;
    for y in 0..baseline_image.height() {
        for x in 0..scene_width {
            let before = baseline_image.get_pixel(x, y).0;
            let after = candidate_image.get_pixel(x, y).0;
            changed += usize::from(before != after);
            let distance = after[0].abs_diff(clear[0]) as u16
                + after[1].abs_diff(clear[1]) as u16
                + after[2].abs_diff(clear[2]) as u16;
            non_background += usize::from(distance > 12);
        }
    }
    let baseline_hash = hash(Path::new(&baseline))?;
    let candidate_hash = hash(Path::new(&candidate))?;
    if baseline_hash == candidate_hash {
        return Err("candidate SHA-256 is identical to the rejected baseline".to_owned());
    }
    if changed < minimum || non_background < minimum {
        return Err(format!(
            "scene proof is too small: changed={changed}, non_background={non_background}, required={minimum}"
        ));
    }
    println!(
        "PASS changed_scene_pixels={changed} non_background_scene_pixels={non_background} dominant_background_rgb={clear:?} baseline_sha256={baseline_hash} candidate_sha256={candidate_hash}"
    );
    Ok(())
}

fn decode(path: &str) -> Result<image::RgbaImage, String> {
    ImageReader::open(path)
        .map_err(|error| format!("could not open {path}: {error}"))?
        .decode()
        .map_err(|error| format!("could not decode {path}: {error}"))
        .map(|image| image.to_rgba8())
}

fn hash(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("could not read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
