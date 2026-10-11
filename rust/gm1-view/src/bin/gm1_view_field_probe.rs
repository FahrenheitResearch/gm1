//! Strict CPU probe for an admitted GM1 View field's exact extrema location.

use gm1_view::input::load_artifact;
use std::{env, path::Path};

fn main() {
    if let Err(error) = run() {
        eprintln!("gm1-view-field-probe: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let input = args.next().ok_or_else(|| {
        "usage: gm1-view-field-probe INPUT.cm1fcp CONTRACT_SHA256 FIELD_ID".to_owned()
    })?;
    let contract = args
        .next()
        .ok_or_else(|| "missing CONTRACT_SHA256".to_owned())?;
    let field_id = args.next().ok_or_else(|| "missing FIELD_ID".to_owned())?;
    if args.next().is_some() {
        return Err("unexpected extra argument".to_owned());
    }
    let dataset = load_artifact(Path::new(&input), &contract)?;
    let field = dataset
        .fields
        .iter()
        .find(|field| field.id == field_id)
        .ok_or_else(|| format!("field {field_id:?} is absent"))?;
    let (flat, &maximum) = field
        .values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .ok_or_else(|| "field is empty".to_owned())?;
    let [nz, ny, nx] = field.shape;
    let k = flat / (ny * nx);
    let remainder = flat % (ny * nx);
    let j = remainder / nx;
    let i = remainder % nx;
    if k >= nz {
        return Err("computed maximum index lies outside admitted shape".to_owned());
    }
    println!(
        "field={} shape=[{nz},{ny},{nx}] maximum_zyx=[{k},{j},{i}] flat_index={flat} value={maximum:.9e} bits={:08x} field_sha256={}",
        field.id,
        maximum.to_bits(),
        field.values_sha256,
    );
    Ok(())
}
