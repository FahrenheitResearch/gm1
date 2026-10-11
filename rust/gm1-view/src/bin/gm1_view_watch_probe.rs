//! CPU-only strict admission probe for one atomic GM1 View watch root.

use gm1_view::watch::load_initial_watch_dataset;
use std::{env, path::Path};

const USAGE: &str = "usage: gm1-view-watch-probe WATCH_DIR";

fn run(arguments: &[String]) -> Result<String, String> {
    let [root] = arguments else {
        return Err(USAGE.to_owned());
    };
    let dataset = load_initial_watch_dataset(Path::new(root))?;
    if dataset.fields.is_empty() {
        return Err("strictly admitted watch dataset has no volume fields".to_owned());
    }
    let fields = dataset
        .fields
        .iter()
        .map(|field| {
            format!(
                "{}:[{},{},{}]:{}",
                field.id, field.shape[0], field.shape[1], field.shape[2], field.values_sha256
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "PASS_GM1_VIEW_WATCH_ADMISSION step={} model_time_f64_bits={:016x} dt_f64_bits={:016x} artifact_kind={:?} artifact_sha256={} payload_sha256={} contract_sha256={} profile_id={} source={:?} fields={}",
        dataset.step,
        dataset.model_time_s.to_bits(),
        dataset.dt_s.to_bits(),
        dataset.artifact_kind,
        dataset.artifact_sha256,
        dataset.payload_sha256,
        dataset.contract_sha256,
        dataset.profile_id,
        dataset.source_path,
        fields,
    ))
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match run(&arguments) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("gm1-view-watch-probe: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argument_inventory_is_exact() {
        assert_eq!(run(&[]).unwrap_err(), USAGE);
        assert_eq!(
            run(&["one".to_owned(), "two".to_owned()]).unwrap_err(),
            USAGE
        );
    }
}
