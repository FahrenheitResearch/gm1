//! CPU-only strict admission and exact-f32 A/B playback probe.

use gm1_view::production_playback::{
    EXACT_F32_BLEND_FORMULA, ProductionFramePair, ProductionLoadEvent, ProductionPairLoader,
    load_production_frame_pair, load_production_pack, scan_production_series,
};
use std::{env, path::Path, time::Duration};

const USAGE: &str = "usage: gm1-production-playback-probe PACK_A | PACK_A PACK_B ALPHA | --series SERIES_ROOT ALPHA";

fn frame_inventory(frame: &gm1_view::production_playback::ExactProductionFrame) -> String {
    frame
        .fields
        .iter()
        .map(|field| {
            format!(
                "{}:[{}]:{}",
                field.name,
                field
                    .shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                field.values_sha256
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn run(arguments: &[String]) -> Result<String, String> {
    match arguments {
        [pack] => {
            let frame = load_production_pack(Path::new(pack))?;
            let residency = frame.renderer_residency()?;
            let resident_fields = residency
                .fields
                .iter()
                .map(|field| {
                    format!(
                        "{}:{:?}:{:?}:{}",
                        field.descriptor.id,
                        field.residency,
                        field.descriptor.geometry,
                        field.descriptor.values_sha256
                    )
                })
                .collect::<Vec<_>>()
                .join(";");
            Ok(format!(
                "PASS_GM1_PRODUCTION_VIEW_V2 step={} model_time_f64_bits={:016x} manifest_sha256={} payload_set_sha256={} history_contract_sha256={} operator_manifest_sha256={} render_adapter={} source_profile={} profile_class={:?} domain={:?} vertical={:?} terrain={:?} resident_fields={} fields={}",
                frame.step,
                frame.model_time_s.to_bits(),
                frame.manifest_sha256,
                frame.payload_set_sha256,
                frame.history_contract_sha256,
                frame.operator_manifest_sha256,
                residency.frame.adapter_id,
                residency.source_profile.id,
                residency.source_profile.class,
                residency.source_profile.domain_label(),
                residency.source_profile.vertical_grid_label(),
                residency.source_profile.terrain_label(),
                resident_fields,
                frame_inventory(&frame),
            ))
        }
        [mode, root, alpha] if mode == "--series" => {
            let scan = scan_production_series(Path::new(root), None);
            let request = scan.request.ok_or_else(|| {
                let reasons = scan
                    .refusals
                    .iter()
                    .map(|refusal| format!("{}: {}", refusal.path.display(), refusal.reason))
                    .collect::<Vec<_>>()
                    .join("; ");
                format!(
                    "strict production series has no admissible real A/B request; visible_steps={:?}; refusals={reasons}",
                    scan.visible_steps
                )
            })?;
            let loader = ProductionPairLoader::new();
            loader.request(request)?;
            match loader
                .receive_timeout(Duration::from_secs(60))
                .map_err(|error| {
                    format!("bounded production pair load did not complete: {error}")
                })? {
                ProductionLoadEvent::Ready(pair) => blend_output(pair, parse_alpha(alpha)?),
                ProductionLoadEvent::Refused { reason, .. } => Err(format!(
                    "strict production second-frame admission refused: {reason}"
                )),
            }
        }
        [pack_a, pack_b, alpha] => {
            let pair = load_production_frame_pair(Path::new(pack_a), Path::new(pack_b), 1)?;
            blend_output(pair, parse_alpha(alpha)?)
        }
        _ => Err(USAGE.to_owned()),
    }
}

fn parse_alpha(alpha: &str) -> Result<f32, String> {
    alpha
        .parse::<f32>()
        .map_err(|error| format!("alpha is not binary32: {error}"))
}

fn blend_output(pair: ProductionFramePair, alpha: f32) -> Result<String, String> {
    let blended = pair.blend_exact_f32(alpha)?;
    let fields = blended
        .fields
        .iter()
        .map(|field| format!("{}:{}", field.name, field.values_sha256))
        .collect::<Vec<_>>()
        .join(";");
    Ok(format!(
        "PASS_GM1_PRODUCTION_VIEW_AB generation={} steps={:?} source_profile={} alpha_f32_bits={} model_time_f64_bits={:016x} formula={:?} source_manifests={:?} fields={fields}",
        pair.generation,
        blended.source_steps,
        pair.a.source_profile.id,
        blended.alpha_f32_bits,
        blended.model_time_s.to_bits(),
        EXACT_F32_BLEND_FORMULA,
        blended.source_manifest_sha256,
    ))
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match run(&arguments) {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("gm1-production-playback-probe: {error}");
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
        assert_eq!(run(&["a".to_owned(), "b".to_owned()]).unwrap_err(), USAGE);
        assert_eq!(
            run(&[
                "a".to_owned(),
                "b".to_owned(),
                "0.5".to_owned(),
                "extra".to_owned(),
            ])
            .unwrap_err(),
            USAGE
        );
    }
}
