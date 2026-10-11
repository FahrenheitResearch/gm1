use cm1_state::scientific::ScientificState;
use cm1_state::snapshot::{read_checkpoint, read_history, write_checkpoint_new, write_history_new};
use cm1_state::Container;
use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

const ABI: &str = "gpuwm-cm1-state/v1";

fn usage() -> &'static str {
    "usage:\n  cm1-state --abi\n  cm1-state inspect INPUT.cm1szs\n  cm1-state verify INPUT.cm1szs\n  cm1-state roundtrip INPUT.cm1szs OUTPUT.cm1szs\n  cm1-state initialize-scientific OUTPUT_DIR\n  cm1-state compare-scientific NATIVE_POST_GETSET_DIR OUTPUT_DIR\n  cm1-state render-scientific-comparison NATIVE_POST_GETSET_DIR OUTPUT.png\n  cm1-state checkpoint-scientific OUTPUT.cm1ckpt\n  cm1-state restore-checkpoint INPUT.cm1ckpt\n  cm1-state history-scientific OUTPUT.cm1hist\n  cm1-state verify-history INPUT.cm1hist"
}

fn run(arguments: Vec<OsString>) -> cm1_state::Result<()> {
    if arguments.is_empty() {
        return Err(error(usage()));
    }
    let command = arguments[0]
        .to_str()
        .ok_or_else(|| error("command is not valid UTF-8"))?;
    match (command, &arguments[1..]) {
        ("--abi", []) => {
            println!("{ABI}");
            Ok(())
        }
        ("inspect", [input]) => {
            let container = Container::read(PathBuf::from(input))?;
            container.verify_internal_roundtrip()?;
            println!("{}", container.inspect_json());
            Ok(())
        }
        ("verify", [input]) => {
            let container = Container::read(PathBuf::from(input))?;
            container.verify_internal_roundtrip()?;
            println!(
                "verified {}: {} payloads, {} bytes, {}",
                container.artifact_sha256(),
                container.payloads().len(),
                container.total_payload_bytes(),
                cm1_state::ARRAY_LAYOUT
            );
            Ok(())
        }
        ("roundtrip", [input, output]) => {
            let input = PathBuf::from(input);
            let output = PathBuf::from(output);
            if input == output {
                return Err(error("roundtrip INPUT and OUTPUT must differ"));
            }
            if output.exists() {
                return Err(error(format!(
                    "roundtrip output already exists: {}",
                    output.display()
                )));
            }
            let container = Container::read(&input)?;
            container.verify_internal_roundtrip()?;
            container.write_roundtrip(&output)?;
            println!(
                "roundtripped {} byte-exactly to {} ({})",
                container.artifact_sha256(),
                output.display(),
                cm1_state::ARRAY_LAYOUT
            );
            Ok(())
        }
        ("initialize-scientific", [output]) => {
            let state = ScientificState::initialize_sealed_v1();
            let receipt = state.write_new(PathBuf::from(output))?;
            println!(
                "initialized {} arrays ({} bytes) in {}; manifest {}",
                receipt.arrays,
                receipt.total_array_bytes,
                receipt.output_dir.display(),
                receipt.manifest_sha256
            );
            Ok(())
        }
        ("compare-scientific", [native, output]) => {
            let state = ScientificState::initialize_sealed_v1();
            let report = state.compare_native(PathBuf::from(native))?;
            let receipt = state.write_new(PathBuf::from(output))?;
            let comparison_path = receipt.output_dir.join("comparison.json");
            let comparison_sha256 = report.write_new(&comparison_path)?;
            let png_path = receipt.output_dir.join("comparison.png");
            let png_sha256 = state.write_comparison_png(&report, native, &png_path)?;
            for field in &report.fields {
                println!(
                    "{} {:?}: {}/{} exact, max {} ULP, max abs {:.9e}; interior {}/{}, boundary {}/{}",
                    field.name,
                    field.bounds.shape_ijk(),
                    field.all.exact_bits,
                    field.all.words,
                    field.all.max_ulp,
                    field.all.max_abs,
                    field.interior.exact_bits,
                    field.interior.words,
                    field.boundary.exact_bits,
                    field.boundary.words
                );
            }
            println!(
                "compared {} fields ({} bit-exact); report {} ({}); PNG {} ({})",
                report.fields.len(),
                report.exact_fields(),
                comparison_path.display(),
                comparison_sha256,
                png_path.display(),
                png_sha256
            );
            Ok(())
        }
        ("render-scientific-comparison", [native, output]) => {
            let state = ScientificState::initialize_sealed_v1();
            let report = state.compare_native(PathBuf::from(native))?;
            let output = PathBuf::from(output);
            let sha256 = state.write_comparison_png(&report, native, &output)?;
            println!(
                "rendered native scientific comparison to {} ({})",
                output.display(),
                sha256
            );
            Ok(())
        }
        ("checkpoint-scientific", [output]) => {
            let state = ScientificState::initialize_sealed_v1();
            let receipt = write_checkpoint_new(&state, PathBuf::from(output))?;
            println!("{}", receipt.canonical_json());
            Ok(())
        }
        ("restore-checkpoint", [input]) => {
            let restored = read_checkpoint(PathBuf::from(input))?;
            println!("{}", restored.receipt().canonical_json());
            Ok(())
        }
        ("history-scientific", [output]) => {
            let state = ScientificState::initialize_sealed_v1();
            let receipt = write_history_new(&state, PathBuf::from(output))?;
            println!("{}", receipt.canonical_json());
            Ok(())
        }
        ("verify-history", [input]) => {
            let restored = read_history(PathBuf::from(input))?;
            println!("{}", restored.receipt().canonical_json());
            Ok(())
        }
        _ => Err(error(usage())),
    }
}

fn error(message: impl Into<String>) -> cm1_state::StateError {
    let message = message.into();
    cm1_state::StateError::from(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message,
    ))
}

fn main() -> ExitCode {
    match run(env::args_os().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cm1-state: {error}");
            ExitCode::from(2)
        }
    }
}
