use cm1_state::child_view_pack::{
    inspect_child_production_view_pack, inspect_child_view_pack,
    publish_child_production_view_pack_new, publish_child_view_pack_new,
    CHILD_PRODUCTION_VIEW_PACK_ABI, CHILD_PRODUCTION_VIEW_PACK_MANIFEST_SCHEMA,
    CHILD_PRODUCTION_VIEW_PACK_RECEIPT_SCHEMA, CHILD_VIEW_PACK_ABI,
    CHILD_VIEW_PACK_MANIFEST_SCHEMA, CHILD_VIEW_PACK_RECEIPT_SCHEMA,
};
use cm1_state::{Result, StateError};
use std::env;
use std::process::ExitCode;

const USAGE: &str = "usage:\n  gm1-child-view-pack --abi\n  gm1-child-view-pack --production-abi\n  gm1-child-view-pack publish PAIR_DIR PAIR_MANIFEST_SHA PAIR_RECEIPT_SHA PRODUCER_SOURCE PRODUCER_COMMIT NEW_DIR\n  gm1-child-view-pack inspect PAIR_DIR PAIR_MANIFEST_SHA PAIR_RECEIPT_SHA PRODUCER_SOURCE PRODUCER_EXE PRODUCER_COMMIT DIRECTORY MANIFEST_SHA PAYLOAD_SHA RECEIPT_SHA\n  gm1-child-view-pack publish-production PAIR_DIR PAIR_MANIFEST_SHA PAIR_RECEIPT_SHA PRODUCER_SOURCE PRODUCER_COMMIT NEW_DIR\n  gm1-child-view-pack inspect-production PAIR_DIR PAIR_MANIFEST_SHA PAIR_RECEIPT_SHA PRODUCER_SOURCE PRODUCER_EXE PRODUCER_COMMIT DIRECTORY MANIFEST_SHA PAYLOAD_SET_SHA RECEIPT_SHA";

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn run(arguments: &[String]) -> Result<String> {
    if arguments == ["--abi"] {
        return Ok(format!(
            "manifest={CHILD_VIEW_PACK_MANIFEST_SCHEMA};receipt={CHILD_VIEW_PACK_RECEIPT_SCHEMA};abi={CHILD_VIEW_PACK_ABI}"
        ));
    }
    if arguments == ["--production-abi"] {
        return Ok(format!(
            "manifest={CHILD_PRODUCTION_VIEW_PACK_MANIFEST_SCHEMA};receipt={CHILD_PRODUCTION_VIEW_PACK_RECEIPT_SCHEMA};abi={CHILD_PRODUCTION_VIEW_PACK_ABI}"
        ));
    }
    match arguments {
        [command, pair, pair_manifest, pair_receipt, producer_source, producer_commit, output]
            if command == "publish" =>
        {
            let executable = env::current_exe()?;
            let receipt = publish_child_view_pack_new(
                pair,
                pair_manifest,
                pair_receipt,
                producer_source,
                &executable,
                producer_commit,
                output,
            )?;
            Ok(format!(
                "PASS_CHILD_VIEW_PACK step={} model_time_f64_bits={} manifest_sha256={} payload_sha256={} receipt_sha256={}",
                receipt.step,
                receipt.model_time_f64_bits,
                receipt.manifest_sha256,
                receipt.payload_sha256,
                receipt.receipt_sha256
            ))
        }
        [command, pair, pair_manifest, pair_receipt, producer_source, producer_executable, producer_commit, directory, manifest, payload, receipt]
            if command == "inspect" =>
        {
            let restored = inspect_child_view_pack(
                pair,
                pair_manifest,
                pair_receipt,
                producer_source,
                producer_executable,
                producer_commit,
                directory,
                manifest,
                payload,
                receipt,
            )?;
            Ok(format!(
                "PASS_CHILD_VIEW_PACK_INSPECT step={} model_time_f64_bits={} manifest_sha256={} payload_sha256={} receipt_sha256={}",
                restored.step,
                restored.model_time_f64_bits,
                restored.manifest_sha256,
                restored.payload_sha256,
                restored.receipt_sha256
            ))
        }
        [command, pair, pair_manifest, pair_receipt, producer_source, producer_commit, output]
            if command == "publish-production" =>
        {
            let executable = env::current_exe()?;
            let receipt = publish_child_production_view_pack_new(
                pair,
                pair_manifest,
                pair_receipt,
                producer_source,
                &executable,
                producer_commit,
                output,
            )?;
            Ok(format!(
                "PASS_CHILD_PRODUCTION_VIEW_PACK step={} model_time_f64_bits={} manifest_sha256={} payload_set_sha256={} receipt_sha256={}",
                receipt.step,
                receipt.model_time_f64_bits,
                receipt.manifest_sha256,
                receipt.payload_set_sha256,
                receipt.receipt_sha256
            ))
        }
        [command, pair, pair_manifest, pair_receipt, producer_source, producer_executable, producer_commit, directory, manifest, payload_set, receipt]
            if command == "inspect-production" =>
        {
            let restored = inspect_child_production_view_pack(
                pair,
                pair_manifest,
                pair_receipt,
                producer_source,
                producer_executable,
                producer_commit,
                directory,
                manifest,
                payload_set,
                receipt,
            )?;
            Ok(format!(
                "PASS_CHILD_PRODUCTION_VIEW_PACK_INSPECT step={} model_time_f64_bits={} manifest_sha256={} payload_set_sha256={} receipt_sha256={}",
                restored.step,
                restored.model_time_f64_bits,
                restored.manifest_sha256,
                restored.payload_set_sha256,
                restored.receipt_sha256
            ))
        }
        _ => Err(invalid(USAGE)),
    }
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match run(&arguments) {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("gm1-child-view-pack: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_and_usage_are_fail_closed() {
        let abi = run(&["--abi".to_owned()]).unwrap();
        assert!(abi.contains(CHILD_VIEW_PACK_MANIFEST_SCHEMA));
        assert!(abi.contains(CHILD_VIEW_PACK_RECEIPT_SCHEMA));
        assert!(abi.contains(CHILD_VIEW_PACK_ABI));
        let production = run(&["--production-abi".to_owned()]).unwrap();
        assert!(production.contains(CHILD_PRODUCTION_VIEW_PACK_MANIFEST_SCHEMA));
        assert!(production.contains(CHILD_PRODUCTION_VIEW_PACK_RECEIPT_SCHEMA));
        assert!(production.contains(CHILD_PRODUCTION_VIEW_PACK_ABI));
        assert!(run(&[]).is_err());
    }
}
