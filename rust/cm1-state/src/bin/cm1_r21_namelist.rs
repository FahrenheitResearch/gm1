use cm1_state::cm1_namelist::{
    cm1_r21_namelist_support_json, inspect_cm1_r21_namelist, parse_cm1_r21_namelist,
    Cm1UnknownPolicy, CM1_NAMELIST_SUPPORT_SHA256,
};
use cm1_state::{Result, StateError};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

/// One line, because the wheel seals every native tool by its `--abi` line.
/// v2 added `inspect --json --unknown report` and the inspection schema.
const ABI: &str = concat!(
    "gm1.cm1-r21.1-namelist-cli/v2",
    ";document_schema=gm1.cm1-r21.1-namelist-document/v1",
    ";support_schema=gm1.cm1-r21.1-configuration-support/v1",
    ";binding_schema=gm1.cm1-r21.1-configuration-binding/v1",
    ";inspection_schema=gm1.cm1-r21.1-namelist-inspection/v1"
);
const USAGE: &str = concat!(
    "usage:\n",
    "  cm1-r21-namelist --abi\n",
    "  cm1-r21-namelist schema\n",
    "  cm1-r21-namelist inspect INPUT.nml\n",
    "  cm1-r21-namelist inspect --json [--unknown refuse|report] INPUT.nml\n",
    "  cm1-r21-namelist normalize INPUT.nml OUTPUT.nml\n",
    "  cm1-r21-namelist bind INPUT.nml OUTPUT.json\n",
    "  cm1-r21-namelist check-checkpoint INPUT.nml"
);

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn read_regular(path: &str) -> Result<Vec<u8>> {
    let path = Path::new(path);
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect NAMELIST {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(invalid(
            "CM1 NAMELIST input must be a regular non-symlink file",
        ));
    }
    fs::read(path).map_err(StateError::from)
}

fn write_new(path: &str, bytes: &[u8]) -> Result<()> {
    let path = Path::new(path);
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(invalid(format!(
            "CM1 NAMELIST output parent does not exist: {}",
            parent.display()
        )));
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

/// `--json` selects the inspection document; `--unknown` defaults to refuse,
/// which keeps the strict identity contract unless a caller asks otherwise.
fn inspection_arguments(arguments: &[String]) -> Result<(Cm1UnknownPolicy, &str)> {
    let mut json = false;
    let mut policy = Cm1UnknownPolicy::Refuse;
    let mut input = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--json" => json = true,
            "--unknown" => {
                index += 1;
                let value = arguments
                    .get(index)
                    .ok_or_else(|| invalid("--unknown needs refuse or report"))?;
                policy = Cm1UnknownPolicy::parse(value)?;
            }
            value if value.starts_with("--") => {
                return Err(invalid(format!(
                    "unknown inspect option {value:?}\n{USAGE}"
                )))
            }
            value if input.is_none() => input = Some(value),
            _ => return Err(invalid(USAGE)),
        }
        index += 1;
    }
    if !json {
        return Err(invalid(format!("inspect options need --json\n{USAGE}")));
    }
    let input = input.ok_or_else(|| invalid(USAGE))?;
    Ok((policy, input))
}

fn run(arguments: &[String]) -> Result<String> {
    match arguments {
        [flag] if flag == "--abi" => Ok(ABI.to_string()),
        [command] if command == "schema" => {
            Ok(String::from_utf8(cm1_r21_namelist_support_json()?.to_vec())
                .expect("compiled schema is UTF-8"))
        }
        [command, input] if command == "inspect" && !input.starts_with("--") => {
            let document = parse_cm1_r21_namelist(&read_regular(input)?)?;
            Ok(String::from_utf8(document.canonical_json().to_vec())
                .expect("canonical document is UTF-8"))
        }
        [command, rest @ ..] if command == "inspect" => {
            let (policy, input) = inspection_arguments(rest)?;
            let inspection = inspect_cm1_r21_namelist(&read_regular(input)?, policy)?;
            Ok(String::from_utf8(inspection.canonical_json()?).expect("inspection JSON is UTF-8"))
        }
        [command, input, output] if command == "normalize" => {
            let document = parse_cm1_r21_namelist(&read_regular(input)?)?;
            document.verify_semantic_roundtrip()?;
            let normalized = document.canonical_namelist();
            write_new(output, normalized.as_bytes())?;
            Ok(format!(
                "{{\"assignment_count\":{},\"document_sha256\":\"{}\",\"group_count\":{},\"schema\":\"gm1.cm1-r21.1-namelist-normalize-receipt/v1\",\"support_schema_sha256\":\"{}\"}}",
                document.assignment_count(),
                document.sha256(),
                document.group_count(),
                CM1_NAMELIST_SUPPORT_SHA256
            ))
        }
        [command, input, output] if command == "bind" => {
            let document = parse_cm1_r21_namelist(&read_regular(input)?)?;
            let binding = document.configuration_binding()?;
            write_new(output, binding.bytes())?;
            Ok(format!(
                "{{\"binding_sha256\":\"{}\",\"blocker_count\":{},\"document_sha256\":\"{}\",\"schema\":\"gm1.cm1-r21.1-configuration-binding-receipt/v1\"}}",
                binding.sha256(),
                binding.blockers().len(),
                binding.document_sha256()
            ))
        }
        [command, input] if command == "check-checkpoint" => {
            let document = parse_cm1_r21_namelist(&read_regular(input)?)?;
            document.require_arbitrary_checkpoint_supported()?;
            unreachable!("arbitrary checkpoint admission is fail-closed in v1")
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
            eprintln!("cm1-r21-namelist: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cm1_state::cm1_namelist::{
        CM1_CONFIGURATION_BINDING_SCHEMA, CM1_NAMELIST_DOCUMENT_SCHEMA,
        CM1_NAMELIST_INSPECTION_SCHEMA, CM1_NAMELIST_SUPPORT_SCHEMA,
    };

    #[test]
    fn abi_exposes_all_four_versioned_schemas_on_one_line() {
        let output = run(&["--abi".to_string()]).unwrap();
        assert_eq!(output, ABI);
        assert!(!output.contains('\n'));
        assert!(output.contains(&format!("document_schema={CM1_NAMELIST_DOCUMENT_SCHEMA}")));
        assert!(output.contains(&format!("support_schema={CM1_NAMELIST_SUPPORT_SCHEMA}")));
        assert!(output.contains(&format!(
            "binding_schema={CM1_CONFIGURATION_BINDING_SCHEMA}"
        )));
        assert!(output.contains(&format!(
            "inspection_schema={CM1_NAMELIST_INSPECTION_SCHEMA}"
        )));
    }

    #[test]
    fn inspect_options_are_parsed_strictly() {
        let arguments = |values: &[&str]| {
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
        };
        let parsed = arguments(&["--json", "--unknown", "report", "deck.nml"]);
        let (policy, input) = inspection_arguments(&parsed).unwrap();
        assert_eq!(policy, Cm1UnknownPolicy::Report);
        assert_eq!(input, "deck.nml");
        let parsed = arguments(&["deck.nml", "--json"]);
        assert_eq!(
            inspection_arguments(&parsed).unwrap().0,
            Cm1UnknownPolicy::Refuse
        );
        for bad in [
            &["--unknown", "report", "deck.nml"][..],
            &["--json", "--unknown", "ignore", "deck.nml"][..],
            &["--json", "--frobnicate", "deck.nml"][..],
            &["--json", "a.nml", "b.nml"][..],
            &["--json", "--unknown"][..],
        ] {
            assert!(inspection_arguments(&arguments(bad)).is_err(), "{bad:?}");
        }
    }
}
