use cm1_state::forcing_history_export::{
    export_forcing_history_new, GM1_FORCING_HISTORY_EXPORT_ABI, GM1_FORCING_HISTORY_EXPORT_SCHEMA,
};
use cm1_state::{Result, StateError};
use std::env;
use std::process::ExitCode;

const USAGE: &str = concat!(
    "usage:\n",
    "  gm1-forcing-history-export --abi\n",
    "  gm1-forcing-history-export INPUT.cm1fhs EXPECTED_CONTRACT_SHA256 NEW_OUTPUT_DIRECTORY"
);

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn run(arguments: &[String]) -> Result<String> {
    match arguments {
        [flag] if flag == "--abi" => Ok(format!(
            "{GM1_FORCING_HISTORY_EXPORT_ABI}\nmanifest_schema={GM1_FORCING_HISTORY_EXPORT_SCHEMA}"
        )),
        [input, expected_contract, output] => {
            Ok(export_forcing_history_new(input, expected_contract, output)?.canonical_json())
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
            eprintln!("gm1-forcing-history-export: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_is_versioned_and_usage_is_strict() {
        let abi = run(&["--abi".to_string()]).unwrap();
        assert!(abi.contains(GM1_FORCING_HISTORY_EXPORT_ABI));
        assert!(abi.contains(GM1_FORCING_HISTORY_EXPORT_SCHEMA));
        assert!(run(&[]).unwrap_err().to_string().contains("usage:"));
    }
}
