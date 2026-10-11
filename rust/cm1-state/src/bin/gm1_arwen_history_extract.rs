use cm1_state::arwen_history::{
    extract_history_record, CHAIN_VARIABLES, GM1_ARWEN_HISTORY_EXTRACT_ABI,
    GM1_ARWEN_HISTORY_FRAME_SCHEMA,
};
use cm1_state::{Result, StateError};
use std::env;
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = concat!(
    "usage:\n",
    "  gm1-arwen-history-extract --abi\n",
    "  gm1-arwen-history-extract WRFOUT RECORD NEW_OUTPUT_DIRECTORY"
);

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn run(arguments: &[String]) -> Result<String> {
    match arguments {
        [flag] if flag == "--abi" => Ok(format!(
            "{GM1_ARWEN_HISTORY_EXTRACT_ABI}\nframe_schema={GM1_ARWEN_HISTORY_FRAME_SCHEMA}"
        )),
        [input, record, output] => {
            let record: u32 = record
                .parse()
                .map_err(|_| invalid(format!("RECORD must be a non-negative integer, not {record:?}")))?;
            let frame = extract_history_record(Path::new(input), record, Path::new(output), CHAIN_VARIABLES)?;
            Ok(format!(
                "{{\"fields\":{},\"skipped\":{},\"times\":\"{}\"}}",
                frame.fields.len(),
                frame.skipped.len(),
                frame.times
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
            eprintln!("gm1-arwen-history-extract: {error}");
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
        assert!(abi.contains(GM1_ARWEN_HISTORY_EXTRACT_ABI));
        assert!(run(&[]).unwrap_err().to_string().contains("usage:"));
        assert!(run(&["a".into(), "x".into(), "b".into()]).unwrap_err().to_string().contains("RECORD"));
    }
}
