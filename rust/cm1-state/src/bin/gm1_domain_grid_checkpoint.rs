use cm1_state::domain_grid_checkpoint::{
    write_moving_domain_component_new, write_stretch_z1_component_new, F02InteriorRoiReceipt,
    MovingDomainMode, MovingDomainSelection, MovingDomainState, StretchZ1Parameters,
    F02_INTERIOR_ROI_SCHEMA, MOVING_DOMAIN_COMPONENT_SCHEMA, STRETCH_Z1_COMPONENT_SCHEMA,
};
use cm1_state::{Result, StateError};
use std::env;
use std::process::ExitCode;

const ABI: &str = "gm1.domain-grid-checkpoint-cli/v1";
const USAGE: &str = "usage:\n  gm1-domain-grid-checkpoint --abi\n  gm1-domain-grid-checkpoint stretch-z1 NEW.json\n  gm1-domain-grid-checkpoint fixed-f02-roi NEW.json\n  gm1-domain-grid-checkpoint f02-roi-receipt NEW.json";

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn fixed_f02_state() -> MovingDomainState {
    MovingDomainState {
        icenter: 151,
        jcenter: 151,
        xcenter: 15_000.0,
        ycenter: 15_000.0,
        umove: 0.0,
        umovem1: 0.0,
        umovem2: 0.0,
        vmove: 0.0,
        vmovem1: 0.0,
        vmovem2: 0.0,
        domainlocx: 0.0,
        domainlocy: 0.0,
        adaptmovetim: 0.0,
        mvrec: 1,
        nwritemv: 1,
    }
}

fn run(arguments: &[String]) -> Result<String> {
    if arguments == ["--abi"] {
        return Ok(format!(
            "{ABI};stretch_schema={STRETCH_Z1_COMPONENT_SCHEMA};moving_schema={MOVING_DOMAIN_COMPONENT_SCHEMA};roi_schema={F02_INTERIOR_ROI_SCHEMA}"
        ));
    }
    let [command, output] = arguments else {
        return Err(invalid(USAGE));
    };
    match command.as_str() {
        "stretch-z1" => {
            write_stretch_z1_component_new(StretchZ1Parameters::gm1_selected(), output)?;
            Ok("PASS_STRETCH_Z1_COMPONENT".to_owned())
        }
        "fixed-f02-roi" => {
            let selection = MovingDomainSelection {
                mode: MovingDomainMode::Fixed,
                terrain: true,
                axisymm: false,
                initsfc: 1,
                lateral_bc_west_east_south_north: [2; 4],
                adapt_move_frq: 0.0,
            };
            write_moving_domain_component_new(selection, fixed_f02_state(), 300, 300, output)?;
            Ok("PASS_FIXED_F02_ROI_COMPONENT".to_owned())
        }
        "f02-roi-receipt" => {
            F02InteriorRoiReceipt::sealed().write_new(output)?;
            Ok("PASS_F02_ROI_RECEIPT".to_owned())
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
            eprintln!("gm1-domain-grid-checkpoint: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_and_usage_are_strict() {
        let abi = run(&["--abi".to_owned()]).unwrap();
        assert!(abi.contains(STRETCH_Z1_COMPONENT_SCHEMA));
        assert!(abi.contains(MOVING_DOMAIN_COMPONENT_SCHEMA));
        assert!(abi.contains(F02_INTERIOR_ROI_SCHEMA));
        assert!(run(&[]).is_err());
    }
}
