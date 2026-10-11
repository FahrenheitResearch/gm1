use cm1_state::child_remap::{
    read_parent_binding, write_child_producer_requirements_new, write_geometry_preflight_new,
    write_validated_parent_plan_new, ParentAnchor, ParentBinding, ParentClock,
    CHILD_PRODUCER_GATE_ABI, CHILD_PRODUCER_REQUIREMENTS_SCHEMA, CHILD_REMAP_BUILDER_ABI,
    CHILD_REMAP_PLAN_SCHEMA, CHILD_REMAP_PREFLIGHT_SCHEMA, CHILD_STRETCH_ABI, CHILD_STRETCH_SCHEMA,
    T300_PARENT_BINDING_SCHEMA,
};
use cm1_state::{Result, StateError};
use std::env;
use std::process::ExitCode;

const USAGE: &str = "usage:\n  gm1-t300-child-remap --abi\n  gm1-t300-child-remap geometry-preflight NEW.json\n  gm1-t300-child-remap bind-parent ARTIFACT_SHA CONTRACT_SHA ARENA_SHA OPERATOR_SHA DIAGNOSTICS_SHA PDEFQ_SHA COMPLETED_STEP REFL_I REFL_J REFL_F32_BITS UPDRAFT_I UPDRAFT_J UPDRAFT_F32_BITS VORT_I VORT_J VORT_F32_BITS NEW.json\n  gm1-t300-child-remap validate-parent BINDING.json PARENT.cm1fcp DIAGNOSTICS_RECEIPT PDEFQ_RECEIPT NEW.json\n  gm1-t300-child-remap producer-requirements VALIDATED_PLAN.json NEW.json";

fn invalid(message: impl Into<String>) -> StateError {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}

fn run(arguments: &[String]) -> Result<String> {
    if arguments == ["--abi"] {
        return Ok(format!(
            "{CHILD_REMAP_BUILDER_ABI};binding={T300_PARENT_BINDING_SCHEMA};preflight={CHILD_REMAP_PREFLIGHT_SCHEMA};plan={CHILD_REMAP_PLAN_SCHEMA};producer_gate={CHILD_PRODUCER_GATE_ABI};producer_requirements={CHILD_PRODUCER_REQUIREMENTS_SCHEMA};stretch_schema={CHILD_STRETCH_SCHEMA};stretch_abi={CHILD_STRETCH_ABI}"
        ));
    }
    match arguments {
        [command, output] if command == "geometry-preflight" => {
            write_geometry_preflight_new(output)?;
            Ok("PASS_T300_CHILD_GEOMETRY_PREFLIGHT".to_owned())
        }
        [command, artifact, contract, arena, operator, diagnostics, pdefq, step, refl_i, refl_j, refl_bits, updraft_i, updraft_j, updraft_bits, vort_i, vort_j, vort_bits, output]
            if command == "bind-parent" =>
        {
            let binding = ParentBinding::new(
                artifact,
                contract,
                arena,
                operator,
                diagnostics,
                pdefq,
                ParentClock::completed_large_step(parse_u64(step, "COMPLETED_STEP")?)?,
                ParentAnchor {
                    i: parse_usize(refl_i, "REFL_I")?,
                    j: parse_usize(refl_j, "REFL_J")?,
                    value: parse_f32_bits(refl_bits, "REFL_F32_BITS")?,
                },
                ParentAnchor {
                    i: parse_usize(updraft_i, "UPDRAFT_I")?,
                    j: parse_usize(updraft_j, "UPDRAFT_J")?,
                    value: parse_f32_bits(updraft_bits, "UPDRAFT_F32_BITS")?,
                },
                ParentAnchor {
                    i: parse_usize(vort_i, "VORT_I")?,
                    j: parse_usize(vort_j, "VORT_J")?,
                    value: parse_f32_bits(vort_bits, "VORT_F32_BITS")?,
                },
            )?;
            binding.write_new(output)?;
            Ok("PASS_PDEFQ_CLEAN_T300_PARENT_BINDING".to_owned())
        }
        [command, binding, parent, diagnostics, pdefq, output] if command == "validate-parent" => {
            let binding = read_parent_binding(binding)?;
            write_validated_parent_plan_new(&binding, parent, diagnostics, pdefq, output)?;
            Ok("PASS_PDEFQ_CLEAN_T300_PARENT_CHILD_PLAN".to_owned())
        }
        [command, plan, output] if command == "producer-requirements" => {
            write_child_producer_requirements_new(plan, output)?;
            Ok("PASS_CHILD_PRODUCER_REQUIREMENTS_BLOCKED_ON_NUMERICAL_PRODUCERS".to_owned())
        }
        _ => Err(invalid(USAGE)),
    }
}

fn parse_u64(value: &str, label: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|_| invalid(format!("{label} is not an unsigned integer")))
}

fn parse_usize(value: &str, label: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| invalid(format!("{label} is not an unsigned integer")))
}

fn parse_f32_bits(value: &str, label: &str) -> Result<f32> {
    if value.len() != 8 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(format!("{label} is not eight hex digits")));
    }
    let result = f32::from_bits(
        u32::from_str_radix(value, 16).map_err(|_| invalid(format!("{label} does not parse")))?,
    );
    if !result.is_finite() {
        return Err(invalid(format!("{label} is not finite")));
    }
    Ok(result)
}

fn main() -> ExitCode {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match run(&arguments) {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("gm1-t300-child-remap: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn abi_and_usage_are_strict() {
        let abi = run(&["--abi".to_owned()]).unwrap();
        assert!(abi.contains(CHILD_REMAP_BUILDER_ABI));
        assert!(abi.contains(T300_PARENT_BINDING_SCHEMA));
        assert!(run(&[]).is_err());
    }

    #[test]
    fn bind_parent_command_publishes_rust_canonical_input() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("gm1-child-cli-{nonce}"));
        fs::create_dir(&root).unwrap();
        let output = root.join("binding.json");
        let arguments = vec![
            "bind-parent".to_owned(),
            "11".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
            "44".repeat(32),
            "55".repeat(32),
            "66".repeat(32),
            "600".to_owned(),
            "243".to_owned(),
            "217".to_owned(),
            format!("{:08x}", 63.182f32.to_bits()),
            "264".to_owned(),
            "199".to_owned(),
            format!("{:08x}", 37.312f32.to_bits()),
            "247".to_owned(),
            "258".to_owned(),
            format!("{:08x}", 0.04631f32.to_bits()),
            output.to_string_lossy().into_owned(),
        ];
        assert_eq!(
            run(&arguments).unwrap(),
            "PASS_PDEFQ_CLEAN_T300_PARENT_BINDING"
        );
        let binding = read_parent_binding(&output).unwrap();
        assert_eq!(binding.low_level_reflectivity.i, 243);
        assert_eq!(binding.clock.step(), 600);
        assert_eq!(binding.clock.model_time_s(), 300.0);
        // A different completed large step is admitted the same way.
        let later = root.join("binding-later.json");
        let mut arguments = arguments.clone();
        arguments[7] = "4321".to_owned();
        let last = arguments.len() - 1;
        arguments[last] = later.to_string_lossy().into_owned();
        assert_eq!(
            run(&arguments).unwrap(),
            "PASS_PDEFQ_CLEAN_T300_PARENT_BINDING"
        );
        let later_binding = read_parent_binding(&later).unwrap();
        assert_eq!(later_binding.clock.step(), 4321);
        assert_eq!(later_binding.clock.model_time_s(), 2160.5);
        fs::remove_dir_all(root).unwrap();
    }
}
