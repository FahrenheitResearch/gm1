//! Strict raw-f32 to CM1FHS1/CM1FCP1 forecast-envelope publisher.
//!
//! The CUDA driver writes explicitly named little-endian binary32 carriers to
//! new files.  This Rust boundary validates their complete byte inventory and
//! finite values, constructs the semantic forecast contract, and delegates
//! deterministic envelope publication to `cm1_state::forecast`.

use cm1_state::domain_location::{DomainLocationIntegrator, DOMAIN_LOCATION_SCALAR_NAMES};
use cm1_state::forecast::{
    read_forecast_checkpoint_embedded, write_forecast_checkpoint_new_raw,
    write_forecast_history_new_raw, ForecastArraySpec, ForecastClock, ForecastContract,
    RawArrayPayload, SchedulerScalar, SchedulerScalarSpec, SchedulerScalarValue,
};
use cm1_state::publish_timing;
use cm1_state::{Result, StateError};
use std::env;
use std::fs::{self, File};
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

/// Words checked for finiteness per branch-free pass; a block with a
/// non-finite word is rescanned word by word to name its index.
const FINITE_BLOCK_WORDS: usize = 1 << 16;
/// Bytes per read call.  One call per multi-gibibyte carrier asks the kernel
/// for a single enormous copy; bounded calls keep each copy cache-sized.
const READ_WINDOW_BYTES: usize = 64 << 20;

// v2 added the required SCALAR_TABLE argument.  It is required rather than
// optional because an optional one lets a caller that knows nothing about the
// CM1 moving-frame scalars publish a frame with no domain location in it and
// no sign that one is missing; `--abi` is how the Python driver refuses a
// publisher that predates them.
//
// v3 adds `restore`.  Until it existed this binary was half of CM1's restart
// pair: `restart_write.F` had a counterpart here and `restart_read.F`
// (`src/cm1.F:2666`, under `irst.eq.1`) had none, so a checkpoint could be
// written and never continued from.  A caller cannot discover that from the
// v2 surface -- the `checkpoint` subcommand looks complete on its own -- so
// the ABI names it, and a driver that needs to resume refuses a publisher
// that cannot read its own artifact back.
const ABI: &str = "gpuwm-cm1-forecast-publish/v3";
const USAGE: &str = concat!(
    "usage:\n",
    "  cm1-forecast-publish --abi\n",
    "  cm1-forecast-publish history|checkpoint NEW_OUTPUT PROFILE_ID ",
    "CONFIG_SHA256 EXECUTION_SHA256 STEP MODEL_TIME_S DT_S FIELD_TABLE.tsv ",
    "SCALAR_TABLE.tsv\n",
    "  cm1-forecast-publish restore INPUT.cm1fcp CONTRACT_SHA256 NEW_OUTPUT_DIR\n\n",
    "Each non-empty FIELD_TABLE row has eight tab-separated columns:\n",
    "NAME  MEANING  UNITS  STAGGER  SHAPE_CSV  BOUNDS_CSV  RAW_F32_PATH  OFFSET_BYTES\n",
    "SHAPE_CSV is slow-to-fast dimension order; BOUNDS_CSV contains lower,upper ",
    "pairs in the same order. Raw carriers are finite little-endian IEEE f32.\n\n",
    "Each non-empty SCALAR_TABLE row has three tab-separated columns:\n",
    "NAME  TYPE  VALUE\n",
    "TYPE is i64, u64, f64-bits or bool. An f64-bits VALUE is exactly sixteen ",
    "lowercase hexadecimal characters, so a scalar crosses this boundary ",
    "bit-for-bit rather than through a decimal round trip. A table naming the ",
    "CM1 moving-frame set (domainlocx, domainlocy, umove, vmove) is checked ",
    "against cm1.F:3512-3513 for the frame's own STEP and DT_S before anything ",
    "is written. An empty SCALAR_TABLE publishes no scheduler scalars.\n\n",
    "restore is the inverse of checkpoint: it validates one CM1FCP1 envelope ",
    "against the independently supplied CONTRACT_SHA256 and writes the arena, ",
    "field table and scalar table back into a new directory, so a driver reads ",
    "a checkpoint back through the same two tables it wrote."
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    History,
    Checkpoint,
}

impl Kind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "history" => Ok(Self::History),
            "checkpoint" => Ok(Self::Checkpoint),
            _ => Err(invalid("artifact kind must be history or checkpoint")),
        }
    }
}

#[derive(Debug)]
struct FieldRow {
    spec: ForecastArraySpec,
    path: PathBuf,
    offset: u64,
}

fn invalid(message: impl Into<String>) -> StateError {
    Error::new(ErrorKind::InvalidInput, message.into()).into()
}

fn parse_u64(value: &str, label: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|_| invalid(format!("{label} must be an unsigned integer")))
}

fn parse_f64(value: &str, label: &str) -> Result<f64> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid(format!("{label} must be a finite number")))?;
    if !parsed.is_finite() {
        return Err(invalid(format!("{label} must be finite")));
    }
    Ok(parsed)
}

fn parse_shape(value: &str, label: &str) -> Result<Vec<usize>> {
    if value.is_empty() {
        return Err(invalid(format!("{label} cannot be empty")));
    }
    value
        .split(',')
        .map(|word| {
            let parsed = word
                .parse::<usize>()
                .map_err(|_| invalid(format!("{label} contains invalid extent {word:?}")))?;
            if parsed == 0 {
                return Err(invalid(format!("{label} contains a zero extent")));
            }
            Ok(parsed)
        })
        .collect()
}

fn parse_bounds(value: &str, label: &str) -> Result<Vec<i32>> {
    if value.is_empty() {
        return Err(invalid(format!("{label} cannot be empty")));
    }
    value
        .split(',')
        .map(|word| {
            word.parse::<i32>()
                .map_err(|_| invalid(format!("{label} contains invalid bound {word:?}")))
        })
        .collect()
}

fn read_rows(path: &Path) -> Result<Vec<FieldRow>> {
    let metadata = fs::metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect field table {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(invalid(format!(
            "field table is not a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() > 1024 * 1024 {
        return Err(invalid("field table exceeds the 1 MiB v1 cap"));
    }
    let bytes = fs::read(path).map_err(|error| {
        invalid(format!(
            "could not read field table {}: {error}",
            path.display()
        ))
    })?;
    let text =
        std::str::from_utf8(&bytes).map_err(|_| invalid("field table must be strict UTF-8"))?;
    if text.contains('\r') {
        return Err(invalid("field table must use canonical LF line endings"));
    }
    let mut rows = Vec::new();
    for (zero_index, line) in text.lines().enumerate() {
        let line_number = zero_index + 1;
        if line.is_empty() {
            return Err(invalid(format!(
                "field table line {line_number} is empty; blank rows are not admitted"
            )));
        }
        let columns = line.split('\t').collect::<Vec<_>>();
        if columns.len() != 8 {
            return Err(invalid(format!(
                "field table line {line_number} has {} columns; expected eight",
                columns.len()
            )));
        }
        let shape = parse_shape(columns[4], &format!("line {line_number} shape"))?;
        let bounds = parse_bounds(columns[5], &format!("line {line_number} bounds"))?;
        let spec = ForecastArraySpec::new(
            columns[0], columns[1], columns[2], columns[3], bounds, shape,
        )?;
        let raw_path = PathBuf::from(columns[6]);
        if !raw_path.is_absolute() {
            return Err(invalid(format!(
                "field table line {line_number} raw path must be absolute"
            )));
        }
        let offset = columns[7].parse::<u64>().map_err(|_| {
            invalid(format!(
                "field table line {line_number} offset must be an unsigned byte count"
            ))
        })?;
        if offset % 4 != 0 {
            return Err(invalid(format!(
                "field table line {line_number} offset must be four-byte aligned"
            )));
        }
        rows.push(FieldRow {
            spec,
            path: raw_path,
            offset,
        });
    }
    if rows.is_empty() {
        return Err(invalid("field table must name at least one carrier"));
    }
    Ok(rows)
}

/// Read the scheduler-scalar table: `NAME`, `TYPE`, `VALUE` per row.
///
/// An `f64-bits` value crosses this boundary as sixteen lowercase hexadecimal
/// characters rather than a decimal literal, so `domainlocx` arrives with the
/// bits the runtime accumulated and not the nearest value some formatter
/// chose.  The forecast envelope stores f64 scalars the same way
/// (`forecast.rs` `f64_bits`), so the whole path is bit-for-bit.
fn read_scalars(path: &Path) -> Result<Vec<(SchedulerScalarSpec, SchedulerScalar)>> {
    let metadata = fs::metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect scalar table {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(invalid(format!(
            "scalar table is not a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() > 64 * 1024 {
        return Err(invalid("scalar table exceeds the 64 KiB v2 cap"));
    }
    let bytes = fs::read(path).map_err(|error| {
        invalid(format!(
            "could not read scalar table {}: {error}",
            path.display()
        ))
    })?;
    let text =
        std::str::from_utf8(&bytes).map_err(|_| invalid("scalar table must be strict UTF-8"))?;
    if text.contains('\r') {
        return Err(invalid("scalar table must use canonical LF line endings"));
    }
    let mut rows: Vec<(SchedulerScalarSpec, SchedulerScalar)> = Vec::new();
    for (zero_index, line) in text.lines().enumerate() {
        let line_number = zero_index + 1;
        if line.is_empty() {
            return Err(invalid(format!(
                "scalar table line {line_number} is empty; blank rows are not admitted"
            )));
        }
        let columns = line.split('\t').collect::<Vec<_>>();
        if columns.len() != 3 {
            return Err(invalid(format!(
                "scalar table line {line_number} has {} columns; expected three",
                columns.len()
            )));
        }
        let (name, kind, raw) = (columns[0], columns[1], columns[2]);
        let value = match kind {
            "i64" => SchedulerScalarValue::I64(raw.parse::<i64>().map_err(|_| {
                invalid(format!(
                    "scalar table line {line_number} value {raw:?} is not an i64"
                ))
            })?),
            "u64" => SchedulerScalarValue::U64(raw.parse::<u64>().map_err(|_| {
                invalid(format!(
                    "scalar table line {line_number} value {raw:?} is not a u64"
                ))
            })?),
            "f64-bits" => SchedulerScalarValue::F64(parse_f64_bits(raw, line_number)?),
            "bool" => match raw {
                "true" => SchedulerScalarValue::Bool(true),
                "false" => SchedulerScalarValue::Bool(false),
                _ => {
                    return Err(invalid(format!(
                        "scalar table line {line_number} bool must be true or false"
                    )))
                }
            },
            _ => {
                return Err(invalid(format!(
                    "scalar table line {line_number} type {kind:?} is not i64, u64, \
                     f64-bits or bool"
                )))
            }
        };
        if rows.iter().any(|(spec, _)| spec.name() == name) {
            return Err(invalid(format!(
                "scalar table line {line_number} repeats {name:?}; one frame carries \
                 one value per scalar and two cannot both be published"
            )));
        }
        rows.push((
            SchedulerScalarSpec::new(name, value.scalar_type())?,
            SchedulerScalar::new(name, value)?,
        ));
    }
    Ok(rows)
}

fn parse_f64_bits(value: &str, line_number: usize) -> Result<f64> {
    if value.len() != 16
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(invalid(format!(
            "scalar table line {line_number} f64 bits must be exactly sixteen \
             lowercase hexadecimal characters"
        )));
    }
    let bits = u64::from_str_radix(value, 16)
        .map_err(|error| invalid(format!("scalar table line {line_number}: {error}")))?;
    Ok(f64::from_bits(bits))
}

/// Re-derive `cm1.F:3512-3513` for this frame and refuse a value that has not
/// been integrated the way CM1 integrates it.
///
/// The driver accumulates the integral on the host, one addition per large
/// step.  This is the boundary that will not publish the result until it has
/// been checked against the frame's own clock, so a mis-wired, drifted or
/// reset accumulator stops the run instead of stamping a wrong ground
/// position onto an artifact that outlives the run.
fn require_cm1_domain_location(clock: ForecastClock, scalars: &[SchedulerScalar]) -> Result<()> {
    let present = DOMAIN_LOCATION_SCALAR_NAMES
        .iter()
        .filter(|name| scalars.iter().any(|scalar| scalar.name() == **name))
        .count();
    if present == 0 {
        return Ok(());
    }
    if present != DOMAIN_LOCATION_SCALAR_NAMES.len() {
        return Err(invalid(format!(
            "the frame carries {present} of the {} CM1 moving-frame scalars \
             {DOMAIN_LOCATION_SCALAR_NAMES:?}; cm1.F:3512-3513's integral cannot be \
             checked or continued from a partial set, and writeout.F:9614-9648 \
             writes the four together or not at all",
            DOMAIN_LOCATION_SCALAR_NAMES.len()
        )));
    }
    DomainLocationIntegrator::from_frame(clock, scalars).map(|_| ())
}

/// Read one carrier's published little-endian bytes straight into its
/// payload window, refusing a range outside the file or any non-finite word.
fn read_f32_bytes_into(path: &Path, offset: u64, window: &mut [u8], name: &str) -> Result<()> {
    let expected_bytes = window.len();
    let metadata = fs::metadata(path).map_err(|error| {
        invalid(format!(
            "could not inspect raw field {}: {error}",
            path.display()
        ))
    })?;
    let end = offset
        .checked_add(expected_bytes as u64)
        .ok_or_else(|| invalid(format!("raw field {name} byte range overflows")))?;
    if !metadata.is_file() || end > metadata.len() {
        return Err(invalid(format!(
            "raw field {name} range {offset}..{end} exceeds regular file {} ({} bytes)",
            path.display(),
            metadata.len()
        )));
    }
    let mut file = File::open(path).map_err(|error| {
        invalid(format!(
            "could not open raw field {}: {error}",
            path.display()
        ))
    })?;
    file.seek(SeekFrom::Start(offset)).map_err(|error| {
        invalid(format!(
            "could not seek raw field {} to {offset}: {error}",
            path.display()
        ))
    })?;
    for chunk in window.chunks_mut(READ_WINDOW_BYTES) {
        file.read_exact(chunk).map_err(|error| {
            invalid(format!(
                "could not read raw field {} range {offset}..{end}: {error}",
                path.display()
            ))
        })?;
    }
    // A binary32 word is non-finite exactly when its eight exponent bits are
    // all ones, which is what f32::is_finite tests.
    const EXPONENT: u32 = 0x7f80_0000;
    for (block_index, block) in window.chunks(4 * FINITE_BLOCK_WORDS).enumerate() {
        let non_finite = block.chunks_exact(4).fold(false, |seen, word| {
            seen | (u32::from_le_bytes(word.try_into().expect("four-byte chunk")) & EXPONENT
                == EXPONENT)
        });
        if !non_finite {
            continue;
        }
        for (word_index, word) in block.chunks_exact(4).enumerate() {
            let value = f32::from_bits(u32::from_le_bytes(
                word.try_into().expect("four-byte chunk"),
            ));
            if !value.is_finite() {
                let index = block_index * FINITE_BLOCK_WORDS + word_index;
                return Err(invalid(format!(
                    "raw field {name} contains a non-finite word at index {index}"
                )));
            }
        }
    }
    Ok(())
}

fn run(arguments: Vec<String>) -> Result<String> {
    if arguments.as_slice() == ["--abi"] {
        return Ok(ABI.to_string());
    }
    // src/cm1.F:2666: the reading half of the restart pair.
    if arguments.first().map(String::as_str) == Some("restore") {
        if arguments.len() != 4 {
            return Err(invalid(USAGE));
        }
        return restore(
            Path::new(&arguments[1]),
            &arguments[2],
            Path::new(&arguments[3]),
        );
    }
    if arguments.len() != 10 {
        return Err(invalid(USAGE));
    }
    let kind = Kind::parse(&arguments[0])?;
    let output = PathBuf::from(&arguments[1]);
    if output.exists() {
        return Err(invalid(format!(
            "forecast output already exists: {}",
            output.display()
        )));
    }
    let step = parse_u64(&arguments[5], "STEP")?;
    let model_time = parse_f64(&arguments[6], "MODEL_TIME_S")?;
    let dt = parse_f64(&arguments[7], "DT_S")?;
    let started = Instant::now();
    let rows = read_rows(Path::new(&arguments[8]))?;
    let scalar_rows = read_scalars(Path::new(&arguments[9]))?;
    publish_timing::record("read_tables", started.elapsed().as_secs_f64(), 0);
    let specs = rows.iter().map(|row| row.spec.clone()).collect::<Vec<_>>();
    let scalar_specs = scalar_rows
        .iter()
        .map(|(spec, _)| spec.clone())
        .collect::<Vec<_>>();
    let scalars = scalar_rows
        .into_iter()
        .map(|(_, scalar)| scalar)
        .collect::<Vec<_>>();
    let contract = ForecastContract::new(
        &arguments[2],
        &arguments[3],
        &arguments[4],
        specs,
        scalar_specs,
        Vec::new(),
    )?;
    // Every carrier is read, in field-table order, straight into its window
    // of the payload the envelope publishes: the arena bytes already are the
    // published little-endian words, so none is decoded and re-encoded and
    // the publisher holds one copy of the state, not two.
    let started = Instant::now();
    let mut payload = RawArrayPayload::new(&contract)?;
    publish_timing::record("allocate_payload", started.elapsed().as_secs_f64(), 0);
    let started = Instant::now();
    let mut arena_bytes = 0u64;
    for row in &rows {
        let window = payload
            .carrier_mut(row.spec.name())
            .expect("the contract was built from these rows");
        read_f32_bytes_into(&row.path, row.offset, window, row.spec.name())?;
        arena_bytes += window.len() as u64;
    }
    publish_timing::record(
        "read_arena_and_finite_check",
        started.elapsed().as_secs_f64(),
        arena_bytes,
    );
    let clock = ForecastClock::new(step, model_time, dt)?;
    // cm1.F:3512-3513, checked before a byte is written.
    require_cm1_domain_location(clock, &scalars)?;
    let receipt = match kind {
        Kind::History => {
            write_forecast_history_new_raw(&contract, payload, clock, scalars, Vec::new(), &output)?
        }
        Kind::Checkpoint => write_forecast_checkpoint_new_raw(
            &contract,
            payload,
            clock,
            scalars,
            Vec::new(),
            &output,
        )?,
    };
    Ok(format!(
        "abi={ABI}\ncontract_sha256={}\n{}",
        contract.sha256(),
        receipt.canonical_json()
    ))
}


// ---------------------------------------------------------------- restore
//
// CM1 has two halves of this: `restart_write.F` puts the model's carried
// state on disk and `restart_read.F` (`src/cm1.F:2666`, under `irst.eq.1`)
// reads the identical inventory straight back into the same arrays.  This
// binary had only the writing half, so an envelope it produced could be
// inspected but never continued from.  `restore` is the reading half, and it
// is deliberately the exact inverse of `publish`: it re-emits the raw arena,
// the eight-column field table and the three-column scalar table that a
// caller handed in, so the driver that wrote a checkpoint reads it back
// through the same two tables it wrote.

/// `restart_read.F:2` reads what `restart_write.F:2` wrote, under the same
/// names; these are the file names of that pair on this side.
const RESTORE_ARENA_FILE: &str = "restart-carriers.f32";
const RESTORE_FIELD_TABLE_FILE: &str = "restart-field-table.tsv";
const RESTORE_SCALAR_TABLE_FILE: &str = "restart-scheduler-scalars.tsv";
const RESTORE_RECEIPT_FILE: &str = "restart-restore.json";
const RESTORE_SCHEMA: &str = "gpuwm-cm1.forecast-restore-directory/v1";
const RESTORE_CONTRACT: &str = "one exact little-endian float32 arena plus the field and scheduler-scalar tables that describe it, written into a new sibling staging directory, synchronized, and renamed without replacing an existing destination";

fn absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let current = env::current_dir()
        .map_err(|error| invalid(format!("could not resolve the working directory: {error}")))?;
    Ok(current.join(path))
}

fn write_new_fsynced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| invalid(format!("could not create {}: {error}", path.display())))?;
    file.write_all(bytes)
        .map_err(|error| invalid(format!("could not write {}: {error}", path.display())))?;
    file.sync_all()
        .map_err(|error| invalid(format!("could not synchronize {}: {error}", path.display())))?;
    Ok(())
}

fn fsync_directory(path: &Path) {
    if let Ok(handle) = File::open(path) {
        let _ = handle.sync_all();
    }
}

fn scalar_table_row(scalar: &SchedulerScalar) -> String {
    match scalar.value() {
        SchedulerScalarValue::I64(value) => format!("{}\ti64\t{value}\n", scalar.name()),
        SchedulerScalarValue::U64(value) => format!("{}\tu64\t{value}\n", scalar.name()),
        // The same sixteen lowercase hexadecimal characters the writer took,
        // so a checkpoint's f64 clock and domain location come back as the
        // bits that were accumulated and not a decimal round trip.
        SchedulerScalarValue::F64(value) => {
            format!("{}\tf64-bits\t{:016x}\n", scalar.name(), value.to_bits())
        }
        SchedulerScalarValue::Bool(value) => format!("{}\tbool\t{value}\n", scalar.name()),
    }
}

fn restore(input: &Path, expected_contract_sha256: &str, output: &Path) -> Result<String> {
    if output.exists() {
        return Err(invalid(format!(
            "restore output already exists: {}",
            output.display()
        )));
    }
    let output = absolute(output)?;
    let parent = output
        .parent()
        .ok_or_else(|| invalid("restore output must name a directory inside an existing parent"))?
        .to_path_buf();
    if !parent.is_dir() {
        return Err(invalid(format!(
            "restore output parent is not an existing directory: {}",
            parent.display()
        )));
    }
    let name = output
        .file_name()
        .ok_or_else(|| invalid("restore output must name a directory"))?
        .to_string_lossy()
        .into_owned();
    let (contract, restored) =
        read_forecast_checkpoint_embedded(input, expected_contract_sha256)?;
    let frame = restored.frame();

    // One arena in the contract's own array order.  `ForecastContract::new`
    // sorts the array specs by name, so this is canonical order rather than
    // the writer's declaration order, and the re-emitted field table is what
    // says where each carrier landed.  A caller that assumed the writer's
    // layout instead of reading the table back would restore every carrier
    // under another carrier's name, which is exactly why the table is
    // re-emitted rather than left to be reconstructed.
    let mut arena = Vec::new();
    let mut table = String::new();
    let arena_path = output.join(RESTORE_ARENA_FILE);
    let arena_display = arena_path.to_string_lossy().into_owned();
    if arena_display.contains('\t') || arena_display.contains('\n') {
        return Err(invalid(
            "restore output path contains a tab or newline; the field table is strictly tab-separated and the row would split",
        ));
    }
    for spec in contract.arrays() {
        let array = frame
            .arrays()
            .iter()
            .find(|array| array.name() == spec.name())
            .ok_or_else(|| {
                invalid(format!(
                    "the envelope carries no array {}; the contract names it",
                    spec.name()
                ))
            })?;
        let offset = arena.len();
        for value in array.values() {
            arena.extend_from_slice(&value.to_le_bytes());
        }
        let shape = spec
            .shape()
            .iter()
            .map(|extent| extent.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let bounds = spec
            .bounds()
            .iter()
            .map(|bound| bound.to_string())
            .collect::<Vec<_>>()
            .join(",");
        table.push_str(&format!(
            "{}\t{}\t{}\t{}\t{shape}\t{bounds}\t{arena_display}\t{offset}\n",
            spec.name(),
            spec.meaning(),
            spec.units(),
            spec.stagger()
        ));
    }
    let scalars = frame
        .scheduler_scalars()
        .iter()
        .map(scalar_table_row)
        .collect::<String>();

    let clock = frame.clock();
    let receipt = restored.receipt();
    let document = format!(
        concat!(
            "{{\"arena_file\":\"{}\",\"arena_nbytes\":{},\"array_payload_sha256\":\"{}\",",
            "\"arrays\":{},\"artifact_sha256\":\"{}\",\"contract_sha256\":\"{}\",",
            "\"dt_f64_bits\":\"{}\",\"field_table_file\":\"{}\",\"kind\":\"checkpoint\",",
            "\"model_time_f64_bits\":\"{}\",\"publication_contract\":\"{}\",",
            "\"scalar_table_file\":\"{}\",\"schema\":\"{}\",\"scheduler_scalars\":{},",
            "\"step\":{}}}\n"
        ),
        RESTORE_ARENA_FILE,
        arena.len(),
        restored.array_payload_sha256(),
        receipt.arrays,
        receipt.artifact_sha256,
        receipt.contract_sha256,
        receipt.dt_f64_bits,
        RESTORE_FIELD_TABLE_FILE,
        receipt.model_time_f64_bits,
        RESTORE_CONTRACT,
        RESTORE_SCALAR_TABLE_FILE,
        RESTORE_SCHEMA,
        receipt.scheduler_scalars,
        clock.step()
    );

    let staging = parent.join(format!(".{name}.incomplete"));
    fs::create_dir(&staging).map_err(|error| {
        invalid(format!(
            "could not create the restore staging directory {}: {error}",
            staging.display()
        ))
    })?;
    let write_all = || -> Result<()> {
        write_new_fsynced(&staging.join(RESTORE_ARENA_FILE), &arena)?;
        write_new_fsynced(&staging.join(RESTORE_FIELD_TABLE_FILE), table.as_bytes())?;
        write_new_fsynced(&staging.join(RESTORE_SCALAR_TABLE_FILE), scalars.as_bytes())?;
        write_new_fsynced(&staging.join(RESTORE_RECEIPT_FILE), document.as_bytes())?;
        Ok(())
    };
    if let Err(error) = write_all() {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    fsync_directory(&staging);
    if output.exists() {
        let _ = fs::remove_dir_all(&staging);
        return Err(invalid(format!(
            "restore output appeared while it was being written: {}",
            output.display()
        )));
    }
    fs::rename(&staging, &output).map_err(|error| {
        let _ = fs::remove_dir_all(&staging);
        invalid(format!(
            "could not publish the restore directory {}: {error}",
            output.display()
        ))
    })?;
    fsync_directory(&parent);
    Ok(format!("abi={ABI}\n{document}"))
}

fn main() -> ExitCode {
    let started = Instant::now();
    let result = run(env::args().skip(1).collect());
    if publish_timing::enabled() {
        publish_timing::record("total", started.elapsed().as_secs_f64(), 0);
        eprint!("{}", publish_timing::take_lines());
    }
    match result {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("cm1-forecast-publish: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_bounds_parsers_are_strict() {
        assert_eq!(parse_shape("2,3,4", "shape").unwrap(), vec![2, 3, 4]);
        assert_eq!(
            parse_bounds("1,2,-1,1", "bounds").unwrap(),
            vec![1, 2, -1, 1]
        );
        assert!(parse_shape("2,0", "shape")
            .unwrap_err()
            .to_string()
            .contains("zero"));
        assert!(parse_bounds("1,x", "bounds").is_err());
    }

    /// One complete checkpoint publication and its restore, through the real
    /// CM1FCP1 envelope bytes, in a temporary directory of this test's own.
    ///
    /// This is CM1's `restart_write.F` / `restart_read.F` pair on this side:
    /// `src/cm1.F:4252` writes and `src/cm1.F:2666` reads the same inventory
    /// back.  What is asserted is that the second half returns exactly the
    /// carrier words and scalar bits the first half was handed -- an arena
    /// that came back merely close would resume onto a different trajectory.
    #[test]
    fn a_published_checkpoint_restores_to_the_exact_arena_and_scalars() {
        let root = std::env::temp_dir().join(format!(
            "cm1-forecast-publish-restore-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let words: [f32; 6] = [1.5, -2.25, 3.125, 0.0, 6.5e-3, -7.75];
        let arena = root.join("arena.f32");
        let mut raw = Vec::new();
        for value in words {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(&arena, &raw).unwrap();
        let fields = root.join("fields.tsv");
        fs::write(
            &fields,
            format!(
                "tkea\tsubgrid turbulence kinetic energy\tm2 s-2\tw\t2,3\t1,2,1,3\t{}\t0\n",
                arena.display()
            ),
        )
        .unwrap();
        // The four CM1 moving-frame scalars for 120 steps of 0.5 s at
        // umove=12.5, vmove=3.0 (cm1.F:3512-3513), plus the kind of extra
        // scheduler scalar a restart carries: mass1 and a step counter.
        let scalars = root.join("scalars.tsv");
        let f64_row = |name: &str, value: f64| format!("{name}\tf64-bits\t{:016x}\n", value.to_bits());
        fs::write(
            &scalars,
            format!(
                "{}{}{}{}{}nstep\tu64\t120\niconly\tbool\tfalse\n",
                f64_row("domainlocx", 750.0),
                f64_row("domainlocy", 180.0),
                f64_row("umove", 12.5),
                f64_row("vmove", 3.0),
                f64_row("mass1", 0.1),
            ),
        )
        .unwrap();

        let envelope = root.join("state.cm1fcp");
        let published = run(vec![
            "checkpoint".into(),
            envelope.display().to_string(),
            "gm1.cartesian-restart/v1".into(),
            "11".repeat(32),
            "22".repeat(32),
            "120".into(),
            "60.0".into(),
            "0.5".into(),
            fields.display().to_string(),
            scalars.display().to_string(),
        ])
        .unwrap();
        assert!(published.starts_with("abi=gpuwm-cm1-forecast-publish/v3"));
        assert!(fs::read(&envelope).unwrap().starts_with(b"CM1FCP1\0"));
        let contract_sha = published
            .lines()
            .find_map(|line| line.strip_prefix("contract_sha256="))
            .unwrap()
            .to_string();

        // A restore anchored to a different contract must not hand back a
        // state: a checkpoint written for another carrier inventory would be
        // decoded under this run's names and resume onto scrambled fields.
        let wrong = root.join("wrong");
        assert!(run(vec![
            "restore".into(),
            envelope.display().to_string(),
            "33".repeat(32),
            wrong.display().to_string(),
        ])
        .is_err());
        assert!(!wrong.exists());
        assert!(!root.join(".wrong.incomplete").exists());

        let restored = root.join("restored");
        run(vec![
            "restore".into(),
            envelope.display().to_string(),
            contract_sha.clone(),
            restored.display().to_string(),
        ])
        .unwrap();
        assert_eq!(
            fs::read(restored.join(RESTORE_ARENA_FILE)).unwrap(),
            raw,
            "restored arena is not the published arena, word for word"
        );
        let table = fs::read_to_string(restored.join(RESTORE_FIELD_TABLE_FILE)).unwrap();
        let columns = table.trim_end().split('\t').collect::<Vec<_>>();
        assert_eq!(columns[0], "tkea");
        assert_eq!(columns[1], "subgrid turbulence kinetic energy");
        assert_eq!(columns[2], "m2 s-2");
        assert_eq!(columns[3], "w");
        assert_eq!(columns[4], "2,3");
        assert_eq!(columns[5], "1,2,1,3");
        assert_eq!(
            columns[6],
            restored.join(RESTORE_ARENA_FILE).display().to_string()
        );
        assert_eq!(columns[7], "0");
        let back = fs::read_to_string(restored.join(RESTORE_SCALAR_TABLE_FILE)).unwrap();
        for row in [
            f64_row("domainlocx", 750.0),
            f64_row("domainlocy", 180.0),
            f64_row("mass1", 0.1),
            f64_row("umove", 12.5),
            f64_row("vmove", 3.0),
            "nstep\tu64\t120\n".to_string(),
            "iconly\tbool\tfalse\n".to_string(),
        ] {
            assert!(back.contains(&row), "{row:?} missing from {back:?}");
        }

        // The destination is never replaced: a second restore onto the same
        // directory would otherwise silently mix two model times.
        assert!(run(vec![
            "restore".into(),
            envelope.display().to_string(),
            contract_sha,
            restored.display().to_string(),
        ])
        .is_err());

        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn kind_refuses_unknown_values() {
        assert_eq!(Kind::parse("history").unwrap(), Kind::History);
        assert_eq!(Kind::parse("checkpoint").unwrap(), Kind::Checkpoint);
        assert!(Kind::parse("grads").is_err());
    }

    fn scalar(name: &str, value: f64) -> SchedulerScalar {
        SchedulerScalar::new(name, SchedulerScalarValue::F64(value)).unwrap()
    }

    #[test]
    fn f64_bits_cross_the_boundary_without_a_decimal_round_trip() {
        // 0.1 has no exact decimal-free binary form, so a formatter is the
        // one thing this column must not depend on.
        assert_eq!(parse_f64_bits("3fb999999999999a", 1).unwrap(), 0.1_f64);
        assert!(parse_f64_bits("3FB999999999999A", 1).is_err());
        assert!(parse_f64_bits("3fb99999999999", 1).is_err());
    }

    #[test]
    fn a_correct_domain_location_is_admitted_and_a_drifted_one_is_refused() {
        // Two hours of 0.5 s steps at CM1's own supercell frame velocity.
        let clock = ForecastClock::new(14_400, 7_200.0, 0.5).unwrap();
        let good = vec![
            scalar("domainlocx", 90_000.0),
            scalar("domainlocy", 21_600.0),
            scalar("umove", 12.5),
            scalar("vmove", 3.0),
        ];
        require_cm1_domain_location(clock, &good).unwrap();

        let mut drifted = good.clone();
        drifted[0] = scalar("domainlocx", 90_000.000_000_001);
        let message = require_cm1_domain_location(clock, &drifted)
            .unwrap_err()
            .to_string();
        assert!(message.contains("domainlocx"), "{message}");
    }

    #[test]
    fn a_partial_moving_frame_set_is_refused_and_an_empty_one_is_allowed() {
        let clock = ForecastClock::new(14_400, 7_200.0, 0.5).unwrap();
        require_cm1_domain_location(clock, &[]).unwrap();
        let message = require_cm1_domain_location(clock, &[scalar("umove", 12.5)])
            .unwrap_err()
            .to_string();
        assert!(message.contains("1 of the 4"), "{message}");
    }

    #[test]
    fn the_scalar_table_parser_is_strict_and_admits_every_contract_type() {
        let directory = std::env::temp_dir().join(format!(
            "gm1-scalar-table-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("scalars.tsv");

        fs::write(
            &path,
            "domainlocx\tf64-bits\t40f5f90000000000\ncompleted\tu64\t7\nnrk\ti64\t-1\nready\tbool\ttrue\n",
        )
        .unwrap();
        let rows = read_scalars(&path).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].1.value(), SchedulerScalarValue::F64(90_000.0));
        assert_eq!(rows[1].1.value(), SchedulerScalarValue::U64(7));
        assert_eq!(rows[2].1.value(), SchedulerScalarValue::I64(-1));
        assert_eq!(rows[3].1.value(), SchedulerScalarValue::Bool(true));

        // An empty table is a frame with no scheduler scalars, not an error.
        fs::write(&path, "").unwrap();
        assert!(read_scalars(&path).unwrap().is_empty());

        for (bad, expected) in [
            (
                "domainlocx\tf64-bits\t40f5f90000000000\ndomainlocx\tf64-bits\t0000000000000000\n",
                "repeats",
            ),
            (
                "domainlocx\tf32-bits\t40f5f900\n",
                "not i64, u64, f64-bits or bool",
            ),
            ("domainlocx\tf64-bits\n", "expected three"),
            ("domainlocx\tbool\tyes\n", "true or false"),
        ] {
            fs::write(&path, bad).unwrap();
            let message = read_scalars(&path).unwrap_err().to_string();
            assert!(message.contains(expected), "{message}");
        }

        fs::remove_dir_all(&directory).unwrap();
    }
}
