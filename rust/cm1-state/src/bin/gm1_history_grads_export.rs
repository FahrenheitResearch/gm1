//! Export one published GM1 history frame as CM1 r21.1 `output_format=1`.
//!
//! CM1's own history format is a GrADS direct-access binary32 `.dat` plus a
//! `.ctl` descriptor (`src/writeout.F:8761-8995` writes the records,
//! `:9265-9528` the descriptor).  Nothing in the meteorological ecosystem can
//! open GM1's private `CM1FHS1` envelope; every tool in it can open this.
//!
//! The conversion is a copy, not a transform.  GM1's arena already holds each
//! carrier as contiguous `[z][y][x]` little-endian binary32 with x fastest,
//! which is exactly GrADS record order, so a carrier becomes a GrADS variable
//! by being written where the descriptor says it is.  The three things that
//! are not a straight copy are stated in the descriptor and the receipt:
//!
//!   * carriers are split by C-grid stagger into CM1's own four files
//!     (`_s`, `_u`, `_v`, `_w`), because GrADS has one grid per descriptor
//!     and CM1 splits them the same way;
//!   * the six-species `q3d` block becomes CM1's six named variables;
//!   * a horizontally uniform base-state profile is broadcast to the 3-D
//!     record CM1 writes, so `th0` in a GM1 export and `th0` in a native CM1
//!     run are the same shape to the same reader.
//!
//! Names are CM1 `writeout.F` `name_output` strings.  Where GM1's carrier name
//! differs, the map below is explicit and one-way; in particular `th3d` is
//! CM1's `thpert` (the potential-temperature PERTURBATION, `solve1.F:893`) and
//! `pp3d` is CM1's `pipert`, and exporting either under the absolute name
//! would hand a CM1 tool a field that is wrong by the whole base state.

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const ABI: &str = "gpuwm-cm1-history-grads-export/v1";
const FIELD_TABLE: &str = "history-field-table.tsv";
const GRADS_UNDEF: f32 = -999999.0;
const CLAIM_LIMIT: &str = "one published GM1 history frame rewritten as CM1 r21.1 output_format=1 GrADS records and descriptors; a format conversion of the frame's own binary32 words, not a re-derivation, a regrid, an interpolation to height, or a forecast claim";

/// GM1 carrier name -> CM1 `writeout.F` `name_output` string.
fn cm1_name(carrier: &str) -> &str {
    match carrier {
        // solve1.F:893 forms total theta as th0+tha; th3d IS tha.
        "th3d" => "thpert",
        "pp3d" => "pipert",
        "u3d" => "u",
        "v3d" => "v",
        "w3d" => "w",
        "closure.tkea" => "tke",
        "terrain_msl" => "zs",
        other => other,
    }
}

const SPECIES: [&str; 6] = ["qv", "qc", "qr", "qi", "qs", "qg"];

#[derive(Clone)]
struct Row {
    name: String,
    meaning: String,
    units: String,
    stagger: String,
    shape: Vec<usize>,
    arena: PathBuf,
    offset: u64,
}

/// The stagger the publisher gives CM1's ground-fixed swath twins
/// (`sws2`..`shs2`, writeout.F:568-720).  They sit on a ground-anchored
/// canvas larger than the model grid, and a frame does not yet record the
/// canvas origin, so no GrADS descriptor of this grid can place them.
const GROUND_CANVAS_STAGGER: &str = "ground-canvas";

impl Row {
    fn words(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Family {
    Scalar,
    U,
    V,
    W,
}

impl Family {
    fn suffix(self) -> &'static str {
        match self {
            Self::Scalar => "s",
            Self::U => "u",
            Self::V => "v",
            Self::W => "w",
        }
    }
}

struct Geometry {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
}

/// One GrADS variable: a name, a level count (0 means a single 2-D surface),
/// and where its words come from.
struct Emitted {
    name: String,
    levels: usize,
    description: String,
    units: String,
    source: Source,
}

enum Source {
    /// Copy `words` binary32 words starting at `offset` in `arena`.
    Arena {
        arena: PathBuf,
        offset: u64,
        words: usize,
    },
    /// Broadcast an `nz`-long column across every horizontal point.
    Profile {
        arena: PathBuf,
        offset: u64,
        levels: usize,
        plane: usize,
    },
}

fn fail(message: String) -> ExitCode {
    eprintln!("gm1-history-grads-export: {message}");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.iter().any(|value| value == "--abi") {
        println!("{ABI}");
        return ExitCode::SUCCESS;
    }
    let mut frame: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut prefix = String::from("cm1out");
    let mut dx: Option<f64> = None;
    let mut dy: Option<f64> = None;
    let mut dz: Option<f64> = None;
    let mut index = 0usize;
    while index < arguments.len() {
        let key = arguments[index].as_str();
        let mut value = || -> Option<String> {
            index += 1;
            arguments.get(index).cloned()
        };
        match key {
            "--frame" => frame = value().map(PathBuf::from),
            "--out" => out = value().map(PathBuf::from),
            "--prefix" => {
                if let Some(text) = value() {
                    prefix = text;
                }
            }
            "--dx" => dx = value().and_then(|text| text.parse().ok()),
            "--dy" => dy = value().and_then(|text| text.parse().ok()),
            "--dz" => dz = value().and_then(|text| text.parse().ok()),
            other => return fail(format!("unknown argument {other}")),
        }
        index += 1;
    }
    let (frame, out) = match (frame, out) {
        (Some(frame), Some(out)) => (frame, out),
        _ => {
            return fail(
                "usage: --frame FRAME_DIR --out OUT_DIR --dx M --dy M --dz M [--prefix NAME]"
                    .into(),
            )
        }
    };
    let (dx, dy, dz) = match (dx, dy, dz) {
        (Some(dx), Some(dy), Some(dz)) if dx > 0.0 && dy > 0.0 && dz > 0.0 => (dx, dy, dz),
        _ => {
            return fail(
                "--dx, --dy and --dz are required and must be positive: the GrADS \
                 descriptor's xdef/ydef/zdef are physical coordinates, and a \
                 descriptor written with a guessed spacing would place every \
                 field on a grid the run never used"
                    .into(),
            )
        }
    };
    match run(&frame, &out, &prefix, dx, dy, dz) {
        Ok(receipt) => {
            println!("{receipt}");
            ExitCode::SUCCESS
        }
        Err(message) => fail(message),
    }
}

fn run(
    frame: &Path,
    out: &Path,
    prefix: &str,
    dx: f64,
    dy: f64,
    dz: f64,
) -> Result<String, String> {
    let rows = read_field_table(frame, &frame.join(FIELD_TABLE))?;
    let geometry = derive_geometry(&rows, dx, dy, dz)?;
    std::fs::create_dir_all(out).map_err(|error| format!("cannot create {out:?}: {error}"))?;
    let mut families: BTreeMap<Family, Vec<Emitted>> = BTreeMap::new();
    let mut not_exported = Vec::new();
    for row in &rows {
        if row.stagger == GROUND_CANVAS_STAGGER {
            // Named in the receipt rather than refused: every other carrier
            // of the frame is on the model grid and exports exactly.
            not_exported.push(format!(
                "{{\"name\":{},\"stagger\":{},\"shape\":[{}],\"reason\":{}}}",
                json_string(&row.name),
                json_string(&row.stagger),
                row.shape
                    .iter()
                    .map(|extent| extent.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                json_string(
                    "a ground-anchored swath canvas, not one of this grid's C-grid \
                     families; the frame does not record the canvas origin, so no \
                     descriptor of this grid can place it"
                )
            ));
            continue;
        }
        let (family, emitted) = classify(row, &geometry)?;
        families.entry(family).or_default().extend(emitted);
    }
    let mut written = Vec::new();
    for (family, variables) in &families {
        let (extents, levels_axis) = family_extent(*family, &geometry);
        let data = out.join(format!("{prefix}_{}.dat", family.suffix()));
        let descriptor = out.join(format!("{prefix}_{}.ctl", family.suffix()));
        let bytes = write_data(&data, variables, extents)?;
        write_ctl(
            &descriptor,
            &data,
            variables,
            *family,
            &geometry,
            levels_axis,
        )?;
        written.push(format!(
            "{{\"grid\":\"{}\",\"data\":{},\"descriptor\":{},\"payload_bytes\":{},\"variables\":[{}]}}",
            family.suffix(),
            json_string(&data.to_string_lossy()),
            json_string(&descriptor.to_string_lossy()),
            bytes,
            variables
                .iter()
                .map(|variable| json_string(&variable.name))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    Ok(format!(
        "{{\"schema\":\"gm1.history-grads-export/v1\",\"abi\":{},\"frame\":{},\"arena\":{},\"nx\":{},\"ny\":{},\"nz\":{},\"dx_m\":{},\"dy_m\":{},\"dz_m\":{},\"files\":[{}],\"not_exported\":[{}],\"claim_limit\":{}}}",
        json_string(ABI),
        json_string(&frame.to_string_lossy()),
        json_string(&rows[0].arena.to_string_lossy()),
        geometry.nx,
        geometry.ny,
        geometry.nz,
        dx,
        dy,
        dz,
        written.join(","),
        not_exported.join(","),
        json_string(CLAIM_LIMIT)
    ))
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Where column 7 of a field table puts the arena, for a frame read at `frame`.
///
/// The publisher writes the arena's file name relative to the frame directory
/// (cartesian_history.py `_field_table_bytes`), so a frame reads the same
/// after `--integrate` renames it out of `.incomplete/` and after it is
/// copied to another machine.  Frames published before that wrote the
/// absolute staging path, which no longer exists once the frame is renamed;
/// for those, the frame's own file of the same name is the arena that path
/// named, and it is read instead.
fn resolve_arena(frame: &Path, column: &str) -> PathBuf {
    let named = PathBuf::from(column);
    if named.is_relative() {
        return frame.join(named);
    }
    if !named.exists() {
        if let Some(name) = named.file_name() {
            let local = frame.join(name);
            if local.is_file() {
                return local;
            }
        }
    }
    named
}

fn read_field_table(frame: &Path, path: &Path) -> Result<Vec<Row>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read the field table {path:?}: {error}"))?;
    let mut rows = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let columns: Vec<&str> = line.split('\t').collect();
        if columns.len() != 8 {
            return Err(format!(
                "field table line {} has {} columns; the publisher writes exactly 8",
                number + 1,
                columns.len()
            ));
        }
        let shape = columns[4]
            .split(',')
            .map(|value| value.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("field table line {}: bad shape: {error}", number + 1))?;
        let offset = columns[7]
            .trim()
            .parse::<u64>()
            .map_err(|error| format!("field table line {}: bad offset: {error}", number + 1))?;
        rows.push(Row {
            name: columns[0].to_string(),
            meaning: columns[1].to_string(),
            units: columns[2].to_string(),
            stagger: columns[3].to_string(),
            shape,
            arena: resolve_arena(frame, columns[6]),
            offset,
        });
    }
    if rows.is_empty() {
        return Err("the field table is empty".into());
    }
    Ok(rows)
}

/// The grid comes from the frame's own carriers, not from an argument: a
/// mass-point volume is (nz, ny, nx) by construction, so a mismatch between
/// the declared spacing and the declared extent cannot silently reshape it.
fn derive_geometry(rows: &[Row], dx: f64, dy: f64, dz: f64) -> Result<Geometry, String> {
    let mass = rows
        .iter()
        .find(|row| row.name == "prs")
        .ok_or_else(|| "the frame carries no prs volume to take the grid extent from".to_string())?;
    if mass.shape.len() != 3 {
        return Err("prs is not a three-dimensional carrier".into());
    }
    Ok(Geometry {
        nx: mass.shape[2],
        ny: mass.shape[1],
        nz: mass.shape[0],
        dx,
        dy,
        dz,
    })
}

fn classify(row: &Row, geometry: &Geometry) -> Result<(Family, Vec<Emitted>), String> {
    let (nx, ny, nz) = (geometry.nx, geometry.ny, geometry.nz);
    let plane = nx * ny;
    let name = cm1_name(&row.name).to_string();
    match row.shape.as_slice() {
        // The six-species block becomes CM1's six named variables.
        [6, z, y, x] if (*z, *y, *x) == (nz, ny, nx) => Ok((
            Family::Scalar,
            SPECIES
                .iter()
                .enumerate()
                .map(|(index, species)| Emitted {
                    name: (*species).to_string(),
                    levels: nz,
                    description: format!("{species} mixing ratio"),
                    units: row.units.clone(),
                    source: Source::Arena {
                        arena: row.arena.clone(),
                        offset: row.offset + (index * nz * plane * 4) as u64,
                        words: nz * plane,
                    },
                })
                .collect(),
        )),
        [z, y, x] if (*z, *y, *x) == (nz, ny, nx) => Ok((
            Family::Scalar,
            vec![Emitted {
                name,
                levels: nz,
                description: row.meaning.clone(),
                units: row.units.clone(),
                source: Source::Arena {
                    arena: row.arena.clone(),
                    offset: row.offset,
                    words: row.words(),
                },
            }],
        )),
        [z, y, x] if (*z, *y, *x) == (nz, ny, nx + 1) => Ok((
            Family::U,
            vec![Emitted {
                name,
                levels: nz,
                description: row.meaning.clone(),
                units: row.units.clone(),
                source: Source::Arena {
                    arena: row.arena.clone(),
                    offset: row.offset,
                    words: row.words(),
                },
            }],
        )),
        [z, y, x] if (*z, *y, *x) == (nz, ny + 1, nx) => Ok((
            Family::V,
            vec![Emitted {
                name,
                levels: nz,
                description: row.meaning.clone(),
                units: row.units.clone(),
                source: Source::Arena {
                    arena: row.arena.clone(),
                    offset: row.offset,
                    words: row.words(),
                },
            }],
        )),
        [z, y, x] if (*z, *y, *x) == (nz + 1, ny, nx) => Ok((
            Family::W,
            vec![Emitted {
                name,
                levels: nz + 1,
                description: row.meaning.clone(),
                units: row.units.clone(),
                source: Source::Arena {
                    arena: row.arena.clone(),
                    offset: row.offset,
                    words: row.words(),
                },
            }],
        )),
        // A 2-D surface field: GrADS writes it with a level count of 0.
        [y, x] if (*y, *x) == (ny, nx) => Ok((
            Family::Scalar,
            vec![Emitted {
                name,
                levels: 0,
                description: row.meaning.clone(),
                units: row.units.clone(),
                source: Source::Arena {
                    arena: row.arena.clone(),
                    offset: row.offset,
                    words: row.words(),
                },
            }],
        )),
        // A horizontally uniform base-state column: broadcast to CM1's own
        // three-dimensional record so the same reader sees the same shape.
        [z] if *z == nz => Ok((
            Family::Scalar,
            vec![Emitted {
                name,
                levels: nz,
                description: format!("{} (horizontally uniform)", row.meaning),
                units: row.units.clone(),
                source: Source::Profile {
                    arena: row.arena.clone(),
                    offset: row.offset,
                    levels: nz,
                    plane,
                },
            }],
        )),
        other => Err(format!(
            "carrier {} has extent {other:?}, which is none of this grid's C-grid \
             families at nx={nx}, ny={ny}, nz={nz}; exporting it would place its \
             words on a grid GrADS would read as another field",
            row.name
        )),
    }
}

fn family_extent(family: Family, geometry: &Geometry) -> ((usize, usize), usize) {
    match family {
        Family::Scalar => ((geometry.nx, geometry.ny), geometry.nz),
        Family::U => ((geometry.nx + 1, geometry.ny), geometry.nz),
        Family::V => ((geometry.nx, geometry.ny + 1), geometry.nz),
        Family::W => ((geometry.nx, geometry.ny), geometry.nz + 1),
    }
}

fn write_data(
    path: &Path,
    variables: &[Emitted],
    extents: (usize, usize),
) -> Result<u64, String> {
    let mut sink = BufWriter::new(
        File::create(path).map_err(|error| format!("cannot create {path:?}: {error}"))?,
    );
    let plane = extents.0 * extents.1;
    let mut total = 0u64;
    for variable in variables {
        match &variable.source {
            Source::Arena {
                arena,
                offset,
                words,
            } => {
                let expected = variable.levels.max(1) * plane;
                if *words != expected {
                    return Err(format!(
                        "variable {} declares {} levels on a {}-point plane ({} words) \
                         but its carrier holds {words}",
                        variable.name,
                        variable.levels.max(1),
                        plane,
                        expected
                    ));
                }
                total += copy_words(arena, *offset, *words, &mut sink)?;
            }
            Source::Profile {
                arena,
                offset,
                levels,
                plane: declared,
            } => {
                if *declared != plane {
                    return Err(format!(
                        "profile {} was classified on a {declared}-point plane but \
                         its file's plane is {plane}",
                        variable.name
                    ));
                }
                let column = read_words(arena, *offset, *levels)?;
                for word in &column {
                    let bytes = word.to_le_bytes();
                    for _ in 0..plane {
                        sink.write_all(&bytes)
                            .map_err(|error| format!("cannot write {path:?}: {error}"))?;
                    }
                }
                total += (levels * plane * 4) as u64;
            }
        }
    }
    sink.flush()
        .map_err(|error| format!("cannot flush {path:?}: {error}"))?;
    Ok(total)
}

fn read_words(arena: &Path, offset: u64, words: usize) -> Result<Vec<f32>, String> {
    let mut file =
        File::open(arena).map_err(|error| format!("cannot open the arena {arena:?}: {error}"))?;
    seek(&mut file, offset)?;
    let mut raw = vec![0u8; words * 4];
    file.read_exact(&mut raw)
        .map_err(|error| format!("cannot read {words} words from {arena:?}: {error}"))?;
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn copy_words(
    arena: &Path,
    offset: u64,
    words: usize,
    sink: &mut BufWriter<File>,
) -> Result<u64, String> {
    let mut file =
        File::open(arena).map_err(|error| format!("cannot open the arena {arena:?}: {error}"))?;
    seek(&mut file, offset)?;
    let mut remaining = words * 4;
    let mut buffer = vec![0u8; 1 << 20];
    while remaining > 0 {
        let take = remaining.min(buffer.len());
        file.read_exact(&mut buffer[..take])
            .map_err(|error| format!("cannot read from {arena:?}: {error}"))?;
        sink.write_all(&buffer[..take])
            .map_err(|error| format!("cannot write the GrADS record: {error}"))?;
        remaining -= take;
    }
    Ok((words * 4) as u64)
}

fn seek(file: &mut File, offset: u64) -> Result<(), String> {
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))
        .map(|_| ())
        .map_err(|error| format!("cannot seek to {offset}: {error}"))
}

/// `writeout.F:9265-9528`'s descriptor, in its own record order and formats.
fn write_ctl(
    path: &Path,
    data: &Path,
    variables: &[Emitted],
    family: Family,
    geometry: &Geometry,
    levels_axis: usize,
) -> Result<(), String> {
    let ((nx, ny), _) = (family_extent(family, geometry).0, ());
    let file_name = data
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "the data file has no usable name".to_string())?;
    let mut text = String::new();
    text.push_str(&format!("dset ^{file_name}\n"));
    text.push_str(
        "title gpuwm-cm1 history frame exported as CM1 r21.1 output_format=1; \
         units of x,y,z are meters\n",
    );
    text.push_str(&format!("undef {:10.1}\n", GRADS_UNDEF));
    // CM1 writes face coordinates for a staggered axis and centres otherwise
    // (writeout.F:9386-9424).
    let (x0, y0) = match family {
        Family::U => (0.0, 0.5 * geometry.dy),
        Family::V => (0.5 * geometry.dx, 0.0),
        _ => (0.5 * geometry.dx, 0.5 * geometry.dy),
    };
    text.push_str(&format!("xdef {:6} linear {:14.6e} {:14.6e}\n", nx, x0, geometry.dx));
    text.push_str(&format!("ydef {:6} linear {:14.6e} {:14.6e}\n", ny, y0, geometry.dy));
    let z0 = if matches!(family, Family::W) {
        0.0
    } else {
        0.5 * geometry.dz
    };
    text.push_str(&format!(
        "zdef {:6} linear {:14.6e} {:14.6e}\n",
        levels_axis, z0, geometry.dz
    ));
    text.push_str("tdef          1 linear 00:00Z01JAN0001 1YR\n");
    text.push_str(&format!("vars {:6}\n", variables.len()));
    for variable in variables {
        let description: String = variable
            .description
            .chars()
            .filter(|character| *character != '\n' && *character != '\r')
            .take(60)
            .collect();
        let units: String = variable
            .units
            .chars()
            .filter(|character| *character != '\n' && *character != '\r')
            .take(12)
            .collect();
        text.push_str(&format!(
            "{:<12} {:6} 99 {:<60} {:<12}\n",
            variable.name, variable.levels, description, units
        ));
    }
    text.push_str("endvars\n");
    std::fs::write(path, text).map_err(|error| format!("cannot write {path:?}: {error}"))
}
