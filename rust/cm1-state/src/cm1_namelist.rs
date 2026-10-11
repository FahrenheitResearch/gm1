//! Exhaustive CM1 r21.1 NAMELIST identity transport and support admission.
//!
//! This module parses every scalar field declared by the pristine authority's
//! 25 NAMELIST groups.  Parsing is deliberately separate from execution:
//! arbitrary configurations can be represented and round-tripped exactly at
//! their CM1 storage precision, while forecast checkpoint admission remains
//! fail-closed unless a complete branch-specific carrier contract exists.

use crate::forecast::ComponentBlob;
use crate::json::{self, JsonValue};
use crate::sha256;
use crate::{Result, StateError};
use std::collections::{BTreeMap, BTreeSet};

pub const CM1_NAMELIST_SUPPORT_SCHEMA: &str = "gm1.cm1-r21.1-configuration-support/v1";
pub const CM1_NAMELIST_DOCUMENT_SCHEMA: &str = "gm1.cm1-r21.1-namelist-document/v1";
pub const CM1_CONFIGURATION_BINDING_SCHEMA: &str = "gm1.cm1-r21.1-configuration-binding/v1";
pub const CM1_NAMELIST_INSPECTION_SCHEMA: &str = "gm1.cm1-r21.1-namelist-inspection/v1";
pub const CM1_CONFIGURATION_BLOB_NAME: &str = "cm1_configuration";
pub const CM1_NAMELIST_SUPPORT_SHA256: &str =
    "ea89820ca430ca7dfce385bfc0bc8fe10288c3ccf460f5d32a435a366c33605b";
pub const CM1_NAMELIST_GROUP_COUNT: usize = 25;
pub const CM1_NAMELIST_FIELD_COUNT: usize = 583;
pub const CM1_CORE_NAMELIST_FIELD_COUNT: usize = 341;
pub const CM1_RESTART_WRITE_ARGUMENT_COUNT: usize = 290;
pub const CM1_NAMELIST_MAX_NBYTES: usize = 4 * 1024 * 1024;
pub const CM1_CONFIGURATION_BINDING_MAX_NBYTES: usize = 512 * 1024;
pub const CM1_ARBITRARY_CHECKPOINT_CLAIM_LIMIT: &str = "all 25 CM1 r21.1 NAMELIST groups and 583 declared scalar fields parse and precision-normalize for identity transport; arbitrary branch execution, native restart compatibility, MPI remapping, nesting, output production, and arbitrary-config trajectory restart remain unsupported";

const SUPPORT_BYTES: &[u8] = include_bytes!("../schema/cm1-r21.1-namelist-support.json");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cm1NamelistType {
    Integer,
    Logical,
    Real32,
    Real64,
    String,
}

impl Cm1NamelistType {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "integer" => Ok(Self::Integer),
            "logical" => Ok(Self::Logical),
            "real32" => Ok(Self::Real32),
            "real64" => Ok(Self::Real64),
            "string" => Ok(Self::String),
            _ => Err(invalid(format!(
                "CM1 support schema contains unknown field type {value:?}"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Integer => "integer",
            Self::Logical => "logical",
            Self::Real32 => "real32",
            Self::Real64 => "real64",
            Self::String => "string",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Cm1NamelistValue {
    Integer(i64),
    Logical(bool),
    Real32(u32),
    Real64(u64),
    String(String),
}

impl Cm1NamelistValue {
    pub fn field_type(&self) -> Cm1NamelistType {
        match self {
            Self::Integer(_) => Cm1NamelistType::Integer,
            Self::Logical(_) => Cm1NamelistType::Logical,
            Self::Real32(_) => Cm1NamelistType::Real32,
            Self::Real64(_) => Cm1NamelistType::Real64,
            Self::String(_) => Cm1NamelistType::String,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_logical(&self) -> Option<bool> {
        match self {
            Self::Logical(value) => Some(*value),
            _ => None,
        }
    }

    pub fn real32_bits(&self) -> Option<u32> {
        match self {
            Self::Real32(value) => Some(*value),
            _ => None,
        }
    }

    pub fn real64_bits(&self) -> Option<u64> {
        match self {
            Self::Real64(value) => Some(*value),
            _ => None,
        }
    }

    fn json(&self) -> JsonValue {
        let value = match self {
            Self::Integer(value) => JsonValue::Number(value.to_string()),
            Self::Logical(value) => JsonValue::Bool(*value),
            Self::Real32(bits) => string(&format!("{bits:08x}")),
            Self::Real64(bits) => string(&format!("{bits:016x}")),
            Self::String(value) => string(value),
        };
        object(vec![
            ("type", string(self.field_type().as_str())),
            ("value", value),
        ])
    }

    fn fortran(&self) -> String {
        match self {
            Self::Integer(value) => value.to_string(),
            Self::Logical(true) => ".true.".to_string(),
            Self::Logical(false) => ".false.".to_string(),
            Self::Real32(bits) => f32::from_bits(*bits).to_string(),
            Self::Real64(bits) => f64::from_bits(*bits).to_string(),
            Self::String(value) => format!("'{}'", value.replace('\'', "''")),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cm1NamelistDocument {
    groups: BTreeMap<String, BTreeMap<String, Cm1NamelistValue>>,
    canonical_json: Vec<u8>,
    sha256: String,
}

impl Cm1NamelistDocument {
    pub fn groups(&self) -> &BTreeMap<String, BTreeMap<String, Cm1NamelistValue>> {
        &self.groups
    }

    pub fn value(&self, group: &str, field: &str) -> Option<&Cm1NamelistValue> {
        self.groups
            .get(&group.to_ascii_lowercase())
            .and_then(|values| values.get(&field.to_ascii_lowercase()))
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn assignment_count(&self) -> usize {
        self.groups.values().map(BTreeMap::len).sum()
    }

    pub fn canonical_json(&self) -> &[u8] {
        &self.canonical_json
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn canonical_namelist(&self) -> String {
        let mut output = String::new();
        for (group, values) in &self.groups {
            output.push('&');
            output.push_str(group);
            output.push('\n');
            for (name, value) in values {
                output.push(' ');
                output.push_str(name);
                output.push_str(" = ");
                output.push_str(&value.fortran());
                output.push_str(",\n");
            }
            output.push_str("/\n");
        }
        output
    }

    pub fn verify_semantic_roundtrip(&self) -> Result<()> {
        let reparsed = parse_cm1_r21_namelist(self.canonical_namelist().as_bytes())?;
        if reparsed != *self {
            return Err(invalid(
                "CM1 canonical NAMELIST does not preserve every typed assignment",
            ));
        }
        Ok(())
    }

    pub fn checkpoint_blockers(&self) -> Vec<String> {
        let mut blockers =
            BTreeSet::from(["CM1-ARBITRARY-CONFIG-CARRIER-CONTRACT-MISSING".to_string()]);
        for (group, field, predicate, code) in [
            ("param2", "adapt_dt", 0, "CM1-ADAPTIVE-CLOCK-STATE-MISSING"),
            ("param2", "irst", 0, "CM1-NATIVE-RESTART-BRIDGE-MISSING"),
            ("param11", "radopt", 0, "CM1-RADIATION-STATE-MISSING"),
            ("param12", "sfcmodel", 0, "CM1-SURFACE-MODEL-STATE-MISSING"),
            ("param12", "oceanmodel", 0, "CM1-OCEAN-MODEL-STATE-MISSING"),
            (
                "param20",
                "ib_init",
                0,
                "CM1-IMMERSED-BOUNDARY-STATE-MISSING",
            ),
        ] {
            if self
                .value(group, field)
                .and_then(Cm1NamelistValue::as_i64)
                .is_some_and(|value| value != predicate)
            {
                blockers.insert(code.to_string());
            }
        }
        for (group, field, code) in [
            (
                "param15",
                "do_adapt_move",
                "CM1-ADAPTIVE-MOVE-STATE-MISSING",
            ),
            ("param18", "do_recycle_w", "CM1-RECYCLE-STATE-MISSING"),
            ("param18", "do_recycle_s", "CM1-RECYCLE-STATE-MISSING"),
            ("param18", "do_recycle_e", "CM1-RECYCLE-STATE-MISSING"),
            ("param18", "do_recycle_n", "CM1-RECYCLE-STATE-MISSING"),
            ("param19", "do_lsnudge", "CM1-LSNUDGE-STATE-MISSING"),
            ("param20", "do_ib", "CM1-IMMERSED-BOUNDARY-STATE-MISSING"),
        ] {
            if self
                .value(group, field)
                .and_then(Cm1NamelistValue::as_logical)
                == Some(true)
            {
                blockers.insert(code.to_string());
            }
        }
        blockers.into_iter().collect()
    }

    pub fn require_arbitrary_checkpoint_supported(&self) -> Result<()> {
        let blockers = self.checkpoint_blockers();
        Err(invalid(format!(
            "arbitrary CM1 r21.1 checkpoint execution is fail-closed: {}",
            blockers.join(",")
        )))
    }

    pub fn configuration_binding(&self) -> Result<Cm1ConfigurationBinding> {
        Cm1ConfigurationBinding::new(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cm1ConfigurationBinding {
    bytes: Vec<u8>,
    sha256: String,
    document_sha256: String,
    blockers: Vec<String>,
}

impl Cm1ConfigurationBinding {
    fn new(document: &Cm1NamelistDocument) -> Result<Self> {
        let blockers = document.checkpoint_blockers();
        let value = object(vec![
            ("blockers", string_array_json(&blockers)),
            ("claim_limit", string(CM1_ARBITRARY_CHECKPOINT_CLAIM_LIMIT)),
            (
                "document",
                json::parse(document.canonical_json()).map_err(invalid)?,
            ),
            ("document_sha256", string(document.sha256())),
            ("execution_admission", string("fail-closed")),
            ("model", string("GM1")),
            ("schema", string(CM1_CONFIGURATION_BINDING_SCHEMA)),
            ("support_schema_sha256", string(CM1_NAMELIST_SUPPORT_SHA256)),
        ]);
        let bytes = value.canonical_bytes();
        if bytes.len() > CM1_CONFIGURATION_BINDING_MAX_NBYTES {
            return Err(invalid("CM1 configuration binding exceeds its v1 size cap"));
        }
        Ok(Self {
            sha256: sha256::digest_hex(&bytes),
            document_sha256: document.sha256().to_string(),
            bytes,
            blockers,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn document_sha256(&self) -> &str {
        &self.document_sha256
    }

    pub fn blockers(&self) -> &[String] {
        &self.blockers
    }

    pub fn component_blob(&self) -> ComponentBlob {
        ComponentBlob::new(
            CM1_CONFIGURATION_BLOB_NAME,
            CM1_CONFIGURATION_BINDING_SCHEMA,
            self.bytes.clone(),
        )
    }
}

pub fn cm1_r21_namelist_support_json() -> Result<&'static [u8]> {
    validate_support_schema()?;
    Ok(SUPPORT_BYTES)
}

/// What the parser does with an assignment r21.1 does not declare.
///
/// `Refuse` is the identity-transport contract: a document that names a field
/// r21.1 cannot hold is not an r21.1 document.  `Report` is the importer's
/// contract: a user's file from another CM1 release (an r22.0 key, a key
/// renamed since r18) is parsed completely, and every such assignment comes
/// back as a row the caller must classify, so none of them vanishes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cm1UnknownPolicy {
    Refuse,
    Report,
}

impl Cm1UnknownPolicy {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "refuse" => Ok(Self::Refuse),
            "report" => Ok(Self::Report),
            _ => Err(invalid(format!(
                "unknown-field policy must be 'refuse' or 'report', not {value:?}"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Refuse => "refuse",
            Self::Report => "report",
        }
    }
}

/// One assignment r21.1 does not declare, as the file wrote it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cm1UnknownAssignment {
    pub group: String,
    pub field: String,
    /// Whether r21.1 declares the group.  gfortran skips a group it is not
    /// asked to read, but stops on an undeclared name inside one it reads.
    pub group_declared: bool,
    pub text: String,
}

/// One line of text outside every group, which gfortran's NAMELIST read skips
/// while it looks for the group it was asked for (`!` comments excluded).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cm1OutsideText {
    /// 1-based line number in the file.
    pub line: usize,
    pub text: String,
}

/// A parsed document plus the literal text of every assignment, the
/// assignments r21.1 does not declare, the order the groups appear in, and
/// any text outside the groups.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cm1NamelistInspection {
    document: Cm1NamelistDocument,
    literals: BTreeMap<String, BTreeMap<String, String>>,
    unknown: Vec<Cm1UnknownAssignment>,
    group_order: Vec<String>,
    outside: Vec<Cm1OutsideText>,
    policy: Cm1UnknownPolicy,
}

impl Cm1NamelistInspection {
    pub fn document(&self) -> &Cm1NamelistDocument {
        &self.document
    }

    pub fn unknown(&self) -> &[Cm1UnknownAssignment] {
        &self.unknown
    }

    /// Every group, declared or not, in the order the file opens it.
    /// cm1.F:480-482 reads param0, param2 and param8 one after another
    /// without rewinding, so their order decides whether CM1 starts.
    pub fn group_order(&self) -> &[String] {
        &self.group_order
    }

    /// Text outside every group.  Only `Report` admits it; `Refuse` stops
    /// on it, as the identity transport always has.
    pub fn outside(&self) -> &[Cm1OutsideText] {
        &self.outside
    }

    pub fn literal(&self, group: &str, field: &str) -> Option<&str> {
        self.literals
            .get(group)
            .and_then(|values| values.get(field))
            .map(String::as_str)
    }

    /// The `gm1.cm1-r21.1-namelist-inspection/v1` JSON: the canonical
    /// document, each declared value decoded for a reader that does not
    /// decode binary32 words, and every undeclared assignment in file order.
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        let document = json::parse(self.document.canonical_json()).map_err(invalid)?;
        let mut groups = Vec::new();
        for (group, values) in self.document.groups() {
            let mut fields = Vec::new();
            for (field, value) in values {
                let text = self.literal(group, field).unwrap_or_default();
                fields.push((
                    field.clone(),
                    object(vec![
                        ("text", string(text)),
                        ("type", string(value.field_type().as_str())),
                        ("value", decoded_json(value, text)?),
                    ]),
                ));
            }
            groups.push((group.clone(), JsonValue::Object(fields)));
        }
        let unknown = self
            .unknown
            .iter()
            .map(|row| {
                let (value_type, value) = guessed_json(&row.text);
                object(vec![
                    ("field", string(&row.field)),
                    ("group", string(&row.group)),
                    ("group_declared", JsonValue::Bool(row.group_declared)),
                    ("text", string(&row.text)),
                    ("value", value),
                    ("value_type", string(value_type)),
                ])
            })
            .collect();
        let group_order = self.group_order.iter().map(|group| string(group)).collect();
        let outside = self
            .outside
            .iter()
            .map(|row| {
                object(vec![
                    ("line", JsonValue::Number(row.line.to_string())),
                    ("text", string(&row.text)),
                ])
            })
            .collect();
        Ok(object(vec![
            ("document", document),
            ("document_sha256", string(self.document.sha256())),
            ("group_order", JsonValue::Array(group_order)),
            ("outside_groups", JsonValue::Array(outside)),
            ("schema", string(CM1_NAMELIST_INSPECTION_SCHEMA)),
            ("support_schema_sha256", string(CM1_NAMELIST_SUPPORT_SHA256)),
            ("unknown", JsonValue::Array(unknown)),
            ("unknown_policy", string(self.policy.as_str())),
            ("values", JsonValue::Object(groups)),
        ])
        .canonical_bytes())
    }
}

pub fn parse_cm1_r21_namelist(bytes: &[u8]) -> Result<Cm1NamelistDocument> {
    inspect_cm1_r21_namelist(bytes, Cm1UnknownPolicy::Refuse).map(|inspection| inspection.document)
}

/// Parse a CM1 NAMELIST file with the Fortran list-directed grammar gfortran
/// reads: several assignments per line, blank or comma separators, a missing
/// final comma, `/` or `&end` terminators, `!` comments outside strings,
/// case-insensitive names and every logical spelling (`T`, `.T.`, `.TRUE.`,
/// `true`).  Declared fields are typed against the r21.1 inventory; the
/// canonical document is unchanged by how the file spelled them.
pub fn inspect_cm1_r21_namelist(
    bytes: &[u8],
    policy: Cm1UnknownPolicy,
) -> Result<Cm1NamelistInspection> {
    if bytes.is_empty() || bytes.len() > CM1_NAMELIST_MAX_NBYTES {
        return Err(invalid("CM1 NAMELIST length is outside its v1 cap"));
    }
    if !bytes.is_ascii() {
        return Err(invalid("CM1 NAMELIST must use ASCII source bytes"));
    }
    let inventory = support_inventory()?;
    let text =
        std::str::from_utf8(bytes).map_err(|_| invalid("CM1 NAMELIST is not valid UTF-8/ASCII"))?;
    let clean = strip_fortran_comments(text)?;
    let mut scanner = Scanner::new(&clean);
    let mut groups = BTreeMap::new();
    let mut literals: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    let mut unknown = Vec::new();
    let mut undeclared_groups = BTreeSet::new();
    let mut group_order = Vec::new();
    let mut outside = Vec::new();
    while scanner.skip_space_and_commas() {
        let at = scanner.position;
        let opened = scanner.consume_byte(b'&') || scanner.consume_byte(b'$');
        if policy == Cm1UnknownPolicy::Report && (!opened || scanner.next_word_is("end")) {
            // gfortran's NAMELIST read looks for '&name' (or '$name') and
            // skips everything else on the way, so text outside the groups --
            // a title line, a note between groups, a stray '&end' -- never
            // stops CM1.  Keep each line so a caller can show that CM1
            // ignores it.
            scanner.position = at;
            let skipped = scanner.outside_text();
            let first_line = clean[..at].matches('\n').count() + 1;
            for (offset, line) in skipped.split('\n').enumerate() {
                let line_text = line.trim();
                if !line_text.is_empty() {
                    outside.push(Cm1OutsideText {
                        line: first_line + offset,
                        text: line_text.to_string(),
                    });
                }
            }
            continue;
        }
        if !opened {
            return Err(invalid(format!(
                "CM1 NAMELIST byte {at}: expected '&group' at top level"
            )));
        }
        let group = scanner.identifier("NAMELIST group")?;
        group_order.push(group.clone());
        let fields = inventory.get(&group);
        if fields.is_none() {
            if policy == Cm1UnknownPolicy::Refuse {
                return Err(invalid(format!(
                    "CM1 NAMELIST group {group:?} is not declared by r21.1"
                )));
            }
            if !undeclared_groups.insert(group.clone()) {
                return Err(invalid(format!(
                    "CM1 NAMELIST group {group:?} appears more than once"
                )));
            }
        } else if groups.contains_key(&group) {
            return Err(invalid(format!(
                "CM1 NAMELIST group {group:?} appears more than once"
            )));
        }
        let mut values = BTreeMap::new();
        let mut texts = BTreeMap::new();
        loop {
            scanner.skip_space_and_commas();
            if scanner.consume_byte(b'/') {
                break;
            }
            if scanner.finished() {
                return Err(invalid(format!(
                    "CM1 NAMELIST group {group:?} is missing its '/' terminator"
                )));
            }
            if scanner.consume_byte(b'&') || scanner.consume_byte(b'$') {
                // The pre-Fortran-90 terminator gfortran still reads.
                let word = scanner.identifier("NAMELIST terminator")?;
                if word == "end" {
                    break;
                }
                return Err(invalid(format!(
                    "CM1 NAMELIST group {group:?} is missing its '/' terminator before '&{word}'"
                )));
            }
            let field = scanner.identifier("NAMELIST field")?;
            scanner.skip_space();
            scanner.expect_byte(b'=', "expected '=' after NAMELIST field")?;
            scanner.skip_space();
            let token = scanner.value_token()?;
            let declared = fields.and_then(|fields| fields.get(&field).copied());
            let Some(field_type) = declared else {
                if policy == Cm1UnknownPolicy::Refuse {
                    return Err(invalid(format!(
                        "CM1 NAMELIST field {group}.{field} is not declared by r21.1"
                    )));
                }
                // An undeclared name may carry a value list (another
                // release's array); keep every element so none is lost.
                let mut text = token.to_string();
                while let Some(element) = scanner.continuation_element()? {
                    text.push_str(", ");
                    text.push_str(element);
                }
                unknown.push(Cm1UnknownAssignment {
                    group: group.clone(),
                    field,
                    group_declared: fields.is_some(),
                    text,
                });
                continue;
            };
            if values.contains_key(&field) {
                return Err(invalid(format!(
                    "CM1 NAMELIST field {group}.{field} is assigned more than once"
                )));
            }
            let value = parse_value(field_type, token, &format!("{group}.{field}"))?;
            texts.insert(field.clone(), token.to_string());
            values.insert(field, value);
        }
        if fields.is_some() {
            groups.insert(group.clone(), values);
            literals.insert(group, texts);
        }
    }
    if groups.is_empty() && unknown.is_empty() {
        return Err(invalid("CM1 NAMELIST contains no groups"));
    }
    let canonical_json = document_json(&groups).canonical_bytes();
    let sha256 = sha256::digest_hex(&canonical_json);
    Ok(Cm1NamelistInspection {
        document: Cm1NamelistDocument {
            groups,
            canonical_json,
            sha256,
        },
        literals,
        unknown,
        group_order,
        outside,
        policy,
    })
}

fn decoded_json(value: &Cm1NamelistValue, text: &str) -> Result<JsonValue> {
    Ok(match value {
        Cm1NamelistValue::Integer(value) => JsonValue::Number(value.to_string()),
        Cm1NamelistValue::Logical(value) => JsonValue::Bool(*value),
        // The literal the file wrote, read as binary64: what a reader of the
        // file means by the number.  The document keeps the binary32 word
        // CM1 stores.  `{:?}` always spells a fraction or an exponent, so a
        // JSON reader never mistakes 1e30 for an integer.
        Cm1NamelistValue::Real32(_) | Cm1NamelistValue::Real64(_) => {
            let normalized = normalized_real(text, "decoded value")?;
            let parsed = normalized
                .parse::<f64>()
                .map_err(|_| invalid(format!("CM1 NAMELIST literal {text:?} is not a real")))?;
            JsonValue::Number(format!("{parsed:?}"))
        }
        Cm1NamelistValue::String(value) => string(value),
    })
}

/// A best reading of an undeclared value: the caller classifies the key, and
/// its value decides whether the setting matches CM1 r21.1.
fn guessed_json(text: &str) -> (&'static str, JsonValue) {
    let token = text.trim();
    if signed_decimal(token) {
        if let Ok(value) = token.parse::<i64>() {
            return ("integer", JsonValue::Number(value.to_string()));
        }
    }
    if let Some(value) = fortran_logical(token) {
        return ("logical", JsonValue::Bool(value));
    }
    if let Ok(value) = parse_string(token, "undeclared value") {
        return ("string", string(&value));
    }
    if let Ok(normalized) = normalized_real(token, "undeclared value") {
        if let Ok(value) = normalized.parse::<f64>() {
            if value.is_finite() {
                return ("real", JsonValue::Number(format!("{value:?}")));
            }
        }
    }
    ("unparsed", JsonValue::Null)
}

/// Fortran's L-edit reading, which list-directed and NAMELIST input use: an
/// optional period, then T or F, then anything (`T`, `.t.`, `.TRUE.`, `true`).
fn fortran_logical(token: &str) -> Option<bool> {
    let body = token.strip_prefix('.').unwrap_or(token);
    let mut characters = body.chars();
    let value = match characters.next()?.to_ascii_lowercase() {
        't' => true,
        'f' => false,
        _ => return None,
    };
    characters
        .all(|character| character.is_ascii_alphabetic() || character == '.')
        .then_some(value)
}

fn validate_support_schema() -> Result<()> {
    let actual = sha256::digest_hex(SUPPORT_BYTES);
    if actual != CM1_NAMELIST_SUPPORT_SHA256 {
        return Err(invalid(format!(
            "compiled CM1 support schema SHA-256 is {actual}; expected {CM1_NAMELIST_SUPPORT_SHA256}"
        )));
    }
    let value = json::parse(SUPPORT_BYTES).map_err(|error| {
        invalid(format!(
            "compiled CM1 support schema is invalid JSON: {error}"
        ))
    })?;
    if value.canonical_bytes() != SUPPORT_BYTES {
        return Err(invalid("compiled CM1 support schema is not canonical JSON"));
    }
    fixed_string(
        &value,
        "schema",
        CM1_NAMELIST_SUPPORT_SCHEMA,
        "CM1 support schema",
    )?;
    let counts = value
        .get("counts")
        .map_err(|error| invalid(format!("CM1 support schema counts: {error}")))?;
    for (name, expected) in [
        ("groups", CM1_NAMELIST_GROUP_COUNT),
        ("fields", CM1_NAMELIST_FIELD_COUNT),
        ("core_fields", CM1_CORE_NAMELIST_FIELD_COUNT),
        ("restart_write_arguments", CM1_RESTART_WRITE_ARGUMENT_COUNT),
    ] {
        let actual = counts
            .get(name)
            .and_then(JsonValue::as_u64)
            .map_err(|error| invalid(format!("CM1 support schema count {name}: {error}")))?;
        if actual != expected as u64 {
            return Err(invalid(format!(
                "CM1 support schema count {name} is {actual}; expected {expected}"
            )));
        }
    }
    Ok(())
}

fn support_inventory() -> Result<BTreeMap<String, BTreeMap<String, Cm1NamelistType>>> {
    validate_support_schema()?;
    let value = json::parse(SUPPORT_BYTES).expect("validated CM1 support JSON parses");
    let groups = value
        .get("groups")
        .and_then(JsonValue::as_array)
        .map_err(|error| invalid(format!("CM1 support schema groups: {error}")))?;
    let mut inventory = BTreeMap::new();
    let mut total = 0usize;
    for group in groups {
        let name = required_string(group, "name", "CM1 support group")?.to_string();
        let fields = group
            .get("fields")
            .and_then(JsonValue::as_array)
            .map_err(|error| invalid(format!("CM1 support group {name} fields: {error}")))?;
        let mut entries = BTreeMap::new();
        for field in fields {
            let field_name = required_string(field, "name", "CM1 support field")?.to_string();
            let field_type =
                Cm1NamelistType::parse(required_string(field, "type", "CM1 support field")?)?;
            if entries.insert(field_name.clone(), field_type).is_some() {
                return Err(invalid(format!(
                    "CM1 support field {name}.{field_name} is duplicated"
                )));
            }
        }
        total += entries.len();
        if inventory.insert(name.clone(), entries).is_some() {
            return Err(invalid(format!("CM1 support group {name} is duplicated")));
        }
    }
    if inventory.len() != CM1_NAMELIST_GROUP_COUNT || total != CM1_NAMELIST_FIELD_COUNT {
        return Err(invalid("CM1 support inventory counts changed after decode"));
    }
    Ok(inventory)
}

fn parse_value(field_type: Cm1NamelistType, token: &str, label: &str) -> Result<Cm1NamelistValue> {
    let token = token.trim();
    if token.is_empty() {
        return Err(invalid(format!("CM1 NAMELIST {label} has an empty value")));
    }
    match field_type {
        Cm1NamelistType::Integer => {
            if !signed_decimal(token) {
                return Err(invalid(format!(
                    "CM1 NAMELIST {label} must be a scalar decimal integer"
                )));
            }
            token
                .parse::<i64>()
                .map(Cm1NamelistValue::Integer)
                .map_err(|_| invalid(format!("CM1 NAMELIST {label} exceeds signed 64-bit range")))
        }
        Cm1NamelistType::Logical => fortran_logical(token)
            .map(Cm1NamelistValue::Logical)
            .ok_or_else(|| {
                invalid(format!(
                    "CM1 NAMELIST {label} must be a Fortran logical (.true., .false., T or F)"
                ))
            }),
        Cm1NamelistType::Real32 => {
            let normalized = normalized_real(token, label)?;
            let value = normalized.parse::<f32>().map_err(|_| {
                invalid(format!(
                    "CM1 NAMELIST {label} is not a default REAL literal"
                ))
            })?;
            if !value.is_finite() {
                return Err(invalid(format!(
                    "CM1 NAMELIST {label} must round to finite binary32"
                )));
            }
            Ok(Cm1NamelistValue::Real32(value.to_bits()))
        }
        Cm1NamelistType::Real64 => {
            let normalized = normalized_real(token, label)?;
            let value = normalized.parse::<f64>().map_err(|_| {
                invalid(format!(
                    "CM1 NAMELIST {label} is not a DOUBLE PRECISION literal"
                ))
            })?;
            if !value.is_finite() {
                return Err(invalid(format!(
                    "CM1 NAMELIST {label} must be finite binary64"
                )));
            }
            Ok(Cm1NamelistValue::Real64(value.to_bits()))
        }
        Cm1NamelistType::String => parse_string(token, label).map(Cm1NamelistValue::String),
    }
}

fn normalized_real(token: &str, label: &str) -> Result<String> {
    if token.contains('_') || token.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(invalid(format!(
            "CM1 NAMELIST {label} must be one scalar REAL literal without a kind suffix"
        )));
    }
    let mut normalized = token.replace(['d', 'D'], "e");
    if !normalized.contains(['e', 'E']) {
        let sign = normalized
            .bytes()
            .enumerate()
            .skip(1)
            .rev()
            .find(|(_, byte)| matches!(byte, b'+' | b'-'))
            .map(|(index, _)| index);
        if let Some(index) = sign {
            let mantissa = &normalized[..index];
            let exponent = &normalized[index + 1..];
            let unsigned_mantissa = mantissa
                .strip_prefix('+')
                .or_else(|| mantissa.strip_prefix('-'))
                .unwrap_or(mantissa);
            let mantissa_is_numeric = unsigned_mantissa
                .bytes()
                .all(|byte| byte.is_ascii_digit() || byte == b'.')
                && unsigned_mantissa
                    .bytes()
                    .filter(|byte| *byte == b'.')
                    .count()
                    <= 1
                && unsigned_mantissa.bytes().any(|byte| byte.is_ascii_digit());
            if mantissa_is_numeric
                && !exponent.is_empty()
                && exponent.bytes().all(|byte| byte.is_ascii_digit())
            {
                normalized.insert(index, 'e');
            }
        }
    }
    if !normalized.bytes().any(|byte| byte.is_ascii_digit()) {
        return Err(invalid(format!(
            "CM1 NAMELIST {label} REAL literal lacks digits"
        )));
    }
    Ok(normalized)
}

fn signed_decimal(value: &str) -> bool {
    let bytes = value.as_bytes();
    let first = usize::from(matches!(bytes.first(), Some(b'+') | Some(b'-')));
    first < bytes.len() && bytes[first..].iter().all(u8::is_ascii_digit)
}

fn parse_string(token: &str, label: &str) -> Result<String> {
    let bytes = token.as_bytes();
    if bytes.len() < 2 || !matches!(bytes[0], b'\'' | b'"') || bytes[bytes.len() - 1] != bytes[0] {
        return Err(invalid(format!(
            "CM1 NAMELIST {label} must be one quoted scalar string"
        )));
    }
    let quote = bytes[0] as char;
    let doubled = format!("{quote}{quote}");
    Ok(token[1..token.len() - 1].replace(&doubled, &quote.to_string()))
}

fn strip_fortran_comments(text: &str) -> Result<String> {
    let mut output = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let mut quote = None;
        for character in line.chars() {
            if character == '\n' {
                output.push(character);
                break;
            }
            match quote {
                Some(active) if character == active => {
                    quote = None;
                    output.push(character);
                }
                Some(_) => output.push(character),
                None if character == '\'' || character == '"' => {
                    quote = Some(character);
                    output.push(character);
                }
                None if character == '!' => break,
                None => output.push(character),
            }
        }
        if !output.ends_with('\n') {
            output.push('\n');
        }
    }
    Ok(output)
}

fn document_json(groups: &BTreeMap<String, BTreeMap<String, Cm1NamelistValue>>) -> JsonValue {
    let groups = groups
        .iter()
        .map(|(name, values)| {
            let values = values
                .iter()
                .map(|(field, value)| (field.clone(), value.json()))
                .collect();
            (name.clone(), JsonValue::Object(values))
        })
        .collect();
    object(vec![
        ("authority_release", string("CM1 r21.1")),
        ("groups", JsonValue::Object(groups)),
        ("schema", string(CM1_NAMELIST_DOCUMENT_SCHEMA)),
        ("support_schema_sha256", string(CM1_NAMELIST_SUPPORT_SHA256)),
    ])
}

struct Scanner<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Scanner<'a> {
    fn new(value: &'a str) -> Self {
        Self {
            bytes: value.as_bytes(),
            position: 0,
        }
    }

    fn finished(&self) -> bool {
        self.position == self.bytes.len()
    }

    fn skip_space(&mut self) {
        while self
            .bytes
            .get(self.position)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.position += 1;
        }
    }

    fn skip_space_and_commas(&mut self) -> bool {
        while self
            .bytes
            .get(self.position)
            .is_some_and(|byte| byte.is_ascii_whitespace() || *byte == b',')
        {
            self.position += 1;
        }
        !self.finished()
    }

    fn consume_byte(&mut self, expected: u8) -> bool {
        if self.bytes.get(self.position) == Some(&expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn expect_byte(&mut self, expected: u8, message: &str) -> Result<()> {
        if self.consume_byte(expected) {
            Ok(())
        } else {
            Err(invalid(format!(
                "CM1 NAMELIST byte {}: {message}",
                self.position
            )))
        }
    }

    fn identifier(&mut self, label: &str) -> Result<String> {
        let start = self.position;
        if !self
            .bytes
            .get(self.position)
            .is_some_and(u8::is_ascii_alphabetic)
        {
            return Err(invalid(format!(
                "CM1 NAMELIST byte {}: expected {label}",
                self.position
            )));
        }
        self.position += 1;
        while self
            .bytes
            .get(self.position)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            self.position += 1;
        }
        Ok(std::str::from_utf8(&self.bytes[start..self.position])
            .expect("ASCII identifier")
            .to_ascii_lowercase())
    }

    /// One value.  Fortran list-directed input separates values by a comma, a
    /// slash or blanks, so an unquoted blank ends the value too: `nx = 20 ny =
    /// 30` is two assignments, and a line with no final comma still ends its
    /// value.  No scalar literal CM1 declares contains a blank.
    fn value_token(&mut self) -> Result<&'a str> {
        let start = self.position;
        let mut quote = None;
        while let Some(byte) = self.bytes.get(self.position).copied() {
            match quote {
                Some(active) if byte == active => quote = None,
                Some(_) => {}
                None if byte == b'\'' || byte == b'"' => quote = Some(byte),
                None if byte == b',' || byte == b'/' || byte.is_ascii_whitespace() => break,
                None => {}
            }
            self.position += 1;
        }
        if quote.is_some() {
            return Err(invalid("CM1 NAMELIST contains an unterminated string"));
        }
        std::str::from_utf8(&self.bytes[start..self.position])
            .map(str::trim)
            .map_err(|_| invalid("CM1 NAMELIST value is not ASCII"))
    }

    /// The next element of a value list, if the text after the current value
    /// is one: separators followed by something that is not a new `name =`.
    fn continuation_element(&mut self) -> Result<Option<&'a str>> {
        let saved = self.position;
        self.skip_space_and_commas();
        let starts_value = self.bytes.get(self.position).is_some_and(|byte| {
            byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.' | b'\'' | b'"')
        });
        if !starts_value {
            self.position = saved;
            return Ok(None);
        }
        let token = self.value_token()?;
        if token.is_empty() {
            self.position = saved;
            return Ok(None);
        }
        Ok(Some(token))
    }

    /// Whether the name at the current position is `word` (any case), with
    /// no further name characters after it.  Consumes nothing.
    fn next_word_is(&self, word: &str) -> bool {
        let end = self.position + word.len();
        self.bytes
            .get(self.position..end)
            .is_some_and(|bytes| bytes.eq_ignore_ascii_case(word.as_bytes()))
            && !self
                .bytes
                .get(end)
                .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
    }

    /// Text outside every group, as gfortran's NAMELIST read passes over it:
    /// from here up to the next `&` or `$` that begins a name, or the end.
    /// Always consumes at least one byte, so a stray `&end` is passed over.
    fn outside_text(&mut self) -> &'a str {
        let start = self.position;
        self.position += 1;
        while let Some(byte) = self.bytes.get(self.position) {
            let opens_group = matches!(byte, b'&' | b'$')
                && self
                    .bytes
                    .get(self.position + 1)
                    .is_some_and(u8::is_ascii_alphabetic);
            if opens_group {
                break;
            }
            self.position += 1;
        }
        std::str::from_utf8(&self.bytes[start..self.position]).unwrap_or_default()
    }
}

fn required_string<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map_err(|error| invalid(format!("{label}.{key}: {error}")))
}

fn fixed_string(value: &JsonValue, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = required_string(value, key, label)?;
    if actual != expected {
        return Err(invalid(format!(
            "{label}.{key} is {actual:?}; expected {expected:?}"
        )));
    }
    Ok(())
}

fn string_array_json(values: &[String]) -> JsonValue {
    JsonValue::Array(values.iter().map(|value| string(value)).collect())
}

fn object(entries: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    )
}

fn string(value: &str) -> JsonValue {
    JsonValue::String(value.to_string())
}

fn invalid(message: impl Into<String>) -> StateError {
    StateError::invalid(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"
! strict mixed-type fixture
&param0
 NX=300, ny=300, nz=75, terrain_flag=.TRUE., procfiles=.false.,
/
&param1
 dtl=5.0d-1, tapfrq=300.0, rstfrq=-9.0d2,
/
&param2
 adapt_dt=0, irst=0, psolver=3, imoist=1, ptype=5,
/
"#;

    #[test]
    fn compiled_support_inventory_is_exhaustive_and_pinned() {
        let bytes = cm1_r21_namelist_support_json().unwrap();
        assert_eq!(sha256::digest_hex(bytes), CM1_NAMELIST_SUPPORT_SHA256);
        let inventory = support_inventory().unwrap();
        assert_eq!(inventory.len(), 25);
        assert_eq!(inventory.values().map(BTreeMap::len).sum::<usize>(), 583);
        assert_eq!(inventory["param1"]["tapfrq"], Cm1NamelistType::Real64);
        assert_eq!(
            inventory["param0"]["terrain_flag"],
            Cm1NamelistType::Logical
        );
        assert_eq!(inventory["nssl_mp_params"].len(), 212);
    }

    #[test]
    fn mixed_case_fortran_document_roundtrips_at_declared_precision() {
        let document = parse_cm1_r21_namelist(FIXTURE.as_bytes()).unwrap();
        assert_eq!(document.group_count(), 3);
        assert_eq!(document.assignment_count(), 13);
        assert_eq!(
            document
                .value("param0", "nx")
                .and_then(Cm1NamelistValue::as_i64),
            Some(300)
        );
        assert_eq!(
            document
                .value("param1", "dtl")
                .and_then(Cm1NamelistValue::real32_bits),
            Some(0.5f32.to_bits())
        );
        assert_eq!(
            document
                .value("param1", "tapfrq")
                .and_then(Cm1NamelistValue::real64_bits),
            Some(300.0f64.to_bits())
        );
        document.verify_semantic_roundtrip().unwrap();
        let rebound = parse_cm1_r21_namelist(document.canonical_namelist().as_bytes()).unwrap();
        assert_eq!(rebound.canonical_json(), document.canonical_json());
        assert_eq!(rebound.sha256(), document.sha256());

        let omitted_exponent = parse_cm1_r21_namelist(b"&param3 fcor=7.90038-5,/").unwrap();
        assert_eq!(
            omitted_exponent
                .value("param3", "fcor")
                .and_then(Cm1NamelistValue::real32_bits),
            Some(7.90038e-5f32.to_bits())
        );
    }

    #[test]
    fn unknown_duplicate_wrong_type_and_nonfinite_values_refuse() {
        for source in [
            "&not_a_group x=1,/",
            "&param0 no_such_field=1,/",
            "&param0 nx=1,nx=2,/",
            "&param0 nx=.true.,/",
            "&param1 dtl=NaN,/",
            "&param0 terrain_flag=1,/",
        ] {
            assert!(
                parse_cm1_r21_namelist(source.as_bytes()).is_err(),
                "{source}"
            );
        }
    }

    fn integer(document: &Cm1NamelistDocument, group: &str, field: &str) -> Option<i64> {
        document
            .value(group, field)
            .and_then(Cm1NamelistValue::as_i64)
    }

    fn logical(document: &Cm1NamelistDocument, group: &str, field: &str) -> Option<bool> {
        document
            .value(group, field)
            .and_then(Cm1NamelistValue::as_logical)
    }

    #[test]
    fn two_assignments_on_one_line_with_commas_or_blanks() {
        for source in [
            "&param0\n nx = 20, ny = 30,\n/\n",
            "&param0\n nx=20,ny=30\n/\n",
            "&param0\n nx = 20  ny = 30\n/\n",
        ] {
            let document = parse_cm1_r21_namelist(source.as_bytes()).unwrap();
            assert_eq!(integer(&document, "param0", "nx"), Some(20), "{source}");
            assert_eq!(integer(&document, "param0", "ny"), Some(30), "{source}");
        }
    }

    #[test]
    fn upper_case_names_and_groups_read_as_lower_case() {
        let document =
            parse_cm1_r21_namelist(b"&PARAM0\n NX = 20, Ny = 30, TERRAIN_FLAG = .FALSE.,\n/\n")
                .unwrap();
        assert_eq!(integer(&document, "param0", "nx"), Some(20));
        assert_eq!(integer(&document, "param0", "ny"), Some(30));
        assert_eq!(logical(&document, "param0", "terrain_flag"), Some(false));
    }

    #[test]
    fn every_fortran_logical_spelling() {
        for (spelling, expected) in [
            (".TRUE.", true),
            (".true.", true),
            ("T", true),
            ("t", true),
            (".T.", true),
            (".t", true),
            ("true", true),
            (".FALSE.", false),
            ("F", false),
            (".f.", false),
            ("False", false),
        ] {
            let source = format!("&param0 terrain_flag = {spelling} /");
            let document = parse_cm1_r21_namelist(source.as_bytes()).unwrap();
            assert_eq!(
                logical(&document, "param0", "terrain_flag"),
                Some(expected),
                "{spelling}"
            );
        }
        for bad in ["1", "yes", ".x.", "0"] {
            let source = format!("&param0 terrain_flag = {bad} /");
            assert!(parse_cm1_r21_namelist(source.as_bytes()).is_err(), "{bad}");
        }
        // The spelling never reaches the canonical document.
        let a = parse_cm1_r21_namelist(b"&param0 terrain_flag = T /").unwrap();
        let b = parse_cm1_r21_namelist(b"&param0 terrain_flag = .true., /").unwrap();
        assert_eq!(a.sha256(), b.sha256());
    }

    #[test]
    fn ampersand_end_and_dollar_end_terminate_a_group() {
        for source in [
            "&param0\n nx = 20,\n&end\n&param1\n dx = 500.0,\n&END\n",
            "$param0\n nx = 20,\n$end\n&param1 dx = 500.0 /\n",
        ] {
            let document = parse_cm1_r21_namelist(source.as_bytes()).unwrap();
            assert_eq!(integer(&document, "param0", "nx"), Some(20), "{source}");
            assert_eq!(
                document
                    .value("param1", "dx")
                    .and_then(Cm1NamelistValue::real32_bits),
                Some(500.0f32.to_bits())
            );
        }
        // A new group without a terminator is refused, not merged.
        assert!(parse_cm1_r21_namelist(b"&param0 nx = 20, &param1 dx = 1.0 /").is_err());
    }

    #[test]
    fn exclamation_marks_inside_strings_are_not_comments() {
        let inspection = inspect_cm1_r21_namelist(
            b"&logging\n compute_tasks_silent = .false.,\n/ ! trailing comment\n&param0 nx = 20 /\n",
            Cm1UnknownPolicy::Report,
        )
        .unwrap();
        assert_eq!(integer(inspection.document(), "param0", "nx"), Some(20));
        let report = inspect_cm1_r21_namelist(
            b"&param0 nx = 20 /\n&param99 note = 'wow! not a comment', x = \"a!b\" ! real comment\n/\n",
            Cm1UnknownPolicy::Report,
        )
        .unwrap();
        let texts: Vec<_> = report
            .unknown()
            .iter()
            .map(|row| row.text.as_str())
            .collect();
        assert_eq!(texts, ["'wow! not a comment'", "\"a!b\""]);
    }

    #[test]
    fn a_missing_trailing_comma_still_ends_the_value() {
        for source in [
            "&param0\n nx = 20\n ny = 30\n/\n",
            "&param0\n nx = 20\n ny = 30 /\n",
            "&param0\n nx = 20, ny = 30/\n",
            "&param0\r\n nx = 20\r\n ny = 30\r\n/\r\n",
        ] {
            let document = parse_cm1_r21_namelist(source.as_bytes()).unwrap();
            assert_eq!(integer(&document, "param0", "nx"), Some(20), "{source:?}");
            assert_eq!(integer(&document, "param0", "ny"), Some(30), "{source:?}");
        }
    }

    #[test]
    fn report_mode_returns_every_undeclared_assignment_in_file_order() {
        let source = b"&param1\n dx = 1000.0, cfl_limit = 1.0,\n/\n\
&param2\n pertcor = 1, ptype = 5, ramp_down = 1\n/\n\
&param23\n timeavg_num = 5, timeavg_len = 60.0,\n/\n";
        assert!(parse_cm1_r21_namelist(source).is_err());
        let inspection = inspect_cm1_r21_namelist(source, Cm1UnknownPolicy::Report).unwrap();
        let rows: Vec<_> = inspection
            .unknown()
            .iter()
            .map(|row| {
                (
                    row.group.as_str(),
                    row.field.as_str(),
                    row.group_declared,
                    row.text.as_str(),
                )
            })
            .collect();
        assert_eq!(
            rows,
            [
                ("param1", "cfl_limit", true, "1.0"),
                ("param2", "pertcor", true, "1"),
                ("param2", "ramp_down", true, "1"),
                ("param23", "timeavg_num", false, "5"),
                ("param23", "timeavg_len", false, "60.0"),
            ]
        );
        // The declared part is exactly what a strict parse of it gives.
        let strict =
            parse_cm1_r21_namelist(b"&param1 dx = 1000.0 /\n&param2 ptype = 5 /\n").unwrap();
        assert_eq!(inspection.document().sha256(), strict.sha256());
        let text = String::from_utf8(inspection.canonical_json().unwrap()).unwrap();
        assert!(text.contains("\"schema\":\"gm1.cm1-r21.1-namelist-inspection/v1\""));
        assert!(text.contains("\"dx\":{\"text\":\"1000.0\",\"type\":\"real32\",\"value\":1000.0}"));
        assert!(text.contains(
            "{\"field\":\"pertcor\",\"group\":\"param2\",\"group_declared\":true,\"text\":\"1\",\"value\":1,\"value_type\":\"integer\"}"
        ));
        assert!(text.contains("\"unknown_policy\":\"report\""));
    }

    #[test]
    fn report_mode_passes_over_text_outside_the_groups_as_gfortran_does() {
        // gfortran reads this file: it skips everything that is not the
        // '&name' it is looking for.  The strict identity parse still stops.
        let source = b"supercell test, 250 m\n&param0 nx = 20 /\nnotes: ny next\n\
&param2 ptype = 5 / trailing words\n&end\n$param8 var1 = 7.0 $end\n";
        assert!(parse_cm1_r21_namelist(source).is_err());
        let inspection = inspect_cm1_r21_namelist(source, Cm1UnknownPolicy::Report).unwrap();
        let rows: Vec<_> = inspection
            .outside()
            .iter()
            .map(|row| (row.line, row.text.as_str()))
            .collect();
        assert_eq!(
            rows,
            [
                (1, "supercell test, 250 m"),
                (3, "notes: ny next"),
                (4, "trailing words"),
                (5, "&end"),
            ]
        );
        assert_eq!(inspection.group_order(), ["param0", "param2", "param8"]);
        let clean = parse_cm1_r21_namelist(
            b"&param0 nx = 20 /\n&param2 ptype = 5 /\n&param8 var1 = 7.0 /\n",
        )
        .unwrap();
        assert_eq!(inspection.document().sha256(), clean.sha256());
        let text = String::from_utf8(inspection.canonical_json().unwrap()).unwrap();
        assert!(text.contains("\"group_order\":[\"param0\",\"param2\",\"param8\"]"));
        assert!(text.contains("{\"line\":1,\"text\":\"supercell test, 250 m\"}"));
    }

    #[test]
    fn group_order_is_the_file_order_including_undeclared_groups() {
        let inspection = inspect_cm1_r21_namelist(
            b"&param8 var1 = 7.0 /\n&param99 foo = 1 /\n&param0 nx = 20 /\n&param2 ptype = 5 /\n",
            Cm1UnknownPolicy::Report,
        )
        .unwrap();
        assert_eq!(
            inspection.group_order(),
            ["param8", "param99", "param0", "param2"]
        );
        assert!(inspection.outside().is_empty());
    }

    #[test]
    fn decoded_reals_are_the_literal_the_file_wrote() {
        let inspection = inspect_cm1_r21_namelist(
            b"&param3 fcor = 5.0e-5, zd = 15000.0 /\n&param1 tapfrq = 1.5d3 /",
            Cm1UnknownPolicy::Refuse,
        )
        .unwrap();
        let text = String::from_utf8(inspection.canonical_json().unwrap()).unwrap();
        assert!(text.contains("\"fcor\":{\"text\":\"5.0e-5\",\"type\":\"real32\",\"value\":5e-5}"));
        assert!(
            text.contains("\"tapfrq\":{\"text\":\"1.5d3\",\"type\":\"real64\",\"value\":1500.0}")
        );
    }

    #[test]
    fn arbitrary_checkpoint_is_explicitly_fail_closed_but_bindable() {
        let document = parse_cm1_r21_namelist(FIXTURE.as_bytes()).unwrap();
        let error = document
            .require_arbitrary_checkpoint_supported()
            .unwrap_err()
            .to_string();
        assert!(error.contains("CM1-ARBITRARY-CONFIG-CARRIER-CONTRACT-MISSING"));
        let binding = document.configuration_binding().unwrap();
        assert_eq!(binding.document_sha256(), document.sha256());
        assert_eq!(binding.component_blob().name(), CM1_CONFIGURATION_BLOB_NAME);
        assert_eq!(
            binding.component_blob().schema(),
            CM1_CONFIGURATION_BINDING_SCHEMA
        );
        assert_eq!(sha256::digest_hex(binding.bytes()), binding.sha256());
    }
}
