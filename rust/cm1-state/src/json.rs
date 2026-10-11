//! Minimal strict JSON used to verify the Python canonical header bytes.
//!
//! This is deliberately not a general serialization framework.  It retains
//! JSON number spellings after checking the normalized forms emitted by the
//! v1 producer, rejects duplicate keys, and emits Python-compatible
//! `ensure_ascii=True`, sorted-key, compact JSON.

use std::fmt::Write;

const MAX_DEPTH: usize = 128;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub(crate) fn as_object(&self) -> Result<&[(String, JsonValue)], String> {
        match self {
            Self::Object(value) => Ok(value),
            _ => Err("expected a JSON object".into()),
        }
    }

    pub(crate) fn as_array(&self) -> Result<&[JsonValue], String> {
        match self {
            Self::Array(value) => Ok(value),
            _ => Err("expected a JSON array".into()),
        }
    }

    pub(crate) fn as_str(&self) -> Result<&str, String> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err("expected a JSON string".into()),
        }
    }

    pub(crate) fn as_u64(&self) -> Result<u64, String> {
        match self {
            Self::Number(value)
                if !value.starts_with('-') && !value.contains('.') && !value.contains('e') =>
            {
                value
                    .parse::<u64>()
                    .map_err(|_| "expected an unsigned 64-bit JSON integer".into())
            }
            _ => Err("expected an unsigned JSON integer".into()),
        }
    }

    pub(crate) fn as_i64(&self) -> Result<i64, String> {
        match self {
            Self::Number(value) if !value.contains('.') && !value.contains('e') => value
                .parse::<i64>()
                .map_err(|_| "expected a signed 64-bit JSON integer".into()),
            _ => Err("expected a signed JSON integer".into()),
        }
    }

    pub(crate) fn as_f64(&self) -> Result<f64, String> {
        match self {
            Self::Number(value) => value
                .parse::<f64>()
                .map_err(|_| "expected a binary64 JSON number".into()),
            _ => Err("expected a JSON number".into()),
        }
    }

    pub(crate) fn get(&self, key: &str) -> Result<&JsonValue, String> {
        self.as_object()?
            .iter()
            .find(|(candidate, _)| candidate == key)
            .map(|(_, value)| value)
            .ok_or_else(|| format!("missing JSON key {key:?}"))
    }

    pub(crate) fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = String::new();
        self.write_canonical(&mut output);
        output.into_bytes()
    }

    fn write_canonical(&self, output: &mut String) {
        match self {
            Self::Null => output.push_str("null"),
            Self::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Self::Number(value) => output.push_str(value),
            Self::String(value) => write_string(value, output),
            Self::Array(values) => {
                output.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        output.push(',');
                    }
                    value.write_canonical(output);
                }
                output.push(']');
            }
            Self::Object(entries) => {
                let mut sorted: Vec<_> = entries.iter().collect();
                sorted.sort_by(|left, right| left.0.cmp(&right.0));
                output.push('{');
                for (index, (key, value)) in sorted.into_iter().enumerate() {
                    if index != 0 {
                        output.push(',');
                    }
                    write_string(key, output);
                    output.push(':');
                    value.write_canonical(output);
                }
                output.push('}');
            }
        }
    }
}

pub(crate) fn parse(bytes: &[u8]) -> Result<JsonValue, String> {
    let text =
        std::str::from_utf8(bytes).map_err(|error| format!("header is not UTF-8: {error}"))?;
    let mut parser = Parser { text, offset: 0 };
    let value = parser.value(0)?;
    parser.whitespace();
    if parser.offset != parser.text.len() {
        return Err(format!(
            "unexpected trailing JSON byte at offset {}",
            parser.offset
        ));
    }
    Ok(value)
}

struct Parser<'a> {
    text: &'a str,
    offset: usize,
}

impl Parser<'_> {
    fn value(&mut self, depth: usize) -> Result<JsonValue, String> {
        if depth > MAX_DEPTH {
            return Err(format!("JSON nesting exceeds {MAX_DEPTH}"));
        }
        self.whitespace();
        match self.peek() {
            Some(b'n') => {
                self.literal("null")?;
                Ok(JsonValue::Null)
            }
            Some(b't') => {
                self.literal("true")?;
                Ok(JsonValue::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(JsonValue::Bool(false))
            }
            Some(b'"') => Ok(JsonValue::String(self.string()?)),
            Some(b'[') => self.array(depth + 1),
            Some(b'{') => self.object(depth + 1),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(byte) => Err(format!(
                "unexpected JSON byte 0x{byte:02x} at offset {}",
                self.offset
            )),
            None => Err("unexpected end of JSON".into()),
        }
    }

    fn array(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.expect(b'[')?;
        self.whitespace();
        let mut values = Vec::new();
        if self.consume(b']') {
            return Ok(JsonValue::Array(values));
        }
        loop {
            values.push(self.value(depth)?);
            self.whitespace();
            if self.consume(b']') {
                return Ok(JsonValue::Array(values));
            }
            self.expect(b',')?;
        }
    }

    fn object(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.expect(b'{')?;
        self.whitespace();
        let mut entries: Vec<(String, JsonValue)> = Vec::new();
        if self.consume(b'}') {
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.whitespace();
            if self.peek() != Some(b'"') {
                return Err(format!(
                    "object key is not a string at offset {}",
                    self.offset
                ));
            }
            let key = self.string()?;
            if entries.iter().any(|(existing, _)| existing == &key) {
                return Err(format!("duplicate JSON object key {key:?}"));
            }
            self.whitespace();
            self.expect(b':')?;
            let value = self.value(depth)?;
            entries.push((key, value));
            self.whitespace();
            if self.consume(b'}') {
                return Ok(JsonValue::Object(entries));
            }
            self.expect(b',')?;
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut output = String::new();
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| "unterminated JSON string".to_string())?;
            match byte {
                b'"' => {
                    self.offset += 1;
                    return Ok(output);
                }
                b'\\' => {
                    self.offset += 1;
                    let escaped = self
                        .peek()
                        .ok_or_else(|| "unterminated JSON escape".to_string())?;
                    self.offset += 1;
                    match escaped {
                        b'"' => output.push('"'),
                        b'\\' => output.push('\\'),
                        b'/' => output.push('/'),
                        b'b' => output.push('\u{0008}'),
                        b'f' => output.push('\u{000c}'),
                        b'n' => output.push('\n'),
                        b'r' => output.push('\r'),
                        b't' => output.push('\t'),
                        b'u' => output.push(self.unicode_escape()?),
                        _ => {
                            return Err(format!(
                                "invalid JSON escape at offset {}",
                                self.offset - 1
                            ));
                        }
                    }
                }
                0x00..=0x1f => {
                    return Err(format!(
                        "unescaped control character at offset {}",
                        self.offset
                    ));
                }
                0x20..=0x7f => {
                    output.push(byte as char);
                    self.offset += 1;
                }
                _ => {
                    let character = self.text[self.offset..]
                        .chars()
                        .next()
                        .ok_or_else(|| "invalid UTF-8 string tail".to_string())?;
                    output.push(character);
                    self.offset += character.len_utf8();
                }
            }
        }
    }

    fn unicode_escape(&mut self) -> Result<char, String> {
        let first = self.hex_quad()?;
        let scalar = if (0xd800..=0xdbff).contains(&first) {
            if self.remaining().len() < 2 || &self.remaining().as_bytes()[..2] != b"\\u" {
                return Err("high surrogate is not followed by a low surrogate".into());
            }
            self.offset += 2;
            let second = self.hex_quad()?;
            if !(0xdc00..=0xdfff).contains(&second) {
                return Err("high surrogate is not followed by a low surrogate".into());
            }
            0x1_0000 + (((first as u32 - 0xd800) << 10) | (second as u32 - 0xdc00))
        } else if (0xdc00..=0xdfff).contains(&first) {
            return Err("unpaired low surrogate in JSON string".into());
        } else {
            first as u32
        };
        char::from_u32(scalar).ok_or_else(|| "invalid Unicode scalar in JSON string".into())
    }

    fn hex_quad(&mut self) -> Result<u16, String> {
        if self.remaining().len() < 4 {
            return Err("truncated JSON Unicode escape".into());
        }
        let bytes = &self.remaining().as_bytes()[..4];
        let mut value = 0u16;
        for byte in bytes {
            value = (value << 4)
                | match byte {
                    b'0'..=b'9' => (byte - b'0') as u16,
                    b'a'..=b'f' => (byte - b'a' + 10) as u16,
                    b'A'..=b'F' => (byte - b'A' + 10) as u16,
                    _ => return Err("invalid hex digit in JSON Unicode escape".into()),
                };
        }
        self.offset += 4;
        Ok(value)
    }

    fn number(&mut self) -> Result<JsonValue, String> {
        let start = self.offset;
        self.consume(b'-');
        match self.peek() {
            Some(b'0') => {
                self.offset += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(format!("leading zero in JSON number at offset {start}"));
                }
            }
            Some(b'1'..=b'9') => {
                self.offset += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.offset += 1;
                }
            }
            _ => return Err(format!("invalid JSON number at offset {start}")),
        }
        if self.consume(b'.') {
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("missing fraction digits at offset {start}"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.offset += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.offset += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.offset += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("missing exponent digits at offset {start}"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.offset += 1;
            }
        }
        let spelling = &self.text[start..self.offset];
        validate_python_number_spelling(spelling)?;
        Ok(JsonValue::Number(spelling.to_string()))
    }

    fn literal(&mut self, literal: &str) -> Result<(), String> {
        if !self.remaining().starts_with(literal) {
            return Err(format!("invalid JSON literal at offset {}", self.offset));
        }
        self.offset += literal.len();
        Ok(())
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.offset += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), String> {
        if self.consume(expected) {
            Ok(())
        } else {
            Err(format!(
                "expected JSON byte {:?} at offset {}",
                expected as char, self.offset
            ))
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.offset).copied()
    }

    fn remaining(&self) -> &str {
        &self.text[self.offset..]
    }
}

fn validate_python_number_spelling(value: &str) -> Result<(), String> {
    let unsigned = value.strip_prefix('-').unwrap_or(value);
    if !unsigned.contains('.') && !unsigned.contains('e') && !unsigned.contains('E') {
        if value == "-0" {
            return Err("-0 is not the Python canonical integer spelling".into());
        }
        return Ok(());
    }
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("JSON number {value:?} is not binary64"))?;
    if !parsed.is_finite() {
        return Err("non-finite JSON numbers are forbidden".into());
    }
    if value.contains('E') {
        return Err("Python canonical JSON uses lower-case e".into());
    }

    let (mantissa, exponent) = match value.split_once('e') {
        Some(parts) => parts,
        None => {
            validate_fraction_tail(unsigned)?;
            return Ok(());
        }
    };
    if parsed == 0.0 {
        return Err("Python canonical JSON does not use exponent notation for zero".into());
    }
    let exponent_sign = exponent
        .as_bytes()
        .first()
        .copied()
        .ok_or_else(|| "missing exponent sign".to_string())?;
    if !matches!(exponent_sign, b'+' | b'-') {
        return Err("Python canonical exponent includes an explicit sign".into());
    }
    let exponent_digits = &exponent[1..];
    if exponent_digits.len() < 2 {
        return Err("Python canonical exponent has at least two digits".into());
    }
    if exponent_digits.len() > 2 && exponent_digits.starts_with('0') {
        return Err("Python canonical exponent has no redundant leading zero".into());
    }
    let exponent_value: i32 = exponent
        .parse()
        .map_err(|_| "JSON exponent is out of range".to_string())?;
    if (-4..16).contains(&exponent_value) {
        return Err("Python canonical binary64 spelling uses decimal notation here".into());
    }
    validate_fraction_tail(mantissa.strip_prefix('-').unwrap_or(mantissa))?;
    if mantissa.ends_with(".0") {
        return Err("Python canonical scientific mantissa omits .0".into());
    }
    Ok(())
}

fn validate_fraction_tail(unsigned: &str) -> Result<(), String> {
    if let Some((_, fraction)) = unsigned.split_once('.') {
        let fraction = fraction
            .split('e')
            .next()
            .expect("split always returns one item");
        if fraction.len() > 1 && fraction.ends_with('0') {
            return Err("Python canonical fraction has no redundant trailing zero".into());
        }
    }
    Ok(())
}

fn write_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0000}'..='\u{001f}' => {
                write!(output, "\\u{:04x}", character as u32)
                    .expect("writing to String cannot fail");
            }
            '\u{0020}'..='\u{007e}' => output.push(character),
            '\u{007f}'..='\u{ffff}' => {
                write!(output, "\\u{:04x}", character as u32)
                    .expect("writing to String cannot fail");
            }
            _ => {
                let scalar = character as u32 - 0x1_0000;
                let high = 0xd800 + (scalar >> 10);
                let low = 0xdc00 + (scalar & 0x3ff);
                write!(output, "\\u{high:04x}\\u{low:04x}").expect("writing to String cannot fail");
            }
        }
    }
    output.push('"');
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn canonicalizes_sorted_ascii_json() {
        let parsed =
            parse(br#"{ "z":"\ud83c\udf2a", "a":"\u00e9", "n":1000.0 }"#).expect("valid JSON");
        assert_eq!(
            parsed.canonical_bytes(),
            br#"{"a":"\u00e9","n":1000.0,"z":"\ud83c\udf2a"}"#
        );
    }

    #[test]
    fn rejects_duplicate_keys_and_noncanonical_number_forms() {
        assert!(parse(br#"{"a":1,"a":2}"#)
            .unwrap_err()
            .contains("duplicate"));
        assert!(parse(b"-0").unwrap_err().contains("canonical"));
        assert!(parse(b"1.00").unwrap_err().contains("trailing"));
        assert!(parse(b"1e+0").unwrap_err().contains("at least two"));
    }
}
