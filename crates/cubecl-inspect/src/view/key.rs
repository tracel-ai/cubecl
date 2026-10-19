use ciborium::Value;
use std::fmt::{self, Write as _};

/// A decoded autotune key on one line: `field=value` pairs, a nested struct's
/// fields prefixed with its own name, an enum as `Variant(value)`.
///
/// `{"elem": {"Float": "F16"}, "k": 1024}` reads `elem=Float(F16) k=1024`.
pub struct KeyText<'a>(pub &'a Value);

impl fmt::Display for KeyText<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut fields = Vec::new();
        flatten(self.0, "", &mut fields);
        let mut separator = "";
        for (path, value) in fields {
            if path.is_empty() {
                write!(f, "{separator}{value}")?;
            } else {
                write!(f, "{separator}{path}={value}")?;
            }
            separator = " ";
        }
        Ok(())
    }
}

/// Collect the `(path, value)` pairs of a struct, descending into nested
/// structs; anything else is one pair at `path`.
fn flatten(value: &Value, path: &str, fields: &mut Vec<(String, String)>) {
    match value {
        Value::Map(entries) if variant(entries).is_none() => {
            for (name, field) in entries {
                let name = scalar(name);
                let path = if path.is_empty() {
                    name
                } else {
                    format!("{path}.{name}")
                };
                flatten(field, &path, fields);
            }
        }
        _ => fields.push((path.to_string(), scalar(value))),
    }
}

/// serde's encoding of an enum variant carrying data: a one-entry map whose
/// key is the variant's name.
fn variant(entries: &[(Value, Value)]) -> Option<(&str, &Value)> {
    match entries {
        [(Value::Text(name), inner)] if name.starts_with(char::is_uppercase) => Some((name, inner)),
        _ => None,
    }
}

/// A value on its own, compactly.
fn scalar(value: &Value) -> String {
    match value {
        Value::Integer(integer) => i128::from(*integer).to_string(),
        Value::Text(text) => text.clone(),
        Value::Bool(flag) => flag.to_string(),
        Value::Float(float) => float.to_string(),
        Value::Null => "none".to_string(),
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(scalar).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Map(entries) => match variant(entries) {
            Some((name, inner)) => format!("{name}({})", scalar(inner)),
            None => {
                let mut text = String::from("{");
                let mut fields = Vec::new();
                flatten(value, "", &mut fields);
                for (index, (path, field)) in fields.iter().enumerate() {
                    let separator = if index == 0 { "" } else { " " };
                    let _ = write!(text, "{separator}{path}={field}");
                }
                text.push('}');
                text
            }
        },
        Value::Bytes(bytes) => format!("<{} bytes>", bytes.len()),
        Value::Tag(_, inner) => scalar(inner),
        _ => format!("{value:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(name: &str) -> Value {
        Value::Text(name.to_string())
    }

    fn map(entries: Vec<(&str, Value)>) -> Value {
        Value::Map(
            entries
                .into_iter()
                .map(|(name, value)| (text(name), value))
                .collect(),
        )
    }

    #[test]
    fn a_key_reads_as_its_fields() {
        let key = map(vec![
            ("elem", map(vec![("Float", text("F16"))])),
            ("k", Value::Integer(1024.into())),
            ("causal", Value::Bool(true)),
            ("operand", text("Float")),
        ]);
        assert_eq!(
            KeyText(&key).to_string(),
            "elem=Float(F16) k=1024 causal=true operand=Float"
        );
    }

    #[test]
    fn a_nested_struct_prefixes_its_fields() {
        let key = map(vec![
            ("accumulator_len", Value::Integer(1.into())),
            (
                "reduce",
                map(vec![("axis_is_contiguous", Value::Bool(true))]),
            ),
        ]);
        assert_eq!(
            KeyText(&key).to_string(),
            "accumulator_len=1 reduce.axis_is_contiguous=true"
        );
    }

    #[test]
    fn a_bare_value_reads_as_itself() {
        assert_eq!(KeyText(&Value::Integer(7.into())).to_string(), "7");
    }
}
