use alloc::format;
use alloc::string::String;

/// Print string without quotes
pub struct DebugRaw<'a>(pub &'a str);

impl<'a> core::fmt::Debug for DebugRaw<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Format strings for use in identifiers and types.
pub fn format_str(string: &str, markers: &[(char, char)], include_space: bool) -> String {
    let mut result = String::new();
    let mut depth = 0;
    let indentation = 4;

    let mut prev = ' ';
    let mut in_string = false;

    for c in string.chars() {
        if c == ' ' {
            if in_string {
                result.push(c);
            }

            continue;
        }
        if c == '"' {
            in_string = !in_string;
        }

        let mut found_marker = false;

        for (start, end) in markers {
            let (start, end) = (*start, *end);

            if c == start {
                depth += 1;
                if prev != ' ' && include_space {
                    result.push(' ');
                }
                result.push(start);
                result.push('\n');
                result.push_str(&" ".repeat(indentation * depth));
                found_marker = true;
            } else if c == end {
                depth -= 1;
                if prev != start {
                    if prev == ' ' {
                        result.pop();
                    }
                    result.push_str(",\n");
                    result.push_str(&" ".repeat(indentation * depth));
                    result.push(end);
                } else {
                    for _ in 0..(&" ".repeat(indentation * depth).len()) + 1 + indentation {
                        result.pop();
                    }
                    result.push(end);
                }
                found_marker = true;
            }
        }

        if found_marker {
            prev = c;
            continue;
        }

        if c == ',' && depth > 0 {
            if prev == ' ' {
                result.pop();
            }

            result.push_str(",\n");
            result.push_str(&" ".repeat(indentation * depth));
            continue;
        }

        if c == ':' && include_space {
            result.push(c);
            result.push(' ');
            prev = ' ';
        } else {
            result.push(c);
            prev = c;
        }
    }

    result
}

/// Format a debug type.
pub fn format_debug<F: core::fmt::Debug>(string: &F) -> String {
    let string = format!("{string:?}");
    format_str(&string, &[('(', ')'), ('[', ']'), ('{', '}')], true)
}

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use super::*;

    #[derive(Debug)]
    #[allow(unused)]
    struct Test {
        map: HashMap<String, u32>,
    }

    #[test_log::test]
    fn test_format_debug() {
        let test = Test {
            map: HashMap::from_iter([("Hey with space".to_string(), 8)]),
        };

        let formatted = format_debug(&test);
        let expected = r#"Test {
    map: {
        "Hey with space": 8,
    },
}"#;

        assert_eq!(expected, formatted);
    }
}
