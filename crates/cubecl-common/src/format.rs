use alloc::format;
use alloc::string::String;

/// Format strings for use in identifiers and types.
pub fn format_str(string: &str, markers: &[(char, char)], include_space: bool) -> String {
    let mut result = String::new();
    let mut depth = 0;
    let indentation = 4;

    let mut prev = ' ';

    for c in string.chars() {
        if c == ' ' {
            continue;
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
