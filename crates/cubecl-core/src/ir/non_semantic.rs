use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::ir::fmt_vararg;

use super::Variable;

/// Operations that don't change the semantics of the kernel. In other words, operations that do not
/// perform any computation, if they run at all. i.e. `println`, comments and debug symbols.
///
/// Can be safely removed or ignored without changing the kernel result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NonSemantic {
    Source {
        name: String,
        file_name: String,
        line: u32,
        col: u32,
    },
    BeginCall {
        name: String,
        line: u32,
        col: u32,
    },
    EndCall,
    Line {
        line: u32,
        col: u32,
    },
    Print {
        format_string: String,
        args: Vec<Variable>,
    },
    Comment {
        content: String,
    },
}

impl Display for NonSemantic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NonSemantic::Print {
                format_string,
                args,
            } => {
                write!(f, "print({format_string}, {})", fmt_vararg(args))
            }
            NonSemantic::Comment { content } => write!(f, "//{content}"),
            _ => {
                // Debug info has no semantic meaning
                Ok(())
            }
        }
    }
}
