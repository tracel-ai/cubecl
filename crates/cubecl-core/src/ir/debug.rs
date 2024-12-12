use serde::{Deserialize, Serialize};

use super::Variable;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DebugInfo {
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
    Span {
        line: u32,
        col: u32,
    },
    Print {
        format_string: String,
        args: Vec<Variable>,
    },
}
