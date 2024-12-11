use serde::{Deserialize, Serialize};

use super::Variable;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DebugInfo {
    Source {
        file_name: String,
        source: String,
        line_offset: u32,
    },
    BeginCall {
        name: String,
        line: u32,
        col: u32,
    },
    EndCall,
    Print {
        format_string: String,
        args: Vec<Variable>,
    },
}
