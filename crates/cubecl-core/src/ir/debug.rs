use serde::{Deserialize, Serialize};

use super::Variable;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DebugInfo {
    BeginCall {
        name: String,
    },
    EndCall,
    Print {
        format_string: String,
        args: Vec<Variable>,
    },
}
