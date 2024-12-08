use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DebugInfo {
    BeginCall { name: String },
    EndCall,
}
