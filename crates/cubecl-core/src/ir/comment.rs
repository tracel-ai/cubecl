use serde::{Deserialize, Serialize};
use std::fmt::Display;

/// A comment
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct Comment {
    pub content: String,
}

impl Display for Comment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "comment({})", self.content)
    }
}
