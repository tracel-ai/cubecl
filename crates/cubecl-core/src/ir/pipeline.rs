use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::Variable;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PipelineOps {
    MemCopyAsync {
        pipeline: Variable,
        source: Variable,
        destination: Variable,
    },
}

impl Display for PipelineOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineOps::MemCopyAsync {
                pipeline,
                source,
                destination,
            } => write!(
                f,
                "mem_copy_async({pipeline}, source: {source}, destination: {destination})",
            ),
        }
    }
}
