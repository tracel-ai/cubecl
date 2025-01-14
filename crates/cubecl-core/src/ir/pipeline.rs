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
    ProducerAcquire {
        pipeline: Variable,
    },
    ProducerCommit {
        pipeline: Variable,
    },
    ConsumerAwait {
        pipeline: Variable,
    },
    ConsumerRelease {
        pipeline: Variable,
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
            PipelineOps::ProducerAcquire { pipeline } => write!(f, "producer_acquire({pipeline})"),
            PipelineOps::ProducerCommit { pipeline } => write!(f, "producer_commit({pipeline})"),
            PipelineOps::ConsumerAwait { pipeline } => write!(f, "consumer_await({pipeline})"),
            PipelineOps::ConsumerRelease { pipeline } => write!(f, "consumer_release({pipeline})"),
        }
    }
}
