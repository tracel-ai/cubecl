use std::fmt::Display;
use crate::TypeHash;

use crate::OperationReflect;

use super::Variable;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = PipelineOpCode)]
/// Operations available on a pipeline
pub enum PipelineOps {
    /// Copy source to destination
    MemCopyAsync {
        pipeline: Variable,
        source: Variable,
        destination: Variable,
    },
    /// Reserves a specific stage for the producer to work on.
    ProducerAcquire { pipeline: Variable },
    /// Signals that the producer is done and the stage is ready for the consumer.
    ProducerCommit { pipeline: Variable },
    /// Waits until the producer has finished with the stage.
    ConsumerWait { pipeline: Variable },
    /// Frees the stage after the consumer is done using it.
    ConsumerRelease { pipeline: Variable },
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
            PipelineOps::ConsumerWait { pipeline } => write!(f, "consumer_wait({pipeline})"),
            PipelineOps::ConsumerRelease { pipeline } => write!(f, "consumer_release({pipeline})"),
        }
    }
}
