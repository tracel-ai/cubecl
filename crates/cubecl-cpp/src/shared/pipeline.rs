use std::fmt::Display;

use super::{Component, Dialect, Variable};

#[derive(Debug, Clone)]
pub enum PipelineOps<D: Dialect> {
    Init {
        pipeline: Variable<D>,
        num_stages: u8,
    },
    MemCopyAsync {
        pipeline: Variable<D>,
        source: Variable<D>,
        destination: Variable<D>,
    },
    ProducerAcquire {
        pipeline: Variable<D>,
    },
    ProducerCommit {
        pipeline: Variable<D>,
    },
    ConsumerWait {
        pipeline: Variable<D>,
    },
    ConsumerRelease {
        pipeline: Variable<D>,
    },
}

impl<D: Dialect> PipelineOps<D> {
    pub fn pipeline_id(&self) -> u32 {
        match self {
            PipelineOps::MemCopyAsync { pipeline, .. } => pipeline.id().unwrap(),
            PipelineOps::Init { pipeline, .. } => pipeline.id().unwrap(),
            PipelineOps::ProducerAcquire { pipeline } => pipeline.id().unwrap(),
            PipelineOps::ProducerCommit { pipeline } => pipeline.id().unwrap(),
            PipelineOps::ConsumerWait { pipeline } => pipeline.id().unwrap(),
            PipelineOps::ConsumerRelease { pipeline } => pipeline.id().unwrap(),
        }
    }
}

impl<D: Dialect> Display for PipelineOps<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineOps::MemCopyAsync {
                pipeline,
                source,
                destination,
            } => {
                let item = source.item();
                let size = format!("sizeof({item})");
                write!(
                    f,
                    "
cooperative_groups::memcpy_async({pipeline}_block, {destination}, {source}, {source}_length * {size});
                "
                )
            }
            PipelineOps::Init { pipeline, .. } => {
                write!(
                    f,
                    "
auto {pipeline}_block = cooperative_groups::this_thread();
                "
                )
            }
            PipelineOps::ProducerAcquire { .. } => {
                write!(
                    f,
                    "
                "
                )
            }
            PipelineOps::ProducerCommit { .. } => {
                write!(
                    f,
                    "
            "
                )
            }
            PipelineOps::ConsumerWait { pipeline } => {
                write!(
                    f,
                    "
cooperative_groups::wait({pipeline}_block);
            "
                )
            }
            PipelineOps::ConsumerRelease { .. } => {
                write!(
                    f,
                    "
            "
                )
            }
        }
    }
}
