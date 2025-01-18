use std::fmt::Display;

use super::{Component, Dialect, Variable};

#[derive(Debug, Clone)]
pub enum PipelineOps<D: Dialect> {
    Init {
        pipeline: Variable<D>,
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
                write!(f, "
cuda::memcpy_async(cooperative_groups::this_thread_block(), {destination}, {source}, {source}_length * {size}, {pipeline});
                ")
            }
            PipelineOps::Init { pipeline } => {
                write!(
                    f,
                    "
__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 1> {pipeline}_state;
auto {pipeline} = cuda::make_pipeline(cooperative_groups::this_thread_block(), &{pipeline}_state);
                "
                )
            }
            PipelineOps::ProducerAcquire { pipeline } => {
                write!(
                    f,
                    "
{pipeline}.producer_acquire();
                "
                )
            }
            PipelineOps::ProducerCommit { pipeline } => {
                write!(
                    f,
                    "
{pipeline}.producer_commit();
            "
                )
            }
            PipelineOps::ConsumerWait { pipeline } => {
                write!(
                    f,
                    "
{pipeline}.consumer_wait();
            "
                )
            }
            PipelineOps::ConsumerRelease { pipeline } => {
                write!(
                    f,
                    "
{pipeline}.consumer_release();
            "
                )
            }
        }
    }
}
