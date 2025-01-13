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
}

impl<D: Dialect> PipelineOps<D> {
    pub fn pipeline_id(&self) -> u32 {
        match self {
            PipelineOps::MemCopyAsync { pipeline, .. } => pipeline.id().unwrap(),
            PipelineOps::Init { pipeline, .. } => pipeline.id().unwrap(),
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
                let size = item.elem().size() * item.vectorization;
                write!(f, "
cuda::memcpy_async(cooperative_groups::this_thread_block(), {destination}, {source}, {source}_length * {size}, {pipeline});
                ")
            }
            PipelineOps::Init { pipeline } => {
                write!(
                    f,
                    "
__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count> {pipeline}_state;
auto {pipeline} = cuda::make_pipeline(block, &{pipeline}_state);
                "
                )
            }
        }
    }
}
