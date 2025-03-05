use std::fmt::Display;

use cubecl_core::ir::BarrierLevel;

use super::{Component, Dialect, Variable};

#[derive(Debug, Clone)]
pub enum BarrierOps<D: Dialect> {
    Init {
        barrier: Variable<D>,
        level: BarrierLevel,
    },
    MemCopyAsync {
        barrier: Variable<D>,
        source: Variable<D>,
        destination: Variable<D>,
        level: BarrierLevel,
    },
    Wait {
        barrier: Variable<D>,
        level: BarrierLevel,
    },
}

impl<D: Dialect> BarrierOps<D> {
    pub fn barrier_id(&self) -> u32 {
        match self {
            BarrierOps::MemCopyAsync { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::Init { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::Wait { barrier, .. } => barrier.id().unwrap(),
        }
    }
}

impl<D: Dialect> Display for BarrierOps<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BarrierOps::Init { barrier, level } => match level {
                BarrierLevel::Unit => write!(
                    f,
                    "
cuda::barrier<cuda::thread_scope_thread> {barrier};
init(&{barrier}, 1);
                "
                ),
                BarrierLevel::CubeCoop(elected_unit) => write!(
                    f,
                    "
cooperative_groups::thread_block block_{barrier} = cooperative_groups::this_thread_block();
__shared__ cuda::barrier<cuda::thread_scope_block> {barrier};
if (threadIdxGlobal == {elected_unit}) {{
   init(&{barrier}, blockDimGlobal);
}}
__syncthreads();
"
                ),
                BarrierLevel::CubeManual(elected_unit) => write!(
                    f,
                    "
__shared__ cuda::barrier<cuda::thread_scope_block> {barrier};
if (threadIdxGlobal == {elected_unit}) {{
   init(&{barrier}, blockDimGlobal);
}}
__syncthreads();
"
                ),
            },
            BarrierOps::MemCopyAsync {
                barrier,
                source,
                destination,
                level,
            } => {
                let item = source.item();
                let size = format!("sizeof({item})");
                match level {
                    BarrierLevel::Unit => write!(
                        f,
                        "
cuda::memcpy_async({destination}, {source}, {source}_length * {size}, {barrier});
                    "
                    ),
                    BarrierLevel::CubeCoop(_) => write!(
                        f,
                        "
cuda::memcpy_async(block_{barrier}, {destination}, {source}, {source}_length * {size}, {barrier});
                        "
                    ),
                    BarrierLevel::CubeManual(_) => write!(
                        f,
                        "
cuda::memcpy_async({destination}, {source}, {source}_length * {size}, {barrier});
                        "
                    ),
                }
            }

            BarrierOps::Wait { barrier, level: _ } => write!(
                f,
                "
{barrier}.arrive_and_wait();
            "
            ),
        }
    }
}
