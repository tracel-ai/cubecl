use std::fmt::{Display, Write};

use cubecl_core::ir::BarrierLevel;

use super::{Component, Dialect, Variable};

#[derive(Debug, Clone)]
pub enum BarrierOps<D: Dialect> {
    Init {
        barrier: Variable<D>,
        level: BarrierLevel,
        with_cta_fence: bool,
    },
    MemCopyAsync {
        barrier: Variable<D>,
        source: Variable<D>,
        destination: Variable<D>,
        level: BarrierLevel,
    },
    MemCopyAsyncTensorGlobalToShared {
        barrier: Variable<D>,
        smem_buffer: Variable<D>,
        tensor_map: Variable<D>,
        indices: Vec<Variable<D>>,
    },
    Arrive {
        barrier: Variable<D>,
        level: BarrierLevel,
    },
    ArriveTx {
        barrier: Variable<D>,
        arrive_count_update: Variable<D>,
        transaction_count_update: Variable<D>,
    },
    ExpectTx {
        barrier: Variable<D>,
        transaction_count_update: Variable<D>,
    },
    Wait {
        barrier: Variable<D>,
        level: BarrierLevel,
    },
    ArriveAndWait {
        barrier: Variable<D>,
        level: BarrierLevel,
    },
}

impl<D: Dialect> BarrierOps<D> {
    pub fn barrier_id(&self) -> u32 {
        match self {
            BarrierOps::MemCopyAsync { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::Init { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::ArriveAndWait { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::Arrive { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::ArriveTx { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::Wait { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::MemCopyAsyncTensorGlobalToShared { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::ExpectTx { barrier, .. } => barrier.id().unwrap(),
        }
    }
}

impl<D: Dialect> Display for BarrierOps<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BarrierOps::Init {
                barrier,
                level,
                with_cta_fence: with_proxy_fence,
            } => {
                let proxy_fence = match with_proxy_fence {
                    true => "cuda::device::experimental::fence_proxy_async_shared_cta();",
                    false => "",
                };
                match level {
                    // Note: Arrival token exists for cuda::thread_scope_thread, but is not public.
                    // So skip creating the token for unit barriers.
                    BarrierLevel::Unit => write!(
                        f,
                        "
cuda::barrier<cuda::thread_scope_thread> {barrier};
init(&{barrier}, 1);
{proxy_fence}
                "
                    ),
                    BarrierLevel::CubeCoop(elected_unit) => write!(
                        f,
                        "
cooperative_groups::thread_block block_{barrier} = cooperative_groups::this_thread_block();
__shared__ cuda::barrier<cuda::thread_scope_block> {barrier};
cuda::barrier<cuda::thread_scope_block>::arrival_token {barrier}_token;
if (threadIdxGlobal == {elected_unit}) {{
   init(&{barrier}, blockDimGlobal);
   {proxy_fence}
}}
__syncthreads();
"
                    ),
                    BarrierLevel::CubeManual(elected_unit) => write!(
                        f,
                        "
__shared__ cuda::barrier<cuda::thread_scope_block> {barrier};
cuda::barrier<cuda::thread_scope_block>::arrival_token {barrier}_token;
if (threadIdxGlobal == {elected_unit}) {{
   init(&{barrier}, blockDimGlobal);
   {proxy_fence}
}}
__syncthreads();
"
                    ),
                }
            }
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
            BarrierOps::MemCopyAsyncTensorGlobalToShared {
                barrier,
                smem_buffer,
                tensor_map,
                indices,
            } => {
                let rank = indices.len();
                let indices = indices.iter().rev().fold(String::new(), |mut s, it| {
                    let _ = write!(s, "{it}, ");
                    s
                });
                writeln!(
                    f,
                    "cuda::device::experimental::cp_async_bulk_tensor_{rank}d_global_to_shared(&{smem_buffer}, &{tensor_map}, {indices} {barrier});"
                )
            }
            BarrierOps::Arrive { barrier, .. } => {
                writeln!(f, "{barrier}_token = {barrier}.arrive();")
            }
            BarrierOps::ArriveTx {
                barrier,
                arrive_count_update,
                transaction_count_update,
            } => {
                writeln!(
                    f,
                    "{barrier}_token = cuda::device::barrier_arrive_tx({barrier}, {arrive_count_update}, {transaction_count_update});"
                )
            }
            BarrierOps::ExpectTx {
                barrier,
                transaction_count_update,
            } => {
                writeln!(
                    f,
                    "cuda::device::barrier_expect_tx({barrier}, {transaction_count_update});"
                )
            }
            BarrierOps::Wait { barrier, .. } => {
                writeln!(f, "{barrier}.wait(std::move({barrier}_token));")
            }
            BarrierOps::ArriveAndWait { barrier, .. } => {
                writeln!(f, "{barrier}.arrive_and_wait();")
            }
        }
    }
}
