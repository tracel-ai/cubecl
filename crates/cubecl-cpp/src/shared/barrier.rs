use std::fmt::Display;

use super::{Component, Dialect, Variable};

#[derive(Debug, Clone)]
pub enum BarrierOps<D: Dialect> {
    Init {
        barrier: Variable<D>,
        num_units: u32,
        elected_unit: u32,
    },
    MemCopyAsync {
        barrier: Variable<D>,
        source: Variable<D>,
        destination: Variable<D>,
        elected_unit: Variable<D>,
    },
    Wait {
        barrier: Variable<D>,
    },
}

impl<D: Dialect> BarrierOps<D> {
    pub fn barrier_id(&self) -> u32 {
        match self {
            BarrierOps::MemCopyAsync { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::Init { barrier, .. } => barrier.id().unwrap(),
            BarrierOps::Wait { barrier } => barrier.id().unwrap(),
        }
    }
}

impl<D: Dialect> Display for BarrierOps<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BarrierOps::MemCopyAsync {
                barrier,
                source,
                destination,
                elected_unit,
            } => {
                let item = source.item();
                let size = format!("sizeof({item})");
                write!(
                    f,
                    "
if (threadIdx.x == {elected_unit}) {{
    cuda::memcpy_async({destination}, {source}, {source}_length * {size}, {barrier});
}}
"
                )
            }
            BarrierOps::Init {
                barrier,
                num_units,
                elected_unit,
            } => match num_units {
                1 => write!(
                    f,
                    "
cuda::barrier<cuda::thread_scope_thread> {barrier};
init(&{barrier}, 1);
                "
                ),
                _ => write!(
                    f,
                    "
__shared__ cuda::barrier<cuda::thread_scope_block> {barrier};
if (threadIdx.x == {elected_unit}) {{
   init(&{barrier}, {num_units});
}}
"
                ),
            },
            BarrierOps::Wait { barrier } => {
                write!(
                    f,
                    "
{barrier}.arrive_and_wait();
"
                )
            }
        }
    }
}
