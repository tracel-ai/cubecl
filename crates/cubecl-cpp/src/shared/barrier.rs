use std::fmt::Display;

use super::{Component, Dialect, Variable};

#[derive(Debug, Clone)]
pub enum BarrierOps<D: Dialect> {
    Init {
        barrier: Variable<D>,
        level: u8,
        num_units: u32,
        elected_unit: u32,
    },
    MemCopyAsync {
        barrier: Variable<D>,
        source: Variable<D>,
        destination: Variable<D>,
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
            } => {
                let item = source.item();
                let size = format!("sizeof({item})");
                write!(
                    f,
                    "
cuda::memcpy_async({destination}, {source}, {size}, {barrier});
                "
                )
            }
            BarrierOps::Init {
                barrier,
                level,
                num_units,
                elected_unit,
            } => match level {
                0 => write!(
                    f,
                    "
cuda::barrier<cuda::thread_scope_thread> {barrier};
{barrier}.init(1);
                "
                ),
                1 => write!(
                    f,
                    "
__shared__ cuda::barrier<cuda::thread_scope_block> {barrier};
if (threadIdx.x == {elected_unit}) {{
   init(&{barrier}, {num_units});
}}
                    "
                ),
                _ => unreachable!(),
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
