use std::fmt::Display;

use super::{Component, Dialect, Variable};

#[derive(Debug, Clone)]
pub enum BarrierOps<D: Dialect> {
    New {
        barrier: Variable<D>,
        unit_count: u32,
    },
    Init {
        barrier: Variable<D>,
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
            BarrierOps::New { barrier, .. } => barrier.id().unwrap(),
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
cuda::memcpy_async({destination}, {source}, {source}_length * {size}, {barrier});
"
                )
            }
            BarrierOps::New {
                barrier,
                unit_count,
            } => match unit_count {
                1 => write!(
                    f,
                    "
cuda::barrier<cuda::thread_scope_thread> {barrier};
                "
                ),
                _ => write!(
                    f,
                    "
__shared__ cuda::barrier<cuda::thread_scope_block> {barrier};
"
                ),
            },
            BarrierOps::Init { barrier } => {
                if let Variable::Barrier {
                    id: _,
                    item: _,
                    unit_count,
                } = barrier
                {
                    match unit_count {
                        1 => write!(
                            f,
                            "
    init(&{barrier}, 1);
                    "
                        ),
                        _ => write!(
                            f,
                            "
       init(&{barrier}, {unit_count});
    "
                        ),
                    }
                } else {
                    unreachable!()
                }
            }
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
