use crate::{Instruction, TypeHash};
use alloc::{format, string::String, vec::Vec};
use core::fmt::Display;

use crate::OperationReflect;

use super::Variable;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, Copy)]
pub enum BarrierLevel {
    Unit,
    CubeCoop(u32),
    CubeManual(u32),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = BarrierOpCode)]
/// Operations available on a barrier
pub enum BarrierOps {
    InitProxied {
        barrier: Variable,
    },
    /// Copy source to destination
    MemCopyAsync {
        barrier: Variable,
        source: Variable,
    },
    MemCopyAsyncBulkGlobalToShared {
        barrier: Variable,
        tensor_map: Variable,
        indices: Vec<Variable>,
    },
    /// Arrives at the barrier (decrements barrier count)
    Arrive {
        barrier: Variable,
    },
    ArriveTx {
        barrier: Variable,
        arrive_count_update: Variable,
        transaction_count_update: Variable,
    },
    Wait {
        barrier: Variable,
        token: Variable,
    },
    /// Waits until data is loaded
    ArriveAndWait {
        barrier: Variable,
    },
}

impl Display for BarrierOps {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BarrierOps::InitProxied { barrier } => {
                write!(f, "init_barrier({barrier})")
            }
            BarrierOps::MemCopyAsync { barrier, source } => {
                write!(f, "mem_copy_async({barrier}, source: {source})",)
            }
            BarrierOps::ArriveAndWait { barrier } => write!(f, "arrive_and_wait({barrier})"),
            BarrierOps::MemCopyAsyncBulkGlobalToShared {
                barrier,
                tensor_map,
                indices,
            } => {
                let rank = indices.len();
                let indices = indices
                    .iter()
                    .map(|it| format!("{it}, "))
                    .collect::<String>();
                write!(
                            f,
                            "mem_copy_async_bulk_global_to_shared::<{rank}>({barrier}, {tensor_map}, {indices})"
                        )
            }
            BarrierOps::Arrive { barrier } => write!(f, "arrive({barrier})"),
            BarrierOps::ArriveTx {
                barrier,
                arrive_count_update,
                transaction_count_update,
            } => write!(
                f,
                "arrive_tx({barrier}, {arrive_count_update}, {transaction_count_update})"
            ),
            BarrierOps::Wait { barrier, token } => write!(f, "wait({barrier}, {token})"),
        }
    }
}

impl From<BarrierOps> for Instruction {
    fn from(value: BarrierOps) -> Self {
        Instruction::no_out(value)
    }
}
