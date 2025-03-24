use crate::{Instruction, TypeHash};
use alloc::{string::String, vec::Vec};
use core::fmt::{Display, Write};

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
    /// Initialize the barrier, optionally with a cta proxy fence
    Init {
        barrier: Variable,
        with_cta_fence: bool,
    },
    /// Copy source to destination
    MemCopyAsync {
        barrier: Variable,
        source: Variable,
    },
    MemCopyAsyncTensorGlobalToShared {
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
    ExpectTx {
        barrier: Variable,
        transaction_count_update: Variable,
    },
    Wait {
        barrier: Variable,
    },
    /// Waits until data is loaded
    ArriveAndWait {
        barrier: Variable,
    },
}

impl Display for BarrierOps {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BarrierOps::Init {
                barrier,
                with_cta_fence,
            } => match with_cta_fence {
                true => write!(f, "init_barrier_tma({barrier})"),
                false => write!(f, "init_barrier({barrier})"),
            },
            BarrierOps::MemCopyAsync { barrier, source } => {
                write!(f, "mem_copy_async({barrier}, source: {source})",)
            }
            BarrierOps::ArriveAndWait { barrier } => write!(f, "arrive_and_wait({barrier})"),
            BarrierOps::MemCopyAsyncTensorGlobalToShared {
                barrier,
                tensor_map,
                indices,
            } => {
                let rank = indices.len();
                let indices = indices.iter().fold(String::new(), |mut s, it| {
                    let _ = write!(s, "{it}, ");
                    s
                });
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
            BarrierOps::ExpectTx {
                barrier,
                transaction_count_update,
            } => write!(f, "expect_tx({barrier}, {transaction_count_update})"),
            BarrierOps::Wait { barrier } => write!(f, "wait({barrier})"),
        }
    }
}

impl From<BarrierOps> for Instruction {
    fn from(value: BarrierOps) -> Self {
        Instruction::no_out(value)
    }
}
