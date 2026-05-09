use crate::{Instruction, TypeHash};
use alloc::{format, string::String, vec::Vec};
use core::fmt::{Display, Write};

use crate::OperationReflect;

use super::Variable;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, Copy, PartialOrd, Ord)]
pub enum BarrierLevel {
    Unit,
    Cube,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = BarrierOpCode)]
/// Operations available on a barrier
pub enum BarrierOps {
    /// Declare the barrier, without doing any initialization
    Declare {
        barrier: Variable,
    },
    /// Initialize the barrier, optionally with a cta proxy fence
    Init {
        barrier: Variable,
        is_elected: Variable,
        arrival_count: Variable,
    },
    /// Manually initialize the barrier with an arrival count, without any sync or election handling
    InitManual {
        barrier: Variable,
        arrival_count: Variable,
    },
    /// Copy source to destination
    MemCopyAsync {
        barrier: Variable,
        #[args(allow_ptr, ptr_read)]
        source: Variable,
        #[args(allow_ptr, ptr_write)]
        destination: Variable,
        source_length: Variable,
    },
    /// Copy source to destination, with cooperative behaviour
    MemCopyAsyncCooperative {
        barrier: Variable,
        #[args(allow_ptr, ptr_read)]
        source: Variable,
        #[args(allow_ptr, ptr_write)]
        destination: Variable,
        source_length: Variable,
    },
    /// Copy source to destination, with transaction count
    MemCopyAsyncTx {
        barrier: Variable,
        #[args(allow_ptr, ptr_read)]
        source: Variable,
        #[args(allow_ptr, ptr_write)]
        destination: Variable,
        source_length: Variable,
    },
    /// Copy source to destination
    CopyAsync {
        #[args(allow_ptr, ptr_read)]
        source: Variable,
        #[args(allow_ptr, ptr_write)]
        destination: Variable,
        source_length: Variable,
        copy_length: u32,
        checked: bool,
    },
    TmaLoad {
        barrier: Variable,
        tensor_map: Variable,
        #[args(allow_ptr, ptr_write)]
        destination: Variable,
        indices: Vec<Variable>,
    },
    TmaLoadIm2col {
        barrier: Variable,
        tensor_map: Variable,
        #[args(allow_ptr, ptr_write)]
        destination: Variable,
        indices: Vec<Variable>,
        offsets: Vec<Variable>,
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
    CommitCopyAsync {
        barrier: Variable,
    },
    ExpectTx {
        barrier: Variable,
        transaction_count_update: Variable,
    },
    Wait {
        barrier: Variable,
        token: Variable,
    },
    WaitParity {
        barrier: Variable,
        phase: Variable,
    },
    /// Waits until data is loaded
    ArriveAndWait {
        barrier: Variable,
    },
}

impl Display for BarrierOps {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BarrierOps::Declare { .. } => Ok(()),
            BarrierOps::Init {
                barrier,
                arrival_count,
                ..
            } => write!(f, "{barrier}.init_barrier({arrival_count})"),
            BarrierOps::InitManual {
                barrier,
                arrival_count,
            } => write!(f, "{barrier}.init_barrier({arrival_count})"),
            BarrierOps::MemCopyAsync {
                barrier, source, ..
            } => {
                write!(f, "mem_copy_async({barrier}, source: {source})",)
            }
            BarrierOps::MemCopyAsyncCooperative {
                barrier, source, ..
            } => {
                write!(f, "mem_copy_async_cooperative({barrier}, source: {source})",)
            }
            BarrierOps::MemCopyAsyncTx {
                barrier, source, ..
            } => {
                write!(f, "mem_copy_async_tx({barrier}, source: {source})",)
            }
            BarrierOps::CopyAsync {
                source,
                destination,
                source_length,
                copy_length,
                checked,
            } => {
                let source_slice = if *checked {
                    format!("[..{source_length}]")
                } else {
                    String::new()
                };
                write!(
                    f,
                    "copy_async(source: {source}{source_slice}, destination: {destination}, bytes: {copy_length})",
                )
            }
            BarrierOps::ArriveAndWait { barrier } => write!(f, "arrive_and_wait({barrier})"),
            BarrierOps::TmaLoad {
                barrier,
                tensor_map,
                destination,
                indices,
            } => {
                let rank = indices.len();
                let indices = indices.iter().fold(String::new(), |mut s, it| {
                    let _ = write!(s, "{it}, ");
                    s
                });
                write!(
                    f,
                    "tma_load::<{rank}>(bar: {barrier}, from: {tensor_map}, to: {destination}, indices: {indices})"
                )
            }
            BarrierOps::TmaLoadIm2col {
                barrier,
                tensor_map,
                destination,
                indices,
                offsets,
            } => {
                let rank = indices.len();
                let indices = indices.iter().fold(String::new(), |mut s, it| {
                    let _ = write!(s, "{it}, ");
                    s
                });
                let offsets = offsets.iter().fold(String::new(), |mut s, it| {
                    let _ = write!(s, "{it}, ");
                    s
                });
                write!(
                    f,
                    "tma_load_im2col::<{rank}>(bar: {barrier}, from: {tensor_map}, to: {destination}, indices: ({indices}), offsets: ({offsets}))"
                )
            }
            BarrierOps::Arrive { barrier } => write!(f, "arrive({barrier})"),
            BarrierOps::CommitCopyAsync { barrier } => write!(f, "commit_copy_async({barrier})"),
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
            BarrierOps::Wait { barrier, token } => write!(f, "wait({barrier}, {token})"),
            BarrierOps::WaitParity { barrier, phase } => {
                write!(f, "wait_parity({barrier}, {phase})")
            }
        }
    }
}

impl Display for BarrierLevel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BarrierLevel::Unit => f.write_str("unit"),
            BarrierLevel::Cube => f.write_str("cube"),
        }
    }
}

impl From<BarrierOps> for Instruction {
    fn from(value: BarrierOps) -> Self {
        Instruction::no_out(value)
    }
}
