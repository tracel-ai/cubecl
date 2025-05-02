use crate::{Instruction, TypeHash};
use alloc::{string::String, vec::Vec};
use core::fmt::{Display, Write};

use crate::OperationReflect;

use super::Variable;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = TmaOpCode)]
/// Operations available on a barrier
pub enum TmaOps {
    TmaStore {
        source: Variable,
        coordinates: Vec<Variable>,
        offset_source: Variable,
    },
    CommitGroup,
    WaitGroup {
        max_pending: u32,
    },
    WaitGroupRead {
        max_pending: u32,
    },
}

impl Display for TmaOps {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TmaOps::TmaStore {
                source,
                coordinates,
                offset_source,
            } => {
                let rank = coordinates.len();
                let coords = coordinates.iter().fold(String::new(), |mut s, coord| {
                    let _ = write!(s, ", {coord}");
                    s
                });
                write!(f, "tma_load::<{rank}>({source} + {offset_source} {coords})")
            }
            TmaOps::CommitGroup => write!(f, "memcpy_async_bulk_commit_group()"),
            TmaOps::WaitGroup { max_pending } => {
                write!(f, "tma_wait_group::<{max_pending}>()")
            }
            TmaOps::WaitGroupRead { max_pending } => {
                write!(f, "tma_wait_group_read::<{max_pending}>()")
            }
        }
    }
}

impl From<TmaOps> for Instruction {
    fn from(value: TmaOps) -> Self {
        Instruction::no_out(value)
    }
}
