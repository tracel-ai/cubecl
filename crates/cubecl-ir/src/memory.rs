use core::fmt::Display;

use crate::{CopyMemoryOperands, IndexOperands, StoreOperands, TypeHash, Variable};

use crate::OperationReflect;

/// Bitwise operations
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = MemoryOpCode)]
pub enum Memory {
    #[operation(pure)]
    Reference(Variable),
    #[operation(pure)]
    Index(IndexOperands),
    Load(#[args(allow_ptr, ptr_read)] Variable),
    Store(StoreOperands),
    CopyMemory(CopyMemoryOperands),
}

impl Display for Memory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Memory::Reference(variable) => write!(f, "&{variable}"),
            Memory::Index(op) => write!(f, "&{}[{}]", op.list, op.index),
            Memory::Load(variable) => write!(f, "load({variable})"),
            Memory::Store(op) => write!(f, "store({}, {})", op.ptr, op.value),
            Memory::CopyMemory(op) => {
                write!(f, "memcpy({}, {}, {})", op.source, op.target, op.len)
            }
        }
    }
}
