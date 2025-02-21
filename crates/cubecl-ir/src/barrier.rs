use crate::{Instruction, TypeHash};
use core::fmt::Display;

use crate::OperationReflect;

use super::Variable;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = BarrierOpCode)]
/// Operations available on a barrier
pub enum BarrierOps {
    /// Copy source to destination
    MemCopyAsync {
        barrier: Variable,
        source: Variable,
        destination: Variable,
        elected_unit: Variable,
    },
    /// Waits until data is loaded
    Wait { barrier: Variable },
}

impl Display for BarrierOps {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BarrierOps::MemCopyAsync {
                barrier,
                source,
                destination,
                elected_unit,
            } => write!(
                f,
                "mem_copy_async({barrier}, source: {source}, destination: {destination}, elected_unit: {elected_unit})",
            ),
            BarrierOps::Wait { barrier } => write!(f, "wait({barrier})"),
        }
    }
}

impl From<BarrierOps> for Instruction {
    fn from(value: BarrierOps) -> Self {
        Instruction::no_out(value)
    }
}
