use core::{cell::Ref, fmt::Display};

use alloc::string::ToString;
use alloc::vec::Vec;

use crate::{GlobalState, GlobalStateInner};

use super::{Instruction, Variable};

pub trait Processor: core::fmt::Debug {
    fn transform(&self, processing: ScopeProcessing) -> ScopeProcessing;
}

/// Information necessary when compiling a scope.
pub struct ScopeProcessing {
    /// The variable declarations.
    pub variables: Vec<Variable>,
    /// The operations.
    pub instructions: Vec<Instruction>,
    /// The global state
    pub global_state: GlobalState,
}

impl ScopeProcessing {
    pub fn state(&self) -> Ref<'_, GlobalStateInner> {
        self.global_state.borrow()
    }
}

impl Display for ScopeProcessing {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "{{")?;
        for instruction in self.instructions.iter() {
            let instruction_str = instruction.to_string();
            if !instruction_str.is_empty() {
                writeln!(f, "    {instruction_str}")?;
            }
        }
        write!(f, "}}")?;
        Ok(())
    }
}
