use std::fmt::{Debug, Display};

use cubecl_core::CompilerRepresentation;
use rspirv::{
    binary::{Assemble, Disassemble},
    dr::Module,
};

mod branch;
mod compiler;
mod instruction;
mod item;
mod lookups;
mod subgroup;
mod target;
mod variable;

pub use compiler::*;
pub use target::*;

#[derive(Debug, Clone)]
pub struct SpirvKernel {
    pub module: Module,
    pub num_bindings: usize,
}

impl CompilerRepresentation for SpirvKernel {
    fn shared_memory_size(&self) -> usize {
        // not used in wgsl compiler
        0
    }
}

impl Display for SpirvKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.module.disassemble())
    }
}

impl SpirvKernel {
    pub fn assemble(&self) -> Vec<u32> {
        self.module.assemble()
    }
}
