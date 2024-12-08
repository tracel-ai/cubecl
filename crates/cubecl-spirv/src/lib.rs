use std::fmt::{Debug, Display};

use cubecl_core::{ir::Binding, CompilerRepresentation};
use cubecl_opt::Optimizer;
use rspirv::{
    binary::{Assemble, Disassemble},
    dr::Module,
};

mod atomic;
mod branch;
mod cmma;
mod compiler;
mod debug;
mod extensions;
mod globals;
mod instruction;
mod item;
mod lookups;
mod metadata;
mod subgroup;
mod sync;
mod target;
mod variable;

pub use compiler::*;
pub use target::*;

#[derive(Debug, Clone)]
pub struct SpirvKernel {
    pub module: Module,
    pub optimizer: Optimizer,
    pub bindings: Vec<Binding>,
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
