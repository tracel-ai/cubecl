#![allow(unknown_lints, unnecessary_transmutes)]

use std::fmt::{Debug, Display};

use cubecl_core::compute::Binding;
use cubecl_opt::Optimizer;
use item::Elem;
use rspirv::{
    binary::{Assemble, Disassemble},
    dr::Module,
};

mod arithmetic;
mod atomic;
mod bitwise;
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
mod transformers;
mod variable;

pub use compiler::*;
pub use target::*;

#[derive(Debug, Clone)]
pub struct SpirvKernel {
    pub module: Module,
    pub optimizer: Optimizer,
    pub bindings: Vec<Binding>,
    pub scalars: Vec<(Elem, usize)>,
    pub has_metadata: bool,
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
