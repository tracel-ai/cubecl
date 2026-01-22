#![allow(unknown_lints, unnecessary_transmutes)]

use std::fmt::{Debug, Display};

use cubecl_core::{CubeDim, prelude::Binding};
use cubecl_opt::Optimizer;
use item::Elem;
use rspirv::{binary::Disassemble, dr::Module};

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
use serde::{Deserialize, Serialize};
pub use target::*;

#[derive(Debug, Clone)]
pub struct SpirvKernel {
    pub module: Option<Module>,
    pub optimizer: Option<Optimizer>,

    pub assembled_module: Vec<u32>,
    pub bindings: Vec<Binding>,
    pub scalars: Vec<(Elem, usize)>,
    pub has_metadata: bool,
}

impl Eq for SpirvKernel {}
impl PartialEq for SpirvKernel {
    fn eq(&self, other: &Self) -> bool {
        self.assembled_module == other.assembled_module
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpirvCacheEntry {
    pub entrypoint_name: String,
    pub cube_dim: CubeDim,
    pub assembled_module: Vec<u32>,
    pub bindings: Vec<Binding>,
    pub scalars: Vec<(Elem, usize)>,
    pub has_metadata: bool,
}

impl SpirvCacheEntry {
    pub fn new(entrypoint_name: &str, cube_dim: CubeDim, kernel: &SpirvKernel) -> Self {
        SpirvCacheEntry {
            entrypoint_name: entrypoint_name.to_string(),
            cube_dim,
            assembled_module: kernel.assembled_module.clone(),
            bindings: kernel.bindings.clone(),
            scalars: kernel.scalars.clone(),
            has_metadata: kernel.has_metadata,
        }
    }

    pub fn kernel(&self) -> SpirvKernel {
        SpirvKernel {
            module: None,
            optimizer: None,
            assembled_module: self.assembled_module.clone(),
            bindings: self.bindings.clone(),
            scalars: self.scalars.clone(),
            has_metadata: self.has_metadata,
        }
    }
}

impl Display for SpirvKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(module) = &self.module {
            write!(f, "{}", module.disassemble())
        } else {
            f.write_str("SPIR-V")
        }
    }
}
