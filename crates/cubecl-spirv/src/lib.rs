#![allow(unknown_lints, unnecessary_transmutes)]

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpirvKernel {
    #[serde(skip)]
    pub module: Option<Arc<Module>>,
    #[serde(skip)]
    pub optimizer: Option<Arc<Optimizer>>,

    pub assembled_module: Vec<u32>,
    pub bindings: Vec<Binding>,
    pub scalars: Vec<(Elem, usize)>,
    pub has_metadata: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpirvAssembly {}

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
    pub kernel: SpirvKernel,
}

impl SpirvCacheEntry {
    pub fn new(entrypoint_name: String, cube_dim: CubeDim, kernel: SpirvKernel) -> Self {
        SpirvCacheEntry {
            entrypoint_name,
            cube_dim,
            kernel,
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
