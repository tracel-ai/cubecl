#![allow(unknown_lints, unnecessary_transmutes)]

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use cubecl_core::prelude::Visibility;
use rspirv::{binary::Disassemble, dr::Module};

mod atomic;
pub mod attributes;
pub mod builtin;
mod cmma;
mod compiler;
mod debug;
mod extensions;
mod globals;
mod item;
mod lookups;
pub mod lower;
pub mod ops;
mod subgroup;
mod sync;
mod target;
mod tensor_indexing;
mod transformers;
pub mod types;
mod value;

pub use compiler::*;
use serde::{Deserialize, Serialize};
pub use target::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpirvKernel {
    #[serde(skip)]
    pub module: Option<Arc<Module>>,

    pub assembled_module: Vec<u32>,
    pub bindings: Vec<Visibility>,
    pub shared_size: usize,
    pub immediate_size: Option<usize>,
    pub info_visibility: Visibility,
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
    pub kernel: SpirvKernel,
}

impl SpirvCacheEntry {
    pub fn new(entrypoint_name: String, kernel: SpirvKernel) -> Self {
        SpirvCacheEntry {
            entrypoint_name,
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
