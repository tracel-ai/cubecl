#![allow(unknown_lints, unnecessary_transmutes)]

use cubecl_runtime::kernel::BufferIOAttr;
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use cubecl_core::prelude::Visibility;
use rspirv::{binary::Disassemble, dr::Module};

pub mod attributes;
pub mod compiler;
pub mod lower;
pub mod ops;
pub mod target;
pub mod types;

pub use compiler::*;
use serde::{Deserialize, Serialize};
pub use target::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpirvKernel {
    #[serde(skip)]
    pub module: Option<Arc<Module>>,

    pub assembled_module: Vec<u32>,
    pub bindings: Vec<Visibility>,
    /// What the kernel does with each buffer binding, by buffer position --
    /// the four-state answer the launch path's taint bookkeeping consumes,
    /// never collapsed the way [`bindings`](Self::bindings) is. Defaulted for
    /// entries persisted before the field existed, which reads as no answer:
    /// every buffer both read and written.
    #[serde(default)]
    pub io: Option<Vec<BufferIOAttr>>,
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
