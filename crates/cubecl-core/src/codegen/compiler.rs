use cubecl_common::ExecutionMode;

use crate::{compute::KernelDefinition, ir::Elem};
use std::fmt::Display;

/// Trait for compiled code representation
pub trait CompilerRepresentation: Display {
    /// Computes and returns the shared memory size
    fn shared_memory_size(&self) -> usize;
}

/// Compiles the representation into its own representation that can be formatted into tokens.
pub trait Compiler: Sync + Send + 'static + Clone + core::fmt::Debug {
    /// The representation for the compiled code.
    type Representation: CompilerRepresentation;
    type CompilationOptions: Send + Default + core::fmt::Debug;

    /// Compiles the [kernel definition](KernelDefinition) into the compiler's representation.
    fn compile(
        &mut self,
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Self::Representation;
    /// The size of the given element in bytes.
    fn elem_size(&self, elem: Elem) -> usize;
}

#[derive(Clone, Debug, Default)]
pub struct WgpuCompilationOptions {
    pub supports_fp_fast_math: bool,
}
