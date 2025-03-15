use cubecl_common::ExecutionMode;

use crate::{compute::KernelDefinition, ir::Elem};

/// Compiles the representation into its own representation that can be formatted into tokens.
pub trait Compiler: Sync + Send + 'static + Clone + core::fmt::Debug {
    /// The representation for the compiled code.
    type Representation: core::fmt::Display;
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

    /// The default extension for the runtime's kernel/shader code.
    /// Might change based on which compiler is used.
    fn extension(&self) -> &'static str;
}

#[derive(Clone, Debug, Default)]
pub struct WgpuCompilationOptions {
    pub supports_fp_fast_math: bool,
}
