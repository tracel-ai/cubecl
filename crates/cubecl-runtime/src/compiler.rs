use crate::kernel::{CompiledKernel, KernelDefinition, KernelMetadata};
use alloc::{string::String, vec::Vec};
use cubecl_common::ExecutionMode;
use cubecl_ir::ElemType;

/// Kernel trait with the ComputeShader that will be compiled and cached based on the
/// provided id.
pub trait CubeTask<C: Compiler>: KernelMetadata + Send + Sync {
    /// Compile a kernel and return the compiled form with an optional non-text representation
    fn compile(
        &self,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
    ) -> Result<CompiledKernel<C>, CompilationError>;
}

/// JIT compilation error.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum CompilationError {
    /// An instruction isn't supported.
    UnsupportedInstruction {
        /// The error context.
        context: String,
    },

    /// When multiple compilation errors are detected.
    Multiple {
        /// The error context.
        context: String,
        /// The errors.
        errors: Vec<Self>,
    },

    /// A generic compilation error.
    Generic {
        /// The error context.
        context: String,
    },
}

impl core::fmt::Display for CompilationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // TODO Make it better
        f.write_fmt(format_args!("{:?}", self))
    }
}

/// Compiles the representation into its own representation that can be formatted into tokens.
pub trait Compiler: Sync + Send + 'static + Clone + core::fmt::Debug {
    /// The representation for the compiled code.
    type Representation: core::fmt::Display;
    /// The compilation options used to configure the compiler
    type CompilationOptions: Send + Default + core::fmt::Debug;

    /// Compiles the [kernel definition](KernelDefinition) into the compiler's representation.
    fn compile(
        &mut self,
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Result<Self::Representation, CompilationError>;

    /// The size of the given element in bytes.
    fn elem_size(&self, elem: ElemType) -> usize;

    /// The default extension for the runtime's kernel/shader code.
    /// Might change based on which compiler is used.
    fn extension(&self) -> &'static str;
}
