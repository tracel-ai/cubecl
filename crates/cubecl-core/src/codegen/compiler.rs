use crate::ir::{Elem, KernelDefinition, Allocator};
use cubecl_runtime::ExecutionMode;
use std::fmt::Display;

/// Trait for compiled code representation
pub trait CompilerRepresentation: Display {
    /// Computes and returns the shared memory size
    fn shared_memory_size(&self) -> usize;
}

/// Compiles the representation into its own representation that can be formatted into tokens.
pub trait Compiler: Sync + Send + 'static + Clone + Default + core::fmt::Debug {
    /// The representation for the compiled code.
    type Representation: CompilerRepresentation;
    type CompilationOptions: Send + Default + core::fmt::Debug;

    /// Compiles the [kernel definition](KernelDefinition) into the compiler's representation.
    fn compile(
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
    ) -> Self::Representation;
    /// The size of the given element in bytes.
    fn elem_size(elem: Elem) -> usize;
    fn local_allocator() -> Allocator;
    /// The maximal size of a shared memory, in bytes
    fn max_shared_memory_size() -> usize;
}
