use crate::kernel::KernelDefinition;
use alloc::string::{String, ToString};
use cubecl_environment::backtrace::BackTrace;
use thiserror::Error;

/// JIT compilation error.
#[derive(Error, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum CompilationError {
    /// An instruction isn't supported.
    #[error(
        "An unsupported instruction caused the compilation to fail\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    UnsupportedInstruction {
        /// The caused of the error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// A generic compilation error.
    #[error(
        "An error caused the compilation to fail\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    Generic {
        /// The error context.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
    /// A generic compilation error.
    #[error(
        "A validation error caused the compilation to fail\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    Validation {
        /// The error context.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
}

impl core::fmt::Debug for CompilationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

impl From<pliron::result::Error> for CompilationError {
    fn from(value: pliron::result::Error) -> Self {
        CompilationError::Validation {
            reason: value.to_string(),
            backtrace: BackTrace::capture(),
        }
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
    ) -> Result<Self::Representation, CompilationError>;

    /// What the compiled kernel does with each buffer binding, by buffer
    /// position — the visibility analysis's answer, when the representation
    /// kept it (see [`BufferIOAttr`](crate::kernel::BufferIOAttr)).
    ///
    /// `None` reads as every buffer both read and written, the conservative
    /// direction. A compiler overriding this must answer from the IR
    /// attributes the annotate pass stamped, never from what its shader
    /// language kept — wgpu's shader visibility, for one, is deliberately
    /// forced wider than the kernel's own behavior.
    fn buffer_io(
        _repr: &Self::Representation,
    ) -> Option<alloc::vec::Vec<crate::kernel::BufferIOAttr>> {
        None
    }

    /// The default extension for the runtime's kernel/shader code.
    /// Might change based on which compiler is used.
    fn extension(&self) -> &'static str;

    /// Short identifier of the language this compiler produces, such as
    /// `"wgsl"` or `"cuda"`.
    ///
    /// What a [`PrecompiledSource`](crate::kernel::PrecompiledSource) has to
    /// name to be accepted by this compiler.
    fn lang_tag(&self) -> &'static str;
}
