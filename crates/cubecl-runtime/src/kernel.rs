use alloc::string::String;
use core::hash::Hash;

use cubecl_ir::{
    ElemType, Scope,
    metadata::Info,
    pliron::{format, value::Value},
    settings::KernelSettings,
};
use serde::{Deserialize, Serialize};

use crate::id::KernelId;

/// Implement this trait to create a [kernel definition](KernelDefinition).
pub trait KernelMetadata: core::any::Any + Send + Sync + 'static {
    /// Name of the kernel for debugging.
    fn name(&self) -> &'static str {
        core::any::type_name::<Self>()
    }

    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> KernelId;

    /// Type of addresses in this kernel
    fn address_type(&self) -> ElemType;
}

#[allow(missing_docs)]
pub struct KernelDefinition {
    pub body: Scope,
    pub info: Info,
    pub settings: KernelSettings,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
/// Global argument of a kernel.
pub struct KernelArg {
    /// The index of the arg.
    pub id: usize,
    /// The value the argument is bound to.
    pub value: Value,
    /// Whether the argument has metadata.
    pub has_extended_meta: bool,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScalarKernelArg {
    pub ty: ElemType,
    pub count: usize,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize, Hash)]
#[allow(missing_docs)]
#[format]
pub enum Visibility {
    Uniform,
    Read,
    ReadWrite,
}

/// What a compiled kernel does with each buffer binding, by buffer position.
///
/// The IR owns this concept — the visibility analysis stamps it on the entry
/// function's arguments — so the launch path reads that enum rather than a
/// copy of it that could drift. Re-exported here because a backend reaching
/// for it is holding a `CompiledKernel` (in `cubecl-server`), not an IR
/// context.
pub use cubecl_ir::attributes::BufferIOAttr;

/// A hand-written kernel's own compiled text, standing in for what the
/// compiler would have produced from a [`KernelDefinition`].
///
/// The text goes to the backend as is, so two things the compiler would have
/// settled are the kernel's to settle:
///
/// - `lang` names the language the text is written in, and must equal the
///   [`lang_tag`](crate::compiler::Compiler::lang_tag) of the compiler the
///   client runs. `CompiledKernel::compile` in `cubecl-server` refuses a
///   mismatch, so CUDA C++ handed to a wgpu client is a
///   [`CompilationError`](crate::compiler::CompilationError), not a naga
///   parse error at first launch.
/// - The kernel's [`id`](crate::kernel::KernelMetadata::id) must cover the
///   text, for instance through [`KernelId::info`](crate::id::KernelId::info)
///   with a hash of it. Every compilation cache, in memory and on disk, is
///   keyed by that id and never sees the source, so two kernels with the
///   same id and different text would share one compiled artifact.
///
/// There is no representation to read a dynamic shared memory size from, so
/// a precompiled kernel is launched with none: what it needs, it declares
/// statically in the text.
pub struct PrecompiledSource {
    /// The compiled source, in the target language.
    pub source: String,
    /// The name of the entrypoint within `source`.
    pub entrypoint_name: String,
    /// The language `source` is written in, as the target compiler tags it.
    pub lang: &'static str,
}

/// Kernel that can be defined
pub trait CubeKernel: KernelMetadata {
    /// Define the kernel for compilation
    fn define(&self) -> KernelDefinition;

    /// The kernel's own compiled source, for a hand-written kernel that
    /// carries target-language text rather than IR to compile.
    ///
    /// `None`, the default, compiles what [`define`](Self::define) returns.
    fn source(&self) -> Option<PrecompiledSource> {
        None
    }
}
