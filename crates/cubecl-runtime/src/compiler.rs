use crate::{
    id::KernelId,
    kernel::{CompiledKernel, KernelDefinition, KernelMetadata},
    server::ExecutionMode,
};
use alloc::string::String;
use cubecl_common::hash::StableHash;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::persistence::{
    CacheOption, Namespace, Store, StoreKey, StoreOptions, StoreValue,
};
use cubecl_ir::{ElemType, StorageType};
use thiserror::Error;

/// A store for `backend`'s compiled artifacts, or `None` when compilation
/// caching is disabled or the target has nowhere durable to put them.
///
/// `fingerprint` names what the artifacts were built for — an architecture, a
/// device — and becomes part of the namespace. Compiled code is not portable
/// across those, so this is what keeps a bundle shipped between machines from
/// serving the wrong binary. It needs no sanitizing: a namespace is a database
/// column, never a path.
pub fn compilation_store<K: StoreKey, V: StoreValue>(
    backend: &'static str,
    fingerprint: impl AsRef<str>,
) -> Option<Store<K, V>> {
    #[cfg(std_io)]
    {
        use crate::config::RuntimeConfig;

        if !crate::config::CubeClRuntimeConfig::get().compilation.cache {
            return None;
        }

        Some(Store::new(
            StoreOptions::new()
                .storage(Namespace::scoped(backend, fingerprint))
                .cache(CacheOption::Lazy),
        ))
    }

    // No file system to persist to; the caller keeps its in-memory map.
    #[cfg(not(std_io))]
    {
        let _ = (backend, fingerprint);
        None
    }
}

/// Records a freshly compiled artifact, logging rather than failing.
///
/// A refused write is routine, not exceptional: another process sharing the
/// environment may have written the key first, or the backing store may have
/// declined it. The artifact was just compiled either way, so the whole cost
/// is compiling it again next run.
pub fn store_compiled<K: StoreKey, V: StoreValue>(store: &mut Store<K, V>, key: K, value: V) {
    if let Err(err) = store.insert(key, value) {
        log::warn!("Unable to cache the compiled kernel: {}", err.reason());
    }
}

/// Kernel trait with the `ComputeShader` that will be compiled and cached based on the
/// provided id.
pub trait CubeTask<C: Compiler>: KernelMetadata + Send + Sync {
    /// Expand the kernel into its [definition](KernelDefinition).
    ///
    /// Kept separate from [`CubeTask::compile`] so a server can hash the definition to key the
    /// compilation cache, then hand the same definition back on a miss instead of expanding twice.
    fn define(&self) -> KernelDefinition;

    /// Compile a kernel definition and return the compiled form with an optional non-text
    /// representation.
    fn compile(
        &self,
        definition: KernelDefinition,
        compiler: &mut C,
        compilation_options: &C::CompilationOptions,
        mode: ExecutionMode,
        address_type: StorageType,
    ) -> Result<CompiledKernel<C>, CompilationError>;
}

/// Key for an entry in the persistent compilation cache.
///
/// The [id](KernelId) alone doesn't describe what a kernel does: it covers the kernel type, its
/// comptime arguments and its launch settings, but nothing of the body. Pairing it with a hash of
/// the expanded IR is what lets a cached artifact be invalidated when the code behind it changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct KernelCacheKey {
    /// Hash of the [kernel id](KernelId).
    pub id: StableHash,
    /// Hash of the [kernel definition](KernelDefinition).
    pub ir: StableHash,
}

impl KernelCacheKey {
    /// Create a key from a kernel id and its expanded definition.
    pub fn new(id: &KernelId, definition: &KernelDefinition) -> Self {
        Self {
            id: id.stable_hash(),
            ir: definition.stable_hash(),
        }
    }
}

/// The environment an in-memory compilation cache was built under.
///
/// A server memoizes its compiled artifacts — pipelines, loaded modules — in a
/// plain map consulted *before* the persistent [`compilation_store`]. That map
/// is bound to an environment exactly as the store is, and an environment
/// switch has to reach it: entries served from it describe the environment that
/// is gone, and, worse, a kernel answered from memory is never written to the
/// new environment's store, so a bundle exported from that environment would
/// silently be missing it.
///
/// Hold one beside the map and consult [`switched`](Self::switched) before the
/// lookup; `true` means clear the map and compile again. This is the same
/// contract [`Store`] applies to itself, for the state a store cannot see —
/// see [`cubecl_environment::environment::generation`].
#[derive(Debug)]
pub struct CompilationEnvironment {
    /// `None` when the map mirrors no store, and so is unbound.
    generation: Option<u32>,
}

impl CompilationEnvironment {
    /// Binds to the active environment when `persistent`, i.e. when the map
    /// this guards mirrors a [`compilation_store`].
    ///
    /// Unbound otherwise: with nothing persisted, a switch changes nothing
    /// about what the map holds, so resetting it would only buy a redundant
    /// compilation — the same reason the autotune cache survives a switch when
    /// its persistent cache is off.
    pub fn new(persistent: bool) -> Self {
        Self {
            generation: persistent.then(cubecl_environment::environment::generation),
        }
    }

    /// Whether the environment switched since the last call.
    ///
    /// Adopts the new generation, so one switch is reported exactly once and
    /// the caller clears its map exactly once.
    pub fn switched(&mut self) -> bool {
        let Some(generation) = self.generation else {
            return false;
        };

        let current = cubecl_environment::environment::generation();
        if current == generation {
            return false;
        }

        self.generation = Some(current);
        true
    }
}

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
        addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError>;

    /// The size of the given element in bytes.
    fn elem_size(&self, elem: ElemType) -> usize;

    /// The default extension for the runtime's kernel/shader code.
    /// Might change based on which compiler is used.
    fn extension(&self) -> &'static str;
}
