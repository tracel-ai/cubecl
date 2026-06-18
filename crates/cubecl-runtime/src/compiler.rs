use crate::kernel::{CompiledKernel, KernelDefinition, KernelMetadata};
use alloc::string::String;
use core::hash::Hash;
use cubecl_common::hash::StableHash;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_environment::persistence::{
    CacheOption, Namespace, Store, StoreKey, StoreOptions, StoreValue,
};
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

/// A server's in-memory compilation cache: the compiled artifacts it memoizes
/// — pipelines, loaded modules — in front of a persistent [`compilation_store`].
///
/// Entries are dropped when the environment switches, because the map is bound
/// to an environment exactly as the store it mirrors is. One served after a
/// switch would describe the environment that is gone, and, worse, would never
/// be written to the new environment's store, so a bundle exported from that
/// environment would silently be missing that kernel. This is the same contract
/// [`Store`] applies to itself, for the state a store cannot see — see
/// [`cubecl_environment::environment::generation`].
///
/// Every accessor resets before it answers, so a backend has nothing to
/// remember beyond using this in place of a plain map.
#[derive(Debug)]
pub struct CompilationCache<K, V> {
    entries: HashMap<K, V>,
    /// The generation the entries were built under, or `None` when the cache
    /// mirrors no store and so is unbound.
    generation: Option<u32>,
}

impl<K: Eq + Hash, V> CompilationCache<K, V> {
    /// An empty cache in front of `store`, bound to the active environment
    /// exactly when that store exists.
    ///
    /// Unbound otherwise: with nothing persisted, a switch changes nothing
    /// about what the cache holds, so resetting it would only buy a redundant
    /// compilation — the same reason the autotune cache survives a switch when
    /// its persistent cache is off.
    pub fn mirroring<SK: StoreKey, SV: StoreValue>(store: &Option<Store<SK, SV>>) -> Self {
        Self {
            entries: HashMap::new(),
            generation: store
                .is_some()
                .then(cubecl_environment::environment::generation),
        }
    }

    /// An empty cache that no environment switch ever resets, for a backend
    /// with no persistent store to mirror.
    pub fn unbound() -> Self {
        Self {
            entries: HashMap::new(),
            generation: None,
        }
    }

    /// The artifact compiled for `key`, if it is still valid.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.reset_if_switched();
        self.entries.get(key)
    }

    /// Whether an artifact for `key` is cached and still valid.
    pub fn contains(&mut self, key: &K) -> bool {
        self.reset_if_switched();
        self.entries.contains_key(key)
    }

    /// Records a freshly compiled artifact.
    pub fn insert(&mut self, key: K, value: V) {
        self.reset_if_switched();
        self.entries.insert(key, value);
    }

    /// Drops every entry when the environment switched since the last access,
    /// adopting the new generation so one switch costs one reset.
    fn reset_if_switched(&mut self) {
        let Some(generation) = self.generation else {
            return;
        };

        let current = cubecl_environment::environment::generation();
        if current == generation {
            return;
        }

        log::debug!("Environment switched, dropping the in-memory compilation cache");
        self.generation = Some(current);
        self.entries.clear();
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
    ) -> Result<Self::Representation, CompilationError>;

    /// The default extension for the runtime's kernel/shader code.
    /// Might change based on which compiler is used.
    fn extension(&self) -> &'static str;
}
