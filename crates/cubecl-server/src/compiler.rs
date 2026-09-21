//! Compilation caching for a runtime: the persistent store and the in-memory
//! cache in front of it. The [`Compiler`] contract itself lives in
//! `cubecl-runtime` and is re-exported here.

pub use cubecl_runtime::compiler::*;

use crate::id::KernelId;
use core::hash::Hash;
use cubecl_common::hash::{StableHash, StableHasher};
use cubecl_environment::collections::HashMap;
#[cfg(std_io)]
use cubecl_environment::persistence::{CacheOption, Namespace, StoreOptions};
use cubecl_environment::persistence::{Store, StoreKey, StoreValue};
use cubecl_environment::records::{Record, RecordEffect, RecordLevel, Span};

/// Platform-specific build identifier, changes on rebuild
pub type BuildId = Option<&'static [u8]>;

/// Pre-hashed build ID
pub fn build_id_hash() -> StableHash {
    StableHasher::hash_one(&buildid::build_id())
}

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

/// Stores a freshly compiled artifact, logging rather than failing, and says
/// whether the store took it.
///
/// A refused write is routine, not exceptional: another process sharing the
/// environment may have written the key first, or the backing store may have
/// declined it. The artifact was just compiled either way, so the whole cost
/// is compiling it again next run.
pub fn store_compiled<K: StoreKey, V: StoreValue>(
    store: &mut Store<K, V>,
    key: K,
    value: V,
) -> bool {
    match store.insert(key, value) {
        Ok(()) => true,
        Err(err) => {
            log::warn!("Unable to cache the compiled kernel: {}", err.reason());
            false
        }
    }
}

/// One kernel's trip through a backend's compilation path, as the environment
/// records it: compiled fresh, or loaded from the compilation store.
///
/// A hit in a server's in-memory cache is not a trip and is not recorded:
/// nothing here runs per launch.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompilationRecord {
    /// The kernel's type.
    pub kernel: alloc::string::String,
    /// The store entry naming the artifact: what tells two instances of one
    /// kernel type apart.
    pub key: KernelCacheKey,
    /// The kernel as cubecl defined it, before the backend's compiler — the
    /// IR's textual form, for a reader to render — at [`RecordLevel::Full`].
    /// Only a fresh compile defines the kernel, so a store load carries none.
    pub ir: Option<alloc::string::String>,
    /// How the artifact was obtained, and what it cost.
    pub outcome: CompilationOutcome,
    /// The source the backend compiled, at [`RecordLevel::Full`].
    pub source: Option<alloc::string::String>,
}

/// How a backend obtained a kernel's artifact.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompilationOutcome {
    /// Compiled from its definition: expanded, compiled, loaded on the device.
    Compiled {
        /// From the store miss to the artifact loaded.
        duration: core::time::Duration,
    },
    /// Read from the compilation store and loaded on the device.
    Loaded {
        /// From the lookup to the artifact loaded.
        duration: core::time::Duration,
    },
}

impl CompilationOutcome {
    /// What obtaining the artifact took, however it was obtained.
    pub fn duration(&self) -> core::time::Duration {
        match self {
            Self::Compiled { duration } | Self::Loaded { duration } => *duration,
        }
    }
}

impl Record for CompilationRecord {
    const KIND: &'static str = "compilation";
}

/// A compilation being recorded: a backend opens one where its compilation
/// path starts — past its in-memory cache — and closes it with how the
/// artifact was obtained. `None` when the environment records nothing.
#[derive(Debug)]
pub struct CompilationRecording {
    span: Span,
    kernel: &'static str,
    key: KernelCacheKey,
    ir: Option<alloc::string::String>,
}

impl CompilationRecording {
    /// Start recording `kernel_id`'s trip, when the environment records.
    pub fn new(kernel_id: &KernelId) -> Option<Self> {
        Some(Self {
            span: Span::new()?,
            kernel: kernel_id.type_name(),
            key: KernelCacheKey::new(kernel_id, build_id_hash()),
            ir: None,
        })
    }

    /// Keep the kernel's IR, from the definition the backend is about to
    /// compile, when the record [keeps code](Self::keeps_code): the textual
    /// IR runs to hundreds of KB per kernel, where the compiled artifact is
    /// tens.
    pub fn defined(&mut self, definition: &crate::kernel::KernelDefinition) {
        if self.keeps_code() {
            self.ir = Some(alloc::format!("{}", definition.body));
        }
    }

    /// The artifact came from the compilation store: the environment did not
    /// change.
    pub fn loaded(self) {
        self.close(
            |duration| CompilationOutcome::Loaded { duration },
            RecordEffect::Observed,
            None,
        );
    }

    /// Whether the record keeps the kernel's code, its IR and its source:
    /// only at [`RecordLevel::Full`], code being the heaviest thing a record
    /// can carry.
    pub fn keeps_code(&self) -> bool {
        cubecl_environment::records::level() == RecordLevel::Full
    }

    /// The artifact was compiled, from `source` when the record
    /// [keeps code](Self::keeps_code). `stored` is whether the store took
    /// it, as [`store_compiled`] answers: a compile the store did not take,
    /// or with no store to take it, changed nothing.
    pub fn compiled(self, source: Option<alloc::string::String>, stored: bool) {
        let effect = if stored {
            RecordEffect::Changed
        } else {
            RecordEffect::Observed
        };
        self.close(
            |duration| CompilationOutcome::Compiled { duration },
            effect,
            source,
        );
    }

    fn close(
        self,
        outcome: impl FnOnce(core::time::Duration) -> CompilationOutcome,
        effect: RecordEffect,
        source: Option<alloc::string::String>,
    ) {
        let Some(duration) = self.span.elapsed() else {
            return;
        };
        let record = CompilationRecord {
            kernel: self.kernel.into(),
            key: self.key,
            ir: self.ir,
            outcome: outcome(duration),
            source,
        };
        self.span.close(effect, &record);
    }
}

/// Key for an entry in the persistent compilation cache.
///
/// The [id](KernelId) alone doesn't describe what a kernel does: it covers the kernel type, its
/// comptime arguments and its launch settings, but nothing of the body. Pairing it with a hash of
/// the expanded IR is what lets a cached artifact be invalidated when the code behind it changes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct KernelCacheKey {
    /// Hash of the [kernel id](KernelId).
    pub id: StableHash,
    /// Hash of the [build id](buildid::build_id).
    pub build_id: StableHash,
}

impl KernelCacheKey {
    /// Create a key from a kernel id and the current build ID.
    pub fn new(id: &KernelId, build_id: StableHash) -> Self {
        Self {
            id: id.stable_hash(),
            build_id,
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
