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
/// nothing here runs per launch. Neither is a trip that fails: the launch
/// error carries that account, and the environment holds nothing of it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompilationRecord {
    /// The kernel's type.
    pub kernel: alloc::string::String,
    /// The store entry naming the artifact: what tells two instances of one
    /// kernel type apart.
    pub key: KernelCacheKey,
    /// The kernel as cubecl defined it, before the backend's compiler — the
    /// IR's textual form, for a reader to render — at [`RecordLevel::Full`].
    /// Only a trip that misses the store defines the kernel, so a
    /// [`Loaded`](CompilationOutcome::Loaded) one carries none.
    pub ir: Option<alloc::string::String>,
    /// How the artifact was obtained.
    pub outcome: CompilationOutcome,
    /// What obtaining it took, from where the trip started to the artifact
    /// loaded on the device.
    pub duration: core::time::Duration,
    /// The source the backend compiled, at [`RecordLevel::Full`].
    pub source: Option<alloc::string::String>,
}

/// How a backend obtained a kernel's artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompilationOutcome {
    /// Compiled from its definition: expanded, compiled by the backend's
    /// compiler, loaded on the device.
    Compiled,
    /// Read from the compilation store and loaded on the device.
    Loaded,
    /// Expanded to a source the store already held an artifact for, under
    /// another key: the artifact was moved under this one and loaded, and the
    /// backend's compiler never ran.
    Rekeyed,
}

impl Record for CompilationRecord {
    const KIND: &'static str = "compilation";
}

/// A compilation being recorded: a backend opens one where its compilation
/// path starts — past its in-memory cache — tells it what the trip goes
/// through, and closes it with how the artifact was obtained. Every call is a
/// no-op when the environment records nothing, and one dropped unclosed, by a
/// trip that failed, records nothing.
#[derive(Debug)]
pub struct CompilationRecording {
    open: Option<OpenRecording>,
}

/// What a [`CompilationRecording`] holds while the environment records.
#[derive(Debug)]
struct OpenRecording {
    span: Span,
    kernel: &'static str,
    key: KernelCacheKey,
    ir: Option<alloc::string::String>,
    source: Option<alloc::string::String>,
}

impl CompilationRecording {
    /// Start recording `kernel_id`'s trip.
    pub fn new(kernel_id: &KernelId) -> Self {
        let open = Span::new().map(|span| OpenRecording {
            span,
            kernel: kernel_id.type_name(),
            key: KernelCacheKey::new(kernel_id, build_id_hash()),
            ir: None,
            source: None,
        });
        Self { open }
    }

    /// The kernel was defined: keep its IR, at [`RecordLevel::Full`] only.
    /// The textual IR runs to hundreds of KB per kernel, where the compiled
    /// artifact is tens.
    pub fn defined(&mut self, definition: &crate::kernel::KernelDefinition) {
        if let Some(open) = self.open.as_mut().filter(|_| keeps_code()) {
            open.ir = Some(alloc::format!("{}", definition.body));
        }
    }

    /// The backend's compiler produced `source`: keep it, at
    /// [`RecordLevel::Full`] only.
    pub fn source(&mut self, source: &str) {
        if let Some(open) = self.open.as_mut().filter(|_| keeps_code()) {
            open.source = Some(source.into());
        }
    }

    /// The artifact came from the compilation store: the environment did not
    /// change.
    pub fn loaded(self) {
        self.close(CompilationOutcome::Loaded, RecordEffect::Observed);
    }

    /// The artifact was compiled. `stored` is whether the store took it, as
    /// [`store_compiled`] answers: a compile the store did not take, or with
    /// no store to take it, changed nothing.
    pub fn compiled(self, stored: bool) {
        self.close(CompilationOutcome::Compiled, effect(stored));
    }

    /// The artifact was already stored under another key, and moved under
    /// this one. `stored` is whether the store took it there.
    pub fn rekeyed(self, stored: bool) {
        self.close(CompilationOutcome::Rekeyed, effect(stored));
    }

    fn close(self, outcome: CompilationOutcome, effect: RecordEffect) {
        let Some(open) = self.open else {
            return;
        };
        let Some(duration) = open.span.elapsed() else {
            return;
        };
        let record = CompilationRecord {
            kernel: open.kernel.into(),
            key: open.key,
            ir: open.ir,
            outcome,
            duration,
            source: open.source,
        };
        open.span.close(effect, &record);
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

/// Whether a record keeps the kernel's code, its IR and its source: code is
/// the heaviest thing a record can carry.
fn keeps_code() -> bool {
    cubecl_environment::records::level() == RecordLevel::Full
}

/// An artifact the store took is the environment changing.
fn effect(stored: bool) -> RecordEffect {
    if stored {
        RecordEffect::Changed
    } else {
        RecordEffect::Observed
    }
}
