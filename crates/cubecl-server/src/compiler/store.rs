//! Where a backend keeps the artifacts it compiled, between runs.

use crate::id::KernelId;
use cubecl_common::hash::{StableHash, StableHasher};
#[cfg(compilation_cache)]
use cubecl_environment::persistence::{CacheOption, Namespace, StoreOptions};
use cubecl_environment::persistence::{Store, StoreKey, StoreValue};

/// The kernels the work loaded in the active environment, by the stable hash
/// of their id: what an environment's prune keeps of the artifacts its stores
/// hold, the rest being what only an autotune race loaded.
///
/// A kernel is recorded the first time a launch issued for the
/// [work](crate::execution::LaunchPurpose::Work) loads it, whatever shape or
/// variant it was launched for, and never for a race's launch. Kept beside the
/// compilation stores, under the same switch and the same configuration: with
/// nothing compiled kept, there is nothing to record either.
#[derive(Debug)]
pub struct WorkLaunchStore {
    store: Option<Store<StableHash, ()>>,
}

impl WorkLaunchStore {
    /// The root of the namespace the kernels are recorded under.
    pub const ROOT: &'static str = "launched";

    /// The store of the active environment, or nothing to record into when
    /// compilation caching is disabled or the target has nowhere durable to
    /// put it.
    pub fn new() -> Self {
        Self {
            store: compilation_store(Self::ROOT, "work"),
        }
    }

    /// Record `kernel` when the launch loading it on this thread is the
    /// work's. Answers whether the kernel needs no recording any more: it was
    /// recorded, or there is nowhere to record it.
    pub fn record(&mut self, kernel: &KernelId) -> bool {
        let Some(store) = self.store.as_mut() else {
            return true;
        };
        if crate::execution::LaunchPurpose::running() == crate::execution::LaunchPurpose::Race {
            return false;
        }
        store_compiled(store, kernel.stable_hash(), ())
    }
}

impl Default for WorkLaunchStore {
    fn default() -> Self {
        Self::new()
    }
}

/// The first segment of a namespace: who wrote it — `autotune`, a backend's
/// compiler, the records, an application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamespaceRoot<'a>(&'a str);

impl<'a> NamespaceRoot<'a> {
    /// The root of `namespace`.
    pub fn new(namespace: &'a str) -> Self {
        Self(namespace.split('/').next().unwrap_or_default())
    }

    /// The root of the records, which every layout of them shares.
    pub fn records() -> Self {
        Self::new(cubecl_environment::records::ROOT)
    }

    /// The root as written.
    pub fn as_str(self) -> &'a str {
        self.0
    }

    /// Whether the namespace may hold compiled artifacts: everything but the
    /// measurements, the records and the work's launches cubecl keeps beside
    /// them.
    pub fn holds_artifacts(self) -> bool {
        ![
            crate::tune::ROOT,
            crate::throughput::ROOT,
            Self::records().0,
            WorkLaunchStore::ROOT,
        ]
        .contains(&self.0)
    }
}

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
    #[cfg(compilation_cache)]
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
    #[cfg(not(compilation_cache))]
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

/// Key for an entry in the persistent compilation cache.
///
/// The [id](KernelId) alone doesn't describe what a kernel does: it covers the kernel type, its
/// comptime arguments and its launch settings, but nothing of the body. Pairing it with a hash of
/// the expanded IR is what lets a cached artifact be invalidated when the code behind it changes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct KernelCacheKey {
    /// Hash of the [kernel id](KernelId): what the
    /// [work's launches](WorkLaunchStore) record, so an artifact is known by
    /// the kernel that loads it, whichever variant it is.
    pub id: StableHash,
    /// Hash of what else decides the artifact for that kernel: the variant a
    /// backend compiles it in, the device properties its code is specialized
    /// for. Zero when the kernel id says everything; not written then, so such
    /// a key encodes as `id` and `build_id` alone.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub variant: StableHash,
    /// Hash of the [build id](buildid::build_id).
    pub build_id: StableHash,
}

impl KernelCacheKey {
    /// Create a key from a kernel id and the current build ID.
    pub fn new(id: &KernelId, build_id: StableHash) -> Self {
        Self {
            id: id.stable_hash(),
            variant: 0,
            build_id,
        }
    }

    /// The key of the same kernel, compiled in `variant`.
    pub fn variant(self, variant: StableHash) -> Self {
        Self { variant, ..self }
    }

    /// Create a key from an artifact id (kernel and variant) and the current build ID.
    pub fn from_artifact_id<V: core::hash::Hash>(
        id: &super::ArtifactId<V>,
        build_id: StableHash,
    ) -> Self {
        Self::new(&id.kernel, build_id).variant(StableHasher::hash_one(&id.variant))
    }

    /// The artifact a compilation store's entry is of: its key for an
    /// artifact, its value for an entry mapping a source to one. `None` for
    /// an entry of anything else.
    #[cfg(compilation_cache)]
    pub fn of_entry(key: &[u8], value: &[u8]) -> Option<Self> {
        let decode = |bytes: &[u8]| ciborium::from_reader::<Self, _>(bytes).ok();
        decode(key).or_else(|| decode(value))
    }
}

/// What an [`ArtifactStore`] kept by kernel and by source names its two maps,
/// as [`compilation_store`] takes a backend's name.
#[derive(Debug, Clone, Copy)]
pub struct StoreNames {
    /// The map from kernel to artifact.
    pub kernels: &'static str,
    /// The map from source to the kernel that first finalized it.
    pub sources: &'static str,
}

/// The compiled artifacts of one backend and device, kept by kernel and, for a
/// backend that opts in, by the source each was finalized from — see
/// [`ArtifactCompiler::source`](super::ArtifactCompiler::source).
///
/// Holds nothing when compilation caching is disabled or the target has
/// nowhere durable to put it: every take then misses and every keep refuses.
#[derive(Debug)]
pub struct ArtifactStore<A> {
    by_kernel: Option<Store<KernelCacheKey, A>>,
    /// Maps a source's hash to the key of the kernel that first finalized it.
    by_source: Option<Store<StableHash, KernelCacheKey>>,
    build_id: StableHash,
}

impl<A: StoreValue> ArtifactStore<A> {
    /// The artifacts `backend` compiled for `fingerprint`, kept by kernel
    /// only. See [`compilation_store`] for what the two name.
    pub fn new(backend: &'static str, fingerprint: impl AsRef<str>) -> Self {
        Self {
            by_kernel: compilation_store(backend, fingerprint),
            by_source: None,
            build_id: build_id_hash(),
        }
    }

    /// The artifacts compiled for `fingerprint`, kept by kernel and by
    /// source, each map under its name in `names`.
    pub fn with_sources(names: StoreNames, fingerprint: impl AsRef<str>) -> Self {
        let by_kernel = compilation_store(names.kernels, fingerprint.as_ref());
        let by_source = by_kernel
            .is_some()
            .then(|| compilation_store(names.sources, fingerprint))
            .flatten();
        Self {
            by_kernel,
            by_source,
            build_id: build_id_hash(),
        }
    }

    /// Whether anything is kept between runs, which is what binds a backend's
    /// loaded kernels to the active environment.
    pub fn persists(&self) -> bool {
        self.by_kernel.is_some()
    }

    /// Whether the compilation store is enabled for this backend.
    pub fn is_enabled(&self) -> bool {
        self.by_kernel.is_some()
    }

    /// Takes the artifact kept for `kernel` out of the store, to be loaded.
    pub fn take(&mut self, kernel: &KernelId) -> Option<A> {
        let key = KernelCacheKey::new(kernel, self.build_id);
        let artifact = self.by_kernel.as_mut()?.remove(&key)?;
        log::trace!("Using the compilation store");
        Some(artifact)
    }

    /// Takes the artifact kept for `id` (kernel and variant) out of the store, to be loaded.
    pub fn take_variant<V: core::hash::Hash>(&mut self, id: &super::ArtifactId<V>) -> Option<A> {
        let key = KernelCacheKey::from_artifact_id(id, self.build_id);
        let artifact = self.by_kernel.as_mut()?.remove(&key)?;
        log::trace!("Using the compilation store for variant");
        Some(artifact)
    }

    /// Stores `artifact` under `id` (kernel and variant).
    pub fn store_variant<V: core::hash::Hash>(
        &mut self,
        id: &super::ArtifactId<V>,
        artifact: A,
    ) -> bool {
        let Some(by_kernel) = self.by_kernel.as_mut() else {
            return false;
        };
        let key = KernelCacheKey::from_artifact_id(id, self.build_id);
        store_compiled(by_kernel, key, artifact)
    }

    /// Purges a corrupted or invalid entry for `kernel`.
    pub fn purge(&mut self, kernel: &KernelId) {
        if let Some(by_kernel) = self.by_kernel.as_mut() {
            let key = KernelCacheKey::new(kernel, self.build_id);
            by_kernel.purge_key(&key);
        }
    }

    /// Purges a corrupted or invalid entry for `id` (kernel and variant).
    pub fn purge_variant<V: core::hash::Hash>(&mut self, id: &super::ArtifactId<V>) {
        if let Some(by_kernel) = self.by_kernel.as_mut() {
            let key = KernelCacheKey::from_artifact_id(id, self.build_id);
            by_kernel.purge_key(&key);
        }
    }

    /// The artifact another kernel finalized from `source`, for
    /// [`keep`](Self::keep) to put under the kernel asking.
    ///
    /// A kernel of this build keeps its own entry: it is as live as the one
    /// asking, and two kernel ids of one build that expand to one source —
    /// several of them in one batch — each keep theirs. One of an earlier
    /// build is moved, deleted under its key: nothing will ask for that key
    /// again.
    pub fn take_by_source(&mut self, source: &str) -> Option<A> {
        let by_kernel = self.by_kernel.as_mut()?;
        let by_source = self.by_source.as_mut()?;
        let hash = StableHasher::hash_one(&source);
        // `get_mut`, not `get`: a lazy store reads through to its storage
        // only there, and the compilation stores are lazy.
        let key = *by_source.get_mut(&hash)?;

        let artifact = if key.build_id == self.build_id {
            by_kernel.get_mut(&key)?.clone()
        } else {
            by_source.purge_key(&hash);
            by_kernel.purge_key(&key)?
        };
        log::trace!("Using the compilation store, by source");
        Some(artifact)
    }

    /// Keeps `artifact` under `kernel` and, when given, under the `source` it
    /// was finalized from. Whether the store took it, as [`store_compiled`]
    /// answers: `false` with nothing to keep it in.
    pub fn keep(&mut self, kernel: &KernelId, artifact: A, source: Option<&str>) -> bool {
        let Some(by_kernel) = self.by_kernel.as_mut() else {
            return false;
        };
        let key = KernelCacheKey::new(kernel, self.build_id);
        let stored = store_compiled(by_kernel, key, artifact);
        if let Some(source) = source
            && let Some(by_source) = self.by_source.as_mut()
        {
            store_compiled(by_source, StableHasher::hash_one(&source), key);
        }
        stored
    }
}

#[cfg(all(test, compilation_cache))]
mod tests {
    use super::*;
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;

    struct First;
    struct Second;
    struct Third;

    /// Stores as [`compilation_store`] opens them: lazy, over storage in an
    /// environment rooted at `root`, which only holds what it is written.
    fn store(root: &std::path::Path) -> ArtifactStore<u32> {
        fn lazy<K: StoreKey, V: StoreValue>(namespace: &str) -> Store<K, V> {
            Store::new(
                StoreOptions::new()
                    .storage(Namespace::new(namespace))
                    .cache(CacheOption::Lazy),
            )
        }

        cubecl_environment::environment::set_root(root);
        ArtifactStore {
            by_kernel: Some(lazy("kernels")),
            by_source: Some(lazy("sources")),
            build_id: build_id_hash(),
        }
    }

    /// Kernels of one build that expand to one source each keep the artifact:
    /// taking it for one leaves it to the others.
    #[test]
    #[serial_test::serial(records)]
    fn kernels_sharing_a_source_each_keep_it() {
        let root = tempfile::tempdir().unwrap();
        let mut store = store(root.path());
        let kernels = [
            KernelId::new::<First>(),
            KernelId::new::<Second>(),
            KernelId::new::<Third>(),
        ];
        store.keep(&kernels[0], 7, Some("shared"));
        for kernel in &kernels[1..] {
            let artifact = store.take_by_source("shared").expect("kept by source");
            store.keep(kernel, artifact, Some("shared"));
        }
        for kernel in &kernels {
            assert_eq!(store.take(kernel), Some(7));
        }
    }

    /// A kernel of an earlier build gives its artifact up: nothing asks for
    /// its key again.
    #[test]
    #[serial_test::serial(records)]
    fn an_earlier_build_is_moved() {
        let root = tempfile::tempdir().unwrap();
        let mut store = store(root.path());
        let current = store.build_id;
        let kernel = KernelId::new::<First>();
        store.build_id = StableHasher::hash_one(&"an earlier build");
        store.keep(&kernel, 7, Some("shared"));

        store.build_id = current;
        assert_eq!(store.take_by_source("shared"), Some(7));
        store.build_id = StableHasher::hash_one(&"an earlier build");
        assert_eq!(store.take(&kernel), None, "moved out");
    }
}

/// Whether a key's variant is the one a kernel id says everything for, which
/// is not written.
fn is_zero(variant: &StableHash) -> bool {
    *variant == 0
}
