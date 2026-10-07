//! Where a backend keeps the artifacts it compiled, between runs.

use crate::id::KernelId;
use cubecl_common::hash::{StableHash, StableHasher};
#[cfg(compilation_cache)]
use cubecl_environment::persistence::{CacheOption, Namespace, StoreOptions};
use cubecl_environment::persistence::{Store, StoreKey, StoreValue};

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

    /// Takes the artifact kept for `kernel` out of the store, to be loaded.
    pub fn take(&mut self, kernel: &KernelId) -> Option<A> {
        let key = KernelCacheKey::new(kernel, self.build_id);
        let artifact = self.by_kernel.as_mut()?.remove(&key)?;
        log::trace!("Using the compilation store");
        Some(artifact)
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
        let key = *by_source.get(&hash)?;

        let artifact = if key.build_id == self.build_id {
            by_kernel.get(&key)?.clone()
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

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_environment::persistence::StoreOptions;

    struct First;
    struct Second;
    struct Third;

    fn store() -> ArtifactStore<u32> {
        ArtifactStore {
            by_kernel: Some(Store::new(StoreOptions::new())),
            by_source: Some(Store::new(StoreOptions::new())),
            build_id: build_id_hash(),
        }
    }

    /// Kernels of one build that expand to one source each keep the artifact:
    /// taking it for one leaves it to the others.
    #[test]
    fn kernels_sharing_a_source_each_keep_it() {
        let mut store = store();
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
    fn an_earlier_build_is_moved() {
        let mut store = store();
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
