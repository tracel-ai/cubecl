//! Where a backend keeps the artifacts it compiled, between runs.

use crate::id::KernelId;
use cubecl_common::hash::{StableHash, StableHasher};
use cubecl_environment::persistence::{Store, StoreValue};

use super::{KernelCacheKey, build_id_hash, compilation_store, store_compiled};

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

    /// The artifacts `backend` compiled for `fingerprint`, kept by kernel and
    /// by source; the second map lives under `source_backend`.
    pub fn with_sources(
        backend: &'static str,
        source_backend: &'static str,
        fingerprint: impl AsRef<str>,
    ) -> Self {
        let by_kernel = compilation_store(backend, fingerprint.as_ref());
        let by_source = by_kernel
            .is_some()
            .then(|| compilation_store(source_backend, fingerprint))
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

    /// Takes the artifact another kernel finalized from `source` out of the
    /// store, deleting it under that kernel: [`keep`](Self::keep) puts it back
    /// under the kernel asking.
    pub fn take_by_source(&mut self, source: &str) -> Option<A> {
        let by_kernel = self.by_kernel.as_mut()?;
        let key = self
            .by_source
            .as_mut()?
            .purge_key(&StableHasher::hash_one(&source))?;
        let artifact = by_kernel.purge_key(&key)?;
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
