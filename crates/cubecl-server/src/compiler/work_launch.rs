//! The kernels the work launched, recorded per build.

use super::persistent_store;
use crate::id::KernelId;
use cubecl_common::hash::StableHash;
use cubecl_environment::environment::BuildStamp;
use cubecl_environment::persistence::{Namespace, Store};

/// The kernels the work launched in the active environment, by the stable
/// hash of their id, kept per build: what an environment's prune keeps of this
/// build's compiled artifacts, the rest being what only an autotune race
/// launched.
///
/// The [loader](super::KernelLoader) records a kernel the first time a launch
/// issued for the [work](crate::execution::LaunchPurpose::Work) reaches it, to
/// load the kernel or to queue it, whatever shape or variant it was launched
/// for, and never for a race's launch. A kernel the environment holds already
/// is read, not written, so a warm start only reads. Kept only beside a store
/// of compiled artifacts, under the same switch: with nothing compiled kept,
/// there is nothing to record either.
#[derive(Debug)]
pub struct WorkLaunchStore {
    /// Lazy, as the compilation stores are: it memoizes each kernel it finds
    /// recorded, and resets itself when the environment switches.
    store: Store<StableHash, ()>,
}

impl WorkLaunchStore {
    /// The root of the namespaces the kernels are recorded under, one per
    /// build.
    pub const ROOT: &'static str = "launched";

    /// The namespace the kernels `build`'s work launched are recorded under.
    pub fn namespace(build: BuildStamp) -> Namespace {
        Namespace::scoped(Self::ROOT, alloc::format!("{build}"))
    }

    /// The records of this build in the active environment, or `None` when
    /// compilation caching is disabled or the target has nowhere durable to
    /// put them.
    pub(crate) fn new() -> Option<Self> {
        persistent_store(Self::namespace(BuildStamp::current())).map(|store| Self { store })
    }

    /// Record that the work launched `kernel`. A storage that refuses the
    /// write is logged, not reported: a lost record costs a compile after a
    /// prune, never a result, and the next launch that reaches here tries
    /// again.
    pub(crate) fn record(&mut self, kernel: &KernelId) {
        let hash = kernel.stable_hash();
        // `get_mut`, not `get`: a lazy store reads through to its storage
        // only there, and memoizes what it finds, so a kernel recorded
        // already, by this process or another, is read once and never
        // written.
        if self.store.get_mut(&hash).is_some() {
            return;
        }
        if let Err(error) = self.store.insert(hash, ()) {
            log::debug!(
                "the work's launch of a kernel was not recorded: {}",
                error.reason()
            );
        }
    }
}

#[cfg(test)]
impl WorkLaunchStore {
    /// The records kept in `storage`, which a test inspects.
    pub(crate) fn over(
        storage: alloc::boxed::Box<dyn cubecl_environment::persistence::Storage>,
    ) -> Self {
        use cubecl_environment::persistence::{CacheOption, StoreOptions};

        Self {
            store: Store::new(
                StoreOptions::new()
                    .storage_with(storage, Namespace::new(Self::ROOT))
                    .cache(CacheOption::Lazy),
            ),
        }
    }

    /// Whether `kernel` is recorded.
    pub(crate) fn holds(&mut self, kernel: &KernelId) -> bool {
        self.store.get_mut(&kernel.stable_hash()).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::string::String;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use cubecl_environment::bytes::Bytes;
    use cubecl_environment::persistence::{Insertion, MemoryStorage, Origin, Storage};
    use cubecl_environment::sync::Arc;

    struct First;
    struct Second;

    /// The work's launch of a kernel is written once, however often it is
    /// recorded, and a process that finds it recorded already writes
    /// nothing: a warm start only reads.
    #[test]
    fn a_kernel_recorded_already_is_not_written_again() {
        let storage = Counting::new("work-launch-written-once");
        let kernel = KernelId::new::<First>();

        let mut first = WorkLaunchStore::over(Box::new(storage.clone()));
        first.record(&kernel);
        first.record(&kernel);
        first.record(&kernel);
        assert_eq!(storage.writes(), 1, "one write for the first record");

        let mut second = WorkLaunchStore::over(Box::new(storage.clone()));
        second.record(&kernel);
        assert_eq!(storage.writes(), 1, "found recorded, not written");
        assert!(second.holds(&kernel));
    }

    /// Kernels are recorded apart: one recorded says nothing of another.
    #[test]
    fn each_kernel_is_recorded_on_its_own() {
        let storage = Counting::new("work-launch-apart");
        let mut work = WorkLaunchStore::over(Box::new(storage.clone()));
        work.record(&KernelId::new::<First>());

        assert!(work.holds(&KernelId::new::<First>()));
        assert!(!work.holds(&KernelId::new::<Second>()));
        assert_eq!(storage.writes(), 1);
    }

    /// A storage over shared memory, counting the writes it is asked for.
    #[derive(Debug, Clone)]
    struct Counting {
        memory: MemoryStorage,
        writes: Arc<AtomicUsize>,
    }

    impl Counting {
        fn new(namespace: &str) -> Self {
            let memory = MemoryStorage::new(namespace);
            memory.purge();
            Self {
                memory,
                writes: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn writes(&self) -> usize {
            self.writes.load(Ordering::Relaxed)
        }
    }

    impl Storage for Counting {
        fn get(&self, key: &[u8]) -> Option<Bytes> {
            self.memory.get(key)
        }

        fn insert(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
            self.writes.fetch_add(1, Ordering::Relaxed);
            self.memory.insert(key, value, origin)
        }

        fn replace(&self, key: &[u8], value: Bytes, origin: Origin) -> Insertion {
            self.writes.fetch_add(1, Ordering::Relaxed);
            self.memory.replace(key, value, origin)
        }

        fn scan(&self, visit: &mut dyn FnMut(&[u8], &[u8])) {
            self.memory.scan(visit)
        }

        fn purge(&self) {
            self.memory.purge()
        }

        fn purge_key(&self, key: &[u8]) {
            self.memory.purge_key(key)
        }

        fn describe(&self) -> String {
            self.memory.describe()
        }
    }
}
