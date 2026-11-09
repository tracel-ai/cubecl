//! The kernels the work launched, recorded per build.

use super::{build_id_hash, compilation_store};
use crate::id::KernelId;
use alloc::string::String;
use cubecl_common::hash::StableHash;
use cubecl_environment::collections::HashSet;
use cubecl_environment::persistence::Store;

/// The kernels the work launched in the active environment, by the stable
/// hash of their id, kept per build: what an environment's prune keeps of this
/// build's compiled artifacts, the rest being what only an autotune race
/// launched.
///
/// A kernel is recorded the first time a launch issued for the
/// [work](crate::execution::LaunchPurpose::Work) reaches the loader, to load
/// it or to queue it, whatever shape or variant it was launched for, and never
/// for a race's launch. A process writes a kernel at most once, and not at
/// all when the environment already holds it, so a warm start only reads.
/// Kept only beside a store of compiled artifacts, under the same switch and
/// the same configuration: with nothing compiled kept, there is nothing to
/// record either.
#[derive(Debug)]
pub struct WorkLaunchStore {
    store: Option<Store<StableHash, ()>>,
    /// The kernels this process recorded or found recorded, in the
    /// environment of `generation`.
    known: HashSet<StableHash>,
    generation: u32,
}

impl WorkLaunchStore {
    /// The root of the namespaces the kernels are recorded under, one per
    /// build.
    pub const ROOT: &'static str = "launched";

    /// The path under [`ROOT`](Self::ROOT) of the records of `build`.
    pub fn build_path(build: StableHash) -> String {
        alloc::format!("{build:032x}")
    }

    /// The records of this build in the active environment when `bound` — the
    /// loader's target keeps what it compiles — and nothing to record into
    /// otherwise, or when compilation caching is disabled.
    pub fn new(bound: bool) -> Self {
        let store = match bound {
            true => compilation_store(Self::ROOT, Self::build_path(build_id_hash())),
            false => None,
        };
        Self {
            store,
            known: HashSet::new(),
            generation: cubecl_environment::environment::generation(),
        }
    }

    /// Record `kernel` when the launch reaching the loader on this thread is
    /// the work's. Answers whether the kernel needs nothing more from here: it
    /// is recorded, or there is nowhere to record it, or the storage refused
    /// the write, which is not tried again: a lost record costs a compile
    /// after a prune, never a result.
    pub fn record(&mut self, kernel: &KernelId) -> bool {
        if self.store.is_none() {
            return true;
        }
        if crate::execution::LaunchPurpose::running() == crate::execution::LaunchPurpose::Race {
            return false;
        }
        self.reset_if_switched();
        let hash = kernel.stable_hash();
        if !self.known.insert(hash) {
            return true;
        }
        let store = self.store.as_mut().expect("checked above");
        // `get_mut`, not `get`: a lazy store reads through to its storage only
        // there, and a kernel another process recorded is read, not written.
        if store.get_mut(&hash).is_none()
            && let Err(error) = store.insert(hash, ())
        {
            log::debug!(
                "the work's launch of a kernel was not recorded: {}",
                error.reason()
            );
        }
        true
    }

    /// Forget what was known of the environment the records were written
    /// into when it switched, as the store it fronts does.
    fn reset_if_switched(&mut self) {
        let current = cubecl_environment::environment::generation();
        if current != self.generation {
            self.generation = current;
            self.known.clear();
        }
    }
}

#[cfg(all(test, compilation_cache))]
mod tests {
    use super::*;
    use cubecl_environment::persistence::{CacheOption, Namespace, StoreOptions};
    // `serial_test`'s macro expands to `vec!`, which a `no_std` crate has to
    // bring in itself.
    use alloc::vec;

    struct First;
    struct Second;

    /// A work store over storage in an environment rooted at `root`.
    fn work_store(root: &std::path::Path) -> WorkLaunchStore {
        cubecl_environment::environment::set_root(root);
        WorkLaunchStore {
            store: Some(Store::new(
                StoreOptions::new()
                    .storage(Namespace::new("launched"))
                    .cache(CacheOption::Lazy),
            )),
            known: HashSet::new(),
            generation: cubecl_environment::environment::generation(),
        }
    }

    /// A race's launch is not recorded and asks again; the work's is, once,
    /// and its kernel needs nothing more.
    #[test]
    #[serial_test::serial(records)]
    fn only_the_works_launch_is_recorded() {
        use crate::execution::LaunchPurpose;

        let root = tempfile::tempdir().unwrap();
        let mut work = work_store(root.path());
        let kernel = KernelId::new::<First>();

        let raced = LaunchPurpose::Race.apply(|| work.record(&kernel));
        assert!(!raced, "a race's launch leaves the kernel to the work");
        assert!(work.known.is_empty());

        assert!(LaunchPurpose::Work.apply(|| work.record(&kernel)));
        assert!(work.known.contains(&kernel.stable_hash()));
        let stored = work.store.as_mut().unwrap();
        assert_eq!(stored.get_mut(&kernel.stable_hash()), Some(&mut ()));
    }

    /// A process that finds a kernel recorded already reads it and writes
    /// nothing, and one with nowhere to record has nothing to do.
    #[test]
    #[serial_test::serial(records)]
    fn a_kernel_recorded_already_is_not_written_again() {
        use crate::execution::LaunchPurpose;

        let root = tempfile::tempdir().unwrap();
        let kernel = KernelId::new::<Second>();
        let mut first = work_store(root.path());
        assert!(LaunchPurpose::Work.apply(|| first.record(&kernel)));
        let mut second = work_store(root.path());
        assert!(LaunchPurpose::Work.apply(|| second.record(&kernel)));
        assert!(second.known.contains(&kernel.stable_hash()));

        let mut nowhere = WorkLaunchStore::new(false);
        assert!(nowhere.record(&kernel));
        assert!(nowhere.known.is_empty());
    }
}
