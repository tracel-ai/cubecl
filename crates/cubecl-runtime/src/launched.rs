//! The kernels a workload launches, collected while it replays: what an
//! environment shipped for that workload has to keep, and all it has to.
//!
//! A build compiles far more than it ships — every candidate an autotune race
//! measured, of which only the winner is ever launched again. Replaying the
//! workload the environment is shipped for, under a [`LaunchedKernels`], names
//! exactly the kernels it runs; everything else in the compilation store can
//! go. The replay may be a dry run: a dropped launch is still issued, and so
//! still collected.
//!
//! Collected on the issuing thread, where every backend's launch passes, by
//! the [stable hash](crate::id::KernelId::stable_hash) that also names a
//! kernel's artifact in the compilation store. Outside a collection a launch
//! pays one relaxed atomic load.

use crate::id::KernelId;
use cubecl_common::hash::StableHash;
use cubecl_environment::collections::HashSet;
use cubecl_environment::sync::{AtomicUsize, LazyLock, Mutex, Ordering};

/// How many collections are open, so overlapping ones compose.
static COLLECTING: AtomicUsize = AtomicUsize::new(0);

/// Every kernel launched while a collection was open.
static LAUNCHED: LazyLock<Mutex<HashSet<StableHash>>> =
    LazyLock::new(|| Mutex::new(HashSet::default()));

/// Collects every kernel launched, on every thread and device, from
/// [`collect`](Self::collect) to [`finish`](Self::finish).
#[must_use = "the kernels are collected until `finish`"]
#[derive(Debug)]
pub struct LaunchedKernels {
    finished: bool,
}

impl LaunchedKernels {
    /// Start collecting.
    pub fn collect() -> Self {
        if COLLECTING.fetch_add(1, Ordering::AcqRel) == 0 {
            LAUNCHED.lock().clear();
        }
        Self { finished: false }
    }

    /// Stop collecting, and hand back the stable hash of every kernel
    /// launched since [`collect`](Self::collect).
    pub fn finish(mut self) -> HashSet<StableHash> {
        self.finished = true;
        let launched = LAUNCHED.lock().clone();
        COLLECTING.fetch_sub(1, Ordering::AcqRel);
        launched
    }
}

impl Drop for LaunchedKernels {
    fn drop(&mut self) {
        if !self.finished {
            COLLECTING.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Note a launch, when a collection is open. The id is asked for only then:
/// outside a collection a launch computes nothing here.
pub(crate) fn note(kernel: impl FnOnce() -> KernelId) {
    if COLLECTING.load(Ordering::Relaxed) > 0 {
        LAUNCHED.lock().insert(kernel().stable_hash());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Launched;

    #[test]
    fn a_collection_names_what_was_launched_while_it_was_open() {
        let before = KernelId::new::<Launched>().info(0u32);
        let during = KernelId::new::<Launched>().info(1u32);
        note(|| before.clone());

        let collection = LaunchedKernels::collect();
        note(|| during.clone());
        note(|| during.clone());
        let launched = collection.finish();

        assert_eq!(launched.len(), 1);
        assert!(launched.contains(&during.stable_hash()));
        assert!(!launched.contains(&before.stable_hash()));
    }
}
