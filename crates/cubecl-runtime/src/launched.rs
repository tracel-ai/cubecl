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
use cubecl_environment::collections::{HashMap, HashSet};
use cubecl_environment::sync::{AtomicUsize, LazyLock, Mutex, Ordering};

/// How many collections are open: what a launch reads before it takes the
/// lock, so one outside every collection pays a relaxed load and nothing more.
static COLLECTING: AtomicUsize = AtomicUsize::new(0);

/// The collections open, each with what was launched since it opened.
static OPEN: LazyLock<Mutex<Collections>> = LazyLock::new(|| Mutex::new(Collections::default()));

#[derive(Default)]
struct Collections {
    next: u64,
    open: HashMap<u64, HashSet<StableHash>>,
}

/// Collects every kernel launched, on every thread and device, from
/// [`new`](Self::new) to [`finish`](Self::finish). Collections may overlap:
/// each sees what was launched while it was open, and nothing before.
#[must_use = "the kernels are collected until `finish`"]
#[derive(Debug)]
pub struct LaunchedKernels {
    id: u64,
}

impl LaunchedKernels {
    /// Start collecting.
    #[allow(
        clippy::new_without_default,
        reason = "starting a collection is not a value"
    )]
    pub fn new() -> Self {
        let mut collections = OPEN.lock();
        let id = collections.next;
        collections.next += 1;
        collections.open.insert(id, HashSet::default());
        // Raised under the lock: a launch that sees it finds the set there.
        COLLECTING.fetch_add(1, Ordering::AcqRel);
        Self { id }
    }

    /// Stop collecting, and hand back the stable hash of every kernel
    /// launched since [`new`](Self::new).
    pub fn finish(self) -> HashSet<StableHash> {
        let launched = close(self.id);
        core::mem::forget(self);
        launched
    }
}

impl Drop for LaunchedKernels {
    fn drop(&mut self) {
        close(self.id);
    }
}

/// Close collection `id`, handing back what it collected.
fn close(id: u64) -> HashSet<StableHash> {
    let mut collections = OPEN.lock();
    let launched = collections.open.remove(&id).unwrap_or_default();
    COLLECTING.fetch_sub(1, Ordering::AcqRel);
    launched
}

/// Note a launch in every open collection. The id is asked for only then:
/// outside a collection a launch computes nothing here.
pub(crate) fn note(kernel: impl FnOnce() -> KernelId) {
    if COLLECTING.load(Ordering::Relaxed) == 0 {
        return;
    }
    // Hashed before the lock, which every launching thread shares.
    let hash = kernel().stable_hash();
    let mut collections = OPEN.lock();
    for launched in collections.open.values_mut() {
        launched.insert(hash);
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

        let collection = LaunchedKernels::new();
        note(|| during.clone());
        note(|| during.clone());
        let launched = collection.finish();

        // Other tests launch in parallel: only this test's kernels are known.
        assert!(launched.contains(&during.stable_hash()));
        assert!(!launched.contains(&before.stable_hash()));
    }

    /// An inner collection sees only what was launched while it was open,
    /// and closing it leaves the outer one collecting.
    #[test]
    fn overlapping_collections_each_see_their_own_launches() {
        let early = KernelId::new::<Launched>().info(10u32);
        let late = KernelId::new::<Launched>().info(11u32);

        let outer = LaunchedKernels::new();
        note(|| early.clone());
        let inner = LaunchedKernels::new();
        note(|| late.clone());
        let inner = inner.finish();
        let outer = outer.finish();

        assert!(inner.contains(&late.stable_hash()));
        assert!(!inner.contains(&early.stable_hash()));
        assert!(outer.contains(&early.stable_hash()) && outer.contains(&late.stable_hash()));
    }
}
