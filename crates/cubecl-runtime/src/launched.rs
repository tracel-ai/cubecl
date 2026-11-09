//! The kernels a workload launches, collected while it replays: what an
//! environment shipped for that workload has to keep, and all it has to.
//!
//! A build compiles far more than it ships — every candidate an autotune race
//! measured, of which only the winner is ever launched again. Replaying the
//! workload the environment is shipped for, under a [`LaunchedKernels`], names
//! exactly the kernels it runs; everything else in the compilation store can
//! go. The replay may run under a process mode that discards launches: a discarded launch
//! is still issued, and so still collected.
//!
//! Collected on the issuing thread, where every backend's launch passes, by
//! the [stable hash](crate::id::KernelId::stable_hash) that also names a
//! kernel's artifact in the compilation store. Outside a collection a launch
//! pays two relaxed atomic loads.
//!
//! A tune collects too, without a replay: it lists what each of
//! its candidates launched, which the stored answer keeps beside the pick. A
//! kernel only the losers launched is one no workload loads again, which is
//! how an environment is pruned.

use crate::id::KernelId;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use cubecl_common::hash::StableHash;
use cubecl_environment::collections::{HashMap, HashSet};
use cubecl_environment::stream::StreamId;
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

/// Note a launch in every open collection, and in the tune open on this thread
/// under the candidate running. The id is asked for only then: outside both a
/// launch computes nothing here.
pub(crate) fn note(kernel: impl FnOnce() -> KernelId) {
    let collecting = COLLECTING.load(Ordering::Relaxed) > 0;
    let tuning = TUNING.load(Ordering::Relaxed) > 0;
    if !collecting && !tuning {
        return;
    }
    // Hashed before the lock, which every launching thread shares.
    let hash = kernel().stable_hash();
    if tuning {
        note_in_tune(hash);
    }
    if collecting {
        let mut collections = OPEN.lock();
        for launched in collections.open.values_mut() {
            launched.insert(hash);
        }
    }
}

/// How many tunes are collecting their candidates' launches, on any stream:
/// what a launch reads before it takes the lock of [`TUNES`].
static TUNING: AtomicUsize = AtomicUsize::new(0);

/// The tunes open, by the stream that opened them, innermost last.
static TUNES: LazyLock<Mutex<HashMap<StreamId, Vec<Tune>>>> =
    LazyLock::new(|| Mutex::new(HashMap::default()));

/// One open tune: the candidate it is running, and what each one launched.
#[derive(Default)]
struct Tune {
    running: Option<String>,
    launched: HashMap<String, HashSet<StableHash>>,
}

/// The kernels each candidate of a tune launched, by the candidate's name.
pub type CandidateKernelMap = HashMap<String, Vec<StableHash>>;

/// Collects the kernels each candidate of one tune launches, from
/// [`open`](Self::open) to [`finish`](Self::finish): what tells a candidate's
/// kernels from its rivals' once the tune has picked one.
///
/// Kept by the [stream](StreamId::current) that opened it, not the thread: a
/// device whose work runs on a thread of its own runs a candidate there, under
/// the caller's stream. A launch counts toward the candidate [running](running)
/// in the tune its stream opened last; a tune reached inside a candidate
/// collects its own candidates' launches, not the outer one's, and a launch
/// outside any running candidate counts toward none.
#[must_use = "the launches are collected until `finish`"]
pub(crate) struct TuneLaunches {
    stream: StreamId,
}

impl TuneLaunches {
    /// Start collecting on the current stream.
    pub(crate) fn open() -> Self {
        let stream = StreamId::current();
        TUNES
            .lock()
            .entry(stream)
            .or_default()
            .push(Tune::default());
        TUNING.fetch_add(1, Ordering::AcqRel);
        Self { stream }
    }

    /// Stop collecting, and hand back the kernels each candidate launched,
    /// each list sorted.
    pub(crate) fn finish(self) -> CandidateKernelMap {
        let tune = self.close();
        core::mem::forget(self);
        tune.launched
            .into_iter()
            .map(|(candidate, kernels)| {
                let mut kernels: Vec<StableHash> = kernels.into_iter().collect();
                kernels.sort_unstable();
                (candidate, kernels)
            })
            .collect()
    }

    fn close(&self) -> Tune {
        let mut tunes = TUNES.lock();
        let tune = match tunes.get_mut(&self.stream) {
            Some(open) => {
                let tune = open.pop().unwrap_or_default();
                if open.is_empty() {
                    tunes.remove(&self.stream);
                }
                tune
            }
            None => Tune::default(),
        };
        TUNING.fetch_sub(1, Ordering::AcqRel);
        tune
    }
}

impl Drop for TuneLaunches {
    fn drop(&mut self) {
        self.close();
    }
}

/// Run `candidate`'s work: the launches it issues on the current stream count
/// toward it in the tune open there. The candidate running before is restored
/// after, so a candidate run inside another (an answered key's pick) collects
/// under its own name, and the outer one's list never carries it.
pub(crate) fn running<R>(candidate: &str, work: impl FnOnce() -> R) -> R {
    if TUNING.load(Ordering::Relaxed) == 0 {
        return work();
    }
    let restore = Running::enter(candidate);
    let output = work();
    core::mem::drop(restore);
    output
}

/// Note `hash` toward the candidate running in the tune the current stream
/// opened last, if there is one.
fn note_in_tune(hash: StableHash) {
    let mut tunes = TUNES.lock();
    let Some(tune) = tunes
        .get_mut(&StreamId::current())
        .and_then(|open| open.last_mut())
    else {
        return;
    };
    if let Some(candidate) = &tune.running {
        let candidate = candidate.clone();
        tune.launched.entry(candidate).or_default().insert(hash);
    }
}

/// Restores, when dropped, the candidate that was running in the innermost
/// tune of its stream before [`enter`](Self::enter).
struct Running {
    stream: StreamId,
    /// `None` when no tune was open on the stream: there is nothing to restore.
    previous: Option<Option<String>>,
}

impl Running {
    fn enter(candidate: &str) -> Self {
        let stream = StreamId::current();
        let previous = TUNES
            .lock()
            .get_mut(&stream)
            .and_then(|open| open.last_mut())
            .map(|tune| tune.running.replace(candidate.to_string()));
        Self { stream, previous }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        let Some(previous) = self.previous.take() else {
            return;
        };
        if let Some(tune) = TUNES
            .lock()
            .get_mut(&self.stream)
            .and_then(|open| open.last_mut())
        {
            tune.running = previous;
        }
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

    mod tunes {
        use super::*;
        use alloc::vec;

        struct Candidate;

        fn kernel(info: u32) -> KernelId {
            KernelId::new::<Candidate>().info(info)
        }

        #[test]
        fn a_tune_lists_what_each_candidate_launched() {
            let tune = TuneLaunches::open();
            running("fast", || note(|| kernel(0)));
            running("slow", || {
                note(|| kernel(1));
                note(|| kernel(2));
            });
            let launched = tune.finish();

            assert_eq!(launched["fast"], vec![kernel(0).stable_hash()]);
            let mut slow = vec![kernel(1).stable_hash(), kernel(2).stable_hash()];
            slow.sort_unstable();
            assert_eq!(launched["slow"], slow);
        }

        #[test]
        fn a_launch_outside_a_running_candidate_counts_toward_none() {
            let tune = TuneLaunches::open();
            note(|| kernel(10));
            let launched = tune.finish();

            assert!(launched.is_empty());
        }

        /// A key reached inside a candidate tunes its own candidates: their
        /// launches are the inner tune's, never the outer candidate's.
        #[test]
        fn a_tune_inside_a_candidate_keeps_its_candidates_launches() {
            let outer = TuneLaunches::open();
            let inner = running("outer", || {
                note(|| kernel(20));
                let inner = TuneLaunches::open();
                running("inner", || note(|| kernel(21)));
                inner.finish()
            });
            let outer = outer.finish();

            assert_eq!(outer["outer"], vec![kernel(20).stable_hash()]);
            assert_eq!(inner["inner"], vec![kernel(21).stable_hash()]);
        }

        /// An answered key's pick runs inside a candidate under its own name, and
        /// the candidate keeps collecting once it returns.
        #[test]
        fn a_candidate_run_inside_another_collects_under_its_own_name() {
            let tune = TuneLaunches::open();
            running("outer", || {
                running("picked", || note(|| kernel(30)));
                note(|| kernel(31));
            });
            let launched = tune.finish();

            assert_eq!(launched["picked"], vec![kernel(30).stable_hash()]);
            assert_eq!(launched["outer"], vec![kernel(31).stable_hash()]);
        }

        #[test]
        fn running_outside_a_tune_only_runs() {
            assert_eq!(running("alone", || 7), 7);
        }
    }
}
