//! What a capture learns from planning its work before recording it.
//!
//! A graph on a stream replays its temporaries over the room the stream's other graphs
//! replay theirs on: that is what lets a stream hold about the largest graph's working set
//! rather than the sum of all of them (see
//! [`PinnedPages`](crate::memory_management::PinnedPages)). A buffer the graph's work leaves
//! behind — an output the caller reads — must never sit on that room, or another graph's
//! replay overwrites it. Whether an allocation is a temporary or an output is only known once
//! the work is over, by which time a recording has baked its address.
//!
//! So the work runs first as a planning run, its launches discarded: each run lists its
//! allocations in order and, once it is over, which of them something still holds. From the
//! second run on, each allocation is placed by what the run before found where it matches: an
//! output off the room other graphs pinned, a temporary on it. The planning run repeats until two
//! runs in a row allocate alike, since a first run can take another path than the steady state
//! (a tune, a fused optimization being built). The recorded run then allocates under the same
//! plan, so it lands where the last planning did, in room that planning grew.
//!
//! A run need not allocate its temporaries exactly as the run before did. A framework writes in
//! place into a buffer nothing else holds and into a fresh one otherwise, and what else holds a
//! buffer when it decides can depend on how far another queue has drained: a run then allocates
//! a temporary more or one fewer than its plan. So an allocation matches the first one of its
//! size among the next few of the plan, skipping the planned ones the
//! run did not make, and one that matches none is placed as a temporary. What a temporary sits
//! on is the run's own business. The recording refuses to seal only when it keeps a buffer it
//! placed as a temporary, since that buffer may sit on room another graph's replay writes.

use crate::memory_management::{ManagedMemoryBinding, WeakMemoryBinding};
use alloc::vec::Vec;

/// How many allocations of the plan, from the one after the last matched, an allocation is
/// matched against by its size: enough to step over the temporaries a run allocates otherwise
/// than its plan, few enough that a common size does not match far down the plan, past the
/// allocations the run is still to make.
const MATCH_WINDOW_ALLOCATIONS: usize = 16;

/// One allocation of a finished planning run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedAllocation {
    /// The size the allocation asked for, in bytes.
    pub size_bytes: u64,
    /// Whether something still held it once the run was over: an output.
    pub outlived: bool,
}

/// Where a run's next allocation goes, by what the run before found where it matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// An output: off the room the stream's other graphs pinned, which their replays write.
    Output,
    /// A temporary, or work outside a capture: the pools decide.
    Pools,
}

/// One allocation of the current run, as it was made.
#[derive(Debug)]
struct RunAllocation {
    size_bytes: u64,
    /// Unheld, so the record never keeps the allocation alive.
    memory: WeakMemoryBinding,
    placement: Placement,
}

/// The planning run of one capture and the plan the recording follows.
///
/// Built and cleared with the capture's lifecycle by
/// [`StreamCapture`](super::StreamCapture), which is its only owner.
#[derive(Debug, Default)]
pub struct AllocationPlanner {
    /// The current run's allocations, in order. Arrival usage: pushed per allocation, drained
    /// in order when the run ends.
    run: Vec<RunAllocation>,
    /// The last finished run, in allocation order: the plan the next run follows. Position
    /// usage: a run matches its allocations against the entries from its cursor on.
    last: Vec<PlannedAllocation>,
    /// The run that just ended, before it becomes `last`. Scratch usage: filled as a run ends,
    /// compared with `last`, then swapped with it.
    ended: Vec<PlannedAllocation>,
    /// What the planning runs left alive when they ended, unheld: the outputs the caller let go
    /// of since, and whatever the work kept, a cache's buffer say, which no later run allocates
    /// again. Arrival usage: pushed as each planning run ends, drained as the window opens.
    kept: Vec<WeakMemoryBinding>,
    /// Whether a run of the work is under way: a planning run, from the capture's prepare or
    /// [`start_run`](Self::start_run) to [`finish_run`](Self::finish_run), or the recorded run.
    /// What the capture's owner allocates otherwise is not the work's.
    running: bool,
    /// Whether any run finished since the capture was prepared.
    planned: bool,
    /// The position in `last` after the current run's last matched allocation.
    cursor: usize,
}

/// A buffer the recorded run keeps on room other graphs' replays write: it placed the buffer as
/// a temporary, since its plan has no output of that size where the buffer was allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Divergence {
    /// The position of the allocation in the recorded run.
    pub position: usize,
    /// The size the allocation asked for, in bytes.
    pub size_bytes: u64,
}

impl AllocationPlanner {
    /// Note an allocation of `size_bytes` bound to `memory` by the current run, placed as
    /// `placement` says.
    pub fn allocated(
        &mut self,
        size_bytes: u64,
        memory: &ManagedMemoryBinding,
        placement: Placement,
    ) {
        self.run.push(RunAllocation {
            size_bytes,
            memory: memory.downgrade(),
            placement,
        });
    }

    /// End a planning run, once its deferred frees are reclaimed, and answer whether it
    /// allocated what the run before it did: the work has reached its steady state, and the
    /// recording can follow it.
    pub fn finish_run(&mut self) -> bool {
        self.kept.extend(
            self.run
                .iter()
                .filter(|allocation| allocation.memory.upgrade().is_some())
                .map(|allocation| allocation.memory.clone()),
        );
        self.end_run_log();
        self.running = false;
        let stable = self.planned && self.ended == self.last;
        core::mem::swap(&mut self.last, &mut self.ended);
        self.planned = true;
        stable
    }

    /// Whether any planning run finished: a recording has a plan to follow.
    pub fn has_plan(&self) -> bool {
        self.planned
    }

    /// Start a planning run: what the work allocates from here is logged, and placed by the
    /// plan, until [`finish_run`](Self::finish_run).
    pub fn start_run(&mut self) {
        self.running = true;
    }

    /// Whether a run of the work is under way, see [`start_run`](Self::start_run).
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// What the planning runs left alive that still is, taken as the window opens.
    pub fn drain_kept(&mut self) -> impl Iterator<Item = ManagedMemoryBinding> + '_ {
        self.kept.drain(..).filter_map(|memory| memory.upgrade())
    }

    /// Start the recorded run at the plan's first allocation, its log empty: what was allocated
    /// since the last planning run ended is not the work's.
    pub fn start_recording(&mut self) {
        self.running = true;
        self.run.clear();
        self.cursor = 0;
    }

    /// Place the current run's next allocation, of `size_bytes`: off the shared room when the
    /// plan's first allocation of that size a few from its cursor on
    /// was an output, which moves the cursor past it. One that matches none goes to the pools,
    /// and leaves the cursor where it was.
    pub fn place_bytes(&mut self, size_bytes: u64) -> Placement {
        let matched = self
            .last
            .iter()
            .enumerate()
            .skip(self.cursor)
            .take(MATCH_WINDOW_ALLOCATIONS)
            .find(|(_, planned)| planned.size_bytes == size_bytes);
        let Some((position, planned)) = matched else {
            return Placement::Pools;
        };
        self.cursor = position + 1;
        match planned.outlived {
            true => Placement::Output,
            false => Placement::Pools,
        }
    }

    /// End the recorded run, once its deferred frees are reclaimed: the first buffer it keeps
    /// that it placed as a temporary, which may sit on room another graph's replay writes. A
    /// temporary allocated otherwise than planned is no divergence, nor is an output it let go:
    /// its graph holds the room, which every replay still writes.
    pub fn finish_recording(&mut self) -> Option<Divergence> {
        self.running = false;
        let divergence = self
            .run
            .iter()
            .enumerate()
            .find(|(_, allocation)| {
                allocation.placement == Placement::Pools && allocation.memory.upgrade().is_some()
            })
            .map(|(position, allocation)| Divergence {
                position,
                size_bytes: allocation.size_bytes,
            });
        self.end_run_log();
        divergence
    }

    /// Forget every run and the plan, for a capture starting over.
    pub fn clear(&mut self) {
        self.run.clear();
        self.last.clear();
        self.ended.clear();
        self.kept.clear();
        self.running = false;
        self.planned = false;
        self.cursor = 0;
    }

    /// End the current run's log into `ended`, each allocation with whether something still
    /// holds it, and start the next run at the plan's first allocation.
    fn end_run_log(&mut self) {
        self.cursor = 0;
        self.ended.clear();
        self.ended
            .extend(self.run.drain(..).map(|allocation| PlannedAllocation {
                size_bytes: allocation.size_bytes,
                outlived: allocation.memory.upgrade().is_some(),
            }));
    }
}

impl core::fmt::Display for Divergence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "allocation {} of the recorded run, of {} B, is still held on room other graphs' \
             replays write, since the plan kept no buffer of its size where it was allocated: \
             the work must keep the same buffers, allocated in the same order, on every run",
            self.position, self.size_bytes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::Placement::{Output, Pools};
    use super::*;
    use crate::memory_management::ManagedMemoryHandle;

    /// An allocation as a run makes it: the pool's own reference, which a real slice keeps, and
    /// the binding the work holds until it lets go.
    fn allocation() -> (ManagedMemoryHandle, ManagedMemoryBinding) {
        let pool = ManagedMemoryHandle::new();
        let held = pool.clone().binding();
        (pool, held)
    }

    /// What a run allocating `sizes` left: the pool's references, since a slice outlives the
    /// run that freed it, the bindings of the buffers it kept, and where each allocation went.
    struct Allocated {
        _pools: Vec<ManagedMemoryHandle>,
        _held: Vec<ManagedMemoryBinding>,
        placements: Vec<Placement>,
    }

    /// Allocate `sizes` as a run does, each placed by the plan, keeping those `kept` names.
    fn allocate(planner: &mut AllocationPlanner, sizes: &[u64], kept: &[bool]) -> Allocated {
        let (mut pools, mut held, mut placements) = (Vec::new(), Vec::new(), Vec::new());
        for (&size, &keep) in sizes.iter().zip(kept) {
            let placement = planner.place_bytes(size);
            let (pool, binding) = allocation();
            planner.allocated(size, &binding, placement);
            pools.push(pool);
            placements.push(placement);
            if keep {
                held.push(binding);
            }
        }
        Allocated {
            _pools: pools,
            _held: held,
            placements,
        }
    }

    /// Run `planner` once over allocations of `sizes`, keeping those `kept` names, and answer
    /// whether it settled.
    fn run(planner: &mut AllocationPlanner, sizes: &[u64], kept: &[bool]) -> (bool, Allocated) {
        let allocated = allocate(planner, sizes, kept);
        (planner.finish_run(), allocated)
    }

    /// Record a run of `sizes` on `planner`, keeping those `kept` names: where each allocation
    /// went, and the divergence the recording ends on.
    fn record(
        planner: &mut AllocationPlanner,
        sizes: &[u64],
        kept: &[bool],
    ) -> (Vec<Placement>, Option<Divergence>) {
        planner.start_recording();
        let allocated = allocate(planner, sizes, kept);
        (allocated.placements, planner.finish_recording())
    }

    /// A planning run settles once two runs in a row allocate alike: the first never does, having
    /// nothing to compare with, and a run that kept another buffer has not settled.
    #[test]
    fn a_planning_settles_when_two_runs_allocate_alike() {
        let mut planner = AllocationPlanner::default();
        assert!(
            !run(&mut planner, &[8, 16], &[false, true]).0,
            "a first run"
        );
        assert!(
            !run(&mut planner, &[8, 16], &[true, true]).0,
            "it kept another buffer"
        );
        assert!(!run(&mut planner, &[8, 16], &[false, true]).0);
        assert!(run(&mut planner, &[8, 16], &[false, true]).0);
        assert!(planner.has_plan());
    }

    /// From the second run on, an allocation the run before found to be an output is placed off
    /// the shared room; a temporary goes to the pools, and a run that allocates as planned does
    /// not diverge.
    #[test]
    fn outputs_are_placed_where_the_plan_kept_them() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 16, 8], &[false, true, true]);

        let (placements, divergence) = record(&mut planner, &[8, 16, 8], &[false, true, true]);
        assert_eq!(placements, [Pools, Output, Output]);
        assert_eq!(divergence, None);
    }

    /// A first planning run has no plan to follow: everything it allocates goes to the pools.
    #[test]
    fn a_first_run_places_everything_in_the_pools() {
        let mut planner = AllocationPlanner::default();
        assert_eq!(planner.place_bytes(8), Pools);
        assert_eq!(planner.place_bytes(16), Pools);
    }

    /// A run allocating a temporary its plan did not make, as a framework writing into a fresh
    /// buffer where the planning run wrote in place does, places it in the pools, and the output
    /// after it still lands off the shared room.
    #[test]
    fn an_unplanned_temporary_leaves_the_outputs_in_place() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 16], &[false, true]);

        let (placements, divergence) = record(&mut planner, &[8, 32, 16], &[false, false, true]);
        assert_eq!(placements, [Pools, Pools, Output]);
        assert_eq!(divergence, None);
    }

    /// A run skipping a temporary its plan made, as one writing in place where the planning run
    /// wrote into a fresh buffer does, matches its next allocation past it.
    #[test]
    fn a_skipped_temporary_leaves_the_outputs_in_place() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 32, 16], &[false, false, true]);

        let (placements, divergence) = record(&mut planner, &[8, 16], &[false, true]);
        assert_eq!(placements, [Pools, Output]);
        assert_eq!(divergence, None);
    }

    /// An allocation is matched only within the window from the cursor: one whose size the plan
    /// allocates further down goes to the pools.
    #[test]
    fn a_size_planned_past_the_window_matches_nothing() {
        let mut planner = AllocationPlanner::default();
        let mut sizes = [8; MATCH_WINDOW_ALLOCATIONS + 1];
        sizes[MATCH_WINDOW_ALLOCATIONS] = 16;
        let mut kept = [false; MATCH_WINDOW_ALLOCATIONS + 1];
        kept[MATCH_WINDOW_ALLOCATIONS] = true;
        let _planned = run(&mut planner, &sizes, &kept);

        planner.start_recording();
        assert_eq!(planner.place_bytes(16), Pools);
    }

    /// A recorded run that keeps a planned temporary diverges there: that buffer sits on the
    /// shared room. One that lets a planned output go does not: its graph holds that room.
    #[test]
    fn keeping_a_temporary_diverges_and_freeing_an_output_does_not() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 16], &[false, true]);

        let (_, divergence) = record(&mut planner, &[8, 16], &[true, false]);
        assert_eq!(
            divergence,
            Some(Divergence {
                position: 0,
                size_bytes: 8
            })
        );
    }

    /// A recorded run that keeps a buffer its plan never allocated diverges there: it went to the
    /// pools, which may carve it from the shared room.
    #[test]
    fn keeping_an_unplanned_buffer_diverges() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 16], &[false, true]);

        let (_, divergence) = record(&mut planner, &[8, 16, 32], &[false, true, true]);
        assert_eq!(
            divergence,
            Some(Divergence {
                position: 2,
                size_bytes: 32
            })
        );
    }
}
