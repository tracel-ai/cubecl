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
//! second run on, each allocation is placed by the run before: an output off the room other
//! graphs pinned, a temporary on it. The planning runs repeat until two runs in a row keep the
//! same buffers, since a first run can take another path than the steady state (a tune, a fused
//! optimization being built). The recorded run then allocates under the same plan, so it lands
//! where the last planning did, in room that planning grew.
//!
//! A run need not allocate its temporaries as the run before did. A framework writes in place
//! into a buffer nothing else holds and into a fresh one otherwise, and what else holds a buffer
//! when it decides can depend on how far another queue has drained. So an allocation takes the
//! plan's entry at the run's cursor when it has the same size, and otherwise the plan's next
//! output when that has its size, skipping the temporaries the run did not make; one that
//! matches neither is placed as a temporary. What a temporary sits on is the run's own business.
//!
//! The recording refuses to seal only when it keeps a buffer it placed as a temporary, since that
//! buffer may sit on room another graph's replay writes. A run can always place a buffer that way:
//! a temporary of the size of the output after it looks like that output when it comes. The
//! refused recording's own allocations then become the plan of the capture prepared next on the
//! stream, which may record at once: a recording that allocates as the refused one did follows
//! that plan exactly, and seals.

use crate::memory_management::{ManagedMemoryBinding, WeakMemoryBinding};
use alloc::vec::Vec;

/// One allocation of a finished run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedAllocation {
    /// The size the allocation asked for, in bytes.
    pub size_bytes: u64,
    /// Whether something still held it once the run was over: an output.
    pub outlived: bool,
}

/// Where a run's next allocation goes, by the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// An output: off the room the stream's other graphs pinned, which their replays write.
    Output,
    /// A temporary, or work outside a capture: the pools decide.
    Pools,
}

/// One allocation of the current run, as it was made.
#[derive(Debug)]
pub struct RunAllocation {
    /// The size the allocation asked for, in bytes.
    pub size_bytes: u64,
    /// The memory it was bound to, unheld, so the record never keeps the allocation alive.
    pub memory: WeakMemoryBinding,
    /// Where it was placed.
    pub placement: Placement,
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
    /// The plan the next run follows, in allocation order: the last finished planning run, or
    /// the last refused recording. Position usage: a run reads the entry at its cursor and
    /// looks for the next output from there.
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
    /// Whether `last` holds a plan.
    planned: bool,
    /// Whether a planning run of this capture finished, so `last` is this capture's own.
    ran: bool,
    /// Whether `last` is a refused recording's, for the capture prepared next to follow.
    carried: bool,
    /// The position in `last` after the current run's last matched allocation.
    cursor: usize,
}

/// A buffer the recorded run keeps on room other graphs' replays write: it was placed as a
/// temporary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Divergence {
    /// How many allocations the recorded run made before it.
    pub position: usize,
    /// The size it asked for, in bytes.
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

    /// End a planning run, once its deferred frees are reclaimed, and answer whether it kept
    /// the buffers the run before it of this capture kept, the same sizes in the same order:
    /// what the recording's seal depends on has settled, and the recording can follow it.
    pub fn finish_run(&mut self) -> bool {
        self.kept.extend(
            self.run
                .iter()
                .filter(|allocation| allocation.memory.upgrade().is_some())
                .map(|allocation| allocation.memory.clone()),
        );
        self.end_run_log();
        self.running = false;
        let outputs = |run: &[PlannedAllocation]| {
            run.iter()
                .filter(|planned| planned.outlived)
                .map(|planned| planned.size_bytes)
                .collect::<Vec<_>>()
        };
        let stable = self.ran && outputs(&self.ended) == outputs(&self.last);
        core::mem::swap(&mut self.last, &mut self.ended);
        self.planned = true;
        self.ran = true;
        stable
    }

    /// Whether a recording has a plan to follow: a planning run finished, or the capture
    /// carries a refused recording's allocations.
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

    /// Place the current run's next allocation, of `size_bytes`. The plan's entry at the cursor
    /// places it when it has that size, and otherwise the plan's next output does when it has
    /// that size, past the temporaries before it; either moves the cursor past the entry. One
    /// that matches neither goes to the pools, and leaves the cursor where it was.
    pub fn place_bytes(&mut self, size_bytes: u64) -> Placement {
        let at_cursor = self
            .last
            .get(self.cursor)
            .filter(|planned| planned.size_bytes == size_bytes)
            .map(|planned| (self.cursor, planned));
        let next_output = || {
            self.last
                .iter()
                .enumerate()
                .skip(self.cursor)
                .find(|(_, planned)| planned.outlived)
                .filter(|(_, planned)| planned.size_bytes == size_bytes)
        };
        let Some((position, planned)) = at_cursor.or_else(next_output) else {
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
    ///
    /// A recording that diverges leaves its own allocations as the plan of the capture prepared
    /// next on the stream.
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
        self.carried = divergence.is_some();
        if self.carried {
            core::mem::swap(&mut self.last, &mut self.ended);
        }
        divergence
    }

    /// Start over for a capture being prepared: every run is forgotten, and so is the plan,
    /// unless the last recording on the stream was refused, whose allocations the new capture
    /// follows until a planning run of its own finishes.
    pub fn restart(&mut self) {
        let carried = core::mem::take(&mut self.carried);
        let plan = core::mem::take(&mut self.last);
        self.clear();
        if carried {
            self.last = plan;
            self.planned = true;
        }
    }

    /// Forget every run and the plan.
    pub fn clear(&mut self) {
        self.run.clear();
        self.last.clear();
        self.ended.clear();
        self.kept.clear();
        self.running = false;
        self.planned = false;
        self.ran = false;
        self.carried = false;
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
            "a buffer of {} B, allocated after {} others, is still held but was placed as a \
             temporary, on room other graphs' replays write; a capture prepared next on this \
             stream follows the refused recording's allocations, and may record at once",
            self.size_bytes, self.position
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

    /// What a run left: the pool's references, since a slice outlives the run that freed it,
    /// the bindings of the buffers it kept, and where each allocation went.
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

    /// A planning run settles once two runs in a row keep the same buffers: the first never
    /// does, having nothing to compare with, and a run that kept another buffer has not settled.
    #[test]
    fn a_planning_settles_when_two_runs_keep_alike() {
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

    /// Runs that keep the same buffers settle whatever temporaries they allocate around them.
    #[test]
    fn a_planning_settles_over_other_temporaries() {
        let mut planner = AllocationPlanner::default();
        let _first = run(&mut planner, &[8, 16], &[false, true]);
        assert!(run(&mut planner, &[8, 8, 32, 16], &[false, false, false, true]).0);
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

    /// A run allocating a temporary of a size the plan allocates again later, as a model step's
    /// hidden-size temporaries do, still places the output after it off the shared room.
    #[test]
    fn an_extra_temporary_of_a_recurring_size_leaves_the_output_in_place() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 16, 8], &[false, true, false]);

        let (placements, divergence) =
            record(&mut planner, &[8, 8, 16, 8], &[false, false, true, false]);
        assert_eq!(placements, [Pools, Pools, Output, Pools]);
        assert_eq!(divergence, None);
    }

    /// A run skipping a temporary its plan made, as one writing in place where the planning run
    /// wrote into a fresh buffer does, places the output after it off the shared room.
    #[test]
    fn a_skipped_temporary_leaves_the_output_in_place() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 32, 16], &[false, false, true]);

        let (placements, divergence) = record(&mut planner, &[8, 16], &[false, true]);
        assert_eq!(placements, [Pools, Output]);
        assert_eq!(divergence, None);
    }

    /// The next output matches however far down the plan it sits: there is no window.
    #[test]
    fn an_output_matches_past_any_number_of_skipped_temporaries() {
        let mut planner = AllocationPlanner::default();
        let mut sizes = [8; 101];
        sizes[100] = 16;
        let mut kept = [false; 101];
        kept[100] = true;
        let _planned = run(&mut planner, &sizes, &kept);

        let (placements, divergence) = record(&mut planner, &[16], &[true]);
        assert_eq!(placements, [Output]);
        assert_eq!(divergence, None);
    }

    /// A temporary of another size than the next output matches the plan only at the cursor:
    /// one the plan makes past an output is not reached across it.
    #[test]
    fn a_temporary_is_not_matched_across_an_output() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 16, 32], &[false, true, false]);

        let (placements, divergence) =
            record(&mut planner, &[8, 32, 16, 32], &[false, false, true, false]);
        assert_eq!(placements, [Pools, Pools, Output, Pools]);
        assert_eq!(divergence, None);
    }

    /// A recorded run that writes in place where the plan wrote a temporary and then a fresh
    /// output of its size keeps a buffer it placed as that temporary, and is refused. The next
    /// capture follows the refused recording, and a recording allocating as it did seals.
    #[test]
    fn an_output_planned_after_a_temporary_of_its_size_seals_on_the_next_recording() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 8], &[false, true]);

        let (placements, divergence) = record(&mut planner, &[8], &[true]);
        assert_eq!(placements, [Pools]);
        assert_eq!(
            divergence,
            Some(Divergence {
                position: 0,
                size_bytes: 8
            })
        );

        planner.restart();
        assert!(planner.has_plan(), "the refused recording is the plan");
        let (placements, divergence) = record(&mut planner, &[8], &[true]);
        assert_eq!(placements, [Output]);
        assert_eq!(divergence, None);
    }

    /// A recorded run that allocates a temporary before an output of the same size, where the
    /// plan wrote that output in place, gives the temporary the output's room and is refused.
    /// The next capture follows the refused recording, and a recording allocating as it did
    /// seals.
    #[test]
    fn a_temporary_before_an_output_of_its_size_seals_on_the_next_recording() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8], &[true]);

        let (placements, divergence) = record(&mut planner, &[8, 8], &[false, true]);
        assert_eq!(placements, [Output, Pools]);
        assert_eq!(
            divergence,
            Some(Divergence {
                position: 1,
                size_bytes: 8
            })
        );

        planner.restart();
        let (placements, divergence) = record(&mut planner, &[8, 8], &[false, true]);
        assert_eq!(placements, [Pools, Output]);
        assert_eq!(divergence, None);
    }

    /// A capture prepared after one that sealed, or after a planning run, starts with no plan.
    #[test]
    fn only_a_refused_recording_carries_its_plan() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8], &[true]);
        let (_, divergence) = record(&mut planner, &[8], &[true]);
        assert_eq!(divergence, None);

        planner.restart();
        assert!(!planner.has_plan());
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
