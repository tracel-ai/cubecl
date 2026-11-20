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
//! second run on, each allocation is placed by what the run before found at its position: an
//! output off the room other graphs pinned, a temporary on it. The planning run repeats until two
//! runs in a row allocate alike, since a first run can take another path than the steady state
//! (a tune, a fused optimization being built). The recorded run then allocates the same
//! sequence under the same plan, so it lands where the last planning did, in room that
//! planning grew. A recording that allocates otherwise than its planning run does not seal.

use crate::memory_management::{ManagedMemoryBinding, WeakMemoryBinding};
use alloc::vec::Vec;

/// One allocation of a finished planning run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedAllocation {
    /// The size the allocation asked for, in bytes.
    pub size_bytes: u64,
    /// Whether something still held it once the run was over: an output.
    pub outlived: bool,
}

/// Where a run's next allocation goes, by what the run before found at its position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// An output: off the room the stream's other graphs pinned, which their replays write.
    Output,
    /// A temporary, or work outside a capture: the pools decide.
    Pools,
}

/// The planning run of one capture and the plan the recording follows.
///
/// Built and cleared with the capture's lifecycle by
/// [`StreamCapture`](super::StreamCapture), which is its only owner.
#[derive(Debug, Default)]
pub struct AllocationPlanner {
    /// The current run's allocations: what each asked for and the memory it got, unheld so
    /// the record never keeps one alive. Arrival usage: pushed per allocation, drained in
    /// order when the run ends.
    run: Vec<(u64, WeakMemoryBinding)>,
    /// The last finished run, in allocation order: the plan the next run follows. Position
    /// usage: a run reads the entry at its cursor.
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
    /// How many allocations the current run made so far: its position in `last`.
    cursor: usize,
    /// The first place the recorded run allocated otherwise than the plan.
    diverged: Option<Divergence>,
}

/// Where a recorded run first allocated otherwise than its planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Divergence {
    /// The position of the allocation in the run.
    pub position: usize,
    /// What the planning run allocated there, if it allocated that many.
    pub planned: Option<PlannedAllocation>,
    /// The size the recorded run asked for there, if it allocated that many.
    pub recorded_bytes: Option<u64>,
}

impl AllocationPlanner {
    /// Note an allocation of `size_bytes` bound to `memory` by the current run.
    pub fn allocated(&mut self, size_bytes: u64, memory: &ManagedMemoryBinding) {
        self.run.push((size_bytes, memory.downgrade()));
    }

    /// End a planning run, once its deferred frees are reclaimed, and answer whether it
    /// allocated what the run before it did: the work has reached its steady state, and the
    /// recording can follow it.
    pub fn finish_run(&mut self) -> bool {
        self.kept.extend(
            self.run
                .iter()
                .filter(|(_, memory)| memory.upgrade().is_some())
                .map(|(_, memory)| memory.clone()),
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
        self.diverged = None;
    }

    /// Place the current run's next allocation, of `size_bytes`: off the shared room when the
    /// plan has an output of that size at that position. Anything else goes to the pools, and an
    /// allocation the plan does not have at that position is the divergence
    /// [`finish_recording`](Self::finish_recording) reports.
    pub fn place_bytes(&mut self, size_bytes: u64) -> Placement {
        let position = self.cursor;
        self.cursor += 1;
        let planned = self.last.get(position).copied();
        let Some(planned) = planned.filter(|planned| planned.size_bytes == size_bytes) else {
            self.diverge(position, planned, Some(size_bytes));
            return Placement::Pools;
        };
        match planned.outlived {
            true => Placement::Output,
            false => Placement::Pools,
        }
    }

    /// End the recorded run, once its deferred frees are reclaimed: the first place it
    /// allocated otherwise than the plan, if there is one — another size, another count, or a
    /// temporary it kept, which may sit on room another graph's replay writes. An output it
    /// let go is no divergence: its graph holds the room, which every replay still writes.
    pub fn finish_recording(&mut self) -> Option<Divergence> {
        self.running = false;
        let diverged = self.diverged;
        self.end_run_log();
        if diverged.is_some() {
            return diverged;
        }
        let recorded = &self.ended;
        // A kept temporary may sit on the shared room; a freed output's room is its graph's.
        let position = recorded
            .iter()
            .zip(&self.last)
            .position(|(recorded, planned)| recorded.outlived && !planned.outlived)
            .or_else(|| {
                (recorded.len() != self.last.len()).then(|| recorded.len().min(self.last.len()))
            })?;
        Some(Divergence {
            position,
            planned: self.last.get(position).copied(),
            recorded_bytes: recorded.get(position).map(|recorded| recorded.size_bytes),
        })
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
        self.diverged = None;
    }

    /// End the current run's log into `ended`, each allocation with whether something still
    /// holds it, and start the next run at the plan's first allocation.
    fn end_run_log(&mut self) {
        self.cursor = 0;
        self.diverged = None;
        self.ended.clear();
        self.ended.extend(
            self.run
                .drain(..)
                .map(|(size_bytes, memory)| PlannedAllocation {
                    size_bytes,
                    outlived: memory.upgrade().is_some(),
                }),
        );
    }

    fn diverge(
        &mut self,
        position: usize,
        planned: Option<PlannedAllocation>,
        recorded: Option<u64>,
    ) {
        self.diverged.get_or_insert(Divergence {
            position,
            planned,
            recorded_bytes: recorded,
        });
    }
}

impl core::fmt::Display for Divergence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let planned = match self.planned {
            Some(PlannedAllocation {
                size_bytes,
                outlived: true,
            }) => alloc::format!("an output of {size_bytes} B"),
            Some(PlannedAllocation { size_bytes, .. }) => {
                alloc::format!("a temporary of {size_bytes} B")
            }
            None => "nothing".into(),
        };
        let recorded = match self.recorded_bytes {
            Some(size_bytes) => alloc::format!("{size_bytes} B"),
            None => "nothing".into(),
        };
        write!(
            f,
            "allocation {} of the recorded run was {recorded} where the planning run allocated \
             {planned}: the work must allocate the same buffers, and keep the same ones, on \
             every run",
            self.position
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_management::ManagedMemoryHandle;

    /// An allocation as a run makes it: the pool's own reference, which a real slice keeps, and
    /// the binding the work holds until it lets go.
    fn allocation() -> (ManagedMemoryHandle, ManagedMemoryBinding) {
        let pool = ManagedMemoryHandle::new();
        let held = pool.clone().binding();
        (pool, held)
    }

    /// Run `planner` once over allocations of `sizes`, keeping those `kept` names, and answer
    /// whether it settled. The pool's references are handed back, since a slice outlives the run
    /// that freed it.
    fn run(
        planner: &mut AllocationPlanner,
        sizes: &[u64],
        kept: &[bool],
    ) -> (bool, Vec<ManagedMemoryHandle>, Vec<ManagedMemoryBinding>) {
        let (mut pools, mut held) = (Vec::new(), Vec::new());
        for (&size, &keep) in sizes.iter().zip(kept) {
            let (pool, binding) = allocation();
            planner.allocated(size, &binding);
            pools.push(pool);
            if keep {
                held.push(binding);
            }
        }
        (planner.finish_run(), pools, held)
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
    /// the shared room, by its position; a temporary goes to the pools, and a run that allocates
    /// as planned does not diverge.
    #[test]
    fn outputs_are_placed_by_position() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 16, 8], &[false, true, true]);

        planner.start_recording();
        assert_eq!(planner.place_bytes(8), Placement::Pools);
        assert_eq!(planner.place_bytes(16), Placement::Output);
        assert_eq!(planner.place_bytes(8), Placement::Output);
        let (_pools, _held) = record(&mut planner, &[8, 16, 8], &[false, true, true]);
        assert_eq!(planner.finish_recording(), None);
    }

    /// A first planning run has no plan to follow: everything it allocates goes to the pools.
    #[test]
    fn a_first_run_places_everything_in_the_pools() {
        let mut planner = AllocationPlanner::default();
        assert_eq!(planner.place_bytes(8), Placement::Pools);
        assert_eq!(planner.place_bytes(16), Placement::Pools);
    }

    /// Note a recorded run's allocations, as [`run`] does for a planning run's.
    fn record(
        planner: &mut AllocationPlanner,
        sizes: &[u64],
        kept: &[bool],
    ) -> (Vec<ManagedMemoryHandle>, Vec<ManagedMemoryBinding>) {
        let (mut pools, mut held) = (Vec::new(), Vec::new());
        for (&size, &keep) in sizes.iter().zip(kept) {
            let (pool, binding) = allocation();
            planner.allocated(size, &binding);
            pools.push(pool);
            if keep {
                held.push(binding);
            }
        }
        (pools, held)
    }

    /// A recorded run that keeps a planned temporary diverges there: that buffer sits on the
    /// shared room. One that lets a planned output go does not: its graph holds that room.
    #[test]
    fn keeping_a_temporary_diverges_and_freeing_an_output_does_not() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 16], &[false, true]);
        planner.start_recording();
        let (_pools, _held) = record(&mut planner, &[8, 16], &[true, false]);
        let divergence = planner.finish_recording().expect("a kept temporary");
        assert_eq!(divergence.position, 0);
        assert_eq!(divergence.recorded_bytes, Some(8));
    }

    /// A recorded run that asks for another size, or for more or fewer buffers, than its plan
    /// diverges at the first place it does.
    #[test]
    fn another_size_or_count_diverges() {
        let mut planner = AllocationPlanner::default();
        let _planned = run(&mut planner, &[8, 16], &[false, false]);

        planner.start_recording();
        assert_eq!(planner.place_bytes(8), Placement::Pools);
        assert_eq!(planner.place_bytes(32), Placement::Pools);
        let (_pools, _held) = record(&mut planner, &[8, 32], &[false, false]);
        let divergence = planner.finish_recording().expect("another size");
        assert_eq!(divergence.position, 1);
        assert_eq!(divergence.recorded_bytes, Some(32));

        planner.start_recording();
        let (_pools, _held) = record(&mut planner, &[8], &[false]);
        let divergence = planner.finish_recording().expect("fewer buffers");
        assert_eq!((divergence.position, divergence.recorded_bytes), (1, None));
    }
}
