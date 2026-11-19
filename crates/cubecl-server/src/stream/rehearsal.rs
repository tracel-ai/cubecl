//! What a capture learns from rehearsing its work before recording it.
//!
//! A graph on a stream replays its temporaries over the room the stream's other graphs
//! replay theirs on: that is what lets a stream hold about the largest graph's working set
//! rather than the sum of all of them (see
//! [`PinnedPages`](crate::memory_management::PinnedPages)). A buffer the graph's work leaves
//! behind — an output the caller reads — must never sit on that room, or another graph's
//! replay overwrites it. Whether an allocation is a temporary or an output is only known once
//! the work is over, by which time a recording has baked its address.
//!
//! So the work runs first as a rehearsal, its launches discarded: each run lists its
//! allocations in order and, once it is over, which of them something still holds. The
//! rehearsal repeats until two runs in a row allocate alike, since a first run can take
//! another path than the steady state (a tune, a fused optimization being built). The
//! recorded run then allocates the same sequence, and each allocation is placed by what the
//! rehearsal found at its position: an output on room set aside for it before the window
//! opens, a temporary on the shared room. A recording that allocates otherwise than its
//! rehearsal does not seal.

use crate::memory_management::{ManagedMemoryBinding, ManagedMemoryHandle, WeakMemoryBinding};
use alloc::vec::Vec;

/// One allocation of a finished rehearsal run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rehearsed {
    /// The size the allocation asked for, in bytes.
    pub size_bytes: u64,
    /// Whether something still held it once the run was over: an output.
    pub outlived: bool,
}

/// Where the recorded run's next allocation goes, by what the rehearsal found at its
/// position.
#[derive(Debug)]
pub enum Placement {
    /// An output: the room set aside for it.
    Output(ManagedMemoryHandle),
    /// A temporary, or work outside a recording: the pools decide.
    Pools,
}

/// The rehearsal of one capture and the plan the recording follows.
///
/// Built and cleared with the capture's lifecycle by
/// [`StreamCapture`](super::StreamCapture), which is its only owner.
#[derive(Debug, Default)]
pub struct Rehearsal {
    /// The current run's allocations: what each asked for and the memory it got, unheld so
    /// the record never keeps one alive. Arrival usage: pushed per allocation, drained in
    /// order when the run ends.
    run: Vec<(u64, WeakMemoryBinding)>,
    /// The last finished run, in allocation order. Position usage: the recording reads the
    /// entry at its cursor.
    last: Vec<Rehearsed>,
    /// Whether any run finished since the capture was prepared.
    rehearsed: bool,
    /// The room set aside for each output of `last`, at its position; `None` for a
    /// temporary or a slot already handed out. Position usage, as `last`.
    slots: Vec<Option<ManagedMemoryHandle>>,
    /// How many allocations the recorded run made so far: its position in `last`.
    cursor: usize,
    /// The first place the recorded run allocated otherwise than its rehearsal.
    diverged: Option<Divergence>,
}

/// Where a recorded run first allocated otherwise than its rehearsal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Divergence {
    /// The position of the allocation in the run.
    pub position: usize,
    /// What the rehearsal allocated there, if it allocated that many.
    pub rehearsed: Option<Rehearsed>,
    /// The size the recorded run asked for there, if it allocated that many.
    pub recorded_bytes: Option<u64>,
}

impl Rehearsal {
    /// Note an allocation of `size_bytes` bound to `memory` by the current run.
    pub fn allocated(&mut self, size_bytes: u64, memory: &ManagedMemoryBinding) {
        self.run.push((size_bytes, memory.downgrade()));
    }

    /// End a rehearsal run, once its deferred frees are reclaimed, and answer whether it
    /// allocated what the run before it did: the work has reached its steady state, and the
    /// recording can follow it.
    pub fn finish_run(&mut self) -> bool {
        let finished: Vec<Rehearsed> = self
            .run
            .drain(..)
            .map(|(size_bytes, memory)| Rehearsed {
                size_bytes,
                outlived: memory.upgrade().is_some(),
            })
            .collect();
        let stable = self.rehearsed && finished == self.last;
        self.last = finished;
        self.rehearsed = true;
        stable
    }

    /// Whether any rehearsal run finished: a recording has a plan to follow.
    pub fn has_plan(&self) -> bool {
        self.rehearsed
    }

    /// The size of every output of the plan, in allocation order: what
    /// [`set_aside`](Self::set_aside) needs room for.
    pub fn output_sizes(&self) -> impl Iterator<Item = u64> + '_ {
        self.last
            .iter()
            .filter(|rehearsed| rehearsed.outlived)
            .map(|rehearsed| rehearsed.size_bytes)
    }

    /// Hold `slots`, one per [`output_sizes`](Self::output_sizes) entry in that order, for the
    /// recorded run's outputs, and start the recording at the plan's first allocation.
    pub fn set_aside(&mut self, slots: impl IntoIterator<Item = ManagedMemoryHandle>) {
        let mut slots = slots.into_iter();
        self.slots.clear();
        self.slots.extend(
            self.last
                .iter()
                .map(|rehearsed| rehearsed.outlived.then(|| slots.next()).flatten()),
        );
        self.cursor = 0;
        self.diverged = None;
    }

    /// Place the recorded run's next allocation, of `size_bytes`: on its slot when the plan
    /// has an output of that size at that position. Anything else goes to the pools, and an
    /// allocation the plan does not have at that position is the divergence
    /// [`finish_recording`](Self::finish_recording) reports.
    pub fn place(&mut self, size_bytes: u64) -> Placement {
        let position = self.cursor;
        self.cursor += 1;
        let rehearsed = self.last.get(position).copied();
        if rehearsed.map(|rehearsed| rehearsed.size_bytes) != Some(size_bytes) {
            self.diverge(position, rehearsed, Some(size_bytes));
            return Placement::Pools;
        }
        match self.slots[position].take() {
            Some(slot) => Placement::Output(slot),
            None => Placement::Pools,
        }
    }

    /// End the recorded run, once its deferred frees are reclaimed: the first place it
    /// allocated otherwise than the plan, if there is one — another size, another count, or a
    /// temporary it kept, which may sit on room another graph's replay writes. An output it
    /// let go is no divergence: its slot only went back to the pools. The slots it never claimed go back to the pools.
    pub fn finish_recording(&mut self) -> Option<Divergence> {
        let recorded: Vec<Rehearsed> = self
            .run
            .drain(..)
            .map(|(size_bytes, memory)| Rehearsed {
                size_bytes,
                outlived: memory.upgrade().is_some(),
            })
            .collect();
        self.slots.clear();
        if let Some(diverged) = self.diverged {
            return Some(diverged);
        }
        // A kept temporary may sit on the shared room; a freed output only gave its slot back.
        let position = recorded
            .iter()
            .zip(&self.last)
            .position(|(recorded, rehearsed)| recorded.outlived && !rehearsed.outlived)
            .or_else(|| {
                (recorded.len() != self.last.len()).then(|| recorded.len().min(self.last.len()))
            })?;
        Some(Divergence {
            position,
            rehearsed: self.last.get(position).copied(),
            recorded_bytes: recorded.get(position).map(|recorded| recorded.size_bytes),
        })
    }

    /// Forget every run and the plan, for a capture starting over.
    pub fn clear(&mut self) {
        self.run.clear();
        self.last.clear();
        self.rehearsed = false;
        self.slots.clear();
        self.cursor = 0;
        self.diverged = None;
    }

    fn diverge(&mut self, position: usize, rehearsed: Option<Rehearsed>, recorded: Option<u64>) {
        self.diverged.get_or_insert(Divergence {
            position,
            rehearsed,
            recorded_bytes: recorded,
        });
    }
}

impl core::fmt::Display for Divergence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let rehearsed = match self.rehearsed {
            Some(Rehearsed {
                size_bytes,
                outlived: true,
            }) => alloc::format!("an output of {size_bytes} B"),
            Some(Rehearsed { size_bytes, .. }) => alloc::format!("a temporary of {size_bytes} B"),
            None => "nothing".into(),
        };
        let recorded = match self.recorded_bytes {
            Some(size_bytes) => alloc::format!("{size_bytes} B"),
            None => "nothing".into(),
        };
        write!(
            f,
            "allocation {} of the recorded run was {recorded} where the rehearsal allocated \
             {rehearsed}: the work must allocate the same buffers, and keep the same ones, on \
             every run",
            self.position
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An allocation as a run makes it: the pool's own reference, which a real slice keeps, and
    /// the binding the work holds until it lets go.
    fn allocation() -> (ManagedMemoryHandle, ManagedMemoryBinding) {
        let pool = ManagedMemoryHandle::new();
        let held = pool.clone().binding();
        (pool, held)
    }

    /// Run `rehearsal` once over allocations of `sizes`, keeping those `kept` names, and answer
    /// whether it settled. The pool's references are handed back, since a slice outlives the run
    /// that freed it.
    fn run(
        rehearsal: &mut Rehearsal,
        sizes: &[u64],
        kept: &[bool],
    ) -> (bool, Vec<ManagedMemoryHandle>, Vec<ManagedMemoryBinding>) {
        let (mut pools, mut held) = (Vec::new(), Vec::new());
        for (&size, &keep) in sizes.iter().zip(kept) {
            let (pool, binding) = allocation();
            rehearsal.allocated(size, &binding);
            pools.push(pool);
            if keep {
                held.push(binding);
            }
        }
        (rehearsal.finish_run(), pools, held)
    }

    /// A rehearsal settles once two runs in a row allocate alike: the first never does, having
    /// nothing to compare with, and a run that kept another buffer has not settled.
    #[test]
    fn a_rehearsal_settles_when_two_runs_allocate_alike() {
        let mut rehearsal = Rehearsal::default();
        assert!(
            !run(&mut rehearsal, &[8, 16], &[false, true]).0,
            "a first run"
        );
        assert!(
            !run(&mut rehearsal, &[8, 16], &[true, true]).0,
            "it kept another buffer"
        );
        assert!(!run(&mut rehearsal, &[8, 16], &[false, true]).0);
        assert!(run(&mut rehearsal, &[8, 16], &[false, true]).0);
        assert!(rehearsal.has_plan());
    }

    /// The recorded run's outputs take the room set aside for them, by their position in the
    /// plan, and its temporaries go to the pools; a run allocating as rehearsed does not diverge.
    #[test]
    fn outputs_take_their_slots_by_position() {
        let mut rehearsal = Rehearsal::default();
        let _rehearsed = run(&mut rehearsal, &[8, 16, 8], &[false, true, true]);
        assert_eq!(rehearsal.output_sizes().collect::<Vec<_>>(), [16, 8]);
        let (first, second) = (ManagedMemoryHandle::new(), ManagedMemoryHandle::new());
        let first_id = first.clone().binding().id();
        rehearsal.set_aside([first, second]);

        assert!(matches!(rehearsal.place(8), Placement::Pools));
        match rehearsal.place(16) {
            Placement::Output(slot) => assert_eq!(slot.binding().id(), first_id),
            Placement::Pools => panic!("the output at position 1 has a slot"),
        }
        assert!(matches!(rehearsal.place(8), Placement::Output(_)));

        let (_pools, _held) = record(&mut rehearsal, &[8, 16, 8], &[false, true, true]);
        assert_eq!(rehearsal.finish_recording(), None);
    }

    /// Note a recorded run's allocations, as [`run`] does for a rehearsal's.
    fn record(
        rehearsal: &mut Rehearsal,
        sizes: &[u64],
        kept: &[bool],
    ) -> (Vec<ManagedMemoryHandle>, Vec<ManagedMemoryBinding>) {
        let (mut pools, mut held) = (Vec::new(), Vec::new());
        for (&size, &keep) in sizes.iter().zip(kept) {
            let (pool, binding) = allocation();
            rehearsal.allocated(size, &binding);
            pools.push(pool);
            if keep {
                held.push(binding);
            }
        }
        (pools, held)
    }

    /// A recorded run that keeps a rehearsed temporary diverges there: that buffer sits on the
    /// shared room. One that lets a rehearsed output go does not: its slot only went back.
    #[test]
    fn keeping_a_temporary_diverges_and_freeing_an_output_does_not() {
        let mut rehearsal = Rehearsal::default();
        let _rehearsed = run(&mut rehearsal, &[8, 16], &[false, true]);
        rehearsal.set_aside([ManagedMemoryHandle::new()]);
        let (_pools, _held) = record(&mut rehearsal, &[8, 16], &[true, false]);
        let divergence = rehearsal.finish_recording().expect("a kept temporary");
        assert_eq!(divergence.position, 0);
        assert_eq!(divergence.recorded_bytes, Some(8));
    }

    /// A recorded run that asks for another size, or for more or fewer buffers, than its plan
    /// diverges at the first place it does.
    #[test]
    fn another_size_or_count_diverges() {
        let mut rehearsal = Rehearsal::default();
        let _rehearsed = run(&mut rehearsal, &[8, 16], &[false, false]);

        rehearsal.set_aside([]);
        assert!(matches!(rehearsal.place(8), Placement::Pools));
        assert!(matches!(rehearsal.place(32), Placement::Pools));
        let (_pools, _held) = record(&mut rehearsal, &[8, 32], &[false, false]);
        let divergence = rehearsal.finish_recording().expect("another size");
        assert_eq!(divergence.position, 1);
        assert_eq!(divergence.recorded_bytes, Some(32));

        rehearsal.set_aside([]);
        let (_pools, _held) = record(&mut rehearsal, &[8], &[false]);
        let divergence = rehearsal.finish_recording().expect("fewer buffers");
        assert_eq!((divergence.position, divergence.recorded_bytes), (1, None));
    }
}
