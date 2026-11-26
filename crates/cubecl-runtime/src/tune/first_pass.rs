use alloc::vec;
use alloc::vec::Vec;
use core::time::Duration;

use crate::tune::Batch;
use crate::tune::patience::PatienceTable;

/// The next candidate a batch's first pass measures, and the candidates to compile before it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FirstPassStep {
    pub(crate) slot: usize,
    /// Ascending, and empty when `slot` was compiled with an earlier window.
    pub(crate) compile: Vec<usize>,
}

/// How many candidates a first pass compiles together: in its first window, and in each window
/// after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompileWindows {
    pub(crate) first_count: usize,
    pub(crate) later_count: usize,
}

/// Plans a batch's first pass: its candidates are measured one at a time in priority order, past
/// the ones their groups' patience skips, and compiled ahead of measuring, a window at a time.
///
/// Which candidates are measured, and so which can win, depends only on their order and their
/// measurements: the windows decide how many compile together, never what is measured. A pass
/// that stops compiles at most one window it never measures.
#[derive(Debug)]
pub(crate) struct FirstPassPlanner {
    patience: PatienceTable,
    /// Whether each slot's kernels were compiled, by slot (Position Usage).
    compiled: Vec<bool>,
    /// The first slot not planned yet.
    next: usize,
    measured: usize,
    min_measured: usize,
    /// The size of the next window to compile, `later_window` once one was.
    window: usize,
    later_window: usize,
}

impl FirstPassPlanner {
    /// A planner that compiles `windows` candidates together, at least one a window.
    pub(crate) fn new(batch: &Batch, windows: CompileWindows) -> Self {
        let patience = PatienceTable::new(batch);
        Self {
            min_measured: patience.min_measured(),
            compiled: vec![false; batch.entries.len()],
            patience,
            next: 0,
            measured: 0,
            window: windows.first_count.max(1),
            later_window: windows.later_count.max(1),
        }
    }

    /// The next slot to measure, with the window to compile first when that slot is not compiled
    /// yet: the next slots from it that are neither compiled nor skipped, as many as the window
    /// holds. `None` once
    /// every remaining slot is skipped.
    pub(crate) fn plan_next(&mut self) -> Option<FirstPassStep> {
        let slot = (self.next..self.compiled.len()).find(|slot| !self.patience.skips(*slot))?;
        self.next = slot + 1;
        let compile = if self.compiled[slot] {
            Vec::new()
        } else {
            self.claim_window(slot)
        };
        Some(FirstPassStep { slot, compile })
    }

    /// Count the candidate at `slot` as measured, by its best first-pass time.
    pub(crate) fn record(&mut self, slot: usize, time: Duration) {
        self.measured += 1;
        self.patience.record(slot, time);
    }

    /// Whether the batch measured enough candidates for the bounds' time limit to end it.
    pub(crate) fn may_short_circuit(&self) -> bool {
        self.measured >= self.min_measured
    }
}

impl FirstPassPlanner {
    /// Mark the window starting at `slot` compiled, and return it.
    fn claim_window(&mut self, slot: usize) -> Vec<usize> {
        let window: Vec<usize> = (slot..self.compiled.len())
            .filter(|slot| !self.compiled[*slot] && !self.patience.skips(*slot))
            .take(self.window)
            .collect();
        for slot in &window {
            self.compiled[*slot] = true;
        }
        self.window = self.later_window;
        window
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tune::{BatchEntry, GroupId, Patience};

    /// A batch of `len` candidates of one ordered group with `patience`.
    fn batch(len: usize, patience: Option<Patience>) -> Batch {
        let group = GroupId::new(0);
        Batch {
            entries: (0..len)
                .map(|index| BatchEntry {
                    index,
                    groups: vec![group],
                    priority: Some(0),
                })
                .collect(),
            patience: patience
                .map(|patience| (group, patience))
                .into_iter()
                .collect(),
        }
    }

    fn patience(min_measured: usize, max_misses: usize) -> Patience {
        Patience {
            min_measured,
            max_misses,
        }
    }

    /// What a pass that measures `times` (microseconds, by slot) plans: the slots it measures and
    /// the windows it compiles, in order.
    fn walk(batch: &Batch, window: usize, times: &[u64]) -> (Vec<usize>, Vec<Vec<usize>>) {
        walk_windows(batch, windows(window, window), times)
    }

    fn walk_windows(
        batch: &Batch,
        windows: CompileWindows,
        times: &[u64],
    ) -> (Vec<usize>, Vec<Vec<usize>>) {
        let mut planner = FirstPassPlanner::new(batch, windows);
        let mut measured = Vec::new();
        let mut windows = Vec::new();
        while let Some(step) = planner.plan_next() {
            if !step.compile.is_empty() {
                windows.push(step.compile);
            }
            planner.record(step.slot, Duration::from_micros(times[step.slot]));
            measured.push(step.slot);
        }
        (measured, windows)
    }

    fn windows(first_count: usize, later_count: usize) -> CompileWindows {
        CompileWindows {
            first_count,
            later_count,
        }
    }

    const TIMES: [u64; 10] = [100, 90, 95, 92, 99, 98, 97, 96, 50, 40];

    #[test]
    fn a_patient_batch_compiles_a_window_at_a_time() {
        let (_, windows) = walk(&batch(10, Some(patience(1, 100))), 4, &TIMES);

        assert_eq!(
            windows,
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]
        );
    }

    #[test]
    fn a_batch_compiles_its_first_window_then_the_rest_at_once() {
        let (measured, compiled) = walk_windows(&batch(10, None), windows(4, 10), &TIMES);

        assert_eq!(measured, (0..10).collect::<Vec<_>>());
        assert_eq!(
            compiled,
            vec![vec![0, 1, 2, 3], (4..10).collect::<Vec<_>>()]
        );
    }

    #[test]
    fn the_measured_candidates_do_not_depend_on_the_window() {
        let batch = batch(10, Some(patience(2, 3)));

        let (one, _) = walk(&batch, 1, &TIMES);
        let (wide, _) = walk(&batch, 64, &TIMES);

        // 90 leads, then 95, 92 and 99 miss in a row.
        assert_eq!(one, vec![0, 1, 2, 3, 4]);
        assert_eq!(wide, one);
    }

    #[test]
    fn a_pass_that_stops_mid_window_leaves_the_rest_compiled_unmeasured() {
        let (measured, windows) = walk(&batch(10, Some(patience(2, 3))), 8, &TIMES);

        assert_eq!(measured, vec![0, 1, 2, 3, 4]);
        assert_eq!(windows, vec![(0..8).collect::<Vec<_>>()]);
    }

    #[test]
    fn the_short_circuit_waits_for_the_minimum_measured() {
        let mut planner = FirstPassPlanner::new(&batch(10, Some(patience(3, 100))), windows(4, 4));

        for _ in 0..2 {
            let step = planner.plan_next().expect("the group should not stop yet");
            planner.record(step.slot, Duration::from_micros(1));
            assert!(!planner.may_short_circuit());
        }
        let step = planner.plan_next().expect("the group should not stop yet");
        planner.record(step.slot, Duration::from_micros(1));

        assert!(planner.may_short_circuit());
    }

    #[test]
    fn without_patience_the_first_measured_may_short_circuit() {
        let mut planner = FirstPassPlanner::new(&batch(10, None), windows(4, 10));
        assert!(!planner.may_short_circuit());

        let step = planner
            .plan_next()
            .expect("a batch without patience measures everything");
        planner.record(step.slot, Duration::from_micros(1));

        assert!(planner.may_short_circuit());
    }
}
