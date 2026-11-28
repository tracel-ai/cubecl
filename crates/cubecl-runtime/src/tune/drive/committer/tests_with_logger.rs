//! Tests that build a [`TuneLogger`] from the runtime configuration, a process-wide global.

use super::*;
use alloc::vec;

fn logger() -> TuneLogger {
    TuneLogger::new(&crate::config::Logger::new(), true)
}

fn skipped(step: Option<(String, Duration)>) -> ResolvedBench {
    ResolvedBench {
        index: 0,
        step,
        result: AutotuneResult::from_error(AutotuneError::Skip { name: "k0".into() }),
    }
}

fn unmeasured() -> Vec<AutotuneResult> {
    vec![AutotuneResult::from_error(AutotuneError::NotMeasured {
        name: "k0".into(),
    })]
}

fn logged_steps(logger: &TuneLogger) -> usize {
    logger
        .as_context()
        .expect("a recorded tune should keep its context")
        .events
        .len()
}

#[test]
fn a_recorded_bench_keeps_its_result() {
    let mut results = unmeasured();

    record_bench(&mut logger(), &mut results, skipped(None));

    assert!(matches!(
        results[0].outcome,
        Err(AutotuneError::Skip { .. })
    ));
}

#[test]
fn a_recorded_bench_logs_its_step() {
    let mut logger = logger();

    record_bench(
        &mut logger,
        &mut unmeasured(),
        skipped(Some(("k0".into(), Duration::from_micros(2)))),
    );

    assert_eq!(logged_steps(&logger), 1);
}

#[test]
fn a_bench_with_no_step_logs_none() {
    let mut logger = logger();

    record_bench(&mut logger, &mut unmeasured(), skipped(None));

    assert_eq!(logged_steps(&logger), 0);
}

#[cfg(persistence)]
mod commit {
    use super::*;
    use crate::tune::cache::record::TuneRecorder;
    use crate::tune::cache::table::TuneTable;
    use crate::tune::cache::tracker::TuneTracker;
    use crate::tune::drive::pending_unmarker::PendingUnmarker;
    use crate::tune::drive::persister::TunePersister;
    use crate::tune::drive::settler::TuneSettler;
    use alloc::string::ToString;
    use alloc::sync::Arc;
    use cubecl_common::profile::TimingMethod;
    use cubecl_environment::sync::Mutex;

    /// A tracker with no store, the key tracked pending.
    fn tracker() -> Arc<Mutex<TuneTracker<String>>> {
        let mut tracker = TuneTracker::with_store(
            None,
            TuneTable {
                namespace: "committer-tests".into(),
                generation: cubecl_environment::environment::generation(),
            },
        );
        tracker.track_pending("key".to_string());
        Arc::new(Mutex::new(tracker))
    }

    /// A committer of the tune of the key over two candidates, settling into `tracker`.
    fn committer(tracker: &Arc<Mutex<TuneTracker<String>>>) -> TuneCommitter<String> {
        let table = tracker.lock().table().clone();
        let unmarker = PendingUnmarker::new(tracker.clone(), &tracker.lock(), "key".to_string());
        let recorder = TuneRecorder::new(&table, &"key".to_string(), "checksum");
        let settler = TuneSettler::new(
            tracker.clone(),
            Arc::new(Mutex::new(crate::config::Logger::new())),
            None,
            unmarker,
            TunePersister::new("checksum".into(), None, None, recorder),
        );
        TuneCommitter::new(
            "key".to_string(),
            ["k0", "k1"].into_iter(),
            logger(),
            settler,
        )
    }

    fn measured_at(index: usize, micros: u64) -> ResolvedBench {
        ResolvedBench {
            index,
            step: None,
            result: AutotuneResult::from_outcome(AutotuneOutcome::new(
                alloc::format!("k{index}"),
                index,
                BenchmarkComputations::new(&BenchmarkDurations::from_durations(
                    TimingMethod::System,
                    vec![Duration::from_micros(micros)],
                )),
            )),
        }
    }

    #[test]
    fn a_strategy_s_decision_wins_over_the_best_score() {
        let tracker = tracker();
        let mut committer = committer(&tracker);
        committer.record_bench(measured_at(0, 10));
        committer.record_bench(measured_at(1, 50));

        assert!(matches!(
            committer.commit(Vec::new(), Some(1)),
            TuneAnswer::Hit { winner_index: 1 }
        ));
    }

    #[test]
    fn with_no_strategy_winner_the_best_score_wins() {
        let tracker = tracker();
        let mut committer = committer(&tracker);
        committer.record_bench(measured_at(0, 50));
        committer.record_bench(measured_at(1, 10));

        assert!(matches!(
            committer.commit(Vec::new(), None),
            TuneAnswer::Hit { winner_index: 1 }
        ));
    }
}
