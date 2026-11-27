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
