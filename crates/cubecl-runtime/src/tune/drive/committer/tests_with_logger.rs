//! Tests that build a [`TuneLogger`] from the runtime configuration, a process-wide global.

use super::*;
use alloc::vec;

#[test]
fn a_recorded_bench_logs_its_step_and_keeps_its_result() {
    let failure = ResolvedBench {
        index: 0,
        step: Some(("k0".into(), Duration::from_micros(2))),
        result: AutotuneResult::from_error(AutotuneError::Skip { name: "k0".into() }),
    };
    let mut logger = TuneLogger::new(&crate::config::Logger::new(), true);
    let mut results = vec![AutotuneResult::from_error(AutotuneError::NotMeasured {
        name: "k0".into(),
    })];

    record_benches(&mut logger, &mut results, vec![failure]);

    assert!(matches!(
        results[0].outcome,
        Err(AutotuneError::Skip { .. })
    ));
    assert_eq!(
        logger
            .context()
            .expect("a recorded tune should keep its context")
            .events
            .len(),
        1
    );
}
