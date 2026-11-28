use cubecl_runtime::throughput::{ThroughputError, ThroughputValue};

/// How many times a probe runs, the fastest kept.
///
/// The first probe of a process runs on clocks that have not ramped, which reads it low, and
/// another process, a thermal cap or an unlucky schedule read it low too. Nothing reads it high,
/// so the fastest of a few is the closest to what the device can do.
pub(super) const PROBE_REPEAT_COUNT: usize = 5;

/// The fastest value `probe` answers over [`PROBE_REPEAT_COUNT`] runs.
///
/// # Errors
///
/// The error of the last run, where no run measured a value.
pub(super) fn fastest_of(
    mut probe: impl FnMut() -> Result<ThroughputValue, ThroughputError>,
) -> Result<ThroughputValue, ThroughputError> {
    (0..PROBE_REPEAT_COUNT)
        .map(|_| probe())
        .reduce(faster)
        .unwrap_or(Err(ThroughputError::NoTiming))
}

/// The faster of two answers: a value over an error, and the later error over the earlier.
fn faster(
    first: Result<ThroughputValue, ThroughputError>,
    second: Result<ThroughputValue, ThroughputError>,
) -> Result<ThroughputValue, ThroughputError> {
    match (first, second) {
        (Ok(first), Ok(second)) => Ok(core::cmp::max_by(first, second, by_rate)),
        (Ok(value), Err(_)) | (Err(_), Ok(value)) => Ok(value),
        (Err(_), Err(error)) => Err(error),
    }
}

/// Orders two values by the rate they measured.
fn by_rate(first: &ThroughputValue, second: &ThroughputValue) -> core::cmp::Ordering {
    first.ops_per_s().total_cmp(&second.ops_per_s())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::time::Duration;

    fn rate(ops_count: usize) -> ThroughputValue {
        ThroughputValue {
            ops_count,
            duration: Duration::from_secs(1),
            samples: None,
        }
    }

    #[test]
    fn the_fastest_run_is_kept() {
        let mut answers = vec![rate(3), rate(9), rate(5), rate(1), rate(7)].into_iter();

        let fastest = fastest_of(|| Ok(answers.next().expect("one answer a run")));

        assert_eq!(fastest, Ok(rate(9)));
    }

    #[test]
    fn a_value_is_kept_over_a_failed_run() {
        let mut runs = 0;

        let fastest = fastest_of(|| {
            runs += 1;
            match runs {
                1 => Ok(rate(4)),
                _ => Err(ThroughputError::Allocation),
            }
        });

        assert_eq!(fastest, Ok(rate(4)));
    }

    #[test]
    fn every_run_failing_answers_the_last_error() {
        let mut runs = 0;

        let fastest = fastest_of(|| {
            runs += 1;
            match runs {
                PROBE_REPEAT_COUNT => Err(ThroughputError::NoTiming),
                _ => Err(ThroughputError::Allocation),
            }
        });

        assert_eq!(fastest, Err(ThroughputError::NoTiming));
    }
}
