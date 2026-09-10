use cubecl_runtime::throughput::{
    KernelConfig, ThroughputBenchmarker, ThroughputError, ThroughputValue,
};

/// The shapes a probe can be launched in, ranked against each other so that
/// only the winner is measured in full.
pub(super) struct ShapeSweep<S> {
    shapes: alloc::vec::Vec<S>,
}

impl<S: Copy> ShapeSweep<S> {
    pub(super) fn new(shapes: alloc::vec::Vec<S>) -> Self {
        Self { shapes }
    }

    /// What the device answers fastest, and the shape that answered it.
    ///
    /// Shapes are built one at a time: a memory probe's pool is a large fraction
    /// of what the device will allocate. A rate that is not finite did not run.
    pub(super) fn fastest(
        &self,
        build: impl Fn(S) -> KernelConfig,
    ) -> Result<(ThroughputValue, S), ThroughputError> {
        let (fastest, warmed) = match self.shapes.len() {
            0 => return Err(ThroughputError::NoTiming),
            1 => (self.shapes[0], None),
            _ => {
                let (fastest, iterations) = self.ranked(&build).ok_or(ThroughputError::NoTiming)?;

                (fastest, Some(iterations))
            }
        };

        let config = build(fastest);
        let value = match warmed {
            Some(iterations) => ThroughputBenchmarker::sample_at(&config, iterations),
            None => ThroughputBenchmarker::sample(config),
        };

        value
            .ops_per_s()
            .is_finite()
            .then_some((value, fastest))
            .ok_or(ThroughputError::NoTiming)
    }

    /// The shape that answers fastest over a ranking pass, and the count the
    /// device was warmed at. Every shape is timed at that same count, so they
    /// are ordered on what they do rather than on which was warmed.
    fn ranked(&self, build: impl Fn(S) -> KernelConfig) -> Option<(S, usize)> {
        let mut warmed = None;
        let mut fastest: Option<(f64, S)> = None;

        for shape in &self.shapes {
            let config = build(*shape);
            let start = *warmed.get_or_insert_with(|| ThroughputBenchmarker::warm(&config));
            let ranked = ThroughputBenchmarker::rank(&config, start);
            let rate = ranked.value.ops_per_s();

            // The next shape starts from what this one settled on, so a sweep
            // whose shapes drift in cost keeps its settling launch near target.
            warmed = Some(ranked.iterations);

            if rate.is_finite() && fastest.is_none_or(|(best, _)| rate > best) {
                fastest = Some((rate, *shape));
            }
        }

        Some((fastest?.1, warmed?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reported value is a full measurement of the winner, not the
    /// ranking pass that found it.
    #[test]
    fn the_shape_that_ranks_fastest_is_the_one_measured() {
        // A shape here is its rate: the faster it is, the less time a pass takes.
        let build = |rate: u64| KernelConfig {
            sample: alloc::boxed::Box::new(move |iterations| {
                core::time::Duration::from_nanos(iterations as u64 * 1000 / rate)
            }),
            ops_count: 1,
            min_iterations: 1,
        };

        let (value, fastest) = ShapeSweep::new(alloc::vec![1, 4, 2])
            .fastest(build)
            .expect("a shape ran");

        assert_eq!(fastest, 4);
        assert!(value.ops_per_s().is_finite());
    }
}
