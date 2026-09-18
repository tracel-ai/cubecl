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
    ///
    /// # Errors
    ///
    /// Whatever `build` reports, and [`ThroughputError::NoTiming`] where no
    /// shape ran.
    pub(super) async fn fastest<F, Fut>(
        &self,
        build: F,
    ) -> Result<(ThroughputValue, S), ThroughputError>
    where
        F: Fn(S) -> Fut,
        Fut: core::future::Future<Output = Result<KernelConfig, ThroughputError>>,
    {
        let (fastest, warmed) = match self.shapes.len() {
            0 => return Err(ThroughputError::NoTiming),
            1 => (self.shapes[0], None),
            _ => {
                let (fastest, iterations) = self.ranked(&build).await?;

                (fastest, Some(iterations))
            }
        };

        let config = build(fastest).await?;
        let value = match warmed {
            Some(iterations) => ThroughputBenchmarker::sample_at(&config, iterations).await,
            None => ThroughputBenchmarker::sample(config).await,
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
    async fn ranked<F, Fut>(&self, build: &F) -> Result<(S, usize), ThroughputError>
    where
        F: Fn(S) -> Fut,
        Fut: core::future::Future<Output = Result<KernelConfig, ThroughputError>>,
    {
        let mut warmed = None;
        let mut fastest: Option<(f64, S)> = None;

        for shape in &self.shapes {
            let config = build(*shape).await?;
            let start = match warmed {
                Some(iterations) => iterations,
                None => ThroughputBenchmarker::warm(&config).await,
            };
            let ranked = ThroughputBenchmarker::rank(&config, start).await;
            let rate = ranked.value.ops_per_s();

            // The next shape starts from what this one settled on, so a sweep
            // whose shapes drift in cost keeps its settling launch near target.
            warmed = Some(ranked.iterations);

            if rate.is_finite() && fastest.is_none_or(|(best, _)| rate > best) {
                fastest = Some((rate, *shape));
            }
        }

        fastest
            .zip(warmed)
            .map(|((_, shape), iterations)| (shape, iterations))
            .ok_or(ThroughputError::NoTiming)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kernel whose pass costs what the shape's rate says it does.
    fn config_at(rate: u64) -> KernelConfig {
        KernelConfig {
            sample: alloc::boxed::Box::new(
                move |iterations| -> cubecl_environment::future::DynFut<_> {
                    Box::pin(async move {
                        core::time::Duration::from_nanos(iterations as u64 * 1000 / rate)
                    })
                },
            ),
            ops_count: 1,
            min_iterations: 1,
        }
    }

    /// The reported value is a full measurement of the winner, not the
    /// ranking pass that found it.
    #[test]
    fn the_shape_that_ranks_fastest_is_the_one_measured() {
        // A shape here is its rate: the faster it is, the less time a pass takes.
        let (value, fastest) = cubecl_environment::future::block_on(
            ShapeSweep::new(alloc::vec![1, 4, 2])
                .fastest(|rate| async move { Ok(config_at(rate)) }),
        )
        .expect("a shape ran");

        assert_eq!(fastest, 4);
        assert!(value.ops_per_s().is_finite());
    }

    /// A shape that cannot be built stops the sweep with its own error, rather
    /// than being outranked by the shapes that could.
    #[test]
    fn a_shape_that_cannot_be_built_fails_the_sweep() {
        for shapes in [alloc::vec![1], alloc::vec![1, 2]] {
            let failed = cubecl_environment::future::block_on(ShapeSweep::new(shapes).fastest(
                |_: u64| async move { Err::<KernelConfig, _>(ThroughputError::Allocation) },
            ));

            assert_eq!(failed.err(), Some(ThroughputError::Allocation));
        }
    }
}
