use crate::{
    config::CubeClRuntimeConfig,
    throughput::{ThroughputCache, ThroughputError, ThroughputKey, ThroughputValue},
};
use alloc::boxed::Box;
use alloc::sync::Arc;
use cubecl_common::profile::{Duration, Instant};
use cubecl_environment::config::RuntimeConfig;
use cubecl_environment::sync::Mutex;

type Cache = Arc<Mutex<ThroughputCache>>;

/// Wall clock a warmup may spend growing its iteration count. Generous next to
/// the tens of milliseconds a converging one needs, and the only thing bounding
/// a probe whose timer is too coarse to ever reach the target.
const WARMUP_BUDGET: Duration = Duration::from_secs(2);

/// Wall clock a plateau must hold across before it is accepted, sized against a
/// clock transition, which takes hundreds of milliseconds.
const PLATEAU_FLOOR: Duration = Duration::from_millis(250);

/// Wall clock one shape's peak is sampled over. A sample count would price a
/// probe filling the duration target forty times one whose pass is microseconds.
const SAMPLE_BUDGET: Duration = Duration::from_millis(200);

/// Samples that may go by without improving before the peak is accepted. Also
/// the floor on samples, since the first always improves on infinity and only
/// the ones after it can go stale.
const SAMPLE_PATIENCE: usize = 12;

/// Wall clock one launch of a shape is grown or cut toward, so that a sweep
/// times every shape over the same span and none of them wears a larger share
/// of the fixed cost of a launch than the rest.
const TARGET_DURATION: Duration = Duration::from_millis(20);

/// Samples a ranking pass keeps the fastest of, per shape.
///
/// A count and not a wall clock, so every shape of a sweep is ordered on the
/// same number of draws. Under a time budget a shape whose pass happens to be
/// short draws more of them, and the lowest of more draws is lower, which
/// decides a close pair on how long its passes are rather than on their rate.
const RANK_SAMPLES: usize = 3;

/// Configuration and payload for a benchmarkable compute kernel.
pub struct KernelConfig {
    /// A closure that executes the kernel for the given number of iterations and returns the duration.
    pub sample: Box<dyn Fn(usize) -> Duration>,
    /// The number of operations processed in one iteration.
    pub ops_count: usize,
    /// Iterations a launch must carry however quickly they turn out to run,
    /// which a duration target cannot express.
    pub min_iterations: usize,
}

/// What a ranking pass found: how fast the shape answered, and the iteration
/// count it settled on for the next shape to start from.
pub struct Ranked {
    /// The shape's rate, measured briefly enough to order it and not to report
    /// it.
    pub value: ThroughputValue,
    /// What one launch of this shape was timed over.
    pub iterations: usize,
}

/// A marker for measuring throughput of compute kernels.
pub struct ThroughputBenchmarker {
    cache: Cache,
    cache_enabled: bool,
}

impl ThroughputBenchmarker {
    /// Creates a new `ThroughputBenchmarker` with the given cache.
    pub fn new(cache: Cache) -> Self {
        let cache_enabled = !CubeClRuntimeConfig::get().throughput.disable_cache;
        Self {
            cache,
            cache_enabled,
        }
    }

    /// The value for `key`, measured by `probe` unless the cache holds one.
    ///
    /// # Errors
    ///
    /// Whatever `probe` reports. Only a measurement is cached.
    pub fn measure(
        &mut self,
        key: ThroughputKey,
        probe: impl FnOnce() -> Result<ThroughputValue, ThroughputError>,
    ) -> Result<ThroughputValue, ThroughputError> {
        if self.cache_enabled
            && let Some(cached_value) = self.cache.lock().get(&key)
        {
            return Ok(*cached_value);
        }

        let value = probe()?;

        if self.cache_enabled {
            self.cache.lock().insert(key, value);
        }

        Ok(value)
    }

    /// Warm one shape of a kernel up to its plateau, then keep its fastest sample.
    pub fn sample(kernel_config: KernelConfig) -> ThroughputValue {
        let sample = kernel_config.sample;

        let iterations = Self::warmup(kernel_config.min_iterations, WARMUP_BUDGET, &sample);
        let duration =
            Self::sample_peak_duration(iterations, &sample, SAMPLE_BUDGET, SAMPLE_PATIENCE);

        ThroughputValue {
            ops_count: kernel_config.ops_count,
            duration,
        }
    }

    /// Warms the device on one shape and reports the iteration count a sample of
    /// it should carry, for [`rank`](Self::rank) to reuse across the rest.
    ///
    /// A warmup is most of what timing a shape costs, and it is the device it
    /// warms rather than the shape, so a sweep pays for one.
    pub fn warm(kernel_config: &KernelConfig) -> usize {
        Self::warmup(
            kernel_config.min_iterations,
            WARMUP_BUDGET,
            &kernel_config.sample,
        )
    }

    /// Keeps the fastest sample of a shape the device is already warm on, at the
    /// iteration count [`warm`](Self::warm) settled.
    ///
    /// A warmup is what a measurement mostly costs, so a sweep that has already
    /// paid one must not pay it again for the shape it picked.
    pub fn sample_at(kernel_config: &KernelConfig, iterations: usize) -> ThroughputValue {
        let iterations = iterations.max(kernel_config.min_iterations).max(1);
        let duration = Self::sample_peak_duration(
            iterations,
            &kernel_config.sample,
            SAMPLE_BUDGET,
            SAMPLE_PATIENCE,
        );

        ThroughputValue {
            ops_count: kernel_config.ops_count,
            duration,
        }
    }

    /// Times one shape briefly, to order it against the others rather than to
    /// report its peak, and reports the iteration count the next shape should
    /// start from.
    ///
    /// Two launches are spent before timing. The shape a sweep warmed on has
    /// its buffers, its pages and its compiled kernel settled and the rest do
    /// not, and a first launch carries all of that, enough that shapes would
    /// rank on which of them the sweep had already touched. The second is what
    /// this shape's own iteration count is settled from, so a shape retiring
    /// ten times more per pass than the one before it is still timed over
    /// [`TARGET_DURATION`] and carries the same share of a launch's fixed cost.
    ///
    /// `iterations` is where that starts, so passing what the previous shape
    /// settled on keeps those two launches near the target too.
    pub fn rank(kernel_config: &KernelConfig, iterations: usize) -> Ranked {
        let settling = iterations.max(kernel_config.min_iterations).max(1);
        // What a shape costs the first time is what it costs to compile and to
        // fault in, not what a pass of it costs, so the first launch is spent
        // and a second one is what the count is settled from. Reading the first
        // ranks a shape the sweep has not compiled before below one it has.
        let _ = (kernel_config.sample)(settling);
        let took = (kernel_config.sample)(settling);

        let iterations = Self::retarget(settling, took)
            .max(kernel_config.min_iterations)
            .max(1);

        let mut fastest = Duration::MAX;
        for _ in 0..RANK_SAMPLES {
            fastest = fastest.min((kernel_config.sample)(iterations));
        }

        let duration = fastest / iterations as u32;

        Ranked {
            value: ThroughputValue {
                ops_count: kernel_config.ops_count,
                duration,
            },
            iterations,
        }
    }

    /// The count that would have taken [`TARGET_DURATION`], given what
    /// `iterations` of them took.
    ///
    /// A timer reading zero says nothing to scale by, so the count stands.
    fn retarget(iterations: usize, took: Duration) -> usize {
        let took = took.as_secs_f64();

        if took <= 0.0 {
            return iterations;
        }

        let scaled = iterations as f64 * (TARGET_DURATION.as_secs_f64() / took);

        if scaled.is_finite() {
            (scaled as usize).max(1)
        } else {
            iterations
        }
    }

    /// Warms up the device by running the kernel multiple times
    /// and estimating the number of iterations needed to reach a stable duration.
    ///
    /// Never returns fewer than `min_iterations`, which the kernel needs to be
    /// measuring what it claims rather than merely to be timed accurately.
    ///
    /// `budget` bounds the growing, not the sampling: a timer too coarse to
    /// ever reach the target reports the same reading at every count, which
    /// asks for a larger one each round without converging.
    fn warmup(
        min_iterations: usize,
        budget: Duration,
        sample: impl Fn(usize) -> Duration,
    ) -> usize {
        const MAX_WARMUP: usize = 50;
        const MAX_ITERATIONS: usize = 1 << 24;
        // A timer reading zero says nothing about the pass, so doubling against
        // it converges on nothing and stops early. An iteration is a real launch
        // for the probe that measures launches, which pays for every one.
        const MAX_BLIND_ITERATIONS: usize = 1 << 10;
        const PLATEAU_TOL: f64 = 0.03;
        const PATIENCE: usize = 3;
        let target_ms = TARGET_DURATION.as_secs_f64() * 1000.0;

        let mut best = f64::INFINITY;
        let mut stable = 0;
        let mut iterations = min_iterations.max(1);
        let start = Instant::now();
        let mut plateau_start = start;

        for _ in 0..MAX_WARMUP {
            let duration = sample(iterations).as_secs_f64() * 1000.0;
            if duration < target_ms {
                let (extra_iters, ceiling) = if duration > 1e-6 {
                    let duration_per_iter = duration / iterations as f64;
                    (
                        ((target_ms - duration) / duration_per_iter).ceil() as usize,
                        MAX_ITERATIONS,
                    )
                } else {
                    (iterations, MAX_BLIND_ITERATIONS)
                };

                let ceiling = ceiling.max(min_iterations);
                if iterations >= ceiling || start.elapsed() >= budget {
                    break;
                }
                iterations = (iterations + extra_iters.max(1)).min(ceiling);
                best = f64::INFINITY;
                stable = 0;
                continue;
            }

            let duration_per_iter = duration / iterations as f64;
            if duration_per_iter < best * (1.0 - PLATEAU_TOL) {
                best = duration_per_iter;
                stable = 0;
                // Growth clears `best`, so the window restarts at the settled count.
                plateau_start = Instant::now();
            } else {
                best = best.min(duration_per_iter);
                stable += 1;
                if stable >= PATIENCE && plateau_start.elapsed() >= PLATEAU_FLOOR {
                    break;
                }
            }
        }

        iterations
    }

    /// Sample the peak throughput of the kernel by running it multiple times
    /// and measuring the duration of each iteration.
    fn sample_peak_duration(
        iterations: usize,
        sample_once: impl Fn(usize) -> Duration,
        budget: Duration,
        patience: usize,
    ) -> Duration {
        debug_assert!(
            iterations > 0,
            "iterations must be positive to avoid division by zero"
        );

        const MAX_SAMPLES: usize = 200;
        const REL_TOL: f64 = 0.01;

        let mut best = f64::INFINITY;
        let mut stale = 0;
        // Wall clock, not the sum of what the samples report: a probe whose
        // timer reads zero would otherwise never spend any of the budget.
        let start = Instant::now();

        for _ in 0..MAX_SAMPLES {
            let s = sample_once(iterations).as_secs_f64();
            if s < best * (1.0 - REL_TOL) {
                best = s;
                stale = 0;
            } else {
                best = best.min(s);
                stale += 1;
            }
            if stale >= patience || start.elapsed() >= budget {
                break;
            }
        }

        Duration::from_secs_f64(best / iterations as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    fn spin(duration: Duration) {
        let start = Instant::now();
        while start.elapsed() < duration {}
    }

    /// A device that takes as long as it reports, at whatever rate it is asked for.
    fn timed_device(per_iter_nanos: impl Fn() -> u64) -> impl Fn(usize) -> Duration {
        move |iterations| {
            let duration = Duration::from_nanos(per_iter_nanos() * iterations as u64);
            spin(duration);

            duration
        }
    }

    /// One iteration of the launch probe is a real launch, so a device whose
    /// timer reads zero must not be answered by doubling toward the ceiling the
    /// duration target drives.
    #[test]
    fn a_timer_reading_zero_does_not_climb_to_the_duration_ceiling() {
        let iterations = ThroughputBenchmarker::warmup(1, WARMUP_BUDGET, |_| Duration::ZERO);

        assert!(iterations <= 1 << 10, "climbed to {iterations}");
    }

    /// The passes a probe needs to be measuring what it claims are not the
    /// timer's to give away.
    #[test]
    fn a_blind_timer_never_cuts_below_the_passes_a_probe_needs() {
        let needed = 1 << 20;

        assert!(ThroughputBenchmarker::warmup(needed, WARMUP_BUDGET, |_| Duration::ZERO) >= needed);
    }

    /// A timer too coarse to resolve the target reports the same reading at
    /// every count, so each round divides it by a larger number and asks for a
    /// larger one still. The iteration ceiling alone stops that only after the
    /// rounds it takes to reach it, which the probe pays for in real launches.
    #[test]
    fn a_timer_that_never_reaches_the_target_stops_growing_on_the_budget() {
        let iterations = ThroughputBenchmarker::warmup(1, Duration::from_millis(12), |_| {
            spin(Duration::from_millis(5));

            Duration::from_millis(1)
        });

        assert!(iterations < 1 << 20, "climbed to {iterations}");
    }

    #[test]
    fn a_timer_reading_zero_still_stops_sampling() {
        let calls = Cell::new(0);
        let _ = ThroughputBenchmarker::sample_peak_duration(
            1,
            |_| {
                calls.set(calls.get() + 1);
                Duration::ZERO
            },
            SAMPLE_BUDGET,
            SAMPLE_PATIENCE,
        );

        assert!(calls.get() < 200, "ran {} samples", calls.get());
    }

    /// Passes at one iteration count sit microseconds apart, so a plateau of
    /// them is evidence about a moment rather than about the device.
    #[test]
    fn a_clock_that_lifts_inside_the_floor_does_not_release_the_warmup() {
        let lift = Duration::from_millis(100);
        let clock = Instant::now();
        let lifts_once = timed_device(move || if clock.elapsed() < lift { 3000 } else { 1000 });

        let start = Instant::now();
        ThroughputBenchmarker::warmup(1, WARMUP_BUDGET, lifts_once);

        assert!(
            start.elapsed() >= lift + PLATEAU_FLOOR,
            "released after {:?}",
            start.elapsed()
        );
    }

    /// A probe runs on first use of every key, so the quiet device is the cost
    /// every one of them pays.
    #[test]
    fn a_steady_device_pays_the_floor_and_nothing_more() {
        let passes = Cell::new(0);
        let steady = timed_device(|| {
            passes.set(passes.get() + 1);
            1000
        });

        let start = Instant::now();
        ThroughputBenchmarker::warmup(1, WARMUP_BUDGET, steady);

        assert!(
            start.elapsed() >= PLATEAU_FLOOR,
            "left after {:?}",
            start.elapsed()
        );
        assert!(passes.get() <= 20, "ran {} passes", passes.get());
    }

    /// Contention that outlasts the whole warmup is measured, not rejected: a
    /// device that is genuinely slow answers the same however long it is held.
    #[test]
    fn a_device_slow_for_the_whole_measurement_reports_its_slow_rate() {
        let value = ThroughputBenchmarker::sample(KernelConfig {
            sample: Box::new(timed_device(|| 3000)),
            ops_count: 1,
            min_iterations: 1,
        });

        assert!(
            (Duration::from_nanos(2900)..Duration::from_nanos(3100)).contains(&value.duration),
            "kept {:?}",
            value.duration
        );
    }

    /// A sweep ranks shapes to order them, not to report them, so it must
    /// separate a slow shape from a fast one without paying for a measurement
    /// of each.
    #[test]
    fn ranking_orders_shapes_for_less_than_one_measurement() {
        let config = |per_iter_nanos: u64| KernelConfig {
            sample: Box::new(timed_device(move || per_iter_nanos)),
            ops_count: 1,
            min_iterations: 1,
        };
        let (fast, slow) = (config(1000), config(3000));

        let iterations = ThroughputBenchmarker::warm(&fast);
        let start = Instant::now();
        let fast_rate = ThroughputBenchmarker::rank(&fast, iterations)
            .value
            .ops_per_s();
        let slow_rate = ThroughputBenchmarker::rank(&slow, iterations)
            .value
            .ops_per_s();

        assert!(fast_rate > slow_rate, "{fast_rate} against {slow_rate}");
        assert!(
            start.elapsed() < PLATEAU_FLOOR + SAMPLE_BUDGET,
            "ranked two shapes in {:?}",
            start.elapsed()
        );
    }

    /// Shapes of one sweep differ in what a pass costs, and timing a fast one
    /// over a tenth of the span would make it wear ten times the share of a
    /// launch's fixed cost. Each settles its own count toward the target.
    #[test]
    fn ranking_times_each_shape_over_the_same_span() {
        let config = |per_iter_nanos: u64| KernelConfig {
            sample: Box::new(move |iterations| {
                Duration::from_nanos(per_iter_nanos * iterations as u64)
            }),
            ops_count: 1,
            min_iterations: 1,
        };

        let slow = ThroughputBenchmarker::rank(&config(1000), 1000);
        let fast = ThroughputBenchmarker::rank(&config(100), 1000);

        assert_eq!(slow.iterations, 20_000);
        assert_eq!(fast.iterations, 200_000);
    }

    /// A shape's first launch carries compiling and faulting it in, which is
    /// not what a pass of it costs. Settling the count from that ranks a shape
    /// the sweep has not seen before below one it has.
    #[test]
    fn ranking_settles_its_count_after_the_first_launch() {
        let launches = Cell::new(0);
        let config = KernelConfig {
            sample: Box::new(move |iterations| {
                launches.set(launches.get() + 1);
                let per_iter = if launches.get() == 1 { 100_000 } else { 1_000 };

                Duration::from_nanos(per_iter * iterations as u64)
            }),
            ops_count: 1,
            min_iterations: 1,
        };

        let ranked = ThroughputBenchmarker::rank(&config, 1_000);

        assert_eq!(ranked.iterations, 20_000);
    }

    /// Every shape of a sweep is timed over the same work, or a shape that
    /// happened to be warmed at a different count would rank on that instead.
    #[test]
    fn ranking_never_carries_fewer_passes_than_a_shape_needs() {
        let needed = 64;
        let config = KernelConfig {
            sample: Box::new(|iterations| Duration::from_nanos(iterations as u64)),
            ops_count: 1,
            min_iterations: needed,
        };

        let ranked = ThroughputBenchmarker::rank(&config, 1);

        assert_eq!(ranked.value.duration, Duration::from_nanos(1));
        assert!(ranked.iterations >= needed);
    }

    /// A working timer still drives the count to the duration target.
    #[test]
    fn a_pass_far_under_the_target_grows_until_it_reaches_it() {
        let iterations = ThroughputBenchmarker::warmup(1, WARMUP_BUDGET, |iterations| {
            Duration::from_micros(iterations as u64)
        });

        assert_eq!(iterations, 20_000);
    }
}
