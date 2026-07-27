use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Display;
use core::time::Duration;

pub use crate::profile::{Instant, TimingMethod};

#[cfg(feature = "std")]
pub use crate::profile::ProfileDuration;

/// Results of a benchmark run.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(new, Debug, Clone)]
pub struct BenchmarkDurations {
    /// How these durations were measured.
    pub timing_method: TimingMethod,
    /// All durations of the run, in the order they were benchmarked
    pub durations: Vec<Duration>,
}

impl BenchmarkDurations {
    /// Construct from a list of durations.
    pub fn from_durations(timing_method: TimingMethod, durations: Vec<Duration>) -> Self {
        Self {
            timing_method,
            durations,
        }
    }

    /// Returns a tuple of durations: (min, max, median)
    fn min_max_median_durations(&self) -> (Duration, Duration, Duration) {
        let mut sorted = self.durations.clone();
        sorted.sort();
        let min = *sorted.first().unwrap();
        let max = *sorted.last().unwrap();
        let median = *sorted.get(sorted.len() / 2).unwrap();
        (min, max, median)
    }

    /// Returns the median duration among all durations
    pub(crate) fn mean_duration(&self) -> Duration {
        self.durations.iter().sum::<Duration>() / self.durations.len() as u32
    }

    /// Returns the variance durations for the durations
    pub(crate) fn variance_duration(&self, mean: Duration) -> Duration {
        self.durations
            .iter()
            .map(|duration| {
                let tmp = duration.as_secs_f64() - mean.as_secs_f64();
                Duration::from_secs_f64(tmp * tmp)
            })
            .sum::<Duration>()
            / self.durations.len() as u32
    }
}

impl Display for BenchmarkDurations {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let computed = BenchmarkComputations::new(self);
        let BenchmarkComputations {
            mean,
            median,
            variance,
            min,
            max,
        } = computed;
        let num_sample = self.durations.len();
        let timing_method = self.timing_method;

        f.write_str(
            format!(
                "
―――――――― Result ―――――――――
  Timing      {timing_method}
  Samples     {num_sample}
  Mean        {mean:.3?}
  Variance    {variance:.3?}
  Median      {median:.3?}
  Min         {min:.3?}
  Max         {max:.3?}
―――――――――――――――――――――――――"
            )
            .as_str(),
        )
    }
}

/// Computed values from benchmark durations.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize, PartialEq, Eq)
)]
#[derive(Debug, Default, Clone)]
pub struct BenchmarkComputations {
    /// Mean of all the durations.
    pub mean: Duration,
    /// Median of all the durations.
    pub median: Duration,
    /// Variance of all the durations.
    pub variance: Duration,
    /// Minimum duration amongst all durations.
    pub min: Duration,
    /// Maximum duration amongst all durations.
    pub max: Duration,
}

impl BenchmarkComputations {
    /// Compute duration values and return a `BenchmarkComputations` struct
    pub fn new(durations: &BenchmarkDurations) -> Self {
        let mean = durations.mean_duration();
        let (min, max, median) = durations.min_max_median_durations();
        Self {
            mean,
            median,
            min,
            max,
            variance: durations.variance_duration(mean),
        }
    }

    /// Returns the score of the current benchmark.
    pub fn score(&self) -> u64 {
        // How much optimism we have regarding the benchmark.
        //
        // The higher the value, the more we prioritize the fastest run regardless of variation.
        const ALPHA: f64 = 0.8;

        let min_ns = self.min.as_nanos() as f64;
        let median_ns = self.median.as_nanos() as f64;
        let variance_ns = self.variance.as_nanos() as f64;
        let mean_ns = self.mean.as_nanos() as f64;

        // The base score is based on the fastest run and the median duration.
        let base_score = (min_ns * ALPHA) + (median_ns * (1.0 - ALPHA));

        // If the standard deviation is high relative to the mean,
        // we inflate the score (making it less desirable).
        let std_dev = num_traits::Float::sqrt(variance_ns);

        // Lower is better
        let coefficient_of_variation = 1.0
            + (std_dev
                / (
                    // The `1.0` is only for numerical stability with small numbers.
                    // Since we work with nanos, this is negligible.
                    1.0 + mean_ns
                ));

        // Return score (Lower is better)
        (base_score * coefficient_of_variation) as u64
    }
}

/// Benchmark trait.
pub trait Benchmark {
    /// Benchmark input arguments.
    type Input: Clone;
    /// The benchmark output.
    type Output;

    /// Prepare the benchmark, run anything that is essential for the benchmark, but shouldn't
    /// count as included in the duration.
    ///
    /// # Notes
    ///
    /// This should not include warmup, the benchmark will be run at least one time without
    /// measuring the execution time.
    fn prepare(&self) -> Self::Input;

    /// Execute the benchmark and returns the logical output of the task executed.
    ///
    /// It is important to return the output since otherwise deadcode optimization might optimize
    /// away code that should be benchmarked.
    fn execute(&self, input: Self::Input) -> Result<Self::Output, String>;

    /// Number of samples per run required to have a statistical significance.
    fn num_samples(&self) -> usize {
        const DEFAULT: usize = 15;
        #[cfg(feature = "std")]
        {
            std::env::var("BENCH_NUM_SAMPLES")
                .map(|val| str::parse::<usize>(&val).unwrap_or(DEFAULT))
                .unwrap_or(DEFAULT)
        }

        #[cfg(not(feature = "std"))]
        {
            DEFAULT
        }
    }

    /// Name of the benchmark, should be short and it should match the name
    /// defined in the crate Cargo.toml
    fn name(&self) -> String;

    /// The options passed to the benchmark.
    fn options(&self) -> Option<String> {
        None
    }

    /// Shapes dimensions
    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![]
    }

    /// Wait for all queued work to complete, surfacing any error left on the stream.
    ///
    /// On asynchronous backends, kernel launches are fire-and-forget: compilation and
    /// validation errors recorded at launch time, as well as GPU execution faults, may
    /// only become observable at synchronization. Implementations must return these as
    /// `Err` — e.g. `block_on(client.sync()).map_err(|err| format!("{err}"))` — rather
    /// than panic, so a failing benchmark fails its own [`Benchmark::run`] instead of
    /// aborting the process.
    fn sync(&self) -> Result<(), String>;

    /// Start measuring the computation duration.
    #[cfg(feature = "std")]
    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.profile_full(args)
    }

    /// Start measuring the computation duration. Use the full duration irregardless of whether
    /// device duration is available or not.
    #[cfg(feature = "std")]
    fn profile_full(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        // Surfaces faults left queued by previous work before the timer starts.
        self.sync()?;
        let start_time = Instant::now();
        let out = self.execute(args)?;
        // Surfaces queued launch errors (compilation/validation) and execution
        // faults belonging to this workload.
        self.sync()?;
        core::mem::drop(out);
        Ok(ProfileDuration::new_system_time(start_time, Instant::now()))
    }

    /// Run the benchmark a number of times.
    #[allow(unused_variables)]
    fn run(&self, timing_method: TimingMethod) -> Result<BenchmarkDurations, String> {
        #[cfg(not(feature = "std"))]
        {
            Err(String::from(
                "Running a benchmark is not supported in a no-std environment",
            ))
        }

        #[cfg(feature = "std")]
        {
            let execute = |args: &Self::Input| -> Result<crate::profile::ProfileTicks, String> {
                let profile = match timing_method {
                    TimingMethod::System => self.profile_full(args.clone()),
                    TimingMethod::Device => self.profile(args.clone()),
                }?;
                Ok(crate::future::block_on(profile.resolve()))
            };
            let args = self.prepare();

            let collect = || -> Result<Vec<Duration>, String> {
                // Triggers JIT-compilation and performs a warmup.
                //
                // We are using 5 iterations, where the first one probably triggers the
                // JIT-compilation and it is then followed by 4 warmup executions.
                //
                // Errors are propagated: JIT compilation happens here, so queued
                // compilation/launch errors and GPU faults surface at a warmup
                // sync. Swallowing them would report timings for a strategy that
                // never ran correctly.
                for _ in 0..5 {
                    execute(&args)?;
                }

                // Real execution.
                let mut durations = Vec::with_capacity(self.num_samples());
                for _ in 0..self.num_samples() {
                    durations.push(execute(&args)?.duration());
                }
                Ok(durations)
            };

            let result = collect();

            // Drain any failure still queued on the stream, regardless of outcome.
            // Device-timing overrides of `profile()` bypass the trait's `sync()`
            // entirely and `ProfileDuration::resolve()` has no error channel, so a
            // fault from the final samples could otherwise go unreported — or worse,
            // surface at the NEXT benchmark's leading sync and be misattributed.
            let drained = self.sync();

            // The in-run error wins; a drain error only surfaces when the run was
            // otherwise clean.
            let durations = result?;
            drained?;

            Ok(BenchmarkDurations {
                timing_method,
                durations,
            })
        }
    }
}

/// Result of a benchmark run, with metadata
#[derive(Clone)]
pub struct BenchmarkResult {
    /// Individual raw results of the run
    pub raw: BenchmarkDurations,
    /// Computed values for the run
    pub computed: BenchmarkComputations,
    /// Git commit hash of the commit in which the run occurred
    pub git_hash: String,
    /// Name of the benchmark
    pub name: String,
    /// Options passed to the benchmark
    pub options: Option<String>,
    /// Shape dimensions
    pub shapes: Vec<Vec<usize>>,
    /// Time just before the run
    pub timestamp: u128,
}

impl Display for BenchmarkResult {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            format!(
                "
        Timestamp: {}
        Git Hash: {}
        Benchmarking - {}{}
        ",
                self.timestamp, self.git_hash, self.name, self.raw
            )
            .as_str(),
        )
    }
}

#[cfg(feature = "std")]
/// Runs the given benchmark on the device and prints result and information.
pub fn run_benchmark<BM>(benchmark: BM) -> Result<BenchmarkResult, String>
where
    BM: Benchmark,
{
    use std::string::ToString;

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap().trim().to_string();
    let durations = benchmark.run(TimingMethod::System)?;

    Ok(BenchmarkResult {
        raw: durations.clone(),
        computed: BenchmarkComputations::new(&durations),
        git_hash,
        name: benchmark.name(),
        options: benchmark.options(),
        shapes: benchmark.shapes(),
        timestamp,
    })
}

#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use super::*;
    use alloc::vec;

    #[test_log::test]
    fn test_min_max_median_durations_even_number_of_samples() {
        let durations = BenchmarkDurations {
            timing_method: TimingMethod::System,
            durations: vec![
                Duration::new(10, 0),
                Duration::new(20, 0),
                Duration::new(30, 0),
                Duration::new(40, 0),
                Duration::new(50, 0),
            ],
        };
        let (min, max, median) = durations.min_max_median_durations();
        assert_eq!(min, Duration::from_secs(10));
        assert_eq!(max, Duration::from_secs(50));
        assert_eq!(median, Duration::from_secs(30));
    }

    #[test_log::test]
    fn test_min_max_median_durations_odd_number_of_samples() {
        let durations = BenchmarkDurations {
            timing_method: TimingMethod::System,
            durations: vec![
                Duration::new(18, 5),
                Duration::new(20, 0),
                Duration::new(30, 0),
                Duration::new(40, 0),
            ],
        };
        let (min, max, median) = durations.min_max_median_durations();
        assert_eq!(min, Duration::from_nanos(18000000005_u64));
        assert_eq!(max, Duration::from_secs(40));
        assert_eq!(median, Duration::from_secs(30));
    }

    #[test_log::test]
    fn test_mean_duration() {
        let durations = BenchmarkDurations {
            timing_method: TimingMethod::System,
            durations: vec![
                Duration::new(10, 0),
                Duration::new(20, 0),
                Duration::new(30, 0),
                Duration::new(40, 0),
            ],
        };
        let mean = durations.mean_duration();
        assert_eq!(mean, Duration::from_secs(25));
    }

    #[test_log::test]
    fn test_variance_duration() {
        let durations = BenchmarkDurations {
            timing_method: TimingMethod::System,
            durations: vec![
                Duration::new(10, 0),
                Duration::new(20, 0),
                Duration::new(30, 0),
                Duration::new(40, 0),
                Duration::new(50, 0),
            ],
        };
        let mean = durations.mean_duration();
        let variance = durations.variance_duration(mean);
        assert_eq!(variance, Duration::from_secs(200));
    }

    #[cfg(feature = "std")]
    mod fallible_sync {
        use super::super::*;
        use core::cell::Cell;

        /// A benchmark whose `sync` fails on a chosen call, counting calls so
        /// tests can target the leading sync, the post-execute sync, or the
        /// trailing drain in `run()`.
        struct MockBench {
            fail_sync_on_call: Option<usize>,
            sync_calls: Cell<usize>,
            samples: usize,
        }

        impl MockBench {
            fn new(fail_sync_on_call: Option<usize>) -> Self {
                Self {
                    fail_sync_on_call,
                    sync_calls: Cell::new(0),
                    samples: 3,
                }
            }
        }

        impl Benchmark for MockBench {
            type Input = ();
            type Output = ();

            fn prepare(&self) -> Self::Input {}

            fn execute(&self, _input: Self::Input) -> Result<Self::Output, String> {
                Ok(())
            }

            fn num_samples(&self) -> usize {
                self.samples
            }

            fn name(&self) -> String {
                String::from("mock")
            }

            fn sync(&self) -> Result<(), String> {
                let call = self.sync_calls.get();
                self.sync_calls.set(call + 1);
                match self.fail_sync_on_call {
                    Some(fail_on) if call == fail_on => Err(String::from("queued fault")),
                    _ => Ok(()),
                }
            }
        }

        #[test_log::test]
        fn clean_run_returns_num_samples_durations() {
            let bench = MockBench::new(None);
            let durations = bench.run(TimingMethod::System).unwrap();
            assert_eq!(durations.durations.len(), bench.num_samples());
        }

        #[test_log::test]
        fn sync_error_during_warmup_fails_the_run() {
            // Call 1 is the post-execute sync of the first warmup iteration,
            // where a queued JIT compilation error would surface.
            let bench = MockBench::new(Some(1));
            let result = bench.run(TimingMethod::System);
            assert_eq!(result.unwrap_err(), "queued fault");
        }

        #[test_log::test]
        fn sync_error_in_trailing_drain_fails_an_otherwise_clean_run() {
            // (5 warmups + 3 samples) * 2 syncs in profile_full = 16 calls,
            // so call 16 is the trailing drain in `run()`.
            let bench = MockBench::new(Some(16));
            let result = bench.run(TimingMethod::System);
            assert_eq!(bench.sync_calls.get(), 17);
            assert_eq!(result.unwrap_err(), "queued fault");
        }

        #[test_log::test]
        fn loop_error_wins_over_drain_error() {
            struct BothFail {
                sync_calls: Cell<usize>,
            }
            impl Benchmark for BothFail {
                type Input = ();
                type Output = ();
                fn prepare(&self) -> Self::Input {}
                fn execute(&self, _input: Self::Input) -> Result<Self::Output, String> {
                    Err(String::from("execute failed"))
                }
                fn num_samples(&self) -> usize {
                    1
                }
                fn name(&self) -> String {
                    String::from("both-fail")
                }
                fn sync(&self) -> Result<(), String> {
                    let call = self.sync_calls.get();
                    self.sync_calls.set(call + 1);
                    match call {
                        // Call 0 is the leading sync of the first warmup; call 1
                        // is the trailing drain, reached because execute errored.
                        0 => Ok(()),
                        _ => Err(String::from("drain failed")),
                    }
                }
            }
            let bench = BothFail {
                sync_calls: Cell::new(0),
            };
            let result = bench.run(TimingMethod::System);
            assert_eq!(bench.sync_calls.get(), 2);
            assert_eq!(result.unwrap_err(), "execute failed");
        }
    }
}
