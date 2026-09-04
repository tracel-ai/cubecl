use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Display;
use core::time::Duration;

pub use crate::profile::{Instant, TimingMethod};

use crate::work::Work;

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

/// Launches the warmup makes however long they take: one to compile, and the
/// four a kernel slower than the budget gets today.
#[cfg(feature = "std")]
const MIN_WARMUP_RUNS: usize = 5;

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

    /// Wall clock the warmup holds the device for before sampling starts.
    ///
    /// A device answers from idle clocks and takes hundreds of milliseconds to
    /// reach the ones it sustains, which a warmup counted in launches gives a
    /// microsecond kernel no way to reach. 500 ms is where that stops showing:
    /// a 4070 Ti SUPER under Vulkan reports every reduce row at its full rate
    /// from there, and 12 of 36 rows below it. A sweep of thousands of rows can
    /// buy the wall clock back with `BENCH_WARMUP_MS`, and pay in accuracy.
    fn warmup_budget(&self) -> Duration {
        const DEFAULT_MS: u64 = 500;
        #[cfg(feature = "std")]
        {
            Duration::from_millis(
                std::env::var("BENCH_WARMUP_MS")
                    .map(|val| str::parse::<u64>(&val).unwrap_or(DEFAULT_MS))
                    .unwrap_or(DEFAULT_MS),
            )
        }

        #[cfg(not(feature = "std"))]
        {
            Duration::from_millis(DEFAULT_MS)
        }
    }

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

    /// The work one execution performs, for scoring the run against measured
    /// peak throughput. `None` when the benchmark has no such figure to report.
    /// Coarse by necessity: this crate cannot name a throughput key, so
    /// `calculate_bounds` (in `cubecl-runtime`) is what turns this single
    /// figure into per-resource bounds, and a caller wanting the achieved
    /// rate against each of those scores them at the client layer, which
    /// does have keys.
    fn work(&self) -> Option<Work> {
        None
    }

    /// Wait for computation to complete.
    fn sync(&self);

    /// Start measuring the computation duration.
    #[cfg(feature = "std")]
    fn profile(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.profile_full(args)
    }

    /// Start measuring the computation duration. Use the full duration irregardless of whether
    /// device duration is available or not.
    #[cfg(feature = "std")]
    fn profile_full(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.sync();
        let start_time = Instant::now();
        let out = self.execute(args)?;
        self.sync();
        core::mem::drop(out);
        Ok(ProfileDuration::new_system_time(start_time, Instant::now()))
    }

    /// Run the benchmark a number of times.
    #[allow(unused_variables)]
    fn run(&self, timing_method: TimingMethod) -> Result<BenchmarkDurations, String> {
        #[cfg(not(feature = "std"))]
        panic!("Attempting to run benchmark in a no-std environment");

        #[cfg(feature = "std")]
        {
            let execute = |args: &Self::Input| {
                let profile: Result<ProfileDuration, String> = match timing_method {
                    TimingMethod::System => self.profile_full(args.clone()),
                    TimingMethod::Device => self.profile(args.clone()),
                };
                let profile = match profile {
                    Ok(val) => val,
                    Err(err) => return Err(err),
                };
                Ok(cubecl_environment::future::block_on(profile.resolve()))
            };
            let args = self.prepare();

            // Compiles on the first launch, then holds the device until it
            // answers at the clocks a sampled run will see.
            let budget = self.warmup_budget();
            let warmup = Instant::now();
            let mut warmups = 0;
            while warmups < MIN_WARMUP_RUNS || warmup.elapsed() < budget {
                let warmed: Result<crate::profile::ProfileTicks, _> = execute(&args);
                if warmed.is_err() {
                    break;
                }
                warmups += 1;
            }

            // Real execution.
            let mut durations = Vec::with_capacity(self.num_samples());
            for _ in 0..self.num_samples() {
                match execute(&args) {
                    Ok(val) => durations.push(val.duration()),
                    Err(err) => {
                        return Err(err);
                    }
                }
            }

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
    use core::cell::Cell;

    /// A device that answers in whatever time it is given, counting the
    /// launches it was asked for.
    struct TimedBench {
        per_execution: Duration,
        executions: Cell<usize>,
    }

    impl TimedBench {
        fn new(per_execution: Duration) -> Self {
            Self {
                per_execution,
                executions: Cell::new(0),
            }
        }
    }

    impl Benchmark for TimedBench {
        type Input = ();
        type Output = ();

        fn prepare(&self) -> Self::Input {}

        fn execute(&self, _input: Self::Input) -> Result<Self::Output, String> {
            self.executions.set(self.executions.get() + 1);
            let start = Instant::now();
            while start.elapsed() < self.per_execution {}

            Ok(())
        }

        fn num_samples(&self) -> usize {
            1
        }

        fn name(&self) -> String {
            "timed".into()
        }

        fn sync(&self) {}
    }

    /// The warmup is there to lift a device off its idle clocks, which takes
    /// hundreds of milliseconds whatever the kernel costs. Counted in launches
    /// it would leave a fast one sampled cold.
    #[test_log::test]
    fn a_kernel_the_launch_count_cannot_warm_is_held_for_the_budget() {
        let bench = TimedBench::new(Duration::ZERO);
        let start = Instant::now();

        bench.run(TimingMethod::System).expect("the bench runs");

        assert!(
            start.elapsed() >= bench.warmup_budget(),
            "warmed {:?}",
            start.elapsed()
        );
        assert!(bench.executions.get() > MIN_WARMUP_RUNS);
    }

    /// A budget that added launches to a kernel already filling it would make
    /// every slow row pay a warmup it does not need.
    #[test_log::test]
    fn a_kernel_that_fills_the_budget_pays_no_extra_launch() {
        let mut bench = TimedBench::new(Duration::ZERO);
        bench.per_execution = bench.warmup_budget() / 4;

        let durations = bench.run(TimingMethod::System).expect("the bench runs");

        assert_eq!(
            bench.executions.get(),
            MIN_WARMUP_RUNS + durations.durations.len()
        );
    }

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
}
