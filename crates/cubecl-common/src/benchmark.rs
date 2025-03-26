use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Display;

use super::stub::Duration;

/// How a benchmark's execution times are measured.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Default, Clone, Copy)]
pub enum TimingMethod {
    /// Time measurements come from full timing of execution + sync
    /// calls.
    #[default]
    Full,
    /// Time measurements come from hardware reported timestamps
    /// coming from a sync call.
    DeviceOnly,
}

impl Display for TimingMethod {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TimingMethod::Full => f.write_str("full"),
            TimingMethod::DeviceOnly => f.write_str("device_only"),
        }
    }
}

/// Error that can occurred when collecting timestamps from a device.
#[derive(Debug)]
pub enum TimestampsError {
    /// Collecting timestamps is disabled, make sure to enable it.
    Disabled,
    /// Collecting timestamps isn't available.
    Unavailable,
    /// An unknown error occurred while collecting timestamps.
    Unknown(String),
}

/// Result when collecting timestamps.
pub type TimestampsResult = Result<Duration, TimestampsError>;

/// Results of a benchmark run.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(new, Debug, Default, Clone)]
pub struct BenchmarkDurations {
    /// How these durations were measured.
    pub timing_method: TimingMethod,
    /// All durations of the run, in the order they were benchmarked
    pub durations: Vec<Duration>,
}

impl BenchmarkDurations {
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
    /// Compute duration values and return a BenchmarkComputations struct
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
}

/// Benchmark trait.
pub trait Benchmark {
    /// Benchmark arguments.
    type Args: Clone;

    /// Prepare the benchmark, run anything that is essential for the benchmark, but shouldn't
    /// count as included in the duration.
    ///
    /// # Notes
    ///
    /// This should not include warmup, the benchmark will be run at least one time without
    /// measuring the execution time.
    fn prepare(&self) -> Self::Args;
    /// Execute the benchmark and returns the time it took to complete.
    fn execute(&self, args: Self::Args);
    /// Number of samples per run required to have a statistical significance.
    fn num_samples(&self) -> usize {
        const DEFAULT: usize = 10;

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
    /// Wait for computation to complete.
    fn sync(&self);

    /// Wait for computation to complete and return hardware reported
    /// computation duration.
    fn sync_elapsed(&self) -> TimestampsResult {
        Err(TimestampsError::Unavailable)
    }

    /// Run the benchmark a number of times.
    #[allow(unused_variables)]
    fn run(&self, timing_method: TimingMethod) -> BenchmarkDurations {
        #[cfg(not(feature = "std"))]
        panic!("Attempting to run benchmark in a no-std environment");
        #[cfg(feature = "std")]
        {
            // Warmup
            let args = self.prepare();

            for _ in 0..self.num_samples() {
                self.execute(args.clone());
            }

            match timing_method {
                TimingMethod::Full => self.sync(),
                TimingMethod::DeviceOnly => {
                    let _ = self.sync_elapsed();
                }
            }
            std::thread::sleep(Duration::from_secs(1));

            let mut durations = Vec::with_capacity(self.num_samples());

            for _ in 0..self.num_samples() {
                match timing_method {
                    TimingMethod::Full => durations.push(self.run_one_full(args.clone())),
                    TimingMethod::DeviceOnly => {
                        durations.push(self.run_one_device_only(args.clone()))
                    }
                }
            }

            BenchmarkDurations {
                timing_method,
                durations,
            }
        }
    }
    #[cfg(feature = "std")]
    /// Collect one sample directly measuring the full execute + sync
    /// step.
    fn run_one_full(&self, args: Self::Args) -> Duration {
        let start = std::time::Instant::now();
        self.execute(args);
        self.sync();
        start.elapsed()
    }
    /// Collect one sample using timing measurements reported by the
    /// device.
    #[cfg(feature = "std")]
    fn run_one_device_only(&self, args: Self::Args) -> Duration {
        let start = std::time::Instant::now();

        self.execute(args);

        let result = self.sync_elapsed();

        match result {
            Ok(time) => time,
            Err(err) => match err {
                TimestampsError::Disabled => {
                    panic!(
                        "Collecting timestamps is deactivated, make sure to enable it before running the benchmark"
                    );
                }
                TimestampsError::Unavailable => start.elapsed(),
                TimestampsError::Unknown(err) => {
                    panic!(
                        "An unknown error occurred while collecting the timestamps when benchmarking: {err}"
                    );
                }
            },
        }
    }
}

/// Result of a benchmark run, with metadata
#[derive(Default, Clone)]
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
pub fn run_benchmark<BM>(benchmark: BM) -> BenchmarkResult
where
    BM: Benchmark,
{
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout).unwrap().trim().to_string();
    let durations = benchmark.run(TimingMethod::Full);

    BenchmarkResult {
        raw: durations.clone(),
        computed: BenchmarkComputations::new(&durations),
        git_hash,
        name: benchmark.name(),
        options: benchmark.options(),
        shapes: benchmark.shapes(),
        timestamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_min_max_median_durations_even_number_of_samples() {
        let durations = BenchmarkDurations {
            timing_method: TimingMethod::Full,
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

    #[test]
    fn test_min_max_median_durations_odd_number_of_samples() {
        let durations = BenchmarkDurations {
            timing_method: TimingMethod::Full,
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

    #[test]
    fn test_mean_duration() {
        let durations = BenchmarkDurations {
            timing_method: TimingMethod::Full,
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

    #[test]
    fn test_variance_duration() {
        let durations = BenchmarkDurations {
            timing_method: TimingMethod::Full,
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
