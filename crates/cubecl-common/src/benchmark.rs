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
                Ok(crate::future::block_on(profile.resolve()))
            };
            let args = self.prepare();

            // Warmup
            for _ in 0..3 {
                let _duration: Result<crate::profile::ProfileTicks, _> = execute(&args);
            }
            std::thread::sleep(Duration::from_secs(1));

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
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
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

    #[test]
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

    #[test]
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

    #[test]
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
