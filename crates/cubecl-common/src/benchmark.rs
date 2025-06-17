use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Display;

use core::pin::Pin;
use core::time::Duration;

/// Result from profiling between two measurements. This can either be a duration or a future that resolves to a duration.
pub enum ProfileDuration {
    /// Client profile contains a full duration.
    Full(Duration),
    /// Client profile measures the device duration, and requires to be resolved.
    DeviceDuration(Pin<Box<dyn Future<Output = Duration> + Send + 'static>>),
}

impl core::fmt::Debug for ProfileDuration {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProfileDuration::Full(duration) => write!(f, "Full({:?})", duration),
            ProfileDuration::DeviceDuration(_) => write!(f, "DeviceDuration"),
        }
    }
}

impl ProfileDuration {
    /// Create a new `ProfileDuration` straight from a duration.
    pub fn from_duration(duration: Duration) -> Self {
        ProfileDuration::Full(duration)
    }

    /// Create a new `ProfileDuration` from a future that resolves to a duration.
    pub fn from_future(future: impl Future<Output = Duration> + Send + 'static) -> Self {
        ProfileDuration::DeviceDuration(Box::pin(future))
    }

    /// The method used to measure the execution time.
    pub fn timing_method(&self) -> TimingMethod {
        match self {
            ProfileDuration::Full(_) => TimingMethod::System,
            ProfileDuration::DeviceDuration(_) => TimingMethod::Device,
        }
    }

    /// Resolve the actual duration of the profile, possibly by waiting for the future to complete.
    pub async fn resolve(self) -> Duration {
        match self {
            ProfileDuration::Full(duration) => duration,
            ProfileDuration::DeviceDuration(future) => future.await,
        }
    }
}

/// How a benchmark's execution times are measured.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TimingMethod {
    /// Time measurements come from full timing of execution + sync
    /// calls.
    System,
    /// Time measurements come from hardware reported timestamps
    /// coming from a sync call.
    Device,
}

impl Display for TimingMethod {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TimingMethod::System => f.write_str("system"),
            TimingMethod::Device => f.write_str("device"),
        }
    }
}

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

    /// Construct from a list of profiles.
    pub async fn from_profiles(profiles: Vec<ProfileDuration>) -> Self {
        let mut durations = Vec::new();
        let mut types = Vec::new();

        for profile in profiles {
            types.push(profile.timing_method());
            durations.push(profile.resolve().await);
        }

        let timing_method = *types.first().expect("need at least 1 profile");
        if types.iter().any(|&t| t != timing_method) {
            panic!("all profiles must use the same timing method");
        }

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
    fn execute(&self, input: Self::Input) -> Self::Output;

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
    fn profile(&self, args: Self::Input) -> ProfileDuration {
        self.profile_full(args)
    }

    /// Start measuring the computation duration. Use the full duration irregardless of whether
    /// device duration is available or not.
    #[cfg(feature = "std")]
    fn profile_full(&self, args: Self::Input) -> ProfileDuration {
        self.sync();
        let start_time = std::time::Instant::now();
        let out = self.execute(args);
        self.sync();
        core::mem::drop(out);
        ProfileDuration::from_duration(start_time.elapsed())
    }

    /// Run the benchmark a number of times.
    #[allow(unused_variables)]
    fn run(&self, timing_method: TimingMethod) -> BenchmarkDurations {
        #[cfg(not(feature = "std"))]
        panic!("Attempting to run benchmark in a no-std environment");

        #[cfg(feature = "std")]
        {
            let execute = |args: &Self::Input| {
                let profile = match timing_method {
                    TimingMethod::System => self.profile_full(args.clone()),
                    TimingMethod::Device => self.profile(args.clone()),
                };
                crate::future::block_on(profile.resolve())
            };
            let args = self.prepare();

            // Warmup
            for _ in 0..3 {
                let _duration = execute(&args);
            }
            std::thread::sleep(Duration::from_secs(1));

            // Real execution.
            let mut durations = Vec::with_capacity(self.num_samples());
            for _ in 0..self.num_samples() {
                durations.push(execute(&args));
            }

            BenchmarkDurations {
                timing_method,
                durations,
            }
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
    let durations = benchmark.run(TimingMethod::System);

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
