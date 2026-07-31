use super::logger::{LogLevel, LoggerConfig};

/// Configuration for autotuning in `CubeCL`.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AutotuneConfig {
    /// Logger configuration for autotune logs, using autotune-specific log levels.
    #[serde(default)]
    pub logger: LoggerConfig<AutotuneLogLevel>,

    /// Recorder configuration: where to write one [`AutotuneRecord`](crate::tune::AutotuneRecord)
    /// per tuning decision, as JSON, for a tool to read back.
    ///
    /// Independent of [`logger`](Self::logger), because the two answer different questions and both
    /// can be wanted at once: the logger's level says how much to tell a human, the recorder says
    /// where to put the machine-readable record.
    #[serde(default)]
    pub recorder: LoggerConfig<RecorderLevel>,

    /// Autotune level, controlling the intensity of autotuning.
    #[serde(default)]
    pub level: AutotuneLevel,

    /// Whether to disable the persistent cache of autotune results.
    ///
    /// The in-memory cache is unaffected: a key is still tuned only once per process.
    #[serde(default)]
    pub disable_cache: bool,

    /// Whether to disable the short circuit logic during autotuning.
    #[serde(default)]
    pub disable_short_circuit: bool,

    /// Sampling budget and elimination thresholds used while benchmarking candidates.
    #[serde(default)]
    pub bench: BenchConfig,
}

/// Controls how many samples autotune collects per candidate and when candidates are dropped.
///
/// Only [`max_samples`](Self::max_samples) and [`adaptive`](Self::adaptive) mean anything to the
/// fixed-count pass; the rest describe elimination, which only the adaptive scheduler performs.
/// Each field says so, because a knob that silently does nothing on the strategy actually running
/// is worse than no knob at all — and `adaptive` is native-only, so on wasm that is every run.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct BenchConfig {
    /// Samples every surviving candidate gets before any elimination happens.
    ///
    /// Adaptive only: the fixed-count pass eliminates nothing, so every candidate gets
    /// [`max_samples`](Self::max_samples) regardless.
    pub min_samples: usize,

    /// Upper bound on samples collected for a single candidate.
    ///
    /// Read by both strategies: the ceiling for the adaptive scheduler, and the flat count for
    /// the fixed pass, which has no elimination to spend a smaller budget on.
    pub max_samples: usize,

    /// Samples that must independently land under the time limit to short circuit.
    ///
    /// A short circuit decision is written to the persistent cache and reused on later runs, so
    /// it is confirmed rather than taken from a single possibly-lucky sample.
    ///
    /// Adaptive only: the fixed pass has the whole sample set in hand before it tests the limit,
    /// so it has nothing to confirm.
    pub short_circuit_samples: usize,

    /// How many times slower than the current best a candidate may be before elimination.
    ///
    /// Adaptive only. Values below `1.0` are read as `1.0`, which eliminates every candidate
    /// slower than the leader that the survivor floor allows.
    pub speed_factor: f64,

    /// Whether to use the adaptive round robin benchmark instead of a fixed sample count.
    ///
    /// Ignored on wasm, which cannot resolve samples between rounds and so always takes the
    /// fixed-count pass.
    pub adaptive: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            min_samples: 3,
            max_samples: 10,
            short_circuit_samples: 2,
            speed_factor: 1.5,
            adaptive: true,
        }
    }
}

/// Every knob is read through an accessor that clamps it into its usable range, so a config file
/// can hold a nonsensical value without any single call site having to remember the repair.
impl BenchConfig {
    /// The sample budget, clamped so the range is always usable.
    pub fn samples(&self) -> (usize, usize) {
        let min = self.min_samples.max(1);
        (min, self.max_samples.max(min))
    }

    /// How many samples must land under the limit, clamped so a short circuit always needs one.
    pub fn short_circuit_samples(&self) -> usize {
        self.short_circuit_samples.max(1)
    }

    /// The elimination threshold, clamped so it can never sit below the leader's own time.
    pub fn speed_factor(&self) -> f64 {
        self.speed_factor.max(1.0)
    }
}

/// Log levels for autotune logging in `CubeCL`.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneLogLevel {
    /// Autotune logging is disabled.
    #[serde(rename = "disabled")]
    Disabled,

    /// Minimal autotune information is logged such as the fastest kernel selected and a few
    /// statistics (default).
    #[default]
    #[serde(rename = "minimal")]
    Minimal,

    /// Full autotune details are logged.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for AutotuneLogLevel {}

/// The recorder's (absent) verbosity.
///
/// A record is one fixed schema, which is the whole point: a tool reads it back and depends on its
/// shape, so there is no "how much" to choose. The recorder is simply on when it has a sink
/// (see [`AutotuneConfig::recording_enabled`]); this type exists only so it can reuse
/// [`LoggerConfig`]'s sinks.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecorderLevel;

impl LogLevel for RecorderLevel {}

impl AutotuneConfig {
    /// Whether tuning decisions are being recorded, i.e. the recorder has somewhere to write.
    pub fn recording_enabled(&self) -> bool {
        #[cfg(std_io)]
        let has_file = self.recorder.file.is_some();
        #[cfg(not(std_io))]
        let has_file = false;

        has_file || self.recorder.stdout || self.recorder.stderr
    }
}

/// Autotune levels controlling the intensity of autotuning.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneLevel {
    /// Minimal autotuning effort.
    #[serde(rename = "minimal")]
    Minimal,

    /// Balanced autotuning effort (default).
    #[default]
    #[serde(rename = "balanced")]
    Balanced,

    /// Increased autotuning effort.
    #[serde(rename = "extensive")]
    Extensive,

    /// Maximum autotuning effort.
    #[serde(rename = "full")]
    Full,
}
