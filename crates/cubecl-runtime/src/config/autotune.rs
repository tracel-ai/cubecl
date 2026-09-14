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

/// Autotune benchmark settings.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct BenchConfig {
    /// Samples collected before adaptive elimination begins.
    pub min_samples: usize,

    /// Maximum samples per candidate.
    pub max_samples: usize,

    /// Samples below the time limit required to short circuit.
    pub short_circuit_samples: usize,

    /// Maximum slowdown relative to the best candidate before elimination.
    pub speed_factor: f64,

    /// Enables adaptive sampling on supported targets.
    pub adaptive: bool,

    /// Warm-up runs per candidate.
    #[serde(default = "default_warmup_samples")]
    pub warmup_samples: usize,

    /// Target device time per browser tuning round, in milliseconds.
    #[serde(default = "default_browser_round_ms")]
    pub browser_round_ms: u64,

    /// Device-time budget per browser tuning key, in milliseconds. Zero is unlimited.
    #[serde(default = "default_browser_budget_ms")]
    pub browser_budget_ms: u64,
}

fn default_warmup_samples() -> usize {
    3
}

fn default_browser_round_ms() -> u64 {
    250
}

fn default_browser_budget_ms() -> u64 {
    5000
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            min_samples: 3,
            max_samples: 10,
            short_circuit_samples: 2,
            speed_factor: 1.5,
            adaptive: true,
            warmup_samples: default_warmup_samples(),
            browser_round_ms: default_browser_round_ms(),
            browser_budget_ms: default_browser_budget_ms(),
        }
    }
}

impl BenchConfig {
    /// Returns the normalized sample range.
    pub fn samples(&self) -> (usize, usize) {
        let min = self.min_samples.max(1);
        (min, self.max_samples.max(min))
    }

    /// Returns the normalized short-circuit sample count.
    pub fn short_circuit_samples(&self) -> usize {
        self.short_circuit_samples.max(1)
    }

    /// Returns the normalized elimination threshold.
    pub fn speed_factor(&self) -> f64 {
        self.speed_factor.max(1.0)
    }

    /// Returns the browser round duration.
    pub fn browser_round(&self) -> core::time::Duration {
        core::time::Duration::from_millis(self.browser_round_ms.max(1))
    }

    /// Returns the browser budget, or `None` when unlimited.
    pub fn browser_budget(&self) -> Option<core::time::Duration> {
        (self.browser_budget_ms > 0)
            .then(|| core::time::Duration::from_millis(self.browser_budget_ms))
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
