#[cfg(std_io)]
use super::cache::CacheConfig;
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

    /// Cache location for storing autotune results.
    #[serde(default)]
    #[cfg(std_io)]
    pub cache: CacheConfig,

    /// Whether to disable the persistent cache of autotune results.
    ///
    /// The in-memory cache is unaffected: a key is still tuned only once per process.
    #[serde(default)]
    pub disable_cache: bool,

    /// Whether to disable the short circuit logic during autotuning.
    #[serde(default)]
    pub disable_short_circuit: bool,
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
