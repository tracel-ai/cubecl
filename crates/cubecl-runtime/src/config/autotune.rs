#[cfg(std_io)]
use super::cache::CacheConfig;
use super::logger::{LogLevel, LoggerConfig};

/// Configuration for autotuning in `CubeCL`.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AutotuneConfig {
    /// Logger configuration for autotune logs, using autotune-specific log levels.
    #[serde(default)]
    pub logger: LoggerConfig<AutotuneLogLevel>,

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

    /// Emits telemetry data as structured JSON.
    #[serde(rename = "telemetry")]
    Telemetry,
}

impl LogLevel for AutotuneLogLevel {}

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
