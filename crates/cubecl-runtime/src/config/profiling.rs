use super::logger::{LogLevel, LoggerConfig};

/// Configuration for profiling settings in CubeCL.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProfilingConfig {
    /// Logger configuration for profiling logs, using profiling-specific log levels.
    #[serde(default)]
    pub logger: LoggerConfig<ProfilingLogLevel>,
}

/// Log levels for profiling in CubeCL.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProfilingLogLevel {
    /// Profiling logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Basic profiling information is logged.
    #[serde(rename = "basic")]
    Basic,

    /// Medium level of profiling details is logged.
    #[serde(rename = "medium")]
    Medium,

    /// Full profiling details are logged.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for ProfilingLogLevel {}
