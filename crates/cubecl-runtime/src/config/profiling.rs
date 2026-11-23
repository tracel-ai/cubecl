use super::{LogLevel, LoggerConfig};

/// What the profiling log reports.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProfilingConfig {
    /// Where the profiling log goes, and how much it says.
    #[serde(default)]
    pub logger: LoggerConfig<ProfilingLogLevel>,
}

/// How much the profiling log says.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProfilingLogLevel {
    /// No profiling log is written.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Only the kernels that run are logged without timing.
    #[serde(rename = "minimal")]
    Minimal,

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
