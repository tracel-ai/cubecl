use super::logger::{LogLevel, LoggerConfig};

/// Configuration for streaming settings in CubeCL.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StreamingConfig {
    /// Logger configuration for streaming logs, using binary log levels.
    #[serde(default)]
    pub logger: LoggerConfig<StreamingLogLevel>,
    /// The maximum number of streams to be used.
    #[serde(default = "default_max_streams")]
    pub max_streams: u8,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            logger: Default::default(),
            max_streams: default_max_streams(),
        }
    }
}

fn default_max_streams() -> u8 {
    1
}

/// Log levels for streaming in CubeCL.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum StreamingLogLevel {
    /// Compilation logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Basic streaming information is logged such as when streams are merged.
    #[serde(rename = "basic")]
    Basic,

    /// Full streaming details are logged.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for StreamingLogLevel {}
