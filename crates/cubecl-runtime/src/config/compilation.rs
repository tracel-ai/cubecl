#[cfg(std_io)]
use super::cache::CacheConfig;
use super::logger::{BinaryLogLevel, LoggerConfig};

/// Configuration for compilation settings in CubeCL.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompilationConfig {
    /// Logger configuration for compilation logs, using binary log levels.
    #[serde(default)]
    pub logger: LoggerConfig<CompilationLogLevel>,
    /// Cache location for storing compiled kernels.
    #[serde(default)]
    #[cfg(std_io)]
    pub cache: Option<CacheConfig>,
}

/// Type alias for the log level used in compilation logging.
pub type CompilationLogLevel = BinaryLogLevel;
