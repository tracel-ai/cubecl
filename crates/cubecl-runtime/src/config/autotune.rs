use super::logger::{LogLevel, LoggerConfig};

/// Configuration for autotuning in CubeCL.
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
    #[cfg(feature = "std")]
    pub cache: AutotuneCache,
}

/// Cache location options for autotune results.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[cfg(feature = "std")]
pub enum AutotuneCache {
    /// Stores cache in the current working directory.
    Local,

    /// Stores cache in the project's `target` directory (default).
    #[default]
    Target,

    /// Stores cache in the system's local configuration directory.
    Global,

    /// Stores cache in a user-specified file path.
    File(std::path::PathBuf),
}

/// Log levels for autotune logging in CubeCL.
#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneLogLevel {
    /// Autotune logging is disabled.
    #[serde(rename = "disabled")]
    Disabled,

    /// Minimal autotune information is logged (default).
    #[default]
    #[serde(rename = "minimal")]
    Minimal,

    /// Full autotune details are logged.
    #[serde(rename = "full")]
    Full,
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
    Exhaustive,
}

#[cfg(feature = "std")]
impl AutotuneCache {
    /// Returns the root directory for the autotune cache.
    ///
    /// Determines the cache location based on the [AutotuneCache] variant:
    pub fn root(&self) -> std::path::PathBuf {
        match self {
            AutotuneCache::Local => std::env::current_dir().unwrap(),
            AutotuneCache::Target => {
                let dir_original = std::env::current_dir().unwrap();
                let mut dir = dir_original.clone();

                // Search for Cargo.toml in parent directories to locate project root.
                loop {
                    if let Ok(true) = std::fs::exists(dir.join("Cargo.toml")) {
                        return dir.join("target");
                    }

                    if !dir.pop() {
                        break;
                    }
                }

                dir_original.join("target")
            }
            AutotuneCache::Global => dirs::config_local_dir().unwrap(),
            AutotuneCache::File(path_buf) => path_buf.clone(),
        }
    }
}
