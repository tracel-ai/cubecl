use super::{autotune::AutotuneConfig, compilation::CompilationConfig, profiling::ProfilingConfig};
use alloc::sync::Arc;

/// Static mutex holding the global configuration, initialized as `None`.
static CUBE_GLOBAL_CONFIG: spin::Mutex<Option<Arc<GlobalConfig>>> = spin::Mutex::new(None);

/// Represents the global configuration for CubeCL, combining profiling, autotuning, and compilation settings.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GlobalConfig {
    /// Configuration for profiling CubeCL operations.
    #[serde(default)]
    pub profiling: ProfilingConfig,

    /// Configuration for autotuning performance parameters.
    #[serde(default)]
    pub autotune: AutotuneConfig,

    /// Configuration for compilation settings.
    #[serde(default)]
    pub compilation: CompilationConfig,
}

impl GlobalConfig {
    /// Retrieves the current global configuration, loading it from the current directory if not set.
    ///
    /// If no configuration is set, it attempts to load one from `cubecl.toml` or `CubeCL.toml` in the
    /// current directory or its parents. If no file is found, a default configuration is used.
    pub fn get() -> Arc<Self> {
        let mut state = CUBE_GLOBAL_CONFIG.lock();
        if let None = state.as_ref() {
            #[cfg(feature = "std")]
            let config = Self::from_current_dir();
            #[cfg(feature = "std")]
            let config = config.override_from_env();
            #[cfg(not(feature = "std"))]
            let config = Self::default();
            *state = Some(Arc::new(config));
        }

        state.as_ref().cloned().unwrap()
    }

    /// Sets the global configuration to the provided value.
    ///
    /// # Panics
    /// Panics if the configuration has already been set or read, as it cannot be overridden.
    ///
    /// # Warning
    /// This method must be called at the start of the program, before any calls to `get`. Attempting
    /// to set the configuration after it has been initialized will cause a panic.
    pub fn set(config: Self) {
        let mut state = CUBE_GLOBAL_CONFIG.lock();
        if state.is_some() {
            panic!("Cannot set the global configuration multiple times.");
        }
        *state = Some(Arc::new(config));
    }

    #[cfg(feature = "std")]
    /// Overrides configuration fields based on environment variables.
    pub fn override_from_env(mut self) -> Self {
        use super::compilation::CompilationLogLevel;
        use crate::config::{
            autotune::{AutotuneLevel, AutotuneLogLevel},
            profiling::ProfilingLogLevel,
        };

        match std::env::var("CUBECL_DEBUG_LOG") {
            Ok(val) => match val.as_str() {
                "1" | "true" => {
                    self.compilation.logger.level = CompilationLogLevel::Full;
                }
                "0" | "false" => {
                    self.compilation.logger.level = CompilationLogLevel::Disabled;
                }
                file_path => {
                    self.compilation.logger.level = CompilationLogLevel::Full;
                    self.profiling.logger.level = ProfilingLogLevel::Basic;
                    self.autotune.logger.level = AutotuneLogLevel::Full;

                    self.compilation.logger.file = Some(file_path.into());
                    self.profiling.logger.file = Some(file_path.into());
                    self.autotune.logger.file = Some(file_path.into());
                }
            },
            Err(_) => {}
        };

        match std::env::var("CUBECL_DEBUG_OPTION") {
            Ok(val) => match val.as_str() {
                "debug" => {
                    self.compilation.logger.level = CompilationLogLevel::Full;
                    self.profiling.logger.level = ProfilingLogLevel::Medium;
                    self.autotune.logger.level = AutotuneLogLevel::Full;
                }
                "debug-full" => {
                    self.compilation.logger.level = CompilationLogLevel::Full;
                    self.profiling.logger.level = ProfilingLogLevel::Full;
                    self.autotune.logger.level = AutotuneLogLevel::Full;
                }
                "profile" => {
                    self.profiling.logger.level = ProfilingLogLevel::Basic;
                }
                "profile-medium" => {
                    self.profiling.logger.level = ProfilingLogLevel::Medium;
                }
                "profile-full" => {
                    self.profiling.logger.level = ProfilingLogLevel::Full;
                }
                _ => {}
            },
            Err(_) => {}
        };

        match std::env::var("CUBECL_AUTOTUNE_LEVEL") {
            Ok(val) => match val.as_str() {
                "minimal" | "0" => {
                    self.autotune.level = AutotuneLevel::Minimal;
                }
                "balanced" | "1" => {
                    self.autotune.level = AutotuneLevel::Balanced;
                }
                "extensive" | "2" => {
                    self.autotune.level = AutotuneLevel::Extensive;
                }
                "exhaustive" | "3" => {
                    self.autotune.level = AutotuneLevel::Exhaustive;
                }
                _ => {}
            },
            Err(_) => {}
        }

        self
    }

    // Loads configuration from `cubecl.toml` or `CubeCL.toml` in the current directory or its parents.
    //
    // Traverses up the directory tree until a valid configuration file is found or the root is reached.
    // Returns a default configuration if no file is found.
    #[cfg(feature = "std")]
    fn from_current_dir() -> Self {
        let mut dir = std::env::current_dir().unwrap();

        loop {
            if let Ok(content) = Self::from_file_path(dir.join("cubecl.toml")) {
                return content;
            }

            if let Ok(content) = Self::from_file_path(dir.join("CubeCL.toml")) {
                return content;
            }

            if !dir.pop() {
                break;
            }
        }

        Self::default()
    }

    // Loads configuration from a specified file path.
    #[cfg(feature = "std")]
    fn from_file_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = match toml::from_str(&content) {
            Ok(val) => val,
            Err(err) => panic!("The file provided doesn't have the right format => {err:?}"),
        };

        Ok(config)
    }
}
