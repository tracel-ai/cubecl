use super::{autotune::AutotuneConfig, compilation::CompilationConfig, profiling::ProfilingConfig};
use alloc::format;
use alloc::string::{String, ToString};
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
    ///
    /// # Notes
    ///
    /// Calling this function is somewhat expensive, because of a global static lock. The config format
    /// is optimized for parsing, not for consumption. A good practice is to use a local static atomic
    /// value that you can populate with the appropriate value from the global config during
    /// initialization of the atomic value.
    ///
    /// For example, the autotune level uses a [core::sync::atomic::AtomicI32] with an initial
    /// value of `-1` to indicate an uninitialized state. It is then set to the proper value based on
    /// the [super::autotune::AutotuneLevel] config. All subsequent fetches of the value are
    /// lock-free.
    pub fn get() -> Arc<Self> {
        let mut state = CUBE_GLOBAL_CONFIG.lock();
        if state.as_ref().is_none() {
            cfg_if::cfg_if! {
                if #[cfg(std_io)]  {
                    let config = Self::from_current_dir();
                    let config = config.override_from_env();
                } else {
                    let config = Self::default();
                }
            }

            *state = Some(Arc::new(config));
        }

        state.as_ref().cloned().unwrap()
    }

    #[cfg(std_io)]
    /// Save the default configuration to the provided file path.
    pub fn save_default<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<()> {
        use std::io::Write;

        let config = Self::get();
        let content =
            toml::to_string_pretty(config.as_ref()).expect("Default config should be serializable");
        let mut file = std::fs::File::create(path)?;
        file.write_all(content.as_bytes())?;

        Ok(())
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

    #[cfg(std_io)]
    /// Overrides configuration fields based on environment variables.
    pub fn override_from_env(mut self) -> Self {
        use super::compilation::CompilationLogLevel;
        use crate::config::{
            autotune::{AutotuneLevel, AutotuneLogLevel},
            profiling::ProfilingLogLevel,
        };

        if let Ok(val) = std::env::var("CUBECL_DEBUG_LOG") {
            self.compilation.logger.level = CompilationLogLevel::Full;
            self.profiling.logger.level = ProfilingLogLevel::Medium;
            self.autotune.logger.level = AutotuneLogLevel::Full;

            match val.as_str() {
                "stdout" => {
                    self.compilation.logger.stdout = true;
                    self.profiling.logger.stdout = true;
                    self.autotune.logger.stdout = true;
                }
                "stderr" => {
                    self.compilation.logger.stderr = true;
                    self.profiling.logger.stderr = true;
                    self.autotune.logger.stderr = true;
                }
                "1" | "true" => {
                    let file_path = "/tmp/cubecl.log";
                    self.compilation.logger.file = Some(file_path.into());
                    self.profiling.logger.file = Some(file_path.into());
                    self.autotune.logger.file = Some(file_path.into());
                }
                "0" | "false" => {
                    self.compilation.logger.level = CompilationLogLevel::Disabled;
                    self.profiling.logger.level = ProfilingLogLevel::Disabled;
                    self.autotune.logger.level = AutotuneLogLevel::Disabled;
                }
                file_path => {
                    self.compilation.logger.file = Some(file_path.into());
                    self.profiling.logger.file = Some(file_path.into());
                    self.autotune.logger.file = Some(file_path.into());
                }
            }
        };

        if let Ok(val) = std::env::var("CUBECL_DEBUG_OPTION") {
            match val.as_str() {
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
            }
        };

        if let Ok(val) = std::env::var("CUBECL_AUTOTUNE_LEVEL") {
            match val.as_str() {
                "minimal" | "0" => {
                    self.autotune.level = AutotuneLevel::Minimal;
                }
                "balanced" | "1" => {
                    self.autotune.level = AutotuneLevel::Balanced;
                }
                "extensive" | "2" => {
                    self.autotune.level = AutotuneLevel::Extensive;
                }
                "full" | "3" => {
                    self.autotune.level = AutotuneLevel::Full;
                }
                _ => {}
            }
        }

        self
    }

    // Loads configuration from `cubecl.toml` or `CubeCL.toml` in the current directory or its parents.
    //
    // Traverses up the directory tree until a valid configuration file is found or the root is reached.
    // Returns a default configuration if no file is found.
    #[cfg(std_io)]
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
    #[cfg(std_io)]
    fn from_file_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = match toml::from_str(&content) {
            Ok(val) => val,
            Err(err) => panic!("The file provided doesn't have the right format => {err:?}"),
        };

        Ok(config)
    }
}

#[derive(Clone, Copy, Debug)]
/// How to format cubecl type names.
pub enum TypeNameFormatLevel {
    /// No formatting apply, full information is included.
    Full,
    /// Most information is removed for a small formatted name.
    Short,
    /// Balanced info is kept.
    Balanced,
}

/// Format a type name with different options.
pub fn type_name_format(name: &str, level: TypeNameFormatLevel) -> String {
    match level {
        TypeNameFormatLevel::Full => name.to_string(),
        TypeNameFormatLevel::Short => {
            if let Some(val) = name.split("<").next() {
                val.split("::").last().unwrap_or(name).to_string()
            } else {
                name.to_string()
            }
        }
        TypeNameFormatLevel::Balanced => {
            let mut split = name.split("<");
            let before_generic = split.next();
            let after_generic = split.next();

            let before_generic = match before_generic {
                None => return name.to_string(),
                Some(val) => val
                    .split("::")
                    .last()
                    .unwrap_or(val)
                    .trim()
                    .replace(">", "")
                    .to_string(),
            };
            let inside_generic = match after_generic {
                None => return before_generic.to_string(),
                Some(val) => {
                    let mut val = val.to_string();
                    for s in split {
                        val += "<";
                        val += s;
                    }
                    val
                }
            };

            let inside = type_name_list_format(&inside_generic, level);

            format!("{before_generic}{inside}")
        }
    }
}

fn type_name_list_format(name: &str, level: TypeNameFormatLevel) -> String {
    let mut acc = String::new();
    let splits = name.split(", ");

    for a in splits {
        acc += " | ";
        acc += &type_name_format(a, level);
    }

    acc
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_format_name() {
        let full_name = "burn_cubecl::kernel::unary_numeric::unary_numeric::UnaryNumeric<f32, burn_cubecl::tensor::base::CubeTensor<_>::copy::Copy, cubecl_cuda::runtime::CudaRuntime>";
        let name = type_name_format(full_name, TypeNameFormatLevel::Balanced);

        assert_eq!(name, "UnaryNumeric | f32 | CubeTensor | Copy | CudaRuntime");
    }
}
