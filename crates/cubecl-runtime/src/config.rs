use alloc::sync::Arc;

static CUBE_GLOBAL_CONFIG: spin::Mutex<Option<Arc<CubeGlobalConfig>>> = spin::Mutex::new(None);

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CubeGlobalConfig {
    #[serde(default)]
    pub profiling: ProfilingConfig,
    #[serde(default)]
    pub autotune: AutotuneConfig,
    #[serde(default)]
    pub compilation: CompilationConfig,
}

impl LogLevel for u32 {}
impl LogLevel for BinaryLogLevel {}
impl LogLevel for ProfilingLogLevel {}
impl LogLevel for AutotuneLogLevel {}

pub trait LogLevel:
    serde::de::DeserializeOwned + serde::Serialize + Clone + core::fmt::Debug + Default
{
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct SingleLoggerConfig<L: LogLevel> {
    #[serde(default)]
    #[cfg(feature = "std")]
    file: Option<std::path::PathBuf>,
    #[serde(default = "append_default")]
    append: bool,
    #[serde(default)]
    stdout: bool,
    #[serde(default)]
    stderr: bool,
    #[serde(default)]
    log: Option<LogCrateLevel>,
    #[serde(default)]
    pub level: L,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum LogCrateLevel {
    #[default]
    #[serde(rename = "info")]
    Info,
    #[serde(rename = "debug")]
    Debug,
    #[serde(rename = "trace")]
    Trace,
}

fn append_default() -> bool {
    true
}

/// CubeCL logger.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum TestConfig {
    /// Log information into a file.
    #[cfg(feature = "std")]
    #[serde(rename = "file")]
    File(std::path::PathBuf),
    /// Log information into standard output.
    #[default]
    #[serde(rename = "stdout")]
    Stdout,
    /// Log information into standard error output.
    #[serde(rename = "stderr")]
    Stderr,
    /// Log information with the log crate.
    #[serde(rename = "log")]
    Log,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CompilationConfig {
    #[serde(default)]
    pub logger: SingleLoggerConfig<CompilationLogLevel>,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ProfilingConfig {
    #[serde(default)]
    pub logger: SingleLoggerConfig<ProfilingLogLevel>,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AutotuneConfig {
    #[serde(default)]
    pub logger: SingleLoggerConfig<AutotuneLogLevel>,
    #[serde(default)]
    pub level: AutotuneLevel,
    #[serde(default)]
    pub cache: AutotuneCache,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneCache {
    Local,
    #[default]
    Target,
    Global,
    File(std::path::PathBuf),
}

impl AutotuneCache {
    pub fn root(&self) -> std::path::PathBuf {
        match self {
            AutotuneCache::Local => std::env::current_dir().unwrap(),
            AutotuneCache::Target => {
                let dir_original = std::env::current_dir().unwrap();
                let mut dir = dir_original.clone();

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

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum BinaryLogLevel {
    #[default]
    #[serde(rename = "disabled")]
    Disabled,
    #[serde(rename = "full")]
    Full,
}

pub type CompilationLogLevel = BinaryLogLevel;

#[derive(Default, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneLogLevel {
    #[serde(rename = "disabled")]
    Disabled,
    #[default]
    #[serde(rename = "minimal")]
    Minmal,
    #[serde(rename = "full")]
    Full,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProfilingLogLevel {
    #[default]
    #[serde(rename = "disabled")]
    Disabled,
    #[serde(rename = "basic")]
    Basic,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "full")]
    Full,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneLevel {
    #[serde(rename = "minimal")]
    Minimal,
    #[default]
    #[serde(rename = "balanced")]
    Medium,
    #[serde(rename = "more")]
    More,
    #[serde(rename = "full")]
    Full,
}

impl CubeGlobalConfig {
    pub fn get() -> Arc<Self> {
        let mut state = CUBE_GLOBAL_CONFIG.lock();
        if let None = state.as_ref() {
            let config = Self::from_current_dir();
            *state = Some(Arc::new(config));
        }

        state.as_ref().cloned().unwrap()
    }

    pub fn set(config: Self) {
        let mut state = CUBE_GLOBAL_CONFIG.lock();
        *state = Some(Arc::new(config));
    }

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

    fn from_file_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = match toml::from_str(&content) {
            Ok(val) => val,
            Err(err) => panic!("The file provided doesn't have the right format => {err:?}"),
        };

        Ok(config)
    }
}

pub use logger::Logger;

#[cfg(not(feature = "std"))]
mod logger {
    use super::*;

    #[derive(Debug)]
    pub struct Logger;

    impl Logger {
        pub fn new() -> Self {
            Self
        }
        pub fn log_compilation<S: Display>(&mut self, _msg: &S) {}
        pub fn log_profiling<S: Display>(&mut self, _msg: &S) {}
        pub fn log_autotune<S: Display>(&mut self, _msg: &S) {}
    }
}

#[cfg(feature = "std")]
mod logger {
    use hashbrown::HashMap;
    use std::{fmt::Display, path::PathBuf};

    use super::{file::FileLogger, *};

    #[derive(Debug)]
    pub struct Logger {
        loggers: Vec<LoggerKind>,
        compilation_index: Vec<usize>,
        profiling_index: Vec<usize>,
        autotune_index: Vec<usize>,
        pub config: Arc<CubeGlobalConfig>,
    }

    impl Logger {
        pub fn autotune_log_level(&self) -> AutotuneLogLevel {
            self.config.autotune.logger.level
        }
        pub fn new() -> Self {
            let config = CubeGlobalConfig::get();
            let mut loggers = Vec::new();
            let mut compilation_index = Vec::new();
            let mut profiling_index = Vec::new();
            let mut autotune_index = Vec::new();

            let mut path2index = HashMap::<PathBuf, usize>::new();

            fn new_file_logger<L: LogLevel>(
                setting_index: &mut Vec<usize>,
                path_buf: &PathBuf,
                append: bool,
                loggers: &mut Vec<LoggerKind>,
                path2index: &mut HashMap<PathBuf, usize>,
            ) {
                if let Some(index) = path2index.get(path_buf) {
                    setting_index.push(*index);
                } else {
                    let logger = LoggerKind::File(FileLogger::new(path_buf.to_str(), append));
                    let index = loggers.len();
                    path2index.insert(path_buf.clone(), index);
                    loggers.push(logger);
                    setting_index.push(index);
                }
                loggers.push(LoggerKind::Stdout);
            }

            fn new_stdout_logger(setting_index: &mut Vec<usize>, loggers: &mut Vec<LoggerKind>) {
                setting_index.push(loggers.len());
                loggers.push(LoggerKind::Stdout);
            }

            fn new_log_logger(
                setting_index: &mut Vec<usize>,
                loggers: &mut Vec<LoggerKind>,
                level: LogCrateLevel,
            ) {
                setting_index.push(loggers.len());
                loggers.push(LoggerKind::Log(level));
            }

            fn new_stderr_logger(setting_index: &mut Vec<usize>, loggers: &mut Vec<LoggerKind>) {
                setting_index.push(loggers.len());
                loggers.push(LoggerKind::Stderr);
            }

            fn register_logger<L: LogLevel>(
                kind: &SingleLoggerConfig<L>,
                append: bool,
                level: Option<LogCrateLevel>,
                setting_index: &mut Vec<usize>,
                loggers: &mut Vec<LoggerKind>,
                path2index: &mut HashMap<PathBuf, usize>,
            ) {
                if let Some(file) = &kind.file {
                    new_file_logger::<L>(setting_index, file, append, loggers, path2index)
                }

                if kind.stdout {
                    new_stdout_logger(setting_index, loggers)
                }

                if kind.stderr {
                    new_stderr_logger(setting_index, loggers)
                }

                if let Some(level) = level {
                    new_log_logger(setting_index, loggers, level)
                }
            }

            if let CompilationLogLevel::Full = config.compilation.logger.level {
                register_logger(
                    &config.compilation.logger,
                    config.compilation.logger.append,
                    config.compilation.logger.log,
                    &mut compilation_index,
                    &mut loggers,
                    &mut path2index,
                )
            }

            if let ProfilingLogLevel::Basic | ProfilingLogLevel::Medium | ProfilingLogLevel::Full =
                config.profiling.logger.level
            {
                register_logger(
                    &config.profiling.logger,
                    config.profiling.logger.append,
                    config.profiling.logger.log,
                    &mut profiling_index,
                    &mut loggers,
                    &mut path2index,
                )
            }

            if let AutotuneLogLevel::Full | AutotuneLogLevel::Minmal = config.autotune.logger.level
            {
                register_logger(
                    &config.autotune.logger,
                    config.autotune.logger.append,
                    config.autotune.logger.log,
                    &mut autotune_index,
                    &mut loggers,
                    &mut path2index,
                )
            }

            Self {
                loggers,
                compilation_index,
                profiling_index,
                autotune_index,
                config,
            }
        }

        pub fn log_compilation<S: Display>(&mut self, msg: &S) {
            let length = self.compilation_index.len();
            if length > 1 {
                let msg = msg.to_string();
                for i in 0..length {
                    let index = self.compilation_index[i];
                    self.log(&msg, index)
                }
            } else if let Some(index) = self.compilation_index.get(0) {
                self.log(&msg, *index)
            }
        }
        pub fn log_profiling<S: Display>(&mut self, msg: &S) {
            let length = self.profiling_index.len();
            if length > 1 {
                let msg = msg.to_string();
                for i in 0..length {
                    let index = self.profiling_index[i];
                    self.log(&msg, index)
                }
            } else if let Some(index) = self.profiling_index.get(0) {
                self.log(&msg, *index)
            }
        }
        pub fn log_autotune<S: Display>(&mut self, msg: &S) {
            let length = self.autotune_index.len();
            if length > 1 {
                let msg = msg.to_string();
                for i in 0..length {
                    let index = self.autotune_index[i];
                    self.log(&msg, index)
                }
            } else if let Some(index) = self.autotune_index.get(0) {
                self.log(&msg, *index)
            }
        }
        fn log<S: Display>(&mut self, msg: &S, index: usize) {
            let logger = &mut self.loggers[index];
            logger.log(msg);
        }
    }

    #[derive(Debug)]
    enum LoggerKind {
        /// Log debugging information into a file.
        File(file::FileLogger),
        /// Log debugging information into standard output.
        Stdout,
        /// Log debugging information into standard output.
        Stderr,
        Log(LogCrateLevel),
    }

    impl LoggerKind {
        fn log<S: Display>(&mut self, msg: &S) {
            match self {
                LoggerKind::File(file_logger) => file_logger.log(msg),
                LoggerKind::Stdout => println!("{msg}"),
                LoggerKind::Stderr => eprintln!("{msg}"),
                LoggerKind::Log(level) => match level {
                    LogCrateLevel::Info => log::info!("{msg}"),
                    LogCrateLevel::Trace => log::debug!("{msg}"),
                    LogCrateLevel::Debug => log::trace!("{msg}"),
                },
            }
        }
    }
}

#[cfg(feature = "std")]
mod file {
    use core::fmt::Display;
    use std::{
        fs::{File, OpenOptions},
        io::{BufWriter, Write},
        path::PathBuf,
    };

    /// Log debugging information into a file.
    #[derive(Debug)]
    pub struct FileLogger {
        writer: BufWriter<File>,
    }

    impl FileLogger {
        pub fn new(file_path: Option<&str>, append: bool) -> Self {
            let path = match file_path {
                Some(path) => PathBuf::from(path),
                None => PathBuf::from("/tmp/cubecl.log"),
            };

            let file = OpenOptions::new()
                .append(append)
                .create(true)
                .open(path)
                .unwrap();

            Self {
                writer: BufWriter::new(file),
            }
        }
        pub fn log<S: Display>(&mut self, msg: &S) {
            writeln!(self.writer, "{msg}").expect("Should be able to log debug information.");
            self.writer.flush().expect("Can complete write operation.");
        }
    }
}
