use core::{
    fmt::Display,
    sync::atomic::{AtomicBool, Ordering},
};

static CUBE_GLOBAL_CONFIG: spin::Mutex<CubeGlobalConfig> =
    spin::Mutex::new(CubeGlobalConfig::default());

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CubeGlobalConfig {
    pub compilation_log_level: CompilationLogLevel,
    pub autotune_log_level: AutotuneLogLevel,
    pub autotune_level: AutotuneLevel,
    pub profiling_level: ProfilingLevel,
    pub logger_compilation: CubeLogger,
    pub logger_profiling: CubeLogger,
    pub logger_autotune: CubeLogger,
}

static MANUAL_SET: AtomicBool = AtomicBool::new(false);

impl CubeGlobalConfig {
    pub fn get() -> Self {
        if !MANUAL_SET.load(Ordering::Relaxed) {
            Self::from_current_dir();
            MANUAL_SET.store(true, Ordering::Relaxed);
        }

        let state = CUBE_GLOBAL_CONFIG.lock();
        state.clone()
    }

    pub fn set(config: Self) {
        let mut state = CUBE_GLOBAL_CONFIG.lock();
        *state = config;
    }

    pub fn from_current_dir() {
        if let Ok(_) = Self::from_file_path("./CubeCL.toml") {
            return;
        }
        if let Ok(_) = Self::from_file_path("./cubecl.toml") {
            return;
        }
    }

    pub fn from_file_path<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<()> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = match toml::from_str(&content) {
            Ok(val) => val,
            Err(err) => panic!("The file provided doesn't have the right format => {err:?}"),
        };

        Self::set(config);

        Ok(())
    }

    pub const fn default() -> Self {
        Self {
            compilation_log_level: CompilationLogLevel::Disabled,
            profiling_level: ProfilingLevel::Disabled,
            autotune_log_level: AutotuneLogLevel::Disabled,
            autotune_level: AutotuneLevel::Medium,
            logger_compilation: CubeLogger::Stdout,
            logger_profiling: CubeLogger::Stdout,
            logger_autotune: CubeLogger::Stdout,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum BinaryLogLevel {
    Disabled,
    Full,
}
pub type CompilationLogLevel = BinaryLogLevel;
pub type AutotuneLogLevel = BinaryLogLevel;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ProfilingLevel {
    Disabled,
    Basic,
    Medium,
    Full,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AutotuneLevel {
    Minimal,
    Medium,
    More,
    Full,
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
    use std::path::PathBuf;

    use super::{file::FileLogger, *};

    #[derive(Debug)]
    pub struct Logger {
        loggers: Vec<LoggerKind>,
        compilation_index: Option<usize>,
        profiling_index: Option<usize>,
        autotune_index: Option<usize>,
        pub config: CubeGlobalConfig,
    }

    impl Logger {
        pub fn new() -> Self {
            let config = CubeGlobalConfig::get();
            let mut loggers = Vec::new();
            let mut compilation_index = None;
            let mut profiling_index = None;
            let mut autotune_index = None;

            let mut path2index = HashMap::<PathBuf, usize>::new();

            let mut new_file_logger =
                |setting_index: &mut Option<usize>,
                 path_buf: &PathBuf,
                 loggers: &mut Vec<LoggerKind>| {
                    if let Some(index) = path2index.get(path_buf) {
                        *setting_index = Some(*index);
                    } else {
                        let logger = LoggerKind::File(FileLogger::new(path_buf.to_str()));
                        let index = loggers.len();
                        path2index.insert(path_buf.clone(), index);
                        loggers.push(logger);
                        *setting_index = Some(index);
                    }
                    loggers.push(LoggerKind::Stdout);
                };

            let new_stdout_logger =
                |setting_index: &mut Option<usize>, loggers: &mut Vec<LoggerKind>| {
                    *setting_index = Some(loggers.len());
                    loggers.push(LoggerKind::Stderr);
                };

            let new_stderr_logger =
                |setting_index: &mut Option<usize>, loggers: &mut Vec<LoggerKind>| {
                    *setting_index = Some(loggers.len());
                    loggers.push(LoggerKind::Stderr);
                };

            let mut register_logger =
                |kind: &CubeLogger, setting_index: &mut Option<usize>| match kind {
                    CubeLogger::File(path_buf) => {
                        new_file_logger(setting_index, path_buf, &mut loggers)
                    }
                    CubeLogger::Stdout => new_stdout_logger(setting_index, &mut loggers),
                    CubeLogger::Stderr => new_stderr_logger(setting_index, &mut loggers),
                };

            if let CompilationLogLevel::Full = config.compilation_log_level {
                register_logger(&config.logger_compilation, &mut compilation_index)
            }

            if let ProfilingLevel::Basic | ProfilingLevel::Medium | ProfilingLevel::Full =
                config.profiling_level
            {
                register_logger(&config.logger_profiling, &mut profiling_index)
            }

            if let BinaryLogLevel::Full = config.autotune_log_level {
                register_logger(&config.logger_autotune, &mut autotune_index)
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
            if let Some(index) = self.compilation_index {
                self.log(msg, index)
            }
        }
        pub fn log_profiling<S: Display>(&mut self, msg: &S) {
            if let Some(index) = self.profiling_index {
                self.log(msg, index)
            }
        }
        pub fn log_autotune<S: Display>(&mut self, msg: &S) {
            if let Some(index) = self.autotune_index {
                self.log(msg, index)
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
    }

    impl LoggerKind {
        fn log<S: Display>(&mut self, msg: &S) {
            match self {
                LoggerKind::File(file_logger) => file_logger.log(msg),
                LoggerKind::Stdout => println!("{msg}"),
                LoggerKind::Stderr => eprintln!("{msg}"),
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
        pub fn new(file_path: Option<&str>) -> Self {
            let path = match file_path {
                Some(path) => PathBuf::from(path),
                None => PathBuf::from("/tmp/cubecl.log"),
            };

            let file = OpenOptions::new()
                .append(true)
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

/// CubeCL logger.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum CubeLogger {
    /// Log information into a file.
    #[cfg(feature = "std")]
    File(std::path::PathBuf),
    /// Log information into standard output.
    #[default]
    Stdout,
    /// Log information into standard error output.
    Stderr,
}
