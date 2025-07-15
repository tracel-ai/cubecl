use super::GlobalConfig;
use crate::config::{
    autotune::AutotuneLogLevel, compilation::CompilationLogLevel, profiling::ProfilingLogLevel,
};
use alloc::{string::ToString, sync::Arc, vec::Vec};
use core::fmt::Display;
use hashbrown::HashMap;

#[cfg(std_io)]
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};

/// Configuration for logging in CubeCL, parameterized by a log level type.
///
/// Note that you can use multiple loggers at the same time.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct LoggerConfig<L: LogLevel> {
    /// Path to the log file, if file logging is enabled (requires `std` feature).
    #[serde(default)]
    #[cfg(std_io)]
    pub file: Option<PathBuf>,

    /// Whether to append to the log file (true) or overwrite it (false). Defaults to true.
    ///
    /// ## Notes
    ///
    /// This parameter might get ignored based on other loggers config.
    #[serde(default = "append_default")]
    pub append: bool,

    /// Whether to log to standard output.
    #[serde(default)]
    pub stdout: bool,

    /// Whether to log to standard error.
    #[serde(default)]
    pub stderr: bool,

    /// Optional crate-level logging configuration (e.g., info, debug, trace).
    #[serde(default)]
    pub log: Option<LogCrateLevel>,

    /// The log level for this logger, determining verbosity.
    #[serde(default)]
    pub level: L,
}

impl<L: LogLevel> Default for LoggerConfig<L> {
    fn default() -> Self {
        Self {
            #[cfg(std_io)]
            file: None,
            append: true,
            #[cfg(feature = "autotune-checks")]
            stdout: true,
            #[cfg(not(feature = "autotune-checks"))]
            stdout: false,
            stderr: false,
            log: None,
            level: L::default(),
        }
    }
}

/// Log levels using the `log` crate.
///
/// This enum defines verbosity levels for crate-level logging.
#[derive(
    Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize, Hash, PartialEq, Eq,
)]
pub enum LogCrateLevel {
    /// Logs informational messages.
    #[default]
    #[serde(rename = "info")]
    Info,

    /// Logs debugging messages.
    #[serde(rename = "debug")]
    Debug,

    /// Logs trace-level messages.
    #[serde(rename = "trace")]
    Trace,
}

impl LogLevel for u32 {}

fn append_default() -> bool {
    true
}

/// Trait for types that can be used as log levels in `LoggerConfig`.
pub trait LogLevel:
    serde::de::DeserializeOwned + serde::Serialize + Clone + Copy + core::fmt::Debug + Default
{
}

/// Central logging utility for CubeCL, managing multiple log outputs.
#[derive(Debug)]
pub struct Logger {
    /// Collection of logger instances (file, stdout, stderr, or crate-level).
    loggers: Vec<LoggerKind>,

    /// Indices of loggers used for compilation logging.
    compilation_index: Vec<usize>,

    /// Indices of loggers used for profiling logging.
    profiling_index: Vec<usize>,

    /// Indices of loggers used for autotuning logging.
    autotune_index: Vec<usize>,

    /// Global configuration for logging settings.
    pub config: Arc<GlobalConfig>,
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}

impl Logger {
    /// Creates a new `Logger` instance based on the global configuration.
    ///
    /// Initializes loggers for compilation, profiling, and autotuning based on the settings in
    /// `GlobalConfig`.
    ///
    /// Note that creating a logger is quite expensive.
    pub fn new() -> Self {
        let config = GlobalConfig::get();
        let mut loggers = Vec::new();
        let mut compilation_index = Vec::new();
        let mut profiling_index = Vec::new();
        let mut autotune_index = Vec::new();

        #[derive(Hash, PartialEq, Eq)]
        enum LoggerId {
            #[cfg(std_io)]
            File(PathBuf),
            #[cfg(feature = "std")]
            Stdout,
            #[cfg(feature = "std")]
            Stderr,
            LogCrate(LogCrateLevel),
        }

        let mut logger2index = HashMap::<LoggerId, usize>::new();

        fn new_logger<S: Clone, ID: Fn(S) -> LoggerId, LG: Fn(S) -> LoggerKind>(
            setting_index: &mut Vec<usize>,
            loggers: &mut Vec<LoggerKind>,
            logger2index: &mut HashMap<LoggerId, usize>,
            state: S,
            func_id: ID,
            func_logger: LG,
        ) {
            let id = func_id(state.clone());

            if let Some(index) = logger2index.get(&id) {
                setting_index.push(*index);
            } else {
                let logger = func_logger(state);
                let index = loggers.len();
                logger2index.insert(id, index);
                loggers.push(logger);
                setting_index.push(index);
            }
        }

        fn register_logger<L: LogLevel>(
            #[allow(unused_variables)] kind: &LoggerConfig<L>, // not used in no-std
            #[allow(unused_variables)] append: bool,           // not used in no-std
            level: Option<LogCrateLevel>,
            setting_index: &mut Vec<usize>,
            loggers: &mut Vec<LoggerKind>,
            logger2index: &mut HashMap<LoggerId, usize>,
        ) {
            #[cfg(std_io)]
            if let Some(file) = &kind.file {
                new_logger(
                    setting_index,
                    loggers,
                    logger2index,
                    (file, append),
                    |(file, _append)| LoggerId::File(file.clone()),
                    |(file, append)| LoggerKind::File(FileLogger::new(file, append)),
                );
            }

            #[cfg(feature = "std")]
            if kind.stdout {
                new_logger(
                    setting_index,
                    loggers,
                    logger2index,
                    (),
                    |_| LoggerId::Stdout,
                    |_| LoggerKind::Stdout,
                );
            }

            #[cfg(feature = "std")]
            if kind.stderr {
                new_logger(
                    setting_index,
                    loggers,
                    logger2index,
                    (),
                    |_| LoggerId::Stderr,
                    |_| LoggerKind::Stderr,
                );
            }

            if let Some(level) = level {
                new_logger(
                    setting_index,
                    loggers,
                    logger2index,
                    level,
                    LoggerId::LogCrate,
                    LoggerKind::Log,
                );
            }
        }

        if let CompilationLogLevel::Disabled = config.compilation.logger.level {
        } else {
            register_logger(
                &config.compilation.logger,
                config.compilation.logger.append,
                config.compilation.logger.log,
                &mut compilation_index,
                &mut loggers,
                &mut logger2index,
            )
        }

        if let ProfilingLogLevel::Disabled = config.profiling.logger.level {
        } else {
            register_logger(
                &config.profiling.logger,
                config.profiling.logger.append,
                config.profiling.logger.log,
                &mut profiling_index,
                &mut loggers,
                &mut logger2index,
            )
        }

        if let AutotuneLogLevel::Disabled = config.autotune.logger.level {
        } else {
            register_logger(
                &config.autotune.logger,
                config.autotune.logger.append,
                config.autotune.logger.log,
                &mut autotune_index,
                &mut loggers,
                &mut logger2index,
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

    /// Logs a message for compilation, directing it to all configured compilation loggers.
    pub fn log_compilation<S: Display>(&mut self, msg: &S) {
        let length = self.compilation_index.len();
        if length > 1 {
            let msg = msg.to_string();
            for i in 0..length {
                let index = self.compilation_index[i];
                self.log(&msg, index)
            }
        } else if let Some(index) = self.compilation_index.first() {
            self.log(&msg, *index)
        }
    }

    /// Logs a message for profiling, directing it to all configured profiling loggers.
    pub fn log_profiling<S: Display>(&mut self, msg: &S) {
        let length = self.profiling_index.len();
        if length > 1 {
            let msg = msg.to_string();
            for i in 0..length {
                let index = self.profiling_index[i];
                self.log(&msg, index)
            }
        } else if let Some(index) = self.profiling_index.first() {
            self.log(&msg, *index)
        }
    }

    /// Logs a message for autotuning, directing it to all configured autotuning loggers.
    pub fn log_autotune<S: Display>(&mut self, msg: &S) {
        let length = self.autotune_index.len();
        if length > 1 {
            let msg = msg.to_string();
            for i in 0..length {
                let index = self.autotune_index[i];
                self.log(&msg, index)
            }
        } else if let Some(index) = self.autotune_index.first() {
            self.log(&msg, *index)
        }
    }

    /// Returns the current autotune log level from the global configuration.
    pub fn log_level_autotune(&self) -> AutotuneLogLevel {
        self.config.autotune.logger.level
    }

    /// Returns the current compilation log level from the global configuration.
    pub fn log_level_compilation(&self) -> CompilationLogLevel {
        self.config.compilation.logger.level
    }

    /// Returns the current profiling log level from the global configuration.
    pub fn log_level_profiling(&self) -> ProfilingLogLevel {
        self.config.profiling.logger.level
    }

    fn log<S: Display>(&mut self, msg: &S, index: usize) {
        let logger = &mut self.loggers[index];
        logger.log(msg);
    }
}

/// Binary log level for enabling or disabling logging.
///
/// This enum provides a simple on/off toggle for logging.
#[derive(Default, Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum BinaryLogLevel {
    /// Logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Logging is fully enabled.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for BinaryLogLevel {}

/// Represents different types of loggers.
#[derive(Debug)]
enum LoggerKind {
    /// Logs to a file.
    #[cfg(std_io)]
    File(FileLogger),

    /// Logs to standard output.
    #[cfg(feature = "std")]
    Stdout,

    /// Logs to standard error.
    #[cfg(feature = "std")]
    Stderr,

    /// Logs using the `log` crate with a specified level.
    Log(LogCrateLevel),
}

impl LoggerKind {
    fn log<S: Display>(&mut self, msg: &S) {
        match self {
            #[cfg(std_io)]
            LoggerKind::File(file_logger) => file_logger.log(msg),
            #[cfg(feature = "std")]
            LoggerKind::Stdout => println!("{msg}"),
            #[cfg(feature = "std")]
            LoggerKind::Stderr => eprintln!("{msg}"),
            LoggerKind::Log(level) => match level {
                LogCrateLevel::Info => log::info!("{msg}"),
                LogCrateLevel::Trace => log::debug!("{msg}"),
                LogCrateLevel::Debug => log::trace!("{msg}"),
            },
        }
    }
}

/// Logger that writes messages to a file.
#[derive(Debug)]
#[cfg(std_io)]
struct FileLogger {
    writer: BufWriter<File>,
}

#[cfg(std_io)]
impl FileLogger {
    // Creates a new file logger.
    fn new(path: &PathBuf, append: bool) -> Self {
        let file = OpenOptions::new()
            .write(true)
            .append(append)
            .create(true)
            .open(path)
            .unwrap();

        Self {
            writer: BufWriter::new(file),
        }
    }

    // Logs a message to the file, flushing the buffer to ensure immediate write.
    fn log<S: Display>(&mut self, msg: &S) {
        writeln!(self.writer, "{msg}").expect("Should be able to log debug information.");
        self.writer.flush().expect("Can complete write operation.");
    }
}
