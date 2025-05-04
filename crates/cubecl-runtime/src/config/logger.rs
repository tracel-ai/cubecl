use hashbrown::HashMap;
use std::{
    fmt::Display,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
};

use crate::config::{
    autotune::AutotuneLogLevel, compilation::CompilationLogLevel, profiling::ProfilingLogLevel,
};

use super::GlobalConfig;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct LoggerConfig<L: LogLevel> {
    #[serde(default)]
    #[cfg(feature = "std")]
    pub file: Option<std::path::PathBuf>,
    #[serde(default = "append_default")]
    pub append: bool,
    #[serde(default)]
    pub stdout: bool,
    #[serde(default)]
    pub stderr: bool,
    #[serde(default)]
    pub log: Option<LogCrateLevel>,
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

impl LogLevel for u32 {}

fn append_default() -> bool {
    true
}
pub trait LogLevel:
    serde::de::DeserializeOwned + serde::Serialize + Clone + Copy + core::fmt::Debug + Default
{
}

#[derive(Debug)]
pub struct Logger {
    loggers: Vec<LoggerKind>,
    compilation_index: Vec<usize>,
    profiling_index: Vec<usize>,
    autotune_index: Vec<usize>,
    pub config: Arc<GlobalConfig>,
}

impl Logger {
    pub fn new() -> Self {
        let config = GlobalConfig::get();
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
            kind: &LoggerConfig<L>,
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

        if let AutotuneLogLevel::Full | AutotuneLogLevel::Minmal = config.autotune.logger.level {
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

    pub fn log_level_autotune(&self) -> AutotuneLogLevel {
        self.config.autotune.logger.level
    }

    pub fn log_level_compilation(&self) -> CompilationLogLevel {
        self.config.compilation.logger.level
    }

    pub fn log_level_profiling(&self) -> ProfilingLogLevel {
        self.config.profiling.logger.level
    }

    fn log<S: Display>(&mut self, msg: &S, index: usize) {
        let logger = &mut self.loggers[index];
        logger.log(msg);
    }
}

#[derive(Default, Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum BinaryLogLevel {
    #[default]
    #[serde(rename = "disabled")]
    Disabled,
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for BinaryLogLevel {}

#[derive(Debug)]
enum LoggerKind {
    /// Log debugging information into a file.
    File(FileLogger),
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

/// Log debugging information into a file.
#[derive(Debug)]
struct FileLogger {
    writer: BufWriter<File>,
}

impl FileLogger {
    fn new(file_path: Option<&str>, append: bool) -> Self {
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
    fn log<S: Display>(&mut self, msg: &S) {
        writeln!(self.writer, "{msg}").expect("Should be able to log debug information.");
        self.writer.flush().expect("Can complete write operation.");
    }
}
