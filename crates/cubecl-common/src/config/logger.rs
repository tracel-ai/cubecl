use alloc::string::ToString;
use alloc::vec::Vec;
use core::fmt::Display;
use hashbrown::HashMap;
use serde::Serialize;
use serde::de::DeserializeOwned;

#[cfg(std_io)]
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};

#[cfg(feature = "std")]
use std::{eprintln, println};

/// Configuration for a log sink, parameterized by a subsystem-specific log level.
///
/// Multiple sinks can be enabled at the same time (e.g. both `stdout` and a file).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct LoggerConfig<L: LogLevel> {
    /// Path to the log file, if file logging is enabled (requires filesystem access).
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

    /// The verbosity level for this subsystem.
    #[serde(default)]
    pub level: L,
}

impl<L: LogLevel> Default for LoggerConfig<L> {
    fn default() -> Self {
        Self {
            #[cfg(std_io)]
            file: None,
            append: true,
            stdout: false,
            stderr: false,
            log: None,
            level: L::default(),
        }
    }
}

/// Log levels forwarded to the `log` crate.
#[derive(
    Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize, Hash, PartialEq, Eq,
)]
pub enum LogCrateLevel {
    /// Informational messages.
    #[default]
    #[serde(rename = "info")]
    Info,

    /// Debugging messages.
    #[serde(rename = "debug")]
    Debug,

    /// Trace-level messages.
    #[serde(rename = "trace")]
    Trace,
}

fn append_default() -> bool {
    true
}

/// Trait for types usable as a subsystem-specific log level.
pub trait LogLevel:
    DeserializeOwned + Serialize + Clone + Copy + core::fmt::Debug + Default
{
}

impl LogLevel for u32 {}

/// Registry of log sinks, deduplicating by output target so that multiple subsystems can share
/// a single file or stdout/stderr stream.
#[derive(Debug, Default)]
pub struct LoggerSinks {
    loggers: Vec<LoggerKind>,
    logger2index: HashMap<LoggerId, usize>,
}

impl LoggerSinks {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers every sink described by `config` and returns their indices.
    ///
    /// Subsequent calls with a sink already registered (same file path, stdout, stderr or
    /// log-crate level) reuse the existing index instead of creating a new sink.
    pub fn register<L: LogLevel>(&mut self, config: &LoggerConfig<L>) -> Vec<usize> {
        let mut indices = Vec::new();

        #[cfg(std_io)]
        if let Some(file) = &config.file {
            self.insert(&mut indices, LoggerId::File(file.clone()), || {
                LoggerKind::File(FileLogger::new(file, config.append))
            });
        }

        #[cfg(feature = "std")]
        if config.stdout {
            self.insert(&mut indices, LoggerId::Stdout, || LoggerKind::Stdout);
        }

        #[cfg(feature = "std")]
        if config.stderr {
            self.insert(&mut indices, LoggerId::Stderr, || LoggerKind::Stderr);
        }

        if let Some(level) = config.log {
            self.insert(&mut indices, LoggerId::LogCrate(level), || {
                LoggerKind::Log(level)
            });
        }

        indices
    }

    /// Writes `msg` to every sink in `indices`.
    pub fn log<S: Display>(&mut self, indices: &[usize], msg: &S) {
        match indices.len() {
            0 => {}
            1 => self.loggers[indices[0]].log(msg),
            _ => {
                let msg = msg.to_string();
                for &index in indices {
                    self.loggers[index].log(&msg);
                }
            }
        }
    }

    fn insert<F: FnOnce() -> LoggerKind>(
        &mut self,
        indices: &mut Vec<usize>,
        id: LoggerId,
        make: F,
    ) {
        if let Some(index) = self.logger2index.get(&id) {
            indices.push(*index);
        } else {
            let index = self.loggers.len();
            self.loggers.push(make());
            self.logger2index.insert(id, index);
            indices.push(index);
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
enum LoggerId {
    #[cfg(std_io)]
    File(PathBuf),
    #[cfg(feature = "std")]
    Stdout,
    #[cfg(feature = "std")]
    Stderr,
    LogCrate(LogCrateLevel),
}

#[derive(Debug)]
enum LoggerKind {
    #[cfg(std_io)]
    File(FileLogger),
    #[cfg(feature = "std")]
    Stdout,
    #[cfg(feature = "std")]
    Stderr,
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
                LogCrateLevel::Debug => log::debug!("{msg}"),
                LogCrateLevel::Trace => log::trace!("{msg}"),
            },
        }
    }
}

#[cfg(std_io)]
#[derive(Debug)]
struct FileLogger {
    writer: BufWriter<File>,
}

#[cfg(std_io)]
impl FileLogger {
    fn new(path: &PathBuf, append: bool) -> Self {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
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

    fn log<S: Display>(&mut self, msg: &S) {
        writeln!(self.writer, "{msg}").expect("Should be able to log debug information.");
        self.writer.flush().expect("Can complete write operation.");
    }
}
