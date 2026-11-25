use super::{CubeClRuntimeConfig, LogLevel, LoggerConfig, RuntimeConfig};
use crate::config::{
    autotune::AutotuneLogLevel, compilation::CompilationLogLevel, memory::MemoryLogLevel,
    streaming::StreamingLogLevel,
};
use alloc::{sync::Arc, vec::Vec};
use core::fmt::Display;

use cubecl_environment::config::logger::LoggerSinks;

/// Writes each of `CubeCL`'s logs to the sinks the configuration gives it.
#[derive(Debug)]
pub struct Logger {
    sinks: LoggerSinks,
    /// The sinks of the compilation log, by their index in `sinks`, read in order by every
    /// `log_compilation`. Empty while that log is disabled.
    compilation_index: Vec<usize>,
    /// The sinks of the autotune log, read in order by every `log_autotune`.
    autotune_index: Vec<usize>,
    /// The sinks of the autotune decisions, read in order by every `log_autotune_decision`.
    autotune_decisions_index: Vec<usize>,
    /// The sinks of the streaming log, read in order by every `log_streaming`.
    streaming_index: Vec<usize>,
    /// The sinks of the memory log, read in order by every `log_memory`.
    memory_index: Vec<usize>,
    config: Arc<CubeClRuntimeConfig>,
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}

impl Logger {
    /// The configuration this logger was made for, levels included.
    pub fn config(&self) -> &CubeClRuntimeConfig {
        &self.config
    }

    /// A logger for the global configuration's sinks and levels.
    ///
    /// Note that creating a logger is quite expensive.
    pub fn new() -> Self {
        let config = CubeClRuntimeConfig::get();
        let mut sinks = LoggerSinks::new();

        let compilation_index = register(&mut sinks, &config.compilation.logger);
        let autotune_index = register(&mut sinks, &config.autotune.logger);
        // Decisions have their own sink, so they land wherever it points regardless of what the
        // logger above is set to, including with the logger disabled.
        let autotune_decisions_index = register_enabled(
            &mut sinks,
            &config.autotune.decisions,
            config.autotune.decisions_enabled(),
        );
        let streaming_index = register(&mut sinks, &config.streaming.logger);
        let memory_index = register(&mut sinks, &config.memory.logger);

        Self {
            sinks,
            compilation_index,
            autotune_index,
            autotune_decisions_index,
            streaming_index,
            memory_index,
            config,
        }
    }

    /// Logs a message for streaming, directing it to all configured streaming loggers.
    pub fn log_streaming<S: Display>(&mut self, msg: &S) {
        self.sinks
            .log(&self.streaming_index, "cubecl::streaming", msg);
    }

    /// Logs a message for memory, directing it to all configured memory loggers.
    pub fn log_memory<S: Display>(&mut self, msg: &S) {
        self.sinks.log(&self.memory_index, "cubecl::memory", msg);
    }

    /// Logs a message for compilation, directing it to all configured compilation loggers.
    pub fn log_compilation<S: Display>(&mut self, msg: &S) {
        self.sinks
            .log(&self.compilation_index, "cubecl::compilation", msg);
    }

    /// Logs a message for autotuning, directing it to all configured autotuning loggers.
    pub fn log_autotune<S: Display>(&mut self, msg: &S) {
        self.sinks
            .log(&self.autotune_index, "cubecl::autotune", msg);
    }

    /// Writes one autotune decision, directing it to all configured decision sinks.
    pub fn log_autotune_decision<S: Display>(&mut self, msg: &S) {
        self.sinks.log(
            &self.autotune_decisions_index,
            // The target keeps the spelling log filters already name.
            "cubecl::autotune::record",
            msg,
        );
    }
}

/// A log level one of whose values turns its log off.
trait Disableable {
    /// Whether this level turns its log off.
    fn is_disabled(&self) -> bool;
}

impl Disableable for CompilationLogLevel {
    fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }
}

impl Disableable for AutotuneLogLevel {
    fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }
}

impl Disableable for StreamingLogLevel {
    fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }
}

impl Disableable for MemoryLogLevel {
    fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }
}

/// The sinks of `logger` registered in `sinks`, or none while its level turns it off.
fn register<L: LogLevel + Disableable>(
    sinks: &mut LoggerSinks,
    logger: &LoggerConfig<L>,
) -> Vec<usize> {
    register_enabled(sinks, logger, !logger.level.is_disabled())
}

/// The sinks of `config` registered in `sinks` when `enabled`, or none.
fn register_enabled<L: LogLevel>(
    sinks: &mut LoggerSinks,
    config: &LoggerConfig<L>,
    enabled: bool,
) -> Vec<usize> {
    if enabled {
        sinks.register(config)
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disabled_log_registers_no_sink() {
        let logger = LoggerConfig::<MemoryLogLevel> {
            stdout: true,
            ..Default::default()
        };

        assert!(register(&mut LoggerSinks::new(), &logger).is_empty());
    }

    #[test]
    fn an_enabled_log_registers_its_sinks() {
        let logger = LoggerConfig::<MemoryLogLevel> {
            stdout: true,
            level: MemoryLogLevel::Basic,
            ..Default::default()
        };

        assert!(!register(&mut LoggerSinks::new(), &logger).is_empty());
    }

    #[test]
    fn only_the_disabled_level_is_disabled() {
        assert!(!CompilationLogLevel::Basic.is_disabled());
    }
}
