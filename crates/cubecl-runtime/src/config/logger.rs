use super::{CubeClRuntimeConfig, RuntimeConfig};
use crate::config::{
    autotune::AutotuneLogLevel, compilation::CompilationLogLevel, memory::MemoryLogLevel,
    profiling::ProfilingLogLevel, streaming::StreamingLogLevel,
};
use alloc::{sync::Arc, vec::Vec};
use core::fmt::Display;

pub(crate) use cubecl_common::config::logger::{LogLevel, LoggerConfig};
use cubecl_common::config::logger::LoggerSinks;

/// Central logging utility for `CubeCL`, managing multiple log outputs.
#[derive(Debug)]
pub struct Logger {
    sinks: LoggerSinks,
    compilation_index: Vec<usize>,
    profiling_index: Vec<usize>,
    autotune_index: Vec<usize>,
    streaming_index: Vec<usize>,
    memory_index: Vec<usize>,
    /// Global configuration for logging settings.
    pub config: Arc<CubeClRuntimeConfig>,
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}

impl Logger {
    /// Creates a new `Logger` instance based on the global configuration.
    ///
    /// Note that creating a logger is quite expensive.
    pub fn new() -> Self {
        let config = CubeClRuntimeConfig::get();
        let mut sinks = LoggerSinks::new();

        let compilation_index = register_enabled(
            &mut sinks,
            &config.compilation.logger,
            !matches!(config.compilation.logger.level, CompilationLogLevel::Disabled),
        );
        let profiling_index = register_enabled(
            &mut sinks,
            &config.profiling.logger,
            !matches!(config.profiling.logger.level, ProfilingLogLevel::Disabled),
        );
        let autotune_index = register_enabled(
            &mut sinks,
            &config.autotune.logger,
            !matches!(config.autotune.logger.level, AutotuneLogLevel::Disabled),
        );
        let streaming_index = register_enabled(
            &mut sinks,
            &config.streaming.logger,
            !matches!(config.streaming.logger.level, StreamingLogLevel::Disabled),
        );
        let memory_index = register_enabled(
            &mut sinks,
            &config.memory.logger,
            !matches!(config.memory.logger.level, MemoryLogLevel::Disabled),
        );

        Self {
            sinks,
            compilation_index,
            profiling_index,
            autotune_index,
            streaming_index,
            memory_index,
            config,
        }
    }

    /// Logs a message for streaming, directing it to all configured streaming loggers.
    pub fn log_streaming<S: Display>(&mut self, msg: &S) {
        self.sinks.log(&self.streaming_index, msg);
    }

    /// Logs a message for memory, directing it to all configured memory loggers.
    pub fn log_memory<S: Display>(&mut self, msg: &S) {
        self.sinks.log(&self.memory_index, msg);
    }

    /// Logs a message for compilation, directing it to all configured compilation loggers.
    pub fn log_compilation<S: Display>(&mut self, msg: &S) {
        self.sinks.log(&self.compilation_index, msg);
    }

    /// Logs a message for profiling, directing it to all configured profiling loggers.
    pub fn log_profiling<S: Display>(&mut self, msg: &S) {
        self.sinks.log(&self.profiling_index, msg);
    }

    /// Logs a message for autotuning, directing it to all configured autotuning loggers.
    pub fn log_autotune<S: Display>(&mut self, msg: &S) {
        self.sinks.log(&self.autotune_index, msg);
    }

    /// Returns the current streaming log level from the global configuration.
    pub fn log_level_streaming(&self) -> StreamingLogLevel {
        self.config.streaming.logger.level
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
}

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
