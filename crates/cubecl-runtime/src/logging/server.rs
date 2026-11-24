use core::fmt::Display;

use crate::config::memory::MemoryLogLevel;
use crate::config::streaming::StreamingLogLevel;
use crate::config::{Logger, compilation::CompilationLogLevel, profiling::ProfilingLogLevel};
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use cubecl_common::profile::ProfileDuration;
use cubecl_environment::future::channel::{Receiver, Sender};
use cubecl_environment::future::spawn_detached;

use super::{ProfileAggregator, ProfileLevel};

enum LogMessage {
    Execution(String),
    Compilation(String),
    Streaming(String),
    Memory(String),
    Profile(String, ProfileDuration),
    ProfileSummary,
}

/// What a server logs through: the compilation, streaming, memory and
/// profiling logs, written by a task of their own so a launch never waits on a
/// sink.
#[derive(Debug)]
pub struct ServerLogger {
    profile_level: Option<ProfileLevel>,
    log_compile_info: bool,
    log_compile_source: bool,
    log_streaming: StreamingLogLevel,
    log_channel: Option<Sender<LogMessage>>,
    log_memory: MemoryLogLevel,
}

impl Default for ServerLogger {
    fn default() -> Self {
        let logger = Logger::new();
        let config = logger.config();

        let profile_level = match config.profiling.logger.level {
            ProfilingLogLevel::Disabled => None,
            ProfilingLogLevel::Minimal => Some(ProfileLevel::ExecutionOnly),
            ProfilingLogLevel::Basic => Some(ProfileLevel::Basic),
            ProfilingLogLevel::Medium => Some(ProfileLevel::Medium),
            ProfilingLogLevel::Full => Some(ProfileLevel::Full),
        };
        let log_compile_info = !matches!(
            config.compilation.logger.level,
            CompilationLogLevel::Disabled
        );
        let log_compile_source =
            matches!(config.compilation.logger.level, CompilationLogLevel::Full);
        let log_streaming = config.streaming.logger.level;
        let log_memory = config.memory.logger.level;

        // Nothing is spawned while every log is off: no message would reach it.
        let enabled = profile_level.is_some()
            || log_compile_info
            || !matches!(log_streaming, StreamingLogLevel::Disabled)
            || !matches!(log_memory, MemoryLogLevel::Disabled);
        let log_channel = enabled.then(|| {
            let (send, receive) = cubecl_environment::future::channel::unbounded();
            spawn_detached(
                AsyncLogger {
                    messages: receive,
                    logger,
                    aggregator: ProfileAggregator::default(),
                }
                .log_messages(),
            );
            send
        });

        Self {
            profile_level,
            log_compile_info,
            log_compile_source,
            log_streaming,
            log_channel,
            log_memory,
        }
    }
}

impl ServerLogger {
    /// Returns the profile level, none if profiling is deactivated.
    pub fn profile_level(&self) -> Option<ProfileLevel> {
        self.profile_level
    }

    /// Returns true if compilation info should be logged.
    pub fn compilation_activated(&self) -> bool {
        self.log_compile_info
    }

    /// Returns true if compilation logging includes kernel sources
    /// ([`CompilationLogLevel::Full`]) — the only level where attaching debug
    /// info and formatting the source pays off. `Basic` prints a name-only
    /// line, so runtimes must not spend a `clang-format` subprocess per fresh
    /// kernel on it.
    pub fn compilation_source_activated(&self) -> bool {
        self.log_compile_source
    }

    /// Log the argument to a file when the compilation logger is activated.
    pub fn log_compilation<I>(&self, arg: &I)
    where
        I: Display,
    {
        self.send(self.log_compile_info, || {
            LogMessage::Compilation(arg.to_string())
        });
    }

    /// Log the argument to the logger when the streaming logger is activated.
    pub fn log_streaming<I: FnOnce() -> String, C: FnOnce(StreamingLogLevel) -> bool>(
        &self,
        cond: C,
        format: I,
    ) {
        self.send(cond(self.log_streaming), || LogMessage::Streaming(format()));
    }

    /// Log the argument to the logger when the memory logger is activated.
    pub fn log_memory<I: FnOnce() -> String, C: FnOnce(MemoryLogLevel) -> bool>(
        &self,
        cond: C,
        format: I,
    ) {
        self.send(cond(self.log_memory), || LogMessage::Memory(format()));
    }

    /// Log a failure at the moment it happens — the backstop of the lazy
    /// model, for the failure nobody ever observes through a side effect.
    ///
    /// Always on, unlike the opt-in channels above: a kernel that will not
    /// compile is a programming error, and a line at the failure site beats a
    /// `Result` handed over some arbitrary distance later. Reads, syncs and
    /// profiles still fail loudly on the buffers a failure claimed; this line
    /// is for the launch whose output nobody ever asks about.
    pub fn log_failure(&self, error: &impl Display) {
        log::warn!("{error}");
    }

    /// Register a profiled task without timing.
    pub fn register_execution(&self, name: impl Display) {
        self.send(
            matches!(self.profile_level, Some(ProfileLevel::ExecutionOnly)),
            || LogMessage::Execution(name.to_string()),
        );
    }

    /// Register a profiled task.
    pub fn register_profiled(&self, name: impl Display, duration: ProfileDuration) {
        self.send(self.profile_level.is_some(), || {
            LogMessage::Profile(name.to_string(), duration)
        });
    }

    /// Show the profiling summary if activated and reset its state.
    pub fn profile_summary(&self) {
        self.send(self.profile_level.is_some(), || LogMessage::ProfileSummary);
    }
}

impl ServerLogger {
    /// Sends the message `message` builds to the logging task, when `enabled`
    /// and a task runs: building it costs nothing otherwise.
    fn send(&self, enabled: bool, message: impl FnOnce() -> LogMessage) {
        let Some(channel) = self.log_channel.as_ref().filter(|_| enabled) else {
            return;
        };
        // Unbounded, so never full. A closed channel means the task is gone,
        // and the message with it.
        let _ = channel.try_send(message());
    }
}

/// Writes the messages a [`ServerLogger`] sends, off the thread that sends
/// them.
struct AsyncLogger {
    messages: Receiver<LogMessage>,
    logger: Logger,
    aggregator: ProfileAggregator,
}

impl AsyncLogger {
    /// Writes every message until the last sender drops.
    async fn log_messages(mut self) {
        while let Ok(message) = self.messages.recv().await {
            match message {
                LogMessage::Compilation(message) => self.logger.log_compilation(&message),
                LogMessage::Streaming(message) => self.logger.log_streaming(&message),
                LogMessage::Memory(message) => self.logger.log_memory(&message),
                LogMessage::Profile(name, profile) => self.log_profile(&name, profile).await,
                LogMessage::Execution(name) => {
                    self.logger.log_profiling(&format!("Executing {name}"))
                }
                LogMessage::ProfileSummary => self.log_summary(),
            }
        }
    }

    /// Writes one profiled launch, and aggregates it for the summary.
    async fn log_profile(&mut self, name: &str, profile: ProfileDuration) {
        // Named but not counted. Folding an absence into the summary as a
        // zero would drag every average it joins down towards a speed nothing
        // achieved.
        let Some(ticks) = profile.resolve().await else {
            self.logger
                .log_profiling(&format!("| {:<10} | {name}", "not measured"));
            return;
        };
        let duration = ticks.duration();
        self.aggregator.aggregate(name, duration);
        self.logger
            .log_profiling(&format!("| {duration:<10?} | {name}"));
    }

    /// Writes the summary of what was profiled since the last one, if
    /// anything was.
    fn log_summary(&mut self) {
        if self.aggregator.is_empty() {
            return;
        }
        self.logger.log_profiling(&self.aggregator);
        self.aggregator.clear();
    }
}
