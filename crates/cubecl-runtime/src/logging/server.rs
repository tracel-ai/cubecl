use core::fmt::Display;

use crate::config::memory::MemoryLogLevel;
use crate::config::streaming::StreamingLogLevel;
use crate::config::{
    CubeClRuntimeConfig, Logger, compilation::CompilationLogLevel, profiling::ProfilingLogLevel,
};
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use cubecl_common::profile::ProfileDuration;
use cubecl_environment::future::channel::{Receiver, Sender};
use cubecl_environment::future::spawn_detached;

use super::{ProfileAggregator, ProfileLevel};

/// What a server logs through: the compilation, streaming, memory and
/// profiling logs, written by a task of their own so a launch never waits on a
/// sink.
///
/// # Guarantees
///
/// A message is built only for a log that is on, and no task runs while every
/// log is off.
#[derive(Debug)]
pub struct ServerLogger {
    sender: LogSender,
}

impl Default for ServerLogger {
    fn default() -> Self {
        Self {
            sender: LogSender::spawn(Logger::new()),
        }
    }
}

impl ServerLogger {
    /// The profiling log's level, `None` while it is off.
    pub fn profile_level(&self) -> Option<ProfileLevel> {
        self.sender.levels().profile
    }

    /// Whether compilations are logged.
    pub fn compilation_activated(&self) -> bool {
        self.sender.levels().compilation
    }

    /// Whether compilation logging includes kernel sources
    /// ([`CompilationLogLevel::Full`]) — the only level where attaching debug
    /// info and formatting the source pays off. `Basic` prints a name-only
    /// line, so runtimes must not spend a `clang-format` subprocess per fresh
    /// kernel on it.
    pub fn compilation_source_activated(&self) -> bool {
        self.sender.levels().compilation_source
    }

    /// Logs `arg` to the compilation log.
    pub fn log_compilation<I>(&self, arg: &I)
    where
        I: Display,
    {
        self.sender.send_compilation(arg);
    }

    /// Logs the line `format` builds to the streaming log, when `cond` accepts
    /// its level.
    pub fn log_streaming<I: FnOnce() -> String, C: FnOnce(StreamingLogLevel) -> bool>(
        &self,
        cond: C,
        format: I,
    ) {
        self.sender.send_streaming(cond, format);
    }

    /// Logs the line `format` builds to the memory log, when `cond` accepts
    /// its level.
    pub fn log_memory<I: FnOnce() -> String, C: FnOnce(MemoryLogLevel) -> bool>(
        &self,
        cond: C,
        format: I,
    ) {
        self.sender.send_memory(cond, format);
    }

    /// Log a failure at the moment it happens — the backstop of the lazy
    /// model, for the failure nobody ever observes through a side effect.
    ///
    /// Always on, unlike the opt-in logs above: a kernel that will not
    /// compile is a programming error, and a line at the failure site beats a
    /// `Result` handed over some arbitrary distance later. Reads, syncs and
    /// profiles still fail loudly on the buffers a failure claimed; this line
    /// is for the launch whose output nobody ever asks about.
    pub fn log_failure(&self, error: &impl Display) {
        log::warn!("{error}");
    }

    /// Logs that `name` ran, at [`ProfileLevel::ExecutionOnly`].
    pub fn register_execution(&self, name: impl Display) {
        self.sender.send_execution(name);
    }

    /// Logs how long `name` took, and keeps it for the summary.
    pub fn register_profiled(&self, name: impl Display, duration: ProfileDuration) {
        self.sender.send_profile(name, duration);
    }

    /// Logs the summary of what was profiled since the last one, and starts
    /// the next.
    pub fn profile_summary(&self) {
        self.sender.send_summary();
    }
}

/// The level of each log a [`ServerLogger`] writes.
#[derive(Debug, Clone, Copy)]
struct LogLevels {
    /// The profiling log's, `None` while it is off.
    pub profile: Option<ProfileLevel>,
    /// Whether compilations are logged.
    pub compilation: bool,
    /// Whether they are logged with their sources.
    pub compilation_source: bool,
    /// The streaming log's.
    pub streaming: StreamingLogLevel,
    /// The memory log's.
    pub memory: MemoryLogLevel,
}

impl LogLevels {
    /// The levels `config` sets.
    fn from_config(config: &CubeClRuntimeConfig) -> Self {
        Self {
            profile: match config.profiling.logger.level {
                ProfilingLogLevel::Disabled => None,
                ProfilingLogLevel::Minimal => Some(ProfileLevel::ExecutionOnly),
                ProfilingLogLevel::Basic => Some(ProfileLevel::Basic),
                ProfilingLogLevel::Medium => Some(ProfileLevel::Medium),
                ProfilingLogLevel::Full => Some(ProfileLevel::Full),
            },
            compilation: !matches!(
                config.compilation.logger.level,
                CompilationLogLevel::Disabled
            ),
            compilation_source: matches!(
                config.compilation.logger.level,
                CompilationLogLevel::Full
            ),
            streaming: config.streaming.logger.level,
            memory: config.memory.logger.level,
        }
    }

    /// Whether any log is on.
    fn any_on(self) -> bool {
        self.profile.is_some()
            || self.compilation
            || !matches!(self.streaming, StreamingLogLevel::Disabled)
            || !matches!(self.memory, MemoryLogLevel::Disabled)
    }
}

/// Sends what a [`ServerLogger`] logs to the task that writes it, for the
/// logs that are on.
#[derive(Debug)]
struct LogSender {
    levels: LogLevels,
    /// `None` while every log is off: no task runs to receive.
    channel: Option<Sender<LogMessage>>,
}

impl LogSender {
    /// A sender to a task writing to `logger`'s sinks, at the levels of its
    /// configuration, spawned only when a log is on.
    pub fn spawn(logger: Logger) -> Self {
        let levels = LogLevels::from_config(logger.config());
        let channel = levels.any_on().then(|| {
            let (sender, messages) = cubecl_environment::future::channel::unbounded();
            spawn_detached(
                AsyncLogger {
                    messages,
                    logger,
                    aggregator: ProfileAggregator::default(),
                }
                .log_messages(),
            );
            sender
        });
        Self { levels, channel }
    }

    /// The levels it sends at.
    pub fn levels(&self) -> &LogLevels {
        &self.levels
    }

    /// Sends `arg` to the compilation log, when it is on.
    pub fn send_compilation(&self, arg: &impl Display) {
        self.send(self.levels.compilation, || {
            LogMessage::Compilation(arg.to_string())
        });
    }

    /// Sends the line `format` builds to the streaming log, when `cond`
    /// accepts its level.
    pub fn send_streaming(
        &self,
        cond: impl FnOnce(StreamingLogLevel) -> bool,
        format: impl FnOnce() -> String,
    ) {
        self.send(cond(self.levels.streaming), || {
            LogMessage::Streaming(format())
        });
    }

    /// Sends the line `format` builds to the memory log, when `cond` accepts
    /// its level.
    pub fn send_memory(
        &self,
        cond: impl FnOnce(MemoryLogLevel) -> bool,
        format: impl FnOnce() -> String,
    ) {
        self.send(cond(self.levels.memory), || LogMessage::Memory(format()));
    }

    /// Sends that `name` ran, at [`ProfileLevel::ExecutionOnly`].
    pub fn send_execution(&self, name: impl Display) {
        self.send(
            matches!(self.levels.profile, Some(ProfileLevel::ExecutionOnly)),
            || LogMessage::Execution(name.to_string()),
        );
    }

    /// Sends how long `name` took, while profiling.
    pub fn send_profile(&self, name: impl Display, duration: ProfileDuration) {
        self.send(self.levels.profile.is_some(), || {
            LogMessage::Profile(name.to_string(), duration)
        });
    }

    /// Asks for the summary, while profiling.
    pub fn send_summary(&self) {
        self.send(self.levels.profile.is_some(), || LogMessage::ProfileSummary);
    }

    /// Sends the message `message` builds, when `enabled` and a task runs:
    /// building it costs nothing otherwise.
    fn send(&self, enabled: bool, message: impl FnOnce() -> LogMessage) {
        let Some(channel) = self.channel.as_ref().filter(|_| enabled) else {
            return;
        };
        // Unbounded, so never full. A closed channel means the task is gone,
        // and the message with it.
        let _ = channel.try_send(message());
    }
}

/// What a [`ServerLogger`] sends its logging task.
enum LogMessage {
    Execution(String),
    Compilation(String),
    Streaming(String),
    Memory(String),
    Profile(String, ProfileDuration),
    ProfileSummary,
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
    pub async fn log_messages(mut self) {
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
}

impl AsyncLogger {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compilation::CompilationLogLevel;

    fn levels(configure: impl FnOnce(&mut CubeClRuntimeConfig)) -> LogLevels {
        let mut config = CubeClRuntimeConfig::default();
        configure(&mut config);
        LogLevels::from_config(&config)
    }

    /// A sender at `levels`, and what receives its messages.
    fn sender(levels: LogLevels) -> (LogSender, Receiver<LogMessage>) {
        let (channel, messages) = cubecl_environment::future::channel::unbounded();
        let sender = LogSender {
            levels,
            channel: Some(channel),
        };
        (sender, messages)
    }

    fn disabled() -> LogLevels {
        LogLevels {
            profile: None,
            compilation: false,
            compilation_source: false,
            streaming: StreamingLogLevel::Disabled,
            memory: MemoryLogLevel::Disabled,
        }
    }

    #[test]
    fn the_minimal_profiling_level_logs_executions_only() {
        let levels = levels(|config| {
            config.profiling.logger.level = ProfilingLogLevel::Minimal;
        });
        assert!(matches!(levels.profile, Some(ProfileLevel::ExecutionOnly)));
    }

    #[test]
    fn the_full_compilation_level_logs_sources() {
        let levels = levels(|config| {
            config.compilation.logger.level = CompilationLogLevel::Full;
        });
        assert!(levels.compilation_source);
    }

    #[test]
    fn no_log_is_on_when_every_log_is_disabled() {
        assert!(!disabled().any_on());
    }

    #[test]
    fn a_log_that_is_on_makes_the_levels_on() {
        let levels = LogLevels {
            compilation: true,
            ..disabled()
        };
        assert!(levels.any_on());
    }

    #[test]
    fn a_compilation_is_sent_while_its_log_is_on() {
        let (sender, messages) = sender(LogLevels {
            compilation: true,
            ..disabled()
        });
        sender.send_compilation(&"kernel");
        assert!(matches!(
            messages.try_recv(),
            Ok(LogMessage::Compilation(line)) if line == "kernel"
        ));
    }

    #[test]
    fn nothing_is_built_for_a_log_that_is_off() {
        let (sender, messages) = sender(disabled());
        sender.send_streaming(
            |level| !matches!(level, StreamingLogLevel::Disabled),
            || unreachable!("built for a log that is off"),
        );
        assert!(messages.try_recv().is_err());
    }

    #[test]
    fn an_execution_is_sent_only_at_the_execution_only_level() {
        let (sender, messages) = sender(LogLevels {
            profile: Some(ProfileLevel::Basic),
            ..disabled()
        });
        sender.send_execution("kernel");
        assert!(messages.try_recv().is_err());
    }

    #[test]
    fn a_summary_is_asked_for_while_profiling() {
        let (sender, messages) = sender(LogLevels {
            profile: Some(ProfileLevel::Basic),
            ..disabled()
        });
        sender.send_summary();
        assert!(matches!(
            messages.try_recv(),
            Ok(LogMessage::ProfileSummary)
        ));
    }

    #[test]
    fn nothing_is_sent_without_a_task() {
        let sender = LogSender {
            levels: LogLevels {
                compilation: true,
                ..disabled()
            },
            channel: None,
        };
        sender.send_compilation(&unreachable_display());
    }

    /// A value whose formatting fails the test: what nothing should format.
    fn unreachable_display() -> impl Display {
        struct Unreachable;
        impl Display for Unreachable {
            fn fmt(&self, _: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                unreachable!("formatted with no task to send it to")
            }
        }
        Unreachable
    }
}
