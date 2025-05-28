use core::fmt::Display;

use crate::config::{Logger, compilation::CompilationLogLevel, profiling::ProfilingLogLevel};
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use async_channel::{Receiver, Sender};
use cubecl_common::future::spawn_detached_fut;
use cubecl_common::profile::ProfileDuration;

use super::{ProfileLevel, Profiled};

enum LogMessage {
    Execution(String),
    Compilation(String),
    Profile(String, ProfileDuration),
    ProfileSummary,
}

/// Server logger.
#[derive(Debug)]
pub struct ServerLogger {
    profile_level: Option<ProfileLevel>,
    log_compile_info: bool,
    log_channel: Option<Sender<LogMessage>>,
}

impl Default for ServerLogger {
    fn default() -> Self {
        let logger = Logger::new();

        let disabled = matches!(
            logger.config.compilation.logger.level,
            CompilationLogLevel::Disabled
        ) && matches!(
            logger.config.profiling.logger.level,
            ProfilingLogLevel::Disabled
        );

        if disabled {
            return Self {
                profile_level: None,
                log_compile_info: false,
                log_channel: None,
            };
        }
        let profile_level = match logger.config.profiling.logger.level {
            ProfilingLogLevel::Disabled => None,
            ProfilingLogLevel::Minimal => Some(ProfileLevel::ExecutionOnly),
            ProfilingLogLevel::Basic => Some(ProfileLevel::Basic),
            ProfilingLogLevel::Medium => Some(ProfileLevel::Medium),
            ProfilingLogLevel::Full => Some(ProfileLevel::Full),
        };

        let log_compile_info = match logger.config.compilation.logger.level {
            CompilationLogLevel::Disabled => false,
            CompilationLogLevel::Basic => true,
            CompilationLogLevel::Full => true,
        };

        let (send, rec) = async_channel::unbounded();

        // Spawn the logger as a detached task.
        let async_logger = AsyncLogger {
            message: rec,
            logger,
            profiled: Default::default(),
        };
        // Spawn the future in the background to logs messages / durations.
        spawn_detached_fut(async_logger.process());

        Self {
            profile_level,
            log_compile_info,
            log_channel: Some(send),
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

    /// Log the argument to a file when the compilation logger is activated.
    pub fn log_compilation<I>(&self, arg: &I)
    where
        I: Display,
    {
        if let Some(channel) = &self.log_channel {
            if self.log_compile_info {
                // Channel will never be full, don't care if it's closed.
                let _ = channel.try_send(LogMessage::Compilation(arg.to_string()));
            }
        }
    }

    /// Register a profiled task without timing.
    pub fn register_execution(&self, name: impl Display) {
        if let Some(channel) = &self.log_channel {
            if matches!(self.profile_level, Some(ProfileLevel::ExecutionOnly)) {
                // Channel will never be full, don't care if it's closed.
                let _ = channel.try_send(LogMessage::Execution(name.to_string()));
            }
        }
    }

    /// Register a profiled task.
    pub fn register_profiled(&self, name: impl Display, duration: ProfileDuration) {
        if let Some(channel) = &self.log_channel {
            if self.profile_level.is_some() {
                // Channel will never be full, don't care if it's closed.
                let _ = channel.try_send(LogMessage::Profile(name.to_string(), duration));
            }
        }
    }

    /// Show the profiling summary if activated and reset its state.
    pub fn profile_summary(&self) {
        if let Some(channel) = &self.log_channel {
            if self.profile_level.is_some() {
                // Channel will never be full, don't care if it's closed.
                let _ = channel.try_send(LogMessage::ProfileSummary);
            }
        }
    }
}

struct AsyncLogger {
    message: Receiver<LogMessage>,
    logger: Logger,
    profiled: Profiled,
}

impl AsyncLogger {
    async fn process(mut self) {
        while let Ok(msg) = self.message.recv().await {
            match msg {
                LogMessage::Compilation(msg) => {
                    self.logger.log_compilation(&msg);
                }
                LogMessage::Profile(name, profile) => {
                    let duration = profile.resolve().await.duration();
                    self.profiled.update(&name, duration);
                    self.logger
                        .log_profiling(&format!("| {duration:<10?} | {name}"));
                }
                LogMessage::Execution(name) => {
                    self.logger.log_profiling(&format!("Executing {name}"));
                }
                LogMessage::ProfileSummary => {
                    if !self.profiled.is_empty() {
                        self.logger.log_profiling(&self.profiled);
                        self.profiled = Profiled::default();
                    }
                }
            }
        }
    }
}
