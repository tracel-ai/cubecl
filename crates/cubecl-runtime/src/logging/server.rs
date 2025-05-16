use core::fmt::Display;

use crate::config::{Logger, compilation::CompilationLogLevel, profiling::ProfilingLogLevel};
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use async_channel::{Receiver, Sender};
use cubecl_common::benchmark::ProfileDuration;

use super::{ProfileLevel, Profiled};

#[cfg(multi_threading)]
use std::thread::{self, JoinHandle};

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
    log_channel: Sender<LogMessage>,
    #[cfg(multi_threading)]
    handle: Option<JoinHandle<()>>,
}

// Make sure we wait for the async logger to be done before exiting.
#[cfg(multi_threading)]
impl Drop for ServerLogger {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
        }
    }
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

        let handle = if !disabled {
            // Spawn a background thread that processes log messages / durations.
            // TODO: On Wasm, this should instead spawn a future.
            #[cfg(not(multi_threading))]
            {
                panic!("Logging compilation/profiling is currently not supported on no_std");
                Some(())
            }

            #[cfg(multi_threading)]
            Some(thread::spawn(|| {
                let as_logger = AsyncLogger {
                    message: rec,
                    logger,
                    profiled: Default::default(),
                };
                cubecl_common::future::block_on(as_logger.process());
            }))
        } else {
            None
        };

        Self {
            profile_level,
            log_compile_info,
            log_channel: send,
            #[cfg(multi_threading)]
            handle,
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

    /// Register a profiled task without timing.
    pub fn register_execution(&self, name: &str) {
        if matches!(self.profile_level, Some(ProfileLevel::ExecutionOnly)) {
            // Channel will never be full, don't care if it's closed.
            let _ = self
                .log_channel
                .try_send(LogMessage::Execution(name.to_string()));
        }
    }

    /// Log the argument to a file when the compilation logger is activated.
    pub fn log_compilation<I>(&self, arg: &I)
    where
        I: Display,
    {
        if self.log_compile_info {
            // Channel will never be full, don't care if it's closed.
            let _ = self
                .log_channel
                .try_send(LogMessage::Compilation(arg.to_string()));
        }
    }

    /// Register a profiled task.
    pub fn register_profiled<Name>(&self, name: Name, duration: ProfileDuration)
    where
        Name: Display,
    {
        if self.profile_level.is_some() {
            // Channel will never be full, don't care if it's closed.
            let _ = self
                .log_channel
                .try_send(LogMessage::Profile(name.to_string(), duration));
        }
    }

    /// Show the profiling summary if activated and reset its state.
    pub fn profile_summary(&self) {
        if self.profile_level.is_some() {
            // Channel will never be full, don't care if it's closed.
            let _ = self.log_channel.try_send(LogMessage::ProfileSummary);
        }
    }
}

#[derive(Debug)]
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
                LogMessage::Profile(name, duration) => {
                    let duration = duration.resolve().await;
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
