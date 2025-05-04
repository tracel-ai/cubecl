use core::fmt::Display;

use crate::config::Logger;

use super::{ProfileLevel, Profiled};

/// Server logger.
#[derive(Debug, Default)]
pub struct SeverLogger {
    kind: DebugLoggerKind,
    #[cfg(feature = "std")]
    profiled: Profiled,
}

#[derive(Debug)]
/// The various logging options available.
enum ServerLoggerOptions {
    /// Debug only the compilation.
    CompilationOnly,
    /// Profile each kernel executed.
    #[cfg(feature = "std")]
    ProfileOnly(ProfileLevel),
    /// Enable all options.
    #[cfg(feature = "std")]
    All(ProfileLevel),
}

/// Debugging logger.
#[derive(Debug)]
enum DebugLoggerKind {
    /// Activated logger.
    Activated(Logger, ServerLoggerOptions),
    /// Don't log information.
    None,
}

impl Default for DebugLoggerKind {
    fn default() -> Self {
        Self::new()
    }
}

impl SeverLogger {
    /// Returns the profile level, none if profiling is deactivated.
    pub fn profile_level(&self) -> Option<ProfileLevel> {
        self.kind.profile_level()
    }

    /// Returns true if compilation info should be logged.
    pub fn compilation_activated(&self) -> bool {
        match &self.kind {
            DebugLoggerKind::Activated(_, options) => match options {
                ServerLoggerOptions::CompilationOnly => true,
                ServerLoggerOptions::All(..) => true,
                ServerLoggerOptions::ProfileOnly(..) => false,
            },
            DebugLoggerKind::None => false,
        }
    }

    /// Register a profiled task.
    #[cfg_attr(not(feature = "std"), expect(unused))]
    pub fn register_profiled<Name>(&mut self, name: Name, duration: core::time::Duration)
    where
        Name: Display,
    {
        #[cfg(feature = "std")]
        {
            let name = name.to_string();
            self.profiled.update(&name, duration);

            match self.kind.profile_level().unwrap_or(ProfileLevel::Basic) {
                ProfileLevel::Basic => {}
                _ => self.kind.register_profiled(name, duration),
            }
        }
    }
    /// Log the argument to a file when the compilation logger is activated.
    pub fn log_compilation<I>(&mut self, arg: I) -> I
    where
        I: Display,
    {
        self.kind.log_compilation(arg)
    }

    /// Show the profiling summary if activated and reset its state.
    pub fn profile_summary(&mut self) {
        #[cfg(feature = "std")]
        if self.profile_level().is_some() {
            let mut profiled = Default::default();
            core::mem::swap(&mut self.profiled, &mut profiled);

            match &mut self.kind {
                DebugLoggerKind::Activated(logger, _) => {
                    logger.log_profiling(&profiled);
                }
                _ => (),
            }
        }
    }
}

impl DebugLoggerKind {
    #[cfg(not(feature = "std"))]
    /// Create a new server logger.
    pub fn new() -> Self {
        Self::None
    }

    /// Create a new server logger.
    #[cfg(feature = "std")]
    pub fn new() -> Self {
        use crate::config::{compilation::CompilationLogLevel, profiling::ProfilingLogLevel};

        let logger = Logger::new();

        let mut profile = None;

        match logger.config.profiling.logger.level {
            ProfilingLogLevel::Disabled => {}
            ProfilingLogLevel::Basic => {
                profile = Some(ProfileLevel::Basic);
            }
            ProfilingLogLevel::Medium => {
                profile = Some(ProfileLevel::Medium);
            }
            ProfilingLogLevel::Full => {
                profile = Some(ProfileLevel::Full);
            }
        };

        let option = if let Some(level) = profile {
            if let CompilationLogLevel::Full = logger.config.compilation.logger.level {
                ServerLoggerOptions::All(level)
            } else {
                ServerLoggerOptions::ProfileOnly(level)
            }
        } else {
            if let CompilationLogLevel::Disabled = logger.config.compilation.logger.level {
                return Self::None;
            }

            ServerLoggerOptions::CompilationOnly
        };

        Self::Activated(logger, option)
    }

    /// Returns the profile level, none if profiling is deactivated.
    fn profile_level(&self) -> Option<ProfileLevel> {
        let option = match self {
            DebugLoggerKind::Activated(_, option) => option,
            DebugLoggerKind::None => {
                return None;
            }
        };
        match option {
            ServerLoggerOptions::CompilationOnly => None,
            ServerLoggerOptions::ProfileOnly(level) => Some(*level),
            ServerLoggerOptions::All(level) => Some(*level),
        }
    }

    /// Returns the profile level, none if profiling is deactivated.
    #[cfg(not(feature = "std"))]
    fn profile_level(&self) -> Option<ProfileLevel> {
        None
    }

    #[cfg(feature = "std")]
    fn register_profiled(&mut self, name: String, duration: core::time::Duration) {
        match self {
            DebugLoggerKind::Activated(logger, _) => {
                logger.log_profiling(&format!("| {duration:<10?} | {name}"));
            }
            _ => (),
        }
    }

    fn log_compilation<I>(&mut self, arg: I) -> I
    where
        I: Display,
    {
        match self {
            DebugLoggerKind::Activated(logger, option) => {
                match option {
                    ServerLoggerOptions::CompilationOnly | ServerLoggerOptions::All(_) => {
                        logger.log_compilation(&arg);
                    }
                    ServerLoggerOptions::ProfileOnly(_) => (),
                };
                arg
            }
            DebugLoggerKind::None => arg,
        }
    }
}
