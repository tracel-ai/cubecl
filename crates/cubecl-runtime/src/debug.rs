use core::fmt::Display;

#[cfg(feature = "std")]
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};

#[cfg(feature = "std")]
use profile::*;

use crate::config::Logger;

#[cfg(feature = "std")]
mod profile {
    use core::fmt::Display;
    use std::collections::HashMap;

    #[derive(Debug, Default)]
    pub(crate) struct Profiled {
        durations: HashMap<String, ProfileItem>,
    }

    #[derive(Debug, Default, Clone)]
    pub(crate) struct ProfileItem {
        total_duration: core::time::Duration,
        num_computed: usize,
    }

    impl Profiled {
        pub fn update(&mut self, name: &String, duration: core::time::Duration) {
            let name = if name.contains("\n") {
                name.split("\n").next().unwrap()
            } else {
                name
            };
            if let Some(item) = self.durations.get_mut(name) {
                item.update(duration);
            } else {
                self.durations.insert(
                    name.to_string(),
                    ProfileItem {
                        total_duration: duration,
                        num_computed: 1,
                    },
                );
            }
        }
    }

    impl Display for Profiled {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            let header_name = "Name";
            let header_num_computed = "Num Computed";
            let header_duration = "Duration";
            let header_ratio = "Ratio";

            let mut ratio_len = header_ratio.len();
            let mut name_len = header_name.len();
            let mut num_computed_len = header_num_computed.len();
            let mut duration_len = header_duration.len();

            let mut total_duration = core::time::Duration::from_secs(0);
            let mut total_computed = 0;

            let mut items: Vec<(String, String, String, core::time::Duration)> = self
                .durations
                .iter()
                .map(|(key, item)| {
                    let name = key.clone();
                    let num_computed = format!("{}", item.num_computed);
                    let duration = format!("{:?}", item.total_duration);

                    name_len = usize::max(name_len, name.len());
                    num_computed_len = usize::max(num_computed_len, num_computed.len());
                    duration_len = usize::max(duration_len, duration.len());

                    total_duration += item.total_duration;
                    total_computed += item.num_computed;

                    (name, num_computed, duration, item.total_duration)
                })
                .collect();

            let total_duration_fmt = format!("{:?}", total_duration);
            let total_compute_fmt = format!("{:?}", total_computed);
            let total_ratio_fmt = "100 %";

            duration_len = usize::max(duration_len, total_duration_fmt.len());
            num_computed_len = usize::max(num_computed_len, total_compute_fmt.len());
            ratio_len = usize::max(ratio_len, total_ratio_fmt.len());

            let line_length = name_len + duration_len + num_computed_len + ratio_len + 11;

            let write_line = |char: &str, f: &mut core::fmt::Formatter<'_>| {
                writeln!(f, "|{}| ", char.repeat(line_length))
            };
            items.sort_by(|(_, _, _, a), (_, _, _, b)| b.cmp(a));

            write_line("⎺", f)?;

            writeln!(
                f,
                "| {:<width_name$} | {:<width_duration$} | {:<width_num_computed$} | {:<width_ratio$} |",
                header_name,
                header_duration,
                header_num_computed,
                header_ratio,
                width_name = name_len,
                width_duration = duration_len,
                width_num_computed = num_computed_len,
                width_ratio = ratio_len,
            )?;

            write_line("⎼", f)?;

            for (name, num_computed, duration, num) in items {
                let ratio = (100 * num.as_micros()) / total_duration.as_micros();

                writeln!(
                    f,
                    "| {:<width_name$} | {:<width_duration$} | {:<width_num_computed$} | {:<width_ratio$} |",
                    name,
                    duration,
                    num_computed,
                    format!("{} %", ratio),
                    width_name = name_len,
                    width_duration = duration_len,
                    width_num_computed = num_computed_len,
                    width_ratio = ratio_len,
                )?;
            }

            write_line("⎼", f)?;

            writeln!(
                f,
                "| {:<width_name$} | {:<width_duration$} | {:<width_num_computed$} | {:<width_ratio$} |",
                "Total",
                total_duration_fmt,
                total_compute_fmt,
                total_ratio_fmt,
                width_name = name_len,
                width_duration = duration_len,
                width_num_computed = num_computed_len,
                width_ratio = ratio_len,
            )?;

            write_line("⎯", f)?;

            Ok(())
        }
    }

    impl ProfileItem {
        pub fn update(&mut self, duration: core::time::Duration) {
            self.total_duration += duration;
            self.num_computed += 1;
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// Control the amount of info being display when profiling.
pub enum ProfileLevel {
    /// Provide only the summary information about kernels being run.
    Basic,
    /// Provide the summary information about kernels being run with their trace.
    Medium,
    /// Provide more information about kernels being run.
    Full,
}

#[derive(Debug)]
/// The various debugging options available.
pub enum DebugOptions {
    /// Debug the compilation.
    Debug,
    /// Profile each kernel executed.
    #[cfg(feature = "std")]
    Profile(ProfileLevel),
    /// Enable all options.
    #[cfg(feature = "std")]
    All(ProfileLevel),
}

/// Debugging logger.
#[derive(Debug, Default)]
pub struct DebugLogger {
    kind: DebugLoggerKind,
    #[cfg(feature = "std")]
    profiled: Profiled,
}

/// Debugging logger.
#[derive(Debug)]
pub enum DebugLoggerKind {
    Activated(Logger, DebugOptions),
    /// Don't log debugging information.
    None,
}

impl Default for DebugLoggerKind {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugLogger {
    /// Returns the profile level, none if profiling is deactivated.
    pub fn profile_level(&self) -> Option<ProfileLevel> {
        self.kind.profile_level()
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
    /// Returns whether the debug logger is activated.
    pub fn is_activated(&self) -> bool {
        !matches!(self.kind, DebugLoggerKind::None)
    }
    /// Log the argument to a file when the debug logger is activated.
    pub fn debug<I>(&mut self, arg: I) -> I
    where
        I: Display,
    {
        self.kind.debug(arg)
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
    /// Create a new debug logger.
    pub fn new() -> Self {
        Self::None
    }

    /// Create a new debug logger.
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
                DebugOptions::All(level)
            } else {
                DebugOptions::Profile(level)
            }
        } else {
            if let CompilationLogLevel::Disabled = logger.config.compilation.logger.level {
                return Self::None;
            }

            DebugOptions::Debug
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
            DebugOptions::Debug => None,
            DebugOptions::Profile(level) => Some(*level),
            DebugOptions::All(level) => Some(*level),
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

    fn debug<I>(&mut self, arg: I) -> I
    where
        I: Display,
    {
        match self {
            DebugLoggerKind::Activated(logger, option) => {
                match option {
                    DebugOptions::Debug | DebugOptions::All(_) => {
                        logger.log_compilation(&arg);
                    }
                    DebugOptions::Profile(_) => (),
                };
                arg
            }
            DebugLoggerKind::None => arg,
        }
    }
}

/// Log debugging information into a file.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct DebugFileLogger {
    writer: BufWriter<File>,
}

#[cfg(feature = "std")]
impl DebugFileLogger {
    fn new(file_path: Option<&str>) -> Self {
        let path = match file_path {
            Some(path) => PathBuf::from(path),
            None => PathBuf::from("/tmp/cubecl.log"),
        };

        let file = OpenOptions::new()
            .append(true)
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
