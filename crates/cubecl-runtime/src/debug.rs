use core::fmt::Display;

#[cfg(feature = "std")]
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};

#[cfg(feature = "std")]
use profile::*;

#[cfg(feature = "std")]
mod profile;

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
    #[cfg(feature = "std")]
    /// Log debugging information into a file.
    File(DebugFileLogger, DebugOptions),
    #[cfg(feature = "std")]
    /// Log debugging information into standard output.
    Stdout(DebugOptions),
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
                #[cfg(feature = "std")]
                DebugLoggerKind::File(file, _) => {
                    file.log(&format!("{}", profiled));
                }
                #[cfg(feature = "std")]
                DebugLoggerKind::Stdout(_) => println!("{profiled}"),
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
        let flag = match std::env::var("CUBECL_DEBUG_LOG") {
            Ok(val) => val,
            Err(_) => return Self::None,
        };
        let level = match std::env::var("CUBECL_DEBUG_OPTION") {
            Ok(val) => val,
            Err(_) => "debug|profile".to_string(),
        };

        let mut debug = false;
        let mut profile = None;
        level.as_str().split("|").for_each(|flag| match flag {
            "debug" => {
                debug = true;
            }
            "profile" => {
                profile = Some(ProfileLevel::Basic);
            }
            "profile-medium" => {
                profile = Some(ProfileLevel::Medium);
            }
            "profile-full" => {
                profile = Some(ProfileLevel::Full);
            }
            _ => {}
        });

        let option = if let Some(level) = profile {
            if debug {
                DebugOptions::All(level)
            } else {
                DebugOptions::Profile(level)
            }
        } else {
            DebugOptions::Debug
        };

        if let Ok(activated) = str::parse::<u8>(&flag) {
            if activated == 1 {
                return Self::File(DebugFileLogger::new(None), option);
            } else {
                return Self::None;
            }
        };

        if let Ok(activated) = str::parse::<bool>(&flag) {
            if activated {
                return Self::File(DebugFileLogger::new(None), option);
            } else {
                return Self::None;
            }
        };

        if let "stdout" = flag.as_str() {
            Self::Stdout(option)
        } else {
            Self::File(DebugFileLogger::new(Some(&flag)), option)
        }
    }

    /// Returns the profile level, none if profiling is deactivated.
    #[cfg(feature = "std")]
    fn profile_level(&self) -> Option<ProfileLevel> {
        let option = match self {
            DebugLoggerKind::File(_, option) => option,
            DebugLoggerKind::Stdout(option) => option,
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
            #[cfg(feature = "std")]
            DebugLoggerKind::File(file, _) => {
                file.log(&format!("| {duration:<10?} | {name}"));
            }
            #[cfg(feature = "std")]
            DebugLoggerKind::Stdout(_) => println!("| {duration:<10?} | {name}"),
            _ => (),
        }
    }

    fn debug<I>(&mut self, arg: I) -> I
    where
        I: Display,
    {
        match self {
            #[cfg(feature = "std")]
            DebugLoggerKind::File(file, option) => {
                match option {
                    DebugOptions::Debug | DebugOptions::All(_) => {
                        file.log(&arg);
                    }
                    DebugOptions::Profile(_) => (),
                };
                arg
            }
            #[cfg(feature = "std")]
            DebugLoggerKind::Stdout(option) => {
                match option {
                    DebugOptions::Debug | DebugOptions::All(_) => {
                        println!("{arg}");
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
