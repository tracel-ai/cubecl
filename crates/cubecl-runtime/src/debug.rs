use core::fmt::Display;

#[cfg(feature = "std")]
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};

#[derive(Debug)]
/// The various debugging options available.
pub enum DebugOptions {
    /// Debug the compilation.
    Debug,
    /// Profile each kernel executed.
    Profile(ProfileLevel),
    /// Enable all options.
    All(ProfileLevel),
}

#[derive(Debug, Copy, Clone)]
/// Control the amount of info being display when profiling.
pub enum ProfileLevel {
    /// Provide only minimal information about kernels being run.
    Basic,
    /// Provide more information about kernels being run.
    Full,
}

/// Debugging logger.
#[derive(Debug)]
pub enum DebugLogger {
    #[cfg(feature = "std")]
    /// Log debugging information into a file.
    File(DebugFileLogger, DebugOptions),
    #[cfg(feature = "std")]
    /// Log debugging information into standard output.
    Stdout(DebugOptions),
    /// Don't log debugging information.
    None,
}

impl Default for DebugLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugLogger {
    #[cfg(not(feature = "std"))]
    /// Create a new debug logger.
    pub fn new() -> Self {
        Self::None
    }

    /// Returns whether the debug logger is activated.
    pub fn is_activated(&self) -> bool {
        !matches!(self, Self::None)
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
    pub fn profile_level(&self) -> Option<ProfileLevel> {
        let option = match self {
            #[cfg(feature = "std")]
            DebugLogger::File(_, option) => option,
            #[cfg(feature = "std")]
            DebugLogger::Stdout(option) => option,
            DebugLogger::None => {
                return None;
            }
        };
        match option {
            DebugOptions::Debug => None,
            DebugOptions::Profile(level) => Some(level.clone()),
            DebugOptions::All(level) => Some(level.clone()),
        }
    }

    /// Register a profiled task.
    pub fn register_profiled<Name>(&mut self, name: Name, duration: core::time::Duration)
    where
        Name: Display,
    {
        match self {
            #[cfg(feature = "std")]
            DebugLogger::File(file, _) => {
                file.log(&format!("| {duration:<10?} | {name}"));
            }
            #[cfg(feature = "std")]
            DebugLogger::Stdout(_) => println!("| {duration:<10?} | {name}"),
            _ => (),
        }
    }

    /// Log the argument to a file when the debug logger is activated.
    pub fn debug<I>(&mut self, arg: I) -> I
    where
        I: Display,
    {
        match self {
            #[cfg(feature = "std")]
            DebugLogger::File(file, option) => {
                match option {
                    DebugOptions::Debug | DebugOptions::All(_) => {
                        file.log(&arg);
                    }
                    DebugOptions::Profile(_) => (),
                };
                arg
            }
            #[cfg(feature = "std")]
            DebugLogger::Stdout(option) => {
                match option {
                    DebugOptions::Debug | DebugOptions::All(_) => {
                        println!("{arg}");
                    }
                    DebugOptions::Profile(_) => (),
                };
                arg
            }
            DebugLogger::None => arg,
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
