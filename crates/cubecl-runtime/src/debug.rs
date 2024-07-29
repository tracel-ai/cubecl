use core::fmt::Display;

#[cfg(feature = "std")]
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::PathBuf,
};

/// Debugging logger.
#[derive(Debug)]
pub enum DebugLogger {
    #[cfg(feature = "std")]
    /// Log debugging information into a file.
    File(DebugFileLogger),
    #[cfg(feature = "std")]
    /// Log debugging information into standard output.
    Stdout,
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

        if let Ok(activated) = str::parse::<u8>(&flag) {
            if activated == 1 {
                return Self::File(DebugFileLogger::new(None));
            } else {
                return Self::None;
            }
        };

        if let Ok(activated) = str::parse::<bool>(&flag) {
            if activated {
                return Self::File(DebugFileLogger::new(None));
            } else {
                return Self::None;
            }
        };

        if let "stdout" = flag.as_str() {
            Self::Stdout
        } else {
            Self::File(DebugFileLogger::new(Some(&flag)))
        }
    }

    /// Log the argument to a file when the debug logger is activated.
    pub fn debug<I>(&mut self, arg: I) -> I
    where
        I: Display,
    {
        match self {
            #[cfg(feature = "std")]
            DebugLogger::File(file) => {
                file.log(&arg);
                arg
            }
            #[cfg(feature = "std")]
            DebugLogger::Stdout => {
                println!("{arg}");
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
