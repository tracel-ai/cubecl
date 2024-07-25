use core::fmt::Display;
#[cfg(feature = "std")]
use std::{
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

impl DebugLogger {
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

        match str::parse::<u8>(&flag) {
            Ok(activated) => {
                if activated == 1 {
                    return Self::File(DebugFileLogger::new(None));
                } else {
                    return Self::None;
                }
            }
            Err(_) => (),
        };

        match str::parse::<bool>(&flag) {
            Ok(activated) => {
                if activated {
                    return Self::File(DebugFileLogger::new(None));
                } else {
                    return Self::None;
                }
            }
            Err(_) => (),
        };

        match flag.as_str() {
            "stdout" => Self::Stdout,
            _ => Self::File(DebugFileLogger::new(Some(&flag))),
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
    writer: BufWriter<std::fs::File>,
}

/// Log debugging information into a standard output.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct DebugStdioLogger;

#[cfg(feature = "std")]
impl DebugFileLogger {
    fn new(file_path: Option<&str>) -> Self {
        let path = match file_path {
            Some(path) => PathBuf::from(path),
            None => PathBuf::from("/tmp/cubecl.log"),
        };

        let file = std::fs::File::create(path).unwrap();

        Self {
            writer: BufWriter::new(file),
        }
    }
    fn log<S: Display>(&mut self, msg: &S) {
        writeln!(self.writer, "{msg}").expect("Should be able to log debug information.");
    }
}
