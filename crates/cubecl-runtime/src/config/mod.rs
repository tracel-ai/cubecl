/// Autotune config module.
pub mod autotune;
/// Compilation config module.
pub mod compilation;
/// Profiling config module.
pub mod profiling;

mod base;

#[cfg(feature = "std")]
mod logger;

#[cfg(not(feature = "std"))]
mod logger {
    use super::*;

    #[derive(Debug)]
    pub struct Logger;

    impl Logger {
        pub fn new() -> Self {
            Self
        }
        pub fn log_compilation<S: Display>(&mut self, _msg: &S) {}
        pub fn log_profiling<S: Display>(&mut self, _msg: &S) {}
        pub fn log_autotune<S: Display>(&mut self, _msg: &S) {}
    }
}

pub use base::*;
pub use logger::Logger;
