use alloc::format;
use alloc::string::String;

/// Contains the backtrace information if available.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BackTrace {
    inner: String,
}

impl BackTrace {
    /// Creates a new backtrace from the current thread.
    pub fn capture() -> Self {
        Self {
            #[cfg(feature = "std")]
            inner: format!("{}", std::backtrace::Backtrace::force_capture()),
            #[cfg(not(feature = "std"))]
            inner: format!("No backtrace available"),
        }
    }
}

impl core::fmt::Debug for BackTrace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", self.inner))
    }
}

impl core::fmt::Display for BackTrace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{}", self.inner))
    }
}
