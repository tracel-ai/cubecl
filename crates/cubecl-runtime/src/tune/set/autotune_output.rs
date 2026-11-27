#[cfg(feature = "autotune-checks")]
use alloc::string::String;

/// The output of an autotune's candidates: one that crosses the device thread and, under the
/// `autotune-checks` feature, is checked against another candidate's output on the same problem.
///
/// # Guarantees
///
/// - Under `autotune-checks`, every implementation returns an error for every mismatch: one it
///   lets pass reads as a passed check.
pub trait AutotuneOutput: Send + 'static {
    /// Whether `other`, another candidate's output on the same problem, matches this one.
    ///
    /// # Errors
    ///
    /// How the two differ, when they do not match.
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) -> Result<(), String>;
}

impl AutotuneOutput for () {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, _other: Self) -> Result<(), String> {
        Ok(())
    }
}
