#[cfg(feature = "autotune-checks")]
use alloc::string::String;

/// What a candidate's output needs to be an autotune's: to cross the device thread, and, under
/// the `autotune-checks` feature, to be checked against another candidate's output on the same
/// problem.
///
/// # Assumptions
///
/// - [`check_equivalence`](Self::check_equivalence) returns an error for every mismatch: one it
///   lets pass reads as a passed check.
pub trait AutotuneCheckable: Send + 'static {
    /// Whether `other`, another candidate's output on the same problem, matches this one.
    ///
    /// # Errors
    ///
    /// Every implementation returns an error, saying how the two differ, when they do not match.
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) -> Result<(), String>;
}

impl AutotuneCheckable for () {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, _other: Self) -> Result<(), String> {
        Ok(())
    }
}
