use alloc::string::{String, ToString};
use derive_more::Display;

use crate::server::{LaunchError, ProfileError, ServerError};

/// Why a candidate has no measurement.
///
/// [`Unknown`](Self::Unknown) is the only kind a tune that measured nothing falls back on. Every
/// other kind says the candidate did not run, or runs wrong, so deciding it would fail the call it
/// is decided for.
#[derive(Clone, Display)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum AutotuneError {
    /// The candidate ran, and the profile around it failed: its work was queued, and only its
    /// timing is missing.
    #[display("{name}: An unknown error happened.\n{err}")]
    Unknown {
        /// The candidate's name.
        name: String,
        /// What the harness reported.
        err: String,
    },
    /// The candidate's own function returned an error, so it queued no work for this problem: it
    /// does not serve these inputs, or failed before launching.
    #[display("{name}: Rejected the problem.\n{reason}")]
    Rejected {
        /// The candidate's name.
        name: String,
        /// What its function returned.
        reason: String,
    },
    /// The tune could not hold the device to measure the candidate, so its measurement never
    /// started.
    #[display("{name}: The device could not be held to measure it.\n{reason}")]
    HoldFailed {
        /// The candidate's name.
        name: String,
        /// Why the device could not be held.
        reason: String,
    },
    /// The harness could not open the profile of a sample, so the candidate never ran in it.
    #[display("{name}: The profile of a sample could not be opened.\n{reason}")]
    Unopened {
        /// The candidate's name.
        name: String,
        /// Why the profile could not be opened.
        reason: String,
    },
    /// A sample's profiled window carried no measurement.
    ///
    /// Disqualifying, and deliberately not folded into a zero: zero is the fastest duration there
    /// is, so a candidate carrying one wins every comparison it enters and gets cached as the best
    /// row in the table. The window may have timed a kernel that never launched, so it is never a
    /// candidate to fall back on either.
    #[display("{name}: A profiled sample carried no measurement.")]
    NotMeasured {
        /// The candidate's name.
        name: String,
    },
    /// The candidate was never measured: the tune ended before it, or its groups' patience
    /// skipped it.
    #[display(
        "{name}: Not measured: the tune ended before it, or its groups' patience skipped it."
    )]
    Skip {
        /// The candidate's name.
        name: String,
    },

    /// One of the candidate's kernels failed to launch: it does not compile on this device, or
    /// asks for more than the device holds.
    Launch(LaunchError),

    /// The candidate panicked, or one of its kernels did while it expanded: a defect in it, which
    /// running it again would hit again.
    #[display("{name}: Panicked.\n{message}")]
    Panicked {
        /// The candidate's name.
        name: String,
        /// What the panic said.
        message: String,
    },
}

impl AutotuneError {
    /// [`HoldFailed`](Self::HoldFailed) for the candidate `name`.
    pub(crate) fn hold_failed(name: &str, reason: &ServerError) -> Self {
        Self::HoldFailed {
            name: name.into(),
            reason: reason.to_string(),
        }
    }

    /// [`Unopened`](Self::Unopened) for the candidate `name`, whose profile failed before it ran.
    pub(crate) fn unopened(name: &str, reason: &ProfileError) -> Self {
        Self::Unopened {
            name: name.into(),
            reason: reason.to_string(),
        }
    }

    /// What a profile that failed after the candidate `name` ran says of it: a panic while its
    /// kernels expanded, a launch that failed, or a failure of the harness around work it queued.
    pub(crate) fn unprofiled(name: &str, error: ProfileError) -> Self {
        match error {
            error if error.is_expansion_panic() => Self::Panicked {
                name: name.into(),
                message: error.to_string(),
            },
            ProfileError::Launch(launch) => Self::Launch(launch),
            error => Self::Unknown {
                name: name.into(),
                err: error.to_string(),
            },
        }
    }

    /// Whether the harness kept the candidate from running, through no fault of its own: the
    /// device could not be held, or the profile could not be opened.
    pub(crate) fn kept_from_running(&self) -> bool {
        matches!(self, Self::HoldFailed { .. } | Self::Unopened { .. })
    }

    /// Whether a tune that measured nothing may decide the candidate.
    pub(crate) fn executed_unmeasured(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }
}

impl core::fmt::Debug for AutotuneError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

impl From<LaunchError> for AutotuneError {
    fn from(value: LaunchError) -> Self {
        Self::Launch(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::CompilationError;

    #[test]
    fn a_harness_failure_is_executed_unmeasured() {
        let unknown = AutotuneError::Unknown {
            name: "candidate".into(),
            err: "no timestamp query".into(),
        };

        assert!(unknown.executed_unmeasured());
    }

    #[test]
    fn a_failed_hold_is_not_executed_unmeasured() {
        let error = ServerError::Generic {
            reason: "the device thread is gone".into(),
            backtrace: Default::default(),
        };

        assert!(!AutotuneError::hold_failed("candidate", &error).executed_unmeasured());
    }

    #[test]
    fn an_unopened_profile_kept_its_candidate_from_running() {
        let error = ProfileError::NotRegistered {
            backtrace: Default::default(),
        };

        assert!(AutotuneError::unopened("candidate", &error).kept_from_running());
    }

    #[test]
    fn a_rejection_did_not_keep_its_candidate_from_running() {
        let rejected = AutotuneError::Rejected {
            name: "candidate".into(),
            reason: "not this shape".into(),
        };

        assert!(!rejected.kept_from_running());
    }

    #[test]
    fn an_unopened_profile_is_not_executed_unmeasured() {
        let error = ProfileError::NotRegistered {
            backtrace: Default::default(),
        };

        assert!(!AutotuneError::unopened("candidate", &error).executed_unmeasured());
    }

    #[test]
    fn a_sample_with_no_measurement_is_not_executed_unmeasured() {
        let unmeasured = AutotuneError::NotMeasured {
            name: "candidate".into(),
        };

        assert!(!unmeasured.executed_unmeasured());
    }

    #[test]
    fn a_profile_stopped_by_an_expansion_panic_is_panicked() {
        let error = ProfileError::Launch(LaunchError::CompilationError(
            CompilationError::ExpansionPanicked {
                kernel: "kernel".into(),
                message: "an assertion failed".into(),
                backtrace: Default::default(),
            },
        ));

        assert!(matches!(
            AutotuneError::unprofiled("candidate", error),
            AutotuneError::Panicked { .. }
        ));
    }

    #[test]
    fn a_profile_whose_launch_failed_is_a_launch_failure() {
        let error = ProfileError::Launch(LaunchError::OutOfMemory {
            reason: "the pool is full".into(),
            backtrace: Default::default(),
        });

        assert!(matches!(
            AutotuneError::unprofiled("candidate", error),
            AutotuneError::Launch(_)
        ));
    }

    #[test]
    fn a_profile_that_failed_otherwise_is_unknown() {
        let error = ProfileError::NotRegistered {
            backtrace: Default::default(),
        };

        assert!(matches!(
            AutotuneError::unprofiled("candidate", error),
            AutotuneError::Unknown { .. }
        ));
    }
}
