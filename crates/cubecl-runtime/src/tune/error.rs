use alloc::string::String;
use cubecl_common::benchmark::BenchmarkComputations;
use derive_more::Display;

use crate::server::LaunchError;

/// One candidate's measurement, as a tune ranks it.
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct AutotuneOutcome {
    /// The candidate's name.
    pub name: String,
    /// The candidate's index in its set.
    pub index: usize,
    /// What its samples measured.
    pub computation: BenchmarkComputations,
}

impl core::fmt::Display for AutotuneOutcome {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Autotune[{}] name {} => {:?}",
            self.index, self.name, self.computation
        )
    }
}

/// Why a candidate has no measurement.
///
/// [`Unknown`](Self::Unknown) is the only kind a tune that measured nothing falls back on: the
/// candidate executed and only its timing failed. Every other kind says the candidate did not run,
/// or runs wrong, so deciding it would fail the call it is decided for.
#[derive(Clone, Display)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum AutotuneError {
    /// The harness failed around work the candidate queued: a profile it could not take or
    /// resolve, an execution that failed, memory that ran out while it ran.
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
    #[display("{name}: The autotune is skipped manually.")]
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
    /// Whether the candidate executed and only its timing failed, so a tune that measured nothing
    /// may decide it.
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
    use alloc::string::ToString;

    #[test]
    fn only_a_timing_failure_is_executed_unmeasured() {
        let name = || "candidate".to_string();
        let unknown = AutotuneError::Unknown {
            name: name(),
            err: "no timestamp query".into(),
        };
        let held = AutotuneError::HoldFailed {
            name: name(),
            reason: "the device thread is gone".into(),
        };
        let unmeasured = AutotuneError::NotMeasured { name: name() };

        assert!(unknown.executed_unmeasured());
        assert!(!held.executed_unmeasured());
        assert!(!unmeasured.executed_unmeasured());
    }
}
