#[cfg(std_io)]
use crate::tune::AutotuneResult;
#[cfg(std_io)]
use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

/// Events that occurred during autotuning, useful for observability and logging.
#[derive(Debug, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum AutotuneLogEvent {
    /// Tracks a tunable kernel that was executed during autotuning.
    TuningStep(String, Duration),
    /// A short circuit event where autotuning stopped early because this candidate
    /// achieved sufficient throughput.
    ShortCircuit(String),
}

/// The context containing bounds, limits, and events that happened during autotuning.
#[derive(Debug, Clone, Default)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub struct AutotuneLogContext {
    /// Calculated bounds for autotuning.
    pub bounds: Option<crate::tune::Bounds>,
    /// The time limit to exceed for early short-circuiting.
    pub limit: Option<Duration>,
    /// The tuning events, in the order they happened (Arrival Usage).
    pub events: Vec<AutotuneLogEvent>,
    /// What every candidate's cross-check found, under the `autotune-checks` feature, the
    /// reference candidate last (Arrival Usage).
    pub checks: Option<Vec<CheckResult>>,
}

impl core::fmt::Display for AutotuneLogContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for event in &self.events {
            match event {
                AutotuneLogEvent::TuningStep(step, duration) => {
                    write!(f, "\n - Tuning: {step} (compilation & bench: {duration:?})")?
                }
                AutotuneLogEvent::ShortCircuit(name) => write!(
                    f,
                    "\nShort circuiting autotune. {name} is close enough to peak throughput."
                )?,
            }
        }
        Ok(())
    }
}

/// The complete account of one tuning decision, written as JSON when autotune decisions have a
/// sink configured. One entry per line, per decision, in a fixed schema for tools to read back.
#[cfg(std_io)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "K: Clone + serde::Deserialize<'de>"))]
pub struct AutotuneDecision<'a, K: Clone> {
    /// The key for the autotuning job.
    pub key: Cow<'a, K>,
    /// The index of the candidate the key runs.
    pub decided_index: usize,
    /// The median time of the candidate the key runs.
    pub decided_time: Duration,
    /// Every candidate's result, measured ones first, fastest first (Arrival Usage).
    pub results: Cow<'a, [AutotuneResult]>,
    /// Logging context with bounds, limit, and events.
    pub log_context: Option<Cow<'a, AutotuneLogContext>>,
    /// Check results if autotune-checks is enabled else None, the reference candidate last
    /// (Arrival Usage).
    pub checks: Option<Cow<'a, [CheckResult]>>,
}

/// The cross-check of one candidate's output against the reference candidate's.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CheckResult {
    /// The candidate's name.
    pub name: String,
    /// Whether its output matched the reference's.
    pub passed: bool,
}
