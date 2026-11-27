#[cfg(std_io)]
use crate::tune::AutotuneResult;
#[cfg(std_io)]
use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

/// Events that occurred during autotuning.
#[derive(Debug, Clone)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub enum AutotuneLogEvent {
    /// A candidate ran, compiling and benchmarking, for the duration.
    TuningStep(String, Duration),
    /// The candidate ended the tune under the time limit.
    ShortCircuit(String),
}

/// What one tune logged.
#[derive(Debug, Clone, Default)]
#[cfg_attr(serializable, derive(serde::Serialize, serde::Deserialize))]
pub struct AutotuneLogContext {
    /// What the tune was measured against, when its set has a bounds generator.
    pub bounds: Option<crate::tune::Bounds>,
    /// The time limit a candidate's median has to be under to short circuit the tune.
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

/// The complete account of one tune, written as JSON when logged tunes have a
/// sink configured. One entry per line, per tune, in a fixed schema for tools to read back.
#[cfg(std_io)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "K: Clone + serde::Deserialize<'de>"))]
pub struct LoggedTune<'a, K: Clone> {
    /// The key for the autotuning job.
    pub key: Cow<'a, K>,
    /// The candidate the tune decided, as the table stores it.
    pub decided_index: usize,
    /// Its median time, `None` where it was decided with nothing measured.
    pub decided_time: Option<Duration>,
    /// Every candidate's result, ranked as the table stores them (Arrival Usage).
    pub results: Cow<'a, [AutotuneResult]>,
    /// What the tune logged, when something read it.
    pub log_context: Option<Cow<'a, AutotuneLogContext>>,
    /// The context's [checks](AutotuneLogContext::checks), written beside it, in its order
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
