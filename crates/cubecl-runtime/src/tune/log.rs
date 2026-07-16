use crate::config::{Logger, autotune::AutotuneLogLevel};
use crate::tune::{AutotuneKey, AutotuneResult};
#[cfg(std_io)]
use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

/// Events that occurred during autotuning, useful for observability and logging.
#[derive(Debug, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum AutotuneLogEvent {
    /// Tracks a tunable kernel that was executed during autotuning.
    TuningStep(String, Duration),
    /// A short circuit event where autotuning stopped early because this candidate
    /// achieved sufficient throughput.
    ShortCircuit(String),
}

/// The context containing bounds, limits, and events that happened during autotuning.
#[derive(Debug, Clone, Default)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub struct AutotuneLogContext {
    /// Calculated bounds for autotuning.
    pub bounds: Option<crate::tune::Bounds<crate::tune::AutotuneBound>>,
    /// The time limit to exceed for early short-circuiting.
    pub limit: Option<Duration>,
    /// The chronological list of tuning events.
    pub events: Vec<AutotuneLogEvent>,
}

impl AutotuneLogContext {
    /// Creates a new log context if the logger is enabled for autotuning.
    pub fn new(
        logger: &mut Logger,
        bounds: Option<crate::tune::Bounds<crate::tune::AutotuneBound>>,
        limit: Option<Duration>,
    ) -> Option<Self> {
        match logger.log_level_autotune() {
            AutotuneLogLevel::Disabled => None,
            _ => Some(Self {
                bounds,
                limit,
                events: Vec::new(),
            }),
        }
    }
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

/// Extension trait for `Option<AutotuneLogContext>` and `Option<&mut AutotuneLogContext>`.
pub trait AutotuneLoggerExt {
    /// Pushes a short circuit event if logging is enabled.
    fn push_short_circuit(&mut self, name: String);
    /// Pushes a tuning step event if logging is enabled.
    fn push_tuning_step(&mut self, name: String, duration: Duration);
    /// Logs the benchmark result if logging is enabled.
    fn log_result<K: AutotuneKey>(
        &self,
        logger: &mut Logger,
        key: &K,
        results: &[AutotuneResult],
        #[cfg(feature = "autotune-checks")] check_results: Option<&[CheckResult]>,
    );
}

impl AutotuneLoggerExt for Option<AutotuneLogContext> {
    fn push_short_circuit(&mut self, name: String) {
        if let Some(ctx) = self.as_mut() {
            ctx.events.push(AutotuneLogEvent::ShortCircuit(name));
        }
    }

    fn push_tuning_step(&mut self, name: String, duration: Duration) {
        if let Some(ctx) = self.as_mut() {
            ctx.events
                .push(AutotuneLogEvent::TuningStep(name, duration));
        }
    }

    fn log_result<K: AutotuneKey>(
        &self,
        logger: &mut Logger,
        key: &K,
        results: &[AutotuneResult],
        #[cfg(feature = "autotune-checks")] check_results: Option<&[CheckResult]>,
    ) {
        crate::tune::log_result(
            logger,
            key,
            results,
            self.as_ref(),
            #[cfg(feature = "autotune-checks")]
            check_results,
        );
    }
}

impl<'a> AutotuneLoggerExt for Option<&'a mut AutotuneLogContext> {
    fn push_short_circuit(&mut self, name: String) {
        if let Some(ctx) = self.as_deref_mut() {
            ctx.events.push(AutotuneLogEvent::ShortCircuit(name));
        }
    }

    fn push_tuning_step(&mut self, name: String, duration: Duration) {
        if let Some(ctx) = self.as_deref_mut() {
            ctx.events
                .push(AutotuneLogEvent::TuningStep(name, duration));
        }
    }

    fn log_result<K: AutotuneKey>(
        &self,
        logger: &mut Logger,
        key: &K,
        results: &[AutotuneResult],
        #[cfg(feature = "autotune-checks")] check_results: Option<&[CheckResult]>,
    ) {
        crate::tune::log_result(
            logger,
            key,
            results,
            self.as_deref(),
            #[cfg(feature = "autotune-checks")]
            check_results,
        );
    }
}

/// Telemetry information emitted when autotune runs.
#[cfg(std_io)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "K: Clone + serde::Deserialize<'de>"))]
pub struct AutotuneTelemetry<'a, K: Clone> {
    /// The key for the autotuning job.
    pub key: Cow<'a, K>,
    /// The index of the fastest candidate.
    pub fastest_index: usize,
    /// The time taken by the fastest candidate.
    pub fastest_time: Duration,
    /// All benchmarking results.
    pub results: Cow<'a, [AutotuneResult]>,
    /// Logging context with bounds, limit, and events.
    pub log_context: Option<Cow<'a, AutotuneLogContext>>,
    /// Check results if autotune-checks is enabled.
    #[cfg(feature = "autotune-checks")]
    pub checks: Option<Cow<'a, [CheckResult]>>,
}

/// The check result for a single benchmark.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckResult {
    /// The name of the benchmark.
    pub name: String,
    /// Whether the check passed.
    pub passed: bool,
}

/// Emit the autotune result through the logger at the currently configured level.
pub(crate) fn log_result<K: AutotuneKey>(
    logger: &mut Logger,
    key: &K,
    results: &[AutotuneResult],
    log_context: Option<&AutotuneLogContext>,
    #[cfg(feature = "autotune-checks")] check_results: Option<&[CheckResult]>,
) {
    let level = logger.log_level_autotune();
    if matches!(level, AutotuneLogLevel::Disabled) {
        return;
    }

    let fastest_result = results
        .first()
        .expect("At least one kernel needed.")
        .outcome
        .as_ref()
        .expect("At least one kernel has to succeed.");

    match level {
        AutotuneLogLevel::Telemetry => {
            #[cfg(std_io)]
            {
                let telemetry = AutotuneTelemetry {
                    key: Cow::Borrowed(key),
                    fastest_index: fastest_result.index,
                    fastest_time: fastest_result.computation.median,
                    results: Cow::Borrowed(results),
                    log_context: log_context.map(Cow::Borrowed),
                    #[cfg(feature = "autotune-checks")]
                    checks: check_results.map(Cow::Borrowed),
                };

                let msg = serde_json::to_string(&telemetry).unwrap_or_else(|err| {
                    format!("{{\"error\": \"Failed to serialize telemetry: {err}\"}}")
                });
                logger.log_autotune(&msg);
            }
            #[cfg(not(std_io))]
            {
                logger.log_autotune(&"{\"error\": \"Telemetry not available without std_io\"}");
            }
        }
        AutotuneLogLevel::Minimal => {
            let top_times = results
                .iter()
                .filter_map(|r| {
                    r.outcome
                        .as_ref()
                        .ok()
                        .map(|o| (o.index, o.computation.median))
                })
                .take(3)
                .collect::<Vec<_>>();

            let context_str = log_context
                .map(|c| format!(", context: {}", c))
                .unwrap_or_default();
            logger.log_autotune(&format!(
                "Fastest result {}-{key}. Top 3 times: {top_times:?}{context_str}",
                fastest_result.name,
            ));
        }
        AutotuneLogLevel::Full => {
            let mut context_str = String::new();
            if let Some(ctx) = log_context {
                use core::fmt::Write;
                if let Some(b) = &ctx.bounds {
                    let _ = writeln!(
                        &mut context_str,
                        "Calculated bounds: {:?} - limit: {:?}",
                        b, ctx.limit
                    );
                }
                let _ = write!(&mut context_str, "{}", ctx);
            }

            logger.log_autotune(&format!(
                "Fastest result {}-{key}.\nContext:\n{context_str}",
                fastest_result.name,
            ));

            for result in results.iter() {
                match &result.outcome {
                    Ok(val) => {
                        logger.log_autotune(&format!("{val}"));
                    }
                    Err(err) => logger.log_autotune(&format!("{err}")),
                }
            }
        }
        AutotuneLogLevel::Disabled => {}
    }
}
