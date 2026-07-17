use crate::config::{Logger, autotune::AutotuneLogLevel};
use crate::tune::{AutotuneKey, AutotuneOutcome, AutotuneResult};
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
    pub bounds: Option<crate::tune::Bounds>,
    /// The time limit to exceed for early short-circuiting.
    pub limit: Option<Duration>,
    /// The chronological list of tuning events.
    pub events: Vec<AutotuneLogEvent>,
    /// The results of the checks.
    pub checks: Option<Vec<crate::tune::log::CheckResult>>,
}

impl AutotuneLogContext {
    /// Creates a new log context if the logger is enabled for autotuning.
    pub fn new(logger: &mut Logger) -> Option<Self> {
        match logger.log_level_autotune() {
            AutotuneLogLevel::Disabled => None,
            _ => Some(Self {
                bounds: None,
                limit: None,
                events: Vec::new(),
                checks: None,
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
    /// Sets the tuning bounds if logging is active.
    fn set_bounds(&mut self, bounds: Option<crate::tune::Bounds>);
    /// Sets the tuning limit if logging is active.
    fn set_limit(&mut self, limit: Option<Duration>);
    /// Sets checks if logging is active.
    fn set_checks(&mut self, checks: impl FnOnce() -> Vec<CheckResult>);
    /// Logs the benchmark result if logging is enabled.
    fn log_result<K: AutotuneKey>(&self, logger: &mut Logger, key: &K, results: &[AutotuneResult]);
}

/// Implements `AutotuneLoggerExt` for an `Option`-like wrapper around `AutotuneLogContext`.
/// The two concrete types (`Option<AutotuneLogContext>` and `Option<&mut AutotuneLogContext>`)
/// differ only in how they reach the inner `&mut AutotuneLogContext`: `.as_mut()` vs
/// `.as_deref_mut()`, and `.as_ref()` vs `.as_deref()`.
macro_rules! impl_autotune_logger_ext {
    ($ty:ty, $as_mut:ident, $as_ref:ident) => {
        impl AutotuneLoggerExt for $ty {
            fn push_short_circuit(&mut self, name: String) {
                if let Some(ctx) = self.$as_mut() {
                    ctx.events.push(AutotuneLogEvent::ShortCircuit(name));
                }
            }

            fn push_tuning_step(&mut self, name: String, duration: Duration) {
                if let Some(ctx) = self.$as_mut() {
                    ctx.events
                        .push(AutotuneLogEvent::TuningStep(name, duration));
                }
            }

            fn set_bounds(&mut self, bounds: Option<crate::tune::Bounds>) {
                if let Some(ctx) = self.$as_mut() {
                    ctx.bounds = bounds;
                }
            }

            fn set_limit(&mut self, limit: Option<Duration>) {
                if let Some(ctx) = self.$as_mut() {
                    ctx.limit = limit;
                }
            }

            fn set_checks(&mut self, checks: impl FnOnce() -> Vec<CheckResult>) {
                if let Some(ctx) = self.$as_mut() {
                    ctx.checks = Some(checks());
                }
            }

            fn log_result<K: AutotuneKey>(
                &self,
                logger: &mut Logger,
                key: &K,
                results: &[AutotuneResult],
            ) {
                log_result(logger, key, results, self.$as_ref());
            }
        }
    };
}

impl_autotune_logger_ext!(Option<AutotuneLogContext>, as_mut, as_ref);
impl_autotune_logger_ext!(Option<&'_ mut AutotuneLogContext>, as_deref_mut, as_deref);

/// The complete record of one tuning decision, written as JSON when the autotune recorder has a
/// sink configured. One record per line, per decision, in a fixed schema for tools to read back.
#[cfg(std_io)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(deserialize = "K: Clone + serde::Deserialize<'de>"))]
pub struct AutotuneRecord<'a, K: Clone> {
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
    /// Check results if autotune-checks is enabled else None.
    pub checks: Option<Cow<'a, [CheckResult]>>,
}

/// The check result for a single benchmark.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckResult {
    /// The name of the benchmark.
    pub name: String,
    /// Whether the check passed.
    pub passed: bool,
}

/// Emit the autotune result: a line for humans at the logger's level and, independently, the
/// [`AutotuneRecord`] for tools if the recorder has a sink. Either, both, or neither.
fn log_result<K: AutotuneKey>(
    logger: &mut Logger,
    key: &K,
    results: &[AutotuneResult],
    log_context: Option<&AutotuneLogContext>,
) {
    let level = logger.log_level_autotune();
    let recording = logger.autotune_recording_enabled();
    if matches!(level, AutotuneLogLevel::Disabled) && !recording {
        return;
    }

    // Shared by both sinks, and resolved only once one of them is listening: it assumes a candidate
    // succeeded, which is not true of every tuning pass.
    let fastest = results
        .first()
        .expect("At least one kernel needed.")
        .outcome
        .as_ref()
        .expect("At least one kernel has to succeed.");

    if recording {
        write_record(logger, key, results, log_context, fastest);
    }
    write_log(logger, level, key, results, log_context, fastest);
}

/// The record, for tools: one JSON object on the recorder's sink.
#[cfg_attr(not(std_io), allow(unused_variables))]
fn write_record<K: AutotuneKey>(
    logger: &mut Logger,
    key: &K,
    results: &[AutotuneResult],
    log_context: Option<&AutotuneLogContext>,
    fastest: &AutotuneOutcome,
) {
    #[cfg(std_io)]
    {
        let record = AutotuneRecord {
            key: Cow::Borrowed(key),
            fastest_index: fastest.index,
            fastest_time: fastest.computation.median,
            results: Cow::Borrowed(results),
            log_context: log_context.map(Cow::Borrowed),
            checks: log_context
                .and_then(|c| c.checks.as_deref())
                .map(Cow::Borrowed),
        };

        let msg = serde_json::to_string(&record).unwrap_or_else(|err| {
            format!("{{\"error\": \"Failed to serialize the autotune record: {err}\"}}")
        });
        logger.log_autotune_record(&msg);
    }
    #[cfg(not(std_io))]
    {
        logger.log_autotune_record(
            &"{\"error\": \"Recording autotune is not available without std_io\"}",
        );
    }
}

/// The line for humans, at whatever the logger's level asks for.
fn write_log<K: AutotuneKey>(
    logger: &mut Logger,
    level: AutotuneLogLevel,
    key: &K,
    results: &[AutotuneResult],
    log_context: Option<&AutotuneLogContext>,
    fastest: &AutotuneOutcome,
) {
    match level {
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
                fastest.name,
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
                fastest.name,
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
