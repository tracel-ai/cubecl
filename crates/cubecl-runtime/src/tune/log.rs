use crate::config::{Logger, autotune::AutotuneLogLevel};
use crate::tune::{AutotuneKey, AutotuneResult};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

/// Events that occurred during autotuning, useful for observability and logging.
#[derive(Debug, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize))]
pub enum AutotuneLogEvent {
    /// Tracks a batch of tunable kernels that were executed during autotuning.
    TuningSteps(Vec<String>),
    /// A short circuit event where autotuning stopped early because this candidate
    /// achieved sufficient throughput.
    ShortCircuit(String),
}

/// The context containing bounds, limits, and events that happened during autotuning.
#[derive(Debug, Clone, Default)]
#[cfg_attr(std_io, derive(serde::Serialize))]
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
                AutotuneLogEvent::TuningSteps(steps) => write!(f, "\n - Tuning: {steps:?}")?,
                AutotuneLogEvent::ShortCircuit(name) => write!(
                    f,
                    "\nShort circuiting autotune. {name} is close enough to peak throughput."
                )?,
            }
        }
        Ok(())
    }
}

/// Extension trait for `Option<AutotuneLogContext>` and `Option<&mut AutotuneLogContext>` to make logging events cleaner.
pub trait AutotuneLoggerExt {
    /// Pushes a short circuit event if logging is enabled.
    fn push_short_circuit(&mut self, name: String);
    /// Pushes a tuning steps event if logging is enabled.
    fn push_tuning_steps(&mut self, names: Vec<String>);
    /// Logs the benchmark result if logging is enabled.
    fn log_result<K: AutotuneKey>(&self, logger: &mut Logger, key: &K, results: &[AutotuneResult]);
}

impl AutotuneLoggerExt for Option<AutotuneLogContext> {
    fn push_short_circuit(&mut self, name: String) {
        if let Some(ctx) = self.as_mut() {
            ctx.events.push(AutotuneLogEvent::ShortCircuit(name));
        }
    }

    fn push_tuning_steps(&mut self, names: Vec<String>) {
        if let Some(ctx) = self.as_mut() {
            ctx.events.push(AutotuneLogEvent::TuningSteps(names));
        }
    }

    fn log_result<K: AutotuneKey>(&self, logger: &mut Logger, key: &K, results: &[AutotuneResult]) {
        crate::tune::log_result(logger, key, results, self.as_ref());
    }
}

impl<'a> AutotuneLoggerExt for Option<&'a mut AutotuneLogContext> {
    fn push_short_circuit(&mut self, name: String) {
        if let Some(ctx) = self.as_deref_mut() {
            ctx.events.push(AutotuneLogEvent::ShortCircuit(name));
        }
    }

    fn push_tuning_steps(&mut self, names: Vec<String>) {
        if let Some(ctx) = self.as_deref_mut() {
            ctx.events.push(AutotuneLogEvent::TuningSteps(names));
        }
    }

    fn log_result<K: AutotuneKey>(&self, logger: &mut Logger, key: &K, results: &[AutotuneResult]) {
        crate::tune::log_result(logger, key, results, self.as_deref());
    }
}

/// Telemetry information emitted when autotune runs.
#[cfg(std_io)]
#[derive(serde::Serialize)]
pub struct AutotuneTelemetry<'a, K> {
    /// The key for the autotuning job.
    pub key: &'a K,
    /// The index of the fastest candidate.
    pub fastest_index: usize,
    /// The time taken by the fastest candidate.
    pub fastest_time: Duration,
    /// All benchmarking results.
    pub results: &'a [AutotuneResult],
    /// Logging context with bounds, limit, and events.
    pub log_context: Option<&'a AutotuneLogContext>,
}

/// Emit the autotune result through the logger at the currently configured level.
pub(crate) fn log_result<K: AutotuneKey>(
    logger: &mut Logger,
    key: &K,
    results: &[AutotuneResult],
    log_context: Option<&AutotuneLogContext>,
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
                    key,
                    fastest_index: fastest_result.index,
                    fastest_time: fastest_result.computation.median,
                    results,
                    log_context,
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
                .map(|r| {
                    let time = r
                        .outcome
                        .as_ref()
                        .map(|r| r.computation.median)
                        .unwrap_or(Duration::MAX);

                    let index = r.outcome.as_ref().map(|r| r.index).unwrap_or_default();
                    (index, time)
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
