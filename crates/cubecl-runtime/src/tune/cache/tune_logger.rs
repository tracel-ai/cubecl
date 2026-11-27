use alloc::format;
use alloc::string::String;
#[cfg(feature = "autotune-checks")]
use alloc::vec::Vec;
use core::time::Duration;

use crate::config::{Logger, autotune::AutotuneLogLevel};
#[cfg(feature = "autotune-checks")]
use crate::tune::CheckResult;
use crate::tune::{
    AutotuneKeyable, AutotuneLogContext, AutotuneLogEvent, AutotuneOutcome, AutotuneResult, Bounds,
};

/// Logs what one tune did into its [`AutotuneLogContext`], kept only while something reads it:
/// the human log, the decisions' sink, or the record of the tune.
#[derive(Debug)]
pub struct TuneLogger {
    context: Option<AutotuneLogContext>,
}

impl TuneLogger {
    /// A logger of one tune, keeping its context when `logger` logs or writes decisions, or when
    /// the tune is `recorded`: the record's trials are the steps the context collects.
    pub fn new(logger: &Logger, recorded: bool) -> Self {
        let autotune = &logger.config().autotune;
        let logging = !matches!(autotune.logger.level, AutotuneLogLevel::Disabled);
        let read = logging || autotune.decisions_enabled() || recorded;
        Self {
            context: read.then(AutotuneLogContext::default),
        }
    }

    /// The context logged so far, when something reads it.
    pub fn context(&self) -> Option<&AutotuneLogContext> {
        self.context.as_ref()
    }

    /// Log that the candidate `name` ran for `duration`, compiling and benchmarking.
    pub fn log_step(&mut self, name: String, duration: Duration) {
        self.log_event(AutotuneLogEvent::TuningStep(name, duration));
    }

    #[cfg(not(target_family = "wasm"))]
    /// Log that the candidate `name` ended the tune under the time limit.
    pub fn log_short_circuit(&mut self, name: String) {
        self.log_event(AutotuneLogEvent::ShortCircuit(name));
    }

    /// Log the bounds the tune was measured against, and the time limit they set.
    pub fn log_bounds(&mut self, bounds: Option<Bounds>, limit: Option<Duration>) {
        if let Some(context) = self.context.as_mut() {
            *context = AutotuneLogContext {
                bounds,
                limit,
                ..core::mem::take(context)
            };
        }
    }

    /// Log what the candidates' cross-checks found.
    #[cfg(feature = "autotune-checks")]
    pub fn log_checks(&mut self, checks: Vec<CheckResult>) {
        if let Some(context) = self.context.as_mut() {
            context.checks = Some(checks);
        }
    }

    /// Write the tune's result to `logger`: the
    /// [`AutotuneDecision`](crate::tune::AutotuneDecision) for tools if decisions have a sink,
    /// and a line for humans at the logger's level. Either, both, or neither.
    ///
    /// A decided candidate at `decided_index` with no measurement still writes its decision,
    /// with no time, so the cross-checks it carries are reported; it writes no line for humans:
    /// an all-failed plan can decide an unmeasured winner, which already warned with the full
    /// results.
    pub fn log_result<K: AutotuneKeyable>(
        &self,
        logger: &mut Logger,
        key: &K,
        results: &[AutotuneResult],
        decided_index: usize,
    ) {
        let reporter = ResultReporter {
            key,
            results,
            log_context: self.context.as_ref(),
            decided_index,
        };
        reporter.report_decision(logger);
        if let Some(decided) = measured_at(results, decided_index) {
            reporter.report_log(logger, decided);
        }
    }
}

impl TuneLogger {
    fn log_event(&mut self, event: AutotuneLogEvent) {
        if let Some(context) = self.context.as_mut() {
            context.events.push(event);
        }
    }
}

/// The measurement of the candidate at `index` among `results`, if it has one.
fn measured_at(results: &[AutotuneResult], index: usize) -> Option<&AutotuneOutcome> {
    results
        .iter()
        .filter_map(AutotuneResult::measured)
        .find(|outcome| outcome.index == index)
}

/// Reports a tune's result: its key, every result, the context, and the decided candidate's
/// measurement.
struct ResultReporter<'a, K> {
    key: &'a K,
    results: &'a [AutotuneResult],
    log_context: Option<&'a AutotuneLogContext>,
    #[cfg_attr(not(std_io), allow(dead_code))]
    decided_index: usize,
}

impl<K: AutotuneKeyable> ResultReporter<'_, K> {
    /// The decision, for tools: one JSON object on the decisions' sink.
    pub fn report_decision(&self, logger: &mut Logger) {
        if !logger.config().autotune.decisions_enabled() {
            return;
        }
        #[cfg(std_io)]
        {
            use alloc::borrow::Cow;

            let decision = crate::tune::AutotuneDecision {
                key: Cow::Borrowed(self.key),
                decided_index: self.decided_index,
                decided_time: measured_at(self.results, self.decided_index)
                    .map(|decided| decided.computation.median),
                results: Cow::Borrowed(self.results),
                log_context: self.log_context.map(Cow::Borrowed),
                checks: self
                    .log_context
                    .and_then(|context| context.checks.as_deref())
                    .map(Cow::Borrowed),
            };

            let msg = serde_json::to_string(&decision).unwrap_or_else(|err| {
                format!("{{\"error\": \"Failed to serialize the autotune decision: {err}\"}}")
            });
            logger.log_autotune_decision(&msg);
        }
        #[cfg(not(std_io))]
        {
            logger.log_autotune_decision(
                &"{\"error\": \"Writing autotune decisions is not available without std_io\"}",
            );
        }
    }

    /// The line for humans, at whatever the logger's level asks for, `decided` the decided
    /// candidate's measurement.
    pub fn report_log(&self, logger: &mut Logger, decided: &AutotuneOutcome) {
        match logger.config().autotune.logger.level {
            AutotuneLogLevel::Minimal => self.report_minimal(logger, decided),
            AutotuneLogLevel::Full => self.report_full(logger, decided),
            AutotuneLogLevel::Disabled => {}
        }
    }

    fn report_minimal(&self, logger: &mut Logger, decided: &AutotuneOutcome) {
        let top_times = self
            .results
            .iter()
            .filter_map(AutotuneResult::measured)
            .map(|outcome| (outcome.index, outcome.computation.median))
            .take(TOP_TIME_COUNT)
            .collect::<alloc::vec::Vec<_>>();
        let context = self
            .log_context
            .map(|context| format!(", context: {context}"))
            .unwrap_or_default();
        logger.log_autotune(&format!(
            "Fastest result {}-{}. Top {TOP_TIME_COUNT} times: {top_times:?}{context}",
            decided.name, self.key,
        ));
    }

    fn report_full(&self, logger: &mut Logger, decided: &AutotuneOutcome) {
        let context = self.log_context.map(full_context).unwrap_or_default();
        logger.log_autotune(&format!(
            "Fastest result {}-{}.\nContext:\n{context}",
            decided.name, self.key,
        ));
        for result in self.results {
            logger.log_autotune(&outcome_line(result));
        }
    }
}

/// How many of the fastest times the minimal log line lists.
const TOP_TIME_COUNT: usize = 3;

/// The context as the full log level writes it: its bounds and limit, then its events.
fn full_context(context: &AutotuneLogContext) -> String {
    let bounds = context
        .bounds
        .as_ref()
        .map(|bounds| {
            format!(
                "Calculated bounds: {bounds:?} - limit: {:?}\n",
                context.limit
            )
        })
        .unwrap_or_default();
    format!("{bounds}{context}")
}

/// One result as the full log level writes it.
fn outcome_line(result: &AutotuneResult) -> String {
    match &result.outcome {
        Ok(outcome) => format!("{outcome}"),
        Err(err) => format!("{err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};
    use cubecl_common::profile::TimingMethod;

    fn logging() -> TuneLogger {
        TuneLogger {
            context: Some(AutotuneLogContext::default()),
        }
    }

    fn measured(index: usize, micros: u64) -> AutotuneResult {
        AutotuneResult::from_outcome(AutotuneOutcome::new(
            alloc::format!("k{index}"),
            index,
            BenchmarkComputations::new(&BenchmarkDurations::from_durations(
                TimingMethod::System,
                vec![Duration::from_micros(micros)],
            )),
        ))
    }

    #[test]
    fn a_logger_nothing_reads_logs_nothing() {
        let mut logger = TuneLogger { context: None };
        logger.log_step("k0".to_string(), Duration::from_micros(1));
        logger.log_bounds(None, Some(Duration::from_micros(1)));

        assert!(logger.context().is_none());
    }

    #[test]
    fn steps_and_short_circuits_are_logged_in_order() {
        let mut logger = logging();
        logger.log_step("k0".to_string(), Duration::from_micros(1));
        logger.log_short_circuit("k0".to_string());

        let events = &logger
            .context()
            .expect("a read logger should keep its context")
            .events;
        assert!(matches!(
            events[..],
            [
                AutotuneLogEvent::TuningStep(..),
                AutotuneLogEvent::ShortCircuit(..)
            ]
        ));
    }

    #[test]
    fn the_bounds_are_logged_beside_the_events() {
        let mut logger = logging();
        logger.log_step("k0".to_string(), Duration::from_micros(1));
        logger.log_bounds(None, Some(Duration::from_micros(5)));

        let context = logger
            .context()
            .expect("a read logger should keep its context");
        assert_eq!(context.limit, Some(Duration::from_micros(5)));
        assert_eq!(context.events.len(), 1);
    }

    #[test]
    fn the_report_names_the_decided_candidate_not_the_best_scored() {
        // Sorted by score, candidate 0 leads, but the scheduler decided candidate 1.
        let results = [measured(0, 40), measured(1, 50)];

        assert_eq!(
            measured_at(&results, 1).map(|outcome| outcome.index),
            Some(1)
        );
    }

    #[test]
    fn an_unmeasured_decision_has_nothing_to_report() {
        let results = [AutotuneResult::from_error(
            crate::tune::AutotuneError::Skip { name: "k0".into() },
        )];

        assert!(measured_at(&results, 0).is_none());
    }

    #[test]
    fn a_full_context_leads_with_its_bounds() {
        let context = AutotuneLogContext {
            bounds: Some(Bounds {
                bounds: vec![],
                launch_overhead: Duration::ZERO,
            }),
            ..Default::default()
        };

        assert!(full_context(&context).starts_with("Calculated bounds:"));
        assert_eq!(full_context(&AutotuneLogContext::default()), "");
    }
}
