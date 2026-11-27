use alloc::string::String;
use core::time::Duration;
use cubecl_common::profile::{Instant, ProfileDuration, ProfileTicks, TimingMethod};

use crate::tune::adaptive::sample_keeper::{SampleKeeper, SampleSummary};
use crate::tune::drive::sample_time::measured;
use crate::tune::{AutotuneError, AutotuneOutcome, AutotuneResult};

/// Tracks one candidate kernel of a batch and the evidence gathered about it so far.
#[derive(Debug)]
pub struct CandidateTracker {
    index: usize,
    name: String,
    samples: SampleKeeper,
    method: Option<TimingMethod>,
    error: Option<AutotuneError>,
    elapsed: Duration,
    live: bool,
}

/// What a [`CandidateTracker`] knows of its candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateState<'a> {
    /// The candidate's index in its set.
    pub index: usize,
    pub name: &'a str,
    /// Whether it is still in the round.
    pub live: bool,
    pub summary: SampleSummary,
    /// What it cost so far, where steps are timed.
    pub elapsed: Duration,
    /// Whether it was never measured and never failed: skipped, or not reached.
    pub unmeasured: bool,
}

impl CandidateState<'_> {
    /// Its best time while it is still in the round.
    pub fn live_best(&self) -> Option<Duration> {
        self.summary.best.filter(|_| self.live)
    }
}

impl CandidateTracker {
    /// A tracker of the candidate at `index`, named `name`, live and unmeasured.
    pub fn new(index: usize, name: String) -> Self {
        Self {
            index,
            name,
            samples: SampleKeeper::default(),
            method: None,
            error: None,
            elapsed: Duration::ZERO,
            live: true,
        }
    }

    pub fn state(&self) -> CandidateState<'_> {
        CandidateState {
            index: self.index,
            name: &self.name,
            live: self.live,
            summary: self.samples.summary(),
            elapsed: self.elapsed,
            unmeasured: self.samples.summary().count == 0 && self.error.is_none(),
        }
    }

    /// Whether it is still in the round with `required` samples under `limit`.
    pub fn confirmed_under(&self, limit: Duration, required: usize) -> bool {
        self.live && self.samples.confirmed_under(limit, required)
    }

    /// Whether any of its samples landed under `limit`.
    pub fn any_under(&self, limit: Duration) -> bool {
        self.samples.any_under(limit)
    }

    /// Track `duration` as its latest sample.
    pub fn track_duration(&mut self, duration: Duration) {
        self.samples.keep(duration);
    }

    /// Track what a resolved sample measured, or disqualify the candidate where it measured no
    /// time: recorded as a zero, it would be the fastest sample in the round and would short
    /// circuit on it.
    pub fn track_ticks(&mut self, ticks: Option<ProfileTicks>) {
        match measured(ticks) {
            Some(duration) => self.track_duration(duration),
            None => self.fail(AutotuneError::NotMeasured {
                name: self.name.clone(),
            }),
        }
    }

    /// `profile`, a sample of the candidate just queued, its timing method kept.
    pub fn track_queued(&mut self, profile: ProfileDuration) -> ProfileDuration {
        self.method.get_or_insert(profile.timing_method());
        profile
    }

    /// Add `spent` to what the candidate cost, when it was timed.
    pub fn track_cost(&mut self, spent: Option<Duration>) {
        self.elapsed += spent.unwrap_or_default();
    }

    /// Add the time since `launched` to what the candidate cost, when it was timed.
    pub fn track_cost_since(&mut self, launched: Option<Instant>) {
        self.track_cost(launched.map(|launched| launched.elapsed()));
    }

    /// Take it out of the round on `error`: a candidate that failed at any point is disqualified
    /// even if earlier samples succeeded, so a kernel that crashes sporadically can never win on
    /// its good runs.
    pub fn fail(&mut self, error: AutotuneError) {
        self.error = Some(error);
        self.live = false;
    }

    /// [Fail](Self::fail) the candidate, answering that it did not survive.
    pub fn disqualify(&mut self, error: AutotuneError) -> bool {
        self.fail(error);
        false
    }

    /// Take it out of the round on the evidence, keeping its samples.
    pub fn eliminate(&mut self) {
        self.live = false;
    }

    /// Leave it out of the batch before its first sample. With no samples and no error, it comes
    /// out as [skipped](AutotuneError::Skip).
    pub fn skip(&mut self) {
        self.live = false;
    }

    pub fn into_result(self) -> AutotuneResult {
        if let Some(error) = self.error {
            return AutotuneResult::from_error(error);
        }
        if self.samples.summary().count == 0 {
            return AutotuneResult::from_error(AutotuneError::Skip { name: self.name });
        }
        let computation = self
            .samples
            .computation(self.method.unwrap_or(TimingMethod::System));
        AutotuneResult::from_outcome(AutotuneOutcome::new(self.name, self.index, computation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker(samples: &[u64]) -> CandidateTracker {
        let mut tracker = CandidateTracker::new(1, "k1".into());
        for micros in samples {
            tracker.track_duration(Duration::from_micros(*micros));
        }
        tracker
    }

    #[test]
    fn an_unreached_candidate_is_skipped() {
        assert!(matches!(
            tracker(&[]).into_result().outcome,
            Err(AutotuneError::Skip { .. })
        ));
    }

    #[test]
    fn a_candidate_that_failed_late_is_disqualified_despite_good_samples() {
        let mut tracker = tracker(&[10, 10, 10]);
        tracker.fail(AutotuneError::NotMeasured { name: "k1".into() });

        assert!(matches!(
            tracker.into_result().outcome,
            Err(AutotuneError::NotMeasured { .. })
        ));
    }

    #[test]
    fn a_sample_that_measured_no_time_disqualifies() {
        let mut tracker = tracker(&[10]);

        tracker.track_ticks(None);

        assert!(!tracker.state().live);
    }

    #[test]
    fn a_sampled_candidate_reports_its_reliable_samples() {
        let outcome = tracker(&[50, 60, 70])
            .into_result()
            .outcome
            .expect("a sampled candidate should succeed");

        assert_eq!(outcome.computation.min, Duration::from_micros(60));
    }

    #[test]
    fn an_eliminated_candidate_has_no_live_best() {
        let mut tracker = tracker(&[10]);
        tracker.eliminate();

        assert_eq!(tracker.state().live_best(), None);
    }

    #[test]
    fn a_skipped_candidate_is_unmeasured() {
        let mut tracker = tracker(&[]);
        tracker.skip();

        assert!(tracker.state().unmeasured);
    }

    #[test]
    fn its_cost_adds_up_where_it_was_timed() {
        let mut tracker = tracker(&[]);
        tracker.track_cost(Some(Duration::from_micros(3)));
        tracker.track_cost(None);

        assert_eq!(tracker.state().elapsed, Duration::from_micros(3));
    }
}
