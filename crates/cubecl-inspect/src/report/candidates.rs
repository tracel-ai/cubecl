use super::{AutotuneReport, CandidateOutcome};
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::Duration;

/// Every candidate of every tuner, across all the keys it raced for: the table
/// that says what a candidate list could do without.
#[derive(Clone, Debug, Serialize)]
pub struct CandidateReport {
    /// Grouped by tuner; within one, the candidates that never win first, the
    /// costliest first — by wall where the tunes were recorded, by slowdown
    /// where they were not.
    pub candidates: Vec<CandidateRow>,
}

/// One candidate's record over the keys its tuner raced.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CandidateRow {
    pub tuner: String,
    pub candidate: String,
    /// Keys it was benchmarked for.
    pub measured: u64,
    /// Keys it is the answer for.
    pub won: u64,
    /// Keys the tuner declined to run it for.
    pub skipped: u64,
    pub failed: u64,
    /// The geometric mean, over the keys it was measured for, of its score
    /// over the winner's: 1 when it always wins, 3 when it is typically three
    /// times slower than the answer. `None` when it was never measured against
    /// a measured winner.
    pub slowdown: Option<f64>,
    /// What running it cost, over the recorded tunes; `None` when none of its
    /// tunes was recorded.
    pub wall: Option<Duration>,
}

/// The running sums a [`CandidateRow`] is folded from.
struct Tally {
    row: CandidateRow,
    /// The sum of the log score ratios, whose mean is the slowdown's log.
    log_slowdown: f64,
    compared: u64,
}

impl Tally {
    fn new(tuner: &str, candidate: &str) -> Self {
        Self {
            row: CandidateRow {
                tuner: tuner.to_string(),
                candidate: candidate.to_string(),
                measured: 0,
                won: 0,
                skipped: 0,
                failed: 0,
                slowdown: None,
                wall: None,
            },
            log_slowdown: 0.0,
            compared: 0,
        }
    }

    /// The tally of `candidate` of `tuner`, started if there is none.
    fn of<'a>(
        tallies: &'a mut BTreeMap<(String, String), Tally>,
        tuner: &str,
        candidate: &str,
    ) -> &'a mut Tally {
        tallies
            .entry((tuner.to_string(), candidate.to_string()))
            .or_insert_with(|| Self::new(tuner, candidate))
    }

    fn finish(self) -> CandidateRow {
        let compared = self.compared;
        CandidateRow {
            slowdown: (compared > 0).then(|| (self.log_slowdown / compared as f64).exp()),
            ..self.row
        }
    }
}

impl From<&AutotuneReport> for CandidateReport {
    fn from(report: &AutotuneReport) -> Self {
        let mut tallies = BTreeMap::<(String, String), Tally>::new();
        for key in &report.keys {
            for trial in key.trials() {
                let tally = Tally::of(&mut tallies, &key.table.tuner, &trial.name);
                *tally.row.wall.get_or_insert_default() += trial.wall;
            }
            for result in &key.results {
                let Some(candidate) = &result.candidate else {
                    continue;
                };
                let tally = Tally::of(&mut tallies, &key.table.tuner, candidate);
                let row = &mut tally.row;
                match &result.outcome {
                    CandidateOutcome::Measured(_) => row.measured += 1,
                    CandidateOutcome::Skipped => row.skipped += 1,
                    CandidateOutcome::Failed { .. } => row.failed += 1,
                }
                if result.index == Some(key.winner) {
                    row.won += 1;
                }
                if let Some(slowdown) = key.slowdown(result) {
                    tally.log_slowdown += slowdown.ln();
                    tally.compared += 1;
                }
            }
        }

        let mut candidates: Vec<CandidateRow> = tallies.into_values().map(Tally::finish).collect();
        candidates.sort_by(|a, b| {
            a.tuner
                .cmp(&b.tuner)
                .then(a.won.min(1).cmp(&b.won.min(1)))
                .then(b.wall.cmp(&a.wall))
                .then(
                    b.slowdown
                        .unwrap_or(0.0)
                        .total_cmp(&a.slowdown.unwrap_or(0.0)),
                )
                .then(a.candidate.cmp(&b.candidate))
        });
        Self { candidates }
    }
}
