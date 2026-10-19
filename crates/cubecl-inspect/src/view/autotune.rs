use super::paint::Role;
use super::table::{Align, Table};
use super::{KeyText, Maybe, Micros, Ratio, Text, Wall};
use crate::report::{AutotuneReport, CandidateOutcome, CandidateReport, Fate, TuneTrace, TunedKey};
use cubecl_server::tune::Placement;
use std::fmt;

impl fmt::Display for Text<'_, AutotuneReport> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let report = self.0;
        let mut table = Table::new(&[
            ("id", Align::Left),
            ("tuner", Align::Left),
            ("winner", Align::Left),
            ("wall", Align::Right),
            ("compiling", Align::Right),
            ("wasted", Align::Right),
            ("measured", Align::Right),
            ("failed", Align::Right),
            ("margin", Align::Right),
            ("key", Align::Left),
        ])
        .paint(&["wall", "compiling", "wasted"], Role::Measure);
        for key in &report.keys {
            table.row(vec![
                key.id.to_string(),
                key.table.tuner.clone(),
                key.winner_name(),
                Maybe(key.wall().map(Wall)).to_string(),
                Maybe(key.compiling.map(Wall)).to_string(),
                Maybe(key.wasted().map(Wall)).to_string(),
                key.measured().to_string(),
                key.failed().to_string(),
                Ratio(key.margin()).to_string(),
                KeyText(&key.key).to_string(),
            ]);
        }
        fmt::Display::fmt(&table, f)?;
        let recorded: Vec<_> = report.keys.iter().filter_map(TunedKey::wall).collect();
        write!(f, "{} keys", report.keys.len())?;
        if !recorded.is_empty() {
            write!(
                f,
                ", {} of them recorded, tuned in {}",
                recorded.len(),
                Wall(recorded.iter().sum())
            )?;
        }
        if report.undecoded > 0 {
            write!(f, ", {} entries that did not decode", report.undecoded)?;
        }
        writeln!(f)
    }
}

impl fmt::Display for Text<'_, TunedKey> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = self.0;
        fmt::Display::fmt(&Header(key), f)?;
        writeln!(f)?;
        match &key.trace {
            Some(_) if !key.plan().is_empty() => fmt::Display::fmt(&PlanOrder(key), f)?,
            Some(trace) => fmt::Display::fmt(&RunOrder(key, trace), f)?,
            None => fmt::Display::fmt(&Ranking(key), f)?,
        }
        fmt::Display::fmt(&Failures(key), f)
    }
}

/// What a key is, what it runs, and what bounded its tune.
struct Header<'a>(&'a TunedKey);

impl fmt::Display for Header<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = self.0;
        Role::Label.write(f, "id        ")?;
        writeln!(f, "{}", key.id)?;
        Role::Label.write(f, "tuner     ")?;
        writeln!(
            f,
            "{} on {} (cubecl {})",
            key.table.tuner, key.table.device, key.table.version
        )?;
        Role::Label.write(f, "key       ")?;
        writeln!(f, "{}", KeyText(&key.key))?;
        Role::Label.write(f, "checksum  ")?;
        writeln!(f, "{}", key.checksum)?;
        Role::Label.write(f, "winner    ")?;
        writeln!(
            f,
            "{} (#{}), margin {}",
            key.winner_name(),
            key.winner,
            Ratio(key.margin())
        )?;
        if let Some(trace) = &key.trace {
            let record = &trace.record;
            Role::Label.write(f, "tuned     ")?;
            write!(
                f,
                "in {} at +{:.1} s of session {}: {} compiling, {} wasted",
                Wall(record.wall),
                trace.stamp.offset.as_secs_f64(),
                trace.stamp.session,
                Wall(key.compiling.unwrap_or_default()),
                Wall(key.wasted().unwrap_or_default())
            )?;
            if record.launches_discarded {
                write!(f, ", with its launches discarded")?;
            }
            writeln!(f)?;
        }
        if let Some(roofline) = key.roofline() {
            Role::Label.write(f, "roofline  ")?;
            writeln!(
                f,
                "{}, the winner at {} of it",
                Micros(roofline),
                Percent(key.efficiency())
            )?;
        }
        if let Some(limit) = key.limit {
            let met = match key.met_limit() {
                Some(true) => "met",
                Some(false) => "not met",
                None => "-",
            };
            Role::Label.write(f, "limit     ")?;
            writeln!(f, "{} ({met})", Micros(limit))?;
        }
        Ok(())
    }
}

/// A tune recorded with its plan: every candidate in the order the plan
/// released it, where it sat in its groups, and what became of it.
struct PlanOrder<'a>(&'a TunedKey);

impl fmt::Display for PlanOrder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = self.0;
        let mut table = Table::new(&[
            ("ran", Align::Right),
            ("batch", Align::Right),
            ("candidate", Align::Left),
            ("group", Align::Left),
            ("level", Align::Right),
            ("priority", Align::Right),
            ("wall", Align::Right),
            ("outcome", Align::Left),
            ("median", Align::Right),
            ("vs winner", Align::Right),
        ])
        .paint(&["wall", "median"], Role::Measure);
        for planned in key.plan() {
            let winner = planned.index == key.winner;
            let result = key.result(&planned.name);
            let fate = key.fate(planned);
            let (ran, wall, status) = match fate {
                Fate::Ran { order } => (
                    format!("{}{}", order + 1, if winner { " *" } else { "  " }),
                    Wall(key.trials()[order].wall).to_string(),
                    match result.map(|result| &result.outcome) {
                        Some(CandidateOutcome::Measured(_)) => "measured",
                        Some(CandidateOutcome::Failed { .. }) => "failed",
                        Some(CandidateOutcome::Skipped) | None => "-",
                    },
                ),
                Fate::CutShort => ("-  ".to_string(), "-".to_string(), "cut short"),
                Fate::NotReached => ("-  ".to_string(), "-".to_string(), "not reached"),
                Fate::Excluded => ("-  ".to_string(), "-".to_string(), "excluded"),
            };
            let median = match result.map(|result| &result.outcome) {
                Some(CandidateOutcome::Measured(computations)) => {
                    Micros(computations.median).to_string()
                }
                _ => "-".to_string(),
            };
            let placements = |cell: fn(&Placement) -> String| match planned.groups.as_slice() {
                [] => "-".to_string(),
                groups => groups.iter().map(cell).collect::<Vec<_>>().join(", "),
            };
            let role = match fate {
                Fate::Ran { .. } => Outcome::of(winner, result.map(|result| &result.outcome)),
                Fate::CutShort | Fate::NotReached | Fate::Excluded => Outcome::NotMeasured,
            }
            .role();
            table.row_in(
                role,
                vec![
                    ran,
                    Maybe(planned.batch.map(|batch| batch + 1)).to_string(),
                    planned.name.clone(),
                    placements(|placement| match placement.ordered {
                        true => format!("{} (ordered)", placement.group),
                        false => placement.group.clone(),
                    }),
                    placements(|placement| placement.group_priority.to_string()),
                    placements(|placement| placement.priority.to_string()),
                    wall,
                    status.to_string(),
                    median,
                    Ratio(result.and_then(|result| key.slowdown(result))).to_string(),
                ],
            );
        }
        fmt::Display::fmt(&table, f)?;
        let stopper = key
            .trace
            .as_ref()
            .and_then(|trace| trace.record.short_circuit.as_ref())
            .map(|stopper| format!("; the tune stopped once {stopper} met the limit"))
            .unwrap_or_default();
        Role::Note.write(
            f,
            format_args!(
                "in the order the plan released them; * runs{stopper}.\n\
                 level: the group's priority, higher groups first; priority: the candidate's within it; either negative excludes it;\n\
                 a batch runs only when every batch before it found nothing valid."
            ),
        )?;
        writeln!(f)
    }
}

/// A recorded tune: the trials in the order they ran, then what never ran.
struct RunOrder<'a>(&'a TunedKey, &'a TuneTrace);

impl fmt::Display for RunOrder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (key, trace) = (self.0, &self.1.record);
        let mut table = Table::new(&[
            ("#", Align::Right),
            ("candidate", Align::Left),
            ("wall", Align::Right),
            ("outcome", Align::Left),
            ("median", Align::Right),
            ("vs winner", Align::Right),
        ])
        .paint(&["wall", "median"], Role::Measure);
        for (order, trial) in trace.trials.iter().enumerate() {
            let result = key.result(&trial.name);
            let winner = result.is_some_and(|result| result.index == Some(key.winner));
            let marker = if winner { " *" } else { "  " };
            let (outcome, median) = match result.map(|result| &result.outcome) {
                Some(CandidateOutcome::Measured(computations)) => {
                    ("measured", Micros(computations.median).to_string())
                }
                Some(CandidateOutcome::Failed { .. }) => ("failed", "-".to_string()),
                Some(CandidateOutcome::Skipped) | None => ("-", "-".to_string()),
            };
            table.row_in(
                Outcome::of(winner, result.map(|result| &result.outcome)).role(),
                vec![
                    format!("{}{marker}", order + 1),
                    trial.name.clone(),
                    Wall(trial.wall).to_string(),
                    outcome.to_string(),
                    median,
                    Ratio(result.and_then(|result| key.slowdown(result))).to_string(),
                ],
            );
        }
        fmt::Display::fmt(&table, f)?;
        Role::Note.write(f, "in the order they ran; * runs.")?;
        writeln!(f)?;

        let not_run: Vec<&str> = key
            .results
            .iter()
            .filter_map(|result| result.candidate.as_deref())
            .filter(|name| !trace.trials.iter().any(|trial| trial.name == *name))
            .collect();
        if !not_run.is_empty() {
            writeln!(f)?;
            Role::Title.write(f, format_args!("not run ({})", not_run.len()))?;
            if let Some(stopper) = &trace.short_circuit {
                write!(f, ", the tune stopped once {stopper} met the limit")?;
            }
            writeln!(f, ":\n  {}", not_run.join(", "))?;
        }
        Ok(())
    }
}

/// An unrecorded tune: every result, ranked the way cubecl stored them.
struct Ranking<'a>(&'a TunedKey);

impl fmt::Display for Ranking<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = self.0;
        let mut table = Table::new(&[
            ("rank", Align::Right),
            ("candidate", Align::Left),
            ("index", Align::Right),
            ("outcome", Align::Left),
            ("median", Align::Right),
            ("min", Align::Right),
            ("max", Align::Right),
            ("score", Align::Right),
        ])
        .paint(&["median", "min", "max"], Role::Measure);
        for (rank, result) in key.results.iter().enumerate() {
            let winner = result.index == Some(key.winner);
            let marker = if winner { " *" } else { "  " };
            let name = result.candidate.as_deref().unwrap_or("-");
            let mut row = vec![
                format!("{}{marker}", rank + 1),
                name.to_string(),
                Maybe(result.index).to_string(),
            ];
            match &result.outcome {
                CandidateOutcome::Measured(computations) => row.extend([
                    "measured".to_string(),
                    Micros(computations.median).to_string(),
                    Micros(computations.min).to_string(),
                    Micros(computations.max).to_string(),
                    computations.score().to_string(),
                ]),
                CandidateOutcome::Skipped => row.extend(unmeasured("skipped")),
                CandidateOutcome::Failed { .. } => row.extend(unmeasured("failed")),
            }
            table.row_in(Outcome::of(winner, Some(&result.outcome)).role(), row);
        }
        fmt::Display::fmt(&table, f)?;
        Role::Note.write(
            f,
            "ranked by score (lower is better); * runs. The order they ran in was not recorded.",
        )?;
        writeln!(f)
    }
}

/// Every failure's reason once, with the candidates that failed for it: one
/// reason is usually a whole family's.
struct Failures<'a>(&'a TunedKey);

impl fmt::Display for Failures<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut failures: Vec<(&str, Vec<&str>)> = Vec::new();
        for result in &self.0.results {
            let CandidateOutcome::Failed { reason } = &result.outcome else {
                continue;
            };
            let name = result.candidate.as_deref().unwrap_or("-");
            match failures.iter_mut().find(|(seen, _)| seen == reason) {
                Some((_, names)) => names.push(name),
                None => failures.push((reason, vec![name])),
            }
        }
        for (reason, names) in failures {
            writeln!(f)?;
            Role::Title.write(f, format_args!("failed ({})", names.len()))?;
            writeln!(f, ": {}", reason.trim_end())?;
            writeln!(f, "  {}", names.join(", "))?;
        }
        Ok(())
    }
}

impl fmt::Display for Text<'_, CandidateReport> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut table = Table::new(&[
            ("tuner", Align::Left),
            ("candidate", Align::Left),
            ("measured", Align::Right),
            ("won", Align::Right),
            ("skipped", Align::Right),
            ("failed", Align::Right),
            ("slowdown", Align::Right),
            ("wall", Align::Right),
        ])
        .paint(&["wall"], Role::Measure)
        .group(0);
        for row in &self.0.candidates {
            let role = match (row.won, row.measured) {
                (1.., _) => Some(Role::Strong),
                (_, 0) => Some(Role::Idle),
                _ => None,
            };
            table.row_in(
                role,
                vec![
                    row.tuner.clone(),
                    row.candidate.clone(),
                    row.measured.to_string(),
                    row.won.to_string(),
                    row.skipped.to_string(),
                    row.failed.to_string(),
                    Ratio(row.slowdown).to_string(),
                    Maybe(row.wall.map(Wall)).to_string(),
                ],
            );
        }
        fmt::Display::fmt(&table, f)?;
        Role::Note.write(f, "slowdown: geometric mean of the candidate's score over the winner's, where both were measured;\n\
             wall: compiling and benchmarking it, over the recorded tunes;\n\
             a candidate that won is bold, one never measured is muted.")?;
        writeln!(f)
    }
}

/// How a candidate fared in one tune, as far as painting its row goes.
enum Outcome {
    Won,
    Measured,
    NotMeasured,
}

impl Outcome {
    fn of(winner: bool, outcome: Option<&CandidateOutcome>) -> Self {
        match (winner, outcome) {
            (true, _) => Self::Won,
            (false, Some(CandidateOutcome::Measured(_))) => Self::Measured,
            (false, _) => Self::NotMeasured,
        }
    }

    fn role(self) -> Option<Role> {
        match self {
            Self::Won => Some(Role::Strong),
            Self::Measured => None,
            Self::NotMeasured => Some(Role::Idle),
        }
    }
}

/// The trailing cells of a result nothing was measured for.
fn unmeasured(outcome: &str) -> [String; 5] {
    [
        outcome.to_string(),
        "-".to_string(),
        "-".to_string(),
        "-".to_string(),
        "-".to_string(),
    ]
}

/// A fraction as a percentage, `-` when there is none.
struct Percent(Option<f64>);

impl fmt::Display for Percent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(fraction) => write!(f, "{:.0}%", fraction * 100.0),
            None => f.write_str("-"),
        }
    }
}
