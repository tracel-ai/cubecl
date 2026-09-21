use super::table::{Align, Table};
use super::{KeyText, Maybe, Micros, Ratio, Text, Wall};
use crate::report::{AutotuneReport, CandidateOutcome, CandidateReport, TuneTrace, TunedKey};
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
        ]);
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
        write!(f, "{table}")?;
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
        write!(f, "{}", Header(key))?;
        writeln!(f)?;
        match &key.trace {
            Some(trace) => write!(f, "{}", RunOrder(key, trace))?,
            None => write!(f, "{}", Ranking(key))?,
        }
        write!(f, "{}", Failures(key))
    }
}

/// What a key is, what it runs, and what bounded its tune.
struct Header<'a>(&'a TunedKey);

impl fmt::Display for Header<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = self.0;
        writeln!(f, "id        {}", key.id)?;
        writeln!(
            f,
            "tuner     {} on {} (cubecl {})",
            key.table.tuner, key.table.device, key.table.version
        )?;
        writeln!(f, "key       {}", KeyText(&key.key))?;
        writeln!(f, "checksum  {}", key.checksum)?;
        writeln!(
            f,
            "winner    {} (#{}), margin {}",
            key.winner_name(),
            key.winner,
            Ratio(key.margin())
        )?;
        if let Some(trace) = &key.trace {
            let record = &trace.record;
            write!(
                f,
                "tuned     in {} at +{:.1} s of session {}: {} compiling, {} wasted",
                Wall(record.wall),
                trace.stamp.offset.as_secs_f64(),
                trace.stamp.session,
                Wall(key.compiling.unwrap_or_default()),
                Wall(key.wasted().unwrap_or_default())
            )?;
            if record.dry_run {
                write!(f, ", in a dry run")?;
            }
            writeln!(f)?;
        }
        if let Some(roofline) = key.roofline() {
            writeln!(
                f,
                "roofline  {}, the winner at {} of it",
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
            writeln!(f, "limit     {} ({met})", Micros(limit))?;
        }
        Ok(())
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
        ]);
        for (order, trial) in trace.trials.iter().enumerate() {
            let result = key.result(&trial.name);
            let winner = result.is_some_and(|result| result.index == Some(key.winner));
            let marker = if winner { " *" } else { "" };
            let (outcome, median) = match result.map(|result| &result.outcome) {
                Some(CandidateOutcome::Measured(computations)) => {
                    ("measured", Micros(computations.median).to_string())
                }
                Some(CandidateOutcome::Failed { .. }) => ("failed", "-".to_string()),
                Some(CandidateOutcome::Skipped) | None => ("-", "-".to_string()),
            };
            table.row(vec![
                format!("{}{marker}", order + 1),
                trial.name.clone(),
                Wall(trial.wall).to_string(),
                outcome.to_string(),
                median,
                Ratio(result.and_then(|result| key.slowdown(result))).to_string(),
            ]);
        }
        write!(f, "{table}")?;
        writeln!(f, "in the order they ran; * runs.")?;

        let not_run: Vec<&str> = key
            .results
            .iter()
            .filter_map(|result| result.candidate.as_deref())
            .filter(|name| !trace.trials.iter().any(|trial| trial.name == *name))
            .collect();
        if !not_run.is_empty() {
            write!(f, "\nnot run ({})", not_run.len())?;
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
        ]);
        for (rank, result) in key.results.iter().enumerate() {
            let marker = if result.index == Some(key.winner) {
                " *"
            } else {
                ""
            };
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
            table.row(row);
        }
        write!(f, "{table}")?;
        writeln!(
            f,
            "ranked by score (lower is better); * runs. The order they ran in was not recorded."
        )
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
            writeln!(f, "\nfailed ({}): {}", names.len(), reason.trim_end())?;
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
        ]);
        for row in &self.0.candidates {
            table.row(vec![
                row.tuner.clone(),
                row.candidate.clone(),
                row.measured.to_string(),
                row.won.to_string(),
                row.skipped.to_string(),
                row.failed.to_string(),
                Ratio(row.slowdown).to_string(),
                Maybe(row.wall.map(Wall)).to_string(),
            ]);
        }
        write!(f, "{table}")?;
        writeln!(
            f,
            "slowdown: geometric mean of the candidate's score over the winner's, where both were measured;\n\
             wall: compiling and benchmarking it, over the recorded tunes."
        )
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
