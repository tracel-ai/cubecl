use super::fields::Fields;
use super::paint::Role;
use super::table::{Align, Table};
use super::{KeyText, Maybe, Micros, Percent, Ratio, Text, Via, Wall};
use crate::report::{
    AutotuneReport, CandidateOutcome, CandidateReport, CandidateResult, Fate, TunedKey,
    UntypedTuneRecord,
};
use cubecl_environment::records::Stamped;
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
                Maybe(key.tune.as_ref().map(|tune| Wall(tune.compiling))).to_string(),
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
        match key.trace() {
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
        let fields = Fields::new(&[
            "id", "tuner", "key", "checksum", "winner", "tuned", "roofline", "limit",
        ]);
        fields.line(f, "id", key.id)?;
        fields.line(
            f,
            "tuner",
            format_args!(
                "{} on {}{} (cubecl {})",
                key.table.tuner,
                key.table.device,
                Via(key.table.fingerprint.as_deref()),
                key.table.version
            ),
        )?;
        fields.line(f, "key", KeyText(&key.key))?;
        fields.line(f, "checksum", &key.checksum)?;
        fields.line(
            f,
            "winner",
            format_args!(
                "{} (#{}), margin {}",
                key.winner_name(),
                key.winner,
                Ratio(key.margin())
            ),
        )?;
        if let Some(tune) = &key.tune {
            let (stamp, record) = (&tune.trace.stamp, &tune.trace.record);
            let discarded = match record.launches_discarded {
                true => ", with its launches discarded",
                false => "",
            };
            fields.line(
                f,
                "tuned",
                format_args!(
                    "in {} at +{:.1} s of session {}: {} compiling, {} wasted{discarded}",
                    Wall(record.wall),
                    stamp.offset.as_secs_f64(),
                    stamp.session,
                    Wall(tune.compiling),
                    Wall(key.wasted().unwrap_or_default())
                ),
            )?;
        }
        if let Some(roofline) = key.roofline() {
            fields.line(
                f,
                "roofline",
                format_args!(
                    "{}, the winner at {} of it",
                    Micros(roofline),
                    Percent(key.efficiency())
                ),
            )?;
        }
        if let Some(limit) = key.limit {
            let met = match key.met_limit() {
                Some(true) => "met",
                Some(false) => "not met",
                None => "-",
            };
            fields.line(f, "limit", format_args!("{} ({met})", Micros(limit)))?;
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
            let cells = ResultCells {
                key,
                result: key.result(&planned.name),
                winner: planned.index == key.winner,
            };
            let (ran, wall, status, role) = match key.fate(planned) {
                Fate::Ran { order } => (
                    format!("{}{}", order + 1, cells.marker()),
                    Wall(key.trials()[order].wall).to_string(),
                    cells.outcome(),
                    cells.role(),
                ),
                Fate::CutShort => not_run("cut short"),
                Fate::NotReached => not_run("not reached"),
                Fate::Excluded => not_run("excluded"),
            };
            let placements = |cell: fn(&Placement) -> String| match planned.groups.as_slice() {
                [] => "-".to_string(),
                groups => groups.iter().map(cell).collect::<Vec<_>>().join(", "),
            };
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
                    cells.median(),
                    cells.slowdown(),
                ],
            );
        }
        fmt::Display::fmt(&table, f)?;
        let stopper = key
            .trace()
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
struct RunOrder<'a>(&'a TunedKey, &'a Stamped<UntypedTuneRecord>);

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
            let cells = ResultCells {
                key,
                result,
                winner: result.is_some_and(|result| result.index == Some(key.winner)),
            };
            table.row_in(
                cells.role(),
                vec![
                    format!("{}{}", order + 1, cells.marker()),
                    trial.name.clone(),
                    Wall(trial.wall).to_string(),
                    cells.outcome().to_string(),
                    cells.median(),
                    cells.slowdown(),
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
            let cells = ResultCells {
                key,
                result: Some(result),
                winner: result.index == Some(key.winner),
            };
            let mut row = vec![
                format!("{}{}", rank + 1, cells.marker()),
                result.candidate.as_deref().unwrap_or("-").to_string(),
                Maybe(result.index).to_string(),
                cells.outcome().to_string(),
                cells.median(),
            ];
            match &result.outcome {
                CandidateOutcome::Measured(computations) => row.extend([
                    Micros(computations.min).to_string(),
                    Micros(computations.max).to_string(),
                    computations.score().to_string(),
                ]),
                CandidateOutcome::Skipped | CandidateOutcome::Failed { .. } => {
                    row.extend(["-".to_string(), "-".to_string(), "-".to_string()])
                }
            }
            table.row_in(cells.role(), row);
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
        .group("tuner");
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

/// The cells a planned candidate that never ran fills before its result's:
/// no position and no wall, `outcome` saying why, the row muted.
fn not_run(outcome: &'static str) -> (String, String, &'static str, Option<Role>) {
    (
        "-  ".to_string(),
        "-".to_string(),
        outcome,
        Some(Role::Idle),
    )
}

/// What a candidate's result fills in the tables of one key's tune: how it
/// ended, against the winner, and whether it is the winner.
struct ResultCells<'a> {
    key: &'a TunedKey,
    /// `None` for a candidate the stored results do not name.
    result: Option<&'a CandidateResult>,
    winner: bool,
}

impl ResultCells<'_> {
    /// What follows its position: a star on the winner, the same width
    /// elsewhere, so the column stays aligned.
    fn marker(&self) -> &'static str {
        match self.winner {
            true => " *",
            false => "  ",
        }
    }

    fn outcome(&self) -> &'static str {
        match self.result.map(|result| &result.outcome) {
            Some(CandidateOutcome::Measured(_)) => "measured",
            Some(CandidateOutcome::Skipped) => "skipped",
            Some(CandidateOutcome::Failed { .. }) => "failed",
            None => "-",
        }
    }

    fn median(&self) -> String {
        match self.result.map(|result| &result.outcome) {
            Some(CandidateOutcome::Measured(computations)) => {
                Micros(computations.median).to_string()
            }
            _ => "-".to_string(),
        }
    }

    /// Its score over the winner's.
    fn slowdown(&self) -> String {
        Ratio(self.result.and_then(|result| self.key.slowdown(result))).to_string()
    }

    /// The winner in bold, a candidate with nothing measured muted.
    fn role(&self) -> Option<Role> {
        match (self.winner, self.result.map(|result| &result.outcome)) {
            (true, _) => Some(Role::Strong),
            (false, Some(CandidateOutcome::Measured(_))) => None,
            (false, _) => Some(Role::Idle),
        }
    }
}
