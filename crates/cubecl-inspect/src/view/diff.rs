use super::table::{Align, Table};
use super::{KeyText, Ratio, Text, Wall};
use crate::report::{EnvironmentDiff, KeyChange};
use std::fmt;

/// How many kept keys the wall comparison lists, largest change first.
const WALL_ROWS: usize = 20;

impl fmt::Display for Text<'_, EnvironmentDiff> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let diff = self.0;
        let count = |change| diff.with(change).count();
        writeln!(f, "before  {}", diff.before.display())?;
        writeln!(f, "after   {}", diff.after.display())?;
        writeln!(
            f,
            "{} keys: {} added, {} removed, {} winners changed ({} under a changed candidate list), {} kept",
            diff.keys.len(),
            count(KeyChange::Added),
            count(KeyChange::Removed),
            count(KeyChange::WinnerChanged),
            diff.with(KeyChange::WinnerChanged)
                .filter(|key| key.candidates_changed())
                .count(),
            count(KeyChange::Kept),
        )?;

        let mut changed = Table::new(&[
            ("tuner", Align::Left),
            ("before", Align::Left),
            ("after", Align::Left),
            ("margins", Align::Right),
            ("list", Align::Left),
            ("key", Align::Left),
        ]);
        for key in diff.with(KeyChange::WinnerChanged) {
            let (Some(before), Some(after)) = (&key.before, &key.after) else {
                continue;
            };
            changed.row(vec![
                key.tuner.clone(),
                before.winner.clone(),
                after.winner.clone(),
                format!("{} / {}", Ratio(before.margin), Ratio(after.margin)),
                if key.candidates_changed() {
                    "changed"
                } else {
                    "same"
                }
                .to_string(),
                KeyText(&key.key).to_string(),
            ]);
        }
        section(f, "winners changed", changed)?;

        for (title, change) in [("added", KeyChange::Added), ("removed", KeyChange::Removed)] {
            let mut table = Table::new(&[
                ("tuner", Align::Left),
                ("winner", Align::Left),
                ("key", Align::Left),
            ]);
            for key in diff.with(change) {
                let answer = key.after.as_ref().or(key.before.as_ref());
                table.row(vec![
                    key.tuner.clone(),
                    answer
                        .map(|answer| answer.winner.clone())
                        .unwrap_or_default(),
                    KeyText(&key.key).to_string(),
                ]);
            }
            section(f, title, table)?;
        }

        let mut walls: Vec<_> = diff
            .keys
            .iter()
            .filter_map(|key| Some((key, key.walls()?)))
            .collect();
        walls.sort_by_key(|(_, (before, after))| std::cmp::Reverse(before.abs_diff(*after)));
        let mut table = Table::new(&[
            ("tuner", Align::Left),
            ("before", Align::Right),
            ("after", Align::Right),
            ("key", Align::Left),
        ]);
        for (key, (before, after)) in walls.into_iter().take(WALL_ROWS) {
            table.row(vec![
                key.tuner.clone(),
                Wall(before).to_string(),
                Wall(after).to_string(),
                KeyText(&key.key).to_string(),
            ]);
        }
        section(f, "tune walls, largest change first", table)
    }
}

/// A titled table, left out when it has no rows.
fn section(f: &mut fmt::Formatter<'_>, title: &str, table: Table) -> fmt::Result {
    if table.is_empty() {
        return Ok(());
    }
    writeln!(f, "\n{title}:")?;
    write!(f, "{table}")
}
