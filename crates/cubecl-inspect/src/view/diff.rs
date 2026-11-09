use super::fields::Fields;
use super::paint::Role;
use super::table::{Align, Table};
use super::{KeyText, Ratio, Text, Via, Wall};
use crate::report::{EnvironmentDiff, KeyChange, KeyDiff};
use std::collections::BTreeSet;
use std::fmt;

/// How many kept keys the wall comparison lists, largest change first.
const WALL_ROWS: usize = 20;

impl fmt::Display for Text<'_, EnvironmentDiff> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let diff = self.0;
        let place = Place::new(diff);
        let count = |change| diff.with(change).count();
        let fields = Fields::new(&["before", "after"]);
        fields.line(f, "before", diff.before.display())?;
        fields.line(f, "after", diff.after.display())?;
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

        let mut changed = Table::new(&place.columns(&[
            ("tuner", Align::Left),
            ("before", Align::Left),
            ("after", Align::Left),
            ("margins", Align::Right),
            ("list", Align::Left),
            ("key", Align::Left),
        ]));
        for key in diff.with(KeyChange::WinnerChanged) {
            let (Some(before), Some(after)) = (&key.before, &key.after) else {
                continue;
            };
            changed.row(place.cells(
                key,
                vec![
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
            ],
            ));
        }
        section(f, "winners changed", changed)?;

        for (title, change) in [("added", KeyChange::Added), ("removed", KeyChange::Removed)] {
            let mut table = Table::new(&place.columns(&[
                ("tuner", Align::Left),
                ("winner", Align::Left),
                ("key", Align::Left),
            ]));
            for key in diff.with(change) {
                let answer = key.after.as_ref().or(key.before.as_ref());
                table.row(place.cells(
                    key,
                    vec![
                    key.tuner.clone(),
                    answer
                        .map(|answer| answer.winner.clone())
                        .unwrap_or_default(),
                    KeyText(&key.key).to_string(),
                ],
                ));
            }
            section(f, title, table)?;
        }

        let mut walls: Vec<_> = diff
            .keys
            .iter()
            .filter_map(|key| Some((key, key.walls()?)))
            .collect();
        walls.sort_by_key(|(_, (before, after))| std::cmp::Reverse(before.abs_diff(*after)));
        let mut table = Table::new(&place.columns(&[
            ("tuner", Align::Left),
            ("before", Align::Right),
            ("after", Align::Right),
            ("key", Align::Left),
        ]))
        .paint(&["before", "after"], Role::Measure);
        for (key, (before, after)) in walls.into_iter().take(WALL_ROWS) {
            table.row(place.cells(
                key,
                vec![
                    key.tuner.clone(),
                    Wall(before).to_string(),
                    Wall(after).to_string(),
                    KeyText(&key.key).to_string(),
                ],
            ));
        }
        section(f, "tune walls, largest change first", table)
    }
}

/// The column that says where a key was tuned, after its tuner: shown only
/// when the diff spans more than one device or adapter, where it is what
/// tells two rows of one tuner and key apart.
struct Place {
    shown: bool,
}

impl Place {
    fn new(diff: &EnvironmentDiff) -> Self {
        let places: BTreeSet<_> = diff
            .keys
            .iter()
            .map(|key| (&key.device, &key.fingerprint))
            .collect();
        Self {
            shown: places.len() > 1,
        }
    }

    /// `columns`, the place's after the first when it is shown.
    fn columns(&self, columns: &[(&'static str, Align)]) -> Vec<(&'static str, Align)> {
        let mut columns = columns.to_vec();
        if self.shown {
            columns.insert(1, ("device", Align::Left));
        }
        columns
    }

    /// `cells` of `key`'s row, its place after the first when it is shown.
    fn cells(&self, key: &KeyDiff, mut cells: Vec<String>) -> Vec<String> {
        if self.shown {
            let place = format!("{}{}", key.device, Via(key.fingerprint.as_deref()));
            cells.insert(1, place);
        }
        cells
    }
}

/// A titled table, left out when it has no rows.
fn section(f: &mut fmt::Formatter<'_>, title: &str, table: Table) -> fmt::Result {
    if table.is_empty() {
        return Ok(());
    }
    writeln!(f)?;
    Role::Title.write(f, title)?;
    writeln!(f)?;
    fmt::Display::fmt(&table, f)
}
