use super::paint::{Paint, Role};
use std::fmt;

/// The space between two columns.
const GAP: &str = "  ";

/// Columns of text, padded to their widest cell, under a rule. Numbers read
/// best right-aligned, so a column is [`Align::Right`] unless it holds names.
pub(crate) struct Table {
    columns: Vec<Column>,
    rows: Vec<Vec<String>>,
    /// The role each row is painted in, beside it in `rows`; `None` keeps the
    /// terminal's own colour.
    roles: Vec<Option<Role>>,
    /// A column whose value is written once per run of equal rows, blank
    /// below it, so a table sorted by that column reads as groups.
    grouped: Option<usize>,
}

#[derive(Clone, Copy)]
pub(crate) enum Align {
    Left,
    Right,
}

struct Column {
    header: &'static str,
    align: Align,
    /// The role its cells are painted in, under their row's.
    role: Option<Role>,
}

impl Table {
    pub(crate) fn new(columns: &[(&'static str, Align)]) -> Self {
        Self {
            columns: columns
                .iter()
                .map(|&(header, align)| Column {
                    header,
                    align,
                    role: None,
                })
                .collect(),
            rows: Vec::new(),
            roles: Vec::new(),
            grouped: None,
        }
    }

    /// Write `column` once per run of equal values instead of on every row.
    pub(crate) fn group(mut self, column: usize) -> Self {
        self.grouped = Some(column);
        self
    }

    /// Paint the cells of the columns named `headers` in `role`, such as
    /// [`Role::Measure`] for the columns that hold times.
    pub(crate) fn paint(mut self, headers: &[&str], role: Role) -> Self {
        for header in headers {
            let column = self
                .columns
                .iter_mut()
                .find(|column| column.header == *header);
            debug_assert!(column.is_some(), "no column named {header}");
            if let Some(column) = column {
                column.role = Some(role);
            }
        }
        self
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub(crate) fn row(&mut self, cells: Vec<String>) {
        self.row_in(None, cells);
    }

    /// A row painted in `role` when the table renders the alternate form.
    pub(crate) fn row_in(&mut self, role: Option<Role>, cells: Vec<String>) {
        debug_assert_eq!(cells.len(), self.columns.len());
        self.rows.push(cells);
        self.roles.push(role);
    }

    /// The table as lines, padded and trimmed: the column names, then one
    /// line per row in the order the rows were added.
    #[cfg(feature = "tui")]
    pub(crate) fn lines(&self) -> Vec<String> {
        let widths = self.widths();
        self.cells()
            .map(|cells| {
                Line {
                    table: self,
                    widths: &widths,
                    cells,
                    row: None,
                }
                .to_string()
            })
            .collect()
    }

    fn widths(&self) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .map(|(column, Column { header, .. })| {
                self.rows
                    .iter()
                    .map(|row| row[column].chars().count())
                    .chain([header.chars().count()])
                    .max()
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Every line's cells as written: the column names, then each row's, a
    /// repeated value of the grouped column left blank.
    fn cells(&self) -> impl Iterator<Item = Vec<&str>> {
        let headers = self.columns.iter().map(|column| column.header).collect();
        let previous = [None].into_iter().chain(self.rows.iter().map(Some));
        let rows = self.rows.iter().zip(previous).map(|(row, previous)| {
            row.iter()
                .enumerate()
                .map(|(column, cell)| {
                    let repeated = self.grouped == Some(column)
                        && previous.is_some_and(|previous: &Vec<String>| previous[column] == *cell);
                    if repeated { "" } else { cell.as_str() }
                })
                .collect()
        });
        [headers].into_iter().chain(rows)
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let widths = self.widths();
        let roles = [Some(Role::Header)]
            .into_iter()
            .chain(self.roles.iter().copied());
        for (index, (cells, row)) in self.cells().zip(roles).enumerate() {
            let line = Line {
                table: self,
                widths: &widths,
                cells,
                row,
            };
            fmt::Display::fmt(&line, f)?;
            writeln!(f)?;
            if index == 0 {
                let rule: Vec<String> = widths.iter().map(|width| "─".repeat(*width)).collect();
                Role::Rule.write(f, rule.join(GAP))?;
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// One line of a [`Table`]: its cells padded to the column widths, each
/// painted in its row's role, then its column's, and the line trimmed.
struct Line<'a> {
    table: &'a Table,
    widths: &'a [usize],
    cells: Vec<&'a str>,
    row: Option<Role>,
}

impl fmt::Display for Line<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(last) = self.cells.iter().rposition(|cell| !cell.is_empty()) else {
            return Ok(());
        };
        let header = self.row == Some(Role::Header);
        for (index, cell) in self.cells[..=last].iter().enumerate() {
            let column = &self.table.columns[index];
            if index > 0 {
                f.write_str(GAP)?;
            }
            if cell.is_empty() {
                f.write_str(&" ".repeat(self.widths[index]))?;
                continue;
            }
            let pad = " ".repeat(self.widths[index] - cell.chars().count());
            let roles: Vec<Role> = self
                .row
                .into_iter()
                .chain(column.role.filter(|_| !header))
                .collect();
            let paint = Paint(&roles);
            match column.align {
                Align::Left => {
                    paint.write(f, cell)?;
                    if index < last {
                        f.write_str(&pad)?;
                    }
                }
                Align::Right => {
                    f.write_str(&pad)?;
                    paint.write(f, cell)?;
                }
            }
        }
        Ok(())
    }
}
