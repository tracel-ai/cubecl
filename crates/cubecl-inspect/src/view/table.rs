use std::fmt;

/// Columns of text, padded to their widest cell. Numbers read best
/// right-aligned, so a column is [`Align::Right`] unless it holds names.
pub(crate) struct Table {
    columns: Vec<(&'static str, Align)>,
    rows: Vec<Vec<String>>,
}

#[derive(Clone, Copy)]
pub(crate) enum Align {
    Left,
    Right,
}

impl Table {
    pub(crate) fn new(columns: &[(&'static str, Align)]) -> Self {
        Self {
            columns: columns.to_vec(),
            rows: Vec::new(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub(crate) fn row(&mut self, cells: Vec<String>) {
        debug_assert_eq!(cells.len(), self.columns.len());
        self.rows.push(cells);
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let widths: Vec<usize> = self
            .columns
            .iter()
            .enumerate()
            .map(|(column, (header, _))| {
                self.rows
                    .iter()
                    .map(|row| row[column].chars().count())
                    .chain([header.chars().count()])
                    .max()
                    .unwrap_or_default()
            })
            .collect();

        let headers: Vec<String> = self
            .columns
            .iter()
            .map(|(header, _)| header.to_string())
            .collect();
        for row in [&headers].into_iter().chain(&self.rows) {
            let mut line = String::new();
            for (column, cell) in row.iter().enumerate() {
                if column > 0 {
                    line.push_str("  ");
                }
                let pad = widths[column] - cell.chars().count();
                match self.columns[column].1 {
                    Align::Left => {
                        line.push_str(cell);
                        line.extend(std::iter::repeat_n(' ', pad));
                    }
                    Align::Right => {
                        line.extend(std::iter::repeat_n(' ', pad));
                        line.push_str(cell);
                    }
                }
            }
            writeln!(f, "{}", line.trim_end())?;
        }
        Ok(())
    }
}
