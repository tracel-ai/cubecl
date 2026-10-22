//! The table a [`Partitioning`] prints as: one row a level, leaf up.

use std::fmt::{self, Display, Formatter};

use crate::{Axis, ComputeScope, Coverage, Level, Partitioning, Space, Spread};

/// The leaf row's glyph.
const LEAF: char = '◦';
const MARGIN: &str = "  ";
const LEVEL: &str = "  ";
const TIMES: &str = " × ";
const GAP: &str = "    ";

/// A [`Partitioning`] with a name for each of its axes; unnamed axes print their index.
pub struct LevelTable<'a> {
    partitioning: &'a Partitioning,
    labels: &'a [(Axis, &'a str)],
}

impl<'a> LevelTable<'a> {
    pub(crate) fn new(partitioning: &'a Partitioning, labels: &'a [(Axis, &'a str)]) -> Self {
        LevelTable {
            partitioning,
            labels,
        }
    }

    fn label(&self, axis: Axis) -> String {
        match self.labels.iter().find(|&&(named, _)| named == axis) {
            Some(&(_, name)) => name.to_string(),
            None => format!("a{}", axis.0),
        }
    }

    /// Every row of the table, leaf up.
    fn rows(&self) -> Vec<Row> {
        let axes: Vec<Axis> = self.partitioning.space().axes().collect();
        let mut space = self.partitioning.space().clone();
        let mut rows: Vec<Row> = self
            .partitioning
            .levels()
            .iter()
            .map(|level| {
                let row = Row::of(level, &space, &axes);
                space = level.child(&space);
                row
            })
            .collect();
        rows.reverse();
        rows.insert(0, Row::leaf(&space, &axes));
        rows
    }
}

impl Display for LevelTable<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let axes: Vec<Axis> = self.partitioning.space().axes().collect();
        let header: Vec<String> = axes.iter().map(|&axis| self.label(axis)).collect();
        let rows = self.rows();

        let counts = Block::new("count", &header, rows.iter().map(|row| &row.counts));
        let tiles = Block::new("tile", &header, rows.iter().map(|row| &row.tile));
        let coverage_wide = rows
            .iter()
            .map(|row| row.coverage.chars().count())
            .max()
            .unwrap_or(0)
            + GAP.chars().count();
        let indent = format!("{MARGIN} {LEVEL}{:coverage_wide$}", "");

        writeln!(
            f,
            "{indent}{}{}{GAP}{}{}",
            counts.indent(),
            counts.line(&header),
            tiles.indent(),
            tiles.line(&header)
        )?;
        writeln!(f)?;
        for row in &rows {
            writeln!(
                f,
                "{MARGIN}{}{LEVEL}{:coverage_wide$}{}{}{GAP}{}{}",
                row.glyph,
                row.coverage,
                counts.indent(),
                counts.line(&row.counts),
                tiles.indent(),
                tiles.line(&row.tile)
            )?;
        }
        writeln!(f)?;
        write!(f, "{indent}{}{GAP}{}", counts.rule(), tiles.rule())
    }
}

/// Prints the table with no axis named; see [`Partitioning::table`].
impl Display for Partitioning {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.table(&[]).fmt(f)
    }
}

/// One line of the table.
struct Row {
    glyph: char,
    coverage: String,
    counts: Vec<String>,
    tile: Vec<String>,
}

impl Row {
    fn of(level: &Level, space: &Space, axes: &[Axis]) -> Row {
        Row {
            glyph: Row::glyph(level.coverage()),
            coverage: Row::coverage(level, space),
            counts: axes
                .iter()
                .map(|&axis| Row::count(level, space, axis))
                .collect(),
            tile: axes.iter().map(|&axis| Row::extent(space, axis)).collect(),
        }
    }

    /// The leaf row: every count is a one.
    fn leaf(space: &Space, axes: &[Axis]) -> Row {
        Row {
            glyph: LEAF,
            coverage: String::new(),
            counts: axes.iter().map(|_| "·".to_string()).collect(),
            tile: axes.iter().map(|&axis| Row::extent(space, axis)).collect(),
        }
    }

    /// The glyph a level's coverage prints as.
    fn glyph(coverage: Coverage) -> char {
        match coverage {
            Coverage::Distribute(ComputeScope::Cube) => '▣',
            Coverage::Distribute(ComputeScope::Plane) => '▤',
            Coverage::Distribute(ComputeScope::PlaneGroup { .. }) => '▥',
            Coverage::Distribute(ComputeScope::Unit) => '▪',
            Coverage::Walk => '↻',
        }
    }

    /// What a level covers, in words.
    fn coverage(level: &Level, space: &Space) -> String {
        let counts: Vec<Option<usize>> = level
            .axes()
            .iter()
            .map(|&axis| level.tiles_const(space, axis))
            .collect();
        let total = counts
            .iter()
            .try_fold(1usize, |acc, count| count.map(|n| acc * n));
        let many = match total {
            Some(n) => n.to_string(),
            None => "?".to_string(),
        };
        let interleaved = level
            .axes()
            .iter()
            .any(|&axis| level.spread(axis) == Some(Spread::Interleaved));
        let mut note = match level.coverage() {
            Coverage::Distribute(ComputeScope::Cube) => match level.shared_by() {
                Some(cubes) => format!("{cubes} cubes sharing {many} boxes"),
                None => format!("{many} cubes"),
            },
            Coverage::Distribute(ComputeScope::Plane) => match level.shared_by() {
                Some(planes) => format!("{planes} planes sharing {many} boxes"),
                None => format!("{many} planes a cube"),
            },
            Coverage::Distribute(ComputeScope::PlaneGroup { planes }) => match level.shared_by() {
                Some(groups) => format!("{groups} groups of {planes} planes sharing {many} boxes"),
                None => format!("{many} groups of {planes} planes a cube"),
            },
            Coverage::Distribute(ComputeScope::Unit) => format!("{many} units"),
            Coverage::Walk => format!("{many} steps"),
        };
        if interleaved {
            note += " interleaved";
        }
        match level.fillers() {
            0 => {}
            1 => note += ", 1 filling plane",
            n => note += &format!(", {n} filling planes"),
        }
        note
    }

    /// How many tiles a level takes along one axis, as printed.
    fn count(level: &Level, space: &Space, axis: Axis) -> String {
        match level.count(axis) {
            None => "·".to_string(),
            Some(_) => match level.tiles_const(space, axis) {
                Some(tiles) => tiles.to_string(),
                None => "?".to_string(),
            },
        }
    }

    /// `axis`'s extent, or the mark a dynamic one prints as.
    fn extent(space: &Space, axis: Axis) -> String {
        match space.is_dynamic(axis) {
            true => "?".to_string(),
            false => space.extent(axis).to_string(),
        }
    }
}

/// One named block of columns, the counts or the tiles: every cell right-aligned in its column,
/// `×` between, a rule naming the block beneath.
struct Block<'n> {
    name: &'n str,
    /// The widest cell of each column, the header's own label included.
    widths: Vec<usize>,
}

impl<'n> Block<'n> {
    fn new<'c>(
        name: &'n str,
        header: &[String],
        cells: impl Iterator<Item = &'c Vec<String>>,
    ) -> Self {
        let widths = cells.fold(
            header.iter().map(|label| label.chars().count()).collect(),
            |widest: Vec<usize>, row| {
                widest
                    .iter()
                    .zip(row)
                    .map(|(&widest, cell)| widest.max(cell.chars().count()))
                    .collect()
            },
        );
        Block { name, widths }
    }

    /// One line's cells of this block.
    fn line(&self, cells: &[String]) -> String {
        cells
            .iter()
            .zip(&self.widths)
            .map(|(cell, &width)| format!("{cell:>width$}"))
            .collect::<Vec<_>>()
            .join(TIMES)
    }

    /// The rule naming this block.
    fn rule(&self) -> String {
        format!(
            "└─ {} {}┘",
            self.name,
            "─".repeat(self.wide() - self.ruled() + 1)
        )
    }

    /// The indent before a line's cells, so they end where the rule does.
    fn indent(&self) -> String {
        " ".repeat(self.wide() - self.spanned())
    }

    /// How wide this block prints: its cells, or its rule where that is wider.
    fn wide(&self) -> usize {
        self.spanned().max(self.ruled())
    }

    /// How wide this block's cells print, separators included.
    fn spanned(&self) -> usize {
        self.widths.iter().sum::<usize>()
            + TIMES.chars().count() * (self.widths.len().saturating_sub(1))
    }

    /// The narrowest this block can print and still carry its rule.
    fn ruled(&self) -> usize {
        self.name.chars().count() + 6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Levels, Space};

    const B: Axis = Axis(0);
    const M: Axis = Axis(1);
    const N: Axis = Axis(2);
    const K: Axis = Axis(3);

    /// A staged matmul's five levels.
    fn staged() -> Partitioning {
        Partitioning::new(
            Space::new(&[(B, 4), (M, 512), (N, 1024), (K, 4096)]),
            Levels::leaf(&[(M, 16), (N, 16), (K, 16)])
                .walk(&[(M, 2), (N, 2)])
                .walk(&[(K, 2)])
                .planes(&[(M, 2), (N, 2)])
                .walk_every(&[K])
                .cubes(&[M, N])
                .batches(&[B])
                .build(),
        )
    }

    #[test]
    fn a_row_is_the_row_below_it_times_its_count() {
        let labels = [(B, "b"), (M, "m"), (N, "n"), (K, "k")];

        assert_eq!(
            staged().table(&labels).to_string(),
            [
                "                        b × m ×  n ×   k    b ×   m ×    n ×    k",
                "",
                "  ◦                     · × · ×  · ×   ·    1 ×  16 ×   16 ×   16",
                "  ↻  4 steps            · × 2 ×  2 ×   ·    1 ×  32 ×   32 ×   16",
                "  ↻  2 steps            · × · ×  · ×   2    1 ×  32 ×   32 ×   32",
                "  ▤  4 planes a cube    · × 2 ×  2 ×   ·    1 ×  64 ×   64 ×   32",
                "  ↻  128 steps          · × · ×  · × 128    1 ×  64 ×   64 × 4096",
                "  ▣  512 cubes          4 × 8 × 16 ×   ·    4 × 512 × 1024 × 4096",
                "",
                "                        └─ count ──────┘    └─ tile ────────────┘",
            ]
            .join("\n")
        );
    }

    /// An unlabelled axis prints its index.
    #[test]
    fn an_unlabelled_axis_prints_its_index() {
        let table = staged().to_string();

        assert!(table.contains("a0 × a1 × a2 ×  a3"), "{table}");
    }

    /// A dynamic count prints as a question.
    #[test]
    fn a_dynamic_axis_prints_a_question() {
        let partitioning = Partitioning::new(
            Space::new(&[(M, 512), (K, 4096)]).with_dynamic(&[K]),
            Levels::leaf(&[(M, 64), (K, 32)])
                .walk_every(&[K])
                .cubes(&[M])
                .build(),
        );
        let table = partitioning.table(&[(M, "m"), (K, "k")]).to_string();

        assert!(table.lines().any(|line| line.contains("× ?")), "{table}");
    }

    /// A shared level still prints as the cube grid.
    #[test]
    fn a_shared_level_still_prints_as_the_cube_grid() {
        let partitioning = Partitioning::new(
            Space::new(&[(M, 512), (N, 1024)]),
            Levels::leaf(&[(M, 64), (N, 64)])
                .cubes(&[M, N])
                .shared_by(8)
                .build(),
        );

        assert!(partitioning.to_string().contains('▣'));
    }

    /// An overhanging axis counts its partial tile.
    #[test]
    fn an_overhanging_axis_counts_its_partial_tile() {
        let partitioning = Partitioning::new(
            Space::new(&[(M, 500), (K, 4096)]),
            Levels::leaf(&[(M, 128), (K, 32)])
                .walk_every(&[K])
                .cubes(&[M])
                .build(),
        );

        assert_eq!(
            partitioning.table(&[(M, "m"), (K, "k")]).to_string(),
            [
                "                      m ×   k      m ×    k",
                "",
                "  ◦                   · ×   ·    128 ×   32",
                "  ↻  128 steps        · × 128    128 × 4096",
                "  ▣  4 cubes          4 ×   ·    500 × 4096",
                "",
                "                  └─ count ─┘    └─ tile ─┘",
            ]
            .join("\n")
        );
    }
}
