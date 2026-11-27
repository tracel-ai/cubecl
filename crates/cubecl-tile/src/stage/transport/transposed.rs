//! The fill of a stage laid along a run ([`StageOrder::AlongRun`]): each unit reads blocks of
//! `run` consecutive source lines along the run, transposes each block in its registers, and
//! stores every value of a source line's width as one line of `run` values of the run.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

/// The blocks a fill that transposes into a stage laid along a run deals its units: `run`
/// consecutive source lines along the run each, one `width`-wide source line apart from the next
/// by `inner` lines, counted so that neighbouring blocks read neighbouring source lines of one
/// row.
#[derive(Clone, Debug)]
pub(crate) struct RunBlocks {
    /// The values of the run one store writes, and the source lines a block reads.
    pub(crate) run: usize,
    /// The values one source line holds.
    pub(crate) width: usize,
    /// The source lines from one value of the run to the next.
    pub(crate) inner: usize,
    /// The blocks along the run that neighbouring units take before the next source line of a
    /// row: their stores then land on distinct banks, as the values of one column of every block
    /// sit a whole number of 16-byte chunks apart.
    pub(crate) across: usize,
    /// The blocks the stage holds.
    pub(crate) count: usize,
}

impl RunBlocks {
    /// The blocks of a stage laid out as `layout`, filled from `width`-wide source lines of
    /// `elem_bits`-bit values and dealt to the units `fill` names: the first run of
    /// [`STORE_BITS`](Self::STORE_BITS) within the stage's ([`StageOrder::AlongRun`]) that deals
    /// every unit as many blocks, or the stage's own where none does.
    ///
    /// # Panics
    ///
    /// - Where `layout` is not laid along a run, or a block's values do not land as whole,
    ///   aligned lines of its run: the Lowering lays a stage along a run only where they do
    ///   ([`StageOrder::for_fragments`]).
    pub(crate) fn new(
        layout: &ConstantLayout,
        width: usize,
        elem_bits: usize,
        fill: &Level,
    ) -> Self {
        let StageOrder::AlongRun { run, .. } = layout.order else {
            panic!("RunBlocks: a stage in its source's order should be filled line for line");
        };
        let mut extents = layout.logical_extents();
        if let Some(innermost) = extents.last_mut() {
            *innermost /= width;
        }
        // A stage addresses its cells by position, so the run is the axis whose next cell lies
        // next in the buffer.
        let rank = extents.len();
        let origin = layout.cell_offset(&vec![0; rank], width, 0);
        let at = (0..rank - 1)
            .find(|&p| {
                let mut next = vec![0; rank];
                next[p] = 1;
                layout.cell_offset(&next, width, 0) == origin.map(|cell| cell + 1)
            })
            .expect("RunBlocks: a stage laid along a run should lay one axis's cells contiguous");
        assert!(
            extents[at].is_multiple_of(run),
            "RunBlocks: the run's {} values should be whole blocks of {run}",
            extents[at]
        );
        let inner: usize = extents[at + 1..].iter().product();
        let lines: usize = extents.iter().product();
        let of = |run: usize| {
            let along = extents[at] / run;
            let across = [Self::MOST_ACROSS, Self::MOST_ACROSS / 2, 1]
                .into_iter()
                .find(|&across| along.is_multiple_of(across))
                .unwrap_or(1);
            RunBlocks {
                run,
                width,
                inner,
                across,
                count: lines / run,
            }
        };
        let lands = |blocks: &RunBlocks| {
            (0..blocks.count).all(|block| blocks.columns(layout, block).is_some())
        };
        let units = fill.copied_units().1;
        let balanced = Self::STORE_BITS
            .into_iter()
            .map(|bits| bits / elem_bits)
            .filter(|&narrower| narrower >= 2 && narrower <= run && run.is_multiple_of(narrower))
            .map(of)
            .find(|blocks| {
                units.is_some_and(|units| blocks.count.is_multiple_of(units)) && lands(blocks)
            });
        let blocks = balanced.unwrap_or_else(|| of(run));
        assert!(
            lands(&blocks),
            "RunBlocks: every block should land as whole, aligned lines of {} values",
            blocks.run
        );
        blocks
    }

    /// The bits one store of a run writes where every unit takes as many blocks, in the order the
    /// fill prefers them: eight bytes, which the compiler pairs into one 16-byte store while a
    /// block of half a chunk's rows keeps half the registers live, so its loads issue together;
    /// then four. A run of a whole chunk deals twice as few blocks, so it balances no units the
    /// eight-byte run leaves unbalanced: it is the fill's run only where none balances.
    const STORE_BITS: [usize; 2] = [64, 32];

    /// The most blocks along the run neighbouring units take ([`across`](Self::across)): four
    /// keep each unit group's reads of one row whole 128-byte lines, eight 16-byte chunks.
    const MOST_ACROSS: usize = 4;

    /// The source line block `block` starts at.
    pub(crate) fn first_line(&self, block: usize) -> usize {
        let rest = block / self.across;
        let along = (rest / self.inner) * self.across + block % self.across;
        along * self.run * self.inner + rest % self.inner
    }

    /// Where each of block `block`'s `width` columns lands, in lines of `run` values, as
    /// [`store_block`] stores it; `None` where a column's values do not lie contiguous, starting
    /// on a whole line.
    pub(crate) fn columns(&self, layout: &ConstantLayout, block: usize) -> Option<Vec<usize>> {
        let first = self.first_line(block);
        (0..self.width)
            .map(|v| {
                let at = |r: usize| {
                    let line = layout.source_line(first + r * self.inner, self.width);
                    layout.cell_offset(&line, self.width, v)
                };
                let start = at(0)?;
                let contiguous = (1..self.run).all(|r| at(r) == Some(start + r));
                (contiguous && start.is_multiple_of(self.run)).then_some(start / self.run)
            })
            .collect()
    }

    /// What each turn of `fill` over these blocks adds to where its unit's first block's first
    /// column lands, one value a column, where every unit's turns agree
    /// ([`Walk::uniform_turn_offsets`]).
    pub(crate) fn uniform_offsets(
        &self,
        layout: &ConstantLayout,
        fill: &Level,
    ) -> Option<Vec<Vec<usize>>> {
        Walk::uniform_turn_offsets(fill, Some(self.count), |block| self.columns(layout, block))
    }

    /// The registers' source lines one unit holds over the stage where `fill` deals it the
    /// blocks: `run` for each of its turns.
    pub(crate) fn fetched_lines(&self, fill: &Level) -> Option<usize> {
        fill.copy_turns(self.count).map(|turns| turns * self.run)
    }
}

/// The blocks of the stage laid out as `layout`, filled from `W`-wide source lines of `T` dealt
/// to the units `fill` names ([`RunBlocks::new`]).
#[cube]
pub(crate) fn run_blocks<T: Numeric, W: Size>(
    layout: &BufferLayout,
    #[comptime] fill: &Level,
) -> comptime_type!(RunBlocks) {
    let constants = layout.constants();
    let elem = elem_type_of::<T>();
    let width = W::value();
    comptime!(RunBlocks::new(
        &constants.expect("RunBlocks: a stage laid along a run should have a constant shape"),
        width,
        elem.size_bits(),
        fill
    ))
}

/// The source line block `block` of `blocks` starts at ([`RunBlocks::first_line`]).
#[cube]
pub(crate) fn block_first_line(#[comptime] blocks: RunBlocks, block: usize) -> usize {
    let inner = comptime!(blocks.inner);
    let across = comptime!(blocks.across);
    let step = comptime!(blocks.run * blocks.inner);
    let rest = block / across;
    ((rest / inner) * across + block % across) * step + rest % inner
}

/// Read block `block`'s `run` source lines of `s` into `rows`, from `at` on.
#[cube]
pub(crate) fn fetch_block<T: Numeric, W: Size>(
    s: &Masked<'_, Vector<T, W>, CoordsDyn>,
    layout: &BufferLayout,
    #[comptime] blocks: RunBlocks,
    block: usize,
    rows: &mut Array<Vector<T, W>>,
    #[comptime] at: usize,
) {
    let inner = comptime!(blocks.inner);
    let first = block_first_line(comptime!(blocks.clone()), block);
    #[unroll]
    for r in 0..blocks.run {
        let line = first + comptime!(r * inner);
        rows[comptime!(at + r)] = s.read(layout.source_line_coords(line, W::value()));
    }
}

/// Block `block`'s rows, `rows[at..at + run]`, transposed and stored into the stage `d`, read as
/// lines of `run` values, laid out as `layout`: column `v` at `base` plus `offsets[v]`, or where
/// `layout` places it where no offsets are stated.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn store_block<T: Numeric, R: Size, W: Size>(
    d: &mut [Vector<T, R>],
    layout: &BufferLayout,
    #[comptime] blocks: RunBlocks,
    block: usize,
    rows: &Array<Vector<T, W>>,
    #[comptime] at: usize,
    base: usize,
    #[comptime] offsets: Option<Vec<usize>>,
) {
    #[unroll]
    for v in 0..W::value() {
        let mut column = Vector::<T, R>::empty();
        #[unroll]
        for r in 0..R::value() {
            column.insert(r, rows[comptime!(at + r)].extract(v));
        }
        #[comptime]
        match &offsets {
            Some(offsets) => {
                let offset = comptime!(offsets[v]);
                d[base + offset] = column;
            }
            None => {
                let first = block_first_line(comptime!(blocks.clone()), block);
                let line = layout.source_line_coords(first, W::value());
                let cell = layout.cell_offset(&line, W::value(), v);
                d[cell / R::value()] = column;
            }
        }
    }
}

/// Where the first column of block `block` lands in the stage laid out as `layout`, in lines of
/// `run` values.
#[cube]
pub(crate) fn first_column(
    layout: &BufferLayout,
    #[comptime] blocks: RunBlocks,
    block: usize,
) -> usize {
    let first = block_first_line(comptime!(blocks.clone()), block);
    let line = layout.source_line_coords(first, comptime!(blocks.width));
    let cell = layout.cell_offset(&line, comptime!(blocks.width), 0usize);
    cell / comptime!(blocks.run)
}

/// Read every block `fill` deals this unit of `s` into `rows`, turn `t`'s from `t · run` on:
/// all of a unit's loads issue before any of its stores.
///
/// # Assumptions
///
/// - `fill` writes its turns out over the blocks ([`RunBlocks::fetched_lines`] is known).
#[cube]
pub(crate) fn fetch_blocks<T: Numeric, W: Size>(
    s: &Masked<'_, Vector<T, W>, CoordsDyn>,
    layout: &BufferLayout,
    #[comptime] blocks: RunBlocks,
    #[comptime] fill: &Level,
    rows: &mut Array<Vector<T, W>>,
) {
    let turns = comptime!(
        fill.copy_turns(blocks.count)
            .expect("RunBlocks: a fill fetched block by block should have a known share")
    );
    let walk = Walk::copy(fill, comptime!(blocks.count).runtime());
    #[unroll]
    for t in 0..turns {
        if walk.takes_turn(t) {
            fetch_block::<T, W>(
                s,
                layout,
                comptime!(blocks.clone()),
                walk.copied_line(t),
                rows,
                comptime!(t * blocks.run),
            );
        }
    }
}

/// Transpose and store the blocks [`fetch_blocks`] read into `rows` into the stage `d`, read as
/// lines of `R` values, each column at the comptime offset its turn adds where every unit's turns
/// agree ([`RunBlocks::uniform_offsets`]).
///
/// # Assumptions
///
/// - As [`fetch_blocks`].
#[cube]
pub(crate) fn store_blocks<T: Numeric, R: Size, W: Size>(
    d: &mut [Vector<T, R>],
    layout: &BufferLayout,
    #[comptime] blocks: RunBlocks,
    #[comptime] fill: &Level,
    rows: &Array<Vector<T, W>>,
) {
    let constants = layout.constants();
    let offsets = comptime!(constants.and_then(|c| blocks.uniform_offsets(&c, fill)));
    let turns = comptime!(
        fill.copy_turns(blocks.count)
            .expect("RunBlocks: a fill fetched block by block should have a known share")
    );
    let walk = Walk::copy(fill, comptime!(blocks.count).runtime());
    let base = first_column(layout, comptime!(blocks.clone()), walk.copied_line(0));
    #[unroll]
    for t in 0..turns {
        if walk.takes_turn(t) {
            store_block::<T, R, W>(
                d,
                layout,
                comptime!(blocks.clone()),
                walk.copied_line(t),
                rows,
                comptime!(t * blocks.run),
                base,
                comptime!(offsets.as_ref().map(|offsets| offsets[t].clone())),
            );
        }
    }
}

#[cube]
impl Memory {
    /// Write every `SW`-wide line of the source of a stage laid along a run, counted in the
    /// source's order, in blocks of lines along the run dealt to the units `fill` names
    /// ([`Walk::copy`]), each block transposed in registers and stored a line of its run at a time:
    /// all of a unit's blocks read before any is stored where `fill` writes its turns out, block by
    /// block otherwise.
    pub(crate) fn transpose_lines<I2: Numeric, D: Size, SW: Size>(
        d: &mut [Vector<I2, D>],
        s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
        layout: &BufferLayout,
        #[comptime] fill: Level,
    ) {
        let blocks = run_blocks::<I2, SW>(layout, &fill);
        let size!(R) = comptime!(blocks.run);
        let d = d.with_vector_size_mut::<R>();
        let written = comptime!(
            blocks
                .fetched_lines(&fill)
                .filter(|_| Walk::copy_turns_written(&fill, Some(blocks.count)))
        );
        #[comptime]
        match written {
            Some(lines) => {
                let mut rows = Array::<Vector<I2, SW>>::new(lines);
                fetch_blocks::<I2, SW>(s, layout, comptime!(blocks.clone()), &fill, &mut rows);
                store_blocks::<I2, R, SW>(d, layout, comptime!(blocks.clone()), &fill, &rows);
            }
            None => {
                let mut rows = Array::<Vector<I2, SW>>::new(comptime!(blocks.run));
                for step in Walk::copy(&fill, comptime!(blocks.count).runtime()) {
                    let block = step.copied_line();
                    fetch_block::<I2, SW>(
                        s,
                        layout,
                        comptime!(blocks.clone()),
                        block,
                        &mut rows,
                        0usize,
                    );
                    store_block::<I2, R, SW>(
                        d,
                        layout,
                        comptime!(blocks.clone()),
                        block,
                        &rows,
                        0usize,
                        0usize,
                        comptime!(None),
                    );
                }
            }
        }
    }

    /// Read this unit's blocks of `s` into `fetched`: the register fetch of a stage laid along a
    /// run ([`fetch_blocks`]).
    pub(crate) fn fetch_blocks<T: Numeric, W: Size>(
        &self,
        s: &Masked<'_, Vector<T, W>, CoordsDyn>,
        fetched: &mut Array<Vector<T, W>>,
    ) {
        let layout = self.layout.clone();
        let fill = comptime!(self.access.copy.clone());
        let blocks = run_blocks::<T, W>(&layout, &fill);
        fetch_blocks::<T, W>(s, &layout, blocks, &fill, fetched);
    }

    /// Transpose and store what [`fetch_blocks`](Memory::fetch_blocks) read into this stage.
    pub(crate) fn store_blocks<T: Numeric, W: Size>(&mut self, fetched: &Array<Vector<T, W>>) {
        let layout = self.layout.clone();
        let fill = comptime!(self.access.copy.clone());
        let blocks = run_blocks::<T, W>(&layout, &fill);
        let size!(R) = comptime!(blocks.run);
        let d = self.lines_storage_mut::<T, R>();
        store_blocks::<T, R, W>(d, &layout, blocks, &fill, fetched);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `16 × 16` `[k, n]` stage laid `k` innermost, rows of 16 `f16` pitched to 24, stored up to
    /// a chunk of eight values at a time.
    fn along_k() -> ConstantLayout {
        let (k, n) = (Dimension(0), Dimension(1));
        ConstantLayout {
            shape: vec![16, 16],
            strides: vec![1, 24],
            projection: Projection::direct(&[k, n]),
            rows: RowArrangement::Padded { lines: 8 },
            order: StageOrder::AlongRun {
                dimension: k,
                run: 8,
            },
        }
    }

    /// Four units take the stage's eight blocks of four rows of `k`, an eight-byte store each:
    /// the four blocks along `k` of a chunk of `n` first, then the next chunk.
    #[test]
    fn neighbouring_units_take_blocks_along_the_run_then_the_next_chunk() {
        let blocks = RunBlocks::new(&along_k(), 8, 16, &Level::copied_by_cube(4));
        assert_eq!((blocks.run, blocks.count, blocks.across), (4, 8, 4));
        let first: Vec<usize> = (0..6).map(|block| blocks.first_line(block)).collect();
        assert_eq!(first, vec![0, 8, 16, 24, 1, 9]);
    }

    /// Each column of a block lands as one line of four values of `k`, a pitched row apart.
    #[test]
    fn a_block_lands_each_column_as_one_line_of_the_run() {
        let blocks = RunBlocks::new(&along_k(), 8, 16, &Level::copied_by_cube(4));
        let columns = blocks.columns(&along_k(), 4).unwrap();
        // Block 4 is rows 0..4 of the second chunk of `n`: cell `(0, 8 + v)` at `(8 + v) · 24`.
        let expected: Vec<usize> = (0..8).map(|v| (8 + v) * 24 / 4).collect();
        assert_eq!(columns, expected);
    }

    /// Where no eight-byte run deals every unit as many blocks, the fill narrows to four bytes.
    #[test]
    fn the_run_narrows_until_every_unit_takes_as_many_blocks() {
        let blocks = RunBlocks::new(&along_k(), 8, 16, &Level::copied_by_cube(16));
        assert_eq!((blocks.run, blocks.count), (2, 16));
    }

    /// Where no run balances the units, the fill stores the stage's widest run.
    #[test]
    fn unbalanced_units_take_the_widest_run() {
        let blocks = RunBlocks::new(&along_k(), 8, 16, &Level::copied_by_cube(64));
        assert_eq!(blocks.run, 8);
    }
}
