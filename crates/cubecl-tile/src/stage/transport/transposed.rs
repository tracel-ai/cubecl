//! The fill of a stage laid along a run ([`StageOrder::AlongRun`]): each unit reads blocks of
//! consecutive source lines along the run, transposes each block in its registers, and stores
//! every value of a source line's width as one line of the run.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use super::straight::fill_extent;
use crate::*;

/// The blocks a fill that transposes into a stage laid along a run deals its units: `run`
/// consecutive source lines along the run each, one `width`-wide source line apart from the next
/// by `inner` lines, counted so that neighbouring units take neighbouring blocks along the run.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct RunBlocks {
    /// The values of the run one store writes, and the source lines a block reads.
    pub(crate) run: usize,
    /// The values one source line holds.
    pub(crate) width: usize,
    /// The source lines from one value of the run to the next.
    pub(crate) inner: usize,
    /// The blocks along the run that neighbouring units take before the next source line of a
    /// row, so that their stores of one column land side by side in one row of the stage.
    pub(crate) across: usize,
    /// The blocks the stage holds.
    pub(crate) count: usize,
}

impl RunBlocks {
    /// The bytes a store of the fill writes, in the order the fill prefers them where each
    /// deals every unit as many blocks: eight, then four. The compiler pairs two eight-byte
    /// stores into one instruction, and a block of half a chunk's lines keeps half the registers
    /// live, so the loads of a split or few-row stage issue together. A chunk-wide store deals
    /// half the blocks of an eight-byte one, so it balances no units the eight-byte one leaves
    /// unbalanced, and it measured slower on few rows and no faster elsewhere (gfx1151, f16
    /// `m × 4096 × 4096`).
    const PREFERRED_RUN_BYTES: [usize; 2] = [8, 4];

    /// The most blocks along the run neighbouring units take ([`across`](Self::across)): four
    /// keep the reads of a group of units within one row of the source whole 128-byte lines.
    const MOST_ACROSS: usize = 4;

    /// The blocks of a stage over `space` laid out in `order`, filled from `width`-wide source
    /// lines of `elem_bits`-bit values and dealt to the units `fill` names: stored the first run
    /// of [`PREFERRED_RUN_BYTES`](Self::PREFERRED_RUN_BYTES) that divides the order's run and
    /// deals every unit as many blocks, or, where none does, the narrowest such run, which keeps
    /// the most units busy.
    ///
    /// # Panics
    ///
    /// - Where `order` does not lay the stage along a run of a Dimension of `space` other than
    ///   its last, or the last's extent is not whole source lines: the Lowering lays a stage along
    ///   a run only where it is ([`StageOrder::for_fragments`]).
    pub(crate) fn new(
        space: &Space,
        order: StageOrder,
        width: usize,
        elem_bits: usize,
        fill: &Level,
    ) -> Self {
        let StageOrder::AlongRun {
            dimension,
            run: stored,
            ..
        } = order
        else {
            panic!("RunBlocks: a stage in its source's order should be filled line for line");
        };
        let rank = space.rank();
        let at = space.position(dimension);
        assert!(
            at + 1 < rank,
            "RunBlocks: a stage should be laid along a run of a Dimension its source does not lay \
             innermost"
        );
        let mut extents: Vec<usize> = (0..rank).map(|p| space.extent_at(p)).collect();
        assert!(
            extents[rank - 1].is_multiple_of(width),
            "RunBlocks: the source's innermost extent {} should be whole {width}-wide lines",
            extents[rank - 1]
        );
        extents[rank - 1] /= width;
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
        // Every run that divides the stage's own lands its blocks whole within a row: the stage's
        // rows and their padding are whole runs of it ([`StageOrder::stored_run`]).
        let runs: Vec<usize> = Self::PREFERRED_RUN_BYTES
            .into_iter()
            .map(|bytes| bytes * 8 / elem_bits)
            .filter(|&run| run >= 2 && run <= stored && stored.is_multiple_of(run))
            .collect();
        let units = fill.copied_units().1;
        let balanced = runs
            .iter()
            .copied()
            .map(of)
            .find(|blocks| units.is_some_and(|units| blocks.count.is_multiple_of(units)));
        let narrowest = runs.iter().copied().min().unwrap_or(stored);
        balanced.unwrap_or_else(|| of(narrowest))
    }

    /// The turns one unit takes over the blocks `fill` deals, where the count of its units is
    /// known.
    pub(crate) fn turns(&self, fill: &Level) -> Option<usize> {
        fill.copy_turns(self.count)
    }

    /// The source lines one unit holds in its registers where `fill` deals it the blocks: `run`
    /// for each of its turns.
    pub(crate) fn fetched_lines(&self, fill: &Level) -> Option<usize> {
        self.turns(fill).map(|turns| turns * self.run)
    }

    /// Whether every unit `fill` names takes as many blocks, each turn of each unit a block.
    pub(crate) fn balanced(&self, fill: &Level) -> bool {
        fill.copied_units()
            .1
            .is_some_and(|units| self.count.is_multiple_of(units))
    }
}

/// The source line block `block` of `blocks` starts at: neighbouring blocks take `across`
/// steps along the run, then the next source line of a row, then the next `across` steps.
#[cube]
fn block_first_line(#[comptime] blocks: RunBlocks, block: usize) -> usize {
    let inner = comptime!(blocks.inner);
    let across = comptime!(blocks.across);
    let step = comptime!(blocks.run * blocks.inner);
    let rest = block / across;
    ((rest / inner) * across + block % across) * step + rest % inner
}

/// Read block `block`'s `run` source lines of `s` into `rows`, from `at` on, the stage laid out
/// as `layout`.
#[cube]
fn fetch_block<T: Numeric, W: Size>(
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

/// Block `block`'s rows, `rows[at..at + run]`, transposed and stored into the stage `d` laid out
/// as `layout`, read as lines of `R` values: each of a source line's `W` values a column of the
/// block, stored as one line of the run where the stage places its first value, a constant step
/// past the first column's where the stage holds every source line within one block
/// ([`BufferLayout::line_cell_step`]).
#[cube]
fn store_block<T: Numeric, R: Size, W: Size>(
    d: &mut [Vector<T, R>],
    layout: &BufferLayout,
    #[comptime] blocks: RunBlocks,
    block: usize,
    rows: &Array<Vector<T, W>>,
    #[comptime] at: usize,
) {
    let first = block_first_line(comptime!(blocks.clone()), block);
    let line = layout.source_line_coords(first, W::value());
    let run = R::value();
    let step = layout.line_cell_step(W::value());
    // Each column starts on a whole line of the run, so the step between two is whole lines.
    let step = comptime!(step.filter(|step| step.is_multiple_of(run)));
    let base = layout.cell_offset(&line, W::value(), 0usize) / run;
    #[unroll]
    for v in 0..W::value() {
        let mut column = Vector::<T, R>::empty();
        #[unroll]
        for r in 0..R::value() {
            column.insert(r, rows[comptime!(at + r)].extract(v));
        }
        match comptime!(step) {
            Some(step) => {
                d[base + comptime!(v * step / run)] = column;
            }
            None => {
                let cell = layout.cell_offset(&line, W::value(), v);
                d[cell / run] = column;
            }
        }
    }
}

/// Read every block `fill` deals this unit of `s` into `rows`, turn `t`'s from `t · run` on.
///
/// # Assumptions
///
/// - `fill` writes its turns out over the blocks ([`RunBlocks::fetched_lines`] is known).
#[cube]
fn fetch_blocks<T: Numeric, W: Size>(
    s: &Masked<'_, Vector<T, W>, CoordsDyn>,
    layout: &BufferLayout,
    #[comptime] blocks: RunBlocks,
    #[comptime] fill: &Level,
    rows: &mut Array<Vector<T, W>>,
) {
    let turns = comptime!(
        blocks
            .turns(fill)
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
/// lines of `R` values.
///
/// # Assumptions
///
/// - As [`fetch_blocks`].
#[cube]
fn store_blocks<T: Numeric, R: Size, W: Size>(
    d: &mut [Vector<T, R>],
    layout: &BufferLayout,
    #[comptime] blocks: RunBlocks,
    #[comptime] fill: &Level,
    rows: &Array<Vector<T, W>>,
) {
    let turns = comptime!(
        blocks
            .turns(fill)
            .expect("RunBlocks: a fill fetched block by block should have a known share")
    );
    let walk = Walk::copy(fill, comptime!(blocks.count).runtime());
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
            );
        }
    }
}

#[cube]
impl Memory {
    /// The blocks this stage over `space`, laid along a run, deals its fill's units, filled from
    /// `width`-wide source lines of `elem_bits`-bit values ([`RunBlocks::new`]).
    pub(crate) fn run_blocks(
        &self,
        #[comptime] space: &Space,
        #[comptime] width: usize,
        #[comptime] elem_bits: usize,
    ) -> comptime_type!(RunBlocks) {
        let fill = comptime!(self.access.copy.clone());
        let order = comptime!(self.layout.order);
        comptime!(RunBlocks::new(space, order, width, elem_bits, &fill))
    }

    /// Fill this stage over `space`, laid along a run, from `src`'s `SW`-wide lines: blocks of
    /// source lines along the run dealt to the units the fill names ([`Walk::copy`]), each block
    /// transposed in registers and stored a line of the run a column. Every unit reads all of its
    /// blocks before it stores any where each takes as many, its turns are written out and their
    /// lines fit the registers a prefetch may hold ([`Prefetch::fits_registers`](crate::stage::Prefetch::fits_registers)); block by block
    /// otherwise.
    pub(crate) fn fill_transposed<T: Numeric, SW: Size>(
        &mut self,
        src: &Memory,
        #[comptime] space: &Space,
    ) {
        let check = comptime!(src.access.overhang.masks());
        let sw = SW::value();
        comptime!(fill_extent(space, sw, sw, check));
        let elem = elem_type_of::<T>();
        let bits = comptime!(elem.size_bits());
        let blocks = self.run_blocks(space, sw, bits);
        let fill = comptime!(self.access.copy.clone());
        let layout = self.layout.clone();
        let s = src.edged(
            src.window_view_storage::<T, SW>(comptime!(Guard::Checked)),
            check,
        );
        let size!(R) = comptime!(blocks.run);
        let d = self.lines_storage_mut::<T, R>();
        let held = comptime!(blocks.fetched_lines(&fill).filter(|&lines| {
            blocks.balanced(&fill)
                && Walk::copy_turns_written(&fill, Some(blocks.count))
                && crate::stage::Prefetch::fits_registers(
                    crate::stage::Prefetch::straight_registers(lines, sw, bits),
                )
        }));
        #[comptime]
        match held {
            Some(lines) => {
                let mut rows = Array::<Vector<T, SW>>::new(lines);
                fetch_blocks::<T, SW>(&s, &layout, comptime!(blocks.clone()), &fill, &mut rows);
                store_blocks::<T, R, SW>(d, &layout, comptime!(blocks.clone()), &fill, &rows);
            }
            None => {
                let mut rows = Array::<Vector<T, SW>>::new(comptime!(blocks.run));
                for step in Walk::copy(&fill, comptime!(blocks.count).runtime()) {
                    let block = step.copied_line();
                    fetch_block::<T, SW>(
                        &s,
                        &layout,
                        comptime!(blocks.clone()),
                        block,
                        &mut rows,
                        0usize,
                    );
                    store_block::<T, R, SW>(
                        d,
                        &layout,
                        comptime!(blocks.clone()),
                        block,
                        &rows,
                        0usize,
                    );
                }
            }
        }
    }

    /// Whether this stage over `space`, laid along a run, is filled from `src` by the blocks a
    /// register fetch can hold ([`fetch_blocks`](Memory::fetch_blocks)): a plain, whole stage
    /// one value a line that replaces, filled by the units' own loads from a plain, direct source
    /// of `elem_bits`-bit values, each unit's share known.
    #[allow(dead_code)] // Reached through its expand, from `Tile::fetches_transposed_from`.
    pub(crate) fn fetches_transposed(
        &self,
        src: &Memory,
        #[comptime] space: &Space,
        #[comptime] elem_bits: usize,
    ) -> comptime_type!(bool) {
        let order = comptime!(self.layout.order);
        let fill = comptime!(self.access.copy.clone());
        let width = comptime!(src.store.vector_size);
        comptime!(
            order != StageOrder::AsSource
                && self.store.packing.as_stored()
                && src.store.packing.as_stored()
                && self.access.whole
                && !self.access.overhang.masks()
                && self.access.write == Write::Replace
                && self.store.vector_size == 1
                && self.projection.is_direct()
                && src.projection.is_direct()
                && src.access.delivery == Delivery::SyncPerUnit
                && RunBlocks::new(space, order, width, elem_bits, &fill)
                    .turns(&fill)
                    .is_some()
        )
    }

    /// Read this unit's blocks of `src` into `fetched`: the register fetch of a stage over
    /// `space` laid along a run, its blocks transposed at the store
    /// ([`store_blocks`](Memory::store_blocks)).
    #[allow(dead_code)] // Reached through its expand, from `Tile::fetch_transposed_from`.
    pub(crate) fn fetch_blocks<T: Numeric, W: Size>(
        &self,
        src: &Memory,
        #[comptime] space: &Space,
        fetched: &mut Array<Vector<T, W>>,
    ) {
        let elem = elem_type_of::<T>();
        let bits = comptime!(elem.size_bits());
        let transposed = self.fetches_transposed(src, space, bits);
        comptime!(assert!(
            transposed,
            "Memory::fetch_blocks: only a whole stage laid along a run, filled from a plain, \
             direct source of a known share, should be fetched into registers by blocks"
        ));
        let check = comptime!(src.access.overhang.masks());
        let w = W::value();
        comptime!(fill_extent(space, w, w, check));
        let blocks = self.run_blocks(space, w, bits);
        let fill = comptime!(self.access.copy.clone());
        let s = src.edged(
            src.window_view_storage::<T, W>(comptime!(Guard::Checked)),
            check,
        );
        let layout = self.layout.clone();
        fetch_blocks::<T, W>(&s, &layout, blocks, &fill, fetched);
    }

    /// Transpose and store what [`fetch_blocks`](Memory::fetch_blocks) read into this stage over
    /// `space`. The caller synchronizes the stage's owner as around any register store.
    #[allow(dead_code)] // Reached through its expand, from `Tile::store_transposed`.
    pub(crate) fn store_blocks<T: Numeric, W: Size>(
        &mut self,
        #[comptime] space: &Space,
        fetched: &Array<Vector<T, W>>,
    ) {
        let elem = elem_type_of::<T>();
        let blocks = self.run_blocks(space, W::value(), comptime!(elem.size_bits()));
        let fill = comptime!(self.access.copy.clone());
        let layout = self.layout.clone();
        let size!(R) = comptime!(blocks.run);
        let d = self.lines_storage_mut::<T, R>();
        store_blocks::<T, R, W>(d, &layout, blocks, &fill, fetched);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: Dimension = Dimension(0);
    const N: Dimension = Dimension(1);

    /// A `16 × 16` `[k, n]` stage laid `k` innermost, stored up to a chunk of eight `f16` at a
    /// time.
    fn along_k() -> (Space, StageOrder) {
        let order = StageOrder::AlongRun {
            dimension: K,
            run: 8,
        };
        (Space::new(&[(K, 16), (N, 16)]), order)
    }

    fn blocks(units: usize) -> RunBlocks {
        let (space, order) = along_k();
        RunBlocks::new(&space, order, 8, 16, &Level::copied_by_cube(units))
    }

    /// Two units take the stage's blocks of four rows of `k`, an 8-byte store each, four along
    /// `k` before the next source line of a row.
    #[test]
    fn a_balanced_fill_stores_eight_bytes_first() {
        let blocks = blocks(2);
        assert_eq!((blocks.run, blocks.count, blocks.across), (4, 8, 4));
        assert_eq!(blocks.inner, 2);
    }

    /// Where no 8-byte run deals every unit as many blocks, the fill narrows its stores to four
    /// bytes: sixteen units.
    #[test]
    fn the_run_narrows_until_every_unit_takes_as_many_blocks() {
        assert_eq!((blocks(8).run, blocks(8).count), (4, 8));
        assert_eq!((blocks(16).run, blocks(16).count), (2, 16));
        assert!(blocks(16).balanced(&Level::copied_by_cube(16)));
    }

    /// Where no run balances the units, the fill stores the narrowest run, which leaves the
    /// fewest of them idle.
    #[test]
    fn unbalanced_units_take_the_narrowest_run() {
        let blocks = blocks(64);
        assert_eq!(blocks.run, 2);
        assert!(!blocks.balanced(&Level::copied_by_cube(64)));
    }

    /// A unit holds `run` source lines for each of its turns.
    #[test]
    fn a_unit_holds_a_run_of_lines_a_turn() {
        let fill = Level::copied_by_cube(4);
        let blocks = blocks(4);
        assert_eq!(blocks.fetched_lines(&fill), Some(8));
    }
}
