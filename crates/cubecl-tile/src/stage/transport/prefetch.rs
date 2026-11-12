//! The straight transport split around a contraction: lines fetched into registers before it,
//! stored into the stage after.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use super::padded::read_stage_line;
use super::straight::fill_extent;
use crate::*;

/// Most scalars of both operands' stages one unit may hold in registers beside an accumulator.
pub(crate) const MOST_FETCHED_SCALARS: usize = 128;

#[cube]
impl Memory {
    /// The turns each unit takes over this stage's lines, unless its shape is not static or its
    /// fill does not state the units it spreads over.
    #[allow(dead_code)] // Reached through its expand, from `fetched_turns` and `fetched_scalars`.
    fn known_turns(&self) -> comptime_type!(Option<usize>) {
        let stage_lines = self.stage_lines().constant();
        let fill = comptime!(self.access.fill);
        comptime!(
            stage_lines
                .filter(|_| fill.count > 0)
                .and_then(|lines| fill.unrolled(lines as usize).count())
        )
    }

    /// The turns each unit takes over this stage's lines, every one written out; known wherever
    /// [`fetches_straight`](Memory::fetches_straight) holds.
    #[allow(dead_code)] // Reached through its expand, from `fetch_straight` and `store_fetched`.
    fn fetched_turns(&self) -> comptime_type!(Turns) {
        let stage_lines = self.stage_lines().constant();
        let fill = comptime!(self.access.fill);
        comptime!(fill.unrolled(
            stage_lines.expect("Memory: a stage fetched into registers has a known share") as usize
        ))
    }

    /// Scalars of this stage one unit holds in registers between fetch and store, unless its share
    /// is unknown.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetches_in_registers`.
    pub(crate) fn fetched_scalars(&self) -> comptime_type!(Option<usize>) {
        let turns = self.known_turns();
        comptime!(turns.map(|turns| turns * self.store.vector_size))
    }

    /// Whether this stage is filled from `src` by copying its lines as they lie, each unit's share
    /// known: all [`fetch_straight`](Memory::fetch_straight) moves.
    #[allow(dead_code)] // Reached through its expand, from `Tile::fetches_straight_from`.
    pub(crate) fn fetches_straight(&self, src: &Memory) -> comptime_type!(bool) {
        let turns = self.known_turns();
        comptime!(
            turns.is_some()
                && self.store.packing.as_stored()
                && src.store.packing.as_stored()
                && self.access.whole
                && !self.access.overhang.masks()
                && self.access.write == Write::Replace
                && self.store.vector_size == src.store.vector_size
                && self.projection.is_direct()
                && src.projection.is_direct()
        )
    }

    /// This stage's line count.
    fn stage_lines(&self) -> usize {
        let shape = self.layout.physical_shape.clone();
        let plen = shape.len().comptime();
        shape
            .product(comptime!((0..plen).collect::<Vec<_>>()))
            .cast::<usize>()
    }

    /// Read this unit's lines of `src` into `fetched`; only a plain, direct, whole stage at its
    /// source's width.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn fetch_straight<T: Numeric, W: Size>(
        &self,
        src: &Memory,
        #[comptime] space: &Space,
        fetched: &mut Array<Vector<T, W>>,
    ) {
        self.refuse_unfetchable(src);
        let w = comptime!(self.store.vector_size);
        let fetched_width = W::value();
        comptime!(assert_eq!(
            fetched_width, w,
            "Memory::fetch_straight: registers of one width fetch a stage of another"
        ));
        let check = comptime!(src.access.overhang.masks());
        comptime!(fill_extent(space, w, w, check));
        let fill = comptime!(self.access.fill);
        let turns = self.fetched_turns();
        let total = self.stage_lines();
        let s = Masked::new(
            src.window_view_storage::<T, W>(comptime!(Guard::Checked)),
            check,
        );
        let mut lines = FetchedLines::<T, W>::new(fetched, &s, &self.layout);
        FillUnits::move_lines::<FetchedLines<T, W>>(fill, total, turns, &mut lines);
    }

    /// Write this unit's `fetched` lines into the stage. The caller must `sync_cube` before (after
    /// the stage's last read) and after (before the next read).
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn store_fetched<T: Numeric, W: Size>(&mut self, fetched: &Array<Vector<T, W>>) {
        let fill = comptime!(self.access.fill);
        let turns = self.fetched_turns();
        let total = self.stage_lines();
        let layout = self.layout.clone();
        let d = self.lines_storage_mut::<T, W>();
        let mut lines = StoredLines::<T, W>::new(d, fetched, &layout);
        FillUnits::move_lines::<StoredLines<T, W>>(fill, total, turns, &mut lines);
    }

    /// Refuse a pairing this transport cannot move.
    fn refuse_unfetchable(&self, src: &Memory) {
        let straight = self.fetches_straight(src);
        comptime!(assert!(
            straight,
            "Memory::fetch_straight: only a plain, direct, whole stage of a known share, filled \
             at its source's width, is fetched into registers"
        ));
    }
}

/// A register prefetch's line: the source's line at the coordinates the stage's line holds, read
/// into the register of this unit's turn.
#[derive(CubeType)]
struct FetchedLines<'a, T: Numeric, W: Size> {
    fetched: &'a mut Array<Vector<T, W>>,
    s: &'a Masked<'a, Vector<T, W>, CoordsDyn>,
    layout: &'a BufferLayout,
}

#[cube]
impl<'a, T: Numeric, W: Size> FetchedLines<'a, T, W> {
    fn new(
        fetched: &'a mut Array<Vector<T, W>>,
        s: &'a Masked<'a, Vector<T, W>, CoordsDyn>,
        layout: &'a BufferLayout,
    ) -> Self {
        FetchedLines::<'a, T, W> { fetched, s, layout }
    }
}

#[cube]
impl<'a, T: Numeric, W: Size> LineMove for FetchedLines<'a, T, W> {
    fn move_line(&mut self, line: usize, #[comptime] turn: Option<usize>) {
        let turn = comptime!(turn.expect("FetchedLines: a fetch writes every turn out"));
        self.fetched[turn] =
            read_stage_line::<T, W, W>(self.s, &self.layout.line_coords(line), comptime!(None));
    }
}

/// A register prefetch's line stored: the register of this unit's turn, into the stage's line.
#[derive(CubeType)]
struct StoredLines<'a, T: Numeric, W: Size> {
    d: &'a mut [Vector<T, W>],
    fetched: &'a Array<Vector<T, W>>,
    layout: &'a BufferLayout,
}

#[cube]
impl<'a, T: Numeric, W: Size> StoredLines<'a, T, W> {
    fn new(
        d: &'a mut [Vector<T, W>],
        fetched: &'a Array<Vector<T, W>>,
        layout: &'a BufferLayout,
    ) -> Self {
        StoredLines::<'a, T, W> { d, fetched, layout }
    }
}

#[cube]
impl<'a, T: Numeric, W: Size> LineMove for StoredLines<'a, T, W> {
    fn move_line(&mut self, line: usize, #[comptime] turn: Option<usize>) {
        let turn = comptime!(turn.expect("StoredLines: a store writes every turn out"));
        self.d[self.layout.line_offset(line)] = self.fetched[turn];
    }
}
