//! The straight transport split around a contraction: lines fetched into registers before it,
//! stored into the stage after.

use cubecl::prelude::*;

use super::padded::read_stage_line;
use super::straight::fill_extent;
use crate::*;

/// Most scalars of both operands' stages one unit may hold in registers beside an accumulator.
pub(crate) const MOST_FETCHED_SCALARS: usize = 128;

#[cube]
impl Memory {
    /// The turns one unit takes over this stage's lines, unless its shape is not static or its
    /// fill does not state the units it deals them to.
    #[allow(dead_code)] // Reached through its expand, from `turns` and `fetched_scalars`.
    fn known_turns(&self) -> comptime_type!(Option<usize>) {
        let stage_lines = self.stage_lines().constant();
        let fill = comptime!(self.access.copy.clone());
        comptime!(stage_lines.and_then(|lines| fill.copy_turns(lines as usize)))
    }

    /// The turns one unit takes over this stage's lines; known wherever
    /// [`fetches_straight`](Memory::fetches_straight) holds.
    #[allow(dead_code)] // Reached through its expand, from `fetch_straight` and `store_fetched`.
    fn turns(&self) -> comptime_type!(usize) {
        let turns = self.known_turns();
        comptime!(turns.expect("Memory: a stage fetched into registers has a known share"))
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
        let layout = self.layout.clone();
        let turns = self.turns();
        let lines = Walk::copy(&comptime!(self.access.copy.clone()), self.stage_lines());
        let s = Masked::new(
            src.window_view_storage::<T, W>(comptime!(Guard::Checked)),
            check,
        );
        // Each turn's line is held in a register of its own, so every turn is written out.
        #[unroll]
        for t in 0..turns {
            if lines.takes_turn(t) {
                let i = lines.region(t).coord(Level::COPIED);
                fetched[t] =
                    read_stage_line::<T, W, W>(&s, &layout.line_coords(i), comptime!(None));
            }
        }
    }

    /// Write this unit's `fetched` lines into the stage. The caller must `sync_cube` before (after
    /// the stage's last read) and after (before the next read).
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn store_fetched<T: Numeric, W: Size>(&mut self, fetched: &Array<Vector<T, W>>) {
        let turns = self.turns();
        let lines = Walk::copy(&comptime!(self.access.copy.clone()), self.stage_lines());
        let layout = self.layout.clone();
        let d = self.lines_storage_mut::<T, W>();
        #[unroll]
        for t in 0..turns {
            if lines.takes_turn(t) {
                d[layout.line_offset(lines.region(t).coord(Level::COPIED))] = fetched[t];
            }
        }
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
