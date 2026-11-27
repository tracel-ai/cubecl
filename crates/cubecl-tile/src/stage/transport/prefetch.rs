//! The straight transport split around a contraction: lines fetched into registers before it,
//! stored into the stage after.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use super::base::{store_line, uniform_store_offsets};
use super::padded::read_stage_line;
use super::straight::fill_extent;
use super::transposed::RunBlocks;
use crate::*;

/// Most 32-bit registers of both operands' stages one unit may hold beside an accumulator: a
/// quarter of the 256 a unit addresses on the GPUs this was measured on, the 128 halves a stage
/// was first held at.
pub(crate) const MOST_FETCHED_REGISTERS: usize = 64;

#[cube]
impl Memory {
    /// The turns one unit takes over this stage's `width`-wide source lines, unless its shape is
    /// not static or its fill does not state the units it deals them to.
    #[allow(dead_code)] // Reached through its expand, from `turns` and `fetched_scalars`.
    fn known_turns(
        &self,
        #[comptime] width: usize,
        #[comptime] elem_bits: usize,
    ) -> comptime_type!(Option<usize>) {
        let stage_lines = self.stage_lines().constant();
        let fill = comptime!(self.access.copy.clone());
        let vector_size = comptime!(self.store.vector_size);
        let constants = self.layout.constants();
        let order = comptime!(self.layout.order);
        comptime!(stage_lines.and_then(|lines| match order {
            StageOrder::AsSource => fill.copy_turns(lines as usize * vector_size / width),
            // A stage laid along a run is fetched a block of lines along the run a turn.
            StageOrder::AlongRun { .. } => constants.and_then(|layout| {
                RunBlocks::new(&layout, width, elem_bits, &fill).fetched_lines(&fill)
            }),
        }))
    }

    /// The turns one unit takes over this stage's `width`-wide source lines; known wherever
    /// [`fetches_straight`](Memory::fetches_straight) holds.
    #[allow(dead_code)] // Reached through its expand, from `fetch_straight` and `store_fetched`.
    fn turns(
        &self,
        #[comptime] width: usize,
        #[comptime] elem_bits: usize,
    ) -> comptime_type!(usize) {
        let turns = self.known_turns(width, elem_bits);
        comptime!(turns.expect("Memory: a stage fetched into registers should have a known share"))
    }

    /// The width a fill of this stage from `src` reads each source line at: the source's own where
    /// this stage holds one value a line and the source more, its blocks transposed in registers
    /// ([`StageOrder::AlongRun`]), this stage's own otherwise.
    #[allow(dead_code)] // Reached through its expand, from the register fetch's counts.
    pub(crate) fn fill_width(&self, src: &Memory) -> comptime_type!(usize) {
        comptime!(match (self.store.vector_size, src.store.vector_size) {
            (1, source) => source,
            (own, _) => own,
        })
    }

    /// Scalars of this stage's fill from `src` one unit holds in registers between fetch and
    /// store, unless its share is unknown.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetches_in_registers`.
    pub(crate) fn fetched_scalars(
        &self,
        src: &Memory,
        #[comptime] elem_bits: usize,
    ) -> comptime_type!(Option<usize>) {
        let width = self.fill_width(src);
        let turns = self.known_turns(width, elem_bits);
        comptime!(turns.map(|turns| turns * width))
    }

    /// Whether this stage is filled from `src` by copying its lines as they lie, each unit's share
    /// known: all [`fetch_straight`](Memory::fetch_straight) moves. A stage laid along a run takes
    /// its source's lines whole, a block of them along the run a turn.
    #[allow(dead_code)] // Reached through its expand, from `Tile::fetches_straight_from`.
    pub(crate) fn fetches_straight(
        &self,
        src: &Memory,
        #[comptime] elem_bits: usize,
    ) -> comptime_type!(bool) {
        let width = self.fill_width(src);
        let turns = self.known_turns(width, elem_bits);
        comptime!(
            turns.is_some()
                && self.store.packing.as_stored()
                && src.store.packing.as_stored()
                && self.access.whole
                && !self.access.overhang.masks()
                && self.access.write == Write::Replace
                && width == src.store.vector_size
                && self.projection.is_direct()
                && src.projection.is_direct()
        )
    }

    /// This stage's line count.
    pub(crate) fn stage_lines(&self) -> usize {
        let shape = self.layout.physical_shape.clone();
        let plen = shape.len().comptime();
        shape
            .product(comptime!((0..plen).collect::<Vec<_>>()))
            .cast::<usize>()
    }

    /// The count of `width`-wide source lines this stage's fill reads.
    pub(crate) fn source_lines(&self, #[comptime] width: usize) -> usize {
        let scale = comptime!(self.store.vector_size);
        self.stage_lines()
            .times(comptime!(scale).runtime())
            .divided_by(comptime!(width).runtime())
    }

    /// Read this unit's lines of `src` into `fetched`; only a plain, direct, whole stage at its
    /// source's width, or one value a line laid out of the source's lines whole.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn fetch_straight<T: Numeric, W: Size>(
        &self,
        src: &Memory,
        #[comptime] space: &Space,
        fetched: &mut Array<Vector<T, W>>,
    ) {
        // Only past `fetches_straight`: the schedule fetches no other stage into registers.
        let elem = elem_type_of::<T>();
        let straight = self.fetches_straight(src, comptime!(elem.size_bits()));
        comptime!(assert!(
            straight,
            "Memory::fetch_straight: only a plain, direct, whole stage of a known share, filled \
             at its source's width, should be fetched into registers"
        ));
        let w = self.fill_width(src);
        let fetched_width = W::value();
        comptime!(assert_eq!(
            fetched_width, w,
            "Memory::fetch_straight: registers of one width fetch a stage of another"
        ));
        let check = comptime!(src.access.overhang.masks());
        let transposes = comptime!(self.layout.order != StageOrder::AsSource);
        comptime!(fill_extent(space, w, w, check));
        let s = src.edged(
            src.window_view_storage::<T, W>(comptime!(Guard::Checked)),
            check,
        );
        if comptime!(transposes) {
            self.fetch_blocks::<T, W>(&s, fetched);
        } else {
            self.fetch_lines::<T, W>(&s, fetched);
        }
    }

    /// [`fetch_straight`](Memory::fetch_straight) of a stage filled line for line.
    fn fetch_lines<T: Numeric, W: Size>(
        &self,
        s: &Masked<'_, Vector<T, W>, CoordsDyn>,
        fetched: &mut Array<Vector<T, W>>,
    ) {
        let w = W::value();
        let layout = self.layout.clone();
        let elem = elem_type_of::<T>();
        let turns = self.turns(w, comptime!(elem.size_bits()));
        let lines = Walk::copy(&comptime!(self.access.copy.clone()), self.source_lines(w));
        // Each turn's line is held in a register of its own, so every turn is written out.
        #[unroll]
        for t in 0..turns {
            if lines.takes_turn(t) {
                let coords = layout.line_coords(lines.copied_line(t));
                fetched[t] = read_stage_line::<T, W, W>(s, &coords, comptime!(None));
            }
        }
    }

    /// Write this unit's `fetched` lines into the stage, each at the comptime offset from the
    /// unit's first line its turn adds where every unit's turns agree
    /// ([`Walk::uniform_turn_offsets`]). The caller must synchronize the stage's owner before
    /// (after the stage's last read) and after (before the next read): the cube, or the plane for a
    /// stage a plane owns.
    #[allow(dead_code)] // Reached through its expand, from `Stages::prefetched`.
    pub(crate) fn store_fetched<T: Numeric, W: Size>(&mut self, fetched: &Array<Vector<T, W>>) {
        if comptime!(self.layout.order != StageOrder::AsSource) {
            self.store_blocks::<T, W>(fetched);
        } else {
            self.store_lines::<T, W>(fetched);
        }
    }

    /// [`store_fetched`](Memory::store_fetched) of a stage filled line for line.
    fn store_lines<T: Numeric, W: Size>(&mut self, fetched: &Array<Vector<T, W>>) {
        let w = W::value();
        let elem = elem_type_of::<T>();
        let turns = self.turns(w, comptime!(elem.size_bits()));
        let lines = Walk::copy(&comptime!(self.access.copy.clone()), self.source_lines(w));
        let layout = self.layout.clone();
        let fill = comptime!(self.access.copy.clone());
        let offsets = uniform_store_offsets(&layout, &fill, self.source_lines(w));
        let size!(SW) = comptime!(self.store.vector_size);
        let d = self.lines_storage_mut::<T, SW>();
        let base = layout.line_offset(lines.copied_line(0));
        #[unroll]
        for t in 0..turns {
            if lines.takes_turn(t) {
                let value = Vector::<T, SW>::cast_from(fetched[t]);
                let offset = comptime!(offsets.as_ref().map(|offsets| offsets[t][0]));
                #[comptime]
                match offset {
                    Some(offset) => {
                        d[base + offset] = value;
                    }
                    None => store_line::<T, SW>(d, &layout, lines.copied_line(t), value),
                }
            }
        }
    }
}
