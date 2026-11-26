//! The straight transport split around a contraction: lines fetched into registers before it,
//! stored into the stage after.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use super::base::{store_line, store_scattered, uniform_store_offsets};
use super::padded::read_stage_line;
use super::straight::fill_extent;
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
    fn known_turns(&self, #[comptime] width: usize) -> comptime_type!(Option<usize>) {
        let stage_lines = self.stage_lines().constant();
        let fill = comptime!(self.access.copy.clone());
        let vector_size = comptime!(self.store.vector_size);
        comptime!(stage_lines.and_then(|lines| {
            let source_lines = lines as usize * vector_size / width;
            fill.copy_turns(source_lines)
        }))
    }

    /// The turns one unit takes over this stage's `width`-wide source lines; known wherever
    /// [`fetches_straight`](Memory::fetches_straight) holds.
    #[allow(dead_code)] // Reached through its expand, from `fetch_straight` and `store_fetched`.
    fn turns(&self, #[comptime] width: usize) -> comptime_type!(usize) {
        let turns = self.known_turns(width);
        comptime!(turns.expect("Memory: a stage fetched into registers should have a known share"))
    }

    /// The width a fill of this stage from `src` reads each source line at: the source's own where
    /// this stage holds one value a line and the source more, each line's values scattered over
    /// the stage's cells, this stage's own otherwise.
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
    pub(crate) fn fetched_scalars(&self, src: &Memory) -> comptime_type!(Option<usize>) {
        let width = self.fill_width(src);
        let turns = self.known_turns(width);
        comptime!(turns.map(|turns| turns * width))
    }

    /// Whether this stage is filled from `src` by copying its lines as they lie, each unit's share
    /// known: all [`fetch_straight`](Memory::fetch_straight) moves. A stage of one value a line
    /// takes a wider source's lines whole and scatters their values.
    #[allow(dead_code)] // Reached through its expand, from `Tile::fetches_straight_from`.
    pub(crate) fn fetches_straight(&self, src: &Memory) -> comptime_type!(bool) {
        let width = self.fill_width(src);
        let turns = self.known_turns(width);
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
        let straight = self.fetches_straight(src);
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
        let scatters = comptime!(self.store.vector_size != w);
        comptime!(fill_extent(space, w, w, check));
        let layout = self.layout.clone();
        let turns = self.turns(w);
        let lines = Walk::copy(&comptime!(self.access.copy.clone()), self.source_lines(w));
        let s = src.edged(
            src.window_view_storage::<T, W>(comptime!(Guard::Checked)),
            check,
        );
        // Each turn's line is held in a register of its own, so every turn is written out.
        #[unroll]
        for t in 0..turns {
            if lines.takes_turn(t) {
                let coords = fill_line_coords(&layout, lines.copied_line(t), w, scatters);
                fetched[t] = read_stage_line::<T, W, W>(&s, &coords, comptime!(None));
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
        let w = W::value();
        let scatters = comptime!(self.store.vector_size != w);
        let turns = self.turns(w);
        let lines = Walk::copy(&comptime!(self.access.copy.clone()), self.source_lines(w));
        let layout = self.layout.clone();
        let fill = comptime!(self.access.copy.clone());
        let offsets = uniform_store_offsets(&layout, &fill, self.source_lines(w), w, scatters);
        let size!(SW) = comptime!(self.store.vector_size);
        let d = self.lines_storage_mut::<T, SW>();
        let base = first_store_offset(&layout, lines.copied_line(0), w, scatters);
        #[unroll]
        for t in 0..turns {
            if lines.takes_turn(t) {
                let line = lines.copied_line(t);
                store_turn::<T, SW, W>(
                    d,
                    &layout,
                    line,
                    base,
                    fetched[t],
                    comptime!(offsets.as_ref().map(|offsets| offsets[t].clone())),
                    scatters,
                );
            }
        }
    }
}

/// The logical coordinates the fill of a stage laid out as `layout` reads its `width`-wide source
/// line `j` at: the stage's own line `j` where it is filled at its own width, the source's line
/// `j` counted in the source's order where it `scatters` each line's values.
#[cube]
pub(crate) fn fill_line_coords(
    layout: &BufferLayout,
    j: usize,
    #[comptime] width: usize,
    #[comptime] scatters: bool,
) -> CoordsDyn {
    match scatters {
        true => layout.source_line_coords(j, width),
        false => layout.line_coords(j),
    }
}

/// Where the first value of source line `j` lands in a stage laid out as `layout`: its own cell
/// where the fill `scatters` `width`-wide lines' values, the stage's line `j` otherwise.
#[cube]
pub(crate) fn first_store_offset(
    layout: &BufferLayout,
    j: usize,
    #[comptime] width: usize,
    #[comptime] scatters: bool,
) -> usize {
    match scatters {
        true => layout.cell_offset(&layout.source_line_coords(j, width), width, 0usize),
        false => layout.line_offset(j),
    }
}

/// Source line `line`'s `value` written into the stage `d` laid out as `layout`: at `base`, the
/// unit's first line's offset, plus the comptime `offsets` its turn adds where every unit agrees,
/// or at offsets worked out for the line otherwise; each value at its own cell where the fill
/// `scatters` them.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn store_turn<T: Numeric, D: Size, W: Size>(
    d: &mut [Vector<T, D>],
    layout: &BufferLayout,
    line: usize,
    base: usize,
    value: Vector<T, W>,
    #[comptime] offsets: Option<Vec<usize>>,
    #[comptime] scatters: bool,
) {
    match offsets {
        Some(offsets) => store_at_offsets::<T, D, W>(d, base, value, offsets, scatters),
        None => store_walked::<T, D, W>(d, layout, line, value, scatters),
    }
}

/// `value` written at `base` plus each of `offsets`: one offset a value where the fill
/// `scatters`, the whole line at the one offset otherwise.
#[cube]
#[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
fn store_at_offsets<T: Numeric, D: Size, W: Size>(
    d: &mut [Vector<T, D>],
    base: usize,
    value: Vector<T, W>,
    #[comptime] offsets: Vec<usize>,
    #[comptime] scatters: bool,
) {
    if scatters {
        #[unroll]
        for v in 0..offsets.len() {
            let offset = comptime!(offsets[v]);
            d[base + offset] = Vector::<T, D>::cast_from(value.extract(v));
        }
    } else {
        let offset = comptime!(offsets[0]);
        d[base + offset] = Vector::<T, D>::cast_from(value);
    }
}

/// `value` of source line `line` written where the stage laid out as `layout` holds it, its
/// offsets worked out for the line.
#[cube]
fn store_walked<T: Numeric, D: Size, W: Size>(
    d: &mut [Vector<T, D>],
    layout: &BufferLayout,
    line: usize,
    value: Vector<T, W>,
    #[comptime] scatters: bool,
) {
    if scatters {
        store_scattered::<T, D, W>(
            d,
            layout,
            &layout.source_line_coords(line, W::value()),
            value,
        );
    } else {
        store_line::<T, D>(d, layout, line, Vector::<T, D>::cast_from(value));
    }
}
