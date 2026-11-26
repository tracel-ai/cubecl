//! [`Delivery::SyncPerUnit`]: every unit loads its own lines and stores them into the stage, so
//! they have landed when the fill returns, and a line can be decoded, padded, scanned or summed on
//! its way. The units walk the stage in its own order: their reads go through the cache, which
//! gathers a warp's lines from several rows back into whole transactions, and their writes stay
//! free of bank conflicts.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use crate::stage::transport::base::{store_line, uniform_store_offsets};
use crate::stage::transport::padded::read_stage_line;
use crate::stage::transport::prefetch::{first_store_offset, store_turn};
use crate::*;

/// What the units' own loads and stores state.
pub(crate) const SYNC_PER_UNIT: Engine = Engine {
    issuer: Issuer::Unit,
    route: Route::Registers,
    present: |_| true,
    line_bytes: None,
    box_edge: None,
    stage_alignment: 0,
};

#[cube]
impl Memory {
    /// Write the stage's `total` lines, dealt to the units `fill` names ([`Walk::copy`]): each
    /// the source's line at the coordinates the stage holds it at, assembled out of narrower
    /// cells where the stage is padded ([`Padding`]), stored at its unit's first line's offset
    /// plus a comptime one where every unit's turns agree ([`Walk::uniform_turn_offsets`]).
    pub(crate) fn sync_lines<I2: Numeric, WP2: Size, SW: Size>(
        d: &mut [Vector<I2, WP2>],
        s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
        layout: &BufferLayout,
        total: usize,
        #[comptime] fill: Level,
        #[comptime] padding: Option<Padding>,
    ) {
        let offsets = uniform_store_offsets(layout, &fill, total, 1usize, false);
        match comptime!(offsets) {
            Some(offsets) => Memory::sync_lines_in_turns::<I2, WP2, SW>(
                d, s, layout, total, fill, padding, offsets,
            ),
            None => Memory::sync_lines_walked::<I2, WP2, SW>(d, s, layout, total, fill, padding),
        }
    }

    /// [`sync_lines`](Memory::sync_lines) where every unit's turns add the comptime `offsets`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
    fn sync_lines_in_turns<I2: Numeric, WP2: Size, SW: Size>(
        d: &mut [Vector<I2, WP2>],
        s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
        layout: &BufferLayout,
        total: usize,
        #[comptime] fill: Level,
        #[comptime] padding: Option<Padding>,
        #[comptime] offsets: Vec<Vec<usize>>,
    ) {
        let lines = Walk::copy(&fill, total);
        let base = layout.line_offset(lines.copied_line(0));
        #[unroll]
        for t in 0..offsets.len() {
            if lines.takes_turn(t) {
                let i = lines.copied_line(t);
                let value =
                    read_stage_line::<I2, WP2, SW>(s, &layout.line_coords(i), comptime!(padding));
                let offset = comptime!(offsets[t][0]);
                d[base + offset] = value;
            }
        }
    }

    /// [`sync_lines`](Memory::sync_lines) with each line's offset worked out for the line.
    fn sync_lines_walked<I2: Numeric, WP2: Size, SW: Size>(
        d: &mut [Vector<I2, WP2>],
        s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
        layout: &BufferLayout,
        total: usize,
        #[comptime] fill: Level,
        #[comptime] padding: Option<Padding>,
    ) {
        for line in Walk::copy(&fill, total) {
            let i = line.copied_line();
            let value =
                read_stage_line::<I2, WP2, SW>(s, &layout.line_coords(i), comptime!(padding));
            store_line::<I2, WP2>(d, layout, i, value);
        }
    }

    /// Write every `SW`-wide line of the source of a stage of one value a line, `lines` of them
    /// counted in the source's order and dealt to the units `fill` names ([`Walk::copy`]), each
    /// of its values at its own cell of the stage: what fills a stage laid down in another order
    /// than its source's ([`StageOrder::AlongRun`]).
    pub(crate) fn scatter_lines<I2: Numeric, D: Size, SW: Size>(
        d: &mut [Vector<I2, D>],
        s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
        layout: &BufferLayout,
        lines: usize,
        #[comptime] fill: Level,
    ) {
        let width = SW::value();
        let offsets = uniform_store_offsets(layout, &fill, lines, width, true);
        match comptime!(offsets) {
            Some(offsets) => {
                Memory::scatter_lines_in_turns::<I2, D, SW>(d, s, layout, lines, fill, offsets)
            }
            None => Memory::scatter_lines_walked::<I2, D, SW>(d, s, layout, lines, fill),
        }
    }

    /// [`scatter_lines`](Memory::scatter_lines) where every unit's turns add the comptime
    /// `offsets`.
    #[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
    fn scatter_lines_in_turns<I2: Numeric, D: Size, SW: Size>(
        d: &mut [Vector<I2, D>],
        s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
        layout: &BufferLayout,
        lines: usize,
        #[comptime] fill: Level,
        #[comptime] offsets: Vec<Vec<usize>>,
    ) {
        let width = SW::value();
        let walk = Walk::copy(&fill, lines);
        let base = first_store_offset(layout, walk.copied_line(0), width, true);
        #[unroll]
        for t in 0..offsets.len() {
            if walk.takes_turn(t) {
                let line = walk.copied_line(t);
                let value = s.read(layout.source_line_coords(line, width));
                store_turn::<I2, D, SW>(
                    d,
                    layout,
                    line,
                    base,
                    value,
                    comptime!(Some(offsets[t].clone())),
                    true,
                );
            }
        }
    }

    /// [`scatter_lines`](Memory::scatter_lines) with each value's cell worked out for its line.
    fn scatter_lines_walked<I2: Numeric, D: Size, SW: Size>(
        d: &mut [Vector<I2, D>],
        s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
        layout: &BufferLayout,
        lines: usize,
        #[comptime] fill: Level,
    ) {
        for line in Walk::copy(&fill, lines) {
            let line = line.copied_line();
            let value = s.read(layout.source_line_coords(line, SW::value()));
            store_turn::<I2, D, SW>(d, layout, line, 0usize, value, comptime!(None), true);
        }
    }
}
