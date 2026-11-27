//! [`Delivery::SyncPerUnit`]: every unit loads its own lines and stores them into the stage, so
//! they have landed when the fill returns, and a line can be decoded, padded, scanned or summed on
//! its way. The units walk the stage in its own order: their reads go through the cache, which
//! gathers a warp's lines from several rows back into whole transactions, and their writes stay
//! free of bank conflicts.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use crate::stage::transport::base::{store_line, uniform_store_offsets};
use crate::stage::transport::padded::read_stage_line;
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
        let offsets = uniform_store_offsets(layout, &fill, total);
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
}
