//! [`Delivery::AsyncPerUnit`]: each unit hands its lines to the copy engine (`cp.async`) rather
//! than loading and storing them, so they land in the stage without passing through its
//! registers, after the fill has returned, on the arrivals the slot's barrier counts.
//!
//! A unit takes as many lines as under the straight fill, but walks them in the source's order
//! rather than the stage's. The copy bypasses L1, so neighbouring lanes must read neighbouring
//! source bytes: walking a stage blocked into instruction tiles in its own order hands a warp a
//! sliver of each of many rows, and reads more from DRAM than the stage holds.
//!
//! Nothing here waits: the slot's barrier is handed the copies and flips once they land
//! ([`Meeting::release_write`](crate::Meeting)), or a blocking copy waits on one of its own
//! ([`Memory::load_from`]).

use cubecl::{
    ir::DeviceProperties,
    prelude::barrier::{copy_async, copy_async_checked},
    prelude::*,
};

use cubecl_std::tensor::layout::{CoordsDyn, Layout, LayoutExpand};

use crate::*;

/// The widths of one copy the engine takes, in bytes: `cp.async` moves 4, 8 or 16.
const COPY_BYTES: [usize; 3] = [4, 8, 16];

/// What the copy engine states.
pub(crate) const ASYNC_PER_UNIT: Engine = Engine {
    issuer: Issuer::Unit,
    route: Route::MemoryToMemory,
    present: |device: &DeviceProperties| device.features.copy_async,
    line_bytes: Some(&COPY_BYTES),
    box_edge: None,
    stage_alignment: 0,
};

#[cube]
impl Memory {
    /// [`Memory::sync_lines`] with every line handed to the copy engine: the same lines per unit,
    /// dealt the same way, numbered in the source's order ([`BufferLayout::row_major_coords`]).
    ///
    /// A move of its own rather than an arm of `sync_lines`: the copy reads and writes lines of one
    /// width, which `sync_lines`, generic over a source served narrower, cannot hand it.
    pub(crate) fn async_lines<I2: Numeric, W: Size>(
        d: &mut [Vector<I2, W>],
        s: &Masked<'_, Vector<I2, W>, CoordsDyn>,
        layout: &BufferLayout,
        total: usize,
        #[comptime] fill: Level,
    ) {
        // Both read outside `comptime!`: in there, `size` is the host's `size_of` of the generic and
        // `value` is not expanded at all.
        let width = W::value();
        let elem_size = I2::size().comptime();
        let bytes = comptime!(width * elem_size);
        comptime!(assert!(
            COPY_BYTES.contains(&bytes),
            "Memory::async_lines: the copy engine moves 4, 8 or 16 bytes at once, and this stage's lines \
         are {bytes}: serve the operand in lines one copy moves"
        ));
        for line in Walk::copy(&fill, total) {
            copy_line_async::<I2, W>(d, s, layout, line.coord(Level::COPIED));
        }
    }
}

/// Copy the `i`-th line of the stage, counted in the source's order, asynchronously: the source
/// line at those coordinates, whole, or zeros where the source masks it off. It has not landed
/// when this returns.
#[cube]
fn copy_line_async<I2: Numeric, W: Size>(
    d: &mut [Vector<I2, W>],
    s: &Masked<'_, Vector<I2, W>, CoordsDyn>,
    layout: &BufferLayout,
    i: usize,
) {
    let coords = layout.row_major_coords(i);
    let offset = layout.to_source_pos(coords.clone());
    let run = s.item_run(coords);
    let width = W::value();
    let dst = &mut d[offset..offset + 1];
    if comptime!(s.check) {
        copy_async_checked(run, dst, comptime!(width as u32));
    } else {
        copy_async(run, dst, comptime!(width as u32));
    }
}
