//! The asynchronous fill under a straight or packed transport ([`Delivery::AsyncPerUnit`]): each
//! unit hands its lines to the copy engine (`cp.async`) rather than loading and storing them, so
//! they land in the stage without passing through its registers, after the fill has returned.
//!
//! A unit takes as many lines as under the straight fill, but walks them in the source's order
//! rather than the stage's. The copy bypasses L1, so neighbouring lanes must read neighbouring
//! source bytes: walking a stage blocked into instruction tiles in its own order hands a warp a
//! sliver of each of many rows, and reads more from DRAM than the stage holds.
//!
//! Nothing here waits: the slot's barrier is handed the copies and flips once they land
//! ([`Slot::release_write`](crate::Slot)), or a blocking copy waits on one of its own
//! ([`Memory::load_from`]).

use cubecl::{
    prelude::barrier::{Barrier, copy_async, copy_async_checked},
    prelude::*,
};

use cubecl_std::tensor::layout::{CoordsDyn, Layout, LayoutExpand};

use crate::*;

/// The widths of one copy the engine takes, in bytes: `cp.async` moves 4, 8 or 16.
const COPY_BYTES: [usize; 3] = [4, 8, 16];

/// [`fill_lines`](super::straight::fill_lines) with every line handed to the copy engine: the
/// same lines per unit on the same schedule, numbered in the source's order ([`BufferLayout::row_major_coords`]).
///
/// A schedule of its own rather than an arm of `fill_lines`: the copy reads and writes lines of
/// one width, which `fill_lines`, generic over a source served narrower, cannot hand it.
#[cube]
pub(crate) fn fill_lines_async<I2: Numeric, W: Size>(
    d: &mut [Vector<I2, W>],
    s: &Masked<'_, Vector<I2, W>, CoordsDyn>,
    layout: &BufferLayout,
    total: usize,
    #[comptime] total_c: Option<u64>,
    #[comptime] fill: FillUnits,
    #[comptime] straight: bool,
) {
    // Both read outside `comptime!`: in there, `size` is the host's `size_of` of the generic and
    // `value` is not expanded at all.
    let width = W::value();
    let elem_size = I2::size().comptime();
    let bytes = comptime!(width * elem_size);
    comptime!(assert!(
        COPY_BYTES.contains(&bytes),
        "fill_lines_async: the copy engine moves 4, 8 or 16 bytes at once, and this stage's lines \
         are {bytes}: serve the operand in lines one copy moves"
    ));
    if comptime!(straight) {
        let units = comptime!(fill.count);
        let tasks = comptime!((total_c.unwrap() as usize).div_ceil(units));
        #[unroll]
        for t in 0..tasks {
            let i = FillUnits::worker(fill) + comptime!(t * units);
            if comptime!((t + 1) * units > total_c.unwrap() as usize) {
                if i < total {
                    copy_line_async::<I2, W>(d, s, layout, i);
                }
            } else {
                copy_line_async::<I2, W>(d, s, layout, i);
            }
        }
    } else {
        let stride = FillUnits::workers(fill);
        let mut i = FillUnits::worker(fill);
        while i < total {
            copy_line_async::<I2, W>(d, s, layout, i);
            i += stride;
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

#[cube]
impl Memory {
    /// Fill `self` from `src` and wait for it: [`fill_from`](Memory::fill_from), plus the barrier
    /// an async copy needs before its bytes can be read. What a blocking
    /// [`copy_from`](Tile::copy_from) runs; a pipelined fill leaves the wait to its slot.
    ///
    /// The contract is `fill_from`'s either way: a `sync_cube` must still separate the fill from
    /// its readers.
    pub(crate) fn load_from<T: Numeric>(&mut self, src: &Memory, #[comptime] space: Space) {
        if comptime!(src.access.delivery == Delivery::AsyncPerUnit) {
            // The wait is the cube's: a stage one plane fills on its own would leave the others'
            // arrivals missing.
            comptime!(assert!(
                self.access.fill.scope == ComputeScope::Cube,
                "Tile::copy_from: an async copy into a stage waits on a barrier the whole cube \
                 arrives at, and this stage is filled by one plane"
            ));
            let landed = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
            self.fill_from::<T>(src, space);
            landed.commit_copy_async();
            landed.arrive_and_wait();
        } else {
            self.fill_from::<T>(src, space);
        }
    }
}
