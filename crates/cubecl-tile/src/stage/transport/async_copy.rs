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
//! ([`Meeting::release_write`](crate::Meeting)), or a blocking copy waits on one of its own
//! ([`Memory::load_from`]).

use cubecl::{
    prelude::barrier::{Barrier, copy_async, copy_async_checked},
    prelude::*,
};

use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

/// A straight fill's line handed to the copy engine: the source line at the coordinates the line
/// holds in the delivery's order ([`LineOrder::Source`]), whole, or zeros where the source masks
/// it off. It has not landed when the move returns.
///
/// A line of its own rather than a [`StraightLines`](super::straight) arm: the copy reads and
/// writes lines of one width, which a straight line, served narrower out of a padded stage,
/// cannot hand it.
#[derive(CubeType)]
pub(crate) struct AsyncLines<'a, I2: Numeric, W: Size> {
    d: &'a mut [Vector<I2, W>],
    s: &'a Masked<'a, Vector<I2, W>, CoordsDyn>,
    layout: &'a BufferLayout,
    #[cube(comptime)]
    order: LineOrder,
}

#[cube]
impl<'a, I2: Numeric, W: Size> AsyncLines<'a, I2, W> {
    pub(crate) fn new(
        d: &'a mut [Vector<I2, W>],
        s: &'a Masked<'a, Vector<I2, W>, CoordsDyn>,
        layout: &'a BufferLayout,
        #[comptime] order: LineOrder,
    ) -> Self {
        AsyncLines::<'a, I2, W> {
            d,
            s,
            layout,
            order,
        }
    }
}

#[cube]
impl<'a, I2: Numeric, W: Size> LineMove for AsyncLines<'a, I2, W> {
    fn move_line(&mut self, line: usize, #[comptime] _turn: Option<usize>) {
        // Both read outside `comptime!`: in there, `size` is the host's `size_of` of the generic
        // and `value` is not expanded at all.
        let width = W::value();
        let elem_size = I2::size().comptime();
        let bytes = comptime!(width * elem_size);
        comptime!(assert!(
            COPY_ENGINE_LINE_BYTES.contains(&bytes),
            "AsyncLines: the copy engine moves 4, 8 or 16 bytes at once, and this stage's lines \
             are {bytes}: serve the operand in lines one copy moves"
        ));
        let (coords, offset) = self.layout.line_in(line, comptime!(self.order));
        let run = self.s.item_run(coords);
        let dst = &mut self.d[offset..offset + 1];
        if comptime!(self.s.check) {
            copy_async_checked(run, dst, comptime!(width as u32));
        } else {
            copy_async(run, dst, comptime!(width as u32));
        }
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
    pub(crate) fn load_from<T: Numeric>(&mut self, src: &Memory, #[comptime] space: &Space) {
        match comptime!(src.access.delivery.landing()) {
            Landing::OnReturn => self.fill_from::<T>(src, space),
            Landing::Arrivals => {
                // The wait is the cube's: a stage one plane fills on its own would leave the
                // others' arrivals missing.
                comptime!(assert!(
                    self.access.fill.scope == ComputeScope::Cube,
                    "Tile::copy_from: an async copy into a stage waits on a barrier the whole \
                     cube arrives at, and this stage is filled by one plane"
                ));
                let landed = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
                self.fill_from::<T>(src, space);
                landed.commit_copy_async();
                landed.arrive_and_wait();
            }
            Landing::Bytes => comptime!(panic!(
                "Tile::copy_from: a stage counted in bytes is filled from a tensor map's tile, \
                 which the stage's slot waits on"
            )),
        }
    }
}
