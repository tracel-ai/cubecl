//! Handing a fill to the engine its delivery names ([`move_lines`]), and a blocking fill that
//! waits for it ([`load_from`]).

use cubecl::{prelude::barrier::Barrier, prelude::*};

use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

/// Move the stage's `total` whole lines out of `s` into `d`, dealt to the units `fill` names
/// ([`Walk::copy`]), by the engine `delivery` names.
#[cube]
pub fn move_lines<I2: Numeric, W: Size>(
    d: &mut [Vector<I2, W>],
    s: &Masked<'_, Vector<I2, W>, CoordsDyn>,
    layout: &BufferLayout,
    total: usize,
    #[comptime] delivery: Delivery,
    #[comptime] fill: Level,
) {
    match delivery {
        Delivery::SyncPerUnit => sync_lines::<I2, W, W>(d, s, layout, total, fill, comptime!(None)),
        Delivery::AsyncPerUnit => async_lines::<I2, W>(d, s, layout, total, fill),
        Delivery::Tma => comptime!(panic!(
            "move_lines: a tensor map's box should be moved whole, by one unit"
        )),
    }
}

/// Fill `dest` from `src` and wait for it: [`fill_from`], plus what the slot would wait
/// on where the lines land after the fill returns. What a blocking
/// [`copy_from`](Tile::copy_from) runs; a pipelined fill leaves the wait to its slot.
///
/// The contract is `fill_from`'s either way: its barrier must still separate the fill from its
/// readers.
#[cube]
pub fn load_from<T: Numeric>(dest: &mut Memory, src: &Memory, #[comptime] space: &Space) {
    match comptime!(src.access.delivery.landing()) {
        Landing::OnReturn => fill_from::<T>(dest, src, space),
        Landing::Arrivals => {
            // The wait is the cube's: a stage one plane fills on its own would leave the
            // others' arrivals missing.
            comptime!(assert!(
                dest.access.copy.copied_within() == ComputeScope::Cube,
                "Tile::copy_from: an async copy into a stage should wait on a barrier the whole \
                 cube arrives at, and this stage is filled by one plane"
            ));
            let landed = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
            fill_from::<T>(dest, src, space);
            landed.commit_copy_async();
            landed.arrive_and_wait();
        }
        Landing::Bytes => comptime!(panic!(
            "Tile::copy_from: a stage counted in bytes should be filled from a tensor map's box, \
             by load_box"
        )),
    }
}
