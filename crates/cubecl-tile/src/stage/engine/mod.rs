//! The engines a stage is filled by, one file each: what the engine states about itself
//! ([`Engine`](crate::launch::Engine)) and the move it makes.
//!
//! A new engine is a file here, its arm of [`Delivery`](crate::Delivery), and its arm in the two
//! places a fill is handed to an engine: [`Delivery::engine`](crate::Delivery) for what it states,
//! [`Memory::move_lines`] for the lines its units move.

pub(crate) mod async_per_unit;
pub(crate) mod sync_per_unit;
pub(crate) mod tma;

pub(crate) use async_per_unit::ASYNC_PER_UNIT;
pub(crate) use sync_per_unit::SYNC_PER_UNIT;
pub(crate) use tma::TMA;

use cubecl::{prelude::barrier::Barrier, prelude::*};

use cubecl_std::tensor::layout::CoordsDyn;

use crate::*;

#[cube]
impl Memory {
    /// Move the stage's `total` whole lines out of `s`, dealt to the units `fill` names
    /// ([`FillUnits::deal`]), by the engine `delivery` names.
    pub(crate) fn move_lines<I2: Numeric, W: Size>(
        d: &mut [Vector<I2, W>],
        s: &Masked<'_, Vector<I2, W>, CoordsDyn>,
        layout: &BufferLayout,
        total: usize,
        #[comptime] delivery: Delivery,
        #[comptime] fill: FillUnits,
    ) {
        match delivery {
            Delivery::SyncPerUnit => {
                Memory::sync_lines::<I2, W, W>(d, s, layout, total, fill, comptime!(None))
            }
            Delivery::AsyncPerUnit => Memory::async_lines::<I2, W>(d, s, layout, total, fill),
            Delivery::Tma => comptime!(panic!(
                "Memory::move_lines: a tensor map's box is moved whole, by one unit"
            )),
        }
    }

    /// Fill `self` from `src` and wait for it: [`fill_from`](Memory::fill_from), plus what the
    /// slot would wait on where the lines land after the fill returns. What a blocking
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
                    self.access.fill.scope() == ComputeScope::Cube,
                    "Tile::copy_from: an async copy into a stage waits on a barrier the whole \
                     cube arrives at, and this stage is filled by one plane"
                ));
                let landed = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
                self.fill_from::<T>(src, space);
                landed.commit_copy_async();
                landed.arrive_and_wait();
            }
            Landing::Bytes => comptime!(panic!(
                "Tile::copy_from: a stage counted in bytes is filled from a tensor map's box \
                 ([`Memory::load_box`])"
            )),
        }
    }
}
