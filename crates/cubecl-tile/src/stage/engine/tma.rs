//! [`Delivery::Tma`]: one elected unit hands the stage's whole box to the TMA engine through a
//! tensor map (`cp.async.bulk.tensor`). The box lands after the fill returns, counted in bytes on
//! a barrier the issuer announced them to, through the async proxy. The tensor map is the source's
//! binding ([`TmaData`]), which knows how a box lands; who issues it, and how a copy waits for it,
//! is here.

use cubecl::{ir::DeviceProperties, ir::features::Tma, prelude::barrier::Barrier, prelude::*};

use crate::*;

/// What the TMA engine states. A box edge is at most CUDA's cap on a descriptor's box dimension,
/// and a stage starts on the span a 128-byte swizzle repeats over, which the engine keys off the
/// address, so a stage's first row takes the first key as the fragment reading it does.
pub const TMA: Engine = Engine {
    issuer: Issuer::Cube,
    route: Route::MemoryToMemory,
    present: |device: &DeviceProperties| device.features.tma.contains(Tma::Base),
    line_bytes: None,
    box_edge: Some(256),
    stage_alignment: 1024,
};

/// Issue `src`'s box into the stage `dest` on the slot's `full` barrier: the `elected` unit alone
/// announces the bytes and issues, since the transaction count is the issuer's alone.
#[cube]
pub fn issue_box<T: Numeric>(
    dest: &mut Memory,
    src: &TmaData,
    full: &Shared<Barrier>,
    elected: u32,
) {
    if UNIT_POS == elected {
        full.expect_tx(dest.size_bytes::<T>());
        src.stage_into::<T>(dest, full);
    }
}

/// Bulk-copy `src`'s box into the stage `dest` and wait for it, on a barrier of its own: what a
/// blocking [`copy_from`](Tile::copy_from) runs.
#[cube]
pub fn load_box<T: Numeric>(dest: &mut Memory, src: &TmaData) {
    comptime!(assert!(
        dest.address == AddressSpace::Shared,
        "load_box: a tensor map should bulk-copy into shared memory only"
    ));
    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let expected = select(UNIT_POS == 0, dest.size_bytes::<T>(), 0);
    if UNIT_POS == 0 {
        src.stage_into::<T>(dest, &barrier);
    }
    let token = barrier.arrive_and_expect_tx(1, expected);
    barrier.wait(token);
}
