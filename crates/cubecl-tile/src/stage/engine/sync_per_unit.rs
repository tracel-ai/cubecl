//! [`Delivery::SyncPerUnit`]: every unit loads its own lines and stores them into the stage, so
//! they have landed when the fill returns, and a line can be decoded, padded, scanned or summed on
//! its way. The units walk the stage in its own order: their reads go through the cache, which
//! gathers a warp's lines from several rows back into whole transactions, and their writes stay
//! free of bank conflicts.

use cubecl::prelude::*;

use cubecl_std::tensor::layout::CoordsDyn;

use crate::stage::transport::base::store_line;
use crate::stage::transport::padded::read_stage_line;
use crate::*;

/// What the units' own loads and stores state.
pub const SYNC_PER_UNIT: Engine = Engine {
    issuer: Issuer::Unit,
    route: Route::Registers,
    present: |_| true,
    line_bytes: None,
    box_edge: None,
    stage_alignment: 0,
};

/// Write the stage's `total` lines into `d`, dealt to the units `fill` names ([`Walk::copy`]):
/// each the source's line at the coordinates the stage holds it at, assembled out of narrower
/// cells where the stage is padded ([`Padding`]).
#[cube]
pub fn sync_lines<I2: Numeric, WP2: Size, SW: Size>(
    d: &mut [Vector<I2, WP2>],
    s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
    layout: &BufferLayout,
    total: usize,
    #[comptime] fill: Level,
    #[comptime] padding: Option<Padding>,
) {
    for line in Walk::copy(&fill, total) {
        let i = line.copied_line();
        let value = read_stage_line::<I2, WP2, SW>(s, &layout.line_coords(i), comptime!(padding));
        store_line::<I2, WP2>(d, layout, i, value);
    }
}
