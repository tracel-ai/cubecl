use cubecl_core as cubecl;
use cubecl_core::prelude::*;

// Using plane operations, return the lowest coordinate for each line element
// for which the item equal the target.
#[cube]
pub(crate) fn lowest_coordinate_matching<E: CubePrimitive>(
    target: Line<E>,
    item: Line<E>,
    coordinate: Line<u32>,
) -> Line<u32> {
    let line_size = item.size();
    let is_candidate = item.equal(target);
    sync_units();
    let candidate_coordinate = select_many(
        is_candidate,
        coordinate,
        Line::empty(line_size).fill(u32::MAX),
    );
    sync_units();
    plane_min(candidate_coordinate)
}
