//! The element positions a target states, as a kernel computes them.

use crate::prelude::*;
use cubecl_ir::{CoordinatePart, ElementIndex, ElementPositions, Scope};

/// The row and the column of the `nth` element `lane` holds.
pub(crate) fn expand_position(
    scope: &Scope,
    positions: &ElementPositions,
    lane: NativeExpand<u32>,
    nth: NativeExpand<u32>,
) -> (NativeExpand<u32>, NativeExpand<u32>) {
    let coordinate = |parts: &[CoordinatePart]| {
        parts
            .iter()
            .map(|part| expand_part(scope, part, lane.clone(), nth.clone()))
            .reduce(|sum, part| sum.__expand_add_method(scope, part))
            .expect("a coordinate should have a part")
    };
    (coordinate(&positions.row), coordinate(&positions.col))
}

fn expand_part(
    scope: &Scope,
    part: &CoordinatePart,
    lane: NativeExpand<u32>,
    nth: NativeExpand<u32>,
) -> NativeExpand<u32> {
    let literal = |value: usize| NativeExpand::<u32>::from_lit(scope, value as u32);
    let index = match part.index {
        ElementIndex::Lane => lane,
        ElementIndex::Nth => nth,
    };
    let step = index.__expand_div_method(scope, literal(part.run));
    let step = match part.count {
        Some(count) => step.__expand_rem_method(scope, literal(count)),
        None => step,
    };
    step.__expand_mul_method(scope, literal(part.stride))
}
