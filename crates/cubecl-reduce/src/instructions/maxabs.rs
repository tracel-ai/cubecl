use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{Reduce, ReduceCoordinate, ReduceInstruction};

// TODO Add to test framework.
/// Return the item with the maximum absolute value.
#[derive(Debug)]
pub struct MaxAbs;

impl Reduce for MaxAbs {
    type Instruction<In: Numeric> = Self;
}

#[cube]
impl<In: Numeric> ReduceInstruction<In> for MaxAbs {
    const REQUIRES_COORDINATE: bool = false;

    type AccumulatorItem = Line<In>;
    type SharedAccumulator = SharedMemory<Line<In>>;

    fn null_input(#[comptime] line_size: u32) -> Line<In> {
        Line::empty(line_size).fill(In::min_value())
    }

    fn null_accumulator(#[comptime] line_size: u32) -> Self::AccumulatorItem {
        Self::null_input(line_size)
    }

    fn assign_accumulator(destination: &mut Self::AccumulatorItem, source: &Self::AccumulatorItem) {
        *destination = *source;
    }

    fn reduce(
        accumulator: &Self::AccumulatorItem,
        item: Line<In>,
        _coordinate: ReduceCoordinate,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        if use_planes {
            let candidate_item = plane_max(Line::abs(item));
            select_many(
                accumulator.greater_than(candidate_item),
                *accumulator,
                candidate_item,
            )
        } else {
            let item_abs = Line::abs(item);
            select_many(accumulator.greater_than(item_abs), *accumulator, item_abs)
        }
    }

    fn fuse_accumulators(
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        lhs + rhs
    }

    fn merge_line<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Out {
        let mut sum = In::from_int(0);
        #[unroll]
        for k in 0..accumulator.size() {
            sum += accumulator[k];
        }
        Out::cast_from(sum)
    }

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator)
    }
}
