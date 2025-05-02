use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::{instructions::ReduceRequirements, precision::ReducePrecision};

use super::{ReduceCoordinate, ReduceFamily, ReduceInstruction};

// TODO Add to test framework.
/// Return the item with the maximum absolute value.
#[derive(Debug, CubeType, Clone)]
pub struct MaxAbs;

impl ReduceFamily for MaxAbs {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for MaxAbs {
    type AccumulatorItem = Line<P::EA>;
    type SharedAccumulator = SharedMemory<Line<P::EA>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn from_config(_config: Self::Config) -> Self {
        MaxAbs {}
    }

    fn null_input(_this: &Self, #[comptime] line_size: u32) -> Line<P::EI> {
        Line::empty(line_size).fill(P::EI::from_int(0))
    }

    fn null_accumulator(_this: &Self, #[comptime] line_size: u32) -> Self::AccumulatorItem {
        Line::empty(line_size).fill(P::EA::from_int(0))
    }

    fn assign_accumulator(
        _this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        *destination = *source;
    }

    fn reduce(
        _this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Line<P::EI>,
        _coordinate: ReduceCoordinate,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        if use_planes {
            let candidate_item = Line::cast_from(plane_max(Line::abs(item)));
            select_many(
                accumulator.greater_than(candidate_item),
                *accumulator,
                candidate_item,
            )
        } else {
            let item_abs = Line::cast_from(Line::abs(item));
            select_many(accumulator.greater_than(item_abs), *accumulator, item_abs)
        }
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        select_many(lhs.greater_than(rhs), lhs, rhs)
    }

    fn merge_line<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Out {
        let mut max = P::EA::from_int(0);
        #[unroll]
        for k in 0..accumulator.size() {
            let candidate = accumulator[k];
            max = select(candidate > max, candidate, max);
        }
        Out::cast_from(max)
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator)
    }
}
