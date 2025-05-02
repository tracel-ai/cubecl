use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::precision::ReducePrecision;

use super::{ReduceCoordinate, ReduceFamily, ReduceInstruction, ReduceRequirements};

#[derive(Debug, CubeType, Clone)]
pub struct Sum {}

impl ReduceFamily for Sum {
    type Instruction<P: ReducePrecision> = Self;
    type Config = ();
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for Sum {
    type AccumulatorItem = Line<P::EA>;
    type SharedAccumulator = SharedMemory<Line<P::EA>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn from_config(_config: Self::Config) -> Self {
        Sum {}
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
            *accumulator + plane_sum(Line::cast_from(item))
        } else {
            *accumulator + Line::cast_from(item)
        }
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        lhs + rhs
    }

    fn merge_line<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Out {
        let mut sum = P::EA::from_int(0);
        #[unroll]
        for k in 0..accumulator.size() {
            sum += accumulator[k];
        }
        Out::cast_from(sum)
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator)
    }
}
