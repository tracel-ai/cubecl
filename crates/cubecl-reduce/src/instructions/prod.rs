use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::instructions::ReduceRequirements;

use super::{ReduceCoordinate, ReduceFamily, ReduceInstruction};

#[derive(Debug, CubeType, Clone)]
pub struct Prod {}

impl ReduceFamily for Prod {
    type Instruction<In: Numeric> = Self;
    type Config = ();
}

#[cube]
impl<In: Numeric> ReduceInstruction<In> for Prod {
    type AccumulatorItem = Line<In>;
    type SharedAccumulator = SharedMemory<Line<In>>;
    type Config = ();

    fn requirements(_this: &Self) -> ReduceRequirements {
        ReduceRequirements { coordinates: false }
    }

    fn from_config(_config: Self::Config) -> Self {
        Prod {}
    }
    fn null_input(_this: &Self, #[comptime] line_size: u32) -> Line<In> {
        Line::empty(line_size).fill(In::from_int(1))
    }

    fn null_accumulator(this: &Self, #[comptime] line_size: u32) -> Self::AccumulatorItem {
        Self::null_input(this, line_size)
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
        item: Line<In>,
        _coordinate: ReduceCoordinate,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        if use_planes {
            *accumulator * plane_prod(item)
        } else {
            *accumulator * item
        }
    }

    fn fuse_accumulators(
        _this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        lhs * rhs
    }

    fn merge_line<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Out {
        let mut prod = In::from_int(1);
        #[unroll]
        for k in 0..accumulator.size() {
            prod *= accumulator[k];
        }
        Out::cast_from(prod)
    }

    fn to_output_perpendicular<Out: Numeric>(
        _this: &Self,
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator)
    }
}
