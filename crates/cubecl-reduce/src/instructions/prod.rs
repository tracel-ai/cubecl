use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{Reduce, ReduceInstruction};

#[derive(Debug)]
pub struct Prod;

impl Reduce for Prod {
    type Instruction<In: Numeric> = Self;
}

#[cube]
impl<In: Numeric> ReduceInstruction<In> for Prod {
    type AccumulatorItem = Line<In>;
    type SharedAccumulator = SharedMemory<Line<In>>;

    fn null_input(#[comptime] line_size: u32) -> Line<In> {
        Line::empty(line_size).fill(In::from_int(1))
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
        _coordinate: Line<u32>,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        if use_planes {
            *accumulator * plane_prod(item)
        } else {
            *accumulator * item
        }
    }

    fn fuse_accumulators(
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        lhs * rhs
    }

    fn merge_line<Out: Numeric>(
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
        accumulator: Self::AccumulatorItem,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator)
    }
}
