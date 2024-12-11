use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{Reduce, Sum};

pub struct Mean;

#[cube]
impl<In: Numeric> Reduce<In> for Mean {
    type AccumulatorItem = Line<In>;
    type SharedAccumulator = SharedMemory<Line<In>>;

    fn null_input(#[comptime] line_size: u32) -> Line<In> {
        Sum::null_input(line_size)
    }

    fn null_accumulator(#[comptime] line_size: u32) -> Self::AccumulatorItem {
        Sum::null_accumulator(line_size)
    }

    fn update_accumulator(destination: &mut Self::AccumulatorItem, source: &Self::AccumulatorItem) {
        Sum::update_accumulator(destination, source);
    }

    fn reduce(
        accumulator: &Self::AccumulatorItem,
        item: Line<In>,
        _coordinate: Line<u32>,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        Sum::reduce(accumulator, item, _coordinate, use_planes)
    }

    fn fuse_accumulators(
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        Sum::fuse_accumulators(lhs, rhs)
    }

    fn merge_line<Out: Numeric>(accumulator: Self::AccumulatorItem, shape_axis_reduce: u32) -> Out {
        Sum::merge_line::<Out>(accumulator, shape_axis_reduce) / Out::cast_from(shape_axis_reduce)
    }

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Line<Out> {
        let sum = Sum::to_output_perpendicular::<Out>(accumulator, shape_axis_reduce);
        sum / Line::empty(accumulator.size()).fill(Out::cast_from(shape_axis_reduce))
    }
}
