use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{Reduce, ReduceInstruction, ReduceShared, Sum};

pub struct Mean;

#[cube]
impl<In: Numeric> ReduceInstruction<In> for Mean {}

#[cube]
impl<In: Numeric> Reduce<In> for Mean {
    type Accumulator = Line<In>;

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator {
        Line::empty(line_size).fill(In::from_int(0))
    }

    fn null_value() -> In {
        In::from_int(0)
    }

    fn reduce(
        accumulator: &mut Self::Accumulator,
        item: Line<In>,
        _coordinate: Line<u32>,
        #[comptime] use_plane: bool,
    ) {
        if use_plane {
            *accumulator += plane_sum(item);
        } else {
            *accumulator += item;
        }
    }

    fn merge_line<Out: Numeric>(accumulator: Self::Accumulator, shape_axis_reduce: u32) -> Out {
        let mut sum = In::from_int(0);
        #[unroll]
        for k in 0..accumulator.size() {
            sum += accumulator[k];
        }
        Out::cast_from(sum) / Out::cast_from(shape_axis_reduce)
    }

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::Accumulator,
        shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator)
            / Line::empty(accumulator.size()).fill(Out::cast_from(shape_axis_reduce))
    }
}

#[cube]
impl<In: Numeric> ReduceShared<In> for Mean {
    type Accumulator = SharedMemory<Line<In>>;
    type AccumulatorItem = Line<In>;

    fn create_accumulator(
        #[comptime] length: u32,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        Sum::create_accumulator(length, line_size)
    }

    fn init_accumulator(
        accumulator: &mut Self::Accumulator,
        index: u32,
        #[comptime] line_size: u32,
    ) {
        accumulator[index] = Line::empty(line_size).fill(In::from_int(0))
    }

    fn null_value() -> In {
        In::from_int(0)
    }

    fn reduce(
        accumulator: &mut Self::Accumulator,
        destination: u32,
        item: Line<In>,
        _coordinate: Line<u32>,
        #[comptime] use_plane: bool,
    ) {
        if use_plane {
            accumulator[destination] += plane_sum(item);
        } else {
            accumulator[destination] += item;
        }
    }

    fn fuse_accumulator(accumulator: &mut Self::Accumulator, destination: u32, origin: u32) {
        Sum::fuse_accumulator(accumulator, destination, origin)
    }

    fn get_first(accumulator: Self::Accumulator) -> Self::AccumulatorItem {
        Sum::get_first(accumulator)
    }

    fn merge_line<Out: Numeric>(accumulator: Self::AccumulatorItem, shape_axis_reduce: u32) -> Out {
        let mut sum = In::from_int(0);
        #[unroll]
        for k in 0..accumulator.size() {
            sum += accumulator[k];
        }
        Out::cast_from(sum) / Out::cast_from(shape_axis_reduce)
    }

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator)
            / Line::empty(accumulator.size()).fill(Out::cast_from(shape_axis_reduce))
    }
}
