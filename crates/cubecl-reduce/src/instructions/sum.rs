use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{Reduce, ReduceInstruction, ReduceShared};

pub struct Sum;

#[cube]
impl<In: Numeric> ReduceInstruction<In> for Sum {}

#[cube]
impl<In: Numeric> Reduce<In> for Sum {
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

    fn merge_line<Out: Numeric>(accumulator: Self::Accumulator, _shape_axis_reduce: u32) -> Out {
        let mut sum = In::from_int(0);
        #[unroll]
        for k in 0..accumulator.size() {
            sum += accumulator[k];
        }
        Out::cast_from(sum)
    }

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::Accumulator,
        _shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator)
    }
}

#[cube]
impl<In: Numeric> ReduceShared<In> for Sum {
    type Accumulator = SharedMemory<Line<In>>;
    type AccumulatorItem = Line<In>;

    fn create_accumulator(
        #[comptime] length: u32,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator {
        SharedMemory::new_lined(length, line_size)
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
        let item = accumulator[origin];
        accumulator[destination] += item;
    }

    fn get_first(accumulator: Self::Accumulator) -> Self::AccumulatorItem {
        accumulator[0]
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
