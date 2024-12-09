use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{Reduce, ReduceInstruction, Sum};

pub struct Mean;

#[cube]
impl<In: Numeric> ReduceInstruction<In> for Mean {}

#[cube]
impl<In: Numeric> Reduce<In> for Mean {
    type Accumulator = Line<In>;

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator {
        Sum::init_accumulator(line_size)
    }

    fn null_value() -> In {
        In::from_int(0)
    }

    fn reduce(
        accumulator: &mut Self::Accumulator,
        item: Line<In>,
        coordinate: Line<u32>,
        #[comptime] use_plane: bool,
    ) {
        Sum::reduce(accumulator, item, coordinate, use_plane)
    }

    fn merge_line<Out: Numeric>(accumulator: Self::Accumulator, shape_axis_reduce: u32) -> Out {
        let sum = Sum::merge_line::<Out>(accumulator, shape_axis_reduce);
        sum / Out::cast_from(shape_axis_reduce)
    }

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::Accumulator,
        shape_axis_reduce: u32,
    ) -> Line<Out> {
        Line::cast_from(accumulator)
            / Line::empty(accumulator.size()).fill(Out::cast_from(shape_axis_reduce))
    }
}
