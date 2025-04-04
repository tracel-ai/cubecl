use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{Reduce, ReduceCoordinate, ReduceInstruction, Sum};

#[derive(Debug, CubeType, Clone)]
pub struct Mean {
    pub(crate) sum: Sum,
}

impl Reduce for Mean {
    type Instruction<In: Numeric> = Self;
    type Config = ();
}

#[cube]
impl<In: Numeric> ReduceInstruction<In> for Mean {
    const REQUIRES_COORDINATE: bool = false;

    type AccumulatorItem = Line<In>;
    type SharedAccumulator = SharedMemory<Line<In>>;
    type Config = ();
    fn from_config(_config: Self::Config) -> Self {
        Mean { sum: Sum {} }
    }

    fn null_input(this: &Self, #[comptime] line_size: u32) -> Line<In> {
        Sum::null_input(&this.sum, line_size)
    }

    fn null_accumulator(this: &Self, #[comptime] line_size: u32) -> Self::AccumulatorItem {
        Sum::null_accumulator(&this.sum, line_size)
    }

    fn assign_accumulator(
        this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        Sum::assign_accumulator(&this.sum, destination, source);
    }

    fn reduce(
        this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Line<In>,
        _coordinate: ReduceCoordinate,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        Sum::reduce(&this.sum, accumulator, item, _coordinate, use_planes)
    }

    fn fuse_accumulators(
        this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        Sum::fuse_accumulators(&this.sum, lhs, rhs)
    }

    // TODO Remove shape_axis_reduce when fusion-on-write is well supported for reduce instructions.
    //      Then, an instruction like Mean can be implemented by fusing a Sum reduction and a element-wise division.
    fn merge_line<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Out {
        Sum::merge_line::<Out>(&this.sum, accumulator, shape_axis_reduce)
            / Out::cast_from(shape_axis_reduce)
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Line<Out> {
        let sum = Sum::to_output_perpendicular::<Out>(&this.sum, accumulator, shape_axis_reduce);
        sum / Line::empty(accumulator.size()).fill(Out::cast_from(shape_axis_reduce))
    }
}
