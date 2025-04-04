use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{Mean, Prod, Reduce, ReduceCoordinate, ReduceInstruction, Sum};

#[derive(Debug, CubeType, Clone)]
pub enum Dynamic {
    Sum(Sum),
    Prod(Prod),
    Mean(Mean),
}

#[derive_cube_comptime]
pub enum DynamicConfig {
    Sum,
    Prod,
    Mean,
}

impl Reduce for Dynamic {
    type Instruction<In: Numeric> = Self;
    type Config = DynamicConfig;
}

#[cube]
impl<In: Numeric> ReduceInstruction<In> for Dynamic {
    const REQUIRES_COORDINATE: bool = false;

    type AccumulatorItem = Line<In>;
    type SharedAccumulator = SharedMemory<Line<In>>;
    type Config = DynamicConfig;

    fn from_config(#[comptime] config: Self::Config) -> Self {
        match config {
            DynamicConfig::Sum => Dynamic::new_Sum(Sum {}),
            DynamicConfig::Prod => Dynamic::new_Prod(Prod {}),
            DynamicConfig::Mean => Dynamic::new_Mean(Mean { sum: Sum {} }),
        }
    }

    fn null_input(this: &Self, #[comptime] line_size: u32) -> Line<In> {
        match this {
            Dynamic::Sum(sum) => Sum::null_input(sum, line_size),
            Dynamic::Prod(prod) => Prod::null_input(prod, line_size),
            Dynamic::Mean(mean) => Mean::null_input(mean, line_size),
        }
    }

    fn null_accumulator(this: &Self, #[comptime] line_size: u32) -> Self::AccumulatorItem {
        match this {
            Dynamic::Sum(sum) => Sum::null_accumulator(sum, line_size),
            Dynamic::Prod(prod) => Prod::null_accumulator(prod, line_size),
            Dynamic::Mean(mean) => Mean::null_accumulator(mean, line_size),
        }
    }

    fn assign_accumulator(
        this: &Self,
        destination: &mut Self::AccumulatorItem,
        source: &Self::AccumulatorItem,
    ) {
        match this {
            Dynamic::Sum(sum) => Sum::assign_accumulator(sum, destination, source),
            Dynamic::Prod(prod) => Prod::assign_accumulator(prod, destination, source),
            Dynamic::Mean(mean) => Mean::assign_accumulator(mean, destination, source),
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &Self::AccumulatorItem,
        item: Line<In>,
        coordinate: ReduceCoordinate,
        #[comptime] use_planes: bool,
    ) -> Self::AccumulatorItem {
        match this {
            Dynamic::Sum(sum) => Sum::reduce(sum, accumulator, item, coordinate, use_planes),
            Dynamic::Prod(prod) => Prod::reduce(prod, accumulator, item, coordinate, use_planes),
            Dynamic::Mean(mean) => Mean::reduce(mean, accumulator, item, coordinate, use_planes),
        }
    }

    fn fuse_accumulators(
        this: &Self,
        lhs: Self::AccumulatorItem,
        rhs: Self::AccumulatorItem,
    ) -> Self::AccumulatorItem {
        match this {
            Dynamic::Sum(sum) => Sum::fuse_accumulators(sum, lhs, rhs),
            Dynamic::Prod(prod) => Prod::fuse_accumulators(prod, lhs, rhs),
            Dynamic::Mean(mean) => Mean::fuse_accumulators(mean, lhs, rhs),
        }
    }

    // TODO Remove shape_axis_reduce when fusion-on-write is well supported for reduce instructions.
    //      Then, an instruction like Dynamic can be implemented by fusing a Sum reduction and a element-wise division.
    fn merge_line<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Out {
        match this {
            Dynamic::Sum(sum) => Sum::merge_line::<Out>(sum, accumulator, shape_axis_reduce),
            Dynamic::Prod(prod) => Prod::merge_line::<Out>(prod, accumulator, shape_axis_reduce),
            Dynamic::Mean(mean) => Mean::merge_line::<Out>(mean, accumulator, shape_axis_reduce),
        }
    }

    fn to_output_perpendicular<Out: Numeric>(
        this: &Self,
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Line<Out> {
        match this {
            Dynamic::Sum(sum) => {
                Sum::to_output_perpendicular::<Out>(sum, accumulator, shape_axis_reduce)
            }
            Dynamic::Prod(prod) => {
                Prod::to_output_perpendicular::<Out>(prod, accumulator, shape_axis_reduce)
            }
            Dynamic::Mean(mean) => {
                Mean::to_output_perpendicular::<Out>(mean, accumulator, shape_axis_reduce)
            }
        }
    }
}
