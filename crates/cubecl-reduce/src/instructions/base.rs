use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait ReduceInstruction<In: Numeric>: Reduce<In> {}

#[cube]
pub trait Reduce<In: Numeric>: Send + Sync + 'static {
    type Accumulator: CubeType;

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator;

    fn reduce(
        accumulator: &mut Self::Accumulator,
        item: Line<In>,
        coordinate: Line<u32>,
        #[comptime] use_plane: bool,
    );

    fn merge_line<Out: Numeric>(accumulator: Self::Accumulator, shape_axis_reduce: u32) -> Out;

    fn to_output_parallel<Out: Numeric>(
        accumulator: Self::Accumulator,
        shape_axis_reduce: u32,
    ) -> Line<Out>;
}

// #[cube]
// pub trait ReduceShared<In: Numeric> {
//     type Accumulator: CubeType;
//     type AccumulatorItem: CubeType;

//     fn create_accumulator(#[comptime] length: u32, #[comptime] line_size: u32)
//         -> Self::Accumulator;

//     fn init_accumulator(
//         accumulator: &mut Self::Accumulator,
//         index: u32,
//         #[comptime] line_size: u32,
//     );

//     fn reduce(
//         accumulator: &mut Self::Accumulator,
//         destination: u32,
//         item: Line<In>,
//         coordinate: Line<u32>,
//         #[comptime] use_plane: bool,
//     );

//     fn fuse_accumulator(accumulator: &mut Self::Accumulator, destination: u32, origin: u32);

//     fn get_first(accumulator: Self::Accumulator) -> Self::AccumulatorItem;

//     fn merge_line<Out: Numeric>(accumulator: Self::AccumulatorItem, shape_axis_reduce: u32) -> Out;

//     fn to_output_parallel<Out: Numeric>(
//         accumulator: Self::AccumulatorItem,
//         shape_axis_reduce: u32,
//     ) -> Line<Out>;
// }
