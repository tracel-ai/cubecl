use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait ReduceInstruction<In: Numeric>: Reduce<In> + ReduceShared<In> {}

#[cube]
pub trait Reduce<In: Numeric>: Send + Sync + 'static {
    type Accumulator: CubeType;

    fn init_accumulator(#[comptime] line_size: u32) -> Self::Accumulator;

    fn null_value() -> In;

    fn reduce(
        accumulator: &mut Self::Accumulator,
        item: Line<In>,
        coordinate: Line<u32>,
        #[comptime] use_plane: bool,
    );

    fn merge_line<Out: Numeric>(accumulator: Self::Accumulator, shape_axis_reduce: u32) -> Out;

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::Accumulator,
        shape_axis_reduce: u32,
    ) -> Line<Out>;
}

#[cube]
pub trait ReduceShared<In: Numeric>: Send + Sync + 'static {
    type Accumulator: CubeType;
    type AccumulatorItem: CubeType;

    fn create_accumulator(#[comptime] length: u32, #[comptime] line_size: u32)
        -> Self::Accumulator;

    fn init_accumulator(
        accumulator: &mut Self::Accumulator,
        index: u32,
        #[comptime] line_size: u32,
    );

    fn null_value() -> In;

    fn reduce(
        accumulator: &mut Self::Accumulator,
        destination: u32,
        item: Line<In>,
        coordinate: Line<u32>,
        #[comptime] use_plane: bool,
    );

    fn fuse_accumulator(accumulator: &mut Self::Accumulator, destination: u32, origin: u32);

    fn get_first(accumulator: Self::Accumulator) -> Self::AccumulatorItem;

    fn merge_line<Out: Numeric>(accumulator: Self::AccumulatorItem, shape_axis_reduce: u32) -> Out;

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::AccumulatorItem,
        shape_axis_reduce: u32,
    ) -> Line<Out>;
}

#[cube]
pub trait SharedAccumulator<In: Numeric>: Send + Sync + 'static {
    type Item;

    fn allocate(#[comptime] length: u32, #[comptime] line_size: u32) -> Self;

    fn init(&mut self, index: u32, #[comptime] line_size: u32);

    fn read(&self, index: u32) -> Self::Item;

    fn write(&mut self, index: u32, item: Self::Item);
}
