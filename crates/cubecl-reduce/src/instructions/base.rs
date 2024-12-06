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

    fn to_output_perpendicular<Out: Numeric>(
        accumulator: Self::Accumulator,
        shape_axis_reduce: u32,
    ) -> Line<Out>;
}
