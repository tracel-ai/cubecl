use crate::matmul::stage_info::StageInfo;
use crate::matmul::data::{GlobalView, Tile};
use crate::matmul::matrix_layout::MatrixLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Stage<E: Numeric>: CubeType + Clone + Copy + IntoRuntime + Send + Sync + 'static {
    type GlobalView: GlobalView<E>;
    type Underlying: CubeType;

    fn new(
        layout: MatrixLayout,
        #[comptime] block_info: StageInfo,
        #[comptime] line_size: u32,
    ) -> Self;

    fn fill(block: &mut Self, gmem: &Self::GlobalView);

    fn get_tile(block: &Self, x: u32, y: u32) -> Tile<'_, E>;

    fn layout(block: &Self) -> MatrixLayout;
}
