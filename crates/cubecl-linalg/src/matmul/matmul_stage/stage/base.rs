use crate::matmul::matmul_global::GlobalView;
use crate::matmul::matmul_tile::RefTile;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Stage<E: Numeric>: CubeType + Clone + Copy + IntoRuntime + Send + Sync + 'static {
    type Underlying: CubeType;

    fn new(
        layout: MatrixLayout,
        #[comptime] block_info: StageInfo,
        #[comptime] line_size: u32,
    ) -> Self;

    fn fill<EG: Numeric, G: GlobalView<EG>>(stage: &mut Self, global: &G);

    fn get_tile(stage: &Self, x: u32, y: u32) -> RefTile<'_, E>;

    fn layout(stage: &Self) -> MatrixLayout;
}
