use crate::matmul::matmul_global::ReadView;
use crate::matmul::matmul_stage::SmmConfig;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Stage<ES: Numeric>:
    CubeType + Clone + Copy + IntoRuntime + Send + Sync + 'static
{
    type Underlying: CubeType;
    type Config: SmmConfig;

    fn new(
        layout: MatrixLayout,
        #[comptime] block_info: StageInfo,
        #[comptime] line_size: u32,
    ) -> Self;

    fn fill<EG: Numeric, RV: ReadView<EG, Config = Self::Config>>(
        stage: &mut Self,
        global: &RV,
        config: Self::Config,
    );

    fn get_tile(stage: &Self, x: u32, y: u32) -> (&Slice<'_, Line<ES>>, MatrixLayout);

    fn layout(stage: &Self) -> MatrixLayout;
}
