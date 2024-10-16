use crate::matmul::matmul_global::{GlobalView, GmmConfig};
use crate::matmul::matmul_stage::StageReader;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Loader<EG: Numeric, ES: Numeric>: CubeType + 'static + Send + Sync {
    type GlobalView: GlobalView<EG>;
    type StageReader: StageReader<ES>;
    type Config: GmmConfig;

    fn new(
        gmem: <Self::GlobalView as GlobalView<EG>>::Global,
        #[comptime] layout: MatrixLayout,
        #[comptime] stage_info: StageInfo,
    ) -> Self;

    fn fill_stage(loader: &mut Self, config: Self::Config) -> Self::StageReader;

    fn init_view(loader: &mut Self, cube_offset: u32, k_start: u32);

    fn advance_view(loader: &mut Self, k_offset: u32);
}
