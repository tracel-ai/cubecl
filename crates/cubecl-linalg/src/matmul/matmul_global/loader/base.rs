use crate::matmul::matmul_global::{GmmConfig, ReadView};
use crate::matmul::matmul_stage::StageReader;
use crate::matmul::matrix_layout::MatrixLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Loader<EG: Numeric, ES: Numeric>: CubeType + 'static + Send + Sync {
    type ReadView: ReadView<EG>;
    type StageReader: StageReader<ES>;
    type Config: GmmConfig;

    fn new(
        gmem: <Self::ReadView as ReadView<EG>>::Global,
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self;

    fn fill_stage(loader: &mut Self, config: Self::Config) -> Self::StageReader;

    fn init_view(loader: &mut Self, cube_offset: u32, k_start: u32);

    fn advance_view(loader: &mut Self, k_offset: u32);
}
