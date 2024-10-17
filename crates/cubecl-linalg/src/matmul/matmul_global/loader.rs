use crate::matmul::matmul_global::ReadView;
use crate::matmul::matmul_stage::{SmmConfig, StageReader};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait Loader<EG: Numeric, ES: Numeric>: CubeType + 'static + Send + Sync {
    type ReadView: ReadView<EG>;
    type StageReader: StageReader<ES>;
    type Config: SmmConfig;

    // new in trait: maybe bad
    fn new(
        gmem: <Self::ReadView as ReadView<EG>>::Global,
        #[comptime] config: Self::Config,
    ) -> Self;

    fn fill_stage(loader: &mut Self, config: Self::Config) -> Self::StageReader;

    fn init_view(loader: &mut Self, cube_offset: u32, k_start: u32);

    fn advance_view(loader: &mut Self, k_offset: u32);
}
