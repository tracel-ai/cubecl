use crate::matmul::matmul_stage::StageReader;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::GmmConfig;

#[cube]
pub trait Loader<EG: Numeric, ES: Numeric, G: GmmConfig>: CubeType + 'static + Send + Sync {
    type StageReader: StageReader<ES, G::SmmConfig>;

    fn fill_stage(loader: &mut Self, #[comptime] config: G) -> Self::StageReader;
    fn advance_view(loader: &mut Self, k_offset: u32);
}
