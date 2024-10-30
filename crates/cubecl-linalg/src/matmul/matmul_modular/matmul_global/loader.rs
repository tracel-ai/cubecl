use crate::matmul::matmul_modular::matmul_stage::StageReader;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::GmmConfig;

#[cube]
/// Input to the global matmul, responsible of filling the stage and providing a reader for it.
/// Advances along the k-dimension to fill the stage with further data.
pub trait Loader<EG: Numeric, ES: Numeric, G: GmmConfig>: CubeType + 'static + Send + Sync {
    /// The stage reader which matches the input of the underlying stage matmul.
    type StageReader: StageReader<ES, G::SmmConfig>;

    /// Fills the stage at the current k offset and returns a reader for it.
    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader;

    /// Move the k offset by k_offset
    fn advance_view(this: &mut Self, k_offset: u32);
}
