use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{
    MatmulPrecision,
    global::GlobalConfig,
    stage::{StageReader, TilingLayout},
};

#[cube]
/// Input to the global matmul, responsible of filling the stage and providing a reader for it.
/// Advances along the k-dimension to fill the stage with further data.
pub trait Loader<MP: MatmulPrecision, G: GlobalConfig>: CubeType + 'static + Send + Sync {
    type TilingLayout: TilingLayout;

    /// Returns a reader for the stage at the current k offset
    fn reader(this: &Self) -> StageReader<MP::ES, Self::TilingLayout>;

    /// Move the k offset by k_offset
    fn advance_view(this: &mut Self, k_offset: u32);
}
