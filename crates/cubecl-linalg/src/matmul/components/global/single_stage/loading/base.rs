use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{
    MatmulPrecision,
    global::{CopyMechanism, GlobalConfig},
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

#[cube]
pub trait SyncLoader<MP: MatmulPrecision, G: GlobalConfig>: Loader<MP, G> {
    /// Fills the stage at the current k offset.
    fn fill_stage(this: &mut Self, #[comptime] config: G);
}

#[cube]
pub trait AsyncLoader<MP: MatmulPrecision, G: GlobalConfig>: Loader<MP, G> {
    /// Fills the stage at the current k offset.
    fn fill_stage<CM: CopyMechanism<MP::ES>>(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] config: G,
    );

    /// Fills the stage with zeros
    fn clear_stage(this: &mut Self, #[comptime] config: G);
}
