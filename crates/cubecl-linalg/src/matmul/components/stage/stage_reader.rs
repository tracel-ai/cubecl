use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::SmmConfig;

#[cube]
/// Input to the stage matmul, responsible of handing slices of data
/// at precise locations in the stage
pub trait StageReader<ES: Numeric, S: SmmConfig>: CubeType {
    /// Hands a portion of data from the stage, whose location is function of the
    /// plane, buffer and accumulator indexes.
    fn read_tile(
        this: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: S,
    ) -> &Slice<'_, Line<ES>>;
}
