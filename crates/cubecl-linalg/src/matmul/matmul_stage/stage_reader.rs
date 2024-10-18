use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::SmmConfig;

#[cube]
pub trait StageReader<ES: Numeric, S: SmmConfig>: CubeType {
    fn read_tile(
        self_: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: S,
    ) -> &Slice<'_, Line<ES>>;
}
