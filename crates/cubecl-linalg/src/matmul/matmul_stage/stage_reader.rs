use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::SmmConfig;

#[cube]
pub trait StageReader<ES: Numeric>: CubeType {
    type Config: SmmConfig;

    fn read_tile(
        self_: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: Self::Config,
    ) -> &Slice<'_, Line<ES>>;
}
