use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::SmmConfig;

#[cube]
pub trait StageWriter<EG: Numeric>: CubeType + 'static + Send + Sync {
    type Config: SmmConfig;

    fn write<ES: Numeric>(
        tile_writer: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] slice_line_size: u32,
        #[comptime] config: Self::Config,
    );
}
