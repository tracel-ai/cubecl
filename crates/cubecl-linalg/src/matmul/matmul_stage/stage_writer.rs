use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matmul_global::GmmConfig;

#[cube]
pub trait StageWriter<EG: Numeric>: CubeType + 'static + Send + Sync {
    fn write<ES: Numeric, G: GmmConfig>(
        tile_writer: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        #[comptime] config: G,
    );
}
