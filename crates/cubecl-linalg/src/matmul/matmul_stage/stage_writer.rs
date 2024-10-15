use cubecl_core as cubecl;
use cubecl_core::prelude::*;


#[cube]
pub trait StageWriter<EG: Numeric>: CubeType + 'static + Send + Sync {
    fn write<ES: Numeric>(
        tile_writer: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    );
}
