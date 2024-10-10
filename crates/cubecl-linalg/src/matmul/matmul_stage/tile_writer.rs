use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait StageWriter<E: Numeric>: CubeType + 'static + Send + Sync {
    fn write_with_cast<C: Numeric>(
        tile_writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    );
}
