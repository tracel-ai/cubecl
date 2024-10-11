use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::stage_info::StageInfo;

#[cube]
// TODO separate load from write, they should never be the same instance
pub trait GlobalView<E: Numeric>: CubeType {
    type Global: CubeType;

    fn line_size(view: &Self) -> u32;

    fn load_coalesced(
        view: &Self,
        tile_x: u32,
        tile_y: u32,
        load_id: u32,
        tile_size_x: u32,
        tile_size_y: u32,
    ) -> Line<E>;
    fn load_shared_memory<ES: Numeric>(
        view: &Self,
        shared_memory: &mut SharedMemory<Line<ES>>,
        #[comptime] stage_info: StageInfo,
    );

    fn write_single<C: CubePrimitive>(view: &mut Self, write_row: u32, write_col: u32, value: C);
    fn write_slice<C: CubePrimitive>(
        view: &mut Self,
        slice: &Slice<'_, C>,
        write_row: u32,
        write_col: u32,
        #[comptime] stage_info: StageInfo,
    );

    fn init_view(view: &mut Self, x_offset: u32, y_offset: u32);
    fn update_view(view: &mut Self, x_offset: u32, y_offset: u32);
}
