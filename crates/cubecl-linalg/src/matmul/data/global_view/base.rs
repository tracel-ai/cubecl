use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::stage_info::StageInfo;

#[cube]
pub trait GlobalView<E: Numeric>: CubeType {
    type Global: CubeType;

    fn load_single(view: &Self, read_row: u32, read_col: u32) -> Line<E>;
    fn load_shared_memory<ES: Numeric>(
        view: &Self,
        shared_memory: &mut SharedMemory<Line<ES>>,
        #[comptime] stage_info: StageInfo,
    );

    fn init_view(view: &mut Self, x_offset: u32, y_offset: u32);
    fn update_view(view: &mut Self, x_offset: u32, y_offset: u32);
}
