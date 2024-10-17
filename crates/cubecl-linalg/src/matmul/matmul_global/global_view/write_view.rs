use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::stage_info::StageInfo;

#[cube]
pub trait WriteView<E: Numeric>: CubeType {
    type Global: CubeType;
    type Config: GmmConfig;

    fn write_coalesced<ES: Numeric>(
        view: &mut Self,
        write_row: u32,
        write_col: u32,
        value: Line<ES>,
    );

    fn write_slice<ES: Numeric>(
        view: &mut Self,
        slice: &Slice<'_, Line<ES>>,
        write_row: u32,
        write_col: u32,
        #[comptime] stage_info: StageInfo,
        #[comptime] slice_tile_size: u32,
        #[comptime] config: Self::Config,
    );

    fn init_view(view: &mut Self, x_offset: u32, y_offset: u32);
}
