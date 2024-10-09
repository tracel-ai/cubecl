use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait GlobalView<E: Numeric>: CubeType {
    type Global: CubeType;

    fn init_view(view: &mut Self, x_offset: u32, y_offset: u32);
    fn load_single(view: &Self, read_row: u32, read_col: u32) -> Line<E>;
    fn update_view(view: &mut Self, x_offset: u32, y_offset: u32);
}
