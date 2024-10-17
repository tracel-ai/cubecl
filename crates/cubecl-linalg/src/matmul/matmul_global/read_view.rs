use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matmul_stage::TilingOrder;
use crate::matmul::matrix::Ident;

#[cube]
pub trait ReadView<E: Numeric>: CubeType {
    type Global: CubeType;
    type Config: GmmConfig;

    fn load_coalesced(
        view: &Self,
        tile_x: u32,
        tile_y: u32,
        load_id: u32,
        #[comptime] ident: Ident,
        #[comptime] config: Self::Config,
    ) -> Line<E>;

    fn load_shared_memory<ES: Numeric, O: TilingOrder>(
        view: &Self,
        shared_memory: &mut SharedMemory<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: Self::Config,
    );

    fn init_view(view: &mut Self, x_offset: u32, y_offset: u32);
    fn update_view(view: &mut Self, x_offset: u32, y_offset: u32);
}
