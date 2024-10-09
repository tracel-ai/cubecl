use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::{
    stage_info::StageInfo,
    data::{GlobalView, Tile},
    matrix_layout::MatrixLayout,
};

#[cube]
pub trait StageReader<E: Numeric>: CubeType {
    fn read_tile(
        tile_reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> Tile<'_, E>;

    // Maybe delete if we don't need layout prior to slice
    fn slice_layout(tile_reader: &Self) -> MatrixLayout;
}

#[cube]
pub trait Loader<E: Numeric>: CubeType + 'static + Send + Sync {
    type GmemView: GlobalView<E>;
    type StageReader: StageReader<E>;

    fn new(
        gmem: <Self::GmemView as GlobalView<E>>::Global,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: StageInfo,
    ) -> Self;

    fn fill_block(loader: &mut Self) -> Self::StageReader;

    fn init_view(loader: &mut Self, cube_offset: u32, k_start: u32);

    fn advance_view(loader: &mut Self, k_offset: u32);
}

#[cube]
pub trait TileWriter<E: CubePrimitive>: CubeType + 'static + Send + Sync {
    fn write_with_cast<C: Numeric>(
        tile_writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    );
}
