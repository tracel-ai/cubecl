use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::{
    block_info::BlockInfo,
    data::{GmemView, Tile},
    matrix_layout::MatrixLayout,
};

#[cube]
pub trait BlockReader<E: Numeric>: CubeType {
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
    type GmemView: GmemView<E>;
    type BlockReader: BlockReader<E>;

    fn new(
        gmem: <Self::GmemView as GmemView<E>>::Gmem,
        #[comptime] layout: MatrixLayout,
        #[comptime] block_info: BlockInfo,
    ) -> Self;

    fn fill_block(loader: &mut Self) -> Self::BlockReader;

    fn advance(loader: &mut Self, k_offset: u32);
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
