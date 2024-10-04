use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait TileReader<E: CubePrimitive>: CubeType {
    fn read(
        tile_reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> &Slice<'_, E>;
}

#[cube]
pub trait TensorLoader<E: CubePrimitive>: CubeType + 'static + Send + Sync {
    type TileReader: TileReader<E>;

    fn load_block(tensor_loader: &mut Self, k_offset: u32) -> Self::TileReader;
}

#[cube]
pub trait TileWriter<E: CubePrimitive>: CubeType + 'static + Send + Sync {
    type Gmem: CubeType;

    fn write_with_cast<C: Numeric>(
        tile_writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    );
}
