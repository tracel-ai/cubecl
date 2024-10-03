use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
/// Defines the number of tiles and their size in each plane
pub trait TileReader<E: CubePrimitive>: CubeType + 'static + Send + Sync {
    fn read(
        reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> &Slice<'_, E>;
}

#[cube]
/// Defines the number of tiles and their size in each plane
pub trait TileWriter<E: CubePrimitive>: CubeType + 'static + Send + Sync {
    // fn write(writer: &mut Self, slice: &Slice<'_, E>, pos_x: u32, pos_y: u32);
    fn write_with_cast<C: Numeric>(
        writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    );
}
