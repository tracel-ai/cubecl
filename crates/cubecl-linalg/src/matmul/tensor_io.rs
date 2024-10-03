use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::tile_io::{TileReader, TileWriter};

#[cube]
pub trait TensorReader<E: Numeric>: CubeType + 'static + Send + Sync {
    type TileReader: TileReader<Line<E>>;

    fn read(reader: &Self, k_offset: u32) -> Self::TileReader;
}

#[cube]
pub trait TensorWriter<E: Numeric>: CubeType + 'static + Send + Sync {
    type TileWriter: TileWriter<Line<E>>;

    fn make_tile_writer() -> Self::TileWriter;
    fn write(writer: &mut Self, tile_writer: Self::TileWriter);
}
