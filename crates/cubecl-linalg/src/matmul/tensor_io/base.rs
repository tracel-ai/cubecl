use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::tile_io::{TileReader, TileWriter};

#[cube]
pub trait TensorLoader<E: Numeric>: CubeType + 'static + Send + Sync {
    type TileReader: TileReader<Line<E>>;

    fn load_tile(reader: &mut Self, k_offset: u32) -> Self::TileReader;
}

#[cube]
// This type is a useless wrapper at the moment
// Maybe we will want to differentiate writing to a slice and writing to gmem...
pub trait TensorWriter<E: Numeric>: CubeType + 'static + Send + Sync {
    type TileWriter: TileWriter<Line<E>>;

    fn as_tile_writer(writer: Self) -> Self::TileWriter;
}
