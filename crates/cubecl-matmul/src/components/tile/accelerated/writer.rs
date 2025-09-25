use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::components::{
    as_cmma_layout,
    tile::{
        StridedTile,
        io::{StageWriter, Strided},
    },
};

/// Writer using the cmma store function.
#[derive(CubeType)]
pub struct CmmaStageWriter {}

#[cube]
impl CmmaStageWriter {
    pub fn store_fragment<E: Numeric, V: Numeric>(
        tile: &mut StridedTile<V, ReadWrite>,
        fragment: &cmma::Matrix<E>,
        #[comptime] line_size: u32,
    ) {
        let layout = as_cmma_layout(tile.layout);
        let (mut slice, stride) = tile.as_unlined(line_size);
        cmma::store(&mut slice, fragment, stride, layout);
    }
}

impl StageWriter for CmmaStageWriter {
    type TileKind = Strided;
}
