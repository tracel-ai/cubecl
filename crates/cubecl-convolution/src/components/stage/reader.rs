use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    MatrixLayout, StageIdent,
    stage::{StageMemory, StageMemoryConfig, TilingLayout},
    tile::Tile,
};

#[derive(Clone, Copy)]
/// Tiling layout specific for bias, which is one-dimensional with stride 0
pub struct BiasTilingLayout {}

#[cube]
impl TilingLayout for BiasTilingLayout {
    fn get_tile<ES: Numeric, S: StageMemoryConfig>(
        stage: &StageMemory<ES, Self>,
        _x: u32, // m is ignored
        y: u32,
        #[comptime] _buffer_index: u32,
        #[comptime] ident: StageIdent,
        #[comptime] config: S,
    ) -> Tile<ES> {
        if comptime!(config.num_stages(ident) > 1) {
            unimplemented!()
        }

        let stage_line_size = config.stage_line_size(ident);
        let tiling_scheme = config.tiling_scheme();

        let tile_size_y = tiling_scheme.elements_in_tile_n() / stage_line_size;

        let length = tile_size_y;
        let start = y * tile_size_y;

        Tile::new_strided(
            stage.as_slice(stage_line_size).slice(start, start + length),
            0,
            MatrixLayout::RowMajor,
        )
    }
}
