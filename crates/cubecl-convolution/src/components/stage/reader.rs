use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    MatrixLayout, StageIdent,
    stage::{StageMemory, StageMemoryConfig, TilingLayout},
    tile::Tile,
};
use cubecl_std::tensor::layout::Coords2d;

#[derive(Clone, Copy)]
/// Tiling layout specific for bias, which is one-dimensional with stride 0
pub struct BiasTilingLayout {}

#[cube]
impl TilingLayout for BiasTilingLayout {
    fn get_tile<ES: Numeric, S: StageMemoryConfig>(
        stage: &StageMemory<ES, Self>,
        tile: Coords2d,
        #[comptime] _buffer_index: u32,
        #[comptime] ident: StageIdent,
        #[comptime] config: S,
    ) -> Tile<ES> {
        if comptime!(config.num_stages(ident) > 1) {
            unimplemented!()
        }

        let (_, col) = tile;

        let stage_line_size = config.stage_line_size(ident);
        let tiling_scheme = config.tiling_scheme();

        let tile_size_col = tiling_scheme.elements_in_tile_n() / stage_line_size;

        let length = tile_size_col;
        let start = col * tile_size_col;

        Tile::new_strided(
            stage.as_slice(stage_line_size).slice(start, start + length),
            0,
            MatrixLayout::RowMajor,
        )
    }
}
