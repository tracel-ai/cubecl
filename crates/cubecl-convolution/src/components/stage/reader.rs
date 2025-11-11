use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    InvalidConfigError, MatrixLayout,
    global::memory::GlobalMemoryConfig,
    stage::{StageMemoryConfig, StridedStage, TilingLayout, TilingValidation},
    tile::StridedTile,
};
use cubecl_std::tensor::layout::Coords2d;

#[derive(Clone, Copy)]
/// Tiling layout specific for bias, which is one-dimensional with stride 0
pub struct BiasTilingLayout {}

#[cube]
impl TilingLayout for BiasTilingLayout {
    fn get_tile<ES: Numeric>(
        stage: &StridedStage<ES, Self>,
        tile: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES> {
        if comptime!(config.num_stages > 1) {
            unimplemented!()
        }

        let (_, col) = tile;

        let stage_line_size = config.stage_line_size;
        let tile_size_col = config.elements_in_tile_col / stage_line_size;

        let length = tile_size_col;
        let start = col * tile_size_col;

        StridedTile::new_strided(
            stage.as_slice(stage_line_size).slice(start, start + length),
            0,
            MatrixLayout::RowMajor,
        )
    }
}

impl TilingValidation for BiasTilingLayout {
    fn check(config: GlobalMemoryConfig) -> Result<(), InvalidConfigError> {
        let stage_width = config.elements_in_stage_col();
        if config.line_size() > stage_width {
            return Err(Box::new("Invalid line size"));
        }
        Ok(())
    }
}
