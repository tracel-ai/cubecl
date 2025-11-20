use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_matmul::components::{
    InvalidConfigError, MatrixLayout,
    stage::{
        StageMemoryConfig, StridedStageMemory, TilingLayout, TilingLayoutEnum, TilingValidation,
    },
    tile::StridedTile,
};
use cubecl_std::tensor::layout::Coords2d;

#[derive(Clone, Copy)]
/// Tiling layout specific for bias, which is one-dimensional with stride 0
pub struct BiasTilingLayout {}

#[cube]
impl TilingLayout for BiasTilingLayout {
    fn get_tile<ES: Numeric>(
        stage: &StridedStageMemory<ES, Self>,
        tile: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> StridedTile<ES> {
        if comptime!(config.num_stages > 1) {
            unimplemented!()
        }

        let (_, col) = tile;

        let stage_line_size = config.line_size;
        let tile_size_col = config.elements_per_tile_along_col / stage_line_size;

        let length = tile_size_col;
        let start = col * tile_size_col;

        StridedTile::new_strided(
            stage.as_slice(stage_line_size),
            start,
            start + length,
            0,
            stage.swizzle,
            MatrixLayout::RowMajor,
            stage_line_size,
        )
    }

    fn to_enum() -> comptime_type!(TilingLayoutEnum) {
        comptime![TilingLayoutEnum::Other]
    }
}

impl TilingValidation for BiasTilingLayout {
    fn check(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
        let stage_width = config.elements_per_stage_along_col();
        if config.line_size > stage_width {
            return Err(Box::new("Invalid line size"));
        }
        Ok(())
    }
}
