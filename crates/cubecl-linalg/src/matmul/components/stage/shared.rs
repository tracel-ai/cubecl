use crate::matmul::components::{
    tile::TileConfig, CompleteStageTiling, Ident, InputIdent, MatmulConfig, MatmulSize,
    MatrixLayout, StageTiling,
};

use super::{StageConfig, TilingLayout};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the single buffer matmul
pub struct CommonStageConfig<T: TileConfig> {
    pub tmm_config: T,
    pub tiling: CompleteStageTiling,
    pub num_planes: u32,
    pub lhs_tiling_layout: TilingLayout,
    pub rhs_tiling_layout: TilingLayout,
    pub quantized: bool,
}

impl<T: TileConfig> StageConfig for CommonStageConfig<T> {
    type TmmConfig = T;

    fn to_tmm_config(self) -> Self::TmmConfig {
        self.tmm_config
    }

    fn line_size(&self, ident: Ident) -> u32 {
        self.tmm_config.line_size(ident)
    }

    fn tiling(&self, ident: Ident) -> StageTiling {
        self.tiling.get(ident)
    }

    fn layout(&self, ident: Ident) -> MatrixLayout {
        self.tmm_config.layout(ident)
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn plane_dim(&self) -> u32 {
        self.tmm_config.plane_dim()
    }

    fn tiling_layout(&self, ident: Ident) -> TilingLayout {
        match ident.as_input() {
            InputIdent::Lhs => self.lhs_tiling_layout,
            InputIdent::Rhs => self.rhs_tiling_layout,
        }
    }

    fn tile_count(&self) -> &MatmulSize {
        &self.tiling.tile_count
    }
}

impl<T: TileConfig> MatmulConfig for CommonStageConfig<T> {}

impl<T: TileConfig> CommonStageConfig<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tmm_config: T,
        tiling: CompleteStageTiling,
        num_planes: u32,
        lhs_tiling_layout: TilingLayout,
        rhs_tiling_layout: TilingLayout,
        quantized: bool,
    ) -> Self {
        Self {
            tmm_config,
            tiling,
            num_planes,
            lhs_tiling_layout,
            rhs_tiling_layout,
            quantized,
        }
    }
}
