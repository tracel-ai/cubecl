use super::{StageBuffering, StageConfig};
use crate::matmul::components::{
    CompleteStageTiling, Ident, MatmulConfig, MatmulSize, MatrixLayout, TilingDimensions,
    tile::TileConfig,
};
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration for the single buffer matmul
pub struct CommonStageConfig<T: TileConfig> {
    pub tmm_config: T,
    pub tiling: CompleteStageTiling,
    pub num_planes: u32,
    pub quantized: bool,
    pub buffering: StageBuffering,
}

impl<T: TileConfig> StageConfig for CommonStageConfig<T> {
    type TmmConfig = T;

    fn to_tmm_config(self) -> Self::TmmConfig {
        self.tmm_config
    }

    fn line_size(&self, ident: Ident) -> u32 {
        self.tmm_config.line_size(ident)
    }

    fn tiling_dimensions(&self, ident: Ident) -> TilingDimensions {
        self.tiling.get(ident)
    }

    fn matrix_layout(&self, ident: Ident) -> MatrixLayout {
        self.tmm_config.matrix_layout(ident)
    }

    fn num_planes(&self) -> u32 {
        self.num_planes
    }

    fn plane_dim(&self) -> u32 {
        self.tmm_config.plane_dim()
    }

    fn tile_count(&self) -> &MatmulSize {
        &self.tiling.tile_count
    }

    fn buffering(&self) -> StageBuffering {
        self.buffering
    }
}

impl<T: TileConfig> MatmulConfig for CommonStageConfig<T> {}

impl<T: TileConfig> CommonStageConfig<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tmm_config: T,
        tiling: CompleteStageTiling,
        num_planes: u32,
        quantized: bool,
        buffering: StageBuffering,
    ) -> Self {
        Self {
            tmm_config,
            tiling,
            num_planes,
            quantized,
            buffering,
        }
    }
}

#[derive(CubeType)]
pub enum RhsTile<Rhs: CubeType> {
    Single(Rhs),
    Double((Rhs, Rhs)),
}
