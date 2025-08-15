use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_matmul::components::MatmulSetupError;
use cubecl_matmul::components::tile::{TileConfig, TileMatmul, TileSetupInfo};

use crate::components::AttentionPrecision;
use crate::components::tile::shared::{KeyValue, ScoreProb};

#[cube]
pub trait ValueMatmul<AP: AttentionPrecision>:
    TileMatmul<AP::MatmulPrecision, Lhs = ScoreProb<AP>, Rhs = KeyValue>
{
}

pub trait ValueMatmulFamily: Send + Sync + 'static {
    type Matmul<AP: AttentionPrecision>: ValueMatmul<AP>;
    type Config: TileConfig;

    fn setup<AP: AttentionPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        tile_setup_info: TileSetupInfo,
    ) -> Result<Self::Config, MatmulSetupError>;
}
