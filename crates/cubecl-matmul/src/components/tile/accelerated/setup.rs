use crate::components::tile::accelerated::config::AcceleratedConfig;
use crate::components::tile::accelerated::matmul::AcceleratedMatmul;
use crate::components::tile::{
    TileMatmulFamily,
    accelerated::reader::{CmmaFragmentReader, CmmaStageReader},
};
use crate::components::{InvalidConfigError, MatmulLineSizes, MatmulProblem, MatmulSelection};
use crate::components::{error::MatmulSetupError, tile::io::Strided};
use crate::components::{resource::ComputeResources, tile::io::TileKind};
use cubecl_core::prelude::*;

impl<Tile: TileKind> TileMatmulFamily for AcceleratedMatmul<Tile>
where
    CmmaStageReader<Tile>: CmmaFragmentReader<TileKind = Tile>,
{
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = AcceleratedMatmul<Tile>;
    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Tile;
    type OutTile = Strided;

    type Config = AcceleratedConfig;

    fn requires_accelerator() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        AcceleratedConfig::new::<Lhs, Rhs, Acc, R>(
            client,
            selection.tiling_scheme.tile_size,
            selection.plane_dim,
            problem.lhs_layout,
            problem.rhs_layout,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
            matmul_line_sizes.out as u32,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
        )
    }
}
