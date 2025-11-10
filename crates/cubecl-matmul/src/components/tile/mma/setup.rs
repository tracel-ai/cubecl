use crate::components::{
    InvalidConfigError, MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection,
};
use crate::components::{
    TileSize,
    tile::{
        TileMatmulFamily,
        mma::{
            MmaMatmul,
            config::MmaMatmulConfig,
            reader::{MmaFragmentReader, MmaStageReader},
        },
    },
};
use crate::components::{error::MatmulSetupError, tile::io::Strided};
use crate::components::{resource::ComputeResources, tile::io::TileKind};
use cubecl_core::{ir::StorageType, prelude::*};
use cubecl_runtime::MmaConfig;

impl<Tile: TileKind> TileMatmulFamily for MmaMatmul<Tile>
where
    MmaStageReader<Tile>: MmaFragmentReader<TileKind = Tile>,
{
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = MmaMatmul<Tile>;
    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Tile;
    type OutTile = Strided;

    type Config = MmaMatmulConfig;

    fn requires_accelerator() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        MmaMatmulConfig::new::<R>(
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
            dtypes,
        )
    }

    fn is_supported<R: Runtime>(client: &ComputeClient<R::Server>, config: MmaConfig) -> bool {
        client.properties().features.mma.contains(&config)
    }

    fn supported_sizes<R: Runtime>(
        client: &ComputeClient<R::Server>,
        lhs_ty: StorageType,
        rhs_ty: StorageType,
        acc_ty: StorageType,
    ) -> Vec<TileSize> {
        client
            .properties()
            .features
            .mma
            .iter()
            .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
            .map(|it| (it.m, it.n, it.k).into())
            .collect()
    }
}
