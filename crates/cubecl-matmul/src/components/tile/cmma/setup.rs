use crate::components::TileSize;
use crate::components::tile::SharedTileConfig;
use crate::components::tile::cmma::matmul::CmmaMatmul;
use crate::components::tile::{
    TileMatmulFamily,
    cmma::reader::{CmmaFragmentReader, CmmaStageReader},
};
use crate::components::{
    InvalidConfigError, MatmulAvailabilityError, MatmulElems, MatmulLineSizes, MatmulProblem,
    MatmulSelection,
};
use crate::components::{error::MatmulSetupError, tile::io::Strided};
use crate::components::{resource::ComputeResources, tile::io::TileKind};
use cubecl_core::{ir::StorageType, prelude::*};
use cubecl_runtime::MmaConfig;

impl<Tile: TileKind> TileMatmulFamily for CmmaMatmul<Tile>
where
    CmmaStageReader<Tile>: CmmaFragmentReader<TileKind = Tile>,
{
    type Config = SharedTileConfig;
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = CmmaMatmul<Tile>;
    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Tile;
    type OutTile = Strided;

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
    ) -> Result<SharedTileConfig, MatmulSetupError> {
        let tile_config = SharedTileConfig::new(
            selection.tiling_scheme.tile_size,
            selection.plane_dim,
            problem.lhs_layout,
            problem.rhs_layout,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
            matmul_line_sizes.out as u32,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
        );

        Self::validate::<R>(tile_config, client, dtypes)
    }

    fn validate<R: Runtime>(
        tile_config: SharedTileConfig,
        client: &ComputeClient<R::Server>,
        dtypes: &MatmulElems,
    ) -> Result<SharedTileConfig, MatmulSetupError> {
        let lhs = dtypes.lhs_register;
        let rhs = dtypes.rhs_register;
        let acc = dtypes.acc_register;

        let size = tile_config.tile_size;
        if !client.properties().features.cmma.contains(&MmaConfig {
            a_type: lhs,
            b_type: rhs,
            cd_type: acc,
            m: size.m(),
            k: size.k(),
            n: size.n(),
        }) {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::CmmaInstructionUnavailable {
                    lhs,
                    rhs,
                    output: acc,
                    size: Some(TileSize::new(size.m(), size.n(), size.k())),
                },
            ));
        }

        Ok(tile_config)
    }

    fn is_supported<R: Runtime>(client: &ComputeClient<R::Server>, config: MmaConfig) -> bool {
        client.properties().features.cmma.contains(&config)
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
            .cmma
            .iter()
            .filter(|it| it.a_type == lhs_ty && it.b_type == rhs_ty && it.cd_type == acc_ty)
            .map(|it| (it.m, it.n, it.k).into())
            .collect()
    }
}
