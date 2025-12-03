use crate::components::tile::{SharedTileConfig, mma::config::StoreMethod};
use crate::components::{
    InvalidConfigError, MatmulAvailabilityError, MatmulElems, MatmulLineSizes, MatmulProblem,
    MatmulSelection,
};
use crate::components::{
    TileSize,
    tile::{
        TileMatmulFamily,
        mma::{
            MmaMatmul,
            reader::{MmaFragmentReader, MmaStageReader},
        },
    },
};
use crate::components::{error::MatmulSetupError, tile::io::Strided};
use crate::components::{resource::ComputeResources, tile::io::TileKind};
use crate::{
    components::tile::mma::config::{LoadMethod, MmaMatmulConfig},
    tune_key::MatmulElemType,
};
use cubecl_core::{ir::StorageType, prelude::*};
use cubecl_runtime::MmaConfig;

impl<LhsTile: TileKind, RhsTile: TileKind, AccTile: TileKind> TileMatmulFamily
    for MmaMatmul<LhsTile, RhsTile, AccTile>
where
    MmaStageReader<LhsTile>: MmaFragmentReader<TileKind = LhsTile>,
    MmaStageReader<RhsTile>: MmaFragmentReader<TileKind = RhsTile>,
    MmaStageReader<AccTile>: MmaFragmentReader<TileKind = AccTile>,
{
    type Config = MmaMatmulConfig;

    type Matmul<L: Numeric, R: Numeric, A: Numeric> = MmaMatmul<LhsTile, RhsTile, AccTile>;
    type LhsTile = LhsTile;
    type RhsTile = RhsTile;
    type AccTile = AccTile;
    type OutTile = Strided;

    fn requires_accelerator() -> bool {
        true
    }

    fn can_cast_stage_element() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        _problem: &MatmulProblem,
        selection: &MatmulSelection,
        _matmul_line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = MmaMatmulConfig::from_shared_tile_config(
            SharedTileConfig {
                tile_size: selection.tiling_scheme.tile_size,
                plane_dim: selection.plane_dim,
                swizzle_config: selection.shared_swizzle,
            },
            load_method(client, dtypes.lhs_stage),
            load_method(client, dtypes.rhs_stage),
            load_method(client, dtypes.acc_stage),
            store_method(client, dtypes.acc_stage),
        );

        validate(tile_config, client, dtypes)
    }

    fn should_swizzle<R: Runtime>(client: &ComputeClient<R>) -> bool {
        // No alignment means swizzling can't be properly used, since it needs to be applied to
        // the address, and alignment guarantees the offset is aligned to the pattern repeat.
        client.properties().features.alignment
    }

    fn is_supported<R: Runtime>(client: &ComputeClient<R>, config: MmaConfig) -> bool {
        client.properties().features.mma.contains(&config)
    }

    fn supported_sizes<R: Runtime>(
        client: &ComputeClient<R>,
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

fn validate<R: Runtime>(
    tile_config: MmaMatmulConfig,
    client: &ComputeClient<R>,
    dtypes: &MatmulElems,
) -> Result<MmaMatmulConfig, MatmulSetupError> {
    let lhs = *dtypes.lhs_register;
    let rhs = *dtypes.rhs_register;
    let acc = *dtypes.acc_register;

    let size = tile_config.shared.tile_size;
    if !client.properties().features.mma.contains(&MmaConfig {
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

fn load_method<R: Runtime>(client: &ComputeClient<R>, dtype: MatmulElemType) -> LoadMethod {
    if !dtype.quantized && client.properties().features.ldmatrix.contains(&dtype) {
        LoadMethod::LoadMatrix
    } else {
        LoadMethod::Manual
    }
}

fn store_method<R: Runtime>(client: &ComputeClient<R>, dtype: MatmulElemType) -> StoreMethod {
    if !dtype.quantized && client.properties().features.stmatrix.contains(&dtype) {
        StoreMethod::StoreMatrix
    } else {
        StoreMethod::Manual
    }
}
