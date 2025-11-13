use crate::components::tile::SharedTileConfig;
use crate::components::tile::register::config::RegisterMatmulConfig;
use crate::components::tile::register::matmul::RegisterMatmul;
use crate::components::tile::{TileMatmulFamily, io::Strided};
use crate::components::{
    AvailableLineSizes, InvalidConfigError, MatmulAvailabilityError, MatmulElems, MatmulLineSizes,
    MatmulProblem, MatmulSelection, MatrixLayout,
};
use crate::components::{error::MatmulSetupError, tile::io::TileKind};
use crate::components::{
    resource::ComputeResources,
    tile::register::reader::{RegisterFragmentReader, RegisterStageReader},
};
use cubecl_core::ir::{ElemType, FloatKind};
use cubecl_core::prelude::*;
use cubecl_runtime::TypeUsage;

impl<AccTile: TileKind> TileMatmulFamily for RegisterMatmul<AccTile>
where
    RegisterStageReader<AccTile>: RegisterFragmentReader<TileKind = AccTile>,
{
    type Config = RegisterMatmulConfig;
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = RegisterMatmul<AccTile>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = AccTile;
    type OutTile = Strided;

    fn requires_accelerator() -> bool {
        false
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Units(1))
    }

    fn setup<R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = RegisterMatmulConfig::from_shared_tile_config(SharedTileConfig::new(
            selection.tiling_scheme.tile_size,
            selection.plane_dim,
            problem.lhs_layout,
            problem.rhs_layout,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
            matmul_line_sizes.out as u32,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
        ));

        Self::validate::<R>(tile_config, client, dtypes)
    }

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
    }

    fn validate<R: Runtime>(
        tile_config: RegisterMatmulConfig,
        client: &ComputeClient<R::Server>,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let tile_config = check_availability::<R>(tile_config, client, dtypes)?;

        let m = tile_config.shared.tile_size.m();
        let n = tile_config.shared.tile_size.n();
        let k = tile_config.shared.tile_size.k();

        let lhs = tile_config.shared.lhs_stage_line_size;
        let rhs = tile_config.shared.rhs_stage_line_size;
        let out = tile_config.shared.out_global_line_size;

        match tile_config.shared.lhs_layout {
            MatrixLayout::RowMajor => {
                if !k.is_multiple_of(lhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis k({k:?}) should be divisible by line size lhs({lhs:?})"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if !m.is_multiple_of(lhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis m({m:?}) should be divisible by line size lhs({lhs:?})"
                    ))));
                }
            }
        }
        match tile_config.shared.rhs_layout {
            MatrixLayout::RowMajor => {
                if !n.is_multiple_of(rhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis n({n:?}) should be divisible by line size rhs({rhs:?})"
                    ))));
                }
            }
            MatrixLayout::ColMajor => {
                if !k.is_multiple_of(rhs) {
                    return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                        "Tile shape in lined axis k({k:?}) should be divisible by line size rhs({rhs:?})"
                    ))));
                }
            }
        }

        if !n.is_multiple_of(out) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Tile shape in lined axis n({n:?}) should be divisible by line size out({out:?})"
            ))));
        }

        Ok(tile_config)
    }
}

fn check_availability<R: Runtime>(
    tile_config: RegisterMatmulConfig,
    client: &ComputeClient<R::Server>,
    dtypes: &MatmulElems,
) -> Result<RegisterMatmulConfig, MatmulSetupError> {
    let lhs = dtypes.lhs_register;
    let rhs = dtypes.rhs_register;
    let acc = dtypes.acc_register;

    let lhs = match lhs {
        StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
            ElemType::Float(FloatKind::F32).into()
        }
        _ => lhs,
    };
    let rhs = match rhs {
        StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
            ElemType::Float(FloatKind::F32).into()
        }
        _ => rhs,
    };

    let output = match acc {
        StorageType::Scalar(ElemType::Float(FloatKind::Flex32)) => {
            ElemType::Float(FloatKind::F32).into()
        }
        _ => acc,
    };

    if !(client
        .properties()
        .features
        .type_usage(lhs)
        .contains(TypeUsage::Arithmetic)
        && client
            .properties()
            .features
            .type_usage(rhs)
            .contains(TypeUsage::Arithmetic)
        && client
            .properties()
            .features
            .type_usage(output)
            .contains(TypeUsage::Arithmetic))
    {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::TypesUnavailable { lhs, rhs, output },
        ));
    }

    Ok(tile_config)
}
