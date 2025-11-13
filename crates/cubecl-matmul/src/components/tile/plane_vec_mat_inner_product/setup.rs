use crate::components::tile::SharedTileConfig;
use crate::components::tile::plane_vec_mat_inner_product::matmul::PlaneVecMatInnerProduct;
use crate::components::tile::{TileMatmulFamily, io::Strided};
use crate::components::{
    InvalidConfigError, MatmulAvailabilityError, MatmulElems, MatmulLineSizes, MatmulProblem,
    MatmulSelection, MatrixLayout,
};
use crate::components::{error::MatmulSetupError, tile::io::TileKind};
use crate::components::{
    resource::ComputeResources,
    tile::plane_vec_mat_inner_product::reader::{MatrixFragmentReader, MatrixStageReader},
};
use cubecl_core::ir::{ElemType, FloatKind};
use cubecl_core::prelude::*;
use cubecl_runtime::{Plane, TypeUsage};

impl<Kind: TileKind> TileMatmulFamily for PlaneVecMatInnerProduct<Kind>
where
    MatrixStageReader<Kind>: MatrixFragmentReader<TileKind = Kind>,
{
    type Config = SharedTileConfig;
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = PlaneVecMatInnerProduct<Kind>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Kind;
    type OutTile = Strided;

    fn requires_accelerator() -> bool {
        false
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
        let tile_config = check_availability::<R>(tile_config, client, dtypes)?;

        if tile_config.lhs_layout != MatrixLayout::RowMajor {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Only Row Major layout is supported for Lhs",
            )));
        }

        if tile_config.rhs_layout != MatrixLayout::ColMajor {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Only Col Major layout is supported for Rhs",
            )));
        }

        let m = tile_config.tile_size.m();
        let n = tile_config.tile_size.n();
        let k = tile_config.tile_size.k();

        let lhs_line = tile_config.lhs_stage_line_size;
        let rhs_line = tile_config.rhs_stage_line_size;
        let out_line = tile_config.out_global_line_size;

        if m != 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Only m=1 is supported, got m={m:?}",
            ))));
        }

        if lhs_line != rhs_line {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Lhs and Rhs must have same line size, got lhs={lhs_line:?} and rhs={rhs_line:?}",
            ))));
        }

        if k != tile_config.plane_dim * lhs_line {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "k must be equal to plane_dim times line size (of both lhs and rhs), got k={:?}, plane_dim={:?} line_size={:?}",
                k, tile_config.plane_dim, lhs_line
            ))));
        }

        if !n.is_multiple_of(out_line) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "n must be divisible by out line size, got n={n:?}, out_line_size={out_line:?}",
            ))));
        }

        Ok(tile_config)
    }
}

fn check_availability<R: Runtime>(
    tile_config: SharedTileConfig,
    client: &ComputeClient<R::Server>,
    dtypes: &MatmulElems,
) -> Result<SharedTileConfig, MatmulSetupError> {
    if !client.properties().features.plane.contains(Plane::Ops) {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::PlaneOpsUnavailable,
        ));
    }

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
