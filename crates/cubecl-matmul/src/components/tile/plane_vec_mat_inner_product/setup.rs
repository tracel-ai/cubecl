use crate::components::tile::SharedTileConfig;
use crate::components::tile::plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig;
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
    type Config = PlaneVecMatInnerProductConfig;
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = PlaneVecMatInnerProduct<Kind>;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Kind;
    type OutTile = Strided;

    fn requires_accelerator() -> bool {
        false
    }

    fn can_cast_stage_element() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<PlaneVecMatInnerProductConfig, MatmulSetupError> {
        let tile_config = PlaneVecMatInnerProductConfig::new(
            SharedTileConfig::new(
                selection.tiling_scheme.tile_size,
                selection.plane_dim,
                selection.shared_swizzle,
            ),
            matmul_line_sizes.lhs as u32,
        );

        validate(
            tile_config,
            problem.lhs_layout,
            problem.rhs_layout,
            matmul_line_sizes,
            client,
            dtypes,
        )
    }

    fn should_swizzle<R: Runtime>(_client: &ComputeClient<R>) -> bool {
        // Supported but need to find good settings for this tiling. Currently tuned for `ldmatrix`.
        // Need to profile at some point
        false
    }
}

fn validate<R: Runtime>(
    tile_config: PlaneVecMatInnerProductConfig,
    lhs_layout: MatrixLayout,
    rhs_layout: MatrixLayout,
    matmul_line_sizes: &MatmulLineSizes,
    client: &ComputeClient<R>,
    dtypes: &MatmulElems,
) -> Result<PlaneVecMatInnerProductConfig, MatmulSetupError> {
    let tile_config = check_availability(tile_config, client, dtypes)?;

    if lhs_layout != MatrixLayout::RowMajor {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Only Row Major layout is supported for Lhs",
        )));
    }

    if rhs_layout != MatrixLayout::ColMajor {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Only Col Major layout is supported for Rhs",
        )));
    }

    let m = tile_config.shared.tile_size.m();
    let n = tile_config.shared.tile_size.n();
    let k = tile_config.shared.tile_size.k();

    let lhs_line = matmul_line_sizes.lhs as u32;
    let rhs_line = matmul_line_sizes.rhs as u32;
    let out_line = matmul_line_sizes.out as u32;

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

    if k != tile_config.shared.plane_dim * lhs_line {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "k must be equal to plane_dim times line size (of both lhs and rhs), got k={:?}, plane_dim={:?} line_size={:?}",
            k, tile_config.shared.plane_dim, lhs_line
        ))));
    }

    if !n.is_multiple_of(out_line) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "n must be divisible by out line size, got n={n:?}, out_line_size={out_line:?}",
        ))));
    }

    Ok(tile_config)
}

fn check_availability<R: Runtime>(
    tile_config: PlaneVecMatInnerProductConfig,
    client: &ComputeClient<R>,
    dtypes: &MatmulElems,
) -> Result<PlaneVecMatInnerProductConfig, MatmulSetupError> {
    if !client.properties().features.plane.contains(Plane::Ops) {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::PlaneOpsUnavailable,
        ));
    }

    let lhs = *dtypes.lhs_register;
    let rhs = *dtypes.rhs_register;
    let acc = *dtypes.acc_register;

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
