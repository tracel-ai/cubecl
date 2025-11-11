use crate::components::tile::plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig;
use crate::components::tile::plane_vec_mat_inner_product::matmul::PlaneVecMatInnerProduct;
use crate::components::tile::{TileMatmulFamily, io::Strided};
use crate::components::{
    InvalidConfigError, MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSelection,
};
use crate::components::{error::MatmulSetupError, tile::io::TileKind};
use crate::components::{
    resource::ComputeResources,
    tile::plane_vec_mat_inner_product::reader::{MatrixFragmentReader, MatrixStageReader},
};
use cubecl_core::prelude::*;

impl<Kind: TileKind> TileMatmulFamily for PlaneVecMatInnerProduct<Kind>
where
    MatrixStageReader<Kind>: MatrixFragmentReader<TileKind = Kind>,
{
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = PlaneVecMatInnerProduct<Kind>;
    type Config = PlaneVecMatInnerProductConfig;

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
    ) -> Result<Self::Config, MatmulSetupError> {
        PlaneVecMatInnerProductConfig::new::<R>(
            client,
            selection.tiling_scheme,
            selection.plane_dim,
            problem.lhs_layout,
            problem.rhs_layout,
            selection.swizzling,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
            matmul_line_sizes.out as u32,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
            dtypes,
        )
    }
}
