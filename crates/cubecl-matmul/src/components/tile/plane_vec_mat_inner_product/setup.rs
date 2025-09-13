use crate::components::tile::plane_vec_mat_inner_product::config::PlaneVecMatInnerProductConfig;
use crate::components::tile::plane_vec_mat_inner_product::matmul::PlaneVecMatInnerProduct;
use crate::components::tile::{TileMatmulFamily, loader::Strided};
use crate::components::{InvalidConfigError, MatmulLineSizes, MatmulProblem, MatmulSelection};
use crate::components::{error::MatmulSetupError, tile::loader::TileKind};
use crate::components::{
    resource::ComputeResources,
    tile::plane_vec_mat_inner_product::loader::{MatrixLoader, TileMatrixLoader},
};
use cubecl_core::prelude::*;

impl<Kind: TileKind> TileMatmulFamily for PlaneVecMatInnerProduct<Kind>
where
    MatrixLoader<Kind>: TileMatrixLoader<TileKind = Kind>,
{
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = PlaneVecMatInnerProduct<Kind>;
    type Config = PlaneVecMatInnerProductConfig;

    type LhsTile = Strided;
    type RhsTile = Strided;
    type AccTile = Kind;

    fn requires_accelerator() -> bool {
        false
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        PlaneVecMatInnerProductConfig::new::<Lhs, Rhs, Acc, R>(
            client,
            selection.tiling_scheme,
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
