use crate::components::tile::register::config::RegisterConfig;
use crate::components::tile::register::matmul::RegisterMatmul;
use crate::components::tile::{TileMatmulFamily, io::Strided};
use crate::components::{
    AvailableLineSizes, InvalidConfigError, MatmulLineSizes, MatmulProblem, MatmulSelection,
};
use crate::components::{error::MatmulSetupError, tile::io::TileKind};
use crate::components::{
    resource::ComputeResources,
    tile::register::reader::{RegisterFragmentReader, RegisterStageReader},
};
use cubecl_core::prelude::*;

impl<AccTile: TileKind> TileMatmulFamily for RegisterMatmul<AccTile>
where
    RegisterStageReader<AccTile>: RegisterFragmentReader<TileKind = AccTile>,
{
    type Matmul<L: Numeric, R: Numeric, A: Numeric> = RegisterMatmul<AccTile>;
    type Config = RegisterConfig;

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

    fn setup<Lhs: Numeric, Rhs: Numeric, Acc: Numeric, R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        RegisterConfig::new::<Lhs, Rhs, Acc, R>(
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

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
        // .filter_lhs(|ls| *ls <= 4)
        // .filter_rhs(|ls| *ls <= 4)
        // .filter_out(|ls| *ls <= 4)
    }
}
