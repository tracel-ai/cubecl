use std::fmt::Display;

use crate::components::resource::ComputeResources;
use crate::components::tile::TileMatmulFamily;
use crate::components::tile::register::config::RegisterConfig;
use crate::components::tile::register::matmul::RegisterMatmul;
use crate::components::{
    AvailableLineSizes, InvalidConfigError, MatmulLineSizes, MatmulPrecision, MatmulProblem,
};
use crate::kernels::MatmulSetupError;
use crate::kernels::matmul::{MatmulSelection, unit_matmul_selection};
use cubecl_core::ir::Elem;
use cubecl_core::prelude::*;

impl TileMatmulFamily for RegisterMatmul {
    type Matmul<MP: MatmulPrecision> = RegisterMatmul;
    type Config = RegisterConfig;

    fn requires_tensor_cores() -> bool {
        false
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Units(1))
    }

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        matmul_line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let stage_vectorization = selection.stage_vectorization;
        let (lhs_stage_line_size, rhs_stage_line_size, stage_line_size_update) =
            if stage_vectorization.stage_line_size == 0 {
                (
                    matmul_line_sizes.lhs as u32,
                    matmul_line_sizes.rhs as u32,
                    false,
                )
            } else {
                (
                    stage_vectorization.stage_line_size as u32,
                    stage_vectorization.stage_line_size as u32,
                    true,
                )
            };

        RegisterConfig::new::<MP, R>(
            client,
            selection.tiling_scheme,
            selection.plane_dim,
            problem.lhs_layout,
            problem.rhs_layout,
            stage_line_size_update,
            matmul_line_sizes.lhs as u32,
            matmul_line_sizes.rhs as u32,
            matmul_line_sizes.out as u32,
            lhs_stage_line_size,
            rhs_stage_line_size,
        )
    }

    fn selection<R: Runtime>(
        _client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        _elem_stage: Elem,
        _elem_acc: Elem,
    ) -> MatmulSelection {
        unit_matmul_selection(problem, plane_dim)
    }

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        available_line_sizes
            .filter_lhs(|ls| *ls <= 4)
            .filter_rhs(|ls| *ls <= 4)
    }
}

pub struct RegisterMatmulConfigError {
    func: Box<dyn Fn() -> String>,
}

impl Display for RegisterMatmulConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = (self.func)();
        write!(f, "{string}")
    }
}
