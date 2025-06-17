use std::fmt::Display;

use crate::components::resource::ComputeResources;
use crate::components::tile::TileMatmulFamily;
use crate::components::tile::register::config::RegisterConfig;
use crate::components::tile::register::matmul::RegisterMatmul;
use crate::components::{
    AvailableLineSizes, InvalidConfigError, MatmulPrecision, MatmulProblem, MatrixLayout,
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
        available_line_sizes: AvailableLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let max_lhs = match problem.lhs_layout {
            MatrixLayout::RowMajor => selection.tiling_scheme.elements_in_tile_k(),
            MatrixLayout::ColMajor => selection.tiling_scheme.elements_in_tile_m(),
        } as u8;
        let max_rhs = match problem.rhs_layout {
            MatrixLayout::RowMajor => selection.tiling_scheme.elements_in_tile_n(),
            MatrixLayout::ColMajor => selection.tiling_scheme.elements_in_tile_k(),
        } as u8;
        let max_out = selection.tiling_scheme.elements_in_tile_n() as u8;

        let lhs_global_line_size = available_line_sizes.maximize_lhs(problem, Some(max_lhs))?;
        let rhs_global_line_size = available_line_sizes.maximize_rhs(problem, Some(max_rhs))?;
        let out_global_line_size = available_line_sizes.maximize_out(problem, Some(max_out))?;

        let stage_vectorization = selection.stage_vectorization;

        let (lhs_stage_line_size, rhs_stage_line_size, stage_line_size_update) =
            if stage_vectorization.stage_line_size == 0 {
                (
                    lhs_global_line_size as u32,
                    rhs_global_line_size as u32,
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
            lhs_global_line_size as u32,
            rhs_global_line_size as u32,
            out_global_line_size as u32,
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
