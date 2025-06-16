use crate::components::resource::ComputeResources;
use crate::components::tile::accelerated::config::AcceleratedConfig;
use crate::components::tile::accelerated::matmul::AcceleratedMatmul;
use crate::components::tile::{TileConfig, TileMatmulFamily};
use crate::components::{
    AvailableLineSizes, InvalidConfigError, MatmulPrecision, MatmulProblem, TileSize,
};
use crate::kernels::matmul::{MatmulSelection, MultiRowStrategy, plane_matmul_selection};
use crate::kernels::{MatmulAvailabilityError, MatmulSetupError};
use cubecl_core::Feature;
use cubecl_core::ir::{Elem, FloatKind};
use cubecl_core::prelude::*;

impl TileMatmulFamily for AcceleratedMatmul {
    type Matmul<MP: MatmulPrecision> = AcceleratedMatmul;
    type Config = AcceleratedConfig;

    fn requires_tensor_cores() -> bool {
        true
    }

    fn computation_resources() -> Result<ComputeResources, InvalidConfigError> {
        Ok(ComputeResources::Planes(1))
    }

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: AvailableLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let lhs_global_line_size = available_line_sizes.maximize_lhs(problem, None)?;
        let rhs_global_line_size = available_line_sizes.maximize_rhs(problem, None)?;
        let out_global_line_size = available_line_sizes.maximize_out(problem, None)?;

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

        AcceleratedConfig::new::<MP, R>(
            client,
            selection.tiling_scheme,
            selection.plane_dim,
            problem.lhs_layout,
            problem.rhs_layout,
            stage_line_size_update,
            lhs_global_line_size as u32,
            rhs_global_line_size as u32,
            out_global_line_size as u32,
            lhs_stage_line_size as u32,
            rhs_stage_line_size as u32,
        )
    }

    fn selection<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        plane_dim: u32,
        elem_stage: Elem,
        elem_acc: Elem,
    ) -> MatmulSelection {
        plane_matmul_selection::<Self, R>(
            client,
            problem,
            plane_dim,
            MultiRowStrategy::Adaptive {
                minimum_stage_count: 8,
            },
            elem_stage,
            elem_acc,
        )
    }
}
