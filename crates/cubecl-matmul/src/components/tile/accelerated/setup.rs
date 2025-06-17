use crate::components::resource::ComputeResources;
use crate::components::tile::TileMatmulFamily;
use crate::components::tile::accelerated::config::AcceleratedConfig;
use crate::components::tile::accelerated::matmul::AcceleratedMatmul;
use crate::components::{InvalidConfigError, MatmulLineSizes, MatmulPrecision, MatmulProblem};
use crate::kernels::MatmulSetupError;
use crate::kernels::matmul::{MatmulSelection, MultiRowStrategy, plane_matmul_selection};
use cubecl_core::ir::Elem;
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
        matmul_line_sizes: MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let (lhs_global_line_size, rhs_global_line_size, out_global_line_size) =
            matmul_line_sizes.into();

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
            lhs_stage_line_size,
            rhs_stage_line_size,
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
