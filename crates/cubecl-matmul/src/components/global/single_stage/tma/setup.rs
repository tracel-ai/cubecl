use crate::components::MatmulPrecision;
use crate::components::MatmulSelection;
use crate::components::error::MatmulSetupError;
use crate::components::global::load::NoLoadingValidation;
use crate::components::global::load::TmaTiling;
use crate::components::global::single_stage::tma::SimpleTmaConfig;
use crate::components::global::single_stage::tma::matmul::SimpleTmaMatmul;
use crate::components::stage::StageConfig;
use crate::components::{AvailableLineSizes, stage::NoTilingLayout};
use crate::components::{MatmulLineSizes, stage::FillStageReaderFamily};
use std::marker::PhantomData;

use cubecl_core::Runtime;
use cubecl_core::client::ComputeClient;
use cubecl_std::tensor::layout::Coords3d;

use crate::components::{
    MatmulProblem,
    global::GlobalMatmulFamily,
    stage::{self, FullStageReaderFamily},
};

/// Simple TMA matmul family for any precision
pub struct SimpleTmaMatmulFamily<SMM: stage::StageMatmulFamily> {
    _stage_matmul: PhantomData<SMM>,
}

impl<SMM> GlobalMatmulFamily for SimpleTmaMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily<
            LhsStageReader = FullStageReaderFamily,
            RhsStageReader = FullStageReaderFamily,
            AccStageReader = FillStageReaderFamily,
            WriteCoords = Coords3d,
        >,
{
    type Matmul<MP: MatmulPrecision> =
        SimpleTmaMatmul<MP, SMM::Matmul<MP, TmaTiling, TmaTiling, NoTilingLayout>>;
    type Config = SimpleTmaConfig<SMM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        assert!(line_sizes.lhs == 1);
        assert!(line_sizes.rhs == 1);

        let stage_config = SMM::setup::<MP, R>(
            client,
            problem,
            selection,
            line_sizes,
            (1, 1).into(),
            None,
            false,
        )?;

        let stage_shape_m = stage_config.tiling_scheme().elements_in_stage_m();
        let stage_shape_n = stage_config.tiling_scheme().elements_in_stage_n();
        let stage_shape_k = stage_config.tiling_scheme().elements_in_stage_k();

        let num_planes = if !selection.load_specialization_config.has_specialization() {
            stage_config.num_main_flow_planes()
        } else {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Error: Specialization is unavailable for simple tma matmul.",
            )));
        };

        SimpleTmaConfig::new::<NoLoadingValidation, NoLoadingValidation, MP, R>(
            client,
            stage_config,
            num_planes,
            !(problem.m as u32).is_multiple_of(stage_shape_m),
            !(problem.n as u32).is_multiple_of(stage_shape_n),
            !(problem.k as u32).is_multiple_of(stage_shape_k),
            stage_shape_k,
            selection.loading_precompute_strategy,
            selection.loader_mode,
        )
    }

    fn filter_line_sizes(available_line_sizes: AvailableLineSizes) -> AvailableLineSizes {
        // We need smem to be unlined so slicing is simpler. TMA doesn't use the vector
        // type anyways and treats it as a void* with the actual type being set by the `TensorMap`
        available_line_sizes
            .filter_lhs(|ls| *ls == 1)
            .filter_rhs(|ls| *ls == 1)
    }
}
