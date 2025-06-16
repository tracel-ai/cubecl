use std::marker::PhantomData;

use crate::components::AvailableLineSizes;
use crate::components::LoadSpecializationConfig;
use crate::components::MatmulPrecision;
use crate::components::global::load::AsyncFullLoadingStrategy;
use crate::components::global::single_stage::barrier::SimpleBarrierConfig;
use crate::components::global::single_stage::barrier::matmul::SimpleBarrierMatmul;
use crate::components::stage::FullReaderFamily;
use crate::components::stage::StageConfig;
use crate::components::{MatmulProblem, global::GlobalMatmulFamily, stage};
use crate::kernels::MatmulSetupError;
use crate::kernels::matmul::MatmulSelection;
use cubecl_core::{Runtime, client::ComputeClient};

pub struct SimpleBarrierMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for SimpleBarrierMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        SimpleBarrierMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;
    type Config = SimpleBarrierConfig<SMM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: AvailableLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        // TODO inject loader info here
        let stage_config = SMM::setup::<MP, R>(
            client,
            problem,
            selection,
            available_line_sizes,
            (1, 1).into(),
        )?;

        let stage_shape_m = stage_config.tiling_scheme().elements_in_stage_m();
        let stage_shape_n = stage_config.tiling_scheme().elements_in_stage_n();
        let stage_shape_k = stage_config.tiling_scheme().elements_in_stage_k();

        let num_planes =
            if let LoadSpecializationConfig::None = selection.load_specialization_config {
                stage_config.num_main_flow_planes()
            } else {
                return Err(MatmulSetupError::InvalidConfig(Box::new(
                    "Error: Specialization is unavailable for simple tma matmul.",
                )));
            };

        SimpleBarrierConfig::new::<LL, RL, R>(
            client,
            stage_config,
            num_planes,
            problem.m as u32 % stage_shape_m != 0,
            problem.n as u32 % stage_shape_n != 0,
            problem.k as u32 % stage_shape_k != 0,
            stage_shape_k,
            selection.loading_precompute_strategy,
            selection.loader_mode,
        )
    }
}
