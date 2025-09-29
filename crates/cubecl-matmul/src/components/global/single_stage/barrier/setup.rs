use std::marker::PhantomData;

use crate::components::global::single_stage::barrier::SimpleBarrierConfig;
use crate::components::global::single_stage::barrier::matmul::SimpleBarrierMatmul;
use crate::components::global::{GlobalWriterFamily, read::AsyncFullLoadingStrategy};
use crate::components::stage::StageConfig;
use crate::components::{MatmulLineSizes, stage::NoTilingLayout};
use crate::components::{MatmulPrecision, stage::StridedStageFamily};
use crate::components::{MatmulProblem, global::GlobalMatmulFamily, stage};
use crate::components::{MatmulSelection, stage::FilledStageFamily};
use crate::components::{error::MatmulSetupError, global::WriteTiling};
use cubecl_core::{Runtime, client::ComputeClient};

/// Simple Barrier matmul family for any precision
pub struct SimpleBarrierMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _writer: PhantomData<GW>,
}

impl<SMM, LL, RL, GW> GlobalMatmulFamily for SimpleBarrierMatmulFamily<SMM, LL, RL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = FilledStageFamily,
            OutStage = GW::Stage,
        >,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = SimpleBarrierMatmul<
        MP,
        SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout, NoTilingLayout, WriteTiling>,
        LL,
        RL,
        GW::Writer<MP::Acc>,
    >;
    type Config = SimpleBarrierConfig<SMM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
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

        SimpleBarrierConfig::new::<LL, RL, R>(
            client,
            stage_config,
            num_planes,
            !(problem.m as u32).is_multiple_of(stage_shape_m),
            !(problem.n as u32).is_multiple_of(stage_shape_n),
            !(problem.k as u32).is_multiple_of(stage_shape_k),
            stage_shape_k,
            selection.loading_precompute_strategy,
            selection.reader_mode,
        )
    }
}
