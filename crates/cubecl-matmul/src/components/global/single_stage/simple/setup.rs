use crate::components::{
    MatmulLineSizes, MatmulPrecision, MatmulSelection,
    error::MatmulSetupError,
    global::{
        read::SyncFullLoadingStrategy,
        single_stage::simple::{SimpleConfig, matmul::SimpleMatmul},
    },
    stage::{FillStageReaderFamily, NoTilingLayout, StageConfig},
};
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;
use std::marker::PhantomData;

use crate::components::{
    MatmulProblem,
    global::GlobalMatmulFamily,
    stage::{self, FullStageReaderFamily},
};

/// Simple matmul family for any precision
pub struct SimpleMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for SimpleMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<
            LhsStageReader = FullStageReaderFamily,
            RhsStageReader = FullStageReaderFamily,
            AccStageReader = FillStageReaderFamily,
            WriteCoords = Coords2d,
        >,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> = SimpleMatmul<
        MP,
        SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout, NoTilingLayout>,
        LL,
        RL,
    >;
    type Config = SimpleConfig<SMM::Config>;

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

        SimpleConfig::new::<LL, RL, MP, R>(
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
