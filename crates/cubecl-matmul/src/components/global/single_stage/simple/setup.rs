use crate::components::{
    MatmulElems, MatmulIdent, MatmulLineSizes, MatmulPrecision, MatmulSelection,
    error::MatmulSetupError,
    global::{
        GlobalConfig as _, GlobalReaderConfig, GlobalWriterFamily, SharedGlobalConfig, WriteTiling,
        cube_dim_validation,
        read::{FullLoadingStrategy, LoadingValidation},
        single_stage::simple::matmul::SimpleMatmul,
    },
    stage::{FilledStageFamily, NoTilingLayout, StageConfig, StridedStageFamily},
};
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::components::{
    MatmulProblem,
    global::GlobalMatmulFamily,
    stage::{self},
};

/// Simple matmul family for any precision
pub struct SimpleMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _writer: PhantomData<GW>,
}

impl<SMM, LL, RL, GW> GlobalMatmulFamily for SimpleMatmulFamily<SMM, LL, RL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = FilledStageFamily,
            OutStage = GW::Stage,
        >,
    LL: FullLoadingStrategy,
    RL: FullLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = SimpleMatmul<
        MP,
        SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout, NoTilingLayout, WriteTiling>,
        LL,
        RL,
        GW::Writer<MP::Acc>,
    >;
    type Config = SharedGlobalConfig<SMM::Config>;

    fn setup<R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let stage_config = SMM::setup::<R>(
            client,
            problem,
            selection,
            line_sizes,
            (1, 1).into(),
            None,
            false,
            dtypes,
        )?;

        let stage_shape_m = stage_config.elements_in_stage_m();
        let stage_shape_n = stage_config.elements_in_stage_n();
        let stage_shape_k = stage_config.elements_in_stage_k();

        let num_planes = if !selection.load_specialization_config.has_specialization() {
            stage_config.num_main_flow_planes()
        } else {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "Error: Specialization is unavailable for simple tma matmul.",
            )));
        };

        let config = SharedGlobalConfig {
            stage_config,
            num_planes,
            lhs_reader_config: GlobalReaderConfig {
                gmem_config: todo!(),
                smem_config: todo!(),
                precompute_job: todo!(),
                plane_dim: todo!(),
                loading_planes_count: todo!(),
                plane_role_config: todo!(),
                reader_mode: todo!(),
                stage_ident: todo!(),
                event_loading_mode: todo!(),
                specialization_tensor_config: todo!(),
            },
            rhs_reader_config: GlobalReaderConfig {
                gmem_config: todo!(),
                smem_config: todo!(),
                precompute_job: todo!(),
                plane_dim: todo!(),
                loading_planes_count: todo!(),
                plane_role_config: todo!(),
                reader_mode: todo!(),
                stage_ident: todo!(),
                event_loading_mode: todo!(),
                specialization_tensor_config: todo!(),
            },
        };
        // !(problem.k as u32).is_multiple_of(stage_shape_k),
        // !(problem.m as u32).is_multiple_of(stage_shape_m),
        // !(problem.n as u32).is_multiple_of(stage_shape_n),
        // selection.loading_precompute_strategy,
        // selection.reader_mode,

        validate::<LL, RL, SMM::Config, R>(config, client)
    }
}

fn validate<LL: LoadingValidation, RL: LoadingValidation, S: StageConfig, R: Runtime>(
    config: SharedGlobalConfig<S>,
    client: &ComputeClient<R::Server>,
) -> Result<SharedGlobalConfig<S>, MatmulSetupError> {
    LL::check::<R>(client, &config.lhs_reader_config)?;
    RL::check::<R>(client, &config.rhs_reader_config)?;
    cube_dim_validation(config.cube_dim())?;
    Ok(config)
}
