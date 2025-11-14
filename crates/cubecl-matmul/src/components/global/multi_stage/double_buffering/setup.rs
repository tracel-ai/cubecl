use crate::components::global::read::LoadingValidation;
use crate::components::global::{SharedGlobalConfig, cube_dim_validation};
use crate::components::global::{WriteTiling, read::PartialLoadingStrategy};
use crate::components::stage::StageConfig;
use crate::components::{
    MatmulElems,
    global::{
        GlobalWriterFamily,
        multi_stage::double_buffering::{DoubleBufferingGlobalConfig, DoubleBufferingMatmul},
    },
};
use crate::components::{MatmulIdent, MatmulLineSizes, MatmulSelection};
use crate::components::{MatmulPrecision, MatmulProblem, stage};
use crate::components::{error::MatmulSetupError, stage::StridedStageFamily};
use crate::components::{global::GlobalMatmulFamily, stage::FilledStageFamily};
use crate::components::{global::MaxGlobalReaderPlanes, stage::NoTilingLayout};
use cubecl_core::prelude::*;
use std::marker::PhantomData;

/// Double buffering matmul family for any precision
pub struct DoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: PartialLoadingStrategy,
    RL: PartialLoadingStrategy,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
    _writer: PhantomData<GW>,
}

impl<SMM, LL, RL, GW> GlobalMatmulFamily for DoubleBufferingMatmulFamily<SMM, LL, RL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = FilledStageFamily,
            OutStage = GW::Stage,
        >,
    LL: PartialLoadingStrategy,
    RL: PartialLoadingStrategy<SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = DoubleBufferingMatmul<
        MP,
        SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout, NoTilingLayout, WriteTiling>,
        LL,
        RL,
        GW::Writer<MP::Acc>,
    >;
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;

    fn setup<R: Runtime>(
        client: &ComputeClient<R::Server>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
        dtypes: &MatmulElems,
    ) -> Result<Self::Config, MatmulSetupError> {
        let max_global_readers = selection
            .load_specialization_config
            .has_specialization()
            .then(|| {
                MaxGlobalReaderPlanes::new::<LL, RL>(
                    &selection.tiling_scheme,
                    line_sizes,
                    selection.plane_dim,
                )
            });

        let stage_config = SMM::setup::<R>(
            client,
            problem,
            selection,
            line_sizes,
            (2, 2).into(),
            max_global_readers,
            false,
            dtypes,
        )?;

        let stage_shape_m = stage_config.elements_in_stage_m();
        let stage_shape_n = stage_config.elements_in_stage_n();
        let stage_shape_k = stage_config.elements_in_stage_k();

        let num_planes = stage_config.plane_role_config().plane_roles.total_count();

        let config = DoubleBufferingGlobalConfig::from_shared_global_config(
            SharedGlobalConfig {
                stage_config,
                num_planes,
            },
            !(problem.m as u32).is_multiple_of(stage_shape_m),
            !(problem.n as u32).is_multiple_of(stage_shape_n),
            !(problem.k as u32).is_multiple_of(2 * stage_shape_k),
            selection.loading_precompute_strategy,
            selection.reader_mode,
            selection.load_specialization_config.into(),
        );

        validate::<LL, RL, SMM::Config, R>(config, client)
    }
}

fn validate<LL: LoadingValidation, RL: LoadingValidation, S: StageConfig, R: Runtime>(
    config: DoubleBufferingGlobalConfig<S>,
    client: &ComputeClient<R::Server>,
) -> Result<DoubleBufferingGlobalConfig<S>, MatmulSetupError> {
    LL::check::<DoubleBufferingGlobalConfig<S>, R>(client, &config, MatmulIdent::Lhs)?;
    RL::check::<DoubleBufferingGlobalConfig<S>, R>(client, &config, MatmulIdent::Rhs)?;
    cube_dim_validation(config.shared.cube_dim())?;

    Ok(config)
}
