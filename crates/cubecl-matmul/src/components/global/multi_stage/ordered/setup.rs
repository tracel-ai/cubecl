use crate::components::global::{GlobalReaderConfig, SharedGlobalConfig, cube_dim_validation};
use crate::components::stage::StageConfig;
use crate::components::{
    MatmulElems,
    global::{
        GlobalWriterFamily,
        read::{FullLoadingStrategy, PartialLoadingStrategy, sync::Synchronous},
    },
};
use crate::components::{MatmulIdent, MatmulLineSizes, MatmulSelection};
use crate::components::{MatmulPrecision, MatmulProblem, stage};
use crate::components::{
    TilingScheme,
    global::{
        WriteTiling,
        multi_stage::ordered::{LL, OrderedDoubleBufferingMatmul},
        read::LoadingValidation,
    },
};
use crate::components::{error::MatmulSetupError, stage::StridedStageFamily};
use crate::components::{global::GlobalMatmulFamily, stage::FilledStageFamily};
use crate::components::{global::MaxGlobalReaderPlanes, stage::NoTilingLayout};
use cubecl_core::prelude::*;
use std::marker::PhantomData;

/// Ordered double buffering matmul family for any precision
pub struct OrderedDoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    RL: PartialLoadingStrategy,
    GW: GlobalWriterFamily,
> {
    _stage_matmul: PhantomData<SMM>,
    _rhs_loading: PhantomData<RL>,
    _writer: PhantomData<GW>,
}

impl<SMM, RL, GW> GlobalMatmulFamily for OrderedDoubleBufferingMatmulFamily<SMM, RL, GW>
where
    SMM: stage::StageMatmulFamily<
            LhsStage = StridedStageFamily,
            RhsStage = StridedStageFamily,
            AccStage = FilledStageFamily,
            OutStage = GW::Stage,
        >,
    RL: PartialLoadingStrategy<SyncStrategy = Synchronous>,
    GW: GlobalWriterFamily,
{
    type Matmul<MP: MatmulPrecision> = OrderedDoubleBufferingMatmul<
        MP,
        SMM::Matmul<
            MP,
            <LL as FullLoadingStrategy>::TilingLayout,
            RL::TilingLayout,
            NoTilingLayout,
            WriteTiling,
        >,
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
            (1, 2).into(),
            max_global_readers,
            true,
            dtypes,
        )?;

        let stage_shape_m = stage_config.elements_in_stage_m();
        let stage_shape_n = stage_config.elements_in_stage_n();
        let stage_shape_k = stage_config.elements_in_stage_k();

        let num_planes = stage_config.plane_role_config().plane_roles.total_count();

        let config = SharedGlobalConfig {
            stage_config,
            num_planes,
            lhs_reader_config: GlobalReaderConfig {
                gmem_config: todo!(),
                smem_config: todo!(),
                precompute_job: selection.loading_precompute_strategy.into(),
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
                precompute_job: selection.loading_precompute_strategy.into(),
                plane_dim: todo!(),
                loading_planes_count: todo!(),
                plane_role_config: todo!(),
                reader_mode: todo!(),
                stage_ident: todo!(),
                event_loading_mode: todo!(),
                specialization_tensor_config: todo!(),
            },
        };
        // !(problem.m as u32).is_multiple_of(stage_shape_m),
        // !(problem.n as u32).is_multiple_of(stage_shape_n),
        // !(problem.k as u32).is_multiple_of(2 * stage_shape_k),
        // selection.loading_precompute_strategy,
        // selection.reader_mode,
        // selection.load_specialization_config.into(),

        validate::<LL, RL, SMM::Config, R>(config, client, selection.tiling_scheme)
    }
}

fn validate<LL: LoadingValidation, RL: LoadingValidation, S: StageConfig, R: Runtime>(
    config: SharedGlobalConfig<S>,
    client: &ComputeClient<R::Server>,
    tiling_scheme: TilingScheme,
) -> Result<SharedGlobalConfig<S>, MatmulSetupError> {
    LL::check::<R>(client, &config.lhs_reader_config)?;
    RL::check::<R>(client, &config.rhs_reader_config)?;
    cube_dim_validation(config.cube_dim())?;

    if tiling_scheme.stage_partitions_in_stage_n() > 1 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Ordered does not support number of stage partitions > 1 in n",
        )));
    }

    if config
        .lhs_reader_config
        .specialization_tensor_config
        .has_specialization()
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Error: In Ordered lhs loading cannot be outside of main flow",
        )));
    }

    Ok(config)
}
