use crate::components::error::MatmulSetupError;
use crate::components::global::MaxLoaderPlanes;
use crate::components::global::load::SyncPartialLoadingStrategy;
use crate::components::global::multi_stage::double_buffering::{
    DoubleBufferingGlobalConfig, DoubleBufferingMatmul,
};
use crate::components::stage::StageConfig;
use crate::components::{MatmulLineSizes, MatmulSelection};
use crate::components::{MatmulPrecision, MatmulProblem, stage};
use crate::components::{global::GlobalMatmulFamily, stage::PartialReaderFamily};
use cubecl_core::prelude::*;
use std::marker::PhantomData;

/// Double buffering matmul family for any precision
pub struct DoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: SyncPartialLoadingStrategy,
    RL: SyncPartialLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for DoubleBufferingMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = PartialReaderFamily, RhsReader = PartialReaderFamily>,
    LL: SyncPartialLoadingStrategy,
    RL: SyncPartialLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        DoubleBufferingMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;

    fn setup<MP: MatmulPrecision, R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        line_sizes: &MatmulLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        let max_loaders = selection
            .load_specialization_config
            .has_specialization()
            .then(|| {
                MaxLoaderPlanes::new::<LL, RL>(
                    &selection.tiling_scheme,
                    line_sizes,
                    selection.plane_dim,
                )
            });

        let stage_config = SMM::setup::<MP, R>(
            client,
            problem,
            selection,
            line_sizes,
            (2, 2).into(),
            max_loaders,
            false,
        )?;

        let stage_shape_m = stage_config.tiling_scheme().elements_in_stage_m();
        let stage_shape_n = stage_config.tiling_scheme().elements_in_stage_n();
        let stage_shape_k = stage_config.tiling_scheme().elements_in_stage_k();

        let num_planes = stage_config.plane_role_config().plane_roles.total_count();

        DoubleBufferingGlobalConfig::new::<LL, RL, MP, R>(
            client,
            stage_config,
            num_planes,
            problem.m as u32 % stage_shape_m != 0,
            problem.n as u32 % stage_shape_n != 0,
            problem.k as u32 % (2 * stage_shape_k) != 0,
            selection.loading_precompute_strategy,
            selection.loader_mode,
            selection.load_specialization_config.into(),
        )
    }
}
