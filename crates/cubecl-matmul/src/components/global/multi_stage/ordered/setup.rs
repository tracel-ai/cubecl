use crate::components::AvailableLineSizes;
use crate::components::global::GlobalConfig;
use crate::components::global::load::{
    LoadingValidation, SyncBufferLoadingStrategy, SyncFullLoadingStrategy,
};
use crate::components::global::multi_stage::ordered::{LL, OrderedDoubleBufferingMatmul};
use crate::components::stage::FullReaderFamily;
use crate::components::stage::StageConfig;
use crate::components::{
    Ident, InvalidConfigError, LoadSpecializationConfig, MatmulChecker, MatmulPrecision,
    MatmulProblem, stage,
};
use crate::components::{global::GlobalMatmulFamily, stage::BufferReaderFamily};
use crate::kernels::matmul::{GlobalInput, MatmulSelection};
use crate::kernels::{MatmulAvailabilityError, MatmulSetupError};
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::OrderedDoubleBufferingGlobalConfig;

pub struct OrderedDoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    RL: SyncBufferLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, RL> GlobalMatmulFamily for OrderedDoubleBufferingMatmulFamily<SMM, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = BufferReaderFamily>,
    RL: SyncBufferLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> = OrderedDoubleBufferingMatmul<
        MP,
        SMM::Matmul<MP, <LL as SyncFullLoadingStrategy>::TilingLayout, RL::TilingLayout>,
        RL,
    >;
    type Input = GlobalInput<SMM::Input>;

    fn setup(
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: &mut AvailableLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        // TODO inject loader info here
        let stage_config = SMM::setup(problem, selection, available_line_sizes, (1, 2).into())?;

        let stage_shape_m = stage_config.tiling_scheme().elements_in_stage_m();
        let stage_shape_n = stage_config.tiling_scheme().elements_in_stage_n();
        let stage_shape_k = stage_config.tiling_scheme().elements_in_stage_k();

        // num_planes = ..
        // fn cube_dim(
        //     selection: &MatmulSelection,
        //     load_specialization: LoadSpecializationConfig,
        // ) -> Result<CubeDim, InvalidConfigError> {
        //     let main_flow_planes = SMM::computation_resources(&selection.tiling_scheme)?
        //         .as_plane_resources(selection.plane_dim)?
        //         .get_count();
        //     Ok(CubeDim::new_2d(
        //         selection.plane_dim,
        //         load_specialization
        //             .to_plane_roles(main_flow_planes)
        //             .total_count(),
        //     ))
        // }

        Ok(OrderedDoubleBufferingGlobalConfig::new(
            stage_config,
            // num_planes,
            todo!(),
            problem.m as u32 % stage_shape_m != 0,
            problem.n as u32 % stage_shape_n != 0,
            problem.k as u32 % (2 * stage_shape_k) != 0,
            selection.loading_precompute_strategy,
            selection.loader_mode,
        ))
    }
}

impl<SMM, RL> MatmulChecker for OrderedDoubleBufferingMatmulFamily<SMM, RL>
where
    SMM: stage::StageMatmulFamily,
    RL: SyncBufferLoadingStrategy,
{
    type Config = OrderedDoubleBufferingGlobalConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        <LL as LoadingValidation>::check::<Self::Config>(config, Ident::Lhs)?;
        RL::check::<Self::Config>(config, Ident::Rhs)?;
        SMM::check_config(&config.stage_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.stage_config)
    }
}
