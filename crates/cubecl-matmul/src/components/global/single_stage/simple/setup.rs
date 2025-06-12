use crate::{
    components::{
        AvailableLineSizes, LoadSpecializationConfig, MatmulChecker, MatmulPrecision,
        global::{
            load::SyncFullLoadingStrategy,
            single_stage::{SingleStageConfig, simple::matmul::SimpleMatmul},
        },
        stage::StageConfig,
    },
    kernels::{
        MatmulSetupError,
        matmul::{GlobalInput, MatmulSelection},
    },
};
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use crate::{
    components::{
        Ident, InvalidConfigError, MatmulProblem,
        global::{GlobalConfig, GlobalMatmulFamily},
        stage::{self, FullReaderFamily},
    },
    kernels::MatmulAvailabilityError,
};

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
    SMM: stage::StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        SimpleMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;
    type Input = GlobalInput<SMM::Input>;

    fn setup(
        problem: &MatmulProblem,
        selection: &MatmulSelection,
        available_line_sizes: &mut AvailableLineSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        // TODO inject loader info here
        let stage_config = SMM::setup(problem, selection, available_line_sizes, (1, 1).into())?;

        let stage_shape_m = stage_config.tiling_scheme().elements_in_stage_m();
        let stage_shape_n = stage_config.tiling_scheme().elements_in_stage_n();
        let stage_shape_k = stage_config.tiling_scheme().elements_in_stage_k();

        // fn cube_dim(
        //     selection: &MatmulSelection,
        //     load_specialization: LoadSpecializationConfig,
        // ) -> Result<CubeDim, InvalidConfigError> {
        //     let main_flow_planes = SMM::computation_resources(&selection.tiling_scheme)?
        //         .as_plane_resources(selection.plane_dim)?
        //         .get_count();

        //     if let LoadSpecializationConfig::None = load_specialization {
        //         Ok(CubeDim::new_2d(selection.plane_dim, main_flow_planes))
        //     } else {
        //         Err(Box::new(
        //             "Error: Specialization is unavailable for simple matmul.",
        //         ))
        //     }
        // }

        Ok(SingleStageConfig::new(
            stage_config,
            // num_planes,
            todo!(),
            problem.m as u32 % stage_shape_m != 0,
            problem.n as u32 % stage_shape_n != 0,
            problem.k as u32 % stage_shape_k != 0,
            stage_shape_k,
            selection.loading_precompute_strategy,
            selection.loader_mode,
        ))
    }
}

impl<SMM, LL, RL> MatmulChecker for SimpleMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type Config = SingleStageConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        LL::check(config, Ident::Lhs)?;
        RL::check(config, Ident::Rhs)?;
        SMM::check_config(&config.stage_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.stage_config())
    }
}
