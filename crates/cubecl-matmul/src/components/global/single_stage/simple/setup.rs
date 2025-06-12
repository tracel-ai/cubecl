use crate::{
    components::{
        LoadSpecializationConfig, MatmulChecker, MatmulPrecision,
        global::{
            load::SyncFullLoadingStrategy,
            single_stage::{SingleStageConfig, simple::matmul::SimpleMatmul},
        },
        problem::MatmulLineSizes,
        stage::StageConfig,
    },
    kernels::matmul::{GlobalInput, MatmulSelection},
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

    fn cube_dim(
        selection: &MatmulSelection,
        load_specialization: LoadSpecializationConfig,
    ) -> Result<CubeDim, InvalidConfigError> {
        let main_flow_planes = SMM::computation_resources(&selection.tiling_scheme)?
            .as_plane_resources(selection.plane_dim)?
            .get_count();

        if let LoadSpecializationConfig::None = load_specialization {
            Ok(CubeDim::new_2d(selection.plane_dim, main_flow_planes))
        } else {
            Err(Box::new(
                "Error: Specialization is unavailable for simple matmul.",
            ))
        }
    }

    fn setup(
        input: Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        quantized: bool,
    ) -> Self::Config {
        let stage_config = SMM::setup(input.stage_input, problem, line_sizes, cube_dim, quantized);
        let stage_shape_m = stage_config.tiling_scheme().elements_in_stage_m();
        let stage_shape_n = stage_config.tiling_scheme().elements_in_stage_n();
        let stage_shape_k = stage_config.tiling_scheme().elements_in_stage_k();

        SingleStageConfig::new(
            stage_config,
            problem.m as u32 % stage_shape_m != 0,
            problem.n as u32 % stage_shape_n != 0,
            problem.k as u32 % stage_shape_k != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            line_sizes.lhs as u32,
            line_sizes.rhs as u32,
            line_sizes.out as u32,
            stage_shape_k,
            input.loading_precompute_strategy,
            input.loader_mode,
        )
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
