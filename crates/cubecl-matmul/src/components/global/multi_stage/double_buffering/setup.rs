use crate::components::MatmulLineSizes;
use crate::components::global::GlobalConfig;
use crate::components::global::load::SyncBufferLoadingStrategy;
use crate::components::global::multi_stage::double_buffering::{
    DoubleBufferingGlobalConfig, DoubleBufferingMatmul,
};
use crate::components::stage::StageConfig;
use crate::components::{
    Ident, InvalidConfigError, LoadSpecializationConfig, MatmulChecker, MatmulPrecision,
    MatmulProblem, stage,
};
use crate::components::{global::GlobalMatmulFamily, stage::BufferReaderFamily};
use crate::kernels::MatmulAvailabilityError;
use crate::kernels::matmul::{GlobalInput, MatmulSelection};
use cubecl_core::prelude::*;
use std::marker::PhantomData;

pub struct DoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for DoubleBufferingMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = BufferReaderFamily, RhsReader = BufferReaderFamily>,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        DoubleBufferingMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;
    type Input = GlobalInput<SMM::Input>;

    fn cube_dim(
        selection: &MatmulSelection,
        load_specialization: LoadSpecializationConfig,
    ) -> Result<CubeDim, InvalidConfigError> {
        let main_flow_planes = SMM::computation_resources(&selection.tiling_scheme)?
            .as_plane_resources(selection.plane_dim)?
            .get_count();
        Ok(CubeDim::new_2d(
            selection.plane_dim,
            load_specialization
                .to_plane_roles(main_flow_planes)
                .total_count(),
        ))
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

        DoubleBufferingGlobalConfig::new(
            stage_config,
            problem.m as u32 % stage_shape_m != 0,
            problem.n as u32 % stage_shape_n != 0,
            problem.k as u32 % (2 * stage_shape_k) != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            line_sizes.lhs as u32,
            line_sizes.rhs as u32,
            line_sizes.out as u32,
            cube_dim.y,
            input.loading_precompute_strategy,
            input.loader_mode,
        )
    }
}

impl<SMM, LL, RL> MatmulChecker for DoubleBufferingMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        LL::check::<Self::Config>(config, Ident::Lhs)?;
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
