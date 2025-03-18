use crate::matmul::components::global::base::AsyncInputLoader;
use crate::matmul::components::global::base::InputLoader;
use crate::matmul::components::global::output_loader::Unloader;
use crate::matmul::components::global::single_stage::Config;
use crate::matmul::components::global::single_stage::AsyncFullLoadingStrategy;
use crate::matmul::components::global::GlobalMatmul;
use crate::matmul::components::global::ZeroAccumulatorLoader;
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::StageMatmul;
use crate::matmul::components::MatmulPrecision;

use barrier::Barrier;
use cubecl_core::prelude::*;
use cubecl_core::Feature;
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::r#virtual::ReadWrite;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

use cubecl_core::{client::ComputeClient, CubeCount, CubeDim, Runtime};

use crate::matmul::{
    components::{
        global::{GlobalConfig, GlobalMatmulFamily},
        stage::{
            self,
            multi_buffer::{LhsReaderFamily, RhsReaderFamily},
        },
        Ident, InvalidConfigError, MatmulConfigFactory, MatmulProblem,
    },
    kernels::MatmulAvailabilityError,
};

pub struct SimpleBarrierMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for SimpleBarrierMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = LhsReaderFamily, RhsReader = RhsReaderFamily>,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> = SimpleBarrierMatmul<
        MP,
        SMM::Matmul<MP::ES, MP::EG, MP::EA, LL::TilingLayout, RL::TilingLayout>,
        LL,
        RL,
    >;
}

impl<SMM, LL, RL> MatmulConfigFactory for SimpleBarrierMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
{
    type Input = SMM::Input;
    type Config = Config<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        LL::check(config, Ident::Lhs)?;
        RL::check(config, Ident::Rhs)?;
        SMM::check_config(&config.to_smm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.to_smm_config())?;

        if !client.properties().feature_enabled(Feature::Barrier) {
            return Err(MatmulAvailabilityError::BarrierUnavailable);
        }

        Ok(())
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let smm_config = SMM::make_config(input, problem, cube_dim, cube_count, quantized);
        let stage_shape = SMM::stage_shape(&smm_config);

        Config::new(
            smm_config,
            problem.m as u32 % stage_shape.m != 0,
            problem.n as u32 % stage_shape.n != 0,
            problem.k as u32 % stage_shape.k != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
            stage_shape.k,
        )
    }
}

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct SimpleBarrierMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP::ES, MP::EG, MP::EA>,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL> GlobalMatmul<MP> for SimpleBarrierMatmul<MP, SMM, LL, RL>
where
    SMM: StageMatmul<
        MP::ES,
        MP::EG,
        MP::EA,
        LhsReader = LhsReader<MP::ES, LL::TilingLayout>,
        RhsReader = RhsReader<MP::ES, RL::TilingLayout>,
    >,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
{
    type Config = Config<SMM::Config>;
    type LhsLoader = AsyncLhsLoader<MP::EG, MP::ES, SMM::Config, LL>;
    type RhsLoader = AsyncRhsLoader<MP::EG, MP::ES, SMM::Config, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<MP::EG>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());
        SMM::zero_accumulator(acc, config.to_smm_config());

        let barrier_level = LL::barrier_level();
        comptime!(assert!(barrier_level == RL::barrier_level()));
        let barrier = Barrier::<MP::ES>::new(barrier_level);

        for loop_iter in 0..num_loops {
            sync_units();

            #[allow(clippy::collapsible_if)]
            if comptime!(config.check_k_bounds()) {
                if loop_iter == num_loops - 1 {
                    Self::LhsLoader::clear_stage(&mut lhs_loader, config);
                    Self::RhsLoader::clear_stage(&mut rhs_loader, config);
                    sync_units();
                }
            }

            // Start loading
            Self::LhsLoader::fill_stage::<Barrier<MP::ES>>(&mut lhs_loader, &barrier, config);
            Self::RhsLoader::fill_stage::<Barrier<MP::ES>>(&mut rhs_loader, &barrier, config);

            let lhs_stage_reader = &Self::LhsLoader::as_stage_reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::as_stage_reader(&rhs_loader);

            barrier.wait();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.to_smm_config(),
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_unloader,
            config.to_smm_config(),
            config,
        );
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new::<Self::Config>(lhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new::<Self::Config>(rhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_unloader(
        out: VirtualTensor<MP::EG, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
    ) -> Self::Out {
        Self::Out::new(out, x_offset, y_offset, batch_offset)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.to_smm_config())
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        SMM::zero_accumulator(acc, config.to_smm_config());
    }
}
