use crate::matmul::components::{
    MatmulPrecision,
    global::{
        GlobalMatmul, IndexedQuantization, ZeroAccumulatorLoader,
        output_loader::Unloader,
        single_stage::{
            Config, FullLoader, SyncFullLhsLoader, SyncFullLoader, SyncFullLoadingStrategy,
            SyncFullRhsLoader,
        },
    },
    stage::{
        StageMatmul,
        multi_buffer::{LhsReader, RhsReader},
    },
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use cubecl_std::{CubeOption, CubeOptionExpand};
use std::marker::PhantomData;

use crate::matmul::{
    components::{
        Ident, InvalidConfigError, MatmulConfigFactory, MatmulProblem,
        global::{GlobalConfig, GlobalMatmulFamily},
        stage::{
            self,
            multi_buffer::{LhsReaderFamily, RhsReaderFamily},
        },
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
    SMM: stage::StageMatmulFamily<LhsReader = LhsReaderFamily, RhsReader = RhsReaderFamily>,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> = SimpleMatmul<
        MP,
        SMM::Matmul<MP::ES, MP::EG, MP::EA, LL::TilingLayout, RL::TilingLayout>,
        LL,
        RL,
    >;
}

impl<SMM, LL, RL> MatmulConfigFactory for SimpleMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
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
        SMM::check_availability::<R, MP>(client, &config.to_smm_config())
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
pub struct SimpleMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP::ES, MP::EG, MP::EA>,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL> GlobalMatmul<MP> for SimpleMatmul<MP, SMM, LL, RL>
where
    SMM: StageMatmul<
            MP::ES,
            MP::EG,
            MP::EA,
            LhsReader = LhsReader<MP::ES, LL::TilingLayout>,
            RhsReader = RhsReader<MP::ES, RL::TilingLayout>,
        >,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type Config = Config<SMM::Config>;
    type LhsLoader = SyncFullLhsLoader<MP::EG, MP::ES, SMM::Config, LL>;
    type RhsLoader = SyncFullRhsLoader<MP::EG, MP::ES, SMM::Config, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Out = Unloader<MP::EG>;
    type Accumulator = SMM::Accumulator;

    #[allow(clippy::clone_on_copy, clippy::manual_map, clippy::single_match)]
    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        quantization: CubeOption<IndexedQuantization<MP::EG>>,
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());
        SMM::zero_accumulator(acc, config.to_smm_config());

        for _ in 0..num_loops {
            sync_units();

            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config);

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader);

            sync_units();

            let scaling = match quantization.clone() {
                CubeOption::Some(quantization) => CubeOption::new_Some(
                    quantization.read_current_scale_lhs(config.global_line_size(Ident::Lhs))
                        * quantization.read_current_scale_rhs(config.global_line_size(Ident::Rhs)),
                ),
                CubeOption::None => CubeOption::new_None(),
            };

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                scaling,
                config.to_smm_config(),
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);

            match quantization.clone() {
                CubeOption::Some(mut quantization) => quantization.advance_indices(),
                CubeOption::None => {}
            }
        }

        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_unloader,
            quantization,
            config.to_smm_config(),
            config,
        );
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new::<Self::Config>(lhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new::<Self::Config>(rhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_unloader(
        out: VirtualTensor<MP::EG, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
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
