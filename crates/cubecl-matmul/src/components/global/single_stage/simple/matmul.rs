use crate::{
    components::{
        InputIdent, MatmulPrecision,
        global::{
            GlobalMatmul, Quantization, ZeroAccumulatorLoader,
            load::{SyncFullLoader, SyncFullLoadingStrategy},
            single_stage::Config,
        },
        problem::MatmulLineSizes,
        stage::{FullStageToTileReader, StageConfig, StageMatmul},
    },
    kernels::matmul::GlobalInput,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use std::marker::PhantomData;

use crate::{
    components::{
        Ident, InvalidConfigError, MatmulConfigFactory, MatmulProblem,
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
}

impl<SMM, LL, RL> MatmulConfigFactory for SimpleMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type Input = GlobalInput<SMM::Input>;
    type Config = Config<SMM::Config>;

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

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let stage_config = SMM::make_config(
            input.stage_input,
            problem,
            line_sizes,
            cube_dim,
            cube_count,
            quantized,
        );
        let stage_shape_m = stage_config.tiling_scheme().elements_in_stage_m();
        let stage_shape_n = stage_config.tiling_scheme().elements_in_stage_n();
        let stage_shape_k = stage_config.tiling_scheme().elements_in_stage_k();

        Config::new(
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

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct SimpleMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP>,
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
            MP,
            LhsReader = FullStageToTileReader<MP::ES, LL::TilingLayout>,
            RhsReader = FullStageToTileReader<MP::ES, RL::TilingLayout>,
        >,
    LL: SyncFullLoadingStrategy,
    RL: SyncFullLoadingStrategy,
{
    type Config = Config<SMM::Config>;
    type LhsLoader = SyncFullLoader<MP, Self::Config, LL>;
    type RhsLoader = SyncFullLoader<MP, Self::Config, RL>;
    type AccumulatorLoader = ZeroAccumulatorLoader;
    type Writer = SMM::Writer;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut out_writer: Self::Writer,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        SMM::zero_accumulator(acc, config.stage_config());

        let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader);
        let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader);

        for _ in 0..num_loops {
            sync_cube();

            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config);

            sync_cube();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.stage_config(),
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        SMM::write_results::<Self::Config>(acc, &mut out_writer, config.stage_config(), config);
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(
            lhs,
            x_offset,
            y_offset,
            batch_offset,
            quantization,
            InputIdent::Lhs,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new(
            rhs,
            x_offset,
            y_offset,
            batch_offset,
            quantization,
            InputIdent::Rhs,
            config,
        )
    }

    fn init_writer(
        out: VirtualTensor<MP::EO, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
        _nth_batch: u32,
        batch_offset: u32,
    ) -> Self::Writer {
        SMM::init_writer(out, x_offset, y_offset, batch_offset)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.stage_config())
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        SMM::zero_accumulator(acc, config.stage_config());
    }
}
