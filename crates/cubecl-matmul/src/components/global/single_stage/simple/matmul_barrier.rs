use std::marker::PhantomData;

use crate::components::InputIdent;
use crate::components::LoadSpecializationConfig;
use crate::components::MatmulPrecision;
use crate::components::global::GlobalMatmul;
use crate::components::global::Quantization;
use crate::components::global::ZeroAccumulatorLoader;
use crate::components::global::load::AsyncFullLoadingStrategy;
use crate::components::global::load::AsyncLoader;
use crate::components::global::single_stage::Config;
use crate::components::problem::MatmulLineSizes;
use crate::components::stage::StageConfig;
use crate::components::stage::StageMatmul;
use crate::components::stage::{FullReaderFamily, FullStageToTileReader};
use crate::kernels::matmul::GlobalInput;
use crate::kernels::matmul::MatmulSelection;
use crate::{
    components::{
        Ident, InvalidConfigError, MatmulConfigFactory, MatmulProblem,
        global::{GlobalConfig, GlobalMatmulFamily},
        stage,
    },
    kernels::MatmulAvailabilityError,
};
use barrier::Barrier;
use cubecl_core::Feature;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_core::{CubeCount, CubeDim, Runtime, client::ComputeClient};
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::ReadWrite;
use cubecl_std::tensor::r#virtual::VirtualTensor;

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
    SMM: stage::StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        SimpleBarrierMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;

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
                "Error: Specialization is unavailable for simple barrier matmul.",
            ))
        }
    }
}

impl<SMM, LL, RL> MatmulConfigFactory for SimpleBarrierMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
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
        SMM::check_availability::<R, MP>(client, &config.stage_config())?;

        if !client.properties().feature_enabled(Feature::Barrier) {
            return Err(MatmulAvailabilityError::BarrierUnavailable);
        }

        Ok(())
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
pub struct SimpleBarrierMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP>,
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
            MP,
            LhsReader = FullStageToTileReader<MP::ES, LL::TilingLayout>,
            RhsReader = FullStageToTileReader<MP::ES, RL::TilingLayout>,
        >,
    LL: AsyncFullLoadingStrategy,
    RL: AsyncFullLoadingStrategy,
{
    type Config = Config<SMM::Config>;
    type LhsLoader = AsyncLoader<MP, Barrier<MP::ES>, SMM::Config, LL>;
    type RhsLoader = AsyncLoader<MP, Barrier<MP::ES>, SMM::Config, RL>;
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

        let barrier_level = LL::barrier_level();
        let barrier = Barrier::<MP::ES>::new(barrier_level);

        for loop_iter in 0..num_loops {
            sync_cube();

            #[allow(clippy::collapsible_if)]
            if comptime!(config.check_k_bounds()) {
                if loop_iter == num_loops - 1 {
                    Self::LhsLoader::clear_stage(&mut lhs_loader, config);
                    Self::RhsLoader::clear_stage(&mut rhs_loader, config);
                    sync_cube();
                }
            }

            // Start loading
            Self::LhsLoader::fill_stage(&mut lhs_loader, &barrier, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &barrier, config);

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader);

            barrier.arrive_and_wait();

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
        Self::LhsLoader::new::<Self::Config>(
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
        Self::RhsLoader::new::<Self::Config>(
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
