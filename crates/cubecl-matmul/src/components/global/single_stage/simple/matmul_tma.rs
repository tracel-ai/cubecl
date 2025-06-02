use crate::components::InputIdent;
use crate::components::MatmulPrecision;
use crate::components::global::ZeroAccumulatorLoader;
use crate::components::global::load::TmaLoader;
use crate::components::global::load::arrive_tma;
use crate::components::global::single_stage::Config;
use crate::components::global::{GlobalMatmul, load::TmaTiling};
use crate::components::global::{Quantization, load::TmaReader};
use crate::components::problem::MatmulLineSizes;
use crate::components::stage::StageConfig;
use crate::components::stage::StageMatmul;
use crate::kernels::matmul::GlobalInput;

use barrier::Barrier;
use cubecl_core::prelude::{barrier::BarrierLevel, *};
use cubecl_core::{self as cubecl};
use cubecl_core::{Feature, TmaFeature};
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, tensor::r#virtual::ReadWrite};
use std::{any::TypeId, marker::PhantomData};

use cubecl_core::{CubeCount, CubeDim, Runtime, client::ComputeClient};

use crate::{
    components::{
        InvalidConfigError, MatmulConfigFactory, MatmulProblem,
        global::{GlobalConfig, GlobalMatmulFamily},
        stage::{self, FullReaderFamily},
    },
    kernels::MatmulAvailabilityError,
};

pub struct SimpleTmaMatmulFamily<SMM: stage::StageMatmulFamily> {
    _stage_matmul: PhantomData<SMM>,
}

impl<SMM> GlobalMatmulFamily for SimpleTmaMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = FullReaderFamily>,
{
    type Matmul<MP: MatmulPrecision> = SimpleTmaMatmul<MP, SMM::Matmul<MP, TmaTiling, TmaTiling>>;
}

impl<SMM> MatmulConfigFactory for SimpleTmaMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily,
{
    type Input = GlobalInput<SMM::Input>;
    type Config = Config<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        SMM::check_config(&config.stage_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.stage_config())?;

        let ei_id = TypeId::of::<MP::EI>();
        let es_id = TypeId::of::<MP::ES>();
        let is_tf32 = ei_id == TypeId::of::<f32>() && es_id == TypeId::of::<tf32>();

        if ei_id != es_id && !is_tf32 {
            return Err(MatmulAvailabilityError::TmaUnavailable);
        }

        let ei_id = TypeId::of::<MP::EI>();
        let es_id = TypeId::of::<MP::ES>();
        let is_tf32 = ei_id == TypeId::of::<f32>() && es_id == TypeId::of::<tf32>();

        if ei_id != es_id && !is_tf32 {
            return Err(MatmulAvailabilityError::TmaUnavailable);
        }

        if !client
            .properties()
            .feature_enabled(Feature::Tma(TmaFeature::Base))
        {
            return Err(MatmulAvailabilityError::TmaUnavailable);
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
        let mut line_sizes = line_sizes.clone();

        // We need smem to be unlined so slicing is simpler. TMA doesn't use the vector
        // type anyways and treats it as a void* with the actual type being set by the `TensorMap`
        line_sizes.lhs = 1;
        line_sizes.rhs = 1;

        let stage_config = SMM::make_config(
            input.stage_input,
            problem,
            &line_sizes,
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
pub struct SimpleTmaMatmul<MP: MatmulPrecision, SMM: StageMatmul<MP>> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> GlobalMatmul<MP> for SimpleTmaMatmul<MP, SMM>
where
    SMM: StageMatmul<MP, LhsReader = TmaReader<MP>, RhsReader = TmaReader<MP>>,
{
    type Config = Config<SMM::Config>;
    type LhsLoader = TmaLoader<MP, SMM::Config>;
    type RhsLoader = TmaLoader<MP, SMM::Config>;
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
        let num_elems_stages = config.tiling_scheme().elements_in_stage_nk()
            + config.tiling_scheme().elements_in_stage_mk();

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());
        SMM::zero_accumulator(acc, config.stage_config());

        let barrier = Barrier::<MP::ES>::new_with_tma_proxy(BarrierLevel::cube_coop(0u32));

        for _ in 0..num_loops {
            sync_cube();

            // Start loading
            Self::LhsLoader::fill_stage(&mut lhs_loader, &barrier, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, &barrier, config);

            arrive_tma::<MP::ES>(&barrier, num_elems_stages);

            barrier.wait();

            let lhs_stage_reader = &Self::LhsLoader::reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::reader(&rhs_loader);

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
        nth_batch: u32,
        _batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new::<Self::Config>(
            lhs.as_tensor_map(),
            x_offset,
            y_offset,
            nth_batch,
            quantization,
            InputIdent::Lhs,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        nth_batch: u32,
        _batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new::<Self::Config>(
            rhs.as_tensor_map(),
            x_offset,
            y_offset,
            nth_batch,
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
