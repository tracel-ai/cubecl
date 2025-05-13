use crate::matmul::components::InputIdent;
use crate::matmul::components::global::ZeroAccumulatorLoader;
use crate::matmul::components::global::load::TmaLoader;
use crate::matmul::components::global::load::arrive_tma;
use crate::matmul::components::global::single_stage::Config;
use crate::matmul::components::global::write::TilewiseWriter;
use crate::matmul::components::global::{GlobalMatmul, load::TmaTiling};
use crate::matmul::components::global::{Quantization, load::TmaReader};
use crate::matmul::components::stage::StageMatmul;
use crate::matmul::components::{Ident, MatmulPrecision};
use crate::matmul::kernels::matmul::LoadingPrecomputeStrategy;

use barrier::Barrier;
use cubecl_core::prelude::{barrier::BarrierLevel, *};
use cubecl_core::{self as cubecl};
use cubecl_core::{Feature, TmaFeature};
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, tensor::r#virtual::ReadWrite};
use std::{any::TypeId, marker::PhantomData};

use cubecl_core::{CubeCount, CubeDim, Runtime, client::ComputeClient};

use crate::matmul::{
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
    type Input = (SMM::Input, LoadingPrecomputeStrategy);
    type Config = Config<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        SMM::check_config(&config.to_smm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.to_smm_config())?;

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
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let mut problem = problem.clone();

        // We need smem to be unlined so slicing is simpler. TMA doesn't use the vector
        // type anyways and treats it as a void* with the actual type being set by the `TensorMap`
        problem.lhs_line_size = 1;
        problem.rhs_line_size = 1;

        let smm_config = SMM::make_config(input.0, &problem, cube_dim, cube_count, quantized);
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
            input.1,
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
    type Out = TilewiseWriter<MP::EO>;
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
        let num_elems_stages = config.tiling_dimensions(Ident::Rhs).total_size()
            + config.tiling_dimensions(Ident::Lhs).total_size();

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());
        SMM::zero_accumulator(acc, config.to_smm_config());

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

    fn init_unloader(
        out: VirtualTensor<MP::EO, ReadWrite>,
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
