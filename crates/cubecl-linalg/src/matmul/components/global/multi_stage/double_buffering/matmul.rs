use crate::matmul::components::global::base::InputLoader;
use crate::matmul::components::global::output_loader::Unloader;
use crate::matmul::components::global::{self, CommonGlobalConfig, SyncInputLoader};
use crate::matmul::components::global::{GlobalConfig, ZeroAccumulatorLoader};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::Ident;
use crate::matmul::components::{stage, MatmulPrecision};
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::loader::{LhsBufferLoader, RhsBufferLoader};

use crate::matmul::components::global::multi_stage::buffer_loading::BufferLoading;
use crate::matmul::components::global::{GlobalMatmulFamily, LoadingValidation};
use crate::matmul::components::stage::single_buffer::{
    LhsBufferReaderFamily, RhsBufferReaderFamily,
};
use crate::matmul::components::stage::{
    ColMajorTilingOrder, ContiguousTilingLayout, RowMajorTilingOrder, StageConfig,
};
use crate::matmul::components::tile::TileConfig;
use crate::matmul::components::InvalidConfigError;
use crate::matmul::components::MatmulConfigFactory;
use crate::matmul::components::MatmulProblem;
use crate::matmul::kernels::MatmulAvailabilityError;

pub struct DoubleBufferingMatmulFamily<SMM: stage::StageMatmulFamily> {
    _stage_matmul: PhantomData<SMM>,
}

type LhsTilingLayout = ContiguousTilingLayout<ColMajorTilingOrder>;
type RhsTilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

impl<SMM> GlobalMatmulFamily for DoubleBufferingMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily<
        LhsReader = LhsBufferReaderFamily,
        RhsReader = RhsBufferReaderFamily,
    >,
{
    type Matmul<MP: MatmulPrecision> = DoubleBufferingMatmul<
        MP,
        SMM::Matmul<MP::ES, MP::EG, MP::EA, LhsTilingLayout, RhsTilingLayout>,
    >;
}

impl<SMM> MatmulConfigFactory for DoubleBufferingMatmulFamily<SMM>
where
    SMM: stage::StageMatmulFamily,
{
    type Input = SMM::Input;
    type Config = CommonGlobalConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        let tmm_config = config.smm_config.to_tmm_config();
        let tile_shape = tmm_config.tile_shape();

        if tile_shape.m != tile_shape.n || tile_shape.n != tile_shape.k {
            return Err(Box::new("Only support square tiling"));
        }

        BufferLoading::check::<Self::Config>(config, Ident::Lhs)?;
        BufferLoading::check::<Self::Config>(config, Ident::Rhs)?;

        if config.tiling_dimensions(Ident::Lhs).tile_count_col() != 2 {
            return Err(Box::new("Double buffering matmul needs exactly 2 buffers."));
        }

        SMM::check_config(&config.to_smm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.smm_config)
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

        CommonGlobalConfig::new(
            smm_config,
            problem.m as u32 % stage_shape.m != 0,
            problem.n as u32 % stage_shape.n != 0,
            problem.k as u32 % stage_shape.k != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
            cube_dim.y,
        )
    }
}

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on buffer A,
/// they trigger a computation event from tensor cores on buffer B. Then buffers are switched.
pub struct DoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP::ES, MP::EG, MP::EA>,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> global::GlobalMatmul<MP> for DoubleBufferingMatmul<MP, SMM>
where
    SMM: stage::StageMatmul<
        MP::ES,
        MP::EG,
        MP::EA,
        LhsReader = LhsBufferReader<MP::ES, LhsTilingLayout>,
        RhsReader = RhsBufferReader<MP::ES, RhsTilingLayout>,
    >,
{
    type Config = CommonGlobalConfig<SMM::Config>;
    type LhsLoader = LhsBufferLoader<MP::EG, MP::ES, SMM::Config, LhsTilingLayout>;
    type RhsLoader = RhsBufferLoader<MP::EG, MP::ES, SMM::Config, RhsTilingLayout>;
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
        let num_buffers = 2;
        let buffer_step = config.tiling_dimensions(Ident::Lhs).tile_shape_col();
        let k_step = num_buffers * buffer_step; // equal to SMM::K

        let range = k_range.1 - k_range.0;
        let num_stages = (range + k_step - 1) / k_step;
        let num_loops = num_stages;

        SMM::zero_accumulator(acc, config.to_smm_config());

        let (mut lhs_tile_a, mut rhs_tile_a) = SMM::init_tile_inputs(config.to_smm_config());
        let (mut lhs_tile_b, mut rhs_tile_b) = SMM::init_tile_inputs(config.to_smm_config());

        ///////////////
        // Load A
        Self::LhsLoader::fill_stage(&mut lhs_loader, config);
        Self::RhsLoader::fill_stage(&mut rhs_loader, config);

        let lhs_buffer_reader_a = Self::LhsLoader::as_stage_reader(&lhs_loader);
        let rhs_buffer_reader_a = Self::RhsLoader::as_stage_reader(&rhs_loader);

        ///////////////
        // Get B
        Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
        Self::RhsLoader::advance_view(&mut rhs_loader, buffer_step);

        let lhs_buffer_reader_b = Self::LhsLoader::as_stage_reader(&lhs_loader);
        let rhs_buffer_reader_b = Self::RhsLoader::as_stage_reader(&rhs_loader);

        for _ in 0..num_loops {
            sync_units();

            ///////////////
            // Load B & Advance
            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config);

            Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, buffer_step);

            ///////////////
            // Execute A
            SMM::execute(
                &lhs_buffer_reader_a,
                &rhs_buffer_reader_a,
                &mut lhs_tile_a,
                &mut rhs_tile_a,
                acc,
                config.to_smm_config(),
            );

            sync_units();

            ///////////////
            // Load Next A
            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config);

            Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, buffer_step);

            ///////////////
            // Execute B
            SMM::execute(
                &lhs_buffer_reader_b,
                &rhs_buffer_reader_b,
                &mut lhs_tile_b,
                &mut rhs_tile_b,
                acc,
                config.to_smm_config(),
            );
        }

        sync_units();

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
        Self::LhsLoader::new(lhs, x_offset, y_offset, batch_offset, config)
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new(rhs, x_offset, y_offset, batch_offset, config)
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
