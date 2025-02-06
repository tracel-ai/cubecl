use crate::matmul::components::global::output_loader::Unloader;
use crate::matmul::components::global::{self, CommonGlobalConfig, InputLoader};
use crate::matmul::components::global::{GlobalConfig, ZeroAccumulatorLoader};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::Ident;
use crate::matmul::components::{stage, MatmulPrecision};
use crate::tensor::{ReadWrite, VirtualTensor};

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

use super::loader::{LhsBufferLoader, RhsBufferLoader};

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on buffer A,
/// they trigger a computation event from tensor cores on buffer B. Then buffers are switched.
pub struct PipelinedMatmul<MP: MatmulPrecision, SMM: stage::StageMatmul<MP::ES, MP::EG, MP::EA>> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> global::GlobalMatmul<MP> for PipelinedMatmul<MP, SMM>
where
    SMM: stage::StageMatmul<
        MP::ES,
        MP::EG,
        MP::EA,
        LhsReader = LhsBufferReader<MP::ES>,
        RhsReader = RhsBufferReader<MP::ES>,
    >,
{
    type Config = CommonGlobalConfig<SMM::Config>;
    type LhsLoader = LhsBufferLoader<MP::EG, MP::ES, SMM::Config>;
    type RhsLoader = RhsBufferLoader<MP::EG, MP::ES, SMM::Config>;
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
        let buffer_step = config.stage_dim(Ident::Lhs).tile_size_col();
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
