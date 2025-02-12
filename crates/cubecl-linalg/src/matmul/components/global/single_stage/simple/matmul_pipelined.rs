use crate::matmul::components::global::output_loader::Unloader;
use crate::matmul::components::global::single_stage::loader::{
    LhsLoader, LoadingStrategy, RhsLoader,
};
use crate::matmul::components::global::single_stage::Config;
use crate::matmul::components::global::ZeroAccumulatorLoader;
use crate::matmul::components::global::{GlobalConfig as _, GlobalMatmul, InputLoader};
use crate::matmul::components::stage::multi_buffer::{LhsReader, RhsReader};
use crate::matmul::components::stage::StageMatmul;
use crate::matmul::components::MatmulPrecision;
use crate::tensor::{ReadWrite, VirtualTensor};

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use pipeline::Pipeline;
use std::marker::PhantomData;

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct SimplePipelinedMatmul<
    MP: MatmulPrecision,
    SMM: StageMatmul<MP::ES, MP::EG, MP::EA>,
    LL: LoadingStrategy,
    RL: LoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL> GlobalMatmul<MP> for SimplePipelinedMatmul<MP, SMM, LL, RL>
where
    SMM: StageMatmul<
        MP::ES,
        MP::EG,
        MP::EA,
        LhsReader = LhsReader<MP::ES>,
        RhsReader = RhsReader<MP::ES>,
    >,
    LL: LoadingStrategy,
    RL: LoadingStrategy,
{
    type Config = Config<SMM::Config>;
    type LhsLoader = LhsLoader<MP::EG, MP::ES, SMM::Config, LL>;
    type RhsLoader = RhsLoader<MP::EG, MP::ES, SMM::Config, RL>;
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

        // Pipeline is declared with two stages, one for lhs and one for rhs
        let pipeline = Pipeline::<MP::ES>::new(2);

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());
        SMM::zero_accumulator(acc, config.to_smm_config());

        for _ in 0..num_loops {
            // Start loading
            pipeline.producer_acquire();
            Self::LhsLoader::fill_stage_window(&mut lhs_loader, pipeline, config);
            Self::RhsLoader::fill_stage_window(&mut rhs_loader, pipeline, config);
            pipeline.producer_commit();

            let lhs_stage_reader = &Self::LhsLoader::as_stage_reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::as_stage_reader(&rhs_loader);

            // Wait for load to finish for this thread, then sync to make sure all planes have finished
            pipeline.consumer_wait();
            sync_units();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.to_smm_config(),
            );

            pipeline.consumer_release();

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
