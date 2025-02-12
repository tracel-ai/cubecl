use crate::matmul::components::global;
use crate::matmul::components::global::output_loader::Unloader;
use crate::matmul::components::global::ZeroAccumulatorLoader;
use crate::matmul::components::global::{GlobalConfig as _, GlobalMatmul, InputLoader};
use crate::matmul::components::stage::single_buffer::{LhsBufferReader, RhsBufferReader};
use crate::matmul::components::stage::StageMatmul;
use crate::matmul::components::Ident;
use crate::matmul::components::MatmulPrecision;
use crate::tensor::{ReadWrite, VirtualTensor};

use super::config::Config;
use super::loader::{LhsBufferLoader, RhsBufferLoader};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use std::marker::PhantomData;

/// Performs matrix multiplication at the global level, with planes split between two roles:
/// - First n planes are used in the stage matmul computation, with n the number of planes needed by the underlying stage matmul
/// - Remaining planes load data to the stage
///
/// Both roles alternate the buffer (tile index in dimension k) they are working on
pub struct SpecializedMatmul<MP: MatmulPrecision, SMM: StageMatmul<MP::ES, MP::EG, MP::EA>> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<MP: MatmulPrecision, SMM> global::GlobalMatmul<MP> for SpecializedMatmul<MP, SMM>
where
    SMM: StageMatmul<
        MP::ES,
        MP::EG,
        MP::EA,
        LhsReader = LhsBufferReader<MP::ES>,
        RhsReader = RhsBufferReader<MP::ES>,
    >,
{
    type Config = Config<SMM::Config>;
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
        let is_consumer = Self::is_consumer(config);

        let num_buffers = config.stage_tiling(Ident::Lhs).tile_count_col();
        let buffer_step = config.stage_tiling(Ident::Lhs).tile_shape_col();
        let k_step = num_buffers * buffer_step; // equal to SMM::K

        let range = k_range.1 - k_range.0;
        let num_stages = (range + k_step - 1) / k_step;
        let num_loops = num_stages * num_buffers;

        SMM::zero_accumulator(acc, config.to_smm_config());

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());

        for _ in 0..num_loops {
            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config);

            let lhs_stage_reader = &Self::LhsLoader::as_stage_reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::as_stage_reader(&rhs_loader);

            sync_units();

            if is_consumer {
                SMM::execute(
                    lhs_stage_reader,
                    rhs_stage_reader,
                    &mut lhs_tile,
                    &mut rhs_tile,
                    acc,
                    config.to_smm_config(),
                );
            }

            Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, buffer_step);
        }

        if is_consumer {
            SMM::read_accumulator::<Self::Out, Self::Config>(
                acc,
                &mut out_unloader,
                config.to_smm_config(),
                config,
            );
        }
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(
            lhs,
            x_offset,
            y_offset,
            batch_offset,
            !Self::is_consumer(config),
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<MP::EG>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new(
            rhs,
            x_offset,
            y_offset,
            batch_offset,
            !Self::is_consumer(config),
            config,
        )
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

#[cube]
impl<
        MP: MatmulPrecision,
        SMM: StageMatmul<
            MP::ES,
            MP::EG,
            MP::EA,
            LhsReader = LhsBufferReader<MP::ES>,
            RhsReader = RhsBufferReader<MP::ES>,
        >,
    > SpecializedMatmul<MP, SMM>
{
    fn is_consumer(#[comptime] config: <Self as GlobalMatmul<MP>>::Config) -> bool {
        UNIT_POS_Y < config.num_consumers()
    }
}
