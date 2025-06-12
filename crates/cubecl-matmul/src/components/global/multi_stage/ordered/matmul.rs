use crate::components::global::load::{
    BufferId, LoadingValidation, SyncBufferLoader, SyncBufferLoadingStrategy, SyncFullLoader,
    SyncFullLoadingStrategy, sync_full_ordered,
};
use crate::components::global::multi_stage::execute::{
    execute_current_and_load_next, execute_last_and_write_results,
};
use crate::components::global::{self, GlobalConfig, ZeroAccumulatorLoader};
use crate::components::global::{Quantization, Specializer};
use crate::components::problem::MatmulLineSizes;
use crate::components::stage::FullReaderFamily;
use crate::components::stage::FullStageToTileReader;
use crate::components::stage::{BufferStageToTileReader, StageConfig};
use crate::components::{
    Ident, InputIdent, InvalidConfigError, LoadSpecializationConfig, MatmulPrecision,
    MatmulProblem, stage,
};
use crate::components::{global::GlobalMatmulFamily, stage::BufferReaderFamily};
use crate::kernels::MatmulAvailabilityError;
use crate::kernels::matmul::{GlobalInput, MatmulSelection};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use cubecl_std::{CubeOption, div_ceil};
use std::marker::PhantomData;

use super::OrderedDoubleBufferingGlobalConfig;

pub struct OrderedDoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    RL: SyncBufferLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _rhs_loading: PhantomData<RL>,
}

/// The ordered double buffering global matmul
/// requires tilewise loading on `Lhs` to guarantee that planes
/// only use data they have loaded themselves.
pub type LL = sync_full_ordered::LoadingStrategy;

impl<SMM, RL> GlobalMatmulFamily for OrderedDoubleBufferingMatmulFamily<SMM, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = FullReaderFamily, RhsReader = BufferReaderFamily>,
    RL: SyncBufferLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> = OrderedDoubleBufferingMatmul<
        MP,
        SMM::Matmul<MP, <LL as SyncFullLoadingStrategy>::TilingLayout, RL::TilingLayout>,
        RL,
    >;

    type Config = OrderedDoubleBufferingGlobalConfig<SMM::Config>;
    type Input = GlobalInput<SMM::Input>;

    fn cube_dim(
        selection: &MatmulSelection,
        load_specialization: LoadSpecializationConfig,
    ) -> Result<CubeDim, InvalidConfigError> {
        let main_flow_planes = SMM::computation_resources(&selection.tiling_scheme)?
            .as_plane_resources(selection.plane_dim)?
            .get_count();
        Ok(CubeDim::new_2d(
            selection.plane_dim,
            load_specialization
                .to_plane_roles(main_flow_planes)
                .total_count(),
        ))
    }

    fn setup(
        input: Self::Input,
        problem: &MatmulProblem,
        line_sizes: &MatmulLineSizes,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let stage_config = SMM::setup(
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

        OrderedDoubleBufferingGlobalConfig::new(
            stage_config,
            problem.m as u32 % stage_shape_m != 0,
            problem.n as u32 % stage_shape_n != 0,
            problem.k as u32 % (2 * stage_shape_k) != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            line_sizes.lhs as u32,
            line_sizes.rhs as u32,
            line_sizes.out as u32,
            cube_dim.y,
            input.loading_precompute_strategy,
            input.loader_mode,
        )
    }
}

// impl<SMM, RL> MatmulChecker for OrderedDoubleBufferingMatmulFamily<SMM, RL>
// where
//     SMM: stage::StageMatmulFamily,
//     RL: SyncBufferLoadingStrategy,
// {
//     type Input = GlobalInput<SMM::Input>;
//     type Config = OrderedDoubleBufferingGlobalConfig<SMM::Config>;

//     fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
//         <LL as LoadingValidation>::check::<Self::Config>(config, Ident::Lhs)?;
//         RL::check::<Self::Config>(config, Ident::Rhs)?;
//         SMM::check_config(&config.stage_config())
//     }

//     fn check_availability<R: Runtime, MP: MatmulPrecision>(
//         client: &ComputeClient<R::Server, R::Channel>,
//         config: &Self::Config,
//     ) -> Result<(), MatmulAvailabilityError> {
//         SMM::check_availability::<R, MP>(client, &config.stage_config)
//     }
// }

/// Performs matrix multiplication at the global level.
/// Uses double buffering with two shared memory buffers for `Rhs`,
/// but only one for `Lhs`—the second "buffer" for `Lhs` is the fragments themselves.
/// For this to work, the `Lhs` loader planes must compute using
/// only the data they have loaded themselves.
pub struct OrderedDoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    RL: SyncBufferLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, RL> global::GlobalMatmul<MP>
    for OrderedDoubleBufferingMatmul<MP, SMM, RL>
where
    SMM: stage::StageMatmul<
            MP,
            LhsReader = FullStageToTileReader<
                MP::ES,
                <LL as SyncFullLoadingStrategy>::TilingLayout,
            >,
            RhsReader = BufferStageToTileReader<MP::ES, RL::TilingLayout>,
        >,
    RL: SyncBufferLoadingStrategy,
{
    type Config = OrderedDoubleBufferingGlobalConfig<SMM::Config>;
    type LhsLoader = SyncFullLoader<MP, Self::Config, LL>;
    type RhsLoader = SyncBufferLoader<MP, Self::Config, RL>;
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
        let buffer_step = config.tiling_scheme().elements_in_stage_k();
        let loop_step = buffer_step * 2;
        let range = k_range.1 - k_range.0;
        let needed_stage_matmuls = div_ceil(range, buffer_step);

        // Algorithm assumes an even number of stages
        let num_stage_matmuls = needed_stage_matmuls + (needed_stage_matmuls % 2);
        let num_loops = (num_stage_matmuls - 2) / 2;

        SMM::zero_accumulator(acc, config.stage_config());

        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.stage_config());

        let lhs_reader = Self::LhsLoader::reader(&lhs_loader);
        let rhs_reader_a = Self::RhsLoader::reader(&rhs_loader, BufferId::A);
        let rhs_reader_b = Self::RhsLoader::reader(&rhs_loader, BufferId::B);

        Self::LhsLoader::fill_stage(&mut lhs_loader, config);
        Self::RhsLoader::fill_stage(&mut rhs_loader, BufferId::A, config);

        Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);

        let specializer = Specializer::new(
            config.plane_role_config(),
            config.specialized_loading_sides(),
        );

        sync_cube();

        for _ in 0..num_loops {
            execute_current_and_load_next::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
                &lhs_reader,
                &rhs_reader_a,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                &mut lhs_loader,
                &mut rhs_loader,
                &specializer,
                BufferId::B,
                config,
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, loop_step);

            sync_cube();

            execute_current_and_load_next::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
                &lhs_reader,
                &rhs_reader_b,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                &mut lhs_loader,
                &mut rhs_loader,
                &specializer,
                BufferId::A,
                config,
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, buffer_step);

            sync_cube();
        }

        execute_current_and_load_next::<MP, SMM, Self::LhsLoader, Self::RhsLoader, Self::Config>(
            &lhs_reader,
            &rhs_reader_a,
            &mut lhs_tile,
            &mut rhs_tile,
            acc,
            &mut lhs_loader,
            &mut rhs_loader,
            &specializer,
            BufferId::B,
            config,
        );

        sync_cube();

        execute_last_and_write_results::<MP, SMM, Self::Config>(
            &lhs_reader,
            &rhs_reader_b,
            &mut lhs_tile,
            &mut rhs_tile,
            acc,
            &mut out_writer,
            &specializer,
            config,
        );
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
        SyncFullLoader::<MP, Self::Config, LL>::new(
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
        SyncBufferLoader::<MP, Self::Config, RL>::new(
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
