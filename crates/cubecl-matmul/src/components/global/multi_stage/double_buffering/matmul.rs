use crate::components::global::load::{BufferId, SyncBufferLoader, SyncBufferLoadingStrategy};
use crate::components::global::multi_stage::double_buffering::DoubleBufferingGlobalConfig;
use crate::components::global::multi_stage::{DoubleBufferingEventListener, EventLoadingRange};
use crate::components::global::{GlobalConfig, ZeroAccumulatorLoader};
use crate::components::global::{Quantization, Specializer};
use crate::components::stage::{BufferStageToTileReader, StageConfig};
use crate::components::{
    Ident, InputIdent, InvalidConfigError, LoadingPlaneCount, MatmulConfigFactory, MatmulPrecision,
    MatmulProblem, stage,
};
use crate::components::{MatmulLineSizes, global};
use crate::components::{global::GlobalMatmulFamily, stage::BufferReaderFamily};
use crate::kernels::MatmulAvailabilityError;
use crate::kernels::matmul::{GlobalInput, MatmulSelection};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};
use cubecl_std::{CubeOption, div_ceil};
use std::marker::PhantomData;

pub struct DoubleBufferingMatmulFamily<
    SMM: stage::StageMatmulFamily,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
> {
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

impl<SMM, LL, RL> GlobalMatmulFamily for DoubleBufferingMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily<LhsReader = BufferReaderFamily, RhsReader = BufferReaderFamily>,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Matmul<MP: MatmulPrecision> =
        DoubleBufferingMatmul<MP, SMM::Matmul<MP, LL::TilingLayout, RL::TilingLayout>, LL, RL>;

    fn cube_dim(
        selection: &MatmulSelection,
        loading_plane_count: LoadingPlaneCount,
    ) -> Result<CubeDim, InvalidConfigError> {
        let compute_planes = SMM::computation_resources(&selection.tiling_scheme)?.get_count();
        let load_only_planes = loading_plane_count.load_only.resolve(compute_planes);
        Ok(CubeDim::new_2d(
            selection.plane_dim,
            compute_planes + load_only_planes,
        ))
    }
}

impl<SMM, LL, RL> MatmulConfigFactory for DoubleBufferingMatmulFamily<SMM, LL, RL>
where
    SMM: stage::StageMatmulFamily,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Input = GlobalInput<SMM::Input>;
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        LL::check::<Self::Config>(config, Ident::Lhs)?;
        RL::check::<Self::Config>(config, Ident::Rhs)?;
        SMM::check_config(&config.stage_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        SMM::check_availability::<R, MP>(client, &config.stage_config)
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

        DoubleBufferingGlobalConfig::new(
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

/// Performs matrix multiplication at the global level, with planes pipelining their work using two buffers:
/// While they trigger a load event from global memory to shared memory on buffer A,
/// they trigger a computation event from tensor cores on buffer B. Then buffers are switched.
pub struct DoubleBufferingMatmul<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
> {
    _ms: PhantomData<MP>,
    _stage_matmul: PhantomData<SMM>,
    _lhs_loading: PhantomData<LL>,
    _rhs_loading: PhantomData<RL>,
}

#[cube]
impl<MP: MatmulPrecision, SMM, LL, RL> global::GlobalMatmul<MP>
    for DoubleBufferingMatmul<MP, SMM, LL, RL>
where
    SMM: stage::StageMatmul<
            MP,
            LhsReader = BufferStageToTileReader<MP::ES, LL::TilingLayout>,
            RhsReader = BufferStageToTileReader<MP::ES, RL::TilingLayout>,
        >,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
{
    type Config = DoubleBufferingGlobalConfig<SMM::Config>;
    type LhsLoader = SyncBufferLoader<MP, Self::Config, LL>;
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

        let lhs_reader_a = Self::LhsLoader::reader(&lhs_loader, BufferId::A);
        let lhs_reader_b = Self::LhsLoader::reader(&lhs_loader, BufferId::B);
        let rhs_reader_a = Self::RhsLoader::reader(&rhs_loader, BufferId::A);
        let rhs_reader_b = Self::RhsLoader::reader(&rhs_loader, BufferId::B);

        Self::LhsLoader::fill_stage(&mut lhs_loader, BufferId::A, config);
        Self::RhsLoader::fill_stage(&mut rhs_loader, BufferId::A, config);

        let specializer = Specializer::new(config.specializer_config());

        sync_cube();

        for _ in 0..num_loops {
            execute_current_stage_and_load_next_buffer::<MP, SMM, LL, RL>(
                &lhs_reader_a,
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

            // We always advance by 2 * k because Buffer B shares the same global memory state as Buffer A,
            // but it is implicitly offset by one buffer's worth (k elements) when reading.
            Self::LhsLoader::advance_view(&mut lhs_loader, loop_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, loop_step);

            sync_cube();

            execute_current_stage_and_load_next_buffer::<MP, SMM, LL, RL>(
                &lhs_reader_b,
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

            sync_cube();
        }

        execute_current_stage_and_load_next_buffer::<MP, SMM, LL, RL>(
            &lhs_reader_a,
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

        execute_last_stage_and_write_results::<MP, SMM>(
            &lhs_reader_b,
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
        SyncBufferLoader::<MP, Self::Config, LL>::new(
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
#[cube]
fn execute_current_stage_and_load_next_buffer<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
>(
    lhs_reader: &SMM::LhsReader,
    rhs_reader: &SMM::RhsReader,
    lhs_tile: &mut SMM::LhsTile,
    rhs_tile: &mut SMM::RhsTile,
    acc: &mut SMM::Accumulator,
    lhs_loader: &mut SyncBufferLoader<MP, DoubleBufferingGlobalConfig<SMM::Config>, LL>,
    rhs_loader: &mut SyncBufferLoader<MP, DoubleBufferingGlobalConfig<SMM::Config>, RL>,
    specializer: &Specializer,
    #[comptime] buffer_to_load: BufferId,
    #[comptime] config: DoubleBufferingGlobalConfig<SMM::Config>,
) {
    if specializer.must_check_if_computer() {
        if specializer.must_check_if_loader() {
            if specializer.is_computer() && specializer.is_loader() {
                execute_with_listener::<MP, SMM, LL, RL>(
                    lhs_reader,
                    rhs_reader,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    lhs_loader,
                    rhs_loader,
                    buffer_to_load,
                    config,
                    EventLoadingRange::Full,
                );
            } else if specializer.is_computer() {
                execute_with_listener::<MP, SMM, LL, RL>(
                    lhs_reader,
                    rhs_reader,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    lhs_loader,
                    rhs_loader,
                    buffer_to_load,
                    config,
                    EventLoadingRange::None,
                );
            } else if specializer.is_loader() {
                fill_stage::<MP, SMM, LL, RL>(lhs_loader, rhs_loader, buffer_to_load, config);
            }
        } else {
            if specializer.is_computer() {
                execute_with_listener::<MP, SMM, LL, RL>(
                    lhs_reader,
                    rhs_reader,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    lhs_loader,
                    rhs_loader,
                    buffer_to_load,
                    config,
                    EventLoadingRange::Full,
                );
            } else {
                fill_stage::<MP, SMM, LL, RL>(lhs_loader, rhs_loader, buffer_to_load, config);
            }
        }
    } else {
        if specializer.must_check_if_loader() {
            if specializer.is_loader() {
                execute_with_listener::<MP, SMM, LL, RL>(
                    lhs_reader,
                    rhs_reader,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    lhs_loader,
                    rhs_loader,
                    buffer_to_load,
                    config,
                    EventLoadingRange::Full,
                );
            } else {
                execute_with_listener::<MP, SMM, LL, RL>(
                    lhs_reader,
                    rhs_reader,
                    lhs_tile,
                    rhs_tile,
                    acc,
                    lhs_loader,
                    rhs_loader,
                    buffer_to_load,
                    config,
                    EventLoadingRange::None,
                );
            }
        } else {
            execute_with_listener::<MP, SMM, LL, RL>(
                lhs_reader,
                rhs_reader,
                lhs_tile,
                rhs_tile,
                acc,
                lhs_loader,
                rhs_loader,
                buffer_to_load,
                config,
                EventLoadingRange::Full,
            );
        }
    }
}

#[cube]
fn execute_with_listener<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
>(
    lhs_reader: &SMM::LhsReader,
    rhs_reader: &SMM::RhsReader,
    lhs_tile: &mut SMM::LhsTile,
    rhs_tile: &mut SMM::RhsTile,
    acc: &mut SMM::Accumulator,
    lhs_loader: &mut SyncBufferLoader<MP, DoubleBufferingGlobalConfig<SMM::Config>, LL>,
    rhs_loader: &mut SyncBufferLoader<MP, DoubleBufferingGlobalConfig<SMM::Config>, RL>,
    #[comptime] buffer_to_load: BufferId,
    #[comptime] config: DoubleBufferingGlobalConfig<SMM::Config>,
    #[comptime] range: EventLoadingRange,
) {
    SMM::execute_with_listener::<
        DoubleBufferingEventListener<
            SyncBufferLoader<MP, DoubleBufferingGlobalConfig<SMM::Config>, LL>,
            SyncBufferLoader<MP, DoubleBufferingGlobalConfig<SMM::Config>, RL>,
            DoubleBufferingGlobalConfig<SMM::Config>,
        >,
    >(
        lhs_reader,
        rhs_reader,
        lhs_tile,
        rhs_tile,
        acc,
        config.stage_config(),
        DoubleBufferingEventListener::new(buffer_to_load, lhs_loader, rhs_loader, config, range),
    );
}

#[cube]
fn fill_stage<
    MP: MatmulPrecision,
    SMM: stage::StageMatmul<MP>,
    LL: SyncBufferLoadingStrategy,
    RL: SyncBufferLoadingStrategy,
>(
    lhs_loader: &mut SyncBufferLoader<MP, DoubleBufferingGlobalConfig<SMM::Config>, LL>,
    rhs_loader: &mut SyncBufferLoader<MP, DoubleBufferingGlobalConfig<SMM::Config>, RL>,
    #[comptime] buffer_to_load: BufferId,
    #[comptime] config: DoubleBufferingGlobalConfig<SMM::Config>,
) {
    SyncBufferLoader::<MP, _, LL>::fill_stage(lhs_loader, buffer_to_load, config);
    SyncBufferLoader::<MP, _, RL>::fill_stage(rhs_loader, buffer_to_load, config);
}

#[cube]
fn execute_last_stage_and_write_results<MP: MatmulPrecision, SMM: stage::StageMatmul<MP>>(
    lhs_reader: &SMM::LhsReader,
    rhs_reader: &SMM::RhsReader,
    lhs_tile: &mut SMM::LhsTile,
    rhs_tile: &mut SMM::RhsTile,
    acc: &mut SMM::Accumulator,
    out_writer: &mut SMM::Writer,
    specializer: &Specializer,
    #[comptime] config: DoubleBufferingGlobalConfig<SMM::Config>,
) {
    if specializer.must_check_if_computer() {
        if specializer.is_computer() {
            execute_and_write::<MP, SMM>(
                lhs_reader, rhs_reader, lhs_tile, rhs_tile, acc, out_writer, config,
            );
        }
    } else {
        execute_and_write::<MP, SMM>(
            lhs_reader, rhs_reader, lhs_tile, rhs_tile, acc, out_writer, config,
        );
    }
}

#[cube]
fn execute_and_write<MP: MatmulPrecision, SMM: stage::StageMatmul<MP>>(
    lhs_reader: &SMM::LhsReader,
    rhs_reader: &SMM::RhsReader,
    lhs_tile: &mut SMM::LhsTile,
    rhs_tile: &mut SMM::RhsTile,
    acc: &mut SMM::Accumulator,
    out_writer: &mut SMM::Writer,
    #[comptime] config: DoubleBufferingGlobalConfig<SMM::Config>,
) {
    SMM::execute(
        lhs_reader,
        rhs_reader,
        lhs_tile,
        rhs_tile,
        acc,
        config.stage_config(),
    );

    SMM::write_results::<DoubleBufferingGlobalConfig<SMM::Config>>(
        acc,
        out_writer,
        config.stage_config(),
        config,
    );
}
