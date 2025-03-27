use core::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

use crate::matmul::components::global::IndexedQuantization;
use crate::matmul::components::stage::shared::{CommonStageConfig, RhsTile, RhsTileExpand};
use crate::matmul::components::stage::{
    Buffering, LazyTask, NoTask, StageEvent, StageMatmul, StageMatmulFamily, TilingLayout,
};
use crate::matmul::components::tile::{TileMatmul, TileMatmulFamily};
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulPrecision, MatmulSize,
};
use crate::matmul::components::{
    Ident, MatmulConfigFactory, MatmulProblem,
    global::{self, AccumulatorLoader},
    stage::{StageConfig as _, StageWriter},
};
use crate::matmul::kernels::MatmulAvailabilityError;

use super::{LhsBufferReader, LhsBufferReaderFamily, RhsBufferReader, RhsBufferReaderFamily};

pub struct SingleBufferMatmulFamily<TMM: TileMatmulFamily> {
    _instruction: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> StageMatmulFamily for SingleBufferMatmulFamily<TMM> {
    fn stage_shape(config: &Self::Config) -> MatmulSize {
        config.tiling.total_shape()
    }

    fn tile_count(config: &Self::Config) -> MatmulSize {
        config.tiling.tile_count
    }

    type LhsReader = LhsBufferReaderFamily;
    type RhsReader = RhsBufferReaderFamily;
    type Matmul<I: Numeric, O: Numeric, Acc: Numeric, TL: TilingLayout, TR: TilingLayout> =
        SingleBufferMatmul<I, O, Acc, TMM::Matmul<I, Acc>, TL, TR>;
}

impl<TMM> MatmulConfigFactory for SingleBufferMatmulFamily<TMM>
where
    TMM: TileMatmulFamily,
{
    type Input = (CompleteStageTiling, Buffering);
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        TMM::check_config(&config.to_tmm_config())
    }

    fn check_availability<R: Runtime, MP: MatmulPrecision>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &Self::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        TMM::check_availability::<R, MP>(client, &config.tmm_config)
    }

    fn make_config(
        input: Self::Input,
        problem: &MatmulProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        quantized: bool,
    ) -> Self::Config {
        let tile_shape = input.0.tile_shape;
        let tile_count = input.0.tile_count;

        let tmm_config = TMM::make_config(tile_shape, problem, cube_dim, cube_count, quantized);

        let tiling = CompleteStageTiling {
            tile_count,
            tile_shape,
        };

        CommonStageConfig::new(tmm_config, tiling, tile_count.m, quantized, input.1)
    }
}

/// Performs matrix multiplication at the stage level, where each plane is responsible for a row of tiles:
/// - One plane per tile in m dimension,
/// - One accumulator per tile in n dimension
///
/// Very similar to multi buffer, except is unable to have more than one buffer, and takes BufferReaders for StageReaders
///
/// # Assumptions
/// - There are at least as many planes as the stage size in m
pub struct SingleBufferMatmul<
    I: Numeric,
    O: Numeric,
    EA: Numeric,
    TMM: TileMatmul<I, EA>,
    TL: TilingLayout,
    TR: TilingLayout,
> {
    _input_precision: PhantomData<I>,
    _output_precision: PhantomData<O>,
    _accumulator_precision: PhantomData<EA>,
    _instruction: PhantomData<TMM>,
    _tiling_layout_lhs: PhantomData<TL>,
    _tiling_layout_rhs: PhantomData<TR>,
}

#[cube]
impl<I, O, EA, TMM, TL, TR> StageMatmul<I, O, EA> for SingleBufferMatmul<I, O, EA, TMM, TL, TR>
where
    I: Numeric,
    O: Numeric,
    EA: Numeric,
    TMM: TileMatmul<I, EA>,
    TL: TilingLayout,
    TR: TilingLayout,
{
    type Config = CommonStageConfig<TMM::Config>;
    type LhsReader = LhsBufferReader<I, TL>;
    type RhsReader = RhsBufferReader<I, TR>;
    type Accumulator = Sequence<TMM::Accumulator>;
    type LhsTile = TMM::Lhs;
    type RhsTile = RhsTile<TMM::Rhs>;

    fn execute(
        lhs_reader: &Self::LhsReader,
        rhs_reader: &Self::RhsReader,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        scaling: CubeOption<f32>,
        #[comptime] config: Self::Config,
    ) {
        Self::execute_with_task::<NoTask>(
            lhs_reader,
            rhs_reader,
            lhs_fragment,
            rhs_fragments,
            acc,
            scaling,
            config,
            NoTask::new(),
        );
    }

    fn execute_with_task<TK: LazyTask>(
        lhs_reader: &Self::LhsReader,
        rhs_reader: &Self::RhsReader,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        scaling: CubeOption<f32>,
        #[comptime] config: Self::Config,
        task: TK,
    ) {
        comptime! {
            if scaling.is_some() {
                todo!();
            }
        }

        match rhs_fragments {
            RhsTile::Single(rhs_fragment) => Self::execute_single_buffer::<TK>(
                lhs_reader,
                rhs_reader,
                lhs_fragment,
                rhs_fragment,
                acc,
                config,
                task,
            ),
            RhsTile::Double(rhs_fragments) => Self::execute_double_buffer::<TK>(
                lhs_reader,
                rhs_reader,
                lhs_fragment,
                rhs_fragments,
                acc,
                config,
                task,
            ),
        }
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile) {
        (
            TMM::allocate_lhs(config.to_tmm_config()),
            match config.buffering {
                Buffering::Single => RhsTile::new_Single(TMM::allocate_rhs(config.to_tmm_config())),
                Buffering::Double => RhsTile::new_Double((
                    TMM::allocate_rhs(config.to_tmm_config()),
                    TMM::allocate_rhs(config.to_tmm_config()),
                )),
            },
        )
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let mut accumulators = Sequence::<TMM::Accumulator>::new();

        #[unroll]
        for _ in 0..config.tile_count().n {
            accumulators.push(TMM::allocate_accumulator(config.to_tmm_config()));
        }

        accumulators
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        #[unroll]
        for i in 0..config.tile_count().n {
            TMM::zero_accumulator(acc.index_mut(i), config.to_tmm_config());
        }
    }

    fn fill_accumulator<L: AccumulatorLoader<O, EA, Self::Config>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        #[unroll]
        for i in 0..config.tile_count().n {
            let acc = acc.index_mut(i);
            L::load::<I, TMM>(loader, acc, i, config.to_tmm_config());
        }
    }

    fn read_accumulator<SW: StageWriter<O>, G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut SW,
        quantization: CubeOption<IndexedQuantization<O>>,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let out_smem_line_size = global_config.stage_line_size(Ident::Out);
        let num_tile_lines =
            stage_config.tiling_dimensions(Ident::Out).tile_size() / out_smem_line_size;

        let start = num_tile_lines * UNIT_POS_Y;
        let mut out_smem = SharedMemory::<O>::new_lined(
            num_tile_lines * stage_config.num_planes(),
            out_smem_line_size,
        );

        #[unroll]
        for accumulator_iter in 0..acc.len() {
            let accumulator = acc.index(accumulator_iter);
            let mut smem_slice = out_smem.slice_mut(start, start + num_tile_lines);
            TMM::read_accumulator(accumulator, &mut smem_slice, stage_config.to_tmm_config());
            SW::write::<O, G>(
                out,
                smem_slice.to_slice(),
                UNIT_POS_Y,
                accumulator_iter,
                global_config,
            );
        }
    }
}

#[cube]
impl<I, O, EA, TMM, TL, TR> SingleBufferMatmul<I, O, EA, TMM, TL, TR>
where
    I: Numeric,
    O: Numeric,
    EA: Numeric,
    TMM: TileMatmul<I, EA>,
    TL: TilingLayout,
    TR: TilingLayout,
{
    // Execute stage matmul with a single buffer for rhs.
    fn execute_single_buffer<TK: LazyTask>(
        lhs_reader: &LhsBufferReader<I, TL>,
        rhs_reader: &RhsBufferReader<I, TR>,
        lhs_fragment: &mut TMM::Lhs,
        rhs_fragment: &mut TMM::Rhs,
        acc: &mut <Self as StageMatmul<I, O, EA>>::Accumulator,
        #[comptime] config: <Self as StageMatmul<I, O, EA>>::Config,
        mut task: TK,
    ) {
        TK::on_event(&mut task, StageEvent::Begin);

        let lhs_tile = LhsBufferReader::read_tile::<TMM::Config>(lhs_reader, UNIT_POS_Y, config);
        TMM::fill_lhs(&lhs_tile, lhs_fragment, config.to_tmm_config());
        TK::on_event(&mut task, StageEvent::LhsLoaded);

        // Comptime iterator
        let mut acc_iter = comptime![0u32];
        let acc_len = acc.len();

        #[unroll]
        for _ in 0..acc_len {
            let remaining = comptime! {acc_len - acc_iter};

            TK::on_event(&mut task, comptime!(StageEvent::RhsRemaining(remaining)));
            let rhs_tile = RhsBufferReader::read_tile::<TMM::Config>(rhs_reader, acc_iter, config);
            TMM::fill_rhs(&rhs_tile, rhs_fragment, config.to_tmm_config());
            TK::on_event(&mut task, comptime!(StageEvent::RhsLoaded(acc_iter)));

            let accumulator = acc.index_mut(acc_iter);

            TK::on_event(&mut task, comptime!(StageEvent::TmmRemaining(remaining)));
            TMM::execute(
                lhs_fragment,
                rhs_fragment,
                accumulator,
                config.to_tmm_config(),
            );
            TK::on_event(&mut task, comptime!(StageEvent::TmmCompleted(acc_iter)));

            comptime![acc_iter += 1];
        }

        TK::on_event(&mut task, comptime!(StageEvent::Finish));
    }

    // Execute stage matmul with two alternating buffers for rhs.
    fn execute_double_buffer<TK: LazyTask>(
        lhs_reader: &LhsBufferReader<I, TL>,
        rhs_reader: &RhsBufferReader<I, TR>,
        lhs_fragment: &mut TMM::Lhs,
        rhs_fragments: &mut (TMM::Rhs, TMM::Rhs),
        acc: &mut <Self as StageMatmul<I, O, EA>>::Accumulator,
        #[comptime] config: <Self as StageMatmul<I, O, EA>>::Config,
        mut _task: TK,
    ) {
        let lhs_tile = LhsBufferReader::read_tile::<TMM::Config>(lhs_reader, UNIT_POS_Y, config);
        TMM::fill_lhs(&lhs_tile, lhs_fragment, config.to_tmm_config());

        let mut accumulator_iter = comptime![0];

        let rhs_tile_first =
            RhsBufferReader::read_tile::<TMM::Config>(rhs_reader, accumulator_iter, config);
        TMM::fill_rhs(
            &rhs_tile_first,
            &mut rhs_fragments.0,
            config.to_tmm_config(),
        );

        #[unroll]
        for _ in 1..acc.len() {
            let (current, next) = if comptime! {accumulator_iter % 2 == 0} {
                (&mut rhs_fragments.0, &mut rhs_fragments.1)
            } else {
                (&mut rhs_fragments.1, &mut rhs_fragments.0)
            };

            let rhs_tile_next = RhsBufferReader::read_tile::<TMM::Config>(
                rhs_reader,
                comptime![accumulator_iter + 1],
                config,
            );
            TMM::fill_rhs(&rhs_tile_next, next, config.to_tmm_config());

            let accumulator = acc.index_mut(accumulator_iter);
            TMM::execute(lhs_fragment, current, accumulator, config.to_tmm_config());

            comptime![accumulator_iter += 1];
        }

        let last = if comptime! {accumulator_iter % 2 == 0} {
            &mut rhs_fragments.0
        } else {
            &mut rhs_fragments.1
        };

        let accumulator = acc.index_mut(accumulator_iter);
        TMM::execute(lhs_fragment, last, accumulator, config.to_tmm_config());
    }
}
