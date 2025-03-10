use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::{StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::{TileMatmul, TileMatmulFamily};
use crate::matmul::components::{
    global::{self, AccumulatorLoader},
    stage::{self, StageConfig as _, StageWriter},
    Ident, MatmulConfigFactory, MatmulProblem,
};
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulPrecision, MatmulSize,
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
    type Input = CompleteStageTiling;
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
        let tile_shape = input.tile_shape;
        let tile_count = input.tile_count;

        let tmm_config = TMM::make_config(tile_shape, problem, cube_dim, cube_count, quantized);

        let tiling = CompleteStageTiling {
            tile_count,
            tile_shape,
        };

        CommonStageConfig::new(tmm_config, tiling, tile_count.m, quantized)
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
impl<I, O, EA, TMM, TL, TR> stage::StageMatmul<I, O, EA>
    for SingleBufferMatmul<I, O, EA, TMM, TL, TR>
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
    type RhsTile = TMM::Rhs;

    fn execute(
        lhs_reader: &LhsBufferReader<I, TL>,
        rhs_reader: &RhsBufferReader<I, TR>,
        lhs_tile: &mut Self::LhsTile,
        rhs_tile: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        let tile_lhs = LhsBufferReader::read_tile::<TMM::Config>(lhs_reader, UNIT_POS_Y, config);
        TMM::fill_lhs(&tile_lhs, lhs_tile, config.to_tmm_config());

        #[unroll]
        for accumulator_iter in 0..acc.len() {
            let tile_rhs =
                RhsBufferReader::read_tile::<TMM::Config>(rhs_reader, accumulator_iter, config);
            TMM::fill_rhs(&tile_rhs, rhs_tile, config.to_tmm_config());

            let accumulator = acc.index_mut(accumulator_iter);
            TMM::execute(lhs_tile, rhs_tile, accumulator, config.to_tmm_config());
        }
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (TMM::Lhs, TMM::Rhs) {
        (
            TMM::allocate_lhs(config.to_tmm_config()),
            TMM::allocate_rhs(config.to_tmm_config()),
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
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
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
