use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::global::AccumulatorLoader;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::{StageMatmul, StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::TileMatmulFamily;
use crate::matmul::components::{
    global,
    stage::{StageConfig as _, StageWriter},
    tile, Ident, MatmulProblem,
};
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulSize,
};
use crate::matmul::kernels::MatmulAvailabilityError;

use super::reader::{LhsReader, RhsReader};
use super::{LhsReaderFamily, RhsReaderFamily};

pub struct MultiBufferMatmulFamily<TMM: TileMatmulFamily> {
    _instruction: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> StageMatmulFamily for MultiBufferMatmulFamily<TMM> {
    fn stage_shape(config: &Self::Config) -> MatmulSize {
        config.tiling.total_shape()
    }

    fn tile_count(config: &Self::Config) -> MatmulSize {
        config.tiling.tile_count
    }

    type LhsReader = LhsReaderFamily;
    type RhsReader = RhsReaderFamily;
    type Matmul<I: Numeric, O: Numeric, Acc: Numeric, TL: TilingLayout, TR: TilingLayout> =
        MultiBufferMatmul<I, O, Acc, TMM::Matmul<I, Acc>, TL, TR>;
}

impl<TMM: TileMatmulFamily> MatmulConfigFactory for MultiBufferMatmulFamily<TMM> {
    type Input = CompleteStageTiling;
    type Config = CommonStageConfig<TMM::Config>;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        check_num_planes(
            config.tiling_dimensions(Ident::Lhs).tile_count_row(),
            config.num_planes(),
        )?;
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
            tile_shape,
            tile_count,
        };

        CommonStageConfig::new(tmm_config, tiling, cube_dim.y, quantized)
    }
}

/// Performs matrix multiplication at the stage level, where each plane is responsible for a row of tiles:
/// - One plane per tile in m dimension,
/// - One accumulator per tile in n dimension
///
/// # Assumptions
/// - There are as many planes as the stage size in m
pub struct MultiBufferMatmul<
    I: Numeric,
    O: Numeric,
    EA: Numeric,
    TMM: tile::TileMatmul<I, EA>,
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
impl<I, O, EA, TMM, TL, TR> StageMatmul<I, O, EA> for MultiBufferMatmul<I, O, EA, TMM, TL, TR>
where
    I: Numeric,
    O: Numeric,
    EA: Numeric,
    TMM: tile::TileMatmul<I, EA>,
    TL: TilingLayout,
    TR: TilingLayout,
{
    type Config = CommonStageConfig<TMM::Config>;

    type LhsReader = LhsReader<I, TL>;
    type RhsReader = RhsReader<I, TR>;
    type Accumulator = Sequence<TMM::Accumulator>;
    type LhsTile = TMM::Lhs;
    type RhsTile = TMM::Rhs;

    fn execute(
        lhs_reader: &LhsReader<I, TL>,
        rhs_reader: &RhsReader<I, TR>,
        lhs_tile: &mut Self::LhsTile,
        rhs_tile: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        #[unroll]
        for buffer_iter in 0..config.tile_count().k {
            let tile_lhs =
                LhsReader::read_tile::<TMM::Config>(lhs_reader, UNIT_POS_Y, buffer_iter, config);
            TMM::fill_lhs(&tile_lhs, lhs_tile, config.to_tmm_config());

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let tile_rhs = RhsReader::read_tile::<TMM::Config>(
                    rhs_reader,
                    buffer_iter,
                    accumulator_iter,
                    config,
                );
                TMM::fill_rhs(&tile_rhs, rhs_tile, config.to_tmm_config());

                let accumulator = acc.index_mut(accumulator_iter);
                TMM::execute(lhs_tile, rhs_tile, accumulator, config.to_tmm_config());
            }
        }
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (TMM::Lhs, TMM::Rhs) {
        (
            TMM::allocate_lhs(config.to_tmm_config()),
            TMM::allocate_rhs(config.to_tmm_config()),
        )
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

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let mut acc = Sequence::<TMM::Accumulator>::new();

        #[unroll]
        for _ in 0..config.tile_count().n {
            acc.push(TMM::allocate_accumulator(config.to_tmm_config()));
        }

        acc
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
}

fn check_num_planes(
    expected_num_planes: u32,
    actual_num_planes: u32,
) -> Result<(), InvalidConfigError> {
    if expected_num_planes != actual_num_planes {
        return Err(Box::new("Error: Expected {expected_num_planes} planes, but found {actual_num_planes}. 
        The number of planes is equal to cube dimension y which should be set to {expected_num_planes}."));
    }

    Ok(())
}
