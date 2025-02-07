use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::StageConfig as _;
use crate::matmul::components::stage::{StageConfig, StageMatmulFamily};
use crate::matmul::components::tile::{TileMatmul, TileMatmulFamily};
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulPrecision, MatmulSize, MatrixLayout,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use crate::matmul::{
    components::{
        global::{self, AccumulatorLoader},
        stage::{self, StageWriter},
        Ident, MatmulConfigFactory, MatmulProblem,
    },
    kernels::matmul::AdvancedConfig,
};

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
    type Matmul<I: Numeric, O: Numeric, Acc: Numeric> =
        SingleBufferMatmul<I, O, Acc, TMM::Matmul<I, Acc>>;
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
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let tile_shape = input.tile_shape;
        let tile_count = input.tile_count;

        let tmm_config =
            TMM::make_config(tile_shape, problem, cube_dim, cube_count, advanced_config);

        let tiling = CompleteStageTiling {
            tile_count,
            tile_shape,
        };

        CommonStageConfig::new(
            tmm_config,
            tiling,
            cube_dim.y,
            advanced_config.lhs_tiling_order,
            advanced_config.rhs_tiling_order,
        )
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
pub struct SingleBufferMatmul<I: Numeric, O: Numeric, EA: Numeric, TMM: TileMatmul<I, EA>> {
    _input_precision: PhantomData<I>,
    _output_precision: PhantomData<O>,
    _accumulator_precision: PhantomData<EA>,
    _instruction: PhantomData<TMM>,
}

#[cube]
impl<I, O, EA, TMM> stage::StageMatmul<I, O, EA> for SingleBufferMatmul<I, O, EA, TMM>
where
    I: Numeric,
    O: Numeric,
    EA: Numeric,
    TMM: TileMatmul<I, EA>,
{
    type Config = CommonStageConfig<TMM::Config>;
    type LhsReader = LhsBufferReader<I>;
    type RhsReader = RhsBufferReader<I>;
    type Accumulator = Sequence<TMM::Accumulator>;
    type LhsTile = TMM::Lhs;
    type RhsTile = TMM::Rhs;

    fn execute(
        lhs_reader: &LhsBufferReader<I>,
        rhs_reader: &RhsBufferReader<I>,
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
        let num_tile_lines = stage_config.tiling(Ident::Out).tile_size() / out_smem_line_size;

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

    fn init_acc_sum_lhs_rows(#[comptime] config: Self::Config) -> SharedMemory<Line<EA>> {
        let line_size = match comptime!(config.layout(Ident::Lhs)) {
            MatrixLayout::RowMajor => comptime!(1),
            MatrixLayout::ColMajor => config.line_size(Ident::Lhs),
        };
        SharedMemory::new_lined(
            config.tiling.get(Ident::Lhs).total_row() / line_size,
            line_size,
        )
    }

    fn sums_lhs_rows(
        lhs: &Self::LhsReader,
        acc: &mut SliceMut<Line<EA>>,
        #[comptime] config: Self::Config,
    ) {
        let tiling = config.tiling.get(Ident::Lhs);
        let num_rows = tiling.tile_shape_row();
        let num_cols = tiling.tile_shape_col();
        let line_size = config.line_size(Ident::Lhs);

        match comptime!(config.layout(Ident::Lhs)) {
            MatrixLayout::RowMajor => {
                let num_sums_per_unit = num_rows / config.plane_dim(); // TODO Is this always an exact division?

                for k in 0..num_sums_per_unit {
                    let row = UNIT_POS_X * num_sums_per_unit + k;
                    if row < num_rows {
                        let start = row * num_cols;
                        let end = start + num_cols;

                        let tile =
                            LhsBufferReader::read_tile::<TMM::Config>(lhs, UNIT_POS_Y, config);

                        acc[row + UNIT_POS_Y * num_rows] +=
                            sum_range_parallel(&tile, start, end, line_size);
                    }
                }
            }
            MatrixLayout::ColMajor => {
                // TODO Are these always exact divisions?
                let num_lines_row_axis = num_rows / line_size;
                let num_sums_per_unit = num_lines_row_axis / config.plane_dim();

                for k in 0..num_sums_per_unit {
                    let start = UNIT_POS_X * num_sums_per_unit + k;
                    if start < num_lines_row_axis {
                        let stride = num_lines_row_axis;

                        let tile =
                            LhsBufferReader::read_tile::<TMM::Config>(lhs, UNIT_POS_Y, config);

                        acc[start] += sum_range_perpendicular(&tile, start, stride, line_size);
                    }
                }
            }
        }
    }

    /// Reads the result of the accumulator and hands it to the stage writer
    fn write_output_quantized<SW: StageWriter<O>, G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        sums_lhs_rows: &SharedMemory<Line<EA>>,
        out: &mut SW,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = global_config.stage_line_size(Ident::Out);
        let line_count_per_tile = stage_config.tiling(Ident::Out).tile_size() / out_smem_line_size;

        let start = line_count_per_tile * UNIT_POS_Y;
        let mut acc_smem = SharedMemory::<EA>::new_lined(
            line_count_per_tile * stage_config.num_planes(),
            out_smem_line_size,
        );

        let mut out_smem = SharedMemory::<O>::new_lined(
            line_count_per_tile * stage_config.num_planes(),
            out_smem_line_size,
        );

        #[unroll]
        for accumulator_iter in 0..acc.len() {
            let accumulator = acc.index(accumulator_iter);
            let mut acc_slice = acc_smem.slice_mut(start, start + line_count_per_tile);
            let mut out_slice = out_smem.slice_mut(start, start + line_count_per_tile);

            TMM::read_accumulator(accumulator, &mut acc_slice, stage_config.to_tmm_config());

            match stage_config.layout(Ident::Lhs) {
                MatrixLayout::RowMajor => {
                    // Add lhs_row_sums to acc_slice.
                    let row_tile = UNIT_POS_X;
                    if row_tile < stage_config.tiling.get(Ident::Out).tile_shape_row() {
                        let row = row_tile
                            + UNIT_POS_Y * stage_config.tiling.get(Ident::Out).tile_shape_row();
                        let line_count_per_col =
                            stage_config.tiling.get(Ident::Out).tile_shape_col()
                                / out_smem_line_size;
                        for col in 0..line_count_per_col {
                            acc_slice[row * line_count_per_col + col] =
                                Line::empty(out_smem_line_size).fill(sums_lhs_rows[row][0]);
                            // sums_lhs_rows line size == 1.
                        }
                    }

                    // Convert acc_slice to out_slice. For now use a dummy technic for testing.
                    // TODO Check out-of-bound.
                    let cast_per_unit = line_count_per_tile / stage_config.plane_dim();
                    for k in 0..cast_per_unit {
                        let index = UNIT_POS_X * cast_per_unit + k;
                        out_slice[index] = Line::cast_from(acc_slice[index]);
                    }

                }
                MatrixLayout::ColMajor => {
                    // TODO
                }
            }

            SW::write::<O, G>(
                out,
                out_slice.to_slice(),
                UNIT_POS_Y,
                accumulator_iter,
                global_config,
            );
        }
    }
}

#[cube]
fn sum_range_parallel<In: Numeric, Acc: Numeric>(
    slice: &Slice<Line<In>>,
    start: u32,
    end: u32,
    #[comptime] line_size: u32,
) -> Line<Acc> {
    // Sum all lines together
    let mut acc = Line::<Acc>::empty(line_size).fill(Acc::from_int(0));
    for index in start..end {
        acc += Line::<Acc>::cast_from(slice[index]);
    }

    // Sum the elements withing the final line.
    let mut sum = Acc::from_int(0);
    #[unroll]
    for k in 0..line_size {
        sum += acc[k];
    }

    // Cast to a line with a size of 1 for type compatibility.
    Line::empty(1).fill(sum)
}

#[cube]
fn sum_range_perpendicular<In: Numeric, Acc: Numeric>(
    slice: &Slice<Line<In>>,
    start: u32,
    stride: u32,
    #[comptime] line_size: u32,
) -> Line<Acc> {
    let mut acc = Line::<Acc>::empty(line_size).fill(Acc::from_int(0));
    for index in range_stepped(start, slice.len(), stride) {
        acc += Line::<Acc>::cast_from(slice[index]);
    }
    acc
}
