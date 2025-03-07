use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::{
    components::{
        global::{self, output_loader::Quantizer, AccumulatorLoader},
        stage::{
            shared::CommonStageConfig, StageConfig, StageMatmul, StageMatmulFamily, StageWriter,
            TilingLayout,
        },
        tile::{self, Tile, TileMatmulFamily},
        CompleteStageTiling, Ident, InvalidConfigError, MatmulConfigFactory, MatmulPrecision,
        MatmulProblem, MatmulSize, MatrixLayout,
    },
    kernels::{matmul::AdvancedConfig, MatmulAvailabilityError},
};

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
        advanced_config: &AdvancedConfig,
        quantized: bool,
    ) -> Self::Config {
        let tile_shape = input.tile_shape;
        let tile_count = input.tile_count;

        let tmm_config = TMM::make_config(
            tile_shape,
            problem,
            cube_dim,
            cube_count,
            advanced_config,
            quantized,
        );

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
impl<I, O, Acc, TMM, TL, TR> StageMatmul<I, O, Acc> for MultiBufferMatmul<I, O, Acc, TMM, TL, TR>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    TMM: tile::TileMatmul<I, Acc>,
    TL: TilingLayout,
    TR: TilingLayout,
{
    type Config = CommonStageConfig<TMM::Config>;

    type LhsReader = LhsReader<I, TL>;
    type RhsReader = RhsReader<I, TR>;
    type Accumulator = Sequence<TMM::Accumulator>;
    type LhsTile = TMM::Lhs;
    type RhsTile = TMM::Rhs;

    #[allow(clippy::single_match)]
    fn execute(
        lhs_reader: &LhsReader<I, TL>,
        rhs_reader: &RhsReader<I, TR>,
        lhs_tile: &mut Self::LhsTile,
        rhs_tile: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        quantizer: &mut Option<Quantizer<Acc>>,
        #[comptime] config: Self::Config,
    ) {
        // TODO Hacker ici pour faire les sommes.

        // stage lhs
        // [ tile00 tile01 tile02 UNIT_POS_Y = 0
        //   tile10 tile11 tile11 UNIT_POS_Y = 1
        // ]
        //
        // stage rhs
        // [ tile00 tile01 tile02
        //   tile10 tile11 tile12
        //   tile20 tile21 tile22
        // ] ai0    ai1    ai2
        //
        //
        // stage out
        // #tiles = NUM_PLANES x NUM_ACC
        #[unroll]
        for buffer_iter in 0..config.tile_count().k {
            let lhs_slice =
                LhsReader::read_tile::<TMM::Config>(lhs_reader, UNIT_POS_Y, buffer_iter, config);

            TMM::fill_lhs(&lhs_slice, lhs_tile, config.to_tmm_config());

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let rhs_slice = RhsReader::read_tile::<TMM::Config>(
                    rhs_reader,
                    buffer_iter,
                    accumulator_iter,
                    config,
                );
                TMM::fill_rhs(&rhs_slice, rhs_tile, config.to_tmm_config());

                let accumulator = acc.index_mut(accumulator_iter);
                TMM::execute(lhs_tile, rhs_tile, accumulator, config.to_tmm_config());

                match comptime!(quantizer.clone()) {
                    Some(mut quantizer) => sum_rhs_cols::<I, Acc, Self::Config>(
                        &rhs_slice,
                        &mut quantizer.rhs_sums.to_slice_mut(),
                        config,
                    ),
                    None => {}
                }
            }

            match comptime!(quantizer.clone()) {
                Some(mut quantizer) => sum_lhs_rows::<I, Acc, Self::Config>(
                    &lhs_slice,
                    &mut quantizer.lhs_sums.to_slice_mut(),
                    config,
                ),
                None => {}
            }
        }
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (TMM::Lhs, TMM::Rhs) {
        (
            TMM::allocate_lhs(config.to_tmm_config()),
            TMM::allocate_rhs(config.to_tmm_config()),
        )
    }

    #[allow(clippy::single_match)]
    fn read_accumulator<SW: StageWriter<O>, G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut SW,
        quantizer: Option<Quantizer<Acc>>,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let smem_line_size = global_config.stage_line_size(Ident::Out);
        let num_tile_lines =
            stage_config.tiling_dimensions(Ident::Out).tile_size() / smem_line_size;
        let smem_length = num_tile_lines * stage_config.num_planes();

        let start = num_tile_lines * UNIT_POS_Y;

        match comptime!(quantizer) {
            Some(quantizer) => {
                let mut acc_smem = SharedMemory::<Acc>::new_lined(smem_length, smem_line_size);
                #[unroll]
                for accumulator_iter in 0..acc.len() {
                    let accumulator = acc.index(accumulator_iter);

                    let mut smem_slice = acc_smem.slice_mut(start, start + num_tile_lines);
                    TMM::read_accumulator(
                        accumulator,
                        &mut smem_slice,
                        stage_config.to_tmm_config(),
                    );
                    quantizer.add_quantization_into::<G>(&mut smem_slice, global_config);
                    SW::write::<Acc, G>(
                        out,
                        smem_slice,
                        UNIT_POS_Y,
                        accumulator_iter,
                        global_config,
                    );
                }
            }
            _ => {
                let mut out_smem = SharedMemory::<O>::new_lined(smem_length, smem_line_size);
                #[unroll]
                for accumulator_iter in 0..acc.len() {
                    let accumulator = acc.index(accumulator_iter);

                    let mut smem_slice = out_smem.slice_mut(start, start + num_tile_lines);
                    TMM::read_accumulator(
                        accumulator,
                        &mut smem_slice,
                        stage_config.to_tmm_config(),
                    );
                    SW::write::<O, G>(out, smem_slice, UNIT_POS_Y, accumulator_iter, global_config);
                }
            }
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

    fn fill_accumulator<L: AccumulatorLoader<O, Acc, Self::Config>>(
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

#[cube]
fn sum_lhs_rows<In: Numeric, Acc: Numeric, Config: StageConfig>(
    tile: &Tile<In>,
    sums: &mut SliceMut<Acc>,
    #[comptime] config: Config,
) {
    let tiling = config.tiling_dimensions(Ident::Lhs);
    let num_rows = tiling.tile_shape_row();
    let num_cols = tiling.tile_shape_col();
    let line_size = config.line_size(Ident::Lhs);

    match comptime!(config.matrix_layout(Ident::Lhs)) {
        MatrixLayout::RowMajor => {
            let num_sums_per_unit = div_ceil(num_rows, config.plane_dim());
            for k in 0..num_sums_per_unit {
                let row = UNIT_POS_X * num_sums_per_unit + k;
                if row < num_rows {
                    let start = row * num_cols;
                    let end = start + num_cols;

                    sums[row + UNIT_POS_Y * num_rows] +=
                        sum_range_parallel::<In, Acc>(tile, start, end, line_size);
                }
            }
        }
        MatrixLayout::ColMajor => {
            // TODO Are these always exact divisions? Fix with div_ceil
            let num_lines_row_axis = num_rows / line_size;
            let num_sums_per_unit = num_lines_row_axis / config.plane_dim();

            for k in 0..num_sums_per_unit {
                let start = UNIT_POS_X * num_sums_per_unit + k;
                if start < num_lines_row_axis {
                    let stride = num_lines_row_axis;
                    let line = sum_range_perpendicular(tile, start, stride, line_size);
                    #[unroll]
                    for k in 0..line.size() {
                        sums[start * line.size() + k] += line[k];
                    }
                }
            }
        }
    }
}

#[cube]
fn sum_rhs_cols<In: Numeric, Acc: Numeric, Config: StageConfig>(
    tile: &Tile<In>,
    sums: &mut SliceMut<Acc>,
    #[comptime] config: Config,
) {
    let tiling = config.tiling_dimensions(Ident::Lhs);
    let num_rows = tiling.tile_shape_row();
    let num_cols = tiling.tile_shape_col();
    let line_size = config.line_size(Ident::Lhs);

    match comptime!(config.matrix_layout(Ident::Lhs)) {
        MatrixLayout::RowMajor => {
            let num_sums_per_unit = div_ceil(num_rows, config.plane_dim());
            for k in 0..num_sums_per_unit {
                let row = UNIT_POS_X * num_sums_per_unit + k;
                if row < num_rows {
                    let start = row * num_cols;
                    let end = start + num_cols;

                    sums[row + UNIT_POS_Y * num_rows] +=
                        sum_range_parallel::<In, Acc>(tile, start, end, line_size);
                }
            }
        }
        MatrixLayout::ColMajor => {
            comptime!(todo!());
            // // TODO Are these always exact divisions? Fix with div_ceil
            // let num_lines_row_axis = num_rows / line_size;
            // let num_sums_per_unit = num_lines_row_axis / config.plane_dim();

            // let tile = LhsBufferReader::read_tile::<TMM::Config>(reader, UNIT_POS_Y, config);

            // for k in 0..num_sums_per_unit {
            //     let start = UNIT_POS_X * num_sums_per_unit + k;
            //     if start < num_lines_row_axis {
            //         let stride = num_lines_row_axis;

            //         sums[start] += sum_range_perpendicular(&tile, start, stride, line_size);
            //     }
            // }
        }
    }
}

#[cube]
fn sum_range_parallel<In: Numeric, Acc: Numeric>(
    tile: &Tile<In>,
    start: u32,
    end: u32,
    #[comptime] line_size: u32,
) -> Acc {
    let slice = tile.slice;
    // Sum all lines together
    let mut acc = Line::<Acc>::empty(line_size).fill(Acc::from_int(0));
    for index in start..end {
        acc += Line::<Acc>::cast_from(slice[index]);
    }

    // Sum the elements within the final line.
    let mut sum = Acc::from_int(0);
    #[unroll]
    for k in 0..line_size {
        sum += acc[k];
    }

    sum
}

#[cube]
fn sum_range_perpendicular<In: Numeric, Acc: Numeric>(
    tile: &Tile<In>,
    start: u32,
    stride: u32,
    #[comptime] line_size: u32,
) -> Line<Acc> {
    let slice = tile.slice;

    let mut acc = Line::<Acc>::empty(line_size).fill(Acc::from_int(0));
    for index in range_stepped(start, slice.len(), stride) {
        acc += Line::<Acc>::cast_from(slice[index]);
    }
    acc
}

#[cube]
#[allow(clippy::manual_div_ceil)]
fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
