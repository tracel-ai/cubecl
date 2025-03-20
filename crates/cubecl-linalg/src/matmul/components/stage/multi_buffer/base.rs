use crate::matmul::components::global::AccumulatorLoader;
use crate::matmul::components::global::IndexedQuantization;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::{StageConfig, StageMatmul, StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::TileMatmul;
use crate::matmul::components::tile::{TileConfig, TileMatmulFamily};
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulSize,
};
use crate::matmul::components::{Ident, MatmulProblem, global, stage::StageWriter, tile};
use crate::matmul::kernels::MatmulAvailabilityError;
use core::any::TypeId;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_reduce::ReduceInstruction;
use cubecl_reduce::instructions::MaxAbs;
use cubecl_reduce::primitives::ReduceRange;
use cubecl_reduce::primitives::reduce_slice_shared;
use cubecl_reduce::primitives::reduce_tree;
use cubecl_std::{CubeOption, CubeOptionExpand};

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
impl<ES, EG, EA, TMM, TL, TR> StageMatmul<ES, EG, EA> for MultiBufferMatmul<ES, EG, EA, TMM, TL, TR>
where
    ES: Numeric,
    EG: Numeric,
    EA: Numeric,
    TMM: tile::TileMatmul<ES, EA>,
    TL: TilingLayout,
    TR: TilingLayout,
{
    type Config = CommonStageConfig<TMM::Config>;

    type LhsReader = LhsReader<ES, TL>;
    type RhsReader = RhsReader<ES, TR>;
    type Accumulator = StageAcc<ES, EA, TMM>;
    type LhsTile = TMM::Lhs;
    type RhsTile = TMM::Rhs;

    fn execute(
        lhs_reader: &LhsReader<ES, TL>,
        rhs_reader: &RhsReader<ES, TR>,
        lhs_tile: &mut Self::LhsTile,
        rhs_tile: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        scaling: CubeOption<f32>,
        #[comptime] config: Self::Config,
    ) {
        #[unroll]
        for buffer_iter in 0..config.tile_count().k {
            let tile_lhs =
                LhsReader::read_tile::<TMM::Config>(lhs_reader, UNIT_POS_Y, buffer_iter, config);
            TMM::fill_lhs(&tile_lhs, lhs_tile, config.to_tmm_config());

            #[unroll]
            for accumulator_iter in 0..acc.tmm_accumulators.len() {
                let tile_rhs = RhsReader::read_tile::<TMM::Config>(
                    rhs_reader,
                    buffer_iter,
                    accumulator_iter,
                    config,
                );
                TMM::fill_rhs(&tile_rhs, rhs_tile, config.to_tmm_config());

                let accumulator = acc.tmm_accumulators.index_mut(accumulator_iter);
                TMM::execute(lhs_tile, rhs_tile, accumulator, config.to_tmm_config());
            }
        }
        acc.accumulate_dequantized_if_quantized(scaling, config);
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (TMM::Lhs, TMM::Rhs) {
        (
            TMM::allocate_lhs(config.to_tmm_config()),
            TMM::allocate_rhs(config.to_tmm_config()),
        )
    }

    fn read_accumulator<SW: StageWriter<EG>, G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut SW,
        quantization: CubeOption<IndexedQuantization<EG>>,
        #[comptime] stage_config: Self::Config,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = global_config.stage_line_size(Ident::Out);
        let num_tile_lines =
            stage_config.tiling_dimensions(Ident::Out).tile_size() / out_smem_line_size;

        let start = num_tile_lines * UNIT_POS_Y;

        let quantization_memories = acc.quantization_memories;
        let acc = &acc.tmm_accumulators;

        match (quantization, quantization_memories) {
            (CubeOption::Some(quantization), CubeOption::Some(memories)) => {
                // let mut acc_smem = SharedMemory::<EA>::new_lined(
                //     num_tile_lines * stage_config.num_planes(),
                //     out_smem_line_size,
                // );
                // let mut acc_slice = acc_smem.slice_mut(start, start + num_tile_lines);

                let mut mem_dequantized = memories.dequantized;
                let slice_mut = mem_dequantized.to_slice_mut();

                requantize(slice_mut, quantization, stage_config);
                sync_units();

                #[unroll]
                for accumulator_iter in 0..acc.len() {
                    SW::write::<f32, G>(
                        out,
                        slice_mut.to_slice(),
                        UNIT_POS_Y,
                        accumulator_iter,
                        global_config,
                    );
                }
            }
            _ => {
                let mut out_smem = SharedMemory::<EG>::new_lined(
                    num_tile_lines * stage_config.num_planes(),
                    out_smem_line_size,
                );
                let mut smem_slice = out_smem.slice_mut(start, start + num_tile_lines);

                #[unroll]
                for accumulator_iter in 0..acc.len() {
                    let accumulator = acc.index(accumulator_iter);
                    TMM::read_accumulator(
                        accumulator,
                        &mut smem_slice,
                        stage_config.to_tmm_config(),
                    );
                    SW::write::<EG, G>(
                        out,
                        smem_slice.to_slice(),
                        UNIT_POS_Y,
                        accumulator_iter,
                        global_config,
                    );
                }
            }
        }
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        let mut tmm_accumulators = Sequence::<TMM::Accumulator>::new();

        #[unroll]
        for _ in 0..config.tile_count().n {
            tmm_accumulators.push(TMM::allocate_accumulator(config.to_tmm_config()));
        }

        let quantization_memories = if config.quantized {
            let line_size = config.line_size(Ident::Out);
            let mem_size = config.tiling_dimensions(Ident::Out).total_size() / line_size;
            CubeOption::new_Some(QuantizationMemories::<EA> {
                quantized: SharedMemory::<EA>::new_lined(mem_size, line_size),
                dequantized: SharedMemory::<f32>::new_lined(mem_size, line_size),
            })
        } else {
            CubeOption::new_None()
        };

        StageAcc::<ES, EA, TMM> {
            tmm_accumulators,
            quantization_memories,
        }
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        #[unroll]
        for i in 0..config.tile_count().n {
            TMM::zero_accumulator(acc.tmm_accumulators.index_mut(i), config.to_tmm_config());
        }
    }

    fn fill_accumulator<L: AccumulatorLoader<EG, EA, Self::Config>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        #[unroll]
        for i in 0..config.tile_count().n {
            let acc = acc.tmm_accumulators.index_mut(i);
            L::load::<ES, TMM>(loader, acc, i, config.to_tmm_config());
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

// This currently assumes that EG is i8.
// The types are generics simply to please the rust type system.
// As such all type casts should be no-ops (I hope).
#[cube]
fn requantize<EG: Numeric, TMM: TileConfig>(
    slice: SliceMut<Line<f32>>,
    mut quantization: IndexedQuantization<EG>,
    #[comptime] config: CommonStageConfig<TMM>,
) {
    if comptime!(TypeId::of::<EG>() != TypeId::of::<i8>()) {
        comptime!(panic!(
            "invalid types for requantization (expected EG = i8)"
        ));
    }

    let line_size = config.line_size(Ident::Out);
    let length = config.tiling_dimensions(Ident::Out).total_size() / line_size;

    // Use cubecl_reduce primitives to find the value with the maximum absolute value.
    let mut accumulator = reduce_slice_shared::<f32, SliceMut<Line<f32>>, MaxAbs>(
        &slice,
        ReduceRange {
            start: 0,
            end: length,
            step: 1,
        },
        config.num_planes(),
        line_size,
        cubecl_reduce::LineMode::Parallel,
        true,
        cubecl_reduce::BoundChecksInner::Branch,
    );
    let max_abs = reduce_tree::<f32, MaxAbs>(&mut accumulator, config.num_planes());
    let max_abs = MaxAbs::merge_line::<f32>(max_abs, 0); // The 0 here is a dummy value

    let scale_out = 1.0 / max_abs;
    quantization.write_scale_out(1.0 / max_abs, line_size);
    rescale::<f32>(
        slice,
        scale_out,
        length,
        line_size,
        config.num_planes() * config.plane_dim(),
    );
}

#[cube]
fn rescale<EA: Numeric>(
    mut slice: SliceMut<Line<EA>>,
    scale_out: f32,
    #[comptime] length: u32,
    #[comptime] line_size: u32,
    #[comptime] num_units: u32,
) {
    let num_elems_per_unit = length / num_units; // TODO is this always exact.
    #[unroll]
    for k in 0..num_elems_per_unit {
        let index = num_units * k + UNIT_POS;
        let scale_out = Line::empty(line_size).fill(scale_out);
        let value = Line::<f32>::cast_from(slice[index]);
        let value = Line::<EA>::cast_from(Line::round(value / scale_out));
        slice[index] = value;
    }
}

#[derive(CubeType, Clone)]
pub struct StageAcc<ES: Numeric, EA: Numeric, TMM: TileMatmul<ES, EA>> {
    tmm_accumulators: Sequence<TMM::Accumulator>,
    quantization_memories: CubeOption<QuantizationMemories<EA>>,
}

#[derive(CubeType, Clone, Copy)]
struct QuantizationMemories<EA: Numeric> {
    quantized: SharedMemory<Line<EA>>,
    dequantized: SharedMemory<Line<f32>>,
}

#[cube]
impl<EA: Numeric> QuantizationMemories<EA> {
    pub fn quantized_slice_mut<TMM: TileConfig>(
        &mut self,
        acc_index: u32,
        #[comptime] config: CommonStageConfig<TMM>,
    ) -> SliceMut<Line<EA>> {
        let line_size = config.line_size(Ident::Out);
        let tile_size = config.tiling_dimensions(Ident::Out).total_size() / line_size;
        let num_tiles_per_plane = config.tiling_dimensions(Ident::Out).tile_count_row();

        let start = UNIT_POS_Y * tile_size * num_tiles_per_plane + acc_index * tile_size;
        let end = start + tile_size;

        self.quantized.slice_mut(start, end)
    }

    fn add_dequantized<TMM: TileConfig>(
        &mut self,
        scaling: f32,
        #[comptime] config: CommonStageConfig<TMM>,
    ) {
        let line_size = config.line_size(Ident::Out);
        let tile_size = config.tiling_dimensions(Ident::Out).total_size() / line_size;
        let num_tiles_per_plane = config.tiling_dimensions(Ident::Out).tile_count_row();
        let cube_dim = config.num_planes * config.plane_dim();

        let scaling = Line::<f32>::empty(line_size).fill(scaling);
        let elems_per_unit = tile_size * num_tiles_per_plane / cube_dim;
        #[unroll]
        for index in 0..elems_per_unit {
            self.dequantized[index] = Line::<f32>::cast_from(self.quantized[index]) * scaling;
        }
    }
}

#[cube]
impl<ES: Numeric, EA: Numeric, TMM: TileMatmul<ES, EA>> StageAcc<ES, EA, TMM> {
    fn accumulate_dequantized_if_quantized(
        &mut self,
        scaling: CubeOption<f32>,
        #[comptime] config: CommonStageConfig<TMM::Config>,
    ) {
        let quantization_memories = self.quantization_memories;
        let acc = &self.tmm_accumulators;
        #[allow(clippy::single_match)]
        match (quantization_memories, scaling) {
            (CubeOption::Some(memories), CubeOption::Some(scaling)) => {
                for acc_index in 0..acc.len() {
                    let mut slice = memories.clone().quantized_slice_mut(acc_index, config);
                    TMM::read_accumulator(acc.index(acc_index), &mut slice, config.to_tmm_config());
                    memories.clone().add_dequantized(scaling, config);
                }
            }
            _ => {}
        }
    }
}
