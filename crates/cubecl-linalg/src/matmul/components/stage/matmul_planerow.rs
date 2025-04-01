use crate::matmul::components::global::AccumulatorLoader;
use crate::matmul::components::global::IndexedQuantization;
use crate::matmul::components::stage::shared::CommonStageConfig;
use crate::matmul::components::stage::shared::{RhsTile, RhsTileExpand};
use crate::matmul::components::stage::{Buffering, NoEvent, StageEvent, StageEventListener};
use crate::matmul::components::stage::{StageConfig, StageMatmul, StageMatmulFamily, TilingLayout};
use crate::matmul::components::tile::TileMatmul;
use crate::matmul::components::tile::{TileConfig, TileMatmulFamily};
use crate::matmul::components::{global, stage::StageWriter, tile, Ident, MatmulProblem};
use crate::matmul::components::{
    CompleteStageTiling, InvalidConfigError, MatmulConfigFactory, MatmulPrecision, MatmulSize,
};
use crate::matmul::kernels::MatmulAvailabilityError;
use core::any::TypeId;
use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_reduce::instructions::MaxAbs;
use cubecl_reduce::primitives::reduce_slice_shared;
use cubecl_reduce::primitives::reduce_tree;
use cubecl_reduce::primitives::ReduceRange;
use cubecl_reduce::ReduceInstruction;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::staging::StageReader;

pub struct PlaneRowMatmulFamily<TMM: TileMatmulFamily> {
    _instruction: PhantomData<TMM>,
}

impl<TMM: TileMatmulFamily> StageMatmulFamily for PlaneRowMatmulFamily<TMM> {
    fn stage_shape(config: &Self::Config) -> MatmulSize {
        config.tiling.total_shape()
    }

    fn tile_count(config: &Self::Config) -> MatmulSize {
        config.tiling.tile_count
    }

    type Matmul<MP: MatmulPrecision, TL: TilingLayout, TR: TilingLayout> =
        PlaneRowMatmul<MP, TMM::Matmul<MP>, TL, TR>;
}

impl<TMM: TileMatmulFamily> MatmulConfigFactory for PlaneRowMatmulFamily<TMM> {
    type Input = (CompleteStageTiling, Buffering);
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
        let tile_shape = input.0.tile_shape;
        let tile_count = input.0.tile_count;

        let tmm_config = TMM::make_config(tile_shape, problem, cube_dim, cube_count, quantized);

        let tiling = CompleteStageTiling {
            tile_shape,
            tile_count,
        };

        CommonStageConfig::new(tmm_config, tiling, cube_dim.y, quantized, input.1)
    }
}

/// Performs matrix multiplication at the stage level, where each plane is responsible for a row of tiles:
/// - One plane per tile in m dimension,
/// - One accumulator per tile in n dimension
///
/// # Assumptions
/// - There are as many planes as the stage size in m
pub struct PlaneRowMatmul<
    MP: MatmulPrecision,
    TMM: tile::TileMatmul<MP>,
    TL: TilingLayout,
    TR: TilingLayout,
> {
    _phantom: PhantomData<(MP, TMM, TL, TR)>,
}

#[cube]
impl<MP, TMM, TL, TR> StageMatmul<MP, TL, TR> for PlaneRowMatmul<MP, TMM, TL, TR>
where
    MP: MatmulPrecision,
    TMM: tile::TileMatmul<MP>,
    TL: TilingLayout,
    TR: TilingLayout,
{
    type Config = CommonStageConfig<TMM::Config>;

    type Accumulator = StageAcc<MP, TMM>;
    type LhsTile = TMM::Lhs;
    type RhsTile = RhsTile<TMM::Rhs>;

    fn execute(
        lhs_reader: &StageReader<MP::ES, TL>,
        rhs_reader: &StageReader<MP::ES, TR>,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        scaling: CubeOption<f32>,
        #[comptime] config: Self::Config,
    ) {
        Self::execute_with_listener::<NoEvent>(
            lhs_reader,
            rhs_reader,
            lhs_fragment,
            rhs_fragments,
            acc,
            scaling,
            config,
            NoEvent {},
        )
    }

    fn execute_with_listener<SEL: StageEventListener>(
        lhs_reader: &StageReader<MP::ES, TL>,
        rhs_reader: &StageReader<MP::ES, TR>,
        lhs_fragment: &mut Self::LhsTile,
        rhs_fragments: &mut Self::RhsTile,
        acc: &mut Self::Accumulator,
        scaling: CubeOption<f32>,
        #[comptime] config: Self::Config,
        listener: SEL,
    ) {
        match rhs_fragments {
            RhsTile::Single(rhs_fragment) => Self::execute_single_buffer::<SEL>(
                lhs_reader,
                rhs_reader,
                lhs_fragment,
                rhs_fragment,
                acc,
                scaling,
                config,
                listener,
            ),
            RhsTile::Double(rhs_fragments) => Self::execute_double_buffer::<SEL>(
                lhs_reader,
                rhs_reader,
                lhs_fragment,
                rhs_fragments,
                acc,
                scaling,
                config,
                listener,
            ),
        }
    }

    fn init_tile_inputs(#[comptime] config: Self::Config) -> (Self::LhsTile, Self::RhsTile) {
        let tmm_config = config.to_tmm_config();
        (
            TMM::allocate_lhs(tmm_config),
            match config.buffering() {
                Buffering::Single => RhsTile::new_Single(TMM::allocate_rhs(tmm_config)),
                Buffering::Double => RhsTile::new_Double((
                    TMM::allocate_rhs(tmm_config),
                    TMM::allocate_rhs(tmm_config),
                )),
            },
        )
    }

    fn read_accumulator<SW: StageWriter<MP::EO>, G: global::GlobalConfig>(
        acc: &Self::Accumulator,
        out: &mut SW,
        quantization: CubeOption<IndexedQuantization<MP::EI, MP::EO>>,
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
                let mut out_smem = SharedMemory::<MP::EO>::new_lined(
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
                    SW::write::<MP::EO, G>(
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
            CubeOption::new_Some(QuantizationMemories::<MP::EA> {
                quantized: SharedMemory::<MP::EA>::new_lined(mem_size, line_size),
                dequantized: SharedMemory::<f32>::new_lined(mem_size, line_size),
            })
        } else {
            CubeOption::new_None()
        };

        StageAcc::<MP, TMM> {
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

    fn fill_accumulator<L: AccumulatorLoader<MP, Self::Config>>(
        loader: &mut L,
        acc: &mut Self::Accumulator,
        #[comptime] config: Self::Config,
    ) {
        #[unroll]
        for i in 0..config.tile_count().n {
            let acc = acc.tmm_accumulators.index_mut(i);
            L::load::<TMM>(loader, acc, i, config.to_tmm_config());
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
impl<MP, TMM, TL, TR> PlaneRowMatmul<MP, TMM, TL, TR>
where
    MP: MatmulPrecision,
    TMM: TileMatmul<MP>,
    TL: TilingLayout,
    TR: TilingLayout,
{
    // Execute stage matmul with a single buffer for rhs.
    fn execute_single_buffer<SEL: StageEventListener>(
        lhs_reader: &StageReader<MP::ES, TL>,
        rhs_reader: &StageReader<MP::ES, TR>,
        lhs_fragment: &mut TMM::Lhs,
        rhs_fragment: &mut TMM::Rhs,
        acc: &mut <Self as StageMatmul<MP, TL, TR>>::Accumulator,
        scaling: CubeOption<f32>,
        #[comptime] config: <Self as StageMatmul<MP, TL, TR>>::Config,
        mut listener: SEL,
    ) {
        SEL::on_event(&mut listener, StageEvent::Begin);

        let k_iterations = config.tile_count().k;
        let acc_iterations = acc.tmm_accumulators.len();
        let acc_total_iterations = comptime![k_iterations * acc_iterations];

        let mut k_iter = comptime![0u32];

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in 0..k_iterations {
            let tile_lhs = StageReader::read_tile::<<Self as StageMatmul<MP, TL, TR>>::Config>(
                lhs_reader, UNIT_POS_Y, k_iter, config,
            );
            TMM::fill_lhs(&tile_lhs, lhs_fragment, config.to_tmm_config());
            SEL::on_event(
                &mut listener,
                comptime![StageEvent::LhsLoaded {
                    current: k_iter,
                    total: k_iterations
                }],
            );

            let mut acc_iter = comptime![0u32];

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 0..acc_iterations {
                let current = comptime![k_iter * acc_iterations + acc_iter];
                let rhs_tile_next = StageReader::read_tile::<
                    <Self as StageMatmul<MP, TL, TR>>::Config,
                >(rhs_reader, k_iter, acc_iter, config);
                TMM::fill_rhs(&rhs_tile_next, rhs_fragment, config.to_tmm_config());
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::RhsLoaded {
                        current,
                        total: acc_total_iterations
                    }],
                );

                let accumulator = acc.tmm_accumulators.index_mut(acc_iter);
                TMM::execute(
                    lhs_fragment,
                    rhs_fragment,
                    accumulator,
                    config.to_tmm_config(),
                );
                SEL::on_event(
                    &mut listener,
                    comptime![StageEvent::TmmCompleted {
                        current,
                        total: acc_total_iterations
                    }],
                );

                comptime![acc_iter += 1];
            }

            comptime![k_iter += 1];
        }

        SEL::on_event(&mut listener, comptime!(StageEvent::Finish));

        acc.accumulate_dequantized_if_quantized(scaling, config);
    }

    // Execute stage matmul with two alternating buffers for rhs.
    fn execute_double_buffer<SEL: StageEventListener>(
        lhs_reader: &StageReader<MP::ES, TL>,
        rhs_reader: &StageReader<MP::ES, TR>,
        lhs_fragment: &mut TMM::Lhs,
        rhs_fragments: &mut (TMM::Rhs, TMM::Rhs),
        acc: &mut <Self as StageMatmul<MP, TL, TR>>::Accumulator,
        scaling: CubeOption<f32>,
        #[comptime] config: <Self as StageMatmul<MP, TL, TR>>::Config,
        mut listener: SEL,
    ) {
        SEL::on_event(&mut listener, StageEvent::Begin);

        let k_iterations = config.tile_count().k;
        let acc_iterations = acc.tmm_accumulators.len();
        let acc_total_iterations = comptime![k_iterations * acc_iterations];

        let mut k_iter = comptime![0u32];

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in 0..k_iterations {
            let tile_lhs = StageReader::read_tile::<<Self as StageMatmul<MP, TL, TR>>::Config>(
                lhs_reader, UNIT_POS_Y, k_iter, config,
            );
            TMM::fill_lhs(&tile_lhs, lhs_fragment, config.to_tmm_config());
            SEL::on_event(
                &mut listener,
                comptime![StageEvent::LhsLoaded {
                    current: k_iter,
                    total: k_iterations
                }],
            );

            let mut acc_iter = comptime![0u32];

            let rhs_tile_first = StageReader::read_tile::<<Self as StageMatmul<MP, TL, TR>>::Config>(
                rhs_reader, k_iter, acc_iter, config,
            );
            TMM::fill_rhs(
                &rhs_tile_first,
                &mut rhs_fragments.0,
                config.to_tmm_config(),
            );
            SEL::on_event(
                &mut listener,
                comptime!(StageEvent::RhsLoaded {
                    current: k_iter * acc_iterations,
                    total: acc_total_iterations
                }),
            );

            #[unroll]
            #[allow(clippy::explicit_counter_loop)]
            for _ in 1..acc_iterations {
                let current_event = comptime![k_iter * acc_iterations + acc_iter + 1];
                let (current, next) = if comptime! {acc_iter % 2 == 0} {
                    (&mut rhs_fragments.0, &mut rhs_fragments.1)
                } else {
                    (&mut rhs_fragments.1, &mut rhs_fragments.0)
                };

                let rhs_tile_next = StageReader::read_tile::<
                    <Self as StageMatmul<MP, TL, TR>>::Config,
                >(
                    rhs_reader, k_iter, comptime![acc_iter + 1], config
                );
                TMM::fill_rhs(&rhs_tile_next, next, config.to_tmm_config());
                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::RhsLoaded {
                        current: current_event,
                        total: acc_total_iterations
                    }),
                );

                let accumulator = acc.tmm_accumulators.index_mut(acc_iter);
                TMM::execute(lhs_fragment, current, accumulator, config.to_tmm_config());
                SEL::on_event(
                    &mut listener,
                    comptime!(StageEvent::RhsLoaded {
                        current: current_event,
                        total: acc_total_iterations
                    }),
                );

                comptime![acc_iter += 1];
            }

            let last = if comptime! {acc_iter % 2 == 0} {
                &mut rhs_fragments.0
            } else {
                &mut rhs_fragments.1
            };

            let accumulator = acc.tmm_accumulators.index_mut(acc_iter);
            TMM::execute(lhs_fragment, last, accumulator, config.to_tmm_config());
            SEL::on_event(
                &mut listener,
                comptime!(StageEvent::RhsLoaded {
                    current: k_iter * acc_iterations + acc_iter + 1,
                    total: acc_total_iterations
                }),
            );

            comptime![k_iter += 1];
        }

        SEL::on_event(&mut listener, comptime!(StageEvent::Finish));

        acc.accumulate_dequantized_if_quantized(scaling, config);
    }
}

// Given a slice of f32 data, we compute the maximum absolute value (M).
// Then we rescale all values by 127 / M so that they fit within -127 and 127 (both inclusives).
//
// This also update the corresponding quantization scaling in the output to M / 127.
//
// # Cooperation at cube-level
//
// Unlike other routines in the stage matmul which use plane-level cooperation,
// this is cube-level cooperation. This is expected since we need the whole output stage (the result of all plane operations)
// to compute the output quantization scaling parameter.
//
// This function doesn't sync_units, it is the responsibility of the caller to make sure all planes have finish their
// work on tiling matmul before calling this function.
//
// # Note on generics
//
// This currently assumes that EG is i8.
// The types are generics simply to please the rust type system.
#[cube]
fn requantize<EI: Numeric, EO: Numeric, TMM: TileConfig>(
    slice: SliceMut<Line<f32>>,
    mut quantization: IndexedQuantization<EI, EO>,
    #[comptime] config: CommonStageConfig<TMM>,
) {
    if comptime!(TypeId::of::<EI>() != TypeId::of::<i8>()) {
        comptime!(panic!(
            "invalid types for requantization (expected EI = i8)"
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

    // This is where I assume that EG is i8.
    let scale_out = 127.0 / max_abs;
    rescale(
        slice,
        scale_out,
        length,
        line_size,
        config.num_planes() * config.plane_dim(),
    );

    quantization.write_scale_out(1.0 / scale_out, line_size);
}

// Multiply all element of the slice by scale_out using all units within the cube.
// See comment on [requantize] for why it is ok to use cube-level cooperation.
//
// TODO: Move this in cubecl_std.
#[cube]
fn rescale(
    mut slice: SliceMut<Line<f32>>,
    scale_out: f32,
    #[comptime] length: u32,
    #[comptime] line_size: u32,
    #[comptime] num_units: u32,
) {
    let scale_out = Line::empty(line_size).fill(scale_out);

    let bound_check = length % num_units != 0;
    let num_elems_per_unit = length.div_ceil(num_units);

    #[unroll]
    for k in 0..num_elems_per_unit {
        let index = num_units * k + UNIT_POS;
        if bound_check {
            if index < length {
                slice[index] = Line::round(slice[index] / scale_out);
            }
        } else {
            slice[index] = Line::round(slice[index] / scale_out);
        }
    }
}

/// For a regular matmul (without quantization), this is simply a sequence of TileMatmul accumulators.
///
/// For quantized matmul, we also add a pair of shared memories.
/// A quantized one with `Line<EA>` and a dequantized one with `Line<f32>`.
/// The quantized memory is used to transfer the result of the tmm_accumulators into shared memory.
/// The dequantized memory is persistent for the whole stage matmul.
/// We need to upload the result of each quantized accumulators into the dequantized one
/// since pair of lhs and rhs stages may have different scaling parameters.
/// See the [accumulated_dequantized_if_quantized] method.
#[derive(CubeType, Clone)]
pub struct StageAcc<MP: MatmulPrecision, TMM: TileMatmul<MP>> {
    tmm_accumulators: Sequence<TMM::Accumulator>,
    quantization_memories: CubeOption<QuantizationMemories<MP::EA>>,
}

// Each memory is stored as a single SharedMemory big enough to accommodate all tmm accumulators.
//
// TODO Maybe we can work with a smaller quantized memory as it is only used to read the tmm accumulators
//      into shared memory. Thus the same slice could be reused across many reads.
//
//      Also, when TMM is PlaneMMA, we don't really need an extra SharedMemory as the TMM accumulators
//      are themselves shared memories. I don't know if it is worth it to spend time on this as PlaneMMA are
//      only used during testing. I still think it is good to be aware of that optimization for future TMM implementation
//      or if we find a way to read and rescale lines from cmma fragments of the Acceralted TMM directly into the dequantized f32 memory.
#[derive(CubeType, Clone, Copy)]
struct QuantizationMemories<EA: Numeric> {
    quantized: SharedMemory<Line<EA>>, // Only to read from tmm_accumulators
    dequantized: SharedMemory<Line<f32>>, // The persistent accumulator for the stage matmul.
}

#[cube]
impl<EA: Numeric> QuantizationMemories<EA> {
    // Returns the chunk of quantized memory corresponding to acc_index and UNIT_POS_Y (plane pos)
    // as a mutable slice.
    fn quantized_slice_mut<TMM: TileConfig>(
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

    // Read elements from the quantized memory and multiply them by scaling before
    // adding them to the dequantized memory.
    //
    // This cooperates at the cube-level. See comment on [requantize] for why this is ok.
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
impl<MP: MatmulPrecision, TMM: TileMatmul<MP>> StageAcc<MP, TMM> {
    // Read the tmm_accumulator into the quantized memory.
    // Then convert the element from the quantized memory to f32
    // and multiply them by scaling before adding the result to the dequantized memory.
    //
    // Do nothing if either scaling or self.quantization_memories is None.
    //
    // This cooperates at the cube-level. See comment on [requantize] for why this is ok.
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
                #[unroll]
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
