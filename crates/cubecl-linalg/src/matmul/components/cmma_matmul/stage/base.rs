use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{LhsStageReader, RhsStageReader, StageSize};
use crate::matmul::components::config::PlaneMapper;
use crate::matmul::components::global::GmmConfig;
use crate::matmul::components::tile::TileMatmul;
use crate::matmul::components::matrix::Ident;
use crate::matmul::components::stage::{SmmConfig, StageMatmul, StageReader, StageWriter};
use crate::matmul::components::Matmul;

/// Performs matrix multiplication at the stage level, where each plane is responsible for a row of tiles:
/// - One plane per tile in m dimension,
/// - One accumulator per tile in n dimension
///
/// # Assumptions
/// - There are as many planes as the stage size in m
pub struct PlaneRowStageMatmul<
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Tmm: TileMatmul<I, Acc, S::TmmConfig>,
    SS: StageSize,
    S: SmmConfig,
> {
    _input_precision: PhantomData<I>,
    _output_precision: PhantomData<O>,
    _accumulator_precision: PhantomData<Acc>,
    _instruction: PhantomData<Tmm>,
    _block_size: PhantomData<SS>,
    _config: PhantomData<S>,
}

#[cube]
impl<I, O, Acc, TMM, SS, S> StageMatmul<I, O, LhsStageReader<I, S>, RhsStageReader<I, S>, S>
    for PlaneRowStageMatmul<I, O, Acc, TMM, SS, S>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    TMM: TileMatmul<I, Acc, S::TmmConfig>,
    SS: StageSize,
    S: SmmConfig,
{
    const M: u32 = SS::NUM_M * TMM::M;
    const N: u32 = SS::NUM_N * TMM::N;
    const K: u32 = SS::NUM_K * TMM::K;
    type Accumulator = Sequence<TMM::Out>;

    fn execute(
        lhs: &LhsStageReader<I, S>,
        rhs: &RhsStageReader<I, S>,
        acc: &mut Self::Accumulator,
        #[comptime] config: S,
    ) {
        let mut instruction_lhs = TMM::init_lhs(config.to_tmm_config());
        let mut instruction_rhs = TMM::init_rhs(config.to_tmm_config());

        #[unroll]
        for buffer_iter in 0..SS::NUM_K {
            let tile_lhs =
                LhsStageReader::read_tile(&lhs, Self::plane_id(), buffer_iter, 0u32, config);
            TMM::fill_lhs(tile_lhs, &mut instruction_lhs, config.to_tmm_config());

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let tile_rhs = RhsStageReader::read_tile(
                    &rhs,
                    Self::plane_id(),
                    buffer_iter,
                    accumulator_iter,
                    config,
                );
                TMM::fill_rhs(tile_rhs, &mut instruction_rhs, config.to_tmm_config());

                let accumulator = acc.index_mut(accumulator_iter);
                TMM::execute(
                    &instruction_lhs,
                    &instruction_rhs,
                    accumulator,
                    config.to_tmm_config(),
                );
            }
        }
    }

    fn acc_init_zeros(#[comptime] config: S) -> Self::Accumulator {
        let mut accumulators = Sequence::<TMM::Out>::new();

        #[unroll]
        for _ in 0..SS::NUM_N {
            accumulators.push(TMM::init_output(config.to_tmm_config()));
        }

        accumulators
    }

    fn acc_read<SW: StageWriter<O, G>, G: GmmConfig>(
        acc: &Self::Accumulator,
        out: &mut SW,
        #[comptime] stage_config: S,
        #[comptime] global_config: G,
    ) {
        let out_smem_line_size = global_config.out_smem_line_size();
        let num_tile_lines =
            stage_config.stage_dim(Ident::Out).tile_num_elements() / out_smem_line_size;

        let start = num_tile_lines * Self::plane_id();
        let mut out_smem = SharedMemory::<Acc>::new_lined(
            num_tile_lines * stage_config.num_planes(),
            out_smem_line_size,
        );

        #[unroll]
        for accumulator_iter in 0..acc.len() {
            let accumulator = acc.index(accumulator_iter);
            let smem_slice = out_smem.slice_mut(start, start + num_tile_lines);
            TMM::read_output(accumulator, smem_slice, stage_config.to_tmm_config());
            SW::write(
                out,
                &smem_slice.as_slice(),
                Self::plane_id(),
                accumulator_iter,
                global_config,
            );
        }
    }
}

impl<I, O, Acc, TMM, SS, S> Matmul<I, O> for PlaneRowStageMatmul<I, O, Acc, TMM, SS, S>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    TMM: TileMatmul<I, Acc, S::TmmConfig>,
    SS: StageSize,
    S: SmmConfig,
{
    type Config = S;

    fn check_config(config: Self::Config) {
        let _ = comptime!(check_num_planes(
            config.stage_dim(Ident::Lhs).num_tiles_x,
            config.num_planes()
        ));
        TMM::check_config(config.to_tmm_config());
    }
}

#[cube]
impl<I, O, Acc, Tmm, SS, S> PlaneMapper for PlaneRowStageMatmul<I, O, Acc, Tmm, SS, S>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Tmm: TileMatmul<I, Acc, S::TmmConfig>,
    SS: StageSize,
    S: SmmConfig,
{
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

fn check_num_planes(expected_num_planes: u32, actual_num_planes: u32) {
    assert_eq!(
        expected_num_planes, actual_num_planes,
        "Error: Expected {expected_num_planes} planes, but found {actual_num_planes}. 
        The number of planes is equal to cube dimension y which should be set to {expected_num_planes}.",
    );
}
