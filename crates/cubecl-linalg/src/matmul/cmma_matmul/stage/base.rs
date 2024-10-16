use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::CmmaStageSize;
use crate::matmul::cmma_matmul::config::CmmaConfig;
use crate::matmul::config::PlaneMapper;
use crate::matmul::launch::stage_matmul_launch;
use crate::matmul::matmul_stage::{SmmConfig, StageMatmul, StageReader, StageWriter};
use crate::matmul::matmul_tile::TileMatmul;
use crate::matmul::stage_info::{StageInfo, StageInfos};
use crate::matmul::Matmul;

pub struct CmmaStageMatmul<
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Tmm: TileMatmul<I, Acc>,
    BlockSize: CmmaStageSize,
> {
    _input_precision: PhantomData<I>,
    _output_precision: PhantomData<O>,
    _accumulator_precision: PhantomData<Acc>,
    _instruction: PhantomData<Tmm>,
    _block_size: PhantomData<BlockSize>,
}

#[cube]
impl<I, O, Acc, Tmm, StageSize, Lhs, Rhs, Out> StageMatmul<I, O, Lhs, Rhs, Out>
    for CmmaStageMatmul<I, O, Acc, Tmm, StageSize>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Tmm: TileMatmul<I, Acc, Config = CmmaConfig>,
    StageSize: CmmaStageSize,
    Lhs: StageReader<I>,
    Rhs: StageReader<I>,
    Out: StageWriter<O>,
{
    const M: u32 = StageSize::M;
    const N: u32 = StageSize::N;
    const K: u32 = StageSize::K;
    type Accumulator = Sequence<Tmm::Out>;

    fn execute(lhs: &Lhs, rhs: &Rhs, acc: &mut Self::Accumulator) {
        let num_buffers = StageSize::K / Tmm::K;

        let mut instruction_lhs = Tmm::init_lhs(Lhs::slice_layout(lhs));
        let mut instruction_rhs = Tmm::init_rhs(Rhs::slice_layout(rhs));

        #[unroll]
        for buffer_iter in 0..num_buffers {
            let (tile_lhs, _) = Lhs::read_tile(&lhs, Self::plane_id(), buffer_iter, 0u32);
            Tmm::fill_lhs(tile_lhs, &mut instruction_lhs);

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let (tile_rhs, _) =
                    Rhs::read_tile(&rhs, Self::plane_id(), buffer_iter, accumulator_iter);
                Tmm::fill_rhs(tile_rhs, &mut instruction_rhs);

                let accumulator = acc.index_mut(accumulator_iter);
                Tmm::execute(&instruction_lhs, &instruction_rhs, accumulator);
            }
        }
    }

    fn acc_init_zeros() -> Self::Accumulator {
        let mut accumulators = Sequence::<Tmm::Out>::new();
        let num_accumulators = StageSize::N / Tmm::N;

        #[unroll]
        for _ in 0..num_accumulators {
            accumulators.push(Tmm::init_output());
        }

        accumulators
    }

    fn acc_read(acc: &Self::Accumulator, out: &mut Out, #[comptime] config: &Self::Config) {
        let line_size = config.out_smem_line_size;
        let num_tile_lines = Tmm::M * Tmm::N / line_size;
        let start = num_tile_lines * Self::plane_id();

        let mut smem =
            SharedMemory::<O>::new_lined(num_tile_lines * comptime!(Self::num_planes()), line_size);

        #[unroll]
        for accumulator_iter in 0..acc.len() {
            let accumulator = acc.index(accumulator_iter);
            let smem_slice = smem.slice_mut(start, start + num_tile_lines);
            Tmm::read_output(accumulator, smem_slice);
            Out::write(
                out,
                &smem_slice.as_slice(),
                Self::plane_id(),
                accumulator_iter,
                line_size,
            );
        }
    }
}

impl<I, O, Acc, Tmm, Block> Matmul<I, O> for CmmaStageMatmul<I, O, Acc, Tmm, Block>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Tmm: TileMatmul<I, Acc, Config = CmmaConfig>,
    Block: CmmaStageSize,
{
    type Config = CmmaConfig;

    fn stage_infos() -> StageInfos {
        StageInfos {
            lhs: StageInfo {
                num_tiles_x: Block::M / Tmm::M,
                num_tiles_y: Block::K / Tmm::K,
                tile_size_x: Tmm::M,
                tile_size_y: Tmm::K,
            },
            rhs: StageInfo {
                num_tiles_x: Block::K / Tmm::K,
                num_tiles_y: Block::N / Tmm::N,
                tile_size_x: Tmm::K,
                tile_size_y: Tmm::N,
            },
            out: StageInfo {
                num_tiles_x: Block::M / Tmm::M,
                num_tiles_y: Block::N / Tmm::N,
                tile_size_x: Tmm::M,
                tile_size_y: Tmm::N,
            },
        }
    }

    fn check_config(config: Self::Config) {
        let _ = comptime!(check_num_planes(Block::M / Tmm::M, config.num_planes));
        Tmm::check_config(config.into_tmm_config());
    }

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: Self::Config,
    ) {
        Self::check_config(config);
        stage_matmul_launch::launch_unchecked::<I, O, Self, R>(
            &client,
            cube_count,
            cube_dim,
            lhs,
            rhs,
            out,
            config.layouts,
            Self::stage_infos(),
            config,
        );
    }
}

#[cube]
impl<I, O, Acc, Tmm, Block> PlaneMapper for CmmaStageMatmul<I, O, Acc, Tmm, Block>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Tmm: TileMatmul<I, Acc>,
    Block: CmmaStageSize,
{
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }

    fn num_planes() -> u32 {
        CUBE_DIM_Y
    }

    fn plane_dim() -> u32 {
        CUBE_DIM_X
    }
}

fn check_num_planes(expected_num_planes: u32, actual_num_planes: u32) {
    assert_eq!(
        expected_num_planes, actual_num_planes,
        "Error: Expected {expected_num_planes} planes, but found {actual_num_planes}. 
        The number of planes is equal to cube dimension y which should be set to {expected_num_planes}.",
    );
}
