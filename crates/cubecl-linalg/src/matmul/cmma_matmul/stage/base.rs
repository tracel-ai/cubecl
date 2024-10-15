use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::CmmaStageSize;
use crate::matmul::launch::stage_matmul_launch;
use crate::matmul::matmul_stage::{StageMatmul, StageReader, StageWriter};
use crate::matmul::matmul_tile::MatmulInstruction;
use crate::matmul::stage_info::{StageInfo, StageInfos};
use crate::matmul::tests::matmul_test_launcher::LINE_SIZE_OUT;
use crate::matmul::{
    matrix_layout::MatrixLayout,
    problem::{MatmulProblem, Requirements},
    subroutine::PlaneMapper,
};
use crate::matmul::{FixedShapeMatmul, Matmul};

pub struct CmmaStageMatmul<
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    INSTR: MatmulInstruction<I, Acc>,
    BlockSize: CmmaStageSize,
> {
    _input_precision: PhantomData<I>,
    _output_precision: PhantomData<O>,
    _accumulator_precision: PhantomData<Acc>,
    _instruction: PhantomData<INSTR>,
    _block_size: PhantomData<BlockSize>,
}

#[cube]
impl<I, O, Acc, Instr, Block, Lhs, Rhs, Out> StageMatmul<I, O, Lhs, Rhs, Out>
    for CmmaStageMatmul<I, O, Acc, Instr, Block>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Instr: MatmulInstruction<I, Acc>,
    Block: CmmaStageSize,
    Lhs: StageReader<I>,
    Rhs: StageReader<I>,
    Out: StageWriter<O>,
{
    type Accumulator = Sequence<Instr::Out>;

    fn execute(lhs: &Lhs, rhs: &Rhs, acc: &mut Self::Accumulator) {
        let num_buffers = Block::K / Instr::K;

        let mut instruction_lhs = Instr::init_lhs(Lhs::slice_layout(lhs));
        let mut instruction_rhs = Instr::init_rhs(Rhs::slice_layout(rhs));

        #[unroll]
        for buffer_iter in 0..num_buffers {
            let tile_lhs = Lhs::read_tile(&lhs, Self::plane_id(), buffer_iter, 0u32);
            Instr::fill_lhs(tile_lhs.slice, &mut instruction_lhs);

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let tile_rhs =
                    Rhs::read_tile(&rhs, Self::plane_id(), buffer_iter, accumulator_iter);
                Instr::fill_rhs(tile_rhs.slice, &mut instruction_rhs);

                let accumulator = acc.index_mut(accumulator_iter);
                Instr::execute(&instruction_lhs, &instruction_rhs, accumulator);
            }
        }
    }

    fn acc_init_zeros() -> Self::Accumulator {
        let mut accumulators = Sequence::<Instr::Out>::new();
        let num_accumulators = Block::N / Instr::N;

        #[unroll]
        for _ in 0..num_accumulators {
            accumulators.push(Instr::init_output());
        }

        accumulators
    }

    fn acc_read(acc: &Self::Accumulator, out: &mut Out) {
        let line_size = LINE_SIZE_OUT;

        let num_tile_lines = Instr::M * Instr::N / line_size;
        let start = num_tile_lines * Self::plane_id();

        let mut smem =
            SharedMemory::<O>::new_lined(num_tile_lines * comptime!(Self::num_planes()), line_size);

        #[unroll]
        for accumulator_iter in 0..acc.len() {
            let accumulator = acc.index(accumulator_iter);
            let smem_slice = smem.slice_mut(start, start + num_tile_lines);
            Instr::read_output(accumulator, smem_slice);
            Out::write(
                out,
                smem_slice.as_slice(),
                Self::plane_id(),
                accumulator_iter,
            );
        }
    }
}

impl<I, O, Acc, Instr, BlockSize> FixedShapeMatmul<I, O>
    for CmmaStageMatmul<I, O, Acc, Instr, BlockSize>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Instr: MatmulInstruction<I, Acc>,
    BlockSize: CmmaStageSize,
{
    const M: u32 = BlockSize::M;
    const N: u32 = BlockSize::N;
    const K: u32 = BlockSize::K;

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        lhs: ArrayArg<'_, R>,
        rhs: ArrayArg<'_, R>,
        out: ArrayArg<'_, R>,
        layouts: (MatrixLayout, MatrixLayout),
    ) {
        stage_matmul_launch::launch_unchecked::<I, O, Self, R>(
            &client,
            cube_count,
            cube_dim,
            lhs,
            rhs,
            out,
            layouts,
            Self::stage_infos(),
        );
    }
}

impl<I, O, Acc, Instr, Block> Matmul<I, O> for CmmaStageMatmul<I, O, Acc, Instr, Block>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Instr: MatmulInstruction<I, Acc>,
    Block: CmmaStageSize,
{
    fn can_process(problem: MatmulProblem) -> bool {
        problem.m <= Block::M && problem.n <= Block::N && problem.k <= Block::K
    }

    fn requirements(_problem: MatmulProblem) -> Requirements {
        Requirements {
            num_planes: Block::M / Instr::M,
            num_cubes: 1,
        }
    }

    fn stage_infos() -> StageInfos {
        StageInfos {
            lhs: StageInfo {
                num_tiles_x: Block::M / Instr::M,
                num_tiles_y: Block::K / Instr::K,
                tile_size_x: Instr::M,
                tile_size_y: Instr::K,
            },
            rhs: StageInfo {
                num_tiles_x: Block::K / Instr::K,
                num_tiles_y: Block::N / Instr::N,
                tile_size_x: Instr::K,
                tile_size_y: Instr::N,
            },
            out: StageInfo {
                num_tiles_x: Block::M / Instr::M,
                num_tiles_y: Block::N / Instr::N,
                tile_size_x: Instr::M,
                tile_size_y: Instr::N,
            },
        }
    }
}

#[cube]
impl<I, O, Acc, Instr, Block> PlaneMapper for CmmaStageMatmul<I, O, Acc, Instr, Block>
where
    I: Numeric,
    O: Numeric,
    Acc: Numeric,
    Instr: MatmulInstruction<I, Acc>,
    Block: CmmaStageSize,
{
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }

    fn num_planes() -> u32 {
        comptime!(Block::M / Instr::M)
    }

    fn plane_dim() -> u32 {
        CUBE_COUNT_X
    }
}
