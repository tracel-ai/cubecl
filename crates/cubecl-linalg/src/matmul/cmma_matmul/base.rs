use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::{
    id_map::PlaneMapper,
    matrix_layout::MatrixLayout,
    problem::{MatmulProblem, Requirements},
    tile_io::{BlockReader, TileWriter},
    BlockMatmul, FixedShapeMatmul, Matmul, MatmulInstruction,
};

use crate::matmul::block_info::{BlockInfo, BlockInfos};

use super::CmmaBlockSize;
use crate::matmul::launch::block_matmul_launch;

pub struct CmmaBlockMatmul<E: Numeric, A: Numeric, I: MatmulInstruction<E, A>, Block: CmmaBlockSize>
{
    _accumulator_precision: PhantomData<A>,
    _input_precision: PhantomData<E>,
    _instruction: PhantomData<I>,
    _cms: PhantomData<Block>,
}

#[derive(CubeType, Clone, Copy)]
pub struct CmmaMatmulConfig {
    pub num_accumulators: u32,
}

impl<Elem, ElemAcc, Instr, Block> FixedShapeMatmul<Elem, Elem>
    for CmmaBlockMatmul<Elem, ElemAcc, Instr, Block>
where
    Elem: Numeric,
    ElemAcc: Numeric,
    Instr: MatmulInstruction<Elem, ElemAcc>,
    Block: CmmaBlockSize,
{
    const M: u32 = Block::M;
    const N: u32 = Block::N;
    const K: u32 = Block::K;

    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount<<R as Runtime>::Server>,
        lhs: ArrayArg<'_, R>,
        rhs: ArrayArg<'_, R>,
        out: ArrayArg<'_, R>,
        layouts: (MatrixLayout, MatrixLayout),
    ) {
        block_matmul_launch::launch_unchecked::<Self, Elem, R>(
            &client,
            cube_count,
            cube_dim,
            lhs,
            rhs,
            out,
            layouts,
            Self::block_infos(),
        );
    }
}

impl<Elem, ElemAcc, Instr, Block> Matmul<Elem, Elem>
    for CmmaBlockMatmul<Elem, ElemAcc, Instr, Block>
where
    Elem: Numeric,
    ElemAcc: Numeric,
    Instr: MatmulInstruction<Elem, ElemAcc>,
    Block: CmmaBlockSize,
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

    fn block_infos() -> BlockInfos {
        BlockInfos {
            lhs: BlockInfo {
                num_tiles_x: Block::M / Instr::M,
                num_tiles_y: Block::K / Instr::K,
                tile_size_x: Instr::M,
                tile_size_y: Instr::K,
            },
            rhs: BlockInfo {
                num_tiles_x: Block::K / Instr::K,
                num_tiles_y: Block::N / Instr::N,
                tile_size_x: Instr::K,
                tile_size_y: Instr::N,
            },
            out: BlockInfo {
                num_tiles_x: Block::M / Instr::M,
                num_tiles_y: Block::N / Instr::N,
                tile_size_x: Instr::M,
                tile_size_y: Instr::N,
            },
        }
    }
}

#[cube]
impl<Elem, ElemAcc, Instr, Block> PlaneMapper for CmmaBlockMatmul<Elem, ElemAcc, Instr, Block>
where
    Elem: Numeric,
    ElemAcc: Numeric,
    Instr: MatmulInstruction<Elem, ElemAcc>,
    Block: CmmaBlockSize,
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

#[cube]
impl<Elem, ElemAcc, Instr, Block, Lhs, Rhs, Out> BlockMatmul<Elem, Lhs, Rhs, Out>
    for CmmaBlockMatmul<Elem, ElemAcc, Instr, Block>
where
    Elem: Numeric,
    ElemAcc: Numeric,
    Instr: MatmulInstruction<Elem, ElemAcc>,
    Block: CmmaBlockSize,
    Lhs: BlockReader<Elem>,
    Rhs: BlockReader<Elem>,
    Out: TileWriter<Line<Elem>>,
{
    type Config = CmmaMatmulConfig;
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
                let tile_rhs = Rhs::read_tile(&rhs, Self::plane_id(), buffer_iter, accumulator_iter);
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
        let num_tile_elements = Instr::M * Instr::N;
        let start = num_tile_elements * Self::plane_id();

        let same_type =
            comptime!(std::any::TypeId::of::<Elem>() == std::any::TypeId::of::<ElemAcc>());

        if same_type {
            let mut smem =
                SharedMemory::<Elem>::new(num_tile_elements * comptime!(Self::num_planes()));

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let accumulator = acc.index(accumulator_iter);
                let smem_slice = smem.slice_mut(start, start + num_tile_elements);
                Instr::read_output(accumulator, smem_slice);
                Out::write_with_cast(
                    out,
                    smem_slice.as_slice(),
                    Self::plane_id(),
                    accumulator_iter,
                );
            }
        } else {
            let mut smem =
                SharedMemory::<ElemAcc>::new(num_tile_elements * comptime!(Self::num_planes()));

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let accumulator = acc.index(accumulator_iter);
                let smem_slice = smem.slice_mut(start, start + num_tile_elements);
                Instr::read_output(accumulator, smem_slice);
                Out::write_with_cast(
                    out,
                    smem_slice.as_slice(),
                    Self::plane_id(),
                    accumulator_iter,
                );
            }
        }
    }
}
