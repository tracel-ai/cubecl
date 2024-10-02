use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::{
    cmma_matmul::BlockInfo,
    matrix_layout::MatrixLayout,
    tile_io::{TileReader, TileWriter},
    BlockKind, BlockMatmul, BlockMatmulWrap, MatmulInstruction,
};

use super::CmmaBlockSize;

pub struct CmmaMatmul<E: Numeric, A: Numeric, I: MatmulInstruction<E, A>, Block: CmmaBlockSize> {
    _accumulator_precision: PhantomData<A>,
    _input_precision: PhantomData<E>,
    _instruction: PhantomData<I>,
    _cms: PhantomData<Block>,
}

#[derive(CubeType, Clone, Copy)]
pub struct CmmaMatmulConfig {
    pub num_accumulators: u32,
}

impl<Elem, ElemAcc, Instr, Block> BlockMatmulWrap for CmmaMatmul<Elem, ElemAcc, Instr, Block>
where
    Elem: Numeric,
    ElemAcc: Numeric,
    Instr: MatmulInstruction<Elem, ElemAcc>,
    Block: CmmaBlockSize,
{
    fn resources() -> CubeDim {
        CubeDim {
            x: 32,
            y: Block::M / Instr::M,
            z: 1,
        }
    }
}

#[cube]
impl<Elem, ElemAcc, Instr, Block, Lhs, Rhs, Out> BlockMatmul<Elem, Lhs, Rhs, Out>
    for CmmaMatmul<Elem, ElemAcc, Instr, Block>
where
    Elem: Numeric,
    ElemAcc: Numeric,
    Instr: MatmulInstruction<Elem, ElemAcc>,
    Block: CmmaBlockSize,
    Lhs: TileReader<Line<Elem>>,
    Rhs: TileReader<Line<Elem>>,
    Out: TileWriter<Line<Elem>>,
{
    type Config = CmmaMatmulConfig;
    type Accumulator = Sequence<Instr::Out>;
    const M: u32 = Block::M;
    const N: u32 = Block::N;
    const K: u32 = Block::K;

    fn execute(
        lhs: Lhs,
        rhs: Rhs,
        acc: &mut Self::Accumulator,
        #[comptime] layouts: (MatrixLayout, MatrixLayout),
    ) {
        let num_buffers = Block::K / Instr::K;
        let mut instruction_lhs = Instr::init_lhs(layouts.0);
        let mut instruction_rhs = Instr::init_rhs(layouts.1);

        #[unroll]
        for buffer_iter in 0..num_buffers {
            let tile_lhs = Lhs::read(&lhs, UNIT_POS_Y, buffer_iter, 0u32);
            Instr::fill_lhs(&tile_lhs, &mut instruction_lhs, layouts.0);

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let tile_rhs = Rhs::read(&rhs, UNIT_POS_Y, buffer_iter, accumulator_iter);
                Instr::fill_rhs(&tile_rhs, &mut instruction_rhs, layouts.1);

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
        let num_planes = <Self as BlockMatmul<Elem, Lhs, Rhs, Out>>::M / Instr::M; // TODO config
        let plane_id = UNIT_POS_Y; // TODO some plane mapper

        let num_tile_elements = Instr::M * Instr::N;
        let start = num_tile_elements * plane_id;

        let same_type =
            comptime!(std::any::TypeId::of::<Elem>() == std::any::TypeId::of::<ElemAcc>());

        if same_type {
            let mut smem = SharedMemory::<Elem>::new(num_tile_elements * num_planes);

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let accumulator = acc.index(accumulator_iter);
                let smem_slice = smem.slice_mut(start, start + num_tile_elements);
                Instr::read_output(accumulator, smem_slice);
                Out::write_with_cast(out, smem_slice.as_slice(), plane_id, accumulator_iter);
            }
        } else {
            let mut smem = SharedMemory::<ElemAcc>::new(num_tile_elements * num_planes);

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let accumulator = acc.index(accumulator_iter);
                let smem_slice = smem.slice_mut(start, start + num_tile_elements);
                Instr::read_output(accumulator, smem_slice);
                Out::write_with_cast(out, smem_slice.as_slice(), plane_id, accumulator_iter);
            }
        }
    }

    fn block_info(#[comptime] block: BlockKind) -> BlockInfo {
        match block {
            BlockKind::Lhs => BlockInfo {
                num_tiles_x: Block::M / Instr::M,
                num_tiles_y: Block::K / Instr::K,
                tile_size_x: Instr::M,
                tile_size_y: Instr::K,
            },
            BlockKind::Rhs => BlockInfo {
                num_tiles_x: Block::K / Instr::K,
                num_tiles_y: Block::N / Instr::N,
                tile_size_x: Instr::K,
                tile_size_y: Instr::N,
            },
            BlockKind::Out => BlockInfo {
                num_tiles_x: Block::M / Instr::M,
                num_tiles_y: Block::N / Instr::N,
                tile_size_x: Instr::M,
                tile_size_y: Instr::N,
            },
        }
    }
}
