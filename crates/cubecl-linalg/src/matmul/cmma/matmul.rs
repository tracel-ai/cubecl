use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::{Matmul, MatmulInstruction, MatrixLayout, MatrixMut, TileReader};

use super::instruction::CmmaValid;

pub struct CmmaMatmul<A: CmmaValid, E: CmmaValid, I: MatmulInstruction<E, A>, CMS: CmmaMatmulSize> {
    _accumulator_precision: PhantomData<A>,
    _input_precision: PhantomData<E>,
    _instruction: PhantomData<I>,
    _cms: PhantomData<CMS>,
}

#[derive(CubeType, Clone, Copy)]
pub struct CmmaMatmulConfig {
    pub num_accumulators: u32,
}

pub trait CmmaMatmulSize {
    const M: u32;
    const N: u32;
    const K: u32;
}

pub struct S128_128_16;

impl CmmaMatmulSize for S128_128_16 {
    const M: u32 = 128;
    const N: u32 = 128;
    const K: u32 = 16;
}

#[cube]
impl<A, E, I, CMS, Lhs, Rhs> Matmul<E, Lhs, Rhs> for CmmaMatmul<A, E, I, CMS>
where
    A: CmmaValid,
    E: CmmaValid,
    I: MatmulInstruction<E, A>,
    CMS: CmmaMatmulSize,
    Lhs: TileReader<Line<E>>,
    Rhs: TileReader<Line<E>>,
{
    type Config = CmmaMatmulConfig;
    type Accumulator = Sequence<I::Out>;
    const M: u32 = CMS::M;
    const N: u32 = CMS::N;
    const K: u32 = CMS::K;

    fn execute(
        lhs: Lhs,
        rhs: Rhs,
        acc: &mut Self::Accumulator,
        #[comptime] layouts: (MatrixLayout, MatrixLayout),
        #[comptime] _config: &Self::Config,
    ) {
        let num_buffers = CMS::K / I::K;
        let mut instruction_lhs = I::init_lhs(layouts.0);
        let mut instruction_rhs = I::init_rhs(layouts.1);

        #[unroll]
        for buffer_iter in 0..num_buffers {
            let tile_lhs = Lhs::read(&lhs, 0, buffer_iter);
            I::fill_lhs(&tile_lhs, &mut instruction_lhs);

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let tile_rhs = Rhs::read(&rhs, buffer_iter, accumulator_iter);
                I::fill_rhs(&tile_rhs, &mut instruction_rhs);

                let accumulator = acc.index_mut(accumulator_iter);
                I::execute(&instruction_lhs, &instruction_rhs, accumulator);
            }
        }
    }

    fn acc_init_zeros(#[comptime] _config: &Self::Config) -> Self::Accumulator {
        let mut accumulators = Sequence::<I::Out>::new();

        let num_accumulators = Rhs::NUM_TILES_Y;

        #[unroll]
        for _ in 0..num_accumulators {
            let acc = I::init_output();

            accumulators.push(acc);
        }

        accumulators
    }

    fn acc_read(
        acc: &Self::Accumulator,
        out: &mut MatrixMut<Line<E>>,
        #[comptime] config: &Self::Config,
    ) {
        let plane_id = UNIT_POS_Y;
        let num_planes = 8u32;
        let line_size = 4u32;

        let size = Lhs::TILE_SIZE_X * Rhs::TILE_SIZE_Y;
        let mut output_smem = SharedMemory::<E>::new_lined(
            num_planes * Lhs::TILE_SIZE_X * Rhs::TILE_SIZE_Y,
            line_size,
        );

        for accumulator_iter in 0..acc.len() {
            let accumulator = acc.index(accumulator_iter);

            I::read_output(
                accumulator,
                output_smem.slice_mut(plane_id * size, plane_id * size + size),
            );

            // from output_smem to out, using tile writer
        }
    }
}
