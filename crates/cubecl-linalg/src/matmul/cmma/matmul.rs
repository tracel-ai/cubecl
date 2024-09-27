use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use half::f16;

use crate::matmul::{Matmul, MatmulDefinition, MatmulInstruction, Matrix, MatrixMut};

#[cube]
pub trait PlaneMapper {
    fn tile_index(row_offset: u32, col_offset: u32) -> u32;
    // let compute_plane_id = 0; // TMP

    // let num_planes_per_row = (definition.n / 16) / acc.len();
    // let tile_row = compute_plane_id / num_planes_per_row;
    // let tile_col_base = (compute_plane_id % num_planes_per_row) * acc.len();
}

pub struct CmmaMatmul<
    A: CmmaValid,
    E: CmmaValid,
    SL: PlaneMapper,
    SR: PlaneMapper,
    SO: PlaneMapper,
    I: MatmulInstruction<E, A, 16, 16, 16>,
> {
    _accumulator_precision: PhantomData<A>,
    _input_precision: PhantomData<E>,
    _lhs_smem_loader: PhantomData<SL>,
    _rhs_smem_loader: PhantomData<SR>,
    _writer: PhantomData<SO>,
    _instruction: PhantomData<I>,
}

#[derive(CubeType, Clone, Copy)]
pub struct CmmaMatmulConfig {
    pub num_accumulators: u32,
}

#[cube]
impl<
        A: CmmaValid,
        E: CmmaValid,
        SL: PlaneMapper,
        SR: PlaneMapper,
        SO: PlaneMapper,
        I: MatmulInstruction<E, A, 16, 16, 16>,
    > Matmul<E> for CmmaMatmul<A, E, SL, SR, SO, I>
{
    type Config = CmmaMatmulConfig;
    type Accumulator = Sequence<I::Output>;

    fn execute(
        lhs: &Matrix<Line<E>>,
        rhs: &Matrix<Line<E>>,
        acc: &mut Self::Accumulator,
        #[comptime] definition: MatmulDefinition,
        #[comptime] _config: &Self::Config,
    ) {
        let num_buffers = definition.k / 16;

        #[unroll]
        for buffer_iter in 0..num_buffers {
            let tile_index = SL::tile_index(0, buffer_iter);
            let start = tile_index * I::slice_length();
            let end = start + I::slice_length();

            let lhs_instruction_input = I::fill_input(lhs.slice.slice(start, end));

            #[unroll]
            for accumulator_iter in 0..acc.len() {
                let tile_index = SR::tile_index(buffer_iter, accumulator_iter);
                let start = tile_index * I::slice_length();
                let end = start + I::slice_length();

                let rhs_instruction_input = I::fill_input(rhs.slice.slice(start, end));

                let accumulator = acc.index_mut(accumulator_iter);

                I::execute(&lhs_instruction_input, &rhs_instruction_input, accumulator);
            }
        }
    }

    fn acc_init_zeros(#[comptime] config: &Self::Config) -> Self::Accumulator {
        let mut accumulators = Sequence::<I::Output>::new();

        #[unroll]
        for _ in 0..config.num_accumulators {
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
        for accumulator_iter in 0..config.num_accumulators {
            let accumulator = acc.index(accumulator_iter);

            let tile_index = SO::tile_index(0, accumulator_iter);
            let start = tile_index * I::slice_length();
            let end = start + I::slice_length();

            I::read_output(accumulator, &mut out.slice.slice_mut(start, end));
        }
    }
}

pub struct CmmaInstruction<I: CmmaValid, O: CmmaValid> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

pub trait CmmaValid: Numeric {}
impl CmmaValid for f32 {}
impl CmmaValid for f16 {}

pub struct CmmaInstructionConfig {}

#[cube]
impl<I: CmmaValid, O: CmmaValid> MatmulInstruction<I, O, 16, 16, 16> for CmmaInstruction<I, O> {
    type Config = CmmaInstructionConfig;
    type Input = cmma::Matrix<I>;
    type Output = cmma::Matrix<O>;

    fn execute(lhs: &Self::Input, rhs: &Self::Input, out: &mut Self::Output) {
        cmma::execute::<I, I, O, O>(lhs, rhs, out, out);
    }

    fn fill_input<C: CubePrimitive>(slice: &Slice<'_, C>) -> cmma::Matrix<I> {
        // TODO: this will make a new each time...
        let fragment = cmma::Matrix::<I>::new(
            cmma::MatrixIdent::A,
            16,
            16,
            16,
            cmma::MatrixLayout::RowMajor,
        );

        cmma::load(&fragment, slice, 16);

        fragment
    }

    fn init_output() -> cmma::Matrix<O> {
        let out = cmma::Matrix::<O>::new(
            cmma::MatrixIdent::Accumulator,
            16,
            16,
            16,
            cmma::MatrixLayout::Undefined,
        );

        cmma::fill(&out, O::from_int(0));

        out
    }

    fn read_output<C: CubePrimitive>(out: &cmma::Matrix<O>, slice: &mut SliceMut<'_, C>) {
        cmma::store(slice, out, 16, cmma::MatrixLayout::RowMajor);
    }

    fn slice_length() -> u32 {
        256u32
    }
}
