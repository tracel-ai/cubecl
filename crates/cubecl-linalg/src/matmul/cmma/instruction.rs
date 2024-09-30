use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::{cmma, prelude::*};
use half::f16;

use crate::matmul::matrix_layout::{as_cmma_layout, MatrixLayout};
use crate::matmul::MatmulInstruction;

pub struct CmmaInstruction<I: CmmaValid, O: CmmaValid> {
    _input: PhantomData<I>,
    _output: PhantomData<O>,
}

pub trait CmmaValid: Numeric {}
impl CmmaValid for f32 {}
impl CmmaValid for f16 {}

pub struct CmmaInstructionConfig {}

#[cube]
impl<I: CmmaValid, O: CmmaValid> MatmulInstruction<I, O> for CmmaInstruction<I, O> {
    type Config = CmmaInstructionConfig;
    type Lhs = cmma::Matrix<I>;
    type Rhs = cmma::Matrix<I>;
    type Out = cmma::Matrix<O>;
    const M: u32 = 16;
    const N: u32 = 16;
    const K: u32 = 16;

    fn execute(lhs: &Self::Lhs, rhs: &Self::Rhs, out: &mut Self::Out) {
        cmma::execute::<I, I, O, O>(lhs, rhs, out, out);
    }

    fn init_lhs(#[comptime] layout: MatrixLayout) -> Self::Lhs {
        cmma::Matrix::<I>::new(
            cmma::MatrixIdent::A,
            Self::M,
            Self::N,
            Self::K,
            as_cmma_layout(layout),
        )
    }

    fn init_rhs(#[comptime] layout: MatrixLayout) -> Self::Rhs {
        cmma::Matrix::<I>::new(
            cmma::MatrixIdent::B,
            Self::M,
            Self::N,
            Self::K,
            as_cmma_layout(layout),
        )
    }

    fn fill_lhs<C: CubePrimitive>(slice: &Slice<'_, C>, lhs: &mut Self::Lhs) {
        cmma::load(&lhs, slice, Self::K);
    }

    fn fill_rhs<C: CubePrimitive>(slice: &Slice<'_, C>, rhs: &mut Self::Rhs) {
        cmma::load(&rhs, slice, Self::N);
    }

    fn init_output() -> cmma::Matrix<O> {
        let out = cmma::Matrix::<O>::new(
            cmma::MatrixIdent::Accumulator,
            Self::M,
            Self::N,
            Self::K,
            cmma::MatrixLayout::Undefined,
        );

        cmma::fill(&out, O::from_int(0));

        out
    }

    fn read_output<C: CubePrimitive>(out: &cmma::Matrix<O>, slice: &mut SliceMut<'_, C>) {
        cmma::store(slice, out, 16, cmma::MatrixLayout::RowMajor);
    }
}
